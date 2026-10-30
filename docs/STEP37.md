# Step-3.7-Flash on Strata

Step models are StepFun's (`stepfun-ai`): **Step-3.5-Flash** is text-only and **Step-3.7-Flash** is the same text
decoder plus a vision encoder. Both are 196B parameters with 11B active, and both use the GGUF architecture
`step35`, so one engine path runs the two.

This engine path is **new and incomplete on purpose**. What runs today is the whole model, one token at a time,
with the experts read from the file. What does not run yet is listed at the end, and it is a real list.

## What the file says

Every number below was read from `stepfun-ai/Step-3.7-Flash-GGUF` (IQ3_XXS), not from a model card.

| | |
|---|---|
| architecture | `step35`, 45 blocks |
| dense / experts | blocks 0-2 are dense (FFN 11264); blocks 3-44 are mixtures of experts |
| experts | 288, top-8, FFN 1280 each, **plus** a shared expert of 1280 |
| gating | sigmoid of the router logits, a selection bias that picks the experts, `weights_norm`, scale 3.0 |
| attention | **one full-attention block in four** (0, 4, 8, … 44); the others use a 512-token window |
| heads | **64 query heads on a full block, 96 on a sliding one**, 8 KV heads, head dim 128 |
| RoPE | rotation covers 64 of the 128 dims on a full block and all 128 on a sliding one, base 5e6 / 1e4 |
| rope scaling | `rope_freqs.weight`, 64 entries; only the full-attention blocks read it |
| QK norm | a per-head RMSNorm on Q and on K, every block |
| attention gate | `attn *= sigmoid(g_proj(x))`, one weight per query head |
| SwiGLU clamp | blocks 43 and 44 only: 7.0 for the experts, 16.0 for the shared expert |
| vocabulary | 128 896 tokens, byte-level BPE, pre-tokenizer `deepseek-v3`, chat template in the file |
| MTP | 3 NextN blocks, in a **file of their own** (`Step3.7-flash-mtp-*.gguf`) |

## Three conventions that are silent when wrong

Each of these produced a bug during this work, and each was invisible at short positions:

1. **The pairing is NEOX.** Element *j* turns with element *j + n_rot/2*
   (`llama_model_rope_type()` answers `LLAMA_ROPE_TYPE_NEOX` for `LLM_ARCH_STEP35`). The adjacent pair
   `(2i, 2i+1)` is the Llama/Qwen convention; at 6 tokens it answers within the noise of a quantized matmul.
   At 52 tokens it is 5.7x further from llama.cpp.
2. **`rope_freqs.weight` divides the angle**, and only on the full-attention blocks. Left out, the angle of
   pair 15 shifts by 0.79 rad at position 4096, 6.5 rad at 32768 and 25.9 rad at 131072 — several whole turns.
   A long context cannot be right without it.
3. **The head count changes with the block type** (64 / 96). A kernel compiled for one shape mis-runs the
   other silently; in the reference this is a per-layer array, never a constant.

## Building

```sh
cmake -S . -B build-hip -G Ninja \
  -DSTRATA_ENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES="gfx1201;gfx1030" \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ -DCMAKE_PREFIX_PATH=/opt/rocm -DROCM_PATH=/opt/rocm \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-hip --target strata-step35 -j 12
```

Run with `LD_LIBRARY_PATH=/opt/rocm/lib` (the binary links `libhipblaslt.so.1`, which lives in
`/opt/rocm/lib`).

## Downloading

```sh
hf download stepfun-ai/Step-3.7-Flash-GGUF \
  --include "IQ3_XXS/*" \
  --local-dir "$DATA/models/step-3.7-flash-IQ3_XXS"
hf download stepfun-ai/Step-3.7-Flash-GGUF \
  --include "Step3.7-flash-mtp-Q8_0.gguf" \
  --local-dir "$DATA/mtp/step37"
```

Two shards, 46.65 GB + 29.11 GB, plus 3.71 GB for the MTP heads. `--include` takes **one** pattern: a second
pattern on the same line is read as a filename and silently downloads nothing.

## The three commands

`--inspect` reads the file and checks every tensor against the keys that describe it. It opens no GPU.

```sh
strata-step35 --inspect <first shard> [--mtp <nextn file>]
```

`--ref` is the CPU reference: the same arithmetic in float64, over the real weights. It is what the kernels are
compared against, and it is slow on purpose (about 5 s per token per block).

```sh
strata-step35 --ref <first shard> --tokens "0,671,6102" --layers 3 [--dump out.f32]
```

`--gpu-all` runs every block on the GPU for one token, with the experts streamed out of the file.

```sh
strata-step35 --gpu-all <first shard> --token 671 --pos 0
```

`--gpu-seq` runs a PROMPT of several positions through every block, dense and mixture of experts, and prints the
hidden state summed over all positions after each block - the figure `--ref` prints, so the two line up.

```sh
strata-step35 --gpu-seq <first shard> --tokens "0,671,6102,294,8760,344" --layers 4
```

## What is measured

All figures on this machine (2 x 16 GB AMD, 62 GB RAM, ROCm at `/opt/rocm`), against llama.cpp built from the
same file:

| | |
|---|---|
| tokenizer | identical ids to llama.cpp (0, 671, 6102, 294, 8760, 344 for the test prompt) |
| dense blocks 0-2 | 1.6e-04 to 9.1e-04 of the value scale |
| first expert block | 1.5e-04 |
| **all 45 blocks, one token** | **0.12% on the rms, 0.50% on the sum** |
| a prompt of 6 positions, 3 dense + 1 expert block | 3.3e-04 to 7.5e-04 |
| speed | **1.39 s per token**, warm page cache (2.13 s cold) |

The residual difference is the Q8_1 quantization of the activations, which K-quant matvecs carry by design; it
is not a difference of arithmetic. llama.cpp's own GPU (fp16) and CPU (f32) runs differ by 1.5e-02 to 3.0e-02 on
the same file, so a few 1e-3 per block is inside its spread.

## What does not work yet

- **No KV cache.** A prompt of several positions now runs (dense and expert blocks both), but the K/V of that
  pass is thrown away: a second turn recomputes the first. That reuse is the one thing between here and a chat
  turn, and the figures to beat are the ones above.
- **No MTP.** The three NextN blocks are readable (`--inspect --mtp` checks them) but not run.
- **No vision.** The `mmproj` projector (3.97 GB, llama.cpp's `PROJECTOR_TYPE_STEP3VL`) is not used.
- **No streaming pool.** Every expert is copied per token into three reused buffers, with no VRAM cache: this is
  what makes it fit in 32 GB, and it is also why it is slow.
- **No integration.** `setup.py`, `strata launch --family step35`, the app and `serve` know nothing about the
  family yet. `strata` dispatches a `step35` file to `strata-step35`, and that is all.

## Tests

```sh
ctest -R step35
```

Five tests, none of which needs the 76 GB download:

- `step35_metadata_test` - the family, the guard, and five files the guard must refuse (synthetic).
- `step35_rope_parity` - NEOX pairing and the frequency factors, **bit for bit** against a float64 host; and a
  check that the wrong pairing is visibly wrong at position 131072, so the test cannot pass vacuously.
- `step35_gemv_parity` - the quantized matvec over every type the release uses (IQ3_XXS, IQ2_S, Q4_K, IQ3_S,
  Q5_K, F32), against a float64 host dot.
- `step35_moe_parity` - the router: expert ids must match **exactly**, and a forced tie must resolve to the
  lower index on both sides.
- `step35_attn_parity` - causal GQA with the sliding window, over both head counts.
