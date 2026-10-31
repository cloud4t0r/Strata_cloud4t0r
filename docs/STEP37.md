# Step-3.7-Flash on Strata

Step models are StepFun's (`stepfun-ai`): **Step-3.5-Flash** is text-only and **Step-3.7-Flash** is the same text
decoder plus a vision encoder. Both are 196B parameters with 11B active, and both use the GGUF architecture
`step35`, so one engine path runs the two.

This engine path is **new and incomplete on purpose**. What runs today is the whole model over a prompt of several
positions, then decode steps that read the prompt's own K/V out of a quantized cache, with the experts read from
the file. What does not run yet is listed at the end, and it is a real list.

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

## The four commands

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

`--gpu-chat` runs a PROMPT through every block with the quantized K/V cache, then each `--next` token as a decode
step that attends to the cells the prompt wrote. It prints the same per-position figures `--ref` prints, so the two
outputs are the comparison:

```sh
strata-step35 --gpu-chat <first shard> --tokens "0,671,6102,294,8760,344" --next "671" [--layers N] [--max-context N]
```

Add `--logits` to either GPU command and it also applies the file's `output_norm` and `output.weight` to the last
position, printing the five tokens the model would answer with - the same line `--ref` prints, so the head is held
to the reference the blocks are held to. `strata-step35 --gpu-seq` takes it too.

Omitting `--layers` runs every block, as the help says; it used to run none, and print the embedding as if it
were the model's answer. `--gpu-seq` reads it the same way now.

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
| prompt of 6 positions, 45 blocks, then decodes | prefill **3.9 s** for the 6, then **0.96 s per decoded token** |
| prefill of 1600 positions, 45 blocks | **586 s** reading the cache, **653 s** reading the pass's fp32 K/V - same tokens, same blocks, page cache warm in both |
| one decoded token at 1600 positions of context | **0.93 s** - three decodes cost 2.79 s on top of the same prefill |
| position 1599 of that prompt, cache against no cache | rms **-0.46%** (8.352877 against 8.391474) |
| trunk head, 45 blocks, prompt `0,671,6102,294,8760,344` | `--ref` answers `11111(15.883) 260(13.809) 680(13.513) 17678(13.004) 957(12.724)`; the GPU (with or without the cache) answers the same first four, and the fifth is a near-tie: 12.79 for 270 without the cache, 12.73 for 764 with it |
| the same 6 positions, CPU reference against GPU | 237.8 s against 4.25 s |

The residual difference is the Q8_1 quantization of the activations, which K-quant matvecs carry by design; it
is not a difference of arithmetic. llama.cpp's own GPU (fp16) and CPU (f32) runs differ by 1.5e-02 to 3.0e-02 on
the same file, so a few 1e-3 per block is inside its spread.

## What does not work yet

- **No MTP.** The three NextN blocks are readable (`--inspect --mtp` checks them) but not run. The trunk's own head
  does run now: `--logits` on `--gpu-seq` or `--gpu-chat` applies the file's `output_norm` and `output` to the last
  position and prints the five tokens it would answer with, which is the figure `--ref` prints - and the two agree
  on the first four at 45 blocks (see the table above). Whatever MTP is built on, it is built on that.
- **No vision.** The `mmproj` projector (3.97 GB, llama.cpp's `PROJECTOR_TYPE_STEP3VL`) is not used.
- **No streaming pool.** Every expert is copied per token into three reused buffers, with no VRAM cache: this is
  what makes it fit in 32 GB, and it is also why it is slow.
- **No integration.** `setup.py`, `strata launch --family step35`, the app and `serve` know nothing about the
  family yet. `strata` dispatches a `step35` file to `strata-step35`, and that is all.
- **The cache is not streamed and not captured.** A cache that outgrows VRAM has no host copy yet
  (`kv_stream.hpp` has one for the QSA layers), and the position still crosses to the device as a 4-byte H2D copy,
  so the pass is not a CUDA graph. A decode step is a full pass over 45 blocks - 0.96 s, most of it the
experts streamed out of the file.

## The KV cache: what it is, and what it cost

The prompt and the decode steps differ in nothing but `n_pos`. `src/step35/step35_seq.cu` writes each position's
cells with `strata_kernels`' `kv_append_q8_steps` and then attends over `0 .. pos0+n_pos-1` through
`attn_q8_cached`, which reads the `int8` codes in place. A decode step is that same call with one token at
`pos0 = n_cached`, so the two cannot drift apart:

- `include/strata/step35/session.hpp` / `src/step35/step35_session.cu`: the `Session`. Per layer, K and V as
  `int8` codes plus one `f16` scale per 64 values (`kv_q8.hpp`'s cell), behind a page table of 128-cell pages, and
  a device step buffer that holds the position - the kernels read the position from device memory, not from an
  argument. Pools are allocated when a layer is first used, so `--layers 4` does not pay for 45. Each layer has its
  own table, which is what lets a sliding layer's table repeat and turn its pool into a ring.
- `src/step35/step35_attn_cached.cu`: `attn_q8_cached`, one warp per (row, query head), the online softmax in
  registers, the `int8` codes read in place. A gather into fp16 scratch was the alternative and it is the wrong one
  at a long context: 4 KB per cell per layer (512 MB at 131072) moved on every token, against 1 KB of codes and no
  staging buffer. Both attention kernels now run the same online softmax, so neither has a window bound:
  `attn_causal` used to keep a window's scores in a `scores[512]` of shared memory, which a full-attention block
  overran from position 512 on - measured 1.6e-03 off the float64 answer at 513 keys, 5.0e-02 at 1024, with no CUDA
  error - and it used to clamp a window wider than 512 down to 512.

Size, from the session's own arithmetic (`session_bytes_at`, which `step35_kv_parity` now checks on the model's
real geometry): **2112 B per layer per token** (1024 codes + 16 f16 scales, twice) for a layer that keeps every
position - **95 KB per token over the 45 layers**, 0.78 GB at 8192, 3.1 GB at 32768, 12.4 GB at 131072. The fp32
cache would be 4.5x that, which is why the quantized form was built first.

**The 33 sliding layers keep a ring, not the prefix.** Only one block in four attends over the whole prefix; the
others read a window of 512 positions and never look further back, so the cells they hold past that window are
written by the pass and read by nobody. Those layers hold `window + kKvPassMax - 1` cells instead - 1536 at a
window of 512 - and their page table repeats every `cells / page_size` pages, so position *p* and *p + 1536* land
on the same physical page. The attention kernel is untouched: it already resolved a cell through
`table[cell / page_size]`, and a repeating table is still a table. On the file's own pattern (12 full blocks, 33
sliding) the cache is, from `session_bytes_at` and checked by `step35_kv_parity`:

| positions | with the ring | every layer keeping every position |
|---|---|---|
| 8192 | 0.315 GB | 0.78 GB |
| 32768 | 0.938 GB | 3.11 GB |
| 131072 | **3.431 GB** | **12.46 GB** |

The 33 sliding layers drop from 9.1 GB to 0.11 GB at 131072; the 12 full ones keep their 3.32 GB, because a
full-attention block really does read the whole prefix.

Measured on this machine (2 × 16 GB AMD, IQ3_XXS), reading `rocm-smi` while the 45-block chat ran with
`--max-context 131072`: **3.778 GB of VRAM at the peak, against 12.761 GB before the ring** - the cache plus the
~0.35 GB of expert buffers and working tensors, both runs. The figures printed per position are identical in the
two runs (position 6: rms 13.050220, first four -3.551789 5.127801 0.270378 -4.080585), which is what the ring is
supposed to be: a different place to put the cells, not different cells.

The price is one rule: **a pass writes at most `kKvPassMax` positions (1024)**. A pass writes all of its cells
before any of its rows is attended, so a pass long enough to lap the ring would let its first row read cells its
last row had already overwritten. `--gpu-chat` prefills in passes of 1024 positions and `session_append` refuses a
longer one, naming the ring's size, rather than answering something plausible and wrong.

The same rule applies to reading back: a row whose window the ring has already handed to a newer position is
refused by `session_attend`, with the cell it wanted and the one the ring now holds named in the message. Decoding
never asks for one - each token attends at the newest position - but a caller walking back up the prefix would
otherwise get a row assembled from other tokens' cells.

What the cache costs, measured on the same prompt (`0,671,6102,294,8760,344,671`, 45 blocks) by running it twice -
once reading the cache, once reading the fp32 K/V of the pass:

| | |
|---|---|
| a 7-position prefill against the same 6 + 1 decode | position 6 lands on the same rms and the same first four values to six decimals: prefill and decode agree **bit for bit** |
| rms of each position, cache against no cache | -0.004%, +0.45%, -0.40%, +0.82%, -0.05%, +0.08%, -1.47% (mean 0.47%) |
| block 44's sum over the 7 positions | +0.08% (22156.0 against 22135.5) |
| the 45 block sums | 0.50% mean, 2.8% worst |

Against the float64 reference, the same 7 positions answer 0.03% to 2.43% away when they read the cache, and
0.03% to 3.95% when they do not - block 44's sum is within 0.1% across the three. The ordering there is not a
measure of either path: at 45 blocks a 1e-7 change in one attention reduction (the online softmax reorders a few
hundred additions) moves the last position by a point or two of the rms, the same way the int8 K/V does, because
near-ties at this depth decide a couple of percent either way. It is the model's sensitivity, and it is the same
order as llama.cpp's own GPU-against-CPU spread of 1.5e-02 to 3.0e-02 quoted above.

The comparison that isolates the cache is the one that does not run 45 blocks: over 4 blocks, prefill 6 positions
then decode 1, the 7 positions land within 0.13% of `--ref --layers 4` on the same 7 tokens.

The ring, on the real weights: 1600 positions through 4 blocks (block 0 full, 1-3 sliding, so the 1536-cell ring is
lapped by 64 cells) against `--gpu-seq` on the same tokens, which reads the pass's fp32 K/V and keeps nothing:

| | |
|---|---|
| position 1599, reading the ring | rms 0.285418, first four -0.127015 0.014492 0.046192 -0.025171 |
| position 1599, `--gpu-seq` | rms 0.285613, first four -0.127171 0.014042 0.046040 -0.025360 |
| difference | -0.07% on the rms - the int8 K/V, the same order the table above quotes |
| prefill of the 1600 | 14.9 s with the cache, 17.0 s without (both warm page cache, 4 blocks) |

That run also found an older defect, now fixed: `run_prompt_dense` sized its `q8_1` scratch for `n_embd` and handed
it to projections 11264 and 12288 wide, overrunning it by up to 96 KB on every prompt pass. At 6 positions the
bytes landed in memory the next allocation replaced; at 1600 they faulted the GPU inside `quantize_q8_1_kernel`.
The scratch is now sized per block for the widest input it quantizes.

## Tests

```sh
ctest -R step35
```

Six tests, none of which needs the 76 GB download:

- `step35_metadata_test` - the family, the guard, and five files the guard must refuse (synthetic).
- `step35_rope_parity` - NEOX pairing and the frequency factors, **bit for bit** against a float64 host; and a
  check that the wrong pairing is visibly wrong at position 131072, so the test cannot pass vacuously.
- `step35_gemv_parity` - the quantized matvec over every type the release uses (IQ3_XXS, IQ2_S, Q4_K, IQ3_S,
  Q5_K, F32), against a float64 host dot.
- `step35_moe_parity` - the router: expert ids must match **exactly**, and a forced tie must resolve to the
  lower index on both sides.
- `step35_attn_parity` - causal GQA with the sliding window, over both head counts, and two 520-position
  cases past the 512 keys the kernel used to hold in shared memory (they fail on the previous kernel:
  3.5e-04 and 4.5e-03).
- `step35_kv_parity` - the cache: the `int8` cell compared code by code and scale by scale with a host that
  applies kv_q8's rule; the attention over the dequantized cells at both head counts; a decode row **bit for bit**
  the prefill's row; a **permuted** page table at 8 cells per page that changes nothing; a decode row at
  position 1023, past the 512 the validation vehicle can hold; a **ring** on a sliding layer (window 8, 1152
  cells) lapped twice by 2048 positions, whose page table is read back and whose last row is still the host's; a
  row at position 0 whose cells that ring recycled **refused**; a pass of 2048 positions into the same ring
  **refused** with the ring's size in the message; and the 45-layer sizes at 8192, 32768 and 131072 - 0.315 /
  0.938 / 3.431 GB with the ring against 0.779 / 3.114 / 12.457 GB keeping every position.
