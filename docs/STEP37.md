# Step-3.7-Flash on Strata

Step models are StepFun's (`stepfun-ai`): **Step-3.5-Flash** is text-only and **Step-3.7-Flash** is the same text
decoder plus a vision encoder. Both are 196B parameters with 11B active, and both use the GGUF architecture
`step35`, so one engine path runs the two.

This engine path is **new and incomplete on purpose**. What runs today is the whole model over a prompt of several
positions, then decode steps that read the prompt's own K/V out of a quantized cache, and greedy generation from
that cache, with the experts read from the file. What does not run yet is listed at the end, and it is a real list.

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

`--max-new N` generates N tokens greedily after the prompt (and after any `--next` steps), each step's own top-1
being the next step's token, and prints the per-token time and a summary. The head is opened once and kept
resident - it is 363 MB of Q5_K, and re-uploading it per token would cost more than the pass.

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
| generation, 45 blocks, 16 tokens from a 6-token prompt | **1.063 s per token** (best 0.799, worst 1.461), 17.09 s for the sixteen; the first four average 0.895 |
| what it generates | `11111 16 455 6102 …` - the same four `--gpu-mtp` prints as the trunk's own continuation |

The residual difference is the Q8_1 quantization of the activations, which K-quant matvecs carry by design; it
is not a difference of arithmetic. llama.cpp's own GPU (fp16) and CPU (f32) runs differ by 1.5e-02 to 3.0e-02 on
the same file, so a few 1e-3 per block is inside its spread.

## What does not work yet

- **Speculation pays only at K=1.** The acceptance loop runs (`--gpu-spec`) and saves 16% with one draft token;
  two and three drafts cost more than they save, measured. See the section above.
- **Sampling only in `--max-new`, only greedy in `--gpu-spec`.** `--temperature`, `--top-k`, `--top-p`,
  `--repeat-penalty` and `--seed` work in the generation loop; the speculative one takes the top-1 because accepting
  a draft drawn from a distribution needs its own rejection rule, which is not written. `--gpu-spec` stops on the
  file's eos, `--max-new` runs to the count asked for. Serving samples, because it goes through the same loop.
- **No vision.** The `mmproj` projector (3.97 GB, llama.cpp's `PROJECTOR_TYPE_STEP3VL`) is not used.
- **The expert tier is flat.** `--expert-cache` keeps a FIFO ring of whole experts and nothing else: no routing
  policy, no per-layer budget, no cross-device promotion, and it is sized by the flag rather than by what is free.
- **No release ships it.** The launcher plans the family and writes its config, but `--start` installs the engine
  package from the update feed (0.1.55) and that release does not carry this fork's `--serve` for step35, so the
  start needs an engine built from this branch. The app and the web page do not know the family yet either.
- **Expert routing is host-side.** Every position of every block brings its eight expert ids back to the host, which
  copies the matrices: that round trip is what makes a CUDA graph impossible and what a device-side gather would
  remove. Measured, it is 1.97% of a token - the smallest of the three items this page has profiled, and not worth a
  device-side router until everything above it is gone.
- **No cache spill.** A K/V cache that outgrows VRAM has no host copy (`kv_stream.hpp` has one for the QSA layers);
  it matters above 131072 positions on a card that also wants the tier (6.7 GB at the model's 262144 maximum).
- **One request at a time.** `--serve` answers one GEN at a time and the engine is single-device: the server's
  slots and `--layer-split` are accepted and ignored, not implemented.

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

## The MTP draft heads

```sh
strata-step35 --gpu-mtp <first shard> --mtp <NextN file> --tokens "0,671,6102,294,8760,344" [--drafts N] [--layers N]
strata-step35 --ref      <first shard> --mtp <NextN file> --tokens "..."                  # the same, on the CPU
```

The trunk runs the prompt through its cache, then continues **greedily on its own** for as many tokens as there are
draft heads, then each NextN block is fed the token the trunk itself chose at that position and the previous head's
`h_nextn`. A draft is worth something only if it proposes what the trunk would have said, so every head's top-1 is
printed against the trunk's token next to it, and the run ends with the count.

The wiring is llama.cpp's `graph_mtp`, block for block: `rms_norm(token_embd[token], nextn.enorm)` concatenated
with `rms_norm(h, nextn.hnorm)` (the embedding first), back to n_embd through `nextn.eh_proj`, one **dense** decoder
block of the NextN file (96 query heads, 8 KV, window 512, FFN 11264, no clamp), then `nextn.shared_head_norm` and
`nextn.shared_head_head`. The NextN file carries its own embedding table and its own head, so a draft borrows
nothing from the trunk but the hidden state.

**One cell, and only its own.** A NextN layer runs once per chain: the trunk never computed K/V for it, and no
earlier draft went through it either, so its attention has exactly the cell this step writes. Running it with the
window of 512 it declares is wrong - it reads cells nothing ever wrote, and the first attempt answered `NaN`. What
a draft knows comes from `h_nextn` and the token.

Measured on three prompts (the same six, eight and ten token sequences, IQ3_XXS trunk and the Q8_0 NextN file,
45 blocks). The bold token is the head's top-1, and ✓ means it is the token the trunk chose next:

| prompt | the trunk continues | block 45 | block 46 | block 47 |
|---|---|---|---|---|
| `0,671,6102,294,8760,344` | `11111 16 455 6102` | **16** ✓ | 128007 ✗ (455 third) | **6102** ✓ |
| `… 671,6102` | `294 8760 344 671` | **8760** ✓ | 294 ✗ (344 second) | 270 ✗ |
| `… 671,6102,294,8760` | `344 671 6102 294` | **671** ✓ | 11111 ✗ | 4593 ✗ (294 second) |

The first head proposes the trunk's own next token **3 times out of 3**; the two chained heads **1 time out of 6**.
That is the shape to expect: block 45 is handed the trunk's hidden state, blocks 46 and 47 continue from a hidden
state their predecessor produced. Three prompts are not a benchmark - this says what the wiring does, not what a
draft is worth, and the worth needs the acceptance loop that does not exist yet.

Against the CPU reference the two paths answer the **same top-1 on all three heads**, with `h_nextn` rms within
0.3% (2.5716 against 2.5731, 3.5184 against 3.5215, 3.3113 against 3.3220) and the third to fifth places swapping
on near-ties - a Q8_0 file against a float32 host, which is the size of the difference the page quotes everywhere
else.

## Which expert a full tier replaces

`--tier-policy fifo|lru|lfu`, default **lru**. The tier evicts an expert when it fills; FIFO walks its ring, LRU takes
the one fetched longest ago, LFU the one fetched least often. The routing is skewed, so the choice is worth a lot:
same VRAM, same code, 24 tokens, prompt `0,671,6102,294,8760,344`.

| tier | fifo | **lru** | lfu |
|---|---|---|---|
| 4 GB | 0.524 s/token, 37.2% of fetches served | **0.329 s/token, 41.7%** | 0.340 s/token, 39.2% |
| 8 GB | 0.262 s/token, 57.6% | **0.220 s/token, 67.6%** | 0.238 s/token, 63.4% |

LRU is 37% faster than FIFO at 4 GB and 16% at 8 GB, on the same hit-rate arithmetic - which is why it is the default
and not an option. The choice is a linear scan over the slots on a miss, nothing beside the 5.9 MB copy it decides.

The ladder it produces, all at 24 tokens:

| flags | s/token | against the default |
|---|---|---|
| none | 0.798 | - |
| `--dense-cache 2600` | 0.483 | -39% |
| `+ --expert-cache 4000` | 0.330 | -59% |
| `+ --expert-cache 8000` | 0.220 | -72% |
| `+ --expert-cache 12000` | **0.200** | **-75%** |
| `+ --expert-cache 10000 --gpu-spec --drafts 1` | 0.253 | -68% |

## Speculative decoding, and why K=1

```sh
strata-step35 --gpu-spec <first shard> --mtp <NextN file> --tokens "..." --max-new N [--drafts K]
```

One block at a time: the NextN heads draft K tokens, **one** trunk pass over the K+1 positions
(`n0, d1 .. dK`) checks them, and the longest prefix the trunk agrees with is kept - the trunk's own token at the
first rejection is the correction, and it opens the next block. The emitted tokens are **exactly greedy's**; only
the time changes. The NextN weights are resident (their three blocks and three heads are 3.7 GB) because a draft
that re-uploads them costs four trunk steps; with them open a draft is **4 ms**.

Measured on this machine, 45 blocks, the prompt `0,671,6102,294,8760,344`, eight tokens, two rounds each:

| | s/token | against greedy | drafts accepted |
|---|---|---|---|
| greedy (`--max-new`), round 1 / 2 | 0.811 / 0.809 | - | - |
| `--drafts 1` | **0.680 / 0.679** | **-16%** | 4 of 5 (80%) |
| `--drafts 2` | 1.029 | +27% | 4 of 10 (40%) |
| `--drafts 3` | 1.277 | +57% | 4 of 15 (27%) |

All four runs emit the same eight tokens, `11111 16 455 6102 294 10322 344 17575`. So speculation here is a
**16% saving with one draft token and a loss beyond that**, and the arithmetic says why rather than the numbers
merely showing it: a pass costs its position count (5.086 s for 10 positions, 0.509 s each) because every position
re-reads the experts it routes to, while a block emits `1 + accepted` tokens for `K + 1` position-visits. K=1 pays
when the first head is right more than about 13% of the time - it was right 80% here. K=3 would need the chained
heads to carry their own weight, and they do not: block 45 answers 3 of 3, blocks 46 and 47 about 1 in 6 (see the
table above), so the extra two positions per block buy a quarter of a token.

Two limits, stated plainly: this is eight tokens on one prompt, so the 16% is the shape of the result and not a
benchmark; and the saving is bounded by the trunk's per-position cost, which is the expert streaming, not the
batch overhead. A VRAM expert cache would move that bound; nothing in the draft heads can.

## The VRAM expert tier

```sh
strata-step35 --gpu-chat <first shard> --tokens "..." --max-new N --expert-cache MiB
strata-step35 --gpu-spec <first shard> --mtp <file> --tokens "..." --max-new N --expert-cache MiB
```

Every position copies the eight experts it routes to out of the file, for each of the 42 mixture-of-experts blocks:
~2.2 GB of movement per position, and that is the whole cost of a token. `--expert-cache` keeps as many of those
experts as the MiB allow in VRAM (`src/step35/step35_experts.cu`: a slot ring, one expert's gate/up/down per slot,
and a residency table `(block, expert) -> slot`), so a position reads them where they are. Eviction is FIFO over
the ring: with 45 blocks between two visits to the same block, an exact recency order costs more than it saves.

A hit and a miss compute the same bits - the tier moves bytes, never numbers - and the measured renders are
identical to the digit with the tier on and off (position 9: rms 14.219030, first four -5.773189 -4.102162 -5.326063
-8.670106 either way).

Measured on this machine (2 x 16 GB, IQ3_XXS: 5880 KB per expert, 45 blocks):

| tier | generation, 9 tokens | hit rate | vs no tier |
|---|---|---|---|
| off | 0.811 s/token | - | - |
| 4 GB | 0.684 s/token | 33.1% | -16% |
| 8 GB | 0.605 s/token | 46.9% | -25% |
| 12 GB | **0.559 s/token** | 55.0% | **-31%** |

and on a 10-position pass, 0.509 s per position without the tier against **0.356** with 8 GB (44.3% of the fetches
served). Combined with one draft token (`--gpu-spec --drafts 1 --expert-cache 12000`) the same nine tokens come out
in **0.407 s/token**: **-50% against greedy without the tier**, same tokens, 54.6% of fetches served.

The size is the whole of what it buys: 12.58 GB of a 17.16 GB card leaves room for a 0.3 GB cache at 8192 positions
and the working buffers, and a request larger than the card holds stops early and says so
(`out of device memory after 2723 expert slot(s)`), after which the session that needs the rest fails with a message
that names the tier. Nothing about the policy is learned: no routing hint, no per-layer sizing, no promotion across
two cards - the engine's other families carry all three (`include/strata/core/expert_cache.hpp`) and none of it is
wired here yet.

## Sampling

```sh
strata-step35 --gpu-chat <shard> --tokens "..." --max-new N --temperature 0.8 --top-p 0.9 --seed 1
strata-step35 --gpu-chat <shard> --tokens "..." --max-new N --top-k 40 --repeat-penalty 1.1
```

`--temperature` defaults to 0, which is the argmax and ignores every other knob - so the default path is the one it
was, and the sequence a command emitted before this exists is the sequence it emits now. The filters run in
llama.cpp's order, which matters: repetition penalty over the last 64 tokens (a positive logit divided, a negative
one multiplied, so the penalty always pushes a repeat down), then temperature, then top-k, then the nucleus, then
the draw. A penalty applied after the filters would push mass back into tokens the filters had just removed.

The arithmetic is host code (`src/step35/step35_sampler.cpp`) over the logits that were already copied back: a sort
of 128 896 floats is about 10 ms, a little over 1% of a token here, and it keeps the sampler divisible and testable
without a card.

`step35_sample_test` (ctest, no GPU) holds it to arithmetic rather than to a mood: greedy answers the argmax even
with `top_k`, `top_p` and a penalty set; `top_k 1` and a 100-logit lead are never lost to the draw; temperature 1 is
the softmax to 0.0e+00; **60 000 draws** of a three-way distribution land within 1.7e-03 of it; `top_p 0.5` over
logits `{1,3,2}` never draws the tail and renormalises to 1.000000; a penalty of 2 halves a repeated token
(0.6652 -> 0.3072) and multiplies a negative one (0.0179 -> 0.0066); one seed gives one stream and another gives
another.

On the release, prompt `0,671,6102,294,8760,344`, eight tokens: greedy gives `11111 16 455 6102 294 10322 344 17575`;
`--temperature 0.8 --top-p 0.9 --seed 1` gives `11111 603 671 6102 294 10322 344 17575`, the same command replayed
gives it again, and `--seed 2` gives `11111 16 1884 850 1951 377 8760 14`. That is the sampler working and the
stream reproducible, not a quality claim: eight tokens say nothing about whether those tokens are good.

## Serving: the engine protocol, and what is still missing

```sh
strata-step35 --serve --model <first shard> [--max-context N] [--expert-cache MiB]
```

`--serve` makes the engine the resident process `serve/server.py` expects: weights, K/V cache, head and expert tier
load once, and requests arrive on stdin one line at a time. The protocol is the server's own, spoken and nothing
else on stdout:

| the engine prints | when |
|---|---|
| `READY <ctx> stop` | once, when the model is loaded; `stop` says it honours a STOP mid-request |
| `RESUME <n>` | per request: how many of its prompt tokens were already cached |
| `PP <read> <total> <ms>` | after each prefill chunk of up to 1024 positions |
| `T <id>` | one per generated token, flushed |
| `DONE <generated> <prompt_tokens> <prompt_ms> <decode_ms> <finish>` | the request is finished; `finish` is `length`, `stop` or `cancel` - the five fields are positional and a bare `DONE` raises in the server, leaving the request waiting for ever (measured) |
| `ERR <message>` | it could not be; the cache is dropped and the next request starts from zero |

and on stdin: `GEN <n> [key=value ...] <id,id,...>` with the server's key names (`temperature`, `top_p`, `top_k`,
`penalty_repeat`, `seed`; any other key it may send is accepted and ignored rather than failing the request),
`STOP` to end a request early, `QUIT` to exit. A request whose prompt extends the previous one decodes only the
new suffix; when the prompt is entirely cached the last position is recomputed, because the hidden state in hand
belongs to the last token *generated*, not to the prompt's.

Checked by piping the protocol in (no HTTP, no server):

```sh
printf 'GEN 4 0,671,6102,294,8760,344\nGEN 3 0,671,6102,294,8760,344,11111\nQUIT\n' |
  strata-step35 --serve --model <shard>
# READY 4096 stop / RESUME 0 / PP 6 6 2944 / T 11111 16 455 6102 / DONE
#                  RESUME 6 / PP 7 7 441  / T 16 455 6102 / DONE     <- one position read, not seven
```

and `GEN 40 ...` followed by `STOP` answers five tokens then `DONE`.

**It works end to end against `serve/server.py`.** With a config written by hand - `exe` = `strata`, `args` =
`--model <first shard> --max-context 8192 --expert-cache 4000`, `tokenizer` = the model folder's `tokenizer/`,
`arch` = `step35` - the server reaches `loaded: true` in 3 s and answers OpenAI-shaped requests:

```
POST /v1/chat/completions  {"messages":[{"role":"user","content":"Hello"}],"max_tokens":16}
-> HTTP 200 in 15.6 s, 16 completion tokens, finish "length"
   reasoning_content: "Got it, let's see. The user just said \"Hello\". First I ..."
```

twice, the same text both times (greedy), the second in 10.9 s. The `content` is empty because this model answers in
its reasoning block first, which the server routes to `reasoning_content` - that is the server's parsing, not a
fault in this engine.

Until the launcher knows the family, that config is written by hand or by `strata_cli/launch/adopt.py`, which now
has a Step-3.7 branch for files already on disk (`--model-path`): two shards, the folder's `tokenizer/`, the engine
reading the GGUF directly, and an expert tier sized from what the card has left after the K/V cache - the cache's
figures are the measured ones of this page, reproduced by `adopt.step35_kv_gb` and checked in
`tests/terminal/test_adopt_step35.py`.

## The launcher: what it does now, and the one thing left

`strata launch --family step35 --model-path <folder> --plan` works, and says true things:

```
  model     step35 IQ3_XXS   (your files in .../step-3.7-flash-IQ3_XXS/IQ3_XXS)
  context   64K    pictures off    parallel fit to your card
  download  Nothing to download: your model files are used where they are. ...
  drafting  off (Step-3.7-Flash's NextN blocks are not used by the serving path yet)
```

The family has a catalogue entry (`setup.py` `OWN_PATH_FAMILIES["step35"]`), the model table reports it as an engine
that *streams* its experts (no VRAM or RAM arena, whatever the size of the file), and the launch writes its run
config from the files on disk (`adopt.py`), with the expert tier sized out of the card and the K/V cache. That config,
pointed at an engine that carries this family, is what serves: `serve/server.py` reached `loaded` in 3 s and answered
`HTTP 200 in 4.4 s` with the twelve tokens asked for.

**The one thing left is the engine binary, not the launcher.** `strata launch --start` resolves
`component:strata-engine` to the engine package the update feed installs - 0.1.55 here - and that release does not
contain this fork's `--serve` for step35, so the start failed with the engine's own usage text. On this machine the
family is served by pointing the config's `exe` at the build that has it (`build-hip/strata`); for anyone else it is
served once a release carries the family. Nothing else in the launch path is missing.

The three remaining catalogue questions, for whoever picks this up:

1. The file list is pinned (title, author, repository, pattern, tag, licence, and both shards with the Hub API's own
   sizes and sha256: 46 645 813 024 / `ce04c744…cc5d` and 29 112 867 936 / `aeedcdaa…e31d`, both matching the copies
   here), at the revision `0b69336d…c31482` (2026-06-03). Only IQ3_XXS is offered: the repository also carries
   IQ4_XS, Q3_K_L, Q3_K_M, Q4_K_S, Q8_0 and BF16, and none of them has been run here.
2. The fit words for this family are the ones `own_size` reports (`streams_experts`: no arena), which is a claim
   about *memory*, not about speed: the launch list will say a card can run it, and the speed the tier buys is not in
   the list. The measured line for this machine is 0.68 s per token with one draft token and 12.6 GB of tier.
3. `strata launch` offers `IQ3_XXS` for this family and for Qwen3.8-Flash-Next: quant names are not unique across
   families, so anything keyed by model name alone has to carry the family too (one test did, and said so).

## The blocks' dense weights, resident: the larger half of a token

`--dense-cache MiB` keeps each block's own attention and feed-forward tensors in VRAM instead of uploading and freeing
them once per pass per block. It is the biggest single win measured on this engine, and the profile is what found it:
with the expert tier holding the experts, **66% of a token was the dense weights being re-read**, not the experts.

| flags | s/token | against the default |
|---|---|---|
| none | 0.818 | - |
| `--dense-cache 2600` | **0.503** | **-39%** |
| `--dense-cache 2600 --expert-cache 10000` | **0.264** | **-68%** |
| the same with `--gpu-spec --drafts 1` | **0.258** | **-68%**, 4 of 5 drafts accepted |

All 45 blocks fit: 1.88 GB, `45 block(s) resident, 0 streamed` (the budget is only a ceiling; a block that does not
fit is streamed exactly as before). The tokens are identical in every row - `11111 16 455 6102 294 10322 ...` - so
this is a place things live, not arithmetic.

And speculation no longer pays at all: 0.253 s per token with one draft token against **0.200** for plain greedy with
the same caches. The 16% it measured before the caches existed came from the pass overhead the dense cache removes,
and once that is gone a verify pass over two positions costs more than the token it saves.

## What a CUDA graph would be worth here, measured

The page used to say the pass is not a graph "because the position crosses as a 4-byte H2D copy". That is a reason it
cannot be captured, and no reason at all that capturing it would be faster. `STRATA_STEP35_PROFILE=1` times the four
kinds of copy a token makes, per layer and per position (six tokens, prompt `0,671,6102,294,8760,344`, 45 blocks):

| what | calls | time | of the wall |
|---|---|---|---|
| the position, 4 bytes H2D (per block per pass) | 315 | 7.8 ms | **0.22%** |
| the router's ids and weights, D2H (per block per position) | 504 | 69.3 ms | **1.97%** |
| a pass's positions, H2D (per block per pass) | 315 | 32.1 ms | 0.91% |
| the three matrices of one expert, H2D - with the tier off | 4032 | 5141.6 ms | 145% of 3.5 s... and 103% of the 4.97 s it took |

Read the last line carefully: the expert copies are synchronous, so their time includes waiting for the kernels in
front of them, and the sum can exceed the wall clock. What it says is not a precise figure but a direction: without
the tier those copies *are* the token - with 12 GB of tier there are none left and a token goes from 0.817 s to
0.577 s.

So everything a graph could remove - the position, the router's round trip, the position array - is **2.2% to 3.1%**
of a token, and the thing that costs is the expert streaming, which the tier already removes. A captured graph is
also unreachable today for a reason that is not the 4-byte copy: **the router's ids come back to the host for every
position of every block**, and the host chooses the eight experts to copy. Capture needs that decision on the device,
which is the same work a device-side expert gather would need - and that, not the graph, is what would move the
number.

The other half of the item, streaming the K/V cache to the host, is bounded by the same arithmetic the page already
gives: 12 full blocks at 2112 bytes per position plus a 0.11 GB ring, so 3.43 GB at 131072 and 6.7 GB at the model's
maximum 262144. An 8 GB card holds the cache at 131072 without a tier; the spill would matter above that, or on a
small card that wants the tier instead. Nothing is built for it, and nothing measured here would be improved by it.

## Tests

```sh
ctest -R step35
```

Seven tests, none of which needs the 76 GB download:

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
- `step35_sample_test` - the sampler, against arithmetic and its own statistics (see above).
- `step35_kv_parity` - the cache: the `int8` cell compared code by code and scale by scale with a host that
  applies kv_q8's rule; the attention over the dequantized cells at both head counts; a decode row **bit for bit**
  the prefill's row; a **permuted** page table at 8 cells per page that changes nothing; a decode row at
  position 1023, past the 512 the validation vehicle can hold; a **ring** on a sliding layer (window 8, 1152
  cells) lapped twice by 2048 positions, whose page table is read back and whose last row is still the host's; a
  row at position 0 whose cells that ring recycled **refused**; a pass of 2048 positions into the same ring
  **refused** with the ring's size in the message; and the 45-layer sizes at 8192, 32768 and 131072 - 0.315 /
  0.938 / 3.431 GB with the ring against 0.779 / 3.114 / 12.457 GB keeping every position.
