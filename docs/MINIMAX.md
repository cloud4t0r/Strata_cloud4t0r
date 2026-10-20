# MiniMax-M3: what it would take to run it in Strata

A study, not a plan. It says what MiniMax-M3 is, which parts of Strata already fit it, which parts do not,
and what the memory on a normal PC looks like. Every number here says where it came from.

**Verdict in one line: the engine work is a medium rewrite (a new family, a new sparse attention, a new
activation, a new router, a new tokenizer, and a float path for the indexer), but the memory decides
everything else - the smallest MiniMax-M3 file is 90.5 GB and the model is 4x heavier per token than
Qwen3.8-Flash-Next. MiniMax-M3 is not a "gaming PC" model; it is a workstation model that happens to have a
GGUF, and it would need a 128 GB RAM machine to be usable at all.**

## What MiniMax-M3 is

Facts read from the model card, the config file and llama.cpp's own implementation at the commit Strata pins.

| | Value | Source |
| --- | --- | --- |
| Total / active parameters | 428B / 23B | the model card; my own count from the config gives 426.1B / 24.7B |
| Layers | 60 | `config.json` `text_config.num_hidden_layers` |
| Hidden size | 6144 | `text_config.hidden_size` |
| Attention | 64 heads, 4 KV heads, head_dim 128 | `num_attention_heads`, `num_key_value_heads`, `head_dim` |
| RoPE | theta 5,000,000, **64 of 128 dims rotated** | `rope_theta`, `rotary_dim` (partial rotary 0.5) |
| Experts | 128 routed, top-4, 1 shared expert, FFN width 3072 | `num_local_experts`, `num_experts_per_tok`, `shared_intermediate_size` |
| Leading dense layers | 3 (dense FFN, width 12288, and dense attention) | `moe_layer_freq` starts with three 0s; `sparse_attention_freq` too |
| Activation | `swigluoai`: alpha 1.702, limit 7.0 | `swiglu_alpha`, `swiglu_limit` |
| Routing | sigmoid gate, a per-expert bias, weights normalised, then scaled by 2.0 | `scoring_func`, `use_routing_bias`, `routed_scaling_factor`; llama.cpp's `build_moe_ffn` |
| Norms | Gemma-style `(1 + w)` RMSNorm | `use_gemma_norm`; the converter bakes the `+1` into the weights |
| Attention type | **MiniMax Sparse Attention (MSA)** on layers 3-59, dense on 0-2 | `sparse_attention_config` |
| MSA | blocks of 128 positions, top 16 blocks, 4 index heads of 128 dims, 1 forced local block | `sparse_block_size`, `sparse_topk_blocks`, `sparse_num_index_heads`, `sparse_index_dim`, `sparse_local_block` |
| Context | 1,048,576 | `max_position_embeddings` |
| Pictures | a ViT (32 layers, 1280 wide, patch 14, 2x2 spatial merge) + a patch-merge MLP | `vision_config` |
| Vocabulary | 200,064 tokens, its own BPE (not Qwen's) | `tokenizer.json` |
| Thinking | `<​mm:think>` ... `</​mm:think>`, three modes (enabled / adaptive / disabled) | the chat template |
| MTP (drafting) | **not in the released weights** | llama.cpp's `minimax-3.cpp`: "MTP is not in released model weights" |

The GGUF architecture string is **`minimax-m3`** (a hyphen). The separate string `"minimax_m3"` with an
underscore is the *vision* projector type, in the mmproj file.

### The GGUF file, exactly

Read out of `unsloth/MiniMax-M3-GGUF`'s UD-IQ1_M shard and of Bartowski's Q4_K_M shard. This is the list
`check_architecture_minimax_m3` would have to require:

```
minimax-m3.block_count                       = 60
minimax-m3.context_length                    = 1048576
minimax-m3.embedding_length                  = 6144
minimax-m3.feed_forward_length               = 12288      (the dense MLP of layers 0-2)
minimax-m3.expert_feed_forward_length        = 3072
minimax-m3.expert_count                      = 128
minimax-m3.expert_used_count                 = 4
minimax-m3.expert_shared_count               = 1
minimax-m3.expert_weights_scale              = 2.0
minimax-m3.expert_weights_norm               = true
minimax-m3.expert_gating_func                = 2          (sigmoid)
minimax-m3.leading_dense_block_count         = 3
minimax-m3.attention.head_count              = 64
minimax-m3.attention.head_count_kv           = 4
minimax-m3.attention.key_length              = 128
minimax-m3.attention.value_length            = 128
minimax-m3.attention.layer_norm_rms_epsilon  = 1e-06
minimax-m3.attention.indexer.head_count      = 4
minimax-m3.attention.indexer.key_length      = 128
minimax-m3.attention.indexer.top_k           = 16
minimax-m3.attention.indexer.block_size      = 128
minimax-m3.attention.indexer.local_blocks    = 1
minimax-m3.rope.dimension_count              = 64
minimax-m3.rope.freq_base                    = 5000000.0
tokenizer.ggml.model                         = "gpt2"     (pre = "minimax-m2")
tokenizer.ggml.bos_token_id = 200019  eos = 200020  padding = 200000  unknown = 16474
```

And the tensors:

```
token_embd.weight [6144,200064]     output.weight [6144,200064]    output_norm.weight [6144]
blk.{i}.attn_norm.weight            blk.{i}.attn_q.weight [6144,8192]     (64 heads x 128)
blk.{i}.attn_k.weight [6144,512]    blk.{i}.attn_v.weight [6144,512]
blk.{i}.attn_q_norm.weight [128]    blk.{i}.attn_k_norm.weight [128]
blk.{i}.attn_output.weight [8192,6144]
blk.{i}.ffn_norm.weight             blk.{i}.ffn_gate.weight / ffn_up.weight / ffn_down.weight   (layers 0-2)
blk.{i}.ffn_gate_inp.weight [6144,128]         blk.{i}.exp_probs_b.bias [128]
blk.{i}.ffn_gate_exps.weight [6144,3072,128]   blk.{i}.ffn_down_exps.weight [3072,6144,128]
blk.{i}.ffn_up_exps.weight [6144,3072,128]
blk.{i}.ffn_gate_shexp.weight [6144,3072]      blk.{i}.ffn_down_shexp.weight [3072,6144]
blk.{i}.ffn_up_shexp.weight [6144,3072]
blk.{i}.indexer.q_proj.weight [6144,512]        blk.{i}.indexer.k_proj.weight [6144,128]
blk.{i}.indexer.q_norm.weight [128]            blk.{i}.indexer.k_norm.weight [128]   (layers 3-59 only)
```

**The MoE tensor names are exactly the ones Strata's layout table already uses.**
`core::family_layout.hpp` expects `ffn_gate_exps.weight {N, FF, NE}`, `ffn_up_exps.weight {N, FF, NE}`,
`ffn_down_exps.weight {FF, N, NE}`, `ffn_gate_inp.weight {N, NE}`, `ffn_gate_shexp.weight {N, FS}`,
`ffn_up_shexp.weight {N, FS}`, `ffn_down_shexp.weight {FS, N}` - and llama.cpp writes MiniMax-M3 with those
names and those shapes. The router bias is the one exception: it is `exp_probs_b.bias`, not
`ffn_exp_probs_b.bias`.

**The indexer tensors are F32.** The converter forces every tensor whose name contains `.indexer.` to
F32. Strata's expert and dense paths assume quantised weights; the indexer branch would need a
float path, or the indexer would need to be re-quantised (which llama.cpp's PR explicitly warns against).

The published GGUFs are Unsloth's (`unsloth/MiniMax-M3-GGUF`, UD-IQ1_M to UD-Q8_K_XL), Bartowski's (the
smallest, with an mmproj), NVIDIA's NVFP4 checkpoint, AMD's MXFP4, and MiniMax's own MXFP8.

## The good news: llama.cpp already has it, at the commit Strata pins

Strata builds ggml from llama.cpp commit `3cf03257f219afbe7334045ff7c6a06ac68c627d` (2026-09-20, the
`CMakeLists.txt` `FetchContent_Declare(strata_llamacpp ...)` pin). **That tree already contains MiniMax-M3:**

| File | What it gives |
| --- | --- |
| `conversion/minimax.py` | the HF to GGUF converter (`MiniMaxM3Model`, `MiniMaxM3VisionModel`) |
| `src/models/minimax-3.cpp` | the model's forward pass, 603 lines, MSA included |
| `src/llama-kv-cache-msa.cpp` | a KV cache of its own for MSA (12.5 KB) |
| `tools/mtmd/models/minimax-m3.cpp` | the vision projector (`minimax_m3` clip arch, patch-merge MLP) |
| `models/templates/MiniMax-M3.jinja` | the chat template |
| `common/parsers/minimax-m3.cpp` | the reasoning and tool-call parser |

The relevant upstream PRs:

| PR | | State |
| --- | --- | --- |
| [#24908](https://github.com/ggml-org/llama.cpp/pull/24908) | Add MiniMax-M3 (MSA) support | merged 2026-07-26 (`b1d4c655`) |
| [#25113](https://github.com/ggml-org/llama.cpp/pull/25113) | mtmd: vision support for MiniMax-M3 | merged 2026-07-26 |
| #26210 | common/chat: MiniMax-M3 parser | merged |
| #26180 | quantised KV cache for MiniMax-M3 | merged |
| [#29553](https://github.com/ggml-org/llama.cpp/pull/29553) | MiniMax-M3: MSA sparse attention kernel | **closed, not merged** (2026-09-28) |

Three consequences:

1. **The reference exists.** Strata's rule is that a new model is checked against llama.cpp step by step
   (KL on the same token ids). For MiniMax-M3 there is something to check against on day one.
2. **Pictures come almost free.** `strata-vision` drives llama.cpp's `mtmd`, so it will pick up the
   `minimax_m3` projector on its own. A projector file exists: `bartowski/MiniMax-M3-GGUF` ships
   `mmproj-MiniMax-M3-bf16.gguf` (1.74 GB).
3. **The chat template is not ours to write.** `serve/server.py` reads `tokenizer/chat_template.jinja` from
   the model's own pack, so MiniMax-M3's template (with its thinking modes) arrives with the file.

**What the merged llama.cpp PR warns about, and Strata inherits:**

- **Flash attention is required.** Without it, llama.cpp falls back to dense attention and says the output
  "may be degraded": MSA is trained in, not an optimisation. So Strata cannot offer a CPU or a
  no-flash-attention path for MiniMax-M3 and call it the same model.
- **The indexer weights must stay F32.** (See the note on the tensor names.)
- **Block selection is anchored to absolute KV slots.** The block a query picks is the block at that
  absolute position; a cache whose slots are not the positions diverges.
- **The fused sparse kernel was never merged** (#29553, closed). At Strata's pin, llama.cpp's MSA is a
  gather at decode and a dense mask at prefill, not a dedicated sparse kernel. Strata's implementation
  would be chasing the same target, so the reference is not a fast implementation to copy.

## How Strata is built for a family, and where MiniMax-M3 lands in it

Strata's engine runs one family per build path. `include/strata/artifact/model_family.hpp` holds the list
(`qwen4exp` for Qwen3.8-Flash-Next, `qwen35moe` for Qwen3.6-35B-A3B). `src/program/generate.cpp` reads the
file's `general.architecture` at start-up and switches to the other path when it must (the `qwen35` path,
`src/qwen35/`, for Qwen3.6).

The shape a model must have is **compiled into the kernels**, not read at run time.
`include/strata/core/geometry.hpp` holds the shape; `core::geometry_unsupported` refuses a file whose
metadata says anything else, and `core::compiled_geometry_mismatch` names the key that differs. So a new
family is a new compiled shape, not a new set of options.

Side by side:

| | Qwen3.8-Flash-Next (`qwen4exp`) | Qwen3.6-35B-A3B (`qwen35moe`) | **MiniMax-M3 (`minimax-m3`)** |
| | the compiled geometry | `config.json` of `Qwen/Qwen3.6-35B-A3B` | `config.json` of `MiniMaxAI/MiniMax-M3` |
| --- | --- | --- | --- |
| Engine path | the main one | `src/qwen35/` | **new** |
| Layers / hidden | 48 / 2560 | 40 / 2048 | **60 / 6144** |
| Attention layers | 12 full + 36 GDN | 10 full + 30 GDN | **60 (3 dense + 57 sparse), no GDN** |
| Heads / KV heads | 24 / 2 | 16 / 2 | **64 / 4** |
| Head dim | 256 | 256 | **128** |
| Rotated dims | 64 | 64 | **64 (same)** |
| Attention output gate | yes (`attn_q` carries it) | yes | **no** |
| Experts / top-k | 512 / 10 | 256 / 8 | **128 / 4** |
| FFN width | 640 | 512 | **3072** |
| Residual | gated (`hc_` tensors) | plain pre-norm | **plain pre-norm (like Qwen3.6)** |
| Sparse attention | a DeepSeek-style indexer on the 12 full layers | none | **MSA on 57 layers** |
| Activation | SwiGLU | SwiGLU | **`swigluoai`** |
| Router | softmax, top-k | softmax, top-k | **sigmoid + bias + normalise + scale** |
| Norm | plain RMSNorm | plain RMSNorm | **Gemma `(1+w)` (baked into the weights)** |
| Tokenizer | Qwen (`qwen35` pre-tokenizer) | Qwen | **its own** |
| MTP drafting | yes | yes | **not in the weights** |

## What Strata already has that MiniMax-M3 can use

Read from the code, not assumed:

- **Partial RoPE.** `include/strata/core/geometry.hpp` already has `n_rot = 64` against a head dim of 256:
  Strata rotates a slice of the head. MiniMax-M3 rotates 64 of 128. Same mechanism.
- **The indexer dimensions are identical.** Strata's QSA indexer is `idx_q_heads = 4` heads of
  `idx_key_dim = 128` (`geometry.hpp`). MiniMax-M3's indexer is 4 heads of 128. The projections, the norms
  and the RoPE on the index branch are the same shapes.
- **The QSA indexer pipeline.** `include/strata/kernels/qsa.hpp` documents the whole chain: indexer scores,
  top-k selection, gather through the page table, GQA attention over the scratch. MSA is the same chain with
  a different granularity (blocks of 128 positions instead of pooled cells of 4) and a different pooling
  (max instead of weighted RMSNorm). The selection, the gather and the attention over a sparse set are
  reusable; the scoring and the cache geometry are not.
- **A paged KV cache with a cell/position map.** Strata already maps positions to cache cells
  (`include/strata/core/pinned.hpp`, the `QsaIndexerBuffers`). MSA needs the same map plus a block
  anchoring; llama.cpp wrote a separate `llama-kv-cache-msa.cpp` for it, which is the sign of the work.
- **Gemma norms cost nothing.** The converter adds the `+1` into the weight tensors, so the kernels keep
  their plain RMSNorm.
- **The router's shape.** `include/strata/kernels/native_router.hpp` is pinned to 512 experts, top-10,
  softmax. The mechanism (one product, one top-k, one grouped expert pass) is the same shape; the gating
  function is not.

## The memory: the part that decides whether this is worth doing

Download sizes, read from the Hub's file lists (sum of the shard sizes):

| File | Download | | File | Download |
| --- | ---: | --- | --- | ---: |
| bartowski IQ1_S | **90.5 GB** | | unsloth UD-IQ3_XXS | 160.3 GB |
| bartowski IQ2_XXS | 116.6 GB | | unsloth UD-IQ4_XS | 208.5 GB |
| bartowski IQ2_XS | 129.5 GB | | unsloth UD-Q4_K_XL | 265.8 GB |
| unsloth UD-IQ1_M | 129.3 GB | | unsloth Q8_0 | 453.6 GB |
| unsloth UD-IQ2_XXS | 134.9 GB | | unsloth BF16 | 852.9 GB |
| unsloth UD-Q2_K_XL | 143.9 GB | | | |

For scale: Strata's smallest model today is Q2_0 at **66.4 GB** and its largest is Q8_0 at **188.2 GB**.
So the smallest MiniMax-M3 in existence (bartowski's IQ1_S, 90.5 GB, about 1.7 bits) sits between the two -
it is not out of reach for the kind of machine Strata already serves, but it is not a gaming-PC model either.

Qwen3.8-Flash-Next is 125.3B by my count from Strata's own geometry (the model card says 125B);
MiniMax-M3 is 426.1B by my count from its config (the card says 428B). That is 3.4x.

And it is not only the disk. Per token:

| | Qwen3.8-Flash-Next | MiniMax-M3 | Ratio |
| --- | ---: | ---: | ---: |
| Active parameters | 6.25B | 24.69B | **3.95x** |
| Routed experts read per token | 480 of 24,576 | 228 of 7,296 | |

Counted from Strata's compiled geometry and from MiniMax-M3's config (top-10 of 512 over 48 layers; top-4
of 128 over 57 MoE layers). The model card's own figure for MiniMax-M3 is 23B active, which matches the
24.7B this count gives.

What that means for a normal PC, and this is an estimate, not a measurement:

- The routed experts alone are **119 GB at IQ2_XXS** and **87 GB at IQ1_S** (413.1B parameters at about
  2.3 and 1.7 bits). A 32 GB PC cannot hold them; neither can a 64 GB one. Strata's own largest model
  needs 136 GB of RAM.
- Strata's answer to that is the low-RAM mode: the experts live in one file on the SSD and the OS page
  cache serves them. The limit there is not the SSD but the **PCIe link to the GPU**, because each token
  needs a different slice of the experts.
- **The bytes read per token are the number that decides the speed.** For Qwen3.8-Flash-Next at Q2_0
  (2 bits) that is 480 experts x 1.23 MB = **590 MB per token**. For MiniMax-M3 at IQ2_XXS (2.3 bits) it
  is 228 experts x 16.3 MB = **3.7 GB per token**, **6.3 times more**; at bartowski's IQ1_S (1.7 bits) it
  is 228 experts x 12.0 MB = **2.7 GB per token**, 4.6 times more.
- Strata measures Qwen3.8-Flash-Next Q2_0 at **87.3 tokens/s at 1K** and **60.3 tokens/s at 262K** on an
  RTX 5070 12 GB / Ryzen 5 7600 / 64 GB box, engine 0.1.26, MTP speculative decoding on
  (`docs/DETAILS.md`, "Speed (measured)"). That box streams its experts over PCIe, so the decode speed
  there is close to bandwidth-bound.
- Scaling the measured number by the ratio of expert bytes per token puts MiniMax-M3 at IQ2_XXS in the
  **neighbourhood of 14 tokens/s** on that same box, about 19 tokens/s at IQ1_S, and around 5 tokens/s at
  UD-IQ4_XS. **These are estimates from the byte counts, not measurements.**
- **The KV cache is heavier too.** Every one of MiniMax-M3's 60 layers keeps a KV cache (there are no
  recurrent layers to make it O(1)), 4 KV heads of 128: **about 60 KB per context token at 8 bits**, against
  13.7 KB for Qwen3.8-Flash-Next (`docs/DETAILS.md`). That is 4.4 times more. At 128K it is 7.9 GB, at
  262K it is 16.1 GB, at 1M it is 62.9 GB - on top of the experts.

So MiniMax-M3 on Strata is a **128 GB RAM workstation** at IQ1_S (Strata's own Q8_0 asks for 136 GB of RAM,
so the machine class exists in Strata's catalogue), decoding at roughly a quarter to a third of the speed of
the model Strata ships today, unless a large share of the experts is resident in VRAM - which needs a card
Strata does not support and a machine that does not need Strata.

There is one escape route worth naming: **expert pruning**. Strata already ships a pruned Flash-Next (the
Coder, 256 of 512 experts, `setup.py`'s `MODELS["IQ1_M"]`). Pruned MiniMax-M3 checkpoints exist:
`sparkarena/Minimax-M3-v0-NVFP4-REAP50` (128.9 GB, 50% of the routed experts removed) and `REAP25`
(187 GB, 25% removed), both built for a DGX Spark and NVFP4. Their own card calls them experimental and
says the quality degradation is "not exhaustively tested"; they also need a patched sGLang.

**Pruning fixes the disk and not the speed.** A token still reads top-4 experts, so a half-pruned
MiniMax-M3 would still be about 24.7B active parameters and still about 4x Qwen3.8-Flash-Next per token.
It would be around 220B parameters, about 50 GB at IQ1_S - smaller than Qwen3.8-Flash-Next's Q2_0 - at
roughly a third of its speed. A pruned MiniMax-M3 gets MiniMax-M3 onto a 64 GB PC; it does not make
MiniMax-M3 a fast model.

## The Q3 / IQ3 candidates, and what Strata can actually read

The Q3 tier of MiniMax-M3, with the types read out of the real GGUF headers (first shard of each
directory, over HTTP range requests - no weights downloaded) and checked against Strata's own kernel
lists in `src/kernels/cuda/iq_kernels.cu`:

| Candidate | Repo | Download | gate/up | down | Strata today |
| --- | --- | ---: | --- | --- | --- |
| UD-IQ3_XXS | unsloth | 160.3 GB | IQ3_S / IQ2_S | IQ4_XS / **IQ3_S** | **no** - the IQ3_S down kernel is missing |
| UD-IQ3_S | unsloth | 175.6 GB | IQ3_S / IQ2_S | IQ4_XS / Q6_K | **yes**, with `-DSTRATA_Q6K_EXPERTS=ON` |
| UD-Q3_K_M | unsloth | 195.5 GB | IQ3_XXS / IQ4_XS | IQ4_XS / Q6_K | **yes**, with the same flag |
| UD-Q3_K_XL | unsloth | 195.8 GB | IQ3_XXS / IQ4_XS | IQ4_XS / Q6_K | **yes**, with the same flag |
| IQ3_XXS | bartowski | 180.0 GB | IQ3_XXS | IQ3_XXS / **Q4_K** | **no** - both downs missing |
| IQ3_XS | bartowski | 196.4 GB | IQ3_S | IQ3_S / **Q4_K** | **no** |
| IQ3_M | bartowski | 205.5 GB | IQ3_S | IQ3_S / Q5_K | **no** - the IQ3_S down kernel |
| Q3_K_S | bartowski | 187.3 GB | **Q3_K** | **Q3_K** | **no** - Q3_K has no expert kernel |
| Q3_K_M | bartowski | 196.6 GB | **Q3_K** | **Q3_K** / Q4_K | **no** |
| Q3_K_L | bartowski | 205.0 GB | **Q3_K** | **Q3_K** / Q5_K | **no** |
| Q3_K_XL | bartowski | 206.0 GB | **Q3_K** | **Q3_K** / Q5_K | **no** |

**Why.** Strata's expert kernels have two separate format lists. The **gate/up** list
(`STRATA_GU_FMTS`) has the i-quants (IQ2_XXS, IQ2_XS, IQ3_XXS, IQ3_S, IQ2_S, IQ4_XS, IQ1_M),
Q2_0, Q4_K, Q5_K, Q6_K, the legacy quants, NVFP4 and the trellis types. The **down** list
(`STRATA_D_FMTS`) is much shorter: IQ4_NL, IQ4_XS, Q2_0, the legacy quants, Q5_K, Q6_K, NVFP4 and
the trellis types - **no IQ3_XXS, no IQ3_S, no IQ2_\*, no Q3_K, no Q4_K**. The comment in the source says
why: "Q5_K / Q6_K downs: Qwen3.6-35B-A3B (FF 512); Flash-Next's FF 640 never selects them". MiniMax-M3's
FF is 3072, so the down matrix lands in formats nobody has needed until now.

**The geometry is not the problem.** `native_expert_supported` also wants `n_embd % 256 == 0` and
`n_ff % 256 == 0`: MiniMax-M3's 6144 and 3072 both divide by 256. Every format in the tables above would
work on a 6144 x 3072 expert once its kernel exists.

**So the honest count is three.** Unsloth's `UD-IQ3_S`, `UD-Q3_K_M` and `UD-Q3_K_XL` are the only
Q3/IQ3 files Strata could read today, and all three need the `STRATA_Q6K_EXPERTS` build - an option
that already exists in the repository (it is what the Unsloth Q6_K files of Qwen3.8-Flash-Next use).
The three that matter for size are a `down` matrix at **IQ3_S** (Unsloth's UD-IQ3_XXS) and at
**Q3_K** (all four of bartowski's). Those are two new expert kernels, not a re-quantisation.

Outside the Q3 tier, the same check says **bartowski's `IQ4_XS` (229.7 GB) is usable as it is**
(gate/up and down both IQ4_XS), and **Unsloth's `UD-IQ4_XS` (208.5 GB) works with the Q6_K flag**.

### What the tier costs

| Candidate | bits/weight | MB per token | vs Flash-Next | RAM to hold it | **on this machine** |
| --- | ---: | ---: | ---: | ---: | ---: |
| UD-IQ3_XXS | 3.01 | 4 856 | x8.2 | 178 GB | **1.3 / 1.5 tok/s** |
| UD-IQ3_S | 3.30 | 5 320 | x9.0 | 194 GB | **1.1 / 1.3 tok/s** |
| bartowski IQ3_XXS | 3.38 | 5 453 | x9.2 | 198 GB | **1.1 / 1.4 tok/s** |
| UD-Q3_K_M | 3.67 | 5 923 | x10.0 | 213 GB | **1.0 / 1.1 tok/s** |
| UD-Q3_K_XL | 3.68 | 5 932 | x10.1 | 214 GB | **1.0 / 1.1 tok/s** |
| bartowski IQ3_XS | 3.69 | 5 950 | x10.1 | 214 GB | **1.0 / 1.1 tok/s** |
| bartowski IQ3_M | 3.86 | 6 226 | x10.6 | 223 GB | **0.9 / 1.0 tok/s** |
| bartowski Q3_K_XL | 3.87 | 6 241 | x10.6 | 224 GB | **0.9 / 1.0 tok/s** |

Bits per weight from the file size and the 426.1B count. MB per token from 228 experts x 56.62M
parameters. tok/s is the measured Flash-Next Q2_0 number (87.3 tok/s, RTX 5070 12 GB) scaled by the
ratio of bytes per token - **an estimate, not a measurement**. RAM is the file plus 10 GB for the OS
and the engine, plus 7.9 GB of KV cache at 128K.

Two things this table says plainly. **The whole Q3/IQ3 tier is a 192-224 GB RAM machine** - none of
them fit the kind of box Strata's README targets. And **every one of them reads ten times more bytes
per token than the model Strata ships**, so they all land in the same 8-11 tok/s band whatever their
precision: at this size, the speed is set by the size of the experts, not by their quantisation.

### The Q2 / IQ2 tier, and the one structural blocker

| Candidate | Repo | Download | gate/up | down | embedding | Strata today |
| --- | --- | ---: | --- | --- | --- | --- |
| IQ1_S | bartowski | 90.5 GB | **IQ1_S** | **IQ1_M / IQ1_S** | **Q2_K** | no - three things missing |
| IQ1_M | bartowski | 100.7 GB | IQ1_M | **IQ1_M / IQ2_XXS** | **Q2_K** | no - the down kernel and the embedding |
| IQ2_XXS | bartowski | 116.6 GB | IQ2_XXS | **IQ2_XS / IQ2_XXS** | **Q2_K** | no - the down kernel and the embedding |
| IQ2_XS | bartowski | 129.5 GB | IQ2_XS | **IQ2_S / IQ2_XS** | **Q2_K** | no - the down kernel and the embedding |
| IQ2_S | bartowski | 132.1 GB | IQ2_XS | **IQ2_XS / IQ3_XXS** | IQ3_S | no - the down kernel |
| UD-IQ2_XXS | unsloth | 134.9 GB | IQ2_S / IQ2_XXS | **IQ3_XXS** / IQ4_XS | Q4_K | no - the down kernel |
| UD-IQ2_M | unsloth | 135.1 GB | IQ2_S / IQ2_XXS | **IQ3_XXS** / IQ4_XS | Q5_K | no - the down kernel |
| UD-Q2_K_XL | unsloth | 143.9 GB | IQ2_XS / IQ3_XXS | **IQ3_XXS** / IQ4_XS | Q5_K | no - the down kernel |
| IQ2_M | bartowski | 145.6 GB | IQ2_S | **IQ2_S / IQ3_S** | IQ3_S | no - the down kernel |
| Q2_K | bartowski | 151.9 GB | **Q2_K** | **Q2_K / Q3_K** | **Q2_K** | no - gate/up, down and the embedding |
| Q2_K_L | bartowski | 153.1 GB | **Q2_K** | **Q2_K / Q3_K** | **Q2_K** | no - gate/up, down and the embedding |

**Not one Q2/IQ2 file works today** (against the three that work in the Q3 tier). But the reason is the
same in every row, and it is cheaper than it looks.

**The blocker is one macro.** The down matrix is one templated kernel, `launch_down<T>`, instantiated
once per format by `STRATA_D_FMTS` (`src/kernels/cuda/iq_kernels.cu`). The file has **21 `Fmt<T>`
templates**: 2, 3, 6, 7, 8, 12, 13, 14, 16, 17, 18, 20, 21, 22, 23, 29, 40, 42, 144, 145, 146. Every
i-quant a Q2/IQ2 file uses is already among them - IQ2_XXS (16), IQ2_XS (17), IQ3_XXS (18), IQ3_S (21),
IQ2_S (22), IQ1_M (29) - they are simply **not in the down list**. Moving them there is a change to one
macro line plus the CPU-side lists in `src/kernels/cpu/iq_avx*.cpp`, and the `dot()` they use is the one
that already computes the gate/up rows.

**Two things really are new code.** `Fmt<11>` (Q3_K) **does not exist** - bartowski's four `Q3_K_*` files
and its `Q2_K_L` need a Q3_K dequantizer and dot product. And Q2_K (type 10) has no template and is not
accepted by `embed_type_supported`, so bartowski's `IQ1_S`, `IQ1_M`, `IQ2_XXS` and `IQ2_XS` (their
`token_embd` is Q2_K) need a Q2_K embedding dequantizer.

**What the tier costs:**

| Candidate | bits/weight | MB per token | vs Flash-Next | RAM to hold it | **on this machine** |
| --- | ---: | ---: | ---: | ---: | ---: |
| IQ1_S | 1.70 | 2 742 | x4.6 | 108 GB | **3.6 / 5.9 tok/s** |
| IQ1_M | 1.89 | 3 051 | x5.2 | 119 GB | **2.8 / 4.1 tok/s** |
| IQ2_XXS | 2.19 | 3 533 | x6.0 | 134 GB | **2.1 / 2.8 tok/s** |
| IQ2_XS | 2.43 | 3 923 | x6.6 | 147 GB | **1.8 / 2.2 tok/s** |
| IQ2_S | 2.48 | 4 002 | x6.8 | 150 GB | **1.7 / 2.1 tok/s** |
| UD-IQ2_XXS | 2.53 | 4 087 | x6.9 | 153 GB | **1.7 / 2.0 tok/s** |
| UD-IQ2_M | 2.54 | 4 093 | x6.9 | 153 GB | **1.7 / 2.0 tok/s** |
| UD-Q2_K_XL | 2.70 | 4 360 | x7.4 | 162 GB | **1.6 / 1.9 tok/s** |
| IQ2_M | 2.73 | 4 411 | x7.5 | 164 GB | **1.6 / 1.9 tok/s** |
| Q2_K | 2.85 | 4 602 | x7.8 | 170 GB | **1.5 / 1.8 tok/s** |
| Q2_K_L | 2.87 | 4 638 | x7.9 | 171 GB | **1.5 / 1.8 tok/s** |

Same method as the Q3 table: bits per weight from the file size, MB per token from 228 experts x 56.62M
parameters, RAM = the file plus 10 GB plus 7.9 GB of KV cache at 128K. The last column is the estimate
**on this machine** - one card, then the two cards together; the method is in "On this machine" below.

**This tier is the one that fits.** bartowski's `IQ1_S` needs 108 GB of RAM - a 128 GB machine - where
the whole Q3 tier needs 194-224 GB. And it is the fastest of all the MiniMax-M3 tiers: twice the Q3
tier's speed, and on this machine about 5.9 tok/s with the two cards, still eight times slower than
Qwen3.8-Flash-Next does on the same box.

## On this machine

Everything in the tables above is measured on the machine this study runs on.

| | | |
| --- | --- | --- |
| CPU | AMD Ryzen 9 5950X, 16 cores / 32 threads, **AVX2, no AVX-512** | `lscpu` |
| RAM | 64 GB DDR4-3333 (4 x 16 GB), 62 GiB total, 57 GiB free | `free -h`, `dmidecode` |
| GPU 1 | Radeon RX 6800 XT / 6900 XT (Navi 21, gfx1030), **15.99 GB VRAM**, PCIe 4.0 x16 | `/sys/class/drm`, `current_link_speed` |
| GPU 2 | Radeon RX 9070 XT (Navi 48, gfx1201), **15.92 GB VRAM**, **PCIe 5.0 x16** | idem |
| Disk | GIGABYTE GP-AG41TB NVMe on `/mnt/gygabyte`, **4.3 GB/s** sequential read, 246 GB free | `dd iflag=direct` |
| OS | Fedora 44, Linux 7.3 | `uname -a` |

### The method

The model has 413.1B expert parameters (57 layers x 128) and 13.0B of everything else - the attention, the
shared expert, the dense FFN of layers 0-2, the embeddings and the head - and the 0.224B indexer, which
every candidate stores in **F32**. A token reads **228 experts** (top 4 over 57 MoE layers). Per token:

```
bytes = 228 x 56.62M parameters x bits/weight / 8
```

Where those bytes come from, on this box:

- **VRAM.** One card: 16 GB, minus 1 GB of reserve, minus the KV cache (60 KB per token at 8 bits, so
  1.9 GB at a 32K context), minus the dense weights. Two cards: the same, doubled - Strata has been run on
  an R9700 + RX 9070 XT, so two AMD cards of different generations share a model.
- **RAM.** Strata's hybrid mode keeps a copy of the experts it can (measured on Qwen3.6: "UD-Q4_K_XL on a
  46 GB PC with a 12 GB card, 18.75 GiB of experts pinned"). Here, about **40 GB** of the 64 GB is
  available for that copy - and DDR4-3333 on four DIMMs tops out near 10 GB/s, which caps the CPU pool.
- **Disk.** The rest comes from the NVMe at the 4.3 GB/s measured above. **This is the term that decides
  the speed**, because the smallest MiniMax-M3 file has 87.8 GB of experts and the box has 64 GB of RAM:
  the working set does not fit, so the misses go to the disk.

The estimate takes the largest of the three paths per token (they overlap: the CPU computes one share of the
missed experts while another share crosses PCIe and the disk feeds both). It assumes a 32K context, int8 KV,
the default PCIe share (Strata's own rule, `L / (L + 1.85 R)` with `L` the PCIe link speed and `R` the CPU
pool's rate - which lands on the 0.60 cap here, because a CPU without AVX-512 is slow), and 20 GB/s for
PCIe 4.0 x16.

**The calibration**: the same model on Qwen3.8-Flash-Next Q2_0 (66.4 GB, 2 bits) gives **43 tok/s** on this
box, against the 60-93 tok/s Strata measures for it on an RTX 5070 / Ryzen 5 7600 with MTP drafting. The
model is in the right range.

### What it says

- **No MiniMax-M3 file fits in this box's RAM.** The smallest needs 102 GB (90.5 GB of file plus the OS,
  the engine and the cache); the box has 64 GB. Everything runs in the low-RAM mode, with the disk in the
  path. The disk has 246 GB free, so every candidate up to UD-Q3_K_M (195.5 GB) fits on it; UD-Q4_K_XL
  (265.8 GB) does not.
- **The best case is bartowski's IQ1_S at 3.6 tok/s on one card, 5.9 on the two.** That is one token every
  170-280 ms - readable, but not a conversation. The Q3 tier lands at 0.9-1.5 tok/s, one token per second
  or slower.
- **The two cards nearly double the speed, not ten times it.** They add VRAM for the experts, but the disk
  stays the bottleneck: 27.8% of the experts in VRAM instead of 10.7% takes IQ1_S from 3.6 to 5.9 tok/s.
- **The CPU without AVX-512 hurts.** Strata's AVX-512 CPUs get a one-time ~40 GB pack "for the fast CPU
  kernel"; on AVX2 the i-quant rows fall back to ggml-cpu's `vec_dot` for a lone token in an expert's
  group. It pushes the PCIe share to its cap, which is why the estimate leans on the link and the disk.
- **For comparison, Flash-Next Q2_0 on this box: 43 tok/s.** MiniMax-M3 would be 7 to 12 times slower.

## The same tables for MiniMax-M2

MiniMax-M2 is the predecessor (June 2026), and on this machine it is a far better proposition than M3:
**230B total, 10.4B active** (llama.cpp's `LLM_TYPE_230B_A10B`), against M3's 428B / 24.7B.

### The architecture, side by side

| | **MiniMax-M2** (`minimax-m2`) | **MiniMax-M3** (`minimax-m3`) |
| --- | --- | --- |
| Total / active | 230B / 10.4B | 428B / 24.7B |
| Layers / hidden | 62 / 3072 | 60 / 6144 |
| Heads / KV heads / head dim | 48 / 8 / 128 | 64 / 4 / 128 |
| Attention | **62 full-attention layers, no GDN, no sparse attention** | 3 dense + 57 MSA |
| Experts | 256, top-8, **no shared expert** | 128, top-4, 1 shared expert |
| Expert FFN | 1536 | 3072 |
| Activation | **plain SwiGLU** (`LLM_FFN_SILU`) | `swigluoai` (alpha 1.702, limit 7.0) |
| q_norm / k_norm | **per layer** ([6144] / [1024]) | per head ([128] / [128]) |
| Routing | sigmoid + bias + normalise + scale | the same |
| Residual / RoPE | plain pre-norm, 64 of 128 rotated, theta 5M | the same |
| Context | 196,608 | 1,048,576 |
| MTP drafting | **yes** (`num_mtp_modules: 3`) | no |
| Pictures | no | yes |
| Tokenizer | `pre = "minimax-m2"` | `pre = "minimax-m2"` - **the same one** |

**Three of M3's hard problems do not exist in M2**: no MSA sparse attention to write, no `swigluoai`
activation (Strata's `swiglu.cuh` already has the plain SwiGLU), and no vision path. And M2 keeps the MTP
block, so Strata's speculative drafting works. The tokenizer is the *same* `minimax-m2` pre-tokenizer as
M3 - one implementation serves both.

What M2 still needs: a new family in the reader, a new engine path (62 full-attention layers, 256 experts
top-8, **no shared expert**, per-layer q/k norms), the router with the sigmoid gating, the MTP draft layer,
and the `minimax-m2` tokenizer. That is a **smaller job than M3**, and the reference implementation is
`src/models/minimax-m2.cpp` (169 lines) in the llama.cpp tree Strata already pins.

### What Strata can read

| Candidate | Repo | Download | gate/up | down | Strata today |
| --- | --- | ---: | --- | --- | --- |
| IQ2_XXS | bartowski | 54.7 GB | IQ1_M / IQ2_XS | **IQ1_M / IQ2_XS** | no - the down kernel, and Q2_K for the embedding |
| IQ2_XS | bartowski | 63.1 GB | IQ2_S / IQ2_XXS | **IQ2_S / IQ2_XXS** | no - the down kernel, and Q2_K for the embedding |
| IQ2_S | bartowski | 63.4 GB | IQ2_S / IQ2_XXS | **IQ2_S / IQ2_XXS** | no - the down kernel |
| UD-IQ1_S | unsloth | 64.0 GB | **IQ1_S** / IQ1_M / IQ2_XXS | **IQ2_S / IQ3_S** / IQ4_XS | no - gate/up and the down kernel |
| UD-IQ1_M | unsloth | 68.7 GB | IQ1_M / IQ2_XXS | **IQ3_S / IQ3_XXS** / IQ4_XS | no - the down kernel |
| IQ2_M | bartowski | 72.0 GB | IQ2_XS / IQ3_XXS | **IQ2_XS / IQ3_XXS** | no - the down kernel |
| UD-IQ2_XXS | unsloth | 74.0 GB | IQ2_S / IQ2_XS / IQ2_XXS | **IQ3_S / IQ3_XXS** / IQ4_XS | no - the down kernel |
| UD-IQ2_M | unsloth | 78.1 GB | IQ2_S / IQ2_XS | **IQ3_S / IQ3_XXS** / IQ4_XS | no - the down kernel |
| Q2_K | bartowski | 79.8 GB | **Q2_K** | **Q2_K / Q3_K** | no - gate/up, down and the embedding |
| UD-Q2_K_XL | unsloth | 85.8 GB | **Q2_K** | **Q3_K / Q4_K** | no - gate/up and the down kernel |
| IQ3_XXS | bartowski | 90.1 GB | IQ3_XXS | **IQ3_S / IQ3_XXS** | no - the down kernel |
| UD-IQ3_XXS | unsloth | 93.6 GB | IQ3_S / IQ3_XXS | **IQ3_S / IQ3_XXS** / IQ4_XS | no - the down kernel |
| IQ3_XS | bartowski | 93.8 GB | IQ3_S / IQ3_XXS | **IQ3_S** | no - the down kernel |
| Q3_K_S | bartowski | 99.1 GB | **Q3_K** | **Q3_K** | no - gate/up and the down kernel |
| IQ3_M | bartowski | 104.0 GB | IQ3_S | **IQ3_S / Q4_K** | no - the down kernel |
| Q3_K_M | bartowski | 104.0 GB | **Q3_K** | **Q3_K / Q4_K** | no - gate/up and the down kernel |
| Q3_K_XL | bartowski | 108.7 GB | **Q3_K** | **Q3_K / Q5_K** | no - gate/up and the down kernel |
| **IQ4_XS** | bartowski | 122.2 GB | IQ4_XS | IQ4_XS | **yes, as it is** |
| **IQ4_XS** | unsloth | 121.9 GB | IQ4_XS | IQ4_XS | **yes, as it is** |

The same verdict as M3: **the down matrix is the blocker**, and the same macro change fixes it. The
formats needed are IQ1_M, IQ2_XS, IQ2_S, IQ2_XXS, IQ3_XXS and IQ3_S - all of which already have a `Fmt<T>`
template in `src/kernels/cuda/iq_kernels.cu`. The two new pieces are Q3_K (no `Fmt<11>`) and Q2_K for the
embedding table of bartowski's four smallest files.

### The estimates on this machine

MiniMax-M2 reads **496 experts per token** (top-8 over 62 layers), 14.16M parameters each, and its KV
cache is **124 KB per token** (62 layers x 8 KV heads x 128, int8) - twice M3's, so 4.06 GB at a 32K
context.

| Candidate | bits/weight | Mo per token | RAM to hold it | **on this machine (1 card / 2)** |
| --- | ---: | ---: | ---: | ---: |
| **bartowski IQ2_XXS** | 1.91 | 1.68 | 69 GB | **10.9 / 16.6 tok/s** |
| bartowski IQ2_XS | 2.21 | 1.94 | 77 GB | **9.2 / 12.9 tok/s** |
| bartowski IQ2_S | 2.22 | 1.95 | 78 GB | **9.1 / 12.8 tok/s** |
| unsloth UD-IQ1_S | 2.24 | 1.96 | 78 GB | **9.0 / 12.6 tok/s** |
| bartowski IQ2_M | 2.52 | 2.21 | 86 GB | **6.5 / 10.4 tok/s** |
| unsloth UD-IQ2_XXS | 2.59 | 2.27 | 88 GB | **5.9 / 10.0 tok/s** |
| unsloth UD-IQ2_M | 2.73 | 2.40 | 92 GB | **5.0 / 9.2 tok/s** |
| bartowski Q2_K | 2.79 | 2.45 | 94 GB | **4.8 / 8.9 tok/s** |
| unsloth UD-Q2_K_XL | 3.00 | 2.63 | 100 GB | **3.9 / 6.9 tok/s** |
| bartowski IQ3_XXS | 3.15 | 2.77 | 104 GB | **3.5 / 5.7 tok/s** |
| unsloth UD-IQ3_XXS | 3.27 | 2.87 | 108 GB | **3.2 / 5.0 tok/s** |
| bartowski IQ3_XS | 3.28 | 2.88 | 108 GB | **3.2 / 4.9 tok/s** |
| bartowski IQ3_M | 3.64 | 3.19 | 118 GB | **2.6 / 3.6 tok/s** |
| bartowski Q3_K_XL | 3.80 | 3.34 | 123 GB | **2.4 / 3.2 tok/s** |
| bartowski IQ4_XS | 4.27 | 3.75 | 136 GB | **1.9 / 2.4 tok/s** |

**MiniMax-M2 on this box is two to three times faster than MiniMax-M3**, and the best file,
bartowski's `IQ2_XXS` (54.7 GB), comes close to fitting in RAM: 54.7 GB of weights plus the 10 GB the
engine leaves for the OS and 4.06 GB of KV cache at 32K is 69 GB, against 64 GB of RAM and 57 GiB free.
It does not quite fit, so it still runs in the low-RAM mode - but with only 54.7 GB of experts against
about 40 GB of RAM copy, 74% of the experts are resident, and **10.9 tok/s on one card, 16.6 on the two**,
against 43 tok/s for Qwen3.8-Flash-Next Q2_0.

The Q4 tier is where M2 gets unpleasant: at IQ4_XS (122.2 GB) the speed falls to 1.9 tok/s, because the
working set is far past the RAM and everything comes off the disk.

## The work, item by item

Ordered by how much each one costs. The files are the ones to touch.

**1. The family in the reader (small, mechanical).**
`include/strata/artifact/model_family.hpp`: a `ModelFamily::MiniMaxM3`, the arch string `minimax-m3`, a
title, and a decision on `model_family_runs` (NVIDIA and AMD yes, Intel Arc no, as for Qwen3.6).
`include/strata/artifact/gguf_reader.hpp`: a `check_architecture_minimax_m3` guard that requires the keys
the geometry is read from (`block_count`, `embedding_length`, `attention.head_count`,
`attention.head_count_kv`, `attention.key_length`, `attention.value_length`, `expert_count`,
`expert_used_count`, `expert_feed_forward_length`, `expert_shared_count`, `leading_dense_block_count`,
`attention.indexer.head_count`, `attention.indexer.key_length`, `attention.indexer.top_k`,
`attention.indexer.block_size`, `attention.indexer.local_blocks`, `expert_gating_func`,
`expert_weights_scale`, `expert_weights_norm`, `rope.dimension_count`).
`include/strata/core/family_layout.hpp`: a layout table (the tensor names are in `conversion/minimax.py`
and `src/models/minimax-3.cpp` of llama.cpp), the dense-lead layers, and the indexer tensors on the sparse
layers only.
`include/strata/core/geometry.hpp`: the new compiled shape, and the fields MiniMax-M3 needs that do not
exist yet (the leading dense count, the indexer block size and top-k, the local blocks, the routing bias,
the expert weight scale).

**2. The engine path (big).**
`src/qwen35/` is the template: a path of its own, switched to from `general.architecture`, with its own
kernels (`q36_kernels.cuh`). MiniMax-M3 needs the same, in `src/minimaxm3/`. It has to carry:

- **MSA sparse attention.** The indexer branch (two projections, two norms, partial RoPE), the per-position
  scores against the indexer keys, the **max-pool over a block of 128 positions**, the **top 16 blocks**,
  the **forced local block** (bias `1e30`), the expansion of a block back to key positions, the sparse mask,
  and the attention over the gathered keys. The decode path and the batch path differ (llama.cpp gathers
  for decode and builds a mask for a batch), and both have to exist.
- **A position-to-cell map and a KV cache for MSA.** Blocks are anchored to absolute positions, not to
  cache cells, so the cache needs the map llama.cpp wrote in `llama-kv-cache-msa.cpp`.
- **`swigluoai`.** `x = min(gate, 7.0)`, `y = clamp(up, -7.0, 7.0)`,
  `out = (x / (1 + exp(1.702 * -x))) * (y + 1)`. Strata's `include/strata/kernels/swiglu.cuh` has three
  copies of the plain SwiGLU; this is a fourth expression.
- **The router.** `sigmoid(logits)`, add the per-expert bias (`exp_probs_b.bias`) for the **choice**,
  keep the unbiased sigmoid for the **weight**, top-4, normalise the four weights to sum to one, scale by
  2.0. Strata's `native_router.hpp` is pinned to 512 experts / top-10 / softmax.
- **The leading dense layers** (3 of them, dense FFN and dense attention) and the shared expert, which
  Strata has for Qwen3.6 but with a different width and a different activation.
- **The embeddings.** 200,064 rows of 6144, and a separate output head. Vocabulary is free in Strata
  (`n_vocab` is read from the file), but the head's precision and the softmax over 200k rows are worth
  checking.

**3. The tokenizer (medium, and it is easy to get wrong).**
`tools/strata_tokenizer.py` reads the file's `tokenizer.ggml.pre` and is pinned to the `qwen35`
pre-tokenizer and to Qwen's byte-level BPE. MiniMax-M3's GGUF says **`pre = "minimax-m2"`**, and its
pre-tokenizer is a different regex (it splits digits in groups of three, splits capitalised words from
lower-case ones, and keeps a trailing `[\r\n/]*` with a punctuation run). The file also has 200,000
vocabulary entries, 199,744 merges, 61 added tokens, and token strings that carry a zero-width character.
So the work is a new `minimax-m2` case in the pre-tokenizer, the special tokens (bos 200019, eos 200020,
pad 200000, unk 16474, image 200025, video 200026), and a re-run of the round trip
`decode(encode(s)) == s` over MiniMax-M3's corpus - the check in that file that sees the failures. The
C++ port follows.

**4. The server's thinking markers (small).**
`serve/server.py` closes a reply on `</think>` (`REASONING_CLOSE`) and `serve/frontend.py` marks and
unmarks thinking literals. MiniMax-M3 uses `<​mm:think>` and `</​mm:think>` instead of `<think>` and
`</think>`. **Both markers carry a zero-width space (U+200B) after the `<`**, so they are
not what they look like, and they must become per-family.

**5. Setup, the catalogue and the app (small, but it touches many files).**
`setup.py`: a new `OWN_PATH_FAMILIES` entry (MiniMax-M3 has no pack of its own, like Qwen3.6), the file
pins with sizes and SHA-256, the `--family` choice, and the fit arithmetic. `strata_cli/launch/catalog.py`
reads that table, so it follows. `apps/host/src/lib/native-names.ts` and `apps/host/server/catalog.mjs`
list the families the app knows. `serve/server.py`'s `FAMILY_MODEL_NAMES` maps an arch to a model name.

**6. What is not needed.**
No pack tool (the file is read as it is), no expert profile, no n-gram table, no MTP draft layer (the
released weights have none), no new vision code (llama.cpp's `mtmd` has the projector: a 32-layer CLIP ViT
with 3-axis axial RoPE, then a two-stage patch-merge MLP - `linear_1`/`linear_2` for the per-patch
embedding and `merger.fc1`/`merger.fc2` for the 2x2 merge). A projector file exists:
`bartowski/MiniMax-M3-GGUF`'s `mmproj-MiniMax-M3-bf16.gguf` (1.735 GB).

## A staged plan

**Stage 0 - a reference, before any Strata code (this is the honest first step).**
Build llama.cpp at Strata's pinned commit with MiniMax-M3 support, run bartowski's IQ1_S (90.5 GB) or
Unsloth's UD-IQ2_XXS on it, and record: the logits on a fixed token list, the speed, the memory, and
whether the pictures work. Two things to check early, because they shape everything after: **flash
attention must be on** (without it the model is degraded, not merely slower), and **the MSA's fused kernel
does not exist upstream** - so the reference speed is the gather-at-decode / mask-at-prefill path, which is
the target Strata would have to match. vLLM and SGLang both have MiniMax-M3 merged and their recipes are
the second reference. **This stage is worth doing even if Strata never runs MiniMax-M3.**

**Stage 1 - the family in the reader.**
The guard, the layout table and the geometry, plus the tools that read a GGUF (`tools/strata_inspect.py`).
Ends when `strata inspect` prints MiniMax-M3's shape and refuses a file with the wrong one, with the key
that differs named.

**Stage 2 - the text engine, dense attention only.**
The path, the router, `swigluoai`, the dense lead layers, the shared expert, the embeddings, **and MSA
switched off** (the first three layers are dense anyway, so a run with the sparse layers forced dense is
still a valid model to check against: llama.cpp itself falls back to dense when flash attention is off).
Check against llama.cpp token by token. This is the stage that proves the engine works.

**Stage 3 - MSA.**
The indexer, the block scores, the top-16, the local block, the sparse mask, the gather, the position map.
Checked against llama.cpp's MSA path (which needs flash attention on, and a per-sequence KV cache -
`--no-kv-unified`).

**Stage 4 - the rest of the product.**
The tokenizer, the thinking markers, the chat template's modes, the app's catalogue, setup's fit
arithmetic, the pictures.

**Stage 5 - the memory story.**
This is the stage that decides whether the family ships. It needs an answer to: how much of MiniMax-M3
fits in 32 GB, in 64 GB, in 192 GB, and what each gives in tokens per second. Without those numbers, setup
would be offering a model that fits on paper and does not run.

## The open questions

1. **Is there a pruned MiniMax-M3 that keeps Strata's promise?** The only route to a gaming PC, and it
   only fixes the disk. The two pruned checkpoints that exist are experimental NVFP4 builds for a DGX
   Spark, and their author says the quality loss is not established. A pruned MiniMax-M3 at a supported
   quantisation (2-4 bits, not NVFP4-only) with a measured quality curve does not exist yet.
2. **Does the sparse attention earn its keep at Strata's context sizes?** MSA's gain is at a million
   tokens. Strata's usual contexts are 32K-256K, where a dense attention over a shorter context may cost
   less than the indexer. **This should be measured, not assumed.**
3. **What is the drafting story without MTP?** Strata's other two families draft (Flash-Next's n-gram
   table, Qwen3.6's MTP block). MiniMax-M3 has neither in its released weights: the config declares
   `num_mtp_modules: 7` but the checkpoint ships **zero** `mtp.*` tensors. Two drafts exist for it on the
   Hub - NVIDIA's DSpark head (`nvidia/MiniMax-M3-DSpark`) and `Inferact/MiniMax-M3-EAGLE3` (with an open
   llama.cpp PR, #24925) - but both are another engine's mechanism.
4. **Does the vision path need the 2-D positions?** MiniMax-M3's language model uses plain 1-D positions
   (its 3-D RoPE is inside the ViT), so Strata's 2-D image placement for the Qwen3-VL models does not
   apply. That is simpler, but it has to be confirmed against the encoder's output. The image tokens are
   ordinary vocabulary ids (200025 for a picture, 200026 for a video) whose embedding rows are replaced by
   the projected features, and a 672x672 picture is 576 tokens.
5. **Is the 1M context real on a 128 GB machine?** At 8 bits the KV cache alone is 62.9 GB at 1M
   positions (60 KB per token), beside 87 GB of experts. The two together do not fit. At 128K it is 7.9 GB
   of cache beside 87 GB of experts, which does.
6. **How does MSA meet Strata's KV streaming?** Strata keeps the authoritative K/V in pinned host memory and
   makes the blocks the *selection* names resident in VRAM on demand (`include/strata/kernels/kv_stream.hpp`,
   `kv_stream_resolve`). That model fits MSA well - the page table is already keyed by the absolute block,
   not by the VRAM slot. But the granularities differ: a Strata page is 4 cells (one Qwen indexer block),
   while an MSA block is 128 positions, and the indexer keys for all 128 positions have to be resident to
   score the block. The Qwen3.6 path keeps its KV in VRAM and does not stream; MiniMax-M3 has 60 KV layers
   instead of 10, so at 262K it would be 16.1 GB of cache, more than a 12 GB card holds. **This is the
   design question the engine path has to answer first.**

## What the MiniMax-M2 integration found on the machine (2026-10-20)

The MiniMax-M2 IQ2_XXS file (bartowski, 54.73 GB, two shards) is downloaded and the reader takes it:
809 tensors, the layout table exact, the geometry right (62 layers of full attention, 48/8 heads x 128, 256
experts top-8, FF 1536, no GDN, no shared expert, no indexer, vocab 200064).  The engine path
(`src/minimaxm2/`) loads it: 1.86 GiB of dense weights on the GPU, the routed experts placed, and one
prompt chunk plus one decode step run ("warmed up in 90.5 s").

**The reference answers correctly.**  llama.cpp at the pinned commit (built with ROCm 10.1 for gfx1201)
answers `ĠParis` (8261) for "The capital of France is" with the same file and the same ids.

**The blocker is the memory, and it is a specific one.**  The hybrid mode pins a copy of every expert in
host memory - 49.10 GiB - and the box has 64 GB.  The experts are interleaved in the GGUF ([N, FF, NE]),
so reading one expert's rows touches one page every 227 KB.  With the page cache squeezed out by the
pinned copy, every one of those pages is a cold read: the placement ran at **39-52 MB/s** on a drive that
reads 4.3 GB/s, and 3.5 GB of swap went in and out.  The copy itself is not the problem; the copy is
what kills the page cache.

**The fix is the mode Flash-Next already has.**  Strata's Flash-Next path has a low-RAM mode
(`--mmap-experts`) that keeps no host copy at all: the experts stay in the file and the CPU pool reads
them where they lie, through the mapping (`src/core/expert_source.cpp`, "the GGUF in place": the job's
`blob` is the gate rows and the job's `NativeFmt` carries the byte distances to the up and down tensors).
That mode is what lets the 93 GB Flash-Next file run on this box at all, and it is what MiniMax-M2 needs:
without the 49 GB pinned copy the page cache can hold the 54.7 GB file, and the interleaved reads are
served from RAM.

So the MiniMax-M2 engine path needs the same treatment in three places:
  * the placement: no host copy, the experts stay in the mapping;
  * the CPU pool's job: the blob and the per-job `NativeFmt` point into the mapping (the mechanism is
    already in `expert_source.cpp`, it needs the M2 tensor layout);
  * the prompt path's staged copy and the slot replacement: the same, per expert.
Until then MiniMax-M2 loads and runs a step on this box, but the placement takes half an hour and the
machine thrashes while it does.

### The copy into VRAM, and the driver

The second half of the placement - the slots' copy from the pinned host arena into VRAM - runs at **10 MB/s**
on this box, while the same operation from a 256 MiB pinned buffer runs at **10.15 GB/s** (sync) and
**11.79 GB/s** (async) (`hipMemcpy` on the RX 9070, ROCm 10.1).  The arena is resident and pinned
(`RssShmem` 49.26 GB), the PCIe link is up (32 GT/s), and the stack shows the thread waiting inside
`hipMemcpy`.  Copying one expert at a time instead of a whole layer's slots changed nothing (10 MB/s
either way), so it is not the size of the copy.

A thousand times between a 256 MiB and a 49 GB registered host region is the driver's handling of a very
large registered allocation, not the link and not the code's copy shape.  The next thing to try is
bisecting the arena size (a 4 GB arena with the same code), and then the low-RAM mode above, which does
not need the arena at all.

### The other finding: the two cards

The layer split across GPUs lives in the **Flash-Next** path (`src/program/generate.cpp`, the stage search,
the prompt buffers' loans, the per-stage expert caches).  The Qwen3.6 path does not have it, and it
**refuses** the option: `strata-q36` exits with its usage on `--layer-split`, so a two-GPU config cannot
run Qwen3.6 either.  MiniMax-M2 inherits that gap: its path is built on the Qwen3.6 template, so it is
single-GPU today.  Porting the split to the "own path" families is a separate piece of work from the
model itself.

## The sources

- `https://huggingface.co/MiniMaxAI/MiniMax-M3` - the card, `config.json`, `tokenizer.json`,
  `tokenizer_config.json`, `models/templates/MiniMax-M3.jinja` and the modelling code
- `https://huggingface.co/MiniMaxAI/MiniMax-M3-NVFP4` - NVIDIA's checkpoint
- `https://github.com/MiniMax-AI/MiniMax-M3` - the repository and the technical report (arXiv:2606.13392)
- llama.cpp at `3cf03257f219afbe7334045ff7c6a06ac68c627d` (the commit Strata pins): `conversion/minimax.py`,
  `src/models/minimax-3.cpp`, `src/llama-kv-cache-msa.cpp`, `tools/mtmd/models/minimax-m3.cpp`,
  `common/parsers/minimax-m3.cpp`, `ggml/src/ggml-cpu/ops.cpp` (the `swigluoai` expression),
  `src/llama-arch.cpp` (the metadata key names), and the GGUF headers of
  `unsloth/MiniMax-M3-GGUF/UD-IQ1_M` and `bartowski/MiniMax-M3-GGUF`
- The other engines, all merged: vLLM PR [#45381](https://github.com/vllm-project/vllm/pull/45381) and
  SGLang PRs #28712-#28715. Their recipes are the second reference after llama.cpp.
- The modelling code in transformers (`models/minimax_m3_vl/modeling_minimax_m3_vl.py`), the paper
  arXiv:2606.13392, and the MSA kernels at `github.com/MiniMax-AI/MSa` (SM100 / Blackwell only).
  **The paper's numbers (28.4x less attention compute at 1M, 14.2x prefill, 7.6x decode on an H800) were
  measured by MiniMax on a 109B model, not on the released 428B MiniMax-M3.**
- The Hub's file lists for `unsloth/MiniMax-M3-GGUF`, `bartowski/MiniMax-M3-GGUF`,
  `sparkarena/Minimax-M3-v0-NVFP4-REAP50`, `sparkarena/Minimax-M3-v0-NVFP4-REAP25`
- Strata's own files: `include/strata/core/geometry.hpp`, `include/strata/core/family_layout.hpp`,
  `include/strata/artifact/model_family.hpp`, `include/strata/artifact/gguf_reader.hpp`,
  `include/strata/kernels/qsa.hpp`, `include/strata/kernels/native_router.hpp`,
  `include/strata/kernels/swiglu.cuh`, `src/qwen35/`, `setup.py`, `serve/server.py`,
  `tools/strata_tokenizer.py`, `docs/DETAILS.md`

The speed and memory figures for MiniMax-M3 on Strata are **estimates** from the parameter counts and the
PCIe bandwidth, not measurements. The figures for Qwen3.8-Flash-Next are Strata's own measurements, from
`docs/DETAILS.md`. The parameter counts are this document's own counts from the geometry and the config,
given so they can be checked.
