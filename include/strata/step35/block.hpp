// include/strata/step35/block.hpp - one Step-3.7-Flash block on the GPU, with the file's own weights.
//
// This is the assembly, not a new piece of arithmetic: every step below is one of the kernels next to it, in the
// order the reference uses and the order the release was trained with.
//
//   h -> rms_norm(attn_norm)                    (the per-head kernel with one row of n_embd: same arithmetic)
//     -> q, k, v = gemv(IQ2_S / Q4_K)           -> per-head RMSNorm of q and k
//     -> rope (NEOX, this block's n_rot and base, the frequency factors on a full block)
//     -> causal attention (one position: its own V, through the same kernel)
//     -> head-wise gate: attn *= sigmoid(g_proj(x))
//     -> wo -> +h
//     -> rms_norm(ffn_norm) -> dense MLP (silu * up, clamped) -> down -> +h
//
// The mixture-of-experts path is not in this entry point yet: it needs the expert tensors of 42 blocks and a
// streamed pool, and the dense blocks are what can be checked end to end today.
#pragma once

#include "strata/artifact/gguf_reader.hpp"

#include <cstdint>
#include <string>

namespace strata::step35::kernels {

/// The device buffers of one dense block, uploaded from the file.  Freed by release().
struct DenseBlock {
    void* q = nullptr;      // attn_q, attn_k, attn_v, attn_output, attn_gate: quantized GGUF blocks, as stored
    void* k = nullptr;
    void* v = nullptr;
    void* wo = nullptr;
    void* gate = nullptr;
    void* ffn_gate = nullptr;
    void* ffn_up = nullptr;
    void* ffn_down = nullptr;
    float* attn_norm = nullptr;
    float* ffn_norm = nullptr;
    float* q_norm = nullptr;
    float* k_norm = nullptr;
    float* rope_ff = nullptr;   // shared rope_freqs table, or null
    int n_embd = 0, head_dim = 0, n_head = 0, n_head_kv = 0, n_rot = 0, ff = 0, n_ff_rope = 0;
    int t_q = 0, t_k = 0, t_v = 0, t_wo = 0, t_gate = 0, t_ffn = 0;   // ggml types
    double base = 0;
    bool sliding = false;
    float eps = 0;
};

/// Uploads block `layer` of `file` (a step35 trunk shard set) and its shared rope_freqs table.
/// `with_ffn` false uploads the attention half alone (a mixture-of-experts block has no dense ffn_* tensor).
bool upload_block(const strata::GgufModel& file, int layer, DenseBlock& b, std::string& err, bool with_ffn = true);
void release_block(DenseBlock& b);

/// Runs the block over one token whose embedding is `h` (n_embd floats, device), at `pos`, in place.
/// `scratch_q8` is a device buffer of gemv_q8_bytes(n_embd).  Returns false on a kernel launch error.
bool run_dense_block(const DenseBlock& b, float* h, int pos, void* scratch_q8, int n_scratch, std::string& err);

/// A mixture-of-experts block: the same attention as DenseBlock, then the router and the experts.  The three
/// expert tensors are [n_embd, n_ff_exp, n_expert] with expert e contiguous, so the existing gemv takes one
/// expert by pointer - no grouped kernel is invented here.  EVERY expert of the block is resident: this is the
/// validation path for one block, not the memory plan for 42 (that one streams, as MiniMax-M2 does).
struct MoeBlock {
    DenseBlock attn;             // the attention half, and the shared expert's shape
    void* router = nullptr;      // ffn_gate_inp.weight, F32 [n_embd, n_expert]
    float* select_bias = nullptr;  // exp_probs_b.bias, or null
    void* exps_gate = nullptr;   // ffn_gate_exps.weight, [n_embd, n_ff_exp, n_expert]
    void* exps_up = nullptr;     // ffn_up_exps.weight
    void* exps_down = nullptr;   // ffn_down_exps.weight
    void* shexp_gate = nullptr;  // ffn_gate_shexp.weight, [n_embd, n_ff_shexp]
    void* shexp_up = nullptr;
    void* shexp_down = nullptr;
    int n_expert = 0, top_k = 0, n_ff_exp = 0, n_ff_shexp = 0;
    int t_exps = 0, t_shexp = 0;
    float scale = 0;
    bool weights_norm = false;
    float clamp_exp = 0, clamp_shexp = 0;
    size_t expert_gate_bytes = 0, expert_down_bytes = 0;   // one expert's share of each 3-D tensor
    bool loaded = false;
};

bool upload_moe_block(const strata::GgufModel& file, int layer, MoeBlock& b, std::string& err);
void release_moe_block(MoeBlock& b);
bool run_moe_block(const MoeBlock& b, float* h, int pos, void* scratch_q8, int n_scratch, std::string& err);

/// One mixture-of-experts block whose experts stay in the FILE: the attention half, the router, the selection
/// bias and the shared expert are resident (a few MB), and the eight chosen experts are copied out of the
/// mapping into three reused device buffers for each block.  Slow by design - it is what lets the whole model
/// run on 32 GB of VRAM - and the page cache does the reading, so the second pass over a layer is cheap.
bool run_stream_block(const strata::GgufModel& file, int layer, float* h, int pos, void* scratch_q8,
                      std::string& err);

}  // namespace strata::step35::kernels
