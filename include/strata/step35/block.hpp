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
bool upload_block(const strata::GgufModel& file, int layer, DenseBlock& b, std::string& err);
void release_block(DenseBlock& b);

/// Runs the block over one token whose embedding is `h` (n_embd floats, device), at `pos`, in place.
/// `scratch_q8` is a device buffer of gemv_q8_bytes(n_embd).  Returns false on a kernel launch error.
bool run_dense_block(const DenseBlock& b, float* h, int pos, void* scratch_q8, int n_scratch, std::string& err);

}  // namespace strata::step35::kernels
