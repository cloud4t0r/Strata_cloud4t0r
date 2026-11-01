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

struct Session;     // the quantized K/V cache, session.hpp (forward: only a pointer crosses here)
struct ExpertCache;  // the VRAM-resident expert tier, experts.hpp (same: a pointer)

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

/// The whole model over a prompt of several positions, DENSE blocks then the mixture-of-experts ones: every
/// position goes through each block at once (see src/step35/step35_seq.cu).  `sums` receives, per block, the hidden
/// state summed over every position - the figure `--ref` prints - and `out` the last block's output,
/// [n_pos][n_embd].
///
/// `session`, when given, is the conversation's quantized K/V: the pass writes its cells (`session_append`) and
/// reads the cache in place (`session_attend`) instead of attenting over the tensors it just computed.  A decode
/// step is this same call with one token at `pos0 = session.n_cached`, so prefill and decode cannot drift apart.
/// With `session` null the attention is `attn_causal` over this pass's fp32 K/V - the validation path.
/// `experts`, when given, is the VRAM-resident expert tier (`experts.hpp`): a position's routed experts are read
/// from it instead of copied out of the file.  A hit and a miss compute the same bits - only the copy goes away.
bool run_prompt_dense(const strata::GgufModel& file, const std::vector<int32_t>& tokens, int pos0, int n_blocks,
                      std::vector<float>& out, std::vector<double>& sums, std::string& err,
                      Session* session = nullptr, ExpertCache* experts = nullptr);

/// The vocabulary logits of ONE position: `h` is one row of the last block's output on the HOST (n_embd floats),
/// `logits` receives the file's `output.weight` applied to `rms_norm(h, output_norm)` - n_vocab floats, also on the
/// host.  Same order as `ref::logits`, which is the point: the CPU reference and the GPU can then be held to each
/// other on the token they would answer with, and not only on the hidden state's sum.
///
/// This is the trunk's own head.  An MTP draft head has its own shared_head_head inside the NextN file and does
/// not go through here.
bool logits_last(const strata::GgufModel& file, const float* h, std::vector<float>& logits, std::string& err);

/// The trunk's head, OPEN across calls: `output.weight` is 363 MB of Q5_K, and a generation loop would spend half a
/// second per token uploading it again.  Open once, apply per position, close.
struct Head {
    void* w = nullptr;      ///< output.weight, resident
    float* norm = nullptr;  ///< output_norm.weight, resident
    float* h = nullptr;     ///< one position's hidden state, device
    float* logits = nullptr;///< n_vocab floats, device
    void* q8 = nullptr;     ///< the q8_1 scratch the matvec needs
    int n_embd = 0;
    int n_vocab = 0;
    int type = 0;           ///< output.weight's ggml type
    float eps = 0;          ///< the model's layer_norm_rms_epsilon
};
bool head_open(const strata::GgufModel& file, Head& head, std::string& err);
bool head_apply(Head& head, const float* h, std::vector<float>& logits, std::string& err);
void head_close(Head& head);

/// The NextN file's blocks, RESIDENT: three decoder layers plus three heads is 3.7 GB of weights, and a draft that
/// re-uploads them per token (1.2 GB each) costs four times what a trunk step costs.  Open once, step per draft,
/// close.
struct MtpHead {
    DenseBlock block;            ///< the NextN decoder layer's own tensors
    void* eh = nullptr;          ///< nextn.eh_proj (2 * n_embd -> n_embd)
    void* enorm = nullptr;       ///< nextn.enorm
    void* hnorm = nullptr;       ///< nextn.hnorm
    void* shnorm = nullptr;      ///< nextn.shared_head_norm
    void* shead = nullptr;       ///< nextn.shared_head_head (n_embd -> n_vocab)
    int t_eh = 0, t_sh = 0;
};

struct MtpContext {
    const strata::GgufModel* file = nullptr;   ///< the NextN file, for the per-draft embedding row
    std::vector<MtpHead> heads;
    int first = 0;              ///< the file's first NextN block (45 for this model)
    int n_mtp = 0;
    int n_embd = 0, n_kv = 0, head_dim = 0, n_vocab = 0;
    float eps = 0;
    void* q8 = nullptr;         ///< the q8_1 scratch, sized for the widest input any of the steps quantizes
    float* q = nullptr;
    float* k = nullptr;
    float* v = nullptr;
    float* ao = nullptr;
    float* gt = nullptr;
    float* res = nullptr;
    float* g = nullptr;
    float* u = nullptr;
    float* act = nullptr;
    float* d = nullptr;
    float* s = nullptr;
    float* x = nullptr;
    float* cat = nullptr;
    float* eu = nullptr;
    float* hu = nullptr;
    float* lg = nullptr;        ///< n_vocab floats: the head's output, NOT the residual width
    int* dpos = nullptr;
};

/// Uploads every NextN block and head.  `n_mtp` blocks are read from the file's own `nextn_predict_layers`.
bool mtp_open(const strata::GgufModel& mtp, MtpContext& ctx, std::string& err);
/// One draft step of head `k` (0-based, weights = block `first + k`) at absolute position `pos`, fed `h` and
/// `token`, writing into `cache` and attending over the cell it writes alone.  `h_nextn` is what the next head
/// consumes, `logits` its vocabulary.
bool mtp_step(MtpContext& ctx, int k, int slot, int pos, int32_t token, const float* h, Session* cache,
              std::vector<float>& h_nextn, std::vector<float>& logits, std::string& err);
void mtp_close(MtpContext& ctx);

/// One mixture-of-experts block whose experts stay in the FILE: the attention half, the router, the selection
/// bias and the shared expert are resident (a few MB), and the eight chosen experts are copied out of the
/// mapping into three reused device buffers for each block.  Slow by design - it is what lets the whole model
/// run on 32 GB of VRAM - and the page cache does the reading, so the second pass over a layer is cheap.
bool run_stream_block(const strata::GgufModel& file, int layer, float* h, int pos, void* scratch_q8,
                      std::string& err);

}  // namespace strata::step35::kernels
