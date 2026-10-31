// src/step35/step35_seq.cu - the model over a PROMPT of several positions, the dense blocks first.
//
// Every block function so far handled ONE position, so a chat turn could not be expressed at all.  This file runs
// n_pos positions through the blocks at once, with the kernels that already exist - no new arithmetic:
//
//   * a residual RMSNorm over n_pos rows is per_head_rms_norm called with head_dim = n_embd: each row is one
//     position and is normalised on its own, which is what the layer wants;
//   * a projection over n_pos positions is gemv_multi in chunks of 8 (iq_mmvq's column limit);
//   * rope takes one position per row, so the position array is n_pos * n_head long, row r sitting at
//     pos0 + r / n_head;
//   * the attention kernel already takes the whole sequence, with the block's window;
//   * the gate is per (position, head), and the arrays are laid out exactly that way, so the gate kernel is
//     called with n_pos * n_head rows.
//
// WITH A CACHE (`Session* session`) the pass also OWNS ITS K/V: after rope, each position's cells are written with
// kv_append_q8_steps and the attention reads the cache in place, over `0 .. pos0+n_pos-1` rather than over this
// pass's tensors alone.  That is what makes the second pass cheap - a decode row reads the cells the prompt wrote
// instead of recomputing them - and it is why the same function is both the prefill and the decode step: a decode
// is this function with one token at `pos0 = n_cached`.  Without a session, the attention is `attn_causal` over the
// fp32 tensors of this pass, which is the validation vehicle for the arithmetic.
#include "strata/step35/attn.hpp"
#include "strata/step35/block.hpp"
#include "strata/step35/gemv.hpp"
#include "strata/step35/kernels.hpp"
#include "strata/step35/moe.hpp"
#include "strata/step35/session.hpp"

#include <cuda_runtime.h>
#include <ggml.h>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace strata::step35::kernels {
namespace {

constexpr int kMaxCols = 8;   // iq_mmvq's column limit

/// The gate of every (position, head), applied to that row of the attention output.
__global__ void k_gate_rows(float* attn, const float* gate, int head_dim) {
    const int r = blockIdx.x;
    const float s = 1.0f / (1.0f + expf(-gate[r]));
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) attn[(size_t) r * head_dim + d] *= s;
}

/// y = x @ W over `n_pos` columns, in the chunks the quantized matvec accepts.  Column p of x is at p * n_in,
/// column p of y at p * n_out, and the Q8_1 scratch holds kMaxCols columns.
bool proj(ggml_type t, const void* w, const float* x, float* y, int n_in, int n_out, int n_pos, void* q8,
          std::string& err) {
    for (int p = 0; p < n_pos; p += kMaxCols) {
        const int n = std::min(kMaxCols, n_pos - p);
        if (!gemv_multi(t, w, x + (size_t) p * n_in, y + (size_t) p * n_out, n_in, n_out, n, q8, nullptr)) {
            err = "a projection refused its type or shape";
            return false;
        }
    }
    return true;
}

/// One tensor, raw (the quantized bytes as stored), on the device.
void* upload(const strata::GgufModel& f, const std::string& name, size_t& bytes) {
    size_t shard = 0;
    const strata::TensorInfo* t = f.find(name, &shard);
    if (!t) throw std::runtime_error("missing tensor " + name);
    bytes = strata::tensor_payload_bytes(*t);
    void* d = nullptr;
    if (cudaMalloc(&d, bytes) != cudaSuccess) throw std::runtime_error("out of device memory for " + name);
    if (cudaMemcpy(d, f.shard(shard).tensor_data(*t), bytes, cudaMemcpyHostToDevice) != cudaSuccess)
        throw std::runtime_error("copy failed for " + name);
    return d;
}

__global__ void k_acc_scaled_rows(float* acc, const float* y, float w, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) acc[i] += w * y[i];
}

__global__ void k_add_n(float* x, const float* y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}

__global__ void k_silu_mul_n(float* act, const float* up, int n, float clamp) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = act[i], u = up[i];
    g = 1.0f / (1.0f + expf(-g)) * g;   // silu
    if (clamp > 1e-6f) {
        if (u > clamp) u = clamp;
        if (u < -clamp) u = -clamp;
        if (g > clamp) g = clamp;
    }
    act[i] = g * u;
}

/// One row of a quantized embedding table, as f32.
bool embed_rows(const strata::GgufModel& file, const std::vector<int32_t>& tokens, int64_t n_embd,
                std::vector<float>& out, std::string& err) {
    size_t shard = 0;
    const strata::TensorInfo* t = file.find("token_embd.weight", &shard);
    if (!t) {
        err = "no token_embd.weight";
        return false;
    }
    const size_t row_bytes = ggml_row_size((ggml_type) t->type, n_embd);
    const auto* tr = ggml_get_type_traits((ggml_type) t->type);
    const uint8_t* base = (const uint8_t*) file.shard(shard).tensor_data(*t);
    out.assign(tokens.size() * (size_t) n_embd, 0.0f);
    for (size_t p = 0; p < tokens.size(); ++p) {
        if (tokens[p] < 0) {
            err = "negative token id";
            return false;
        }
        tr->to_float(base + (size_t) tokens[p] * row_bytes, out.data() + p * (size_t) n_embd, n_embd);
    }
    return true;
}

/// Device positions: row r of the Q side sits at pos0 + r / n_head.
std::vector<int> positions_for(int n_pos, int n_heads, int pos0) {
    std::vector<int> pos((size_t) n_pos * n_heads);
    for (int p = 0; p < n_pos; ++p)
        for (int h = 0; h < n_heads; ++h) pos[(size_t) p * n_heads + h] = pos0 + p;
    return pos;
}

}  // namespace

bool run_prompt_dense(const strata::GgufModel& file, const std::vector<int32_t>& tokens, int pos0,
                      int n_blocks, std::vector<float>& out, std::vector<double>& sums, std::string& err,
                      Session* session) {
    const int n_pos = (int) tokens.size();
    const int64_t n_embd = (int64_t) file.meta().get("step35.embedding_length")->num();
    const int dense_layers = (int) file.meta().get("step35.leading_dense_block_count")->num();
    const std::string pfx = "step35.";
    const int n_expert = (int) file.meta().get(pfx + "expert_count")->num();
    const int top_k = (int) file.meta().get(pfx + "expert_used_count")->num();
    const int n_ff_exp = (int) file.meta().get(pfx + "expert_feed_forward_length")->num();
    const int n_ff_sh = (int) file.meta().get(pfx + "expert_shared_feed_forward_length")->num();
    // the sliding window, from the file rather than from a constant: every sliding block has it, and a block whose
    // window the cache applied differently would be a quiet difference between the two attention paths
    const strata::MetaValue* wv = file.meta().get(pfx + "attention.sliding_window");
    const int window = wv ? (int) wv->num() : 512;
    const float exp_scale = (float) file.meta().get(pfx + "expert_weights_scale")->num();
    const strata::MetaValue* normv = file.meta().get(pfx + "expert_weights_norm");
    const bool exp_norm = normv && normv->type == strata::MetaType::BOOL && normv->u != 0;
    const int e = (int) n_embd;
    if (!embed_rows(file, tokens, n_embd, out, err)) return false;

    float* d_h = nullptr;
    if (cudaMalloc(&d_h, out.size() * sizeof(float)) != cudaSuccess) {
        err = "out of device memory";
        return false;
    }
    cudaMemcpy(d_h, out.data(), out.size() * sizeof(float), cudaMemcpyHostToDevice);

    for (int layer = 0; layer < n_blocks; ++layer) {
        const bool is_moe = layer >= dense_layers;
        DenseBlock b;
        if (!upload_block(file, layer, b, err, !is_moe)) return false;
        const int hd = b.head_dim, nh = b.n_head, nkv = b.n_head_kv;
        // The q8_1 scratch, for the WIDEST input this block quantizes - not for n_embd.  `proj` quantizes up to
        // kMaxCols columns of an n_in-wide input and n_in is not always n_embd: the FFN down projection reads the
        // 11264-wide activation and `wo` the attention output (8192 on a full block, 12288 on a sliding one).
        // Sized by n_embd alone, this buffer was overrun by up to 96 KB on every prompt pass - heap corruption that
        // answered correctly only while the scratch happened to sit in front of memory nobody was using yet, and
        // that a 76-position pass turned into a GPU memory fault.
        const int q8_in = std::max(e, std::max((int) b.ff, nh * hd));
        void* d_q8 = nullptr;
        float *x = nullptr, *q = nullptr, *k = nullptr, *v = nullptr, *ao = nullptr, *gt = nullptr, *res = nullptr;
        float *g = nullptr, *u = nullptr, *act = nullptr, *d = nullptr;
        int *dpq = nullptr, *dpk = nullptr;
        const size_t nq = (size_t) n_pos * nh * hd, nk = (size_t) n_pos * nkv * hd;
        bool ok = cudaMalloc(&d_q8, (size_t) kMaxCols * gemv_q8_bytes(q8_in)) == cudaSuccess &&
                  cudaMalloc((void**) &x, (size_t) n_pos * e * sizeof(float)) == cudaSuccess &&
                  cudaMalloc((void**) &q, nq * sizeof(float)) == cudaSuccess &&
                  cudaMalloc((void**) &k, nk * sizeof(float)) == cudaSuccess &&
                  cudaMalloc((void**) &v, nk * sizeof(float)) == cudaSuccess &&
                  cudaMalloc((void**) &ao, nq * sizeof(float)) == cudaSuccess &&
                  cudaMalloc((void**) &gt, (size_t) n_pos * nh * sizeof(float)) == cudaSuccess &&
                  cudaMalloc((void**) &res, (size_t) n_pos * e * sizeof(float)) == cudaSuccess &&
                  cudaMalloc((void**) &g, (size_t) n_pos * b.ff * sizeof(float)) == cudaSuccess &&
                  cudaMalloc((void**) &u, (size_t) n_pos * b.ff * sizeof(float)) == cudaSuccess &&
                  cudaMalloc((void**) &act, (size_t) n_pos * b.ff * sizeof(float)) == cudaSuccess &&
                  cudaMalloc((void**) &d, (size_t) n_pos * e * sizeof(float)) == cudaSuccess;
        const std::vector<int> posq = positions_for(n_pos, nh, pos0), posk = positions_for(n_pos, nkv, pos0);
        ok = ok && cudaMalloc((void**) &dpq, posq.size() * sizeof(int)) == cudaSuccess &&
             cudaMalloc((void**) &dpk, posk.size() * sizeof(int)) == cudaSuccess;
        if (!ok) {
            err = "out of device memory";
            return false;
        }
        cudaMemcpy(dpq, posq.data(), posq.size() * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(dpk, posk.data(), posk.size() * sizeof(int), cudaMemcpyHostToDevice);

        // ---- the attention half
        cudaMemcpy(x, d_h, (size_t) n_pos * e * sizeof(float), cudaMemcpyDeviceToDevice);
        per_head_rms_norm(x, n_pos, e, b.attn_norm, b.eps, nullptr);
        ok = proj((ggml_type) b.t_q, b.q, x, q, e, nh * hd, n_pos, d_q8, err) &&
             proj((ggml_type) b.t_k, b.k, x, k, e, nkv * hd, n_pos, d_q8, err) &&
             proj((ggml_type) b.t_v, b.v, x, v, e, nkv * hd, n_pos, d_q8, err);
        if (ok) per_head_rms_norm(q, n_pos * nh, hd, b.q_norm, b.eps, nullptr);
        if (ok) per_head_rms_norm(k, n_pos * nkv, hd, b.k_norm, b.eps, nullptr);
        if (ok) rope_neox(q, n_pos * nh, hd, b.n_rot, dpq, b.base, b.rope_ff, b.n_ff_rope, nullptr);
        if (ok) rope_neox(k, n_pos * nkv, hd, b.n_rot, dpk, b.base, b.rope_ff, b.n_ff_rope, nullptr);
        if (ok && session) {
            // the cache's own geometry, checked rather than assumed: a session built for another model would
            // otherwise write these cells with this model's numbers and answer from them
            if (session->n_head_kv != nkv || session->head_dim != hd || session->page_size != kKvPageSize) {
                err = "the cache was built for " + std::to_string(session->n_head_kv) + " KV heads of " +
                      std::to_string(session->head_dim) + " dims, this block has " + std::to_string(nkv) + " of " +
                      std::to_string(hd);
                ok = false;
            }
            // The window the cache rings this layer with must be the one this block attends over: a block reading a
            // window out of a pool that keeps every position is merely wasteful, but the reverse reads cells that
            // were reused by the position after it - a wrong answer with nothing to show for it.
            const int w_layer = ok && (size_t) layer < session->windows.size() ? session->windows[(size_t) layer] : 0;
            if (ok && (w_layer > 0) != b.sliding) {
                err = "block " + std::to_string(layer) + " reads " + (b.sliding ? "a window" : "the whole prefix") +
                      ", but its cache was built " + (w_layer > 0 ? "with a ring" : "to keep every position");
                ok = false;
            }
            if (ok && b.sliding && w_layer != window) {
                err = "block " + std::to_string(layer) + " has a window of " + std::to_string(w_layer) +
                      ", the file's sliding_window says " + std::to_string(window);
                ok = false;
            }
            if (ok) ok = session_append(*session, layer, pos0, n_pos, k, v, err);
            if (ok) ok = session_attend(*session, layer, q, ao, pos0, n_pos, nh, b.sliding ? window : 0, err);
        }
        if (ok && !session) attn_causal(q, k, v, ao, n_pos, nh, nkv, hd, b.sliding ? window : 0, nullptr);
        if (ok) ok = proj((ggml_type) b.t_gate, b.gate, x, gt, e, nh, n_pos, d_q8, err);
        if (ok) k_gate_rows<<<n_pos * nh, 128>>>(ao, gt, hd);
        if (ok) ok = proj((ggml_type) b.t_wo, b.wo, ao, res, nh * hd, e, n_pos, d_q8, err);
        if (ok) k_add_n<<<(n_pos * e + 255) / 256, 256>>>(d_h, res, n_pos * e);

        if (ok) cudaMemcpy(x, d_h, (size_t) n_pos * e * sizeof(float), cudaMemcpyDeviceToDevice);
        if (ok) per_head_rms_norm(x, n_pos, e, b.ffn_norm, b.eps, nullptr);
        if (ok && !is_moe) {
            // ---- the dense feed-forward
            ok = proj((ggml_type) b.t_ffn, b.ffn_gate, x, g, e, b.ff, n_pos, d_q8, err);
            if (ok) ok = proj((ggml_type) b.t_ffn, b.ffn_up, x, u, e, b.ff, n_pos, d_q8, err);
            if (ok) cudaMemcpy(act, g, (size_t) n_pos * b.ff * sizeof(float), cudaMemcpyDeviceToDevice);
            if (ok) k_silu_mul_n<<<(n_pos * b.ff + 255) / 256, 256>>>(act, u, n_pos * b.ff, 0.0f);
            if (ok) ok = proj((ggml_type) b.t_ffn, b.ffn_down, act, d, b.ff, e, n_pos, d_q8, err);
            if (ok) k_add_n<<<(n_pos * e + 255) / 256, 256>>>(d_h, d, n_pos * e);
        } else if (ok) {
            // ---- the mixture of experts, ONE POSITION AT A TIME (each position routes on its own)
            const std::string pre = "blk." + std::to_string(layer) + ".";
            size_t shard_g = 0, shard_d = 0;
            const strata::TensorInfo* tg = file.find(pre + "ffn_gate_exps.weight", &shard_g);
            const strata::TensorInfo* td = file.find(pre + "ffn_down_exps.weight", &shard_d);
            if (!tg || !td) { err = "missing expert tensor"; ok = false; }
            const size_t gate_bytes = ok ? strata::tensor_payload_bytes(*tg) / (size_t) n_expert : 0;
            const size_t down_bytes = ok ? strata::tensor_payload_bytes(*td) / (size_t) n_expert : 0;
            const uint8_t* base_g = ok ? (const uint8_t*) file.shard(shard_g).tensor_data(*tg) : nullptr;
            const uint8_t* base_d = ok ? (const uint8_t*) file.shard(shard_d).tensor_data(*td) : nullptr;
            size_t shard_u = 0;
            const strata::TensorInfo* tu = file.find(pre + "ffn_up_exps.weight", &shard_u);
            const uint8_t* base_u = tu ? (const uint8_t*) file.shard(shard_u).tensor_data(*tu) : nullptr;
            if (!tu) { err = "missing ffn_up_exps"; ok = false; }
            size_t rb = 0;
            void* router = nullptr;
            float* sbias = nullptr;
            void *sg = nullptr, *su = nullptr, *sd = nullptr;
            float *logits = nullptr, *eg = nullptr, *eu = nullptr, *eact = nullptr, *ed = nullptr, *eacc = nullptr;
            int32_t* ids = nullptr;
            float* wts = nullptr;
            void *dg = nullptr, *du = nullptr, *dd = nullptr;
            int t_sh = 0;
            if (ok) {
                try {
                    router = upload(file, pre + "ffn_gate_inp.weight", rb);
                    t_sh = (int) file.find(pre + "ffn_gate_shexp.weight")->type;
                    if (file.find(pre + "exp_probs_b.bias")) sbias = (float*) upload(file, pre + "exp_probs_b.bias", rb);
                    sg = upload(file, pre + "ffn_gate_shexp.weight", rb);
                    su = upload(file, pre + "ffn_up_shexp.weight", rb);
                    sd = upload(file, pre + "ffn_down_shexp.weight", rb);
                } catch (const std::exception& ex) {
                    err = ex.what();
                    ok = false;
                }
            }
            ok = ok && cudaMalloc((void**) &logits, (size_t) n_expert * sizeof(float)) == cudaSuccess &&
                 cudaMalloc((void**) &eg, (size_t) n_ff_exp * sizeof(float)) == cudaSuccess &&
                 cudaMalloc((void**) &eu, (size_t) n_ff_exp * sizeof(float)) == cudaSuccess &&
                 cudaMalloc((void**) &eact, (size_t) n_ff_sh * sizeof(float)) == cudaSuccess &&
                 cudaMalloc((void**) &ed, (size_t) e * sizeof(float)) == cudaSuccess &&
                 cudaMalloc((void**) &eacc, (size_t) e * sizeof(float)) == cudaSuccess &&
                 cudaMalloc((void**) &ids, (size_t) top_k * sizeof(int32_t)) == cudaSuccess &&
                 cudaMalloc((void**) &wts, (size_t) top_k * sizeof(float)) == cudaSuccess &&
                 cudaMalloc(&dg, gate_bytes) == cudaSuccess && cudaMalloc(&du, gate_bytes) == cudaSuccess &&
                 cudaMalloc(&dd, down_bytes) == cudaSuccess;
            float clamp_exp = 0, clamp_sh = 0;
            for (const auto& c : {std::make_pair("swiglu_clamp_exp", &clamp_exp),
                                  std::make_pair("swiglu_clamp_shexp", &clamp_sh)}) {
                const strata::MetaValue* v = file.meta().get(pfx + c.first);
                if (v && v->type == strata::MetaType::ARRAY && (int) v->items.size() > layer)
                    *c.second = (float) v->items[(size_t) layer].num();
            }
            std::vector<int32_t> h_ids((size_t) top_k);
            std::vector<float> h_w((size_t) top_k);
            for (int p = 0; ok && p < n_pos; ++p) {
                const float* xp = x + (size_t) p * e;
                float* hp = d_h + (size_t) p * e;
                ok = gemv(GGML_TYPE_F32, router, xp, logits, e, n_expert, d_q8, nullptr);
                if (ok) moe_route(logits, sbias, 1, n_expert, top_k, exp_scale, exp_norm, ids, wts, nullptr);
                if (ok) ok = cudaMemcpy(h_ids.data(), ids, h_ids.size() * sizeof(int32_t), cudaMemcpyDeviceToHost) == cudaSuccess;
                if (ok) ok = cudaMemcpy(h_w.data(), wts, h_w.size() * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess;
                if (ok) ok = cudaMemset(eacc, 0, (size_t) e * sizeof(float)) == cudaSuccess;
                for (int k = 0; ok && k < top_k; ++k) {
                    const size_t ex = (size_t) h_ids[(size_t) k];
                    ok = cudaMemcpy(dg, base_g + ex * gate_bytes, gate_bytes, cudaMemcpyHostToDevice) == cudaSuccess;
                    if (ok) ok = cudaMemcpy(du, base_u + ex * gate_bytes, gate_bytes, cudaMemcpyHostToDevice) == cudaSuccess;
                    if (ok) ok = cudaMemcpy(dd, base_d + ex * down_bytes, down_bytes, cudaMemcpyHostToDevice) == cudaSuccess;
                    if (ok) ok = gemv((ggml_type) tg->type, dg, xp, eg, e, n_ff_exp, d_q8, nullptr);
                    if (ok) ok = gemv((ggml_type) tg->type, du, xp, eu, e, n_ff_exp, d_q8, nullptr);
                    if (ok) ok = cudaMemcpy(eact, eg, (size_t) n_ff_exp * sizeof(float), cudaMemcpyDeviceToDevice) == cudaSuccess;
                    if (ok) k_silu_mul_n<<<(n_ff_exp + 255) / 256, 256>>>(eact, eu, n_ff_exp, clamp_exp);
                    if (ok) ok = gemv((ggml_type) tg->type, dd, eact, ed, n_ff_exp, e, d_q8, nullptr);
                    if (ok) k_acc_scaled_rows<<<(e + 255) / 256, 256>>>(eacc, ed, h_w[(size_t) k], e);
                }
                if (ok) ok = gemv((ggml_type) t_sh, sg, xp, eg, e, n_ff_sh, d_q8, nullptr);
                if (ok) ok = gemv((ggml_type) t_sh, su, xp, eu, e, n_ff_sh, d_q8, nullptr);
                if (ok) ok = cudaMemcpy(eact, eg, (size_t) n_ff_sh * sizeof(float), cudaMemcpyDeviceToDevice) == cudaSuccess;
                if (ok) k_silu_mul_n<<<(n_ff_sh + 255) / 256, 256>>>(eact, eu, n_ff_sh, clamp_sh);
                if (ok) ok = gemv((ggml_type) t_sh, sd, eact, ed, n_ff_sh, e, d_q8, nullptr);
                if (ok) k_acc_scaled_rows<<<(e + 255) / 256, 256>>>(eacc, ed, 1.0f, e);
                if (ok) k_add_n<<<(e + 255) / 256, 256>>>(hp, eacc, e);
            }
            for (void* q : {router, (void*) sbias, sg, su, sd, (void*) logits, (void*) eg, (void*) eu,
                            (void*) eact, (void*) ed, (void*) eacc, (void*) ids, (void*) wts, dg, du, dd})
                if (q) cudaFree(q);
        }
        if (ok && cudaDeviceSynchronize() != cudaSuccess) ok = false;
        if (!ok && err.empty()) err = "a kernel failed";
        for (void* p : {(void*) x, (void*) q, (void*) k, (void*) v, (void*) ao, (void*) gt, (void*) res,
                        (void*) g, (void*) u, (void*) act, (void*) d, (void*) dpq, (void*) dpk, d_q8})
            if (p) cudaFree(p);
        release_block(b);
        if (!ok) return false;
        // what this block handed on, summed over every position: the figure the reference prints
        std::vector<float> check((size_t) n_pos * e);
        cudaMemcpy(check.data(), d_h, check.size() * sizeof(float), cudaMemcpyDeviceToHost);
        double s = 0;
        for (float val : check) s += (double) val;
        sums.push_back(s);
    }
    out.resize((size_t) n_pos * e);
    cudaMemcpy(out.data(), d_h, out.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_h);
    return true;
}

}  // namespace strata::step35::kernels
