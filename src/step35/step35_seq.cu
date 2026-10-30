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
// Scope: the dense blocks.  The mixture-of-experts blocks route PER POSITION, so they go through the same
// machinery one position at a time; that is the next change, and it is named rather than half-done here.
#include "strata/step35/attn.hpp"
#include "strata/step35/block.hpp"
#include "strata/step35/gemv.hpp"
#include "strata/step35/kernels.hpp"

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
                      int n_blocks, std::vector<float>& out, std::vector<double>& sums, std::string& err) {
    const int n_pos = (int) tokens.size();
    const int64_t n_embd = (int64_t) file.meta().get("step35.embedding_length")->num();
    const int e = (int) n_embd;
    if (!embed_rows(file, tokens, n_embd, out, err)) return false;

    float* d_h = nullptr;
    void* d_q8 = nullptr;
    if (cudaMalloc(&d_h, out.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_q8, (size_t) kMaxCols * gemv_q8_bytes(e)) != cudaSuccess) {
        err = "out of device memory";
        return false;
    }
    cudaMemcpy(d_h, out.data(), out.size() * sizeof(float), cudaMemcpyHostToDevice);

    for (int layer = 0; layer < n_blocks; ++layer) {
        DenseBlock b;
        if (!upload_block(file, layer, b, err)) return false;
        const int hd = b.head_dim, nh = b.n_head, nkv = b.n_head_kv;
        float *x = nullptr, *q = nullptr, *k = nullptr, *v = nullptr, *ao = nullptr, *gt = nullptr, *res = nullptr;
        float *g = nullptr, *u = nullptr, *act = nullptr, *d = nullptr;
        int *dpq = nullptr, *dpk = nullptr;
        const size_t nq = (size_t) n_pos * nh * hd, nk = (size_t) n_pos * nkv * hd;
        bool ok = cudaMalloc((void**) &x, (size_t) n_pos * e * sizeof(float)) == cudaSuccess &&
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
        if (ok) attn_causal(q, k, v, ao, n_pos, nh, nkv, hd, b.sliding ? 512 : 0, nullptr);
        if (ok) ok = proj((ggml_type) b.t_gate, b.gate, x, gt, e, nh, n_pos, d_q8, err);
        if (ok) k_gate_rows<<<n_pos * nh, 128>>>(ao, gt, hd);
        if (ok) ok = proj((ggml_type) b.t_wo, b.wo, ao, res, nh * hd, e, n_pos, d_q8, err);
        if (ok) k_add_n<<<(n_pos * e + 255) / 256, 256>>>(d_h, res, n_pos * e);

        // ---- the dense feed-forward
        if (ok) cudaMemcpy(x, d_h, (size_t) n_pos * e * sizeof(float), cudaMemcpyDeviceToDevice);
        if (ok) per_head_rms_norm(x, n_pos, e, b.ffn_norm, b.eps, nullptr);
        if (ok) ok = proj((ggml_type) b.t_ffn, b.ffn_gate, x, g, e, b.ff, n_pos, d_q8, err);
        if (ok) ok = proj((ggml_type) b.t_ffn, b.ffn_up, x, u, e, b.ff, n_pos, d_q8, err);
        if (ok) cudaMemcpy(act, g, (size_t) n_pos * b.ff * sizeof(float), cudaMemcpyDeviceToDevice);
        if (ok) k_silu_mul_n<<<(n_pos * b.ff + 255) / 256, 256>>>(act, u, n_pos * b.ff, 0.0f);
        if (ok) ok = proj((ggml_type) b.t_ffn, b.ffn_down, act, d, b.ff, e, n_pos, d_q8, err);
        if (ok) k_add_n<<<(n_pos * e + 255) / 256, 256>>>(d_h, d, n_pos * e);
        if (ok && cudaDeviceSynchronize() != cudaSuccess) ok = false;
        if (!ok && err.empty()) err = "a kernel failed";
        for (void* p : {(void*) x, (void*) q, (void*) k, (void*) v, (void*) ao, (void*) gt, (void*) res,
                        (void*) g, (void*) u, (void*) act, (void*) d, (void*) dpq, (void*) dpk})
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
    cudaFree(d_q8);
    return true;
}

}  // namespace strata::step35::kernels
