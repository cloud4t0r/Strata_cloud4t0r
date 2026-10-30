// src/step35/step35_attn.cu - causal GQA attention with the sliding window.  See attn.hpp.
//
// One warp per (position, query head).  The softmax is ONLINE: the running max, the denominator and the weighted
// value vector live in registers, so the kernel has no bound on how many keys a row may see.  It used to keep the
// scores of the window in a `scores[512]` of shared memory and clamp the window to that size, which was wrong in
// two ways that a short test cannot see: a full-attention block at position 512 wrote one score past the end of
// the array, and the host clamp turned a window wider than 512 into 512.  Measured against a float64 host, the
// overrun was 1.6e-03 off at 513 keys and 5.0e-02 at 1024, with no CUDA error at all.
//
// The lanes are laid out like `attn_q8_cached` lays them out: lane l owns dimensions l, l+32, l+64, ... so the
// accumulators are per-lane and a head_dim that is not a multiple of 32 costs nothing but idle lanes.
#include "strata/step35/attn.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace strata::step35::kernels {
namespace {

__device__ float warp_sum(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
    return v;
}

constexpr int kMaxAcc = 8;   // 32 lanes x 8 accumulators = head_dim 256, the widest this kernel carries

__global__ void k_attn_causal(const float* q, const float* k, const float* v, float* out, int n_pos, int n_head,
                              int n_head_kv, int head_dim, int window) {
    const int pos = blockIdx.x, h = blockIdx.y;
    const int lane = threadIdx.x;
    const int group = n_head / n_head_kv;
    const int kvh = h / group;
    int first = 0;
    if (window > 0 && pos - first + 1 > window) first = pos - window + 1;

    const float* qh = q + ((size_t) pos * n_head + h) * head_dim;
    const float scale = (float) (1.0 / sqrt((double) head_dim));

    // Every `acc[i]` has a constant index and its own dimension is `lane + 32 * i`, so the array stays in
    // registers: a runtime index would put it in local memory, which is the one thing this loop cannot afford.
    float acc[kMaxAcc];
#pragma unroll
    for (int i = 0; i < kMaxAcc; ++i) acc[i] = 0.0f;
    float mx = -INFINITY, den = 0.0f;

    for (int j = first; j <= pos; ++j) {
        const float* kj = k + ((size_t) j * n_head_kv + kvh) * head_dim;
        float s = 0.0f;
#pragma unroll
        for (int i = 0; i < kMaxAcc; ++i) {
            const int d = lane + 32 * i;
            if (d < head_dim) s += qh[d] * kj[d];
        }
        s = warp_sum(s) * scale;
        // every lane holds the same s after the reduction, so one rescale per lane is enough
        const float mn = fmaxf(mx, s);
        const float c = (mx == -INFINITY) ? 0.0f : expf(mx - mn);
        const float p = expf(s - mn);
        den = den * c + p;
        mx = mn;
        const float* vj = v + ((size_t) j * n_head_kv + kvh) * head_dim;
#pragma unroll
        for (int i = 0; i < kMaxAcc; ++i) {
            const int d = lane + 32 * i;
            if (d < head_dim) acc[i] = acc[i] * c + p * vj[d];
        }
    }

    float* o = out + ((size_t) pos * n_head + h) * head_dim;
    const float inv = den > 0.0f ? 1.0f / den : 0.0f;
#pragma unroll
    for (int i = 0; i < kMaxAcc; ++i) {
        const int d = lane + 32 * i;
        if (d < head_dim) o[d] = acc[i] * inv;
    }
}

}  // namespace

void attn_causal(const float* q, const float* k, const float* v, float* out, int n_pos, int n_head, int n_head_kv,
                 int head_dim, int window, void* stream) {
    if (head_dim <= 0 || head_dim > 32 * kMaxAcc) {
        std::fprintf(stderr, "attn_causal: head_dim %d must be positive and at most %d\n", head_dim, 32 * kMaxAcc);
        std::exit(1);
    }
    if (n_head_kv <= 0 || n_head % n_head_kv != 0) {
        std::fprintf(stderr, "attn_causal: %d query heads over %d KV heads\n", n_head, n_head_kv);
        std::exit(1);
    }
    if (n_pos <= 0) return;
    k_attn_causal<<<dim3(n_pos, n_head), 32, 0, (cudaStream_t) stream>>>(q, k, v, out, n_pos, n_head, n_head_kv,
                                                                        head_dim, window);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) std::fprintf(stderr, "attn_causal launch: %s\n", cudaGetErrorString(e));
}

}  // namespace strata::step35::kernels
