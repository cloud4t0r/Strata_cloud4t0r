// src/step35/step35_attn.cu - causal GQA attention with the sliding window.  See attn.hpp.
//
// One warp per (position, query head).  The scores of the window live in shared memory: the window is at most
// 512 keys, so 2 KB, and carrying them means the dot products are computed once instead of once per pass.
#include "strata/step35/attn.hpp"

#include <cuda_runtime.h>

namespace strata::step35::kernels {
namespace {

__device__ float warp_max(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
    return v;
}

__device__ float warp_sum(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
    return v;
}

constexpr int kMaxWindow = 512;   // the release's sliding_window; a longer prefix gets a different kernel

__global__ void k_attn_causal(const float* q, const float* k, const float* v, float* out, int n_pos, int n_head,
                              int n_head_kv, int head_dim, int window) {
    const int pos = blockIdx.x, h = blockIdx.y;
    const int lane = threadIdx.x;
    const int group = n_head / n_head_kv;
    const int kvh = h / group;
    int first = 0;
    if (window > 0 && pos - first + 1 > window) first = pos - window + 1;
    const int nkeys = pos - first + 1;

    __shared__ float scores[kMaxWindow];
    const float* qh = q + ((size_t) pos * n_head + h) * head_dim;
    const double scale = 1.0 / sqrt((double) head_dim);

    // pass 1: the score of every key in the window, and the max for a stable softmax
    float mx = -INFINITY;
    for (int j = first; j <= pos; ++j) {
        const float* kj = k + ((size_t) j * n_head_kv + kvh) * head_dim;
        float acc = 0;
        for (int d = lane; d < head_dim; d += 32) acc += qh[d] * kj[d];
        acc = warp_sum(acc);
        if (lane == 0) {
            scores[j - first] = acc * (float) scale;
            mx = fmaxf(mx, acc * (float) scale);
        }
    }
    __syncwarp();
    mx = warp_max(mx);

    // softmax denominator
    float denom = 0;
    for (int i = lane; i < nkeys; i += 32) denom += expf(scores[i] - mx);
    denom = warp_sum(denom);

    // pass 2: the weighted sum of the values
    for (int d = lane; d < head_dim; d += 32) {
        float acc = 0;
        for (int j = first; j <= pos; ++j) {
            const float* vj = v + ((size_t) j * n_head_kv + kvh) * head_dim;
            acc += (expf(scores[j - first] - mx) / denom) * vj[d];
        }
        out[((size_t) pos * n_head + h) * head_dim + d] = acc;
    }
}

}  // namespace

void attn_causal(const float* q, const float* k, const float* v, float* out, int n_pos, int n_head, int n_head_kv,
                 int head_dim, int window, void* stream) {
    if (window > kMaxWindow) window = kMaxWindow;
    k_attn_causal<<<dim3(n_pos, n_head), 32, 0, (cudaStream_t) stream>>>(q, k, v, out, n_pos, n_head, n_head_kv,
                                                                        head_dim, window);
}

}  // namespace strata::step35::kernels
