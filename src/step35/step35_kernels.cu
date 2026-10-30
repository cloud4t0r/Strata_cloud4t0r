// src/step35/step35_kernels.cu - the Step-3.7-Flash block pieces strata_kernels does not carry: per-head
// RMSNorm and the NEOX partial RoPE with frequency factors.  See include/strata/step35/kernels.hpp for the two
// conventions this file exists to pin down.
#include "strata/step35/kernels.hpp"

#include <cuda_runtime.h>

#include <cmath>

namespace strata::step35::kernels {
namespace {

__device__ float block_reduce_sum(float v, float* sh) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
    if (lane == 0) sh[warp] = v;
    __syncthreads();
    const int warps = (blockDim.x + 31) >> 5;
    v = threadIdx.x < warps ? sh[threadIdx.x] : 0.0f;
    if (warp == 0) {
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
    }
    return v;
}

/// One block per head.  head_dim is 128 for this model, so one warp-group of 128 threads is enough.
__global__ void k_rms_norm(float* x, int head_dim, const float* w, float eps) {
    float* r = x + (size_t) blockIdx.x * head_dim;
    __shared__ float sh[32];
    float ss = 0;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) ss += r[i] * r[i];
    ss = block_reduce_sum(ss, sh);
    __shared__ float inv;
    if (threadIdx.x == 0) inv = rsqrtf(ss / (float) head_dim + eps);
    __syncthreads();
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) r[i] = r[i] * inv * w[i];
}

/// One block per row, and each row carries its own position (a batch mixes prompt and decode rows).
/// Element j turns with element j + half, and the tail beyond n_rot passes through.
__global__ void k_rope(float* x, int head_dim, int n_rot, const int* positions, double base, const float* ff,
                       int n_ff) {
    const int row = blockIdx.x;
    float* r = x + (size_t) row * head_dim;
    const int pos = positions[row];
    const int half = n_rot / 2;
    for (int j = threadIdx.x; j < half; j += blockDim.x) {
        double theta = (double) pos * pow(base, -2.0 * (double) j / (double) n_rot);
        if (ff != nullptr && j < n_ff && ff[j] != 0.0f) theta /= (double) ff[j];
        const double c = cos(theta), s = sin(theta);
        const float a = r[j], b = r[j + half];
        rope_pair((double) a, (double) b, c, s, r[j], r[j + half]);
    }
}

}  // namespace

void per_head_rms_norm(float* x, int rows, int head_dim, const float* weight, float eps, void* stream) {
    const int threads = head_dim < 128 ? head_dim : (head_dim > 512 ? 512 : head_dim);
    k_rms_norm<<<rows, threads, 0, (cudaStream_t) stream>>>(x, head_dim, weight, eps);
}

void rope_neox(float* x, int rows, int head_dim, int n_rot, const int* positions, double base, const float* ff,
               int n_ff, void* stream) {
    const int threads = n_rot / 2 < 128 ? (n_rot / 2 > 32 ? n_rot / 2 : 32) : 128;
    k_rope<<<rows, threads, 0, (cudaStream_t) stream>>>(x, head_dim, n_rot, positions, base, ff, n_ff);
}

}  // namespace strata::step35::kernels
