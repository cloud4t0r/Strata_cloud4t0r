// src/decider/gpu_kernels.cuh - the decider's kernels, declared once so the full-model path can run them.
//
// Each kernel is defined in the .cu file that owns it (gpu_mlp.cu, gpu_gdn.cu, gpu_attn.cu) and also used by
// gpu_model.cu, which walks the 32 blocks.  They are `__global__` and linked across translation units, so the
// declarations have to be identical - this file is the single copy.
//
// Internal to src/decider: the public surface is include/strata/decider/gpu.hpp.
#pragma once

#include <cstddef>

namespace strata::decider::gpu::kernels {

/// out = x / rms(x) * w, one block per row of `n`.
__global__ void rmsnorm(const float* x, const float* w, float* y, int n, float eps);

/// out = silu(gate) * up, elementwise.
__global__ void swiglu(const float* gate, const float* up, float* out, size_t n);

/// x += y, elementwise (the residual stream).
__global__ void add_in_place(float* x, const float* y, size_t n);

/// The causal depthwise convolution with `k` taps (tap 0 oldest) and SiLU, over [tokens][channels].
__global__ void conv1d_silu(const float* x, const float* w, float* y, int tokens, int channels, int k);

/// softplus with the reference's threshold, so a large a + dt_bias does not overflow to infinity.
__device__ __forceinline__ float softplus(float x) { return x > 20.0f ? x : log1pf(expf(x)); }

/// The scan's block shape: 16 x 16 threads, each holding an 8 x 8 tile of the 128 x 128 state.
constexpr int kHeads = 16;
constexpr int kTile = 8;

/// The delta rule for one value head over the whole sequence; grid = v heads, block = kHeads x kHeads, kTile^2
/// tiles.
__global__ void gdn_scan(const float* conv, const float* a, const float* b, const float* ssm_a, const float* dt,
                         float* core, int tokens, int v_heads, int k_heads, int k_dim, int v_dim, int key_dim,
                         int conv_dim, float scale, float eps);

/// out = rms(x) * w * silu(z), one block per row of `n` (a value head's width).
__global__ void gated_norm(const float* x, const float* z, const float* w, float* out, int n, float eps);

/// RMSNorm over one head then partial RoPE; one block per (token, head).
__global__ void norm_rope(float* buffer, const float* w, int per_token, int row_stride, int head_dim, int n_rot,
                          float theta, float eps);

/// Non-causal grouped-query attention with the per-head gate; one block per query head.
__global__ void attention(const float* qg, const float* k, const float* v, float* attn, int tokens, int heads,
                          int kv_heads, int head_dim, float scale);

}  // namespace strata::decider::gpu::kernels
