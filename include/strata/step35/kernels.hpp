// include/strata/step35/kernels.hpp - the Step-3.7-Flash device kernels that strata_kernels does not already
// carry: the per-head RMSNorm of Q and K, and the NEOX partial RoPE with llama3's frequency factors.
//
// Two conventions are pinned here rather than in each caller, because both are silent when wrong:
//
//   * PAIRING IS NEOX.  Element j turns with element j + n_rot/2, because llama_model_rope_type() answers
//     LLAMA_ROPE_TYPE_NEOX for LLM_ARCH_STEP35 and ggml reads that as (i0, i0 + n_dims/2).  The adjacent
//     pair (2i, 2i+1) is the NORM convention used by Llama and Qwen: at the positions of a short test the two
//     answer within the noise of a quantized matmul, and the mistake only shows in a long context.
//   * THE ANGLE IS DIVIDED BY rope_freqs[j].  That table (llama3 scaling, entries 14-31 of 64 in the release)
//     applies to the FULL-attention blocks only; on sliding blocks the caller passes no table.  It shifts the
//     angle by 0.79 rad at position 4096 and 25.9 at 131072 for a full block, so it is not a rounding detail.
//
// The angle is computed in DOUBLE on the device: one pow, one cos and one sin per pair per row, which costs
// nothing beside the matmuls and keeps the kernel answerable to a float64 host reference without a tolerance
// loose enough to hide a permutation.
#pragma once

#include <cstdint>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define STRATA_STEP35_HD __host__ __device__
#else
#define STRATA_STEP35_HD
#endif

namespace strata::step35::kernels {

/// ONE NEOX PAIR: (a, b) -> (a*cos - b*sin, a*sin + b*cos).  Shared by the kernel and by the host reference
/// of the parity test, so the pairing cannot be written down two ways.
STRATA_STEP35_HD inline void rope_pair(double a, double b, double c, double s, float& oa, float& ob) {
    oa = (float) (a * c - b * s);
    ob = (float) (a * s + b * c);
}

/// x is `rows` rows of `head_dim` floats, normalised in place: y = x / sqrt(mean(x^2) + eps) * weight[i].
/// `weight` is the block's attn_q_norm / attn_k_norm tensor (head_dim floats); it may not be null.
void per_head_rms_norm(float* x, int rows, int head_dim, const float* weight, float eps, void* stream);

/// Rotates the first `n_rot` of every row in place.  `positions` holds one position per row; `ff` is the
/// model's rope_freqs table (`n_ff` entries) or null for a block that does not use it.
void rope_neox(float* x, int rows, int head_dim, int n_rot, const int* positions, double base, const float* ff,
               int n_ff, void* stream);

}  // namespace strata::step35::kernels
