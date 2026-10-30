// src/step35/step35_moe.cu - the router of a Step-3.7-Flash block.  See include/strata/step35/moe.hpp for the
// order of operations, which is llama.cpp's and is the whole point of this file.
#include "strata/step35/moe.hpp"

#include <cuda_runtime.h>

namespace strata::step35::kernels {
namespace {

/// One warp per token.  Each lane holds the experts at lane, lane + 32, ... and the k passes take the largest
/// remaining score; ties go to the lower index in every reduction, so the answer does not depend on the order
/// the hardware happened to shuffle in.
__global__ void k_moe_route(const float* logits, const float* bias, int n_expert, int top_k, float scale,
                            bool weights_norm, int32_t* ids, float* weights) {
    const int row = blockIdx.x;
    const int lane = threadIdx.x;
    const float* lr = logits + (size_t) row * n_expert;

    // the probabilities and the selection scores, recomputed per pass (n_expert is small: 288)
    auto prob = [&](int e) { return 1.0f / (1.0f + expf(-lr[e])); };
    auto score = [&](int e) { return prob(e) + (bias ? bias[e] : 0.0f); };

    for (int k = 0; k < top_k; ++k) {
        // the best not-yet-taken expert over the whole block
        float best = -INFINITY;
        int best_i = -1;
        for (int e = lane; e < n_expert; e += 32) {
            bool taken = false;
            for (int j = 0; j < k; ++j) taken = taken || (ids[(size_t) row * top_k + j] == e);
            if (taken) continue;
            const float s = score(e);
            if (s > best || (s == best && (best_i < 0 || e < best_i))) {
                best = s;
                best_i = e;
            }
        }
        // reduce the (score, index) pair across the warp, ties to the lower index
        for (int off = 16; off > 0; off >>= 1) {
            const float o = __shfl_down_sync(0xffffffffu, best, off);
            const int oi = __shfl_down_sync(0xffffffffu, best_i, off);
            if (o > best || (o == best && oi >= 0 && (best_i < 0 || oi < best_i))) {
                best = o;
                best_i = oi;
            }
        }
        if (lane == 0) ids[(size_t) row * top_k + k] = best_i;
        __syncwarp();
    }

    if (lane == 0) {
        double sum = 0;
        for (int k = 0; k < top_k; ++k) {
            const int e = ids[(size_t) row * top_k + k];
            const double w = (double) prob(e);
            weights[(size_t) row * top_k + k] = (float) w;
            sum += w;
        }
        if (weights_norm) {
            const double den = sum < 6.103515625e-5 ? 6.103515625e-5 : sum;   // llama.cpp's clamp
            for (int k = 0; k < top_k; ++k) weights[(size_t) row * top_k + k] = (float) ((double) weights[(size_t) row * top_k + k] / den);
        }
        if (scale != 0.0f && scale != 1.0f) {
            for (int k = 0; k < top_k; ++k) weights[(size_t) row * top_k + k] *= scale;
        }
    }
}

}  // namespace

void moe_route(const float* logits, const float* bias, int rows, int n_expert, int top_k, float scale,
               bool weights_norm, int32_t* ids, float* weights, void* stream) {
    k_moe_route<<<rows, 32, 0, (cudaStream_t) stream>>>(logits, bias, n_expert, top_k, scale, weights_norm, ids,
                                                        weights);
}

}  // namespace strata::step35::kernels
