// include/strata/step35/moe.hpp - the router of a Step-3.7-Flash mixture-of-experts block.
//
// The order of operations is llama.cpp's build_moe_ffn in its SIGMOID branch, and every step of it matters:
//
//   1. probs     = sigmoid(logits)                       the routing probabilities
//   2. selection = probs + exp_probs_b                   the per-expert SELECTION bias (DeepSeek V3's); it
//                                                       chooses the experts and does NOT enter their weights
//   3. top-k of selection, largest first, ties to the LOWEST index (made a rule here: an unstable order would
//      make two runs of the same model answer differently, and would make a parity test unreproducible)
//   4. weights   = probs[those k]                        the UNBIASED probabilities
//   5. weights  /= max(sum, 6.103515625e-5)              expert_weights_norm, the clamp being llama.cpp's
//   6. weights  *= expert_weights_scale                  (3.0 in this release)
//
// The shared expert is NOT part of this: it is added unweighted by the block.
#pragma once

#include <cstdint>

namespace strata::step35::kernels {

/// Routes `rows` tokens.  `logits` and `probs_out` are [rows][n_expert]; `bias` is the selection bias or null.
/// `ids` receives [rows][top_k] expert indices (largest selection score first) and `weights` [rows][top_k].
/// `weights_out` (optional) receives the sigmoid probabilities themselves, for a caller that wants them.
void moe_route(const float* logits, const float* bias, int rows, int n_expert, int top_k, float scale,
               bool weights_norm, int32_t* ids, float* weights, void* stream);

}  // namespace strata::step35::kernels
