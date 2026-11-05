// include/strata/decider/gpu.hpp - the decider's GPU kernels, one stage at a time.
//
// WHAT IS HERE TODAY.  The dense MLP of one block: the post-attention RMSNorm, the SwiGLU (gate and up as two
// quantized matrix products, SiLU(gate) * up, then down), over a whole token sequence at once.  It takes the
// hidden state the CPU reference wrote and gives back the same shape, so `--ref --dump-dir` and this binary can be
// compared element by element (tools/decider_block_parity.py).
//
// WHY A STAGE AT A TIME.  The decider has four kinds of kernel (norms and dense projections, the Gated DeltaNet
// scan, the non-causal attention, the readout) and each one has a silent failure mode.  Comparing them separately
// against the reference localises a mistake to a kernel instead of to "the logits are off".
//
// The weights are read from the GGUF exactly as the CPU reference reads them (Q8_0 rows through ggml's traits),
// and the matrix product is the engine's own `iq_mmvq`, so this is the arithmetic the final path will run.
#pragma once

#include <cstdint>
#include <string>

namespace strata::decider::gpu {

/// Runs block `block`'s dense MLP over a `[tokens][n_embd]` hidden state.
/// `in_path` is raw little-endian f32, row-major, as `strata-decider --ref --dump-dir` writes it ("xn_post_N.f32");
/// `out_path` receives the block's MLP output in the same layout (what the reference calls "mlp_N.f32").
/// `xn_path`, when not empty, also receives the normalised input the projections saw: the first stage a wrong
/// result can come from, so it is worth being able to compare it on its own.
bool mlp(const std::string& gguf, int64_t block, const std::string& in_path, const std::string& out_path,
         std::string& err, const std::string& xn_path = {});

}  // namespace strata::decider::gpu
