// include/strata/step35/gemv.hpp - one quantized matrix-vector product for the Step-3.7-Flash blocks, on the
// kernels that already exist (strata_kernels: iq_mmvq + quantize_q8_1_rows, the path the Qwen3.6 and MiniMax-M2
// engines take).  Nothing here re-implements a quantized dot product.
//
// The release's matrices are IQ3_XXS (18), IQ2_S (22), Q4_K (12), IQ3_S (21) and Q5_K (13) - every one of them
// in iq_mmvq's table - plus F32 for the norms, which is a plain float dot product.
#pragma once

#include <ggml.h>

#include <cstddef>
#include <cstdint>

namespace strata::step35::kernels {

/// Bytes of Q8_1 scratch one activation row of `n_in` floats needs (n_in a multiple of 32).
std::size_t gemv_q8_bytes(int n_in);

/// y[r] = W[r] . x for one row: `weights` is a device GGUF block matrix (row-major, `n_in` contiguous values per
/// row), `q8_scratch` the device buffer above.  The activation is quantized to Q8_1 first, as llama.cpp does,
/// so the answer carries that quantization (relative error ~1e-3, not bit-exact with an fp32 dot).
/// Returns false (and launches nothing) for a type neither path carries.
bool gemv(ggml_type type, const void* weights, const float* x, float* y, int n_in, int n_out, void* q8_scratch,
          void* stream);

/// The same over `ncols` activation rows (1..8), each `n_in` apart; y column c is `y + c * n_out`.
bool gemv_multi(ggml_type type, const void* weights, const float* x, float* y, int n_in, int n_out, int ncols,
                void* q8_scratch, void* stream);

}  // namespace strata::step35::kernels
