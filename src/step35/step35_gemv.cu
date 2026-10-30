// src/step35/step35_gemv.cu - the quantized matrix-vector product for the Step-3.7-Flash blocks, on the
// kernels strata_kernels already carries.  See include/strata/step35/gemv.hpp.
#include "strata/step35/gemv.hpp"

#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>
#include <ggml.h>

namespace strata::step35::kernels {
namespace {

/// A plain F32 row dot, for the tensors the release keeps as floats.  One block per output row, one warp per
/// dot: n_in is 4096 or 11264 here, so a warp-strided loop with a shuffle reduction is enough.
__global__ void k_dot_f32(const float* w, const float* x, float* y, int n_in, int n_out) {
    const int r = blockIdx.x;
    const float* row = w + (size_t) r * n_in;
    float acc = 0;
    for (int i = threadIdx.x; i < n_in; i += blockDim.x) acc += row[i] * x[i];
    __shared__ float sh[32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) sh[warp] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        const int warps = (blockDim.x + 31) >> 5;
        float total = 0;
        for (int i = 0; i < warps; ++i) total += sh[i];
        y[r] = total;
    }
}

}  // namespace

std::size_t gemv_q8_bytes(int n_in) { return strata::kernels::native_q8_1_bytes(n_in, 1); }

bool gemv(ggml_type type, const void* weights, const float* x, float* y, int n_in, int n_out,
          void* q8_scratch, void* stream) {
    return gemv_multi(type, weights, x, y, n_in, n_out, 1, q8_scratch, stream);
}

bool gemv_multi(ggml_type type, const void* weights, const float* x, float* y, int n_in, int n_out, int ncols,
                void* q8_scratch, void* stream) {
    if (n_in <= 0 || n_out <= 0 || ncols < 1) return false;
    if (type == GGML_TYPE_F32) {
        if (ncols != 1) return false;   // only the norms are F32, and they are per row
        k_dot_f32<<<n_out, 256, 0, (cudaStream_t) stream>>>((const float*) weights, x, y, n_in, n_out);
        return true;
    }
    // quantize the activations once, then the quantized dot every row reuses
    strata::kernels::quantize_q8_1_rows(x, ncols, n_in, q8_scratch, stream);
    strata::kernels::iq_mmvq(type, weights, q8_scratch, y, n_in, n_out, ncols, stream);
    return true;
}

}  // namespace strata::step35::kernels
