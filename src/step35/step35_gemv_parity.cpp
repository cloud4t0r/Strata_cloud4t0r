// src/step35/step35_gemv_parity.cpp - the quantized matrix-vector product of the Step-3.7-Flash blocks, against a
// float64 host dot over the same weights.
//
// The weights are SYNTHETIC and quantized here through ggml's own from_float_ref, so no model file is needed and
// every type the release actually uses gets a case: IQ3_XXS (18), IQ2_S (22), Q4_K (12), IQ3_S (21), Q5_K (13)
// and F32 - the last one for the norms.
//
// The tolerance is not the fp32 one: the kernel quantizes the ACTIVATION to Q8_1 first, exactly as llama.cpp
// does, so a few 1e-3 of relative error is the contract, not a defect.  What the check does catch is the thing
// that actually goes wrong - a wrong row stride, block size or type dispatch - because those are orders of
// magnitude larger than the quantization.
#include "strata/step35/gemv.hpp"

#include "strata/kernels/iq_kernels.hpp"

#include <cuda_runtime.h>
#include <ggml.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace sk = strata::step35::kernels;

namespace {

int failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        ++failures;
        std::printf("FAIL  %s\n", what);
    } else {
        std::printf("  ok  %s\n", what);
    }
}

void check_cuda(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::printf("FAIL  %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

const char* type_name(ggml_type t) {
    switch (t) {
        case GGML_TYPE_IQ3_XXS: return "IQ3_XXS";
        case GGML_TYPE_IQ2_S: return "IQ2_S";
        case GGML_TYPE_Q4_K: return "Q4_K";
        case GGML_TYPE_IQ3_S: return "IQ3_S";
        case GGML_TYPE_Q5_K: return "Q5_K";
        case GGML_TYPE_F32: return "F32";
        default: return "?";
    }
}

/// A synthetic matrix of `n_out` rows of `n_in` values, quantized to `t` by ggml itself.
std::vector<uint8_t> quantized_matrix(ggml_type t, int n_in, int n_out, uint32_t seed) {
    ggml_quantize_init(t);   // the IQ quantizers build their grids here; without it ggml_abort()s
    const size_t row_bytes = ggml_row_size(t, n_in);
    std::vector<uint8_t> out(row_bytes * (size_t) n_out);
    std::vector<float> row((size_t) n_in);
    const ggml_type_traits* tr = ggml_get_type_traits(t);
    for (int r = 0; r < n_out; ++r) {
        for (int k = 0; k < n_in; ++k) {
            seed = seed * 1664525u + 1013904223u;
            row[(size_t) k] = 0.05f * (((float) (seed >> 8) / (float) (1u << 24)) - 0.5f);
        }
        if (t == GGML_TYPE_F32) {
            std::memcpy(out.data() + (size_t) r * row_bytes, row.data(), row_bytes);
        } else {
            tr->from_float_ref(row.data(), out.data() + (size_t) r * row_bytes, n_in);
        }
    }
    return out;
}

/// The host answer: dequantize each row and accumulate in float64.
std::vector<float> host_gemv(ggml_type t, const std::vector<uint8_t>& w, const std::vector<float>& x, int n_in,
                             int n_out) {
    std::vector<float> y((size_t) n_out), row((size_t) n_in);
    const size_t row_bytes = ggml_row_size(t, n_in);
    const ggml_type_traits* tr = ggml_get_type_traits(t);
    for (int r = 0; r < n_out; ++r) {
        const void* src = w.data() + (size_t) r * row_bytes;
        if (t == GGML_TYPE_F32) {
            std::memcpy(row.data(), src, (size_t) n_in * sizeof(float));
        } else {
            tr->to_float(src, row.data(), n_in);
        }
        double acc = 0;
        for (int k = 0; k < n_in; ++k) acc += (double) row[(size_t) k] * (double) x[(size_t) k];
        y[(size_t) r] = (float) acc;
    }
    return y;
}

void gemv_case(ggml_type t, int n_in, int n_out) {
    const std::vector<uint8_t> w = quantized_matrix(t, n_in, n_out, 7u + (uint32_t) t);
    std::vector<float> x((size_t) n_in);
    uint32_t seed = 4242u;
    for (float& v : x) {
        seed = seed * 1664525u + 1013904223u;
        v = ((float) (seed >> 8) / (float) (1u << 24)) - 0.5f;
    }
    const std::vector<float> ref = host_gemv(t, w, x, n_in, n_out);

    const size_t row_bytes = ggml_row_size(t, n_in);
    const size_t q8_bytes = sk::gemv_q8_bytes(n_in);
    uint8_t* d_w = nullptr;
    float* d_x = nullptr;
    float* d_y = nullptr;
    void* d_q8 = nullptr;
    check_cuda(cudaMalloc(&d_w, row_bytes * (size_t) n_out), "malloc w");
    check_cuda(cudaMalloc(&d_x, x.size() * sizeof(float)), "malloc x");
    check_cuda(cudaMalloc(&d_y, (size_t) n_out * sizeof(float)), "malloc y");
    check_cuda(cudaMalloc(&d_q8, q8_bytes), "malloc q8");
    check_cuda(cudaMemcpy(d_w, w.data(), row_bytes * (size_t) n_out, cudaMemcpyHostToDevice), "copy w");
    check_cuda(cudaMemcpy(d_x, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice), "copy x");

    const bool ran = sk::gemv(t, d_w, d_x, d_y, n_in, n_out, d_q8, nullptr);
    char label[160];
    if (!ran) {
        std::snprintf(label, sizeof label, "%s: gemv refused a type it must carry", type_name(t));
        check(false, label);
    } else {
        check_cuda(cudaDeviceSynchronize(), "gemv sync");
        std::vector<float> got((size_t) n_out);
        check_cuda(cudaMemcpy(got.data(), d_y, got.size() * sizeof(float), cudaMemcpyDeviceToHost), "back");
        double num = 0, den = 0;
        for (int r = 0; r < n_out; ++r) {
            const double d = (double) got[(size_t) r] - (double) ref[(size_t) r];
            num += d * d;
            den += (double) ref[(size_t) r] * (double) ref[(size_t) r];
        }
        const double rel = den > 0 ? std::sqrt(num / den) : 0.0;
        std::snprintf(label, sizeof label, "%-8s %5d x %-5d rel_l2 %.2e (Q8_1 activation quantization)",
                      type_name(t), n_in, n_out, rel);
        check(rel < 2e-2, label);   // a wrong stride/block/dispatch is orders of magnitude past this
    }
    cudaFree(d_w);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_q8);
}

}  // namespace

int main() {
    // the shapes the release uses: hidden 4096, the dense FFN 11264, an expert's 1280, the router's 288
    for (ggml_type t : {GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ2_S, GGML_TYPE_Q4_K, GGML_TYPE_IQ3_S, GGML_TYPE_Q5_K,
                        GGML_TYPE_F32}) {
        gemv_case(t, 256, 64);
    }
    gemv_case(GGML_TYPE_IQ3_XXS, 4096, 64);    // a projection at the real hidden width
    gemv_case(GGML_TYPE_IQ3_XXS, 11264, 16);   // the dense FFN's down projection
    gemv_case(GGML_TYPE_IQ2_S, 4096, 16);      // q or k, IQ2_S in the release

    if (failures) {
        std::printf("step35_gemv_parity: %d check(s) failed\n", failures);
        return 1;
    }
    std::printf("step35_gemv_parity: all checks passed\n");
    return 0;
}
