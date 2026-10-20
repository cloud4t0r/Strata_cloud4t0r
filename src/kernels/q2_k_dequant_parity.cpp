// src/kernels/q2_k_dequant_parity.cpp - the GPU's Q2_K dequantizer against the artifact's validated CPU one.
//
// MiniMax-M2's IQ2_XXS file carries its token_embd at Q2_K, so the embedding path reads a type no Strata
// kernel had decoded before.  The reference is `dequantize_q2_K` (include/strata/artifact/dequant.hpp),
// transcribed from ggml's dequantize_row_q2_K at the pinned commit; this test feeds both the same blocks.
//
//   q2_k_dequant_parity [SEED]
//
// The blocks are random bytes with valid fp16 super-block scales, so every branch of the format is exercised
// (the scales and the mins are 6-bit packed, and a group of 32 values takes two of them).
#include "strata/artifact/dequant.hpp"
#include "strata/kernels/iq_kernels.hpp"

#include "ggml.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

constexpr int kQ2K = 10;
constexpr int kBlock = 256;
constexpr int kBlocks = 64;      // 64 super-blocks = 16384 values
constexpr int kRows = 32;        // the embedding table: 32 rows of 512 values (2 super-blocks each)

// ggml's own conversion, so the reference and the test agree on the rounding (the FP16 path of the kernel
// computes in FP32 and converts, like the other formats)
uint16_t f32_to_f16(float f) { return ggml_fp32_to_fp16(f); }

}  // namespace

int main(int argc, char** argv) {
    const uint32_t seed = argc > 1 ? (uint32_t) std::atoi(argv[1]) : 20260720u;
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> byte(0, 255);

    // the blocks: 84 bytes each (scales 16, qs 64, d and dmin as fp16)
    std::vector<uint8_t> blocks((size_t) kBlocks * 84);
    for (size_t i = 0; i < blocks.size(); i += 84) {
        for (int k = 0; k < 80; ++k) blocks[i + (size_t) k] = (uint8_t) byte(rng);
        const float d = 0.05f + 0.01f * (float) byte(rng) / 255.0f;
        const float dmin = 0.01f + 0.01f * (float) byte(rng) / 255.0f;
        const uint16_t hd = f32_to_f16(d), hm = f32_to_f16(dmin);
        std::memcpy(&blocks[i + 80], &hd, 2);
        std::memcpy(&blocks[i + 82], &hm, 2);
    }

    // the CPU reference
    std::vector<float> want((size_t) kBlocks * kBlock);
    for (int b = 0; b < kBlocks; ++b)
        strata::dequantize_q2_K(&blocks[(size_t) b * 84], &want[(size_t) b * kBlock]);

    cudaStream_t s = nullptr;
    cudaStreamCreate(&s);
    float* dq = nullptr;
    uint16_t* dh = nullptr;
    uint8_t* dblocks = nullptr;
    cudaMalloc((void**) &dq, want.size() * 4);
    cudaMalloc((void**) &dh, want.size() * 2);
    cudaMalloc((void**) &dblocks, blocks.size());
    cudaMemcpy(dblocks, blocks.data(), blocks.size(), cudaMemcpyHostToDevice);
    int failures = 0;

    // (a) the FP32 path
    strata::kernels::iq_dequant_f32(kQ2K, dblocks, (int64_t) want.size(), dq, s);
    cudaStreamSynchronize(s);
    std::vector<float> got(want.size());
    cudaMemcpy(got.data(), dq, got.size() * 4, cudaMemcpyDeviceToHost);
    size_t differ = 0;
    double max_rel = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        if (std::memcmp(&got[i], &want[i], 4) != 0) ++differ;
        const double rel = std::fabs((double) got[i] - (double) want[i]) /
                           (std::fabs((double) want[i]) + 1e-6);
        if (rel > max_rel) max_rel = rel;
    }
    std::fflush(stdout);
    std::printf("  Q2_K FP32 dequant vs the CPU reference: %zu of %zu values differ in any bit, max rel %.3e\n",
                differ, got.size(), max_rel);
    if (differ) ++failures;

    // (b) the FP16 path: the CPU value rounded to fp16 must be what the device wrote
    strata::kernels::iq_dequant_f16(kQ2K, dblocks, (int64_t) want.size(), dh, s);
    cudaStreamSynchronize(s);
    std::vector<uint16_t> goth(want.size());
    cudaMemcpy(goth.data(), dh, goth.size() * 2, cudaMemcpyDeviceToHost);
    size_t h_differ = 0;
    for (size_t i = 0; i < goth.size(); ++i) {
        const uint16_t ref = f32_to_f16(want[i]);
        if (goth[i] != ref) ++h_differ;
    }
    std::printf("  Q2_K FP16 dequant vs the CPU reference: %zu of %zu values differ\n", h_differ, goth.size());
    if (h_differ) ++failures;

    // (c) the embedding path: the same rows, gathered by token id
    const int64_t row_values = 512;
    std::vector<uint8_t> table((size_t) kRows * 2 * 84);
    for (size_t i = 0; i < table.size(); i += 84) {
        for (int k = 0; k < 80; ++k) table[i + (size_t) k] = (uint8_t) byte(rng);
        const uint16_t hd = f32_to_f16(0.1f), hm = f32_to_f16(0.02f);
        std::memcpy(&table[i + 80], &hd, 2);
        std::memcpy(&table[i + 82], &hm, 2);
    }
    std::vector<int32_t> tokens(kRows);
    for (int i = 0; i < kRows; ++i) tokens[(size_t) i] = kRows - 1 - i;   // rows in reverse order
    int32_t* dtok = nullptr;
    cudaMalloc((void**) &dtok, tokens.size() * 4);
    cudaMemcpy(dtok, tokens.data(), tokens.size() * 4, cudaMemcpyHostToDevice);
    float* demb = nullptr;
    uint8_t* dtable = nullptr;
    cudaMalloc((void**) &demb, (size_t) kRows * row_values * 4);
    cudaMalloc((void**) &dtable, table.size());
    cudaMemcpy(dtable, table.data(), table.size(), cudaMemcpyHostToDevice);
    strata::kernels::iq_embed_rows(kQ2K, dtable, (size_t) 2 * 84, dtok, kRows, row_values, demb, s);
    cudaStreamSynchronize(s);
    std::vector<float> gemb((size_t) kRows * row_values);
    cudaMemcpy(gemb.data(), demb, gemb.size() * 4, cudaMemcpyDeviceToHost);
    size_t emb_differ = 0;
    for (int r = 0; r < kRows; ++r) {
        std::vector<float> ref((size_t) row_values);
        // a row is two super-blocks: 256 values each
        strata::dequantize_q2_K(&table[(size_t) (kRows - 1 - r) * 2 * 84], ref.data());
        strata::dequantize_q2_K(&table[(size_t) (kRows - 1 - r) * 2 * 84 + 84], ref.data() + 256);
        for (int64_t i = 0; i < row_values; ++i)
            if (std::memcmp(&gemb[(size_t) r * row_values + (size_t) i], &ref[(size_t) i], 4) != 0) ++emb_differ;
    }
    std::printf("  Q2_K embedding rows vs the CPU reference: %zu of %lld values differ\n", emb_differ,
                (long long) (kRows * row_values));
    if (emb_differ) ++failures;

    cudaFree(dq); cudaFree(dh); cudaFree(dtok); cudaFree(demb); cudaFree(dblocks); cudaFree(dtable);
    cudaStreamDestroy(s);
    std::printf("%s\n", failures ? "FAILED" : "ALL PASSED");
    return failures ? 1 : 0;
}
