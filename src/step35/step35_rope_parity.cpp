// src/step35/step35_rope_parity.cpp - the parity test for the two Step-3.7-Flash kernels that are not in
// strata_kernels: the per-head RMSNorm and the NEOX partial RoPE with llama3's frequency factors.
//
// WHY THE POSITIONS ARE FAR.  The failure this test exists for is a PAIRING mistake, and at position 0-5 the
// wrong pairing answers within the noise of a quantized matmul - which is exactly how it went unnoticed in the
// CPU reference for two commits.  The cases below turn at 4096 and 131072, where a swapped pair is a completely
// different number, and the last check makes the point explicit: the adjacent (NORM) pairing is applied to the
// same input and MUST differ, so a green run cannot be vacuous.
//
// The host reference is the same arithmetic in float64 (the kernel computes its angle in double too), so a
// correct kernel agrees to float32 rounding and nothing looser is needed.
#include "strata/step35/kernels.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
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

/// A deterministic stream of values in [-0.5, 0.5): a wrong pairing must not be able to hide behind symmetry.
std::vector<float> values(size_t n, uint32_t seed) {
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) {
        seed = seed * 1664525u + 1013904223u;
        v[i] = ((float) (seed >> 8) / (float) (1u << 24)) - 0.5f;
    }
    return v;
}

/// The host answer: NEOX pairing, float64 angle, the frequency factors for the full-attention blocks only.
std::vector<float> host_rope(const std::vector<float>& x, int rows, int head_dim, int n_rot,
                             const std::vector<int>& pos, double base, const std::vector<float>& ff) {
    std::vector<float> y = x;
    const int half = n_rot / 2;
    for (int r = 0; r < rows; ++r) {
        float* row = y.data() + (size_t) r * head_dim;
        for (int j = 0; j < half; ++j) {
            double theta = (double) pos[(size_t) r] * std::pow(base, -2.0 * (double) j / (double) n_rot);
            if (!ff.empty() && (size_t) j < ff.size() && ff[(size_t) j] != 0.0f) theta /= (double) ff[(size_t) j];
            const double c = std::cos(theta), s = std::sin(theta);
            const float a = row[j], b = row[j + half];
            sk::rope_pair((double) a, (double) b, c, s, row[j], row[j + half]);
        }
    }
    return y;
}

/// The WRONG pairing, to prove the test would have caught it.
std::vector<float> host_rope_adjacent(const std::vector<float>& x, int rows, int head_dim, int n_rot,
                                      const std::vector<int>& pos, double base) {
    std::vector<float> y = x;
    for (int r = 0; r < rows; ++r) {
        float* row = y.data() + (size_t) r * head_dim;
        for (int i = 0; i + 1 < n_rot; i += 2) {
            const double theta = (double) pos[(size_t) r] * std::pow(base, -2.0 * (double) i / (double) n_rot);
            const double c = std::cos(theta), s = std::sin(theta);
            const float a = row[i], b = row[i + 1];
            sk::rope_pair((double) a, (double) b, c, s, row[i], row[i + 1]);
        }
    }
    return y;
}

float max_rel(const std::vector<float>& a, const std::vector<float>& b) {
    float worst = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        const float d = std::fabs(a[i] - b[i]);
        const float s = std::fmax(std::fabs(a[i]), 1e-3f);
        worst = std::fmax(worst, d / s);
    }
    return worst;
}

/// One RoPE case: (head_dim, n_rot, base, with the frequency factors or not, positions of the rows).
void rope_case(int head_dim, int n_rot, double base, bool with_ff, const std::vector<int>& pos, const char* name) {
    const int rows = (int) pos.size();
    std::vector<float> x = values((size_t) rows * head_dim, 12345u + (uint32_t) n_rot);
    const std::vector<float> ref = host_rope(x, rows, head_dim, n_rot, pos, base, with_ff ? std::vector<float>{
        1.0f, 1.0f, 1.13836f, 1.37450f, 1.57644f, 1.73373f, 1.84757f, 1.92565f, 1.97725f, 2.0f, 2.0f, 2.0f
    } : std::vector<float>{});

    std::vector<float> ff = with_ff ? std::vector<float>{1.0f, 1.0f, 1.13836f, 1.37450f, 1.57644f, 1.73373f,
                                                        1.84757f, 1.92565f, 1.97725f, 2.0f, 2.0f, 2.0f}
                                    : std::vector<float>{};
    float* d_x = nullptr;
    int* d_pos = nullptr;
    float* d_ff = nullptr;
    check_cuda(cudaMalloc(&d_x, x.size() * sizeof(float)), "malloc x");
    check_cuda(cudaMalloc(&d_pos, pos.size() * sizeof(int)), "malloc pos");
    if (!ff.empty()) check_cuda(cudaMalloc(&d_ff, ff.size() * sizeof(float)), "malloc ff");
    check_cuda(cudaMemcpy(d_x, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice), "copy x");
    check_cuda(cudaMemcpy(d_pos, pos.data(), pos.size() * sizeof(int), cudaMemcpyHostToDevice), "copy pos");
    if (!ff.empty()) check_cuda(cudaMemcpy(d_ff, ff.data(), ff.size() * sizeof(float), cudaMemcpyHostToDevice), "copy ff");

    sk::rope_neox(d_x, rows, head_dim, n_rot, d_pos, base, ff.empty() ? nullptr : d_ff, (int) ff.size(), nullptr);
    check_cuda(cudaDeviceSynchronize(), "rope sync");
    std::vector<float> got(x.size());
    check_cuda(cudaMemcpy(got.data(), d_x, got.size() * sizeof(float), cudaMemcpyDeviceToHost), "back");

    const float err = max_rel(ref, got);
    char label[160];
    std::snprintf(label, sizeof label, "%s: %d rows, n_rot %d, base %.0f, %s%s -> max rel %.2e", name, rows, n_rot,
                  base, with_ff ? "with freq factors" : "no freq factors",
                  pos[0] == 0 && rows > 1 && pos[1] == 0 ? " at position 0" : "", err);
    check(err < 1e-5f, label);

    // every row at position 0 leaves every pair at angle 0: the rotation must be the identity, bit for bit
    bool all_zero = true;
    for (int q : pos) all_zero = all_zero && q == 0;
    if (all_zero) {
        const bool same = std::memcmp(x.data(), got.data(), x.size() * sizeof(float)) == 0;
        check(same, "position 0 rotates nothing (bit for bit)");
    }

    // OBSERVABILITY: the wrong pairing on the same input must give a different answer, or the check above
    // could not tell a right kernel from a wrong one.  The last row is the far position.
    if (!all_zero) {
        const std::vector<float> wrong = host_rope_adjacent(x, rows, head_dim, n_rot, pos, base);
        const size_t row = (size_t) (rows - 1) * head_dim;
        std::vector<float> w1(wrong.begin() + row, wrong.begin() + row + head_dim);
        std::vector<float> g1(got.begin() + row, got.begin() + row + head_dim);
        const float differs = max_rel(w1, g1);
        char note[128];
        std::snprintf(note, sizeof note, "the adjacent (NORM) pairing differs at position %d (max rel %.2e)",
                      pos[(size_t) (rows - 1)], (double) differs);
        check(differs > 1e-2f, note);
    }

    cudaFree(d_x);
    cudaFree(d_pos);
    if (d_ff) cudaFree(d_ff);
}

/// The per-head RMSNorm: y = x / sqrt(mean(x^2) + eps) * w, in place.
void norm_case(int rows, int head_dim, float eps) {
    std::vector<float> x = values((size_t) rows * head_dim, 999u);
    std::vector<float> w = values((size_t) head_dim, 424242u);
    for (float& v : w) v += 1.0f;   // the real QK-norm weights are near 1
    std::vector<float> ref = x;
    for (int r = 0; r < rows; ++r) {
        float* row = ref.data() + (size_t) r * head_dim;
        double ss = 0;
        for (int i = 0; i < head_dim; ++i) ss += (double) row[i] * (double) row[i];
        const double inv = 1.0 / std::sqrt(ss / (double) head_dim + (double) eps);
        for (int i = 0; i < head_dim; ++i) row[i] = (float) ((double) row[i] * inv * (double) w[(size_t) i]);
    }
    float* d_x = nullptr;
    float* d_w = nullptr;
    check_cuda(cudaMalloc(&d_x, x.size() * sizeof(float)), "malloc x");
    check_cuda(cudaMalloc(&d_w, w.size() * sizeof(float)), "malloc w");
    check_cuda(cudaMemcpy(d_x, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice), "copy x");
    check_cuda(cudaMemcpy(d_w, w.data(), w.size() * sizeof(float), cudaMemcpyHostToDevice), "copy w");
    sk::per_head_rms_norm(d_x, rows, head_dim, d_w, eps, nullptr);
    check_cuda(cudaDeviceSynchronize(), "norm sync");
    std::vector<float> got(x.size());
    check_cuda(cudaMemcpy(got.data(), d_x, got.size() * sizeof(float), cudaMemcpyDeviceToHost), "back");
    const float err = max_rel(ref, got);
    char label[128];
    std::snprintf(label, sizeof label, "per-head RMSNorm, %d heads of %d, eps %.1e -> max rel %.2e", rows, head_dim,
                  (double) eps, err);
    check(err < 1e-6f, label);
    cudaFree(d_x);
    cudaFree(d_w);
}

}  // namespace

int main() {
    // the release's two block kinds: a full-attention block (64 of 128 dims, base 5e6, with the table) and a
    // sliding one (all 128 dims, base 1e4, no table).  Positions include the ends of a long prompt.
    rope_case(128, 64, 5000000.0, true, {0, 1, 4096, 131072}, "full-attention block");
    rope_case(128, 128, 10000.0, false, {0, 1, 4096, 131072}, "sliding block");
    rope_case(128, 64, 5000000.0, true, {0, 0, 0, 0}, "full-attention block, prompt start");
    rope_case(128, 128, 10000.0, false, {0, 0, 0, 0}, "sliding block, prompt start");
    norm_case(96, 128, 1e-5f);
    norm_case(64, 128, 1e-5f);

    if (failures) {
        std::printf("step35_rope_parity: %d check(s) failed\n", failures);
        return 1;
    }
    std::printf("step35_rope_parity: all checks passed\n");
    return 0;
}
