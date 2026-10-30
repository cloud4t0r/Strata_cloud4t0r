// src/step35/step35_moe_parity.cpp - the router of a Step-3.7-Flash block against a float64 host router written
// to llama.cpp's build_moe_ffn (SIGMOID branch).
//
// The expert IDS must match EXACTLY: a router that picks a different set of eight experts is not a rounding
// difference, it is a different model.  The weights are held to 1e-6 because they are sums of the same sigmoids.
//
// Three cases matter and each gets one:
//   * the release's own shape (288 experts, top-8, a selection bias, weights_norm, scale 3.0);
//   * no bias (a block without exp_probs_b), to show the bias is not assumed;
//   * a TIE between two experts, where both sides must take the lower index - the rule that makes two runs of
//     the same model identical, and without which this test could not be reproduced.
#include "strata/step35/moe.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
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

struct Routed {
    std::vector<int32_t> ids;
    std::vector<float> weights;
};

/// llama.cpp's order: sigmoid, + selection bias for the top-k only, unbiased probs as weights, normalise with
/// the F16 clamp, then scale.  Ties go to the lower index.
Routed host_route(const std::vector<float>& logits, const std::vector<float>& bias, int rows, int n_expert,
                  int top_k, float scale, bool weights_norm) {
    Routed out;
    out.ids.assign((size_t) rows * top_k, 0);
    out.weights.assign((size_t) rows * top_k, 0);
    for (int r = 0; r < rows; ++r) {
        const float* lr = logits.data() + (size_t) r * n_expert;
        std::vector<double> probs((size_t) n_expert);
        for (int e = 0; e < n_expert; ++e) probs[(size_t) e] = 1.0 / (1.0 + std::exp(-(double) lr[e]));
        std::vector<char> taken((size_t) n_expert, 0);
        double sum = 0;
        for (int k = 0; k < top_k; ++k) {
            int best_i = -1;
            double best = -1e300;
            for (int e = 0; e < n_expert; ++e) {
                if (taken[(size_t) e]) continue;
                const double sel = probs[(size_t) e] + (bias.empty() ? 0.0 : (double) bias[(size_t) e]);
                if (sel > best || (sel == best && (best_i < 0 || e < best_i))) {
                    best = sel;
                    best_i = e;
                }
            }
            taken[(size_t) best_i] = 1;
            out.ids[(size_t) r * top_k + k] = best_i;
            out.weights[(size_t) r * top_k + k] = (float) probs[(size_t) best_i];
            sum += probs[(size_t) best_i];
        }
        if (weights_norm) {
            const double den = sum < 6.103515625e-5 ? 6.103515625e-5 : sum;
            for (int k = 0; k < top_k; ++k) out.weights[(size_t) r * top_k + k] = (float) ((double) out.weights[(size_t) r * top_k + k] / den);
        }
        if (scale != 0.0f && scale != 1.0f) {
            for (int k = 0; k < top_k; ++k) out.weights[(size_t) r * top_k + k] *= scale;
        }
    }
    return out;
}

void route_case(const std::vector<float>& logits, const std::vector<float>& bias, int rows, int n_expert, int top_k,
                float scale, bool weights_norm, const char* name) {
    const Routed ref = host_route(logits, bias, rows, n_expert, top_k, scale, weights_norm);
    float* d_logits = nullptr;
    float* d_bias = nullptr;
    int32_t* d_ids = nullptr;
    float* d_w = nullptr;
    check_cuda(cudaMalloc(&d_logits, logits.size() * sizeof(float)), "malloc logits");
    check_cuda(cudaMalloc(&d_ids, ref.ids.size() * sizeof(int32_t)), "malloc ids");
    check_cuda(cudaMalloc(&d_w, ref.weights.size() * sizeof(float)), "malloc w");
    if (!bias.empty()) check_cuda(cudaMalloc(&d_bias, bias.size() * sizeof(float)), "malloc bias");
    check_cuda(cudaMemcpy(d_logits, logits.data(), logits.size() * sizeof(float), cudaMemcpyHostToDevice), "copy logits");
    if (!bias.empty()) check_cuda(cudaMemcpy(d_bias, bias.data(), bias.size() * sizeof(float), cudaMemcpyHostToDevice), "copy bias");

    sk::moe_route(d_logits, bias.empty() ? nullptr : d_bias, rows, n_expert, top_k, scale, weights_norm, d_ids, d_w,
                  nullptr);
    check_cuda(cudaDeviceSynchronize(), "route sync");
    std::vector<int32_t> ids(ref.ids.size());
    std::vector<float> w(ref.weights.size());
    check_cuda(cudaMemcpy(ids.data(), d_ids, ids.size() * sizeof(int32_t), cudaMemcpyDeviceToHost), "back ids");
    check_cuda(cudaMemcpy(w.data(), d_w, w.size() * sizeof(float), cudaMemcpyDeviceToHost), "back w");

    bool ids_ok = true;
    for (size_t i = 0; i < ids.size(); ++i) ids_ok = ids_ok && ids[i] == ref.ids[i];
    float worst = 0;
    for (size_t i = 0; i < w.size(); ++i) {
        const float d = std::fabs(w[i] - ref.weights[i]);
        worst = std::fmax(worst, d / std::fmax(std::fabs(ref.weights[i]), 1e-6f));
    }
    char label[160];
    std::snprintf(label, sizeof label, "%s: %d x %d experts, top %d -> ids exact, weights max rel %.1e", name, rows,
                  n_expert, top_k, (double) worst);
    check(ids_ok && worst < 1e-6f, label);
    cudaFree(d_logits);
    if (d_bias) cudaFree(d_bias);
    cudaFree(d_ids);
    cudaFree(d_w);
}

std::vector<float> noise(size_t n, uint32_t seed, float amp) {
    std::vector<float> v(n);
    for (float& x : v) {
        seed = seed * 1664525u + 1013904223u;
        x = amp * (((float) (seed >> 8) / (float) (1u << 24)) - 0.5f);
    }
    return v;
}

}  // namespace

int main() {
    const int rows = 5, n_expert = 288, top_k = 8;
    const std::vector<float> logits = noise((size_t) rows * n_expert, 11u, 12.0f);
    const std::vector<float> bias = noise((size_t) n_expert, 909u, 1.5f);

    route_case(logits, bias, rows, n_expert, top_k, 3.0f, true, "the release's router");
    route_case(logits, {}, rows, n_expert, top_k, 3.0f, true, "no selection bias");
    route_case(logits, bias, rows, n_expert, top_k, 3.0f, false, "no weights_norm");

    // a tie: experts 7 and 200 carry the same score, and both sides must take the lower index first
    std::vector<float> tied = logits;
    for (int r = 0; r < rows; ++r) {
        tied[(size_t) r * n_expert + 7] = 4.0f;
        tied[(size_t) r * n_expert + 200] = 4.0f;
    }
    route_case(tied, {}, rows, n_expert, 2, 3.0f, true, "a tie takes the lower index");

    if (failures) {
        std::printf("step35_moe_parity: %d check(s) failed\n", failures);
        return 1;
    }
    std::printf("step35_moe_parity: all checks passed\n");
    return 0;
}
