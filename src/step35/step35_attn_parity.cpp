// src/step35/step35_attn_parity.cpp - causal grouped-query attention with the sliding window, against a float64
// host attention.
//
// The window is the point.  A window that is off by one, or applied to the wrong end, still produces a
// well-formed attention whose values look plausible: only a comparison position by position shows it.  So the
// cases run a window SHORTER than the sequence (16 keys of 40 positions), where positions 0-15 have a full
// prefix and 16+ have exactly 16 keys, and a second case with no window at all.
//
// Both head widths the model uses are covered (64 and 96 query heads), each with 8 KV heads, and head_dim 128.
//
// Two more cases run 520 positions, past the 512 keys the kernel used to fit in a `scores[512]` of shared memory:
// one with no window, one whose window (600) is wider than that array and used to be silently clamped to 512.
// Both were wrong before the online softmax and both are exact against the float64 host now.
#include "strata/step35/attn.hpp"

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

std::vector<float> noise(size_t n, uint32_t seed) {
    std::vector<float> v(n);
    for (float& x : v) {
        seed = seed * 1664525u + 1013904223u;
        x = ((float) (seed >> 8) / (float) (1u << 24)) - 0.5f;
    }
    return v;
}

std::vector<float> host_attn(const std::vector<float>& q, const std::vector<float>& k, const std::vector<float>& v,
                             int n_pos, int n_head, int n_head_kv, int head_dim, int window) {
    std::vector<float> out(q.size(), 0);
    const int group = n_head / n_head_kv;
    for (int p = 0; p < n_pos; ++p) {
        const int first = (window > 0 && p + 1 > window) ? p - window + 1 : 0;
        for (int h = 0; h < n_head; ++h) {
            const int kvh = h / group;
            const float* qh = &q[((size_t) p * n_head + h) * head_dim];
            std::vector<double> s((size_t) (p - first + 1));
            double mx = -1e300;
            for (int j = first; j <= p; ++j) {
                const float* kj = &k[((size_t) j * n_head_kv + kvh) * head_dim];
                double acc = 0;
                for (int d = 0; d < head_dim; ++d) acc += (double) qh[d] * (double) kj[d];
                acc /= std::sqrt((double) head_dim);
                s[(size_t) (j - first)] = acc;
                mx = std::fmax(mx, acc);
            }
            double den = 0;
            for (double& x : s) {
                x = std::exp(x - mx);
                den += x;
            }
            for (int d = 0; d < head_dim; ++d) {
                double acc = 0;
                for (int j = first; j <= p; ++j) {
                    const float* vj = &v[((size_t) j * n_head_kv + kvh) * head_dim];
                    acc += s[(size_t) (j - first)] / den * (double) vj[d];
                }
                out[((size_t) p * n_head + h) * head_dim + d] = (float) acc;
            }
        }
    }
    return out;
}

void attn_case(int n_pos, int n_head, int n_head_kv, int head_dim, int window, const char* name) {
    std::vector<float> q = noise((size_t) n_pos * n_head * head_dim, 5u);
    std::vector<float> k = noise((size_t) n_pos * n_head_kv * head_dim, 6u);
    std::vector<float> v = noise((size_t) n_pos * n_head_kv * head_dim, 7u);
    const std::vector<float> ref = host_attn(q, k, v, n_pos, n_head, n_head_kv, head_dim, window);

    float *d_q = nullptr, *d_k = nullptr, *d_v = nullptr, *d_o = nullptr;
    check_cuda(cudaMalloc(&d_q, q.size() * sizeof(float)), "malloc q");
    check_cuda(cudaMalloc(&d_k, k.size() * sizeof(float)), "malloc k");
    check_cuda(cudaMalloc(&d_v, v.size() * sizeof(float)), "malloc v");
    check_cuda(cudaMalloc(&d_o, ref.size() * sizeof(float)), "malloc o");
    check_cuda(cudaMemcpy(d_q, q.data(), q.size() * sizeof(float), cudaMemcpyHostToDevice), "copy q");
    check_cuda(cudaMemcpy(d_k, k.data(), k.size() * sizeof(float), cudaMemcpyHostToDevice), "copy k");
    check_cuda(cudaMemcpy(d_v, v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice), "copy v");
    sk::attn_causal(d_q, d_k, d_v, d_o, n_pos, n_head, n_head_kv, head_dim, window, nullptr);
    check_cuda(cudaDeviceSynchronize(), "attn sync");
    std::vector<float> got(ref.size());
    check_cuda(cudaMemcpy(got.data(), d_o, got.size() * sizeof(float), cudaMemcpyDeviceToHost), "back");

    // rel_l2 over the whole tensor: a per-element relative error is meaningless where an output sits near zero,
    // and this is a kernel of fp32 exp against a float64 host, so a few 1e-6 of rel_l2 is the floor.
    double num = 0, den = 0, abs_max = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double d = (double) got[i] - (double) ref[i];
        num += d * d;
        den += (double) ref[i] * (double) ref[i];
        abs_max = std::fmax(abs_max, std::fabs(d));
    }
    const double rel = den > 0 ? std::sqrt(num / den) : 0.0;
    char label[180];
    std::snprintf(label, sizeof label,
                  "%s: %d positions, %d Q / %d KV heads, dim %d, window %d -> rel_l2 %.1e (max abs %.1e)", name,
                  n_pos, n_head, n_head_kv, head_dim, window, rel, abs_max);
    check(rel < 1e-5, label);

    // the first position attends to itself only: its output must be exactly V of its own group
    bool first_ok = true;
    const int group = n_head / n_head_kv;
    for (int h = 0; h < n_head; ++h)
        for (int d = 0; d < head_dim; ++d) {
            const float want = v[((size_t) 0 * n_head_kv + h / group) * head_dim + d];
            first_ok = first_ok && std::fabs(got[(size_t) h * head_dim + d] - want) < 1e-6f;
        }
    check(first_ok, "position 0 attends to itself alone (its output is its group's V)");

    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_o);
}

}  // namespace

int main() {
    attn_case(40, 96, 8, 128, 16, "sliding block, window shorter than the sequence");
    attn_case(40, 96, 8, 128, 0, "sliding block, no window");
    attn_case(24, 64, 8, 128, 16, "full block, window shorter than the sequence");
    attn_case(40, 96, 8, 128, 512, "window larger than the sequence");
    attn_case(520, 16, 8, 128, 0, "no window, past the old 512-key shared-memory array");
    attn_case(520, 16, 8, 128, 600, "window wider than the old 512-key array");

    if (failures) {
        std::printf("step35_attn_parity: %d check(s) failed\n", failures);
        return 1;
    }
    std::printf("step35_attn_parity: all checks passed\n");
    return 0;
}
