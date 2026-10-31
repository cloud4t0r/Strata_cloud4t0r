// src/step35/step35_kv_parity.cpp - the quantized K/V cache, against a float64 host that quantizes the same way.
//
// Four things are checked, and each one is a way this could be wrong while still answering something plausible:
//
//   1. THE CELL ITSELF.  The pool is read back and compared code by code and scale by scale with a host that
//      applies kv_q8.hpp's rule (`f16(max|x| / 127)` per 64 dimensions, `clamp(rint(x / scale), -127, 127)`).
//      A transposed group, a scale taken over the head instead of the group, or a sign slip changes the codes but
//      leaves the attention finite - it would pass a tolerance test on the output alone.
//   2. THE ATTENTION over the dequantized cells, for both query head counts the release uses (96 on a sliding
//      block, 64 on a full one), with no window, a window shorter than the sequence, and a window longer than it.
//   3. A DECODE ROW is the same cells, read the same way, one position at a time: the row a decode produces must
//      be IDENTICAL to the row the prefill of the whole sequence produced, otherwise the cache is not the thing the
//      prompt wrote.  (It is bit for bit - same keys, same order, same arithmetic.)
//   4. A PERMUTED PAGE TABLE, at a granule of 8 cells rather than the session's 128, must not change a single bit
//      of the answer: the mapping is a detail of where a cell sits, never of what it means.
//   5. A DECODE ROW PAST POSITION 512 with no window - 1024 cells - which is the case the validation vehicle
//      (`attn_causal`) cannot express at all: it keeps a window's scores in a fixed 512-entry shared array.
//
// Nothing here needs the 76 GB download: the K/V is generated.
#include "strata/step35/attn.hpp"
#include "strata/step35/session.hpp"

#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/qsa.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace sk = strata::step35::kernels;
namespace kk = strata::kernels;

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

/// The host's copy of kv_q8.hpp's rule, one cell (one position, every KV head): codes and the stored scales.
void host_quantize(const float* x, int nkv, int hd, std::vector<int8_t>& codes, std::vector<uint16_t>& scales) {
    const int groups = hd / kk::KV_Q8_GROUP;
    codes.assign((size_t) nkv * hd, 0);
    scales.assign((size_t) nkv * groups, 0);
    for (int h = 0; h < nkv; ++h) {
        for (int g = 0; g < groups; ++g) {
            const float* src = x + (size_t) h * hd + g * kk::KV_Q8_GROUP;
            float amax = 0.0f;
            for (int t = 0; t < kk::KV_Q8_GROUP; ++t) amax = std::fmax(amax, std::fabs(src[t]));
            const uint16_t sbits = kk::f16_from_f32(amax / 127.0f);
            const float sf = kk::f32_from_f16(sbits);
            for (int t = 0; t < kk::KV_Q8_GROUP; ++t) {
                int c = 0;
                if (sf > 0.0f) {
                    // __float2int_rn on the device and std::nearbyint here are both round-to-nearest-even
                    c = (int) std::nearbyint(src[t] / sf);
                    c = c < -127 ? -127 : (c > 127 ? 127 : c);
                }
                codes[(size_t) h * hd + g * kk::KV_Q8_GROUP + t] = (int8_t) c;
            }
            scales[(size_t) h * groups + g] = sbits;
        }
    }
}

/// The dequantized value of a stored cell, `[nkv][hd]` fp32, exactly as the kernel reconstructs it.
std::vector<float> dequantize(const std::vector<int8_t>& codes, const std::vector<uint16_t>& scales, int nkv, int hd) {
    const int groups = hd / kk::KV_Q8_GROUP;
    std::vector<float> out((size_t) nkv * hd);
    for (int h = 0; h < nkv; ++h)
        for (int g = 0; g < groups; ++g) {
            const float s = kk::f32_from_f16(scales[(size_t) h * groups + g]);
            for (int t = 0; t < kk::KV_Q8_GROUP; ++t)
                out[(size_t) h * hd + g * kk::KV_Q8_GROUP + t] =
                    (float) codes[(size_t) h * hd + g * kk::KV_Q8_GROUP + t] * s;
        }
    return out;
}

/// Causal GQA attention in float64 over the DEQUANTIZED K/V - the same contract as k_attn_q8.  One row at a time,
/// so a single position far past the sliding window can be checked without paying for the whole sequence.
void host_row(const std::vector<float>& q, const std::vector<float>& k, const std::vector<float>& v, int p,
              int n_head, int n_head_kv, int head_dim, int window, std::vector<float>& out) {
    const int group = n_head / n_head_kv;
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
            for (int j = first; j <= p; ++j)
                acc += s[(size_t) (j - first)] / den * (double) v[((size_t) j * n_head_kv + kvh) * head_dim + d];
            out[((size_t) p * n_head + h) * head_dim + d] = (float) acc;
        }
    }
}

std::vector<float> host_attn(const std::vector<float>& q, const std::vector<float>& k, const std::vector<float>& v,
                             int n_pos, int n_head, int n_head_kv, int head_dim, int window) {
    std::vector<float> out(q.size(), 0);
    for (int p = 0; p < n_pos; ++p) host_row(q, k, v, p, n_head, n_head_kv, head_dim, window, out);
    return out;
}

double rel_l2(const std::vector<float>& got, const std::vector<float>& ref) {
    double num = 0, den = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double d = (double) got[i] - (double) ref[i];
        num += d * d;
        den += (double) ref[i] * (double) ref[i];
    }
    return den > 0 ? std::sqrt(num / den) : 0.0;
}

double max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0;
    for (size_t i = 0; i < a.size(); ++i) m = std::fmax(m, std::fabs((double) a[i] - (double) b[i]));
    return m;
}

float* to_device(const std::vector<float>& v) {
    float* d = nullptr;
    check_cuda(cudaMalloc((void**) &d, v.size() * sizeof(float)), "malloc");
    check_cuda(cudaMemcpy(d, v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice), "copy in");
    return d;
}

std::vector<float> from_device(const float* d, size_t n) {
    std::vector<float> v(n);
    check_cuda(cudaMemcpy(v.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost), "copy out");
    return v;
}

// ---- 1, 2 and 3: a prompt written through the session, the cells read back, then one decode row at a time
void session_case(int n_pos, int n_head, int n_head_kv, int head_dim, int window, bool chunked, const char* name) {
    std::vector<float> q = noise((size_t) n_pos * n_head * head_dim, 11u);
    std::vector<float> k = noise((size_t) n_pos * n_head_kv * head_dim, 12u);
    std::vector<float> v = noise((size_t) n_pos * n_head_kv * head_dim, 13u);

    sk::Session s;
    std::string err;
    if (!sk::session_create(s, 1, n_head_kv, head_dim, n_pos, err)) {
        std::printf("FAIL  %s: %s\n", name, err.c_str());
        ++failures;
        return;
    }
    // A second layer must not be allocated just because the session could hold it: 45 layers at a long context is
    // the memory this design exists to keep down.
    check(s.layers.size() == 1 && s.layers[0].k_q == nullptr, "a layer's pools are allocated on first use, not at create");
    check(sk::session_bytes_per_token(s) == (uint64_t) kk::kv_q8_bytes_per_cell({0, n_head_kv, head_dim, 0, 0, 0, 0, 0, 0}),
          "a token's cells are kv_q8's bytes per cell");

    float* d_q = to_device(q);
    float* d_k = to_device(k);
    float* d_v = to_device(v);
    float* d_o = nullptr;
    check_cuda(cudaMalloc((void**) &d_o, q.size() * sizeof(float)), "malloc out");

    const int first_chunk = chunked ? n_pos / 2 : n_pos;
    bool ok = sk::session_append(s, 0, 0, first_chunk, d_k, d_v, err);
    if (ok && chunked) ok = sk::session_append(s, 0, first_chunk, n_pos - first_chunk, d_k + (size_t) first_chunk * n_head_kv * head_dim,
                                               d_v + (size_t) first_chunk * n_head_kv * head_dim, err);
    if (!ok) {
        std::printf("FAIL  %s: append: %s\n", name, err.c_str());
        ++failures;
        return;
    }
    ok = sk::session_attend(s, 0, d_q, d_o, 0, n_pos, n_head, window, err);
    check_cuda(cudaDeviceSynchronize(), "sync");
    if (!ok) {
        std::printf("FAIL  %s: attend: %s\n", name, err.c_str());
        ++failures;
        return;
    }
    const std::vector<float> got = from_device(d_o, q.size());

    // the host's view of the same cells: quantize every position, then dequantize
    std::vector<float> kdq((size_t) n_pos * n_head_kv * head_dim), vdq(kdq.size());
    for (int p = 0; p < n_pos; ++p) {
        std::vector<int8_t> kc, vc;
        std::vector<uint16_t> ks, vs;
        host_quantize(&k[(size_t) p * n_head_kv * head_dim], n_head_kv, head_dim, kc, ks);
        host_quantize(&v[(size_t) p * n_head_kv * head_dim], n_head_kv, head_dim, vc, vs);
        const std::vector<float> kf = dequantize(kc, ks, n_head_kv, head_dim);
        const std::vector<float> vf = dequantize(vc, vs, n_head_kv, head_dim);
        std::copy(kf.begin(), kf.end(), kdq.begin() + (size_t) p * n_head_kv * head_dim);
        std::copy(vf.begin(), vf.end(), vdq.begin() + (size_t) p * n_head_kv * head_dim);
    }
    const std::vector<float> ref = host_attn(q, kdq, vdq, n_pos, n_head, n_head_kv, head_dim, window);
    const double rel = rel_l2(got, ref);
    char label[220];
    std::snprintf(label, sizeof label,
                  "%s: %d positions, %d Q / %d KV heads, dim %d, window %d, appended %s -> rel_l2 %.1e", name, n_pos,
                  n_head, n_head_kv, head_dim, window, chunked ? "in two chunks" : "at once", rel);
    check(rel < 1e-5, label);

    // the stored cell, bit for bit: codes and scales, through the page table the session wrote with
    const int groups = head_dim / kk::KV_Q8_GROUP;
    const int page_size = s.page_size;
    std::vector<int8_t> pool_k((size_t) s.capacity * n_head_kv * head_dim), pool_v(pool_k.size());
    std::vector<uint16_t> scale_k((size_t) s.capacity * n_head_kv * groups), scale_v(scale_k.size());
    check_cuda(cudaMemcpy(pool_k.data(), s.layers[0].k_q, pool_k.size(), cudaMemcpyDeviceToHost), "read k");
    check_cuda(cudaMemcpy(pool_v.data(), s.layers[0].v_q, pool_v.size(), cudaMemcpyDeviceToHost), "read v");
    check_cuda(cudaMemcpy(scale_k.data(), s.layers[0].k_scale, scale_k.size() * 2, cudaMemcpyDeviceToHost), "read ks");
    check_cuda(cudaMemcpy(scale_v.data(), s.layers[0].v_scale, scale_v.size() * 2, cudaMemcpyDeviceToHost), "read vs");
    bool cells_ok = true;
    size_t first_bad = 0;
    for (int p = 0; p < n_pos && cells_ok; ++p) {
        std::vector<int8_t> kc, vc;
        std::vector<uint16_t> ks, vs;
        host_quantize(&k[(size_t) p * n_head_kv * head_dim], n_head_kv, head_dim, kc, ks);
        host_quantize(&v[(size_t) p * n_head_kv * head_dim], n_head_kv, head_dim, vc, vs);
        for (int h = 0; h < n_head_kv && cells_ok; ++h) {
            const size_t row = (size_t) (p / page_size) * n_head_kv * page_size + (size_t) h * page_size + (p % page_size);
            for (int d = 0; d < head_dim; ++d)
                if (pool_k[row * head_dim + d] != kc[(size_t) h * head_dim + d] ||
                    pool_v[row * head_dim + d] != vc[(size_t) h * head_dim + d]) {
                    cells_ok = false;
                    first_bad = (size_t) p;
                    break;
                }
            for (int g = 0; g < groups; ++g)
                if (scale_k[row * groups + g] != ks[(size_t) h * groups + g] ||
                    scale_v[row * groups + g] != vs[(size_t) h * groups + g]) {
                    cells_ok = false;
                    first_bad = (size_t) p;
                    break;
                }
        }
    }
    std::snprintf(label, sizeof label, "%s: the pool holds the host's codes and scales for every position", name);
    if (!cells_ok) std::printf("      first differing position: %zu\n", first_bad);
    check(cells_ok, label);

    // position 0 sees itself alone, so its row is its group's dequantized V: the q -> kv head map, or a wrong
    // group stride, shows here even when the softmax happens to be right
    bool first_ok = true;
    const int group = n_head / n_head_kv;
    for (int h = 0; h < n_head; ++h)
        for (int d = 0; d < head_dim; ++d)
            first_ok = first_ok && std::fabs(got[(size_t) h * head_dim + d] -
                                             vdq[(size_t) (h / group) * head_dim + d]) < 1e-6f;
    check(first_ok, "position 0's row is its KV group's dequantized V");

    // ---- 3: a decode row over cells the prefill wrote, one position at a time
    double worst_decode = 0;
    for (int p = 0; p < n_pos; p += std::max(1, n_pos / 4)) {
        float* d_row = nullptr;
        check_cuda(cudaMalloc((void**) &d_row, (size_t) n_head * head_dim * sizeof(float)), "malloc row");
        ok = sk::session_attend(s, 0, d_q + (size_t) p * n_head * head_dim, d_row, p, 1, n_head, window, err);
        check_cuda(cudaDeviceSynchronize(), "sync row");
        if (!ok) {
            std::printf("FAIL  %s: decode row %d: %s\n", name, p, err.c_str());
            ++failures;
            cudaFree(d_row);
            return;
        }
        const std::vector<float> row = from_device(d_row, (size_t) n_head * head_dim);
        worst_decode = std::fmax(worst_decode, max_abs_diff(row, std::vector<float>(
                                                                    got.begin() + (size_t) p * n_head * head_dim,
                                                                    got.begin() + (size_t) (p + 1) * n_head * head_dim)));
        cudaFree(d_row);
    }
    std::snprintf(label, sizeof label,
                  "%s: a decode row is bit for bit the prefill's row (max abs diff %.1e)", name, worst_decode);
    check(worst_decode == 0.0, label);

    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_o);
    sk::session_destroy(s);
}

// ---- 4: the page table is a detail of where a cell sits.  A granule of 8 cells, permuted pages, and the answer
// must not move by a bit.
void permuted_page_case(int n_pos, int n_head, int n_head_kv, int head_dim, int window) {
    std::vector<float> q = noise((size_t) n_pos * n_head * head_dim, 21u);
    std::vector<float> k = noise((size_t) n_pos * n_head_kv * head_dim, 22u);
    std::vector<float> v = noise((size_t) n_pos * n_head_kv * head_dim, 23u);

    const int page_size = 8;
    const int n_pages = (n_pos + page_size - 1) / page_size;
    kk::QsaShapes shapes;
    shapes.n_head_kv = n_head_kv;
    shapes.head_dim = head_dim;
    shapes.page_size = page_size;

    // two runs: the identity, then a permutation that reverses the pages - the same cells, in another order
    std::vector<float> identity, permuted;
    for (int run = 0; run < 2; ++run) {
        std::vector<int32_t> table((size_t) n_pages);
        for (int i = 0; i < n_pages; ++i) table[(size_t) i] = run == 0 ? i : n_pages - 1 - i;
        int32_t* d_table = nullptr;
        int32_t* d_step = nullptr;
        int8_t *kq = nullptr, *vq = nullptr;
        uint16_t *ks = nullptr, *vs = nullptr;
        check_cuda(cudaMalloc((void**) &d_table, table.size() * sizeof(int32_t)), "malloc table");
        check_cuda(cudaMemcpy(d_table, table.data(), table.size() * sizeof(int32_t), cudaMemcpyHostToDevice), "table");
        check_cuda(cudaMalloc((void**) &d_step, kk::kStepCount * sizeof(int32_t)), "malloc step");
        const size_t cells = (size_t) n_pages * page_size;
        check_cuda(cudaMalloc((void**) &kq, cells * n_head_kv * head_dim), "malloc k");
        check_cuda(cudaMalloc((void**) &vq, cells * n_head_kv * head_dim), "malloc v");
        check_cuda(cudaMalloc((void**) &ks, cells * n_head_kv * (head_dim / kk::KV_Q8_GROUP) * 2), "malloc ks");
        check_cuda(cudaMalloc((void**) &vs, cells * n_head_kv * (head_dim / kk::KV_Q8_GROUP) * 2), "malloc vs");

        float* d_q = to_device(q);
        float* d_k = to_device(k);
        float* d_v = to_device(v);
        float* d_o = nullptr;
        check_cuda(cudaMalloc((void**) &d_o, q.size() * sizeof(float)), "malloc out");

        for (int p = 0; p < n_pos; ++p) {
            int32_t step[kk::kStepCount] = {p, p + 1, 0, p + 1};
            check_cuda(cudaMemcpy(d_step, step, sizeof step, cudaMemcpyHostToDevice), "step");
            kk::kv_append_q8_step(kq, vq, ks, vs, d_table, d_step, &k[(size_t) p * n_head_kv * head_dim],
                                  &v[(size_t) p * n_head_kv * head_dim], shapes, nullptr, nullptr);
        }
        int32_t last = n_pos - 1;
        check_cuda(cudaMemcpy(d_step, &last, sizeof last, cudaMemcpyHostToDevice), "last");
        sk::KvQ8View view;
        view.k_q = kq; view.v_q = vq; view.k_scale = ks; view.v_scale = vs;
        view.page_table = d_table; view.step = d_step;
        view.page_size = page_size; view.n_head_kv = n_head_kv; view.head_dim = head_dim;
        sk::attn_q8_cached(d_q, d_o, view, n_pos, n_head, window, nullptr);
        check_cuda(cudaDeviceSynchronize(), "sync permuted");
        (run == 0 ? identity : permuted) = from_device(d_o, q.size());

        cudaFree(d_q); cudaFree(d_k); cudaFree(d_v); cudaFree(d_o);
        cudaFree(d_table); cudaFree(d_step); cudaFree(kq); cudaFree(vq); cudaFree(ks); cudaFree(vs);
    }
    const double diff = max_abs_diff(identity, permuted);
    char label[200];
    std::snprintf(label, sizeof label,
                  "a permuted page table at 8 cells per page changes nothing (max abs diff %.1e)", diff);
    check(diff == 0.0, label);
}

// ---- 5: a full-attention row PAST the sliding window, which is the case `attn_causal` cannot express: its scores
// live in a fixed 512-entry shared array.  Here a decode row at position 1023 reads 1024 cells and must still
// match the host.
void long_prefix_case() {
    const int n_pos = 1024, n_head = 64, n_head_kv = 8, head_dim = 128;
    std::vector<float> q = noise((size_t) n_pos * n_head * head_dim, 31u);
    std::vector<float> k = noise((size_t) n_pos * n_head_kv * head_dim, 32u);
    std::vector<float> v = noise((size_t) n_pos * n_head_kv * head_dim, 33u);

    sk::Session s;
    std::string err;
    if (!sk::session_create(s, 1, n_head_kv, head_dim, n_pos, err)) {
        std::printf("FAIL  long prefix: %s\n", err.c_str());
        ++failures;
        return;
    }
    float* d_q = to_device(q);
    float* d_k = to_device(k);
    float* d_v = to_device(v);
    float* d_o = nullptr;
    check_cuda(cudaMalloc((void**) &d_o, q.size() * sizeof(float)), "malloc out");
    if (!sk::session_append(s, 0, 0, n_pos, d_k, d_v, err) ||
        !sk::session_attend(s, 0, d_q + (size_t) (n_pos - 1) * n_head * head_dim, d_o, n_pos - 1, 1, n_head, 0, err)) {
        std::printf("FAIL  long prefix: %s\n", err.c_str());
        ++failures;
        return;
    }
    check_cuda(cudaDeviceSynchronize(), "sync long");
    const std::vector<float> got = from_device(d_o, (size_t) n_head * head_dim);

    std::vector<float> kdq(k.size()), vdq(k.size());
    for (int p = 0; p < n_pos; ++p) {
        std::vector<int8_t> kc, vc;
        std::vector<uint16_t> ks, vs;
        host_quantize(&k[(size_t) p * n_head_kv * head_dim], n_head_kv, head_dim, kc, ks);
        host_quantize(&v[(size_t) p * n_head_kv * head_dim], n_head_kv, head_dim, vc, vs);
        const std::vector<float> kf = dequantize(kc, ks, n_head_kv, head_dim);
        const std::vector<float> vf = dequantize(vc, vs, n_head_kv, head_dim);
        std::copy(kf.begin(), kf.end(), kdq.begin() + (size_t) p * n_head_kv * head_dim);
        std::copy(vf.begin(), vf.end(), vdq.begin() + (size_t) p * n_head_kv * head_dim);
    }
    std::vector<float> ref(q.size(), 0);
    host_row(q, kdq, vdq, n_pos - 1, n_head, n_head_kv, head_dim, 0, ref);
    const double rel = rel_l2(got, std::vector<float>(ref.begin() + (size_t) (n_pos - 1) * n_head * head_dim, ref.end()));
    char label[220];
    std::snprintf(label, sizeof label,
                  "a decode row at position 1023 over 1024 cached cells (no window) -> rel_l2 %.1e", rel);
    check(rel < 1e-5, label);

    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_o);
    sk::session_destroy(s);
}

// ---- 6: A RING ON A SLIDING LAYER.  The 33 sliding blocks of this model read a window of 512 and never look
// further back, so their cells past that window are written and read by nobody.  Here a layer with a window of 8
// keeps `8 + kKvPassMax - 1` cells rounded to pages instead of the whole 2048-position prefix, the positions lap
// that ring twice, and the answer must still be the host's - which is the only thing that says the page table's
// repeat lands on the right cells.  A pass too long for the ring must be REFUSED, not answered.
void ring_case() {
    const int n_pos = 2048, n_head = 96, n_head_kv = 8, head_dim = 128, window = 8;
    std::vector<float> q = noise((size_t) n_pos * n_head * head_dim, 41u);
    std::vector<float> k = noise((size_t) n_pos * n_head_kv * head_dim, 42u);
    std::vector<float> v = noise((size_t) n_pos * n_head_kv * head_dim, 43u);

    sk::Session s;
    std::string err;
    const std::vector<int> windows(1, window);
    if (!sk::session_create(s, 1, n_head_kv, head_dim, n_pos, err, windows)) {
        std::printf("FAIL  ring: %s\n", err.c_str());
        ++failures;
        return;
    }
    const int need = std::min(s.capacity, window + sk::kKvPassMax - 1);
    const int want = ((need + s.page_size - 1) / s.page_size) * s.page_size;
    check(sk::session_layer_cells(s, 0) == want && want < s.capacity,
          "a sliding layer's ring is the window plus one pass, not the whole prefix");

    float* d_q = to_device(q);
    float* d_k = to_device(k);
    float* d_v = to_device(v);
    float* d_o = nullptr;
    check_cuda(cudaMalloc((void**) &d_o, (size_t) n_head * head_dim * sizeof(float)), "malloc out");

    // two passes of kKvPassMax positions: 2048 cells written into a ring of `want`, so every slot is reused
    bool ok = true;
    for (int off = 0; ok && off < n_pos; off += sk::kKvPassMax) {
        const int n = std::min(sk::kKvPassMax, n_pos - off);
        ok = sk::session_append(s, 0, off, n, d_k + (size_t) off * n_head_kv * head_dim,
                                d_v + (size_t) off * n_head_kv * head_dim, err);
    }
    if (!ok) {
        std::printf("FAIL  ring: append: %s\n", err.c_str());
        ++failures;
        return;
    }
    check(s.layers[0].cells == want && s.n_cached == n_pos,
          "the pool holds the ring's cells while the sequence holds every position");

    // the table's repeat, read back: page i and page i + ring_pages are the same physical page
    const int n_pages = s.capacity / s.page_size, ring_pages = want / s.page_size;
    std::vector<int32_t> table((size_t) n_pages);
    check_cuda(cudaMemcpy(table.data(), s.layers[0].table, table.size() * sizeof(int32_t), cudaMemcpyDeviceToHost),
               "read table");
    bool table_ok = ring_pages < n_pages;
    for (int i = 0; i < n_pages; ++i) table_ok = table_ok && table[(size_t) i] == i % ring_pages;
    check(table_ok, "the ring's page table repeats every ring_pages, and the prefix's pages all map inside it");

    ok = sk::session_attend(s, 0, d_q + (size_t) (n_pos - 1) * n_head * head_dim, d_o, n_pos - 1, 1, n_head, window,
                            err);
    check_cuda(cudaDeviceSynchronize(), "sync ring");
    if (!ok) {
        std::printf("FAIL  ring: attend: %s\n", err.c_str());
        ++failures;
        return;
    }
    const std::vector<float> got = from_device(d_o, (size_t) n_head * head_dim);

    std::vector<float> kdq(k.size()), vdq(k.size());
    for (int p = 0; p < n_pos; ++p) {
        std::vector<int8_t> kc, vc;
        std::vector<uint16_t> ks, vs;
        host_quantize(&k[(size_t) p * n_head_kv * head_dim], n_head_kv, head_dim, kc, ks);
        host_quantize(&v[(size_t) p * n_head_kv * head_dim], n_head_kv, head_dim, vc, vs);
        const std::vector<float> kf = dequantize(kc, ks, n_head_kv, head_dim);
        const std::vector<float> vf = dequantize(vc, vs, n_head_kv, head_dim);
        std::copy(kf.begin(), kf.end(), kdq.begin() + (size_t) p * n_head_kv * head_dim);
        std::copy(vf.begin(), vf.end(), vdq.begin() + (size_t) p * n_head_kv * head_dim);
    }
    std::vector<float> ref(q.size(), 0);
    host_row(q, kdq, vdq, n_pos - 1, n_head, n_head_kv, head_dim, window, ref);
    const double rel = rel_l2(got, std::vector<float>(ref.begin() + (size_t) (n_pos - 1) * n_head * head_dim, ref.end()));
    char label[240];
    std::snprintf(label, sizeof label,
                  "the row of position %d, after the ring lapped it, is the host's (rel_l2 %.1e)", n_pos - 1, rel);
    check(rel < 1e-5, label);

    // a row whose window the ring has already handed to a newer position must be refused, not assembled from
    // other tokens' cells
    const bool recycled = !sk::session_attend(s, 0, d_q, d_o, 0, 1, n_head, window, err);
    std::snprintf(label, sizeof label, "a row at position 0, whose cells the ring recycled, is refused: %s",
                  recycled ? err.c_str() : "it was answered");
    check(recycled && err.find("handed") != std::string::npos, label);

    // a pass that would lap the ring before its own last row is written must not be answered at all
    const bool refused = !sk::session_append(s, 0, 0, n_pos, d_k, d_v, err);
    std::snprintf(label, sizeof label, "a pass of %d positions into a %d-cell ring is refused: %s", n_pos, want,
                  refused ? err.c_str() : "it was accepted");
    check(refused && err.find("ring holds") != std::string::npos, label);

    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_o);
    sk::session_destroy(s);
}

// ---- 7: THE FIGURES THE DOC QUOTES, on the model's real geometry, from the session's own arithmetic - so the
// numbers in STEP37.md are pinned by a test rather than by a print nobody can re-run without the 76 GB file.
void ring_sizes_case() {
    const int n_layers = 45, n_kv = 8, hd = 128, window = 512;
    std::vector<int> windows((size_t) n_layers, window);
    for (int l = 0; l < n_layers; l += 4) windows[(size_t) l] = 0;   // one full-attention block in four
    for (const int capacity : {8192, 32768, 131072}) {
        sk::Session s;
        std::string err;
        if (!sk::session_create(s, n_layers, n_kv, hd, capacity, err, windows)) {
            std::printf("FAIL  sizes at %d: %s\n", capacity, err.c_str());
            ++failures;
            return;
        }
        const double ring_gb = (double) sk::session_bytes_at(s, s.capacity) / 1e9;
        const double all_gb = (double) s.capacity * (double) sk::session_bytes_per_token(s) / 1e9;
        char label[280];
        std::snprintf(label, sizeof label,
                      "45 layers at %d, one full block in four: %.3f GB with the ring, %.3f GB keeping every "
                      "position", s.capacity, ring_gb, all_gb);
        check(ring_gb < all_gb && (capacity < 32768 || ring_gb < all_gb * 0.32), label);
        sk::session_destroy(s);
    }
}

}  // namespace

int main() {
    session_case(40, 96, 8, 128, 0, false, "sliding block, no window");
    session_case(40, 96, 8, 128, 16, false, "sliding block, window shorter than the sequence");
    session_case(40, 64, 8, 128, 0, true, "full block, no window, appended in two chunks");
    session_case(24, 96, 8, 128, 512, false, "window longer than the sequence");
    permuted_page_case(40, 96, 8, 128, 0);
    permuted_page_case(37, 64, 8, 128, 16);
    long_prefix_case();
    ring_case();
    ring_sizes_case();

    if (failures) {
        std::printf("step35_kv_parity: %d check(s) failed\n", failures);
        return 1;
    }
    std::printf("step35_kv_parity: all checks passed\n");
    return 0;
}
