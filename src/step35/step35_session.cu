// src/step35/step35_session.cu - the quantized K/V store between passes.  See session.hpp.
//
// This file is the OWNER and the glue; it invents no arithmetic.  The cells are `strata_kernels`' kv_q8
// (`kv_append_q8_steps` quantizes a token's K and V into them, `attn_q8_cached` reads them back in place), the
// layout is that header's, and the position lives in the device step buffer qsa.hpp describes.  What is new here
// is only the ownership: one set of pools per layer, a page table, and a count.
#include "strata/step35/session.hpp"

#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/qsa.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <vector>

namespace strata::step35::kernels {
namespace {

using strata::kernels::KV_Q8_GROUP;
using strata::kernels::QsaShapes;
using strata::kernels::kStepCount;
using strata::kernels::kStepNKv;
using strata::kernels::kStepPos;
using strata::kernels::kStepWidth;

/// The geometry the cache kernels are handed.  `n_head` and the rope settings belong to the attention and are not
/// in here: every block of this model has 8 KV heads of 128 dims, and only the query head count changes.
QsaShapes shapes_of(const Session& s) {
    QsaShapes sh;
    sh.n_head_kv = s.n_head_kv;
    sh.head_dim = s.head_dim;
    sh.page_size = s.page_size;
    return sh;
}

int pages_of(const Session& s) { return (s.capacity + s.page_size - 1) / s.page_size; }

bool set_err(std::string& err, const std::string& what) {
    err = what;
    return false;
}

bool cuda_ok(cudaError_t e, const std::string& what, std::string& err) {
    if (e == cudaSuccess) return true;
    err = what + ": " + cudaGetErrorString(e);
    return false;
}

/// Bytes one layer holds: every allocated page, codes and scales.
uint64_t layer_bytes(const Session& s) {
    return (uint64_t) pages_of(s) * s.page_size * strata::kernels::kv_q8_bytes_per_cell(shapes_of(s));
}

/// Allocates one layer's pools, once.  Idempotent, so a caller can simply call it before every use.
bool session_layer(Session& s, int layer, std::string& err) {
    if (layer < 0 || layer >= s.n_layers) return set_err(err, "layer outside the cache's range");
    KvLayer& l = s.layers[(size_t) layer];
    if (l.k_q != nullptr) return true;
    const size_t cells = (size_t) pages_of(s) * s.page_size;
    const size_t codes = cells * (size_t) s.n_head_kv * s.head_dim;
    const size_t scales = cells * (size_t) s.n_head_kv * (s.head_dim / KV_Q8_GROUP);
    const std::string what = "allocate layer " + std::to_string(layer);
    if (!cuda_ok(cudaMalloc((void**) &l.k_q, codes), what, err) ||
        !cuda_ok(cudaMalloc((void**) &l.v_q, codes), what, err) ||
        !cuda_ok(cudaMalloc((void**) &l.k_scale, scales * sizeof(uint16_t)), what, err) ||
        !cuda_ok(cudaMalloc((void**) &l.v_scale, scales * sizeof(uint16_t)), what, err)) {
        cudaFree(l.k_q); cudaFree(l.v_q); cudaFree(l.k_scale); cudaFree(l.v_scale);
        l = KvLayer{};
        return false;
    }
    return true;
}

}  // namespace

uint64_t session_bytes_per_token(const Session& s) {
    return (uint64_t) s.n_layers * strata::kernels::kv_q8_bytes_per_cell(shapes_of(s));
}

uint64_t session_bytes(const Session& s) {
    uint64_t n = (uint64_t) pages_of(s) * sizeof(int32_t) + (uint64_t) kStepCount * sizeof(int32_t) +
                 (uint64_t) s.capacity * kStepCount * sizeof(int32_t);
    for (const KvLayer& l : s.layers)
        if (l.k_q != nullptr) n += layer_bytes(s);
    return n;
}

bool session_create(Session& s, int n_layers, int n_head_kv, int head_dim, int capacity, std::string& err) {
    if (n_layers <= 0 || n_head_kv <= 0 || head_dim <= 0 || capacity <= 0)
        return set_err(err, "a cache needs layers, KV heads, a head width and a capacity");
    if (head_dim % KV_Q8_GROUP != 0)
        return set_err(err, "head_dim must be a multiple of " + std::to_string(KV_Q8_GROUP) + " for the int8 cell");
    s.n_layers = n_layers;
    s.n_head_kv = n_head_kv;
    s.head_dim = head_dim;
    s.page_size = kKvPageSize;
    // Round the capacity up to whole pages: the last partial page is allocated either way, so a `capacity` that is
    // not a multiple of the granule would be a bound the pools do not really have.
    s.capacity = ((capacity + s.page_size - 1) / s.page_size) * s.page_size;
    s.n_cached = 0;
    s.layers.assign((size_t) n_layers, KvLayer{});

    const int n_pages = pages_of(s);
    std::vector<int32_t> table((size_t) n_pages);
    for (int i = 0; i < n_pages; ++i) table[(size_t) i] = i;   // the identity: logical page -> itself
    if (!cuda_ok(cudaMalloc((void**) &s.page_table, (size_t) n_pages * sizeof(int32_t)), "allocate page table", err))
        return false;
    if (!cuda_ok(cudaMemcpy(s.page_table, table.data(), (size_t) n_pages * sizeof(int32_t), cudaMemcpyHostToDevice),
                 "upload page table", err))
        return false;
    if (!cuda_ok(cudaMalloc((void**) &s.step, (size_t) kStepCount * sizeof(int32_t)), "allocate step", err))
        return false;
    if (!cuda_ok(cudaMemset(s.step, 0, (size_t) kStepCount * sizeof(int32_t)), "zero step", err)) return false;
    // One step buffer per position of a pass: the append takes an ARRAY of positions (token j at j * kStepCount),
    // which is what lets one launch write a whole prompt's cells instead of one launch per token.
    if (!cuda_ok(cudaMalloc((void**) &s.steps, (size_t) s.capacity * kStepCount * sizeof(int32_t)),
                 "allocate step array", err))
        return false;
    return true;
}

void session_destroy(Session& s) {
    for (KvLayer& l : s.layers) {
        cudaFree(l.k_q);
        cudaFree(l.v_q);
        cudaFree(l.k_scale);
        cudaFree(l.v_scale);
        l = KvLayer{};
    }
    s.layers.clear();
    cudaFree(s.page_table);
    cudaFree(s.step);
    cudaFree(s.steps);
    s.page_table = nullptr;
    s.step = nullptr;
    s.steps = nullptr;
    s.n_cached = 0;
}

bool session_append(Session& s, int layer, int pos0, int n_tok, const float* kcur, const float* vcur,
                    std::string& err) {
    if (n_tok <= 0) return true;
    if (pos0 < 0 || pos0 + n_tok > s.capacity)
        return set_err(err, "position " + std::to_string(pos0) + "+" + std::to_string(n_tok) +
                                " is outside the cache's " + std::to_string(s.capacity) + " cells");
    if (!session_layer(s, layer, err)) return false;
    const KvLayer& l = s.layers[(size_t) layer];

    // The per-position step buffers.  Only kStepPos is read by the append, but the other three are filled
    // consistently anyway: a buffer half-written is how a caller ends up reading a stale `n_kv`.
    std::vector<int32_t> host((size_t) n_tok * kStepCount, 0);
    for (int j = 0; j < n_tok; ++j) {
        const int pos = pos0 + j;
        host[(size_t) j * kStepCount + kStepPos] = pos;
        host[(size_t) j * kStepCount + kStepNKv] = pos + 1;
        host[(size_t) j * kStepCount + kStepWidth] = pos + 1;   // no selection here: the width is the prefix
    }
    if (!cuda_ok(cudaMemcpy(s.steps, host.data(), host.size() * sizeof(int32_t), cudaMemcpyHostToDevice),
                 "upload positions", err))
        return false;
    strata::kernels::kv_append_q8_steps(l.k_q, l.v_q, l.k_scale, l.v_scale, s.page_table, s.steps, kStepCount, kcur,
                                        vcur, s.n_head_kv * s.head_dim, n_tok, shapes_of(s), nullptr, nullptr);
    if (!cuda_ok(cudaGetLastError(), "append cells", err)) return false;
    if (pos0 + n_tok > s.n_cached) s.n_cached = pos0 + n_tok;
    return true;
}

bool session_attend(Session& s, int layer, const float* q, float* out, int pos0, int n_tok, int n_head, int window,
                    std::string& err) {
    if (n_tok <= 0) return true;
    if (pos0 < 0 || pos0 + n_tok > s.capacity)
        return set_err(err, "attention at " + std::to_string(pos0) + "+" + std::to_string(n_tok) +
                                " is outside the cache's " + std::to_string(s.capacity) + " cells");
    if (!session_layer(s, layer, err)) return false;
    const KvLayer& l = s.layers[(size_t) layer];

    // One writer for the position the attention reads: the last row's.  A per-row host scalar would freeze at the
    // first row in a captured graph, which is exactly the failure qsa.hpp's step buffer exists to prevent.
    const int32_t last = pos0 + n_tok - 1;
    if (!cuda_ok(cudaMemcpy(s.step, &last, sizeof(int32_t), cudaMemcpyHostToDevice), "upload position", err))
        return false;

    KvQ8View kv;
    kv.k_q = l.k_q;
    kv.v_q = l.v_q;
    kv.k_scale = l.k_scale;
    kv.v_scale = l.v_scale;
    kv.page_table = s.page_table;
    kv.step = s.step;
    kv.page_size = s.page_size;
    kv.n_head_kv = s.n_head_kv;
    kv.head_dim = s.head_dim;
    attn_q8_cached(q, out, kv, n_tok, n_head, window, nullptr);
    return cuda_ok(cudaGetLastError(), "attention", err);
}

}  // namespace strata::step35::kernels
