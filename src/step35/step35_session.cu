// src/step35/step35_session.cu - the quantized K/V store between passes.  See session.hpp.
//
// This file is the OWNER and the glue; it invents no arithmetic.  The cells are `strata_kernels`' kv_q8
// (`kv_append_q8_steps` quantizes a token's K and V into them, `attn_q8_cached` reads them back in place), the
// layout is that header's, and the position lives in the device step buffer qsa.hpp describes.  What is new here
// is only the ownership: one set of pools per layer, one page table per layer, and a count.
#include "strata/step35/session.hpp"

#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/qsa.hpp"

#include <cuda_runtime.h>

#include <algorithm>
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

int pages_for(const Session& s, int cells) { return (cells + s.page_size - 1) / s.page_size; }

bool set_err(std::string& err, const std::string& what) {
    err = what;
    return false;
}

bool cuda_ok(cudaError_t e, const std::string& what, std::string& err) {
    if (e == cudaSuccess) return true;
    err = what + ": " + cudaGetErrorString(e);
    return false;
}

/// Bytes a layer holds with `cells` allocated: every page, codes and scales.
uint64_t layer_bytes(const Session& s, int cells) {
    return (uint64_t) pages_for(s, cells) * s.page_size * strata::kernels::kv_q8_bytes_per_cell(shapes_of(s));
}

/// The table of one layer: logical page -> physical page, `capacity / page_size` entries.  A full layer maps every
/// page to itself; a ring layer repeats its own `cells / page_size` pages, so position p and p + cells share a
/// physical page and the kernel - which only ever reads `table[cell / page_size]` - needs no notion of a ring.
std::vector<int32_t> table_of(const Session& s, int layer) {
    const int n_pages = pages_of(s);
    const int ring_pages = pages_for(s, session_layer_cells(s, layer));
    std::vector<int32_t> t((size_t) n_pages);
    for (int i = 0; i < n_pages; ++i) t[(size_t) i] = i % ring_pages;
    return t;
}

/// Allocates one layer's pools and its table, once.  Idempotent, so a caller can simply call it before every use.
bool session_layer(Session& s, int layer, std::string& err) {
    if (layer < 0 || layer >= s.n_layers) return set_err(err, "layer outside the cache's range");
    KvLayer& l = s.layers[(size_t) layer];
    if (l.k_q != nullptr) return true;
    const int cell_cap = session_layer_cells(s, layer);
    const size_t cells = (size_t) pages_for(s, cell_cap) * s.page_size;
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
    l.cells = (int) cells;
    l.ring = s.windows[(size_t) layer] > 0 ? (int) cells : 0;
    const std::vector<int32_t> table = table_of(s, layer);
    if (!cuda_ok(cudaMalloc((void**) &l.table, table.size() * sizeof(int32_t)), what, err) ||
        !cuda_ok(cudaMemcpy(l.table, table.data(), table.size() * sizeof(int32_t), cudaMemcpyHostToDevice), what,
                 err)) {
        cudaFree(l.k_q); cudaFree(l.v_q); cudaFree(l.k_scale); cudaFree(l.v_scale); cudaFree(l.table);
        l = KvLayer{};
        return false;
    }
    return true;
}

}  // namespace

int session_layer_cells(const Session& s, int layer) {
    if (layer < 0 || layer >= s.n_layers || (size_t) layer >= s.windows.size()) return 0;
    const int w = s.windows[(size_t) layer];
    if (w <= 0 || w >= s.capacity) return s.capacity;
    // The window, plus what one pass needs on top of it: a pass of `n` positions at a late position reads
    // `window + n - 1` cells before its last row is done.  Rounded up to whole pages, so the ring's own pages line
    // up with the table's repeat.
    const int need = std::min(s.capacity, w + kKvPassMax - 1);
    return std::min(s.capacity, ((need + s.page_size - 1) / s.page_size) * s.page_size);
}

uint64_t session_bytes_per_token(const Session& s) {
    return (uint64_t) s.n_layers * strata::kernels::kv_q8_bytes_per_cell(shapes_of(s));
}

uint64_t session_bytes_at(const Session& s, int n_positions) {
    const int n = std::max(0, n_positions);
    uint64_t bytes = (uint64_t) pages_of(s) * sizeof(int32_t) * (uint64_t) std::max(1, s.n_layers) +
                     (uint64_t) kStepCount * sizeof(int32_t) +
                     (uint64_t) std::min(s.capacity, n) * kStepCount * sizeof(int32_t);
    const uint64_t cell = strata::kernels::kv_q8_bytes_per_cell(shapes_of(s));
    for (int l = 0; l < s.n_layers; ++l)
        bytes += (uint64_t) pages_for(s, std::min(session_layer_cells(s, l), n)) * s.page_size * cell;
    return bytes;
}

uint64_t session_bytes(const Session& s) {
    uint64_t n = (uint64_t) kStepCount * sizeof(int32_t) + (uint64_t) s.capacity * kStepCount * sizeof(int32_t);
    for (const KvLayer& l : s.layers)
        if (l.k_q != nullptr) n += layer_bytes(s, l.cells) + (uint64_t) pages_of(s) * sizeof(int32_t);
    return n;
}

bool session_create(Session& s, int n_layers, int n_head_kv, int head_dim, int capacity, std::string& err,
                    const std::vector<int>& per_layer_window) {
    if (n_layers <= 0 || n_head_kv <= 0 || head_dim <= 0 || capacity <= 0)
        return set_err(err, "a cache needs layers, KV heads, a head width and a capacity");
    if (head_dim % KV_Q8_GROUP != 0)
        return set_err(err, "head_dim must be a multiple of " + std::to_string(KV_Q8_GROUP) + " for the int8 cell");
    if (!per_layer_window.empty() && (int) per_layer_window.size() != n_layers)
        return set_err(err, "the window list has " + std::to_string(per_layer_window.size()) + " entries for " +
                                std::to_string(n_layers) + " layers");
    s.n_layers = n_layers;
    s.n_head_kv = n_head_kv;
    s.head_dim = head_dim;
    s.page_size = kKvPageSize;
    // Round the capacity up to whole pages: the last partial page is allocated either way, so a `capacity` that is
    // not a multiple of the granule would be a bound the pools do not really have.
    s.capacity = ((capacity + s.page_size - 1) / s.page_size) * s.page_size;
    s.n_cached = 0;
    s.windows = per_layer_window.empty() ? std::vector<int>((size_t) n_layers, 0) : per_layer_window;
    s.layers.assign((size_t) n_layers, KvLayer{});

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
        cudaFree(l.table);
        l = KvLayer{};
    }
    s.layers.clear();
    s.windows.clear();
    cudaFree(s.step);
    cudaFree(s.steps);
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

    // A ring layer holds `window + kKvPassMax - 1` cells, and a pass writes all its cells before any of its rows is
    // attended: if the pass is long enough to lap the ring, the first row would read cells the last row already
    // overwrote.  Refusing is the only honest answer - the alternative is a plausible wrong number.
    if (l.ring > 0 && l.cells < s.capacity) {
        const int span = n_tok + std::min(s.windows[(size_t) layer], pos0 + 1) - 1;
        if (span > l.cells)
            return set_err(err, "a pass of " + std::to_string(n_tok) + " position(s) at " + std::to_string(pos0) +
                                    " needs " + std::to_string(span) + " cells, layer " + std::to_string(layer) +
                                    "'s ring holds " + std::to_string(l.cells) + " (window " +
                                    std::to_string(s.windows[(size_t) layer]) + ", at most " +
                                    std::to_string(kKvPassMax) + " positions per pass)");
    }

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
    strata::kernels::kv_append_q8_steps(l.k_q, l.v_q, l.k_scale, l.v_scale, l.table, s.steps, kStepCount, kcur,
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
    kv.page_table = l.table;
    kv.step = s.step;
    kv.page_size = s.page_size;
    kv.n_head_kv = s.n_head_kv;
    kv.head_dim = s.head_dim;
    attn_q8_cached(q, out, kv, n_tok, n_head, window, nullptr);
    return cuda_ok(cudaGetLastError(), "attention", err);
}

}  // namespace strata::step35::kernels
