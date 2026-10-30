// include/strata/step35/session.hpp - one conversation's K/V, quantized, kept between passes.
//
// A prompt could run but its K/V was thrown away (`step35_seq.cu` allocated q/k/v inside the block loop and freed
// them at the end of the pass), so a second turn recomputed the first.  This is the store that makes a turn cheap:
// per layer, the cells of `strata_kernels`' kv_q8 layout - `int8` codes plus one `f16` scale per KV_Q8_GROUP (64)
// dimensions, for K and for V - behind a page table, so the attention reads a cell through
// `page_table[cell / page_size]` and never assumes the mapping.
//
// WHY THE QUANTIZED FORM FIRST.  8 KV heads x 128 dims = 1024 values per layer per token, so a cell is
// 2112 bytes per layer (1024 codes + 16 f16 scales, twice) and ~95 KB for the 45 layers: 0.78 GB at 8192
// positions, 3.1 GB at 32768, 12.4 GB at 131072.  The same cache in fp32 would be 4.5x that, which does not fit
// beside the experts on 32 GB of VRAM.
//
// THE ONE THING TO GET RIGHT.  `pos` is not an argument of the kernels: it lives in the device step buffer
// (`qsa.hpp`'s kStepPos) that the append and the attention both read, so a captured graph stays valid across
// tokens.  A caller writes it once per pass, through `session_append` / `session_attend` below, and never by hand.
#pragma once

#include "strata/step35/attn.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace strata::step35::kernels {

/// Cells per page.  The granule is coarse on purpose (one sequence per engine, not hundreds): it buys nothing to
/// make it fine and costs one indirection per cell.  It is a free choice - `kv_q8` and the attention resolve a row
/// per cell - and the parity test runs a permuted table to prove the mapping cannot leak into the result.
inline constexpr int kKvPageSize = 128;

/// One layer's cache: K and V codes and their scales, `[page][kv_head][page_size][head_dim]` (head_dim fastest)
/// for the codes and `[page][kv_head][page_size][head_dim / KV_Q8_GROUP]` for the scales, as kv_q8.hpp lays out.
struct KvLayer {
    int8_t* k_q = nullptr;
    int8_t* v_q = nullptr;
    uint16_t* k_scale = nullptr;
    uint16_t* v_scale = nullptr;
};

/// A conversation's cache.  `page_size`, `n_head_kv` and `head_dim` are the artifact's geometry, not per block:
/// Step-3.7-Flash has 8 KV heads of 128 dims on every block, and only the QUERY head count changes (64 / 96).
struct Session {
    int n_layers = 0;
    int n_head_kv = 0;
    int head_dim = 0;
    int page_size = kKvPageSize;
    int capacity = 0;    ///< cells the cache holds
    int n_cached = 0;    ///< cells written so far (the next free position; never used as an argument)
    int32_t* page_table = nullptr;   ///< (n_pages,) i32, identity unless a caller permutes it for a test
    int32_t* step = nullptr;         ///< (kStepCount,) i32 on device: what the kernels read the position from
    int32_t* steps = nullptr;        ///< (capacity * kStepCount,) i32: one step buffer per prefilled position
    std::vector<KvLayer> layers;     ///< one entry per layer, allocated on first use
};

/// Bytes one cell costs, all layers included (the figure the doc quotes: ~95 KB per token for 45 layers).
uint64_t session_bytes_per_token(const Session& s);
/// Bytes currently allocated by the session, page table and step buffers included.
uint64_t session_bytes(const Session& s);

/// Allocates the page table (the identity mapping), the step buffers and the per-layer table; a layer's own
/// buffers are allocated when it is first appended to, so `--layers 4` does not pay for 45.
bool session_create(Session& s, int n_layers, int n_head_kv, int head_dim, int capacity, std::string& err);
void session_destroy(Session& s);

/// Takes the K and V of `n_tok` positions (already normed and rotated, exactly as the attention wants them),
/// starting at sequence position `pos0`, and writes their cells.  K/V are `[n_tok][n_head_kv][head_dim]` fp32 on
/// the device; the cell's scale is `f16(max|x| / 127)` over each 64-dimension group, as kv_q8.hpp specifies.
bool session_append(Session& s, int layer, int pos0, int n_tok, const float* kcur, const float* vcur,
                    std::string& err);

/// Attention for `n_tok` query rows whose positions are `pos0 .. pos0+n_tok-1`, over the cells `0 .. pos0+n_tok-1`
/// of this layer's cache (the row is causal: row r sees `0 .. pos0+r`, and only the last `window` of those when
/// `window > 0`).  `q` and `out` are `[n_tok][n_head][head_dim]` fp32 on the device.  Reads the codes in place -
/// no gather, no fp16 staging - which is what keeps the full-attention layers affordable: a gather of 131072 cells
/// would move 512 MB per layer per token, the in-place read moves the codes once.
bool session_attend(Session& s, int layer, const float* q, float* out, int pos0, int n_tok, int n_head, int window,
                    std::string& err);

}  // namespace strata::step35::kernels
