// include/strata/step35/attn.hpp - causal grouped-query attention with Step-3.7-Flash's sliding window.
//
// Every parameter is per BLOCK, not compiled in: a full-attention block has 64 query heads and no window, a
// sliding one has 96 and a window of 512 (the release's header says so, block by block).  A kernel built for
// one shape would silently mis-run the other, which is why this is a parameter and the parity test covers both.
//
// This entry point takes the whole K/V of the sequence, in fp32, and is the vehicle that says the arithmetic is
// right.  The engine's decode path keeps the cache quantized (strata_kernels' kv_q8) and will read it in place;
// that path is a different kernel with its own parity test, not this one.
#pragma once

#include <cstdint>

namespace strata::step35::kernels {

/// out[pos][h][d] = sum over j in [max(0, pos-window+1), pos] of softmax_j(q[pos][h] . k[j][h/group]) * v[j][h/group],
/// with `group` = n_head / n_head_kv.  `window` <= 0 means the whole prefix.  All tensors are device fp32,
/// `head_dim` contiguous floats per head.
void attn_causal(const float* q, const float* k, const float* v, float* out, int n_pos, int n_head, int n_head_kv,
                 int head_dim, int window, void* stream);

/// One layer's cache as the decode path reads it: kv_q8.hpp's layout, behind a page table, with the position in
/// the device step buffer.  Passing this rather than a Session keeps the kernel answerable to a test that builds
/// the pools by hand.
struct KvQ8View {
    const int8_t* k_q = nullptr;
    const int8_t* v_q = nullptr;
    const uint16_t* k_scale = nullptr;
    const uint16_t* v_scale = nullptr;
    const int32_t* page_table = nullptr;   ///< logical page -> physical page
    const int32_t* step = nullptr;         ///< qsa.hpp's kStepPos: the position of the LAST query row
    int page_size = 0;
    int n_head_kv = 0;
    int head_dim = 0;
};

/// The same arithmetic as `attn_causal`, over the quantized cache instead of an fp32 K/V tensor:
/// out[r][h][d] = softmax over j in [max(0, pos_r - window + 1), pos_r] of q[r][h].k'[j][h/group] * v'[j][h/group],
/// with pos_r = step[kStepPos] - (n_pos - 1 - r) and k'/v' the dequantized cells (`code * scale`).  A `window` <= 0
/// means the whole prefix.  `head_dim` must be a multiple of 4 and at most 512.
///
/// WHY IT IS NOT `attn_causal` WITH A GATHER.  Two reasons, and both matter at a long context: the gathered fp16
/// scratch would be 4 KB per cell per layer (512 MB at 131072) moved on every token, and `attn_causal` keeps a
/// window's scores in a 512-entry shared array, which a full-attention block outgrows at position 512 (513 keys:
/// measured 1.6e-03 off the float64 answer there, 5.0e-02 at 1024, with no error reported).  This kernel keeps the
/// running max, denominator and weighted value vector in registers instead, so it has no window bound at all.
void attn_q8_cached(const float* q, float* out, const KvQ8View& kv, int n_pos, int n_head, int window, void* stream);

}  // namespace strata::step35::kernels
