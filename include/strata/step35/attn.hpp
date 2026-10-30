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

}  // namespace strata::step35::kernels
