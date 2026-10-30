// src/step35/step35_attn_cached.cu - causal GQA attention over the QUANTIZED cache, in place.  See attn.hpp.
//
// One warp per (query row, query head), as `attn_causal` does - but the keys and values are read as `int8` codes
// and one `f16` scale per 64 values, straight out of the paged pool, and the softmax is ONLINE: the running max,
// the denominator and the weighted value vector live in registers, so nothing here is sized by the window.  That
// is the difference that matters at a long context: `attn_causal` carries the window's scores in a fixed
// `scores[512]` of shared memory, which a full-attention block outgrows at position 511, and a gather into fp16
// scratch would move 4 KB per cell per layer (512 MB at 131072) on every token.
//
// The lanes are laid out like the CELLS are: lane l owns dimensions 4l .. 4l+3, i.e. one `char4` load of codes and
// one scale - so `head_dim` 128 is 32 lanes with no leftovers, and each lane's four accumulators are contiguous
// dimensions of the output.  The scale of a group of 64 is shared by 16 lanes and is loaded per lane anyway: 2
// bytes beside 4 bytes of codes, and the cache line is already open.
#include "strata/step35/attn.hpp"

#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/qsa.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace strata::step35::kernels {
namespace {

using strata::kernels::f32_from_f16;

/// One warp's dot product, over the whole warp.
__device__ float warp_sum(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
    return v;
}

constexpr int kMaxQuads = 4;   // 4 dimensions per lane x 32 lanes x 4 = head_dim 512, the widest supported

__global__ void k_attn_q8(const float* __restrict__ q, float* __restrict__ out,
                          const int8_t* __restrict__ k_q, const int8_t* __restrict__ v_q,
                          const uint16_t* __restrict__ k_scale, const uint16_t* __restrict__ v_scale,
                          const int32_t* __restrict__ table, const int32_t* __restrict__ step, int n_pos,
                          int n_head, int n_head_kv, int head_dim, int page_size, int window) {
    const int r = blockIdx.x, h = blockIdx.y;
    const int lane = threadIdx.x;
    const int group = n_head / n_head_kv;
    const int kvh = h / group;
    // the position of this row: the step buffer holds the position of the LAST row, and the rows of one pass are
    // consecutive, so the earlier rows sit behind it.  One writer (session_attend), no per-row host scalar.
    const int pos = __ldg(step + strata::kernels::kStepPos) - (n_pos - 1 - r);
    if (pos < 0) return;
    const int first = (window > 0 && pos + 1 > window) ? pos - window + 1 : 0;
    const int nq = head_dim / 4;
    const int nacc = (nq + 31) / 32;
    const int groups = head_dim / strata::kernels::KV_Q8_GROUP;
    const float scale = rsqrtf((float) head_dim);
    const float* qh = q + ((size_t) r * n_head + h) * head_dim;

    float acc[kMaxQuads][4];
    for (int i = 0; i < nacc; ++i)
        for (int t = 0; t < 4; ++t) acc[i][t] = 0.0f;
    float mx = -INFINITY, den = 0.0f;

    for (int j = first; j <= pos; ++j) {
        const int cell = j;
        const long long page = (long long) __ldg(table + cell / page_size);
        const long long row = (page * n_head_kv + kvh) * page_size + (cell % page_size);
        const int8_t* krow = k_q + row * head_dim;
        const uint16_t* ksc = k_scale + row * groups;
        float s = 0.0f;
        for (int i = 0, d4 = lane; i < nacc; ++i, d4 += 32) {
            if (d4 >= nq) break;
            const float ks = f32_from_f16(__ldg(ksc + (d4 * 4) / strata::kernels::KV_Q8_GROUP));
            const char4 kc = *reinterpret_cast<const char4*>(krow + d4 * 4);
            s += (qh[d4 * 4 + 0] * (float) kc.x + qh[d4 * 4 + 1] * (float) kc.y +
                  qh[d4 * 4 + 2] * (float) kc.z + qh[d4 * 4 + 3] * (float) kc.w) *
                 ks;
        }
        s = warp_sum(s) * scale;
        // online softmax: the whole warp holds the same s after the reduction, so the rescale is one per lane
        const float mn = fmaxf(mx, s);
        const float c = (mx == -INFINITY) ? 0.0f : expf(mx - mn);
        const float p = expf(s - mn);
        den = den * c + p;
        mx = mn;
        const int8_t* vrow = v_q + row * head_dim;
        const uint16_t* vsc = v_scale + row * groups;
        for (int i = 0, d4 = lane; i < nacc; ++i, d4 += 32) {
            if (d4 >= nq) break;
            const float vs = f32_from_f16(vsc[(d4 * 4) / strata::kernels::KV_Q8_GROUP]);
            const char4 vc = *reinterpret_cast<const char4*>(vrow + d4 * 4);
            acc[i][0] = acc[i][0] * c + p * (float) vc.x * vs;
            acc[i][1] = acc[i][1] * c + p * (float) vc.y * vs;
            acc[i][2] = acc[i][2] * c + p * (float) vc.z * vs;
            acc[i][3] = acc[i][3] * c + p * (float) vc.w * vs;
        }
    }

    float* o = out + ((size_t) r * n_head + h) * head_dim;
    const float inv = den > 0.0f ? 1.0f / den : 0.0f;
    for (int i = 0, d4 = lane; i < nacc; ++i, d4 += 32) {
        if (d4 >= nq) break;
        for (int t = 0; t < 4; ++t) o[d4 * 4 + t] = acc[i][t] * inv;
    }
}

}  // namespace

void attn_q8_cached(const float* q, float* out, const KvQ8View& kv, int n_pos, int n_head, int window, void* stream) {
    if (kv.head_dim % 4 != 0 || kv.head_dim <= 0 || kv.head_dim > 128 * kMaxQuads) {
        std::fprintf(stderr, "attn_q8_cached: head_dim %d must be a multiple of 4 and at most %d\n", kv.head_dim,
                     128 * kMaxQuads);
        std::exit(1);
    }
    if (kv.n_head_kv <= 0 || n_head % kv.n_head_kv != 0 || kv.head_dim % strata::kernels::KV_Q8_GROUP != 0) {
        std::fprintf(stderr, "attn_q8_cached: %d query heads over %d KV heads, head_dim %d, needs head_dim a "
                             "multiple of %d\n", n_head, kv.n_head_kv, kv.head_dim, strata::kernels::KV_Q8_GROUP);
        std::exit(1);
    }
    if (n_pos <= 0) return;
    k_attn_q8<<<dim3(n_pos, n_head), 32, 0, (cudaStream_t) stream>>>(
        q, out, kv.k_q, kv.v_q, kv.k_scale, kv.v_scale, kv.page_table, kv.step, n_pos, n_head, kv.n_head_kv,
        kv.head_dim, kv.page_size, window);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) std::fprintf(stderr, "attn_q8_cached launch: %s\n", cudaGetErrorString(e));
}

}  // namespace strata::step35::kernels
