// src/minimaxm2/m2_engine.cu - strata-m2: the engine path for MiniMax-M2 (GGUF architecture minimax-m2).
// A driver of its own beside Flash-Next's (generate/verify/prefill stay untouched, so the Flash-Next identity
// gate cannot move), built from Strata's kernels.
//
//   * weights straight from the GGUF: dense Q4_K / i-quant matrices in their blocks (native MMVQ), the 1-D tensors
//     and the F32 projections (the router, the router's per-expert bias) as f32, the routed experts repacked at
//     load into native expert blobs [gate | up | down] per (layer, expert) for `native_expert_grouped`;
//   * the plain pre-norm residual layer driver: x += attn(rms(x) * attn_norm); x += moe(rms(x) * ffn_norm);
//   * attention without an indexer and without an output gate: dense, 48 query heads over 4 KV heads, the QK norms
//     PER LAYER (one vector for all the query heads, one for all the key heads), split-K over 256-cell chunks
//     with a log-sum-exp merge, causal for a batch of queries (the prompt) and for one (decode);
//   * MoE: MiniMax-M2's router (sigmoid gate, a per-expert selection bias for the choice, the unbiased sigmoid for
//     the weight, the four weights normalised and scaled by routed_scaling_factor), the experts grouped per expert
//     on the device (one weight read per expert for the whole batch), NO shared expert;
//   * the prompt read in chunks of --chunk tokens through the same driver (batched matmuls, ncols <= 8 per launch);
//   * decode one token at a time from a captured graph (positions on the device), or with --no-graph eagerly.
//
// Modes: --serve (serve/server.py's engine protocol: READY / GEN / T / DONE / STOP / PP), --tokens IDS [--n N]
// [--force IDS] [--dump-full-logits F], --bench-pp N / --bench-tg N.
#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/family_layout.hpp"
#include "strata/kernels/fused_gdn.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/native_router.hpp"
#include "strata/platform/memory.hpp"
#include "strata/platform/thread_name.hpp"
#include "strata/prefill/gemm.hpp"
#include "strata/prefill/moe_mmq.hpp"
#include "strata/minimaxm2/engine.hpp"
#include "m2_kernels.cuh"
#if defined(STRATA_Q36_CPU_POOL)
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/pool.hpp"
#endif

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cctype>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <tuple>
#include <type_traits>
#include <vector>

namespace {

namespace K = strata::kernels;
namespace MMQ = strata::prefill::mmq;

#define CK(x)                                                                                                          \
    do {                                                                                                               \
        cudaError_t e_ = (x);                                                                                          \
        if (e_ != cudaSuccess) {                                                                                       \
            std::fprintf(stderr, "strata-m2: %s failed: %s (%s:%d)\n", #x, cudaGetErrorString(e_), __FILE__, __LINE__); \
            std::exit(1);                                                                                              \
        }                                                                                                              \
    } while (0)

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

constexpr int kThreads = 256;
// The attention's split-K chunk, in cells.  It is also the block size of the chunk kernel: each thread
// owns one cell of the chunk and writes its softmax weight into sc[].  With MiniMax-M2's head dim of 128
// the chunk is 128 cells (the Qwen3.6 path's 256 cells with 256 threads each wrote every weight; a
// 256-cell chunk with a 128-thread block would leave half the chunk's weights unwritten).
constexpr int kChunkCells = 128;
constexpr int kHD = 128;           // MiniMax-M2's head dim (load() refuses any other)
constexpr int kMaxKvHeads = 32;    // an INT8 cache's shared scales per cell: --kv int8 is refused above this many KV heads
constexpr int kKvBlock = 32;       // an INT8 scale covers this many dims of a KV head (Q8_0's block: 4 a head at kHD 128)
constexpr int kKvBlocks = kHD / kKvBlock;
static_assert(kKvBlock == 32, "the INT8 path reduces one block per warp");
// The auto prompt chunk (--prefill auto) works with MiniMax-M2's own figures.  MEASURED on this build (RX 6800 + RX 9070,
// a context of 4096, an FP16 cache, the layers 31 on the second card, one session): the free VRAM after init was 12.33
// GiB at a prompt chunk of 512, 11.45 at 2048 and 10.27 at 4096.  A row of the chunk therefore costs 617200 B (0.59 MiB:
// the activations, the router's logits, the expert parts and the CPU pool's copies, the q8 / FP16 operands, the split-K
// partials and the expert scratch) and 0.36 GiB are fixed beside it (the attention scores, the GEMM workspace, the
// dequant scratch).  The rule used to carry q36's numbers over (0.5 MiB per row, kv_one = 45056 per cell) and refused a
// card that had the room.
constexpr double kM2ChunkRowBytes = 617200.0;
constexpr double kM2FixedBufferBytes = 0.36 * double(1ull << 30);
constexpr int kM2ReserveMib = 768;   // the VRAM kept free on a card by default (--vram-reserve-mib overrides it)
constexpr int kConvTile = 16;     // tokens per block of the prompt conv (k_conv_seq tile mode)

// ------------------------------------------------------------------------------------------------ small kernels
__device__ __forceinline__ float silu(float v) { return v / (1.0f + expf(-v)); }
__device__ __forceinline__ float sigm(float v) { return 1.0f / (1.0f + expf(-v)); }
__device__ __forceinline__ float softplus(float v) { return v > 20.f ? v : log1pf(expf(v)); }

__device__ float block_sum(float v, float* sh) {
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int) threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    const float r = sh[0];
    __syncthreads();
    return r;
}
__device__ float block_max(float v, float* sh) {
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int) threadIdx.x < s) sh[threadIdx.x] = fmaxf(sh[threadIdx.x], sh[threadIdx.x + s]);
        __syncthreads();
    }
    const float r = sh[0];
    __syncthreads();
    return r;
}

/// y[r] = rms(x[r]) * w, one block per row of n.
__global__ void k_rms(const float* x, const float* w, float* y, int n, float eps) {
    __shared__ float sh[kThreads];
    const float* xr = x + (size_t) blockIdx.x * n;
    float* yr = y + (size_t) blockIdx.x * n;
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += xr[i] * xr[i];
    s = block_sum(s, sh);
    const float r = rsqrtf(s / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) yr[i] = xr[i] * r * w[i];
}
/// x[r] += y[r]; h[r] = rms(x[r]) * w (one block per row of n): the residual add and the next norm in one launch.
/// out[r * ld_out + i] = rms(x[r * ldx + ..]) * w (one block per row of n): a norm into a slice of a wider row.
__global__ void k_rms_ld(const float* x, int ldx, const float* w, float* out, int ld_out, int n, float eps) {
    __shared__ float sh[kThreads];
    const float* xr = x + (size_t) blockIdx.x * ldx;
    float* yr = out + (size_t) blockIdx.x * ld_out;
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += xr[i] * xr[i];
    s = block_sum(s, sh);
    const float r = rsqrtf(s / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) yr[i] = xr[i] * r * w[i];
}
__global__ void k_add_rms(float* x, const float* y, const float* w, float* h, int n, float eps) {
    __shared__ float sh[kThreads];
    float* xr = x + (size_t) blockIdx.x * n;
    const float* yr = y + (size_t) blockIdx.x * n;
    float* hr = h + (size_t) blockIdx.x * n;
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = xr[i] + yr[i];
        xr[i] = v;
        s += v * v;
    }
    s = block_sum(s, sh);
    const float r = rsqrtf(s / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) hr[i] = xr[i] * r * w[i];
}
/// The MoE block's end for one token row (block per row): x += sum_k w_k part_k (in pick order, FMA after the first
/// product, as native_moe_combine) + sigmoid(sgate) * shared; then, with `wn`, h = rms(x) * wn (the next layer's
/// attention norm).  Replaces combine + gate + add + rms.
__global__ void k_moe_out(float* x, const float* parts, const int32_t* prow, const float* wts, int K_,
                          const float* shared, const float* sgate, int sg_stride, const float* wn, float* h, int n,
                          float eps) {
    __shared__ float sh[kThreads];
    __shared__ const float* pk[16];
    const int t = blockIdx.x;
    float* xr = x + (size_t) t * n;
    if (threadIdx.x < K_)   // pick k's row: t * K + k, or its sorted row (`prow`, the prompt's MMQ order)
        pk[threadIdx.x] = parts + (size_t) (prow ? prow[t * K_ + threadIdx.x] : t * K_ + threadIdx.x) * n;
    __syncthreads();
    const float* wr = wts + (size_t) t * K_;
    const float gs = sigm(sgate[(size_t) t * sg_stride]);
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float acc = wr[0] * pk[0][i];
        for (int k = 1; k < K_; ++k) acc = fmaf(wr[k], pk[k][i], acc);
        acc += gs * shared[(size_t) t * n + i];
        const float v = xr[i] + acc;
        xr[i] = v;
        s += v * v;
    }
    if (wn == nullptr) return;
    s = block_sum(s, sh);
    const float r = rsqrtf(s / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) h[(size_t) t * n + i] = xr[i] * r * wn[i];
}
/// GDN alpha and beta for T <= 8 rows in one launch, their epilogues applied: rows [0, HV) of the output are
/// softplus(alpha . x + dt) * ssm_a, rows [HV, 2 HV) sigmoid(beta . x).  Warp per row (grid (ceil(2 HV / 8), T)).
__global__ void k_gdn_ab(const float* Wa, const float* Wb, const float* x, const float* dt, const float* ssm_a,
                         float* alpha, float* beta, int n_in, int HV) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int o = blockIdx.x * 8 + warp;
    if (o >= 2 * HV) return;
    const bool isb = o >= HV;
    const int r = isb ? o - HV : o;
    const float4* wr = (const float4*) ((isb ? Wb : Wa) + (size_t) r * n_in);
    const float4* xr = (const float4*) (x + (size_t) blockIdx.y * n_in);
    float acc = 0.f;
    for (int i = lane; i < n_in / 4; i += 32) {
        const float4 a = wr[i], b = xr[i];
        acc += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }
    for (int m = 16; m > 0; m >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, m);
    if (lane != 0) return;
    if (isb) beta[(size_t) blockIdx.y * HV + r] = sigm(acc);
    else alpha[(size_t) blockIdx.y * HV + r] = softplus(acc + dt[r]) * ssm_a[r];
}
__global__ void k_add(float* x, const float* y, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}
__global__ void k_swiglu(const float* g, const float* u, float* h, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) h[i] = silu(g[i]) * u[i];
}
/// y[t][o] = W[o] . x[t] for f32 W [n_out][n_in], n_in a multiple of 128: one warp per row, float4 loads, 8 rows per
/// block (grid (ceil(n_out / 8), T)).
__global__ void k_gemv_f32(const float* W, const float* x, float* y, int n_in, int n_out) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int o = blockIdx.x * 8 + warp;
    if (o >= n_out) return;
    const float4* wr = (const float4*) (W + (size_t) o * n_in);
    const float4* xr = (const float4*) (x + (size_t) blockIdx.y * n_in);
    float acc = 0.f;
    for (int i = lane; i < n_in / 4; i += 32) {
        const float4 a = wr[i], b = xr[i];
        acc += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }
    for (int m = 16; m > 0; m >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, m);
    if (lane == 0) y[(size_t) blockIdx.y * n_out + o] = acc;
}
unsigned rows8(int n) { return (unsigned) ((n + 7) / 8); }
/// GDN gate / beta in place for T rows of HV: alpha -> softplus(alpha + dt) * ssm_a, beta -> sigmoid(beta).
__global__ void k_gdn_gb(float* alpha, const float* dt, const float* ssm_a, float* beta, int HV, int T) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= HV * T) return;
    const int h = i % HV;
    alpha[i] = softplus(alpha[i] + dt[h]) * ssm_a[h];
    beta[i] = sigm(beta[i]);
}
/// sh[t] *= sigmoid(gate[t]) (the shared expert's scalar gate), rows of n.
__global__ void k_gate_rows(float* sh, const float* gate, int n, int T) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t) n * T) return;
    sh[i] *= sigm(gate[i / n]);
}

/// The K / V cache's element: FP32 by default, FP16 or INT8 with --kv (half and a quarter of the VRAM for the same
/// context).  The kernels below are templates on it; these helpers are the only places that convert.  INT8's values are
/// what they are: the block's scale (one per cell, per KV head and per kKvBlock dims, `ksc` below) is applied by the
/// caller.
__device__ __forceinline__ float kv_ld(float v) { return v; }
__device__ __forceinline__ float kv_ld(__half v) { return __half2float(v); }
__device__ __forceinline__ float kv_ld(int8_t v) { return (float) v; }
__device__ __forceinline__ void kv_st(float* p, float v) { *p = v; }
__device__ __forceinline__ void kv_st(__half* p, float v) { *p = __float2half(v); }

/// Batch rows (several sessions decoded in one window, one token each): row t reads and writes the cache of session
/// slot[t] at cell[t]; its rotary position is cell[t] + delta[t].  Every field null: one session, the rows are its
/// consecutive cells *pos0 + t (the prompt, a verify window, a decode step).
struct RowSel {
    const int32_t* cell = nullptr;   // device [T]
    const int32_t* slot = nullptr;   // device [T]
    const int32_t* delta = nullptr;  // device [T]
    const void* const* kc = nullptr; // device [n_sessions]: this layer's K / V base of each session (float, __half, int8)
    const void* const* vc = nullptr;
    const __half* const* ks = nullptr;   // device [n_sessions]: their scales (--kv int8), the same indices
    const __half* const* vs = nullptr;
};

/// The attention layer's per-token preparation for T rows at positions *pos0 + t: q (from the [q | gate] pairs) and k
/// RMS-normed per head with their weights, NEOX RoPE over the first n_rot dims, k / v into the cache, the gate kept.
/// Rotary position of a cell: the cell itself (text) plus *rdelta (what the pictures before a text run did not use
/// of their cells, see mrope), or, with `mtab` (a request's prompt cells, device int32 [cell][3] = t, h, w), the
/// interleaved M-RoPE position (llama.cpp's IMROPE with sections 11 / 11 / 10: pair p takes stream p % 3).
/// grid (H + HKV, T), block 256 (HD <= 256).
__global__ void k_attn_prep(const float* qg, const float* kx, const float* vx, const float* qw, const float* kw,
                            float* q, float* gate, float* kc, float* vc, int H, int HKV, int HD, int n_rot,
                            const int32_t* pos0, int ctx, float base, float eps, const int32_t* rdelta,
                            const int32_t* mtab, RowSel rs) {
    __shared__ float sh[kThreads];
    __shared__ float buf[256];
    const int head = blockIdx.x, t = blockIdx.y;
    const int pos = rs.cell ? rs.cell[t] : *pos0 + t;
    const int dl = rs.delta ? rs.delta[t] : (rdelta ? *rdelta : 0);
    if (rs.kc) {
        kc = (float*) rs.kc[rs.slot[t]];
        vc = (float*) rs.vc[rs.slot[t]];
    }
    const bool isq = head < H;
    const float* src = isq ? qg + ((size_t) t * H + head) * 2 * HD : kx + ((size_t) t * HKV + head - H) * HD;
    const int i = threadIdx.x;
    const float v0 = i < HD ? src[i] : 0.f;
    const float s = block_sum(v0 * v0, sh);
    const float r = rsqrtf(s / (float) HD + eps);
    if (i < HD) buf[i] = v0 * r * (isq ? qw[i] : kw[i]);
    __syncthreads();
    if (i >= HD) return;
    float v = buf[i];
    const int half = n_rot / 2;
    if (i < n_rot) {
        const int p = i < half ? i : i - half;
        const int rp = mtab ? __ldg(mtab + (size_t) pos * 3 + p % 3) : pos + dl;
        const float theta = (float) rp * powf(base, -2.0f * (float) p / (float) n_rot);
        float sn, cs;
        sincosf(theta, &sn, &cs);
        v = i < half ? buf[i] * cs - buf[i + half] * sn : buf[i - half] * sn + buf[i] * cs;
    }
    if (isq) {
        q[((size_t) t * H + head) * HD + i] = v;
        gate[((size_t) t * H + head) * HD + i] = qg[((size_t) t * H + head) * 2 * HD + HD + i];
    } else {
        const int kh = head - H;
        kc[((size_t) kh * ctx + pos) * HD + i] = v;
        vc[((size_t) kh * ctx + pos) * HD + i] = vx[((size_t) t * HKV + kh) * HD + i];
    }
}

/// MiniMax-M2's attention preparation, for T rows at positions *pos0 + t: the query and key vectors RMS-normed
/// PER LAYER - one weight vector for all the query heads, one for all the key heads, where the other families
/// norm each head separately - the partial NEOX RoPE over the first n_rot dims, the key and the value into the
/// cache.  MiniMax-M2 has no attention output gate (attention_output_gate false), so the query is a plain [N, HQ]
/// projection and not the [q | gate] pair the other families use.
/// grid (T), block 256.
// The device a pointer lives on (-1: unknown): the layer split's diagnostic.
int dev_of(const void* p) {
#if defined(STRATA_USE_HIP) || defined(STRATA_HIP_GFX906)
    hipPointerAttribute_t at{};
    if (hipPointerGetAttributes(&at, p) != hipSuccess) {
        (void) hipGetLastError();
        return -1;
    }
    return at.device;
#elif defined(__CUDACC__)
    cudaPointerAttributes at{};
    if (cudaPointerGetAttributes(&at, p) != cudaSuccess) {
        (void) cudaGetLastError();
        return -1;
    }
    return at.device;
#else
    (void) p;
    return -1;
#endif
}

template <class KV>
__global__ void k_attn_prep_m2(const float* qg, const float* kx, const float* vx, const float* qw, const float* kw,
                                float* q, KV* kc, KV* vc, __half* ksc, __half* vsc, int H, int HKV, int HD, int n_rot,
                                const int32_t* pos0, int ctx, float base, float eps, const int32_t* rdelta,
                                const int32_t* mtab, RowSel rs) {
    __shared__ float sh[kThreads];
    __shared__ float sks[kMaxKvHeads * kKvBlocks], svs[kMaxKvHeads * kKvBlocks];   // INT8: this cell's scales (32 heads at most)
    extern __shared__ float stage[];   // INT8 only, the host sizes it [2 * HK]: the k and the v row before their scale is known
    const int t = blockIdx.x;
    const int pos = rs.cell ? rs.cell[t] : *pos0 + t;
    const int dl = rs.delta ? rs.delta[t] : (rdelta ? *rdelta : 0);
    if (rs.kc) {
        kc = (KV*) rs.kc[rs.slot[t]];
        vc = (KV*) rs.vc[rs.slot[t]];
        if (rs.ks) {
            ksc = (__half*) rs.ks[rs.slot[t]];
            vsc = (__half*) rs.vs[rs.slot[t]];
        }
    }
    const int HQ = H * HD, HK = HKV * HD;
    float* sk8 = stage;              // --kv int8: the rows while the block reduces them (nothing is read from these
    float* sv8 = stage + HK;         // shared arrays when KV is float or __half: the stores go straight to the cache)
    float sq = 0.f, sk = 0.f;
    for (int i = threadIdx.x; i < HQ; i += kThreads) {
        const float v = qg[(size_t) t * HQ + i];
        sq += v * v;
    }
    for (int i = threadIdx.x; i < HK; i += kThreads) {
        const float v = kx[(size_t) t * HK + i];
        sk += v * v;
    }
    const float rq = rsqrtf(block_sum(sq, sh) / (float) HQ + eps);
    const float rk = rsqrtf(block_sum(sk, sh) / (float) HK + eps);
    const int half = n_rot / 2;
    for (int i = threadIdx.x; i < HQ; i += kThreads) {
        const int h = i / HD, d = i - h * HD;
        float v = qg[(size_t) t * HQ + i] * rq * qw[i];
        if (d < n_rot) {
            const int p = d < half ? d : d - half;
            const int rp = mtab ? __ldg(mtab + (size_t) pos * 3 + p % 3) : pos + dl;
            const float theta = (float) rp * powf(base, -2.0f * (float) p / (float) n_rot);
            float sn, cs;
            sincosf(theta, &sn, &cs);
            const int j = d < half ? i + half : i - half;
            const float other = qg[(size_t) t * HQ + j] * rq * qw[j];
            v = d < half ? v * cs - other * sn : other * sn + v * cs;
        }
        q[(size_t) t * HQ + i] = v;
    }
    for (int i = threadIdx.x; i < HK; i += kThreads) {
        const int h = i / HD, d = i - h * HD;
        float v = kx[(size_t) t * HK + i] * rk * kw[i];
        if (d < n_rot) {
            const int p = d < half ? d : d - half;
            const int rp = mtab ? __ldg(mtab + (size_t) pos * 3 + p % 3) : pos + dl;
            const float theta = (float) rp * powf(base, -2.0f * (float) p / (float) n_rot);
            float sn, cs;
            sincosf(theta, &sn, &cs);
            const int j = d < half ? i + half : i - half;
            const float other = kx[(size_t) t * HK + j] * rk * kw[j];
            v = d < half ? v * cs - other * sn : other * sn + v * cs;
        }
        if constexpr (std::is_same_v<KV, int8_t>) sk8[i] = v;   // --kv int8: staged, its scale is known only after the row
        else kv_st(kc + ((size_t) h * ctx + pos) * HD + d, v);
    }
    for (int i = threadIdx.x; i < HK; i += kThreads) {
        const int h = i / HD, d = i - h * HD;
        const float v = vx[(size_t) t * HK + i];
        if constexpr (std::is_same_v<KV, int8_t>) sv8[i] = v;
        else kv_st(vc + ((size_t) h * ctx + pos) * HD + d, v);
    }
    if constexpr (std::is_same_v<KV, int8_t>) {
        __syncthreads();   // the staged rows are read across the block below
        // One scale per cell, per KV head and per block of kKvBlock dims: the block's largest magnitude / 127 (the
        // Q8_0 rule; 4 blocks a head here).  The score and the value sums carry each block's scale as one more factor,
        // so the values are rounded but every sum stays in FP32.  A warp holds one block of 32, so its reduction is a
        // shuffle: `kKvBlock` is the warp's width.
        const int lane = threadIdx.x & 31, wrp = threadIdx.x / 32;
        for (int r = wrp; r < HKV * kKvBlocks; r += kThreads / 32) {
            const int h = r / kKvBlocks, b = r - h * kKvBlocks;
            float mk = 0.f, mv = 0.f;
            for (int i = b * kKvBlock + lane; i < (b + 1) * kKvBlock; i += 32) {
                mk = fmaxf(mk, fabsf(sk8[h * HD + i]));
                mv = fmaxf(mv, fabsf(sv8[h * HD + i]));
            }
            for (int m = 16; m > 0; m >>= 1) {
                mk = fmaxf(mk, __shfl_xor_sync(0xffffffffu, mk, m));
                mv = fmaxf(mv, __shfl_xor_sync(0xffffffffu, mv, m));
            }
            if (lane == 0) {
                // the scale as a half, as Q8_0 keeps it (2 bytes for 32 values): the values are quantized with the
                // half's own value, so what is stored and what is read back agree
                sks[r] = __half2float(__float2half(mk > 0.f ? mk / 127.f : 1.f));
                svs[r] = __half2float(__float2half(mv > 0.f ? mv / 127.f : 1.f));
                ksc[((size_t) h * ctx + pos) * kKvBlocks + b] = __float2half(sks[r]);
                vsc[((size_t) h * ctx + pos) * kKvBlocks + b] = __float2half(svs[r]);
            }
        }
        __syncthreads();
        for (int i = threadIdx.x; i < HK; i += kThreads) {
            const int h = i / HD, d = i - h * HD, r = h * kKvBlocks + d / kKvBlock;
            const int qk = __float2int_rn(sk8[i] / sks[r]);
            const int qv = __float2int_rn(sv8[i] / svs[r]);
            kc[((size_t) h * ctx + pos) * HD + d] = (KV) (qk > 127 ? 127 : (qk < -127 ? -127 : qk));
            vc[((size_t) h * ctx + pos) * HD + d] = (KV) (qv > 127 ? 127 : (qv < -127 ? -127 : qv));
        }
    }
}

/// Split-K attention, one block per (chunk of 256 cells, query row): query row qi = t * H + h (t from t0) sees the
/// cells [0, *pos0 + t].  Writes the chunk's partial accumulator, max and sum; chunks past the query's cells write
/// nothing (the merge stops before them).  grid (n_chunk, n_rows), block 256, HD == 256.
template <int HD, class KV>
__global__ void k_attn_part(const float* q, const KV* kc, const KV* vc, const __half* ksc, const __half* vsc,
                            const int32_t* pos0, int t0, int H, int HKV, int ctx, float scale, int n_chunk,
                            float* pacc, float* pm, float* pl, RowSel rs) {
    __shared__ float qs[HD], sc[kChunkCells], sh[kThreads];
    const int c = blockIdx.x, row = blockIdx.y;
    const int t = t0 + row / H, h = row % H, kvh = h / (H / HKV);
    const int npos = (rs.cell ? rs.cell[t] : *pos0 + t) + 1;
    if (rs.kc) {
        kc = (const KV*) rs.kc[rs.slot[t]];
        vc = (const KV*) rs.vc[rs.slot[t]];
        ksc = rs.ks ? rs.ks[rs.slot[t]] : nullptr;
        vsc = rs.vs ? rs.vs[rs.slot[t]] : nullptr;
    }
    const bool i8 = ksc != nullptr;   // --kv int8: each block's scale is one more factor of the score and of the sum
    const int p0 = c * kChunkCells;
    if (p0 >= npos) return;
    const int j = threadIdx.x;
    qs[j] = q[((size_t) t * H + h) * HD + j];
    __syncthreads();
    const int p = p0 + j;
    float s = -INFINITY;
    if (p < npos) {
        const KV* kp = kc + ((size_t) kvh * ctx + p) * HD;
        float a = 0.f;
        if (i8) {   // the dot product per block, its own scale applied to its own sum
            const __half* ks = ksc + ((size_t) kvh * ctx + p) * kKvBlocks;
            for (int b = 0; b < kKvBlocks; ++b) {
                const int i0 = b * kKvBlock;
                float part = 0.f;
                for (int i = i0; i < i0 + kKvBlock; ++i) part += qs[i] * kv_ld(kp[i]);
                a += part * __half2float(ks[b]);
            }
        } else {
            for (int i = 0; i < HD; ++i) a += qs[i] * kv_ld(kp[i]);
        }
        s = a * scale;
    }
    const float m = block_max(s, sh);
    const float e = p < npos ? expf(s - m) : 0.f;
    sc[j] = e;
    const float l = block_sum(e, sh);
    const int n = min(kChunkCells, npos - p0);
    float acc = 0.f;
    const KV* vp = vc + ((size_t) kvh * ctx + p0) * HD + j;
    const __half* vs = vsc ? vsc + ((size_t) kvh * ctx + p0) * kKvBlocks : nullptr;
    const int vb = j / kKvBlock;   // this dim's block
    for (int k = 0; k < n; ++k)
        acc += sc[k] * (i8 ? __half2float(vs[k * kKvBlocks + vb]) : 1.f) * kv_ld(vp[(size_t) k * HD]);
    const size_t o = (size_t) row * n_chunk + c;
    pacc[o * HD + j] = acc;
    if (j == 0) {
        pm[o] = m;
        pl[o] = l;
    }
}
/// The chunks merged: grid (n_rows), block HD.  MiniMax-M2 has no attention output gate, so nothing multiplies
/// the merged row (the other families' merge applies sigmoid(gate)).
template <int HD>
__global__ void k_attn_merge(const float* pacc, const float* pm, const float* pl,
                             const int32_t* pos0, int t0, int H, int n_chunk, float* out, RowSel rs) {
    const int row = blockIdx.x, t = t0 + row / H, h = row % H, d = threadIdx.x;
    const int npos = (rs.cell ? rs.cell[t] : *pos0 + t) + 1;
    const int nc = (npos + kChunkCells - 1) / kChunkCells;
    float M = -INFINITY;
    for (int c = 0; c < nc; ++c) M = fmaxf(M, pm[(size_t) row * n_chunk + c]);
    float num = 0.f, den = 0.f;
    for (int c = 0; c < nc; ++c) {
        const size_t o = (size_t) row * n_chunk + c;
        const float w = expf(pm[o] - M);
        num += w * pacc[o * HD + d];
        den += w * pl[o];
    }
    const size_t gi = ((size_t) t * H + h) * HD + d;
    out[gi] = num / den;
}

/// The routed entries (T tokens x K picks, entry e = t * K + k) grouped per expert, on the device: one group per
/// expert used (ascending id), its blob address, its entries in entry order.  One block.
__global__ void k_group(const int32_t* ids, int n_ent, int K_, int n_expert, const uint8_t* blobs, unsigned long long blob,
                        unsigned long long* grp_ptr, int32_t* grp_start, int32_t* n_groups, int32_t* ent_dst,
                        int32_t* ent_tok) {
    __shared__ int cnt[512], off[512];
    for (int x = threadIdx.x; x < n_expert; x += blockDim.x) cnt[x] = 0;
    __syncthreads();
    for (int e = threadIdx.x; e < n_ent; e += blockDim.x) atomicAdd(&cnt[ids[e]], 1);
    __syncthreads();
    if (threadIdx.x == 0) {
        int g = 0, pos = 0;
        for (int x = 0; x < n_expert; ++x) {
            off[x] = pos;
            if (cnt[x] == 0) continue;
            grp_ptr[g] = (unsigned long long) (blobs + (size_t) x * blob);
            grp_start[g] = pos;
            pos += cnt[x];
            ++g;
        }
        grp_start[g] = pos;
        *n_groups = g;
        for (int e = 0; e < n_ent; ++e) {
            const int p = off[ids[e]]++;
            ent_dst[p] = e;
            ent_tok[p] = e / K_;
        }
    }
}
/// k_group for the hybrid mode: only the entries whose expert has a VRAM slot (slot_of[x] >= 0) are grouped, each
/// group's blob at blobs + slot * blob; the others are the CPU pool's (their rows come from k_merge_cpu).  One block.
__global__ void k_group_hy(const int32_t* ids, int n_ent, int K_, int n_expert, const int32_t* slot_of,
                           const uint8_t* blobs, unsigned long long blob, unsigned long long* grp_ptr, int32_t* grp_start,
                           int32_t* n_groups, int32_t* ent_dst, int32_t* ent_tok) {
    __shared__ int cnt[512], off[512];
    for (int x = threadIdx.x; x < n_expert; x += blockDim.x) cnt[x] = 0;
    __syncthreads();
    for (int e = threadIdx.x; e < n_ent; e += blockDim.x)
        if (slot_of[ids[e]] >= 0) atomicAdd(&cnt[ids[e]], 1);
    __syncthreads();
    if (threadIdx.x == 0) {
        int g = 0, pos = 0;
        for (int x = 0; x < n_expert; ++x) {
            off[x] = pos;
            if (cnt[x] == 0) continue;
            grp_ptr[g] = (unsigned long long) (blobs + (size_t) slot_of[x] * blob);
            grp_start[g] = pos;
            pos += cnt[x];
            ++g;
        }
        grp_start[g] = pos;
        *n_groups = g;
        for (int e = 0; e < n_ent; ++e) {
            if (slot_of[ids[e]] < 0) continue;
            const int p = off[ids[e]]++;
            ent_dst[p] = e;
            ent_tok[p] = e / K_;
        }
    }
}
/// The routing grouped per expert when the experts' blobs sit in TWO places: the ones with a VRAM slot in `blobs`
/// (at slot * blob) and the others in `pool` (at x * blob), the pinned host buffer the GPU reads over PCIe.  The
/// prompt path uses it: the slots keep their VRAM (the decode's speed) and only the pool's experts are streamed.
__global__ void k_group_pool(const int32_t* ids, int n_ent, int K_, int n_expert, const int32_t* slot_of,
                             const uint8_t* blobs, unsigned long long blob, const uint8_t* pool,
                             unsigned long long* grp_ptr, int32_t* grp_start, int32_t* n_groups, int32_t* ent_dst,
                             int32_t* ent_tok) {
    __shared__ int cnt[512], off[512];
    for (int x = threadIdx.x; x < n_expert; x += blockDim.x) cnt[x] = 0;
    __syncthreads();
    for (int e = threadIdx.x; e < n_ent; e += blockDim.x) atomicAdd(&cnt[ids[e]], 1);
    __syncthreads();
    if (threadIdx.x == 0) {
        int g = 0, pos = 0;
        for (int x = 0; x < n_expert; ++x) {
            off[x] = pos;
            if (cnt[x] == 0) continue;
            const int s = slot_of[x];
            grp_ptr[g] = (unsigned long long) (s >= 0 ? blobs + (size_t) s * blob : pool + (size_t) x * blob);
            grp_start[g] = pos;
            pos += cnt[x];
            ++g;
        }
        grp_start[g] = pos;
        *n_groups = g;
        for (int e = 0; e < n_ent; ++e) {
            const int p = off[ids[e]]++;
            ent_dst[p] = e;
            ent_tok[p] = e / K_;
        }
    }
}
/// The peer's rows into the host rows: the entries it computed.  `peer_row[e]` is the entry's row in the peer's
/// OWN output (-1: the peer did not compute it).  The rows go where the CPU pool's rows go, so the merge that
/// follows takes them like the CPU's.  grid (n_ent), block 256.
/// The inverse of the grouping's table: the entry whose row is p.  grid (n_rows), block 256.
/// The CPU pool's expert rows into the entries' rows: parts[e] = cpu[e] where miss[e].  grid (n_ent), block 256.
__global__ void k_merge_cpu(float* parts, const float* cpu, const int32_t* miss, int n) {
    const int e = blockIdx.x;
    if (!miss[e]) return;
    for (int i = threadIdx.x; i < n; i += blockDim.x) parts[(size_t) e * n + i] = cpu[(size_t) e * n + i];
}
/// The prompt attention's FP16 operands: cells [0, P) of each KV head's K / V (cache rows ctx apart) into compact
/// [kvh][P][HD] halves.  grid1(HKV * P * HD).
template <class KV>
__global__ void k_kv_f16(const KV* kc, const KV* vc, const __half* ksc, const __half* vsc, __half* k16, __half* v16,
                         int ctx, int P, int HD, int HKV) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t) HKV * P * HD) return;
    const int64_t per = (int64_t) P * HD;
    const int64_t kvh = i / per, rest = i % per;
    const size_t src = (size_t) kvh * ctx * HD + rest;
    if (ksc) {   // --kv int8: the FP16 GEMM operands are the cache's rows back in their own block's scale (they are rounded)
        const size_t sc_ = ((size_t) kvh * ctx + (size_t) (rest / HD)) * kKvBlocks + (size_t) (rest % HD) / kKvBlock;
        k16[i] = __float2half((float) kc[src] * __half2float(ksc[sc_]));
        v16[i] = __float2half((float) vc[src] * __half2float(vsc[sc_]));
        return;
    }
    k16[i] = __float2half(kv_ld(kc[src]));
    v16[i] = __float2half(kv_ld(vc[src]));
}
/// The prompt's causal softmax: row (h, t) of S (ld floats, P cells) -> softmax(S * scale) over the cells
/// [0, pos0 + t], 0 past them, as FP16 probabilities (S16, the same row offsets, ld halves).  grid (T, H), block 256.
__global__ void k_attn_softmax16(const float* S, __half* S16, int ld, int T, int P, int pos0, float scale) {
    __shared__ float sh[kThreads];
    const int t = blockIdx.x, h = blockIdx.y;
    const float* row = S + ((size_t) h * T + t) * ld;
    __half* out = S16 + ((size_t) h * T + t) * ld;
    const int n = min(P, pos0 + t + 1);
    float mx = -INFINITY;
    for (int p = threadIdx.x; p < n; p += blockDim.x) mx = fmaxf(mx, row[p] * scale);
    mx = block_max(mx, sh);
    float sum = 0.f;
    for (int p = threadIdx.x; p < n; p += blockDim.x) sum += expf(row[p] * scale - mx);
    sum = block_sum(sum, sh);
    const float inv = 1.0f / sum;
    for (int p = threadIdx.x; p < P; p += blockDim.x) out[p] = __float2half(p < n ? expf(row[p] * scale - mx) * inv : 0.f);
}
/// att *= sigmoid(gate) (the attention output gate), n values.
__global__ void k_gate_mul(float* att, const float* gate, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) att[i] *= sigm(gate[i]);
}

/// Picture rows over the embeddings of a chunk: x[r] = img[map[r]] for the rows with map[r] >= 0 (the rows of a picture's
/// <|image_pad|> cells).  grid (T), block 256.
__global__ void k_put_rows(float* x, const float* img, const int32_t* map, int n) {
    const int r = blockIdx.x;
    const int m = map[r];
    if (m < 0) return;
    for (int i = threadIdx.x; i < n; i += blockDim.x) x[(size_t) r * n + i] = img[(size_t) m * n + i];
}

/// The prompt's routed entries sorted by expert for MMQ: bounds[x] .. bounds[x + 1] are expert x's rows; sorted row r
/// holds entry dst[r] (= t * K + k, the row of the per-slot outputs) of token src[r].  The order inside an expert is the
/// atomics' (no output depends on it: every row's products are its own).  cnt / cur: n_expert ints of scratch.
__global__ void k_route_count(const int32_t* ids, int n_ent, int32_t* cnt) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e < n_ent) atomicAdd(&cnt[ids[e]], 1);
}
__global__ void k_route_scan(const int32_t* cnt, int n_expert, int32_t* bounds, int32_t* cur, int32_t* max_rows) {
    if (threadIdx.x != 0) return;
    int pos = 0, mx = 0;
    for (int x = 0; x < n_expert; ++x) {
        bounds[x] = pos;
        cur[x] = pos;
        pos += cnt[x];
        mx = max(mx, cnt[x]);
    }
    bounds[n_expert] = pos;
    *max_rows = mx;
}
__global__ void k_route_scatter(const int32_t* ids, int n_ent, int K_, int32_t* cur, int32_t* dst, int32_t* src,
                                int32_t* inv) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_ent) return;
    const int r = atomicAdd(&cur[ids[e]], 1);
    dst[r] = e;
    src[r] = e / K_;
    inv[e] = r;   // entry e's sorted row: where its down product lands
}

/// The greedy pick of one logits row: out = argmax (lowest id on ties).  Stage 1: `gridDim.x` blocks each write the
/// best of their stride into (pv, pi); stage 2 (one block, n = the stage-1 block count) reads those.
__global__ void k_argmax(const float* x, int n, int32_t* out, float* pv, int32_t* pi, bool stage2) {
    __shared__ float bv[kThreads];
    __shared__ int bi[kThreads];
    // row blockIdx.y: its logits (stage 1, n = the vocabulary) and its kArgBlocks partials
    x += (size_t) blockIdx.y * (stage2 ? 0 : n);
    pv += (size_t) blockIdx.y * 128;
    pi += (size_t) blockIdx.y * 128;
    out += blockIdx.y;
    float v = -INFINITY;
    int id = 0x7fffffff;
    if (stage2) {
        for (int i = threadIdx.x; i < n; i += blockDim.x)
            if (pv[i] > v || (pv[i] == v && pi[i] < id)) { v = pv[i]; id = pi[i]; }
    } else {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
            if (x[i] > v) { v = x[i]; id = i; }
    }
    bv[threadIdx.x] = v;
    bi[threadIdx.x] = id;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int) threadIdx.x < s) {
            const float ov = bv[threadIdx.x + s];
            const int oi = bi[threadIdx.x + s];
            if (ov > bv[threadIdx.x] || (ov == bv[threadIdx.x] && oi < bi[threadIdx.x])) {
                bv[threadIdx.x] = ov;
                bi[threadIdx.x] = oi;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        if (stage2) *out = bi[0];
        else { pv[blockIdx.x] = bv[0]; pi[blockIdx.x] = bi[0]; }
    }
}
constexpr int kArgBlocks = 128;   // (the row stride of k_argmax's partials)

unsigned grid1(int64_t n, int b = kThreads) { return (unsigned) ((n + b - 1) / b); }

/// out[t][o] = W[o] . x[t] in FP32 for a prompt chunk's small F32 projections (the router, the GDN alpha / beta,
/// the shared expert's gate: n_out <= 256): a (16 TM) x 64 output tile per block, K in steps of 16 through shared
/// memory, TM x 4 outputs per thread (TM = 4 for the router, 1 for the 32-wide alpha / beta: more blocks).  hipBLAS
/// picked 8 x 8 / 16 x 16 tiles for these shapes (~10% of a prompt).  K % 4 == 0.
template <int TM>
__global__ void __launch_bounds__(256) k_sgemm_nt(const float* __restrict__ W, const float* __restrict__ x,
                                                  float* __restrict__ out, int K, int n_out, int T, int ldo, int kz) {
    constexpr int BM = 16 * TM, BN = 64, BK = 16;
    const int kb = blockIdx.z * kz, ke = min(K, kb + kz);   // split-K: this block's K range, its own partial
    out += (size_t) blockIdx.z * T * ldo;
    __shared__ float As[BK][BM + 4], Bs[BK][BN + 4];
    const int tid = threadIdx.x;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
    const int tm = (tid / 16) * TM, tn = (tid % 16) * 4;
    const int lr = tid / 4, lk = (tid % 4) * 4;   // this thread's loads: row lr of each tile, k lk .. lk + 3
    float acc[TM][4] = {};
    for (int k0 = kb; k0 < ke; k0 += BK) {
        if (lr < BM) {
            float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
            if (m0 + lr < T && k0 + lk < ke) a = *reinterpret_cast<const float4*>(x + (size_t) (m0 + lr) * K + k0 + lk);
            As[lk][lr] = a.x; As[lk + 1][lr] = a.y; As[lk + 2][lr] = a.z; As[lk + 3][lr] = a.w;
        }
        float4 b = make_float4(0.f, 0.f, 0.f, 0.f);
        if (n0 + lr < n_out && k0 + lk < ke) b = *reinterpret_cast<const float4*>(W + (size_t) (n0 + lr) * K + k0 + lk);
        Bs[lk][lr] = b.x; Bs[lk + 1][lr] = b.y; Bs[lk + 2][lr] = b.z; Bs[lk + 3][lr] = b.w;
        __syncthreads();
#pragma unroll
        for (int k = 0; k < BK; ++k) {
            float ar[TM];
#pragma unroll
            for (int i = 0; i < TM; ++i) ar[i] = As[k][tm + i];
            const float4 bv = *reinterpret_cast<const float4*>(&Bs[k][tn]);
            const float br[4] = {bv.x, bv.y, bv.z, bv.w};
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] = fmaf(ar[i], br[j], acc[i][j]);
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
            if (m0 + tm + i < T && n0 + tn + j < n_out) out[(size_t) (m0 + tm + i) * ldo + n0 + tn + j] = acc[i][j];
}

/// out[i] = the sum of the split-K partials part[z][i] in order (deterministic), n values each.
__global__ void k_sum_splits(const float* part, float* out, int64_t n, int splits) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float a = 0.f;
    for (int z = 0; z < splits; ++z) a += part[(size_t) z * n + i];
    out[i] = a;
}

/// x f32 -> FP16 bits (the prompt GEMMs' activations).
__global__ void k_to_f16(const float* x, __half* y, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2half(x[i]);
}

/// The GDN conv for T tokens in ONE launch (the prompt): one block per 128-channel head, thread = channel, the
/// 3-entry history carried in registers across the tokens and written back at the end; SiLU, then the L2 norm of the
/// q/k heads (block < qk_heads) per token - fused_gdn_conv_l2's arithmetic, token after token.  History [channel][3].
__global__ void __launch_bounds__(128) k_conv_seq(float* hist, const float* qkv, const float* w, float* h, int C,
                                                  int qk_heads, float eps, int T, bool write,
                                                  float* const* hist_tab = nullptr, const int32_t* slot_rows = nullptr,
                                                  const int32_t* grow0 = nullptr, const int32_t* gcnt = nullptr) {
    __shared__ float part[4];
    if (hist_tab) {   // batch rows, grid y = group: group g's history is session slot_rows[g]'s; its rows are
                      // grow0[g] .. + gcnt[g] - 1 (one row at row g when there are no group arrays)
        const int g = blockIdx.y, r0 = grow0 ? grow0[g] : g;
        hist = hist_tab[slot_rows[g]];
        if (gcnt) T = gcnt[g];
        qkv += (size_t) r0 * C;
        h += (size_t) r0 * C;
    }
    const int c = blockIdx.x * 128 + threadIdx.x;
    float v0 = hist[c * 3], v1 = hist[c * 3 + 1], v2 = hist[c * 3 + 2];
    // TILE mode (a prompt chunk: grid y > 1, no batch tables): the tiles of kConvTile tokens run side by side.  A tile's history
    // is the three inputs before its first token (exactly the values the sequential loop carries); tile 0 reads the old history,
    // and, as the only writer, stores the new one: the last three of (old history, inputs).  The arithmetic per token is the same.
    int t_first = 0, t_end = T;
    const bool tiled = hist_tab == nullptr && gridDim.y > 1;
    if (tiled) {
        t_first = (int) blockIdx.y * kConvTile;
        t_end = min(T, t_first + kConvTile);
        if (t_first == 0 && write) {
            const float o0 = v0, o1 = v1, o2 = v2;
            const float old3[3] = {o0, o1, o2};
            for (int k = 0; k < 3; ++k) {
                const int idx = T - 3 + k;
                hist[c * 3 + k] = idx >= 0 ? qkv[(size_t) idx * C + c] : old3[idx + 3];
            }
        }
        if (t_first > 0) {
            v0 = qkv[(size_t) (t_first - 3) * C + c];
            v1 = qkv[(size_t) (t_first - 2) * C + c];
            v2 = qkv[(size_t) (t_first - 1) * C + c];
        }
        if (t_first >= T) return;
    }
    const float w0 = w[c * 4], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3];
    const bool qk = (int) blockIdx.x < qk_heads;
    for (int t = t_first; t < t_end; ++t) {
        const float x = qkv[(size_t) t * C + c];
        const float sum = v0 * w0 + v1 * w1 + v2 * w2 + x * w3;
        v0 = v1;
        v1 = v2;
        v2 = x;
        float y = sum / (1.0f + __expf(-sum));
        if (qk) {
            float sq = y * y;
            for (int o = 16; o > 0; o >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, o);
            if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = sq;
            __syncthreads();
            const float ss = part[0] + part[1] + part[2] + part[3];
            __syncthreads();
            y *= rsqrtf(ss + eps);
        }
        h[(size_t) t * C + c] = y;
    }
    if (!write || tiled) return;   // a verify window: the history is commit()'s to move; tiles: tile 0 stored it
    hist[c * 3] = v0;
    hist[c * 3 + 1] = v1;
    hist[c * 3 + 2] = v2;
}

/// The gated delta rule + SiLU-gated norm for T tokens in ONE launch (the prompt): fused_gdn_step_norm's kernel
/// (block = one V head, 128 columns x 4 row groups, state [row][head][col]) with its 32 state rows per thread kept in
/// registers across the tokens.  h (T, C) = q | k | v, gate / beta (T, HV), z / y (T, ZV).
__global__ void __launch_bounds__(512) k_gdn_seq(float* state, const float* h, int C, const float* gate,
                                                 const float* beta, const float* z, const float* gamma, float eps,
                                                 float* y, int h_k, int h_v, int T, bool write,
                                                 float* const* state_tab = nullptr, const int32_t* slot_rows = nullptr,
                                                 const int32_t* grow0 = nullptr, const int32_t* gcnt = nullptr) {
    constexpr int S = 128, RG = 4, RPG = 32;
    if (state_tab) {   // batch rows, grid y = group (see k_conv_seq)
        const int g = blockIdx.y, r = grow0 ? grow0[g] : g;
        state = state_tab[slot_rows[g]];
        if (gcnt) T = gcnt[g];
        h += (size_t) r * C;
        gate += (size_t) r * h_v;
        beta += (size_t) r * h_v;
        z += (size_t) r * h_v * S;
        if (y) y += (size_t) r * h_v * S;
    }
    __shared__ float sk[S], sq[S];
    __shared__ float red_kv[RG][S];
    __shared__ float red_o[RG][S];
    __shared__ float wsum[S * RG / 32];
    const int head = blockIdx.x, col = threadIdx.x, rg = threadIdx.y, tid = rg * S + col;
    const int qh = head % h_k;
    const int ZV = h_v * S;
    float s[RPG];
    float* base = state + ((size_t) (rg * RPG) * h_v + head) * S + col;
    const size_t row_stride = (size_t) h_v * S;
#pragma unroll
    for (int r = 0; r < RPG; ++r) s[r] = base[r * row_stride];
    for (int t = 0; t < T; ++t) {
        const float* ht = h + (size_t) t * C;
        if (tid < S) { sk[tid] = ht[h_k * S + qh * S + tid]; sq[tid] = ht[qh * S + tid]; }
        __syncthreads();
        const float g = __expf(gate[(size_t) t * h_v + head]);
        float kv = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) kv = fmaf(s[r], sk[rg * RPG + r], kv);
        red_kv[rg][col] = kv;
        __syncthreads();
        const float kv_col = red_kv[0][col] + red_kv[1][col] + red_kv[2][col] + red_kv[3][col];
        const float delta = (ht[2 * h_k * S + head * S + col] - g * kv_col) * beta[(size_t) t * h_v + head];
        float o = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) {
            s[r] = fmaf(g, s[r], sk[rg * RPG + r] * delta);
            o = fmaf(s[r], sq[rg * RPG + r], o);
        }
        red_o[rg][col] = o;
        __syncthreads();
        float oc = 0.0f, sq_part = 0.0f;
        if (rg == 0) {
            oc = (red_o[0][col] + red_o[1][col] + red_o[2][col] + red_o[3][col]) * rsqrtf((float) S);
            sq_part = oc * oc;
        }
        for (int o2 = 16; o2 > 0; o2 >>= 1) sq_part += __shfl_xor_sync(0xffffffffu, sq_part, o2);
        if ((tid & 31) == 0) wsum[tid >> 5] = sq_part;
        __syncthreads();
        if (rg == 0 && y != nullptr) {   // (the commit runs without outputs)
            const float ss = wsum[0] + wsum[1] + wsum[2] + wsum[3];
            const float scale = rsqrtf(ss / (float) S + eps);
            const float zz = z[(size_t) t * ZV + head * S + col];
            y[(size_t) t * ZV + head * S + col] = oc * scale * gamma[col] * (zz / (1.0f + __expf(-zz)));
        }
        __syncthreads();   // sk / sq / the reductions are rewritten by the next token
    }
    if (!write) return;   // a verify window: the state is commit()'s to write
#pragma unroll
    for (int r = 0; r < RPG; ++r) base[r * row_stride] = s[r];
}

/// The prompt's gated delta rule with more parallelism: a state column's update needs only that column (kv, delta and
/// the output o all reduce over the rows), so a V head splits into kGdnColGroups blocks of 128 / kGdnColGroups columns,
/// each with 16 row groups of 8 rows (512 threads, 8 state values per thread in registers).  Writes the raw outputs
/// o (T, HV * 128); k_gdn_norm applies the SiLU-gated RMS norm.  grid (HV, kGdnColGroups), block (32, 16).
constexpr int kGdnColGroups = 4;
__global__ void __launch_bounds__(512) k_gdn_seq_cols(float* state, const float* h, int C, const float* gate,
                                                      const float* beta, float* o_raw, int h_k, int h_v, int T) {
    constexpr int S = 128, NC = S / kGdnColGroups, RG = 16, RPG = S / RG;
    // two barriers per token (was three): sk / sq are double-buffered, and the output sum of token t is done after token t + 1's first
    // barrier, off the chain of the recurrence (red_o is read between that barrier and the next one, written after it)
    __shared__ float sk[2][S], sq[2][S];
    __shared__ float red_kv[RG][NC];
    __shared__ float red_o[RG][NC];
    const int head = blockIdx.x, cl = threadIdx.x, rg = threadIdx.y, tid = rg * NC + cl;
    const int col = blockIdx.y * NC + cl;
    const int qh = head % h_k;
    float sv[RPG];
    float* base = state + ((size_t) (rg * RPG) * h_v + head) * S + col;
    const size_t row_stride = (size_t) h_v * S;
#pragma unroll
    for (int r = 0; r < RPG; ++r) sv[r] = base[r * row_stride];
    // the next token's inputs are loaded while this one is computed (the loads were the head of every iteration)
    float nsk = 0.0f, nsq = 0.0f, nvv = 0.0f, ngate = 0.0f, nbeta = 0.0f;
    auto load = [&](int t) {
        const float* ht = h + (size_t) t * C;
        if (tid < S) { nsk = ht[h_k * S + qh * S + tid]; nsq = ht[qh * S + tid]; }
        nvv = ht[2 * h_k * S + head * S + col];
        ngate = gate[(size_t) t * h_v + head];
        nbeta = beta[(size_t) t * h_v + head];
    };
    if (T > 0) load(0);
    for (int t = 0; t < T; ++t) {
        const int par = t & 1;
        if (tid < S) { sk[par][tid] = nsk; sq[par][tid] = nsq; }
        const float vcur = nvv, gcur = ngate, bcur = nbeta;
        if (t + 1 < T) load(t + 1);
        __syncthreads();
        if (t > 0 && rg == 0) {   // token t - 1's output
            float oc = 0.0f;
#pragma unroll
            for (int i = 0; i < RG; ++i) oc += red_o[i][cl];
            o_raw[(size_t) (t - 1) * h_v * S + head * S + col] = oc * rsqrtf((float) S);
        }
        const float g = __expf(gcur);
        float kv = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) kv = fmaf(sv[r], sk[par][rg * RPG + r], kv);
        red_kv[rg][cl] = kv;
        __syncthreads();
        float kv_col = 0.0f;
#pragma unroll
        for (int i = 0; i < RG; ++i) kv_col += red_kv[i][cl];
        const float delta = (vcur - g * kv_col) * bcur;
        float o = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) {
            sv[r] = fmaf(g, sv[r], sk[par][rg * RPG + r] * delta);
            o = fmaf(sv[r], sq[par][rg * RPG + r], o);
        }
        red_o[rg][cl] = o;
    }
    __syncthreads();
    if (T > 0 && rg == 0) {   // the last token's output
        float oc = 0.0f;
#pragma unroll
        for (int i = 0; i < RG; ++i) oc += red_o[i][cl];
        o_raw[(size_t) (T - 1) * h_v * S + head * S + col] = oc * rsqrtf((float) S);
    }
#pragma unroll
    for (int r = 0; r < RPG; ++r) base[r * row_stride] = sv[r];
}
/// y = rms(o) * gamma * SiLU(z) per (token, V head): grid (HV, T), block 128.
__global__ void __launch_bounds__(128) k_gdn_norm(const float* o_raw, const float* z, const float* gamma, float eps,
                                                  float* y, int h_v) {
    __shared__ float part[4];
    const size_t i = ((size_t) blockIdx.y * h_v + blockIdx.x) * 128 + threadIdx.x;
    const float oc = o_raw[i];
    float sq = oc * oc;
    for (int o = 16; o > 0; o >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, o);
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = sq;
    __syncthreads();
    const float ss = part[0] + part[1] + part[2] + part[3];
    const float scale = rsqrtf(ss / 128.0f + eps);
    const float zz = z[i];
    y[i] = oc * scale * gamma[threadIdx.x] * (zz / (1.0f + __expf(-zz)));
}

// ------------------------------------------------------------------------------------------------ the model
// The layer split: the second card's device and the first layer it runs.  The cards do not share their
// memory (no P2P between them), so the second card gets its own copy of the weights it reads.
int g_second_card = -1;
// The end-of-turn id the serve loop stops at (is_eos): the GGUF's tokenizer.ggml.eos_token_id, -1 when absent.
int32_t g_eos_id = -1;
int g_split_first = 0;

struct Mat {
    const uint8_t* d = nullptr;   // device, GGUF blocks
    int type = -1;
    int n_in = 0, n_out = 0;
};

struct Layer {
    bool attn = false;
    const float *attn_norm = nullptr, *post_norm = nullptr;
    // GDN
    Mat qkv, zg, ssm_out;
    const float *alpha = nullptr, *beta = nullptr, *dt = nullptr, *ssm_a = nullptr, *conv = nullptr, *ssm_norm = nullptr;
    // attention
    Mat wq, wk, wv, wo;
    const float *q_norm = nullptr, *k_norm = nullptr;
    // MoE
    const float* router = nullptr;   // [n_expert][N] f32
    float* router_ext = nullptr;     // the same with the shared expert's gate row appended: [n_expert + 1][N]
    const float* router_bias = nullptr;   // [n_expert] f32: the per-expert selection bias (MiniMax-M2)
    Mat sh_gate, sh_up, sh_down;
    const float* sh_gate_inp = nullptr;   // [N]
    K::NativeExpertLayout L;
    uint8_t* blobs = nullptr;   // n_expert blobs, `stride` bytes apart (L.bytes rounded up to whole blocks of every
    size_t stride = 0;          // part's type, so MMQ can step over them by blocks), then kMmqTail zero bytes
                                // (hybrid mode: n_slots slots, see place_experts)
    // the GGUF's expert tensors (the mapped file stays open): place_experts builds the blobs from them
    const uint8_t *src_g = nullptr, *src_u = nullptr, *src_d = nullptr;
    size_t src_gsz = 0, src_dsz = 0;
    uint8_t* hblobs = nullptr;  // hybrid mode: the host copy of the experts [h_first, h_first + h_resident), `stride` apart
    int h_first = 0;            // the first of them: the experts WITHOUT a VRAM slot (the pooled ones), not the slots'
    int h_resident = 0;         // how many of them are in it (--resident-budget-gib; the others are read in place)
    /// Is this expert of the host copy?  (The pinned arena holds the pooled experts: the ones the GPU does not
    /// hold, so a slot's fill never reads it and the prompt's stream takes the pool from RAM, not from the file.)
    bool host_has(int64_t e) const { return hblobs && e >= h_first && e < (int64_t) h_first + h_resident; }
    const uint8_t* host_blob(int64_t e) const { return hblobs + (size_t) (e - h_first) * stride; }
    uint8_t* host_blob(int64_t e) { return hblobs + (size_t) (e - h_first) * stride; }
    uint8_t* peer_blobs = nullptr;   // the peer's slots (the experts the primary does not hold), on its card
    int n_slots = 0;            // hybrid mode: the experts resident in VRAM
};

struct Model {
    strata::core::ModelGeometry g;
    std::shared_ptr<strata::GgufModel> file;   // kept mapped: the experts are placed after the engine's buffers
    bool hybrid = false;                       // not every expert fits in VRAM: slots + the CPU pool
    // the packed Q8_0 decode copies (gguf weights, packed copy, bytes): given back when that lets every expert fit
    std::vector<std::tuple<const uint8_t*, uint8_t*, size_t>> packed;
    // the GGUF-native MTP block (blk.<n_layers>, loaded with --mtp): eh_proj over [enorm(e) | hnorm(h)], then an
    // attention + MoE layer, shared_head_norm, the main head
    bool has_mtp = false;
    Layer mtp;
    Mat eh_proj;
    const float *mtp_enorm = nullptr, *mtp_hnorm = nullptr, *mtp_shnorm = nullptr;
    int64_t top_k = 8;
    float eps = 1e-6f, rope_base = 1e7f;
    float expert_weights_scale = 1.0f;   // the file's routed_scaling_factor (MiniMax-M2: 2.0)
    int n_rot = 64;
    std::vector<Layer> layers;
    // the second card's copy of the layers it runs, and of the head (the layer split)
    std::vector<Layer> layers2;
    const float* out_norm = nullptr;
    const float* out_norm2 = nullptr;
    Mat head;
    Mat head2;
    const uint8_t* embd = nullptr;
    int embd_type = -1;
    size_t embd_row = 0;
};

constexpr size_t kMmqTail = 4096;
size_t gcd_sz(size_t a, size_t b) { return b ? gcd_sz(b, a % b) : a; }
size_t lcm_sz(size_t a, size_t b) { return a / gcd_sz(a, b) * b; }
size_t block_bytes(int type) {
    int be = 0, bb = 0;
    return strata::block_geometry((uint32_t) type, be, bb) ? (size_t) bb : 1;
}   // zeroed bytes after the last expert (MMQ reads a little past a matrix)

size_t row_bytes(int type, int64_t n) {
    int be = 0, bb = 0;
    if (!strata::block_geometry((uint32_t) type, be, bb)) return 0;
    return (size_t) (n / be) * (size_t) bb;
}

float bf16_to_f32(uint16_t b) {
    const uint32_t u = (uint32_t) b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

// IEEE binary16 -> f32 by hand. __ushort_as_half is a __device__ intrinsic and
// upload_f32 runs on the host, so a CUDA toolkit older than 12.5 refuses the
// call; widen the bits here instead, the way bf16_to_f32 does.
float f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t) (h & 0x8000) << 16;
    const uint32_t exponent = (h >> 10) & 0x1f;
    const uint32_t mantissa = h & 0x3ff;
    uint32_t u;
    if (exponent == 0) {
        if (mantissa == 0) {
            u = sign;                                               // +-0
        } else {
            uint32_t m = mantissa, shift = 0;                        // subnormal: normalize it
            while ((m & 0x400) == 0) {
                m <<= 1;
                ++shift;
            }
            u = sign | ((uint32_t) (127 - 14 - shift) << 23) | ((m & 0x3ff) << 13);
        }
    } else if (exponent == 0x1f) {
        u = sign | 0x7f800000u | (mantissa << 13);                   // inf / nan
    } else {
        u = sign | ((exponent + (127 - 15)) << 23) | (mantissa << 13);
    }
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

/// A tensor as f32 on the device (F32 as it is, BF16 / F16 widened).
float* upload_f32(const strata::TensorInfo& t, const uint8_t* src) {
    std::vector<float> h((size_t) t.elements());
    if (t.type == 0) std::memcpy(h.data(), src, h.size() * 4);
    else if (t.type == 30)
        for (size_t i = 0; i < h.size(); ++i) h[i] = bf16_to_f32((uint16_t) (src[2 * i] | (src[2 * i + 1] << 8)));
    else if (t.type == 1)   // F16
        for (size_t i = 0; i < h.size(); ++i)
            h[i] = f16_to_f32((uint16_t) (src[2 * i] | (src[2 * i + 1] << 8)));
    else if (t.type == 8 && h.size() % 32 == 0) {
        // Q8_0 (the Q8_0 file stores ssm_alpha / ssm_beta so): blocks of an fp16 scale and 32 int8, widened here
        for (size_t b = 0; b < h.size() / 32; ++b) {
            const uint8_t* q = src + b * 34;
            const float d = f16_to_f32((uint16_t) (q[0] | (q[1] << 8)));
            for (int j = 0; j < 32; ++j) h[b * 32 + j] = d * (float) (int8_t) q[2 + j];
        }
    } else {
        std::fprintf(stderr, "strata-m2: %s is %s, expected F32, BF16, F16 or Q8_0\n", t.name.c_str(), t.type_name());
        std::exit(1);
    }
    float* d = nullptr;
    CK(cudaMalloc(&d, h.size() * 4));
    CK(cudaMemcpy(d, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
    return d;
}

bool load(const std::string& path, Model& m, std::string& err, bool want_mtp) {
    const auto t0 = Clock::now();
    m.file = std::make_shared<strata::GgufModel>(strata::GgufModel::open(path));
    const strata::GgufModel& gm = *m.file;
    if (!strata::core::geometry_from_gguf(gm.meta(), m.g, m.top_k, err)) return false;
    if (m.g.family != strata::ModelFamily::MiniMaxM2) {
        err = "strata-m2 runs MiniMax-M2 (minimax-m2) files only";
        return false;
    }
    size_t checked = 0;
    if (!strata::core::check_gguf_layout(gm, m.g, err, &checked)) return false;
    const auto& g = m.g;
    // MiniMax-M2: full attention on every layer (no GDN), head_dim 128, 256 experts, top-4 or top-8.
    if (g.head_dim != 128 || g.has_ssm() || g.n_expert > 512 || m.top_k > 15) {
        err = "a geometry this driver's kernels do not take (head 128, no GDN layer, <= 512 experts, top-k <= 15)";
        return false;
    }
    const std::string p = "minimax-m2.";
    if (const strata::MetaValue* v = gm.meta().get(p + "attention.layer_norm_rms_epsilon")) m.eps = (float) v->num();
    if (const strata::MetaValue* v = gm.meta().get(p + "rope.freq_base")) m.rope_base = (float) v->num();
    if (const strata::MetaValue* v = gm.meta().get(p + "rope.dimension_count")) m.n_rot = (int) v->u;
    if (const strata::MetaValue* v = gm.meta().get(p + "expert_weights_scale")) m.expert_weights_scale = (float) v->num();
    if (const strata::MetaValue* v = gm.meta().get("tokenizer.ggml.eos_token_id"); v && v->is_num())
        g_eos_id = (int32_t) v->u;
    uint64_t bytes = 0;
    auto find = [&](const std::string& name, size_t& s) -> const strata::TensorInfo* {
        const strata::TensorInfo* t = gm.find(name, &s);
        if (!t) {
            std::fprintf(stderr, "strata-m2: missing %s\n", name.c_str());
            std::exit(1);
        }
        return t;
    };
    auto f32 = [&](const std::string& name) -> const float* {
        size_t s = 0;
        const strata::TensorInfo* t = find(name, s);
        bytes += t->elements() * 4;
        return upload_f32(*t, gm.shard(s).tensor_data(*t));
    };
    auto mat = [&](const std::string& name) -> Mat {
        size_t s = 0;
        const strata::TensorInfo* t = find(name, s);
        Mat w;
        w.type = (int) t->type;
        w.n_in = (int) t->shape[0];
        w.n_out = (int) t->shape[1];
        if (!K::native_mmvq_supported(w.type)) {
            std::fprintf(stderr, "strata-m2: %s is %s, which the native MMVQ does not take\n", name.c_str(), t->type_name());
            std::exit(1);
        }
        const uint64_t nb = strata::tensor_payload_bytes(*t);
        uint8_t* d = nullptr;
        CK(cudaMalloc(&d, nb));
        CK(cudaMemcpy(d, gm.shard(s).tensor_data(*t), nb, cudaMemcpyHostToDevice));
        bytes += nb;
        w.d = d;
        // STRATA_Q8_PACKED (on by default here): the lossless qs / d plane copy the Q8_0 decode kernels read instead,
        // bitwise the same outputs
        if (w.type == 8 && K::native_q8_0_packed_enabled() && K::native_q8_0_packed_eligible(w.n_in, w.n_out)) {
            std::vector<uint8_t> packed(nb);
            K::native_q8_0_pack_host(gm.shard(s).tensor_data(*t), packed.data(), w.n_in, w.n_out);
            uint8_t* pd = nullptr;
            CK(cudaMalloc(&pd, nb));
            CK(cudaMemcpy(pd, packed.data(), nb, cudaMemcpyHostToDevice));
            K::native_q8_0_packed_register(d, pd, w.n_in, w.n_out);
            m.packed.emplace_back(d, pd, (size_t) nb);
            bytes += nb;
        }
        return w;
    };
    auto L = [](int64_t l, const char* s) { return "blk." + std::to_string(l) + "." + s; };
    m.layers.resize((size_t) g.n_layers);
    // MiniMax-M2's released GGUFs carry no MTP block (the config declares num_mtp_modules 3, the files ship
    // none), so there is no draft layer here: the engine decodes one token at a time.
    (void) want_mtp;
    m.has_mtp = false;
    // The layers' weights are uploaded to the CURRENT device, so the same code builds the second card's copy:
    // the second card runs the layers from g_split_first on, and it reads its own copy of their weights.
    for (int pass = 0; pass < (g_second_card >= 0 ? 2 : 1); ++pass) {
      if (pass == 1) {
        CK(cudaSetDevice(g_second_card));
        CK(cudaGetLastError());
        m.layers2.resize((size_t) (g.n_layers - g_split_first));
      }
      std::vector<Layer>& lay = pass == 0 ? m.layers : m.layers2;
      const int64_t l0 = pass == 0 ? 0 : g_split_first;
      for (int64_t l = l0; l < g.n_layers; ++l) {
        Layer& y = lay[(size_t) (l - l0)];
        y.attn = true;   // full attention on EVERY layer
        y.attn_norm = f32(L(l, "attn_norm.weight"));
        y.post_norm = f32(L(l, "ffn_norm.weight"));   // the MoE's norm (the other families call it post_attention_norm)
        y.wq = mat(L(l, "attn_q.weight"));
        y.wk = mat(L(l, "attn_k.weight"));
        y.wv = mat(L(l, "attn_v.weight"));
        y.wo = mat(L(l, "attn_output.weight"));
        y.q_norm = f32(L(l, "attn_q_norm.weight"));   // per layer: one vector for all the query heads
        y.k_norm = f32(L(l, "attn_k_norm.weight"));   // per layer: one vector for all the key heads
        y.router = f32(L(l, "ffn_gate_inp.weight"));
        y.router_bias = f32(L(l, "exp_probs_b.bias"));
        // MiniMax-M2 has NO shared expert (shared_intermediate_size 0): there is no shared gate row, so
        // router_ext stays null and the combine below adds nothing beside the routed experts.
        // the routed experts: [gate | up | down] per expert, the native expert layout
        size_t sg = 0, su = 0, sd = 0;
        const strata::TensorInfo* tg = find(L(l, "ffn_gate_exps.weight"), sg);
        const strata::TensorInfo* tu = find(L(l, "ffn_up_exps.weight"), su);
        const strata::TensorInfo* td = find(L(l, "ffn_down_exps.weight"), sd);
        if (tg->type != tu->type) {
            err = L(l, "ffn_gate_exps / ffn_up_exps differ in type");
            return false;
        }
        if (!K::native_expert_supported((int) tg->type, (int) td->type, g.n_embd, g.n_ff)) {
            err = L(l, "experts are ") + tg->type_name() + "/" + td->type_name() + ", which the native expert kernels " +
                  "do not take (build with -DSTRATA_Q6K_EXPERTS=ON for Q6_K)";
            return false;
        }
        y.L = K::native_expert_layout((int) tg->type, (int) td->type, g.n_embd, g.n_ff);
        const size_t gsz = (size_t) g.n_ff * y.L.gu_row, dsz = (size_t) g.n_embd * y.L.d_row;
        // a stride in whole blocks of both parts' types (MMQ steps over the experts by blocks), 16-byte aligned
        const size_t al = lcm_sz(lcm_sz(block_bytes(y.L.gu_type), block_bytes(y.L.d_type)), 16);
        y.stride = (y.L.bytes + al - 1) / al * al;
        y.src_g = gm.shard(sg).tensor_data(*tg);
        y.src_u = gm.shard(su).tensor_data(*tu);
        y.src_d = gm.shard(sd).tensor_data(*td);
        // The experts are read from the mapping, one expert at a time, when they are placed in VRAM.  Without a
        // read-ahead the kernel serves those pages one by one: the placement of a 48 GB model then runs at the
        // mmap's page-fault rate (measured: 52 MB/s) instead of the drive's (4.3 GB/s).  Asking for the whole
        // region up front puts the drive in sequential mode.
        strata::platform::advise_willneed(y.src_g, strata::tensor_payload_bytes(*tg));
        strata::platform::advise_willneed(y.src_u, strata::tensor_payload_bytes(*tu));
        strata::platform::advise_willneed(y.src_d, strata::tensor_payload_bytes(*td));
        y.src_gsz = gsz;
        y.src_dsz = dsz;
      }
      if (pass == 1) {   // the second card's head (the last layer's output is read there)
        // the second card does NOT keep its own copy of the head: the head's weights stay on the primary
        // (card 0), and the last row's hidden is copied over (12 Ko) when the head is called.  Keeping the
        // head on card 1 cost 2.46 Gio of VRAM that the card does not have (the head did not fit: cudaMalloc
        // returned null, "invalid argument").
        m.out_norm2 = nullptr;   // overwritten after the pass loop if split mode is active
        m.head2.d = nullptr;      // use the primary's head
        CK(cudaSetDevice(0));
        CK(cudaGetLastError());
        {   // The primary's LAST layer reads the NEXT layer's norm, and the next layer is on the second card:
            // the primary gets a copy of that norm (12 KB) so it can read it.
            const size_t nb = (size_t) g.n_embd * 4;
            // through the host, not a peer copy: a device-to-device copy between two different GPU generations
            // is the path that hung this machine
            float* nh = (float*) std::malloc(nb);
            CK(cudaMemcpy(nh, m.layers2[0].attn_norm, nb, cudaMemcpyDeviceToHost));
            CK(cudaSetDevice(0));
            CK(cudaGetLastError());
            float* n0 = nullptr;
            CK(cudaMalloc(&n0, nb));
            CK(cudaMemcpy(n0, nh, nb, cudaMemcpyHostToDevice));
            std::free(nh);
            m.layers[(size_t) g_split_first].attn_norm = n0;
        }
        std::fprintf(stderr, "strata-m2: the second card has its own copy of the layers %d-%lld\n", g_split_first,
                     (long long) g.n_layers);
      }
    }
    if (m.has_mtp) {
        const int64_t l = g.n_layers;
        m.eh_proj = mat(L(l, "nextn.eh_proj.weight"));
        m.mtp_enorm = f32(L(l, "nextn.enorm.weight"));
        m.mtp_hnorm = f32(L(l, "nextn.hnorm.weight"));
        m.mtp_shnorm = f32(L(l, "nextn.shared_head_norm.weight"));
    }
    m.out_norm = f32("output_norm.weight");
    m.head = mat("output.weight");
    if (g_second_card >= 0) {
        // the second card's last layer applies the output norm (out_norm) to produce hmain; give it its own copy
        const size_t nb = (size_t) g.n_embd * 4;
        float* nh = (float*) std::malloc(nb);
        CK(cudaMemcpy(nh, m.out_norm, nb, cudaMemcpyDeviceToHost));
        CK(cudaSetDevice(g_second_card));
        CK(cudaGetLastError());
        float* n2 = nullptr;
        CK(cudaMalloc(&n2, nb));
        CK(cudaMemcpy(n2, nh, nb, cudaMemcpyHostToDevice));
        std::free(nh);
        m.out_norm2 = n2;
        CK(cudaSetDevice(0));
        CK(cudaGetLastError());
    }
    {
        size_t s = 0;
        const strata::TensorInfo* t = find("token_embd.weight", s);
        m.embd_type = (int) t->type;
        m.embd_row = row_bytes(m.embd_type, (int64_t) t->shape[0]);
        const uint64_t nb = strata::tensor_payload_bytes(*t);
        uint8_t* d = nullptr;
        CK(cudaMalloc(&d, nb));
        CK(cudaMemcpy(d, gm.shard(s).tensor_data(*t), nb, cudaMemcpyHostToDevice));
        m.embd = d;
        bytes += nb;
    }
    std::fprintf(stderr, "strata-m2: %s: %zu tensors (layout checked), %.2f GiB on the GPU in %.1f s (the routed "
                 "experts are placed after the engine's buffers)\n", path.c_str(), checked, (double) bytes / (1 << 30),
                 ms_since(t0) / 1000.0);
    return true;
}

/// Expert e of layer L as its native blob [gate | up | down] (L.L.bytes) at dst (host memory).
void expert_blob(const Layer& L, int64_t e, uint8_t* dst) {
    std::memcpy(dst, L.src_g + (size_t) e * L.src_gsz, L.src_gsz);
    std::memcpy(dst + L.L.up_off, L.src_u + (size_t) e * L.src_gsz, L.src_gsz);
    std::memcpy(dst + L.L.down_off, L.src_d + (size_t) e * L.src_dsz, L.src_dsz);
}

// ------------------------------------------------------------------------------------------------ the engine
float* dalloc(size_t n) {
    float* p = nullptr;
    CK(cudaMalloc(&p, n * 4));
    CK(cudaMemset(p, 0, n * 4));
    return p;
}
template <class T> T* dalloc_t(size_t n) {
    T* p = nullptr;
    CK(cudaMalloc(&p, n * sizeof(T)));
    CK(cudaMemset(p, 0, n * sizeof(T)));
    return p;
}

/// A picture in a prompt: its first cell, its rows, and the hash the conversation cache compares it by.
struct ImgKey {
    int64_t cell = 0, n = 0;
    uint64_t hash = 0;
    bool operator==(const ImgKey& o) const { return cell == o.cell && n == o.n && hash == o.hash; }
};

struct Engine {
    const Model* m = nullptr;
    cudaStream_t st = nullptr;
    int ctx = 32768, tmax = 512;
    int64_t pos = 0;   // cells holding a processed token
    // activations [tmax][...]
    float *x, *h, *y, *qkv, *hc, *z, *alpha, *beta, *gdn_y, *qg, *kx, *vx, *q, *gate, *att, *logits_r, *eparts, *moe_out,
        *sg, *su, *shh, *sy, *sgate, *ewts, *logits;
    float* hmain = nullptr;   // [tmax][N] the main model's final normed hidden of each row (the MTP seeds)
    float* hlast = nullptr;   // [N] that of the last processed position (zeros before the first): the current session's
    void *q8, *xq, *escr;
    int32_t *ids, *grp_start, *n_groups, *ent_dst, *ent_tok, *tok, *pos0, *pick;
    int32_t* hpin = nullptr;   // pinned staging: [0] decode token, [1] position, [2..9] picks, [10] max rows, [16..] tokens
    // A prompt chunk's tokens and a slot table, both pinned: their upload is an asynchronous copy, and the source
    // of an asynchronous copy must be pinned (a pageable source makes the card's SDMA engine read a page the driver
    // has not mapped into it - "Page not present", on a host address, the second card, 2026-10-29).
    int32_t* htoks = nullptr;
    unsigned long long* grp_ptr;
    float *pacc, *pm, *pl, *amv;
    int32_t* ami;
    // ---- The layer split: the two cards' buffers ------------------------------------------------------
    // The Engine's members hold the CURRENT card's pointers; the other card's set waits in `set[1]`, and crossing
    // the boundary exchanges them (use_set).  The list is a macro so the two copies cannot drift apart.
#define STRATA_M2_BUF_LIST(X)                                                                 \
    X(float*, x) X(float*, h) X(float*, y) X(float*, qkv) X(float*, hc) X(float*, z)              \
    X(float*, qg) X(float*, kx) X(float*, vx) X(float*, q) X(float*, gate) X(float*, att)         \
    X(float*, logits_r) X(float*, eparts) X(float*, moe_out) X(float*, pacc) X(float*, pm)        \
    X(float*, pl) X(float*, amv) X(float*, att_s) X(float*, sk_part) X(float*, logits)            \
    X(void*, q8) X(void*, xq) X(void*, escr) X(int32_t*, ids) X(int32_t*, grp_start)             \
    X(int32_t*, n_groups) X(int32_t*, ent_dst) X(int32_t*, ent_tok) X(int32_t*, tok)             \
    X(int32_t*, pos0) X(int32_t*, pick) X(int32_t*, rcnt) X(int32_t*, rcur)                      \
    X(int32_t*, rbounds) X(int32_t*, rmax) X(int32_t*, ident) X(int32_t*, rinv)                   \
    X(int32_t*, ami) X(unsigned long long*, grp_ptr) X(cudaStream_t, st)                         \
    X(float*, ecpu) X(int32_t*, d_miss) X(float*, alpha) X(float*, beta) X(float*, gdn_y)         \
    X(float*, sg) X(float*, su) X(float*, shh) X(float*, sy) X(float*, sgate) X(float*, ewts)     \
    X(float*, hmain) X(float*, hlast) X(void*, mq_x) X(void*, mq_hq) X(__half*, x16)             \
    X(__half*, att_p16) X(__half*, k16) X(__half*, v16) X(__half*, q16) X(float*, mq_gu)          \
    X(float*, mq_h) X(int32_t*, d_delta)                                                         \
    X(cudaEvent_t, ev_route) X(cudaEvent_t, ev_xq) X(cublasHandle_t, blas)                    \
    X(float*, vbuf) X(void*, vq) X(void*, vgrp) X(float*, vx_) X(float*, vh_)                \
    X(int32_t*, d_rcell) X(int32_t*, d_rslot) X(int32_t*, d_rdelta)                           \
    X(int32_t*, d_gslot) X(int32_t*, d_grow0) X(int32_t*, d_gcnt)
    struct Buf {
#define STRATA_M2_BUF_DECL(T, N) T N = nullptr;
        STRATA_M2_BUF_LIST(STRATA_M2_BUF_DECL)
#undef STRATA_M2_BUF_DECL
    };
    /// The layer l on the card that runs it (the layer split: the second card has its own copy of the weights).
    const Layer& layer_of(size_t l) const {
        if (split_dev >= 0 && (int) l >= split_layer && (size_t) (l - split_layer) < m->layers2.size())
            return m->layers2[(size_t) (l - split_layer)];
        return m->layers[l];
    }
    Layer& layer_of(size_t l) {
        if (split_dev >= 0 && (int) l >= split_layer && (size_t) (l - split_layer) < m->layers2.size())
            return const_cast<Layer&>(m->layers2[(size_t) (l - split_layer)]);
        return const_cast<Layer&>(m->layers[l]);
    }
    /// The head (the output matrix) on the card that reads it.
    const Mat& head_mat() const {
        if (split_dev >= 0 && split_layer < (int) m->layers.size() && m->head2.d) return m->head2;
        return m->head;
    }
    float* split_stage = nullptr;   // the crossing's host staging (the peer copy is opt-in)
    bool split_copy_peer = false;    // a device-to-device copy between the two cards (the risky path)
    Buf set[2];           // the two cards' sets
    int cur_set = 0;      // the current card's set
    int split_dev = -1;   // the second card (--layer-split), -1: none
    int split_layer = -1; // the first layer that runs on the second card
    cudaEvent_t ev_cross = nullptr;   // the split's crossing: recorded on the first card, waited for on the second
    /// The cards' buffers: the other set's come in (the current members are NOT saved).
    void load_set(int k) {
        Buf& b = set[k];
#define STRATA_M2_BUF_LOAD(T, N) N = b.N;
        STRATA_M2_BUF_LIST(STRATA_M2_BUF_LOAD)
#undef STRATA_M2_BUF_LOAD
        cur_set = k;
        // the cuBLAS handle follows its card's stream
        if (blas) cublasSetStream(blas, st);
    }
    /// The current members into the set k (the two cards' sets are kept apart).
    void save_set(int k) {
        Buf& a = set[k];
#define STRATA_M2_BUF_SAVE(T, N) a.N = N;
        STRATA_M2_BUF_LIST(STRATA_M2_BUF_SAVE)
#undef STRATA_M2_BUF_SAVE
    }
    /// The cards' buffers: the current members go back into their set, the other set's come in.
    void use_set(int k) {
        if (k == cur_set) return;
        save_set(cur_set);
        load_set(k);
    }
    int n_chunk = 0, att_rows = 0;   // attention partials: att_rows query rows x n_chunk chunks
    std::vector<float*> conv, S;   // per layer; the MTP block's at index n_layers: the CURRENT session's (select)
    std::vector<void*> kc, vc;     // the same indices: the K / V cache (float, __half or int8: --kv)
    std::vector<__half*> ksc, vsc;  // the same indices: their scales, one per cell, KV head and block (--kv int8 only, else null)
    // verify windows (T <= kVerify): the GDN inputs of every row per layer, kept for the commit
    static constexpr int kVerify = 8;
    std::vector<float*> v_qkv, v_hc, v_alpha, v_beta;
    // the MTP block (when loaded)
    float *mtp_cat = nullptr, *hmtp = nullptr, *logits_m = nullptr;
    int32_t *tok_m = nullptr, *pos0_m = nullptr, *pick_m = nullptr;
    // graphs: [0] the decode step, [T] a verify window of T rows, mtp1 the MTP draft step
    // (the graphs hold the current session's state pointers as kernel arguments: one set per session)
    struct GSet {
        cudaGraphExec_t g1 = nullptr, gv[kVerify + 1] = {}, gm1 = nullptr;
    };
    std::vector<GSet> gsets;
    GSet& gs() { return gsets[(size_t) cur]; }
    strata::prefill::Gemm gemm[2];   // one per card (the layer split): each holds its own workspace
    cublasHandle_t blas = nullptr;   // the handle of the current card (the layer split)
    float* att_s = nullptr;            // the prompt attention's scores / probabilities, kAttScoreFloats floats
    static constexpr int kSplitK = 16;
    float* sk_part = nullptr;          // the split-K partials of the F32 projections: [kSplitK][tmax][<= 257]
    __half* att_p16 = nullptr;         // the probabilities in FP16 (kAttScoreFloats halves)
    __half *k16 = nullptr, *v16 = nullptr, *q16 = nullptr;   // the prompt attention's FP16 operands
    // The prompt attention on the FP16 GEMMs, per card (the layer split: the cards can differ).  Off on gfx103x: there
    // the hipBLAS strided-batched FP16-in / FP32-out GEMMs of the scores and the values returned non-finite rows from
    // finite operands inside this engine (RX 6800, ROCm 7.1 / hipBLAS 3: a 25-token prompt went NaN at a different layer
    // each run, the logits were NaN, the server's sampled replies one byte token repeated - the same GEMMs alone,
    // 12000 calls, never failed).  The split-K kernels the decode uses take the chunk instead.  STRATA_M2_ATT_GEMM=1 / 0
    // forces either path.
    bool att_gemm[2] = {true, true};
    size_t kAttScoreFloats = (size_t) 64 << 20;   // 256 MB (16 M floats when the experts' fit is close)
    // the prompt's MMQ experts
    bool use_mmq = false;
    std::unique_ptr<MMQ::Context> mctx;
    int32_t *rcnt = nullptr, *rcur = nullptr, *rbounds = nullptr, *rmax = nullptr, *ident = nullptr, *rinv = nullptr;
    void *mq_x = nullptr, *mq_hq = nullptr;
    float *mq_gu = nullptr, *mq_h = nullptr;
    __half* x16 = nullptr;
    bool use_graph = true;
    int mtp_d = 0;   // MTP drafts per speculative step (0: none; greedy requests only)

    // ---- sessions: the conversation states.  One for a single-request engine; --batch N keeps N, each with its own
    // recurrent state and KV cache, so N requests run at once: their rows share one decode window (forward_rows).  The
    // members conv / S / kc / vc / pos / hlast above are the CURRENT session's (select() swaps them).
    struct Sess {
        std::vector<float*> conv, S;
        std::vector<void*> kc, vc;
        std::vector<__half*> ksc, vsc;
        float* hlast = nullptr;
        int64_t pos = 0;
        int32_t rdelta = 0;                 // rotary offset of the text after the last picture (see k_attn_prep)
        std::vector<int32_t> live;          // the tokens whose state it holds (cell i = live[i])
        std::vector<ImgKey> imgs;           // the pictures among them
        bool active = false;                // a batch slot is decoding in it
        uint64_t used = 0;                  // when it was last used (the LRU pick)
    };
    std::vector<Sess> sess;
    int n_sess = 1, cur = 0;
    uint64_t use_clock = 0;
    // device tables for batch rows: per state index (layer, then the MTP block) the session pointers
    float **d_conv_tab = nullptr, **d_S_tab = nullptr;
    void **d_kc_tab = nullptr, **d_vc_tab = nullptr;
    __half **d_ksc_tab = nullptr, **d_vsc_tab = nullptr;
    int32_t *d_rcell = nullptr, *d_rslot = nullptr, *d_rdelta = nullptr;   // [kVerify] the rows' cell / session / rotary offset
    // the GDN layers run a slot's rows in sequence: per group (a slot) its session, its first row and its row count
    int32_t *d_gslot = nullptr, *d_grow0 = nullptr, *d_gcnt = nullptr;
    int rows_G = 0;                                                        // groups of the rows being run
    int32_t* hrow = nullptr;                                                // pinned staging for the rows
    bool rows_mode = false;                                                 // layer(): the rows are sessions' (not consecutive cells)
    bool bmtp = false;                                                      // --batch-mtp: batch windows verify MTP proposals
    int64_t rows_pmax = 0;                                                  // (eager) the highest cell of the rows
    // pictures (--vision): the request's rows replace the embeddings of its <|image_pad|> cells; M-RoPE
    bool vision = false;
    // The low-RAM mode: no host copy of the experts at all.  The experts stay in the GGUF mapping and the
    // CPU pool reads them where they lie (the job's blob is the gate rows, the job's NativeFmt carries the
    // byte distances to the up and down tensors).  On a 64 GB box the copy of a 49 GB model does not fit
    // beside the file: the placement then ran at 39-52 MB/s and the box thrashed (3.5 GB of swap).
    bool low_ram = false;
    // --kv: the K / V cache's element.  FP32 by default (byte-identical to the engine before the flag), FP16 at half the
    // VRAM, INT8 at a quarter of it (one float scale per cell, per KV head and per block of kKvBlock dims, written with
    // the row: 8 MiB a layer at 64K).  A rounded cache: the values are O(1 .. 50) and every sum stays in FP32 (the
    // split-K kernel and the FP16 GEMMs both accumulate there).
    enum KvMode { KV_F32 = 0, KV_F16 = 1, KV_I8 = 2 };
    KvMode kv_mode = KV_F32;
    /// The cache's element size in bytes (a session's K or V, one cell, one head).
    size_t kv_esz() const { return kv_mode == KV_I8 ? 1 : (kv_mode == KV_F16 ? 2 : 4); }
    const char* kv_word() const { return kv_mode == KV_I8 ? "INT8" : (kv_mode == KV_F16 ? "FP16" : "FP32"); }
    FILE* dump_layers = nullptr;      // the layer outputs, like q36ref's: [pos][layer][n_embd] f32
    std::vector<float> dump_row;
    // The repack buffers: one expert's blob, for the GPU's copy out of the mapping.  ONE PER STREAM: the copy
    // out of the host buffer is ASYNC, so the host buffer stays in use until the stream has taken it - one
    // buffer shared by the prompt path and the slot replacement has them corrupt each other's expert (the
    // two copies overlap, and the answer changes with the speed of the machine).  PINNED for the same reason:
    // the copy is asynchronous, and a pageable source hands the card a page of ours the driver has not pinned -
    // on a machine under memory pressure that page is not resident and the copy faults ("Page not present", a
    // host address, the second card, 2026-10-29).
    uint8_t* repack = nullptr;      // the decode path's (the slot replacement)
    uint8_t* repack_pf = nullptr;   // the prompt path's (the staged copy)
    std::vector<strata::kernels::cpu::NativeFmt> zc_fmt;   // the per-job layouts of the experts read in place
    int32_t* d_mtab = nullptr;          // [ctx][3] t, h, w of the current request's prompt cells (vision)
    const int32_t* mtab_now = nullptr;  // set for a forward over prompt cells that need it, else null
    int32_t* d_delta = nullptr;         // device scalar: the current session's rotary offset (decode graphs read it)
    float* d_img = nullptr;             // the request's picture rows [n][N]
    size_t img_cap = 0;                 // floats
    int32_t* d_map = nullptr;           // [tmax] a chunk's row -> picture row (or -1)
    int32_t* h_map = nullptr;           // pinned
    bool map_on = false;                // this forward's embeddings take picture rows
    std::vector<int32_t> img_map;       // the request's cell -> picture row (-1: text)
    int64_t img_prompt_cells = 0;       // the request's prompt length: cells below it use the table

    // ---- the hybrid mode (Model::hybrid): per state index (the layers, then the MTP block) the slot tables; a
    // batch of up to hybrid_t rows runs its VRAM-resident experts on the GPU and the others on the CPU pool, side by
    // side; a bigger prompt chunk streams each layer's experts into a staging buffer for MMQ (two, double-buffered)
    struct Hy {
        std::vector<int> slot_of, exp_in_slot;
        std::vector<double> score;   // routing counts, halved at each rebalance
        int32_t* slot_of_d = nullptr;
#if defined(STRATA_Q36_CPU_POOL)
        strata::kernels::cpu::NativeFmt fmt;
#endif
    };
    std::vector<Hy> hy;
    // ---- the peer tier: a second card with its own expert slots ----
    // The primary keeps what it had: the dense weights, the KV cache, its own expert slots.  The peer holds the
    // experts the primary does not, and computes the rows of the entries routed to them, straight into the same
    // pinned host rows the CPU pool writes (portable, so both cards see the same addresses).  The rows are
    // bit-identical whichever card computes them: the same kernels, the same activations.
    int peer_device = -1;              // the second card (--peer-device N), -1: none
    cudaStream_t peer_st = nullptr;     // the peer's stream
    cudaEvent_t peer_done = nullptr;    // the peer's share of a layer is done
    struct PeerHy {
        std::vector<int32_t> slot_of;  // [n_expert]: the peer's slot, or -1
        int32_t* slot_of_d = nullptr;  // the same, on the peer's device
    };
    std::vector<PeerHy> peer_hy;
    // the peer's own buffers (on its device)
    unsigned long long* peer_grp_ptr = nullptr;
    int32_t* peer_grp_start = nullptr;
    int32_t* peer_n_groups = nullptr;
    int32_t* peer_ent_dst = nullptr;
    int32_t* peer_ent_tok = nullptr;
    int32_t* peer_ids = nullptr;        // the routed ids, copied over
    void* peer_xq = nullptr;            // the q8_1 activations, copied over
    void* peer_escr = nullptr;          // the expert scratch
    float* peer_eparts = nullptr;       // the experts' rows, on the peer
    int32_t* peer_slot_of_d = nullptr;  // the peer's slot table, ON THE PRIMARY (the primary's kernels read it)
    int32_t* peer_row = nullptr;        // [n_ent] on the peer: the entry's row in the peer's own output (-1: none)
    size_t peer_rows_max = 0;           // the dense rows the peer's buffers hold (the experts of a layer)
    long long hy_peer = 0;
    int hybrid_t = 32;
    bool main_hybrid = false;   // some main layer is hybrid (else only the MTP block is: the decode graph stays)
    bool stream_native = false;   // native (IQ1_M) prompt path streams the pool's experts to the GPU (kill switch
                                  // STRATA_M2_NO_STREAM_NATIVE); it is the path of every hybrid layer, MMQ or not
    bool mtp_graph = true;      // the MTP draft step may be a graph (not when its layer is hybrid)
    /// state index li's layer runs hybrid (its experts not all in VRAM)
    bool lhy(size_t li) const {
        if (!m->hybrid) return false;
        const Layer& L = li < m->layers.size() ? m->layers[li] : m->mtp;
        return L.n_slots < (int) m->g.n_expert;
    }
#if defined(STRATA_Q36_CPU_POOL)
    std::unique_ptr<strata::kernels::cpu::ExpertPool> pool;
    std::vector<strata::kernels::cpu::ExpertJobMulti> jobs;
#endif
    float *h_h = nullptr, *cpu_out = nullptr, *ecpu = nullptr;
    int32_t *h_ids = nullptr, *h_miss = nullptr, *d_miss = nullptr, *h_cnt = nullptr;
    uint8_t* acts = nullptr;   // per token: its quantized activation (kNativeActBytes)
    cudaEvent_t ev_route = nullptr;
    // the activations are ready: recorded after the quantization, so the peer copies them whole
    cudaEvent_t ev_xq = nullptr;
    // The streaming buffers come in one pair PER CARD (the layer split): the kernels that read a layer's streamed
    // experts run on that layer's own card, and a kernel cannot read the other card's memory -- it faults with
    // "page not present" at the first row it touches.  Card 0's pair is stage / ev_copy / ev_used, card 1's stage2...
    // In the DEFAULT mode the pair is PINNED HOST memory (stream_vram == false): the pool's experts are repacked
    // there and the GPU reads them over PCIe, so the streaming costs no VRAM and the slots keep theirs (the decode's
    // speed).  STRATA_M2_STREAM_VRAM=1 brings back the VRAM pair (one per card, 2 x the layer per card).
    // The fill pool: the threads that copy a prompt layer's pooled experts into the pinned staging buffer.
    //
    // `stream_kick` filled them on the calling thread, one `expert_blob` at a time (three memcpys out of the file
    // mapping, each of them paying its page faults).  A staging buffer holds all 256 experts of a layer and is
    // 1.32 GiB, so the pool's share of one layer (256 minus its 62 slots) is ~1 GiB: one thread took 1.6 s per layer
    // on the MiniMax-M2.5 UD-IQ2_XXS pack (a 512-token prompt: 96.6 s), while the GPU has the layer's own ~10-20 ms
    // queued, so it waited for nearly all of it.  The experts are disjoint slices of `dst` and `expert_blob` writes
    // nothing else, so the range splits over these threads with no lock and no ordering.
    // The count is `STRATA_M2_FILL_THREADS` (0: one thread, the A/B arm), started with the staging buffers.
    struct FillPool {
        std::mutex mu;
        std::condition_variable cv, done_cv;
        std::function<void(int, int)> job;   // the current fill: the experts [lo, hi)
        int job_n = 0, gen = 0, n_running = 0, nthreads = 0;
        bool quit = false;
        std::vector<std::thread> threads;
        void start(int n) {
            if (n <= 0 || !threads.empty()) return;
            nthreads = n;
            for (int t = 0; t < n; ++t) threads.emplace_back([this, t] { work(t); });
        }
        ~FillPool() {
            {
                std::lock_guard<std::mutex> lk(mu);
                quit = true;
            }
            cv.notify_all();
            for (auto& t : threads)
                if (t.joinable()) t.join();
        }
        void work(int t) {
            strata::platform::name_this_thread("strata-m2-fill", t);
            int seen = 0;
            for (;;) {
                {
                    std::unique_lock<std::mutex> lk(mu);
                    cv.wait(lk, [&] { return quit || gen != seen; });
                    if (quit) return;
                    seen = gen;
                }
                // A fixed stride per thread: the experts are equal-sized blobs, so this splits the work evenly and
                // each thread walks its own part of the layer (and of the file) in order.
                const int n = job_n;
                const int lo = (int) ((int64_t) n * t / nthreads), hi = (int) ((int64_t) n * (t + 1) / nthreads);
                if (lo < hi) job(lo, hi);
                {
                    std::lock_guard<std::mutex> lk(mu);
                    if (--n_running == 0) done_cv.notify_one();
                }
            }
        }
        /// `fn` over the experts [0, n) on the pool, and return once all of them are done.
        void run(std::function<void(int, int)> fn, int n) {
            if (nthreads <= 0 || n < 2) {
                fn(0, n);
                return;
            }
            {
                std::lock_guard<std::mutex> lk(mu);
                job = std::move(fn);
                job_n = n;
                n_running = nthreads;
                ++gen;
            }
            cv.notify_all();
            std::unique_lock<std::mutex> lk(mu);
            done_cv.wait(lk, [&] { return n_running == 0; });
        }
    };
    FillPool fill;
    uint8_t* stage[2] = {};
    cudaStream_t cst = nullptr;
    cudaEvent_t ev_copy[2] = {}, ev_used[2] = {};
    uint8_t* stage2[2] = {};
    cudaStream_t cst2 = nullptr;
    cudaEvent_t ev_copy2[2] = {}, ev_used2[2] = {};
    bool stream_vram = false;
    /// The card a layer's kernels run on (the layer split).
    int card_of(int li) const { return split_dev >= 0 && li >= split_layer ? split_dev : 0; }
    bool on_card2(int d) const { return split_dev > 0 && d == split_dev; }
    bool vram_stage(int d) const { return stream_vram && on_card2(d); }
    cudaStream_t cst_of(int d) const { return vram_stage(d) ? cst2 : cst; }
    uint8_t* stage_of(int d, int b) const { return vram_stage(d) ? stage2[b] : stage[b]; }
    cudaEvent_t ev_copy_of(int d, int b) const { return vram_stage(d) ? ev_copy2[b] : ev_copy[b]; }
    cudaEvent_t ev_used_of(int d, int b) const { return on_card2(d) ? ev_used2[b] : ev_used[b]; }
    std::vector<int> stream_seq;   // the state indices a streamed forward visits, in order
    int stream_pos = 0;
    long long hy_cpu = 0, hy_gpu = 0, hy_swaps = 0;   // entries computed on the CPU / GPU, slot replacements
    double hy_cpu_ms = 0;
    double hy_wait_ms = 0;   // time the pool's thread spent waiting for the routing's rows to reach the host
    // decode / verify rows: their routing is counted once the rows are kept (commit), so a verify window's rejected
    // drafts never move the slots; the slots follow the counts at every kRebalanceEvery-th position exactly (a
    // window never steps over one), so plain and speculative decoding see the same slot tables at every position
    static constexpr int kRebalanceEvery = 32;
    bool rows_deferred = false;          // set while a decode step or a verify window runs
    std::vector<std::vector<int32_t>> row_ids;   // [main layer][row * top_k + k] of the last such call
    void count_rows(int n) {
        const int Kp = (int) m->top_k;
        for (size_t li = 0; li < row_ids.size() && li < hy.size(); ++li)
            if (!row_ids[li].empty())
                for (int e = 0; e < n * Kp; ++e) hy[li].score[(size_t) row_ids[li][(size_t) e]] += 1.0;
    }
    void after_rows(int n) {   // a decode step / verify window kept n rows: count them, rebalance at the boundary
        if (!m->hybrid) return;
        count_rows(n);
        if (pos % kRebalanceEvery == 0) {
            CK(cudaStreamSynchronize(st));
            rebalance(8);
        }
    }

    /// A session's K or V cache for this context, zeroed: FP32, FP16 (--kv f16) or INT8 (--kv int8).  INT8 needs one
    /// float scale per cell, per KV head and per block of kKvBlock dims, written with the row (8 MiB a layer at 64K):
    /// `scales` takes it, and is left null for the other two.
    void* kv_alloc(__half** scales) const {
        const size_t n = (size_t) m->g.n_head_kv * ctx * m->g.head_dim;
        *scales = nullptr;
        if (kv_mode == KV_I8) {
            *scales = dalloc_t<__half>((size_t) m->g.n_head_kv * ctx * kKvBlocks);
            return (void*) dalloc_t<int8_t>(n);
        }
        return kv_mode == KV_F16 ? (void*) dalloc_t<__half>(n) : (void*) dalloc(n);
    }

    void alloc_session(Sess& ss) const {
        const auto& g = m->g;
        const size_t C = (size_t) g.ssm_conv_channels;
        __half *sk = nullptr, *sv = nullptr;
        int sess_dev = 0;   // the card the layer's K / V are on (the layer split)
        for (size_t li = 0; li < m->layers.size(); ++li) {
            const Layer& L = m->layers[li];
            // the layer split: the layer's K / V live on its card
            if (split_dev >= 0) {
                const int ldev = (int) li >= split_layer ? split_dev : 0;
                if (ldev != sess_dev) {
                    CK(cudaSetDevice(ldev));
                    CK(cudaGetLastError());
                    sess_dev = ldev;
                }
            }
            if (L.attn) {
                ss.conv.push_back(nullptr);
                ss.S.push_back(nullptr);
                ss.kc.push_back(kv_alloc(&sk));
                ss.vc.push_back(kv_alloc(&sv));
                ss.ksc.push_back(sk);
                ss.vsc.push_back(sv);
            } else {
                ss.conv.push_back(dalloc(C * (g.ssm_d_conv - 1)));
                ss.S.push_back(dalloc((size_t) g.ssm_v_heads * g.ssm_state_size * g.ssm_state_size));
                ss.kc.push_back(nullptr);
                ss.vc.push_back(nullptr);
                ss.ksc.push_back(nullptr);
                ss.vsc.push_back(nullptr);
            }
        }
        if (m->has_mtp) {   // the MTP block's own K / V (an attention layer at index n_layers)
            ss.conv.push_back(nullptr);
            ss.S.push_back(nullptr);
            ss.kc.push_back(kv_alloc(&sk));
            ss.vc.push_back(kv_alloc(&sv));
            ss.ksc.push_back(sk);
            ss.vsc.push_back(sv);
        }
        if (split_dev >= 0) {   // the session's hlast is on the first card (the layer split)
            CK(cudaSetDevice(0));
            CK(cudaGetLastError());
        }
        ss.hlast = dalloc((size_t) g.n_embd);
    }

    /// The current session: `i`'s state, cache and position become the engine's (conv / S / kc / vc / pos / hlast).
    void select(int i) {
        if (i == cur) return;
        sess[(size_t) cur].pos = pos;
        cur = i;
        load_cur();
    }
    void load_cur() {
        Sess& ss = sess[(size_t) cur];
        conv = ss.conv; S = ss.S; kc = ss.kc; vc = ss.vc; ksc = ss.ksc; vsc = ss.vsc;
        hlast = ss.hlast;
        pos = ss.pos;
        ss.used = ++use_clock;
        set_delta(ss.rdelta);
    }
    /// The rotary offset the decode graphs read (a device scalar), for the current session.
    void set_delta(int32_t d) {
        sess[(size_t) cur].rdelta = d;
        if (!d_delta) return;
        hpin[60] = d;
        CK(cudaMemcpyAsync(d_delta, hpin + 60, 4, cudaMemcpyHostToDevice, st));
        CK(cudaStreamSynchronize(st));   // (the pinned word is reused)
    }

    /// The K / V cache's element type, for a call site: the attention kernels are templates on it.  `f` receives an
    /// element pointer (whose pointee is only read as a type: the three are float, __half and int8_t).
    template <class F>
    void kv_visit(F&& f) const {
        if (kv_mode == KV_I8) f((int8_t*) nullptr);
        else if (kv_mode == KV_F16) f((__half*) nullptr);
        else f((float*) nullptr);
    }

    /// Frees the MMQ buffers.  A hybrid layer cannot use them (MMQ reads a whole layer's experts from one block,
    /// `L.blobs`, and a hybrid layer's experts live in two places), and they were allocated with the first card
    /// current, so the second card's layers could not reach them either.  The members are part of the two cards'
    /// `set[]`, so both copies are cleared with them.
    void drop_mmq() {
        if (!use_mmq) return;
        int prev = 0;
        CK(cudaGetDevice(&prev));
        CK(cudaSetDevice(0));   // init() allocated them there
        CK(cudaGetLastError());
        for (void* p : {mq_x, mq_hq, (void*) mq_gu, (void*) mq_h})
            if (p) CK(cudaFree(p));
        for (int32_t* p : {rcnt, rcur, rbounds, rmax, ident, rinv})
            if (p) CK(cudaFree(p));
        mctx.reset();
        mq_x = mq_hq = nullptr;
        mq_gu = mq_h = nullptr;
        rcnt = rcur = rbounds = rmax = ident = rinv = nullptr;
        for (Buf& b : set) {
            b.mq_x = b.mq_hq = nullptr;
            b.mq_gu = b.mq_h = nullptr;
            b.rcnt = b.rcur = b.rbounds = b.rmax = b.ident = b.rinv = nullptr;
        }
        use_mmq = false;
        CK(cudaSetDevice(prev));
        CK(cudaGetLastError());
    }

    void init(const Model& model, int ctx_, int tmax_, int n_sessions = 1, bool vision_ = false) {
        m = &model;
        ctx = ctx_;
        n_sess = std::max(1, std::min(n_sessions, kVerify));
        vision = vision_;
        tmax = std::max(tmax_, kVerify);
        const auto& g = m->g;
        // The layer split: each card holds the K / V cache of the layers it runs, so a context that two 16 GB cards
        // cannot hold (a 64K one: 31 layers x 512 MiB = 15.5 GiB per card, beside 2.5 GiB of weights) is said here with
        // the numbers, before anything is allocated - the split had no such check and died in a bare cudaMalloc.
        if (split_dev >= 0) {
            const size_t esz = kv_esz();
            const size_t kv_layer = (size_t) g.n_head_kv * (size_t) (g.head_dim * 2 * esz +
                                     (kv_mode == KV_I8 ? kKvBlocks * 2 * 4 : 0)) * (size_t) ctx;   // K and V, per layer
            const size_t on0 = (size_t) std::min<int64_t>((int64_t) split_layer, (int64_t) g.n_layers);
            const size_t n_kv[2] = {on0, (size_t) g.n_layers - on0 + (m->has_mtp ? 1 : 0)};
            for (int d = 0; d < 2; ++d) {
                CK(cudaSetDevice(d));
                CK(cudaGetLastError());
                size_t f = 0, t = 0;
                CK(cudaMemGetInfo(&f, &t));
                CK(cudaGetLastError());
                const size_t margin = (size_t) 1536 << 20;   // the engine's buffers (chunk 512 .. 4096, the scores)
                const size_t need = n_kv[d] * kv_layer;
                if (need + margin > f) {
                    const size_t room = f > margin ? f - margin : 0;
                    const long long most = n_kv[d] ? (long long) ((double) room / (double) (n_kv[d] * kv_layer) *
                                                                   (double) ctx) / 1024 * 1024
                                                   : 0;
                    std::fprintf(stderr, "strata-m2: not enough VRAM on card %d for a context of %d: the K / V cache of "
                                 "its %zu layers needs %.2f GiB (%s) and %.2f GiB are free after the weights, less "
                                 "%.2f GiB for the engine's buffers: use a shorter --max-context (at most about %lld on "
                                 "this card), --kv f16 (half of it) or --kv int8 (a quarter)\n", d, ctx, n_kv[d],
                                 (double) need / (1 << 30), kv_word(), (double) f / (1 << 30),
                                 (double) margin / (1 << 30), most);
                    std::exit(1);
                }
            }
            CK(cudaSetDevice(0));
            CK(cudaGetLastError());
        }
        const size_t T = (size_t) tmax, N = (size_t) g.n_embd, Kp = (size_t) m->top_k, FF = (size_t) g.n_ff;
        const size_t C = (size_t) g.ssm_conv_channels;
        // The cards' buffers: the same set, once per card (the layer split).  The stream and the events go
        // with it: a stream belongs to its card.
        auto alloc_core = [&](int card) {
            // the decode / verify's rows: per card (the verify's path runs on the card that decodes)
            d_rcell = dalloc_t<int32_t>(kVerify);
            d_rslot = dalloc_t<int32_t>(kVerify);
            d_rdelta = dalloc_t<int32_t>(kVerify);
            d_gslot = dalloc_t<int32_t>(kVerify);
            d_grow0 = dalloc_t<int32_t>(kVerify);
            d_gcnt = dalloc_t<int32_t>(kVerify);
            d_delta = dalloc_t<int32_t>(1);   // the rotary offset: per card (it belongs to its card)
            // the stream, the events and the pool's buffers: per card (a stream and an event belong to their card)
            CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
            CK(cudaEventCreateWithFlags(&ev_route, cudaEventDisableTiming));
            CK(cudaEventCreateWithFlags(&ev_xq, cudaEventDisableTiming));
            ecpu = dalloc(T * Kp * N);
            d_miss = dalloc_t<int32_t>(T * Kp);
            // the verify's scratch: per card too (they were allocated lazily, on whatever card was current)
            vbuf = dalloc(8 * 16384);
            vq = dalloc(K::native_q8_1_bytes((int) g.n_embd, 1) * 8);
            vgrp = dalloc(2048);
            vx_ = dalloc(8 * 2048);
            vh_ = dalloc(8 * 2048);
            x = dalloc(T * N); h = dalloc(T * N); y = dalloc(T * N); hmain = dalloc(T * N);
            qkv = dalloc(T * C); hc = dalloc(T * C); z = dalloc(T * g.ssm_value_dim);
            alpha = dalloc(T * g.ssm_v_heads); beta = dalloc(T * g.ssm_v_heads); gdn_y = dalloc(T * g.ssm_value_dim);
            qg = dalloc(T * 2 * g.n_head * g.head_dim); kx = dalloc(T * g.n_head_kv * g.head_dim);
            vx = dalloc(T * g.n_head_kv * g.head_dim); q = dalloc(T * g.n_head * g.head_dim);
            gate = dalloc(T * g.n_head * g.head_dim); att = dalloc(T * g.n_head * g.head_dim);
            logits_r = dalloc(T * (g.n_expert + 1));
            eparts = dalloc(T * Kp * N); moe_out = dalloc(T * N);
            sg = dalloc(T * g.n_ff_shexp); su = dalloc(T * g.n_ff_shexp); shh = dalloc(T * g.n_ff_shexp); sy = dalloc(T * N);
            sgate = dalloc(T); ewts = dalloc(T * Kp); logits = dalloc((size_t) kVerify * g.n_vocab);
            const int max_in = (int) std::max<int64_t>({2 * g.n_embd, g.ssm_value_dim, g.n_head * g.head_dim, g.n_ff_shexp});
            CK(cudaMalloc(&q8, K::native_q8_1_bytes(max_in, 1) * T));
            CK(cudaMalloc(&xq, K::native_q8_1_bytes((int) N, 1) * T));
            x16 = dalloc_t<__half>(T * (size_t) max_in);
            if (cublasCreate(&blas) != CUBLAS_STATUS_SUCCESS || cublasSetStream(blas, st) != CUBLAS_STATUS_SUCCESS) {
                std::fprintf(stderr, "strata-m2: cuBLAS handle failed\n");
                std::exit(1);
            }
            {   // the prompt attention's path on this card (see att_gemm)
                int dev = 0;
                cudaDeviceProp pr{};
                CK(cudaGetDevice(&dev));
                CK(cudaGetDeviceProperties(&pr, dev));
#if defined(__HIPCC__)
                const bool gfx103 = std::strncmp(pr.gcnArchName, "gfx103", 6) == 0;
                const char* arch = pr.gcnArchName;
#else
                const bool gfx103 = false;
                const char* arch = pr.name;
#endif
                const char* fe = std::getenv("STRATA_M2_ATT_GEMM");
                att_gemm[card] = fe ? fe[0] == '1' : !gfx103;
                if (!att_gemm[card])
                    std::fprintf(stderr, "strata-m2: card %d (%s): the prompt attention on the split-K kernels, not the FP16 "
                                 "GEMMs (%s)\n", card, arch, fe ? "STRATA_M2_ATT_GEMM=0" : "their FP32 output is not reliable "
                                 "on gfx103x; STRATA_M2_ATT_GEMM=1 forces them");
            }
            att_s = dalloc(kAttScoreFloats);
            sk_part = dalloc((size_t) kSplitK * T * (g.n_expert + 1));
            att_p16 = dalloc_t<__half>(kAttScoreFloats);
            k16 = dalloc_t<__half>((size_t) g.n_head_kv * ctx * g.head_dim);
            v16 = dalloc_t<__half>((size_t) g.n_head_kv * ctx * g.head_dim);
            q16 = dalloc_t<__half>(T * g.n_head * g.head_dim);
            use_mmq = MMQ::built() && std::getenv("STRATA_Q36_NO_MMQ") == nullptr;
            auto mmq_ok = [&](const Layer& L) {
                return MMQ::supported(L.L.gu_type) && MMQ::supported(L.L.d_type) && MMQ::fits(L.L.gu_type, 2 * g.n_ff) &&
                       MMQ::fits(L.L.d_type, g.n_embd);
            };
            for (const Layer& L : m->layers) use_mmq = use_mmq && mmq_ok(L);
            if (m->has_mtp) use_mmq = use_mmq && mmq_ok(m->mtp);
            // The native streamed path serves every layer MMQ cannot read: MMQ needs all of a layer's experts in one
            // block, so a hybrid layer (some experts in slots) always takes the native path, and `use_mmq` no longer
            // implies `stream_native == false`.
            stream_native = std::getenv("STRATA_M2_NO_STREAM_NATIVE") == nullptr;
            stream_vram = std::getenv("STRATA_M2_STREAM_VRAM") != nullptr;
            if (use_mmq) {
                const int64_t ne = (int64_t) T * Kp;
                mctx = std::make_unique<MMQ::Context>();
                rcnt = dalloc_t<int32_t>((size_t) g.n_expert);
                rcur = dalloc_t<int32_t>((size_t) g.n_expert);
                rbounds = dalloc_t<int32_t>((size_t) g.n_expert + 1);
                rmax = dalloc_t<int32_t>(1);
                ident = dalloc_t<int32_t>((size_t) ne);
                rinv = dalloc_t<int32_t>((size_t) ne);
                CK(cudaDeviceSynchronize());   // ident's cudaMemset (legacy stream) before the iota on the non-blocking stream
                MMQ::iota(ident, ne, st);
                CK(cudaMalloc(&mq_x, MMQ::q8_bytes(ne, g.n_embd)));
                CK(cudaMalloc(&mq_hq, MMQ::q8_bytes(ne, g.n_ff)));
                mq_gu = dalloc((size_t) ne * 2 * FF);
                mq_h = dalloc((size_t) ne * FF);
            }
            std::fprintf(stderr, "strata-m2: prompt experts on %s\n", use_mmq ? "MMQ (int8)" : "the native grouped kernels");
            {
                std::string gerr;
                int64_t big = 0;
                auto take = [&](const Layer& L) {
                    for (const Mat* w : {&L.qkv, &L.zg, &L.ssm_out, &L.wq, &L.wk, &L.wv, &L.wo, &L.sh_gate, &L.sh_up, &L.sh_down})
                        big = std::max<int64_t>(big, (int64_t) w->n_in * w->n_out);
                };
                for (const Layer& L : m->layers) take(L);
                if (m->has_mtp) {
                    take(m->mtp);
                    big = std::max<int64_t>(big, (int64_t) m->eh_proj.n_in * m->eh_proj.n_out);
                }
                if (!gemm[card].init(st, big, gerr)) {
                    std::fprintf(stderr, "strata-m2: prompt GEMM: %s\n", gerr.c_str());
                    std::exit(1);
                }
            }
            CK(cudaMalloc(&escr, K::native_expert_scratch_bytes((int64_t) (T * Kp), (int64_t) FF)));
            ids = dalloc_t<int32_t>(T * Kp);
            grp_ptr = dalloc_t<unsigned long long>(std::min<size_t>(T * Kp, (size_t) g.n_expert) + 1);
            grp_start = dalloc_t<int32_t>(std::min<size_t>(T * Kp, (size_t) g.n_expert) + 1);
            n_groups = dalloc_t<int32_t>(1);
            ent_dst = dalloc_t<int32_t>(T * Kp);
            ent_tok = dalloc_t<int32_t>(T * Kp);
            tok = dalloc_t<int32_t>(T);
            pos0 = dalloc_t<int32_t>(1);
            pick = dalloc_t<int32_t>(kVerify);
            amv = dalloc((size_t) kArgBlocks * kVerify);
            ami = dalloc_t<int32_t>((size_t) kArgBlocks * kVerify);
            CK(cudaHostAlloc((void**) &hpin, 64 * sizeof(int32_t), cudaHostAllocDefault));
            CK(cudaHostAlloc((void**) &htoks, (size_t) T * 4, cudaHostAllocDefault));
            n_chunk = (ctx + kChunkCells - 1) / kChunkCells;
            att_rows = (int) (g.n_head * std::min<int64_t>((int64_t) T, 64));   // 64 query tokens per attention batch
            pacc = dalloc((size_t) att_rows * n_chunk * g.head_dim);
            pm = dalloc((size_t) att_rows * n_chunk);
            pl = dalloc((size_t) att_rows * n_chunk);
            for (const Layer& L : m->layers) {   // the verify windows' GDN inputs (one set, whichever session runs)
                if (L.attn) {
                    for (auto* v : {&v_qkv, &v_hc, &v_alpha, &v_beta}) v->push_back(nullptr);
                } else {
                    v_qkv.push_back(dalloc((size_t) kVerify * C));
                    v_hc.push_back(dalloc((size_t) kVerify * C));
                    v_alpha.push_back(dalloc((size_t) kVerify * g.ssm_v_heads));
                    v_beta.push_back(dalloc((size_t) kVerify * g.ssm_v_heads));
                }
            }
            if (m->has_mtp) {
                mtp_cat = dalloc(T * 2 * N);
                hmtp = dalloc(T * N);
                logits_m = dalloc((size_t) g.n_vocab);
                tok_m = dalloc_t<int32_t>(T);
                pos0_m = dalloc_t<int32_t>(1);
                pick_m = dalloc_t<int32_t>(1);
            }
            sess.resize((size_t) n_sess);
            gsets.resize((size_t) n_sess);
            {   // the sessions' caches: said before they fail (a slot's K / V at the context, fp32, for the attention layers and
                // the MTP block, and its recurrent state)
                size_t n_attn = m->has_mtp ? 1 : 0, n_gdn = 0;
                for (const Layer& L : m->layers) (L.attn ? n_attn : n_gdn) += 1;
                const size_t esz = kv_esz();   // --kv f16 / int8: the sessions' K / V in half / a quarter of the bytes
                const size_t scale1 = kv_mode == KV_I8 ? (size_t) g.n_head_kv * ctx * kKvBlocks * 4 * 2 : 0;   // their scales (K and V)
                const size_t per = ((size_t) (g.n_head_kv * g.head_dim * 2 * esz) * (size_t) ctx + scale1) * n_attn +
                                   (size_t) g.ssm_v_heads * g.ssm_state_size * g.ssm_state_size * 4 * n_gdn;
                size_t fr0 = 0, tot0 = 0;
                CK(cudaMemGetInfo(&fr0, &tot0));
                if (n_sess > 1)
                    std::fprintf(stderr, "strata-m2: %d sessions (--batch): %.2f GiB of VRAM each at a context of %d (%.2f GiB in "
                                 "all), %.2f GiB free before them\n", n_sess, (double) per / (1 << 30), ctx,
                                 (double) per * n_sess / (1 << 30), (double) fr0 / (1 << 30));
                if (n_sess > 1 && per * (size_t) n_sess + ((size_t) 1 << 30) > fr0) {
                    std::fprintf(stderr, "strata-m2: not enough VRAM for %d slots at a context of %d (%.2f GiB each, %.2f GiB "
                                 "free): use fewer slots (\"parallel\" / --batch) or a shorter --max-context\n", n_sess, ctx,
                                 (double) per / (1 << 30), (double) fr0 / (1 << 30));
                    std::exit(1);
                }
                // one session on one card: its K / V said with the numbers instead of a cudaMalloc failure (setup wrote
                // --max-context 131072 on a 16 GB card: 62 GiB of fp32 K / V).  256 MiB: the hybrid experts' outputs
                // place_experts needs next, so a context refused here would not have started either
                const size_t margin1 = (size_t) 256 << 20;
                if (n_sess == 1 && split_dev < 0 && per + margin1 > fr0) {
                    const size_t room = fr0 > margin1 ? fr0 - margin1 : 0;
                    const long long most = (long long) ((double) ctx * (double) room / (double) per) / 1024 * 1024;
                    std::fprintf(stderr, "strata-m2: not enough VRAM for a context of %d: its K / V cache needs %.2f GiB "
                                 "(%s), %.2f GiB free after the weights and the buffers: use a shorter --max-context "
                                 "(at most about %lld on this card)\n", ctx, (double) per / (1 << 30),
                                 kv_word(), (double) fr0 / (1 << 30), most);
                    std::exit(1);
                }
            }
            // The sessions' KV cache: ONCE.  alloc_session puts each layer's K / V on the card that runs it
            // (the layer split), so the second pass would only allocate a second, unused copy of the whole
            // cache on the second card: 1.94 GiB of the 16 GB card, for nothing.
            if (card == 0) for (Sess& ss : sess) alloc_session(ss);
            if (n_sess > 1) {   // the batch rows' tables: per state index the session pointers, on the device
                const size_t ns = sess[0].conv.size();
                std::vector<float*> hc(ns * (size_t) n_sess), hs(hc);
                std::vector<void*> hk(ns * (size_t) n_sess), hv(ns * (size_t) n_sess);
                for (size_t li = 0; li < ns; ++li)
                    for (int s_ = 0; s_ < n_sess; ++s_) {
                        hc[li * n_sess + s_] = sess[(size_t) s_].conv[li];
                        hs[li * n_sess + s_] = sess[(size_t) s_].S[li];
                        hk[li * n_sess + s_] = sess[(size_t) s_].kc[li];
                        hv[li * n_sess + s_] = sess[(size_t) s_].vc[li];
                    }
                const std::pair<float***, std::vector<float*>*> tabs[2] = {{&d_conv_tab, &hc}, {&d_S_tab, &hs}};
                for (const auto& tb : tabs) {
                    CK(cudaMalloc((void**) tb.first, tb.second->size() * sizeof(float*)));
                    CK(cudaMemcpy((void*) *tb.first, tb.second->data(), tb.second->size() * sizeof(float*),
                                  cudaMemcpyHostToDevice));
                }
                const std::pair<void***, std::vector<void*>*> tabs_kv[2] = {{&d_kc_tab, &hk}, {&d_vc_tab, &hv}};
                for (const auto& tb : tabs_kv) {
                    CK(cudaMalloc((void**) tb.first, tb.second->size() * sizeof(void*)));
                    CK(cudaMemcpy((void*) *tb.first, tb.second->data(), tb.second->size() * sizeof(void*),
                                  cudaMemcpyHostToDevice));
                }
                std::vector<__half*> hks(ns * (size_t) n_sess, nullptr), hvs(hks);
                for (size_t li = 0; li < ns; ++li)
                    for (int s_ = 0; s_ < n_sess; ++s_) {
                        hks[li * n_sess + s_] = sess[(size_t) s_].ksc[li];
                        hvs[li * n_sess + s_] = sess[(size_t) s_].vsc[li];
                    }
                const std::pair<__half***, std::vector<__half*>*> tabs_ks[2] = {{&d_ksc_tab, &hks}, {&d_vsc_tab, &hvs}};
                for (const auto& tb : tabs_ks) {   // --kv int8: the rows' scales, one array per layer and session
                    CK(cudaMalloc((void**) tb.first, tb.second->size() * sizeof(float*)));
                    CK(cudaMemcpy((void*) *tb.first, tb.second->data(), tb.second->size() * sizeof(float*),
                                  cudaMemcpyHostToDevice));
                }
                CK(cudaHostAlloc((void**) &hrow, 128 * sizeof(int32_t), cudaHostAllocDefault));
            }
            if (vision) {
                d_mtab = dalloc_t<int32_t>((size_t) ctx * 3);
                d_map = dalloc_t<int32_t>(T);
                CK(cudaHostAlloc((void**) &h_map, T * sizeof(int32_t), cudaHostAllocDefault));
            }
            cur = 0;
            load_cur();

        };
        CK(cudaSetDevice(0));
        CK(cudaGetLastError());
        alloc_core(0);
        if (split_dev >= 0) {
            save_set(0);
            CK(cudaSetDevice(split_dev));
            CK(cudaGetLastError());
            alloc_core(1);
            CK(cudaSetDevice(0));
            CK(cudaGetLastError());
            save_set(1);
            CK(cudaSetDevice(0));
            CK(cudaGetLastError());
            load_set(0);
        }
        CK(cudaDeviceSynchronize());   // every legacy-stream cudaMemset of the buffers above is done
        size_t fr = 0, tot = 0;
        CK(cudaMemGetInfo(&fr, &tot));
        std::fprintf(stderr, "strata-m2: context %d cells, prompt chunk %d%s, K / V in %s, %.2f GiB of VRAM free\n", ctx, tmax,
                     m->has_mtp ? ", MTP block loaded" : "", kv_word(), (double) fr / (1 << 30));
    }

    /// The recurrent state and the cache to a fresh sequence.
    void reset() {
        const auto& g = m->g;
        // The layer split: the session's state is on the FIRST card, so the memsets run there (a memset of the
        // other card's memory crashes the driver).
        const int cd = cur_set;
        if (split_dev >= 0) {
            use_set(0);
            CK(cudaSetDevice(0));
            CK(cudaGetLastError());
        }
        for (size_t l = 0; l < conv.size(); ++l) {
            if (conv[l]) CK(cudaMemsetAsync(conv[l], 0, (size_t) g.ssm_conv_channels * (g.ssm_d_conv - 1) * 4, st));
            if (S[l]) CK(cudaMemsetAsync(S[l], 0, (size_t) g.ssm_v_heads * g.ssm_state_size * g.ssm_state_size * 4, st));
        }
        if (hlast) CK(cudaMemsetAsync(hlast, 0, (size_t) g.n_embd * 4, st));
        // The attention's K / V too: the warm-up pass (the engine's first, padded forward) leaves its rows in the
        // cache, and the real prompt would read them.
        for (size_t l = 0; l < kc.size(); ++l) {
            if (!kc[l] && !vc[l]) continue;
            const bool on2 = split_dev >= 0 && (int) l >= split_layer;
            if (on2) {
                CK(cudaSetDevice(split_dev));
                CK(cudaGetLastError());
            }
            cudaStream_t ks = on2 ? set[1].st : st;
            const size_t kvb = (size_t) g.n_head_kv * ctx * g.head_dim * kv_esz();
            if (kc[l]) CK(cudaMemsetAsync(kc[l], 0, kvb, ks));   // an INT8 cell's stale scale is harmless beside a zeroed row
            if (vc[l]) CK(cudaMemsetAsync(vc[l], 0, kvb, ks));
            if (on2) {
                CK(cudaSetDevice(0));
                CK(cudaGetLastError());
            }
        }
        CK(cudaStreamSynchronize(st));
        if (split_dev >= 0 && cd != 0) {   // back to the card we came from
            use_set(cd);
            CK(cudaSetDevice(split_dev));
            CK(cudaGetLastError());
        }
        pos = 0;
        sess[(size_t) cur].pos = 0;
        set_delta(0);
    }

    /// y [T][n_out] = W x [T][n_in] on the native MMVQ, 8 columns per launch.
    void mm(const Mat& w, const float* xin, float* out, int T) {
        prep(xin, w.n_in, T, q8);
        apply(w, q8, out, T);
    }
    /// y [T][n_out] = W [n_out][n_in] (f32) x [T][n_in]: a decode batch on the warp-per-row kernel, a prompt chunk on
    /// cuBLAS's FP32 GEMM (the router and the GDN alpha / beta stay in FP32, as llama.cpp keeps them).
    void f32mm(const float* W, const float* xin, float* out, int n_in, int n_out, int T) {
        if (T <= kGemmT) {
            k_gemv_f32<<<dim3(rows8(n_out), T), kThreads, 0, st>>>(W, xin, out, n_in, n_out);
            return;
        }
        static const bool blas_f32 = std::getenv("STRATA_Q36_BLAS_F32") != nullptr;   // the hipBLAS / cuBLAS SGEMM (A/B)
        if (!blas_f32 && n_in % 4 == 0) {
            // K split so that about 128 blocks run (the partials summed in a fixed order: deterministic); TM = 4 rows per
            // thread for the 256-wide router, 1 for the narrow ones (alpha / beta 32, the gate 1)
            const int tm = n_out > 64 ? 4 : 1;
            const int gx = (n_out + 63) / 64, gy = (T + 16 * tm - 1) / (16 * tm);
            const int want = std::max(1, std::min(kSplitK, 128 / std::max(1, gx * gy)));
            const int kz = (n_in / want + 15) / 16 * 16;
            const int splits = (n_in + kz - 1) / kz;
            float* dst = splits > 1 ? sk_part : out;
            if (tm == 4) k_sgemm_nt<4><<<dim3(gx, gy, splits), 256, 0, st>>>(W, xin, dst, n_in, n_out, T, n_out, kz);
            else k_sgemm_nt<1><<<dim3(gx, gy, splits), 256, 0, st>>>(W, xin, dst, n_in, n_out, T, n_out, kz);
            if (splits > 1)
                k_sum_splits<<<grid1((int64_t) T * n_out), kThreads, 0, st>>>(sk_part, out, (int64_t) T * n_out, splits);
            return;
        }
        const float one = 1.0f, zero = 0.0f;
        if (cublasSgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, n_out, T, n_in, &one, W, n_in, xin, n_in, &zero, out, n_out) !=
            CUBLAS_STATUS_SUCCESS) {
            std::fprintf(stderr, "strata-m2: cublasSgemm failed\n");
            std::exit(1);
        }
    }

    /// A batch above kGemmT tokens (a prompt chunk) runs its projections as GEMMs (prefill::Gemm::native: the GGUF
    /// blocks dequantized to FP16, the activations FP16); a decode batch on the native MMVQ (q8_1 activations).
    static constexpr int kGemmT = 8;
    void prep(const float* xin, int n, int T, void* q) {
        if (T > kGemmT) k_to_f16<<<grid1((int64_t) n * T), kThreads, 0, st>>>(xin, x16, (int64_t) n * T);
        else quant(xin, n, T, q);
    }
    void apply(const Mat& w, const void* q, float* out, int T) {
        if (T > kGemmT) gemm[cur_set].native((const uint16_t*) x16, w.type, w.d, out, T, w.n_out, w.n_in);
        else mmq(w, q, out, T);
    }
    /// x [T][n] -> its q8_1 image (T contiguous columns: one vector of T * n), for every matrix that reads x.
    void quant(const float* xin, int n, int T, void* q) { K::native_quantize_q8_1(xin, q, n * T, 1, st); }
    // ---- decode kernels of this engine (q36_kernels.cuh), each the old kernel's bits (STRATA_Q36_VERIFY=1 compares)
    bool fast_k = std::getenv("STRATA_Q36_OLD_KERNELS") == nullptr;
    bool verify_k = std::getenv("STRATA_Q36_VERIFY") != nullptr;
    float* vbuf = nullptr;                       // verify scratch
    long long v_calls = 0, v_bad = 0;
    std::map<std::string, long long> v_by;   // mismatches per check
    void verify_cmp(const char* what, const float* a, const float* b, size_t n, const std::string& tag = "") {
        CK(cudaStreamSynchronize(st));
        std::vector<float> ha(n), hb(n);
        CK(cudaMemcpy(ha.data(), a, n * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(hb.data(), b, n * 4, cudaMemcpyDeviceToHost));
        ++v_calls;
        if (std::memcmp(ha.data(), hb.data(), n * 4) != 0) {
            ++v_bad;
            size_t i = 0;
            while (i < n && std::memcmp(&ha[i], &hb[i], 4) == 0) ++i;
            if (++v_by[what] <= 3)
                std::fprintf(stderr, "strata-m2 VERIFY MISMATCH %s%s: first at %zu of %zu: new %g old %g (bits %08x %08x)\n", what,
                             tag.c_str(), i, n, (double) ha[i], (double) hb[i], *(unsigned*) &ha[i], *(unsigned*) &hb[i]);
        }
    }
    /// A Q8_0 matrix that is not on the packed decode kernel (the shared expert, attention k / v: fewer than 2048
    /// rows) for T <= 8 columns: one warp per row (m2k::k_q8_rows) in place of the 4-warp block of
    /// native_small_mmvq_kernel; the same bits.
    bool q8_small(const Mat& w, const void* q, float* out, int T) {
        if (!fast_k || w.type != 8 || T < 1 || T > 8 || w.n_in % 32 != 0) return false;
        if (K::native_q8_0_packed_eligible(w.n_in, w.n_out) && K::native_q8_0_packed_enabled()) return false;
        const int nb = w.n_in / 32;
        if (nb > 64) return false;
        const void* wp = w.d;
        const half* dpl = nullptr;
        const int NJ = nb <= 32 ? 1 : 2;
#define Q36_Q8(NCV)                                                                                          \
    case NCV:                                                                                                \
        if (NJ == 1) m2k::launch_q8_rows<NCV, 1, false>(wp, dpl, q, out, nb, w.n_out, st);                  \
        else m2k::launch_q8_rows<NCV, 2, false>(wp, dpl, q, out, nb, w.n_out, st);                          \
        break;
        switch (T) { Q36_Q8(1) Q36_Q8(2) Q36_Q8(3) Q36_Q8(4) Q36_Q8(5) Q36_Q8(6) Q36_Q8(7) Q36_Q8(8) }
#undef Q36_Q8
        if (verify_k) {
            if (!vbuf) CK(cudaMalloc((void**) &vbuf, (size_t) 8 * 16384 * 4));
            const size_t qb = K::native_q8_1_bytes(w.n_in, 1);
            for (int c0 = 0; c0 < T; c0 += 8) {
                const int nc = std::min(8, T - c0);
                K::native_mmvq(w.type, w.d, (const char*) q + (size_t) c0 * qb, vbuf + (size_t) c0 * w.n_out, w.n_in, w.n_out,
                               nc, st);
            }
            verify_cmp("q8_small", out, vbuf, (size_t) T * w.n_out,
                       " n_in " + std::to_string(w.n_in) + " n_out " + std::to_string(w.n_out) + " T " + std::to_string(T));
        }
        return true;
    }
    // ---- the q8_1 images produced by the norm kernels themselves (fast_k, T <= kGemmT): xq_ready = the layer's input h is
    // already quantized in xq (k_moe_out_small of the layer before), xq_post = the MoE input h is (k_add_rms_q)
    bool xq_ready = false, xq_post = false;
    void* vq = nullptr;
    void verify_q8(const char* what, const float* hsrc, int T, const std::string& tag) {
        const int N = (int) m->g.n_embd;
        const size_t qb = K::native_q8_1_bytes(N, 1);
        if (!vq) CK(cudaMalloc(&vq, qb * 8));
        K::native_quantize_q8_1(hsrc, vq, N * T, 1, st);
        const long long before = v_bad;
        verify_cmp(what, (const float*) xq, (const float*) vq, qb * (size_t) T / 4, tag);
        if (v_bad != before && std::getenv("STRATA_Q36_VDEBUG") && v_by[what] <= 2) {   // a mismatching block, on the host
            std::vector<uint8_t> a(qb * (size_t) T), b(qb * (size_t) T);
            std::vector<float> hv((size_t) N * T);
            CK(cudaMemcpy(a.data(), xq, a.size(), cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(b.data(), vq, b.size(), cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(hv.data(), hsrc, hv.size() * 4, cudaMemcpyDeviceToHost));
            for (size_t o = 0; o < a.size(); o += 36)
                if (std::memcmp(&a[o], &b[o], 36) != 0) {
                    const size_t blk = o / 36;
                    float v[32], t[32];
                    for (int l = 0; l < 32; ++l) v[l] = t[l] = hv[blk * 32 + (size_t) l];
                    for (int off = 16; off > 0; off >>= 1) {   // the xor butterfly, lane by lane
                        float u[32];
                        for (int l = 0; l < 32; ++l) u[l] = t[l] + t[l ^ off];
                        std::memcpy(t, u, sizeof(u));
                    }
                    unsigned ds_a, ds_b;
                    std::memcpy(&ds_a, &a[o], 4);
                    std::memcpy(&ds_b, &b[o], 4);
                    std::fprintf(stderr, "VDEBUG block %zu: ds new %08x old %08x, qs bytes %s, cpu butterfly sum %.9g (half %04x)\n",
                                 blk, ds_a, ds_b, std::memcmp(&a[o + 4], &b[o + 4], 32) == 0 ? "equal" : "DIFFER", (double) t[0],
                                 (unsigned) __half_as_ushort(__float2half(t[0])));
                    break;
                }
        }
    }
    void prep_in(int T) {
        if (xq_ready && T <= kGemmT) {
            xq_ready = false;
            return;
        }
        xq_ready = false;
        prep(h, (int) m->g.n_embd, T, xq);
    }
    void add_rms_post(const Layer& L, int T) {
        const int N = (int) m->g.n_embd;
        xq_post = false;
        if (fast_k && T <= kGemmT) {
            if (verify_k) {
                if (!vx_) {
                    CK(cudaMalloc((void**) &vx_, (size_t) 8 * 2048 * 4));
                    CK(cudaMalloc((void**) &vh_, (size_t) 8 * 2048 * 4));
                }
                CK(cudaMemcpyAsync(vx_, x, (size_t) T * N * 4, cudaMemcpyDeviceToDevice, st));
                k_add_rms<<<T, kThreads, 0, st>>>(vx_, y, L.post_norm, vh_, N, m->eps);
            }
            m2k::k_add_rms_q<<<T, 256, 0, st>>>(x, y, L.post_norm, h, (m2k::QB*) xq, N, m->eps);
            xq_post = true;
            if (verify_k) {
                verify_cmp("add_rms x", x, vx_, (size_t) T * N, " T " + std::to_string(T));
                verify_cmp("add_rms h", h, vh_, (size_t) T * N, " T " + std::to_string(T));
                verify_q8("add_rms q8", h, T, " T " + std::to_string(T));
            }
            return;
        }
        k_add_rms<<<T, kThreads, 0, st>>>(x, y, L.post_norm, h, N, m->eps);
    }
    /// The shared expert's gate + up + SwiGLU + q8_1 image (into q8) for T <= 4 rows in one launch (m2k::k_sh_gu), else false.
    bool shared_gu(const Layer& L, int T) {
        const int FS = (int) m->g.n_ff_shexp;
        const Mat &a = L.sh_gate, &b = L.sh_up;
        if (!fast_k || T < 1 || T > 4 || a.type != 8 || b.type != 8 || a.n_in != b.n_in || a.n_out != FS || b.n_out != FS ||
            a.n_in % 32 != 0 || a.n_in / 32 > 64 || a.n_in / 32 <= 32 || FS % 32 != 0 ||
            (K::native_q8_0_packed_eligible(a.n_in, a.n_out) && K::native_q8_0_packed_enabled()))
            return false;
        const int nb = a.n_in / 32;
        const dim3 grid((unsigned) (FS / 32));
        const bool pkd = false;
        const uint8_t* wa = a.d;
        const uint8_t* wb = b.d;
        const half* da = nullptr;
        const half* db = nullptr;
#define Q36_SH(NCV)                                                                                          \
    case NCV:                                                                                                \
        if (pkd)                                                                                             \
            m2k::k_sh_gu<NCV, 2, true><<<grid, 1024, 0, st>>>(wa, da, wb, db, (const m2k::QB*) xq,        \
                                                               (m2k::QB*) q8, nb, FS);                      \
        else                                                                                                 \
            m2k::k_sh_gu<NCV, 2, false><<<grid, 1024, 0, st>>>(wa, nullptr, wb, nullptr, (const m2k::QB*) xq, \
                                                                (m2k::QB*) q8, nb, FS);                     \
        break;
        switch (T) { Q36_SH(1) Q36_SH(2) Q36_SH(3) Q36_SH(4) }
#undef Q36_SH
        if (verify_k) {
            if (!vbuf) CK(cudaMalloc((void**) &vbuf, (size_t) 8 * 16384 * 4));
            if (!vq) CK(cudaMalloc(&vq, K::native_q8_1_bytes(m->g.n_embd, 1) * 8));
            float* vg = vbuf;
            float* vu = vbuf + (size_t) 8 * FS;
            mmq_ref(a, xq, vg, T);
            mmq_ref(b, xq, vu, T);
            K::native_swiglu_quantize_q8_1(vg, vu, vq, FS, T, st);
            verify_cmp("shared_gu q8", (const float*) q8, (const float*) vq, K::native_q8_1_bytes(FS, 1) * (size_t) T / 4,
                       " T " + std::to_string(T));
        }
        return true;
    }
    /// The old products (native MMVQ), for the checks.
    void mmq_ref(const Mat& w, const void* q, float* out, int T) {
        const size_t qb = K::native_q8_1_bytes(w.n_in, 1);
        for (int c0 = 0; c0 < T; c0 += 8) {
            const int nc = std::min(8, T - c0);
            K::native_mmvq(w.type, w.d, (const char*) q + (size_t) c0 * qb, out + (size_t) c0 * w.n_out, w.n_in, w.n_out, nc, st);
        }
    }
    void group_old(const Layer& L, const int32_t* slot_of, int T, unsigned long long* o_ptr, int32_t* o_start, int32_t* o_ng,
                   int32_t* o_dst, int32_t* o_tok) {
        const int Kp = (int) m->top_k, n_ent = T * Kp;
        if (slot_of)
            k_group_hy<<<1, kThreads, 0, st>>>(ids, n_ent, Kp, (int) m->g.n_expert, slot_of, L.blobs, L.stride, o_ptr, o_start,
                                               o_ng, o_dst, o_tok);
        else
            k_group<<<1, kThreads, 0, st>>>(ids, n_ent, Kp, (int) m->g.n_expert, L.blobs, L.stride, o_ptr, o_start, o_ng, o_dst,
                                            o_tok);
    }
    /// VERIFY: the old grouping of the current `ids` into scratch, compared with grp_ptr / grp_start / n_groups / ent_dst / ent_tok.
    void group_verify(const Layer& L, const int32_t* slot_of, int T) {
        if (!vgrp) CK(cudaMalloc((void**) &vgrp, 2048));
        unsigned long long* o_ptr = (unsigned long long*) vgrp;
        int32_t* o_start = (int32_t*) (o_ptr + 65);
        int32_t* o_ng = o_start + 66;
        int32_t* o_dst = o_ng + 2;
        int32_t* o_tok = o_dst + 64;
        group_old(L, slot_of, T, o_ptr, o_start, o_ng, o_dst, o_tok);
        CK(cudaStreamSynchronize(st));
        int nn = 0, no = 0;
        CK(cudaMemcpy(&nn, n_groups, 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(&no, o_ng, 4, cudaMemcpyDeviceToHost));
        ++v_calls;
        bool bad = nn != no;
        if (!bad) {
            std::vector<unsigned long long> a((size_t) nn + 1), b((size_t) nn + 1);
            std::vector<int32_t> sa((size_t) nn + 1), sb((size_t) nn + 1), da(64), db(64), ta(64), tb(64);
            CK(cudaMemcpy(a.data(), grp_ptr, (size_t) nn * 8, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(b.data(), o_ptr, (size_t) nn * 8, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(sa.data(), grp_start, ((size_t) nn + 1) * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(sb.data(), o_start, ((size_t) nn + 1) * 4, cudaMemcpyDeviceToHost));
            const int tot = sa[(size_t) nn];
            CK(cudaMemcpy(da.data(), ent_dst, (size_t) tot * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(db.data(), o_dst, (size_t) tot * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(ta.data(), ent_tok, (size_t) tot * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(tb.data(), o_tok, (size_t) tot * 4, cudaMemcpyDeviceToHost));
            bad = std::memcmp(a.data(), b.data(), (size_t) nn * 8) != 0 ||
                  std::memcmp(sa.data(), sb.data(), ((size_t) nn + 1) * 4) != 0 ||
                  std::memcmp(da.data(), db.data(), (size_t) tot * 4) != 0 ||
                  std::memcmp(ta.data(), tb.data(), (size_t) tot * 4) != 0;
        }
        if (bad) {
            ++v_bad;
            if (v_bad <= 20) std::fprintf(stderr, "strata-m2 VERIFY MISMATCH group (T %d, groups %d vs %d)\n", T, nn, no);
        }
    }
    bool grouped_by_route = false;   // route_group already filled the group arrays of this layer
    /// The router's top-k and the grouping of the picks in one launch (T <= 8, 256 experts, 8 picks); the logits are in logits_r.
    /// `slot_of`: the hybrid mode's slot table (null: every expert resident).  Returns false when the old kernels must run.
    bool route_group(const Layer& L, const int32_t* slot_of, int T, int ld) {
        const int Kp = (int) m->top_k;
        grouped_by_route = false;
        // MiniMax-M2 picks 8 experts of 256 (expert_used_count 8); the kernel is instantiated for the two
        // counts the family can carry.  It matters: falling back to native_router_topk would be the SOFTMAX
        // router, and MiniMax-M2's is a sigmoid with a per-expert selection bias - a different choice of
        // experts, and a different answer.
        if (!fast_k || T > 8 || m->g.n_expert != 256 || (Kp != 4 && Kp != 8)) return false;
        // MiniMax-M2's router: the sigmoid gate and the per-expert selection bias (the file's
        // expert_weights_scale, 2.0, rides in the weights).
        const float scale = m->expert_weights_scale;
        if (Kp == 8) {
            if (slot_of)
                m2k::k_route_group_m2<true, 8><<<1, dim3(32, 8), 0, st>>>(logits_r, L.router_bias, ids, ewts, T, ld, scale,
                                                                               slot_of, L.blobs, L.stride, grp_ptr, grp_start, n_groups,
                                                                               ent_dst, ent_tok);
            else
                m2k::k_route_group_m2<false, 8><<<1, dim3(32, 8), 0, st>>>(logits_r, L.router_bias, ids, ewts, T, ld, scale,
                                                                                nullptr, L.blobs, L.stride, grp_ptr, grp_start, n_groups,
                                                                                ent_dst, ent_tok);
        } else {
            if (slot_of)
                m2k::k_route_group_m2<true, 4><<<1, dim3(32, 8), 0, st>>>(logits_r, L.router_bias, ids, ewts, T, ld, scale,
                                                                           slot_of, L.blobs, L.stride, grp_ptr, grp_start, n_groups,
                                                                           ent_dst, ent_tok);
            else
                m2k::k_route_group_m2<false, 4><<<1, dim3(32, 8), 0, st>>>(logits_r, L.router_bias, ids, ewts, T, ld, scale,
                                                                            nullptr, L.blobs, L.stride, grp_ptr, grp_start, n_groups,
                                                                            ent_dst, ent_tok);
        }
        grouped_by_route = true;
        // the verify arm compares against native_router_topk, which is the SOFTMAX router: MiniMax-M2's is a
        // sigmoid with a selection bias, so the two are not comparable and the arm stays off here.
        return true;
    }
    /// The routed entries of T rows grouped per expert (k_group; with slot_of the hybrid mode's k_group_hy).
    void group_entries(const Layer& L, const int32_t* slot_of, int T) {
        if (grouped_by_route) {   // route_group did it
            grouped_by_route = false;
            return;
        }
        const int Kp = (int) m->top_k, n_ent = T * Kp;
        if (fast_k && n_ent <= 64) {
            if (slot_of)
                m2k::k_group_small<true><<<1, 64, 0, st>>>(ids, n_ent, Kp, slot_of, L.blobs, L.stride, grp_ptr, grp_start,
                                                            n_groups, ent_dst, ent_tok);
            else
                m2k::k_group_small<false><<<1, 64, 0, st>>>(ids, n_ent, Kp, nullptr, L.blobs, L.stride, grp_ptr, grp_start,
                                                             n_groups, ent_dst, ent_tok);
            if (verify_k) group_verify(L, slot_of, T);
            return;
        }
        group_old(L, slot_of, T, grp_ptr, grp_start, n_groups, ent_dst, ent_tok);
    }
    void* vgrp = nullptr;
    float *vx_ = nullptr, *vh_ = nullptr;
    /// The MoE block's end (k_moe_out) for T rows.
    void moe_end(int T, const int32_t* prow, const float* shared, const float* sgate, const float* wn, float* hout, int sgs = 1) {
        const int N = (int) m->g.n_embd, Kp = (int) m->top_k;
        const bool small = fast_k && T <= 8 && N <= 2048 && N % 256 == 0 && Kp <= 15;
        if (small && verify_k) {
            if (!vx_) {
                CK(cudaMalloc((void**) &vx_, (size_t) 8 * 2048 * 4));
                CK(cudaMalloc((void**) &vh_, (size_t) 8 * 2048 * 4));
            }
            CK(cudaMemcpyAsync(vx_, x, (size_t) T * N * 4, cudaMemcpyDeviceToDevice, st));
            k_moe_out<<<T, kThreads, 0, st>>>(vx_, eparts, prow, ewts, Kp, shared, sgate, sgs, wn, vh_, N, m->eps);
        }
        if (small) {
            const bool mkq = wn != nullptr && hout == h;   // the next layer's input, quantized here
            m2k::k_moe_out_small<false><<<T, 1024, 0, st>>>(x, eparts, prow, ewts, Kp, shared, sgate, sgs, wn, hout, N, m->eps,
                                                             mkq ? (m2k::QB*) xq : nullptr);
            xq_ready = mkq;
            if (verify_k) {
                if (mkq) verify_q8("moe_out q8", hout, T, " T " + std::to_string(T));
                verify_cmp("moe_out x", x, vx_, (size_t) T * N, " T " + std::to_string(T));
                if (wn) verify_cmp("moe_out h", hout, vh_, (size_t) T * N, " T " + std::to_string(T));
            }
            return;
        }
        xq_ready = false;
        k_moe_out<<<T, kThreads, 0, st>>>(x, eparts, prow, ewts, Kp, shared, sgate, sgs, wn, hout, N, m->eps);
    }
    /// y [T][n_out] = W x from x's q8_1 image, 8 columns per launch.
    void mmq(const Mat& w, const void* q, float* out, int T) {
        if (q8_small(w, q, out, T)) return;
        const size_t qb = K::native_q8_1_bytes(w.n_in, 1);
        for (int c0 = 0; c0 < T; c0 += 8) {
            const int nc = std::min(8, T - c0);
            K::native_mmvq(w.type, w.d, (const char*) q + (size_t) c0 * qb, out + (size_t) c0 * w.n_out, w.n_in,
                           w.n_out, nc, st);
        }
    }

    /// One layer for T rows: x (the residual) += mixer(h); then MoE; x += moe; `hout` = rms(x) * wn (the next norm).
    /// `li` indexes the layer's state; positions *dpos .. + T - 1 (device), p0 the same on the host (eager grids);
    /// `verify`: a verify window - the GDN state is read, not written (commit() writes it for the kept rows).
    void layer(Layer& L, size_t li, int T, int64_t p0, const int32_t* dpos, bool graph_mode, bool verify,
               const float* wn, float* hout) {
        const auto& g = m->g;
        const int N = (int) g.n_embd, Kp = (int) m->top_k, FF = (int) g.n_ff, FS = (int) g.n_ff_shexp;
        const int HK = (int) g.ssm_k_heads, HV = (int) g.ssm_v_heads, D = (int) g.ssm_state_size;
        const int C = (int) g.ssm_conv_channels;
        const int H = (int) g.n_head, HKV = (int) g.n_head_kv, HD = (int) g.head_dim;
        if (!L.attn) {
            float* qkv_b = verify ? v_qkv[li] : qkv;
            float* hc_b = verify ? v_hc[li] : hc;
            float* al_b = verify ? v_alpha[li] : alpha;
            float* be_b = verify ? v_beta[li] : beta;
            prep_in(T);
            apply(L.qkv, xq, qkv_b, T);
            apply(L.zg, xq, z, T);
            if (T <= kGemmT) {
                k_gdn_ab<<<dim3(rows8(2 * HV), T), kThreads, 0, st>>>(L.alpha, L.beta, h, L.dt, L.ssm_a, al_b, be_b, N,
                                                                      HV);
            } else {
                f32mm(L.alpha, h, al_b, N, HV, T);
                f32mm(L.beta, h, be_b, N, HV, T);
                k_gdn_gb<<<grid1((int64_t) HV * T), kThreads, 0, st>>>(al_b, L.dt, L.ssm_a, be_b, HV, T);
            }
            if (rows_mode) {   // batch rows: a slot's rows (one, or a verify window's two) in sequence on its own state
                k_conv_seq<<<dim3(C / 128, rows_G), 128, 0, st>>>(nullptr, qkv_b, L.conv, hc_b, C, 2 * HK, m->eps, 1,
                                                                  !verify, d_conv_tab + li * n_sess, d_gslot, d_grow0, d_gcnt);
                k_gdn_seq<<<dim3(HV, rows_G), dim3(128, 4), 0, st>>>(nullptr, hc_b, C, al_b, be_b, z, L.ssm_norm, m->eps,
                                                                     gdn_y, HK, HV, 1, !verify, d_S_tab + li * n_sess,
                                                                     d_gslot, d_grow0, d_gcnt);
            } else {
                // one token, a window or a chunk: the same kernels (every token in one launch each), so a decode step and a
                // verify window compute bitwise the same rows and the MTP's greedy text is the plain greedy text.  (Strata's
                // fused single-token kernels are not used here: fused_gdn_conv_l2's 4-tap sum contracts into FMAs
                // differently from k_conv_seq's, a 1-ulp difference in h that the Q8_1 activation rounding can turn into a
                // flipped near-tie later; with them the KL is the same, 1.05x / 0.81x the floor, but MTP would not be exact.)
                // a prompt chunk: tiles of kConvTile tokens side by side (tile mode); else one block per head walks the tokens
                const int tiles = (fast_k && T > kGemmT && !verify) ? (T + kConvTile - 1) / kConvTile : 1;
                k_conv_seq<<<dim3(C / 128, tiles), 128, 0, st>>>(conv[li], qkv_b, L.conv, hc_b, C, 2 * HK, m->eps, T, !verify);
                if (T > kGemmT && !verify) {   // a prompt chunk: the column-split delta rule, then the norm
                    k_gdn_seq_cols<<<dim3(HV, kGdnColGroups), dim3(128 / kGdnColGroups, 16), 0, st>>>(S[li], hc_b, C, al_b,
                                                                                                      be_b, gdn_y, HK, HV, T);
                    k_gdn_norm<<<dim3(HV, T), 128, 0, st>>>(gdn_y, z, L.ssm_norm, m->eps, gdn_y, HV);
                } else {   // decode, verify windows and commits: one kernel, so a window's rows are a decode step's rows
                    k_gdn_seq<<<HV, dim3(128, 4), 0, st>>>(S[li], hc_b, C, al_b, be_b, z, L.ssm_norm, m->eps, gdn_y, HK, HV,
                                                           T, !verify);
                }
            }
            mm(L.ssm_out, gdn_y, y, T);
        } else {
            prep_in(T);
            apply(L.wq, xq, qg, T);
            apply(L.wk, xq, kx, T);
            apply(L.wv, xq, vx, T);
            RowSel rs;
            if (rows_mode) {
                rs.cell = d_rcell;
                rs.slot = d_rslot;
                rs.delta = d_rdelta;
                rs.kc = d_kc_tab + li * n_sess;
                rs.vc = d_vc_tab + li * n_sess;
                if (kv_mode == KV_I8) {   // the rows' scales (INT8 only: the tables are never built otherwise)
                    rs.ks = d_ksc_tab + li * n_sess;
                    rs.vs = d_vsc_tab + li * n_sess;
                }
            }
            if (split_dev >= 0 && (int) li >= split_layer && getenv("STRATA_M2_PEER_DEBUG")) {
                static bool once = false;
                if (true) {
                    once = true;
                    std::fprintf(stderr,
                                 "strata-m2: ATTN layer %d q_norm=%p(d%d) k_norm=%p(d%d) kc=%p(d%d) vc=%p(d%d) dpos=%p(d%d) "
                                 "ddelta=%p(d%d) qg=%p(d%d) kx=%p(d%d) vx=%p(d%d) q=%p(d%d)\n",
                                 (int) li,
                                 (const void*) L.q_norm, dev_of(L.q_norm), (const void*) L.k_norm, dev_of(L.k_norm),
                                 (const void*) kc[li], dev_of(kc[li]), (const void*) vc[li], dev_of(vc[li]),
                                 (const void*) dpos, dev_of(dpos), (const void*) d_delta, dev_of(d_delta),
                                 (const void*) qg, dev_of(qg), (const void*) kx, dev_of(kx), (const void*) vx,
                                 dev_of(vx), (const void*) q, dev_of(q));
                }
            }
            kv_visit([&](auto pkv) {
                using KV = std::remove_pointer_t<decltype(pkv)>;
                k_attn_prep_m2<KV><<<T, kThreads, kv_mode == KV_I8 ? 2 * HKV * HD * 4 : 0, st>>>(
                    qg, kx, vx, L.q_norm, L.k_norm, q, rows_mode ? nullptr : (KV*) kc[li],
                    rows_mode ? nullptr : (KV*) vc[li], rows_mode ? nullptr : ksc[li], rows_mode ? nullptr : vsc[li],
                    H, HKV, HD, m->n_rot, dpos, ctx, m->rope_base, m->eps, d_delta, mtab_now, rs);
            });
            if (T > kGemmT && att_gemm[cur_set]) {
                // a prompt chunk: scores = Q K^T and out = P V as FP16 GEMMs (FP32 accumulation, FP32 scores) per KV
                // head (its query heads one strided batch over the shared K / V), the causal softmax in between writing
                // FP16 probabilities; the queries in slices whose scores fit
                const int P = (int) p0 + T;
                const int ldp = (P + 7) / 8 * 8;
                const int tq = (int) std::max<int64_t>(1, std::min<int64_t>(T, (int64_t) (kAttScoreFloats / ((size_t) H * ldp))));
                const int G = H / HKV;
                const float one = 1.0f, zero = 0.0f;
                kv_visit([&](auto pkv) {
                    using KV = std::remove_pointer_t<decltype(pkv)>;
                    k_kv_f16<KV><<<grid1((int64_t) HKV * P * HD), kThreads, 0, st>>>(
                        (const KV*) kc[li], (const KV*) vc[li], ksc[li], vsc[li], k16, v16, ctx, P, HD, HKV);
                });
                k_to_f16<<<grid1((int64_t) T * H * HD), kThreads, 0, st>>>(q, q16, (int64_t) T * H * HD);
                for (int t0 = 0; t0 < T; t0 += tq) {
                    const int nt = std::min(tq, T - t0);
                    for (int kvh = 0; kvh < HKV; ++kvh) {
                        const __half* Kh = k16 + (size_t) kvh * P * HD;
                        const __half* Qh = q16 + ((size_t) t0 * H + kvh * G) * HD;
                        float* Sh = att_s + (size_t) kvh * G * nt * ldp;
                        if (cublasGemmStridedBatchedEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, P, nt, HD, &one, Kh, CUDA_R_16F, HD,
                                                       0, Qh, CUDA_R_16F, H * HD, HD, &zero, Sh, CUDA_R_32F, ldp,
                                                       (long long) nt * ldp, G, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT) !=
                            CUBLAS_STATUS_SUCCESS) {
                            std::fprintf(stderr, "strata-m2: attention scores GEMM failed\n");
                            std::exit(1);
                        }
                    }
                    k_attn_softmax16<<<dim3(nt, H), kThreads, 0, st>>>(att_s, att_p16, ldp, nt, P, (int) p0 + t0,
                                                                       1.0f / sqrtf((float) HD));
                    for (int kvh = 0; kvh < HKV; ++kvh) {
                        const __half* Vh = v16 + (size_t) kvh * P * HD;
                        const __half* Sh = att_p16 + (size_t) kvh * G * nt * ldp;
                        float* Oh = att + ((size_t) t0 * H + kvh * G) * HD;
                        if (cublasGemmStridedBatchedEx(blas, CUBLAS_OP_N, CUBLAS_OP_N, HD, nt, P, &one, Vh, CUDA_R_16F, HD,
                                                       0, Sh, CUDA_R_16F, ldp, (long long) nt * ldp, &zero, Oh, CUDA_R_32F,
                                                       H * HD, HD, G, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT) !=
                            CUBLAS_STATUS_SUCCESS) {
                            std::fprintf(stderr, "strata-m2: attention values GEMM failed\n");
                            std::exit(1);
                        }
                    }
                }
                // MiniMax-M2 has no attention output gate: nothing multiplies the attention output.
            } else {
                // split-K over 256-cell chunks; the chunk grid covers every cell a row can see (graph mode: the whole
                // context, past-the-end chunks return at once)
                const int rows_per = att_rows / H;
                for (int t0 = 0; t0 < T; t0 += rows_per) {
                    const int nt = std::min(rows_per, T - t0);
                    const int nc = graph_mode ? n_chunk
                                              : (int) std::min<int64_t>(n_chunk, (p0 + t0 + nt + kChunkCells - 1) / kChunkCells);
                    kv_visit([&](auto pkv) {
                        using KV = std::remove_pointer_t<decltype(pkv)>;
                        k_attn_part<kHD, KV><<<dim3(nc, nt * H), kChunkCells, 0, st>>>(
                            q, rows_mode ? nullptr : (const KV*) kc[li], rows_mode ? nullptr : (const KV*) vc[li],
                            rows_mode ? nullptr : ksc[li], rows_mode ? nullptr : vsc[li], dpos, t0, H, HKV, ctx,
                            1.0f / sqrtf((float) HD), n_chunk, pacc, pm, pl, rs);
                    });
                    k_attn_merge<kHD><<<nt * H, HD, 0, st>>>(pacc, pm, pl, dpos, t0, H, n_chunk, att, rs);
                }
            }
            mm(L.wo, att, y, T);
        }
        add_rms_post(L, T);
        // routed experts (one token: the router and the shared expert's gate in one product, the gate its last row)
        const bool one = T == 1;
        // a decode batch of up to 8 rows takes the router and the shared expert's gate in one product too (rows of 257) when the fused
        // router follows; the prompt chunks and the checks keep the separate products
        // MiniMax-M2 has no shared expert, so the router is the plain one: no gate row appended, no ext arm.
        const bool ext = false;
        f32mm(L.router, h, logits_r, N, (int) g.n_expert, T);
        // the hybrid rows path: a batch of up to hybrid_t rows; when only the MTP block is hybrid, its every batch
        // (its prompt pass too: no staging buffers then)
        const bool hy_rows = lhy(li) && (T <= hybrid_t || !main_hybrid);
        const bool mmq_rows = !hy_rows && T > kGemmT && use_mmq;
        const bool native_stream_rows = !hy_rows && T > kGemmT && stream_native;
        // Native streaming: handled in the else branch below (k_group uses the streamed buffer directly).
        (void) native_stream_rows;
        // The routing: MiniMax-M2's SIGMOID gate with its per-expert selection bias.  The fast path (the
        // decode's eight tokens at a time) groups the picks as it goes; a prompt of any length goes through
        // the generic kernel, which does the same arithmetic.  The softmax router would pick OTHER experts.
        if (!(!mmq_rows && route_group(L, hy_rows ? hy[li].slot_of_d : nullptr, T, ext ? (int) g.n_expert + 1 : (int) g.n_expert)))
            K::native_router_topk_sigmoid(logits_r, L.router_bias, ids, ewts, (int) g.n_expert, Kp, T,
                                         m->expert_weights_scale, st);
        const int64_t n_ent = (int64_t) T * Kp;
        {   // DIAG: the layer's state (the env var names the layer)
            const char* lay = std::getenv("STRATA_M2_DUMP_STATE");
            if (lay && (int) li == std::atoi(lay)) {
                static std::vector<float> v;
                v.resize((size_t) N);
                const char* names[3] = {"h", "moe_out", "x"};
                const float* srcs[3] = {h, moe_out, x};
                {   // DIAG: the routing's weights and logits (a NaN here poisons the combine)
                    static std::vector<float> w, lg;
                    w.resize((size_t) n_ent);
                    lg.resize((size_t) n_ent);
                    CK(cudaMemcpyAsync(w.data(), ewts, (size_t) n_ent * 4, cudaMemcpyDeviceToHost, st));
                    CK(cudaMemcpyAsync(lg.data(), logits_r, (size_t) n_ent * 4, cudaMemcpyDeviceToHost, st));
                    CK(cudaStreamSynchronize(st));
                    const int e = 12544;   // the first entry of the first bad row (1568)
                    std::fprintf(stderr, "ROUTW layer %d T=%d entry %d (row %d) ewts :", (int) li, T, e, e / (int) m->top_k);
                    std::fprintf(stderr, " %.9g", w[(size_t) e]);
                    std::fprintf(stderr, "  (NaN=%d) | logits :", (int) std::isnan(w[(size_t) e]));
                    std::fprintf(stderr, " %.9g (NaN=%d)\n", lg[(size_t) e], (int) std::isnan(lg[(size_t) e]));
                    std::fprintf(stderr, "\n");
                    std::fflush(stderr);
                }
                for (int k = 0; k < 3; ++k) {
                    CK(cudaMemcpyAsync(v.data(), srcs[k], (size_t) N * 4, cudaMemcpyDeviceToHost, st));
                    CK(cudaStreamSynchronize(st));
                    std::fprintf(stderr, "STATE layer %d T=%d %s :", (int) li, T, names[k]);
                    for (int i = 0; i < 4; ++i) std::fprintf(stderr, " %.9g", v[(size_t) i]);
                    std::fprintf(stderr, "\n");
                    std::fflush(stderr);
                }
            }
        }
        {   // DIAG: the routed ids of a layer (the env var names it)
            const char* lay = std::getenv("STRATA_M2_DUMP_ROUTE");
            if (lay && (int) li == std::atoi(lay)) {
                static std::vector<int32_t> h;
                h.resize((size_t) n_ent);
                CK(cudaMemcpyAsync(h.data(), ids, (size_t) n_ent * 4, cudaMemcpyDeviceToHost, st));
                CK(cudaStreamSynchronize(st));
                std::fprintf(stderr, "ROUTE layer %d T=%d :", (int) li, T);
                for (int i = 0; i < n_ent && i < 16; ++i) std::fprintf(stderr, " %d", h[(size_t) i]);
                std::fprintf(stderr, "\n");
                std::fflush(stderr);
            }
        }
        const int32_t* prow = nullptr;
        if (hy_rows) {
            moe_hybrid_gpu(L, li, T);
        } else if (T > kGemmT && use_mmq && !lhy(li)) {
            // a prompt chunk: the experts on llama.cpp's MMQ (int8, the weights read once per expert): the entries
            // sorted by expert on the device, gate+up as one [2 FF] product per expert, SwiGLU, down into rows the
            // combine reads through entry -> sorted row
            //
            // Only a layer whose experts all sit in one block may take it: MMQ steps over them from ONE base pointer
            // with `L.stride` (`Product.w` / `expert_bytes`), which is `L.blobs` only when every expert has a slot.  A
            // hybrid layer streams just the experts WITHOUT a slot into the staging and leaves the residents' places
            // as they were, so MMQ read the wrong bytes for those experts: their rows came out as garbage, the
            // prompt's logits went to NaN, `k_argmax` left its INT32_MAX sentinel in `pick`, and the next decode step
            // fed that id to the embedding lookup -- `table + 2^31 * row_bytes`, which the card refuses with
            // HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION inside embed_rows_kernel.  Hybrid layers take the streamed
            // native path instead: it reads a resident expert from its blob and the pool's from the staging, and on
            // this machine (RX 9070 + RX 6800, IQ2_XXS) it also reads the prompt faster than MMQ at an equal
            // placement (512 tokens: 10.5 tok/s against 8.8, `--expert-vram 0`).
            const int ne = (int) g.n_expert;
            CK(cudaMemsetAsync(rcnt, 0, (size_t) ne * 4, st));
            k_route_count<<<grid1(n_ent), kThreads, 0, st>>>(ids, (int) n_ent, rcnt);
            k_route_scan<<<1, 32, 0, st>>>(rcnt, ne, rbounds, rcur, rmax);
            k_route_scatter<<<grid1(n_ent), kThreads, 0, st>>>(ids, (int) n_ent, Kp, rcur, ent_dst, ent_tok, rinv);
            CK(cudaMemcpyAsync(hpin + 10, rmax, 4, cudaMemcpyDeviceToHost, st));
            if (lhy(li)) CK(cudaMemcpyAsync(h_cnt + li * g.n_expert, rcnt, (size_t) ne * 4, cudaMemcpyDeviceToHost, st));
            CK(cudaStreamSynchronize(st));
            if (lhy(li))
                for (int x = 0; x < ne; ++x) hy[li].score[(size_t) x] += h_cnt[li * g.n_expert + x];
            const int64_t maxr = hpin[10];
            const uint8_t* wbase = lhy(li) ? stream_take((int) li) : L.blobs;
            MMQ::quantize(h, ent_tok, mq_x, L.L.gu_type, N, N, n_ent, st);
            MMQ::Product gu;
            gu.w = wbase; gu.type = L.L.gu_type; gu.w_rows = 2 * FF; gu.w_cols = N; gu.expert_bytes = L.stride;
            gu.n = ne; gu.xq = mq_x; gu.bounds = rbounds; gu.ids = ident; gu.total_rows = n_ent; gu.max_rows = maxr;
            gu.dst = mq_gu; gu.ld_dst = 2 * FF;
            mctx->run(gu, st);
            MMQ::swiglu(mq_gu, mq_h, n_ent, FF, false, st);
            MMQ::quantize(mq_h, nullptr, mq_hq, L.L.d_type, FF, FF, n_ent, st);
            MMQ::Product dn;
            dn.w = wbase + L.L.down_off; dn.type = L.L.d_type; dn.w_rows = N; dn.w_cols = FF;
            dn.expert_bytes = L.stride; dn.n = ne; dn.xq = mq_hq; dn.bounds = rbounds; dn.ids = ident;
            dn.total_rows = n_ent; dn.max_rows = maxr; dn.dst = eparts; dn.ld_dst = N;
            mctx->run(dn, st);
            if (lhy(li)) stream_done((int) li);
            prow = rinv;
        } else {
            if (native_stream_rows) {
                const int ne = (int) g.n_expert;
                const uint8_t* wb = stream_take((int) li);
                if (stream_vram)   // every expert of the layer sits in the streamed buffer
                    k_group<<<1, kThreads, 0, st>>>(ids, n_ent, Kp, ne, wb, L.stride, grp_ptr, grp_start, n_groups,
                                                    ent_dst, ent_tok);
                else   // the slots from their VRAM blob, the pool's experts from the pinned host buffer
                    k_group_pool<<<1, kThreads, 0, st>>>(ids, n_ent, Kp, ne, hy[li].slot_of_d, L.blobs, L.stride, wb,
                                                         grp_ptr, grp_start, n_groups, ent_dst, ent_tok);
                if (!xq_post) quant(h, N, T, xq);
                K::native_expert_grouped(L.L, grp_ptr, grp_start, n_groups, ent_dst, ent_tok,
                                         std::min<int64_t>(n_ent, g.n_expert), n_ent, xq, escr, eparts, st);
                if (lhy(li)) stream_done((int) li);
            } else {
                group_entries(L, nullptr, T);
                if (!xq_post) quant(h, N, T, xq);
                K::native_expert_grouped(L.L, grp_ptr, grp_start, n_groups, ent_dst, ent_tok,
                                         std::min<int64_t>(n_ent, g.n_expert), n_ent, xq, escr, eparts, st);
            }
        }
        // MiniMax-M2 has NO shared expert: nothing beside the routed experts' parts.
        if (hy_rows) {
            moe_hybrid_peer(li, T);          // the second card's share (its rows go into the host rows)
            moe_hybrid_cpu(li, T);           // the CPU pool's experts (while the GPU runs the rest), merged
        }
        {   // DIAG: a bad entry's rows (the NaN lives in the rows from 1568 on)
            const char* lay = std::getenv("STRATA_M2_DUMP_STATE");
            if (lay && (int) li == std::atoi(lay)) {
                const int e = 1568 * (int) m->top_k;   // the first entry of the first bad row
                static std::vector<float> g, c;
                g.resize((size_t) N); c.resize((size_t) N);
                CK(cudaMemcpyAsync(g.data(), eparts + (size_t) e * N, (size_t) N * 4, cudaMemcpyDeviceToHost, st));
                CK(cudaMemcpyAsync(c.data(), cpu_out + (size_t) e * N, (size_t) N * 4, cudaMemcpyDeviceToHost, st));
                CK(cudaStreamSynchronize(st));
                std::fprintf(stderr, "BADROW layer %d entry %d (row %d): GPU eparts first 4:", (int) li, e, e / (int) m->top_k);
                for (int i = 0; i < 4; ++i) std::fprintf(stderr, " %.9g", g[(size_t) i]);
                std::fprintf(stderr, "  (NaN=%d)\n", (int) std::isnan(g[0]));
                std::fprintf(stderr, "BADROW layer %d entry %d (row %d): pool  cpu_out first 4:", (int) li, e, e / (int) m->top_k);
                for (int i = 0; i < 4; ++i) std::fprintf(stderr, " %.9g", c[(size_t) i]);
                std::fprintf(stderr, "  (NaN=%d)  d_miss=%d  id=%d\n", (int) std::isnan(c[0]), h_miss[e], h_ids[e]);
                std::fprintf(stderr, "WN layer %d first 4:", (int) li);
                if (wn) for (int i = 0; i < 4; ++i) std::fprintf(stderr, " %.9g", wn[(size_t) i]);
                std::fprintf(stderr, "  (NaN=%d)  eps=%.3g\n", (int)(wn && std::isnan(wn[0])), m->eps);
                std::fflush(stderr);
            }
        }
        // the layer split: the crossing's D2H copies MUST happen BEFORE the combine (moe_end), which adds
        // the MoE to x and overwrites h: otherwise the D2H copy reads the residual (0) and writes 0
        // the combine, the residual add and the next norm in one launch
        moe_end(T, prow, sy, sgate, wn, hout, 1);
        {   // DIAG: what the combine wrote (the NaN must be in its inputs or in its write)
            const char* lay = std::getenv("STRATA_M2_DUMP_STATE");
            if (lay && (int) li == std::atoi(lay)) {
                static std::vector<float> vx, vh;
                vx.resize((size_t) N); vh.resize((size_t) N);
                CK(cudaMemcpyAsync(vx.data(), x, (size_t) N * 4, cudaMemcpyDeviceToHost, st));
                CK(cudaMemcpyAsync(vh.data(), hout, (size_t) N * 4, cudaMemcpyDeviceToHost, st));
                CK(cudaStreamSynchronize(st));
                std::fprintf(stderr, "AFTERCOMB layer %d T=%d x first 4:", (int) li, T);
                for (int i = 0; i < 4; ++i) std::fprintf(stderr, " %.9g", vx[(size_t) i]);
                std::fprintf(stderr, " (NaN=%d)\n", (int) std::isnan(vx[0]));
                std::fprintf(stderr, "AFTERCOMB layer %d T=%d hout first 4:", (int) li, T);
                for (int i = 0; i < 4; ++i) std::fprintf(stderr, " %.9g", vh[(size_t) i]);
                std::fprintf(stderr, " (NaN=%d)\n", (int) std::isnan(vh[0]));
                std::fflush(stderr);
            }
        }
    }

    /// The main model over T rows (device `tok`, positions p0 .. p0 + T - 1, also at *pos0): hmain gets every row's
    /// final normed hidden; `heads` rows from the end get logits (rows [T - heads, T): logits row r - (T - heads)) and
    /// their greedy picks (`pick`).  `verify`: a verify window (the GDN state left for commit()).
    void forward(int T, int64_t p0, int heads, bool graph_mode, bool verify) {
        const auto& g = m->g;
        const int N = (int) g.n_embd;
        if (main_hybrid && T > hybrid_t && (use_mmq || stream_native)) {
            std::vector<int> seq;
            for (size_t l = 0; l < m->layers.size(); ++l)
                if (lhy(l)) seq.push_back((int) l);
            stream_begin(seq);
        }
        // the layer split: the embedding and the first layers are on the first card, so the run starts there.
        // The head of the previous call may have left the second card current, so the device is set
        // unconditionally (not only when the set changes).
        if (split_dev >= 0) {
            if (cur_set != 0) use_set(0);
            CK(cudaSetDevice(0));
            CK(cudaGetLastError());
        }
        K::iq_embed_rows(m->embd_type, m->embd, m->embd_row, tok, T, N, x, st);
        if (map_on) k_put_rows<<<T, kThreads, 0, st>>>(x, d_img, d_map, N);   // a picture's cells take its rows
        k_rms<<<T, kThreads, 0, st>>>(x, m->layers[0].attn_norm, h, N, m->eps);   // later layers: from k_moe_out
        for (size_t l = 0; l < m->layers.size(); ++l) {
            if (split_dev >= 0 && (int) l == split_layer) {
                // The layer split: the residual (x), the normed hidden (h), the tokens and the position cross
                // to the second card, which takes over the buffers and the stream.
                const size_t nx = (size_t) T * N;
                if (split_copy_peer) {
                    CK(cudaMemcpyPeerAsync(set[1].x, split_dev, x, 0, nx * 4, st));
                    CK(cudaMemcpyPeerAsync(set[1].h, split_dev, h, 0, nx * 4, st));
                    CK(cudaMemcpyPeerAsync(set[1].tok, split_dev, tok, 0, (size_t) T * 4, st));
                    CK(cudaMemcpyPeerAsync(set[1].pos0, split_dev, pos0, 0, 4, st));
                } else {
                    // through the host: x and h are copied HERE, before use_set(1) switches to card 1's
                    // buffers (the D2H must read card 0's x and h, which are still current)
                    CK(cudaMemcpyAsync(split_stage, x, nx * 4, cudaMemcpyDeviceToHost, st));
                    CK(cudaMemcpyAsync(split_stage + nx, h, nx * 4, cudaMemcpyDeviceToHost, st));
                    CK(cudaMemcpyAsync((int32_t*) (split_stage + 2 * nx), tok, (size_t) T * 4, cudaMemcpyDeviceToHost, st));
                    CK(cudaMemcpyAsync((int32_t*) (split_stage + 2 * nx) + T, pos0, 4, cudaMemcpyDeviceToHost, st));
                }
                CK(cudaEventRecord(ev_cross, st));
                if (getenv("STRATA_M2_DUMP_ROWS")) {   // DIAG: what arrives on the second card
                    std::vector<uint8_t> a((size_t) T * N * 4), b(a.size());
                    std::vector<int32_t> pa(4), pb(4);
                    CK(cudaMemcpyAsync(a.data(), x, a.size(), cudaMemcpyDeviceToHost, st));
                    CK(cudaMemcpyAsync(b.data(), set[1].x, b.size(), cudaMemcpyDeviceToHost, st));
                    CK(cudaMemcpyAsync(pa.data(), tok, 16, cudaMemcpyDeviceToHost, st));
                    CK(cudaMemcpyAsync(pb.data(), set[1].tok, 16, cudaMemcpyDeviceToHost, st));
                    CK(cudaStreamSynchronize(st));
                    long nd = 0, fi = -1;
                    for (size_t z = 0; z < a.size(); ++z)
                        if (a[z] != b[z]) { if (fi < 0) fi = (long) z; ++nd; }
                    std::fprintf(stderr, "CROSS layer %d T=%d : x %ld differing bytes (first %ld of %zu) | tok %d,%d vs %d,%d\n",
                                 (int) l, T, nd, fi, a.size(), pa[0], pa[1], pb[0], pb[1]);
                    std::fflush(stderr);
                }
                use_set(1);
                CK(cudaSetDevice(split_dev));
                CK(cudaGetLastError());
                CK(cudaStreamWaitEvent(st, ev_cross, 0));
                if (!split_copy_peer) {   // the second card's stream takes the rows from the host
                    CK(cudaMemcpyAsync(set[1].x, split_stage, nx * 4, cudaMemcpyHostToDevice, st));
                    CK(cudaMemcpyAsync(set[1].h, split_stage + nx, nx * 4, cudaMemcpyHostToDevice, st));
                    CK(cudaMemcpyAsync(set[1].tok, (int32_t*) (split_stage + 2 * nx), (size_t) T * 4, cudaMemcpyHostToDevice, st));
                    CK(cudaMemcpyAsync(set[1].pos0, (int32_t*) (split_stage + 2 * nx) + T, 4, cudaMemcpyHostToDevice, st));
                }
            }
            const bool last = l + 1 == m->layers.size();
            const bool on2 = split_dev >= 0 && (int) l >= split_layer;   // the second card's layers
            Layer& L = layer_of(l);
            // the next layer's norm: the primary's copy for its own layers (the boundary one is copied there),
            // the second card's copy for the second card's layers
            const float* wn = last ? (on2 ? m->out_norm2 : m->out_norm)
                                   : (on2 ? layer_of(l + 1).attn_norm : m->layers[l + 1].attn_norm);
            layer(L, l, T, p0, pos0, graph_mode, verify, wn, last ? hmain : h);
            if (dump_layers && T > 0) {
                // the layer's output (the residual stream), like q36ref's and llama.cpp's l_out-N
                if (!dump_row.size()) dump_row.resize((size_t) T * N);
                CK(cudaMemcpyAsync(dump_row.data(), x, (size_t) T * N * 4, cudaMemcpyDeviceToHost, st));
                CK(cudaStreamSynchronize(st));
                for (int t = 0; t < T; ++t)
                    std::fwrite(dump_row.data() + (size_t) t * N, 4, (size_t) N, dump_layers);
                std::fflush(dump_layers);   // the engine does not close the file: flush it as it goes
            }
        }
        if (heads > 0) {
            if (split_dev >= 0) {
                // the head runs on the primary (card 0): copy the last row's hidden (12 Ko) from card 1
                const size_t nh = (size_t) N * 4;
                std::vector<float> hb((size_t) N);
                CK(cudaMemcpyAsync(hb.data(), hmain + (size_t) (T - heads) * N, nh, cudaMemcpyDeviceToHost, st));
                CK(cudaStreamSynchronize(st));
                CK(cudaSetDevice(0));
                CK(cudaGetLastError());
                use_set(0);   // the head's GEMM (gemm[]) and its stream belong to card 0
                CK(cudaMemcpy(hlast, hb.data(), nh, cudaMemcpyHostToDevice));
                head_rows(hlast, heads, logits, pick);
            } else {
                head_rows(hmain + (size_t) (T - heads) * N, heads, logits, pick);
            }
        }
        if (split_dev >= 0) {
            // The layer split: the call ends on the first card, its set AND its device current.  A chunk without a
            // head (a long prompt's first chunks) ended with the second card's set current: the next chunk's tokens
            // and position went to the second card's tok / pos0 while the embedding read the first card's (a prompt
            // over one chunk faulted in embed_rows).  After a head the first card's set was current but the second
            // card's device, and the caller's copies of logits / pick ran with the wrong device current.
            if (cur_set != 0) {
                CK(cudaStreamSynchronize(st));   // the second card's layers are done before its set is put away
                use_set(0);
            }
            CK(cudaSetDevice(0));
            CK(cudaGetLastError());
        }
    }
    /// logits [n][V] = W_out hn rows, and each row's greedy pick.
    void head_rows(const float* hn, int n, float* lg, int32_t* pk) {
        const auto& g = m->g;
        mm(head_mat(), hn, lg, n);
        k_argmax<<<dim3(kArgBlocks, n), kThreads, 0, st>>>(lg, (int) g.n_vocab, pk, amv, ami, false);
        k_argmax<<<dim3(1, n), kThreads, 0, st>>>(lg, kArgBlocks, pk, amv, ami, true);
    }

    /// The draft head: the MTP block's pick over the first `draft_n` rows of the output matrix only (the lowest token ids are
    /// the most frequent ones).  A draft only proposes: the verify window decides, so a token outside the subset is a
    /// rejected draft and nothing else - the answer is the plain greedy answer whatever draft_n is.  0 = the whole vocabulary.
    int draft_n = 0;
    void head_draft(const float* hn) {
        const auto& g = m->g;
        const int N = (int) g.n_embd;
        const Mat& hd = m->head;
        if (!fast_k || draft_n <= 0 || draft_n >= g.n_vocab || draft_n % 8 != 0 || hd.type != 8 || hd.n_in != N || N / 32 != 64) {
            head_rows(hn, 1, logits_m, pick_m);
            return;
        }
        quant(hn, N, 1, q8);
        const uint8_t* pk = nullptr;
        for (const auto& p : m->packed)
            if (std::get<0>(p) == hd.d) pk = std::get<1>(p);
        if (pk) {
            const half* dpl = reinterpret_cast<const half*>(pk + (size_t) hd.n_in * hd.n_out);
            m2k::launch_q8_rows<1, 2, true>(pk, dpl, q8, logits_m, 64, draft_n, st);
        } else {
            m2k::launch_q8_rows<1, 2, false>(hd.d, nullptr, q8, logits_m, 64, draft_n, st);
        }
        k_argmax<<<dim3(kArgBlocks, 1), kThreads, 0, st>>>(logits_m, draft_n, pick_m, amv, ami, false);
        k_argmax<<<dim3(1, 1), kThreads, 0, st>>>(logits_m, kArgBlocks, pick_m, amv, ami, true);
    }

    /// The MTP block over T rows: row t reads token tok_m[t] at position p0 + t (also at *pos0_m) and the seed hidden
    /// seeds[t] (the main model's normed hidden of the position before, or the block's own for a chained draft);
    /// hmtp gets each row's normed output.  `draft`: logits of the last row and its pick (`pick_m`).
    void forward_mtp(int T, int64_t p0, const float* seeds, bool draft, bool graph_mode) {
        const auto& g = m->g;
        const int N = (int) g.n_embd;
        if (main_hybrid && lhy(m->layers.size()) && T > hybrid_t && (use_mmq || stream_native)) stream_begin({(int) m->layers.size()});
        K::iq_embed_rows(m->embd_type, m->embd, m->embd_row, tok_m, T, N, y, st);
        if (map_on) k_put_rows<<<T, kThreads, 0, st>>>(y, d_img, d_map, N);
        k_rms_ld<<<T, kThreads, 0, st>>>(y, N, m->mtp_enorm, mtp_cat, 2 * N, N, m->eps);              // [e | ...]
        k_rms_ld<<<T, kThreads, 0, st>>>(seeds, N, m->mtp_hnorm, mtp_cat + N, 2 * N, N, m->eps);      // [... | h]
        mm(m->eh_proj, mtp_cat, x, T);
        k_rms<<<T, kThreads, 0, st>>>(x, m->mtp.attn_norm, h, N, m->eps);
        layer(const_cast<Layer&>(m->mtp), m->layers.size(), T, p0, pos0_m, graph_mode, false, m->mtp_shnorm, hmtp);
        if (draft) head_draft(hmtp + (size_t) (T - 1) * N);
    }

    /// After a verify window of T rows: the GDN state takes the first n_keep rows (conv history and recurrence run
    /// again over the rows' kept inputs).
    void commit(int n_keep) {
        const auto& g = m->g;
        const int HK = (int) g.ssm_k_heads, HV = (int) g.ssm_v_heads, C = (int) g.ssm_conv_channels;
        for (size_t l = 0; l < m->layers.size(); ++l) {
            const Layer& L = m->layers[l];
            if (L.attn) continue;
            k_conv_seq<<<C / 128, 128, 0, st>>>(conv[l], v_qkv[l], L.conv, hc, C, 2 * HK, m->eps, n_keep, true);
            k_gdn_seq<<<HV, dim3(128, 4), 0, st>>>(S[l], v_hc[l], C, v_alpha[l], v_beta[l], z, L.ssm_norm, m->eps,
                                                   nullptr, HK, HV, n_keep, true);
        }
    }

    // ---- the hybrid mode
    const Layer& layer_at(int li) const { return li < (int) m->layers.size() ? m->layers[(size_t) li] : m->mtp; }

    /// A streamed forward over the state indices `seq`: the first two layers' experts start copying now.
    void stream_begin(const std::vector<int>& seq) {
        stream_seq = seq;
        stream_pos = 0;
        for (int i = 0; i < 2 && i < (int) seq.size(); ++i) stream_kick(i);
    }
    void stream_kick(int i) {   // the i-th layer of the sequence into staging buffer i % 2, after its last reader
        const int b = i % 2;
        const int li = stream_seq[(size_t) i];
        const Layer& L = layer_of((size_t) li);   // the layer's own card (the layer split)
        const int ne = (int) m->g.n_expert;
        const Hy& H = hy[(size_t) li];
        if (!stream_vram) {
            // The pool's experts into the pinned host buffer: the CPU repacks them (the mapping, straight from the
            // file) and the GPU reads them over PCIe.  The experts with a slot are NOT copied -- the grouping points
            // at their VRAM blob -- so the traffic is the pool's share of a layer, not the whole layer.
            CK(cudaEventSynchronize(ev_used[b]));   // the buffer's last reader on card 0 is done
            if (split_dev > 0) CK(cudaEventSynchronize(ev_used2[b]));   // ... and on card 1 (the pair is shared)
            uint8_t* dst = stage[b];
            // The experts are copied by the fill pool's threads, not by this one: this thread has the layer's
            // launches to queue (`stream_done` has just recorded that the GPU is done with this buffer).
            fill.run(
                [&](int lo, int hi) {
                    for (int x = lo; x < hi; ++x) {
                        if (H.slot_of[(size_t) x] >= 0) continue;   // read from its VRAM slot: nothing to stream
                        if (L.host_has(x))   // --resident-budget-gib: it is already pinned, take it there
                            std::memcpy(dst + (size_t) x * L.stride, L.host_blob(x), (size_t) L.L.bytes);
                        else
                            expert_blob(L, x, dst + (size_t) x * L.stride);
                    }
                },
                ne);
            return;
        }
        // The copies go to the LAYER's card, on its own copy stream: a kernel of card 1 reading card 0's staging
        // faults (page not present).  The caller's current device is restored on the way out.
        const int d = card_of(li);
        int prev = 0;
        CK(cudaGetDevice(&prev));
        if (prev != d) { CK(cudaSetDevice(d)); CK(cudaGetLastError()); }
        cudaStream_t cs = cst_of(d);
        uint8_t* dst = stage_of(d, b);
        CK(cudaStreamWaitEvent(cs, ev_used_of(d, b), 0));
        // the experts with a VRAM slot come from it (device to device, at VRAM speed), only the others over PCIe; runs
        // of consecutive experts from the same source in one copy each.  STRATA_Q36_STREAM_ALL=1: the whole layer from
        // host memory (the first version: PCIe-bound)
        static const bool all = std::getenv("STRATA_Q36_STREAM_ALL") != nullptr;
        // fewer than half the experts resident (a 12 GB card): the one whole-layer copy is faster than many small ones
        // (RTX 3060, UD-Q4_K_XL at 37 of 256: pp2048 1970 whole vs 1540 in runs)
        if (L.hblobs && L.h_first == 0 && L.h_resident >= ne && (all || 2 * L.n_slots < ne)) {
            CK(cudaMemcpyAsync(dst, L.hblobs, L.stride * (size_t) ne, cudaMemcpyHostToDevice, cs));
        } else {
            // the expert's source: its VRAM slot, the host copy (--resident-budget-gib), or the mapping - which needs
            // each expert repacked on its own (the staging buffer wants the native [gate | up | down] layout)
            for (int x = 0; x < ne;) {
                const bool dev = H.slot_of[(size_t) x] >= 0;
                const bool res = !dev && L.host_has(x);
                int y = x + 1;
                if (dev)   // consecutive slots holding consecutive experts
                    while (y < ne && H.slot_of[(size_t) y] == H.slot_of[(size_t) y - 1] + 1) ++y;
                else if (res)   // consecutive resident experts: one run of the arena
                    while (y < ne && H.slot_of[(size_t) y] < 0 && L.host_has(y)) ++y;
                if (dev) {
                    CK(cudaMemcpyAsync(dst + (size_t) x * L.stride, L.blobs + (size_t) H.slot_of[(size_t) x] * L.stride,
                                       (size_t) (y - x) * L.stride, cudaMemcpyDeviceToDevice, cs));
                } else if (res) {
                    CK(cudaMemcpyAsync(dst + (size_t) x * L.stride, L.host_blob(x),
                                       (size_t) (y - x) * L.stride, cudaMemcpyHostToDevice, cs));
                } else {
                    // the host buffer stays in use until the copy has taken it: wait for the stream before
                    // filling it again
                    CK(cudaStreamSynchronize(cs));
                    expert_blob(L, x, repack_pf);
                    CK(cudaMemcpyAsync(dst + (size_t) x * L.stride, repack_pf, L.L.bytes, cudaMemcpyHostToDevice,
                                       cs));
                }
                x = y;
            }
        }
        CK(cudaEventRecord(ev_copy_of(d, b), cs));
        if (prev != d) { CK(cudaSetDevice(prev)); CK(cudaGetLastError()); }
    }
    const uint8_t* stream_take(int li) {
        if (stream_pos >= (int) stream_seq.size() || stream_seq[(size_t) stream_pos] != li) {
            std::fprintf(stderr, "strata-m2: the expert stream is out of step (layer %d)\n", li);
            std::exit(1);
        }
        const int b = stream_pos % 2;
        const int d = card_of(li);
        if (stream_vram) CK(cudaStreamWaitEvent(st, ev_copy_of(d, b), 0));   // host mode: the fill is the CPU's
        return stage_of(d, b);
    }
    void stream_done(int li) {
        const int b = stream_pos % 2;
        CK(cudaEventRecord(ev_used_of(card_of(li), b), st));
        if (stream_pos + 2 < (int) stream_seq.size()) stream_kick(stream_pos + 2);
        ++stream_pos;
    }

    /// The GPU half of a hybrid MoE: the routing to the host, the resident experts' entries grouped and run.
    void moe_hybrid_gpu(const Layer& L, size_t li, int T) {
        const auto& g = m->g;
        const int N = (int) g.n_embd, Kp = (int) m->top_k;
        const int64_t n_ent = (int64_t) T * Kp;
        CK(cudaMemcpyAsync(h_ids, ids, (size_t) n_ent * 4, cudaMemcpyDeviceToHost, st));
        CK(cudaMemcpyAsync(h_h, h, (size_t) T * N * 4, cudaMemcpyDeviceToHost, st));
        CK(cudaEventRecord(ev_route, st));
        group_entries(L, hy[li].slot_of_d, T);
        if (!xq_post) quant(h, N, T, xq);   // the experts' and the shared expert's input
        CK(cudaEventRecord(ev_xq, st));     // the activations are whole: the peer copies them after this
        K::native_expert_grouped(L.L, grp_ptr, grp_start, n_groups, ent_dst, ent_tok,
                                 std::min<int64_t>(n_ent, g.n_expert), n_ent, xq, escr, eparts, st);
    }
    /// The peer's share of a layer: the entries whose expert the PEER holds (the primary's slot table says no).
    /// The peer groups them with its own slot table, computes their rows with the same kernels as the primary
    /// (so the rows are bit-identical), and writes them into the same pinned host rows the CPU pool writes.
    void moe_hybrid_peer(size_t li, int T) {
        if (peer_device < 0 || li >= peer_hy.size()) return;
        // a layer the peer had no room for: it holds none of it (the table was never built, its experts are on the
        // CPU pool), so there is nothing to group, compute or wait for on that card
        if (peer_hy[li].slot_of_d == nullptr) return;
        const auto& g = m->g;
        const int N = (int) g.n_embd, Kp = (int) m->top_k;
        const int n_ent = T * Kp;
        const Layer& L = layer_of((size_t) li);   // the layer's own copy (the layer split)
        PeerHy& P = peer_hy[li];
        // The card the layer's buffers (ids, xq) are on: 0, or the split's card.  With the peer on that same card
        // (--layer-split N --peer-device <split card>) the two copies below are a plain copy on its own stream.
        const int src_dev = card_of((int) li);
        // The peer's stream waits for the routing (the ids) and for the activations.  EVERY call on the peer's
        // stream needs the PEER's device current: HIP refuses a launch on a stream that belongs to another device.
        // The ids and the activations, copied to the peer's card.  The copy is issued with the SOURCE's device
        // current and on the source's stream: a peer copy reads the source in the SOURCE's address space, and
        // with the peer's device current it would read the peer's memory at that address (garbage).
        CK(cudaSetDevice(src_dev));
        CK(cudaGetLastError());
        if (src_dev == peer_device) {
            CK(cudaMemcpyAsync(peer_ids, ids, (size_t) n_ent * 4, cudaMemcpyDeviceToDevice, st));
            CK(cudaMemcpyAsync(peer_xq, xq, K::native_q8_1_bytes(N, 1) * (size_t) T, cudaMemcpyDeviceToDevice, st));
        } else {
            CK(cudaMemcpyPeerAsync(peer_ids, peer_device, ids, src_dev, (size_t) n_ent * 4, st));
            CK(cudaMemcpyPeerAsync(peer_xq, peer_device, xq, src_dev, K::native_q8_1_bytes(N, 1) * (size_t) T, st));
        }
        CK(cudaEventRecord(ev_xq, st));   // the copies are done (and the quantization was before them)
        if (getenv("STRATA_M2_DUMP_ROWS")) {   // DIAG: the activations, byte for byte
            std::vector<uint8_t> a((size_t) K::native_q8_1_bytes(N, 1) * (size_t) T), b(a.size());
            CK(cudaMemcpyAsync(a.data(), xq, a.size(), cudaMemcpyDeviceToHost, st));
            CK(cudaStreamSynchronize(st));
            CK(cudaMemcpyAsync(b.data(), peer_xq, a.size(), cudaMemcpyDeviceToHost, peer_st));
            CK(cudaStreamSynchronize(peer_st));
            long ndiff = 0, first = -1;
            for (size_t z = 0; z < a.size(); ++z)
                if (a[z] != b[z]) { if (first < 0) first = (long) z; ++ndiff; }
            std::fprintf(stderr, "XQ   layer %d T=%d : %ld differing bytes (first at %ld of %zu)\n", (int) li, T,
                         ndiff, first, a.size());
            std::fflush(stderr);
        }
        // From here on, every call on the peer's stream needs the PEER's device current: HIP refuses a launch
        // on a stream that belongs to another device.
        CK(cudaSetDevice(peer_device));
        CK(cudaGetLastError());
        CK(cudaStreamWaitEvent(peer_st, ev_xq, 0));
        // the grouping on the peer: only the entries whose expert it holds
        // the DENSE rows (one per expert the peer holds), not the entries: the table is the grouping's, and it
        // has one row per expert
        CK(cudaMemsetAsync(peer_ent_dst, 0xff, peer_rows_max * 4, peer_st));
        k_group_hy<<<1, 256, 0, peer_st>>>(peer_ids, n_ent, Kp, (int) g.n_expert, P.slot_of_d, L.peer_blobs, L.stride,
                                             peer_grp_ptr, peer_grp_start, peer_n_groups, peer_ent_dst, peer_ent_tok);
        K::native_expert_grouped(L.L, peer_grp_ptr, peer_grp_start, peer_n_groups, peer_ent_dst, peer_ent_tok,
                                 std::min<int64_t>(n_ent, g.n_expert), n_ent, peer_xq, peer_escr, peer_eparts, peer_st);
        if (getenv("STRATA_M2_DUMP_ROWS")) {   // DIAG: the PRIMARY's own rows (its GPU) for the same entries
            int shown = 0;
            std::fprintf(stderr, "PRIMROW layer %d T=%d :", (int) li, T);
            for (int e = 0; e < n_ent && shown < 4; ++e) {
                const int32_t x = h_ids[e];
                const int slot = (peer_device >= 0 && li < peer_hy.size()) ? peer_hy[(size_t) li].slot_of[(size_t) x] : -1;
                if (slot < 0) continue;
                ++shown;
                const int row = ent_dst[e];   // the entry's row in the primary's own output
                for (int j = 0; j < 3; ++j) std::fprintf(stderr, " %.9g", eparts[(size_t) row * N + j]);
            }
            std::fprintf(stderr, "\n");
            std::fflush(stderr);
        }
        if (getenv("STRATA_M2_DUMP_ROWS")) {   // DIAG: the peer's own rows, for the entries it took
            CK(cudaStreamSynchronize(peer_st));
            int shown = 0;
            std::fprintf(stderr, "PEERROW layer %d T=%d :", (int) li, T);
            for (int e = 0; e < n_ent && shown < 4; ++e) {
                const int32_t x = h_ids[e];
                const int slot = (peer_device >= 0 && li < peer_hy.size()) ? peer_hy[(size_t) li].slot_of[(size_t) x] : -1;
                if (slot < 0) continue;
                ++shown;
                for (int j = 0; j < 3; ++j) std::fprintf(stderr, " %.9g", cpu_out[(size_t) e * N + j]);
            }
            std::fprintf(stderr, "\n");
            std::fflush(stderr);
        }
        CK(cudaEventRecord(peer_done, peer_st));
        // back to the card the layer runs on: the merge and the rest of the layer launch on ITS stream, and a
        // kernel launched on another device's stream fails with "invalid resource handle"
        CK(cudaSetDevice(src_dev));
        CK(cudaGetLastError());
        // the primary waits for the peer's rows before the merge reads them
        CK(cudaStreamWaitEvent(st, peer_done, 0));
    }
    /// The CPU half: the entries without a slot on the pool (ggml-cpu's arithmetic for the layer's formats), their
    /// rows uploaded and merged into the entries' rows.
    void moe_hybrid_cpu(size_t li, int T) {
#if defined(STRATA_Q36_CPU_POOL)
        namespace C = strata::kernels::cpu;
        const auto& g = m->g;
        const int N = (int) g.n_embd, Kp = (int) m->top_k;
        const int n_ent = T * Kp;
        Hy& H = hy[li];
        // the layer's OWN copy (the layer split: the second card's layers have their own weights)
        const Layer& L = layer_of((size_t) li);
        {
            const auto tw = Clock::now();
            while (cudaEventQuery(ev_route) == cudaErrorNotReady) {
            }
            hy_wait_ms += ms_since(tw);
        }
        const auto t0 = Clock::now();
        // the missed entries per expert (up to MAXT tokens per job)
        jobs.clear();
        // The per-job layouts of the experts read in place.  Sized ONCE here: a resize inside the loop
        // would move the vector and leave the jobs already created pointing at freed memory.
        if (zc_fmt.size() < (size_t) n_ent) zc_fmt.resize((size_t) n_ent);
        std::vector<char> quantized((size_t) T, 0);
        int n_miss = 0;   // the entries the primary's GPU does not compute (the CPU pool's and the peer's)
        int n_pool = 0;   // ... and of those, the ones this pool computes
        int n_peer = 0;   // ... and the ones the second card computes
        std::map<int, size_t> open;   // expert -> its job with room
        // The second card's table: the entries it computes are not this pool's.  Its kernels write their rows into
        // the same pinned host buffer (cpu_out is the peer's output too), and the merge below takes them because
        // h_miss says the primary's GPU did not compute the entry.
        const std::vector<int32_t>* peer_slots =
            (peer_device >= 0 && li < peer_hy.size()) ? &peer_hy[li].slot_of : nullptr;
        const bool defer = rows_deferred && li < m->layers.size();
        if (defer) {
            if (row_ids.size() < m->layers.size()) row_ids.resize(m->layers.size());
            row_ids[li].assign(h_ids, h_ids + n_ent);
        }
        for (int e = 0; e < n_ent; ++e) {
            const int x = h_ids[e];
            if (!defer) H.score[(size_t) x] += 1.0;
            h_miss[e] = H.slot_of[(size_t) x] < 0;
            if (!h_miss[e]) continue;
            ++n_miss;
            // the second card computes this entry: a job here would compute it a second time, into the same row
            if (peer_slots && (size_t) x < peer_slots->size() && (*peer_slots)[(size_t) x] >= 0) {
                ++n_peer;
                continue;
            }
            ++n_pool;
            const int t = e / Kp;
            if (!quantized[(size_t) t]) {
                C::native_quant_act(H.fmt, h_h + (size_t) t * N, acts + (size_t) t * C::kNativeActBytes);
                quantized[(size_t) t] = 1;
            }
            auto it = open.find(x);
            if (it == open.end() || jobs[it->second].nt == C::MAXT) {
                C::ExpertJobMulti j;
                if (L.host_has(x)) {
                    j.blob = L.host_blob(x);
                    j.fmt = nullptr;
                } else {
                    // the GGUF read in place: the blob is the expert's gate rows, and its up and down rows
                    // are at their distance in the mapping (the job's own NativeFmt carries them)
                    j.blob = L.src_g + (size_t) x * L.src_gsz;
                    C::NativeFmt& f = zc_fmt[jobs.size()];
                    f = H.fmt;
                    // The distance from THIS expert's gate rows to its up and down rows, in the mapping.  The
                    // job's blob is at src_g + x * src_gsz, so the x terms only cancel for the up matrix
                    // (same size per expert); the down matrix is n_embd rows of d_row, a different size, and
                    // its x term stays.
                    f.up_off = (size_t) ((uintptr_t) L.src_u - (uintptr_t) L.src_g);
                    f.down_off = (size_t) ((uintptr_t) L.src_d - (uintptr_t) L.src_g) +
                                 (size_t) x * (L.src_dsz - L.src_gsz);
                    j.fmt = &f;
                }
                jobs.push_back(j);
                it = open.insert_or_assign(x, jobs.size() - 1).first;
            }
            C::ExpertJobMulti& j = jobs[it->second];
            j.nact[j.nt] = acts + (size_t) t * C::kNativeActBytes;
            j.out[j.nt] = cpu_out + (size_t) e * N;
            ++j.nt;
        }
        hy_cpu += n_pool;
        hy_peer += n_peer;
        hy_gpu += n_ent - n_miss;
        if (n_miss == 0) return;
        if (n_pool > 0) {
            pool->run_split_multi_native(H.fmt, jobs.data(), (int) jobs.size());
            hy_cpu_ms += ms_since(t0);
        }
        {   // DIAG: the host rows (the CPU pool's and the peer's), before they are merged
            const char* lay = std::getenv("STRATA_M2_DUMP_ROWS");
            if (lay && (int) li == std::atoi(lay)) {
                std::fprintf(stderr, "ROWS layer %d T=%d :", (int) li, T);
                for (int e = 0; e < 3; ++e)
                    for (int j = 0; j < 3; ++j) std::fprintf(stderr, " %.9g", cpu_out[(size_t) e * N + j]);
                std::fprintf(stderr, "\n");
                std::fflush(stderr);
            }
        }
        {   // DIAG: the host rows (the CPU pool's and the peer's), and which entries the peer took
            const char* lay = std::getenv("STRATA_M2_DUMP_ROWS");
            if (lay && (int) li == std::atoi(lay)) {
                std::fprintf(stderr, "ROWS layer %d T=%d :", (int) li, T);
                for (int e = 0; e < 8 && e < n_ent; ++e)
                    for (int j = 0; j < 2; ++j) std::fprintf(stderr, " %.9g", cpu_out[(size_t) e * N + j]);
                std::fprintf(stderr, "\n");
                std::fprintf(stderr, "IDS  layer %d T=%d :", (int) li, T);
                for (int e = 0; e < 8 && e < n_ent; ++e) {
                    const int32_t x = h_ids[e];
                    const int slot = (peer_device >= 0 && li < peer_hy.size()) ? peer_hy[(size_t) li].slot_of[(size_t) x] : -1;
                    std::fprintf(stderr, " %d%s", x, slot >= 0 ? "*" : "");
                }
                std::fprintf(stderr, "\n");
                std::fflush(stderr);
            }
        }
        if (getenv("STRATA_M2_DUMP_ROWS")) {   // DIAG: the CPU pool's rows, for the same entries
            int shown = 0;
            std::fprintf(stderr, "CPUROW layer %d T=%d :", (int) li, T);
            for (int e = 0; e < n_ent && shown < 4; ++e) {
                const int32_t x = h_ids[e];
                const int slot = (peer_device >= 0 && li < peer_hy.size()) ? peer_hy[(size_t) li].slot_of[(size_t) x] : -1;
                if (slot < 0) continue;
                ++shown;
                for (int z = 0; z < 3; ++z) std::fprintf(stderr, " %.9g", cpu_out[(size_t) e * N + z]);
            }
            std::fprintf(stderr, "\n");
            std::fflush(stderr);
        }
        CK(cudaMemcpyAsync(ecpu, cpu_out, (size_t) n_ent * N * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(d_miss, h_miss, (size_t) n_ent * 4, cudaMemcpyHostToDevice, st));
        k_merge_cpu<<<n_ent, kThreads, 0, st>>>(eparts, ecpu, d_miss, N);
#else
        (void) li;
        (void) T;
#endif
    }

    /// The slots to the experts the routing used most (the counts halved each time): at most max_swaps replacements
    /// over all layers, between forwards (the copies are on the engine's stream).
    void rebalance(int max_swaps) {
        static const bool off = std::getenv("STRATA_Q36_FIXED_SLOTS") != nullptr;   // the slots as placed (tests)
        if (!m->hybrid || off || warming) return;
        const int64_t ne = m->g.n_expert;
        int left = max_swaps;
        for (size_t li = 0; li < hy.size() && left > 0; ++li) {
            Hy& H = hy[li];
            // the layer's OWN copy: the second card's slots are on its card (layer_at would give the first card's)
            const Layer& L = layer_of((size_t) li);
            const int S = L.n_slots;
            if (S <= 0 || S >= ne) continue;
            std::vector<int> order((size_t) ne);
            for (int x = 0; x < ne; ++x) order[(size_t) x] = x;
            std::sort(order.begin(), order.end(), [&](int a, int b) { return H.score[(size_t) a] > H.score[(size_t) b]; });
            // residents outside the top S, lowest score first: the slots to give away
            std::vector<int> victims;
            for (int s = 0; s < S; ++s) victims.push_back(s);
            std::sort(victims.begin(), victims.end(), [&](int a, int b) {
                return H.score[(size_t) H.exp_in_slot[(size_t) a]] < H.score[(size_t) H.exp_in_slot[(size_t) b]];
            });
            size_t vi = 0;
            bool changed = false;
            for (int i = 0; i < S && left > 0; ++i) {
                const int x = order[(size_t) i];
                if (H.slot_of[(size_t) x] >= 0) continue;
                while (vi < victims.size()) {
                    const int s = victims[vi];
                    const int y = H.exp_in_slot[(size_t) s];
                    if (H.score[(size_t) y] >= H.score[(size_t) x]) { vi = victims.size(); break; }
                    // y must not be in the top S itself
                    ++vi;
                    H.slot_of[(size_t) y] = -1;
                    H.slot_of[(size_t) x] = s;
                    H.exp_in_slot[(size_t) s] = x;
                    if (L.host_has(x)) {
                        CK(cudaMemcpyAsync(L.blobs + (size_t) s * L.stride, L.host_blob(x), L.L.bytes,
                                           cudaMemcpyHostToDevice, st));
                    } else {
                        // the low-RAM mode: the expert is repacked out of the mapping first
                        CK(cudaStreamSynchronize(st));
                        expert_blob(L, x, repack);
                        CK(cudaMemcpyAsync(L.blobs + (size_t) s * L.stride, repack, L.L.bytes,
                                           cudaMemcpyHostToDevice, st));
                    }
                    --left;
                    ++hy_swaps;
                    changed = true;
                    break;
                }
                if (vi >= victims.size()) break;
            }
            if (changed) {
                std::memcpy(htoks, H.slot_of.data(), (size_t) ne * 4);
                CK(cudaMemcpyAsync(H.slot_of_d, htoks, (size_t) ne * 4, cudaMemcpyHostToDevice, st));
            }
            for (double& v : H.score) v *= 0.5;
        }
        CK(cudaStreamSynchronize(st));   // the upload is done before the host table changes
    }

    void shrink_att(size_t floats) {
        CK(cudaDeviceSynchronize());
        CK(cudaFree(att_s));
        CK(cudaFree(att_p16));
        kAttScoreFloats = floats;
        att_s = dalloc(kAttScoreFloats);
        att_p16 = dalloc_t<__half>(kAttScoreFloats);
        CK(cudaDeviceSynchronize());
    }
    void reset_graphs() {
        for (GSet& gset : gsets) {
            for (cudaGraphExec_t* g : {&gset.g1, &gset.gm1})
                if (*g) { CK(cudaGraphExecDestroy(*g)); *g = nullptr; }
            for (auto& g : gset.gv)
                if (g) { CK(cudaGraphExecDestroy(g)); g = nullptr; }
        }
        for (auto& kv : row_graphs)
            if (kv.second) CK(cudaGraphExecDestroy(kv.second));
        row_graphs.clear();
    }

    template <class F> void graph_run(cudaGraphExec_t& gx, F body) {
        if (!use_graph) { body(); return; }
        if (!gx) {
            cudaGraph_t graph = nullptr;
            CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
            body();
            CK(cudaStreamEndCapture(st, &graph));
            CK(cudaGraphInstantiate(&gx, graph, 0));
            CK(cudaGraphDestroy(graph));
        }
        CK(cudaGraphLaunch(gx, st));
    }

    // ---- pictures (--vision).  A request with pictures hands the engine its rows (the encoder's, in prompt order), the
    // cell -> row map of its <|image_pad|> cells and the M-RoPE table of its prompt cells; the cells below the prompt
    // length take their rotary positions from the table, the text after it continues at cell + delta (what the pictures
    // did not use of their cells: llama.cpp's mtmd advances the text position by max(nx, ny) per picture).
    bool img_active = false;
    void set_images(const std::vector<float>& rows, const std::vector<int32_t>& cell_row, const std::vector<int32_t>& mrope,
                    int64_t n_prompt, int32_t delta) {
        const size_t need = rows.size();
        if (need > img_cap) {
            CK(cudaDeviceSynchronize());
            if (d_img) CK(cudaFree(d_img));
            CK(cudaMalloc((void**) &d_img, need * 4));
            img_cap = need;
        }
        CK(cudaMemcpy(d_img, rows.data(), need * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_mtab, mrope.data(), mrope.size() * 4, cudaMemcpyHostToDevice));
        img_map = cell_row;
        img_prompt_cells = n_prompt;
        img_active = true;
        set_delta(delta);
    }
    void clear_images() {
        img_map.clear();
        img_active = false;
        img_prompt_cells = 0;
        mtab_now = nullptr;
        map_on = false;
        set_delta(0);
    }
    /// A forward over the cells [c0, c0 + T): the picture rows of its cells, and the table when it is a prompt's.
    void chunk_pictures(int64_t c0, int T) {
        mtab_now = nullptr;
        map_on = false;
        if (!img_active) return;
        if (c0 < img_prompt_cells) mtab_now = d_mtab;
        bool any = false;
        for (int j = 0; j < T; ++j) {
            const int64_t c = c0 + j;
            const int32_t v = c < (int64_t) img_map.size() ? img_map[(size_t) c] : -1;
            h_map[j] = v;
            any = any || v >= 0;
        }
        if (any) {
            CK(cudaMemcpyAsync(d_map, h_map, (size_t) T * 4, cudaMemcpyHostToDevice, st));
            map_on = true;
        }
    }

    /// The tokens at the current position: a prompt in chunks (eager), one token from the decode graph.  The last
    /// token's logits stay in `logits` (its greedy pick in `pick`), hlast its normed hidden.  With the MTP block the
    /// prompt also runs through it (its K / V for every position: row q reads h_{q-1} and token q).  on_chunk(done) runs
    /// after every chunk; false ends the read there (a cancelled request): the tokens processed are returned.
    template <class F> int64_t process(const int32_t* toks, int64_t n, F on_chunk_) {
        const int N = (int) m->g.n_embd;
        auto on_chunk = [&](int64_t done) -> bool {   // a callback returning void never cancels
            if constexpr (std::is_void_v<decltype(on_chunk_(done))>) {
                on_chunk_(done);
                return true;
            } else {
                return on_chunk_(done);
            }
        };
        if (n == 1) {
            CK(cudaStreamSynchronize(st));   // the pinned staging is free again
            hpin[0] = toks[0];
            hpin[1] = (int32_t) pos;
            CK(cudaMemcpyAsync(tok, hpin, 4, cudaMemcpyHostToDevice, st));
            CK(cudaMemcpyAsync(pos0, hpin + 1, 4, cudaMemcpyHostToDevice, st));
            chunk_pictures(pos, 1);
            rows_deferred = true;
            // a prompt cell of a picture request (its rotary position is in the table, its row a picture's) runs eagerly
            if (mtab_now || map_on) forward(1, pos, 1, false, false);
            else graph_run(gs().g1, [&] { forward(1, pos, 1, use_graph, false); });
            mtab_now = nullptr;
            map_on = false;
            rows_deferred = false;
            if (hlast) CK(cudaMemcpyAsync(hlast, hmain, (size_t) N * 4, cudaMemcpyDeviceToDevice, st));
            pos += 1;
            after_rows(1);
            on_chunk(1);
            return 1;
        }
        for (int64_t i = 0; i < n; i += tmax) {
            const int T = (int) std::min<int64_t>(tmax, n - i);
            CK(cudaStreamSynchronize(st));
            hpin[1] = (int32_t) pos;
            std::memcpy(htoks, toks + i, (size_t) T * 4);
            CK(cudaMemcpyAsync(tok, htoks, (size_t) T * 4, cudaMemcpyHostToDevice, st));
            CK(cudaMemcpyAsync(pos0, hpin + 1, 4, cudaMemcpyHostToDevice, st));
            chunk_pictures(pos, T);
            forward(T, pos, i + T == n ? 1 : 0, false, false);
            if (m->has_mtp && mtp_d > 0) {
                // seeds: h_{q-1} for each row q: hlast, then this chunk's rows 0 .. T - 2 (in `y`, free here)
                CK(cudaMemcpyAsync(mtp_seed(), hlast, (size_t) N * 4, cudaMemcpyDeviceToDevice, st));
                if (T > 1)
                    CK(cudaMemcpyAsync(mtp_seed() + N, hmain, (size_t) (T - 1) * N * 4, cudaMemcpyDeviceToDevice, st));
                CK(cudaMemcpyAsync(tok_m, htoks, (size_t) T * 4, cudaMemcpyHostToDevice, st));
                CK(cudaMemcpyAsync(pos0_m, hpin + 1, 4, cudaMemcpyHostToDevice, st));
                forward_mtp(T, pos, mtp_seed(), false, false);
            }
            CK(cudaMemcpyAsync(hlast, hmain + (size_t) (T - 1) * N, (size_t) N * 4, cudaMemcpyDeviceToDevice, st));
            pos += T;
            CK(cudaStreamSynchronize(st));
            mtab_now = nullptr;
            map_on = false;
            if (!on_chunk(i + T)) {
                if (m->hybrid) rebalance(hybrid_swaps_after_prompt);
                return i + T;
            }
        }
        if (m->hybrid) rebalance(hybrid_swaps_after_prompt);
        return n;
    }

    // ---- batch rows: one decode window for the next token of several sessions
    struct Row { int slot; int32_t tok; };
    long long win_count = 0;
    std::map<int, cudaGraphExec_t> row_graphs;   // by (rows, groups, kind): graphs of the batch windows
    /// A slot's rows of one window: `n` consecutive cells from the session's position (1: the next token; 2: also an MTP
    /// proposal the window verifies).
    struct Group { int slot; int n; int32_t tok[2]; };
    int32_t rows_pick[kVerify] = {};
    int rows_G_last = 0;   // groups of the window run last (commit_rows takes their keep counts)

    void run_graph_key(int key, const std::function<void()>& body) {
        if (!use_graph) { body(); return; }
        graph_run(row_graphs[key], body);
    }

    /// One window over the groups' rows.  `verify`: a verify window - the GDN state is read, not written (commit_rows writes
    /// it for the kept rows) and the sessions' positions stay.  The greedy picks end up in rows_pick (one per row), every
    /// row's logits in `logits`.  Without verify the sessions' positions advance by their rows.
    void run_rows(const Group* gr, int G, bool verify) {
        sess[(size_t) cur].pos = pos;
        CK(cudaStreamSynchronize(st));   // the pinned staging is free again
        int64_t pmax = 0;
        int T = 0;
        for (int g = 0; g < G; ++g) {
            const Sess& ss = sess[(size_t) gr[g].slot];
            hrow[40 + g] = gr[g].slot;
            hrow[48 + g] = T;
            hrow[56 + g] = gr[g].n;
            for (int j = 0; j < gr[g].n; ++j) {
                hrow[T] = gr[g].tok[j];
                hrow[8 + T] = (int32_t) (ss.pos + j);
                hrow[16 + T] = gr[g].slot;
                hrow[24 + T] = ss.rdelta;
                ++T;
            }
            pmax = std::max<int64_t>(pmax, ss.pos + gr[g].n - 1);
        }
        CK(cudaMemcpyAsync(tok, hrow, (size_t) T * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(d_rcell, hrow + 8, (size_t) T * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(d_rslot, hrow + 16, (size_t) T * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(d_rdelta, hrow + 24, (size_t) T * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(d_gslot, hrow + 40, (size_t) G * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(d_grow0, hrow + 48, (size_t) G * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(d_gcnt, hrow + 56, (size_t) G * 4, cudaMemcpyHostToDevice, st));
        mtab_now = nullptr;
        map_on = false;
        rows_mode = true;
        rows_G = G;
        rows_deferred = true;
        run_graph_key((verify ? 1000 : 0) + G * 10 + (T - G), [&] { forward(T, pmax, T, use_graph, verify); });
        rows_deferred = false;
        rows_mode = false;
        CK(cudaMemcpyAsync(hrow + 32, pick, (size_t) T * 4, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
        for (int r = 0; r < T; ++r) rows_pick[r] = hrow[32 + r];
        if (!verify)
            for (int g = 0; g < G; ++g) sess[(size_t) gr[g].slot].pos += gr[g].n;
        pos = sess[(size_t) cur].pos;
        rows_G_last = G;
        if (m->hybrid) {
            count_rows(T);
            if (++win_count % kRebalanceEvery == 0) rebalance(8);
        }
    }
    /// One window for the next token of each of the rows' sessions (no proposals): the greedy picks in rows_pick.
    void step_rows(const Row* rows, int nb) {
        Group gr[kVerify];
        for (int r = 0; r < nb; ++r) gr[r] = {rows[r].slot, 1, {rows[r].tok, 0}};
        run_rows(gr, nb, false);
    }
    /// After run_rows(.., verify = true): the GDN state of group g takes its first keep[g] rows (the conv history and the
    /// recurrence run again over the rows' kept inputs, as commit() does) and the session advances by keep[g].
    void commit_rows(const Group* gr, const int* keep, int G) {
        const auto& g = m->g;
        const int HK = (int) g.ssm_k_heads, HV = (int) g.ssm_v_heads, C = (int) g.ssm_conv_channels;
        sess[(size_t) cur].pos = pos;
        for (int i = 0; i < G; ++i) hrow[56 + i] = keep[i];   // (hrow's slot / first-row words are the window's, unchanged)
        CK(cudaMemcpyAsync(d_gcnt, hrow + 56, (size_t) G * 4, cudaMemcpyHostToDevice, st));
        rows_mode = true;
        rows_G = G;
        run_graph_key(2000 + G, [&] {
            for (size_t l = 0; l < m->layers.size(); ++l) {
                const Layer& L = m->layers[l];
                if (L.attn) continue;
                k_conv_seq<<<dim3(C / 128, G), 128, 0, st>>>(nullptr, v_qkv[l], L.conv, hc, C, 2 * HK, m->eps, 1, true,
                                                             d_conv_tab + l * n_sess, d_gslot, d_grow0, d_gcnt);
                k_gdn_seq<<<dim3(HV, G), dim3(128, 4), 0, st>>>(nullptr, v_hc[l], C, v_alpha[l], v_beta[l], z, L.ssm_norm,
                                                                m->eps, nullptr, HK, HV, 1, true, d_S_tab + l * n_sess,
                                                                d_gslot, d_grow0, d_gcnt);
            }
        });
        rows_mode = false;
        CK(cudaStreamSynchronize(st));
        for (int i = 0; i < G; ++i) sess[(size_t) gr[i].slot].pos += keep[i];
        pos = sess[(size_t) cur].pos;
    }

    /// The MTP block over rows of several sessions: row r reads token tok at cell `cell` of session `slot` and the seed
    /// hidden at `seed` (device, n_embd floats); its draft - the block's greedy pick after that row - ends up in
    /// mtp_pick[r], and the block's K / V cells are written.
    struct MtpRow { int slot; int32_t tok; int64_t cell; const float* seed; };
    int32_t mtp_pick[kVerify] = {};
    float* logits_mr = nullptr;
    int32_t* pick_mr = nullptr;
    void mtp_rows(const MtpRow* rows, int T) {
        const auto& g = m->g;
        const int N = (int) g.n_embd;
        if (!logits_mr) {
            logits_mr = dalloc((size_t) kVerify * g.n_vocab);
            pick_mr = dalloc_t<int32_t>(kVerify);
        }
        sess[(size_t) cur].pos = pos;
        CK(cudaStreamSynchronize(st));
        int64_t pmax = 0;
        for (int r = 0; r < T; ++r) {
            hrow[r] = rows[r].tok;
            hrow[8 + r] = (int32_t) rows[r].cell;
            hrow[16 + r] = rows[r].slot;
            hrow[24 + r] = sess[(size_t) rows[r].slot].rdelta;
            pmax = std::max<int64_t>(pmax, rows[r].cell);
            CK(cudaMemcpyAsync(mtp_seed() + (size_t) r * N, rows[r].seed, (size_t) N * 4, cudaMemcpyDeviceToDevice, st));
        }
        CK(cudaMemcpyAsync(tok_m, hrow, (size_t) T * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(d_rcell, hrow + 8, (size_t) T * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(d_rslot, hrow + 16, (size_t) T * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(d_rdelta, hrow + 24, (size_t) T * 4, cudaMemcpyHostToDevice, st));
        mtab_now = nullptr;
        map_on = false;
        rows_mode = true;
        rows_G = T;
        rows_deferred = true;
        run_graph_key(3000 + T, [&] {
            K::iq_embed_rows(m->embd_type, m->embd, m->embd_row, tok_m, T, N, y, st);
            k_rms_ld<<<T, kThreads, 0, st>>>(y, N, m->mtp_enorm, mtp_cat, 2 * N, N, m->eps);                 // [e | ...]
            k_rms_ld<<<T, kThreads, 0, st>>>(mtp_seed(), N, m->mtp_hnorm, mtp_cat + N, 2 * N, N, m->eps);   // [... | h]
            mm(m->eh_proj, mtp_cat, x, T);
            k_rms<<<T, kThreads, 0, st>>>(x, m->mtp.attn_norm, h, N, m->eps);
            layer(const_cast<Layer&>(m->mtp), m->layers.size(), T, pmax, pos0_m, use_graph, false, m->mtp_shnorm, hmtp);
            mm(m->head, hmtp, logits_mr, T);
            k_argmax<<<dim3(kArgBlocks, T), kThreads, 0, st>>>(logits_mr, (int) g.n_vocab, pick_mr, amv, ami, false);
            k_argmax<<<dim3(1, T), kThreads, 0, st>>>(logits_mr, kArgBlocks, pick_mr, amv, ami, true);
        });
        rows_deferred = false;
        rows_mode = false;
        CK(cudaMemcpyAsync(hrow + 32, pick_mr, (size_t) T * 4, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
        for (int r = 0; r < T; ++r) mtp_pick[r] = hrow[32 + r];
    }
    /// Row r's logits of the last step_rows, on the host.
    void host_logits_row(int r, std::vector<float>& out) {
        out.resize((size_t) m->g.n_vocab);
        CK(cudaMemcpyAsync(out.data(), logits + (size_t) r * m->g.n_vocab, out.size() * 4, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
    }

    /// Session dst takes session src's state: the recurrent state, the cache's first src.pos cells, the position, the
    /// pictures' offset (for a conversation read in one session that continues in another).
    void copy_session(int src, int dst) {
        if (src == dst) return;
        sess[(size_t) cur].pos = pos;
        const auto& g = m->g;
        const Sess& a = sess[(size_t) src];
        Sess& b = sess[(size_t) dst];
        const size_t C = (size_t) g.ssm_conv_channels;
        const size_t esz = kv_esz();
        const size_t row = (size_t) ctx * g.head_dim * esz;
        const size_t srow = (size_t) ctx * 4 * kKvBlocks;   // --kv int8: one float per cell, per head and per block, the
                                                            // same row layout
        for (size_t l = 0; l < a.conv.size(); ++l) {
            if (a.conv[l]) {
                CK(cudaMemcpyAsync(b.conv[l], a.conv[l], C * (g.ssm_d_conv - 1) * 4, cudaMemcpyDeviceToDevice, st));
                CK(cudaMemcpyAsync(b.S[l], a.S[l], (size_t) g.ssm_v_heads * g.ssm_state_size * g.ssm_state_size * 4,
                                   cudaMemcpyDeviceToDevice, st));
            }
            if (a.kc[l] && a.pos > 0) {
                CK(cudaMemcpy2DAsync(b.kc[l], row, a.kc[l], row, (size_t) a.pos * g.head_dim * esz,
                                     (size_t) g.n_head_kv, cudaMemcpyDeviceToDevice, st));
                CK(cudaMemcpy2DAsync(b.vc[l], row, a.vc[l], row, (size_t) a.pos * g.head_dim * esz,
                                     (size_t) g.n_head_kv, cudaMemcpyDeviceToDevice, st));
                if (a.ksc[l]) {   // INT8: the rows' scales, one per cell, head and block
                    CK(cudaMemcpy2DAsync(b.ksc[l], srow, a.ksc[l], srow, (size_t) a.pos * 4 * kKvBlocks,
                                         (size_t) g.n_head_kv, cudaMemcpyDeviceToDevice, st));
                    CK(cudaMemcpy2DAsync(b.vsc[l], srow, a.vsc[l], srow, (size_t) a.pos * 4 * kKvBlocks,
                                         (size_t) g.n_head_kv, cudaMemcpyDeviceToDevice, st));
                }
            }
        }
        CK(cudaMemcpyAsync(b.hlast, a.hlast, (size_t) g.n_embd * 4, cudaMemcpyDeviceToDevice, st));
        CK(cudaStreamSynchronize(st));
        b.pos = a.pos;
        b.rdelta = a.rdelta;
        b.live = a.live;
        b.imgs = a.imgs;
        if (dst == cur) load_cur();
    }
    int hybrid_swaps_after_prompt = 2048;
    /// the MTP seed rows of a prompt chunk: a buffer the block's own forward does not use (`moe_out`, [tmax][N])
    float* mtp_seed() { return moe_out; }

    /// One speculative step with D MTP drafts (greedy): `x` is the next token (decided, not processed); `fix` the
    /// tokens of the rows the block must redo first (the drafts the last window accepted, seeded by hmain rows
    /// 0 .. fix.size() - 1 of that window).  Drafts d1 .. dD, then the window [x, d1 .. dD] through the main model;
    /// returns the accepted count a and writes the picks of rows 0 .. a (row a's pick is the token after them).
    /// `cancel`: a stop asked for before the state moves keeps only x (the caller then emits no draft), so that a request
    /// stopped mid-answer leaves the state at the tokens it has shown: a conversation that continues from them (the server's
    /// hand-over of a stopped request to a batch slot) reuses it instead of reading them again.
    int spec_step(int x_tok, const std::vector<int32_t>& fix, int D, std::vector<int32_t>& drafts, int32_t* picks,
                  const std::atomic<bool>* cancel = nullptr) {
        const int N = (int) m->g.n_embd;
        if (main_hybrid) D = std::min<int>(D, (int) (kRebalanceEvery - pos % kRebalanceEvery) - 1);
        const int T0 = (int) fix.size() + 1;   // the redo rows and the row of x
        CK(cudaStreamSynchronize(st));
        // rows of the first MTP call: (hmain row i, fix[i]) at positions pos - fix.size() + i, then (hlast, x) at pos
        for (int i = 0; i < (int) fix.size(); ++i) hpin[16 + i] = fix[(size_t) i];
        hpin[16 + fix.size()] = x_tok;
        hpin[1] = (int32_t) (pos - (int64_t) fix.size());
        CK(cudaMemcpyAsync(tok_m, hpin + 16, (size_t) T0 * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(pos0_m, hpin + 1, 4, cudaMemcpyHostToDevice, st));
        // seeds: hmain rows 0 .. fix - 1 are the h of the positions before the fixed rows; hlast is that before x
        float* seeds = mtp_seed();
        if (!fix.empty())
            CK(cudaMemcpyAsync(seeds, hmain, fix.size() * (size_t) N * 4, cudaMemcpyDeviceToDevice, st));
        CK(cudaMemcpyAsync(seeds + fix.size() * (size_t) N, hlast, (size_t) N * 4, cudaMemcpyDeviceToDevice, st));
        forward_mtp(T0, pos - (int64_t) fix.size(), seeds, D > 0, false);
        drafts.assign((size_t) D, 0);
        if (D > 0) {
            CK(cudaMemcpyAsync(hpin + 2, pick_m, 4, cudaMemcpyDeviceToHost, st));
            CK(cudaStreamSynchronize(st));
            drafts[0] = hpin[2];
        }
        int last_row = T0 - 1;   // the row of the previous MTP call whose output seeds the next draft
        for (int d = 1; d < D; ++d) {   // chained: (the block's own output, the last draft) at the next position
            hpin[0] = drafts[(size_t) d - 1];
            hpin[1] = (int32_t) (pos + d);
            CK(cudaMemcpyAsync(tok_m, hpin, 4, cudaMemcpyHostToDevice, st));
            CK(cudaMemcpyAsync(pos0_m, hpin + 1, 4, cudaMemcpyHostToDevice, st));
            CK(cudaMemcpyAsync(seeds, hmtp + (size_t) last_row * N, (size_t) N * 4, cudaMemcpyDeviceToDevice, st));
            last_row = 0;
            if (mtp_graph) graph_run(gs().gm1, [&] { forward_mtp(1, pos + d, seeds, true, use_graph); });
            else forward_mtp(1, pos + d, seeds, true, false);
            CK(cudaMemcpyAsync(hpin + 2, pick_m, 4, cudaMemcpyDeviceToHost, st));
            CK(cudaStreamSynchronize(st));
            drafts[(size_t) d] = hpin[2];
        }
        // the verify window
        const int T = D + 1;
        hpin[16] = x_tok;
        for (int d = 0; d < D; ++d) hpin[17 + d] = drafts[(size_t) d];
        hpin[1] = (int32_t) pos;
        CK(cudaMemcpyAsync(tok, hpin + 16, (size_t) T * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(pos0, hpin + 1, 4, cudaMemcpyHostToDevice, st));
        rows_deferred = true;
        graph_run(gs().gv[T], [&] { forward(T, pos, T, use_graph, true); });
        rows_deferred = false;
        CK(cudaMemcpyAsync(hpin + 2, pick, (size_t) T * 4, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
        int a = 0;
        while (a < D && hpin[2 + a] == drafts[(size_t) a]) ++a;
        if (cancel != nullptr && cancel->load()) a = 0;
        for (int i = 0; i <= a; ++i) picks[i] = hpin[2 + i];
        commit(a + 1);
        CK(cudaMemcpyAsync(hlast, hmain + (size_t) a * N, (size_t) N * 4, cudaMemcpyDeviceToDevice, st));
        pos += a + 1;
        after_rows(a + 1);
        return a;
    }

    /// The first prompt after load ran at about half speed (pp4096 rep 0: ~1770 tok/s against ~3290): the GPU's code
    /// objects, the BLAS handles' kernels and the decode graph are made on first use.  A throw-away prompt of one
    /// chunk, a decode step and (with drafts) a speculative step at load do that before the first request; then the
    /// state, the cache position and the hybrid counters are as after load.
    bool warming = false;
    void warmup() {
        const auto t0 = Clock::now();
        warming = true;
        const int32_t filler = 198;   // a newline: any token works
        // Every phase starts from a reset position: the K / V cache holds `ctx` cells exactly, so a phase that runs
        // past it (a chunk of `tmax` = 2048 plus a decode, on a 2048-cell cache) wrote past the cache and faulted in
        // k_attn_prep_m2.  The prompt chunk is clamped to `ctx - 1` for the same reason.
        const int64_t room = std::max<int64_t>(1, (int64_t) ctx - 1);
        const int64_t n_chunk = std::min<int64_t>((int64_t) tmax, room);
        std::vector<int32_t> toks((size_t) n_chunk, filler);
        process(toks.data(), (int64_t) toks.size(), [](int64_t) {});
        reset();
        if (n_chunk > kGemmT + 1) {
            process(toks.data(), kGemmT + 1, [](int64_t) {});   // the smallest GEMM chunk too
            reset();
        }
        process(&filler, 1, [](int64_t) {});
        if (mtp_d > 0) {
            std::vector<int32_t> fix, drafts;
            int32_t picks[16];
            spec_step(filler, fix, mtp_d, drafts, picks);
        }
        CK(cudaStreamSynchronize(st));
        warming = false;
        for (Hy& H : hy) std::fill(H.score.begin(), H.score.end(), 0.0);
        hy_cpu = hy_gpu = hy_swaps = 0;
        hy_peer = 0;
        hy_cpu_ms = 0;
        hy_wait_ms = 0;
        reset();
        std::fprintf(stderr, "strata-m2: warmed up in %.1f s (one prompt chunk, a decode step%s)\n",
                     ms_since(t0) / 1000.0, mtp_d > 0 ? ", a speculative step" : "");
    }

    int greedy() {
        CK(cudaMemcpyAsync(hpin + 2, pick, 4, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
        return hpin[2];
    }
    void host_logits(std::vector<float>& out) {
        out.resize((size_t) m->g.n_vocab);
        CK(cudaMemcpyAsync(out.data(), logits, out.size() * 4, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
    }
};

// ------------------------------------------------------------------------------------------------ expert placement
/// The routed experts after the engine's buffers: all in VRAM when they fit in `vram_gib` (< 0: the free VRAM less a
/// reserve), else the hybrid mode - the experts in host memory (all of them, or the first ones up to `host_gib`, see
/// --resident-budget-gib; the others are read where they lie in the file), the same number of VRAM slots per layer, a
/// decode batch's other experts on the CPU pool and a prompt chunk's streamed through two staging buffers.
bool place_experts(Model& m, Engine& E, double vram_gib, int threads, int hybrid_t, int slots, double host_gib,
                   size_t reserve, size_t reserve_later, bool reserve_given, std::string& err) {
    const auto t0 = Clock::now();
    const auto& g = m.g;
    const int64_t ne = g.n_expert;
    std::vector<Layer*> all;
    for (size_t i = 0; i < m.layers.size(); ++i) {
        // the layer split: the second card's layers are the ones the engine runs, and the placement fills
        // THEIR expert slots (on their card)
        Layer* L = (E.split_dev >= 0 && (int) i >= E.split_layer && i - (size_t) E.split_layer < m.layers2.size())
                       ? &m.layers2[i - (size_t) E.split_layer]
                       : &m.layers[i];
        all.push_back(L);
    }
    if (m.has_mtp) all.push_back(&m.mtp);
    size_t total = 0, per_slot = 0, per_slot0 = 0, per_slot1 = 0, max_layer = 0;
    for (size_t li = 0; li < all.size(); ++li) {
        Layer* L = all[li];
        total += L->stride * (size_t) ne + kMmqTail;
        per_slot += L->stride;
        // The layer split: each card's slots come out of ITS free VRAM, so the slots per layer follow the layers that
        // card runs.  The sum over all of them was divided between the cards by nothing: the RX 6800 and the RX 9070
        // got 19 slots per layer where 38 fit each (the second card's half of the strides was charged to the first's
        // budget too), and the whole model's experts stayed a decode's CPU share away from the card that holds them.
        if (E.split_dev >= 0 && (int) li >= E.split_layer) per_slot1 += L->stride;
        else per_slot0 += L->stride;
        max_layer = std::max(max_layer, L->stride * (size_t) ne);
    }
    size_t fr = 0, tot = 0;
    CK(cudaMemGetInfo(&fr, &tot));
    // `reserve` is the first card's, `reserve_later` the split's / the peer's card (--vram-reserve-mib and
    // --vram-reserve-later-mib; the caller has already defaulted the second to the first).
    // The K / V cache of the card's layers and the engine's buffers are allocated by init(), BEFORE this call: `fr`
    // already leaves them out, so the reserve covers only what the hybrid mode allocates next (the staging pair, the
    // CPU part's outputs, the slots - checked with the numbers just below).  It used to subtract the card's K / V a
    // second time (8.3 GiB at 32K on two 16 GB cards), which left 0 expert slot per layer.
    const size_t budget = vram_gib >= 0 ? (size_t) (vram_gib * (double) (1ull << 30))
                                        : (fr > reserve ? fr - reserve : 0);
    // every expert fits once the packed Q8_0 decode copies are given back (a file a few GiB larger than the card's
    // room, UD-Q6_K_XL on 32 GB): drop them - the decode reads the GGUF layout then, bitwise the same outputs, a little
    // slower - rather than run the hybrid mode with its CPU share and streamed prompt layers.  STRATA_Q36_KEEP_PACKED=1
    // keeps them (and the hybrid mode).
    size_t packed_bytes = 0;
    for (const auto& pk : m.packed) packed_bytes += std::get<2>(pk);
    size_t main_need = 0;   // the main layers' experts (the MTP block's can go to the CPU, see below)
    for (Layer& L : m.layers) main_need += L.stride * (size_t) ne + kMmqTail;
    const size_t slack = ((size_t) 512 << 20) + (E.kAttScoreFloats - ((size_t) 16 << 20)) * 6;   // reserve + scores
    const bool mtp_cpu_ok = m.has_mtp && std::getenv("STRATA_Q36_NO_MTP_CPU") == nullptr;
    if (slots <= 0 && vram_gib < 0 && total > budget && packed_bytes > 0 &&
        (total <= budget + packed_bytes || (mtp_cpu_ok && main_need <= budget + packed_bytes + slack)) &&
        std::getenv("STRATA_Q36_KEEP_PACKED") == nullptr) {
        CK(cudaDeviceSynchronize());
        for (const auto& pk : m.packed) {
            K::native_q8_0_packed_unregister(std::get<0>(pk));
            CK(cudaFree(std::get<1>(pk)));
        }
        m.packed.clear();
        E.reset_graphs();
        CK(cudaMemGetInfo(&fr, &tot));
        std::fprintf(stderr, "strata-m2: the packed Q8_0 decode copies (%.2f GiB) given back so that every routed "
                     "expert fits in VRAM (STRATA_Q36_KEEP_PACKED=1 keeps them and runs the hybrid mode)\n",
                     (double) packed_bytes / (1 << 30));
    }
    // The MMQ buffers only matter to a layer whose experts all sit in one block, and a hybrid layer never takes that
    // path (see the prompt path in layer()).  The experts already do not fit in VRAM (`total > fr`), so the placement
    // is going to be hybrid: free them HERE, before the slot budget is cut, and measure again so the slots take the
    // room (~386 MiB at chunk 2048 on this model: three experts a layer).  A machine whose experts all fit is not
    // hybrid and keeps MMQ.
    if (E.use_mmq && total > fr) {
        E.drop_mmq();
        CK(cudaMemGetInfo(&fr, &tot));
        std::fprintf(stderr, "strata-m2: the experts do not fit in VRAM, so a prompt layer runs on the native kernels: "
                             "the MMQ buffers are freed and their VRAM goes to the slots\n");
    }
    size_t budget2 = vram_gib >= 0 ? budget : (fr > reserve ? fr - reserve : 0);
    // the MAIN layers' experts fit, the MTP block's not all (UD-Q6_K_XL on 32 GB): the main layers all in VRAM - the
    // text is the all-in-VRAM text, the decode graph stays - and the MTP block's experts on slots + the CPU pool (its
    // drafts only).  When that needs it: the attention scores buffer at 64 MB instead of 256 MB (more, smaller
    // slices in a long prompt) and the reserve at 256 MiB instead of 768 (said in the log; --expert-vram sets the
    // budget instead).  STRATA_Q36_NO_MTP_CPU=1: the plain hybrid mode for every layer.
    size_t total_main = 0;
    for (Layer& L : m.layers) total_main += L.stride * (size_t) ne + kMmqTail;
    bool mtp_only = false;
    if (slots <= 0 && vram_gib < 0 && m.has_mtp && total > budget2 && std::getenv("STRATA_Q36_NO_MTP_CPU") == nullptr &&
        !reserve_given) {
        const size_t att_save = (E.kAttScoreFloats - ((size_t) 16 << 20)) * 6;   // att_s (4 B) + att_p16 (2 B)
        const size_t small_reserve = (size_t) 256 << 20;
        const size_t room = fr + att_save > small_reserve ? fr + att_save - small_reserve : 0;
        if (total_main + ((size_t) 64 << 20) <= budget2) {
            mtp_only = true;
        } else if (total_main + ((size_t) 64 << 20) <= room) {
            E.shrink_att((size_t) 16 << 20);
            CK(cudaMemGetInfo(&fr, &tot));
            budget2 = fr > small_reserve ? fr - small_reserve : 0;
            mtp_only = total_main + ((size_t) 64 << 20) <= budget2;
            std::fprintf(stderr, "strata-m2: the experts nearly fit: the attention scores buffer at 64 MB and the VRAM "
                         "reserve at 256 MiB (instead of 768) so that every main layer's experts fit; a driver or "
                         "another program needing more VRAM later can make it fail - --expert-vram GiB sets the experts' "
                         "budget instead\n");
        }
    }
    if (slots <= 0 && total <= budget2 && total + ((size_t) 256 << 20) <= fr) {
        std::vector<uint8_t> blob;
        for (Layer* L : all) {
            CK(cudaMalloc(&L->blobs, L->stride * (size_t) ne + kMmqTail));
            CK(cudaMemset(L->blobs, 0, L->stride * (size_t) ne + kMmqTail));
            blob.assign(L->L.bytes, 0);
            for (int64_t e = 0; e < ne; ++e) {
                expert_blob(*L, e, blob.data());
                CK(cudaMemcpy(L->blobs + (size_t) e * L->stride, blob.data(), L->L.bytes, cudaMemcpyHostToDevice));
            }
            L->n_slots = (int) ne;
        }
        std::fprintf(stderr, "strata-m2: every routed expert in VRAM (%.2f GiB) in %.1f s\n", (double) total / (1 << 30),
                     ms_since(t0) / 1000.0);
        return true;
    }
#if !defined(STRATA_Q36_CPU_POOL)
    (void) threads;
    (void) hybrid_t;
    err = "the routed experts (" + std::to_string(total >> 20) + " MiB) do not fit in VRAM, and this build has no CPU "
          "expert pool (STRATA_NATIVE_EXPERTS=OFF)";
    return false;
#else
    namespace C = strata::kernels::cpu;
    m.hybrid = true;
    E.main_hybrid = !mtp_only;
    if (E.main_hybrid) E.use_graph = false;   // the host takes part in every layer
    E.mtp_graph = false;                      // the MTP block's layer is hybrid either way
    E.hybrid_t = std::max(1, std::min(hybrid_t, E.use_mmq ? E.tmax : E.tmax));
    if (!E.use_mmq && !E.stream_native) E.hybrid_t = E.tmax;   // no streamed prompt path: every batch on slots + the pool
    // one pair per card: each card's layers stream into ITS staging (a kernel cannot read the other card's).
    // In the default (pinned host) mode the pair costs no VRAM at all: the slots keep their share, and the decode
    // does not pay for the prompt's streaming.
    const size_t stream_bytes = (E.use_mmq || E.stream_native) && E.main_hybrid ? 2 * (max_layer + kMmqTail) : 0;
    const size_t staging_card = E.stream_vram ? stream_bytes / 2 : 0;   // VRAM per card (0: pinned host)
    size_t slot_bytes = budget2 > staging_card ? budget2 - staging_card : 0;
    if (slot_bytes + staging_card + ((size_t) 256 << 20) > fr) slot_bytes = fr > staging_card + ((size_t) 512 << 20) ? fr - staging_card - ((size_t) 512 << 20) : 0;
    // --expert-slots: that many per layer (tests: the same split with and without the MTP block).
    // A NEGATIVE count means NO slot at all: every expert goes to the CPU pool.  It is what takes the slots
    // out of the picture when the answer has to be compared with a reference (the placement reads 0 as "the
    // VRAM budget", so 0 could not do it).
    int S = slots < 0 ? 0 : (slots > 0 ? (int) std::min<int64_t>(ne, slots)
                                          : (int) std::min<size_t>((size_t) ne, slot_bytes / per_slot0));
    // The peer's own budget: the slots it takes are on ITS card, with its own free VRAM.  It holds the experts
    // the primary does not, so the two tiers together hold S + S_peer of the layer's experts.
    int S_peer = 0;
    int S_split = -1;   // the split card's slots per layer, when the peer shares that card
    if (E.peer_device >= 0) {
        size_t pfr = 0, ptot = 0, pslot = 0;
        CK(cudaSetDevice(E.peer_device));
        CK(cudaMemGetInfo(&pfr, &ptot));
        CK(cudaGetLastError());
        // The reserve on the peer's card: the first card's rule when the peer sits on the first card, the later
        // card's when it is the second one (--vram-reserve-later-mib).
        const size_t prsv = E.peer_device == 0 ? reserve : reserve_later;
        // The peer's card can also hold expert slots: the layer split's layers (their own slots) or, with the peer on
        // the primary, the primary's.  They come out of the same VRAM, and the free room is measured HERE, before
        // either tier is placed.  Charging the peer alone overcommitted the card (14.00 GiB of peer beside 2.30 GiB
        // of the split's slots on a 16 GB card here), the driver then served the VRAM from host memory, and a kernel
        // read a page that was not there (HSA_STATUS_ERROR_MEMORY_FAULT in k_route_group_m2, GPU 1, 2026-10-26).
        size_t slots_here = 0;
        if (E.split_dev >= 0 && E.peer_device == E.split_dev) {
            const size_t room = pfr > prsv ? pfr - prsv : 0;
            if (slots < 0) S_split = 0;
            else if (slots > 0) S_split = (int) std::min<int64_t>((int64_t) ne - (int64_t) S, slots);
            else   // no --expert-slots: the same count on both tiers, so their bytes add up to the room
                S_split = (int) std::min<int64_t>((int64_t) ne - (int64_t) S,
                                                  (int64_t) (room / (per_slot1 + per_slot)));
            slots_here = (size_t) S_split * per_slot1;
        } else if (E.peer_device == 0) {
            slots_here = (size_t) S * per_slot0;
        }
        pslot = pfr > (prsv + slots_here + (size_t) (1ull << 30))
                    ? pfr - prsv - slots_here - (size_t) (1ull << 30)
                    : 0;
        CK(cudaSetDevice(0));
        CK(cudaGetLastError());
        // The peer holds the experts after the slots on its OWN card: the first card's S, plus the split card's
        // S_split when the peer shares that card.  Starting at S alone gave the peer the same first 24 experts the
        // split card's slots hold: loaded twice, computed twice (the merge keeps the layer's rows, so the peer's work
        // there was wasted).
        const int peer_first = (E.split_dev >= 0 && E.peer_device == E.split_dev) ? (int) (S + S_split) : (int) S;
        S_peer = (int) std::min<size_t>((size_t) std::max<int64_t>(0, ne - peer_first), pslot / per_slot);
        std::fprintf(stderr, "strata-m2: the peer (device %d) takes %d of %d experts per layer (%.2f GiB), from expert "
                     "%d\n", E.peer_device, S_peer, ne, (double) (S_peer * per_slot) / (1 << 30), peer_first);
    }
    if (mtp_only) {   // the main layers all, the MTP block what is left
        const size_t left = budget2 > total_main ? budget2 - total_main : 0;
        S = (int) std::min<size_t>((size_t) ne, left / m.mtp.stride);
    }
    {   // what the hybrid mode allocates next (the staging pair, the CPU part's outputs, the slots) against the VRAM the
        // sessions and the context left: said with the numbers instead of a cudaMalloc failure (8 slots on a 12 GB card)
        const size_t Tn = (size_t) E.tmax, Nn = (size_t) g.n_embd, Kn = (size_t) m.top_k;
        size_t need = staging_card + Tn * Kn * Nn * 4 + Tn * Kn * 4;
        for (size_t li = 0; li < all.size(); ++li) {
            Layer* L = all[li];
            // the layer split: the second card's slots are on ITS card, checked where it computes them
            if (E.split_dev >= 0 && (int) li >= E.split_layer) continue;
            if (mtp_only && L != &m.mtp) need += L->stride * (size_t) ne + kMmqTail;
            else need += (S > 0 ? (size_t) S * L->stride + kMmqTail : 0);
        }
        if (need > fr) {
            err = "not enough VRAM left for the hybrid experts' buffers and slots (" + std::to_string((need >> 20)) +
                  " MiB needed, " + std::to_string(fr >> 20) + " MiB free after the weights, the context and the sessions): use "
                  "fewer slots (\"parallel\" / --batch), a shorter --max-context" +
                  (slots > 0 ? ", or fewer --expert-slots" : "");
            return false;
        }
    }
    // host memory: the POOL's experts -- the ones without a VRAM slot -- pinned where the driver allows it, up to
    // --resident-budget-gib.  The low-RAM mode keeps NO copy: the experts stay in the mapping and are read where they
    // lie.  The slots are placed in the loop below and the pool is what follows them, so each layer's arena is
    // allocated THERE, sized on that layer's pool.  A per-layer share of the budget (cap / layers / stride) sized the
    // arena on the LAYER instead: it came to 158 of card 0's 166 pooled experts, so 8 a layer were still read out of
    // the file at every step, and the 8.4 GiB that no expert used were allocated and pinned all the same.
    if (std::getenv("STRATA_M2_PLACE_TRACE")) {
        std::fprintf(stderr, "PLACE S0=%zu slot_bytes=%.2f GiB per_slot0=%.2f MiB per_slot1=%.2f MiB\n",
                     (size_t) S, (double) slot_bytes / (1 << 30), (double) per_slot0 / 1048576.0,
                     (double) per_slot1 / 1048576.0);
        std::fflush(stderr);
    }
    const size_t host_cap = E.low_ram ? 0
                                     : host_gib > 0 ? (size_t) (host_gib * (double) (1ull << 30)) : (size_t) -1;
    size_t host = 0;      // the bytes pinned so far (the budget's running total)
    {   // the low-RAM mode's repack buffers: one expert's blob, out of the mapping (before the slots' copy,
        // which uses it).  Pinned: their copies are asynchronous (see the members)
        size_t max_bytes = 0;
        for (Layer* L : all) max_bytes = std::max(max_bytes, L->L.bytes);
        CK(cudaHostAlloc((void**) &E.repack, max_bytes, cudaHostAllocDefault));
        CK(cudaHostAlloc((void**) &E.repack_pf, max_bytes, cudaHostAllocDefault));
    }
    E.hy.resize(all.size());
    if (E.peer_device >= 0) E.peer_hy.resize(all.size());
    int place_dev = 0;   // the card the layer's resources are on (the layer split)
    int slot_off = 0;    // the first expert this card holds: the primary takes [0, S), the second card the rest
    const int S_first = S;   // the primary's slots (the layer loop replaces S with the second card's, below)
    for (size_t li = 0; li < all.size(); ++li) {
        Layer& L = *all[li];
        if (std::getenv("STRATA_M2_PLACE_TRACE")) {   // DIAG: where the placement is, its slots and the layer's expert
            std::fprintf(stderr, "PLACE li=%zu S=%d slot_off=%d stride=%.2f MiB bytes=%zu\n", li, S, slot_off,
                         (double) L.stride / 1048576.0, (size_t) L.L.bytes);
            std::fflush(stderr);
        }
        // the layer split: the layer's resources (its expert slots) live on its card, with ITS free VRAM
        if (E.split_dev >= 0) {
            const int ldev = (int) li >= E.split_layer ? E.split_dev : 0;
            if (ldev != place_dev) {
                CK(cudaSetDevice(ldev));
                CK(cudaGetLastError());
                place_dev = ldev;
                size_t f = 0, t = 0;
                CK(cudaMemGetInfo(&f, &t));
                CK(cudaGetLastError());
                // the second card's own room: its K / V and its buffers are already allocated too (see above)
                const size_t srsv = ldev == 0 ? reserve : reserve_later;
                size_t sb = vram_gib >= 0 ? budget : (f > srsv ? f - srsv : 0);
                if (sb + staging_card + ((size_t) 256 << 20) > f)
                    sb = f > staging_card + ((size_t) 512 << 20) ? f - staging_card - ((size_t) 512 << 20) : 0;
                slot_off = S;   // the second card takes the experts the primary does not hold
                // ...and only those: its own slots per layer can be MORE than the primary's (its VRAM is free while the
                // primary's holds its share of the sessions), and S + slot_off used to be able to pass the count of a
                // layer's experts - H.slot_of / exp_in_slot are ne entries long, and writing past them corrupted the
                // heap the placement allocates from.
                const int64_t left = (int64_t) ne - (int64_t) slot_off;
                if (S_split >= 0)   // the peer shares this card: its budget above already kept room for these slots
                    S = (int) std::min<int64_t>(left, S_split);
                else
                    S = slots < 0 ? 0
                                 : (slots > 0 ? (int) std::min<int64_t>(left, slots)
                                              : (int) std::min<int64_t>(left, (int64_t) (sb / per_slot1)));
                std::fprintf(stderr, "strata-m2: the second card takes %d of %d experts per layer (%.2f GiB free)\n",
                             S, ne, (double) f / (double) (1ull << 30));
            }
        }
        if (mtp_only && &L != &m.mtp) {   // a main layer: every expert in VRAM, as in the all-in-VRAM placement
            CK(cudaMalloc(&L.blobs, L.stride * (size_t) ne + kMmqTail));
            CK(cudaMemset(L.blobs, 0, L.stride * (size_t) ne + kMmqTail));
            std::vector<uint8_t> blob(L.L.bytes, 0);
            for (int64_t e = 0; e < ne; ++e) {
                expert_blob(L, e, blob.data());
                CK(cudaMemcpy(L.blobs + (size_t) e * L.stride, blob.data(), L.L.bytes, cudaMemcpyHostToDevice));
            }
            L.n_slots = (int) ne;
            continue;
        }
        L.hblobs = nullptr;   // the pool's copy, allocated below once this layer's slots are known
        L.h_first = 0;
        L.h_resident = 0;
        L.n_slots = S;
        if (S > 0) {
            CK(cudaMalloc(&L.blobs, L.stride * (size_t) S + kMmqTail));
            CK(cudaMemset(L.blobs, 0, L.stride * (size_t) S + kMmqTail));
            // ONE EXPERT AT A TIME: the slot's content is repacked out of the mapping into the host buffer, then
            // copied.  One expert (4 MB) is a DMA; a whole layer's run (48 experts, ~200 MB) out of a big pinned
            // arena went through the driver's bounce buffer at 5 MB/s on this box.
            for (int s = 0; s < S; ++s) {
                const int e = slot_off + s;   // the expert this slot holds (the second card's are the ones the first does not)
                expert_blob(L, e, E.repack);
                CK(cudaMemcpy(L.blobs + (size_t) s * L.stride, E.repack, L.L.bytes, cudaMemcpyHostToDevice));
            }
        }
        Engine::Hy& H = E.hy[li];
        H.slot_of.assign((size_t) ne, -1);
        H.exp_in_slot.assign((size_t) S, 0);
        for (int s = 0; s < S; ++s) {
            H.slot_of[(size_t) (slot_off + s)] = s;
            H.exp_in_slot[(size_t) s] = slot_off + s;
        }
        // The peer's slots: the experts the primary does not hold, on the peer's card.
        if (S_peer > 0) {
            Engine::PeerHy& P = E.peer_hy[li];
            P.slot_of.assign((size_t) ne, -1);
            P.slot_of_d = nullptr;   // until this layer's slots are placed: moe_hybrid_peer skips a layer without one
            CK(cudaSetDevice(E.peer_device));
            // The card's real free VRAM, which can be less than the budget said (another program, the display):
            // the layer takes as many slots as fit, and the ones that do not fit go to the CPU pool.
            size_t pfr = 0, ptot = 0;
            CK(cudaMemGetInfo(&pfr, &ptot));
            CK(cudaGetLastError());
            const size_t want = L.stride * (size_t) S_peer + kMmqTail;
            if (want > pfr) {
                const int fit = (int) std::max<size_t>(0, (pfr - (64u << 20)) / L.stride);
                std::fprintf(stderr, "strata-m2: the peer takes %d of %d experts per layer here (%.2f GiB free)\n",
                             fit, ne, (double) pfr / (double) (1ull << 30));
                S_peer = fit;
            }
            // A layer the peer has no room for keeps NO expert of it -- the table stays all -1, slot_of_d stays null
            // and moe_hybrid_peer skips the layer -- and its experts go to the CPU pool.  This used to `continue`,
            // which also skipped everything the layer needs for itself: H.score stayed empty (the CPU pool's
            // `H.score[x] += 1.0` then wrote past it), H.slot_of_d stayed null (k_group_pool / group_entries read
            // it) and H.fmt stayed zeroed (native_fmt below builds it; the CPU pool and the stream's offsets use it).
            if (S_peer > 0) {
                CK(cudaMalloc(&L.peer_blobs, L.stride * (size_t) S_peer + kMmqTail));
                CK(cudaMemset(L.peer_blobs, 0, L.stride * (size_t) S_peer + kMmqTail));
                for (int s = 0; s < S_peer; ++s) {
                    // after the slots on the PEER's own card: the split card's when the peer shares it, the first
                    // card's otherwise (see peer_first above)
                    const int e = ((E.peer_device == place_dev) ? slot_off + S : S_first) + s;
                    // out of the mapping (the host copy below is the POOL's, and this expert is on a card)
                    expert_blob(L, e, E.repack);
                    CK(cudaMemcpy(L.peer_blobs + (size_t) s * L.stride, E.repack, L.L.bytes, cudaMemcpyHostToDevice));
                    P.slot_of[(size_t) e] = s;
                    if (li == 0 && getenv("STRATA_M2_DUMP_ROWS")) {   // DIAG: the peer's slot content
                        static std::vector<uint8_t> want, got;
                        want.resize(L.L.bytes);
                        got.resize(L.L.bytes);
                        for (int s2 = 0; s2 < S_peer && s2 < 4; ++s2) {
                            const int e2 = e + (s2 - s);
                            expert_blob(L, e2, want.data());
                            CK(cudaMemcpy(got.data(), L.peer_blobs + (size_t) s2 * L.stride, L.L.bytes, cudaMemcpyDeviceToHost));
                            long nd = 0, fi = -1;
                            for (size_t z = 0; z < want.size(); ++z)
                                if (want[z] != got[z]) { if (fi < 0) fi = (long) z; ++nd; }
                            std::fprintf(stderr, "SLOT %d (expert %d) : %ld differing bytes (first at %ld of %zu)\n", s2, e2, nd, fi, want.size());
                        }
                        std::fflush(stderr);
                    }
                }
                CK(cudaMalloc(&P.slot_of_d, (size_t) ne * 4));
                CK(cudaMemcpy(P.slot_of_d, P.slot_of.data(), (size_t) ne * 4, cudaMemcpyHostToDevice));
                CK(cudaSetDevice(0));
                CK(cudaGetLastError());
                // the primary's kernels read the peer's table too (to tell the peer's entries from the CPU's)
                if (!E.peer_slot_of_d) CK(cudaMalloc(&E.peer_slot_of_d, (size_t) ne * 4));
                CK(cudaMemcpy(E.peer_slot_of_d, P.slot_of.data(), (size_t) ne * 4, cudaMemcpyHostToDevice));
            }
            // The rest of the layer is the layer's OWN card: 0, or the split card for the layers it runs.  This used to
            // set 0: a split layer's slot table (and the next split layers' blobs) was then allocated on the OTHER
            // card, and the layer's routing kernel read it on its own card (HSA_STATUS_ERROR_MEMORY_FAULT in
            // k_route_group_m2 on GPU 1, 2026-10-26).
            CK(cudaSetDevice(place_dev));
            CK(cudaGetLastError());
        }
        H.score.assign((size_t) ne, 0.0);
        CK(cudaMalloc(&H.slot_of_d, (size_t) ne * 4));
        CK(cudaMemcpy(H.slot_of_d, H.slot_of.data(), (size_t) ne * 4, cudaMemcpyHostToDevice));
        // The pinned arena holds the POOL's experts -- the ones neither card holds -- and it is allocated here, once
        // the slots are placed: the pool's range is known only now.  It follows the slots ([0, S) on card 0,
        // [S, S + S_peer) on card 1), so the arena costs the pool's bytes, not the whole layer's: a per-layer share of
        // the budget used to be the size of the LAYER, which pinned fewer experts than the pool on some layers and
        // experts the second card already held on others.  What the budget leaves covers the pool's first experts; the
        // others are read out of the mapping, as before.
        int pool_first = 0;
        for (int x = 0; x < ne; ++x)
            if (H.slot_of[(size_t) x] >= 0) pool_first = x + 1;
        if (E.peer_device >= 0 && li < E.peer_hy.size() && !E.peer_hy[li].slot_of.empty())
            for (int x = 0; x < ne; ++x)
                if (E.peer_hy[li].slot_of[(size_t) x] >= 0) pool_first = x + 1;
        const size_t left = host_cap == (size_t) -1 ? (size_t) -1 : host_cap > host ? host_cap - host : 0;
        const int pool = (int) ne - pool_first;
        const int take = (pool > 0 && left > kMmqTail)
                             ? (int) std::min<size_t>((size_t) pool, (left - kMmqTail) / L.stride)
                             : 0;
        int want = take;
        size_t bytes = want > 0 ? (size_t) want * L.stride + kMmqTail : 0;
        // The arena must be PINNED: the pool reads it at every token and the prompt's stream takes it from RAM (the
        // asynchronous copies below need a pinned source), so a refused allocation takes fewer experts instead of
        // falling back to pageable memory, which is both slower than the file and unsafe to copy from asynchronously.
        while (want > 0 && cudaHostAlloc((void**) &L.hblobs, bytes, cudaHostAllocDefault) != cudaSuccess) {
            (void) cudaGetLastError();
            L.hblobs = nullptr;
            want /= 2;
            bytes = want > 0 ? (size_t) want * L.stride + kMmqTail : 0;
        }
        if (want > 0) {
            L.h_first = pool_first;
            L.h_resident = want;
            host += bytes;
            std::memset(L.hblobs, 0, bytes);
            for (int64_t e = L.h_first; e < (int64_t) L.h_first + want; ++e) expert_blob(L, e, L.host_blob(e));
        }
        if (!C::native_fmt(L.L.gu_type, L.L.d_type, g.n_embd, g.n_ff, H.fmt, err)) return false;
        if (H.fmt.up_off != L.L.up_off || H.fmt.down_off != L.L.down_off || H.fmt.bytes != L.L.bytes) {
            err = "the CPU and GPU expert layouts differ";
            return false;
        }
    }
    const size_t T = (size_t) E.tmax, N = (size_t) g.n_embd, Kp = (size_t) m.top_k;
    // the host buffers the two cards share (the pool's rows, its activations): PORTABLE, so a stream on either
    // card can copy from them (a pinned buffer bound to one device cannot be used by the other)
    CK(cudaHostAlloc((void**) &E.h_h, T * N * 4, cudaHostAllocDefault | cudaHostAllocPortable));
    // the crossing's staging: the rows go through the host unless the peer copy is asked for
    E.split_copy_peer = std::getenv("STRATA_M2_SPLIT_COPY") != nullptr &&
                        std::string(std::getenv("STRATA_M2_SPLIT_COPY")) == "peer";
    CK(cudaHostAlloc((void**) &E.split_stage, (2 * T * N + T + 1) * 4, cudaHostAllocDefault | cudaHostAllocPortable));
    // the host rows the CPU pool writes: with a peer, PORTABLE, so the peer card writes into the same
    // addresses (the peer's rows and the CPU's rows meet in the same buffer, and the merge picks both up)
    CK(cudaHostAlloc((void**) &E.cpu_out, T * Kp * N * 4, cudaHostAllocDefault | cudaHostAllocPortable));
    if (E.peer_device >= 0) {
        CK(cudaSetDevice(E.peer_device));
        CK(cudaMalloc(&E.peer_row, T * Kp * 4));
        CK(cudaStreamCreateWithFlags(&E.peer_st, cudaStreamNonBlocking));
        CK(cudaEventCreateWithFlags(&E.peer_done, cudaEventDisableTiming));
        const size_t nmax = std::min<size_t>(T * Kp, (size_t) g.n_expert);
        CK(cudaMalloc(&E.peer_grp_ptr, (nmax + 1) * sizeof(unsigned long long)));
        CK(cudaMalloc(&E.peer_grp_start, (nmax + 1) * 4));
        CK(cudaMalloc(&E.peer_n_groups, 4));
        // The ENTRIES' tables (ent_dst, ent_tok): the expert kernels index them by ENTRY, not by the dense rows,
        // so they hold T * Kp entries, as the primary's do.  The GROUPS' tables (grp_ptr, grp_start) hold one
        // entry per expert.
        CK(cudaMalloc(&E.peer_ent_dst, T * Kp * 4));
        CK(cudaMalloc(&E.peer_ent_tok, T * Kp * 4));
        E.peer_rows_max = T * Kp;
        CK(cudaMalloc(&E.peer_ids, T * Kp * 4));
        CK(cudaMalloc(&E.peer_xq, K::native_q8_1_bytes((int) N, 1) * T));
        // The scratch is sized the way native_expert_grouped sizes it: FOUR buffers of cap_entries x n_ff floats
        // (gate, up, h, hq) and n_ff is the LAYER's (the expert's), not the geometry's.
        int64_t ff_max = 0;
        for (const Layer* L2 : all) ff_max = std::max<int64_t>(ff_max, L2->L.n_ff);
        E.peer_escr = dalloc(K::native_expert_scratch_bytes((int64_t) (T * Kp), ff_max));
        // The peer's rows go STRAIGHT into the host rows (the portable pinned buffer): the expert kernels write
        // them at the entries' positions, which is where the CPU pool's rows and the merge expect them.
        E.peer_eparts = E.cpu_out;
        CK(cudaSetDevice(0));
        CK(cudaGetLastError());
    }
    CK(cudaHostAlloc((void**) &E.h_ids, T * Kp * 4, cudaHostAllocDefault | cudaHostAllocPortable));
    CK(cudaHostAlloc((void**) &E.h_miss, T * Kp * 4, cudaHostAllocDefault | cudaHostAllocPortable));
    CK(cudaHostAlloc((void**) &E.h_cnt, all.size() * (size_t) ne * 4, cudaHostAllocDefault | cudaHostAllocPortable));

    E.acts = (uint8_t*) std::malloc(T * C::kNativeActBytes + 64);

    {   // the split's crossing event: on the FIRST card (the placement may have left the second card current)
        CK(cudaSetDevice(0));
        CK(cudaGetLastError());
        CK(cudaEventCreateWithFlags(&E.ev_cross, cudaEventDisableTiming));
    }
    if (stream_bytes) {
        if (!E.stream_vram) {   // pinned host: the GPU reads the pool's experts over PCIe, no VRAM, no copy stream
            for (int b = 0; b < 2; ++b) {
                CK(cudaHostAlloc((void**) &E.stage[b], max_layer + kMmqTail, cudaHostAllocPortable));
                std::memset(E.stage[b], 0, max_layer + kMmqTail);
                CK(cudaEventCreateWithFlags(&E.ev_copy[b], cudaEventDisableTiming));
                CK(cudaEventCreateWithFlags(&E.ev_used[b], cudaEventDisableTiming));
            }
            {   // the fill pool (see FillPool): the prompt's staging was filled by one thread, ~1.6 s a layer.
                // 16 threads is where the curve bends on a 16-core part (measured on a 512-token prompt, same staging
                // and same prompt each run: 1 thread 96.6 s, 6 -> 28.9 s, 12 -> 23.9 s, 16 -> 19.7 s, 24 -> 20.6 s).
                int nfill = 16;
                if (const char* v = std::getenv("STRATA_M2_FILL_THREADS")) nfill = std::atoi(v);
                const int cpus = (int) std::thread::hardware_concurrency();
                nfill = std::clamp(nfill, 0, std::max(1, cpus - 1));
                E.fill.start(nfill);
                if (nfill > 0)
                    std::fprintf(stderr, "strata-m2: the prompt's staged experts are copied by %d threads\n", nfill);
            }
            if (E.split_dev > 0) {   // the second card's "done reading" events (the buffers are shared, the events
                CK(cudaSetDevice(E.split_dev));   // are not: an event belongs to the card that creates it)
                CK(cudaGetLastError());
                for (int b = 0; b < 2; ++b) CK(cudaEventCreateWithFlags(&E.ev_used2[b], cudaEventDisableTiming));
                CK(cudaSetDevice(0));
                CK(cudaGetLastError());
            }
        } else {
        CK(cudaStreamCreateWithFlags(&E.cst, cudaStreamNonBlocking));
        for (int b = 0; b < 2; ++b) {
            CK(cudaMalloc(&E.stage[b], max_layer + kMmqTail));
            CK(cudaMemset(E.stage[b], 0, max_layer + kMmqTail));
            CK(cudaEventCreateWithFlags(&E.ev_copy[b], cudaEventDisableTiming));
            CK(cudaEventCreateWithFlags(&E.ev_used[b], cudaEventDisableTiming));
            CK(cudaEventRecord(E.ev_used[b], E.st));
        }
        if (E.split_dev > 0) {   // the second card's own pair: its layers' kernels run on ITS device
            CK(cudaSetDevice(E.split_dev));
            CK(cudaGetLastError());
            cudaStream_t tmp = nullptr;
            CK(cudaStreamCreateWithFlags(&tmp, cudaStreamNonBlocking));
            CK(cudaStreamCreateWithFlags(&E.cst2, cudaStreamNonBlocking));
            for (int b = 0; b < 2; ++b) {
                CK(cudaMalloc(&E.stage2[b], max_layer + kMmqTail));
                CK(cudaMemset(E.stage2[b], 0, max_layer + kMmqTail));
                CK(cudaEventCreateWithFlags(&E.ev_copy2[b], cudaEventDisableTiming));
                CK(cudaEventCreateWithFlags(&E.ev_used2[b], cudaEventDisableTiming));
                CK(cudaEventRecord(E.ev_used2[b], tmp));   // nothing to wait for, yet
            }
            CK(cudaDeviceSynchronize());
            CK(cudaStreamDestroy(tmp));
            CK(cudaSetDevice(0));
            CK(cudaGetLastError());
        }
        }
    }
    E.pool = std::make_unique<C::ExpertPool>(threads);
    CK(cudaDeviceSynchronize());
    if (mtp_only)
        std::fprintf(stderr, "strata-m2: every main layer's routed experts in VRAM (%.2f GiB); the MTP block's: %d of "
                     "%lld in VRAM, the rest on the CPU pool (its drafts only)\n", (double) total_main / (1 << 30), S,
                     (long long) ne);
    else {
        size_t host_all = 0;
        for (Layer* L : all)
            if (!mtp_only || L == &m.mtp) host_all += L->stride * (size_t) ne + kMmqTail;
        char hb[192];
        int hmin = 0, hmax = 0, hfirst = 0;
        for (size_t li = 0; li < all.size(); ++li) {
            if (!all[li]->hblobs || all[li]->h_resident <= 0) continue;
            if (!hmin || all[li]->h_resident < hmin) hmin = all[li]->h_resident;
            if (all[li]->h_resident > hmax) { hmax = all[li]->h_resident; hfirst = all[li]->h_first; }
        }
        if (host == 0)
            std::snprintf(hb, sizeof hb, "none in host memory (every expert is read from the file)");
        else if (host >= host_all)
            std::snprintf(hb, sizeof hb, "all %.2f GiB in %s host memory", (double) host / (1 << 30),
                          "pinned");
        else if (hmin == hmax)
            std::snprintf(hb, sizeof hb, "%.2f of %.2f GiB in %s host memory (%d of %lld per layer, the experts %d-%d: "
                          "the pool's)", (double) host / (1 << 30), (double) host_all / (1 << 30),
                          "pinned", hmin, (long long) ne, hfirst, hfirst + hmax - 1);
        else
            std::snprintf(hb, sizeof hb, "%.2f of %.2f GiB in %s host memory (%d-%d of %lld per layer, the experts %d-%d: "
                          "the pool's)", (double) host / (1 << 30), (double) host_all / (1 << 30),
                          "pinned", hmin, hmax, (long long) ne, hfirst, hfirst + hmax - 1);
        if (E.split_dev >= 0)
            std::fprintf(stderr, "strata-m2: hybrid experts: %d of %lld per layer on card 0 (%.2f GiB) and %d on card 1 "
                         "(%.2f GiB), %s, a batch of <= %d rows on %d CPU threads, bigger prompt chunks streamed (%.2f "
                         "GiB staging, %s); %.1f s\n", S_first, (long long) ne, (double) S_first * per_slot0 / (1 << 30), S,
                         (double) S * per_slot1 / (1 << 30), hb, E.hybrid_t,
                         E.pool->workers() + (E.pool->host_works() ? 1 : 0), (double) stream_bytes / (1 << 30),
                         E.stream_vram ? "VRAM" : "pinned host", ms_since(t0) / 1000.0);
        else
            std::fprintf(stderr, "strata-m2: hybrid experts: %d of %lld per layer in VRAM (%.2f GiB), %s, a batch of <= %d "
                         "rows on %d CPU threads, bigger prompt chunks streamed (%.2f GiB staging, %s); %.1f s\n", S,
                         (long long) ne, (double) S * per_slot / (1 << 30), hb, E.hybrid_t,
                         E.pool->workers() + (E.pool->host_works() ? 1 : 0), (double) stream_bytes / (1 << 30),
                         E.stream_vram ? "VRAM" : "pinned host", ms_since(t0) / 1000.0);
    }
    return true;
#endif
}

// ------------------------------------------------------------------------------------------------ sampling
struct Sampling {
    float temperature = 0.f, top_p = 1.f, min_p = 0.f, repeat = 1.f, freq = 0.f, present = 0.f;
    int top_k = 20, last_n = 0;
    unsigned long long seed = 0;
    bool needs_logits() const { return temperature > 0.f || repeat != 1.f || freq != 0.f || present != 0.f; }
};

int sample(std::vector<float>& lg, const Sampling& s, const std::vector<int32_t>& hist, std::mt19937_64& rng) {
    if (s.last_n != 0 && (s.repeat != 1.f || s.freq != 0.f || s.present != 0.f)) {
        std::map<int, int> cnt;
        const size_t from = s.last_n < 0 || (size_t) s.last_n >= hist.size() ? 0 : hist.size() - (size_t) s.last_n;
        for (size_t i = from; i < hist.size(); ++i) ++cnt[hist[i]];
        for (const auto& [id, c] : cnt) {
            float& v = lg[(size_t) id];
            if (s.repeat != 1.f) v = v > 0 ? v / s.repeat : v * s.repeat;
            v -= (float) c * s.freq + s.present;
        }
    }
    if (s.temperature <= 0.f) return (int) (std::max_element(lg.begin(), lg.end()) - lg.begin());
    const int kk = std::max(1, std::min(s.top_k > 0 ? s.top_k : 64, (int) lg.size()));
    std::vector<int> idx(lg.size());
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = (int) i;
    std::partial_sort(idx.begin(), idx.begin() + kk, idx.end(), [&](int a, int b) { return lg[a] > lg[b]; });
    idx.resize((size_t) kk);
    std::vector<double> p(idx.size());
    const double mx = lg[(size_t) idx[0]];
    double sum = 0;
    for (size_t i = 0; i < idx.size(); ++i) sum += (p[i] = std::exp((lg[(size_t) idx[i]] - mx) / s.temperature));
    size_t keep = idx.size();
    double cum = 0;
    for (size_t i = 0; i < idx.size(); ++i) {
        if (p[i] / sum < s.min_p * (p[0] / sum)) { keep = i; break; }
        cum += p[i] / sum;
        if (cum >= s.top_p) { keep = i + 1; break; }
    }
    keep = std::max<size_t>(1, keep);
    double tot = 0;
    for (size_t i = 0; i < keep; ++i) tot += p[i];
    std::uniform_real_distribution<double> u(0.0, tot);
    double r = u(rng);
    for (size_t i = 0; i < keep; ++i) {
        if ((r -= p[i]) <= 0) return idx[i];
    }
    return idx[keep - 1];
}

std::vector<int32_t> parse_ids(const char* s) {
    std::vector<int32_t> v;
    while (*s) {
        while (*s == ',' || *s == ' ') ++s;
        if (!*s) break;
        char* e = nullptr;
        const long x = std::strtol(s, &e, 10);
        if (e == s) break;
        v.push_back((int32_t) x);
        s = e;
    }
    return v;
}

// The end-of-turn ids: the GGUF's tokenizer.ggml.eos_token_id (MiniMax-M2: 200020, "[e~["), set by load(); the Qwen
// ids (<|endoftext|>, <|im_end|>) stood here before and are kept for a file without the key.  With only those, a
// MiniMax-M2 answer never ended: the server read past "[e~[" to max_tokens.
bool is_eos(int t) { return g_eos_id >= 0 ? t == g_eos_id : t == 248044 || t == 248046; }

// ------------------------------------------------------------------------------------------------ pictures
constexpr int32_t kImagePad = 248056;   // <|image_pad|>

uint64_t fnv1a(const void* p, size_t n, uint64_t h = 1469598103934665603ull) {
    const uint8_t* b = (const uint8_t*) p;
    for (size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 1099511628211ull;
    return h;
}

/// A request's pictures: the encoder's rows in prompt order, each picture's cell run, and the positions.
struct ImgReq {
    std::vector<float> rows;
    std::vector<ImgKey> imgs;
    std::vector<int32_t> cell_row;   // per prompt cell: its row in `rows`, or -1 (text)
    std::vector<int32_t> mrope;      // [cell][3] t, h, w of every prompt cell (llama.cpp's mtmd positions)
    int32_t delta = 0;               // the text after the prompt: rotary position = cell + delta
};

/// Reads a strata-vision file (one or more records: int32 'SVE1', n, nx, ny, n_embd, then n x n_embd floats, in prompt
/// order) and lays the pictures over the prompt's <|image_pad|> runs.  Positions as mtmd's: text counts up by one; a
/// picture's cells take (t, h, w) = (p, p + row, p + column) and the text after it continues at p + max(nx, ny).  "" or
/// the error.  `keys`: the server's key for each picture (img_keys=), else the embeddings' hash.
std::string build_images(const std::string& path, const std::vector<int32_t>& ids, int64_t n_embd,
                         const std::vector<uint64_t>& keys, ImgReq& r) {
    struct Img { int64_t n, nx, ny; size_t off; };
    std::vector<Img> imgs;
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return "cannot open " + path;
    std::string ve;
    while (ve.empty()) {
        int32_t hdr[5];
        const size_t got = std::fread(hdr, sizeof(int32_t), 5, f);
        if (got == 0) break;
        if (got != 5 || hdr[0] != 0x31455653 || hdr[1] < 1 || hdr[2] < 1 || hdr[3] < 1 || (int64_t) hdr[2] * hdr[3] != hdr[1] ||
            hdr[4] != (int32_t) n_embd) {
            ve = "bad embeddings file (expected strata-vision records of width " + std::to_string((long long) n_embd) + ")";
            break;
        }
        const size_t off = r.rows.size(), cnt = (size_t) hdr[1] * (size_t) hdr[4];
        r.rows.resize(off + cnt);
        if (std::fread(r.rows.data() + off, sizeof(float), cnt, f) != cnt) { ve = "short embeddings file"; break; }
        imgs.push_back({hdr[1], hdr[2], hdr[3], off});
    }
    std::fclose(f);
    if (!ve.empty()) return ve;
    const int64_t n = (int64_t) ids.size();
    r.cell_row.assign((size_t) n, -1);
    r.mrope.assign((size_t) n * 3, 0);
    auto put = [&](int64_t c, int64_t t, int64_t h, int64_t w) {
        r.mrope[(size_t) c * 3] = (int32_t) t;
        r.mrope[(size_t) c * 3 + 1] = (int32_t) h;
        r.mrope[(size_t) c * 3 + 2] = (int32_t) w;
    };
    int64_t p = 0, i = 0;
    size_t k = 0;
    while (i < n) {
        if (ids[(size_t) i] != kImagePad) { put(i, p, p, p); ++p; ++i; continue; }
        if (k >= imgs.size()) return "the prompt has more images than the embeddings file";
        const Img& im = imgs[k++];
        uint64_t h;
        {   // what the conversation cache compares: a picture is its grid and its key (or its embeddings)
            const int64_t grid[3] = {im.n, im.nx, im.ny};
            h = fnv1a(grid, sizeof grid);
            if (keys.size() == imgs.size()) h = fnv1a(&keys[k - 1], sizeof(uint64_t), h);
            else h = fnv1a(r.rows.data() + im.off, (size_t) im.n * (size_t) n_embd * sizeof(float), h);
        }
        r.imgs.push_back({i, im.n, h});
        for (int64_t j = 0; j < im.n; ++j)
            if (i + j >= n || ids[(size_t) (i + j)] != kImagePad)
                return "image " + std::to_string(k) + " has " + std::to_string((long long) im.n) +
                       " rows but fewer <|image_pad|> tokens";
        for (int64_t j = 0; j < im.n; ++j) {
            const int64_t y = j / im.nx, x = j % im.nx;
            put(i + j, p, p + y, p + x);
            r.cell_row[(size_t) (i + j)] = (int32_t) ((im.off / (size_t) n_embd) + (size_t) j);
        }
        i += im.n;
        p += std::max(im.nx, im.ny);
    }
    if (k != imgs.size()) return "the embeddings file has more images than the prompt";
    if (n > 0 && ids[(size_t) (n - 1)] == kImagePad) return "the prompt cannot end in an image";
    r.delta = (int32_t) (p - n);
    return "";
}

// ------------------------------------------------------------------------------------------------ serve
/// A batch slot: a session decoding (its next token is `x`, not yet in the state).
struct Slot {
    bool active = false, stop = false;
    int32_t x = 0;
    int64_t produced = 0, max_new = 0;
    Sampling smp;
    std::mt19937_64 rng;
    std::vector<int32_t> hist;
    Clock::time_point t0;
    int32_t draft = 0;            // --batch-mtp: the MTP block's proposal for the token after `x`
    bool draft_ready = false;
    long long offered = 0, accepted = 0;
};

/// The engine protocol (serve/server.py).  One request at a time on the control lines: GEN / GENI (alone: speculative
/// steps with the MTP block), or with --batch N BGEN / BGENI <slot> (the request's prompt is read into that slot's
/// session - the decode windows of the other slots go on between its chunks - then its tokens come as BT lines, one
/// window for every slot decoding, until BDONE).  BSTOP <slot> ends a slot at its next window; STOP the request on the
/// control lines.
struct Serve {
    Engine& E;
    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::string> lines;
    bool eof = false;
    std::atomic<bool> stop{false};
    std::vector<Slot> slots;
    int n_active = 0;
    std::vector<float> lg;
    // decode windows of the other slots between two chunks of a prompt read: STRATA_Q36_READ_WINDOWS (default 1; a chunk
    // takes ~0.15 s at 512 tokens, a window ~0.012 s: more windows give the decoding slots more of the time a read takes)
    int read_windows = 1;

    explicit Serve(Engine& e) : E(e), slots((size_t) e.n_sess) {
        if (const char* v = std::getenv("STRATA_Q36_READ_WINDOWS")) read_windows = std::max(0, std::min(64, std::atoi(v)));
    }

    /// BSTOP / BYIELD lines among the queued ones: applied now (a window is a moment away at most).
    void poll_control() {
        std::lock_guard<std::mutex> lk(mu);
        for (auto it = lines.begin(); it != lines.end();) {
            if (it->rfind("BSTOP ", 0) == 0) {
                const int b = std::atoi(it->c_str() + 6);
                if (b >= 0 && b < (int) slots.size()) slots[(size_t) b].stop = true;
                it = lines.erase(it);
            } else if (it->rfind("BYIELD", 0) == 0) {   // (a read here does not give way: the server then goes on)
                it = lines.erase(it);
            } else {
                ++it;
            }
        }
    }

    /// One decode window with MTP proposals (--batch-mtp): every active slot's next token, and for up to kVerify - S of the
    /// slots (the greedy ones, in turn) also the proposal its MTP block made: the window verifies both at once.  A slot whose
    /// proposal the main model confirms moves two tokens in the window.  The proposals for the next window come right after
    /// it, from the hidden states the window just produced (the first one of a slot from the state its prompt left).
    int mtp_turn = 0;
    void window_mtp() {
        poll_control();
        const int N = (int) E.m->g.n_embd;
        std::vector<int> act;
        for (int b = 0; b < E.n_sess; ++b)
            if (slots[(size_t) b].active) act.push_back(b);
        const int S = (int) act.size();
        if (S == 0) return;
        // who verifies a proposal this window: the greedy slots in turn, as many as fit beside one row per slot
        const int room = Engine::kVerify - S;
        std::vector<char> cand((size_t) S, 0);
        int nc = 0;
        for (int k = 0; k < S && nc < room; ++k) {
            const int i = (mtp_turn + k) % S;
            if (!slots[(size_t) act[(size_t) i]].smp.needs_logits()) { cand[(size_t) i] = 1; ++nc; }
        }
        ++mtp_turn;
        // a candidate without a proposal gets its first one from the state its last row left
        std::vector<Engine::MtpRow> first;
        std::vector<int> first_of;
        for (int i = 0; i < S; ++i) {
            Slot& sl = slots[(size_t) act[(size_t) i]];
            if (cand[(size_t) i] && !sl.draft_ready) {
                Engine::Sess& ss = E.sess[(size_t) act[(size_t) i]];
                first.push_back({act[(size_t) i], sl.x, ss.pos, ss.hlast});
                first_of.push_back(i);
            }
        }
        if (!first.empty()) {
            E.mtp_rows(first.data(), (int) first.size());
            for (size_t k = 0; k < first.size(); ++k) {
                Slot& sl = slots[(size_t) act[(size_t) first_of[k]]];
                sl.draft = E.mtp_pick[k];
                sl.draft_ready = true;
            }
        }
        std::vector<Engine::Group> gr((size_t) S);
        std::vector<int> row0((size_t) S);
        int T = 0;
        for (int i = 0; i < S; ++i) {
            const Slot& sl = slots[(size_t) act[(size_t) i]];
            gr[(size_t) i] = {act[(size_t) i], cand[(size_t) i] ? 2 : 1, {sl.x, sl.draft}};
            row0[(size_t) i] = T;
            T += gr[(size_t) i].n;
        }
        E.run_rows(gr.data(), S, true);
        // the picks: emit, decide how many rows each slot keeps
        std::vector<int> keep((size_t) S, 1);
        std::vector<int32_t> last((size_t) S, 0);
        std::vector<char> alive((size_t) S, 0);
        for (int i = 0; i < S; ++i) {
            const int b = act[(size_t) i];
            Slot& sl = slots[(size_t) b];
            Engine::Sess& ss = E.sess[(size_t) b];
            const int n = gr[(size_t) i].n, r0 = row0[(size_t) i];
            int y0 = E.rows_pick[r0];
            if (sl.smp.needs_logits()) {
                E.host_logits_row(r0, lg);
                y0 = sample(lg, sl.smp, sl.hist, sl.rng);
            }
            bool two = n == 2 && y0 == sl.draft && !is_eos(y0) && !sl.stop && sl.produced + 2 <= sl.max_new &&
                       ss.pos + 3 <= E.ctx;   // (ss.pos: the position of the slot's next token: the window did not move it)
            if (n == 2) { ++sl.offered; if (two) ++sl.accepted; }
            const int rows_kept = two ? 2 : 1;
            keep[(size_t) i] = rows_kept;
            const int32_t fed[2] = {sl.x, sl.draft};
            bool done = false;
            int y = y0;
            for (int j = 0; j < rows_kept; ++j) {
                if (j == 1) y = E.rows_pick[r0 + 1];
                ss.live.push_back(fed[j]);
                std::printf("BT %d %d\n", b, y);
                ++sl.produced;
                sl.hist.push_back(y);
                const char* fin = is_eos(y) ? "stop" : sl.stop ? "cancel" : sl.produced >= sl.max_new ? "length"
                                  : ss.pos + j + 2 > E.ctx ? "length" : nullptr;
                if (fin) {
                    std::printf("BDONE %d %lld %s %.1f\n", b, (long long) sl.produced, fin, ms_since(sl.t0));
                    if (sl.offered > 0)
                        std::fprintf(stderr, "strata batch: slot %d: %lld of %lld MTP proposals accepted\n", b, sl.accepted,
                                     sl.offered);
                    sl.active = false;
                    ss.active = false;
                    --n_active;
                    done = true;
                    break;
                }
            }
            last[(size_t) i] = y;
            alive[(size_t) i] = !done;
            if (!done) sl.x = y;
        }
        E.commit_rows(gr.data(), keep.data(), S);
        // the hidden state after each slot's last kept row (the seed of a later first proposal), and the next proposals
        std::vector<Engine::MtpRow> nx;
        std::vector<int> nx_of, nx_last;
        for (int i = 0; i < S; ++i) {
            const int b = act[(size_t) i];
            Engine::Sess& ss = E.sess[(size_t) b];
            const int r0 = row0[(size_t) i];
            CK(cudaMemcpyAsync(ss.hlast, E.hmain + (size_t) (r0 + keep[(size_t) i] - 1) * N, (size_t) N * 4,
                               cudaMemcpyDeviceToDevice, E.st));
            Slot& sl = slots[(size_t) b];
            sl.draft_ready = false;
            if (!alive[(size_t) i] || !cand[(size_t) i]) continue;
            const int64_t p_new = ss.pos;   // commit_rows moved it: the cell of the token the next window feeds
            if (keep[(size_t) i] == 2) {
                nx.push_back({b, sl.draft, p_new - 1, E.hmain + (size_t) r0 * N});
                nx.push_back({b, last[(size_t) i], p_new, E.hmain + (size_t) (r0 + 1) * N});
                nx_last.push_back((int) nx.size() - 1);
            } else {
                nx.push_back({b, last[(size_t) i], p_new, E.hmain + (size_t) r0 * N});
                nx_last.push_back((int) nx.size() - 1);
            }
            nx_of.push_back(i);
        }
        std::fflush(stdout);
        if (!nx.empty()) {
            // (more rows than the window holds: the proposals come in two calls)
            for (size_t at = 0, k = 0; at < nx.size();) {
                size_t take = std::min<size_t>(nx.size() - at, Engine::kVerify);
                while (take > 0 && at + take < nx.size() && nx[at + take].slot == nx[at + take - 1].slot) --take;   // a slot's rows together
                E.mtp_rows(nx.data() + at, (int) take);
                for (; k < nx_last.size() && (size_t) nx_last[k] < at + take; ++k) {
                    Slot& sl = slots[(size_t) act[(size_t) nx_of[k]]];
                    sl.draft = E.mtp_pick[(size_t) nx_last[k] - at];
                    sl.draft_ready = true;
                }
                at += take;
            }
        }
    }

    /// One decode window over every active slot: one token each.
    void window() {
        poll_control();
        if (E.bmtp && E.mtp_d > 0) {   // proposals need a row to spare and a slot that decodes greedily
            int act = 0, greedy = 0;
            for (const Slot& sl : slots)
                if (sl.active) { ++act; greedy += !sl.smp.needs_logits(); }
            if (greedy > 0 && Engine::kVerify - act > 0) { window_mtp(); return; }
        }
        window_plain();
    }

    /// One token of every active slot, no proposals.
    void window_plain() {
        std::vector<Engine::Row> rows;
        for (int b = 0; b < E.n_sess; ++b)
            if (slots[(size_t) b].active) rows.push_back({b, slots[(size_t) b].x});
        if (rows.empty()) return;
        E.step_rows(rows.data(), (int) rows.size());
        const size_t Nb = (size_t) E.m->g.n_embd * 4;
        for (size_t r = 0; r < rows.size(); ++r) {
            const int b = rows[r].slot;
            Slot& sl = slots[(size_t) b];
            Engine::Sess& ss = E.sess[(size_t) b];
            if (E.bmtp)   // (the hidden state a later first proposal starts from)
                CK(cudaMemcpyAsync(ss.hlast, E.hmain + r * (size_t) E.m->g.n_embd, Nb, cudaMemcpyDeviceToDevice, E.st));
            sl.draft_ready = false;
            ss.live.push_back(sl.x);
            int y;
            if (sl.smp.needs_logits()) {
                E.host_logits_row((int) r, lg);
                y = sample(lg, sl.smp, sl.hist, sl.rng);
            } else {
                y = E.rows_pick[r];
            }
            std::printf("BT %d %d\n", b, y);
            ++sl.produced;
            sl.hist.push_back(y);
            const char* fin = is_eos(y) ? "stop" : sl.stop ? "cancel" : sl.produced >= sl.max_new ? "length"
                              : ss.pos + 1 > E.ctx ? "length" : nullptr;
            if (fin) {
                std::printf("BDONE %d %lld %s %.1f\n", b, (long long) sl.produced, fin, ms_since(sl.t0));
                sl.active = false;
                ss.active = false;
                --n_active;
            } else {
                sl.x = y;
            }
        }
        std::fflush(stdout);
    }

    /// The number of tokens at the start of `ids` that session ss already holds (with the same pictures).  The
    /// session's state is rewound to that point before the rest is read, so a prompt that re-renders its history a
    /// little differently (an agent client re-sends the whole conversation every turn, and the reply it shows may not
    /// be the tokens this engine generated) still skips the part it shares.  The last token is always read again:
    /// its logits are this request's first pick, and a stale one would belong to another prompt.
    size_t match_len(const Engine::Sess& ss, const std::vector<int32_t>& ids, const std::vector<ImgKey>& imgs) const {
        const size_t L = ss.live.size(), M = ids.size();
        if (L == 0 || M == 0) return 0;
        size_t k = 0;
        const size_t n = std::min(L, M);
        while (k < n && ss.live[k] == ids[k]) ++k;
        // a picture must not be cut in half: the held state stops before it, on either side
        for (const ImgKey& im : ss.imgs)
            if ((size_t) im.cell < k && k < (size_t) (im.cell + im.n)) k = (size_t) im.cell;
        for (const ImgKey& im : imgs)
            if ((size_t) im.cell < k && k < (size_t) (im.cell + im.n)) k = (size_t) im.cell;
        // the pictures wholly inside the shared start must be the request's too
        std::vector<ImgKey> held, asked;
        for (const ImgKey& im : ss.imgs) if (im.cell + im.n <= (int64_t) k) held.push_back(im);
        for (const ImgKey& im : imgs) if (im.cell + im.n <= (int64_t) k) asked.push_back(im);
        if (held != asked) return 0;
        if (k >= M) k = M - 1;   // the last token is read (its logits are wanted)
        return k;
    }

    /// The session a request alone takes: the one holding the longest start of its prompt, else an empty one, else the
    /// least recently used (never a decoding slot).
    int pick_session(const std::vector<int32_t>& ids, const std::vector<ImgKey>& imgs) const {
        int best = -1, free_empty = -1, lru = -1;
        size_t bl = 0;
        for (int i = 0; i < E.n_sess; ++i) {
            const Engine::Sess& ss = E.sess[(size_t) i];
            if (ss.active) continue;
            const size_t l = match_len(ss, ids, imgs);
            if (l > bl) { bl = l; best = i; }
            if (ss.live.empty() && free_empty < 0) free_empty = i;
            if (lru < 0 || ss.used < E.sess[(size_t) lru].used) lru = i;
        }
        return best >= 0 ? best : free_empty >= 0 ? free_empty : lru;
    }

    void request(const std::string& line) {
        const bool adm = line[0] == 'B';
        const bool geni = line.rfind(adm ? "BGENI " : "GENI ", 0) == 0;
        if (!adm && !geni && line.rfind("GEN ", 0) != 0) {
            std::printf("ERR expected: GEN <max_new> [key=value ...] <id,id,...> or GENI <max_new> [keys] <file> <id,id,...>\n");
            std::fflush(stdout);
            return;
        }
        char* p = const_cast<char*>(line.c_str()) + (adm ? (geni ? 6 : 5) : (geni ? 5 : 4));
        int slot = -1;
        if (adm) {
            char* e1 = nullptr;
            const long b = std::strtol(p, &e1, 10);
            p = e1;
            slot = (int) b;
            if (e1 == line.c_str() || b < 0 || b >= (long) slots.size() || slots[(size_t) b].active || E.n_sess < 2) {
                std::printf("ERR BGEN: no such free slot (--batch %d) or a bad max_new\n", E.n_sess < 2 ? 0 : E.n_sess);
                std::fflush(stdout);
                return;
            }
        }
        const long long max_new = std::strtoll(p, &p, 10);
        Sampling smp;
        std::vector<uint64_t> img_keys;
        for (;;) {
            while (*p == ' ') ++p;
            const char* st0 = p;
            while (*p && *p != ' ') ++p;
            const std::string tk(st0, (size_t) (p - st0));
            const size_t eq = tk.find('=');
            if (tk.empty() || eq == std::string::npos) { p = const_cast<char*>(st0); break; }
            const std::string key = tk.substr(0, eq);
            const char* v = tk.c_str() + eq + 1;
            if (key == "temperature") smp.temperature = std::strtof(v, nullptr);
            else if (key == "top_p") smp.top_p = std::strtof(v, nullptr);
            else if (key == "top_k") smp.top_k = std::atoi(v);
            else if (key == "min_p") smp.min_p = std::strtof(v, nullptr);
            else if (key == "penalty_last_n") smp.last_n = std::atoi(v);
            else if (key == "penalty_repeat") smp.repeat = std::strtof(v, nullptr);
            else if (key == "penalty_freq") smp.freq = std::strtof(v, nullptr);
            else if (key == "penalty_present") smp.present = std::strtof(v, nullptr);
            else if (key == "seed") smp.seed = std::strtoull(v, nullptr, 10);
            else if (key == "img_keys") {
                const char* q = v;
                img_keys.clear();
                while (*q) {
                    char* qe = nullptr;
                    const unsigned long long x = std::strtoull(q, &qe, 16);
                    if (qe == q) { img_keys.clear(); break; }
                    img_keys.push_back((uint64_t) x);
                    q = *qe == ',' ? qe + 1 : qe;
                    if (*qe != ',' && *qe != '\0') { img_keys.clear(); break; }
                }
            }
        }
        std::string emb;
        if (geni) {
            while (*p == ' ') ++p;
            const char* e = p;
            while (*e && *e != ' ') ++e;
            emb.assign(p, (size_t) (e - p));
            p = const_cast<char*>(e);
        }
        const std::vector<int32_t> ids = parse_ids(p);
        if (max_new < 1 || ids.empty() || (geni && emb.empty())) {
            std::printf("ERR bad request: max_new or the ids\n");
            std::fflush(stdout);
            return;
        }
        if (geni && !E.vision) {
            std::printf("ERR this engine was started without --vision\n");
            std::fflush(stdout);
            return;
        }
        if ((int64_t) ids.size() + max_new > E.ctx) {
            std::printf("ERR prompt (%zu tokens) + max_new (%lld) exceeds the context (%d)\n", ids.size(), max_new, E.ctx);
            std::fflush(stdout);
            return;
        }
        bool bad = false;
        for (int32_t t : ids) bad |= t < 0 || t >= (int32_t) E.m->g.n_vocab;
        if (bad) { std::printf("ERR a token id is outside the vocabulary\n"); std::fflush(stdout); return; }
        ImgReq ir;
        if (geni) {
            const std::string ve = build_images(emb, ids, E.m->g.n_embd, img_keys, ir);
            if (!ve.empty()) { std::printf("ERR %s\n", ve.c_str()); std::fflush(stdout); return; }
        }
        stop.store(false);
        // ---- the session: the held state continues when the request extends it (the recurrent state cannot step back)
        const int s = adm ? slot : pick_session(ids, ir.imgs);
        Engine::Sess& ss = E.sess[(size_t) s];
        size_t reuse = match_len(ss, ids, ir.imgs);
        if (adm) {   // another session holding more of this prompt (the conversation was read alone): copy its state
            int src = -1;
            size_t best = reuse;
            for (int i = 0; i < E.n_sess; ++i) {
                if (i == s || E.sess[(size_t) i].active) continue;
                const size_t l = match_len(E.sess[(size_t) i], ids, ir.imgs);
                if (l > best) { best = l; src = i; }
            }
            if (src >= 0) {
                E.copy_session(src, s);
                reuse = best;
            }
        }
        E.select(s);
        if (reuse == 0) {
            E.reset();
            ss.live.clear();
            ss.imgs.clear();
        } else {
            // Rewind to the shared start: the held tail (a differently rendered reply, a shorter prompt) is dropped,
            // and the K / V cells past `reuse` are never read (the attention stops at the cell it writes).  No cache
            // is zeroed: that is what makes a warm turn cheap.
            ss.live.resize(reuse);
            ss.imgs.erase(std::remove_if(ss.imgs.begin(), ss.imgs.end(),
                                         [&](const ImgKey& im) { return im.cell + im.n > (int64_t) reuse; }),
                          ss.imgs.end());
            E.pos = (int64_t) reuse;
            ss.pos = E.pos;
        }
        if (geni) E.set_images(ir.rows, ir.cell_row, ir.mrope, (int64_t) ids.size(), ir.delta);
        else E.clear_images();
        const auto t0 = Clock::now();
        const int64_t fresh = (int64_t) (ids.size() - reuse);
        bool cancelled = false;
        const int64_t got = E.process(ids.data() + reuse, fresh, [&](int64_t done) -> bool {
            std::printf("PP %lld %lld %.0f %.1f\n", (long long) done, (long long) fresh, ms_since(t0),
                        (double) done / std::max(1e-9, ms_since(t0) / 1000.0));
            std::fflush(stdout);
            if (n_active > 0 && done < fresh) {
                poll_control();
                // the slots decoding go on between a read's chunks (the last chunk's
                for (int w = 0; w < read_windows && n_active > 0; ++w) window();   // logits must stay for the first token)
            } else {
                poll_control();
            }
            if (stop.load() && done < fresh) { cancelled = true; return false; }
            return true;
        });
        ss.live.insert(ss.live.end(), ids.begin() + (long) reuse, ids.begin() + (long) reuse + (long) got);
        ss.imgs.clear();
        bool cut = false;
        for (const ImgKey& k : ir.imgs) {
            if (k.cell + k.n <= (int64_t) ss.live.size()) ss.imgs.push_back(k);
            else if (k.cell < (int64_t) ss.live.size()) cut = true;
        }
        if (cancelled && cut) {   // stopped inside a picture: the state cannot be continued, the next read starts over
            ss.live.clear();
            ss.imgs.clear();
        }
        E.sess[(size_t) E.cur].pos = E.pos;
        const double prompt_ms = ms_since(t0);
        if (cancelled) {
            std::printf("DONE 0 %zu %.1f 0.0 cancel 0 0 %zu 0 0 0 0 0.0 %lld 0\n", ids.size(), prompt_ms, reuse, (long long) got);
            if (adm) std::printf("BADM %d 0\n", slot);
            std::fflush(stdout);
            return;
        }
        std::mt19937_64 rng(smp.seed);
        std::vector<int32_t> hist = ids;
        const auto d0 = Clock::now();
        long long produced = 0, offered = 0, accepted = 0;
        const char* finish = "length";
        // greedy requests alone with the MTP block: speculative steps of mtp_d drafts (verified, so the text is the greedy
        // text); sampled ones, and requests beside others, one token at a time
        const bool spec = E.mtp_d > 0 && !smp.needs_logits() && !adm;
        std::vector<int32_t> fix, drafts;
        int32_t picks[16];
        int nx = smp.needs_logits() ? -1 : E.greedy();
        if (adm) {
            if (smp.needs_logits()) {
                E.host_logits(lg);
                nx = sample(lg, smp, hist, rng);
            }
            std::printf("T %d\n", nx);
            ++produced;
            hist.push_back(nx);
            const char* fin = is_eos(nx) ? "stop" : stop.load() ? "cancel" : max_new <= 1 ? "length"
                              : E.pos + 1 > E.ctx ? "length" : nullptr;
            const bool cont = fin == nullptr;
            std::printf("DONE %lld %zu %.1f %.1f %s 0 0 %zu 0 0 0 0 0.0 %lld 0\n", produced, ids.size(), prompt_ms,
                        ms_since(d0), cont ? "length" : fin, reuse, (long long) fresh);
            if (cont) {
                Slot& sl = slots[(size_t) slot];
                sl = Slot{};
                sl.active = true;
                sl.x = nx;
                sl.produced = 1;
                sl.max_new = max_new;
                sl.smp = smp;
                sl.rng = rng;
                sl.hist = hist;
                sl.t0 = Clock::now();
                ss.active = true;
                ++n_active;
            }
            std::printf("BADM %d %d\n", slot, cont ? 1 : 0);
            std::fflush(stdout);
            std::fprintf(stderr, "strata batch: slot %d takes a prompt of %zu tokens = %zu reused + %lld read in %.0f ms (%.1f "
                         "tok/s)%s\n", slot, ids.size(), reuse, (long long) fresh, prompt_ms,
                         prompt_ms > 0 ? 1000.0 * (double) fresh / prompt_ms : 0.0, cont ? "" : ", done at once");
            return;
        }
        for (;;) {
            if (smp.needs_logits()) {
                E.host_logits(lg);
                nx = sample(lg, smp, hist, rng);
            }
            auto emit = [&](int t) {   // false: the answer ends here
                std::printf("T %d\n", t);
                std::fflush(stdout);
                ++produced;
                hist.push_back(t);
                if (is_eos(t)) { finish = "stop"; return false; }
                if (produced >= max_new) return false;
                if (stop.load()) { finish = "cancel"; return false; }
                return true;
            };
            if (!emit(nx)) break;
            if (n_active > 0) window();
            if (spec && n_active == 0 && E.pos + E.mtp_d + 1 <= E.ctx) {
                const int a = E.spec_step(nx, fix, E.mtp_d, drafts, picks, &stop);
                offered += E.mtp_d;
                accepted += a;
                // the state now holds x and every accepted draft, whether or not the answer ends inside them (an end of
                // answer, max_new): the held tokens say so, so a later request that continues this conversation matches them
                // (a prompt that has the same tokens next reuses the state, any other reads from the start) and never reads
                // a token twice
                ss.live.push_back(nx);
                for (int i = 0; i < a; ++i) ss.live.push_back(drafts[(size_t) i]);
                fix.assign(drafts.begin(), drafts.begin() + a);
                bool ended = false;
                for (int i = 0; i < a && !ended; ++i)
                    if (!emit(drafts[(size_t) i])) ended = true;
                if (ended) break;
                nx = picks[a];
                continue;
            }
            if (E.pos + 1 > E.ctx) break;
            const int32_t one = nx;
            E.process(&one, 1, [](int64_t) {});
            ss.live.push_back(nx);
            fix.clear();
            if (!smp.needs_logits()) nx = E.greedy();
        }
        E.sess[(size_t) E.cur].pos = E.pos;
        const double decode_ms = ms_since(d0);
        std::printf("DONE %lld %zu %.1f %.1f %s %lld %lld %zu 0 0 0 0 0.0 %lld 0\n", produced, ids.size(), prompt_ms,
                    decode_ms, finish, accepted, offered, reuse, (long long) fresh);
        std::fflush(stdout);
        std::fprintf(stderr, "strata serve: session %d: prompt %zu tokens = %zu reused + %lld read in %.0f ms (%.1f tok/s), "
                     "%lld generated in %.0f ms (%.1f tok/s), drafts accepted %lld of %lld (%s)\n", s, ids.size(), reuse,
                     (long long) fresh, prompt_ms, prompt_ms > 0 ? 1000.0 * (double) fresh / prompt_ms : 0.0, produced,
                     decode_ms, decode_ms > 0 ? 1000.0 * (double) produced / decode_ms : 0.0, accepted, offered, finish);
    }

    int run() {
        if (E.n_sess > 1) std::printf("INFO batch_slots=%d slot_cache=1\n", E.n_sess);
        std::printf("READY %d stop\n", E.ctx);
        std::fflush(stdout);
        std::thread reader([&] {
            std::string l;
            while (std::getline(std::cin, l)) {
                if (!l.empty() && l.back() == '\r') l.pop_back();
                if (l == "STOP") { stop.store(true); continue; }
                std::lock_guard<std::mutex> lk(mu);
                lines.push_back(l);
                cv.notify_one();
            }
            std::lock_guard<std::mutex> lk(mu);
            eof = true;
            cv.notify_one();
        });
        reader.detach();
        for (;;) {
            std::string line;
            bool have = false;
            {
                std::unique_lock<std::mutex> lk(mu);
                if (n_active == 0) cv.wait(lk, [&] { return eof || !lines.empty(); });
                if (!lines.empty()) {
                    line = lines.front();
                    lines.pop_front();
                    have = true;
                } else if (n_active == 0) {
                    return 0;
                }
            }
            if (!have) { window(); continue; }
            if (line.empty()) continue;
            if (line.rfind("BSTOP ", 0) == 0) {
                const int b = std::atoi(line.c_str() + 6);
                if (b >= 0 && b < (int) slots.size()) slots[(size_t) b].stop = true;
            } else if (line.rfind("BYIELD", 0) == 0) {
                // (a read that has ended)
            } else if (line.rfind("GEN", 0) == 0 || line.rfind("BGEN", 0) == 0) {
                request(line);
            } else {
                std::printf("ERR expected: GEN <max_new> [key=value ...] <id,id,...>\n");
                std::fflush(stdout);
            }
            if (n_active > 0) window();
        }
    }
};

int serve(Engine& E) {
    Serve sv(E);
    return sv.run();
}

}  // namespace

int strata::minimaxm2::run(int argc, char** argv) {
    std::string model, tokens, force, dump_full, emb_file, dump_layers_path;
    bool low_ram = false;
    Engine::KvMode kv = Engine::KV_F32;   // --kv: the K / V cache's element (FP32, FP16 or INT8)
    int peer_device = -1;
    int layer_split = -1;
    int ctx = 32768, chunk = 0, n_gen = 64, bench_pp = 0, bench_tg = 0, mtp_d = 0, cpu_threads = 0, hybrid_rows = 128;
    double expert_vram = -1.0;
    double resident_gib = -1.0;   // --resident-budget-gib: GiB of the routed experts kept in host RAM (< 0: all)
    bool warm = true;
    int expert_slots = 0, batch = 0, draft_vocab = -1;
    int vram_reserve_mib = kM2ReserveMib;   // --vram-reserve-mib: VRAM kept free on the first card
    bool vram_reserve_given = false;
    int vram_reserve_later_mib = -1;        // --vram-reserve-later-mib: ... on the split's / the peer's card (< 0: same)
    bool serve_mode = false, graph = true, vision = false, batch_mtp = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "strata-m2: %s needs a value\n", a.c_str()); std::exit(2); }
            return argv[++i];
        };
        // strata's names where they mean the same thing (the family switch passes strata's command line on)
        if (a == "--model" || a == "--native") model = next();
        else if (a == "--max-new") n_gen = std::atoi(next().c_str());
        else if (a == "--no-token-graph") graph = false;
        else if (a == "--pool-workers") cpu_threads = std::atoi(next().c_str());
        else if (a == "--gpu") (void) next();   // applied by strata before the switch (CUDA/HIP_VISIBLE_DEVICES)
        else if (a == "--greedy") {}            // --tokens decodes greedily here anyway
        else if (a == "--serve") serve_mode = true;
        else if (a == "--max-context" || a == "--ctx") ctx = std::atoi(next().c_str());
        else if (a == "--chunk" || a == "--prefill") { const std::string v = next(); chunk = v == "auto" ? 0 : std::atoi(v.c_str()); }
        else if (a == "--tokens") tokens = next();
        else if (a == "--n") n_gen = std::atoi(next().c_str());
        else if (a == "--force") force = next();
        else if (a == "--emb") emb_file = next();   // --tokens with pictures: a strata-vision file (the ids carry <|image_pad|> runs)
        else if (a == "--dump-full-logits") dump_full = next();
        else if (a == "--bench-pp") bench_pp = std::atoi(next().c_str());
        else if (a == "--bench-tg") bench_tg = std::atoi(next().c_str());
        else if (a == "--no-graph") graph = false;
        else if (a == "--no-warmup") warm = false;
        else if (a == "--mtp") {
            // a number of drafts; Flash-Next's "--mtp <draft layer dir>" means nothing for this family
            const std::string v = next();
            if (!v.empty() && std::isdigit((unsigned char) v[0])) mtp_d = std::atoi(v.c_str());
            else std::fprintf(stderr, "strata-m2: ignoring --mtp %s (this family's MTP is the GGUF's own block: --mtp N "
                                      "drafts)\n", v.c_str());
        }
        else if (a == "--draft-vocab") draft_vocab = std::atoi(next().c_str());   // the MTP drafts pick among the first N token ids
        else if (a == "--vision") vision = true;                                  // pictures (GENI): the encoder is strata-vision's
        else if (a == "--batch" || a == "--slots") batch = std::atoi(next().c_str());   // sessions decoding together
        else if (a == "--batch-mtp") batch_mtp = true;                                  // ... each verifying an MTP proposal
        else if (a == "--expert-vram") expert_vram = std::atof(next().c_str());   // GiB for the routed experts
        else if (a == "--resident-budget-gib") resident_gib = std::atof(next().c_str());   // GiB of experts kept in host RAM
        else if (a == "--vram-reserve-mib") { vram_reserve_mib = std::atoi(next().c_str()); vram_reserve_given = true; }
        else if (a == "--vram-reserve-later-mib") vram_reserve_later_mib = std::atoi(next().c_str());
        else if (a == "--low-ram") low_ram = true;   // no host copy of the experts: they stay in the mapping
        else if (a == "--kv") {   // the K / V cache's element (FP32 by default: the byte-identical path)
            const std::string v = next();
            if (v == "f16" || v == "fp16" || v == "half") kv = Engine::KV_F16;
            else if (v == "int8" || v == "i8" || v == "q8") kv = Engine::KV_I8;
            else if (v == "f32" || v == "fp32" || v == "float") kv = Engine::KV_F32;
            else   // the Terminal's other modes (q4_0, k8v4) belong to the other families: said, not fatal
                std::fprintf(stderr, "strata-m2: --kv %s is not a mode of this engine (f32, f16, int8): the K / V cache "
                             "keeps its mode\n", v.c_str());
        }
        else if (a == "--peer-device") peer_device = std::atoi(next().c_str());   // the second card's expert tier
        else if (a == "--layer-split") layer_split = std::atoi(next().c_str());  // the layers from there on the 2nd card
        else if (a == "--dump-layers") dump_layers_path = next();
        else if (a == "--cpu-threads") cpu_threads = std::atoi(next().c_str());
        else if (a == "--expert-slots") expert_slots = std::atoi(next().c_str());
        else if (a == "--hybrid-rows") hybrid_rows = std::atoi(next().c_str());
        else if (serve_mode && a.rfind("--", 0) == 0) {
            // serve passes the config's args plus a few of its own: one this engine does not have is said and skipped
            // (with its value), not fatal
            const bool val = i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0;
            std::fprintf(stderr, "strata-m2: ignoring %s%s%s (not an option of this engine)\n", a.c_str(),
                         val ? " " : "", val ? argv[i + 1] : "");
            if (val) ++i;
        } else {
            std::fprintf(stderr,
                         "usage: strata-m2 --model GGUF [--max-context 32768] [--chunk 512] [--no-graph]\n"
                         "       (--serve | --tokens IDS [--n 64] [--force IDS] [--dump-full-logits F] | --bench-pp N "
                         "--bench-tg N)\n");
            return 2;
        }
    }
    if (model.empty()) { std::fprintf(stderr, "strata-m2: --model is required\n"); return 2; }
#if !defined(_WIN32)
    setenv("CUDA_MODULE_LOADING", "EAGER", 0);   // every kernel loaded with the context, not in the first prompt
    setenv("STRATA_Q8_PACKED", "1", 0);   // the packed Q8_0 decode copy unless STRATA_Q8_PACKED is set otherwise
    setenv("STRATA_MMQ_KQ", "1", 0);      // the K-quant MMQ prompt kernels (every expert of this family is a K-quant)
    // gfx103x (RX 6800): the prompt's dense GEMMs as SGEMMs on FP32 copies (prefill/gemm.cu rdna2_sgemm; other cards
    // keep the native call).  The native FP16-in / FP32-out rocBLAS GEMMs there gave NaN rows in this engine: a
    // 2048-token chunk went non-finite from layer 1 on (the logits NaN, the pick 0x7fffffff, a fault in embed_rows).
    // STRATA_RDNA2_SGEMM=0 turns it off.
    setenv("STRATA_RDNA2_SGEMM", "1", 0);
#else
    if (std::getenv("STRATA_MMQ_KQ") == nullptr) _putenv_s("STRATA_MMQ_KQ", "1");
    if (std::getenv("STRATA_Q8_PACKED") == nullptr) _putenv_s("STRATA_Q8_PACKED", "1");   // as on Linux (it was missing here)
    if (std::getenv("CUDA_MODULE_LOADING") == nullptr) _putenv_s("CUDA_MODULE_LOADING", "EAGER");
    if (std::getenv("STRATA_RDNA2_SGEMM") == nullptr) _putenv_s("STRATA_RDNA2_SGEMM", "1");   // as on Linux (above)
#endif
    if (mtp_d < 0 || mtp_d > 7) { std::fprintf(stderr, "strata-m2: --mtp takes 0 .. 7 drafts\n"); return 2; }
    if (chunk < 0 || chunk > 4096 || ctx < 256) { std::fprintf(stderr, "strata-m2: bad --chunk / --max-context\n"); return 2; }
    if (resident_gib == 0.0 || (resident_gib > 0 && resident_gib < 0.25)) {
        std::fprintf(stderr, "strata-m2: --resident-budget-gib takes a number of GiB (0.25 or more), e.g. "
                             "--resident-budget-gib 38; leave it out for every expert in RAM, or use --low-ram for none\n");
        return 2;
    }
    Model m;
    // The layer split: the second card gets its own copy of the weights of the layers it runs (the cards do
    // not share their memory), so the loader must know about it BEFORE it uploads the weights.
    if (layer_split > 0 && layer_split <= (int) m.g.n_layers) {
        int n_dev = 1;
        CK(cudaGetDeviceCount(&n_dev));
        CK(cudaGetLastError());
        if (n_dev < 2) {
            std::fprintf(stderr, "strata-m2: --layer-split needs a second visible GPU (%d visible)\n", n_dev);
            return 2;
        }
        g_second_card = 1;
        g_split_first = layer_split;
    }
    std::string err;
    try {
        if (!load(model, m, err, mtp_d > 0)) { std::fprintf(stderr, "strata-m2: %s\n", err.c_str()); return 1; }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strata-m2: %s\n", e.what());
        return 1;
    }
    Engine E;
    // The decode's graph is captured on the first card and replays ITS pointers: with the layer split the
    // decode's layers run on the second card, so the graph would read the first card's memory.  The split
    // runs the layers without the graph.
    E.use_graph = graph && layer_split <= 0;
    // the low-RAM mode: no host copy of the experts (they stay in the GGUF mapping and the CPU pool reads
    // them where they lie).  On a 64 GB box the copy of a 49 GB model does not fit beside the file: the
    // placement ran at 39-52 MB/s and the machine thrashed (3.5 GB of swap in and out).
    E.low_ram = low_ram || std::getenv("STRATA_M2_LOW_RAM") != nullptr;
    E.kv_mode = kv;
    if (E.kv_mode != Engine::KV_F32)
        std::fprintf(stderr, "strata-m2: the K / V cache in %s: %s the VRAM for the same context\n", E.kv_word(),
                     E.kv_mode == Engine::KV_I8 ? "a quarter of" : "half");
    if (E.kv_mode == Engine::KV_I8 && m.g.n_head_kv > kMaxKvHeads) {
        // the scales of a cell live in shared memory, one per KV head and per block (32 heads: 1 KiB for K and V)
        std::fprintf(stderr, "strata-m2: --kv int8 holds up to %d KV heads (this model has %d): the K / V cache stays "
                     "FP16\n", kMaxKvHeads, (int) m.g.n_head_kv);
        E.kv_mode = Engine::KV_F16;
    }
    if (E.low_ram && resident_gib > 0) {
        // both say what to do with the experts in host RAM: --low-ram says none
        std::fprintf(stderr, "strata-m2: --low-ram keeps no host copy: --resident-budget-gib %g ignored\n", resident_gib);
        resident_gib = -1;
    }
    // the second card's expert tier (--peer-device N).  It must be a device the engine can use: the engine
    // runs on device 0, so the peer is any other visible device.
    if (peer_device >= 0) {
        int n_dev = 1;
        CK(cudaGetDeviceCount(&n_dev));
        CK(cudaGetLastError());
        if (n_dev < 2 || peer_device >= n_dev) {
            std::fprintf(stderr, "strata-m2: --peer-device %d needs a second visible GPU (%d visible)\n", peer_device, n_dev);
            return 2;
        }
        // EXPERIMENTAL: the peer's tier is written but not validated on a real card yet (a fault on the peer
        // while the experts run there).  It stays behind an environment variable so it cannot be reached by
        // accident; the layer split (the other way to use a second card) is not written at all.
        if (std::getenv("STRATA_M2_PEER_EXPERIMENTAL") == nullptr) {
            std::fprintf(stderr, "strata-m2: the second card's expert tier is experimental and not validated yet "
                                 "(STRATA_M2_PEER_EXPERIMENTAL=1 to try it)\n");
            return 2;
        }
        E.peer_device = peer_device;
        std::fprintf(stderr, "strata-m2: the second card (device %d) takes part of the expert tier (experimental)\n",
                     peer_device);
    }
    if (layer_split > 0) {
        // The layer split: the layers from layer_split on the second card.  The first card keeps the embedding
        // and the first layers; the second card the rest, and the residual crosses once per window.
        int n_dev = 1;
        CK(cudaGetDeviceCount(&n_dev));
        CK(cudaGetLastError());
        if (n_dev < 2) {
            std::fprintf(stderr, "strata-m2: --layer-split needs a second visible GPU (%d visible)\n", n_dev);
            return 2;
        }
        E.split_dev = 1;
        E.split_layer = layer_split;
        std::fprintf(stderr, "strata-m2: the layers from %d on the second card (device 1)\n", layer_split);
    }
    if (!dump_layers_path.empty()) {
        E.dump_layers = std::fopen(dump_layers_path.c_str(), "wb");
        if (!E.dump_layers) { std::fprintf(stderr, "strata-m2: cannot write %s\n", dump_layers_path.c_str()); return 2; }
    }
    if (E.low_ram) std::fprintf(stderr, "strata-m2: low-RAM mode: the experts stay in the file, no host copy\n");
    // --vram-reserve-mib / --vram-reserve-later-mib: the VRAM kept free, on the first card and on the split's / the
    // peer's card.  A negative count is read as 0; the later card follows the first unless it is given.
    const size_t reserve_bytes = (size_t) std::max(0, vram_reserve_mib) << 20;
    const size_t reserve_later_bytes =
        (size_t) std::max(0, vram_reserve_later_mib >= 0 ? vram_reserve_later_mib : vram_reserve_mib) << 20;
    if (chunk == 0) {
        // auto: the largest chunk (4096 .. 1024) that fits beside every expert in VRAM; 2048 when they will not all fit (a prompt chunk streams every layer's
        // experts over PCIe once, so a bigger chunk pays that once for more tokens)
        size_t fr = 0, tot = 0, experts = 0;
        CK(cudaMemGetInfo(&fr, &tot));
        for (const Layer& L : m.layers) experts += L.stride * (size_t) m.g.n_expert;
        const size_t fr_raw = fr;   // before the packed Q8_0 copies count as given back
        // (the MTP block's experts can go to the CPU, the packed Q8_0 copies can be given back: see place_experts)
        size_t packed = 0;
        for (const auto& pk : m.packed) packed += std::get<2>(pk);
        fr += packed;
        // What has to fit beside the experts: the engine's buffers (kM2ChunkRowBytes a row, kM2FixedBufferBytes fixed),
        // the reserve, the sessions' K / V and the FP16 copies the prompt attention keeps (k16 / v16: 2 x heads x dim x 2
        // B per cell on each card, whatever the cache's own element).  The K / V term counts the layers THIS card runs
        // (--layer-split: the second card's half is on ITS card).
        const size_t n_kv_lay = m.layers.size() + (m.has_mtp ? 1 : 0);
        const size_t on0 = E.split_dev >= 0 ? std::min((size_t) E.split_layer, m.layers.size()) : n_kv_lay;
        const int ns = std::max(1, batch);
        const double kv_cell = (double) on0 * ((double) m.g.n_head_kv * (double) m.g.head_dim * 2.0 * (double) E.kv_esz() +
                                               (E.kv_mode == Engine::KV_I8 ? (double) m.g.n_head_kv * kKvBlocks * 2.0 * 4.0 : 0.0));
        const double kv_prompt = 2.0 * (double) m.g.n_head_kv * (double) m.g.head_dim * 2.0;
        const double base = kM2FixedBufferBytes + (double) reserve_bytes + kM2ChunkRowBytes * 512.0 +
                            kv_cell * (double) ctx * (double) ns + kv_prompt * (double) ctx;
        // the engine's buffers at 512 + the reserve + the caches: the least a chunk of 512 with all the experts costs
        const double want = (double) experts + base;
        const bool fits = expert_slots <= 0 && (expert_vram < 0 ? want <= (double) fr
                                                               : (double) experts <= expert_vram * (double) (1ull << 30));
        chunk = fits ? 512 : 2048;
        const double GiB = (double) (1ull << 30);
        if (fits && expert_vram < 0) {
            // a bigger prompt chunk feeds the MMQ expert kernels more rows per expert (R9700, UD-Q4_K_XL, pp4096: 512 3390, 1024
            // 4495, 2048 5353, 4096 5750 tok/s): the largest that fits beside the experts, whose buffers grow by 617200 B a
            // row on this path (measured, see kM2ChunkRowBytes) and whose caches are the kv_cell / kv_prompt above.
            // With several slots a long prompt would hold the other slots' windows for a whole chunk: 1024 at most there.
            // (the free room is read before init, after the weights: the engine's buffers and the sessions' caches are not
            // allocated yet, so `base` and the row term are what is still to come; no give-back of the packed decode
            // copies for a bigger chunk either - that is place_experts' call)
            const int cap = ns > 1 ? 1024 : 4096;
            for (int c : {4096, 2048, 1024})
                if (c <= cap && (double) experts + base + kM2ChunkRowBytes * (double) (c - 512) <= (double) fr_raw) {
                    chunk = c;
                    break;
                }
        } else if (!fits) {
            // the hybrid mode: 2048 (a chunk streams every layer's experts over PCIe once); a small card, where the chunk's buffers
            // would take the slots' room, falls back to less
            const double fr_gib = (double) fr / GiB;
            chunk = fr_gib >= 7.5 ? 2048 : fr_gib >= 5.5 ? 1024 : 512;
        }
        if (const char* cv = std::getenv("STRATA_Q36_CHUNK")) chunk = std::max(256, std::min(4096, std::atoi(cv)));   // force a chunk (512: the old default)
        if (std::getenv("STRATA_Q36_PLACE_DEBUG"))
            std::fprintf(stderr, "strata-m2: chunk rule: free %.2f GiB (raw %.2f, with %.2f packed), main experts %.2f GiB, chunk %d\n",
                         fr / 1073741824.0, fr_raw / 1073741824.0, packed / 1073741824.0, experts / 1073741824.0, chunk);
    }
    E.init(m, ctx, chunk, batch, vision);
    if (!place_experts(m, E, expert_vram, cpu_threads, hybrid_rows, expert_slots, resident_gib, reserve_bytes,
                       reserve_later_bytes, vram_reserve_given, err)) {
        std::fprintf(stderr, "strata-m2: %s\n", err.c_str());
        return 1;
    }
    struct Report {
        Engine& E;
        ~Report() {
            if (E.verify_k) std::fprintf(stderr, "strata-m2 VERIFY: %lld comparisons, %lld mismatches\n", E.v_calls, E.v_bad);
            if (E.verify_k)
                for (const auto& kv : E.v_by)
                    std::fprintf(stderr, "strata-m2 VERIFY   %s: %lld mismatches\n", kv.first.c_str(), kv.second);
            if (!E.m->hybrid) return;
            const long long n = E.hy_cpu + E.hy_gpu + E.hy_peer;
            char peer_txt[64] = "";
            if (E.hy_peer > 0) std::snprintf(peer_txt, sizeof peer_txt, ", %lld on the peer", E.hy_peer);
            std::fprintf(stderr, "strata-m2: hybrid experts: %lld of %lld batch entries on the GPU (%.1f%%), %lld on the "
                         "CPU pool%s in %.0f ms (waited %.0f ms for the routing), %lld slot replacements\n", E.hy_gpu, n,
                         n ? 100.0 * (double) E.hy_gpu / (double) n : 0.0,
                         E.hy_cpu, peer_txt, E.hy_cpu_ms, E.hy_wait_ms, E.hy_swaps);
        }
    } report{E};
    E.reset();
    E.mtp_d = m.has_mtp ? mtp_d : 0;
    if (draft_vocab < 0 && std::getenv("STRATA_Q36_DRAFT_VOCAB") != nullptr) draft_vocab = std::atoi(std::getenv("STRATA_Q36_DRAFT_VOCAB"));
    E.draft_n = std::max(0, draft_vocab);
    E.bmtp = (batch_mtp || (std::getenv("STRATA_BATCH_MTP") != nullptr && std::string(std::getenv("STRATA_BATCH_MTP")) != "0")) && E.n_sess > 1 && E.mtp_d > 0;
    if (batch_mtp && !E.bmtp) std::fprintf(stderr, "strata-m2: --batch-mtp needs --batch N (2 or more) and --mtp N: ignored\n");
    if (warm) {
        E.warmup();
        // The warm-up's rows are in the K / V cache: the real prompt must not read them.
        E.reset();
    }
    if (serve_mode) return serve(E);
    if (bench_pp > 0 || bench_tg > 0) {
        // llama-bench's shapes: a prompt of bench_pp tokens from a fresh state, then bench_tg tokens one at a time
        std::vector<int32_t> toks((size_t) std::max(bench_pp, 1));
        std::mt19937 rng(1234);
        for (auto& t : toks) t = (int32_t) (rng() % 200000u);
        for (int rep = 0; rep < 3; ++rep) {
            E.reset();
            const auto t0 = Clock::now();
            E.process(toks.data(), (int64_t) toks.size(), [](int64_t) {});
            CK(cudaStreamSynchronize(E.st));
            const double pp_ms = ms_since(t0);
            const auto t1 = Clock::now();
            int nx = E.greedy();
            for (int i = 0; i < bench_tg; ++i) {
                const int32_t one = nx;
                E.process(&one, 1, [](int64_t) {});
                nx = E.greedy();
            }
            const double tg_ms = ms_since(t1);
            std::printf("bench %d: pp%d %.1f tok/s, tg%d %.1f tok/s\n", rep, (int) toks.size(),
                        (double) toks.size() / (pp_ms / 1000.0), bench_tg,
                        bench_tg > 0 ? (double) bench_tg / (tg_ms / 1000.0) : 0.0);
            std::fflush(stdout);
        }
        return 0;
    }
    const std::vector<int32_t> prompt = parse_ids(tokens.c_str()), forced = parse_ids(force.c_str());
    if (prompt.empty()) { std::fprintf(stderr, "strata-m2: --tokens, --serve or --bench-pp is required\n"); return 2; }
    if (!forced.empty()) n_gen = (int) forced.size() + 1;
    FILE* ff = dump_full.empty() ? nullptr : std::fopen(dump_full.c_str(), "wb");
    std::vector<float> lg;
    std::vector<int32_t> gen;
    if (!emb_file.empty()) {
        if (!vision) { std::fprintf(stderr, "strata-m2: --emb needs --vision\n"); return 2; }
        ImgReq ir;
        const std::string ve = build_images(emb_file, prompt, m.g.n_embd, {}, ir);
        if (!ve.empty()) { std::fprintf(stderr, "strata-m2: %s\n", ve.c_str()); return 2; }
        E.set_images(ir.rows, ir.cell_row, ir.mrope, (int64_t) prompt.size(), ir.delta);
        std::fprintf(stderr, "strata-m2: %zu picture(s), %zu rows, text after them at cell + %d\n", ir.imgs.size(),
                     ir.rows.size() / (size_t) m.g.n_embd, (int) ir.delta);
    }
    const auto t0 = Clock::now();
    E.process(prompt.data(), (int64_t) prompt.size(), [](int64_t) {});
    const double pp_ms = ms_since(t0);
    const auto t1 = Clock::now();
    if (E.mtp_d > 0 && forced.empty() && !ff) {
        long long offered = 0, accepted = 0;
        std::vector<int32_t> fix, drafts;
        int32_t picks[16];
        int nx = E.greedy();
        while ((int) gen.size() < n_gen) {
            gen.push_back(nx);
            if ((int) gen.size() >= n_gen) break;
            const int a = E.spec_step(nx, fix, E.mtp_d, drafts, picks);
            offered += E.mtp_d;
            accepted += a;
            fix.assign(drafts.begin(), drafts.begin() + a);
            for (int i = 0; i < a && (int) gen.size() < n_gen; ++i) gen.push_back(drafts[(size_t) i]);
            nx = picks[a];
        }
        gen.resize((size_t) n_gen);
        std::printf("gen:");
        for (int t : gen) std::printf(" %d", t);
        std::printf("\n");
        std::fprintf(stderr, "strata-m2: prompt %zu tokens in %.1f ms, %zu generated in %.1f ms (%.1f tok/s), MTP %d drafts: "
                     "%lld of %lld accepted (%.0f%%)\n", prompt.size(), pp_ms, gen.size(), ms_since(t1),
                     gen.size() / (ms_since(t1) / 1000.0), E.mtp_d, accepted, offered,
                     offered ? 100.0 * accepted / offered : 0.0);
        return 0;
    }
    for (int i = 0; i < n_gen; ++i) {
        const int a = E.greedy();
        gen.push_back(a);
        if (ff) {
            E.host_logits(lg);
            std::fwrite(lg.data(), 4, lg.size(), ff);
        }
        if (i + 1 == n_gen) break;
        const int32_t nx = i < (int) forced.size() ? forced[(size_t) i] : a;
        E.process(&nx, 1, [](int64_t) {});
    }
    if (ff) std::fclose(ff);
    std::printf("gen:");
    for (int t : gen) std::printf(" %d", t);
    std::printf("\n");
    std::fprintf(stderr, "strata-m2: prompt %zu tokens in %.1f ms (%.1f tok/s), %zu generated in %.1f ms (%.1f tok/s)\n",
                 prompt.size(), pp_ms, prompt.size() / (pp_ms / 1000.0), gen.size(), ms_since(t1),
                 gen.size() / (ms_since(t1) / 1000.0));
    return 0;
}
