// src/decider/gpu_gdn.cu - the decider's Gated DeltaNet block on the GPU: causal convolution, the delta-rule
// scan, the gated norm and the output projection.
//
// WHY THIS IS THE HARD ONE.  The MLP has no dependency between tokens; this block has three.  The convolution
// carries four positions of history, the recurrent state is a 128x128 matrix per value head that every token
// rewrites, and the v/k head pairing is the GGUF's [v_per_k][head_k] order (v head g belongs to k head g % head_k,
// NOT g / 2 as the reference package's repeat_interleave would have it).  A mistake in any of them keeps the
// norms plausible and changes the direction.
//
// THE SCAN.  One block per value head, 16x16 threads, each holding an 8x8 tile of the state in registers.  A
// token needs two column reductions (S^T k for the correction, S^T q for the output) and one rank-1 update; the
// reductions go through shared memory, the rest stays in registers.  It is one block per head because the state
// is per head and the sequence is sequential - 32 heads run in parallel, the tokens inside a head do not.
#include "gpu_common.cuh"

#include "strata/decider/gpu.hpp"

namespace strata::decider::gpu {
namespace {

using namespace detail;

constexpr int kTile = 8;      // the state tile a thread holds, per side
constexpr int kHeads = 16;    // threads per side of the block: 16 x 8 = 128 = the head width

/// y[t][c] = silu(sum_j w[c][j] * x[t-(K-1)+j][c]): the causal depthwise convolution, tap j = 0 oldest.
__global__ void conv1d_silu_kernel(const float* x, const float* w, float* y, int T, int C, int K) {
    const long long index = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= (long long) T * C) return;
    const int t = (int) (index / C), c = (int) (index % C);
    float acc = 0.0f;
    for (int j = 0; j < K; ++j) {
        const int src = t - (K - 1) + j;
        if (src >= 0) acc += w[(size_t) c * K + j] * x[(size_t) src * C + c];
    }
    y[index] = acc / (1.0f + expf(-acc));
}

/// softplus with the reference's threshold, so a large a + dt_bias does not overflow to infinity.
__device__ __forceinline__ float softplus(float x) { return x > 20.0f ? x : log1pf(expf(x)); }

/// The delta rule for one value head over the whole sequence: S is [KD][VD], kept in registers.
__global__ void gdn_scan_kernel(const float* conv, const float* a, const float* b, const float* ssm_a,
                                const float* dt, float* core, int T, int VH, int KH, int KD, int VD, int key_dim,
                                int conv_dim, float scale, float eps) {
    const int h = blockIdx.x;
    const int kh = h % KH;   // the GGUF's [v_per_k][head_k] order
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * kHeads + tx;
    const int value_dim = VH * VD;

    __shared__ float qsh[128], ksh[128], vsh[128], kvsh[128], dsh[128];
    __shared__ float red[kHeads][128];
    __shared__ float s_decay, s_beta;

    float S[kTile][kTile];
    for (int i = 0; i < kTile; ++i)
        for (int j = 0; j < kTile; ++j) S[i][j] = 0.0f;

    for (int t = 0; t < T; ++t) {
        const float* row = conv + (size_t) t * conv_dim;
        if (tid < KD) {
            qsh[tid] = row[kh * KD + tid];
            ksh[tid] = row[key_dim + kh * KD + tid];
        }
        if (tid < VD) vsh[tid] = row[2 * key_dim + h * VD + tid];
        __syncthreads();
        if (tid == 0) {
            float qs = 0.0f, ks = 0.0f;
            for (int i = 0; i < KD; ++i) {
                qs += qsh[i] * qsh[i];
                ks += ksh[i] * ksh[i];
            }
            const float qn = rsqrtf(qs + eps), kn = rsqrtf(ks + eps);
            for (int i = 0; i < KD; ++i) {
                qsh[i] *= qn * scale;
                ksh[i] *= kn;
            }
            const float gate = ssm_a[h] * softplus(a[t * VH + h] + dt[h]);   // ssm_a is negative: decay <= 1
            s_decay = expf(gate);
            s_beta = 1.0f / (1.0f + expf(-b[t * VH + h]));
        }
        __syncthreads();
        const float decay = s_decay, beta = s_beta;

        for (int i = 0; i < kTile; ++i)
            for (int j = 0; j < kTile; ++j) S[i][j] *= decay;

        // kv = S^T k: each thread sums its 8 rows, then the 16 row-groups are summed per column
        float part[kTile];
        for (int j = 0; j < kTile; ++j) part[j] = 0.0f;
        for (int i = 0; i < kTile; ++i) {
            const float ki = ksh[ty * kTile + i];
            for (int j = 0; j < kTile; ++j) part[j] += S[i][j] * ki;
        }
        for (int j = 0; j < kTile; ++j) red[ty][tx * kTile + j] = part[j];
        __syncthreads();
        if (ty == 0) {
            for (int j = 0; j < kTile; ++j) {
                const int col = tx * kTile + j;
                float sum = 0.0f;
                for (int r = 0; r < kHeads; ++r) sum += red[r][col];
                kvsh[col] = sum;
            }
        }
        __syncthreads();
        if (ty == 0)
            for (int j = 0; j < kTile; ++j) {
                const int col = tx * kTile + j;
                dsh[col] = (vsh[col] - kvsh[col]) * beta;
            }
        __syncthreads();

        for (int i = 0; i < kTile; ++i) {
            const float ki = ksh[ty * kTile + i];
            for (int j = 0; j < kTile; ++j) S[i][j] += ki * dsh[tx * kTile + j];
        }

        // out = S^T q, the same reduction
        for (int j = 0; j < kTile; ++j) part[j] = 0.0f;
        for (int i = 0; i < kTile; ++i) {
            const float qi = qsh[ty * kTile + i];
            for (int j = 0; j < kTile; ++j) part[j] += S[i][j] * qi;
        }
        __syncthreads();   // red is reused: everyone is done reading the previous reduction
        for (int j = 0; j < kTile; ++j) red[ty][tx * kTile + j] = part[j];
        __syncthreads();
        if (ty == 0) {
            float* out_row = core + (size_t) t * value_dim + h * VD;
            for (int j = 0; j < kTile; ++j) {
                const int col = tx * kTile + j;
                float sum = 0.0f;
                for (int r = 0; r < kHeads; ++r) sum += red[r][col];
                out_row[col] = sum;
            }
        }
        __syncthreads();
    }
}

/// The gated norm: out = rms(x) * w * silu(z), one block per (token, value head) row of VD.
__global__ void gated_norm_kernel(const float* x, const float* z, const float* w, float* out, int n, float eps) {
    const int row = blockIdx.x;
    const float* xr = x + (size_t) row * n;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += xr[i] * xr[i];
    __shared__ float shared[128];
    shared[threadIdx.x] = ss;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
        __syncthreads();
    }
    const float scale = rsqrtf(shared[0] / (float) n + eps);
    float* orow = out + (size_t) row * n;
    const float* zrow = z + (size_t) row * n;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float gate = zrow[i];
        orow[i] = xr[i] * scale * w[i] * (gate / (1.0f + expf(-gate)));
    }
}

}  // namespace

bool gdn(const std::string& gguf, int64_t block, const std::string& in_path, const std::string& out_path,
         std::string& err, const std::string& conv_path, const std::string& core_path) {
    try {
        const GgufFile g(strata::gguf_split_paths(gguf).front());
        const std::string why = strata::check_architecture_qwen35(g);
        if (!why.empty()) fail(why);
        const int64_t E = get_int(g, "qwen35.embedding_length");
        const int64_t blocks = get_int(g, "qwen35.block_count");
        if (block < 0 || block >= blocks)
            fail("block " + std::to_string(block) + " is outside 0.." + std::to_string(blocks - 1));
        const std::string p = "blk." + std::to_string(block) + ".";
        const MetaValue* recurrent = g.get("qwen35.attention.recurrent_layers");
        if (!recurrent || (size_t) block >= recurrent->items.size() || recurrent->items[(size_t) block].u == 0)
            fail("block " + std::to_string(block) + " is not a Gated DeltaNet block");

        const int64_t KH = get_int(g, "qwen35.ssm.group_count");
        const int64_t VH = get_int(g, "qwen35.ssm.time_step_rank");
        const int64_t KD = get_int(g, "qwen35.ssm.state_size");
        const int64_t VD = KD;
        const int64_t K = get_int(g, "qwen35.ssm.conv_kernel");
        const int64_t key_dim = KH * KD, value_dim = VH * VD, conv_dim = 2 * key_dim + value_dim;
        const double eps = get_num(g, "qwen35.attention.layer_norm_rms_epsilon");
        if (KD > 128 || VD > 128) fail("the scan kernel is written for a head width of 128");
        const float scale = 1.0f / sqrtf((float) KD);

        const std::vector<float> input = read_f32(in_path);
        if (input.size() % (size_t) E) fail(in_path + " is not a multiple of the embedding width");
        const int64_t T = (int64_t) (input.size() / (size_t) E);
        if (T < 1) fail("empty input");
        std::fprintf(stderr,
                     "strata-decider-gpu: block %lld GDN, %lld tokens, %lld k heads x %lld, %lld v heads x %lld\n",
                     (long long) block, (long long) T, (long long) KH, (long long) KD, (long long) VH, (long long) VD);

        const std::vector<float> ssm_a = read_tensor(g, *g.find(p + "ssm_a"));
        const std::vector<float> dt = read_tensor(g, *g.find(p + "ssm_dt.bias"));
        const std::vector<float> conv_w = read_tensor(g, *g.find(p + "ssm_conv1d.weight"));
        const std::vector<float> ssm_norm = read_tensor(g, *g.find(p + "ssm_norm.weight"));
        Device<float> d_a = upload_floats(ssm_a);
        Device<float> d_dt = upload_floats(dt);
        Device<float> d_convw = upload_floats(conv_w);
        Device<float> d_norm = upload_floats(ssm_norm);
        Device<float> d_x = upload_floats(input);
        Device<uint8_t> d_qkv = upload_tensor(g, *g.find(p + "attn_qkv.weight"));
        Device<uint8_t> d_z = upload_tensor(g, *g.find(p + "attn_gate.weight"));
        Device<uint8_t> d_alpha = upload_tensor(g, *g.find(p + "ssm_alpha.weight"));
        Device<uint8_t> d_beta = upload_tensor(g, *g.find(p + "ssm_beta.weight"));
        Device<uint8_t> d_outw = upload_tensor(g, *g.find(p + "ssm_out.weight"));

        Device<float> d_mixed((size_t) T * conv_dim);
        Device<float> d_zv((size_t) T * value_dim);
        Device<float> d_av((size_t) T * VH);
        Device<float> d_bv((size_t) T * VH);
        Device<float> d_conv((size_t) T * conv_dim);
        Device<float> d_core((size_t) T * value_dim);
        Device<float> d_normed((size_t) T * value_dim);
        Device<float> d_out((size_t) T * E);
        constexpr int kChunk = 8;
        const size_t scratch = strata::kernels::native_q8_1_bytes((int) (conv_dim > E ? conv_dim : E), kChunk);
        Device<uint8_t> d_q8(scratch);

        gemm_q8(d_qkv.data, d_x.data, d_mixed.data, (int) E, (int) conv_dim, (int) T, d_q8.data, nullptr);
        gemm_q8(d_z.data, d_x.data, d_zv.data, (int) E, (int) value_dim, (int) T, d_q8.data, nullptr);
        gemm_q8(d_alpha.data, d_x.data, d_av.data, (int) E, (int) VH, (int) T, d_q8.data, nullptr);
        gemm_q8(d_beta.data, d_x.data, d_bv.data, (int) E, (int) VH, (int) T, d_q8.data, nullptr);

        const long long cells = (long long) T * conv_dim;
        conv1d_silu_kernel<<<(unsigned) ((cells + 255) / 256), 256>>>(d_mixed.data, d_convw.data, d_conv.data,
                                                                     (int) T, (int) conv_dim, (int) K);
        check(cudaGetLastError(), "conv1d_silu_kernel");
        gdn_scan_kernel<<<(unsigned) VH, dim3(kHeads, kHeads)>>>(d_conv.data, d_av.data, d_bv.data, d_a.data,
                                                                 d_dt.data, d_core.data, (int) T, (int) VH, (int) KH,
                                                                 (int) KD, (int) VD, (int) key_dim, (int) conv_dim,
                                                                 scale, (float) eps);
        check(cudaGetLastError(), "gdn_scan_kernel");
        gated_norm_kernel<<<(unsigned) (T * VH), 128>>>(d_core.data, d_zv.data, d_norm.data, d_normed.data, (int) VD,
                                                        (float) eps);
        check(cudaGetLastError(), "gated_norm_kernel");
        gemm_q8(d_outw.data, d_normed.data, d_out.data, (int) value_dim, (int) E, (int) T, d_q8.data, nullptr);
        check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

        maybe_write(conv_path, d_conv.data, (size_t) T * conv_dim);
        maybe_write(core_path, d_core.data, (size_t) T * value_dim);
        std::vector<float> result((size_t) T * E);
        check(cudaMemcpy(result.data(), d_out.data, result.size() * sizeof(float), cudaMemcpyDeviceToHost),
              "cudaMemcpy result");
        write_f32(out_path, result);
        return true;
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
}

}  // namespace strata::decider::gpu
