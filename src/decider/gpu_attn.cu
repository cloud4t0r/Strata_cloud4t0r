// src/decider/gpu_attn.cu - the decider's full-attention block on the GPU: per-head Q/K norms, partial RoPE,
// NON-CAUSAL grouped-query attention, the per-head gate and the output projection.
//
// WHAT MAKES THIS BLOCK DIFFERENT.  Every other attention Strata runs is causal; this one is not.  A token sees
// the whole sequence, including the questions after it and the assistant suffix that is never generated - the
// checkpoint was trained that way, and a causal mask would still produce a distribution, just the wrong one.  The
// per-head gate (`sigmoid` of the second half of the q projection) is the other thing only this family has.
//
// THE LAYOUT.  `attn_q.weight` holds, per head, [q(head_dim) | gate(head_dim)] - interleaved by head, not
// concatenated - and the GGUF's k/v are plain [kv_heads][head_dim].  RoPE is partial (the first 64 of 256 dims)
// with the rotate-half pairing (i, i + 32), which is what the reference package's rotate_half does.
#include "gpu_common.cuh"

#include "strata/decider/gpu.hpp"

namespace strata::decider::gpu {
namespace {

using namespace detail;

/// Above this the block's score array no longer fits in shared memory; the engine will tile instead.
constexpr int kMaxTokens = 4096;

/// RMSNorm over one head, then partial RoPE.  One block per (token, head); `row_stride` is the distance between
/// two heads of the same token (2 * head_dim for q, whose head carries the gate too; head_dim for k).
__global__ void norm_rope_kernel(float* buffer, const float* w, int per_t, int row_stride, int D, int n_rot,
                                 float theta, float eps) {
    const int row = blockIdx.x;
    const int t = row / per_t, sub = row % per_t;
    float* p = buffer + (size_t) (t * per_t + sub) * row_stride;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) ss += p[i] * p[i];
    __shared__ float shared[256];
    shared[threadIdx.x] = ss;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
        __syncthreads();
    }
    const float scale = rsqrtf(shared[0] / (float) D + eps);
    for (int i = threadIdx.x; i < D; i += blockDim.x) p[i] *= scale * w[i];
    __syncthreads();
    const int half = n_rot / 2;
    if (threadIdx.x < half) {
        const int i = threadIdx.x;
        const float frequency = powf(theta, -2.0f * (float) i / (float) n_rot);
        const float angle = (float) t * frequency;
        const float c = cosf(angle), s = sinf(angle);
        const float a = p[i], b = p[i + half];
        p[i] = a * c - b * s;
        p[i + half] = b * c + a * s;
    }
}

/// Non-causal GQA: one block per query head, the tokens walked in order.  Each key's score is one thread's dot
/// product over the head width; the softmax is over ALL tokens (no mask: the batch is one unpadded sequence), and
/// the output is multiplied by the head's gate before it leaves.
__global__ void attention_kernel(const float* qg, const float* k, const float* v, float* attn, int T, int H, int KV,
                                 int D, float scale) {
    const int h = blockIdx.x;
    const int kvh = h / (H / KV);
    extern __shared__ float shared[];
    float* qsh = shared;                 // [D]
    float* scores = shared + D;          // [T], then the softmax weights in place
    const int stride_q = H * 2 * D, stride_kv = KV * D;
    for (int t = 0; t < T; ++t) {
        if (threadIdx.x < D) qsh[threadIdx.x] = qg[(size_t) t * stride_q + h * 2 * D + threadIdx.x];
        __syncthreads();
        for (int s = threadIdx.x; s < T; s += blockDim.x) {
            const float* kr = k + (size_t) s * stride_kv + kvh * D;
            float acc = 0.0f;
            for (int d = 0; d < D; ++d) acc += qsh[d] * kr[d];
            scores[s] = acc * scale;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            float best = -INFINITY;
            for (int s = 0; s < T; ++s) best = fmaxf(best, scores[s]);
            float sum = 0.0f;
            for (int s = 0; s < T; ++s) {
                scores[s] = expf(scores[s] - best);
                sum += scores[s];
            }
            for (int s = 0; s < T; ++s) scores[s] /= sum;
        }
        __syncthreads();
        if (threadIdx.x < D) {
            float acc = 0.0f;
            for (int s = 0; s < T; ++s) acc += scores[s] * v[(size_t) s * stride_kv + kvh * D + threadIdx.x];
            const float gate = qg[(size_t) t * stride_q + h * 2 * D + D + threadIdx.x];
            attn[(size_t) t * (H * D) + h * D + threadIdx.x] = acc / (1.0f + expf(-gate));
        }
        __syncthreads();
    }
}

}  // namespace

bool attention(const std::string& gguf, int64_t block, const std::string& in_path, const std::string& out_path,
               std::string& err, const std::string& attn_path) {
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
        if (!recurrent || (size_t) block >= recurrent->items.size() || recurrent->items[(size_t) block].u != 0)
            fail("block " + std::to_string(block) + " is not a full-attention block");

        const int64_t H = get_int(g, "qwen35.attention.head_count");
        const int64_t KV = get_int(g, "qwen35.attention.head_count_kv");
        const int64_t D = get_int(g, "qwen35.attention.key_length");
        const int64_t n_rot = get_int(g, "qwen35.rope.dimension_count");
        const double theta = get_num(g, "qwen35.rope.freq_base");
        const double eps = get_num(g, "qwen35.attention.layer_norm_rms_epsilon");
        if (D > 256) fail("the kernels are written for a head width of up to 256");
        if (H % KV) fail("head_count does not split into head_count_kv");
        const float scale = 1.0f / sqrtf((float) D);

        const std::vector<float> input = read_f32(in_path);
        if (input.size() % (size_t) E) fail(in_path + " is not a multiple of the embedding width");
        const int64_t T = (int64_t) (input.size() / (size_t) E);
        if (T < 1) fail("empty input");
        if (T > kMaxTokens) fail("this harness holds the scores in shared memory: at most " +
                                 std::to_string(kMaxTokens) + " tokens");
        std::fprintf(stderr, "strata-decider-gpu: block %lld attention, %lld tokens, %lld heads / %lld kv x %lld\n",
                     (long long) block, (long long) T, (long long) H, (long long) KV, (long long) D);

        const std::vector<float> q_norm = read_tensor(g, *g.find(p + "attn_q_norm.weight"));
        const std::vector<float> k_norm = read_tensor(g, *g.find(p + "attn_k_norm.weight"));
        Device<float> d_qn = upload_floats(q_norm);
        Device<float> d_kn = upload_floats(k_norm);
        Device<float> d_x = upload_floats(input);
        Device<uint8_t> d_q = upload_tensor(g, *g.find(p + "attn_q.weight"));
        Device<uint8_t> d_k = upload_tensor(g, *g.find(p + "attn_k.weight"));
        Device<uint8_t> d_v = upload_tensor(g, *g.find(p + "attn_v.weight"));
        Device<uint8_t> d_o = upload_tensor(g, *g.find(p + "attn_output.weight"));

        Device<float> d_qg((size_t) T * H * 2 * D);
        Device<float> d_kv((size_t) T * KV * D * 2);   // k then v, one buffer each half
        Device<float> d_attn((size_t) T * H * D);
        Device<float> d_out((size_t) T * E);
        constexpr int kChunk = 8;
        const size_t scratch = strata::kernels::native_q8_1_bytes((int) E, kChunk);
        Device<uint8_t> d_q8(scratch);

        gemm_q8(d_q.data, d_x.data, d_qg.data, (int) E, (int) (H * 2 * D), (int) T, d_q8.data, nullptr);
        gemm_q8(d_k.data, d_x.data, d_kv.data, (int) E, (int) (KV * D), (int) T, d_q8.data, nullptr);
        gemm_q8(d_v.data, d_x.data, d_kv.data + (size_t) T * KV * D, (int) E, (int) (KV * D), (int) T, d_q8.data,
                nullptr);

        norm_rope_kernel<<<(unsigned) (T * H), (unsigned) D>>>(d_qg.data, d_qn.data, (int) H, (int) (2 * D), (int) D,
                                                               (int) n_rot, (float) theta, (float) eps);
        check(cudaGetLastError(), "norm_rope_kernel q");
        norm_rope_kernel<<<(unsigned) (T * KV), (unsigned) D>>>(d_kv.data, d_kn.data, (int) KV, (int) D, (int) D,
                                                                (int) n_rot, (float) theta, (float) eps);
        check(cudaGetLastError(), "norm_rope_kernel k");

        const size_t shared_bytes = (size_t) (2 * D + T) * sizeof(float);
        attention_kernel<<<(unsigned) H, 256, shared_bytes>>>(d_qg.data, d_kv.data,
                                                              d_kv.data + (size_t) T * KV * D, d_attn.data, (int) T,
                                                              (int) H, (int) KV, (int) D, scale);
        check(cudaGetLastError(), "attention_kernel");
        gemm_q8(d_o.data, d_attn.data, d_out.data, (int) (H * D), (int) E, (int) T, d_q8.data, nullptr);
        check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

        maybe_write(attn_path, d_attn.data, (size_t) T * H * D);
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
