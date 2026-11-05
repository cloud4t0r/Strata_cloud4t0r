// src/decider/gpu_mlp.cu - the decider's dense MLP on the GPU (see include/strata/decider/gpu.hpp).
//
// The block's MLP is the part of the model with no data dependency between tokens, so it is also the part where a
// kernel mistake cannot hide behind a recurrence: if the norm's epsilon, the SiLU or the matrix product is wrong,
// the output is wrong by a visible amount and nothing compensates.  That makes it the right first kernel.
//
// Everything the sequence needs goes through ONE matrix product per projection: `gemm_q8` hands iq_mmvq as many
// columns as it takes, so a 98-token sequence is 13 calls of the same weights, not 98.
#include "gpu_common.cuh"
#include "gpu_kernels.cuh"

#include "strata/decider/gpu.hpp"

namespace strata::decider::gpu {
namespace kernels {

using namespace detail;

/// out = x / rms(x) * w, one block per row.  `w` is the GGUF's own tensor: the +1 of the model's zero-centered
/// norms is already folded into it (the CPU reference documents why).
__global__ void rmsnorm(const float* x, const float* w, float* y, int n, float eps) {
    const int row = blockIdx.x;
    const float* xr = x + (size_t) row * n;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += xr[i] * xr[i];
    __shared__ float shared[256];
    shared[threadIdx.x] = ss;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
        __syncthreads();
    }
    const float scale = rsqrtf(shared[0] / (float) n + eps);
    float* yr = y + (size_t) row * n;
    for (int i = threadIdx.x; i < n; i += blockDim.x) yr[i] = xr[i] * scale * w[i];
}

/// The readout: one block per code, the row dotted with the hidden state at a marker.
__global__ void readout(const float* hidden, const float* code_rows, float* logits, int codes, int n) {
    const int c = blockIdx.x;
    if (c >= codes) return;
    const float* row = code_rows + (size_t) c * n;
    float acc = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) acc += row[i] * hidden[i];
    __shared__ float shared[256];
    shared[threadIdx.x] = acc;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) logits[c] = shared[0];
}

/// x += y, the residual stream.
__global__ void add_in_place(float* x, const float* y, size_t n) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}

/// h = silu(gate) * up, elementwise over the whole [tokens][n_ff] block.
__global__ void swiglu(const float* gate, const float* up, float* out, size_t n) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float g = gate[i];
    out[i] = (g / (1.0f + expf(-g))) * up[i];
}

}  // namespace kernels

namespace {
using namespace detail;
using kernels::rmsnorm;
using kernels::swiglu;
using kernels::add_in_place;
using kernels::readout;
}  // namespace

bool mlp(const std::string& gguf, int64_t block, const std::string& in_path, const std::string& out_path,
         std::string& err, const std::string& xn_path) {
    try {
        const GgufFile g(strata::gguf_split_paths(gguf).front());
        const std::string why = strata::check_architecture_qwen35(g);
        if (!why.empty()) fail(why);
        const int64_t E = get_int(g, "qwen35.embedding_length");
        const int64_t FF = get_int(g, "qwen35.feed_forward_length");
        const int64_t blocks = get_int(g, "qwen35.block_count");
        if (block < 0 || block >= blocks)
            fail("block " + std::to_string(block) + " is outside 0.." + std::to_string(blocks - 1));
        const double eps = get_num(g, "qwen35.attention.layer_norm_rms_epsilon");
        const std::string p = "blk." + std::to_string(block) + ".";

        const std::vector<float> input = read_f32(in_path);
        if (input.size() % (size_t) E) fail(in_path + " is not a multiple of the embedding width");
        const int64_t T = (int64_t) (input.size() / (size_t) E);
        if (T < 1) fail("empty input");
        std::fprintf(stderr, "strata-decider-gpu: block %lld MLP, %lld tokens, embedding %lld, ffn %lld\n",
                     (long long) block, (long long) T, (long long) E, (long long) FF);

        const std::vector<float> post = read_tensor(g, *g.find(p + "post_attention_norm.weight"));
        Device<float> d_post = upload_floats(post);
        Device<float> d_x = upload_floats(input);
        Device<uint8_t> d_gate = upload_tensor(g, *g.find(p + "ffn_gate.weight"));
        Device<uint8_t> d_up = upload_tensor(g, *g.find(p + "ffn_up.weight"));
        Device<uint8_t> d_down = upload_tensor(g, *g.find(p + "ffn_down.weight"));

        Device<float> d_xn((size_t) T * E);
        Device<float> d_g((size_t) T * FF);
        Device<float> d_u((size_t) T * FF);
        Device<float> d_h((size_t) T * FF);
        Device<float> d_out((size_t) T * E);
        constexpr int kChunk = 8;
        const size_t scratch = std::max(strata::kernels::native_q8_1_bytes((int) E, kChunk),
                                        strata::kernels::native_q8_1_bytes((int) FF, kChunk));
        Device<uint8_t> d_q8(scratch);

        const int threads = 256;
        rmsnorm<<<(unsigned) T, threads>>>(d_x.data, d_post.data, d_xn.data, (int) E, (float) eps);
        check(cudaGetLastError(), "rmsnorm");
        gemm_q8(d_gate.data, d_xn.data, d_g.data, (int) E, (int) FF, (int) T, d_q8.data, nullptr);
        gemm_q8(d_up.data, d_xn.data, d_u.data, (int) E, (int) FF, (int) T, d_q8.data, nullptr);
        const size_t cells = (size_t) T * FF;
        swiglu<<<(unsigned) ((cells + threads - 1) / threads), threads>>>(d_g.data, d_u.data, d_h.data, cells);
        check(cudaGetLastError(), "swiglu");
        gemm_q8(d_down.data, d_h.data, d_out.data, (int) FF, (int) E, (int) T, d_q8.data, nullptr);
        check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

        maybe_write(xn_path, d_xn.data, (size_t) T * E);
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
