// src/decider/gpu_mlp.cu - the decider's dense MLP on the GPU (see include/strata/decider/gpu.hpp).
//
// The block's MLP is the part of the model with no data dependency between tokens, so it is also the part where a
// kernel mistake cannot hide behind a recurrence: if the norm's epsilon, the SiLU or the matrix product is wrong,
// the output is wrong by a visible amount and nothing compensates.  That makes it the right first kernel.
//
// Everything the sequence needs goes through ONE matrix product per projection: `iq_mmvq` takes ncols columns, so
// a 98-token sequence is 98 columns of the same weights, quantized once per call.
#include "strata/decider/gpu.hpp"

#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>
#include <ggml.h>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::decider::gpu {
namespace {

using strata::GgufFile;
using strata::MetaType;
using strata::MetaValue;
using strata::TensorInfo;

[[noreturn]] void fail(const std::string& what) { throw std::runtime_error(what); }

void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) fail(std::string(what) + ": " + cudaGetErrorString(status));
}

int64_t get_int(const GgufFile& g, const std::string& key) {
    const MetaValue* v = g.get(key);
    if (!v) fail("missing " + key);
    if (!v->is_num()) fail(key + " is not a number");
    return (int64_t) v->num();
}

double get_num(const GgufFile& g, const std::string& key) {
    const MetaValue* v = g.get(key);
    if (!v) fail("missing " + key);
    if (!v->is_num()) fail(key + " is not a number");
    return v->num();
}

/// A device buffer freed on the way out (the tool is one shot; a pool would be for the engine).
template<typename T>
struct Device {
    T* data = nullptr;
    size_t count = 0;
    explicit Device(size_t n) : count(n) { check(cudaMalloc((void**) &data, n * sizeof(T)), "cudaMalloc"); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    Device(Device&& other) noexcept : data(other.data), count(other.count) { other.data = nullptr; }
    ~Device() { if (data) (void) cudaFree(data); }
};

/// Uploads a whole tensor as it is stored: Q8_0 rows stay Q8_0 (the matrix product reads them in place).
Device<uint8_t> upload_tensor(const GgufFile& g, const TensorInfo& t) {
    const uint64_t bytes = ggml_row_size((ggml_type) t.type, (int64_t) t.shape[0]) * t.shape[1];
    Device<uint8_t> buffer(bytes);
    check(cudaMemcpy(buffer.data, g.tensor_data(t), bytes, cudaMemcpyHostToDevice), "cudaMemcpy weights");
    return buffer;
}

Device<float> upload_floats(const std::vector<float>& values) {
    Device<float> buffer(values.size());
    check(cudaMemcpy(buffer.data, values.data(), values.size() * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy");
    return buffer;
}

/// out = x / rms(x) * w, one block per row.  `w` is the GGUF's own tensor: the +1 of the model's zero-centered
/// norms is already folded into it (the reference documents why).
__global__ void rmsnorm_kernel(const float* x, const float* w, float* y, int n, float eps) {
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

/// h = silu(gate) * up, elementwise over the whole [tokens][n_ff] block.
__global__ void swiglu_kernel(const float* gate, const float* up, float* out, size_t n) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float g = gate[i];
    out[i] = (g / (1.0f + expf(-g))) * up[i];
}

/// The engine's quantized matrix product takes at most MAX_NCOLS columns per call (8 to 16, per build), so a
/// sequence longer than that is a handful of calls.  `y` is column-major: [columns][n_out], as iq_mmvq writes it.
void gemm_q8(const void* weights, const float* x, float* y, int n_in, int n_out, int ncols, void* scratch,
             void* stream) {
    constexpr int kChunk = 8;
    for (int c = 0; c < ncols; c += kChunk) {
        const int n = ncols - c < kChunk ? ncols - c : kChunk;
        strata::kernels::quantize_q8_1_rows(x + (size_t) c * n_in, n, n_in, scratch, stream);
        strata::kernels::iq_mmvq(GGML_TYPE_Q8_0, weights, scratch, y + (size_t) c * n_out, n_in, n_out, n, stream);
    }
}

std::vector<float> read_f32(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) fail("cannot open " + path);
    const std::streamsize bytes = file.tellg();
    if (bytes <= 0 || bytes % (std::streamsize) sizeof(float)) fail(path + " is not a whole number of floats");
    file.seekg(0);
    std::vector<float> values((size_t) (bytes / (std::streamsize) sizeof(float)));
    if (!file.read((char*) values.data(), bytes)) fail("short read on " + path);
    return values;
}

void write_f32(const std::string& path, const std::vector<float>& values) {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file) fail("cannot write " + path);
    file.write((const char*) values.data(), (std::streamsize) (values.size() * sizeof(float)));
    if (!file) fail("short write on " + path);
}

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
        if (block < 0 || block >= blocks) fail("block " + std::to_string(block) + " is outside 0.." +
                                               std::to_string(blocks - 1));
        const double eps = get_num(g, "qwen35.attention.layer_norm_rms_epsilon");
        const std::string p = "blk." + std::to_string(block) + ".";

        const std::vector<float> input = read_f32(in_path);
        if (input.size() % (size_t) E) fail(in_path + " is not a multiple of the embedding width");
        const int64_t T = (int64_t) (input.size() / (size_t) E);
        if (T < 1) fail("empty input");
        std::fprintf(stderr, "strata-decider-gpu: block %lld, %lld tokens, embedding %lld, ffn %lld\n",
                     (long long) block, (long long) T, (long long) E, (long long) FF);

        const TensorInfo& t_post = *g.find(p + "post_attention_norm.weight");
        const TensorInfo& t_gate = *g.find(p + "ffn_gate.weight");
        const TensorInfo& t_up = *g.find(p + "ffn_up.weight");
        const TensorInfo& t_down = *g.find(p + "ffn_down.weight");

        // the small vectors, as the reference reads them
        std::vector<float> post((size_t) E);
        std::memcpy(post.data(), g.tensor_data(t_post), (size_t) E * sizeof(float));
        Device<float> d_post = upload_floats(post);
        Device<float> d_x = upload_floats(input);
        Device<uint8_t> d_gate = upload_tensor(g, t_gate);
        Device<uint8_t> d_up = upload_tensor(g, t_up);
        Device<uint8_t> d_down = upload_tensor(g, t_down);

        Device<float> d_xn((size_t) T * E);
        Device<float> d_g((size_t) T * FF);
        Device<float> d_u((size_t) T * FF);
        Device<float> d_h((size_t) T * FF);
        Device<float> d_out((size_t) T * E);
        constexpr int kChunk = 8;
        const size_t scratch_act = strata::kernels::native_q8_1_bytes((int) E, kChunk);
        const size_t scratch_h = strata::kernels::native_q8_1_bytes((int) FF, kChunk);
        Device<uint8_t> d_q8_act(scratch_act > scratch_h ? scratch_act : scratch_h);

        const int threads = 256;
        rmsnorm_kernel<<<(unsigned) T, threads>>>(d_x.data, d_post.data, d_xn.data, (int) E, (float) eps);
        check(cudaGetLastError(), "rmsnorm_kernel");
        gemm_q8(d_gate.data, d_xn.data, d_g.data, (int) E, (int) FF, (int) T, d_q8_act.data, nullptr);
        gemm_q8(d_up.data, d_xn.data, d_u.data, (int) E, (int) FF, (int) T, d_q8_act.data, nullptr);
        const size_t cells = (size_t) T * FF;
        const unsigned blocks_1d = (unsigned) ((cells + threads - 1) / threads);
        swiglu_kernel<<<blocks_1d, threads>>>(d_g.data, d_u.data, d_h.data, cells);
        check(cudaGetLastError(), "swiglu_kernel");
        gemm_q8(d_down.data, d_h.data, d_out.data, (int) FF, (int) E, (int) T, d_q8_act.data, nullptr);
        check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

        if (!xn_path.empty()) {
            std::vector<float> xn((size_t) T * E);
            check(cudaMemcpy(xn.data(), d_xn.data, xn.size() * sizeof(float), cudaMemcpyDeviceToHost),
                  "cudaMemcpy xn");
            write_f32(xn_path, xn);
        }
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
