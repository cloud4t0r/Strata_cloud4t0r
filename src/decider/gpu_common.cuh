// src/decider/gpu_common.cuh - what every decider kernel harness needs: the device buffers, the weight upload,
// the quantized matrix product and the raw f32 files the CPU reference writes.
//
// Internal to src/decider: the public surface is include/strata/decider/gpu.hpp.
#pragma once

#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>
#include <ggml.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::decider::gpu::detail {

using strata::GgufFile;
using strata::MetaType;
using strata::MetaValue;
using strata::TensorInfo;

[[noreturn]] inline void fail(const std::string& what) { throw std::runtime_error(what); }

inline void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) fail(std::string(what) + ": " + cudaGetErrorString(status));
}

inline int64_t get_int(const GgufFile& g, const std::string& key) {
    const MetaValue* v = g.get(key);
    if (!v) fail("missing " + key);
    if (!v->is_num()) fail(key + " is not a number");
    return (int64_t) v->num();
}

inline double get_num(const GgufFile& g, const std::string& key) {
    const MetaValue* v = g.get(key);
    if (!v) fail("missing " + key);
    if (!v->is_num()) fail(key + " is not a number");
    return v->num();
}

/// A device buffer freed on the way out (the harness is one shot; the engine will have a pool).
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
inline Device<uint8_t> upload_tensor(const GgufFile& g, const TensorInfo& t) {
    const uint64_t bytes = ggml_row_size((ggml_type) t.type, (int64_t) t.shape[0]) * t.shape[1];
    Device<uint8_t> buffer(bytes);
    check(cudaMemcpy(buffer.data, g.tensor_data(t), bytes, cudaMemcpyHostToDevice), "cudaMemcpy weights");
    return buffer;
}

inline Device<float> upload_floats(const std::vector<float>& values) {
    Device<float> buffer(values.size());
    check(cudaMemcpy(buffer.data, values.data(), values.size() * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy");
    return buffer;
}

/// A whole tensor read as floats, whatever its type (the small vectors: norms, ssm_a, dt_bias, the convolution).
inline std::vector<float> read_tensor(const GgufFile& g, const TensorInfo& t) {
    int64_t n = 1;
    for (uint64_t d : t.shape) n *= (int64_t) d;
    std::vector<float> values((size_t) n);
    if (t.type == GGML_TYPE_F32) {
        std::memcpy(values.data(), g.tensor_data(t), (size_t) n * sizeof(float));
        return values;
    }
    const auto* traits = ggml_get_type_traits((ggml_type) t.type);
    if (!traits || !traits->to_float) fail("no dequantizer for " + t.name);
    traits->to_float(g.tensor_data(t), values.data(), n);
    return values;
}

/// The engine's quantized matrix product takes at most MAX_NCOLS columns per call (8 to 16, per build), so a
/// sequence longer than that is a handful of calls.  `y` is column-major: [columns][n_out], as iq_mmvq writes it.
inline void gemm_q8(const void* weights, const float* x, float* y, int n_in, int n_out, int ncols, void* scratch,
                    void* stream) {
    constexpr int kChunk = 8;
    for (int c = 0; c < ncols; c += kChunk) {
        const int n = ncols - c < kChunk ? ncols - c : kChunk;
        strata::kernels::quantize_q8_1_rows(x + (size_t) c * n_in, n, n_in, scratch, stream);
        strata::kernels::iq_mmvq(GGML_TYPE_Q8_0, weights, scratch, y + (size_t) c * n_out, n_in, n_out, n, stream);
    }
}

inline std::vector<float> read_f32(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) fail("cannot open " + path);
    const std::streamsize bytes = file.tellg();
    if (bytes <= 0 || bytes % (std::streamsize) sizeof(float)) fail(path + " is not a whole number of floats");
    file.seekg(0);
    std::vector<float> values((size_t) (bytes / (std::streamsize) sizeof(float)));
    if (!file.read((char*) values.data(), bytes)) fail("short read on " + path);
    return values;
}

inline void write_f32(const std::string& path, const std::vector<float>& values) {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file) fail("cannot write " + path);
    file.write((const char*) values.data(), (std::streamsize) (values.size() * sizeof(float)));
    if (!file) fail("short write on " + path);
}

/// Copies a device buffer back and writes it, when a path was given.
inline void maybe_write(const std::string& path, const float* device, size_t n) {
    if (path.empty()) return;
    std::vector<float> values(n);
    check(cudaMemcpy(values.data(), device, n * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy dump");
    write_f32(path, values);
}

}  // namespace strata::decider::gpu::detail
