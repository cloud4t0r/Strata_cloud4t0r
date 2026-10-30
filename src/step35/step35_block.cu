// src/step35/step35_block.cu - one dense Step-3.7-Flash block on the GPU, with the file's own weights.
// See include/strata/step35/block.hpp for the order of the steps.
#include "strata/step35/block.hpp"

#include "strata/step35/attn.hpp"
#include "strata/step35/gemv.hpp"
#include "strata/step35/moe.hpp"
#include "strata/step35/kernels.hpp"

#include <cuda_runtime.h>
#include <ggml.h>

#include <cstring>
#include <vector>

namespace strata::step35::kernels {
namespace {

using strata::GgufModel;
using strata::MetaType;
using strata::MetaValue;
using strata::TensorInfo;

double num(const GgufModel& f, const std::string& key, bool required, double fallback = 0) {
    const MetaValue* v = f.meta().get(key);
    if (!v) {
        if (required) throw std::runtime_error("missing " + key);
        return fallback;
    }
    return v->num();
}

/// One tensor, raw (the quantized bytes exactly as the file stores them), on the device.
void* upload(const GgufModel& f, const std::string& name, size_t& bytes) {
    size_t shard = 0;
    const TensorInfo* t = f.find(name, &shard);
    if (!t) throw std::runtime_error("missing tensor " + name);
    bytes = strata::tensor_payload_bytes(*t);
    if (bytes == 0) throw std::runtime_error("tensor " + name + " has no payload size");
    void* d = nullptr;
    if (cudaMalloc(&d, bytes) != cudaSuccess) throw std::runtime_error("out of device memory for " + name);
    if (cudaMemcpy(d, f.shard(shard).tensor_data(*t), bytes, cudaMemcpyHostToDevice) != cudaSuccess)
        throw std::runtime_error("copy failed for " + name);
    return d;
}

__global__ void k_split_gate(float* attn, const float* gate, int n_head, int head_dim) {
    const int h = blockIdx.x;
    const float s = 1.0f / (1.0f + expf(-gate[h]));
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) attn[(size_t) h * head_dim + d] *= s;
}

__global__ void k_silu_mul(float* act, const float* up, int n, float clamp) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = act[i], u = up[i];
    g = 1.0f / (1.0f + expf(-g)) * g;   // silu
    if (clamp > 1e-6f) {
        if (u > clamp) u = clamp;
        if (u < -clamp) u = -clamp;
        if (g > clamp) g = clamp;
    }
    act[i] = g * u;
}

__global__ void k_add(float* x, const float* y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}

}  // namespace

bool upload_block(const GgufModel& file, int layer, DenseBlock& b, std::string& err, bool with_ffn) {
    try {
        const std::string p = "step35.";
        const std::string pre = "blk." + std::to_string(layer) + ".";
        size_t bytes = 0;
        b.n_embd = (int) num(file, p + "embedding_length", true);
        b.head_dim = (int) num(file, p + "attention.key_length", true);
        b.ff = (int) num(file, p + "feed_forward_length", true);
        b.eps = (float) num(file, p + "attention.layer_norm_rms_epsilon", true);
        b.base = num(file, p + "rope.freq_base", true);
        // the per-layer arrays, by hand: the reader samples 64 entries, which covers 45 blocks
        for (const char* key : {"attention.head_count", "attention.head_count_kv", "attention.sliding_window_pattern"}) {
            const MetaValue* v = file.meta().get(p + key);
            if (!v || v->type != MetaType::ARRAY || (int) v->items.size() <= layer)
                throw std::runtime_error(std::string("cannot read ") + p + key + " for block " + std::to_string(layer));
        }
        b.n_head = (int) file.meta().get(p + "attention.head_count")->items[(size_t) layer].num();
        b.n_head_kv = (int) file.meta().get(p + "attention.head_count_kv")->items[(size_t) layer].num();
        b.sliding = file.meta().get(p + "attention.sliding_window_pattern")->items[(size_t) layer].u != 0;
        // llama.cpp's rule for this file: a sliding block rotates head_dim dims, a full one half of them
        b.n_rot = b.sliding ? b.head_dim : b.head_dim / 2;
        if (b.sliding) b.base = num(file, p + "rope.freq_base_swa", false, b.base);

        b.q = upload(file, pre + "attn_q.weight", bytes);
        b.t_q = (int) file.find(pre + "attn_q.weight")->type;
        b.k = upload(file, pre + "attn_k.weight", bytes);
        b.t_k = (int) file.find(pre + "attn_k.weight")->type;
        b.v = upload(file, pre + "attn_v.weight", bytes);
        b.t_v = (int) file.find(pre + "attn_v.weight")->type;
        b.wo = upload(file, pre + "attn_output.weight", bytes);
        b.t_wo = (int) file.find(pre + "attn_output.weight")->type;
        b.gate = upload(file, pre + "attn_gate.weight", bytes);
        b.t_gate = (int) file.find(pre + "attn_gate.weight")->type;
        b.attn_norm = (float*) upload(file, pre + "attn_norm.weight", bytes);
        b.ffn_norm = (float*) upload(file, pre + "ffn_norm.weight", bytes);
        b.q_norm = (float*) upload(file, pre + "attn_q_norm.weight", bytes);
        b.k_norm = (float*) upload(file, pre + "attn_k_norm.weight", bytes);
        if (with_ffn) {
            b.ffn_gate = upload(file, pre + "ffn_gate.weight", bytes);
            b.t_ffn = (int) file.find(pre + "ffn_gate.weight")->type;
            b.ffn_up = upload(file, pre + "ffn_up.weight", bytes);
            b.ffn_down = upload(file, pre + "ffn_down.weight", bytes);
        }
        // the frequency factors are one shared tensor, and only the full-attention blocks use it
        if (!b.sliding && file.find("rope_freqs.weight")) {
            size_t ff_bytes = 0;
            b.rope_ff = (float*) upload(file, "rope_freqs.weight", ff_bytes);
            b.n_ff_rope = (int) (ff_bytes / sizeof(float));
        }
        return true;
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
}

void release_block(DenseBlock& b) {
    for (void** p : {&b.q, &b.k, &b.v, &b.wo, &b.gate, &b.ffn_gate, &b.ffn_up, &b.ffn_down,
                     (void**) &b.attn_norm, (void**) &b.ffn_norm, (void**) &b.q_norm, (void**) &b.k_norm,
                     (void**) &b.rope_ff}) {
        if (*p) cudaFree(*p);
        *p = nullptr;
    }
}

bool run_dense_block(const DenseBlock& b, float* h, int pos, void* scratch_q8, int n_scratch, std::string& err) {
    (void) n_scratch;
    const int e = b.n_embd, hd = b.head_dim, nh = b.n_head, nkv = b.n_head_kv;
    float *x = nullptr, *q = nullptr, *k = nullptr, *v = nullptr, *ao = nullptr, *gt = nullptr, *res = nullptr;
    float *g = nullptr, *u = nullptr, *d = nullptr, *a = nullptr;
    auto alloc = [&](float** p, size_t n) {
        if (cudaMalloc(p, n * sizeof(float)) != cudaSuccess) throw std::runtime_error("out of device memory");
    };
    try {
        alloc(&x, e);
        alloc(&q, (size_t) nh * hd);
        alloc(&k, (size_t) nkv * hd);
        alloc(&v, (size_t) nkv * hd);
        alloc(&ao, (size_t) nh * hd);
        alloc(&gt, nh);
        alloc(&res, e);
        alloc(&g, b.ff);
        alloc(&u, b.ff);
        alloc(&d, e);
        alloc(&a, b.ff);
    } catch (const std::exception& ex) {
        err = ex.what();
        return false;
    }

    cudaMemcpy(x, h, (size_t) e * sizeof(float), cudaMemcpyDeviceToDevice);
    // attn_norm: the same kernel as the per-head RMSNorm, with one row of n_embd
    per_head_rms_norm(x, 1, e, b.attn_norm, b.eps, nullptr);
    gemv((ggml_type) b.t_q, b.q, x, q, e, nh * hd, scratch_q8, nullptr);
    gemv((ggml_type) b.t_k, b.k, x, k, e, nkv * hd, scratch_q8, nullptr);
    gemv((ggml_type) b.t_v, b.v, x, v, e, nkv * hd, scratch_q8, nullptr);
    per_head_rms_norm(q, nh, hd, b.q_norm, b.eps, nullptr);
    per_head_rms_norm(k, nkv, hd, b.k_norm, b.eps, nullptr);
    int* d_pos = nullptr;
    if (cudaMalloc(&d_pos, (size_t) (nh > nkv ? nh : nkv) * sizeof(int)) != cudaSuccess) {
        err = "out of device memory for positions";
        return false;
    }
    std::vector<int> pos_host((size_t) (nh > nkv ? nh : nkv), pos);
    cudaMemcpy(d_pos, pos_host.data(), pos_host.size() * sizeof(int), cudaMemcpyHostToDevice);
    rope_neox(q, nh, hd, b.n_rot, d_pos, b.base, b.rope_ff, b.n_ff_rope, nullptr);
    rope_neox(k, nkv, hd, b.n_rot, d_pos, b.base, b.rope_ff, b.n_ff_rope, nullptr);
    // one position: the attention kernel still runs, so the block goes through the same arithmetic
    attn_causal(q, k, v, ao, 1, nh, nkv, hd, 0, nullptr);
    gemv((ggml_type) b.t_gate, b.gate, x, gt, e, nh, scratch_q8, nullptr);
    k_split_gate<<<nh, 128>>>(ao, gt, nh, hd);
    gemv((ggml_type) b.t_wo, b.wo, ao, res, nh * hd, e, scratch_q8, nullptr);
    k_add<<<(e + 255) / 256, 256>>>(h, res, e);
    // the dense feed-forward
    cudaMemcpy(x, h, (size_t) e * sizeof(float), cudaMemcpyDeviceToDevice);
    per_head_rms_norm(x, 1, e, b.ffn_norm, b.eps, nullptr);
    gemv((ggml_type) b.t_ffn, b.ffn_gate, x, g, e, b.ff, scratch_q8, nullptr);
    gemv((ggml_type) b.t_ffn, b.ffn_up, x, u, e, b.ff, scratch_q8, nullptr);
    cudaMemcpy(a, g, (size_t) b.ff * sizeof(float), cudaMemcpyDeviceToDevice);
    k_silu_mul<<<(b.ff + 255) / 256, 256>>>(a, u, b.ff, 0.0f);
    gemv((ggml_type) b.t_ffn, b.ffn_down, a, d, b.ff, e, scratch_q8, nullptr);
    k_add<<<(e + 255) / 256, 256>>>(h, d, e);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        err = std::string("a kernel failed: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    cudaFree(d_pos);
    for (float* p : {x, q, k, v, ao, gt, res, g, u, d, a}) cudaFree(p);
    return true;
}

// -------------------------------------------------------------------------------------- the mixture of experts

namespace {

__global__ void k_acc_scaled(float* acc, const float* y, float w, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) acc[i] += w * y[i];
}

/// The attention half, shared by both block kinds: everything up to and including the second residual add of
/// the attention (h += wo(gate(attn(q,k,v)))).
bool run_attention(const DenseBlock& b, float* h, int pos, void* q8, std::string& err) {
    const int e = b.n_embd, hd = b.head_dim, nh = b.n_head, nkv = b.n_head_kv;
    float *x = nullptr, *q = nullptr, *k = nullptr, *v = nullptr, *ao = nullptr, *gt = nullptr, *res = nullptr;
    auto alloc = [](float** p, size_t n) { return cudaMalloc(p, n * sizeof(float)) == cudaSuccess; };
    if (!alloc(&x, e) || !alloc(&q, (size_t) nh * hd) || !alloc(&k, (size_t) nkv * hd) ||
        !alloc(&v, (size_t) nkv * hd) || !alloc(&ao, (size_t) nh * hd) || !alloc(&gt, nh) || !alloc(&res, e)) {
        err = "out of device memory";
        return false;
    }
    int* d_pos = nullptr;
    std::vector<int> pos_host((size_t) (nh > nkv ? nh : nkv), pos);
    bool ok = cudaMalloc(&d_pos, pos_host.size() * sizeof(int)) == cudaSuccess;
    if (ok) cudaMemcpy(d_pos, pos_host.data(), pos_host.size() * sizeof(int), cudaMemcpyHostToDevice);
    if (ok) ok = cudaMemcpy(x, h, (size_t) e * sizeof(float), cudaMemcpyDeviceToDevice) == cudaSuccess;
    if (ok) per_head_rms_norm(x, 1, e, b.attn_norm, b.eps, nullptr);
    if (ok) gemv((ggml_type) b.t_q, b.q, x, q, e, nh * hd, q8, nullptr);
    if (ok) gemv((ggml_type) b.t_k, b.k, x, k, e, nkv * hd, q8, nullptr);
    if (ok) gemv((ggml_type) b.t_v, b.v, x, v, e, nkv * hd, q8, nullptr);
    if (ok) per_head_rms_norm(q, nh, hd, b.q_norm, b.eps, nullptr);
    if (ok) per_head_rms_norm(k, nkv, hd, b.k_norm, b.eps, nullptr);
    if (ok) rope_neox(q, nh, hd, b.n_rot, d_pos, b.base, b.rope_ff, b.n_ff_rope, nullptr);
    if (ok) rope_neox(k, nkv, hd, b.n_rot, d_pos, b.base, b.rope_ff, b.n_ff_rope, nullptr);
    if (ok) attn_causal(q, k, v, ao, 1, nh, nkv, hd, 0, nullptr);
    if (ok) gemv((ggml_type) b.t_gate, b.gate, x, gt, e, nh, q8, nullptr);
    if (ok) k_split_gate<<<nh, 128>>>(ao, gt, nh, hd);
    if (ok) gemv((ggml_type) b.t_wo, b.wo, ao, res, nh * hd, e, q8, nullptr);
    if (ok) k_add<<<(e + 255) / 256, 256>>>(h, res, e);
    if (ok && cudaDeviceSynchronize() != cudaSuccess) ok = false;
    if (!ok) err = "a kernel failed in the attention half";
    if (d_pos) cudaFree(d_pos);
    for (float* p : {x, q, k, v, ao, gt, res}) if (p) cudaFree(p);
    return ok;
}

}  // namespace

bool upload_moe_block(const GgufModel& file, int layer, MoeBlock& b, std::string& err) {
    try {
        const std::string p = "step35.";
        const std::string pre = "blk." + std::to_string(layer) + ".";
        if (!upload_block(file, layer, b.attn, err, false)) return false;
        size_t bytes = 0;
        b.n_expert = (int) num(file, p + "expert_count", true);
        b.top_k = (int) num(file, p + "expert_used_count", true);
        b.n_ff_exp = (int) num(file, p + "expert_feed_forward_length", true);
        b.n_ff_shexp = (int) num(file, p + "expert_shared_feed_forward_length", true);
        b.scale = (float) num(file, p + "expert_weights_scale", true);
        const MetaValue* nrm = file.meta().get(p + "expert_weights_norm");
        b.weights_norm = nrm && nrm->type == MetaType::BOOL && nrm->u != 0;
        // the per-block clamp arrays (layers 43 and 44 in the release)
        for (const auto& c : {std::make_pair("swiglu_clamp_exp", &b.clamp_exp),
                              std::make_pair("swiglu_clamp_shexp", &b.clamp_shexp)}) {
            const MetaValue* v = file.meta().get(p + c.first);
            if (v && v->type == MetaType::ARRAY && (int) v->items.size() > layer) *c.second = (float) v->items[(size_t) layer].num();
        }
        b.router = upload(file, pre + "ffn_gate_inp.weight", bytes);
        if (file.find(pre + "exp_probs_b.bias")) b.select_bias = (float*) upload(file, pre + "exp_probs_b.bias", bytes);
        b.exps_gate = upload(file, pre + "ffn_gate_exps.weight", bytes);
        b.expert_gate_bytes = bytes / (size_t) b.n_expert;
        b.t_exps = (int) file.find(pre + "ffn_gate_exps.weight")->type;
        b.exps_up = upload(file, pre + "ffn_up_exps.weight", bytes);
        b.exps_down = upload(file, pre + "ffn_down_exps.weight", bytes);
        b.expert_down_bytes = bytes / (size_t) b.n_expert;
        b.shexp_gate = upload(file, pre + "ffn_gate_shexp.weight", bytes);
        b.t_shexp = (int) file.find(pre + "ffn_gate_shexp.weight")->type;
        b.shexp_up = upload(file, pre + "ffn_up_shexp.weight", bytes);
        b.shexp_down = upload(file, pre + "ffn_down_shexp.weight", bytes);
        b.loaded = true;
        return true;
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
}

void release_moe_block(MoeBlock& b) {
    release_block(b.attn);
    for (void** p : {&b.router, &b.exps_gate, &b.exps_up, &b.exps_down, &b.shexp_gate, &b.shexp_up, &b.shexp_down,
                     (void**) &b.select_bias}) {
        if (*p) cudaFree(*p);
        *p = nullptr;
    }
    b.loaded = false;
}

bool run_moe_block(const MoeBlock& b, float* h, int pos, void* q8, int n_scratch, std::string& err) {
    (void) n_scratch;
    const DenseBlock& a = b.attn;
    const int e = a.n_embd;
    if (!run_attention(a, h, pos, q8, err)) return false;

    float *x = nullptr, *logits = nullptr, *g = nullptr, *u = nullptr, *act = nullptr, *d = nullptr, *acc = nullptr;
    int32_t* ids = nullptr;
    float* w = nullptr;
    auto alloc = [](void** p, size_t bytes) { return cudaMalloc(p, bytes) == cudaSuccess; };
    if (!alloc((void**) &x, (size_t) e * sizeof(float)) || !alloc((void**) &logits, (size_t) b.n_expert * sizeof(float)) ||
        !alloc((void**) &g, (size_t) b.n_ff_exp * sizeof(float)) || !alloc((void**) &u, (size_t) b.n_ff_exp * sizeof(float)) ||
        !alloc((void**) &act, (size_t) b.n_ff_shexp * sizeof(float)) || !alloc((void**) &d, (size_t) e * sizeof(float)) ||
        !alloc((void**) &acc, (size_t) e * sizeof(float)) ||
        !alloc((void**) &ids, (size_t) b.top_k * sizeof(int32_t)) || !alloc((void**) &w, (size_t) b.top_k * sizeof(float))) {
        err = "out of device memory";
        return false;
    }
    bool ok = cudaMemcpy(x, h, (size_t) e * sizeof(float), cudaMemcpyDeviceToDevice) == cudaSuccess;
    if (ok) per_head_rms_norm(x, 1, e, a.ffn_norm, a.eps, nullptr);
    if (ok) gemv(GGML_TYPE_F32, b.router, x, logits, e, b.n_expert, q8, nullptr);
    if (ok) moe_route(logits, b.select_bias, 1, b.n_expert, b.top_k, b.scale, b.weights_norm, ids, w, nullptr);
    // the routing decision comes back to the host: eight expert matvecs are issued one by one.  A production
    // path groups them in one kernel; this is the block that has to be shown right first.
    std::vector<int32_t> h_ids((size_t) b.top_k);
    std::vector<float> h_w((size_t) b.top_k);
    if (ok) ok = cudaMemcpy(h_ids.data(), ids, h_ids.size() * sizeof(int32_t), cudaMemcpyDeviceToHost) == cudaSuccess;
    if (ok) ok = cudaMemcpy(h_w.data(), w, h_w.size() * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess;
    if (ok) ok = cudaMemset(acc, 0, (size_t) e * sizeof(float)) == cudaSuccess;
    for (int k = 0; ok && k < b.top_k; ++k) {
        const size_t ex = (size_t) h_ids[(size_t) k];
        const uint8_t* wg = (const uint8_t*) b.exps_gate + ex * b.expert_gate_bytes;
        const uint8_t* wu = (const uint8_t*) b.exps_up + ex * b.expert_gate_bytes;
        const uint8_t* wd = (const uint8_t*) b.exps_down + ex * b.expert_down_bytes;
        ok = gemv((ggml_type) b.t_exps, wg, x, g, e, b.n_ff_exp, q8, nullptr);
        if (ok) ok = gemv((ggml_type) b.t_exps, wu, x, u, e, b.n_ff_exp, q8, nullptr);
        if (ok) ok = cudaMemcpy(act, g, (size_t) b.n_ff_exp * sizeof(float), cudaMemcpyDeviceToDevice) == cudaSuccess;
        if (ok) k_silu_mul<<<(b.n_ff_exp + 255) / 256, 256>>>(act, u, b.n_ff_exp, b.clamp_exp);
        if (ok) ok = gemv((ggml_type) b.t_exps, wd, act, d, b.n_ff_exp, e, q8, nullptr);
        if (ok) k_acc_scaled<<<(e + 255) / 256, 256>>>(acc, d, h_w[(size_t) k], e);
    }
    // the shared expert, added unweighted
    if (ok) ok = gemv((ggml_type) b.t_shexp, b.shexp_gate, x, g, e, b.n_ff_shexp, q8, nullptr);
    if (ok) ok = gemv((ggml_type) b.t_shexp, b.shexp_up, x, u, e, b.n_ff_shexp, q8, nullptr);
    if (ok) ok = cudaMemcpy(act, g, (size_t) b.n_ff_shexp * sizeof(float), cudaMemcpyDeviceToDevice) == cudaSuccess;
    if (ok) k_silu_mul<<<(b.n_ff_shexp + 255) / 256, 256>>>(act, u, b.n_ff_shexp, b.clamp_shexp);
    if (ok) ok = gemv((ggml_type) b.t_shexp, b.shexp_down, act, d, b.n_ff_shexp, e, q8, nullptr);
    if (ok) k_acc_scaled<<<(e + 255) / 256, 256>>>(acc, d, 1.0f, e);
    if (ok) k_add<<<(e + 255) / 256, 256>>>(h, acc, e);
    if (ok && cudaDeviceSynchronize() != cudaSuccess) ok = false;
    if (!ok) err = "a kernel failed in the expert half";
    for (void* p : {(void*) x, (void*) logits, (void*) g, (void*) u, (void*) act, (void*) d, (void*) acc,
                    (void*) ids, (void*) w})
        if (p) cudaFree(p);
    return ok;
}

}  // namespace strata::step35::kernels
