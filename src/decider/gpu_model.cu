// src/decider/gpu_model.cu - the whole decider on the GPU: embeddings, the 32 blocks, the final norm and the
// readout at the markers.
//
// This is M3 of the integration study: the kernels already exist and are each held to the CPU reference
// (tools/decider_block_parity.py); what is new here is the order they run in, the residual stream between them,
// and the fact that the result can be compared to the checkpoint's own logits (tools/decider_parity.py --gpu).
//
// WHAT STAYS ON THE HOST.  The readout: 255 rows of `output.weight` dotted with the hidden state at each marker.
// That is 0.05% of the arithmetic, and the dequantization of those rows is the code the reference already uses;
// the GPU path will take it when the readout is timed, not before.
#include "gpu_common.cuh"
#include "gpu_kernels.cuh"

#include "strata/decider/gpu.hpp"

#include <cmath>

namespace strata::decider::gpu {
namespace {

using namespace detail;

/// A device allocation that can be moved into a vector (one per tensor, several hundred for the model).
struct Dev {
    void* p = nullptr;
    Dev() = default;
    explicit Dev(size_t bytes) { check(cudaMalloc(&p, bytes), "cudaMalloc"); }
    Dev(Dev&& other) noexcept : p(other.p) { other.p = nullptr; }
    Dev& operator=(Dev&& other) noexcept {
        if (this != &other) {
            if (p) (void) cudaFree(p);
            p = other.p;
            other.p = nullptr;
        }
        return *this;
    }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
    ~Dev() { if (p) (void) cudaFree(p); }
    template<typename T>
    T* as() const { return (T*) p; }
};

Dev upload(const GgufFile& g, const TensorInfo& t) {
    const uint64_t bytes = ggml_row_size((ggml_type) t.type, (int64_t) t.shape[0]) * t.shape[1];
    Dev buffer(bytes);
    check(cudaMemcpy(buffer.p, g.tensor_data(t), bytes, cudaMemcpyHostToDevice), "cudaMemcpy weights");
    return buffer;
}

Dev upload_vector(const std::vector<float>& values) {
    Dev buffer(values.size() * sizeof(float));
    check(cudaMemcpy(buffer.p, values.data(), values.size() * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy");
    return buffer;
}

/// One row of a quantized matrix as floats: the readout's rows, dequantized on the host (the reference does the
/// same, and 255 rows of 2560 is not worth a kernel).
void read_row_f32(const GgufFile& g, const TensorInfo& t, int64_t row, int64_t n_in, float* out) {
    const uint8_t* base = g.tensor_data(t);
    if (t.type == GGML_TYPE_F32) {
        std::memcpy(out, base + (size_t) row * (size_t) n_in * sizeof(float), (size_t) n_in * sizeof(float));
        return;
    }
    const auto* traits = ggml_get_type_traits((ggml_type) t.type);
    if (!traits || !traits->to_float) fail("no dequantizer for " + t.name);
    const size_t row_bytes = ggml_row_size((ggml_type) t.type, n_in);
    traits->to_float(base + (size_t) row * row_bytes, out, n_in);
}

/// Every weight one block reads, whatever its kind (the unused members stay null).
struct BlockWeights {
    Dev attn_norm, post_norm, ffn_gate, ffn_up, ffn_down;
    Dev q, k, v, o, q_norm, k_norm;                       // full attention
    Dev qkv, z, alpha, beta, ssm_a, dt, conv, ssm_norm, ssm_out;   // gated delta net
};

}  // namespace

bool logits(const std::string& gguf, const std::vector<int32_t>& tokens, const std::vector<int32_t>& markers,
            std::vector<std::vector<float>>& out, std::string& err, bool trace) {
    try {
        const GgufFile g(strata::gguf_split_paths(gguf).front());
        const std::string why = strata::check_architecture_qwen35(g);
        if (!why.empty()) fail(why);
        const int64_t E = get_int(g, "qwen35.embedding_length");
        const int64_t FF = get_int(g, "qwen35.feed_forward_length");
        const int64_t blocks = get_int(g, "qwen35.block_count");
        const int64_t H = get_int(g, "qwen35.attention.head_count");
        const int64_t KV = get_int(g, "qwen35.attention.head_count_kv");
        const int64_t D = get_int(g, "qwen35.attention.key_length");
        const int64_t n_rot = get_int(g, "qwen35.rope.dimension_count");
        const double theta = get_num(g, "qwen35.rope.freq_base");
        const int64_t KH = get_int(g, "qwen35.ssm.group_count");
        const int64_t VH = get_int(g, "qwen35.ssm.time_step_rank");
        const int64_t KD = get_int(g, "qwen35.ssm.state_size");
        const int64_t K = get_int(g, "qwen35.ssm.conv_kernel");
        const double eps = get_num(g, "qwen35.attention.layer_norm_rms_epsilon");
        const int64_t key_dim = KH * KD, value_dim = VH * KD, conv_dim = 2 * key_dim + value_dim;
        if (D > 256 || KD > 128) fail("the kernels are written for head widths of 256 (attention) and 128 (GDN)");
        const float scale_attn = 1.0f / sqrtf((float) D);
        const float scale_gdn = 1.0f / sqrtf((float) KD);

        const int64_t T = (int64_t) tokens.size();
        if (T < 1) fail("no tokens");
        const int64_t vocab = (int64_t) g.get("tokenizer.ggml.tokens")->count;
        for (int32_t id : tokens)
            if (id < 0 || id >= vocab) fail("token " + std::to_string(id) + " is outside the vocabulary");
        for (int32_t m : markers)
            if (m < 0 || m >= T) fail("marker " + std::to_string(m) + " is outside the sequence");
        std::fprintf(stderr, "strata-decider-gpu: full model, %lld tokens, %lld blocks\n", (long long) T,
                     (long long) blocks);

        // ---- the weights, once
        const TensorInfo& t_embd = *g.find("token_embd.weight");
        const TensorInfo& t_out = *g.find("output.weight");
        Dev d_embd = upload(g, t_embd);
        Dev d_outw = upload(g, t_out);
        Dev d_final_norm = upload_vector(read_tensor(g, *g.find("output_norm.weight")));
        std::vector<BlockWeights> w((size_t) blocks);
        const MetaValue* recurrent = g.get("qwen35.attention.recurrent_layers");
        for (int64_t l = 0; l < blocks; ++l) {
            const std::string p = "blk." + std::to_string(l) + ".";
            BlockWeights& b = w[(size_t) l];
            b.attn_norm = upload_vector(read_tensor(g, *g.find(p + "attn_norm.weight")));
            b.post_norm = upload_vector(read_tensor(g, *g.find(p + "post_attention_norm.weight")));
            b.ffn_gate = upload(g, *g.find(p + "ffn_gate.weight"));
            b.ffn_up = upload(g, *g.find(p + "ffn_up.weight"));
            b.ffn_down = upload(g, *g.find(p + "ffn_down.weight"));
            const bool gdn = recurrent && (size_t) l < recurrent->items.size() && recurrent->items[(size_t) l].u;
            if (gdn) {
                b.qkv = upload(g, *g.find(p + "attn_qkv.weight"));
                b.z = upload(g, *g.find(p + "attn_gate.weight"));
                b.alpha = upload(g, *g.find(p + "ssm_alpha.weight"));
                b.beta = upload(g, *g.find(p + "ssm_beta.weight"));
                b.ssm_a = upload_vector(read_tensor(g, *g.find(p + "ssm_a")));
                b.dt = upload_vector(read_tensor(g, *g.find(p + "ssm_dt.bias")));
                b.conv = upload_vector(read_tensor(g, *g.find(p + "ssm_conv1d.weight")));
                b.ssm_norm = upload_vector(read_tensor(g, *g.find(p + "ssm_norm.weight")));
                b.ssm_out = upload(g, *g.find(p + "ssm_out.weight"));
            } else {
                b.q = upload(g, *g.find(p + "attn_q.weight"));
                b.k = upload(g, *g.find(p + "attn_k.weight"));
                b.v = upload(g, *g.find(p + "attn_v.weight"));
                b.o = upload(g, *g.find(p + "attn_output.weight"));
                b.q_norm = upload_vector(read_tensor(g, *g.find(p + "attn_q_norm.weight")));
                b.k_norm = upload_vector(read_tensor(g, *g.find(p + "attn_k_norm.weight")));
            }
        }
        std::vector<int32_t> d_tokens(tokens.begin(), tokens.end());
        Dev d_ids((size_t) T * sizeof(int32_t));
        check(cudaMemcpy(d_ids.p, d_tokens.data(), d_tokens.size() * sizeof(int32_t), cudaMemcpyHostToDevice),
              "cudaMemcpy tokens");

        // ---- the buffers
        const size_t n_e = (size_t) T * E;
        Dev d_x(n_e * sizeof(float)), d_xn(n_e * sizeof(float)), d_mix(n_e * sizeof(float)), d_tmp(n_e * sizeof(float));
        Dev d_qg((size_t) T * H * 2 * D * sizeof(float));
        Dev d_kv((size_t) T * KV * D * 2 * sizeof(float));
        Dev d_attn((size_t) T * H * D * sizeof(float));
        Dev d_mixed((size_t) T * conv_dim * sizeof(float));
        Dev d_z((size_t) T * value_dim * sizeof(float));
        Dev d_a((size_t) T * VH * sizeof(float)), d_b((size_t) T * VH * sizeof(float));
        Dev d_conv((size_t) T * conv_dim * sizeof(float));
        Dev d_core((size_t) T * value_dim * sizeof(float));
        Dev d_normed((size_t) T * value_dim * sizeof(float));
        Dev d_gate((size_t) T * FF * sizeof(float)), d_up((size_t) T * FF * sizeof(float));
        Dev d_h((size_t) T * FF * sizeof(float));
        Dev d_hidden((size_t) E * sizeof(float));
        constexpr int kChunk = 8;
        // the widest input any projection reads: the FFN's down projection takes n_ff (9216), wider than the
        // embedding or the GDN's conv width.  Sizing this on E and conv_dim alone overflowed by 9 KB and faulted
        // on the second sequence.
        const int64_t widest = std::max({E, FF, conv_dim, H * 2 * D, value_dim});
        const size_t scratch = strata::kernels::native_q8_1_bytes((int) widest, kChunk);
        Dev d_q8(scratch);

        // ---- embeddings
        strata::kernels::iq_embed_rows((int) t_embd.type, g.tensor_data(t_embd),
                                       ggml_row_size((ggml_type) t_embd.type, E), d_ids.as<int32_t>(), T, E,
                                       d_x.as<float>(), nullptr);
        check(cudaGetLastError(), "iq_embed_rows");

        for (int64_t l = 0; l < blocks; ++l) {
            const BlockWeights& b = w[(size_t) l];
            kernels::rmsnorm<<<(unsigned) T, 256>>>(d_x.as<float>(), b.attn_norm.as<float>(), d_xn.as<float>(), (int) E,
                                                    (float) eps);
            check(cudaGetLastError(), "rmsnorm attn");
            const bool gdn = recurrent && (size_t) l < recurrent->items.size() && recurrent->items[(size_t) l].u;
            if (gdn) {
                gemm_q8(b.qkv.p, d_xn.as<float>(), d_mixed.as<float>(), (int) E, (int) conv_dim, (int) T, d_q8.p,
                        nullptr);
                gemm_q8(b.z.p, d_xn.as<float>(), d_z.as<float>(), (int) E, (int) value_dim, (int) T, d_q8.p, nullptr);
                gemm_q8(b.alpha.p, d_xn.as<float>(), d_a.as<float>(), (int) E, (int) VH, (int) T, d_q8.p, nullptr);
                gemm_q8(b.beta.p, d_xn.as<float>(), d_b.as<float>(), (int) E, (int) VH, (int) T, d_q8.p, nullptr);
                const long long cells = (long long) T * conv_dim;
                kernels::conv1d_silu<<<(unsigned) ((cells + 255) / 256), 256>>>(
                    d_mixed.as<float>(), b.conv.as<float>(), d_conv.as<float>(), (int) T, (int) conv_dim, (int) K);
                check(cudaGetLastError(), "conv1d_silu");
                kernels::gdn_scan<<<(unsigned) VH, dim3(kernels::kHeads, kernels::kHeads)>>>(
                    d_conv.as<float>(), d_a.as<float>(), d_b.as<float>(), b.ssm_a.as<float>(), b.dt.as<float>(),
                    d_core.as<float>(), (int) T, (int) VH, (int) KH, (int) KD, (int) KD, (int) key_dim, (int) conv_dim,
                    scale_gdn, (float) eps);
                check(cudaGetLastError(), "gdn_scan");
                kernels::gated_norm<<<(unsigned) (T * VH), 128>>>(d_core.as<float>(), d_z.as<float>(),
                                                                 b.ssm_norm.as<float>(), d_normed.as<float>(),
                                                                 (int) KD, (float) eps);
                check(cudaGetLastError(), "gated_norm");
                gemm_q8(b.ssm_out.p, d_normed.as<float>(), d_mix.as<float>(), (int) value_dim, (int) E, (int) T,
                        d_q8.p, nullptr);
            } else {
                gemm_q8(b.q.p, d_xn.as<float>(), d_qg.as<float>(), (int) E, (int) (H * 2 * D), (int) T, d_q8.p,
                        nullptr);
                gemm_q8(b.k.p, d_xn.as<float>(), d_kv.as<float>(), (int) E, (int) (KV * D), (int) T, d_q8.p, nullptr);
                gemm_q8(b.v.p, d_xn.as<float>(), d_kv.as<float>() + (size_t) T * KV * D, (int) E, (int) (KV * D),
                        (int) T, d_q8.p, nullptr);
                kernels::norm_rope<<<(unsigned) (T * H), (unsigned) D>>>(d_qg.as<float>(), b.q_norm.as<float>(),
                                                                        (int) H, (int) (2 * D), (int) D, (int) n_rot,
                                                                        (float) theta, (float) eps);
                check(cudaGetLastError(), "norm_rope q");
                kernels::norm_rope<<<(unsigned) (T * KV), (unsigned) D>>>(d_kv.as<float>(), b.k_norm.as<float>(),
                                                                         (int) KV, (int) D, (int) D, (int) n_rot,
                                                                         (float) theta, (float) eps);
                check(cudaGetLastError(), "norm_rope k");
                const size_t shared_bytes = (size_t) (2 * D + T) * sizeof(float);
                kernels::attention<<<(unsigned) H, 256, shared_bytes>>>(
                    d_qg.as<float>(), d_kv.as<float>(), d_kv.as<float>() + (size_t) T * KV * D, d_attn.as<float>(),
                    (int) T, (int) H, (int) KV, (int) D, scale_attn);
                check(cudaGetLastError(), "attention");
                gemm_q8(b.o.p, d_attn.as<float>(), d_mix.as<float>(), (int) (H * D), (int) E, (int) T, d_q8.p,
                        nullptr);
            }
            // the residual, then the dense SwiGLU on the post-attention norm
            kernels::add_in_place<<<(unsigned) ((n_e + 255) / 256), 256>>>(d_x.as<float>(), d_mix.as<float>(), n_e);
            check(cudaGetLastError(), "residual mixer");
            kernels::rmsnorm<<<(unsigned) T, 256>>>(d_x.as<float>(), b.post_norm.as<float>(), d_xn.as<float>(), (int) E,
                                                    (float) eps);
            check(cudaGetLastError(), "rmsnorm post");
            gemm_q8(b.ffn_gate.p, d_xn.as<float>(), d_gate.as<float>(), (int) E, (int) FF, (int) T, d_q8.p, nullptr);
            gemm_q8(b.ffn_up.p, d_xn.as<float>(), d_up.as<float>(), (int) E, (int) FF, (int) T, d_q8.p, nullptr);
            const size_t cells_ff = (size_t) T * FF;
            kernels::swiglu<<<(unsigned) ((cells_ff + 255) / 256), 256>>>(d_gate.as<float>(), d_up.as<float>(),
                                                                         d_h.as<float>(), cells_ff);
            check(cudaGetLastError(), "swiglu");
            gemm_q8(b.ffn_down.p, d_h.as<float>(), d_tmp.as<float>(), (int) FF, (int) E, (int) T, d_q8.p, nullptr);
            kernels::add_in_place<<<(unsigned) ((n_e + 255) / 256), 256>>>(d_x.as<float>(), d_tmp.as<float>(), n_e);
            check(cudaGetLastError(), "residual mlp");
            if (trace) {
                check(cudaDeviceSynchronize(), "cudaDeviceSynchronize trace");
                std::vector<float> snapshot(n_e);
                check(cudaMemcpy(snapshot.data(), d_x.p, n_e * sizeof(float), cudaMemcpyDeviceToHost),
                      "cudaMemcpy trace");
                double sum = 0.0, ss = 0.0;
                for (float value : snapshot) {
                    sum += value;
                    ss += (double) value * value;
                }
                std::fprintf(stderr, "trace block %-3lld sum %12.4f norm %12.4f\n", (long long) l, sum,
                             std::sqrt(ss));
            }
        }

        // ---- the final norm at the markers, then the readout (255 rows, on the host: see the file's header)
        const MetaValue* codes = g.get("pplx_decider.token_ids");
        if (codes->items.size() != (size_t) codes->count)
            fail("the reader kept only " + std::to_string(codes->items.size()) + " of the " +
                 std::to_string(codes->count) + " readout codes");
        out.clear();
        for (int32_t marker : markers) {
            kernels::rmsnorm<<<1, 256>>>(d_x.as<float>() + (size_t) marker * E, d_final_norm.as<float>(),
                                         d_hidden.as<float>(), (int) E, (float) eps);
            check(cudaGetLastError(), "rmsnorm final");
            check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
            std::vector<float> hidden((size_t) E);
            check(cudaMemcpy(hidden.data(), d_hidden.p, hidden.size() * sizeof(float), cudaMemcpyDeviceToHost),
                  "cudaMemcpy hidden");
            std::vector<float> values;
            values.reserve(codes->count);
            std::vector<float> row((size_t) E);
            for (const MetaValue& code : codes->items) {
                read_row_f32(g, t_out, (int64_t) code.u, E, row.data());
                float acc = 0.0f;
                for (int64_t i = 0; i < E; ++i) acc += row[(size_t) i] * hidden[(size_t) i];
                values.push_back(acc);
            }
            out.push_back(std::move(values));
        }
        return true;
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
}

}  // namespace strata::decider::gpu
