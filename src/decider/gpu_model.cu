// src/decider/gpu_model.cu - the whole decider on the GPU: embeddings, the 32 blocks, the final norm and the
// readout at the markers.  Also the `--serve` loop, which keeps the model loaded between requests.
//
// This is M3-M4 of the integration study: the kernels already exist and are each held to the CPU reference
// (tools/decider_block_parity.py); what is new here is the order they run in, the residual stream between them,
// the readout, and the fact that the result is compared to the checkpoint's own logits
// (tools/decider_parity.py --gpu).
//
// THE READOUT.  255 rows of `output.weight`, dequantized once and kept on the device; one block per code, each
// dotted with the hidden state at a marker.  It is 0.05% of the arithmetic, but it is also the one place where
// the head's rows are indexed by VOCABULARY id, not by rank - the rows come from `pplx_decider.token_ids`.
#include "gpu_common.cuh"
#include "gpu_kernels.cuh"

#include "strata/decider/gpu.hpp"

#include <cmath>
#include <iostream>
#include <memory>
#include <sstream>

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

/// One row of a quantized matrix as floats: the readout's rows, dequantized once on the host at load time.
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
    Dev q, k, v, o, q_norm, k_norm;                                // full attention
    Dev qkv, z, alpha, beta, ssm_a, dt, conv, ssm_norm, ssm_out;   // gated delta net
};

/// The model as the GPU holds it: every weight uploaded once, so a request is one forward pass.
struct Model {
    std::unique_ptr<GgufFile> file;
    int64_t n_embd = 0, n_ff = 0, n_blocks = 0, n_head = 0, n_head_kv = 0, head_dim = 0, n_rot = 0;
    int64_t gdn_kh = 0, gdn_vh = 0, gdn_kd = 0, conv_k = 0, key_dim = 0, value_dim = 0, conv_dim = 0;
    double theta = 0, eps = 0;
    std::vector<int64_t> codes;
    std::vector<uint8_t> recurrent;
    Dev embd, final_norm, code_rows;   ///< code_rows: the 255 readout rows, dequantized
    std::vector<BlockWeights> weights;
};

bool open(const std::string& path, Model& m, std::string& err) {
    try {
        m.file = std::make_unique<GgufFile>(strata::gguf_split_paths(path).front());
        const GgufFile& g = *m.file;
        const std::string why = strata::check_architecture_qwen35(g);
        if (!why.empty()) fail(why);
        m.n_embd = get_int(g, "qwen35.embedding_length");
        m.n_ff = get_int(g, "qwen35.feed_forward_length");
        m.n_blocks = get_int(g, "qwen35.block_count");
        m.n_head = get_int(g, "qwen35.attention.head_count");
        m.n_head_kv = get_int(g, "qwen35.attention.head_count_kv");
        m.head_dim = get_int(g, "qwen35.attention.key_length");
        m.n_rot = get_int(g, "qwen35.rope.dimension_count");
        m.theta = get_num(g, "qwen35.rope.freq_base");
        m.gdn_kh = get_int(g, "qwen35.ssm.group_count");
        m.gdn_vh = get_int(g, "qwen35.ssm.time_step_rank");
        m.gdn_kd = get_int(g, "qwen35.ssm.state_size");
        m.conv_k = get_int(g, "qwen35.ssm.conv_kernel");
        m.eps = get_num(g, "qwen35.attention.layer_norm_rms_epsilon");
        m.key_dim = m.gdn_kh * m.gdn_kd;
        m.value_dim = m.gdn_vh * m.gdn_kd;
        m.conv_dim = 2 * m.key_dim + m.value_dim;
        if (m.head_dim > 256 || m.gdn_kd > 128)
            fail("the kernels are written for head widths of 256 (attention) and 128 (GDN)");
        const MetaValue* recurrent = g.get("qwen35.attention.recurrent_layers");
        for (const MetaValue& v : recurrent->items) m.recurrent.push_back(v.u ? 1 : 0);

        const MetaValue* codes = g.get("pplx_decider.token_ids");
        if (codes->items.size() != (size_t) codes->count)
            fail("the reader kept only " + std::to_string(codes->items.size()) + " of the " +
                 std::to_string(codes->count) + " readout codes");
        for (const MetaValue& id : codes->items) m.codes.push_back((int64_t) id.u);

        const TensorInfo& t_embd = *g.find("token_embd.weight");
        const TensorInfo& t_out = *g.find("output.weight");
        m.embd = upload(g, t_embd);
        m.final_norm = upload_vector(read_tensor(g, *g.find("output_norm.weight")));
        std::vector<float> rows((size_t) m.codes.size() * m.n_embd);
        for (size_t c = 0; c < m.codes.size(); ++c)
            read_row_f32(g, t_out, m.codes[c], m.n_embd, rows.data() + c * (size_t) m.n_embd);
        m.code_rows = upload_vector(rows);

        m.weights.resize((size_t) m.n_blocks);
        for (int64_t l = 0; l < m.n_blocks; ++l) {
            const std::string p = "blk." + std::to_string(l) + ".";
            BlockWeights& b = m.weights[(size_t) l];
            b.attn_norm = upload_vector(read_tensor(g, *g.find(p + "attn_norm.weight")));
            b.post_norm = upload_vector(read_tensor(g, *g.find(p + "post_attention_norm.weight")));
            b.ffn_gate = upload(g, *g.find(p + "ffn_gate.weight"));
            b.ffn_up = upload(g, *g.find(p + "ffn_up.weight"));
            b.ffn_down = upload(g, *g.find(p + "ffn_down.weight"));
            if (m.recurrent[(size_t) l]) {
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
        return true;
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
}

bool run(Model& m, const std::vector<int32_t>& tokens, const std::vector<int32_t>& markers,
         std::vector<std::vector<float>>& out, std::string& err, bool trace) {
    try {
        const int64_t E = m.n_embd, FF = m.n_ff, H = m.n_head, KV = m.n_head_kv, D = m.head_dim;
        const int64_t T = (int64_t) tokens.size();
        if (T < 1) fail("no tokens");
        const int64_t vocab = (int64_t) m.file->get("tokenizer.ggml.tokens")->count;
        for (int32_t id : tokens)
            if (id < 0 || id >= vocab) fail("token " + std::to_string(id) + " is outside the vocabulary");
        for (int32_t marker : markers)
            if (marker < 0 || marker >= T) fail("marker " + std::to_string(marker) + " is outside the sequence");
        const float scale_attn = 1.0f / sqrtf((float) D);
        const float scale_gdn = 1.0f / sqrtf((float) m.gdn_kd);

        const size_t n_e = (size_t) T * E;
        Dev d_ids((size_t) T * sizeof(int32_t));
        check(cudaMemcpy(d_ids.p, tokens.data(), tokens.size() * sizeof(int32_t), cudaMemcpyHostToDevice),
              "cudaMemcpy tokens");
        Dev d_x(n_e * sizeof(float)), d_xn(n_e * sizeof(float)), d_mix(n_e * sizeof(float)), d_tmp(n_e * sizeof(float));
        Dev d_qg((size_t) T * H * 2 * D * sizeof(float));
        Dev d_kv((size_t) T * KV * D * 2 * sizeof(float));
        Dev d_attn((size_t) T * H * D * sizeof(float));
        Dev d_mixed((size_t) T * m.conv_dim * sizeof(float));
        Dev d_z((size_t) T * m.value_dim * sizeof(float));
        Dev d_a((size_t) T * m.gdn_vh * sizeof(float)), d_b((size_t) T * m.gdn_vh * sizeof(float));
        Dev d_conv((size_t) T * m.conv_dim * sizeof(float));
        Dev d_core((size_t) T * m.value_dim * sizeof(float));
        Dev d_normed((size_t) T * m.value_dim * sizeof(float));
        Dev d_gate((size_t) T * FF * sizeof(float)), d_up((size_t) T * FF * sizeof(float));
        Dev d_h((size_t) T * FF * sizeof(float));
        Dev d_hidden((size_t) E * sizeof(float));
        constexpr int kChunk = 8;
        // the widest input any projection reads: ffn_down takes n_ff (9216), wider than the embedding or the GDN's
        // conv width.  Sizing this on E and conv_dim alone overflowed by 9 KB and faulted on the second sequence.
        const int64_t widest = std::max({E, FF, m.conv_dim, H * 2 * D, m.value_dim});
        Dev d_q8(strata::kernels::native_q8_1_bytes((int) widest, kChunk));

        strata::kernels::iq_embed_rows((int) m.file->find("token_embd.weight")->type,
                                       m.file->tensor_data(*m.file->find("token_embd.weight")),
                                       ggml_row_size((ggml_type) m.file->find("token_embd.weight")->type, E),
                                       d_ids.as<int32_t>(), T, E, d_x.as<float>(), nullptr);
        check(cudaGetLastError(), "iq_embed_rows");

        for (int64_t l = 0; l < m.n_blocks; ++l) {
            const BlockWeights& b = m.weights[(size_t) l];
            kernels::rmsnorm<<<(unsigned) T, 256>>>(d_x.as<float>(), b.attn_norm.as<float>(), d_xn.as<float>(), (int) E,
                                                    (float) m.eps);
            check(cudaGetLastError(), "rmsnorm attn");
            if (m.recurrent[(size_t) l]) {
                gemm_q8(b.qkv.p, d_xn.as<float>(), d_mixed.as<float>(), (int) E, (int) m.conv_dim, (int) T, d_q8.p,
                        nullptr);
                gemm_q8(b.z.p, d_xn.as<float>(), d_z.as<float>(), (int) E, (int) m.value_dim, (int) T, d_q8.p, nullptr);
                gemm_q8(b.alpha.p, d_xn.as<float>(), d_a.as<float>(), (int) E, (int) m.gdn_vh, (int) T, d_q8.p, nullptr);
                gemm_q8(b.beta.p, d_xn.as<float>(), d_b.as<float>(), (int) E, (int) m.gdn_vh, (int) T, d_q8.p, nullptr);
                const long long cells = (long long) T * m.conv_dim;
                kernels::conv1d_silu<<<(unsigned) ((cells + 255) / 256), 256>>>(
                    d_mixed.as<float>(), b.conv.as<float>(), d_conv.as<float>(), (int) T, (int) m.conv_dim,
                    (int) m.conv_k);
                check(cudaGetLastError(), "conv1d_silu");
                kernels::gdn_scan<<<(unsigned) m.gdn_vh, dim3(kernels::kHeads, kernels::kHeads)>>>(
                    d_conv.as<float>(), d_a.as<float>(), d_b.as<float>(), b.ssm_a.as<float>(), b.dt.as<float>(),
                    d_core.as<float>(), (int) T, (int) m.gdn_vh, (int) m.gdn_kh, (int) m.gdn_kd, (int) m.gdn_kd,
                    (int) m.key_dim, (int) m.conv_dim, scale_gdn, (float) m.eps);
                check(cudaGetLastError(), "gdn_scan");
                kernels::gated_norm<<<(unsigned) (T * m.gdn_vh), 128>>>(d_core.as<float>(), d_z.as<float>(),
                                                                       b.ssm_norm.as<float>(), d_normed.as<float>(),
                                                                       (int) m.gdn_kd, (float) m.eps);
                check(cudaGetLastError(), "gated_norm");
                gemm_q8(b.ssm_out.p, d_normed.as<float>(), d_mix.as<float>(), (int) m.value_dim, (int) E, (int) T,
                        d_q8.p, nullptr);
            } else {
                gemm_q8(b.q.p, d_xn.as<float>(), d_qg.as<float>(), (int) E, (int) (H * 2 * D), (int) T, d_q8.p,
                        nullptr);
                gemm_q8(b.k.p, d_xn.as<float>(), d_kv.as<float>(), (int) E, (int) (KV * D), (int) T, d_q8.p, nullptr);
                gemm_q8(b.v.p, d_xn.as<float>(), d_kv.as<float>() + (size_t) T * KV * D, (int) E, (int) (KV * D),
                        (int) T, d_q8.p, nullptr);
                kernels::norm_rope<<<(unsigned) (T * H), (unsigned) D>>>(d_qg.as<float>(), b.q_norm.as<float>(),
                                                                        (int) H, (int) (2 * D), (int) D, (int) m.n_rot,
                                                                        (float) m.theta, (float) m.eps);
                check(cudaGetLastError(), "norm_rope q");
                kernels::norm_rope<<<(unsigned) (T * KV), (unsigned) D>>>(d_kv.as<float>(), b.k_norm.as<float>(),
                                                                         (int) KV, (int) D, (int) D, (int) m.n_rot,
                                                                         (float) m.theta, (float) m.eps);
                check(cudaGetLastError(), "norm_rope k");
                const size_t shared_bytes = (size_t) (2 * D + T) * sizeof(float);
                kernels::attention<<<(unsigned) H, 256, shared_bytes>>>(
                    d_qg.as<float>(), d_kv.as<float>(), d_kv.as<float>() + (size_t) T * KV * D, d_attn.as<float>(),
                    (int) T, (int) H, (int) KV, (int) D, scale_attn);
                check(cudaGetLastError(), "attention");
                gemm_q8(b.o.p, d_attn.as<float>(), d_mix.as<float>(), (int) (H * D), (int) E, (int) T, d_q8.p,
                        nullptr);
            }
            kernels::add_in_place<<<(unsigned) ((n_e + 255) / 256), 256>>>(d_x.as<float>(), d_mix.as<float>(), n_e);
            check(cudaGetLastError(), "residual mixer");
            kernels::rmsnorm<<<(unsigned) T, 256>>>(d_x.as<float>(), b.post_norm.as<float>(), d_xn.as<float>(), (int) E,
                                                    (float) m.eps);
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

        const size_t n_codes = m.codes.size();
        Dev d_logits(n_codes * sizeof(float));
        out.clear();
        for (int32_t marker : markers) {
            kernels::rmsnorm<<<1, 256>>>(d_x.as<float>() + (size_t) marker * E, m.final_norm.as<float>(),
                                         d_hidden.as<float>(), (int) E, (float) m.eps);
            check(cudaGetLastError(), "rmsnorm final");
            kernels::readout<<<(unsigned) n_codes, 256>>>(d_hidden.as<float>(), m.code_rows.as<float>(),
                                                          d_logits.as<float>(), (int) n_codes, (int) E);
            check(cudaGetLastError(), "readout");
            check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
            std::vector<float> values(n_codes);
            check(cudaMemcpy(values.data(), d_logits.p, n_codes * sizeof(float), cudaMemcpyDeviceToHost),
                  "cudaMemcpy logits");
            out.push_back(std::move(values));
        }
        return true;
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
}

}  // namespace

bool logits(const std::string& gguf, const std::vector<int32_t>& tokens, const std::vector<int32_t>& markers,
            std::vector<std::vector<float>>& out, std::string& err, bool trace) {
    Model model;
    if (!open(gguf, model, err)) return false;
    return run(model, tokens, markers, out, err, trace);
}

int serve(const std::string& gguf, bool trace) {
    Model model;
    std::string err;
    if (!open(gguf, model, err)) {
        std::fprintf(stderr, "strata-decider-gpu: %s\n", err.c_str());
        return 1;
    }
    std::fprintf(stderr, "strata-decider-gpu: serving %lld blocks, %zu readout codes\n", (long long) model.n_blocks,
                 model.codes.size());
    std::fflush(stderr);
    // the handshake serve/server.py waits for: the context it may plan a request with, and that STOP is honoured
    // (a decision cannot be interrupted, so the line is a promise about the protocol, not about the work)
    std::printf("READY %d stop\n", 32768);
    std::fflush(stdout);
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        if (line == "QUIT" || line == "EXIT") break;
        std::istringstream in(line);
        std::string verb;
        in >> verb;
        if (verb != "DEC") {
            std::printf("ERR unknown operation %s\n", verb.c_str());
            std::fflush(stdout);
            continue;
        }
        long long n_tokens = 0;
        in >> n_tokens;
        std::vector<int32_t> tokens((size_t) std::max(0LL, n_tokens));
        for (auto& token : tokens) in >> token;
        long long n_markers = 0;
        in >> n_markers;
        std::vector<int32_t> markers((size_t) std::max(0LL, n_markers));
        for (auto& marker : markers) in >> marker;
        if (!in || tokens.empty() || markers.empty()) {
            std::printf("ERR malformed DEC line\n");
            std::fflush(stdout);
            continue;
        }
        std::vector<std::vector<float>> out;
        if (!run(model, tokens, markers, out, err, trace)) {
            std::printf("ERR %s\n", err.c_str());
            std::fflush(stdout);
            continue;
        }
        std::printf("DEC %zu", out.size());
        for (const auto& row : out) {
            std::printf(" %zu", row.size());
            for (float value : row) std::printf(" %.9g", (double) value);
        }
        std::printf("\nDONE\n");
        std::fflush(stdout);
    }
    return 0;
}

}  // namespace strata::decider::gpu
