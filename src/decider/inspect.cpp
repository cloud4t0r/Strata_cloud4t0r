// src/decider/inspect.cpp - the decider's tensor audit (see include/strata/decider/inspect.hpp).
//
// The expected shapes are derived from the file's own metadata, so the same audit holds for the 32-block release
// and for a small synthetic file the tests write: nothing here is a constant of the 4.8 B model except the schema
// (which tensor exists in a GDN block, which in a full-attention block, and their widths in terms of the geometry).
#include "strata/decider/inspect.hpp"

#include <set>
#include <string>
#include <vector>

namespace strata::decider {
namespace {

constexpr uint32_t kF32 = 0;    ///< GGML_TYPE_F32
constexpr uint32_t kQ8_0 = 8;   ///< GGML_TYPE_Q8_0

struct Expect {
    std::string name;
    std::vector<uint64_t> shape;
    uint32_t type = kQ8_0;
};

std::string shape_text(const std::vector<uint64_t>& shape) {
    std::string out = "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i) out += ", ";
        out += std::to_string(shape[i]);
    }
    return out + "]";
}

/// One expected tensor: the file must hold it, with this shape and this type.  `why` names what the width means
/// when it is wrong, because "expected [2560, 8192], got [2560, 4096]" is only useful with the reason.
std::string want(const GgufFile& g, const std::string& name, const std::vector<uint64_t>& shape, uint32_t type,
                 std::string* problem) {
    if (!problem->empty()) return name;
    const TensorInfo* t = g.find(name);
    if (!t) {
        *problem = "missing tensor " + name + " (expected " + shape_text(shape) + ")";
        return name;
    }
    if (t->shape != shape) {
        *problem = name + " is " + shape_text(t->shape) + ", the geometry implies " + shape_text(shape);
        return name;
    }
    if (t->type != type) {
        *problem = name + " is " + t->type_name() + ", the checkpoint is " + (type == kF32 ? "F32" : "Q8_0");
        return name;
    }
    return name;
}

}  // namespace

std::string inspect(const GgufFile& g, std::string* report) {
    std::string err = check_architecture_qwen35(g);
    if (!err.empty()) return err;

    const uint64_t blocks = g.get("qwen35.block_count")->u;
    const uint64_t interval = g.get("qwen35.full_attention_interval")->u;
    const uint64_t E = g.get("qwen35.embedding_length")->u;
    const uint64_t F = g.get("qwen35.feed_forward_length")->u;
    const uint64_t H = g.get("qwen35.attention.head_count")->u;
    const uint64_t KV = g.get("qwen35.attention.head_count_kv")->u;
    const uint64_t K = g.get("qwen35.attention.key_length")->u;
    const uint64_t V = g.get("qwen35.attention.value_length")->u;
    const uint64_t G = g.get("qwen35.ssm.group_count")->u;
    const uint64_t T = g.get("qwen35.ssm.time_step_rank")->u;
    const uint64_t S = g.get("qwen35.ssm.state_size")->u;
    const uint64_t I = g.get("qwen35.ssm.inner_size")->u;
    const uint64_t C = g.get("qwen35.ssm.conv_kernel")->u;

    // The vocabulary is the tokenizer's, and the readout's codes must be inside it.
    const MetaValue* tokens = g.get("tokenizer.ggml.tokens");
    const uint64_t vocab = tokens && tokens->type == MetaType::ARRAY ? tokens->count : 0;
    if (vocab == 0) return "missing tokenizer.ggml.tokens: the readout's ids cannot be checked against a vocabulary";
    for (const MetaValue& id : g.get("pplx_decider.token_ids")->items)
        if (id.u >= vocab)
            return "pplx_decider.token_ids holds " + std::to_string(id.u) + ", past the " + std::to_string(vocab) +
                   "-token vocabulary";

    std::string problem;
    std::set<std::string> expected;
    auto add = [&](const std::string& name, const std::vector<uint64_t>& shape, uint32_t type = kQ8_0) {
        expected.insert(name);
        want(g, name, shape, type, &problem);
    };
    add("token_embd.weight", {E, vocab});
    add("output.weight", {E, vocab});
    add("output_norm.weight", {E}, kF32);

    for (uint64_t b = 0; b < blocks; ++b) {
        const std::string p = "blk." + std::to_string(b) + ".";
        const bool full = (b + 1) % interval == 0;
        // the residual stream and the dense SwiGLU are the same in both layer kinds
        add(p + "attn_norm.weight", {E}, kF32);
        add(p + "post_attention_norm.weight", {E}, kF32);
        add(p + "ffn_gate.weight", {E, F});
        add(p + "ffn_up.weight", {E, F});
        add(p + "ffn_down.weight", {F, E});
        if (full) {
            // q carries the attention gate interleaved per head: 2 * head_dim per head
            add(p + "attn_q.weight", {E, H * K * 2});
            add(p + "attn_k.weight", {E, KV * K});
            add(p + "attn_v.weight", {E, KV * V});
            add(p + "attn_output.weight", {H * V, E});
            add(p + "attn_q_norm.weight", {K}, kF32);
            add(p + "attn_k_norm.weight", {K}, kF32);
        } else {
            // Gated DeltaNet: q and k are the group heads, v the value heads, and the gate is SiLU(z)
            add(p + "attn_qkv.weight", {E, G * S * 2 + T * S});
            add(p + "attn_gate.weight", {E, T * S});
            add(p + "ssm_alpha.weight", {E, T});
            add(p + "ssm_beta.weight", {E, T});
            add(p + "ssm_a", {T}, kF32);
            add(p + "ssm_dt.bias", {T}, kF32);
            add(p + "ssm_conv1d.weight", {C, G * S * 2 + T * S}, kF32);
            add(p + "ssm_norm.weight", {S}, kF32);
            add(p + "ssm_out.weight", {T * S, E});
        }
    }
    if (!problem.empty()) return problem;

    // A tensor nobody reads is as suspicious as a missing one: a converted file that kept an MTP block, or a
    // projection under another name, would otherwise pass with the engine ignoring it.
    for (const TensorInfo& t : g.tensors())
        if (expected.find(t.name) == expected.end())
            return "unexpected tensor " + t.name + ": this checkpoint's schema has no such weight";

    if (report) {
        const uint64_t full_blocks = blocks / interval;
        *report = "  geometry: " + std::to_string(blocks) + " blocks (" + std::to_string(blocks - full_blocks) +
                  " GDN + " + std::to_string(full_blocks) + " full), embedding " + std::to_string(E) +
                  ", ffn " + std::to_string(F) + ", heads " + std::to_string(H) + "/" + std::to_string(KV) + " x " +
                  std::to_string(K) + ", GDN " + std::to_string(G) + " qk / " + std::to_string(T) + " v x " +
                  std::to_string(S) + " (inner " + std::to_string(I) + ", conv " + std::to_string(C) + ")\n" +
                  "  tensors: " + std::to_string(g.tensors().size()) + " (" +
                  std::to_string(g.tensors().size() - 3) + " per block + 3 global), all shapes and types match\n" +
                  "  readout: " + std::to_string(g.get("pplx_decider.codes")->count) + " codes, ids inside the " +
                  std::to_string(vocab) + "-token vocabulary, temperature " +
                  std::to_string(g.get("pplx_decider.temperature")->num()) + "\n" +
                  "  attention: non-causal full layers, causal GDN (protocol v3, marker " +
                  g.get("pplx_decider.marker")->s + ")";
    }
    return {};
}

}  // namespace strata::decider
