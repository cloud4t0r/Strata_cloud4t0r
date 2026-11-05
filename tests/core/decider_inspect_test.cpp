// tests/core/decider_inspect_test.cpp - the decider's tensor audit (src/decider/inspect.cpp) on a synthetic file.
//
// The geometry is small (8 blocks, embedding 8) so the payloads stay a few KB, but the schema is the release's: a
// GDN block carries attn_qkv/attn_gate/ssm_* and no q/k/v, a full-attention block carries q/k/v/q_norm/k_norm and
// no ssm_*, and every width is derived from the metadata rather than written as a constant.  The audit must accept
// the whole file, and refuse a missing weight, a wrong width, a wrong type and a weight nobody reads.
#include "gguf_fixture.hpp"

#include "strata/decider/inspect.hpp"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        ++failures;
        std::printf("FAIL  %s\n", what.c_str());
    } else {
        std::printf("  ok  %s\n", what.c_str());
    }
}

struct TempDir {
    fs::path path;
    TempDir() {
        path = fs::temp_directory_path() /
               ("strata-decider-inspect-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        fs::create_directories(path);
    }
    ~TempDir() {
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }
};

constexpr uint64_t kBlocks = 8;
constexpr uint64_t kInterval = 4;
// A Q8_0 tensor's first dimension is a multiple of the 32-element block, so every width that ends up as dim 0
// (embedding, feed-forward, head*value and the GDN value width, the vocabulary) is a multiple of 32.
constexpr uint64_t kE = 32;     // embedding
constexpr uint64_t kF = 32;     // feed-forward
constexpr uint64_t kH = 8;      // attention heads
constexpr uint64_t kKV = 2;     // KV heads
constexpr uint64_t kK = 4;      // key length
constexpr uint64_t kV = 4;      // value length (head * value = 32)
constexpr uint64_t kG = 8;      // GDN q/k heads
constexpr uint64_t kT = 4;      // GDN value heads
constexpr uint64_t kS = 8;      // GDN state size (value width = 32)
constexpr uint64_t kC = 2;      // GDN conv kernel
constexpr uint64_t kVocab = 64;

std::vector<uint64_t> recurrent_layers() {
    std::vector<uint64_t> v;
    for (uint64_t i = 0; i < kBlocks; ++i) v.push_back((i + 1) % kInterval == 0 ? 0 : 1);
    return v;
}

std::vector<fixture::Kv> metadata() {
    const std::string p = "qwen35.";
    std::vector<fixture::Kv> kv = {
        fixture::str("general.architecture", "qwen35"),
        fixture::u32(p + "block_count", kBlocks),
        fixture::u32(p + "context_length", 4096),
        fixture::u32(p + "embedding_length", kE),
        fixture::u32(p + "feed_forward_length", kF),
        fixture::u32(p + "attention.head_count", kH),
        fixture::u32(p + "attention.head_count_kv", kKV),
        fixture::u32(p + "attention.key_length", kK),
        fixture::u32(p + "attention.value_length", kV),
        fixture::f32(p + "attention.layer_norm_rms_epsilon", 1e-6f),
        fixture::u32s(p + "rope.dimension_sections", {1, 0, 0, 0}),
        fixture::f32(p + "rope.freq_base", 1e7f),
        fixture::u32(p + "rope.dimension_count", 2),
        fixture::u32(p + "full_attention_interval", kInterval),
        fixture::u32(p + "ssm.conv_kernel", kC),
        fixture::u32(p + "ssm.state_size", kS),
        fixture::u32(p + "ssm.group_count", kG),
        fixture::u32(p + "ssm.time_step_rank", kT),
        fixture::u32(p + "ssm.inner_size", kT * kS),
        fixture::bools(p + "attention.recurrent_layers", recurrent_layers()),
        fixture::boolean(p + "attention.causal", false),
        fixture::u32("pplx_decider.protocol_version", 3),
        fixture::str("pplx_decider.template", "joint-question-inline-readout-v1"),
        fixture::str("pplx_decider.marker", "<|fim_suffix|>"),
        fixture::f32("pplx_decider.temperature", 1.5539092f),
        fixture::strs("pplx_decider.codes", {"A", "B"}),
        fixture::u32s("pplx_decider.token_ids", {1, 2}),
        fixture::strs("tokenizer.ggml.tokens", std::vector<std::string>(kVocab, "x")),
    };
    return kv;
}

/// Every tensor the audit expects, with the widths the metadata implies.
std::vector<fixture::Tensor> tensors() {
    auto t = [](const std::string& name, std::vector<uint64_t> shape, uint32_t type) {
        fixture::Tensor x;
        x.name = name;
        x.shape = std::move(shape);
        x.type = type;
        return x;
    };
    const uint32_t f32 = 0, q8 = 8;
    std::vector<fixture::Tensor> ts = {
        t("token_embd.weight", {kE, kVocab}, q8),
        t("output.weight", {kE, kVocab}, q8),
        t("output_norm.weight", {kE}, f32),
    };
    for (uint64_t b = 0; b < kBlocks; ++b) {
        const std::string p = "blk." + std::to_string(b) + ".";
        ts.push_back(t(p + "attn_norm.weight", {kE}, f32));
        ts.push_back(t(p + "post_attention_norm.weight", {kE}, f32));
        ts.push_back(t(p + "ffn_gate.weight", {kE, kF}, q8));
        ts.push_back(t(p + "ffn_up.weight", {kE, kF}, q8));
        ts.push_back(t(p + "ffn_down.weight", {kF, kE}, q8));
        if ((b + 1) % kInterval == 0) {
            ts.push_back(t(p + "attn_q.weight", {kE, kH * kK * 2}, q8));
            ts.push_back(t(p + "attn_k.weight", {kE, kKV * kK}, q8));
            ts.push_back(t(p + "attn_v.weight", {kE, kKV * kV}, q8));
            ts.push_back(t(p + "attn_output.weight", {kH * kV, kE}, q8));
            ts.push_back(t(p + "attn_q_norm.weight", {kK}, f32));
            ts.push_back(t(p + "attn_k_norm.weight", {kK}, f32));
        } else {
            ts.push_back(t(p + "attn_qkv.weight", {kE, kG * kS * 2 + kT * kS}, q8));
            ts.push_back(t(p + "attn_gate.weight", {kE, kT * kS}, q8));
            ts.push_back(t(p + "ssm_alpha.weight", {kE, kT}, q8));
            ts.push_back(t(p + "ssm_beta.weight", {kE, kT}, q8));
            ts.push_back(t(p + "ssm_a", {kT}, f32));
            ts.push_back(t(p + "ssm_dt.bias", {kT}, f32));
            ts.push_back(t(p + "ssm_conv1d.weight", {kC, kG * kS * 2 + kT * kS}, f32));
            ts.push_back(t(p + "ssm_norm.weight", {kS}, f32));
            ts.push_back(t(p + "ssm_out.weight", {kT * kS, kE}, q8));
        }
    }
    return ts;
}

/// The audit's answer for a file built from `kv` and `ts` (a fresh name each time).
std::string audit(const fs::path& dir, const std::vector<fixture::Kv>& kv, const std::vector<fixture::Tensor>& ts) {
    static int counter = 0;
    const fs::path p = dir / ("case-" + std::to_string(++counter) + ".gguf");
    fixture::write(p, kv, ts);
    const strata::GgufFile f(p.string());
    return strata::decider::inspect(f);
}

/// `ts` without one tensor.
std::vector<fixture::Tensor> without(std::vector<fixture::Tensor> ts, const std::string& name) {
    for (size_t i = 0; i < ts.size(); ++i)
        if (ts[i].name == name) {
            ts.erase(ts.begin() + (long) i);
            break;
        }
    return ts;
}

}  // namespace

int main() {
    TempDir dir;

    // ---- the release's schema, scaled down
    {
        std::string report;
        const fs::path p = dir.path / "whole.gguf";
        fixture::write(p, metadata(), tensors());
        const strata::GgufFile f(p.string());
        const std::string err = strata::decider::inspect(f, &report);
        check(err.empty(), "the whole file passes the audit" + (err.empty() ? "" : ": " + err));
        check(report.find("8 blocks (6 GDN + 2 full)") != std::string::npos, "the report counts the layer kinds");
        check(report.find(std::to_string(tensors().size()) + " (") != std::string::npos,
              "the report counts the tensors it checked");
        check(report.find("2 codes") != std::string::npos, "the report names the readout's code count");
    }

    // ---- what the audit must refuse
    {
        const std::string err = audit(dir.path, metadata(), without(tensors(), "blk.1.ffn_down.weight"));
        check(err.find("missing tensor blk.1.ffn_down.weight") != std::string::npos,
              "a missing weight is named: " + err);
    }
    {
        // q carries the gate interleaved: a file that dropped the gate halves it
        std::vector<fixture::Tensor> ts = tensors();
        for (auto& t : ts)
            if (t.name == "blk.3.attn_q.weight") t.shape = {kE, kH * kK};
        const std::string err = audit(dir.path, metadata(), ts);
        check(err.find("blk.3.attn_q.weight") != std::string::npos && err.find("geometry implies") != std::string::npos,
              "a projection of the wrong width is refused: " + err);
    }
    {
        std::vector<fixture::Tensor> ts = tensors();
        for (auto& t : ts)
            if (t.name == "blk.0.attn_norm.weight") t.type = 8;
        const std::string err = audit(dir.path, metadata(), ts);
        check(err.find("blk.0.attn_norm.weight") != std::string::npos && err.find("F32") != std::string::npos,
              "a norm stored in another type is refused: " + err);
    }
    {
        // an MTP block, or any weight the engine would silently ignore
        std::vector<fixture::Tensor> ts = tensors();
        fixture::Tensor extra;
        extra.name = "blk.0.mtp_proj.weight";
        extra.shape = {kE, kE};
        extra.type = 8;
        ts.push_back(extra);
        const std::string err = audit(dir.path, metadata(), ts);
        check(err.find("unexpected tensor blk.0.mtp_proj.weight") != std::string::npos,
              "a weight nobody reads is refused: " + err);
    }
    {
        // a readout code past the vocabulary: the head is indexed by vocabulary id
        std::vector<fixture::Kv> kv = metadata();
        for (auto& e : kv)
            if (e.key == "pplx_decider.token_ids") e.arr = {1, kVocab + 7};
        const std::string err = audit(dir.path, kv, tensors());
        check(err.find("past the") != std::string::npos, "a readout id outside the vocabulary is refused: " + err);
    }

    if (failures) {
        std::printf("decider_inspect_test: %d check(s) failed\n", failures);
        return 1;
    }
    std::printf("decider_inspect_test: all checks passed\n");
    return 0;
}
