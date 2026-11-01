// src/step35/step35_mtp_parity.cpp - the MTP (NextN) block, GPU against the CPU reference, on a synthetic file.
//
// The MTP block is the one piece of this engine whose parity was a manual run on the 76 GB release, because the
// reference and the kernels both need a file.  This builds a tiny one (gguf_fixture.hpp, the same writer the validator
// tests use) with ONE NextN block, and runs the same two-step chain on both sides:
//
//   step 0: (h0, token 3) at position 0  ->  h1, logits over 16 pieces of vocabulary
//   step 1: (h1, token 5) at position 1  ->  h2, logits
//
// The second step is what makes it worth running: h1 is the reference's own output fed back, exactly as a chain of
// draft heads does, so a mistake in the wiring (hnorm/enorm/eh_proj order, the head, the position) shows in the
// second row even when the first one happens to agree.
//
// Nothing here needs the model: 64 dims, 2 heads, 128-wide feed-forward, 16 tokens, ~40 KB of weights.
#include "gguf_fixture.hpp"   // tests/core: the synthetic GGUF writer

#include "step35ref.hpp"      // the CPU reference (src/step35, on the include path)

#include "strata/artifact/gguf_reader.hpp"
#include "strata/step35/block.hpp"
#include "strata/step35/session.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <vector>

namespace fs = std::filesystem;

namespace {

int failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        ++failures;
        std::printf("FAIL  %s\n", what);
    } else {
        std::printf("  ok  %s\n", what);
    }
}

constexpr int kE = 64, kHeadDim = 64, kHead = 2, kKvHead = 2, kFf = 128, kVocab = 16, kLayers = 2, kNmtp = 1;
constexpr int kMtpLayer = kLayers - kNmtp;   // 1

std::vector<fixture::Tensor> tensors() {
    auto f32 = [](const std::string& name, std::vector<uint64_t> shape, uint32_t seed) {
        fixture::Tensor t;
        t.name = name;
        t.shape = std::move(shape);
        t.type = 0;   // F32
        uint64_t n = 1;
        for (uint64_t d : t.shape) n *= d;
        t.payload = fixture::f32_payload(n, seed);
        return t;
    };
    const std::string p = "blk." + std::to_string(kMtpLayer) + ".";
    const uint64_t q = (uint64_t) kHead * kHeadDim, kv = (uint64_t) kKvHead * kHeadDim;
    return {
        f32("token_embd.weight", {(uint64_t) kE, (uint64_t) kVocab}, 1),
        f32(p + "attn_norm.weight", {(uint64_t) kE}, 2),
        f32(p + "attn_q_norm.weight", {(uint64_t) kHeadDim}, 3),
        f32(p + "attn_k_norm.weight", {(uint64_t) kHeadDim}, 4),
        f32(p + "ffn_norm.weight", {(uint64_t) kE}, 5),
        f32(p + "attn_q.weight", {(uint64_t) kE, q}, 6),
        f32(p + "attn_k.weight", {(uint64_t) kE, kv}, 7),
        f32(p + "attn_v.weight", {(uint64_t) kE, kv}, 8),
        f32(p + "attn_output.weight", {q, (uint64_t) kE}, 9),
        f32(p + "attn_gate.weight", {(uint64_t) kE, (uint64_t) kHead}, 10),
        f32(p + "ffn_gate.weight", {(uint64_t) kE, (uint64_t) kFf}, 11),
        f32(p + "ffn_up.weight", {(uint64_t) kE, (uint64_t) kFf}, 12),
        f32(p + "ffn_down.weight", {(uint64_t) kFf, (uint64_t) kE}, 13),
        f32(p + "nextn.eh_proj.weight", {(uint64_t) (2 * kE), (uint64_t) kE}, 14),
        f32(p + "nextn.enorm.weight", {(uint64_t) kE}, 15),
        f32(p + "nextn.hnorm.weight", {(uint64_t) kE}, 16),
        f32(p + "nextn.shared_head_norm.weight", {(uint64_t) kE}, 17),
        f32(p + "nextn.shared_head_head.weight", {(uint64_t) kE, (uint64_t) kVocab}, 18),
    };
}

std::vector<fixture::Kv> metadata() {
    using namespace fixture;
    const std::vector<uint64_t> heads(kLayers, kHead), kv_heads(kLayers, kKvHead), swa(kLayers, 1), no_clamp(kLayers, 0);
    std::vector<std::string> tokens;
    for (int i = 0; i < kVocab; ++i) tokens.push_back("t" + std::to_string(i));
    return {
        str("general.architecture", "step35"),
        u32("step35.block_count", kLayers),
        u32("step35.nextn_predict_layers", kNmtp),
        u32("step35.embedding_length", kE),
        u32("step35.feed_forward_length", kFf),
        u32("step35.attention.key_length", kHeadDim),
        u32("step35.attention.value_length", kHeadDim),
        u32("step35.attention.sliding_window", 512),
        u32s("step35.attention.head_count", heads),
        u32s("step35.attention.head_count_kv", kv_heads),
        u32s("step35.attention.sliding_window_pattern", swa),
        u32s("step35.swiglu_clamp_exp", no_clamp),
        u32s("step35.swiglu_clamp_shexp", no_clamp),
        f32("step35.attention.layer_norm_rms_epsilon", 1e-5f),
        f32("step35.rope.freq_base", 5e6f),
        f32("step35.rope.freq_base_swa", 1e4f),
        u32("step35.leading_dense_block_count", 1),
        u32("step35.expert_count", 4),
        u32("step35.expert_used_count", 2),
        u32("step35.expert_feed_forward_length", 32),
        u32("step35.expert_shared_feed_forward_length", 32),
        u32("step35.expert_gating_func", 2),
        f32("step35.expert_weights_scale", 1.0f),
        u32s("tokenizer.ggml.token_type", std::vector<uint64_t>(kVocab, 1)),
        str_array("tokenizer.ggml.tokens", tokens),
    };
}

/// The highest-logit token.
int argmax(const std::vector<float>& v) {
    int best = 0;
    for (int i = 1; i < (int) v.size(); ++i)
        if (v[(size_t) i] > v[(size_t) best]) best = i;
    return best;
}

double rel_l2(const std::vector<float>& got, const std::vector<float>& ref) {
    double num = 0, den = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double d = (double) got[i] - (double) ref[i];
        num += d * d;
        den += (double) ref[i] * (double) ref[i];
    }
    return den > 0 ? std::sqrt(num / den) : 0.0;
}

}  // namespace

int main() {
    const fs::path dir = fs::temp_directory_path() / "strata-step35-mtp";
    fs::create_directories(dir);
    const fs::path path = dir / "mtp.gguf";
    try {
        fixture::write(path, metadata(), tensors());
    } catch (const std::exception& ex) {
        std::printf("FAIL  the fixture could not be written: %s\n", ex.what());
        return 1;
    }

    // ---- the reference
    strata::step35::ref::Model ref;
    std::string err;
    if (!strata::step35::ref::load(path.string(), ref, err)) {
        std::printf("FAIL  the reference cannot load the fixture: %s\n", err.c_str());
        return 1;
    }
    check(ref.n_layers == kLayers && ref.n_embd == kE, "the reference reads the fixture's geometry");
    // h0: what a trunk would hand the first head (any vector; both sides get the same one)
    std::vector<float> h0((size_t) kE);
    for (int i = 0; i < kE; ++i) h0[(size_t) i] = 0.1f * std::sin((double) i);
    std::vector<std::vector<strata::step35::ref::PositionKv>> ref_kv((size_t) kNmtp,
                                                                    std::vector<strata::step35::ref::PositionKv>(2));
    std::vector<float> ref_h1, ref_l1, ref_h2, ref_l2;
    if (!strata::step35::ref::mtp_block(ref, kMtpLayer, 0, 3, h0, ref_kv[0], ref_h1, ref_l1, err) ||
        !strata::step35::ref::mtp_block(ref, kMtpLayer, 1, 5, ref_h1, ref_kv[0], ref_h2, ref_l2, err)) {
        std::printf("FAIL  the reference cannot run the fixture: %s\n", err.c_str());
        return 1;
    }

    // ---- the kernels, on the same file
    try {
        strata::GgufModel mtp = strata::GgufModel::open(path.string());
        strata::step35::kernels::MtpContext ctx;
        if (!strata::step35::kernels::mtp_open(mtp, ctx, err)) {
            std::printf("FAIL  mtp_open: %s\n", err.c_str());
            return 1;
        }
        check(ctx.n_mtp == kNmtp && ctx.first == kMtpLayer && ctx.n_vocab == kVocab,
              "the NextN file opens with its own block, head and vocabulary");
        strata::step35::kernels::Session session;
        if (!strata::step35::kernels::session_create(session, kNmtp, kKvHead, kHeadDim, 8, err,
                                                     std::vector<int>(kNmtp, 512))) {
            std::printf("FAIL  session_create: %s\n", err.c_str());
            return 1;
        }
        std::vector<float> gpu_h1, gpu_l1, gpu_h2, gpu_l2;
        const bool ok0 = strata::step35::kernels::mtp_step(ctx, 0, 0, 0, 3, h0.data(), &session, gpu_h1, gpu_l1, err);
        const bool ok1 = ok0 && strata::step35::kernels::mtp_step(ctx, 0, 0, 1, 5, gpu_h1.data(), &session, gpu_h2,
                                                                  gpu_l2, err);
        if (!ok1) {
            std::printf("FAIL  mtp_step: %s\n", err.c_str());
            return 1;
        }
        check(argmax(gpu_l1) == argmax(ref_l1), "step 0: the same top-1 token as the reference");
        const double r1 = rel_l2(gpu_l1, ref_l1);
        char label[240];
        // The kernels quantize every activation to Q8_1 (that is what the K-quant matvec carries by design, and what
        // the gemv parity test accepts at 3.7e-03): 1e-3 is the floor here, not 1e-5.
        std::snprintf(label, sizeof label, "step 0: the logits are the reference's to the quantization (rel_l2 %.1e)", r1);
        check(r1 < 5e-3, label);
        check(argmax(gpu_l2) == argmax(ref_l2), "step 1, on the reference's own h1 fed back: the same top-1 token");
        const double r2 = rel_l2(gpu_l2, ref_l2);
        std::snprintf(label, sizeof label, "step 1: the logits are the reference's to the quantization (rel_l2 %.1e)", r2);
        check(r2 < 5e-3, label);
        const double rh = rel_l2(gpu_h2, ref_h2);
        std::snprintf(label, sizeof label, "step 1: h_nextn matches the reference's to the quantization (rel_l2 %.1e)", rh);
        check(rh < 5e-3, label);
        strata::step35::kernels::session_destroy(session);
        strata::step35::kernels::mtp_close(ctx);
    } catch (const std::exception& ex) {
        std::printf("FAIL  the kernels cannot run the fixture: %s\n", ex.what());
        return 1;
    }
    std::error_code ec;
    fs::remove_all(dir, ec);

    if (failures) {
        std::printf("step35_mtp_parity: %d check(s) failed\n", failures);
        return 1;
    }
    std::printf("step35_mtp_parity: all checks passed\n");
    return 0;
}
