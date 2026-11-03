// src/step35/step35_engine.cu - the Step-3.7-Flash (GGUF architecture step35) engine path.
//
// What is here today is the file's own validator: `strata-step35 --inspect <first shard>` reads the metadata and
// every tensor of the trunk (and, with --mtp, of the NextN file) and checks that they agree - each layer's Q width
// against its own head count, the dense layers against leading_dense_block_count, the expert tensors against
// expert_count, the vocabulary against the tokenizer.  It opens no GPU and writes nothing; a file that fails a
// check is reported with the tensor or key that differs and both values, and the process exits 1.
//
// The forward pass (attention, MoE, MTP) comes next, and this file is where it goes.
#include "strata/step35/engine.hpp"

#include "step35ref.hpp"

#include "strata/artifact/gguf_reader.hpp"
#include "strata/step35/block.hpp"
#include "strata/step35/experts.hpp"
#include "strata/step35/gemv.hpp"
#include "strata/step35/profile.hpp"
#include "strata/step35/sampler.hpp"
#include "strata/step35/session.hpp"

#include <ggml.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace strata::step35 {
namespace {

using strata::GgufModel;
using strata::MetaType;
using strata::MetaValue;
using strata::TensorInfo;

// ------------------------------------------------------------------------------------------------ the report
/// One line per thing checked; `bad` is what failed and decides the exit code.
struct Report {
    int bad = 0;

    void fail(const std::string& what) {
        ++bad;
        std::printf("  FAILED  %s\n", what.c_str());
    }
};

std::string number(double v) {
    char b[64];
    if (v == (double) (int64_t) v && v > -1e15 && v < 1e15) {
        std::snprintf(b, sizeof b, "%lld", (long long) (int64_t) v);
    } else {
        std::snprintf(b, sizeof b, "%.6g", v);
    }
    return b;
}

std::string shape_of(const std::vector<uint64_t>& s) {
    std::string out = "[";
    for (size_t i = 0; i < s.size(); ++i) {
        if (i) out += ", ";
        out += std::to_string(s[i]);
    }
    return out + "]";
}

std::string gigabytes(uint64_t bytes) {
    char b[64];
    std::snprintf(b, sizeof b, "%.2f GB", (double) bytes / 1e9);
    return b;
}

// ------------------------------------------------------------------------------------------------ metadata
/// An integer key.  Absent and not required: `*out` is left alone.
bool get_int(const GgufModel& gm, Report& r, const std::string& key, int64_t& out, bool required = true) {
    const MetaValue* v = gm.meta().get(key);
    if (!v) {
        if (required) r.fail("missing " + key);
        return !required;
    }
    if (!v->is_num() || v->type == MetaType::BOOL) {
        r.fail(key + " is not a whole number");
        return false;
    }
    out = (int64_t) v->num();
    return true;
}

/// A number key that may carry a fraction (epsilon, the RoPE bases, the expert scale).
bool get_num(const GgufModel& gm, Report& r, const std::string& key, double& out, bool required = true) {
    const MetaValue* v = gm.meta().get(key);
    if (!v) {
        if (required) r.fail("missing " + key);
        return !required;
    }
    if (!v->is_num()) {
        r.fail(key + " is not a number");
        return false;
    }
    out = v->num();
    return true;
}

bool get_bool(const GgufModel& gm, Report& r, const std::string& key, bool& out, bool required = true) {
    const MetaValue* v = gm.meta().get(key);
    if (!v) {
        if (required) r.fail("missing " + key);
        return !required;
    }
    if (v->type != MetaType::BOOL) {
        r.fail(key + " is not a boolean");
        return false;
    }
    out = v->u != 0;
    return true;
}

/// A whole array (per-layer head counts, the sliding-window pattern, the clamp widths).  Empty when absent and
/// not required.
bool get_arr(const GgufModel& gm, Report& r, const std::string& key, std::vector<double>& out,
             bool required = true) {
    out.clear();
    const MetaValue* v = gm.meta().get(key);
    if (!v) {
        if (required) r.fail("missing " + key);
        return !required;
    }
    if (v->type != MetaType::ARRAY) {
        r.fail(key + " is not an array");
        return false;
    }
    // GgufFile keeps a sample of the first 64 entries of an array (the count is exact); a per-layer array longer
    // than that would be read short here, so say so instead of planning on a truncated pattern.
    if (v->items.size() != v->count) {
        r.fail(key + " has " + std::to_string(v->count) + " entries; this reader samples the first 64, so a model "
                "with that many layers needs the reader widened first");
        return false;
    }
    for (const MetaValue& item : v->items) {
        if (!item.is_num()) {
            r.fail(key + " holds something that is not a number");
            return false;
        }
        out.push_back(item.num());
    }
    return true;
}

// ------------------------------------------------------------------------------------------------ tensors
/// One tensor, against the shape the metadata says it must have.  The failure names both, as model_family_test
/// does for a wrong QK-norm.
bool want(const GgufModel& gm, Report& r, const std::string& name, const std::vector<uint64_t>& shape) {
    const TensorInfo* t = gm.find(name);
    if (!t) {
        r.fail("missing tensor " + name);
        return false;
    }
    if (t->shape == shape) return true;
    r.fail(name + ": " + shape_of(t->shape) + ", wanted " + shape_of(shape));
    return false;
}

void expect_absent(const GgufModel& gm, Report& r, const std::string& name, const char* why) {
    if (gm.find(name)) r.fail(name + " is present, but " + why);
}

// ------------------------------------------------------------------------------------------------ geometry
struct Geometry {
    int64_t n_layers = 0, n_embd = 0, n_vocab = 0, ctx = 0;
    int64_t head_dim = 0, value_dim = 0;
    int64_t ff_dense = 0;
    int64_t n_expert = 0, top_k = 0, ff_expert = 0, ff_shexp = 0;
    int64_t dense_layers = 0, gating = 0, window = 0;
    int64_t n_rot_full = 0, n_rot_swa = 0;
    double eps = 0, rope_full = 0, rope_swa = 0, expert_scale = 0;
    bool expert_norm = false;
    std::vector<double> n_head, n_head_kv, is_swa, clamp_exp, clamp_shexp;

    int64_t heads(int64_t layer) const { return (int64_t) n_head[(size_t) layer]; }
    int64_t kv_heads(int64_t layer) const { return (int64_t) n_head_kv[(size_t) layer]; }
    bool sliding(int64_t layer) const { return is_swa[(size_t) layer] != 0; }
    bool moe(int64_t layer) const { return layer >= dense_layers; }
};

/// Step-3.7-Flash's own keys, from the file.  Returns false as soon as something is missing or inconsistent.
bool read_metadata(const GgufModel& gm, Report& r, Geometry& g) {
    const std::string p = "step35.";
    get_int(gm, r, p + "block_count", g.n_layers);
    get_int(gm, r, p + "embedding_length", g.n_embd);
    get_int(gm, r, p + "context_length", g.ctx);
    get_int(gm, r, p + "feed_forward_length", g.ff_dense);
    get_int(gm, r, p + "attention.key_length", g.head_dim);
    get_int(gm, r, p + "attention.value_length", g.value_dim);
    get_int(gm, r, p + "attention.sliding_window", g.window);
    get_int(gm, r, p + "expert_count", g.n_expert);
    get_int(gm, r, p + "expert_used_count", g.top_k);
    get_int(gm, r, p + "expert_feed_forward_length", g.ff_expert);
    get_int(gm, r, p + "expert_shared_feed_forward_length", g.ff_shexp);
    get_int(gm, r, p + "leading_dense_block_count", g.dense_layers);
    get_int(gm, r, p + "expert_gating_func", g.gating);
    get_num(gm, r, p + "attention.layer_norm_rms_epsilon", g.eps);
    get_num(gm, r, p + "rope.freq_base", g.rope_full);
    get_num(gm, r, p + "rope.freq_base_swa", g.rope_swa);
    get_num(gm, r, p + "expert_weights_scale", g.expert_scale);
    get_bool(gm, r, p + "expert_weights_norm", g.expert_norm);
    get_arr(gm, r, p + "attention.head_count", g.n_head);
    get_arr(gm, r, p + "attention.head_count_kv", g.n_head_kv);
    get_arr(gm, r, p + "attention.sliding_window_pattern", g.is_swa);
    get_arr(gm, r, p + "swiglu_clamp_exp", g.clamp_exp, false);
    get_arr(gm, r, p + "swiglu_clamp_shexp", g.clamp_shexp, false);
    // llama.cpp's rule for this file, which carries neither key: every layer rotates head_dim dims, and a FULL
    // attention layer keeps half of them (step35.cpp: "full_attention layer only use half of the RoPE dimensions").
    // The 64-entry rope_freqs.weight tensor of the release agrees with the widest of the two.
    g.n_rot_swa = g.head_dim;
    get_int(gm, r, p + "rope.dimension_count", g.n_rot_swa, false);
    g.n_rot_full = g.n_rot_swa / 2;
    get_int(gm, r, p + "rope.dimension_count_full", g.n_rot_full, false);
    if (g.n_layers <= 0) return false;
    // the three per-layer arrays must describe every block, or the layer loop below reads past their end
    auto sized = [&](const char* what, const std::vector<double>& a) {
        if ((int64_t) a.size() == g.n_layers) return true;
        r.fail(std::string("step35.") + what + " has " + std::to_string(a.size()) + " entries for " +
               std::to_string(g.n_layers) + " blocks");
        return false;
    };
    if (!sized("attention.head_count", g.n_head)) return false;
    if (!sized("attention.head_count_kv", g.n_head_kv)) return false;
    if (!sized("attention.sliding_window_pattern", g.is_swa)) return false;
    for (int64_t il = 0; il < g.n_layers; ++il) {
        if (g.heads(il) <= 0 || g.heads(il) % g.kv_heads(il) != 0) {
            r.fail("layer " + std::to_string(il) + ": " + number(g.heads(il)) + " Q heads for " +
                   number(g.kv_heads(il)) + " KV heads is not a whole GQA split");
            return false;
        }
    }
    if (g.top_k > g.n_expert) {
        r.fail("expert_used_count " + number((double) g.top_k) + " is bigger than expert_count " +
               number((double) g.n_expert));
        return false;
    }
    if (g.dense_layers > g.n_layers) {
        r.fail("leading_dense_block_count " + number((double) g.dense_layers) + " is bigger than block_count " +
               number((double) g.n_layers));
        return false;
    }
    // the vocabulary the tensor shapes must match comes from the tokenizer's own list (its count: the reader
    // keeps a 64-entry sample of the list itself)
    if (const MetaValue* tok = gm.meta().get("tokenizer.ggml.tokens")) {
        if (tok->type != MetaType::ARRAY) {
            r.fail("tokenizer.ggml.tokens is not an array");
            return false;
        }
        g.n_vocab = (int64_t) tok->count;
    } else {
        r.fail("missing tokenizer.ggml.tokens");
        return false;
    }
    return r.bad == 0;
}

/// The trunk: every block's tensors, in the shapes its own layer type implies.
bool check_trunk(const GgufModel& gm, Report& r, const Geometry& g) {
    const int64_t e = g.n_embd, hd = g.head_dim;
    want(gm, r, "token_embd.weight", {(uint64_t) e, (uint64_t) g.n_vocab});
    want(gm, r, "output_norm.weight", {(uint64_t) e});
    want(gm, r, "output.weight", {(uint64_t) e, (uint64_t) g.n_vocab});
    // one shared tensor holds the RoPE factors of every layer: llama.cpp sizes it by the widest rotation (a
    // sliding layer keeps all 128 dims, a full one 64) and each layer reads its own prefix
    want(gm, r, "rope_freqs.weight", {(uint64_t) (g.n_rot_swa / 2)});
    for (int64_t il = 0; il < g.n_layers; ++il) {
        const std::string b = "blk." + std::to_string(il) + ".";
        const uint64_t q = (uint64_t) g.heads(il) * (uint64_t) hd;
        const uint64_t kv = (uint64_t) g.kv_heads(il) * (uint64_t) hd;
        want(gm, r, b + "attn_norm.weight", {(uint64_t) e});
        want(gm, r, b + "attn_q.weight", {(uint64_t) e, q});
        want(gm, r, b + "attn_k.weight", {(uint64_t) e, kv});
        want(gm, r, b + "attn_v.weight", {(uint64_t) e, kv});
        want(gm, r, b + "attn_output.weight", {q, (uint64_t) e});
        want(gm, r, b + "attn_q_norm.weight", {(uint64_t) hd});
        want(gm, r, b + "attn_k_norm.weight", {(uint64_t) hd});
        // the head-wise gate: one weight per query head
        want(gm, r, b + "attn_gate.weight", {(uint64_t) e, (uint64_t) g.heads(il)});
        want(gm, r, b + "ffn_norm.weight", {(uint64_t) e});
        if (!g.moe(il)) {
            want(gm, r, b + "ffn_gate.weight", {(uint64_t) e, (uint64_t) g.ff_dense});
            want(gm, r, b + "ffn_up.weight", {(uint64_t) e, (uint64_t) g.ff_dense});
            want(gm, r, b + "ffn_down.weight", {(uint64_t) g.ff_dense, (uint64_t) e});
            expect_absent(gm, r, b + "ffn_gate_inp.weight", "the layer is one of the dense ones");
        } else {
            want(gm, r, b + "ffn_gate_inp.weight", {(uint64_t) e, (uint64_t) g.n_expert});
            want(gm, r, b + "exp_probs_b.bias", {(uint64_t) g.n_expert});
            want(gm, r, b + "ffn_gate_exps.weight", {(uint64_t) e, (uint64_t) g.ff_expert, (uint64_t) g.n_expert});
            want(gm, r, b + "ffn_up_exps.weight", {(uint64_t) e, (uint64_t) g.ff_expert, (uint64_t) g.n_expert});
            want(gm, r, b + "ffn_down_exps.weight",
                 {(uint64_t) g.ff_expert, (uint64_t) e, (uint64_t) g.n_expert});
            want(gm, r, b + "ffn_gate_shexp.weight", {(uint64_t) e, (uint64_t) g.ff_shexp});
            want(gm, r, b + "ffn_up_shexp.weight", {(uint64_t) e, (uint64_t) g.ff_shexp});
            want(gm, r, b + "ffn_down_shexp.weight", {(uint64_t) g.ff_shexp, (uint64_t) e});
            expect_absent(gm, r, b + "ffn_gate.weight", "the layer is a mixture of experts");
        }
    }
    return r.bad == 0;
}

/// The NextN file: `nextn_predict_layers` extra blocks, each a decoder layer plus the eh_proj wiring.  Its own
/// token_embd / output_norm / output are allowed (the release carries copies) and are not checked here.  The
/// per-layer arrays are the MTP file's own: it declares all 48 blocks, the trunk only its 45.
bool check_mtp(const GgufModel& gm, Report& r, const Geometry& trunk, int64_t& n_mtp) {
    Geometry m;
    get_int(gm, r, "step35.block_count", m.n_layers);
    int64_t nextn = 0;
    get_int(gm, r, "step35.nextn_predict_layers", nextn);
    n_mtp = nextn;
    if (nextn <= 0 || nextn > trunk.n_layers) {
        r.fail("step35.nextn_predict_layers " + number((double) nextn) + " is not a plausible count");
        return false;
    }
    if (m.n_layers != trunk.n_layers + nextn) {
        r.fail("step35.block_count " + number((double) m.n_layers) + " is not the trunk's " +
               number((double) trunk.n_layers) + " plus " + number((double) nextn) + " MTP blocks");
        return false;
    }
    get_arr(gm, r, "step35.attention.head_count", m.n_head);
    get_arr(gm, r, "step35.attention.head_count_kv", m.n_head_kv);
    get_arr(gm, r, "step35.attention.sliding_window_pattern", m.is_swa);
    if ((int64_t) m.n_head.size() != m.n_layers || (int64_t) m.n_head_kv.size() != m.n_layers) {
        r.fail("the MTP file's per-layer head counts do not cover its " + number((double) m.n_layers) + " blocks");
        return false;
    }
    const int64_t e = trunk.n_embd, hd = trunk.head_dim;
    for (int64_t il = trunk.n_layers; il < trunk.n_layers + nextn; ++il) {
        const std::string b = "blk." + std::to_string(il) + ".";
        const uint64_t heads = (uint64_t) m.heads(il);
        const uint64_t q = heads * (uint64_t) hd;
        const uint64_t kv = (uint64_t) m.kv_heads(il) * (uint64_t) hd;
        want(gm, r, b + "nextn.eh_proj.weight", {(uint64_t) (2 * e), (uint64_t) e});
        want(gm, r, b + "nextn.enorm.weight", {(uint64_t) e});
        want(gm, r, b + "nextn.hnorm.weight", {(uint64_t) e});
        want(gm, r, b + "attn_norm.weight", {(uint64_t) e});
        want(gm, r, b + "attn_q.weight", {(uint64_t) e, q});
        want(gm, r, b + "attn_k.weight", {(uint64_t) e, kv});
        want(gm, r, b + "attn_v.weight", {(uint64_t) e, kv});
        want(gm, r, b + "attn_output.weight", {q, (uint64_t) e});
        want(gm, r, b + "attn_q_norm.weight", {(uint64_t) hd});
        want(gm, r, b + "attn_k_norm.weight", {(uint64_t) hd});
        want(gm, r, b + "attn_gate.weight", {(uint64_t) e, heads});
        want(gm, r, b + "ffn_norm.weight", {(uint64_t) e});
        // each MTP block is a DENSE decoder layer, not a mixture of experts
        want(gm, r, b + "ffn_gate.weight", {(uint64_t) e, (uint64_t) trunk.ff_dense});
        want(gm, r, b + "ffn_up.weight", {(uint64_t) e, (uint64_t) trunk.ff_dense});
        want(gm, r, b + "ffn_down.weight", {(uint64_t) trunk.ff_dense, (uint64_t) e});
        expect_absent(gm, r, b + "ffn_gate_inp.weight", "an MTP block is a dense decoder layer");
    }
    return r.bad == 0;
}

uint64_t payload_bytes(const GgufModel& gm) {
    uint64_t total = 0;
    for (size_t s = 0; s < gm.size(); ++s)
        for (const TensorInfo& t : gm.shard(s).tensors()) total += strata::tensor_payload_bytes(t);
    return total;
}

void print_geometry(const Geometry& g, int64_t n_mtp) {
    int64_t full = 0, swa = 0;
    for (int64_t il = 0; il < g.n_layers; ++il) (g.sliding(il) ? swa : full) += 1;
    int64_t clamp_layers = 0;
    for (size_t i = 0; i < g.clamp_exp.size(); ++i)
        if (g.clamp_exp[i] != 0 || g.clamp_shexp[i] != 0) ++clamp_layers;
    std::printf("  blocks          %lld (%lld dense, %lld mixture of experts) + %lld MTP\n",
                (long long) g.n_layers, (long long) g.dense_layers,
                (long long) (g.n_layers - g.dense_layers), (long long) n_mtp);
    std::printf("  embedding       %lld, vocabulary %lld, context %lld\n", (long long) g.n_embd,
                (long long) g.n_vocab, (long long) g.ctx);
    std::printf("  attention       %lld full / %lld sliding (window %lld), head dim %lld, rotation %lld dims full "
                "/ %lld sliding\n",
                (long long) full, (long long) swa, (long long) g.window, (long long) g.head_dim,
                (long long) g.n_rot_full, (long long) g.n_rot_swa);
    std::printf("  rope            base %s (full) / %s (sliding), eps %.1e\n", number(g.rope_full).c_str(),
                number(g.rope_swa).c_str(), g.eps);
    std::printf("  experts         %lld, top %lld, ffn %lld + shared %lld, gating %lld, scale %s, weights norm %s\n",
                (long long) g.n_expert, (long long) g.top_k, (long long) g.ff_expert,
                (long long) g.ff_shexp, (long long) g.gating, number(g.expert_scale).c_str(),
                g.expert_norm ? "on" : "off");
    std::printf("  dense ffn       %lld\n", (long long) g.ff_dense);
    std::printf("  swiglu clamp    %lld layers", (long long) clamp_layers);
    if (clamp_layers) {
        std::printf(":");
        for (size_t i = 0; i < g.clamp_exp.size(); ++i)
            if (g.clamp_exp[i] != 0 || g.clamp_shexp[i] != 0)
                std::printf(" %zu(%s/%s)", i, number(g.clamp_exp[i]).c_str(), number(g.clamp_shexp[i]).c_str());
    }
    std::printf("\n");
}

/// What the trunk costs and what its tokenizer is, so a run can be planned before a line of the engine exists.
void print_file(const GgufModel& gm) {
    const MetaValue* model = gm.meta().get("tokenizer.ggml.model");
    const MetaValue* pre = gm.meta().get("tokenizer.ggml.pre");
    const MetaValue* merges = gm.meta().get("tokenizer.ggml.merges");
    const MetaValue* eos = gm.meta().get("tokenizer.ggml.eos_token_id");
    const MetaValue* chat = gm.meta().get("tokenizer.chat_template");
    size_t tensors = 0;
    for (size_t s = 0; s < gm.size(); ++s) tensors += gm.shard(s).tensors().size();
    std::string tok = model && model->type == MetaType::STRING ? model->s : "(none)";
    if (pre && pre->type == MetaType::STRING) tok += " / " + pre->s;
    std::printf("  file            %zu shard(s), %zu tensors, %s of weights\n", gm.size(), tensors,
                gigabytes(payload_bytes(gm)).c_str());
    std::printf("  tokenizer       %s, %zu merges, eos %s, chat template %s\n", tok.c_str(),
                (size_t) (merges && merges->type == MetaType::ARRAY ? merges->count : 0),
                eos && eos->is_num() ? number(eos->num()).c_str() : "(none)",
                chat && chat->type == MetaType::STRING ? "present" : "absent");
}

int inspect(const std::string& trunk, const std::string& mtp) {
    Report r;
    std::printf("strata-step35: %s\n", trunk.c_str());
    GgufModel gm = GgufModel::open(trunk);
    const MetaValue* arch = gm.meta().get("general.architecture");
    if (!arch || arch->type != MetaType::STRING) {
        std::printf("  FAILED  this file has no general.architecture\n");
        return 1;
    }
    if (arch->s != "step35") {
        std::printf("  FAILED  architecture is '%s', this engine runs 'step35'\n", arch->s.c_str());
        return 1;
    }
    Geometry g;
    if (!read_metadata(gm, r, g)) {
        std::printf("strata-step35: the metadata does not describe a model this engine can plan for (%d "
                    "problem(s) above)\n", r.bad);
        return 1;
    }
    print_geometry(g, 0);
    print_file(gm);
    check_trunk(gm, r, g);
    if (!mtp.empty()) {
        std::printf("strata-step35: %s (MTP)\n", mtp.c_str());
        GgufModel mm = GgufModel::open(mtp);
        const MetaValue* march = mm.meta().get("general.architecture");
        if (!march || march->type != MetaType::STRING || march->s != "step35") {
            r.fail("the MTP file is not a step35 file");
        } else {
            int64_t n_mtp = 0;
            check_mtp(mm, r, g, n_mtp);
            std::printf("  MTP blocks      %lld\n", (long long) n_mtp);
        }
    }
    if (r.bad) {
        std::printf("strata-step35: %d check(s) failed\n", r.bad);
        return 1;
    }
    std::printf("strata-step35: the file and its metadata agree; nothing is missing for the forward pass\n");
    return 0;
}

/// `--gpu-block`: one dense block on the GPU with the file's own weights, for one token, compared by the caller
/// against `--ref --layers <block+1>` on the same token.
int gpu_block(const std::string& path, int32_t token, int layer, int pos) {
    try {
        strata::GgufModel file = strata::GgufModel::open(path);
        if (layer < 0 || layer > 2) {
            std::printf("strata-step35: --gpu-block carries the dense blocks only (0, 1, 2)\n");
            return 2;
        }
        // the token's embedding row, dequantized here and handed to the block
        size_t shard = 0;
        const strata::TensorInfo* emb = file.find("token_embd.weight", &shard);
        if (!emb) { std::printf("strata-step35: no token_embd.weight\n"); return 1; }
        const int64_t n_embd = (int64_t) emb->shape[0];
        std::vector<float> row((size_t) n_embd);
        const size_t row_bytes = ggml_row_size((ggml_type) emb->type, n_embd);
        const auto* tr = ggml_get_type_traits((ggml_type) emb->type);
        tr->to_float((const uint8_t*) file.shard(shard).tensor_data(*emb) + (size_t) token * row_bytes, row.data(), n_embd);

        kernels::DenseBlock b;
        std::string err;
        if (!kernels::upload_block(file, layer, b, err)) {
            std::printf("strata-step35: %s\n", err.c_str());
            return 1;
        }
        float* d_h = nullptr;
        void* d_q8 = nullptr;
        cudaMalloc(&d_h, row.size() * sizeof(float));
        cudaMalloc(&d_q8, kernels::gemv_q8_bytes((int) n_embd));
        cudaMemcpy(d_h, row.data(), row.size() * sizeof(float), cudaMemcpyHostToDevice);
        const bool ok = kernels::run_dense_block(b, d_h, pos, d_q8, (int) kernels::gemv_q8_bytes((int) n_embd), err);
        if (!ok) {
            std::printf("strata-step35: %s\n", err.c_str());
            return 1;
        }
        std::vector<float> got((size_t) n_embd);
        cudaMemcpy(got.data(), d_h, got.size() * sizeof(float), cudaMemcpyDeviceToHost);
        double sum = 0, ss = 0;
        for (float x : got) { sum += (double) x; ss += (double) x * (double) x; }
        std::printf("strata-step35: gpu block %d, token %d at position %d: head %d, kv %d, rot %d, %s\n", layer,
                    token, pos, b.n_head, b.n_head_kv, b.n_rot, b.sliding ? "sliding" : "full");
        std::printf("  sum %.6f  rms %.6f  first four: %.6f %.6f %.6f %.6f\n", sum,
                    std::sqrt(ss / (double) n_embd), got[0], got[1], got[2], got[3]);
        cudaFree(d_h);
        cudaFree(d_q8);
        kernels::release_block(b);
        return 0;
    } catch (const std::exception& ex) {
        std::printf("strata-step35: %s\n", ex.what());
        return 1;
    }
}

/// `--gpu-moe`: one mixture-of-experts block on the GPU with the file's own weights, for one token.
int gpu_moe(const std::string& path, int32_t token, int layer, int pos) {
    try {
        strata::GgufModel file = strata::GgufModel::open(path);
        size_t shard = 0;
        const strata::TensorInfo* emb = file.find("token_embd.weight", &shard);
        if (!emb) { std::printf("strata-step35: no token_embd.weight\n"); return 1; }
        const int64_t n_embd = (int64_t) emb->shape[0];
        std::vector<float> row((size_t) n_embd);
        const size_t row_bytes = ggml_row_size((ggml_type) emb->type, n_embd);
        const auto* tr = ggml_get_type_traits((ggml_type) emb->type);
        tr->to_float((const uint8_t*) file.shard(shard).tensor_data(*emb) + (size_t) token * row_bytes, row.data(), n_embd);

        kernels::MoeBlock b;
        std::string err;
        if (!kernels::upload_moe_block(file, layer, b, err)) {
            std::printf("strata-step35: %s\n", err.c_str());
            return 1;
        }
        float* d_h = nullptr;
        void* d_q8 = nullptr;
        cudaMalloc(&d_h, row.size() * sizeof(float));
        cudaMalloc(&d_q8, kernels::gemv_q8_bytes((int) n_embd));
        cudaMemcpy(d_h, row.data(), row.size() * sizeof(float), cudaMemcpyHostToDevice);
        if (!kernels::run_moe_block(b, d_h, pos, d_q8, (int) kernels::gemv_q8_bytes((int) n_embd), err)) {
            std::printf("strata-step35: %s\n", err.c_str());
            return 1;
        }
        std::vector<float> got((size_t) n_embd);
        cudaMemcpy(got.data(), d_h, got.size() * sizeof(float), cudaMemcpyDeviceToHost);
        double sum = 0, ss = 0;
        for (float x : got) { sum += (double) x; ss += (double) x * (double) x; }
        std::printf("strata-step35: gpu MoE block %d, token %d at position %d: %d experts, top %d, %s, scale %.1f\n",
                    layer, token, pos, b.n_expert, b.top_k, b.attn.sliding ? "sliding" : "full", (double) b.scale);
        std::printf("  sum %.6f  rms %.6f  first four: %.6f %.6f %.6f %.6f\n", sum,
                    std::sqrt(ss / (double) n_embd), got[0], got[1], got[2], got[3]);
        cudaFree(d_h);
        cudaFree(d_q8);
        kernels::release_moe_block(b);
        return 0;
    } catch (const std::exception& ex) {
        std::printf("strata-step35: %s\n", ex.what());
        return 1;
    }
}

/// `--gpu-chain`: the first N blocks on the GPU, one after the other, from the token's embedding.  This is the
/// only honest way to compare a block above the first: block 3 works on what block 2 handed it, not on the
/// embedding, so comparing it alone against the reference compares two different inputs.
int gpu_chain(const std::string& path, int32_t token, int n_layers, int pos) {
    try {
        strata::GgufModel file = strata::GgufModel::open(path);
        size_t shard = 0;
        const strata::TensorInfo* emb = file.find("token_embd.weight", &shard);
        if (!emb) { std::printf("strata-step35: no token_embd.weight\n"); return 1; }
        const int64_t n_embd = (int64_t) emb->shape[0];
        const int64_t dense_layers = (int64_t) file.meta().get("step35.leading_dense_block_count")->num();
        std::vector<float> row((size_t) n_embd);
        const size_t row_bytes = ggml_row_size((ggml_type) emb->type, n_embd);
        const auto* tr = ggml_get_type_traits((ggml_type) emb->type);
        tr->to_float((const uint8_t*) file.shard(shard).tensor_data(*emb) + (size_t) token * row_bytes, row.data(), n_embd);

        float* d_h = nullptr;
        void* d_q8 = nullptr;
        cudaMalloc(&d_h, row.size() * sizeof(float));
        cudaMalloc(&d_q8, kernels::gemv_q8_bytes((int) n_embd));
        cudaMemcpy(d_h, row.data(), row.size() * sizeof(float), cudaMemcpyHostToDevice);
        for (int layer = 0; layer < n_layers; ++layer) {
            std::string err;
            std::vector<float> check((size_t) n_embd);
            const bool dense = layer < (int) dense_layers;
            if (dense) {
                kernels::DenseBlock b;
                if (!kernels::upload_block(file, layer, b, err) ||
                    !kernels::run_dense_block(b, d_h, pos, d_q8, (int) kernels::gemv_q8_bytes((int) n_embd), err)) {
                    std::printf("strata-step35: block %d: %s\n", layer, err.c_str());
                    return 1;
                }
                kernels::release_block(b);
            } else {
                kernels::MoeBlock b;
                if (!kernels::upload_moe_block(file, layer, b, err) ||
                    !kernels::run_moe_block(b, d_h, pos, d_q8, (int) kernels::gemv_q8_bytes((int) n_embd), err)) {
                    std::printf("strata-step35: block %d: %s\n", layer, err.c_str());
                    return 1;
                }
                kernels::release_moe_block(b);
            }
            cudaMemcpy(check.data(), d_h, check.size() * sizeof(float), cudaMemcpyDeviceToHost);
            double sum = 0;
            for (float x : check) sum += (double) x;
            std::printf("  block %2d (%s) : sum %.6f\n", layer, dense ? "dense" : "MoE", sum);
        }
        cudaFree(d_h);
        cudaFree(d_q8);
        return 0;
    } catch (const std::exception& ex) {
        std::printf("strata-step35: %s\n", ex.what());
        return 1;
    }
}

/// `--gpu-all`: every block, experts streamed out of the file, one token.  The first time the whole model runs.
int gpu_all(const std::string& path, int32_t token, int pos) {
    try {
        strata::GgufModel file = strata::GgufModel::open(path);
        size_t shard = 0;
        const strata::TensorInfo* emb = file.find("token_embd.weight", &shard);
        const int64_t n_embd = (int64_t) emb->shape[0];
        const int64_t n_layers = (int64_t) file.meta().get("step35.block_count")->num();
        const int64_t dense_layers = (int64_t) file.meta().get("step35.leading_dense_block_count")->num();
        std::vector<float> row((size_t) n_embd);
        const size_t row_bytes = ggml_row_size((ggml_type) emb->type, n_embd);
        const auto* tr = ggml_get_type_traits((ggml_type) emb->type);
        tr->to_float((const uint8_t*) file.shard(shard).tensor_data(*emb) + (size_t) token * row_bytes, row.data(),
                     n_embd);
        float* d_h = nullptr;
        void* d_q8 = nullptr;
        cudaMalloc(&d_h, row.size() * sizeof(float));
        cudaMalloc(&d_q8, kernels::gemv_q8_bytes((int) n_embd));
        cudaMemcpy(d_h, row.data(), row.size() * sizeof(float), cudaMemcpyHostToDevice);
        for (int layer = 0; layer < (int) n_layers; ++layer) {
            std::string err;
            const bool dense = layer < (int) dense_layers;
            if (dense) {
                kernels::DenseBlock b;
                if (!kernels::upload_block(file, layer, b, err) ||
                    !kernels::run_dense_block(b, d_h, pos, d_q8, (int) kernels::gemv_q8_bytes((int) n_embd), err)) {
                    std::printf("strata-step35: block %d: %s\n", layer, err.c_str());
                    return 1;
                }
                kernels::release_block(b);
            } else {
                if (!kernels::run_stream_block(file, layer, d_h, pos, d_q8, err)) {
                    std::printf("strata-step35: block %d: %s\n", layer, err.c_str());
                    return 1;
                }
            }
            std::vector<float> check((size_t) n_embd);
            cudaMemcpy(check.data(), d_h, check.size() * sizeof(float), cudaMemcpyDeviceToHost);
            double sum = 0;
            for (float x : check) sum += (double) x;
            if (layer < 4 || layer + 1 == (int) n_layers) std::printf("  block %2d (%s) : sum %.6f\n", layer, dense ? "dense" : "MoE streamed", sum);
        }
        std::vector<float> got((size_t) n_embd);
        cudaMemcpy(got.data(), d_h, got.size() * sizeof(float), cudaMemcpyDeviceToHost);
        double sum = 0, ss = 0;
        for (float x : got) { sum += (double) x; ss += (double) x * (double) x; }
        std::printf("strata-step35: %lld blocks streamed, token %d at position %d: sum %.6f rms %.6f\n",
                    (long long) n_layers, token, pos, sum, std::sqrt(ss / (double) n_embd));
        cudaFree(d_h);
        cudaFree(d_q8);
        return 0;
    } catch (const std::exception& ex) {
        std::printf("strata-step35: %s\n", ex.what());
        return 1;
    }
}

/// The highest-logit token: what a draft head proposes, and what the next head is fed.
int argmax(const std::vector<float>& lg) {
    int best = 0;
    for (int i = 1; i < (int) lg.size(); ++i)
        if (lg[(size_t) i] > lg[(size_t) best]) best = i;
    return best;
}

/// The VRAM-resident expert tier, sized from a MiB budget and the file's own expert shapes.  The shapes come from
/// block 3 (the first mixture-of-experts block) because that is where the tensors exist.
bool open_tier(const strata::GgufModel& file, int mib, kernels::ExpertCache& tier, std::string& err,
               std::FILE* out = stdout, const std::string& policy = "lru") {
    if (mib <= 0) return true;
    const int dense = (int) file.meta().get("step35.leading_dense_block_count")->num();
    size_t shard = 0;
    const strata::TensorInfo* tg = file.find("blk." + std::to_string(dense) + ".ffn_gate_exps.weight", &shard);
    const strata::TensorInfo* td = file.find("blk." + std::to_string(dense) + ".ffn_down_exps.weight", &shard);
    if (!tg || !td) {
        err = "cannot size the expert tier: no expert tensors";
        return false;
    }
    const int n_expert = (int) file.meta().get("step35.expert_count")->num();
    const size_t gate_bytes = strata::tensor_payload_bytes(*tg) / (size_t) n_expert;
    const size_t down_bytes = strata::tensor_payload_bytes(*td) / (size_t) n_expert;
    const size_t per_expert = 2 * gate_bytes + down_bytes;
    const int n_slots = (int) ((size_t) mib * 1024 * 1024 / per_expert);
    if (n_slots <= 0) {
        err = "the expert tier budget is smaller than one expert (" + std::to_string(per_expert / 1024) + " KB)";
        return false;
    }
    const int asked = n_slots;
    if (!kernels::expert_cache_open(tier, n_slots, gate_bytes, down_bytes, err)) return false;
    if (!kernels::expert_cache_policy(tier, policy)) {
        err = "unknown expert tier policy " + policy + " (fifo, lru or lfu)";
        return false;
    }
    std::fprintf(out, "strata-step35: expert tier: %d slot(s) of %zu KB = %.2f GB of VRAM, policy %s\n", tier.n_slots,
                 per_expert / 1024, (double) tier.n_slots * (double) per_expert / 1e9, policy.c_str());
    if (tier.n_slots < asked)   // the allocation stopped early and said why: fewer slots is still a tier
        std::fprintf(out, "strata-step35: expert tier: %s (asked for %d slot(s))\n", err.c_str(), asked);
    err.clear();
    return true;
}

/// "auto" for --expert-cache and --dense-cache: what the card has left once the K/V cache and a reserve for the head
/// and the working buffers are out of the way.  The blocks come first - 1.9 GB buys the larger half of a token - and the
/// tier takes the rest.  `tier_mib` / `dense_mib` are -1 for auto on the command line; both are replaced in place.
void auto_budgets(const strata::GgufModel& file, int n_blocks, int dense_layers, const kernels::Session& session,
                  int& tier_mib, int& dense_mib, bool kv_host, std::FILE* out = stdout) {
    if (tier_mib >= 0 && dense_mib >= 0) return;
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess || free_b == 0) return;
    // The head is 363 MB, the working buffers (q/k/v, the q8_1 scratch, an expert's three matrices when one is
    // streamed) another ~250 MB with the dense cache in place, and the residency tables and step buffers grow with the
    // context.  1.5 GB is what keeps the LAST allocation of a load from failing: measured, 1 GB let the head fail.
    const uint64_t reserve = 3ull << 29;
    // a cache in pinned host memory costs the card nothing: that VRAM goes to the experts instead
    const uint64_t kv = kv_host ? 0 : kernels::session_bytes_at(session, session.capacity);
    const uint64_t room = free_b > kv + reserve ? (uint64_t) free_b - kv - reserve : 0;
    // the blocks AND their routers and shared experts: leaving the latter out is what made auto stream them for ever
    const uint64_t dense_need = kernels::dense_layers_bytes(file, n_blocks, dense_layers) +
                                kernels::moe_small_total(file, n_blocks, dense_layers);
    // rounded UP: a budget a byte short of the blocks streams the last one for ever
    if (dense_mib < 0) dense_mib = (int) ((std::min(dense_need, room) + (1u << 20) - 1) >> 20);
    // what the dense cache will really take: its budget if the person set one, the blocks' size if it is auto
    const uint64_t dense_take = std::min<uint64_t>(dense_mib >= 0 ? ((uint64_t) dense_mib << 20) : dense_need, room);
    if (tier_mib < 0) tier_mib = (int) ((room - dense_take) >> 20);
    std::fprintf(out, "strata-step35: auto budgets: %.2f GB free, %.2f GB of K/V, %.1f GB reserved -> dense %d MiB, "
                      "tier %d MiB\n",
                 (double) free_b / 1e9, (double) kv / 1e9, (double) reserve / 1e9, dense_mib, tier_mib);
}

/// The K/V bytes the layers [from, to) allocate: what one card's share of the cache really costs.
uint64_t kv_bytes_of_layers(const kernels::Session& s, int from, int to) {
    // the cell's bytes, from the session's own geometry (what session_bytes_at sums, one layer at a time)
    const uint64_t cell = (uint64_t) s.n_head_kv * s.head_dim * 2 +
                          (uint64_t) s.n_head_kv * (s.head_dim / 64) * 2 * 2;   // 64 = kv_q8.hpp's KV_Q8_GROUP
    uint64_t total = 0;
    for (int l = from; l < to; ++l) total += (uint64_t) kernels::session_layer_cells(s, l) * cell;
    return total;
}

/// auto_budgets, per card: `from`..`to` are the layers this card runs, so its K/V share and its blocks are what have to
/// fit.  Prints on the caller's stream like auto_budgets does.
void auto_budgets_range(const strata::GgufModel& file, int n_blocks, int dense_layers, const kernels::Session& session,
                        int from, int to, int& tier_mib, int& dense_mib, bool kv_host, std::FILE* out,
                        uint64_t extra_reserve = 0) {
    if (tier_mib >= 0 && dense_mib >= 0) return;
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess || free_b == 0) return;
    // The engine's own buffers, plus what the CALLER says to leave for everything else: --vram-reserve-mib, the flag
    // the rest of Strata already passes and this engine used to ignore - the desktop's own compositor lives in that
    // difference, and a compositor with no VRAM loses its context (Mesa's DeviceReset) and takes the session with it.
    const uint64_t reserve = (3ull << 29) + extra_reserve;
    const uint64_t kv = kv_host ? 0 : kv_bytes_of_layers(session, from, to);
    const uint64_t room = free_b > kv + reserve ? (uint64_t) free_b - kv - reserve : 0;
    const uint64_t dense_need = (kernels::dense_layers_bytes(file, to, dense_layers) -
                                 kernels::dense_layers_bytes(file, from, dense_layers)) +
                                (kernels::moe_small_total(file, to, dense_layers) -
                                 kernels::moe_small_total(file, from, dense_layers));
    if (dense_mib < 0) dense_mib = (int) ((std::min(dense_need, room) + (1u << 20) - 1) >> 20);
    const uint64_t dense_take = std::min<uint64_t>(dense_mib >= 0 ? ((uint64_t) dense_mib << 20) : dense_need, room);
    if (tier_mib < 0) tier_mib = (int) ((room - dense_take) >> 20);
    std::fprintf(out, "strata-step35: auto budgets (layers %d-%d): %.2f GB free, %.2f GB of K/V, %.1f GB reserved -> "
                      "dense %d MiB, tier %d MiB\n",
                 from, to - 1, (double) free_b / 1e9, (double) kv / 1e9, (double) reserve / 1e9, dense_mib, tier_mib);
}

/// The blocks' dense weights, resident: sized like the tier (MiB on the command line), and reported with what it kept.
bool open_dense_cache(const strata::GgufModel& file, int mib, int n_layers, kernels::DenseCache& cache, std::string& err) {
    if (mib <= 0) return true;
    if (!kernels::dense_cache_open(cache, file, n_layers, (size_t) mib * 1024 * 1024, err)) return false;
    std::printf("strata-step35: dense block cache: up to %d MiB\n", mib);
    return true;
}

void print_dense_cache(const kernels::DenseCache& cache) {
    if (cache.budget == 0) return;
    std::printf("strata-step35: dense block cache: %d block(s) resident (%.2f GB), %d streamed\n", cache.kept,
                (double) cache.used / 1e9, cache.streamed);
}

/// The level BELOW the tier: pinned host memory, expert-granular, filled from the file on its own misses.  A miss in
/// the tier then copies from RAM instead of from the page cache's coin toss over a 75 GB file.
/// What the machine can lend right now, in bytes (0 when it cannot be read): MemAvailable on Linux, which counts what
/// a new allocation can take without swapping - `free` alone does not.
uint64_t available_ram_bytes() {
#if defined(__linux__)
    std::FILE* f = std::fopen("/proc/meminfo", "r");
    if (f == nullptr) return 0;
    char key[64] = {0};
    unsigned long long value = 0;
    char unit[16] = {0};
    uint64_t found = 0;
    while (std::fscanf(f, "%63s %llu %15s", key, &value, unit) == 3)
        if (std::strcmp(key, "MemAvailable:") == 0) {
            found = (uint64_t) value * 1024ull;
            break;
        }
    std::fclose(f);
    return found;
#else
    return 0;
#endif
}

bool open_host_cache(const strata::GgufModel& file, int mib, int n_blocks, kernels::ExpertCache& cache, std::string& err,
                     std::FILE* out) {
    if (mib == -1) {                                  // auto: a quarter of what the machine can lend now, at most 8 GiB
        const uint64_t avail = available_ram_bytes();
        mib = avail ? (int) (std::min<uint64_t>(avail / 4, 8ull << 30) / (1u << 20)) : 2048;
        std::fprintf(out, "strata-step35: host cache auto: %d MiB of %llu MiB available\n", mib,
                     (unsigned long long) (avail / (1u << 20)));
    }
    if (mib <= 0) return true;
    const int dense = (int) file.meta().get("step35.leading_dense_block_count")->num();
    size_t shard = 0;
    const strata::TensorInfo* tg = file.find("blk." + std::to_string(dense) + ".ffn_gate_exps.weight", &shard);
    const strata::TensorInfo* td = file.find("blk." + std::to_string(dense) + ".ffn_down_exps.weight", &shard);
    if (!tg || !td) {
        err = "cannot size the host cache: no expert tensors";
        return false;
    }
    const int n_expert = (int) file.meta().get("step35.expert_count")->num();
    const size_t gate_bytes = strata::tensor_payload_bytes(*tg) / (size_t) n_expert;
    const size_t down_bytes = strata::tensor_payload_bytes(*td) / (size_t) n_expert;
    const size_t per_expert = 2 * gate_bytes + down_bytes;
    int n_slots = (int) ((size_t) mib * 1024 * 1024 / per_expert);
    // Pinned memory is not swapped: asking for more than half of what the machine can lend does not slow the model
    // down, it slows the DESKTOP down until it dies - measured here, 33.6 GB pinned on 62 GB took Cinnamon with it.
    const uint64_t avail = available_ram_bytes();
    uint64_t want = (uint64_t) n_slots * per_expert;
    // Ceiling: a quarter of what the machine can lend, and never more than 8 GiB.  Pageable memory can be swapped,
    // but taking it still shrinks the page cache the experts come from - and a desktop that has to swap is a desktop
    // that stutters.
    const uint64_t cap = avail > 0 ? std::min<uint64_t>(avail / 4, 8ull << 30) : 0;
    if (cap > 0 && want > cap) {
        n_slots = (int) (cap / per_expert);
        want = (uint64_t) n_slots * per_expert;
        std::fprintf(out, "strata-step35: host cache: %d MiB asked, clamped to %d MiB (a quarter of the %llu MiB the "
                          "machine can lend, ceiling 8 GiB)\n",
                     mib, (int) ((uint64_t) n_slots * per_expert / (1u << 20)),
                     (unsigned long long) (avail / (1u << 20)));
        mib = (int) ((uint64_t) n_slots * per_expert / (1u << 20));
    }
    if (n_slots <= 0) return true;
    if (!kernels::expert_cache_open(cache, n_slots, gate_bytes, down_bytes, err, true)) return false;
    std::fprintf(out, "strata-step35: host expert cache: %d slot(s) of %zu KB = %.1f GB of RAM\n", cache.n_slots,
                 per_expert / 1024, (double) cache.n_slots * (double) per_expert / 1e9);
    (void) n_blocks;
    return true;
}

/// What the tier bought, and what it did not: a hit rate with no time next to it would say nothing.
void print_tier(const kernels::ExpertCache& tier, double seconds) {
    if (tier.n_slots <= 0) return;
    const double n = (double) (tier.hits + tier.misses);
    std::printf("strata-step35: expert tier: %llu hit(s) of %llu fetch(es) (%.1f%%), %llu eviction(s), %.3f s\n",
                (unsigned long long) tier.hits, (unsigned long long) (tier.hits + tier.misses),
                n > 0 ? 100.0 * (double) tier.hits / n : 0.0, (unsigned long long) tier.evictions, seconds);
}

/// The five tokens of highest logit, in exactly the shape `--ref` prints them: the figure the CPU reference and the
/// GPU paths are compared on.  A head that is merely mis-scaled still answers with plausible logits, so the ranking
/// and the values have to come out together.
void print_top5(const std::vector<float>& lg, const std::string& label) {
    std::vector<int> top((size_t) 5, 0);
    for (int32_t id = 0; id < (int32_t) lg.size(); ++id) {
        for (int k = 0; k < 5; ++k) {
            if (lg[(size_t) id] > lg[(size_t) top[(size_t) k]]) {
                for (int j = 4; j > k; --j) top[(size_t) j] = top[(size_t) (j - 1)];
                top[(size_t) k] = id;
                break;
            }
        }
    }
    std::printf("  next-token top 5 %s:", label.c_str());
    for (int k = 0; k < 5; ++k) std::printf(" %d(%.3f)", top[(size_t) k], lg[(size_t) top[(size_t) k]]);
    std::printf("\n");
}

/// `--gpu-seq`: a PROMPT of several positions through the dense blocks on the GPU.
int gpu_seq(const std::string& path, const std::string& tokens_csv, int n_blocks, int pos, bool want_logits) {
    try {
        strata::GgufModel file = strata::GgufModel::open(path);
        std::vector<int32_t> ids;
        for (size_t i = 0; i < tokens_csv.size();) {
            size_t j = tokens_csv.find(',', i);
            if (j == std::string::npos) j = tokens_csv.size();
            ids.push_back((int32_t) std::strtol(tokens_csv.substr(i, j - i).c_str(), nullptr, 10));
            i = j + 1;
        }
        if (ids.empty()) {
            std::printf("strata-step35: --gpu-seq needs --tokens <id,id,...>\n");
            return 2;
        }
        // `--layers` omitted is 0, and the help says the prompt goes through every block: a pass over zero blocks
        // prints an empty result and looks like a model that answers nothing.
        const int n_layers_total = (int) file.meta().get("step35.block_count")->num();
        if (n_blocks <= 0 || n_blocks > n_layers_total) n_blocks = n_layers_total;
        std::vector<float> out;
        std::vector<double> sums;
        std::string err;
        if (!kernels::run_prompt_dense(file, ids, pos, n_blocks, out, sums, err)) {
            std::printf("strata-step35: %s\n", err.c_str());
            return 1;
        }
        std::printf("strata-step35: %zu position(s) through %d block(s) on the GPU (dense, then mixture of experts)\n", ids.size(), n_blocks);
        for (size_t l = 0; l < sums.size(); ++l) std::printf("  block %2zu : sum %.6f\n", l, sums[l]);
        // the same per-position figures `--ref` prints, from the fp32 K/V of this pass: the line the cache's own
        // `--gpu-chat` is compared against, position by position
        const int n_embd = out.empty() ? 0 : (int) (out.size() / ids.size());
        for (size_t p = 0; n_embd > 0 && p < ids.size(); ++p) {
            const float* h = out.data() + p * (size_t) n_embd;
            double ss = 0;
            for (int i = 0; i < n_embd; ++i) ss += (double) h[i] * (double) h[i];
            std::printf("  position %zu: rms %.6f, first four:", p, std::sqrt(ss / (double) n_embd));
            for (int i = 0; i < 4; ++i) std::printf(" %.6f", h[i]);
            std::printf("\n");
        }
        if (want_logits && n_embd > 0) {
            std::vector<float> lg;
            if (!kernels::logits_last(file, out.data() + (ids.size() - 1) * (size_t) n_embd, lg, err)) {
                std::printf("strata-step35: %s\n", err.c_str());
                return 1;
            }
            print_top5(lg, "after " + std::to_string(n_blocks) + " block(s) on the GPU");
        }
        return 0;
    } catch (const std::exception& ex) {
        std::printf("strata-step35: %s\n", ex.what());
        return 1;
    }
}

/// `--gpu-mtp`: the trunk over a prompt, greedy steps of its own to have something to check the draft against, then
/// the NextN blocks as a chain of draft heads.  A draft is worth something only if it proposes what the trunk would
/// have said, so each head is fed the token the trunk itself chose at that position and its top-1 is compared with
/// the token the trunk chose next.
int gpu_mtp(const std::string& path, const std::string& mtp_path, const std::string& tokens_csv, int n_blocks,
            int drafts, int tier_mib, int dense_mib, const std::string& tier_policy) {
    try {
        strata::GgufModel file = strata::GgufModel::open(path);
        strata::GgufModel mtp = strata::GgufModel::open(mtp_path);
        std::vector<int32_t> prompt;
        for (size_t i = 0; i < tokens_csv.size();) {
            size_t j = tokens_csv.find(',', i);
            if (j == std::string::npos) j = tokens_csv.size();
            prompt.push_back((int32_t) std::strtol(tokens_csv.substr(i, j - i).c_str(), nullptr, 10));
            i = j + 1;
        }
        if (prompt.empty()) {
            std::printf("strata-step35: --gpu-mtp needs --tokens <id,id,...>\n");
            return 2;
        }
        const int n_layers_total = (int) file.meta().get("step35.block_count")->num();
        if (n_blocks <= 0 || n_blocks > n_layers_total) n_blocks = n_layers_total;
        const int n_embd = (int) file.meta().get("step35.embedding_length")->num();
        const int n_mtp = (int) mtp.meta().get("step35.nextn_predict_layers")->num();
        const int first = (int) mtp.meta().get("step35.block_count")->num() - n_mtp;
        if (n_mtp <= 0) {
            std::printf("strata-step35: %s declares no NextN block\n", mtp_path.c_str());
            return 1;
        }
        const int want = std::min(drafts, n_mtp);
        const int n_kv = (int) file.meta().get("step35.attention.head_count_kv")->items[0].num();
        const int head_dim = (int) file.meta().get("step35.attention.key_length")->num();
        const strata::MetaValue* swa = file.meta().get("step35.attention.sliding_window_pattern");
        const strata::MetaValue* wv = file.meta().get("step35.attention.sliding_window");
        const int window = wv ? (int) wv->num() : 512;
        std::vector<int> windows((size_t) n_blocks, 0);
        for (int l = 0; l < n_blocks; ++l)
            if (swa && swa->type == strata::MetaType::ARRAY && (size_t) l < swa->items.size() &&
                swa->items[(size_t) l].num() != 0)
                windows[(size_t) l] = window;

        // the trunk, over the prompt, with its cache: the same pass `--gpu-chat` makes
        std::string err;
        kernels::ExpertCache tier;
        if (!open_tier(file, tier_mib, tier, err, stdout, tier_policy)) {
            std::printf("strata-step35: %s\n", err.c_str());
            return 1;
        }
        kernels::DenseCache dense;
        if (!open_dense_cache(file, dense_mib, n_blocks, dense, err)) {
            std::printf("strata-step35: %s\n", err.c_str());
            return 1;
        }
        kernels::DenseCache* densep = dense.budget > 0 ? &dense : nullptr;
        kernels::ExpertCache* tierp = tier.n_slots > 0 ? &tier : nullptr;
        kernels::Session trunk;
        const int capacity = (int) prompt.size() + want + 2;
        if (!kernels::session_create(trunk, n_blocks, n_kv, head_dim, capacity, err, windows)) {
            std::printf("strata-step35: %s\n", err.c_str());
            return 1;
        }
        std::vector<float> out;
        std::vector<double> sums;
        if (!kernels::run_prompt_dense(file, prompt, 0, n_blocks, out, sums, err, &trunk, tierp,
                                       dense.budget > 0 ? &dense : nullptr)) {
            std::printf("strata-step35: %s\n", err.c_str());
            kernels::expert_cache_close(tier);
            kernels::session_destroy(trunk);
            return 1;
        }
        // the trunk's own continuation, one greedy token at a time: the figure every draft is held to
        std::vector<std::vector<float>> hidden;                       // hidden[k]: the trunk at prompt_len-1+k
        std::vector<int> tok;                                         // tok[k]: the token it chose there
        hidden.push_back(std::vector<float>(out.end() - n_embd, out.end()));
        for (int k = 0; k <= want; ++k) {
            std::vector<float> lg;
            if (!kernels::logits_last(file, hidden[(size_t) k].data(), lg, err)) {
                std::printf("strata-step35: %s\n", err.c_str());
                kernels::session_destroy(trunk);
                return 1;
            }
            tok.push_back(argmax(lg));
            if (k == want) break;
            const std::vector<int32_t> one(1, tok[(size_t) k]);
            std::vector<float> dout;
            std::vector<double> dsums;
            if (!kernels::run_prompt_dense(file, one, (int) prompt.size() + k, n_blocks, dout, dsums, err, &trunk,
                                           tierp, dense.budget > 0 ? &dense : nullptr)) {
                std::printf("strata-step35: %s\n", err.c_str());
                kernels::session_destroy(trunk);
                return 1;
            }
            hidden.push_back(std::vector<float>(dout.end() - n_embd, dout.end()));
        }
        std::printf("strata-step35: %zu token(s) through %d block(s); the trunk continues", prompt.size(), n_blocks);
        for (int k = 0; k <= want; ++k) std::printf(" %d", tok[(size_t) k]);
        std::printf("\n");

        // the NextN blocks and their heads, resident: a draft that re-uploads them costs four trunk steps
        kernels::MtpContext dctx;
        if (!kernels::mtp_open(mtp, dctx, err)) {
            std::printf("strata-step35: %s\n", err.c_str());
            kernels::session_destroy(trunk);
            return 1;
        }
        const auto t_draft0 = std::chrono::steady_clock::now();
        // the draft cache: one cell per NextN layer (mtp_step says why)
        std::vector<int> kwindows((size_t) n_mtp, 0);
        kernels::Session draft;
        if (!kernels::session_create(draft, n_mtp, n_kv, head_dim, want, err, kwindows)) {
            std::printf("strata-step35: %s\n", err.c_str());
            kernels::session_destroy(trunk);
            return 1;
        }
        int agree = 0;
        std::vector<float> hh = hidden[0];
        for (int k = 0; k < want; ++k) {
            std::vector<float> hn, lg;
            if (!kernels::mtp_step(dctx, k, k, (int) prompt.size() + k, tok[(size_t) k], hh.data(), &draft, hn, lg,
                                   err)) {
                std::printf("strata-step35: %s\n", err.c_str());
                kernels::mtp_close(dctx);
                kernels::session_destroy(draft);
                kernels::session_destroy(trunk);
                return 1;
            }
            const int d = argmax(lg);
            const int t = tok[(size_t) (k + 1)];
            if (d == t) ++agree;
            double ss = 0;
            for (float v : hn) ss += (double) v * (double) v;
            std::printf("  draft %d: block %d at position %d, fed token %d, h_nextn rms %.6f -> %d %s (trunk: %d)\n",
                        k, first + k, (int) prompt.size() + k, tok[(size_t) k], std::sqrt(ss / (double) n_embd), d,
                        d == t ? "AGREES" : "differs", t);
            print_top5(lg, "draft " + std::to_string(k));
            hh = hn;
        }
        const double draft_s =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t_draft0).count();
        kernels::mtp_close(dctx);
        kernels::session_destroy(draft);
        kernels::session_destroy(trunk);
        print_tier(tier, 0.0);
        print_dense_cache(dense);
        kernels::expert_cache_close(tier);
        kernels::dense_cache_close(dense);
        std::printf("strata-step35: %d draft(s) in %.3f s (%.3f s/draft)\n", want, draft_s,
                    want ? draft_s / (double) want : 0.0);
        std::printf("strata-step35: the draft agreed with the trunk on %d of %d token(s)\n", agree, want);
        return 0;
    } catch (const std::exception& ex) {
        std::printf("strata-step35: %s\n", ex.what());
        return 1;
    }
}

/// A strata-vision file: records of `int32 magic('SVE1'), n, nx, ny, n_embd`, then `n * n_embd` floats, in prompt
/// order.  The same format `strata-vision` writes for every family; read here, never written.
bool read_embeddings(const std::string& path, int n_embd, std::vector<float>& rows, int& n_records, std::string& err) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
        err = "cannot open the image embeddings file " + path;
        return false;
    }
    rows.clear();
    n_records = 0;
    std::vector<int32_t> hdr(5);
    while (true) {
        const size_t got = std::fread(hdr.data(), sizeof(int32_t), 5, f);
        if (got == 0) break;
        if (got != 5 || hdr[0] != 0x31455653 || hdr[1] < 1 || hdr[2] < 1 || hdr[3] < 1 ||
            (int64_t) hdr[2] * hdr[3] != hdr[1] || hdr[4] != n_embd) {
            std::fclose(f);
            err = "bad image embeddings file: expected SVE1 records of width " + std::to_string(n_embd);
            return false;
        }
        const size_t count = (size_t) hdr[1] * (size_t) hdr[4];
        const size_t at = rows.size();
        rows.resize(at + count);
        if (std::fread(rows.data() + at, sizeof(float), count, f) != count) {
            std::fclose(f);
            err = "the image embeddings file is short";
            return false;
        }
        ++n_records;
    }
    std::fclose(f);
    return true;
}

/// The id of the prompt's picture token.  It cannot be looked up in the file: the reader keeps a SAMPLE of the first
/// 64 entries of `tokenizer.ggml.tokens`, and `<im_patch>` is 128001 in this release.  So it is a number with a flag
/// over it (`--image-token`, default 128001 - the id printed by reading the vocabulary out of band), the way the other
/// engines of this Terminal carry their own pad token as a constant.
constexpr int kStep35ImageToken = 128001;

/// `--dump-embeddings <file> --tokens <ids>`: this token sequence's own embedding rows, written as an SVE file - the
/// same format `strata-vision` writes and `GENI` reads.  It exists so that feeding a prompt's own rows back through the
/// picture path must answer exactly what the text path answers: the check that the injection lands where it should.
int dump_embeddings(const std::string& path, const std::string& tokens_csv, const std::string& out) {
    try {
        strata::GgufModel file = strata::GgufModel::open(path);
        const int n_embd = (int) file.meta().get("step35.embedding_length")->num();
        std::vector<int32_t> ids;
        for (size_t i = 0; i < tokens_csv.size();) {
            size_t j = tokens_csv.find(',', i);
            if (j == std::string::npos) j = tokens_csv.size();
            ids.push_back((int32_t) std::strtol(tokens_csv.substr(i, j - i).c_str(), nullptr, 10));
            i = j + 1;
        }
        if (ids.empty() || out.empty()) {
            std::printf("strata-step35: --dump-embeddings needs --tokens <ids> and a file to write\n");
            return 2;
        }
        std::vector<float> rows;
        std::string err;
        if (!kernels::embed_rows(file, ids, n_embd, rows, err)) {
            std::printf("strata-step35: %s\n", err.c_str());
            return 1;
        }
        std::FILE* f = std::fopen(out.c_str(), "wb");
        if (f == nullptr) {
            std::printf("strata-step35: cannot write %s\n", out.c_str());
            return 1;
        }
        for (size_t i = 0; i < ids.size(); ++i) {
            const int32_t hdr[5] = {0x31455653, 1, 1, 1, n_embd};
            std::fwrite(hdr, sizeof(int32_t), 5, f);
            std::fwrite(rows.data() + i * (size_t) n_embd, sizeof(float), (size_t) n_embd, f);
        }
        std::fclose(f);
        std::printf("strata-step35: wrote %zu row(s) of %d to %s\n", ids.size(), n_embd, out.c_str());
        return 0;
    } catch (const std::exception& ex) {
        std::printf("strata-step35: %s\n", ex.what());
        return 1;
    }
}

/// `--serve`: the resident engine `serve/server.py` talks to.  The protocol is its own (`READY <ctx> stop`,
/// `GEN <n> [key=value ...] <ids>`, `T <id>` per token, `PP <read> <total> <ms>` while a prompt is read, `DONE`,
/// `STOP` to end a request early, `QUIT` to end the process), and this function speaks exactly that and nothing
/// more: the weights, the cache, the head and the expert tier stay loaded between requests, and a request whose
/// prompt extends the one before decodes only what is new.
int serve(const std::string& path, const std::string& mtp_path, int max_ctx, int tier_mib, int dense_mib,
          const std::string& tier_policy, bool kv_host, int image_token, int layer_split, int host_mib,
          int vram_reserve_mib, int vram_reserve_later_mib, int prefill_chunk) {
    (void)mtp_path;   // drafting in a request is not wired: the protocol has no field for it
    try {
        strata::GgufModel file = strata::GgufModel::open(path);
        std::string err;
        const int n_blocks = (int) file.meta().get("step35.block_count")->num();
        const int n_embd = (int) file.meta().get("step35.embedding_length")->num();
        const int n_kv = (int) file.meta().get("step35.attention.head_count_kv")->items[0].num();
        const int head_dim = (int) file.meta().get("step35.attention.key_length")->num();
        const strata::MetaValue* swa = file.meta().get("step35.attention.sliding_window_pattern");
        const strata::MetaValue* wv = file.meta().get("step35.attention.sliding_window");
        const int window = wv ? (int) wv->num() : 512;
        const strata::MetaValue* eos_v = file.meta().get("tokenizer.ggml.eos_token_id");
        const int eos = eos_v ? (int) eos_v->num() : -1;
        std::vector<int> windows((size_t) n_blocks, 0);
        for (int l = 0; l < n_blocks; ++l)
            if (swa && swa->type == strata::MetaType::ARRAY && (size_t) l < swa->items.size() &&
                swa->items[(size_t) l].num() != 0)
                windows[(size_t) l] = window;

        kernels::Session trunk;
        if (!kernels::session_create(trunk, n_blocks, n_kv, head_dim, max_ctx, err, windows, kv_host)) {
            std::printf("ERR %s\n", err.c_str());
            std::fflush(stdout);
            return 1;
        }
        // "auto" is resolved HERE, as on the command-line paths: before this it reached the caches as -1, which they
        // read as "off" - a served model ran with no dense cache and no expert tier, every expert from the file.
        // A layer split gives each card its own share, sized from its own free VRAM.
        const int split = (layer_split != 0 && n_blocks > 1)
                              ? (layer_split < 0 ? n_blocks / 2 : (int) std::min<int64_t>(layer_split, n_blocks - 1))
                              : 0;
        const int dense_layers = (int) file.meta().get("step35.leading_dense_block_count")->num();
        kernels::ExpertCache tier[2];
        kernels::DenseCache dense[2];
        int tier_mib_by_dev[2] = {(int) tier_mib, (int) tier_mib};
        int dense_mib_by_dev[2] = {(int) dense_mib, (int) dense_mib};
        for (int d = 0; d < (split > 0 ? 2 : 1); ++d) {
            cudaSetDevice(d);
            const int from = d == 0 ? 0 : split, to = d == 0 ? (split > 0 ? split : n_blocks) : n_blocks;
            auto_budgets_range(file, n_blocks, dense_layers, trunk, from, to, tier_mib_by_dev[d],
                               dense_mib_by_dev[d], kv_host, stderr,
                               (uint64_t) std::max<int64_t>(0, d == 0 ? vram_reserve_mib : vram_reserve_later_mib) << 20);
            if (!open_tier(file, tier_mib_by_dev[d], tier[d], err, stderr, tier_policy) ||
                !open_dense_cache(file, dense_mib_by_dev[d], n_blocks, dense[d], err)) {
                std::printf("ERR %s\n", err.c_str());
                std::fflush(stdout);
                cudaSetDevice(0);
                return 1;
            }
        }
        cudaSetDevice(0);
        if (split > 0)
            std::fprintf(stderr, "strata-step35: layer split: layers 0-%d on card 0, %d-%d on card 1\n", split - 1,
                         split, n_blocks - 1);
        kernels::ExpertCache host_tier;
        if (host_mib != 0) {
            if (!open_host_cache(file, (int) host_mib, n_blocks, host_tier, err, stderr)) {
                std::printf("ERR %s\n", err.c_str());
                std::fflush(stdout);
                return 1;
            }
            if (tier[0].n_slots > 0) tier[0].below = &host_tier;
            if (split > 0 && tier[1].n_slots > 0) tier[1].below = &host_tier;
        }
        kernels::ExpertCache* tierp = tier[0].n_slots > 0 ? &tier[0] : nullptr;
        kernels::DenseCache* densep = dense[0].budget > 0 ? &dense[0] : nullptr;
        kernels::ExpertCache* tierp2 = (split > 0 && tier[1].n_slots > 0) ? &tier[1] : nullptr;
        kernels::DenseCache* densep2 = (split > 0 && dense[1].budget > 0) ? &dense[1] : nullptr;
        const int head_dev = split > 0 ? 1 : 0;
        cudaSetDevice(head_dev);
        kernels::Head head;
        if (!kernels::head_open(file, head, err)) {
            std::printf("ERR %s\n", err.c_str());
            std::fflush(stdout);
            return 1;
        }
        // What the engine sees, on stderr (stdout is the protocol): this path runs on ONE card, so a config that names
        // two is worth a line here rather than a slow mystery - the runtime may hand it the other one.
        {
            int n_dev = 0;
            cudaDeviceProp prop {};
            if (cudaGetDeviceCount(&n_dev) == cudaSuccess && n_dev > 0 && cudaGetDeviceProperties(&prop, 0) == cudaSuccess)
                std::fprintf(stderr,
                             "strata-step35: device 0 of %d visible: %s (%.1f GB); this engine runs one card\n", n_dev,
                             prop.name, (double) prop.totalGlobalMem / 1e9);
        }
        std::printf("READY %d stop\n", max_ctx);
        std::fflush(stdout);

        // stdin on its own thread: a STOP (or a QUIT) behind a long answer must be seen while it is being written
        std::mutex mu;
        std::condition_variable cv;
        std::deque<std::string> queue;
        std::atomic<bool> quit{false};
        std::atomic<bool> stop{false};
        std::thread reader([&] {
            std::string line;
            while (std::getline(std::cin, line)) {
                if (line.rfind("STOP", 0) == 0) {   // seen HERE, so a STOP during a request is not queued behind it
                    stop = true;
                    continue;
                }
                {
                    std::lock_guard<std::mutex> lock(mu);
                    queue.push_back(line);
                }
                cv.notify_one();
            }
            quit = true;
            cv.notify_one();
        });

        std::vector<int32_t> cached;      // what the session holds, in order
        std::vector<float> hidden((size_t) n_embd), out, sums;
        std::vector<double> dsums;
        std::vector<int32_t> prompt;
        while (true) {
            std::string line;
            {
                std::unique_lock<std::mutex> lock(mu);
                cv.wait(lock, [&] { return !queue.empty() || quit; });
                if (queue.empty()) break;   // EOF and nothing left
                line = queue.front();
                queue.pop_front();
            }
            if (line.rfind("QUIT", 0) == 0) break;
            if (line.rfind("STOP", 0) == 0) {
                stop = false;
                continue;
            }

            const bool geni = line.rfind("GENI ", 0) == 0;
            if (!geni && line.rfind("GEN ", 0) != 0) {
                std::printf("ERR unknown request: %s\n", line.substr(0, 40).c_str());
                std::fflush(stdout);
                continue;
            }
            // GEN <n> [key=value ...] <id,id,...>   |   GENI <n> [key=value ...] <file> <id,id,...>
            SamplerConfig samp;
            int max_new = 0;
            std::string field, emb_path;
            {
                std::vector<std::string> parts;
                size_t i = geni ? 5 : 4;
                while (i < line.size()) {
                    size_t j = line.find(' ', i);
                    if (j == std::string::npos) j = line.size();
                    parts.push_back(line.substr(i, j - i));
                    i = j + 1;
                }
                size_t at = 0;
                if (at < parts.size()) max_new = (int) std::strtol(parts[at++].c_str(), nullptr, 10);
                for (; at + 1 < parts.size(); ++at) {
                    const std::string& k = parts[at];
                    if (k.find('=') == std::string::npos) break;   // the ids begin here
                    const size_t eq = k.find('=');
                    const std::string key = k.substr(0, eq), val = k.substr(eq + 1);
                    if (key == "temperature") samp.temperature = (float) std::strtod(val.c_str(), nullptr);
                    else if (key == "top_k") samp.top_k = (int) std::strtoll(val.c_str(), nullptr, 10);
                    else if (key == "top_p") samp.top_p = (float) std::strtod(val.c_str(), nullptr);
                    else if (key == "penalty_repeat") samp.repeat_penalty = (float) std::strtod(val.c_str(), nullptr);
                    else if (key == "seed") samp.seed = (uint64_t) std::strtoull(val.c_str(), nullptr, 10);
                    // every other key the server may send (min_p, frequency and presence penalties, logprobs) is
                    // accepted and ignored: refusing one would fail a request over a knob this engine does not have
                    if (samp.top_k > 64) samp.top_k = 64;
                }
                // GENI carries the embeddings file between the keys and the ids
                if (geni) {
                    emb_path = at < parts.size() ? parts[at] : std::string();
                    ++at;
                }
                field = at < parts.size() ? parts[at] : std::string();
            }
            prompt.clear();
            for (size_t i = 0; i < field.size();) {
                size_t j = field.find(',', i);
                if (j == std::string::npos) j = field.size();
                prompt.push_back((int32_t) std::strtol(field.substr(i, j - i).c_str(), nullptr, 10));
                i = j + 1;
            }
            if (prompt.empty()) {
                std::printf("ERR a request needs at least one token\n");
                std::fflush(stdout);
                continue;
            }
            // ---- a picture's cells: the <im_patch> positions take the encoder's rows, one per position in order
            std::vector<float> image_rows;
            std::vector<int32_t> image_row;
            if (geni) {
                int n_records = 0;
                if (!read_embeddings(emb_path, n_embd, image_rows, n_records, err)) {
                    std::printf("ERR %s\n", err.c_str());
                    std::fflush(stdout);
                    continue;
                }
                const int pad = image_token;
                if (pad < 0) {
                    std::printf("ERR the file has no <im_patch> token\n");
                    std::fflush(stdout);
                    continue;
                }
                image_row.assign(prompt.size(), -1);
                size_t row = 0;
                for (size_t i = 0; i < prompt.size(); ++i) {
                    if (prompt[i] == pad) image_row[i] = (int32_t) row++;
                }
                const size_t have = image_rows.size() / (size_t) n_embd;
                if (row == 0) {
                    std::printf("ERR the prompt has no <im_patch> token for %d picture(s)\n", n_records);
                    std::fflush(stdout);
                    continue;
                }
                if (row != have) {
                    std::printf("ERR the prompt has %zu <im_patch> cell(s), the embeddings file %zu row(s)\n", row, have);
                    std::fflush(stdout);
                    continue;
                }
            }

            // the common prefix with what the session already holds: a chat turn usually appends
            size_t keep = 0;
            while (keep < cached.size() && keep < prompt.size() && cached[keep] == prompt[keep]) ++keep;
            // The hidden state this function holds is the one of the LAST position it ran, which after a request is
            // the last token it GENERATED - not the prompt's.  When the whole prompt is already cached, redo its
            // last position so the answer is read from where it belongs; one position, not a prompt.
            if (keep == prompt.size()) keep = prompt.size() - 1;
            trunk.n_cached = (int) keep;   // whatever the session held past this is not part of any answer
            cached.resize(keep);
            const auto t0 = std::chrono::steady_clock::now();
            bool ok = true;
            if (keep < prompt.size()) {
                std::printf("RESUME %zu\n", keep);   // the tokens the request reuses, before the first PP
                std::fflush(stdout);
                for (size_t off = keep; ok && off < prompt.size(); off += (size_t) prefill_chunk) {
                    const size_t n = std::min((size_t) prefill_chunk, prompt.size() - off);
                    const std::vector<int32_t> chunk(prompt.begin() + off, prompt.begin() + off + n);
                    std::vector<float> part;
                    std::vector<double> csum;
                    ok = kernels::run_prompt_dense(file, chunk, (int) off, n_blocks, part, csum, err, &trunk, tierp,
                                                   densep, geni ? &image_rows : nullptr, geni ? &image_row : nullptr,
                                                   split, tierp2, densep2);
                    if (ok) out = part;
                    if (ok) {
                        const long long ms =
                            (long long) std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::steady_clock::now() - t0).count();
                        // PP <read> <total> <ms> <tok/s>: the server takes the fifth field as the prompt's rate (it
                        // showed "-" without it), and the fourth as the milliseconds since the prompt started
                        std::printf("PP %zu %zu %lld %.1f\n", off + n, prompt.size(), ms,
                                    (double) (off + n) / std::max(1e-9, (double) ms / 1000.0));
                        std::fflush(stdout);
                    }
                }
            }
            if (!ok) {
                std::printf("ERR %s\n", err.c_str());
                std::fflush(stdout);
                // the cache's contents are now unknown: the next request starts from zero
                trunk.n_cached = 0;
                cached.clear();
                continue;
            }
            hidden.assign(out.end() - n_embd, out.end());
            cached = prompt;
            int emitted = 0, pos = (int) prompt.size();
            bool stopped = false;
            const auto t_gen = std::chrono::steady_clock::now();
            Rng rng(samp.seed);
            std::vector<int> history = prompt;
            while (ok && emitted < max_new) {
                std::vector<float> lg;
                cudaSetDevice(head_dev);
                if (!kernels::head_apply(head, hidden.data(), lg, err)) {
                    ok = false;
                    break;
                }
                const int tok = sample(lg, samp, history, rng);
                if (eos >= 0 && tok == eos) break;
                std::printf("T %d\n", tok);
                std::fflush(stdout);
                history.push_back(tok);
                ++emitted;
                if (stop.exchange(false)) {
                    stopped = true;
                    break;
                }
                const std::vector<int32_t> one(1, tok);
                std::vector<float> dout;
                std::vector<double> one_sums;
                if (!kernels::run_prompt_dense(file, one, pos, n_blocks, dout, one_sums, err, &trunk, tierp, densep,
                                               nullptr, nullptr, split, tierp2, densep2)) {
                    ok = false;
                    break;
                }
                hidden.assign(dout.end() - n_embd, dout.end());
                cached.push_back(tok);
                ++pos;
            }
            if (!ok) {
                std::printf("ERR %s\n", err.c_str());
                trunk.n_cached = 0;
                cached.clear();
            } else {
                // DONE <generated> <prompt_tokens> <prompt_ms> <decode_ms> <finish> - the five fields the server's
                // _parse_done reads positionally; a bare DONE raises there and leaves the request waiting for ever
                const long long pre_ms = (long long) std::chrono::duration_cast<std::chrono::milliseconds>(
                    t_gen - t0).count();
                const long long dec_ms = (long long) std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - t_gen).count();
                const char* finish = stopped ? "cancel" : (emitted >= max_new ? "length" : "stop");
                std::printf("DONE %d %d %lld %lld %s\n", emitted, (int) prompt.size(), pre_ms, dec_ms, finish);
            }
            std::fflush(stdout);
        }
        quit = true;
        cv.notify_all();
        if (reader.joinable()) reader.join();
        for (int d = 0; d < (split > 0 ? 2 : 1); ++d)   // stderr again: the caller parses stdout
            if (tier[d].n_slots > 0)
                std::fprintf(stderr, "strata-step35: expert tier (card %d): %llu hit(s) of %llu fetch(es)\n", d,
                             (unsigned long long) tier[d].hits,
                             (unsigned long long) (tier[d].hits + tier[d].misses));
        if (host_mib != 0 && host_tier.n_slots > 0)
            std::fprintf(stderr, "strata-step35: host expert cache: %llu hit(s) of %llu fetch(es), %d slot(s)\n",
                         (unsigned long long) host_tier.hits,
                         (unsigned long long) (host_tier.hits + host_tier.misses), host_tier.n_slots);
        kernels::head_close(head);
        kernels::session_destroy(trunk);
        if (host_mib != 0) kernels::expert_cache_close(host_tier);
        kernels::expert_cache_close(tier[0]);
        kernels::dense_cache_close(dense[0]);
        if (split > 0) {
            cudaSetDevice(1);
            kernels::expert_cache_close(tier[1]);
            kernels::dense_cache_close(dense[1]);
            cudaSetDevice(0);
        }
        return 0;
    } catch (const std::exception& ex) {
        std::printf("ERR %s\n", ex.what());
        std::fflush(stdout);
        return 1;
    }
}

/// `--gpu-spec`: the same prompt, generated one BLOCK at a time - the NextN heads draft K tokens, one trunk pass
/// over K+1 positions checks them, and the longest matching prefix is kept.  The trunk's own token at the first
/// rejection is the correction, so the answer is exactly what greedy would have produced: speculation here buys
/// time, never different tokens.
int gpu_spec(const std::string& path, const std::string& mtp_path, const std::string& tokens_csv, int n_blocks,
             int max_new, int drafts, int tier_mib, int dense_mib, const std::string& tier_policy) {
    try {
        strata::GgufModel file = strata::GgufModel::open(path);
        strata::GgufModel mtp = strata::GgufModel::open(mtp_path);
        std::vector<int32_t> prompt;
        for (size_t i = 0; i < tokens_csv.size();) {
            size_t j = tokens_csv.find(',', i);
            if (j == std::string::npos) j = tokens_csv.size();
            prompt.push_back((int32_t) std::strtol(tokens_csv.substr(i, j - i).c_str(), nullptr, 10));
            i = j + 1;
        }
        if (prompt.empty() || max_new <= 0) {
            std::printf("strata-step35: --gpu-spec needs --tokens <id,id,...> and --max-new N\n");
            return 2;
        }
        std::string err;
        const int n_layers_total = (int) file.meta().get("step35.block_count")->num();
        if (n_blocks <= 0 || n_blocks > n_layers_total) n_blocks = n_layers_total;
        const int n_embd = (int) file.meta().get("step35.embedding_length")->num();
        const int n_kv = (int) file.meta().get("step35.attention.head_count_kv")->items[0].num();
        const int head_dim = (int) file.meta().get("step35.attention.key_length")->num();
        const strata::MetaValue* swa = file.meta().get("step35.attention.sliding_window_pattern");
        const strata::MetaValue* wv = file.meta().get("step35.attention.sliding_window");
        const int window = wv ? (int) wv->num() : 512;
        std::vector<int> windows((size_t) n_blocks, 0);
        for (int l = 0; l < n_blocks; ++l)
            if (swa && swa->type == strata::MetaType::ARRAY && (size_t) l < swa->items.size() &&
                swa->items[(size_t) l].num() != 0)
                windows[(size_t) l] = window;

        kernels::ExpertCache tier;
        if (!open_tier(file, tier_mib, tier, err, stdout, tier_policy)) {
            std::printf("strata-step35: %s\n", err.c_str());
            return 1;
        }
        kernels::DenseCache dense;
        if (!open_dense_cache(file, dense_mib, n_blocks, dense, err)) {
            std::printf("strata-step35: %s\n", err.c_str());
            return 1;
        }
        kernels::DenseCache* densep = dense.budget > 0 ? &dense : nullptr;
        kernels::ExpertCache* tierp = tier.n_slots > 0 ? &tier : nullptr;
        kernels::Session trunk;
        if (!kernels::session_create(trunk, n_blocks, n_kv, head_dim,
                                     (int) prompt.size() + max_new + drafts + kernels::kKvPassMax, err, windows)) {
            std::printf("strata-step35: %s\n", err.c_str());
            return 1;
        }
        std::vector<float> out;
        std::vector<double> sums;
        const auto t_pre = std::chrono::steady_clock::now();
        if (!kernels::run_prompt_dense(file, prompt, 0, n_blocks, out, sums, err, &trunk, tierp,
                                       dense.budget > 0 ? &dense : nullptr)) {
            std::printf("strata-step35: %s\n", err.c_str());
            kernels::expert_cache_close(tier);
            kernels::session_destroy(trunk);
            return 1;
        }
        const double prefill_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_pre).count();

        // the file's own end-of-sequence, not a number copied out of this prompt: a model that stops on nothing
        // would run to --max-new every time
        const strata::MetaValue* eos_v = file.meta().get("tokenizer.ggml.eos_token_id");
        const int eos = eos_v ? (int) eos_v->num() : -1;
        kernels::Head head;
        kernels::MtpContext dctx;
        kernels::Session dk;
        int n_mtp = 0;
        if (kernels::head_open(file, head, err) && kernels::mtp_open(mtp, dctx, err)) n_mtp = dctx.n_mtp;
        const int K = std::max(1, std::min(drafts, std::max(1, n_mtp)));
        if (!kernels::session_create(dk, std::max(1, n_mtp), n_kv, head_dim, K + 1, err)) n_mtp = 0;
        // `drafts` blocks of K tokens each: one cell per NextN layer per block is enough, the check pass reads none
        // INVARIANT at the top of the loop: `pos` is what the cache holds, and `h` is the hidden state of the last
        // position written, i.e. the one that predicts the token about to be emitted.  `pending` is a token the
        // trunk already chose at the previous block's first rejection - it is emitted, not re-decided.
        std::vector<float> h(out.end() - n_embd, out.end());
        int pos = (int) prompt.size(), emitted = 0, accepted_total = 0, blocks = 0, passes = 0, pending = -1;
        std::vector<int> emitted_tokens;
        const auto t_gen = std::chrono::steady_clock::now();
        while (emitted < max_new) {
            int n0 = pending;
            if (n0 < 0) {
                std::vector<float> lg;
                if (!kernels::head_apply(head, h.data(), lg, err)) {
                    std::printf("strata-step35: %s\n", err.c_str());
                    break;
                }
                n0 = argmax(lg);
            }
            pending = -1;
            if (eos >= 0 && n0 == eos) {
                std::printf("  eos after %d token(s)\n", emitted);
                break;
            }
            // ---- the draft: K tokens, each head chained on the previous one's h_nextn
            std::vector<int> d;
            std::vector<float> hh = h;
            int fed = n0;
            for (int k = 0; k < K && n_mtp > 0; ++k) {
                std::vector<float> hn, lg2;
                if (!kernels::mtp_step(dctx, k, k, pos + k, fed, hh.data(), &dk, hn, lg2, err)) break;
                d.push_back(argmax(lg2));
                fed = d.back();
                hh = hn;
            }
            // ---- verify: ONE pass over [n0, drafts...], then the trunk's own top-1 at every position
            std::vector<int32_t> batch(1, n0);
            batch.insert(batch.end(), d.begin(), d.end());
            std::vector<float> vout;
            std::vector<double> vsums;
            if (!kernels::run_prompt_dense(file, batch, pos, n_blocks, vout, vsums, err, &trunk, tierp,
                                           dense.budget > 0 ? &dense : nullptr)) {
                std::printf("strata-step35: %s\n", err.c_str());
                break;
            }
            ++passes;
            std::vector<int> t;
            for (size_t i = 0; i < batch.size(); ++i) {
                std::vector<float> lg2;
                if (!kernels::head_apply(head, vout.data() + i * (size_t) n_embd, lg2, err)) break;
                t.push_back(argmax(lg2));
            }
            if (t.size() != batch.size()) break;
            // ---- the longest prefix of the draft the trunk agrees with
            int a = 0;
            while (a < (int) d.size() && d[(size_t) a] == t[(size_t) a]) ++a;
            accepted_total += a;
            ++blocks;
            std::printf("  block %d at position %d: drafted", blocks, pos);
            for (int x : d) std::printf(" %d", x);
            std::printf(", the trunk accepts %d", a);
            if (a < (int) d.size()) std::printf(", corrects with %d", t[(size_t) a]);
            std::printf("\n");
            emitted += 1 + a;
            emitted_tokens.push_back(n0);
            for (int i = 0; i < a; ++i) emitted_tokens.push_back(d[(size_t) i]);
            if (a < (int) d.size()) pending = t[(size_t) a];   // decided here, fed at the next block
            // the state is the hidden state of the last position emitted (row `a`), and nothing past it is kept
            const int row = std::min(a, (int) batch.size() - 1);
            h.assign(vout.begin() + (size_t) row * n_embd, vout.begin() + (size_t) (row + 1) * n_embd);
            pos += a + 1;
            trunk.n_cached = pos;
        }
        if (emitted > max_new) emitted = max_new;   // a block can overshoot by the tokens it emitted at once
        const double gen_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_gen).count();
        std::printf("strata-step35: prefill %zu position(s) in %.2f s\n", prompt.size(), prefill_s);
        std::printf("strata-step35: tokens:");
        for (size_t i = 0; i < emitted_tokens.size() && (int) i < emitted; ++i)
            std::printf(" %d", emitted_tokens[i]);
        std::printf("\n");
        std::printf("strata-step35: %d token(s) in %d block(s) and %d pass(es), %.2f s: %.3f s/token, "
                    "%d draft(s) accepted of %d (%.0f%%)\n",
                    emitted, blocks, passes, gen_s, emitted ? gen_s / emitted : 0.0, accepted_total,
                    blocks * K, blocks * K ? 100.0 * (double) accepted_total / (double) (blocks * K) : 0.0);
        print_tier(tier, gen_s);
        print_dense_cache(dense);
        kernels::expert_cache_close(tier);
        kernels::dense_cache_close(dense);
        kernels::session_destroy(dk);
        kernels::mtp_close(dctx);
        kernels::head_close(head);
        kernels::session_destroy(trunk);
        return 0;
    } catch (const std::exception& ex) {
        std::printf("strata-step35: %s\n", ex.what());
        return 1;
    }
}

/// `--gpu-chat`: a prompt, then decode steps that reuse its K/V through the quantized cache.  The figures printed
/// per position are the ones `--ref` prints, so the two outputs are the comparison: prefill n positions, decode m,
/// and `--ref --tokens "<n + m ids>"` answers for the same sequence in float64.
int gpu_chat(const std::string& path, const std::string& tokens_csv, const std::string& next_csv, int n_blocks,
             int max_ctx, bool want_logits, int max_new, int tier_mib, int dense_mib,
             const std::string& tier_policy, bool kv_host, int layer_split, int host_mib, int vram_reserve_mib,
             int vram_reserve_later_mib, int prefill_chunk, const SamplerConfig& samp) {
    try {
        strata::GgufModel file = strata::GgufModel::open(path);
        auto parse = [](const std::string& csv) {
            std::vector<int32_t> v;
            for (size_t i = 0; i < csv.size();) {
                size_t j = csv.find(',', i);
                if (j == std::string::npos) j = csv.size();
                v.push_back((int32_t) std::strtol(csv.substr(i, j - i).c_str(), nullptr, 10));
                i = j + 1;
            }
            return v;
        };
        const std::vector<int32_t> prompt = parse(tokens_csv), next = parse(next_csv);
        if (prompt.empty()) {
            std::printf("strata-step35: --gpu-chat needs --tokens <id,id,...>\n");
            return 2;
        }
        const int n_layers_total = (int) file.meta().get("step35.block_count")->num();
        if (n_blocks <= 0 || n_blocks > n_layers_total) n_blocks = n_layers_total;
        const int n_embd = (int) file.meta().get("step35.embedding_length")->num();
        const int n_head_kv = (int) file.meta().get("step35.attention.head_count_kv")->items[0].num();
        const int head_dim = (int) file.meta().get("step35.attention.key_length")->num();
        const int needed = (int) prompt.size() + (int) next.size();
        const int capacity = std::max(max_ctx, needed);

        // Which blocks keep the whole prefix and which read a window - from the file, block by block, because a
        // block that attended over a window the cache did not apply (or the reverse) would answer something
        // plausible and wrong.  A sliding layer's cache is a ring of `window + kKvPassMax - 1` cells.
        const strata::MetaValue* swa = file.meta().get("step35.attention.sliding_window_pattern");
        const strata::MetaValue* wv = file.meta().get("step35.attention.sliding_window");
        const int window = wv ? (int) wv->num() : 512;
        std::vector<int> windows((size_t) n_blocks, 0);
        for (int l = 0; l < n_blocks; ++l) {
            // the model-wide pattern first, then the per-block key `upload_block` itself reads: a file that declares
            // one and not the other must not silently get a full cache where a window is attended - step35_seq.cu
            // refuses that combination instead of answering from cells the next position already reused
            bool sliding = false;
            if (swa && swa->type == strata::MetaType::ARRAY && (size_t) l < swa->items.size()) {
                sliding = swa->items[(size_t) l].num() != 0;
            } else if (const strata::MetaValue* pb =
                           file.meta().get("blk." + std::to_string(l) + ".attention.sliding_window_pattern")) {
                if (pb->type == strata::MetaType::ARRAY && (size_t) l < pb->items.size())
                    sliding = pb->items[(size_t) l].num() != 0;
                else if (pb->is_num())
                    sliding = pb->num() != 0;
            }
            if (sliding) windows[(size_t) l] = window;
        }

        kernels::Session s;
        std::string err;
        if (!kernels::session_create(s, n_blocks, n_head_kv, head_dim, capacity, err, windows, kv_host)) {
            std::printf("strata-step35: %s\n", err.c_str());
            return 1;
        }
        if (kv_host)
            std::printf("strata-step35: the K/V cache lives in pinned host memory (device-mapped): it costs no VRAM, "
                        "and every attention reads it over the bus\n");
        // A layer split across two cards: layers 0..split-1 on card 0, the rest on card 1, each with its own K/V share,
        // its own dense cache and its own expert tier - sized from ITS free VRAM.
        const int split = (layer_split != 0 && n_blocks > 1)
                              ? (layer_split < 0 ? n_blocks / 2 : (int) std::min<int64_t>(layer_split, n_blocks - 1))
                              : 0;
        if (split > 0) std::printf("strata-step35: layer split: layers 0-%d on card 0, %d-%d on card 1\n", split - 1,
                                   split, n_blocks - 1);
        const int dense_layers = (int) file.meta().get("step35.leading_dense_block_count")->num();
        kernels::ExpertCache tier[2];
        kernels::DenseCache dense[2];
        int tier_mib_by_dev[2] = {(int) tier_mib, (int) tier_mib};
        int dense_mib_by_dev[2] = {(int) dense_mib, (int) dense_mib};
        for (int d = 0; d < (split > 0 ? 2 : 1); ++d) {
            cudaSetDevice(d);
            const int from = d == 0 ? 0 : split, to = d == 0 ? (split > 0 ? split : n_blocks) : n_blocks;
            auto_budgets_range(file, n_blocks, dense_layers, s, from, to, tier_mib_by_dev[d], dense_mib_by_dev[d],
                               kv_host, stdout,
                               (uint64_t) std::max<int64_t>(0, d == 0 ? vram_reserve_mib : vram_reserve_later_mib) << 20);
            if (!open_tier(file, tier_mib_by_dev[d], tier[d], err, stdout, tier_policy)) {
                std::printf("strata-step35: %s\n", err.c_str());
                cudaSetDevice(0);
                return 1;
            }
            if (!open_dense_cache(file, dense_mib_by_dev[d], n_blocks, dense[d], err)) {
                std::printf("strata-step35: %s\n", err.c_str());
                cudaSetDevice(0);
                return 1;
            }
        }
        cudaSetDevice(0);
        // the level below: one host cache per card's tier (they hold the same experts, so one would do, but a split's
        // two tiers must not fight over one LRU)
        // ONE host cache, shared by both cards' tiers: its keys are (layer, expert), and each layer lives on one card,
        // so a single LRU is correct - and asking for one per card was exactly twice the RAM (33.6 GB pinned on 62 GB).
        kernels::ExpertCache host_tier;
        if (host_mib != 0) {
            if (!open_host_cache(file, (int) host_mib, n_blocks, host_tier, err, stdout)) {
                std::printf("strata-step35: %s\n", err.c_str());
                return 1;
            }
            if (tier[0].n_slots > 0) tier[0].below = &host_tier;
            if (split > 0 && tier[1].n_slots > 0) tier[1].below = &host_tier;
        }
        kernels::ExpertCache* tierp = tier[0].n_slots > 0 ? &tier[0] : nullptr;
        kernels::DenseCache* densep = dense[0].budget > 0 ? &dense[0] : nullptr;
        kernels::ExpertCache* tierp2 = (split > 0 && tier[1].n_slots > 0) ? &tier[1] : nullptr;
        kernels::DenseCache* densep2 = (split > 0 && dense[1].budget > 0) ? &dense[1] : nullptr;
        int n_full = 0, n_ring = 0, ring_cells = 0;
        for (int l = 0; l < n_blocks; ++l) {
            const int c = kernels::session_layer_cells(s, l);
            if (windows[(size_t) l] > 0) { ++n_ring; ring_cells = c; } else { ++n_full; }
        }
        const strata::MetaValue* eos_v = file.meta().get("tokenizer.ggml.eos_token_id");
        const int eos = eos_v ? (int) eos_v->num() : -1;
        std::printf("strata-step35: cache: %d full layer(s) x %d cells, %d sliding layer(s) x %d cells (window %d)"
                    " -> %.3f GB at %d positions, %.3f GB if every layer kept every position\n",
                    n_full, s.capacity, n_ring, ring_cells, window,
                    (double) kernels::session_bytes_at(s, s.capacity) / 1e9, s.capacity,
                    (double) s.capacity * (double) kernels::session_bytes_per_token(s) / 1e9);

        // ---- the prompt, in passes of at most kKvPassMax positions: a sliding layer's ring is sized for one such
        // pass, and the rows of a pass are independent (each reads the cells of the positions before it), so
        // splitting the prompt moves no bit - the parity test appends in two chunks and compares the whole thing.
        std::vector<float> out;
        std::vector<double> sums;
        const auto t_prefill = std::chrono::steady_clock::now();
        for (size_t off = 0; off < prompt.size(); off += (size_t) prefill_chunk) {
            const size_t n = std::min((size_t) prefill_chunk, prompt.size() - off);
            const std::vector<int32_t> chunk(prompt.begin() + off, prompt.begin() + off + n);
            std::vector<float> part;
            std::vector<double> csum;
            if (!kernels::run_prompt_dense(file, chunk, (int) off, n_blocks, part, csum, err, &s, tierp, densep,
                                           nullptr, nullptr, split, tierp2, densep2)) {
                std::printf("strata-step35: %s\n", err.c_str());
                return 1;
            }
            if (sums.empty()) {
                sums = csum;
            } else {
                for (size_t l = 0; l < sums.size() && l < csum.size(); ++l) sums[l] += csum[l];
            }
            out.insert(out.end(), part.begin(), part.end());
        }
        std::printf("strata-step35: prefill of %zu position(s) through %d block(s) in %.3f s (%.3f s/position)\n",
                    prompt.size(), n_blocks,
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - t_prefill).count(),
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - t_prefill).count() /
                        (double) prompt.size());
        for (size_t l = 0; l < sums.size(); ++l) std::printf("  block %2zu : sum %.6f\n", l, sums[l]);
        std::vector<float> rows = out;   // [position][n_embd], the figures --ref prints

        // ---- the decode steps: ONE position each, attending to the cells the prompt left behind
        int pos = (int) prompt.size();
        for (size_t d = 0; d < next.size(); ++d) {
            std::vector<int32_t> one(1, next[d]);
            std::vector<float> dout;
            std::vector<double> dsums;
            if (!kernels::run_prompt_dense(file, one, pos, n_blocks, dout, dsums, err, &s, tierp, densep, nullptr,
                                           nullptr, split, tierp2, densep2)) {
                std::printf("strata-step35: decode %zu: %s\n", d, err.c_str());
                return 1;
            }
            double ss = 0;
            for (float x : dout) ss += (double) x * (double) x;
            std::printf("  decode %zu (token %d at position %d, %d cached): sum %.6f rms %.6f\n", d, next[d], pos,
                        s.n_cached, dsums.empty() ? 0.0 : dsums.back(), std::sqrt(ss / (double) n_embd));
            rows.insert(rows.end(), dout.begin(), dout.end());
            ++pos;
        }
        // ---- greedy generation: each step's own top-1 is the next step's token, through the head kept open
        int generated = 0;
        double gen_s = 0, worst = 0, best = 1e9;
        if (max_new > 0) {
            const auto t_gen0 = std::chrono::steady_clock::now();
            const int head_dev = split > 0 ? 1 : 0;
            cudaSetDevice(head_dev);
            kernels::Head head;
            if (!kernels::head_open(file, head, err)) {
                std::printf("strata-step35: %s\n", err.c_str());
                kernels::session_destroy(s);
                return 1;
            }
            std::vector<float> h = std::vector<float>(rows.end() - n_embd, rows.end());
            std::vector<int> history = prompt;   // what the repetition penalty looks back at
            Rng rng(samp.seed);
            for (int g = 0; g < max_new; ++g) {
                std::vector<float> lg;
                cudaSetDevice(head_dev);
                if (!kernels::head_apply(head, h.data(), lg, err)) {
                    std::printf("strata-step35: %s\n", err.c_str());
                    kernels::head_close(head);
                    kernels::session_destroy(s);
                    return 1;
                }
                const int tok = sample(lg, samp, history, rng);
                if (eos >= 0 && tok == eos) {   // the file's own end token: the answer is over, whatever the count asked
                    std::printf("  end token after %d token(s)\n", generated);
                    break;
                }
                history.push_back(tok);
                const int at = pos;
                const std::vector<int32_t> one(1, tok);
                std::vector<float> dout;
                std::vector<double> dsums;
                const auto t0 = std::chrono::steady_clock::now();
                if (!kernels::run_prompt_dense(file, one, at, n_blocks, dout, dsums, err, &s, tierp, densep, nullptr,
                                               nullptr, split, tierp2, densep2)) {
                    std::printf("strata-step35: %s\n", err.c_str());
                    kernels::head_close(head);
                    kernels::session_destroy(s);
                    return 1;
                }
                const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                gen_s += dt;
                worst = std::max(worst, dt);
                best = std::min(best, dt);
                ++generated;
                std::printf("  generate %d: token %d (logit %.3f) at position %d, %.3f s\n", g, tok,
                            lg[(size_t) tok], at, dt);
                h.assign(dout.end() - n_embd, dout.end());
                rows.insert(rows.end(), dout.begin(), dout.end());
                ++pos;
            }
            kernels::head_close(head);
            cudaSetDevice(0);
            const double total = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_gen0).count();
            std::printf("strata-step35: %d token(s) generated in %.2f s (%.3f s/token, best %.3f, worst %.3f)\n",
                        generated, total, generated ? gen_s / generated : 0.0, best, worst);
            print_profile(total);
        }

        for (size_t p = 0; p * (size_t) n_embd < rows.size(); ++p) {
            const float* h = rows.data() + p * (size_t) n_embd;
            double ss = 0;
            for (int i = 0; i < n_embd; ++i) ss += (double) h[i] * (double) h[i];
            std::printf("  position %zu: rms %.6f, first four:", p, std::sqrt(ss / (double) n_embd));
            for (int i = 0; i < 4; ++i) std::printf(" %.6f", h[i]);
            std::printf("\n");
        }
        if (want_logits && n_embd > 0 && !rows.empty()) {
            std::vector<float> lg;
            const size_t last = rows.size() / (size_t) n_embd - 1;
            if (!kernels::logits_last(file, rows.data() + last * (size_t) n_embd, lg, err)) {
                std::printf("strata-step35: %s\n", err.c_str());
                return 1;
            }
            print_top5(lg, "after " + std::to_string(n_blocks) + " block(s) on the GPU, reading the cache");
        }
        print_tier(tier[0], 0.0);
        print_dense_cache(dense[0]);
        if (host_mib != 0 && host_tier.n_slots > 0)
            std::printf("strata-step35: host expert cache: %llu hit(s) of %llu fetch(es), %d slot(s)\n",
                        (unsigned long long) host_tier.hits,
                        (unsigned long long) (host_tier.hits + host_tier.misses), host_tier.n_slots);
        if (split > 0) {
            print_tier(tier[1], 0.0);
            print_dense_cache(dense[1]);
        }
        if (host_mib != 0) kernels::expert_cache_close(host_tier);
        kernels::expert_cache_close(tier[0]);
        kernels::dense_cache_close(dense[0]);
        if (split > 0) {
            cudaSetDevice(1);
            kernels::expert_cache_close(tier[1]);
            kernels::dense_cache_close(dense[1]);
            cudaSetDevice(0);
        }
        kernels::session_destroy(s);
        return 0;
    } catch (const std::exception& ex) {
        std::printf("strata-step35: %s\n", ex.what());
        return 1;
    }
}

void usage() {
    std::printf(
        "strata-step35: the Step-3.7-Flash engine path (GGUF architecture step35)\n"
        "\n"
        "  strata-step35 --serve --model <first shard> [--max-context N] [--expert-cache MiB]\n"
        "      The resident engine serve/server.py speaks to: READY, then one GEN request per line on stdin, T per\n"
        "      token, DONE at the end, STOP to end a request early, QUIT to exit.  Weights, cache, head and expert\n"
        "      tier stay loaded; a request whose prompt extends the last one decodes only what is new.\n"
        "\n"
        "  strata-step35 --inspect <first shard> [--mtp <nextn file>]\n"
        "      Read the file, print the geometry it declares and check every tensor against it.  No GPU.\n"
        "\n"
        "  strata-step35 --gpu-block <first shard> --layer N --token <id> [--pos N]\n"
        "      One dense block on the GPU with the file's own weights, for one token.  Prints the output sum.\n"
        "\n"
        "  strata-step35 --ref <first shard> --tokens <id,id,...> [--layers N] [--mtp <nextn file>]\n"
        "      Run the CPU reference over the blocks (default: all of them, 3 dense then 42 MoE) and print\n"
        "      what each one produced.  The reference a kernel is compared against; no GPU.  With --mtp it\n"
        "      also runs the NextN draft heads, exactly as --gpu-mtp does, so the two can be compared.\n"
        "\n"
        "  strata-step35 --gpu-spec <first shard> --mtp <nextn file> --tokens <id,id,...> --max-new N [--drafts K]\n"
        "      The same generation, one BLOCK at a time: the NextN heads draft K tokens, ONE trunk pass over K+1\n"
        "      positions checks them, and the longest matching prefix is kept - the trunk's token at the first\n"
        "      rejection is the correction.  The tokens are exactly greedy's; only the time changes.  Prints the\n"
        "      acceptance rate and the seconds per token.\n"
        "\n"
        "  strata-step35 --gpu-mtp <first shard> --mtp <nextn file> --tokens <id,id,...> [--layers N] [--drafts N]\n"
        "      The trunk over the prompt, then its own greedy continuation, then the NextN blocks as a chain of\n"
        "      draft heads.  Each head is fed the token the trunk chose and its top-1 is compared with the token\n"
        "      the trunk chose next: that comparison is what a draft is worth.\n"
        "\n"
        "  strata-step35 --gpu-chat <first shard> --tokens <id,id,...> [--next <id,id,...>] [--layers N]\n"
        "      [--max-context N] [--logits] [--max-new N]\n"
        "      Prefill the prompt with the quantized K/V cache, then decode each --next token against it.  Prints\n"
        "      the per-block sums and, per position, the rms and first four values --ref prints: the two outputs\n"
        "      are the comparison the cache is held to.  `--logits` adds the file's own output_norm and output\n"
        "      applied to the last position, and the five tokens it would answer with - the figure `--ref`\n"
        "      prints, so the head is held to the reference the same way the blocks are.  --gpu-seq takes it too.\n"
        "      --max-new N generates N tokens greedily after them, each step's own top-1 being the next step's\n"
        "      token, and prints the per-token time.  --expert-cache MiB keeps that much of the routed experts in\n"
        "      VRAM and reads them from there; the answer is bit for bit the same, only the copies go away.\n"
        "      --dense-cache MiB keeps the blocks' own attention and feed-forward weights in VRAM instead of uploading\n"
        "      them once per pass per block - 2 GB holds all 45 and is where the larger half of a token goes.\n"
        "      [--tier-policy fifo|lru|lfu] chooses which expert a full tier replaces (default lru: measured 41.7% of\n"
        "      fetches served at 4 GB against fifo's 37.2%, and 67.6% against 57.6% at 8 GB).  Both caches take\n"
        "      `auto`: what the card has left once the K/V cache and 1.5 GB of reserve are out of the way, the blocks\n"
        "      first and the tier with the rest.  --max-new stops on the file's own end token.\n"
        "      --kv-host puts the K/V cache in pinned, device-mapped host memory: it costs no VRAM at all, at the price\n"
        "      of reading it over the bus - what lets a context bigger than the card run at all.\n"
        "      --host-cache MiB puts a second level BEHIND the expert tier, in pinned RAM: a miss in the card's tier\n"
        "      then copies from memory instead of from the file.  Keep it under the free RAM (the model's 75 GB does\n"
        "      not fit; a third of it is what the page cache would hold anyway, but chosen expert by expert).  `auto`\n"
        "      takes a quarter of what the machine can lend (ceiling 8 GiB), and anything above is clamped to that.\n"
        "      The memory is PAGEABLE on purpose: pinned memory cannot be swapped, and gigabytes of it take a desktop\n"
        "      down - measured, twice, with Cinnamon as the victim.  It is off unless asked for.\n"
        "      --vram-reserve-mib N leaves N MiB of the FIRST card free for everything else, and\n"
        "      --vram-reserve-later-mib N of the SECOND, on top of the engine's own buffers.  The desktop's\n"
        "      compositor lives there: a compositor with no VRAM loses its GL context and takes the session with it,\n"
        "      which is what Mesa's DeviceReset in the log means.  With a layer split both cards are used, so reserve\n"
        "      on BOTH - and more on the one that drives the display.\n"
        "      --prefill-chunk N reads the prompt in passes of N positions (default 1024, never more).  A pass works\n"
        "      through 8 x N experts per block, so a smaller pass keeps the card's tier usable: measured below.\n"
        "      [--temperature T] [--top-k K] [--top-p P] [--repeat-penalty R] [--seed S] sample instead of taking\n"
        "      the argmax; T=0 (the default) is greedy and ignores the rest.  --gpu-spec stays greedy: accepting a\n"
        "      draft drawn from a distribution needs its own rejection rule, which is not written yet.\n"
        "\n"
        "Nothing serves in this build: --gpu-chat runs a prompt, its --next steps and --max-new greedy tokens,\n"
        "and no process answers requests.  When the forward pass lands the flags will be strata's own\n"
        "(--model, --max-context, --tokens, --max-new, --serve) plus --mtp for the NextN file.\n");
}

/// `--ref`: the CPU reference over the dense blocks, for a token sequence given by hand.
int reference(const std::string& path, const std::string& tokens_csv, int64_t layers, const std::string& dump,
              const std::string& mtp_path) {
    ref::Model m;
    std::string err;
    if (!ref::load(path, m, err)) {
        std::printf("strata-step35: %s\n", err.c_str());
        return 1;
    }
    // `--layers` omitted is 0 here, and 0 has always meant "every block" in the help text: the comparison that
    // left the range check as `layers < 0` ran no block at all and printed the embedding as if it were an answer.
    if (layers <= 0 || layers > m.n_layers) layers = m.n_layers;
    std::vector<int32_t> ids;
    for (size_t i = 0; i < tokens_csv.size();) {
        size_t j = tokens_csv.find(',', i);
        if (j == std::string::npos) j = tokens_csv.size();
        ids.push_back((int32_t) std::strtol(tokens_csv.substr(i, j - i).c_str(), nullptr, 10));
        i = j + 1;
    }
    if (ids.empty()) {
        std::printf("strata-step35: --ref needs --tokens <id,id,...>\n");
        return 2;
    }
    std::vector<float> hidden;
    std::vector<double> trace;
    if (!ref::forward(m, ids, layers, hidden, err, &trace)) {
        std::printf("strata-step35: %s\n", err.c_str());
        return 1;
    }
    std::printf("strata-step35: %zu token(s) through %lld block(s), embedding %lld\n", ids.size(),
                (long long) layers, (long long) m.n_embd);
    for (size_t l = 0; l < trace.size(); ++l)
        std::printf("  block %2zu : sum %.6f\n", l, trace[l]);
    for (size_t p = 0; p < ids.size(); ++p) {
        const float* h = hidden.data() + p * (size_t) m.n_embd;
        double ss = 0;
        for (int64_t i = 0; i < m.n_embd; ++i) ss += (double) h[i] * (double) h[i];
        std::printf("  position %zu: rms %.6f, first four:", p, std::sqrt(ss / (double) m.n_embd));
        for (int i = 0; i < 4; ++i) std::printf(" %.6f", h[i]);
        std::printf("\n");
    }
    std::vector<float> last(hidden.end() - m.n_embd, hidden.end());
    std::vector<float> lg;
    if (!ref::logits(m, last, lg, err)) {
        std::printf("strata-step35: %s\n", err.c_str());
        return 1;
    }
    if (!dump.empty()) {
        // the raw f32, position-major, behind a one-line header: what a comparison against the oracle reads
        std::FILE* out = std::fopen(dump.c_str(), "wb");
        if (!out) {
            std::printf("strata-step35: cannot write %s\n", dump.c_str());
            return 1;
        }
        const int64_t n_pos = (int64_t) ids.size();
        std::fprintf(out, "step35 hidden %lld %lld\n", (long long) n_pos, (long long) m.n_embd);
        std::fwrite(hidden.data(), sizeof(float), hidden.size(), out);
        std::fprintf(out, "\nstep35 logits %lld\n", (long long) lg.size());
        std::fwrite(lg.data(), sizeof(float), lg.size(), out);
        std::fclose(out);
        std::printf("  wrote %s (%lld x %lld hidden + %lld logits)\n", dump.c_str(), (long long) n_pos,
                    (long long) m.n_embd, (long long) lg.size());
    }
    // the five tokens the file itself would answer with, which is what the oracle can be compared on
    print_top5(lg, "after " + std::to_string((long long) layers) + " block(s)");
    if (!mtp_path.empty()) {
        ref::Model mm;
        if (!ref::load(mtp_path, mm, err)) {
            std::printf("strata-step35: %s\n", err.c_str());
            return 1;
        }
        const int64_t n_mtp = mm.n_layers - m.n_layers;
        if (n_mtp <= 0) {
            std::printf("strata-step35: %s declares no NextN block\n", mtp_path.c_str());
            return 1;
        }
        // The trunk's own continuation, one greedy token at a time - the same figure --gpu-mtp compares against,
        // and the same absolute positions, so the two runs are the same computation and not two opinions.
        std::vector<int32_t> seq = ids;
        std::vector<std::vector<float>> hidden;
        std::vector<int> tok;
        hidden.push_back(last);
        for (int64_t k = 0; k <= n_mtp; ++k) {
            std::vector<float> lg2;
            if (!ref::logits(m, hidden[(size_t) k], lg2, err)) {
                std::printf("strata-step35: %s\n", err.c_str());
                return 1;
            }
            tok.push_back(argmax(lg2));
            if (k == n_mtp) break;
            seq.push_back(tok[(size_t) k]);
            std::vector<float> out2;
            if (!ref::forward(m, seq, layers, out2, err)) {
                std::printf("strata-step35: %s\n", err.c_str());
                return 1;
            }
            hidden.emplace_back(out2.end() - m.n_embd, out2.end());
        }
        std::printf("strata-step35: the trunk continues");
        for (int k = 0; k <= (int) n_mtp; ++k) std::printf(" %d", tok[(size_t) k]);
        std::printf("\n");
        const size_t base = ids.size();
        std::vector<std::vector<ref::PositionKv>> kv((size_t) n_mtp,
                                                    std::vector<ref::PositionKv>(base + (size_t) n_mtp));
        std::vector<float> hh = hidden[0], hn, lg2;
        int agree = 0;
        for (int64_t k = 0; k < n_mtp; ++k) {
            if (!ref::mtp_block(mm, m.n_layers + k, (int64_t) base + k, tok[(size_t) k], hh, kv[(size_t) k], hn, lg2,
                                err)) {
                std::printf("strata-step35: %s\n", err.c_str());
                return 1;
            }
            const int d = argmax(lg2);
            const int t = tok[(size_t) (k + 1)];
            if (d == t) ++agree;
            double ss = 0;
            for (float v : hn) ss += (double) v * (double) v;
            std::printf("  draft %lld: block %lld at position %lld, fed token %d, h_nextn rms %.6f -> %d %s (trunk: "
                        "%d)\n",
                        (long long) k, (long long) (m.n_layers + k), (long long) (base + (size_t) k), tok[(size_t) k],
                        std::sqrt(ss / (double) m.n_embd), d, d == t ? "AGREES" : "differs", t);
            print_top5(lg2, "draft " + std::to_string((long long) k));
            hh = hn;
        }
        std::printf("strata-step35: the draft agreed with the trunk on %d of %lld token(s)\n", agree,
                    (long long) n_mtp);
    }
    return 0;
}

}  // namespace

int run(int argc, char** argv) {
    std::string trunk, mtp, tokens, dump, next;
    int64_t layers = 0, gpu_layer = -1, token = -1, pos = 0, max_ctx = 8192;
    bool want_ref = false, want_gpu = false, want_moe = false, want_chain = false, want_all = false, want_seq = false;
    bool want_chat = false, want_logits = false, want_gpu_mtp = false, want_spec = false, want_serve = false;
    int64_t drafts = 3, max_new = 0, tier_mib = 0, dense_mib = 0, top_k = 0, seed = 0;
    // -1 means "auto": what the card has left once the K/V cache and a reserve are out of the way
    std::string tier_policy = "lru";
    bool kv_host = false, want_dump_emb = false;
    int64_t layer_split = 0, host_mib = 0, vram_reserve_mib = 0, vram_reserve_later_mib = 0, prefill_chunk = 1024;
    int64_t image_token = kStep35ImageToken;
    std::string dump_emb;
    double temperature = 0.0, top_p = 1.0, repeat_penalty = 1.0;
    for (int i = 1; i < argc; ++i) {
        // `--logits` is a switch; the flags this engine takes all take a value, so one with nothing after it is an
        // error and not a silent no-op (the loop used to stop at argc-1, which swallowed a trailing bare flag).
        const bool is_flag = argv[i][0] == '-' && argv[i][1] != '\0';
        // switches this engine takes or tolerates: `--vision` is written by setup for the families that read pictures,
        // and here the encoder is strata-vision's, so it is accepted and does nothing
        const bool is_switch = std::strcmp(argv[i], "--logits") == 0 || std::strcmp(argv[i], "--kv-host") == 0 ||
                               std::strcmp(argv[i], "--vision") == 0;
        if (is_flag && !is_switch && i + 1 >= argc) {
            std::printf("strata-step35: %s needs a value\n", argv[i]);
            return 2;
        }
        if (std::strcmp(argv[i], "--logits") == 0) {
            want_logits = true;
        } else if (std::strcmp(argv[i], "--inspect") == 0 || std::strcmp(argv[i], "--model") == 0 ||
            std::strcmp(argv[i], "--native") == 0) {
            trunk = argv[++i];
        } else if (std::strcmp(argv[i], "--serve") == 0) {
            want_serve = true;
            // the server passes the model as a flag of its own, not as a mode: `--serve` takes no value
            if (i + 1 < argc && argv[i + 1][0] != '-' && trunk.empty()) trunk = argv[++i];
        } else if (std::strcmp(argv[i], "--gpu-spec") == 0) {
            trunk = argv[++i];
            want_spec = true;
        } else if (std::strcmp(argv[i], "--gpu-mtp") == 0) {
            trunk = argv[++i];
            want_gpu_mtp = true;
        } else if (std::strcmp(argv[i], "--gpu-chat") == 0) {
            trunk = argv[++i];
            want_chat = true;
        } else if (std::strcmp(argv[i], "--gpu-seq") == 0) {
            trunk = argv[++i];
            want_seq = true;
        } else if (std::strcmp(argv[i], "--gpu-all") == 0) {
            trunk = argv[++i];
            want_all = true;
        } else if (std::strcmp(argv[i], "--gpu-chain") == 0) {
            trunk = argv[++i];
            want_chain = true;
        } else if (std::strcmp(argv[i], "--gpu-moe") == 0) {
            trunk = argv[++i];
            want_moe = true;
        } else if (std::strcmp(argv[i], "--gpu-block") == 0) {
            trunk = argv[++i];
            want_gpu = true;
        } else if (std::strcmp(argv[i], "--ref") == 0) {
            trunk = argv[++i];
            want_ref = true;
        } else if (std::strcmp(argv[i], "--mtp") == 0) {
            mtp = argv[++i];
        } else if (std::strcmp(argv[i], "--tokens") == 0) {
            tokens = argv[++i];
        } else if (std::strcmp(argv[i], "--temperature") == 0) {
            temperature = std::strtod(argv[++i], nullptr);
        } else if (std::strcmp(argv[i], "--top-k") == 0) {
            top_k = std::strtoll(argv[++i], nullptr, 10);
        } else if (std::strcmp(argv[i], "--top-p") == 0) {
            top_p = std::strtod(argv[++i], nullptr);
        } else if (std::strcmp(argv[i], "--repeat-penalty") == 0) {
            repeat_penalty = std::strtod(argv[++i], nullptr);
        } else if (std::strcmp(argv[i], "--seed") == 0) {
            seed = std::strtoll(argv[++i], nullptr, 10);
        } else if (std::strcmp(argv[i], "--kv-host") == 0) {
            kv_host = true;
        } else if (std::strcmp(argv[i], "--prefill-chunk") == 0) {
            // never above kKvPassMax: a sliding layer's ring is sized for one such pass and nothing more
            prefill_chunk = std::min<int64_t>(kernels::kKvPassMax,
                                              std::max<int64_t>(1, std::strtoll(argv[++i], nullptr, 10)));
        } else if (std::strcmp(argv[i], "--vram-reserve-mib") == 0) {
            vram_reserve_mib = std::strtoll(argv[++i], nullptr, 10);
        } else if (std::strcmp(argv[i], "--vram-reserve-later-mib") == 0) {
            vram_reserve_later_mib = std::strtoll(argv[++i], nullptr, 10);
        } else if (std::strcmp(argv[i], "--host-cache") == 0) {
            const std::string v = argv[++i];
            host_mib = v == "auto" ? -1 : std::strtoll(v.c_str(), nullptr, 10);
        } else if (std::strcmp(argv[i], "--layer-split") == 0) {
            const std::string v = argv[++i];
            layer_split = v == "auto" ? -1 : std::strtoll(v.c_str(), nullptr, 10);
        } else if (std::strcmp(argv[i], "--vision") == 0) {
            // pictures arrive as GENI requests; the flag only says the encoder is there
        } else if (std::strcmp(argv[i], "--image-token") == 0) {
            image_token = std::strtoll(argv[++i], nullptr, 10);
        } else if (std::strcmp(argv[i], "--dump-embeddings") == 0) {
            dump_emb = argv[++i];
            want_dump_emb = true;
        } else if (std::strcmp(argv[i], "--tier-policy") == 0) {
            tier_policy = argv[++i];
        } else if (std::strcmp(argv[i], "--dense-cache") == 0) {
            const std::string v = argv[++i];
            dense_mib = v == "auto" ? -1 : std::strtoll(v.c_str(), nullptr, 10);
        } else if (std::strcmp(argv[i], "--expert-cache") == 0) {
            const std::string v = argv[++i];
            tier_mib = v == "auto" ? -1 : std::strtoll(v.c_str(), nullptr, 10);
        } else if (std::strcmp(argv[i], "--max-new") == 0) {
            max_new = std::strtoll(argv[++i], nullptr, 10);
        } else if (std::strcmp(argv[i], "--drafts") == 0) {
            drafts = std::strtoll(argv[++i], nullptr, 10);
        } else if (std::strcmp(argv[i], "--next") == 0) {
            next = argv[++i];
        } else if (std::strcmp(argv[i], "--dump") == 0) {
            dump = argv[++i];
        } else if (std::strcmp(argv[i], "--layers") == 0) {
            layers = std::strtoll(argv[++i], nullptr, 10);
        } else if (std::strcmp(argv[i], "--max-context") == 0) {
            max_ctx = std::strtoll(argv[++i], nullptr, 10);
        } else if (std::strcmp(argv[i], "--layer") == 0) {
            gpu_layer = std::strtoll(argv[++i], nullptr, 10);
        } else if (std::strcmp(argv[i], "--token") == 0) {
            token = std::strtoll(argv[++i], nullptr, 10);
        } else if (std::strcmp(argv[i], "--pos") == 0) {
            pos = std::strtoll(argv[++i], nullptr, 10);
        } else if (is_flag && i + 1 < argc) {
            // an option of another engine that the launcher passes anyway (--layer-split auto, --slots 2, --batch ...):
            // skipped with its value rather than read as a model path or refused.  This engine is single-device, one
            // request at a time, and says so in docs/STEP37.md instead of pretending otherwise.
            ++i;
        }
    }
    if (trunk.empty()) {
        usage();
        return 2;
    }
    if (want_dump_emb) return dump_embeddings(trunk, tokens, dump_emb);
    if (want_ref) return reference(trunk, tokens, layers, dump, mtp);
    // --gpu-spec stays greedy on purpose: it accepts a draft by comparing argmaxes, and drawing from a distribution
    // would need its own rejection rule to keep the answer's distribution.  `samp` is not passed to it.
    SamplerConfig samp;
    samp.temperature = (float) temperature;
    samp.top_k = (int) top_k;
    samp.top_p = (float) top_p;
    samp.repeat_penalty = (float) repeat_penalty;
    samp.seed = (uint64_t) seed;
    if (want_spec)
        return gpu_spec(trunk, mtp, tokens, (int) layers, (int) max_new, (int) drafts, (int) tier_mib,
                        (int) dense_mib, tier_policy);
    if (want_serve)
        return serve(trunk, mtp, (int) max_ctx, (int) tier_mib, (int) dense_mib, tier_policy, kv_host,
                     (int) image_token, (int) layer_split, (int) host_mib, (int) vram_reserve_mib,
                     (int) vram_reserve_later_mib, (int) prefill_chunk);
    if (want_gpu_mtp)
        return gpu_mtp(trunk, mtp, tokens, (int) layers, (int) drafts, (int) tier_mib, (int) dense_mib, tier_policy);
    if (want_chat)
        return gpu_chat(trunk, tokens, next, (int) layers, (int) max_ctx, want_logits, (int) max_new,
                        (int) tier_mib, (int) dense_mib, tier_policy, kv_host, (int) layer_split, (int) host_mib,
                        (int) vram_reserve_mib, (int) vram_reserve_later_mib, (int) prefill_chunk, samp);
    if (want_seq) return gpu_seq(trunk, tokens, (int) layers, (int) pos, want_logits);
    if (want_all) {
        if (token < 0) {
            std::printf("strata-step35: --gpu-all needs --token <id>\n");
            return 2;
        }
        return gpu_all(trunk, (int32_t) token, (int) pos);
    }
    if (want_chain) {
        if (token < 0 || layers <= 0) {
            std::printf("strata-step35: --gpu-chain needs --token <id> and --layers N\n");
            return 2;
        }
        return gpu_chain(trunk, (int32_t) token, (int) layers, (int) pos);
    }
    if (want_moe) {
        if (token < 0) {
            std::printf("strata-step35: --gpu-moe needs --token <id>\n");
            return 2;
        }
        return gpu_moe(trunk, (int32_t) token, (int) gpu_layer, (int) pos);
    }
    if (want_gpu) {
        if (token < 0) {
            std::printf("strata-step35: --gpu-block needs --token <id>\n");
            return 2;
        }
        return gpu_block(trunk, (int32_t) token, (int) gpu_layer, (int) pos);
    }
    try {
        return inspect(trunk, mtp);
    } catch (const std::exception& e) {
        std::printf("strata-step35: %s\n", e.what());
        return 1;
    }
}

}  // namespace strata::step35
