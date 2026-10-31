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
#include "strata/step35/gemv.hpp"
#include "strata/step35/session.hpp"

#include <ggml.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

/// `--gpu-chat`: a prompt, then decode steps that reuse its K/V through the quantized cache.  The figures printed
/// per position are the ones `--ref` prints, so the two outputs are the comparison: prefill n positions, decode m,
/// and `--ref --tokens "<n + m ids>"` answers for the same sequence in float64.
int gpu_chat(const std::string& path, const std::string& tokens_csv, const std::string& next_csv, int n_blocks,
             int max_ctx, bool want_logits) {
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
        if (!kernels::session_create(s, n_blocks, n_head_kv, head_dim, capacity, err, windows)) {
            std::printf("strata-step35: %s\n", err.c_str());
            return 1;
        }
        int n_full = 0, n_ring = 0, ring_cells = 0;
        for (int l = 0; l < n_blocks; ++l) {
            const int c = kernels::session_layer_cells(s, l);
            if (windows[(size_t) l] > 0) { ++n_ring; ring_cells = c; } else { ++n_full; }
        }
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
        for (size_t off = 0; off < prompt.size(); off += (size_t) kernels::kKvPassMax) {
            const size_t n = std::min((size_t) kernels::kKvPassMax, prompt.size() - off);
            const std::vector<int32_t> chunk(prompt.begin() + off, prompt.begin() + off + n);
            std::vector<float> part;
            std::vector<double> csum;
            if (!kernels::run_prompt_dense(file, chunk, (int) off, n_blocks, part, csum, err, &s)) {
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
        std::printf("strata-step35: prefill of %zu position(s) through %d block(s)\n", prompt.size(), n_blocks);
        for (size_t l = 0; l < sums.size(); ++l) std::printf("  block %2zu : sum %.6f\n", l, sums[l]);
        std::vector<float> rows = out;   // [position][n_embd], the figures --ref prints

        // ---- the decode steps: ONE position each, attending to the cells the prompt left behind
        int pos = (int) prompt.size();
        for (size_t d = 0; d < next.size(); ++d) {
            std::vector<int32_t> one(1, next[d]);
            std::vector<float> dout;
            std::vector<double> dsums;
            if (!kernels::run_prompt_dense(file, one, pos, n_blocks, dout, dsums, err, &s)) {
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
        "  strata-step35 --inspect <first shard> [--mtp <nextn file>]\n"
        "      Read the file, print the geometry it declares and check every tensor against it.  No GPU.\n"
        "\n"
        "  strata-step35 --gpu-block <first shard> --layer N --token <id> [--pos N]\n"
        "      One dense block on the GPU with the file's own weights, for one token.  Prints the output sum.\n"
        "\n"
        "  strata-step35 --ref <first shard> --tokens <id,id,...> [--layers N]\n"
        "      Run the CPU reference over the blocks (default: all of them, 3 dense then 42 MoE) and print\n"
        "      what each one produced.  The reference a kernel is compared against; no GPU.\n"
        "\n"
        "  strata-step35 --gpu-chat <first shard> --tokens <id,id,...> [--next <id,id,...>] [--layers N]\n"
        "      [--max-context N] [--logits]\n"
        "      Prefill the prompt with the quantized K/V cache, then decode each --next token against it.  Prints\n"
        "      the per-block sums and, per position, the rms and first four values --ref prints: the two outputs\n"
        "      are the comparison the cache is held to.  `--logits` adds the file's own output_norm and output\n"
        "      applied to the last position, and the five tokens it would answer with - the figure `--ref`\n"
        "      prints, so the head is held to the reference the same way the blocks are.  --gpu-seq takes it too.\n"
        "\n"
        "There is no forward pass and nothing to serve in this build: --gpu-chat runs a prompt and its decode\n"
        "steps, and no process answers requests.  When the forward pass lands the flags will be strata's own\n"
        "(--model, --max-context, --tokens, --max-new, --serve) plus --mtp for the NextN file.\n");
}

/// `--ref`: the CPU reference over the dense blocks, for a token sequence given by hand.
int reference(const std::string& path, const std::string& tokens_csv, int64_t layers, const std::string& dump) {
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
    return 0;
}

}  // namespace

int run(int argc, char** argv) {
    std::string trunk, mtp, tokens, dump, next;
    int64_t layers = 0, gpu_layer = -1, token = -1, pos = 0, max_ctx = 8192;
    bool want_ref = false, want_gpu = false, want_moe = false, want_chain = false, want_all = false, want_seq = false;
    bool want_chat = false, want_logits = false;
    for (int i = 1; i < argc; ++i) {
        // `--logits` is a switch; everything else takes a value, so a flag with nothing after it is an error and
        // not a silent no-op (the loop used to stop at argc-1, which swallowed a trailing bare flag).
        if (std::strcmp(argv[i], "--logits") != 0 && i + 1 >= argc) {
            std::printf("strata-step35: %s needs a value\n", argv[i]);
            return 2;
        }
        if (std::strcmp(argv[i], "--logits") == 0) {
            want_logits = true;
        } else if (std::strcmp(argv[i], "--inspect") == 0 || std::strcmp(argv[i], "--model") == 0 ||
            std::strcmp(argv[i], "--native") == 0) {
            trunk = argv[++i];
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
        }
    }
    if (trunk.empty()) {
        usage();
        return 2;
    }
    if (want_ref) return reference(trunk, tokens, layers, dump);
    if (want_chat) return gpu_chat(trunk, tokens, next, (int) layers, (int) max_ctx, want_logits);
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
