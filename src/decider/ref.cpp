// src/decider/ref.cpp - the decider's CPU reference (see ref.hpp).
//
// Plain float arithmetic on the file's weights, dequantized row by row through ggml's type traits.  It is slow on
// purpose: it exists to be right, and to be the thing a kernel is compared against.  Nothing here runs on a GPU.
// The row loop is the only parallelism (OpenMP when the build has it), and every projection keeps its rows
// independent, so the result does not depend on the thread count.
#include "strata/decider/ref.hpp"

#include <ggml.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace strata::decider::ref {
namespace {

using strata::MetaType;
using strata::MetaValue;
using strata::TensorInfo;

[[noreturn]] void fail(const std::string& what) { throw std::runtime_error(what); }

/// The active trace (null unless --trace), so a block can report its own stages while it is being debugged.
TraceFn g_trace;
std::string g_prefix;
StageFn g_stage;          ///< the active stage dump (null unless --dump-dir), for the block functions
int64_t g_stage_block = 0;

void trace_vec(const char* stage, const float* v, size_t n) {
    if (!g_trace) return;
    double sum = 0.0, ss = 0.0;
    for (size_t i = 0; i < n; ++i) {
        sum += v[i];
        ss += (double) v[i] * v[i];
    }
    const std::string name = g_prefix + stage;
    g_trace(name.c_str(), sum, std::sqrt(ss));
}

int64_t get_int(const strata::GgufFile& g, const std::string& key) {
    const MetaValue* v = g.get(key);
    if (!v) fail("missing " + key);
    if (!v->is_num()) fail(key + " is not a number");
    return (int64_t) v->num();
}

/// A float key (the norms' epsilon, the rope base, the calibrated temperature): `get_int` would truncate it to a
/// whole number, and an epsilon of 0 changes every RMSNorm whose variance is below 1e-6 - which is most of them.
double get_num(const strata::GgufFile& g, const std::string& key) {
    const MetaValue* v = g.get(key);
    if (!v) fail("missing " + key);
    if (!v->is_num()) fail(key + " is not a number");
    return v->num();
}

/// A row of a matrix, as f32: the GGUF stores dim 0 (the input width) first and contiguously.
void row_f32(const strata::GgufFile& g, const TensorInfo& t, int64_t row, int64_t n_in, float* out) {
    const uint8_t* base = g.tensor_data(t);
    if (t.type == GGML_TYPE_F32) {
        std::memcpy(out, base + (size_t) row * (size_t) n_in * sizeof(float), (size_t) n_in * sizeof(float));
        return;
    }
    const auto* tr = ggml_get_type_traits((ggml_type) t.type);
    if (!tr || !tr->to_float) fail("no dequantizer for " + t.name);
    const size_t row_bytes = ggml_row_size((ggml_type) t.type, n_in);
    tr->to_float(base + (size_t) row * row_bytes, out, n_in);
}

/// y = W x, with W read row by row: the dequantization of one row is all the memory it takes.
void gemv(const strata::GgufFile& g, const TensorInfo& t, const float* x, float* y) {
    if (t.shape.size() != 2) fail(t.name + " is not a matrix");
    const int64_t n_in = (int64_t) t.shape[0], n_out = (int64_t) t.shape[1];
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        std::vector<float> row((size_t) n_in);
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
        for (int64_t r = 0; r < n_out; ++r) {
            row_f32(g, t, r, n_in, row.data());
            float acc = 0.0f;
            for (int64_t i = 0; i < n_in; ++i) acc += row[(size_t) i] * x[i];
            y[r] = acc;
        }
    }
}

const TensorInfo& tensor(const strata::GgufFile& g, const std::string& name) {
    const TensorInfo* t = g.find(name);
    if (!t) fail("missing tensor " + name);
    return *t;
}

float silu(float x) { return x / (1.0f + std::exp(-x)); }

/// softplus, with the overflow guard the reference gets from `F.softplus`'s threshold.
double softplus(double x) { return x > 20.0 ? x : std::log1p(std::exp(x)); }

/// RMSNorm: out = x / rms(x) * w, with the GGUF's own w.
///
/// THE +1 IS ALREADY IN THE FILE.  Qwen3_5RMSNorm keeps a zero-initialised weight and applies `x * (1 + w)`, and
/// the converter folds that 1 into the stored tensor (llama.cpp's RMSNorm is `x * w`).  The stored mean confirms it
/// (attn_norm 1.14, output_norm 3.20); adding another 1 here doubles every normed vector, which is the first of the
/// silent traps.  ssm_norm is a Qwen3_5RMSNormGated, whose weight is initialised to ones and is stored as is.
void rmsnorm(const float* x, const float* w, int64_t n, double eps, float* out) {
    double ss = 0.0;
    for (int64_t i = 0; i < n; ++i) ss += (double) x[i] * x[i];
    const float scale = (float) (1.0 / std::sqrt(ss / (double) n + eps));
    for (int64_t i = 0; i < n; ++i) out[i] = x[i] * scale * w[i];
}

/// L2 normalisation over `n`, the GDN's own (eps inside the square root).
void l2norm(float* x, int64_t n, double eps) {
    double ss = 0.0;
    for (int64_t i = 0; i < n; ++i) ss += (double) x[i] * x[i];
    const float scale = (float) (1.0 / std::sqrt(ss + eps));
    for (int64_t i = 0; i < n; ++i) x[i] *= scale;
}

/// Partial RoPE, rotate-half pairing: dim i pairs with i + n_rot/2, dims past n_rot are untouched.
void rope(float* v, int64_t n_rot, double base, int64_t pos) {
    const int64_t half = n_rot / 2;
    for (int64_t i = 0; i < half; ++i) {
        const double freq = std::pow(base, -(double) (2 * i) / (double) n_rot);
        const double angle = (double) pos * freq;
        const float c = (float) std::cos(angle), s = (float) std::sin(angle);
        const float a = v[i], b = v[i + half];
        v[i] = a * c - b * s;
        v[i + half] = b * c + a * s;
    }
}

struct Layer {
    const TensorInfo* attn_norm = nullptr;
    const TensorInfo* post_norm = nullptr;
    const TensorInfo* ffn_gate = nullptr;
    const TensorInfo* ffn_up = nullptr;
    const TensorInfo* ffn_down = nullptr;
    // full attention
    const TensorInfo* wq = nullptr;      ///< q and the per-head gate, interleaved
    const TensorInfo* wk = nullptr;
    const TensorInfo* wv = nullptr;
    const TensorInfo* wo = nullptr;
    const TensorInfo* q_norm = nullptr;
    const TensorInfo* k_norm = nullptr;
    // gated delta net
    const TensorInfo* wqkv = nullptr;
    const TensorInfo* wgate = nullptr;   ///< z
    const TensorInfo* walpha = nullptr;  ///< a
    const TensorInfo* wbeta = nullptr;   ///< b
    const TensorInfo* ssm_a = nullptr;
    const TensorInfo* ssm_dt = nullptr;
    const TensorInfo* conv = nullptr;
    const TensorInfo* ssm_norm = nullptr;
    const TensorInfo* ssm_out = nullptr;
};

/// The small per-block vectors, as f32 (a norm, a bias, ssm_a: a few hundred floats).
std::vector<float> small(const strata::GgufFile& g, const TensorInfo& t) {
    int64_t n = 1;
    for (uint64_t d : t.shape) n *= (int64_t) d;
    std::vector<float> out((size_t) n);
    row_f32(g, t, 0, n, out.data());
    return out;
}

/// One full-attention block's token mixer: non-causal attention over the whole sequence, then the gate and o_proj.
void attention(const Model& m, const Layer& l, const std::vector<float>& qn, const std::vector<float>& kn,
               const std::vector<float>& x, int64_t T, std::vector<float>& out) {
    const strata::GgufFile& g = *m.file;
    const int64_t E = m.n_embd, H = m.n_head, KV = m.n_head_kv, D = m.head_dim;
    const int64_t qg_dim = H * D * 2, kv_dim = KV * D;
    const double scale = 1.0 / std::sqrt((double) D);

    std::vector<float> qg((size_t) T * qg_dim), k((size_t) T * kv_dim), v((size_t) T * kv_dim);
    for (int64_t t = 0; t < T; ++t) {
        const float* xt = x.data() + t * E;
        gemv(g, *l.wq, xt, qg.data() + t * qg_dim);
        gemv(g, *l.wk, xt, k.data() + t * kv_dim);
        gemv(g, *l.wv, xt, v.data() + t * kv_dim);
        for (int64_t h = 0; h < H; ++h) {
            float* q = qg.data() + t * qg_dim + h * 2 * D;   // [q(D) | gate(D)]
            rmsnorm(q, qn.data(), D, m.eps, q);
            rope(q, m.n_rot, m.rope_base, t);
        }
        for (int64_t h = 0; h < KV; ++h) {
            float* kt = k.data() + t * kv_dim + h * D;
            rmsnorm(kt, kn.data(), D, m.eps, kt);
            rope(kt, m.n_rot, m.rope_base, t);
        }
    }

    // Non-causal: every position attends to every position (there is no padding: the batch is one sequence).
    std::vector<float> scores((size_t) T);
    out.assign((size_t) T * E, 0.0f);
    std::vector<float> attn((size_t) T * H * D);
    for (int64_t h = 0; h < H; ++h) {
        const int64_t kvh = h / (H / KV);
        for (int64_t t = 0; t < T; ++t) {
            const float* q = qg.data() + t * qg_dim + h * 2 * D;
            float best = -std::numeric_limits<float>::infinity();
            for (int64_t s = 0; s < T; ++s) {
                const float* kt = k.data() + s * kv_dim + kvh * D;
                float acc = 0.0f;
                for (int64_t d = 0; d < D; ++d) acc += q[d] * kt[d];
                scores[(size_t) s] = acc * (float) scale;
                best = std::max(best, scores[(size_t) s]);
            }
            float sum = 0.0f;
            for (int64_t s = 0; s < T; ++s) {
                scores[(size_t) s] = std::exp(scores[(size_t) s] - best);
                sum += scores[(size_t) s];
            }
            float* o = attn.data() + (t * H + h) * D;
            std::fill(o, o + D, 0.0f);
            for (int64_t s = 0; s < T; ++s) {
                const float w = scores[(size_t) s] / sum;
                const float* vt = v.data() + s * kv_dim + kvh * D;
                for (int64_t d = 0; d < D; ++d) o[d] += w * vt[d];
            }
            const float* gate = qg.data() + t * qg_dim + h * 2 * D + D;
            for (int64_t d = 0; d < D; ++d) o[d] *= 1.0f / (1.0f + std::exp(-gate[d]));
        }
    }
    trace_vec(" attn", attn.data(), attn.size());
    if (g_stage) g_stage(g_stage_block, "attn", attn);
    std::vector<float> flat((size_t) H * D);
    for (int64_t t = 0; t < T; ++t) {
        for (int64_t h = 0; h < H; ++h)
            std::memcpy(flat.data() + h * D, attn.data() + (t * H + h) * D, (size_t) D * sizeof(float));
        gemv(g, *l.wo, flat.data(), out.data() + t * E);
    }
    trace_vec(" attn_out", out.data(), out.size());
}

/// One Gated DeltaNet block's token mixer: causal convolution, the gated delta rule, the gated norm, o_proj.
void gated_delta_net(const Model& m, const Layer& l, const std::vector<float>& ssm_norm,
                     const std::vector<float>& x, int64_t T, std::vector<float>& out) {
    const strata::GgufFile& g = *m.file;
    const int64_t E = m.n_embd, KD = m.gdn_kd, VD = m.gdn_vd, KH = m.gdn_kh, VH = m.gdn_vh;
    const int64_t key_dim = KH * KD, value_dim = VH * VD, conv_dim = 2 * key_dim + value_dim;
    const int64_t K = m.conv_k;
    const double scale = 1.0 / std::sqrt((double) KD);

    std::vector<float> mixed((size_t) T * conv_dim), z((size_t) T * value_dim);
    std::vector<float> a((size_t) T * VH), b((size_t) T * VH);
    for (int64_t t = 0; t < T; ++t) {
        const float* xt = x.data() + t * E;
        gemv(g, *l.wqkv, xt, mixed.data() + t * conv_dim);
        gemv(g, *l.wgate, xt, z.data() + t * value_dim);
        gemv(g, *l.walpha, xt, a.data() + t * VH);
        gemv(g, *l.wbeta, xt, b.data() + t * VH);
    }

    // The convolution: tap j = 0 is the oldest, and position t reads t-(K-1) .. t (zeros before the start).
    //
    // THE LAYOUT.  `ssm_conv1d.weight` is [K, conv_dim] in GGUF order, so dim 0 (the tap) is contiguous: the taps
    // of channel c are the four floats at c*K.  Reading it as a row-major [K][conv_dim] matrix (the way a
    // projection is read) takes the wrong floats and still produces plausible numbers - the first version of this
    // reference did exactly that, and its convolution came out 2.5x too small.
    std::vector<float> conv((size_t) T * conv_dim, 0.0f);
    {
        if (l.conv->type != GGML_TYPE_F32) fail("ssm_conv1d.weight must be F32");
        const float* w = (const float*) g.tensor_data(*l.conv);
        for (int64_t t = 0; t < T; ++t) {
            float* dst = conv.data() + t * conv_dim;
            for (int64_t j = 0; j < K; ++j) {
                const int64_t src = t - (K - 1) + j;
                if (src < 0) continue;
                const float* src_row = mixed.data() + src * conv_dim;
                for (int64_t c = 0; c < conv_dim; ++c) dst[c] += w[c * K + j] * src_row[c];
            }
            for (int64_t c = 0; c < conv_dim; ++c) dst[c] = silu(dst[c]);
        }
    }
    trace_vec(" mixed", mixed.data(), mixed.size());
    trace_vec(" conv", conv.data(), conv.size());
    if (g_stage) g_stage(g_stage_block, "conv", conv);

    // The delta rule, one state per value head: S[h] is [KD][VD].
    //
    // WHICH K HEAD A V HEAD READS.  The reference package does `repeat_interleave(2)`, so its v head h reads k head
    // h/2 (interleave).  The GGUF is NOT in that order: the converter (`_reorder_v_heads` in llama.cpp's
    // conversion/qwen.py) rewrites the v rows of in_proj_qkv, in_proj_z, in_proj_a, in_proj_b, out_proj and
    // conv1d from [head_v][dim] to [v_per_k][head_k][dim], so a v head g belongs to k head g % head_k.  Reading the
    // file with the reference's interleave pairs every v head with the wrong k head: the output keeps the right
    // norm and loses the right direction (cosine 0.38 against the oracle, and no error anywhere).
    std::vector<float> state((size_t) VH * KD * VD, 0.0f);
    std::vector<float> q((size_t) KD), k((size_t) KD), vh((size_t) VD), kv((size_t) VD), delta((size_t) VD);
    std::vector<float> core((size_t) T * value_dim);
    std::vector<float> dt = small(g, *l.ssm_dt);
    std::vector<float> A = small(g, *l.ssm_a);
    for (int64_t t = 0; t < T; ++t) {
        const float* row = conv.data() + t * conv_dim;
        for (int64_t h = 0; h < VH; ++h) {
            const int64_t kh = KH > 0 ? h % KH : 0;
            std::memcpy(q.data(), row + kh * KD, (size_t) KD * sizeof(float));
            std::memcpy(k.data(), row + key_dim + kh * KD, (size_t) KD * sizeof(float));
            std::memcpy(vh.data(), row + 2 * key_dim + h * VD, (size_t) VD * sizeof(float));
            l2norm(q.data(), KD, 1e-6);
            l2norm(k.data(), KD, 1e-6);
            for (int64_t i = 0; i < KD; ++i) q[(size_t) i] *= (float) scale;
            const double gate = (double) A[(size_t) h] * softplus((double) a[(size_t) t * VH + h] + dt[(size_t) h]);
            const float decay = (float) std::exp(gate);
            const float beta = 1.0f / (1.0f + std::exp(-b[(size_t) t * VH + h]));
            float* S = state.data() + (size_t) h * KD * VD;
            for (int64_t i = 0; i < KD * VD; ++i) S[i] *= decay;
            for (int64_t j = 0; j < VD; ++j) kv[(size_t) j] = 0.0f;
            for (int64_t i = 0; i < KD; ++i) {
                const float ki = k[(size_t) i];
                if (ki == 0.0f) continue;
                for (int64_t j = 0; j < VD; ++j) kv[(size_t) j] += S[i * VD + j] * ki;
            }
            for (int64_t j = 0; j < VD; ++j) delta[(size_t) j] = (vh[(size_t) j] - kv[(size_t) j]) * beta;
            for (int64_t i = 0; i < KD; ++i) {
                const float ki = k[(size_t) i];
                if (ki == 0.0f) continue;
                for (int64_t j = 0; j < VD; ++j) S[i * VD + j] += ki * delta[(size_t) j];
            }
            float* o = core.data() + t * value_dim + h * VD;
            for (int64_t j = 0; j < VD; ++j) o[j] = 0.0f;
            for (int64_t i = 0; i < KD; ++i) {
                const float qi = q[(size_t) i];
                if (qi == 0.0f) continue;
                for (int64_t j = 0; j < VD; ++j) o[j] += S[i * VD + j] * qi;
            }
        }
    }

    trace_vec(" core", core.data(), core.size());
    if (g_stage) g_stage(g_stage_block, "core", core);

    trace_vec(" z", z.data(), z.size());
    // The gated norm, per value head, then o_proj.
    std::vector<float> flat((size_t) value_dim);
    for (int64_t t = 0; t < T; ++t) {
        for (int64_t h = 0; h < VH; ++h) {
            float* o = core.data() + t * value_dim + h * VD;
            rmsnorm(o, ssm_norm.data(), VD, m.eps, o);
            const float* zt = z.data() + t * value_dim + h * VD;
            for (int64_t j = 0; j < VD; ++j) o[j] *= silu(zt[j]);
        }
        std::memcpy(flat.data(), core.data() + t * value_dim, (size_t) value_dim * sizeof(float));
        gemv(g, *l.ssm_out, flat.data(), out.data() + t * E);
    }
    trace_vec(" gdn_out", out.data(), out.size());
}

}  // namespace

bool load(const std::string& path, Model& m, std::string& err) {
    try {
        m.file = std::make_unique<strata::GgufFile>(strata::gguf_split_paths(path).front());
        const strata::GgufFile& g = *m.file;
        const std::string why = strata::check_architecture_qwen35(g);
        if (!why.empty()) {
            err = why;
            return false;
        }
        m.n_layers = get_int(g, "qwen35.block_count");
        m.n_embd = get_int(g, "qwen35.embedding_length");
        m.n_ff = get_int(g, "qwen35.feed_forward_length");
        m.n_head = get_int(g, "qwen35.attention.head_count");
        m.n_head_kv = get_int(g, "qwen35.attention.head_count_kv");
        m.head_dim = get_int(g, "qwen35.attention.key_length");
        m.n_rot = get_int(g, "qwen35.rope.dimension_count");
        m.kv_dim = m.n_head_kv * m.head_dim;
        m.gdn_kh = get_int(g, "qwen35.ssm.group_count");
        m.gdn_vh = get_int(g, "qwen35.ssm.time_step_rank");
        m.gdn_kd = get_int(g, "qwen35.ssm.state_size");
        m.gdn_vd = get_int(g, "qwen35.ssm.state_size");
        m.conv_k = get_int(g, "qwen35.ssm.conv_kernel");
        m.conv_dim = 2 * m.gdn_kh * m.gdn_kd + m.gdn_vh * m.gdn_vd;
        m.eps = get_num(g, "qwen35.attention.layer_norm_rms_epsilon");
        m.rope_base = get_num(g, "qwen35.rope.freq_base");
        m.temperature = get_num(g, "pplx_decider.temperature");
        const MetaValue* tokens = g.get("tokenizer.ggml.tokens");
        if (!tokens || tokens->type != MetaType::ARRAY) fail("missing tokenizer.ggml.tokens");
        m.n_vocab = (int64_t) tokens->count;
        const MetaValue* codes = g.get("pplx_decider.token_ids");
        if (codes->items.size() != (size_t) codes->count)
            fail("the reader kept " + std::to_string(codes->items.size()) + " of the " +
                 std::to_string(codes->count) + " readout codes: the head would answer over a prefix only");
        for (const MetaValue& id : codes->items) m.codes.push_back((int64_t) id.u);
        const MetaValue* recurrent = g.get("qwen35.attention.recurrent_layers");
        for (const MetaValue& v : recurrent->items) m.recurrent.push_back(v.u ? 1 : 0);
        return true;
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
}

bool logits(const Model& m, const std::vector<int32_t>& tokens, const std::vector<int32_t>& markers,
            std::vector<std::vector<float>>& out, std::string& err, const TraceFn& trace, const StageFn& stage) {
    try {
        const strata::GgufFile& g = *m.file;
        const int64_t T = (int64_t) tokens.size(), E = m.n_embd, FF = m.n_ff;
        if (T == 0) fail("no tokens");
        for (int32_t id : tokens)
            if (id < 0 || id >= m.n_vocab) fail("token " + std::to_string(id) + " is outside the vocabulary");
        for (int32_t p : markers)
            if (p < 0 || p >= T) fail("marker " + std::to_string(p) + " is outside the sequence");

        g_trace = trace;
        g_stage = stage;
        auto report = [&](const char* stage, const std::vector<float>& v) {
            if (!trace) return;
            double sum = 0.0, ss = 0.0;
            for (float value : v) {
                sum += value;
                ss += (double) value * value;
            }
            trace(stage, sum, std::sqrt(ss));
        };
        std::vector<float> x((size_t) T * E);
        {
            const std::string emb_name = "token_embd.weight";
            const TensorInfo& emb = tensor(g, emb_name);
            for (int64_t t = 0; t < T; ++t) row_f32(g, emb, tokens[(size_t) t], E, x.data() + t * E);
        }
        report("embd", x);
        const std::vector<float> output_norm = small(g, tensor(g, "output_norm.weight"));

        std::vector<float> normed((size_t) T * E), mixed_out((size_t) T * E), tmp((size_t) T * E);
        for (int64_t l = 0; l < m.n_layers; ++l) {
            const std::string p = "blk." + std::to_string(l) + ".";
            g_prefix = "L" + std::to_string(l);
            g_stage_block = l;
            Layer L;
            L.attn_norm = &tensor(g, p + "attn_norm.weight");
            L.post_norm = &tensor(g, p + "post_attention_norm.weight");
            L.ffn_gate = &tensor(g, p + "ffn_gate.weight");
            L.ffn_up = &tensor(g, p + "ffn_up.weight");
            L.ffn_down = &tensor(g, p + "ffn_down.weight");
            const std::vector<float> attn_norm = small(g, *L.attn_norm);
            const std::vector<float> post_norm = small(g, *L.post_norm);
            for (int64_t t = 0; t < T; ++t)
                rmsnorm(x.data() + t * E, attn_norm.data(), E, m.eps, normed.data() + t * E);

            trace_vec(" xn", normed.data(), normed.size());
            if (stage) stage(l, "xn_attn", normed);
            if (m.recurrent[(size_t) l]) {
                L.wqkv = &tensor(g, p + "attn_qkv.weight");
                L.wgate = &tensor(g, p + "attn_gate.weight");
                L.walpha = &tensor(g, p + "ssm_alpha.weight");
                L.wbeta = &tensor(g, p + "ssm_beta.weight");
                L.ssm_a = &tensor(g, p + "ssm_a");
                L.ssm_dt = &tensor(g, p + "ssm_dt.bias");
                L.conv = &tensor(g, p + "ssm_conv1d.weight");
                L.ssm_norm = &tensor(g, p + "ssm_norm.weight");
                L.ssm_out = &tensor(g, p + "ssm_out.weight");
                const std::vector<float> sn = small(g, *L.ssm_norm);
                gated_delta_net(m, L, sn, normed, T, mixed_out);
            } else {
                L.wq = &tensor(g, p + "attn_q.weight");
                L.wk = &tensor(g, p + "attn_k.weight");
                L.wv = &tensor(g, p + "attn_v.weight");
                L.wo = &tensor(g, p + "attn_output.weight");
                L.q_norm = &tensor(g, p + "attn_q_norm.weight");
                L.k_norm = &tensor(g, p + "attn_k_norm.weight");
                const std::vector<float> qn = small(g, *L.q_norm);
                const std::vector<float> kn = small(g, *L.k_norm);
                attention(m, L, qn, kn, normed, T, mixed_out);
            }
            if (stage) stage(l, "mixer", mixed_out);
            for (int64_t i = 0; i < T * E; ++i) x[(size_t) i] += mixed_out[(size_t) i];
            if (stage) stage(l, "x_post", x);   // the MLP's input before its norm: a kernel test needs the raw one

            // the dense SwiGLU, on the post-attention norm
            for (int64_t t = 0; t < T; ++t)
                rmsnorm(x.data() + t * E, post_norm.data(), E, m.eps, normed.data() + t * E);
            if (stage) stage(l, "xn_post", normed);
            std::vector<float> gate((size_t) FF), up((size_t) FF);
            for (int64_t t = 0; t < T; ++t) {
                gemv(g, *L.ffn_gate, normed.data() + t * E, gate.data());
                gemv(g, *L.ffn_up, normed.data() + t * E, up.data());
                for (int64_t i = 0; i < FF; ++i) gate[(size_t) i] = silu(gate[(size_t) i]) * up[(size_t) i];
                gemv(g, *L.ffn_down, gate.data(), tmp.data() + t * E);
            }
            trace_vec(" mlp", tmp.data(), tmp.size());
            if (stage) stage(l, "mlp", tmp);
            for (int64_t i = 0; i < T * E; ++i) x[(size_t) i] += tmp[(size_t) i];
            const std::string stage = "block " + std::to_string(l);
            report(stage.c_str(), x);
        }

        std::vector<float> hidden((size_t) E);
        out.clear();
        const std::string readout_name = "output.weight";
        const TensorInfo& readout = tensor(g, readout_name);
        std::vector<float> row((size_t) E);
        for (size_t index = 0; index < markers.size(); ++index) {
            const int32_t marker = markers[index];
            rmsnorm(x.data() + (size_t) marker * E, output_norm.data(), E, m.eps, hidden.data());
            if (index == 0) report("final", hidden);
            std::vector<float> values;
            values.reserve(m.codes.size());
            for (int64_t id : m.codes) {
                row_f32(g, readout, id, E, row.data());
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

}  // namespace strata::decider::ref
