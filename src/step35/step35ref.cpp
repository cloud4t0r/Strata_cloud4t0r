// src/step35/step35ref.cpp - the Step-3.7-Flash CPU reference (see step35ref.hpp).
//
// Plain double-precision arithmetic on weights dequantized through ggml's type traits.  It is slow on purpose:
// it exists to be right, and to be the thing a kernel is compared against.  Nothing here runs on a GPU.
#include "step35ref.hpp"

#include <ggml.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace strata::step35::ref {
namespace {

using strata::GgufModel;
using strata::MetaType;
using strata::MetaValue;
using strata::TensorInfo;

double get_num(const GgufModel& g, const std::string& key, bool required, double fallback = 0) {
    const MetaValue* v = g.meta().get(key);
    if (!v) {
        if (required) throw std::runtime_error("missing " + key);
        return fallback;
    }
    if (!v->is_num()) throw std::runtime_error(key + " is not a number");
    return v->num();
}

std::vector<double> get_arr(const GgufModel& g, const std::string& key, int64_t want, bool required = true) {
    std::vector<double> out;
    const MetaValue* v = g.meta().get(key);
    if (!v) {
        if (required) throw std::runtime_error("missing " + key);
        return out;
    }
    if (v->type != MetaType::ARRAY) throw std::runtime_error(key + " is not an array");
    if ((int64_t) v->count != want) {
        throw std::runtime_error(key + " has " + std::to_string(v->count) + " entries, expected " +
                                 std::to_string(want) + " (the reader samples the first 64)");
    }
    for (const MetaValue& item : v->items) out.push_back(item.num());
    return out;
}

const TensorInfo& tensor(const GgufModel& g, const std::string& name, size_t& shard) {
    const TensorInfo* t = g.find(name, &shard);
    if (!t) throw std::runtime_error("missing tensor " + name);
    return *t;
}

/// A row of a quantized matrix, as f32: the GGUF stores dim 0 (the input width) first and contiguously.
void row_f32(const GgufModel& g, const TensorInfo& t, size_t shard, int64_t row, int64_t n_in, float* out) {
    const uint8_t* base = (const uint8_t*) g.shard(shard).tensor_data(t);
    if (t.type == GGML_TYPE_F32) {   // a plain tensor has no dequantizer: the bytes are the values
        std::memcpy(out, base + (size_t) row * (size_t) n_in * sizeof(float), (size_t) n_in * sizeof(float));
        return;
    }
    const auto* tr = ggml_get_type_traits((ggml_type) t.type);
    if (!tr || !tr->to_float) throw std::runtime_error("no dequantizer for " + t.name);
    const size_t row_bytes = ggml_row_size((ggml_type) t.type, n_in);
    tr->to_float(base + (size_t) row * row_bytes, out, n_in);
}

/// y = W x, with W read row by row: the dequantization of one row is all the memory it takes.
std::vector<float> gemv(const GgufModel& g, const std::string& name, const std::vector<float>& x) {
    size_t shard = 0;
    const TensorInfo& t = tensor(g, name, shard);
    if (t.shape.size() != 2) throw std::runtime_error(name + " is not a matrix");
    const int64_t n_in = (int64_t) t.shape[0], n_out = (int64_t) t.shape[1];
    if ((int64_t) x.size() != n_in) {
        throw std::runtime_error(name + " takes " + std::to_string(n_in) + " inputs, got " + std::to_string(x.size()));
    }
    std::vector<float> y((size_t) n_out);
    std::vector<float> row((size_t) n_in);
    for (int64_t r = 0; r < n_out; ++r) {
        row_f32(g, t, shard, r, n_in, row.data());
        double acc = 0;
        for (int64_t k = 0; k < n_in; ++k) acc += (double) row[(size_t) k] * (double) x[(size_t) k];
        y[(size_t) r] = (float) acc;
    }
    return y;
}

/// y = expert e's matrix times x, for a 3-D expert tensor [n_in, n_out, n_expert] (dim 0 fastest, so expert e's
/// block is contiguous).
std::vector<float> gemv_expert(const GgufModel& g, const std::string& name, const std::vector<float>& x, int64_t e) {
    size_t shard = 0;
    const TensorInfo& t = tensor(g, name, shard);
    if (t.shape.size() != 3) throw std::runtime_error(name + " is not a 3-D expert tensor");
    const int64_t n_in = (int64_t) t.shape[0], n_out = (int64_t) t.shape[1], n_ex = (int64_t) t.shape[2];
    if (e < 0 || e >= n_ex) throw std::runtime_error(name + ": expert " + std::to_string(e) + " is out of range");
    if ((int64_t) x.size() != n_in) throw std::runtime_error(name + " takes " + std::to_string(n_in) + " inputs");
    const size_t row_bytes = ggml_row_size((ggml_type) t.type, n_in);
    const uint8_t* base = (const uint8_t*) g.shard(shard).tensor_data(t) + (size_t) e * (size_t) n_out * row_bytes;
    std::vector<float> y((size_t) n_out), row((size_t) n_in);
    const auto* tr = ggml_get_type_traits((ggml_type) t.type);
    for (int64_t r = 0; r < n_out; ++r) {
        if (t.type == GGML_TYPE_F32) {
            std::memcpy(row.data(), base + (size_t) r * row_bytes, (size_t) n_in * sizeof(float));
        } else {
            if (!tr || !tr->to_float) throw std::runtime_error("no dequantizer for " + t.name);
            tr->to_float(base + (size_t) r * row_bytes, row.data(), n_in);
        }
        double acc = 0;
        for (int64_t k = 0; k < n_in; ++k) acc += (double) row[(size_t) k] * (double) x[(size_t) k];
        y[(size_t) r] = (float) acc;
    }
    return y;
}

std::vector<float> vector_of(const GgufModel& g, const std::string& name) {
    size_t shard = 0;
    const TensorInfo& t = tensor(g, name, shard);
    if (t.shape.size() != 1) throw std::runtime_error(name + " is not a vector");
    std::vector<float> out((size_t) t.shape[0]);
    row_f32(g, t, shard, 0, (int64_t) t.shape[0], out.data());
    return out;
}

void rms_norm(std::vector<float>& x, const std::vector<float>& w, double eps) {
    double ss = 0;
    for (float v : x) ss += (double) v * (double) v;
    const double scale = 1.0 / std::sqrt(ss / (double) x.size() + eps);
    for (size_t i = 0; i < x.size(); ++i) x[i] = (float) ((double) x[i] * scale * (double) w[i]);
}

float silu(float v) { return (float) ((double) v / (1.0 + std::exp(-(double) v))); }

/// RoPE on one head: the first `n_rot` of its dims turn, the rest are left alone.
///
/// Pairing is NEOX - element j turns with element j + n_rot/2 - because llama_model_rope_type() returns
/// LLAMA_ROPE_TYPE_NEOX for LLM_ARCH_STEP35 (src/llama-model.cpp), and ggml's rotate_pairs reads it as
/// (i0, i0 + n_dims/2) with i0 stepping by 2 over the first half.  The angle of pair j is
/// pos * base^(-2j/n_rot), divided by `ff[j]` when the model's rope_freqs table is given (llama3 scaling,
/// ggml-cpu ops.cpp:6194).  Adjacent pairing (2i, 2i+1) is the NORM convention and is WRONG here; at the
/// positions of a short test the two differ by little, which is why this is written down.
void rope(float* v, int64_t head_dim, int64_t n_rot, int64_t pos, double base, const std::vector<float>* ff) {
    const int64_t half = n_rot / 2;
    for (int64_t j = 0; j < half; ++j) {
        const double freq = 1.0 / std::pow(base, (double) (2 * j) / (double) n_rot);
        double theta = (double) pos * freq;
        if (ff && (size_t) j < ff->size() && (*ff)[(size_t) j] != 0.0f) {
            theta /= (double) (*ff)[(size_t) j];
        }
        const double c = std::cos(theta), s = std::sin(theta);
        const double a = v[j], b = v[j + half];
        v[j] = (float) (a * c - b * s);
        v[j + half] = (float) (a * s + b * c);
    }
    (void) head_dim;
}

struct Layer {
    std::vector<float> k, v;   // [n_pos][kv_heads*head_dim]
};

}  // namespace

bool load(const std::string& path, Model& m, std::string& err) {
    try {
        m.file = std::make_shared<GgufModel>(GgufModel::open(path));
        const GgufModel& f = *m.file;
        const std::string p = "step35.";
        m.n_layers = (int64_t) get_num(f, p + "block_count", true);
        m.n_embd = (int64_t) get_num(f, p + "embedding_length", true);
        m.head_dim = (int64_t) get_num(f, p + "attention.key_length", true);
        m.window = (int64_t) get_num(f, p + "attention.sliding_window", true);
        m.ff_dense = (int64_t) get_num(f, p + "feed_forward_length", true);
        m.n_expert = (int64_t) get_num(f, p + "expert_count", true);
        m.top_k = (int64_t) get_num(f, p + "expert_used_count", true);
        m.ff_expert = (int64_t) get_num(f, p + "expert_feed_forward_length", true);
        m.ff_shexp = (int64_t) get_num(f, p + "expert_shared_feed_forward_length", true);
        m.dense_layers = (int64_t) get_num(f, p + "leading_dense_block_count", true);
        m.eps = get_num(f, p + "attention.layer_norm_rms_epsilon", true);
        m.rope_full = get_num(f, p + "rope.freq_base", true);
        m.rope_swa = get_num(f, p + "rope.freq_base_swa", true);
        m.expert_scale = get_num(f, p + "expert_weights_scale", true);
        m.n_head = get_arr(f, p + "attention.head_count", m.n_layers);
        m.n_head_kv = get_arr(f, p + "attention.head_count_kv", m.n_layers);
        m.is_swa = get_arr(f, p + "attention.sliding_window_pattern", m.n_layers);
        m.clamp_exp = get_arr(f, p + "swiglu_clamp_exp", m.n_layers, false);
        m.clamp_shexp = get_arr(f, p + "swiglu_clamp_shexp", m.n_layers, false);
        const MetaValue* norm = f.meta().get(p + "expert_weights_norm");
        m.expert_norm = norm && norm->type == MetaType::BOOL && norm->u != 0;
        // llama.cpp's rule for this file (neither key is in it): a sliding block rotates head_dim dims, a full
        // block half of them
        m.n_rot_swa = m.head_dim;
        m.n_rot_full = m.head_dim / 2;
        if (f.find("rope_freqs.weight")) m.rope_freqs = vector_of(f, "rope_freqs.weight");
        const MetaValue* tok = f.meta().get("tokenizer.ggml.tokens");
        if (!tok || tok->type != MetaType::ARRAY) throw std::runtime_error("missing tokenizer.ggml.tokens");
        m.n_vocab = (int64_t) tok->count;
        return true;
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
}

bool forward(const Model& m, const std::vector<int32_t>& tokens, int64_t layers, std::vector<float>& out,
             std::string& err, std::vector<double>* trace) {
    try {
        const GgufModel& f = *m.file;
        if (layers > m.n_layers) {
            err = "the file has " + std::to_string(m.n_layers) + " blocks, " + std::to_string(layers) + " asked";
            return false;
        }
        const int64_t n_pos = (int64_t) tokens.size(), e = m.n_embd;
        // the embeddings, one row each
        std::vector<float> x((size_t) (n_pos * e));
        {
            size_t shard = 0;
            const TensorInfo& t = tensor(f, "token_embd.weight", shard);
            for (int64_t p = 0; p < n_pos; ++p) {
                if (tokens[(size_t) p] < 0 || tokens[(size_t) p] >= m.n_vocab) {
                    err = "token " + std::to_string(tokens[(size_t) p]) + " is outside the vocabulary";
                    return false;
                }
                row_f32(f, t, shard, tokens[(size_t) p], e, x.data() + p * e);
            }
        }
        std::vector<std::vector<Layer>> kv((size_t) layers);
        for (int64_t L = 0; L < layers; ++L) {
            const std::string b = "blk." + std::to_string(L) + ".";
            const auto attn_norm_w = vector_of(f, b + "attn_norm.weight");
            const auto q_norm_w = vector_of(f, b + "attn_q_norm.weight");
            const auto k_norm_w = vector_of(f, b + "attn_k_norm.weight");
            const auto ffn_norm_w = vector_of(f, b + "ffn_norm.weight");
            kv[(size_t) L].resize((size_t) n_pos);
            const int64_t nh = m.heads(L), nkv = m.kv_heads(L), hd = m.head_dim;
            const int64_t rot = m.rot(L);
            const double base = m.rope_base(L);
            for (int64_t p = 0; p < n_pos; ++p) {
                std::vector<float> h(x.begin() + p * e, x.begin() + (p + 1) * e);
                std::vector<float> hn = h;
                rms_norm(hn, attn_norm_w, m.eps);
                auto q = gemv(f, b + "attn_q.weight", hn);
                auto k = gemv(f, b + "attn_k.weight", hn);
                auto v = gemv(f, b + "attn_v.weight", hn);
                auto per_head_norm = [&](std::vector<float>& v, int64_t heads_n, const std::vector<float>& w) {
                    for (int64_t hh = 0; hh < heads_n; ++hh) {
                        float* r = v.data() + hh * hd;
                        double ss = 0;
                        for (int64_t d = 0; d < hd; ++d) ss += (double) r[d] * (double) r[d];
                        const double sc = 1.0 / std::sqrt(ss / (double) hd + m.eps);
                        for (int64_t d = 0; d < hd; ++d) r[d] = (float) ((double) r[d] * sc * (double) w[(size_t) d]);
                    }
                };
                per_head_norm(q, nh, q_norm_w);
                per_head_norm(k, nkv, k_norm_w);
                // llama.cpp: the frequency factors go to the FULL-attention blocks and nowhere else
                const std::vector<float>* ff = m.sliding(L) ? nullptr : &m.rope_freqs;
                if (ff && ff->empty()) ff = nullptr;
                for (int64_t hh = 0; hh < nh; ++hh) rope(q.data() + hh * hd, hd, rot, p, base, ff);
                for (int64_t hh = 0; hh < nkv; ++hh) rope(k.data() + hh * hd, hd, rot, p, base, ff);
                kv[(size_t) L][(size_t) p].k = k;
                kv[(size_t) L][(size_t) p].v = v;
                // attention: causal, and on a sliding block only the last `window` positions
                for (int64_t hh = 0; hh < nh; ++hh) {
                    const int64_t g = hh / (nh / nkv ? (nh / nkv) : 1);
                    int64_t first = 0;
                    if (m.sliding(L) && p - first + 1 > m.window) first = p - m.window + 1;
                    float* qh = q.data() + hh * hd;          // read as the query, then written with the result
                    std::vector<double> score((size_t) (p - first + 1));
                    double best = -1e30;
                    for (int64_t j = first; j <= p; ++j) {
                        const float* kj = kv[(size_t) L][(size_t) j].k.data() + g * hd;
                        double acc = 0;
                        for (int64_t d = 0; d < hd; ++d) acc += (double) qh[d] * (double) kj[d];
                        acc /= std::sqrt((double) hd);
                        score[(size_t) (j - first)] = acc;
                        if (acc > best) best = acc;
                    }
                    double sum = 0;
                    for (double& s : score) {
                        s = std::exp(s - best);
                        sum += s;
                    }
                    std::vector<double> acc((size_t) hd, 0.0);
                    for (int64_t j = first; j <= p; ++j) {
                        const double w = score[(size_t) (j - first)] / sum;
                        const float* vj = kv[(size_t) L][(size_t) j].v.data() + g * hd;
                        for (int64_t d = 0; d < hd; ++d) acc[(size_t) d] += w * (double) vj[d];
                    }
                    for (int64_t d = 0; d < hd; ++d) qh[d] = (float) acc[(size_t) d];
                }
                // the head-wise gate: attn *= sigmoid(g_proj(x)), one weight per head
                auto gate = gemv(f, b + "attn_gate.weight", hn);
                for (int64_t hh = 0; hh < nh; ++hh) {
                    const double sig = 1.0 / (1.0 + std::exp(-(double) gate[(size_t) hh]));
                    for (int64_t d = 0; d < hd; ++d) q[(size_t) (hh * hd + d)] = (float) ((double) q[(size_t) (hh * hd + d)] * sig);
                }
                auto attn = gemv(f, b + "attn_output.weight", q);
                for (int64_t i = 0; i < e; ++i) h[(size_t) i] += attn[(size_t) i];
                // the feed-forward: a dense MLP on the leading blocks, a mixture of experts on the rest
                std::vector<float> fn = h;
                rms_norm(fn, ffn_norm_w, m.eps);
                const double clamp = m.clamp_exp.size() > (size_t) L ? m.clamp_exp[(size_t) L] : 0.0;
                if (!m.moe(L)) {
                    auto g_ = gemv(f, b + "ffn_gate.weight", fn);
                    auto u_ = gemv(f, b + "ffn_up.weight", fn);
                    std::vector<float> act((size_t) m.ff_dense);
                    for (int64_t i = 0; i < m.ff_dense; ++i) {
                        double gv = silu(g_[(size_t) i]), uv = u_[(size_t) i];
                        if (clamp > 1e-6) {           // llama.cpp's order: up clamped both ways, silu clamped above
                            if (uv > clamp) uv = clamp;
                            if (uv < -clamp) uv = -clamp;
                            if (gv > clamp) gv = clamp;
                        }
                        act[(size_t) i] = (float) (gv * uv);
                    }
                    auto down = gemv(f, b + "ffn_down.weight", act);
                    for (int64_t i = 0; i < e; ++i) h[(size_t) i] += down[(size_t) i];
                } else {
                    // ---- the router: sigmoid probs, a selection bias that does NOT enter the weights, top-k,
                    // normalize the chosen probs, then scale (llama.cpp's build_moe_ffn, SIGMOID branch)
                    const auto logits = gemv(f, b + "ffn_gate_inp.weight", fn);
                    std::vector<float> bias;
                    if (f.find(b + "exp_probs_b.bias")) bias = vector_of(f, b + "exp_probs_b.bias");
                    std::vector<double> probs((size_t) m.n_expert);
                    for (int64_t i = 0; i < m.n_expert; ++i) {
                        probs[(size_t) i] = 1.0 / (1.0 + std::exp(-(double) logits[(size_t) i]));
                    }
                    // top-k, largest selection score first, TIES TO THE LOWEST INDEX - the same rule the device
                    // kernel follows, so the two can be compared exactly and two runs never differ
                    std::vector<int64_t> order;
                    std::vector<char> taken((size_t) m.n_expert, 0);
                    for (int64_t k = 0; k < m.top_k; ++k) {
                        int64_t best_i = -1;
                        double best = -1e300;
                        for (int64_t e = 0; e < m.n_expert; ++e) {
                            if (taken[(size_t) e]) continue;
                            const double sel = probs[(size_t) e] + (bias.empty() ? 0.0 : (double) bias[(size_t) e]);
                            if (sel > best || (sel == best && (best_i < 0 || e < best_i))) {
                                best = sel;
                                best_i = e;
                            }
                        }
                        if (best_i < 0) break;
                        taken[(size_t) best_i] = 1;
                        order.push_back(best_i);
                    }
                    std::vector<double> w((size_t) m.top_k);
                    double sum = 0;
                    for (int64_t k = 0; k < m.top_k; ++k) {
                        w[(size_t) k] = probs[(size_t) order[(size_t) k]];
                        sum += w[(size_t) k];
                    }
                    if (m.expert_norm) {
                        const double den = sum < 6.103515625e-5 ? 6.103515625e-5 : sum;   // llama.cpp's clamp
                        for (double& x2 : w) x2 /= den;
                    }
                    if (m.expert_scale != 0.0 && m.expert_scale != 1.0) {
                        for (double& x2 : w) x2 *= m.expert_scale;
                    }
                    const double clamp_sh = m.clamp_shexp.size() > (size_t) L ? m.clamp_shexp[(size_t) L] : 0.0;
                    std::vector<double> acc((size_t) e, 0.0);
                    std::vector<float> act((size_t) m.ff_expert);
                    for (int64_t k = 0; k < m.top_k; ++k) {
                        const int64_t ex = order[(size_t) k];
                        const auto ge = gemv_expert(f, b + "ffn_gate_exps.weight", fn, ex);
                        const auto ue = gemv_expert(f, b + "ffn_up_exps.weight", fn, ex);
                        for (int64_t i = 0; i < m.ff_expert; ++i) {
                            double gv = silu(ge[(size_t) i]), uv = ue[(size_t) i];
                            if (clamp > 1e-6) {
                                if (uv > clamp) uv = clamp;
                                if (uv < -clamp) uv = -clamp;
                                if (gv > clamp) gv = clamp;
                            }
                            act[(size_t) i] = (float) (gv * uv);
                        }
                        const auto de = gemv_expert(f, b + "ffn_down_exps.weight", act, ex);
                        for (int64_t i = 0; i < e; ++i) acc[(size_t) i] += w[(size_t) k] * (double) de[(size_t) i];
                    }
                    // the shared expert: added, never weighted
                    const auto sg = gemv(f, b + "ffn_gate_shexp.weight", fn);
                    const auto su = gemv(f, b + "ffn_up_shexp.weight", fn);
                    std::vector<float> act_sh((size_t) m.ff_shexp);
                    for (int64_t i = 0; i < m.ff_shexp; ++i) {
                        double gv = silu(sg[(size_t) i]), uv = su[(size_t) i];
                        if (clamp_sh > 1e-6) {
                            if (uv > clamp_sh) uv = clamp_sh;
                            if (uv < -clamp_sh) uv = -clamp_sh;
                            if (gv > clamp_sh) gv = clamp_sh;
                        }
                        act_sh[(size_t) i] = (float) (gv * uv);
                    }
                    const auto sh = gemv(f, b + "ffn_down_shexp.weight", act_sh);
                    for (int64_t i = 0; i < e; ++i) acc[(size_t) i] += (double) sh[(size_t) i];
                    for (int64_t i = 0; i < e; ++i) h[(size_t) i] += (float) acc[(size_t) i];
                }
                std::memcpy(x.data() + p * e, h.data(), (size_t) e * sizeof(float));
            }
            if (trace) {   // what the block handed on, summed over every position: the oracle prints the same sum
                double sum = 0;
                for (size_t q = 0; q < x.size(); ++q) sum += (double) x[q];
                trace->push_back(sum);
            }
        }
        out = std::move(x);
        return true;
    } catch (const std::exception& ex) {
        err = ex.what();
        return false;
    }
}

bool logits(const Model& m, const std::vector<float>& hidden_last, std::vector<float>& out, std::string& err) {
    try {
        const GgufModel& f = *m.file;
        const auto norm_w = vector_of(f, "output_norm.weight");
        std::vector<float> h = hidden_last;
        rms_norm(h, norm_w, m.eps);
        out = gemv(f, "output.weight", h);
        return true;
    } catch (const std::exception& ex) {
        err = ex.what();
        return false;
    }
}

}  // namespace strata::step35::ref
