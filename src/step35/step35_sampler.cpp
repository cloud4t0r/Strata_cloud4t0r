// src/step35/step35_sampler.cpp - see sampler.hpp for the order the filters are applied in, and why.
#include "strata/step35/sampler.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace strata::step35 {
namespace {

/// The repetition penalty, llama.cpp's rule: a token already emitted has its positive logit divided and its negative
/// one multiplied, so the penalty always moves it down the ranking, whichever side of zero it sits on.
void apply_penalty(std::vector<float>& lg, const std::vector<int>& recent, const SamplerConfig& cfg) {
    if (cfg.repeat_penalty <= 1.0f || cfg.repeat_last <= 0 || recent.empty()) return;
    const size_t from = recent.size() > (size_t) cfg.repeat_last ? recent.size() - (size_t) cfg.repeat_last : 0;
    for (size_t i = from; i < recent.size(); ++i) {
        const int id = recent[i];
        if (id < 0 || (size_t) id >= lg.size()) continue;
        float& v = lg[(size_t) id];
        v = v > 0.0f ? v / cfg.repeat_penalty : v * cfg.repeat_penalty;
    }
}

}  // namespace

std::vector<float> sample_probs(const std::vector<float>& logits, const SamplerConfig& cfg,
                                const std::vector<int>& recent) {
    std::vector<float> p = logits;
    if (p.empty()) return p;
    apply_penalty(p, recent, cfg);
    if (cfg.temperature <= 0.0f) {   // greedy: one token with all the mass, and nothing else survives
        const size_t best = (size_t) (std::max_element(p.begin(), p.end()) - p.begin());
        std::vector<float> out(p.size(), 0.0f);
        out[best] = 1.0f;
        return out;
    }
    // temperature, then a stable softmax (the max is subtracted: a logit of 20 with a temperature of 0.1 would
    // overflow exp otherwise)
    const float t = cfg.temperature;
    float mx = -INFINITY;
    for (float v : p) mx = std::max(mx, v / t);
    double sum = 0;
    for (float& v : p) {
        v = std::exp(v / t - mx);
        sum += (double) v;
    }
    for (float& v : p) v = (float) ((double) v / sum);

    // top-k: keep the k largest; ties never matter because the survivors are chosen by rank, not by value
    if (cfg.top_k > 0 && (size_t) cfg.top_k < p.size()) {
        std::vector<size_t> idx(p.size());
        std::iota(idx.begin(), idx.end(), (size_t) 0);
        std::partial_sort(idx.begin(), idx.begin() + cfg.top_k, idx.end(),
                          [&](size_t a, size_t b) { return p[a] > p[b]; });
        std::vector<char> keep(p.size(), 0);
        for (int k = 0; k < cfg.top_k; ++k) keep[idx[(size_t) k]] = 1;
        for (size_t i = 0; i < p.size(); ++i)
            if (!keep[i]) p[i] = 0.0f;
    }
    // top-p: the shortest prefix of the descending order whose mass reaches p, then renormalise
    if (cfg.top_p < 1.0f) {
        std::vector<size_t> idx(p.size());
        std::iota(idx.begin(), idx.end(), (size_t) 0);
        std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) { return p[a] > p[b]; });
        double acc = 0;
        size_t cut = p.size();
        for (size_t i = 0; i < idx.size(); ++i) {
            acc += (double) p[idx[i]];
            if (acc >= (double) cfg.top_p) {
                cut = i + 1;
                break;
            }
        }
        std::vector<char> keep(p.size(), 0);
        for (size_t i = 0; i < cut; ++i) keep[idx[i]] = 1;
        for (size_t i = 0; i < p.size(); ++i)
            if (!keep[i]) p[i] = 0.0f;
    }
    double tail = 0;
    for (float v : p) tail += (double) v;
    if (tail > 0)
        for (float& v : p) v = (float) ((double) v / tail);
    return p;
}

int sample(const std::vector<float>& logits, const SamplerConfig& cfg, const std::vector<int>& recent, Rng& rng) {
    const std::vector<float> p = sample_probs(logits, cfg, recent);
    if (p.empty()) return 0;
    if (cfg.temperature <= 0.0f) return (int) (std::max_element(logits.begin(), logits.end()) - logits.begin());
    const double u = rng.uniform();
    double acc = 0;
    for (size_t i = 0; i < p.size(); ++i) {
        acc += (double) p[i];
        if (u < acc) return (int) i;
    }
    // only reachable through a rounding error at the very end of the walk: answer with the last token that has mass
    for (size_t i = p.size(); i-- > 0;)
        if (p[i] > 0.0f) return (int) i;
    return 0;
}

}  // namespace strata::step35
