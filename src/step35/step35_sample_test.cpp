// src/step35/step35_sample_test.cpp - the sampler, against arithmetic and against its own statistics.
//
// Nothing here needs a GPU or the model: the sampler is host code over 128 896 floats, and every claim it makes can
// be checked on three or five of them.  The checks are the ways a sampler is usually wrong: greedy that still
// depends on the other knobs, a nucleus that leaves the tail with mass, a penalty applied to the wrong side of
// zero, and a stream that is not reproducible.
#include "strata/step35/sampler.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

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

/// The hand computation, for a three-way distribution: p_i = exp(x_i / t) / sum.
std::vector<float> softmax3(const std::vector<float>& x, float t) {
    std::vector<float> p(3);
    double sum = 0;
    for (int i = 0; i < 3; ++i) sum += std::exp((double) x[(size_t) i] / (double) t);
    for (int i = 0; i < 3; ++i) p[(size_t) i] = (float) (std::exp((double) x[(size_t) i] / (double) t) / sum);
    return p;
}

}  // namespace

int main() {
    const std::vector<float> lg = {1.0f, 3.0f, 2.0f};
    strata::step35::SamplerConfig greedy;
    greedy.temperature = 0.0f;
    // greedy must not depend on the other knobs, however they are set
    strata::step35::SamplerConfig greedy_odd = greedy;
    greedy_odd.top_k = 2;
    greedy_odd.top_p = 0.1f;
    greedy_odd.repeat_penalty = 5.0f;
    strata::step35::Rng rng(7);
    const std::vector<int> recent = {1};   // the argmax, so a penalty that leaked into greedy would show here
    check(strata::step35::sample(lg, greedy, recent, rng) == 1, "temperature 0 answers the argmax");
    check(strata::step35::sample(lg, greedy_odd, recent, rng) == 1,
          "temperature 0 still answers the argmax with top_k, top_p and a penalty set");

    strata::step35::SamplerConfig one = greedy;
    one.temperature = 1.0f;
    one.top_k = 1;
    bool always = true;
    for (int i = 0; i < 1000; ++i) always = always && strata::step35::sample(lg, one, {}, rng) == 1;
    check(always, "top_k 1 answers the argmax a thousand times");

    strata::step35::SamplerConfig hot = one;
    hot.top_k = 0;
    const std::vector<float> degenerate = {0.0f, 100.0f, 0.0f};
    always = true;
    for (int i = 0; i < 1000; ++i) always = always && strata::step35::sample(degenerate, hot, {}, rng) == 1;
    check(always, "a 100-logit lead is never lost to the draw");

    // the probabilities are the softmax, to the float it is computed in
    const std::vector<float> p = strata::step35::sample_probs(lg, hot, {});
    const std::vector<float> want = softmax3(lg, 1.0f);
    double worst = 0;
    for (size_t i = 0; i < p.size(); ++i) worst = std::fmax(worst, std::fabs((double) p[i] - (double) want[i]));
    char label[200];
    std::snprintf(label, sizeof label, "temperature 1 is the softmax (worst |diff| %.1e)", worst);
    check(worst < 1e-6, label);

    // and the draws follow it: 60 000 draws of a 3-way distribution, within 1% absolute
    const size_t n = 60000;
    std::vector<int> count(3, 0);
    strata::step35::Rng draws(12345);
    for (size_t i = 0; i < n; ++i) ++count[(size_t) strata::step35::sample(lg, hot, {}, draws)];
    double worst_rate = 0;
    for (int i = 0; i < 3; ++i)
        worst_rate = std::fmax(worst_rate, std::fabs((double) count[(size_t) i] / (double) n - (double) want[(size_t) i]));
    std::snprintf(label, sizeof label,
                  "60 000 draws follow the softmax (%d/%d/%d against %.3f/%.3f/%.3f, worst %.1e)", count[0], count[1],
                  count[2], (double) want[0], (double) want[1], (double) want[2], worst_rate);
    check(worst_rate < 0.01, label);

    // the nucleus: nothing outside the smallest set whose mass reaches top_p is ever drawn, and the kept mass is 1
    strata::step35::SamplerConfig nucleus = hot;
    nucleus.top_p = 0.5f;   // 3.0 of 6.06 = 0.669 alone, so the nucleus is {1} and the draw never leaves it
    strata::step35::SamplerConfig probe = nucleus;
    bool inside = true;
    for (int i = 0; i < 2000; ++i) inside = inside && strata::step35::sample(lg, probe, {}, draws) == 1;
    check(inside, "top_p 0.5 over logits {1,3,2} keeps only the 0.669 token and never draws the tail");
    const std::vector<float> pn = strata::step35::sample_probs(lg, probe, {});
    double mass = 0;
    for (float v : pn) mass += (double) v;
    std::snprintf(label, sizeof label, "the nucleus renormalises to 1 (%.6f)", mass);
    check(std::fabs(mass - 1.0) < 1e-6 && pn[0] == 0.0f && pn[2] == 0.0f, label);

    // the repetition penalty moves a repeated token down, on both sides of zero, and by the factor it promises
    strata::step35::SamplerConfig pen = hot;
    pen.repeat_penalty = 2.0f;
    pen.repeat_last = 8;
    const std::vector<float> pp = strata::step35::sample_probs(lg, pen, {1});
    const std::vector<float> base = strata::step35::sample_probs(lg, hot, {});
    std::snprintf(label, sizeof label, "a penalty of 2 halves the repeated token's probability (%.4f -> %.4f)",
                  (double) base[1], (double) pp[1]);
    check(pp[1] < base[1], label);
    const std::vector<float> negatives = {-1.0f, 3.0f, -2.0f};
    const std::vector<float> pneg = strata::step35::sample_probs(negatives, pen, {0});
    const std::vector<float> bneg = strata::step35::sample_probs(negatives, hot, {});
    std::snprintf(label, sizeof label, "a negative logit is multiplied, not divided (%.4f -> %.4f)", (double) bneg[0],
                  (double) pneg[0]);
    check(pneg[0] < bneg[0], label);

    // the stream: same seed, same draws; different seed, different draws
    strata::step35::Rng a(99), b(99), c(100);
    bool same = true, differs = false;
    for (int i = 0; i < 1000; ++i) {
        const int x = strata::step35::sample(lg, hot, {}, a);
        same = same && x == strata::step35::sample(lg, hot, {}, b);
        differs = differs || x != strata::step35::sample(lg, hot, {}, c);
    }
    check(same, "one seed gives one stream");
    check(differs, "another seed gives another stream");

    if (failures) {
        std::printf("step35_sample_test: %d check(s) failed\n", failures);
        return 1;
    }
    std::printf("step35_sample_test: all checks passed\n");
    return 0;
}
