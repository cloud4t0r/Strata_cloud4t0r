// include/strata/step35/sampler.hpp - turning a position's logits into one token, with the knobs a chat needs.
//
// GREEDY IS THE DEFAULT AND THE ONLY PATH THAT EXISTED UNTIL NOW: `--max-new` and `--gpu-spec` took the top-1, which
// is what makes their output exactly comparable to the reference and to each other.  This is what a temperature, a
// top-k, a nucleus and a repetition penalty add, and it is deliberately host-side: 128 896 floats copied back per
// token are already on the host, and a sort of that size costs 10 ms - 1% of a token here, and the code stays
// divisible and testable without a card.
//
// ORDER, which matters and is llama.cpp's: repetition penalty over the recent window, then temperature, then top-k,
// then top-p, then the draw.  A penalty applied after the filters would push mass back into tokens the filters had
// already removed; a draw before the filters would not be the distribution anybody asked for.
#pragma once

#include <cstdint>
#include <vector>

namespace strata::step35 {

struct SamplerConfig {
    float temperature = 0.0f;   ///< 0 = greedy, whatever else is set: the answer must not depend on the others
    int top_k = 0;              ///< 0 = no k limit
    float top_p = 1.0f;         ///< 1.0 = no nucleus cut
    float repeat_penalty = 1.0f;///< 1.0 = off; > 1 divides a repeated token's positive logit and multiplies a negative
    int repeat_last = 64;       ///< how many of the most recent tokens the penalty looks at
    uint64_t seed = 0;          ///< the stream; 0 is the default so two runs of the same command agree
};

/// xorshift64*: a small deterministic stream, so a seeded run is reproducible and a test can assert on draws.
struct Rng {
    uint64_t state = 0x9E3779B97F4A7C15ull;
    explicit Rng(uint64_t seed = 0) : state(seed ? seed : 0x9E3779B97F4A7C15ull) {}
    uint64_t next() {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        return state * 2685821657736338717ull;
    }
    /// [0, 1), 53 bits: enough for a 128 896-way walk without a bias anyone can measure.
    double uniform() { return (double) (next() >> 11) * (1.0 / 9007199254740992.0); }
};

/// The token a position emits.  `logits` is copied, not modified, so a caller can still rank what it was given.
/// `recent` is the tokens already emitted, oldest first; only its tail (`repeat_last`) is read.
int sample(const std::vector<float>& logits, const SamplerConfig& cfg, const std::vector<int>& recent, Rng& rng);

/// The same filters without the draw: the probability each token is left with, out of 1, and 0 for what the filters
/// removed.  Exposed so a test can hold the arithmetic to a hand computation instead of only to a statistic.
std::vector<float> sample_probs(const std::vector<float>& logits, const SamplerConfig& cfg,
                                const std::vector<int>& recent);

}  // namespace strata::step35
