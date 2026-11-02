// include/strata/step35/profile.hpp - what the per-layer host round-trips cost, in the engine's own numbers.
//
// WHY THIS EXISTS.  docs/STEP37.md says the pass is not a CUDA graph "because the position still crosses to the device
// as a 4-byte H2D copy".  That is a reason a graph cannot be captured, not a reason it would be faster: the question
// is what those copies cost.  This counts and times the three kinds - the position, the router's ids and weights
// coming back to the host, and the experts going out to the card - so the answer is measured rather than assumed.
//
// Off by default (`STRATA_STEP35_PROFILE=1` turns it on): the timers are two steady_clock reads around an operation
// that takes microseconds, so leaving them in would cost more than they measure.
#pragma once

#include <cstdint>

namespace strata::step35 {

struct Profile {
    uint64_t position_calls = 0, position_ns = 0;   ///< the 4-byte H2D of a pass's position, per layer
    uint64_t router_calls = 0, router_ns = 0;       ///< the routed ids and weights, D2H, per position per block
    uint64_t expert_calls = 0, expert_ns = 0;       ///< the three matrices of one expert, H2D (no tier)
    uint64_t steps_calls = 0, steps_ns = 0;         ///< a pass's positions, H2D, once per layer
    uint64_t block_calls = 0, block_ns = 0;         ///< one block's dense weights, uploaded per pass and per layer
    uint64_t moe_calls = 0, moe_ns = 0;             ///< a MoE block's router and shared expert, uploaded per pass and per layer
};

bool profiling();                        ///< STRATA_STEP35_PROFILE=1
Profile& profile();                      ///< the counters of this process
void print_profile(double total_s);      ///< one line per kind, with its share of `total_s`

/// Times the enclosing scope into `calls` / `ns` when profiling is on; free when it is off.
struct ProfileScope {
    uint64_t* calls = nullptr;
    uint64_t* ns = nullptr;
    int64_t start = 0;
    ProfileScope(uint64_t& c, uint64_t& n) {
        if (!profiling()) return;
        calls = &c;
        ns = &n;
        start = now_ns();
    }
    ~ProfileScope() {
        if (calls == nullptr) return;
        ++*calls;
        *ns += (uint64_t) (now_ns() - start);
    }
    ProfileScope(const ProfileScope&) = delete;
    ProfileScope& operator=(const ProfileScope&) = delete;
    static int64_t now_ns();
};

}  // namespace strata::step35
