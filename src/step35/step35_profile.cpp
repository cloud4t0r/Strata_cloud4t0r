// src/step35/step35_profile.cpp - the counters behind profile.hpp.
#include "strata/step35/profile.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::step35 {
namespace {

Profile g_profile;

double share(uint64_t ns, double total_s) { return total_s > 0 ? 100.0 * ((double) ns / 1e9) / total_s : 0.0; }

}  // namespace

bool profiling() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_STEP35_PROFILE");
        return v != nullptr && v[0] != '\0' && std::strcmp(v, "0") != 0;
    }();
    return on;
}

Profile& profile() { return g_profile; }

int64_t ProfileScope::now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

void print_profile(double total_s) {
    if (!profiling()) return;
    const Profile& p = g_profile;
    std::fprintf(stderr,
                 "strata-step35: profile over %.2f s - position %llu call(s) %.1f ms (%.2f%%), router %llu call(s) "
                 "%.1f ms (%.2f%%), experts %llu call(s) %.1f ms (%.2f%%), steps %llu call(s) %.1f ms (%.2f%%), "
                 "blocks %llu call(s) %.1f ms (%.2f%%)\n",
                 total_s, (unsigned long long) p.position_calls, (double) p.position_ns / 1e6,
                 share(p.position_ns, total_s), (unsigned long long) p.router_calls, (double) p.router_ns / 1e6,
                 share(p.router_ns, total_s), (unsigned long long) p.expert_calls, (double) p.expert_ns / 1e6,
                 share(p.expert_ns, total_s), (unsigned long long) p.steps_calls, (double) p.steps_ns / 1e6,
                 share(p.steps_ns, total_s), (unsigned long long) p.block_calls, (double) p.block_ns / 1e6,
                 share(p.block_ns, total_s));
}

}  // namespace strata::step35
