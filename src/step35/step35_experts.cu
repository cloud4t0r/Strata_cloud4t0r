// src/step35/step35_experts.cu - the VRAM-resident expert tier.  See experts.hpp.
//
// Eviction is FIFO over the slot ring, not LRU: with one position's eight experts fetched at a time and 45 blocks
// between two visits to the same block, an exact recency order would cost more to maintain than the streaming it
// saves, and the routing skew that makes the cache work at all is a property of the whole run rather than of the
// last few fetches.  A slot's `used` stamp is kept anyway, so a policy change is a comparison, not a rewrite.
#include "strata/step35/experts.hpp"

#include <cuda_runtime.h>

#include <cstdio>

namespace strata::step35::kernels {
namespace {

int64_t key_of(int layer, int expert) { return (int64_t) layer * 1000000 + (int64_t) expert; }

}  // namespace

bool expert_cache_open(ExpertCache& c, int n_slots, size_t gate_bytes, size_t down_bytes, std::string& err) {
    if (n_slots <= 0 || gate_bytes == 0 || down_bytes == 0) {
        err = "an expert cache needs slots and a size";
        return false;
    }
    c.gate_bytes = gate_bytes;
    c.down_bytes = down_bytes;
    c.slots.assign((size_t) n_slots, ExpertSlot{});
    c.fetches.assign((size_t) n_slots, 0);
    c.where.clear();
    c.where.reserve((size_t) n_slots * 2);
    for (int i = 0; i < n_slots; ++i) {
        ExpertSlot& s = c.slots[(size_t) i];
        const bool ok = cudaMalloc(&s.gate, gate_bytes) == cudaSuccess && cudaMalloc(&s.up, gate_bytes) == cudaSuccess &&
                        cudaMalloc(&s.down, down_bytes) == cudaSuccess;
        if (!ok) {
            // a smaller tier is still a tier: keep what was allocated and say how many slots it really holds
            err = std::string("out of device memory after ") + std::to_string(i) + " expert slot(s)";
            c.slots.resize((size_t) i);
            c.n_slots = i;
            return i > 0;
        }
    }
    c.n_slots = n_slots;
    return true;
}

void expert_cache_close(ExpertCache& c) {
    for (ExpertSlot& s : c.slots) {
        cudaFree(s.gate);
        cudaFree(s.up);
        cudaFree(s.down);
    }
    c.slots.clear();
    c.where.clear();
    c.fetches.clear();
    c.n_slots = 0;
}

bool expert_cache_policy(ExpertCache& c, const std::string& name) {
    if (name == "fifo") c.policy = TierPolicy::Fifo;
    else if (name == "lru") c.policy = TierPolicy::Lru;
    else if (name == "lfu") c.policy = TierPolicy::Lfu;
    else return false;
    return true;
}

ExpertSlot* expert_cache_fetch(ExpertCache& c, int layer, int expert, const uint8_t* gate_src, const uint8_t* up_src,
                               const uint8_t* down_src, bool& hit, std::string& err) {
    hit = false;
    if (c.n_slots <= 0) {
        err = "the expert cache has no slot";
        return nullptr;
    }
    const int64_t key = key_of(layer, expert);
    const auto it = c.where.find(key);
    if (it != c.where.end()) {
        ExpertSlot& s = c.slots[(size_t) it->second];
        s.used = ++c.clock;
        ++c.hits;
        hit = true;
        return &s;
    }
    // a miss: pick the slot to reuse.  FIFO walks the ring; LRU takes the least recently filled and LFU the least
    // often filled (ties to the oldest).  The scan is over the slots, once per miss, and the routing is what it is
    // worth optimising for: a linear pass over a few thousand slots is nothing beside the 5.9 MB copy that follows.
    ++c.misses;
    int idx = (int) (c.misses - 1) % c.n_slots;
    if (c.policy != TierPolicy::Fifo) {
        int best = 0;
        for (int i = 0; i < c.n_slots; ++i) {
            const ExpertSlot& s = c.slots[(size_t) i];
            const ExpertSlot& b = c.slots[(size_t) best];
            const bool better = c.policy == TierPolicy::Lru ? s.used < b.used
                               : c.policy == TierPolicy::Lfu ? (c.fetches[(size_t) i] < c.fetches[(size_t) best] ||
                                                                (c.fetches[(size_t) i] == c.fetches[(size_t) best] &&
                                                                 s.used < b.used))
                                                             : false;
            if (better) best = i;
        }
        idx = best;
    }
    ExpertSlot& s = c.slots[(size_t) idx];
    if (s.key >= 0) {
        c.where.erase(s.key);   // the key leaves with the bytes: a stale entry would send a fetch to a reused slot
        ++c.evictions;
    }
    const cudaError_t e1 = cudaMemcpy(s.gate, gate_src, c.gate_bytes, cudaMemcpyHostToDevice);
    const cudaError_t e2 = cudaMemcpy(s.up, up_src, c.gate_bytes, cudaMemcpyHostToDevice);
    const cudaError_t e3 = cudaMemcpy(s.down, down_src, c.down_bytes, cudaMemcpyHostToDevice);
    if (e1 != cudaSuccess || e2 != cudaSuccess || e3 != cudaSuccess) {
        err = std::string("filling an expert slot: ") +
              cudaGetErrorString(e1 != cudaSuccess ? e1 : (e2 != cudaSuccess ? e2 : e3));
        return nullptr;
    }
    s.key = key;
    s.used = ++c.clock;
    ++c.fetches[(size_t) idx];
    c.where[key] = idx;
    return &s;
}

}  // namespace strata::step35::kernels
