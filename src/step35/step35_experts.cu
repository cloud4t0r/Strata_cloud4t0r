// src/step35/step35_experts.cu - the VRAM-resident expert tier.  See experts.hpp.
//
// Eviction is FIFO over the slot ring, not LRU: with one position's eight experts fetched at a time and 45 blocks
// between two visits to the same block, an exact recency order would cost more to maintain than the streaming it
// saves, and the routing skew that makes the cache work at all is a property of the whole run rather than of the
// last few fetches.  A slot's `used` stamp is kept anyway, so a policy change is a comparison, not a rewrite.
#include "strata/step35/experts.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <cstdlib>

namespace strata::step35::kernels {
namespace {

int64_t key_of(int layer, int expert) { return (int64_t) layer * 1000000 + (int64_t) expert; }

}  // namespace

bool expert_cache_open(ExpertCache& c, int n_slots, size_t gate_bytes, size_t down_bytes, std::string& err,
                       bool host_pinned) {
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
    c.host = host_pinned;
    bool pinned = false;
    auto alloc = [&](void** p, size_t bytes) {
        if (!host_pinned) return cudaMalloc(p, bytes) == cudaSuccess;
        // PAGEABLE, on purpose.  cudaHostAlloc locks the pages: they cannot be swapped, and asking for gigabytes of
        // them on a desktop starves it until the session dies - measured on this machine, twice, with Cinnamon as the
        // victim.  A pageable copy is slower (a few GB/s against ~27) and still beats the disk, and the kernel keeps
        // the right to take the memory back instead of taking the desktop down.
        *p = std::malloc(bytes);
        pinned = false;
        return *p != nullptr;
    };
    for (int i = 0; i < n_slots; ++i) {
        ExpertSlot& s = c.slots[(size_t) i];
        const bool ok = alloc(&s.gate, gate_bytes) && alloc(&s.up, gate_bytes) && alloc(&s.down, down_bytes);
        s.pinned = pinned;
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
        if (c.host) {
            if (s.pinned) {
                cudaFreeHost(s.gate);
                cudaFreeHost(s.up);
                cudaFreeHost(s.down);
            } else {
                std::free(s.gate);
                std::free(s.up);
                std::free(s.down);
            }
        } else {
            cudaFree(s.gate);
            cudaFree(s.up);
            cudaFree(s.down);
        }
    }
    c.below = nullptr;
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
    // A MISS HERE IS NOT A MISS BELOW.  If a host cache is behind this one, it holds the bytes or reads them once; the
    // copy that fills this slot then comes from RAM rather than from the file.
    if (c.below != nullptr) {
        bool hit2 = false;
        ExpertSlot* s2 = expert_cache_fetch(*c.below, layer, expert, gate_src, up_src, down_src, hit2, err);
        if (s2 == nullptr) return nullptr;
        ++c.misses;
        int idx2 = (int) (c.misses - 1) % c.n_slots;
        if (c.policy != TierPolicy::Fifo) {
            int best = 0;
            for (int i = 0; i < c.n_slots; ++i)
                if (c.slots[(size_t) i].used < c.slots[(size_t) best].used) best = i;
            idx2 = best;
        }
        ExpertSlot& dst = c.slots[(size_t) idx2];
        if (dst.key >= 0) {
            c.where.erase(dst.key);
            ++c.evictions;
        }
        const cudaError_t e1 = cudaMemcpy(dst.gate, s2->gate, c.gate_bytes, cudaMemcpyHostToDevice);
        const cudaError_t e2 = cudaMemcpy(dst.up, s2->up, c.gate_bytes, cudaMemcpyHostToDevice);
        const cudaError_t e3 = cudaMemcpy(dst.down, s2->down, c.down_bytes, cudaMemcpyHostToDevice);
        if (e1 != cudaSuccess || e2 != cudaSuccess || e3 != cudaSuccess) {
            err = std::string("filling an expert slot from the host cache: ") +
                  cudaGetErrorString(e1 != cudaSuccess ? e1 : (e2 != cudaSuccess ? e2 : e3));
            return nullptr;
        }
        dst.key = key;
        dst.used = ++c.clock;
        ++c.fetches[(size_t) idx2];
        c.where[key] = idx2;
        return &dst;
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
    if (c.host) {   // a host cache's slot IS host memory: the three copies are memcpy, not a device transfer
        std::memcpy(s.gate, gate_src, c.gate_bytes);
        std::memcpy(s.up, up_src, c.gate_bytes);
        std::memcpy(s.down, down_src, c.down_bytes);
        s.key = key;
        s.used = ++c.clock;
        ++c.fetches[(size_t) idx];
        c.where[key] = idx;
        return &s;
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
