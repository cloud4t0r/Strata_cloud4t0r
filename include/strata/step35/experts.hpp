// include/strata/step35/experts.hpp - the VRAM-resident expert tier for the Step-3.7 path.
//
// WHY IT EXISTS, IN ONE PARAGRAPH.  `step35_seq.cu` copies the eight experts a position routes to out of the file
// into three reused device buffers, for every position and every block: 42 mixture-of-experts blocks, ~6.7 MB per
// expert in the release's IQ3_XXS, so a position moves about 2.2 GB and that is the whole cost of a token - measured
// 0.509 s per position in a batch pass, 0.81 s per token alone, against ~0.3 ms of arithmetic.  The experts are 75 GB
// and the card is 16 GB, so all of them cannot stay; but routing is skewed, and the engine's other families already
// keep a residency table (`include/strata/core/expert_cache.hpp`) for exactly this reason.
//
// WHAT THIS IS.  Slot storage and the residency table, nothing else: `expert_cache_fetch` answers the device pointers
// for one (block, expert), filling a slot from the file when it has to.  The arithmetic is untouched, so a cache hit
// and a cache miss compute the same bits - which is what makes `--expert-cache` safe to turn on and off.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace strata::step35::kernels {

/// One expert's three matrices, resident.  `gate` and `up` are the same shape in this model; `down` is not.
struct ExpertSlot {
    void* gate = nullptr;
    void* up = nullptr;
    void* down = nullptr;
    int64_t key = -1;   ///< layer * 1e6 + expert, or -1 while the slot is empty
    bool pinned = false;///< a host slot that cudaHostAlloc gave us (cudaFreeHost), else malloc (free)
    uint64_t used = 0;  ///< fetch order, for the eviction choice
};

/// How a slot is chosen when the ring is full.  FIFO is the ring's own order; LRU evicts the least recently fetched
/// and LFU the least often fetched (ties to the oldest), which is what the routing's skew should reward.
enum class TierPolicy { Fifo, Lru, Lfu };

struct ExpertCache {
    int n_slots = 0;
    TierPolicy policy = TierPolicy::Lru;   // measured best: 41.7% of fetches served at 4 GB against FIFO's 37.2%
    size_t gate_bytes = 0;
    size_t down_bytes = 0;
    std::vector<ExpertSlot> slots;
    /// The level BELOW this one: a cache whose slots are pinned HOST memory.  A miss here asks it first, so the bytes
    /// come from RAM (pinned, DMA) instead of the file - which on a 75 GB model on a 62 GB machine is the page cache's
    /// coin toss, page by page, instead of an expert-granular choice.
    ExpertCache* below = nullptr;
    bool host = false;          ///< this cache's own slots live in pinned host memory
    std::unordered_map<int64_t, int> where;   ///< (layer, expert) -> slot index
    uint64_t clock = 0;
    // counters, for the figure the engine prints
    uint64_t hits = 0, misses = 0, evictions = 0;
    std::vector<uint64_t> fetches;   ///< per slot: how many times it was filled (LFU)
};

/// Allocates `n_slots` experts of `gate_bytes` (twice: gate and up) and `down_bytes`.  A slot is filled on first use.
/// `host_pinned`: the slots live in pinned host memory (about 20 GB/s of copy, no page faults) instead of VRAM - the
/// level below a VRAM tier, not a replacement for it.
bool expert_cache_open(ExpertCache& c, int n_slots, size_t gate_bytes, size_t down_bytes, std::string& err,
                       bool host_pinned = false);
/// "fifo", "lru", "lfu"; anything else leaves the policy alone and returns false.
bool expert_cache_policy(ExpertCache& c, const std::string& name);
void expert_cache_close(ExpertCache& c);

/// The slot holding (layer, expert), filled from the three host pointers when it is not resident already.  Returns
/// null and sets `err` only on a CUDA failure; `hit` says whether the copy was skipped.  The returned pointers are
/// valid until the next fetch that evicts this slot, which is why the caller must use them before fetching again.
ExpertSlot* expert_cache_fetch(ExpertCache& c, int layer, int expert, const uint8_t* gate_src, const uint8_t* up_src,
                               const uint8_t* down_src, bool& hit, std::string& err);

}  // namespace strata::step35::kernels
