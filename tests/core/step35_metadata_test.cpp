// tests/core/step35_metadata_test.cpp - Step-3.7-Flash (GGUF architecture step35) as the reader sees it: the
// family it is recognised as, the guard that refuses a file whose per-layer arrays do not describe every block,
// and the counts that must agree.  Synthetic files only (gguf_fixture.hpp), so no 76 GB download is needed.
//
// The numbers are the release's: stepfun-ai/Step-3.7-Flash-GGUF, IQ3_XXS, whose header says 45 blocks (3 dense,
// 42 mixture of experts), 4096 embedding, one full-attention block in four with window 512, 64 Q heads on a full
// block and 96 on a sliding one, 8 KV heads, 288 experts top-8, a shared expert, and SwiGLU clamping on the last
// two blocks.  A 3-block model keeps the file a few KB and the pattern identical.
#include "gguf_fixture.hpp"

#include "strata/artifact/gguf_reader.hpp"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        ++failures;
        std::printf("FAIL  %s\n", what.c_str());
    } else {
        std::printf("  ok  %s\n", what.c_str());
    }
}

struct TempDir {
    fs::path path;
    TempDir() {
        path = fs::temp_directory_path() /
               ("strata-step35-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        fs::create_directories(path);
    }
    ~TempDir() {
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }
};

const uint64_t kBlocks = 3;
const uint64_t kEmbd = 64;
const uint64_t kVocab = 128;
const uint64_t kExperts = 8;

/// The release's key set, in a size the fixture can write.  `alter` changes one key to build a file that must be
/// refused.
std::vector<fixture::Kv> metadata(void (*alter)(std::vector<fixture::Kv>&) = nullptr) {
    const std::string p = "step35.";
    std::vector<fixture::Kv> kv = {
        fixture::str("general.architecture", "step35"),
        fixture::u32(p + "block_count", kBlocks),
        fixture::u32(p + "embedding_length", kEmbd),
        fixture::u32(p + "context_length", 262144),
        fixture::u32(p + "feed_forward_length", 11264),
        fixture::u32(p + "attention.key_length", 128),
        fixture::u32(p + "attention.value_length", 128),
        fixture::f32(p + "attention.layer_norm_rms_epsilon", 1e-5f),
        fixture::u32(p + "attention.sliding_window", 512),
        // one full-attention block in four: block 0 full, 1 and 2 sliding (the release's pattern is the same)
        fixture::u32s(p + "attention.head_count", {64, 96, 96}),
        fixture::u32s(p + "attention.head_count_kv", {8, 8, 8}),
        fixture::u32s(p + "attention.sliding_window_pattern", {0, 1, 1}),
        fixture::f32(p + "rope.freq_base", 5000000.0f),
        fixture::f32(p + "rope.freq_base_swa", 10000.0f),
        fixture::u32(p + "expert_count", kExperts),
        fixture::u32(p + "expert_used_count", 2),
        fixture::u32(p + "expert_feed_forward_length", 1280),
        fixture::u32(p + "expert_shared_feed_forward_length", 1280),
        fixture::f32(p + "expert_weights_scale", 3.0f),
        fixture::u32(p + "expert_weights_norm", 1),
        fixture::u32(p + "expert_gating_func", 2),
        fixture::u32(p + "leading_dense_block_count", 1),
        fixture::u32s(p + "swiglu_clamp_exp", {0, 0, 7}),
        fixture::u32s(p + "swiglu_clamp_shexp", {0, 0, 16}),
    };
    if (alter) alter(kv);
    std::vector<std::string> tokens;
    for (uint64_t i = 0; i < kVocab; ++i) tokens.push_back("t" + std::to_string(i));
    kv.push_back(fixture::str_array("tokenizer.ggml.tokens", tokens));
    return kv;
}

std::string guard_of(const fs::path& dir, const std::vector<fixture::Kv>& kv) {
    const fs::path p = dir / "m.gguf";
    fixture::write(p, kv, {});
    const strata::GgufFile f(p.string());
    return strata::check_architecture(f);
}

/// The first key whose name contains `needle`, replaced or removed.
void drop(std::vector<fixture::Kv>& kv, const std::string& needle) {
    for (size_t i = 0; i < kv.size(); ++i) {
        if (kv[i].key.find(needle) != std::string::npos) {
            kv.erase(kv.begin() + (long) i);
            return;
        }
    }
}

void set_u32(std::vector<fixture::Kv>& kv, const std::string& key, uint64_t v) {
    for (auto& e : kv)
        if (e.key == key) {
            e.u = v;
            return;
        }
}

void set_arr(std::vector<fixture::Kv>& kv, const std::string& key, std::vector<uint64_t> v) {
    for (auto& e : kv)
        if (e.key == key) {
            e.arr = std::move(v);
            return;
        }
}

}  // namespace

int main() {
    TempDir dir;

    // ---- the family
    {
        const fs::path p = dir.path / "family.gguf";
        fixture::write(p, metadata(), {});
        const strata::GgufFile f(p.string());
        check(strata::model_family(f) == strata::ModelFamily::Step35, "general.architecture step35 is Step35");
        check(std::string(strata::model_family_arch(strata::ModelFamily::Step35)) == "step35",
              "the family's arch string is step35");
        check(std::string(strata::model_family_title(strata::ModelFamily::Step35)) == "Step-3.7-Flash",
              "the family's title is Step-3.7-Flash");
        check(strata::check_architecture(f).empty(), "the guard accepts the release's key set");
        // the engine path is the STEP35 one: `strata` must hand the file over, not try Flash-Next's geometry
        check(!strata::model_family_runs(strata::ModelFamily::Step35),
              "Flash-Next's engine does not claim the family (its own path runs it)");
        const std::string why = strata::model_family_not_supported(strata::ModelFamily::Step35);
        check(why.find("Step-3.7-Flash") != std::string::npos, "the refusal names the model: " + why);
    }

    // ---- what the guard must refuse
    {
        const std::string err = guard_of(dir.path, metadata([](std::vector<fixture::Kv>& kv) {
                                             drop(kv, "sliding_window_pattern");
                                         }));
        check(err.find("sliding_window_pattern") != std::string::npos, "a missing pattern is named: " + err);
    }
    {
        // one head count too few: every block has its own width, so a short array is read past its end
        const std::string err = guard_of(dir.path, metadata([](std::vector<fixture::Kv>& kv) {
                                             set_arr(kv, "step35.attention.head_count", {64, 96});
                                         }));
        check(err.find("2 entries for 3 blocks") != std::string::npos, "a short per-layer array is refused: " + err);
    }
    {
        const std::string err = guard_of(dir.path, metadata([](std::vector<fixture::Kv>& kv) {
                                             set_u32(kv, "step35.expert_used_count", 9);
                                         }));
        check(err.find("expert_used_count") != std::string::npos, "top-k above the expert count is refused: " + err);
    }
    {
        const std::string err = guard_of(dir.path, metadata([](std::vector<fixture::Kv>& kv) {
                                             set_u32(kv, "step35.leading_dense_block_count", 4);
                                         }));
        check(err.find("leading_dense_block_count") != std::string::npos,
              "a dense prefix longer than the file is refused: " + err);
    }
    {
        // 96 heads over 8 KV heads is a whole split; 96 over 7 is not
        const std::string err = guard_of(dir.path, metadata([](std::vector<fixture::Kv>& kv) {
                                             set_arr(kv, "step35.attention.head_count_kv", {8, 7, 8});
                                         }));
        check(err.find("head_count") != std::string::npos, "a broken GQA split is refused: " + err);
    }

    if (failures) {
        std::printf("step35_metadata_test: %d check(s) failed\n", failures);
        return 1;
    }
    std::printf("step35_metadata_test: all checks passed\n");
    return 0;
}
