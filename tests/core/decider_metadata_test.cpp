// tests/core/decider_metadata_test.cpp - PPLX Decider Flash (GGUF architecture qwen35) as the reader sees it: the
// family it is recognised as, and the guard that refuses a file which is not the decider the engine would run.
//
// Synthetic files only (gguf_fixture.hpp), so no 5 GB download is needed.  The numbers are the official release's
// (perplexity-ai/pplx-decider-flash-gguf-v1.2, Q8_0): 32 blocks, 2560 embedding, 9216 feed-forward, 16 Q heads
// over 4 KV heads of 256, 64 RoPE dims in mRoPE pairs 11/11/10/0, one full-attention block in four (the other 24
// are Gated DeltaNet, 16 q/k heads, 32 value heads of 128), NON-CAUSAL full attention, and the v3 readout protocol
// (255 codes, `<|fim_suffix|>`, temperature 1.5539).  The code list is a three-entry subset: the guard checks the
// pairing and the shape, not the release's particular 255 letters.
#include "gguf_fixture.hpp"

#include "strata/artifact/gguf_reader.hpp"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <functional>
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
               ("strata-decider-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        fs::create_directories(path);
    }
    ~TempDir() {
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }
};

const uint64_t kBlocks = 32;
const uint64_t kInterval = 4;

/// The release's layer pattern: a full-attention block every `kInterval`, the rest recurrent (GDN).
std::vector<uint64_t> recurrent_layers() {
    std::vector<uint64_t> v;
    for (uint64_t i = 0; i < kBlocks; ++i) v.push_back((i + 1) % kInterval == 0 ? 0 : 1);
    return v;
}

/// The release's key set, in a size the fixture can write.  `alter` changes one key to build a file that must be
/// refused.
std::vector<fixture::Kv> metadata(const std::function<void(std::vector<fixture::Kv>&)>& alter = {}) {
    const std::string p = "qwen35.";
    std::vector<fixture::Kv> kv = {
        fixture::str("general.architecture", "qwen35"),
        fixture::str("general.type", "model"),
        fixture::str("general.name", "Hf"),
        fixture::str("general.size_label", "4.8B"),
        fixture::u32(p + "block_count", kBlocks),
        fixture::u32(p + "context_length", 262144),
        fixture::u32(p + "embedding_length", 2560),
        fixture::u32(p + "feed_forward_length", 9216),
        fixture::u32(p + "attention.head_count", 16),
        fixture::u32(p + "attention.head_count_kv", 4),
        fixture::u32(p + "attention.key_length", 256),
        fixture::u32(p + "attention.value_length", 256),
        fixture::f32(p + "attention.layer_norm_rms_epsilon", 1e-6f),
        fixture::u32s(p + "rope.dimension_sections", {11, 11, 10, 0}),
        fixture::f32(p + "rope.freq_base", 1e7f),
        fixture::u32(p + "rope.dimension_count", 64),
        fixture::u32(p + "full_attention_interval", kInterval),
        fixture::u32(p + "ssm.conv_kernel", 4),
        fixture::u32(p + "ssm.state_size", 128),
        fixture::u32(p + "ssm.group_count", 16),
        fixture::u32(p + "ssm.time_step_rank", 32),
        fixture::u32(p + "ssm.inner_size", 4096),
        fixture::bools(p + "attention.recurrent_layers", recurrent_layers()),
        fixture::boolean(p + "attention.causal", false),
        fixture::u32("pplx_decider.protocol_version", 3),
        fixture::str("pplx_decider.template", "joint-question-inline-readout-v1"),
        fixture::str("pplx_decider.marker", "<|fim_suffix|>"),
        fixture::f32("pplx_decider.temperature", 1.5539092f),
        fixture::strs("pplx_decider.codes", {"A", "B", "C"}),
        fixture::u32s("pplx_decider.token_ids", {32, 33, 34}),
        fixture::u32("pplx_decider.max_pixels", 1638400),
        fixture::strs("pplx_decider.escape_tokens", {"<|fim_suffix|>"}),
        fixture::str("pplx_decider.base_model", "Qwen/Qwen3.5-4B"),
    };
    if (alter) alter(kv);
    return kv;
}

/// The guard's answer for a file built from `kv` (a fresh name each time: a directory is not rewritten in place).
std::string guard_of(const fs::path& dir, const std::vector<fixture::Kv>& kv, int n = 0) {
    static int counter = 0;
    const fs::path p = dir / ("case-" + std::to_string(++counter) + "-" + std::to_string(n) + ".gguf");
    fixture::write(p, kv, {});
    const strata::GgufFile f(p.string());
    return strata::check_architecture(f);
}

void drop(std::vector<fixture::Kv>& kv, const std::string& key) {
    for (size_t i = 0; i < kv.size(); ++i)
        if (kv[i].key == key) {
            kv.erase(kv.begin() + (long) i);
            return;
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

void set_f32(std::vector<fixture::Kv>& kv, const std::string& key, float v) {
    for (auto& e : kv)
        if (e.key == key) {
            e.f = v;
            return;
        }
}

void set_str(std::vector<fixture::Kv>& kv, const std::string& key, const std::string& v) {
    for (auto& e : kv)
        if (e.key == key) {
            e.s = v;
            return;
        }
}

void set_strs(std::vector<fixture::Kv>& kv, const std::string& key, std::vector<std::string> v) {
    for (auto& e : kv)
        if (e.key == key) {
            e.sarr = std::move(v);
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
        check(strata::model_family(f) == strata::ModelFamily::Decider, "general.architecture qwen35 is Decider");
        check(std::string(strata::model_family_arch(strata::ModelFamily::Decider)) == "qwen35",
              "the family's arch string is qwen35");
        check(std::string(strata::model_family_title(strata::ModelFamily::Decider)) == "PPLX Decider Flash",
              "the family's title is PPLX Decider Flash");
        check(strata::check_architecture(f).empty(), "the guard accepts the release's key set");
        check(!strata::model_family_runs(strata::ModelFamily::Decider),
              "Flash-Next's engine does not claim the family (it is not a chat model)");
        const std::string why = strata::model_family_not_supported(strata::ModelFamily::Decider);
        check(why.find("PPLX Decider Flash") != std::string::npos, "the refusal names the model: " + why);
        check(why.find("decision model") != std::string::npos, "the refusal says it is not a chat model: " + why);
        check(strata::check_engine_architecture(f).find("decision model") != std::string::npos,
              "the engine's guard refuses it with the decision-model reason");
    }

    // ---- the property that makes it a decider, not a chat model
    {
        const std::string err = guard_of(dir.path, metadata([](std::vector<fixture::Kv>& kv) {
                                             set_u32(kv, "qwen35.attention.causal", 1);
                                         }));
        check(err.find("qwen35.attention.causal is true") != std::string::npos,
              "a causal qwen35 file is refused: " + err);
    }
    {
        const std::string err = guard_of(dir.path, metadata([](std::vector<fixture::Kv>& kv) {
                                             drop(kv, "qwen35.attention.causal");
                                         }));
        check(err.find("missing qwen35.attention.causal") != std::string::npos,
              "an absent causal key is refused (the default must not be guessed): " + err);
    }

    // ---- the v3 protocol
    {
        const std::string err = guard_of(dir.path, metadata([](std::vector<fixture::Kv>& kv) {
                                             set_u32(kv, "pplx_decider.protocol_version", 2);
                                         }));
        check(err.find("protocol_version is not 3") != std::string::npos, "protocol v2 is refused: " + err);
    }
    {
        const std::string err = guard_of(dir.path, metadata([](std::vector<fixture::Kv>& kv) {
                                             set_str(kv, "pplx_decider.template", "joint-question-inline-readout-v2");
                                         }));
        check(err.find("pplx_decider.template") != std::string::npos, "another template is refused: " + err);
    }
    {
        const std::string err = guard_of(dir.path, metadata([](std::vector<fixture::Kv>& kv) {
                                             set_arr(kv, "pplx_decider.token_ids", {32, 33});
                                         }));
        check(err.find("3 entries for 2 token ids") != std::string::npos,
              "codes and token ids must pair up: " + err);
    }
    {
        const std::string err = guard_of(dir.path, metadata([](std::vector<fixture::Kv>& kv) {
                                             set_strs(kv, "pplx_decider.codes", {"A", "b", "C"});
                                         }));
        check(err.find("not uppercase A-Z") != std::string::npos, "a non-code token is refused: " + err);
    }
    {
        const std::string err = guard_of(dir.path, metadata([](std::vector<fixture::Kv>& kv) {
                                             set_f32(kv, "pplx_decider.temperature", 0.0f);
                                         }));
        check(err.find("temperature") != std::string::npos, "a zero temperature is refused: " + err);
    }

    // ---- the layer pattern the kernels compile
    {
        std::vector<uint64_t> short_pattern = recurrent_layers();
        short_pattern.pop_back();
        const std::string err = guard_of(dir.path, metadata([&](std::vector<fixture::Kv>& kv) {
                                             set_arr(kv, "qwen35.attention.recurrent_layers", short_pattern);
                                         }));
        check(err.find("31 entries for 32 blocks") != std::string::npos,
              "a short layer pattern is refused: " + err);
    }
    {
        // block 0 claims to be a full-attention block: the pattern disagrees with the interval
        std::vector<uint64_t> flipped = recurrent_layers();
        flipped[0] = 0;
        const std::string err = guard_of(dir.path, metadata([&](std::vector<fixture::Kv>& kv) {
                                             set_arr(kv, "qwen35.attention.recurrent_layers", flipped);
                                         }));
        check(err.find("disagrees with full_attention_interval on block 0") != std::string::npos,
              "a pattern that disagrees with the interval is refused: " + err);
    }
    {
        const std::string err = guard_of(dir.path, metadata([](std::vector<fixture::Kv>& kv) {
                                             set_u32(kv, "qwen35.full_attention_interval", 1);
                                         }));
        check(err.find("full_attention_interval") != std::string::npos, "interval 1 is refused: " + err);
    }

    // ---- the geometry
    {
        const std::string err = guard_of(dir.path, metadata([](std::vector<fixture::Kv>& kv) {
                                             set_u32(kv, "qwen35.attention.head_count", 15);
                                         }));
        check(err.find("head_count does not split") != std::string::npos, "a broken GQA split is refused: " + err);
    }
    {
        const std::string err = guard_of(dir.path, metadata([](std::vector<fixture::Kv>& kv) {
                                             set_arr(kv, "qwen35.rope.dimension_sections", {11, 11, 11, 0});
                                         }));
        check(err.find("dimension_sections") != std::string::npos,
              "mRoPE sections that do not fill the rotary dims are refused: " + err);
    }
    {
        const std::string err = guard_of(dir.path, metadata([](std::vector<fixture::Kv>& kv) {
                                             set_u32(kv, "qwen35.ssm.inner_size", 4000);
                                         }));
        check(err.find("inner_size") != std::string::npos, "a GDN inner size that does not add up is refused: " + err);
    }

    if (failures) {
        std::printf("decider_metadata_test: %d check(s) failed\n", failures);
        return 1;
    }
    std::printf("decider_metadata_test: all checks passed\n");
    return 0;
}
