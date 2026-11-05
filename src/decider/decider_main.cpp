// src/decider/decider_main.cpp - `strata-decider`, the decision model's own entry point.
//
// WHAT EXISTS TODAY.  The tensor audit (`--inspect`) and the CPU reference (`--ref`), neither of which needs a GPU.
// `--inspect` is the first thing to run on a new file; `--ref` is the oracle the kernels will be compared against,
// and the way the whole forward is checked against the checkpoint's own numbers before any kernel exists.  The
// engine itself (the 32 blocks on CUDA/HIP, the readout at the markers, the `DEC` operation of serve/server.py) is
// M2-M4 of the integration study; until it lands, any other request says so and exits 2.
//
// The family is recognised by `strata` (artifact/model_family.hpp: architecture qwen35), which refuses it with the
// decision-model reason instead of running Flash-Next's chat path over it.
#include "strata/artifact/gguf_reader.hpp"
#include "strata/decider/inspect.hpp"
#include "strata/decider/ref.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <exception>
#include <string>
#include <vector>

namespace {

int usage(const char* argv0) {
    std::fprintf(stderr,
                 "usage: %s --inspect <file.gguf>\n"
                 "       %s --ref <file.gguf> --tokens <ids> --markers <positions>\n"
                 "\n"
                 "  --inspect  check the file's 427 tensors against the geometry its metadata declares\n"
                 "  --ref      run the CPU reference and print the readout's raw logits (255 per marker) as JSON\n"
                 "             --tokens   the sequence's token ids, comma separated\n"
                 "             --markers  the readout positions, comma separated\n"
                 "             --trace    print the hidden state's sum and norm after each block (to stderr)\n"
                 "             --dump-dir DIR --dump-block N  write that block's stages as raw f32 (a kernel's target)\n"
                 "\n"
                 "The engine (blocks on GPU, readout, /v1/systemone) is not built yet: see the integration study.\n",
                 argv0, argv0);
    return 2;
}

std::vector<int32_t> ids_of(const char* text) {
    std::vector<int32_t> out;
    const char* p = text;
    while (*p) {
        char* end = nullptr;
        const long value = std::strtol(p, &end, 10);
        if (end == p) throw std::runtime_error(std::string("not a number: ") + p);
        out.push_back((int32_t) value);
        p = end;
        if (*p == ',') ++p;
    }
    return out;
}

int inspect_file(const char* path) {
    const strata::GgufFile file(strata::gguf_split_paths(path).front());
    std::string report;
    const std::string err = strata::decider::inspect(file, &report);
    if (!err.empty()) {
        std::fprintf(stderr, "%s: %s\n", path, err.c_str());
        return 1;
    }
    std::printf("%s\n%s\n  architecture guard: PASS\n", path, report.c_str());
    return 0;
}

/// Writes one stage of one block as raw little-endian f32, row-major [tokens][n_embd]: the layout the GPU
/// kernels read and write, so a comparison needs no format.
void write_stage(const std::string& dir, int64_t block, const char* name, const std::vector<float>& values) {
    const std::string path = dir + "/" + name + "_" + std::to_string(block) + ".f32";
    std::FILE* file = std::fopen(path.c_str(), "wb");
    if (!file) throw std::runtime_error("cannot write " + path);
    if (std::fwrite(values.data(), sizeof(float), values.size(), file) != values.size()) {
        std::fclose(file);
        throw std::runtime_error("short write on " + path);
    }
    std::fclose(file);
}

int run_reference(const char* path, const std::vector<int32_t>& tokens, const std::vector<int32_t>& markers,
                  bool tracing, const std::string& dump_dir, int64_t dump_block) {
    strata::decider::ref::Model model;
    std::string err;
    std::fprintf(stderr, "strata-decider: reading %s ...\n", path);
    if (!strata::decider::ref::load(path, model, err)) {
        std::fprintf(stderr, "%s: %s\n", path, err.c_str());
        return 1;
    }
    std::fprintf(stderr, "strata-decider: %lld tokens, %zu markers, %lld blocks ...\n", (long long) tokens.size(),
                 markers.size(), (long long) model.n_layers);
    std::vector<std::vector<float>> logits;
    strata::decider::ref::TraceFn trace;
    if (tracing)
        trace = [](const char* stage, double sum, double norm) {
            std::fprintf(stderr, "trace %-10s sum %12.4f norm %12.4f\n", stage, sum, norm);
        };
    strata::decider::ref::StageFn stage;
    if (!dump_dir.empty())
        stage = [&](int64_t block, const char* name, const std::vector<float>& values) {
            if (block == dump_block) write_stage(dump_dir, block, name, values);
        };
    if (!strata::decider::ref::logits(model, tokens, markers, logits, err, trace, stage)) {
        std::fprintf(stderr, "%s: %s\n", path, err.c_str());
        return 1;
    }
    std::printf("{\"logits\": [");
    for (size_t i = 0; i < logits.size(); ++i) {
        std::printf("%s\n [", i ? "," : "");
        for (size_t j = 0; j < logits[i].size(); ++j) std::printf("%s%.9g", j ? ", " : "", (double) logits[i][j]);
        std::printf("]");
    }
    std::printf("\n]}\n");
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc >= 3 && std::strcmp(argv[1], "--inspect") == 0) return inspect_file(argv[2]);
        if (argc >= 3 && std::strcmp(argv[1], "--ref") == 0) {
            std::vector<int32_t> tokens, markers;
            bool tracing = false;
            std::string dump_dir;
            int64_t dump_block = 0;
            for (int i = 3; i < argc; ++i) {
                if (std::strcmp(argv[i], "--trace") == 0) {
                    tracing = true;
                    continue;
                }
                if (i + 1 >= argc) break;
                if (std::strcmp(argv[i], "--tokens") == 0) tokens = ids_of(argv[i + 1]);
                else if (std::strcmp(argv[i], "--markers") == 0) markers = ids_of(argv[i + 1]);
                else if (std::strcmp(argv[i], "--dump-dir") == 0) dump_dir = argv[i + 1];
                else if (std::strcmp(argv[i], "--dump-block") == 0) dump_block = std::atoll(argv[i + 1]);
                else continue;
                ++i;
            }
            if (tokens.empty() || markers.empty()) {
                std::fprintf(stderr, "strata-decider: --ref needs --tokens and --markers\n");
                return 2;
            }
            return run_reference(argv[2], tokens, markers, tracing, dump_dir, dump_block);
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strata-decider: %s\n", e.what());
        return 1;
    }
    return usage(argv[0]);
}
