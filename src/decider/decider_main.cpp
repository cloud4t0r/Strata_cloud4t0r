// src/decider/decider_main.cpp - `strata-decider`, the decision model's own entry point.
//
// WHAT EXISTS TODAY.  The tensor audit (`--inspect`), which needs no GPU: it is the first thing to run on a new
// file, and the only thing this build can do with one.  The engine itself (the 32 blocks on CUDA/HIP, the readout
// at the markers, the `DEC` operation of serve/server.py) is M2-M4 of the integration study; until it lands, a
// request that is not `--inspect` says so and exits 2, rather than pretending to decide.
//
// The family is recognised by `strata` (artifact/model_family.hpp: architecture qwen35), which refuses it with the
// decision-model reason instead of running Flash-Next's chat path over it.
#include "strata/artifact/gguf_reader.hpp"
#include "strata/decider/inspect.hpp"

#include <cstdio>
#include <cstring>
#include <exception>
#include <string>

namespace {

int usage(const char* argv0) {
    std::fprintf(stderr,
                 "usage: %s --inspect <file.gguf>\n"
                 "\n"
                 "  --inspect  check the file's 427 tensors against the geometry its metadata declares\n"
                 "\n"
                 "The engine (blocks, readout, /v1/systemone) is not built yet: see the integration study.\n",
                 argv0);
    return 2;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 3 && std::strcmp(argv[1], "--inspect") == 0) {
        try {
            const strata::GgufFile file(strata::gguf_split_paths(argv[2]).front());
            std::string report;
            const std::string err = strata::decider::inspect(file, &report);
            if (!err.empty()) {
                std::fprintf(stderr, "%s: %s\n", argv[2], err.c_str());
                return 1;
            }
            std::printf("%s\n%s\n  architecture guard: PASS\n", argv[2], report.c_str());
            return 0;
        } catch (const std::exception& e) {
            std::fprintf(stderr, "%s: %s\n", argv[2], e.what());
            return 1;
        }
    }
    return usage(argv[0]);
}
