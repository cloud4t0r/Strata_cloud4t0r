// src/decider/decider_engine.cpp - the family switch's landing point for a `qwen35` file (see engine.hpp).
//
// It takes the engine's own spelling of the arguments and nothing else: `--native <file>` names the GGUF, `--serve`
// is the only mode.  A decision model generates no text, so there is no context to plan, no prompt to chunk and no
// drafter to feed - a flag that would change any of those is refused rather than silently ignored.
#include "strata/decider/engine.hpp"

#include "strata/decider/gpu.hpp"

#include <cstdio>
#include <cstring>
#include <string>

namespace strata::decider {

int run(int argc, char** argv) {
    std::string path;
    bool serve = false;
    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        if (std::strcmp(arg, "--native") == 0 || std::strcmp(arg, "--model") == 0) {
            if (i + 1 < argc) path = argv[++i];
        } else if (std::strcmp(arg, "--serve") == 0) {
            serve = true;
        }
    }
    if (path.empty()) {
        std::fprintf(stderr, "strata-decider: --native <file.gguf> is needed\n");
        return 2;
    }
    if (!serve) {
        std::fprintf(stderr, "strata-decider: a decision model only serves (--serve); it generates no text\n");
        return 2;
    }
    return gpu::serve(path);
}

}  // namespace strata::decider
