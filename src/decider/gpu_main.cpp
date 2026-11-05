// src/decider/gpu_main.cpp - `strata-decider-gpu`, the decider's kernel harness.
//
// It takes one stage of one block, runs it on the GPU over the hidden state the CPU reference wrote, and writes the
// result in the same layout.  tools/decider_block_parity.py runs both sides and compares them; a kernel is correct
// when the two agree, and the stage it belongs to says which one is wrong when they do not.
//
//     strata-decider --ref <gguf> --tokens ... --markers ... --dump-dir /tmp/d --dump-block 0
//     strata-decider-gpu --gguf <gguf> --block 0 --in /tmp/d/xn_post_0.f32 --out /tmp/d/mlp_gpu_0.f32
#include "strata/decider/gpu.hpp"

#include <cstdint>
#include <vector>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <exception>
#include <string>

namespace {

int usage(const char* argv0) {
    std::fprintf(stderr,
                 "usage: %s --gguf <file.gguf> --block N --stage mlp|gdn|attn --in <hidden.f32> --out <out.f32>\n"
                 "\n"
                 "  --in   the stage's input, raw f32 [tokens][n_embd], as `strata-decider --ref --dump-dir DIR\n"
                 "         --dump-block N` writes it: x_post_N.f32 for the MLP, xn_attn_N.f32 for the GDN\n"
                 "  --out  receives the stage's output, same layout (mlp_N.f32, mixer_N.f32)\n"
                 "  --xn-out / --conv-out / --core-out / --attn-out  also write an intermediate stage\n"
                 "\n"
                 "       %s --gguf <file.gguf> --stage full --tokens <ids> --markers <positions>\n"
                 "\n"
                 "  --stage full runs the whole model (embeddings, 32 blocks, final norm, readout) and prints the\n"
                 "  readout's raw logits as JSON, the same shape `strata-decider --ref` prints.\n",
                 argv0);
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

}  // namespace

int main(int argc, char** argv) {
    std::string gguf, in_path, out_path, xn_path, conv_path, core_path, attn_path, stage = "mlp";
    std::vector<int32_t> tokens, markers;
    bool trace = false;
    long long block = -1;
    for (int i = 1; i < argc; ++i) {
        const std::string flag = argv[i];
        if (flag == "--trace") {
            trace = true;
            continue;
        }
        if (i + 1 >= argc) break;
        if (flag == "--gguf") gguf = argv[++i];
        else if (flag == "--in") in_path = argv[++i];
        else if (flag == "--out") out_path = argv[++i];
        else if (flag == "--block") block = std::atoll(argv[++i]);
        else if (flag == "--xn-out") xn_path = argv[++i];
        else if (flag == "--conv-out") conv_path = argv[++i];
        else if (flag == "--core-out") core_path = argv[++i];
        else if (flag == "--attn-out") attn_path = argv[++i];
        else if (flag == "--stage") stage = argv[++i];
        else if (flag == "--tokens") tokens = ids_of(argv[++i]);
        else if (flag == "--markers") markers = ids_of(argv[++i]);
    }
    if (stage == "full") {
        if (gguf.empty() || tokens.empty() || markers.empty()) return usage(argv[0]);
        std::string err;
        std::vector<std::vector<float>> logits;
        try {
            if (!strata::decider::gpu::logits(gguf, tokens, markers, logits, err, trace)) {
                std::fprintf(stderr, "strata-decider-gpu: %s\n", err.c_str());
                return 1;
            }
        } catch (const std::exception& e) {
            std::fprintf(stderr, "strata-decider-gpu: %s\n", e.what());
            return 1;
        }
        std::printf("{\"logits\": [");
        for (size_t i = 0; i < logits.size(); ++i) {
            std::printf("%s\n [", i ? "," : "");
            for (size_t j = 0; j < logits[i].size(); ++j)
                std::printf("%s%.9g", j ? ", " : "", (double) logits[i][j]);
            std::printf("]");
        }
        std::printf("\n]}\n");
        return 0;
    }
    if (gguf.empty() || in_path.empty() || out_path.empty() || block < 0) return usage(argv[0]);
    std::string err;
    try {
        bool ok = false;
        if (stage == "gdn") ok = strata::decider::gpu::gdn(gguf, (int64_t) block, in_path, out_path, err, conv_path,
                                                           core_path);
        else if (stage == "attn") ok = strata::decider::gpu::attention(gguf, (int64_t) block, in_path, out_path, err,
                                                                       attn_path);
        else ok = strata::decider::gpu::mlp(gguf, (int64_t) block, in_path, out_path, err, xn_path);
        if (!ok) {
            std::fprintf(stderr, "strata-decider-gpu: %s\n", err.c_str());
            return 1;
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strata-decider-gpu: %s\n", e.what());
        return 1;
    }
    std::printf("%s\n", out_path.c_str());
    return 0;
}
