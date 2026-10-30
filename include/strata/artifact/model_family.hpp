// include/strata/artifact/model_family.hpp - which model family a GGUF holds, from its general.architecture.
//
// Strata was written for one model (Qwen3.8-Flash-Next, `qwen4exp`).  Qwen3.6-35B-A3B (`qwen35moe`) is a close
// cousin (same tokenizer, the same Gated-DeltaNet + gated-attention 3:1 layer pattern, the same head dims), so the
// reader, the geometry and the layout tables know both.  MiniMax-M2 (`minimax-m2`) is a third family: full
// attention on every layer (no GDN, no sparse attention), 256 experts top-8, NO shared expert, per-layer QK
// norms, and a router with a per-expert selection bias.  Step-3.7-Flash (`step35`) is a fourth: hybrid attention
// (one full layer in four, sliding window 512 on the others), a head count that changes with the layer type
// (64 full / 96 SWA), 288 experts top-8 WITH a shared expert, a head-wise attention gate and SwiGLU clamping on
// its last two layers.  The ENGINE runs only the families `model_family_runs` names; a recognised family it does
// not run yet is refused with `model_family_not_supported`, not mis-run.
//
// No dependencies: the GGUF reader (the architecture guard) and the core geometry both include this.
#pragma once

#include <string>

namespace strata {

enum class ModelFamily {
    Unknown = 0,
    Qwen4Exp,    ///< Qwen3.8-Flash-Next and its fine-tunes / pruned variants (Swift 1.5, the Coder)
    Qwen35Moe,   ///< Qwen3.6-35B-A3B (Unsloth's MTP GGUFs): plain pre-norm residual, dense attention, nextn MTP block
    MiniMaxM2,   ///< MiniMax-M2 (230B A10B): full attention on every layer, 256 experts top-8, no shared expert
    Step35,      ///< Step-3.5/3.7-Flash (196B A11B): hybrid SWA/full attention, 288 experts top-8 + shared expert, MTP heads
};

inline ModelFamily model_family_from_arch(const std::string& arch) {
    if (arch == "qwen4exp") return ModelFamily::Qwen4Exp;
    if (arch == "qwen35moe") return ModelFamily::Qwen35Moe;
    if (arch == "minimax-m2") return ModelFamily::MiniMaxM2;
    if (arch == "step35") return ModelFamily::Step35;
    return ModelFamily::Unknown;
}

/// The GGUF architecture string (also the metadata key prefix: "<arch>.block_count").
inline const char* model_family_arch(ModelFamily f) {
    switch (f) {
    case ModelFamily::Qwen4Exp:
        return "qwen4exp";
    case ModelFamily::Qwen35Moe:
        return "qwen35moe";
    case ModelFamily::MiniMaxM2:
        return "minimax-m2";
    case ModelFamily::Step35:
        return "step35";
    default:
        return "";
    }
}

/// The model's name for messages.
inline const char* model_family_title(ModelFamily f) {
    switch (f) {
    case ModelFamily::Qwen4Exp:
        return "Qwen3.8-Flash-Next";
    case ModelFamily::Qwen35Moe:
        return "Qwen3.6-35B-A3B";
    case ModelFamily::MiniMaxM2:
        return "MiniMax-M2";
    case ModelFamily::Step35:
        return "Step-3.7-Flash";
    default:
        return "an unknown model";
    }
}

/// Whether Flash-Next's engine path runs the family.  Qwen3.6 has an engine path of its own (src/qwen35) and
/// MiniMax-M2 has one of its own (src/minimaxm2), which the CUDA and HIP builds of `strata` switch to at the top
/// of main(); the Intel Arc (SYCL) build has none yet, so there (and on Flash-Next's path) the family is refused
/// with `model_family_not_supported`.
inline bool model_family_runs(ModelFamily f) { return f == ModelFamily::Qwen4Exp; }

/// The refusal for a recognised family the engine does not run (empty when it runs it).
inline std::string model_family_not_supported(ModelFamily f) {
    if (model_family_runs(f)) return {};
    if (f == ModelFamily::Unknown) return "this is not a model family Strata knows";
    return std::string(model_family_title(f)) + " (architecture '" + model_family_arch(f) +
           "') runs on the NVIDIA (CUDA) and AMD (HIP) builds of Strata; this build does not run it yet (the Intel "
           "Arc / SYCL port is in development). Use a CUDA or HIP build, or a Qwen3.8-Flash-Next model file";
}

}  // namespace strata
