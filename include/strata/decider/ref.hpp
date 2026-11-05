// src/decider/ref.hpp - PPLX Decider Flash's CPU reference: the same arithmetic the kernels must reproduce, in
// plain C++ over the file's own weights (the `--ref` mode of strata-decider, the step35ref arrangement).
//
// WHAT IT IS FOR.  The decider has six traps that produce plausible numbers when they are wrong (the q/gate
// interleave, the GDN head pairing, the SiLU gate, the zero-centered norms, the assistant suffix, the readout's
// vocabulary indexing).  This reference is what says whether a kernel got them right: it reads the real GGUF,
// dequantizes each row through ggml's own type traits, and runs the 32 blocks in the order the checkpoint does.
//
// WHY THE SEQUENCE MATTERS.  The full-attention layers are NON-CAUSAL (every position sees the whole sequence,
// including the assistant suffix that is never generated), the Gated DeltaNet layers are causal, and the
// convolution carries four positions of history.  So a position cannot be computed alone: `logits` takes the whole
// token sequence and reads the head at the markers.
//
// The text-only path.  mRoPE degenerates for text (the three position streams are equal), so the rotary part is
// the standard partial RoPE on the first 64 of 256 head dims, with the rotate-half pairing (i, i + 32).
#pragma once

#include "strata/artifact/gguf_reader.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace strata::decider::ref {

/// What the blocks need from the file, read once.
struct Model {
    std::unique_ptr<strata::GgufFile> file;   // GgufFile has no default constructor: load() opens it
    int64_t n_layers = 0, n_embd = 0, n_ff = 0;
    int64_t n_head = 0, n_head_kv = 0, head_dim = 0, n_rot = 0, kv_dim = 0;
    int64_t gdn_kh = 0, gdn_vh = 0, gdn_kd = 0, gdn_vd = 0, conv_k = 0, conv_dim = 0;
    int64_t n_vocab = 0;
    double eps = 0, rope_base = 0, temperature = 0;
    std::vector<int64_t> codes;               ///< the 255 readout codes, as vocabulary ids
    std::vector<int64_t> recurrent;           ///< per block: 1 = GDN, 0 = full attention
};

/// Reads the keys and the tensor names the reference needs.  `err` names the first problem; false on any.
bool load(const std::string& path, Model& m, std::string& err);

/// Where a divergence starts: the hidden state's sum and L2 norm after the embeddings, after each block and after
/// the final norm.  One run against the reference package localises a wrong block without a hidden-state dump.
using TraceFn = std::function<void(const char* stage, double sum, double norm)>;

/// A block's intermediate tensor, as it is computed: `stage` is "xn_attn" (the mixer's normalised input),
/// "mixer" (the mixer's output), "x_post" (the residual after it, the MLP's un-normalised input), "xn_post" (the
/// MLP's normalised input) or "mlp" (the MLP's output), and the
/// values are row-major [tokens][n_embd].  This is what a kernel is compared against, block by block.
using StageFn = std::function<void(int64_t block, const char* stage, const std::vector<float>& values)>;

/// The readout's raw logits (255 per marker, before temperature) for `tokens`, read at `markers`.
bool logits(const Model& m, const std::vector<int32_t>& tokens, const std::vector<int32_t>& markers,
            std::vector<std::vector<float>>& out, std::string& err, const TraceFn& trace = {},
            const StageFn& stage = {});

}  // namespace strata::decider::ref
