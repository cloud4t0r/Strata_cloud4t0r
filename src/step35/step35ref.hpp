// src/step35/step35ref.hpp - the Step-3.7-Flash CPU reference: the same arithmetic the kernels must reproduce,
// written once in plain C++ on dequantized weights, so a kernel can be compared against it (the `q36ref`
// arrangement of the Qwen3.6 path, and the `--ref` mode of strata-step35).
//
// It reads the real file: every projection dequantizes its rows through ggml's own type traits, so the values are
// the file's, not a fixture's.  A sequence of N tokens goes through the blocks in order, with causal attention
// (the sliding window on a sliding block) and the block's own head count, which is what makes it usable as an
// oracle for the GPU path.
#pragma once

#include "strata/artifact/gguf_reader.hpp"
#include "strata/step35/engine.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strata::step35::ref {

/// What the blocks need from the file, read once.
struct Model {
    std::shared_ptr<strata::GgufModel> file;   // GgufModel has no default constructor: the file is opened by load()
    int64_t n_layers = 0, n_embd = 0, n_vocab = 0;
    int64_t head_dim = 0, n_rot_full = 0, n_rot_swa = 0, window = 0;
    int64_t ff_dense = 0, n_expert = 0, top_k = 0, ff_expert = 0, ff_shexp = 0, dense_layers = 0;
    double eps = 0, rope_full = 0, rope_swa = 0, expert_scale = 0;
    bool expert_norm = false;
    std::vector<double> n_head, n_head_kv, is_swa, clamp_exp, clamp_shexp;

    int64_t heads(int64_t l) const { return (int64_t) n_head[(size_t) l]; }
    int64_t kv_heads(int64_t l) const { return (int64_t) n_head_kv[(size_t) l]; }
    bool sliding(int64_t l) const { return is_swa[(size_t) l] != 0; }
    bool moe(int64_t l) const { return l >= dense_layers; }
    int64_t rot(int64_t l) const { return sliding(l) ? n_rot_swa : n_rot_full; }
    double rope_base(int64_t l) const { return sliding(l) ? rope_swa : rope_full; }
};

/// Reads the keys the reference needs.  `err` names the first problem; false on any.
bool load(const std::string& path, Model& m, std::string& err);

/// The block's output for `n_pos` positions of `tokens`, in f32, row-major [n_pos][n_embd].  `layers` stops
/// before the end (0 = the embeddings alone); the whole model is 45 blocks, 3 dense and 42 mixture of experts.
/// `trace`, when given, receives the sum of the hidden state after each block (all positions): the same figure
/// llama.cpp's eval callback prints for `attn_norm_in-<block+1>`, so one run localises a divergence.
bool forward(const Model& m, const std::vector<int32_t>& tokens, int64_t layers, std::vector<float>& out,
             std::string& err, std::vector<double>* trace = nullptr);

/// The last position's logits over the vocabulary, through the file's own output tensor.
bool logits(const Model& m, const std::vector<float>& hidden_last, std::vector<float>& out, std::string& err);

}  // namespace strata::step35::ref
