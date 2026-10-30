// include/strata/step35/engine.hpp - the Step-3.7-Flash (GGUF architecture step35) engine path as a library:
// `strata` hands a step35 model to it at the top of main() (the family switch), and strata-step35 is a thin
// wrapper around the same entry point.  See src/step35/step35_engine.cu.
//
// The model's own keys, read from the file (llama.cpp's names under the "step35." prefix):
//   block_count 45, embedding_length 4096, vocabulary 128 896
//   attention: one full layer in four (sliding_window_pattern), sliding window 512 on the others, head count
//    64 on a full layer and 96 on a sliding one, 8 KV heads, head dim 128, per-head Q/K RMSNorm, partial RoPE
//    (64 of the 128 dims on a full layer, all 128 on a sliding one) with base 5e6 / 1e4
//   experts: 288, top-8, expert FFN 1280, shared expert 1280, SiLU, sigmoid gating, expert_weights_norm,
//    expert_weights_scale 3.0; layers 0-2 are dense (FFN 11264) and the last two clamp their SwiGLU input
//    (swiglu_clamp_exp 7.0 / swiglu_clamp_shexp 16.0)
//   MTP: three NextN blocks in a file of their own (--mtp), not in the trunk
#pragma once

namespace strata::step35 {

/// The engine's command line.  Today it carries --inspect (read the file, print and check the geometry it
/// declares, then exit: no GPU work) and the flag names the engine path will take (`--model`, `--max-context`,
/// `--tokens`, `--max-new`, `--mtp`, ...).  Returns the process exit code.
int run(int argc, char** argv);

}  // namespace strata::step35
