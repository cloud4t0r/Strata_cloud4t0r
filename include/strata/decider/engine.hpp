// include/strata/decider/engine.hpp - PPLX Decider Flash's engine path as `strata` hands it over.
//
// The family switch at the top of `strata`'s main reads the file's architecture (`qwen35`) and calls `run` before
// any of the chat engine's state exists, the way `step35` and `minimax-m2` are handed over.  The decision model has
// no generation path, so the only mode is `--serve`: one `DEC` line in, one row of 255 logits per marker out.
#pragma once

namespace strata::decider {

/// `strata --serve --native <file.gguf>`.  Returns the process exit code.
int run(int argc, char** argv);

}  // namespace strata::decider
