// include/strata/minimaxm2/engine.hpp - the MiniMax-M2 (GGUF architecture minimax-m2) engine path as a library:
// `strata` hands a minimax-m2 model to it at the top of main() (the family switch), and strata-m2 is a thin wrapper
// around the same entry point.  See src/minimaxm2/m2_engine.cu.
#pragma once

namespace strata::minimaxm2 {

/// The engine's command line (strata's own flag names where they mean the same thing: --native / --model,
/// --max-context, --prefill / --chunk, --serve, --tokens, --max-new, --no-token-graph; the family's own:
/// --expert-vram, --expert-slots, --hybrid-rows, --cpu-threads).  Returns the process exit code.
int run(int argc, char** argv);

}  // namespace strata::minimaxm2
