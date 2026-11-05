// include/strata/decider/inspect.hpp - PPLX Decider Flash's tensor audit: the file's 427 tensors against the
// geometry its own metadata declares.
//
// WHY IT IS SEPARATE FROM THE ARCHITECTURE GUARD.  `check_architecture_qwen35` (artifact/gguf_reader.hpp) reads
// metadata only, so `strata` can refuse a file before any weight is touched.  This audit goes one level down: it
// checks every tensor name, shape and type the engine will read, which is what catches a conversion that kept the
// right header and changed a projection's width.  It needs no GPU, so it is the first thing to run on a new file:
//
//     strata-decider --inspect decider-flash-v1.2-q8_0.gguf
//
// The schema is the official Q8_0 release's (perplexity-ai/pplx-decider-flash-gguf-v1.2, 427 tensors): 24 Gated
// DeltaNet blocks (attention every fourth block is full), dense SwiGLU, tied embeddings, and a vocabulary-sized
// `output.weight` whose only non-zero rows are the 255 readout codes.  That last property is data, not shape, so
// the audit checks the ids are inside the vocabulary and leaves the rows to the loader.
#pragma once

#include "strata/artifact/gguf_reader.hpp"

#include <string>

namespace strata::decider {

/// "" when every tensor the engine reads is present with the shape and type its metadata implies; otherwise the
/// first problem, naming the tensor and what was expected.  When `report` is not null it receives the audit's
/// human-readable summary (counts per role, the geometry, the readout's code count).
std::string inspect(const GgufFile& g, std::string* report = nullptr);

}  // namespace strata::decider
