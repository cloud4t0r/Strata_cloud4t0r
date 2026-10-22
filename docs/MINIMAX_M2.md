# MiniMax-M2 on Strata

MiniMax-M2 (`minimax-m2` GGUF architecture) is a 230B mixture-of-experts model with 10B active parameters
(256 experts, top-8, full attention on every layer, no shared expert, no MTP block). Strata runs it on a
normal PC with one or two graphics cards through the dedicated engine path `src/minimaxm2/`. The `strata`
engine carries that path and switches to it when it opens a `minimax-m2` file; the `strata-m2` binary is the
same path on its own. The model is ~54.7 GB (IQ2_XXS, bartowski's GGUF, two files).

**Status (0.2.2, engine 0.1.47):** single card and two-card layer split both produce the reference token
**8261** for the 6-token prompt `200034,517,5505,300,5969,355`, matching llama.cpp.

## Hardware

- **One card (mono):** RX 9070 (gfx1201) or equivalent with ~16 GB VRAM + enough RAM for the file's page cache.
  The low-RAM mode (`--low-ram`) keeps no copy of the experts in host RAM; they are read from the file on demand.
  Without it, the engine copies the ~49 GB of experts into RAM. See [Low-RAM mode or normal mode](#low-ram-mode-or-normal-mode).
- **Two cards (split):** RX 9070 (gfx1201) as card 0 + RX 6800 (gfx1030) as card 1. The first card runs
  layers 0-30, the second runs layers 31-61 and the output head. The head runs on the primary (card 0);
  the last row's hidden (12 KB) crosses from card 1.
- Linux with ROCm 10 under `/opt/rocm`. Windows/Mac paths are not validated for M2 yet.

## Install with setup (Linux)

```sh
./setup.sh --family minimax2
```

What setup does for this model:

- **It compiles the engine from source.** A ready-made engine is not used, except with `--prebuilt`. Setup
  checks that the engine runs the model (`strata --families` lists `minimax-m2`) and stops if it does not.
- **It also compiles `strata-m2`** (the CMake target of the same name) in the engine's build folder
  (`build/`, `build-cuda12/` or `build-hip/`) and copies it next to the engine. If that build fails, or cannot
  be done (a ready-made engine, Intel Arc, Windows), setup says so and goes on: the model runs through
  `strata`, which does not need `strata-m2`. To retry by hand: `cmake --build <build folder> --target strata-m2`.
- **It downloads the two files** of IQ2_XXS from `bartowski/MiniMaxAI_MiniMax-M2-GGUF` (folder
  `MiniMaxAI_MiniMax-M2-IQ2_XXS/`): `-00001-of-00002.gguf` (39.9 GB) and `-00002-of-00002.gguf` (14.8 GB),
  54.7 GB in all. It checks the size and SHA-256 of each one.
- **No pictures.** The model has no image reader and no picture file to download. `--vision yes|gpu|cpu`
  only gives a warning; images stay off.
- **One card.** With several cards, setup uses one and says so. The two-card split is run by hand (below).
- **It picks the low-RAM mode or the normal mode** (next section) and writes the start script
  `run_minimax_m2.sh` in the Strata folder.

`./setup.sh --family minimax2 --m2-low-ram on` (or `off`) forces the mode. The end-to-end setup has not been
tested yet; only setup's GPU-free tests (`python tools/test_setup_minimax.py`) and a HIP build have.

### Low-RAM mode or normal mode

| Mode | What it does | RAM setup plans for |
| --- | --- | ---: |
| **Low-RAM** (`--low-ram`) | the experts are read where they lie in the file; no copy in RAM | 24 GB |
| **Normal** (no `--low-ram`) | the ~49 GB of experts are copied into RAM | 96 GB |

`--m2-low-ram auto` (the default) picks the low-RAM mode when the PC has less than 96 GB of RAM in total, and
the normal mode from 96 GB. Setup prints the mode and why. **The 96 GB line is derived from the code, not
measured:** on a 64 GB box the copy did not fit beside the file and the machine thrashed (see
[MINIMAX.md](MINIMAX.md#what-the-minimax-m2-integration-found-on-the-machine-2026-10-20)); 96 GB leaves room for
the copy, the page cache and the OS. With `--m2-low-ram off` on a PC under 96 GB, setup warns and asks first (with `--yes`, it warns and goes on: the
flag counts as your choice).

### The start script: `run_minimax_m2.sh`

```sh
./run_minimax_m2.sh                    # the mode setup chose
M2_LOW_RAM=1 ./run_minimax_m2.sh       # low-RAM mode for this run (same as: ./run_minimax_m2.sh --low-ram)
M2_LOW_RAM=0 ./run_minimax_m2.sh       # normal mode for this run (same as: ./run_minimax_m2.sh --no-low-ram)
M2_DRY_RUN=1 ./run_minimax_m2.sh       # print the command, start nothing
```

It starts the Strata server (`serve/server.py`) with MiniMax-M2's config, on one card. A mode other than
setup's runs from a copy of the config (`.run_minimax_m2.override.json`, next to it); setup's config is not
changed. The script always unsets `STRATA_M2_LOW_RAM`: the engine turns the low-RAM mode on when that variable
exists at all, even set to `0`, so only `--low-ram` decides. On Windows setup writes no such script.

## Build by hand

```sh
cmake -S . -B build-hip -G Ninja \
  -DSTRATA_ENABLE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES="gfx1201;gfx1030" \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_PREFIX_PATH=/opt/rocm -DROCM_PATH=/opt/rocm \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-hip --target strata-m2 -j 12
```

Run with `LD_LIBRARY_PATH=/opt/rocm/lib`.

## Download the model

`./setup.sh --family minimax2` downloads it (entry `minimax2` in `OWN_PATH_FAMILIES`), or fetch it by hand
from `bartowski/MiniMaxAI_MiniMax-M2-GGUF`, folder `MiniMaxAI_MiniMax-M2-IQ2_XXS/`: two shards, 39.9 GB +
14.8 GB = 54.7 GB. Keep both in the same folder and pass the first one (`-00001-of-00002.gguf`): the engine
finds the second beside it.

## Run: single card (mono)

```sh
HIP_VISIBLE_DEVICES=1 LD_LIBRARY_PATH=/opt/rocm/lib \
  ./build-hip/strata-m2 \
    --model /path/to/MiniMaxAI_MiniMax-M2-IQ2_XXS-00001-of-00002.gguf \
    --max-context 4096 \
    --tokens "200034,517,5505,300,5969,355" --n 64 --low-ram
```

Or via the test runner:

```sh
STRATA_M2_DEVICES=1 tools/run_m2_test.sh mono
```

## Run: two cards (layer split)

```sh
HIP_VISIBLE_DEVICES=1,0 LD_LIBRARY_PATH=/opt/rocm/lib \
  ./build-hip/strata-m2 \
    --model /path/to/MiniMaxAI_MiniMax-M2-IQ2_XXS-00001-of-00002.gguf \
    --max-context 4096 \
    --tokens "200034,517,5505,300,5969,355" --n 64 --low-ram \
    --layer-split 31
```

Or via the test runner:

```sh
STRATA_M2_DEVICES=1,0 STRATA_M2_EXTRA="--layer-split 31" tools/run_m2_test.sh split
```

The split point (layer 31) puts 31 layers on each card. The crossing copies the residual and the normed hidden
through host memory (host-staged, not peer-to-peer) because the two cards are different generations
(gfx1201 + gfx1030).

## Environment variables

| Variable | Effect |
|---|---|
| `STRATA_M2_DEVICES` | Comma-separated GPU list for the test runner (e.g. `1,0`) |
| `STRATA_M2_EXTRA` | Extra flags passed to `strata-m2` (e.g. `--layer-split 31`) |
| `STRATA_M2_TIMEOUT` | Max run time in seconds (default 900) |
| `STRATA_M2_SPLIT_COPY` | Set to `peer` to use device-to-device peer copies instead of host staging (experimental) |
| `STRATA_M2_PEER_EXPERIMENTAL` | Set to `1` to enable the peer tier (second card holds experts) |
| `STRATA_M2_DUMP_STATE` | Layer index to dump state for (diagnostics) |
| `STRATA_M2_DUMP_ROWS` | Enable row-level crossing diagnostics |
| `STRATA_M2_DUMP_LAYERS` | Path to write per-layer residual dumps |

## Peer tier (experimental)

A second mechanism keeps the experts on the second card instead of splitting layers. It is behind
`STRATA_M2_PEER_EXPERIMENTAL=1` until bit-exact. Rows are bit-identical to the CPU pool's, and the
answer is 8261.

## Performance (measured on Ryzen 9 5950X, 64 GB DDR4, RX 9070 + RX 6800)

- Decode speed split vs mono: to be measured.
- The layer split runs on both cards; the residual and the normed hidden cross once per verify window
  (a few hundred KB through pinned RAM), not twice per layer.

## Notes

- The build's CMake cache may have a stale `CMAKE_PROJECT_INCLUDE` that must be removed before reconfiguring:
  `sed -i '/CMAKE_PROJECT_INCLUDE/d' build-hip/CMakeCache.txt`
- The machine crashed twice during early testing (VRAM overcommit + two concurrent engine instances). The test
  runner (`tools/run_m2_test.sh`) serializes runs with a timeout.
- M2.5 (same architecture, 62 layers, 256 experts, 8 active) reuses the same engine path; only weights and
  3 config entries change. Smallest GGUF quant is Q2_K (~78-83 GB).
path">
/mnt/gygabyte/llm_models/Strata_cloudat0r/docs/MINIMAX_M2.md