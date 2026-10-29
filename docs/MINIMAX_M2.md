# MiniMax-M2 on Strata

MiniMax-M2 (`minimax-m2` GGUF architecture) is a 230B mixture-of-experts model with 10B active parameters
(256 experts, top-8, full attention on every layer, no shared expert, no MTP block). Strata runs it on a
normal PC with one or two graphics cards through the dedicated engine path `src/minimaxm2/`. The `strata`
engine carries that path and switches to it when it opens a `minimax-m2` file; the `strata-m2` binary is the
same path on its own. The model is ~54.7 GB (IQ2_XXS, bartowski's GGUF, two files).

**Status (0.2.2, engine 0.1.47):** single card and two-card layer split both produce the reference token
**8261** for the 6-token prompt `200034,517,5505,300,5969,355`, matching llama.cpp. The K / V cache has three
modes: FP32 (the engine's default), FP16 (`--kv f16`) and INT8 (`--kv int8`). They all answered 8261 at 4K;
f16 and int8 also at 64K, and int8 at 128K. Against the exact FP32 cache both round the logits a little
(measured: 0.25 and 0.22 on the 50 likeliest tokens at 4K), and either can turn a near-tie
([the cache's precision](#the-caches-precision)).

## Hardware

- **One card (mono):** RX 9070 (gfx1201) or equivalent with ~16 GB VRAM + enough RAM for the file's page cache.
  The low-RAM mode (`--low-ram`) keeps no copy of the experts in host RAM; they are read from the file on demand.
  Without it, the engine copies the ~49 GB of experts into RAM. See [Low-RAM mode or normal mode](#low-ram-mode-or-normal-mode).
- **Two cards (split):** RX 9070 (gfx1201) as card 0 + RX 6800 (gfx1030) as card 1. The first card runs
  layers 0-30, the embedding and the output head. It also holds a copy of every layer's dense weights. The
  second card runs layers 31-61 with its own copy of their dense weights. The last row's hidden (12 KB)
  crosses back from card 1 to the head on card 0. (From the engine's loader code.)
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
- **One card by default.** Without `--gpus`, setup uses one card. The engine is still compiled for every AMD
  card's architecture (RX 9070 + RX 6800: gfx1201 and gfx1030), so `"gpu": 1` in the config does not stop on
  `device kernel image is invalid`.
- **Two cards when you ask** (Linux; AMD or NVIDIA): `./setup.sh --setup --family minimax2 --gpus 1,0`. See
  [Two cards with setup](#two-cards-with-setup). The AMD order is the measured one; NVIDIA runs the same engine
  split through CUDA, but has not been measured here. Windows stays on one card. Qwen3.6 also stays on one card.
- **It picks the context from the VRAM**, with the K / V cache in INT8 (`--m2-kv`, below).
  - One card: the cache must fit on that one card beside the dense weights (1.86 GiB, measured) and the
    engine's buffers (about 1.54 GiB at the prompt chunk setup lets the engine choose, measured). With int8
    setup recommends 64K on a 16 GB card, 64K on 24 GB and 256K on 48 GB; f16 gives 32K, 64K and 128K there.
    It never adds up the VRAM of several cards.
  - Two cards: each card holds the cache of its own layers (about half), its own buffers and its dense
    weights. Setup subtracts the VRAM other programs use right now (the desktop). The tighter card decides.
    Two 16 GB cards get 128K with int8 (measured: 19 and 31 expert slots per layer, 8261, 23.0 tok/s) and 64K
    with f16 (measured: 35 and 47 slots, 8261, 25.2 tok/s). Close other GPU programs before you run setup.
  - For a context that does not fit, setup prints the numbers and asks first. With `--yes`, it goes on only
    if you passed `--context` yourself.
- **It picks the low-RAM mode or the normal mode** (next section) and writes the start script
  `run_minimax_m2.sh` in the Strata folder.
- **It keeps VRAM free when you ask**: `--vram-reserve-mib N` is written to the config's `"args"` as the first
  card's reserve, and `--vram-reserve-later-mib N` as the split's second card's ([VRAM kept
  free](#vram-kept-free)). Without them the engine keeps its own 768 MiB on each card.

`./setup.sh --family minimax2 --m2-low-ram on` (or `off`) forces the mode.
`--m2-kv f32|f16|int8` sets the cache's precision (default `auto`: int8; `f16` and `f32` ask for the wider
caches; [the cache's precision](#the-caches-precision)). The end-to-end setup has not been tested yet; only
setup's GPU-free tests (`python tools/test_setup_minimax.py`) and a HIP build have.

### Low-RAM mode or normal mode

| Mode | What it does | RAM setup plans for |
| --- | --- | ---: |
| **Low-RAM** (`--low-ram`) | the experts are read where they lie in the file; no copy in RAM | 24 GB |
| **Normal** (no `--low-ram`) | a copy of the experts in RAM, capped by `--resident-budget-gib` | 24 GB + what the copy takes |

`--m2-low-ram auto` (the default) keeps the normal mode and writes a budget: the RAM the PC can hand out when
setup runs (`MemAvailable` on Linux, the free physical memory on Windows and macOS) less 8 GiB, and at most
the PC's RAM less 32 GiB (the rest of the machine: measured here, a 30 GiB copy ran on this 62.7 GiB PC and a
34 GiB one swapped), rounded down to a quarter GiB and capped at the 49.10 GiB of experts. The low-RAM mode
takes over only when that leaves less than 4 GiB (under 12 GiB free at setup time, or under 36 GiB of RAM in
total), or when the free RAM cannot be read: a tiny copy is not worth the pinned pages. `--m2-low-ram on`
forces the low-RAM mode; `off` forces the normal mode with the same budget (and, when the free RAM cannot be
read, no budget at all, so the engine copies all 49.10 GiB: setup says so).

**The margin is a snapshot:** it is what was free when you ran setup. If you later keep a virtual machine or a
game beside the model, run setup again or lower `--resident-budget-gib` in the config by hand. The 8 GiB
covers the OS, the desktop, the engine's buffers and the page cache the other experts are read through (the
safe end of the 4 to 8 GiB advised in [A RAM budget for the experts](#a-ram-budget-for-the-experts)), and the
copy is capped at the PC's RAM less 32 GiB (`M2_RESIDENT_OTHER_GIB`): on this 62.7 GiB PC the 30 GiB copy ran
and a 34 GiB one swapped. The old full-copy mode needed 96 GB of RAM in total: on a 64 GB box the copy did not
fit beside the file and the machine thrashed (see
[MINIMAX.md](MINIMAX.md#what-the-minimax-m2-integration-found-on-the-machine-2026-10-20)).

### The cache's precision

The context is bounded by the K / V cache: 8 KV heads x 128 values, K and V, on each of the 62 layers. The
engine's default is **FP32**, the path it has always taken. `--kv f16` keeps the same values in half
precision; `--kv int8` stores 8 bits a value with one scale per cell, per KV head and per block of 32 values
(the Q8_0 rule: the block's largest magnitude / 127, the scale itself a half). What a context token costs on
one card that holds all 62 layers:

| Mode | A context token | VRAM per 32K tokens | Measured |
| --- | ---: | ---: | --- |
| FP32 (engine default) | 507,904 B | 15.5 GiB | the exact path |
| `--kv f16` | 253,952 B | 7.75 GiB | 0.25 from FP32's logits (50 likeliest, 4K) |
| `--kv int8` | 134,912 B (126,976 of cache + 7,936 of scales) | 4.1 GiB | 8261 on the reference prompt; 0.22 from FP32's logits |

Setup writes `--kv int8` (`--m2-kv auto|f32|f16|int8`, default `auto` = int8); `--m2-kv f16` and
`--m2-kv f32` ask for the wider caches, `--m2-kv f32` restoring the engine's default byte for byte.

Measured on this machine (RX 6800 gfx1030 + RX 9070 gfx1201, `--layer-split 31`, `--resident-budget-gib 0.25`
so that the VRAM alone decides, `--prefill 512` for a like-for-like chunk, IQ2_XXS, the reference prompt whose
answer is 8261, one session):

| Context | Mode | VRAM free on card 0 | Expert slots per layer (card 0 / card 1) | Answer |
| ---: | --- | ---: | ---: | --- |
| 4096 | FP32 | 11.84 GiB | 115 / 128 | 8261 |
| 4096 | f16 | 12.33 GiB | 120 / 132 | 8261 |
| 4096 | int8 | 12.57 GiB | 123 / 133 | 8261 |
| 65536 | f16 | 4.11 GiB | 35 / 47 | 8261 |
| 65536 | int8 | 7.75 GiB | 72 / 85 | 8261 |
| 131072 | int8 | 2.62 GiB | 19 / 31 | 8261 |
| 16384, one card | int8 | 9.78 GiB | 51 (all 62 layers there) | 8261 |

A cache that does not fit is refused with the numbers: `not enough VRAM on card 0 for a context of 65536: the
K / V cache of its 31 layers needs 15.50 GiB (fp32) and 13.66 GiB are free after the weights, less 1.50 GiB
for the engine's buffers: use a shorter --max-context (at most about 51200 on this card), --kv f16 (half of
it) or --kv int8 (a quarter)`.

**The precision cost.** FP16 and INT8 both round what the attention reads. Measured at 4K on the reference
prompt with the expert placement pinned (`--expert-slots 115`, so that the modes differ only in their cache),
against the exact FP32 cache over the first six steps: FP16 moves the 50 likeliest logits by 0.25 on average
(0.14 to 0.42 a step), INT8 by 0.22 (0.13 to 0.30) - as close, for half the cache. At the token level a
near-tie decides: the 6th step's top-2 gap is 0.11, so both FP16 and INT8 write 275 where FP32 writes 17795,
and with that placement their texts stay identical to each other's over the 32 tokens measured. Over a longer
run the texts do part: with `--resident-budget-gib 30` (the modes' placements differ there, 111 and 123
against 114 and 126), INT8 kept FP16's 64 tokens for 19, then wrote a full stop where FP16 wrote `, a city`
(tokens 44 416 2839). The scales cover 32 values, as Q8_0 keeps them, where they used to cover a whole KV
head's 128; with one scale a head the same eight steps moved by 0.36 over the 50 likeliest tokens, and the
scale itself is a half (2 bytes for 32 values, 7,936 B a token). The VRAM saved is real - at 64K int8 doubles
the experts kept in VRAM against f16 (72 and 85 a layer against 35 and 47) and it is what makes 128K fit
there (19 and 31 a layer, 2.62 GiB free) - so use it where the context matters more than the exact wording,
and f16 everywhere else.

**The cache and the prompt chunk.** A bigger prompt chunk is faster on a long prompt, and its buffers take
VRAM from the experts. `--prefill auto` (setup's choice) takes 4096 when every expert fits in VRAM, 2048 in
the hybrid mode when 7.5 GiB or more are free, 1024 from 5.5 GiB and 512 below. Measured at 4K with f16:
12.33 GiB free at a chunk of 512, 11.45 at 2048, 10.27 at 4096 - a row of the chunk costs 617,200 B (0.59
MiB) and 0.36 GiB are fixed. At 64K int8 the engine chose 2048, and the chunk's own buffers cost about nine
expert slots a layer: 63 and 76 slots there against 72 and 85 at a chunk of 512.

**What this repository's service runs.** `strata-minimax2-iq2_xxs.json`: 32K, `--kv int8`, `--low-ram`,
`--pool-workers 15`, `--prefill auto`, `--layer-split 31`. On this machine the engine printed
`context 32768 cells, prompt chunk 2048, K / V in INT8, 9.43 GiB of VRAM free`, put 90 of the 256 experts a layer
on card 0 (8.63 GiB) and 102 on card 1 (9.78 GiB), no host copy: 4.5 s of placement and 26.1 s of warm-up with
the file's pages in the page cache (192.4 s when they are not). The API then read a 1,828-token prompt in 11.1 s
(165.6 tok/s) and generated at 24.7 tok/s, and a 5,428-token one in 39.8 s with 912 tokens reused. The prompt's
experts are streamed through host RAM ([Prompt streaming](#prompt-streaming)). The Terminal's other `--kv` modes
(`q4_0`, `k8v4`) belong to the Flash-Next engine: `strata-m2` says they are not modes of this engine and keeps the
mode it has (`f32`, `f16` or `int8`).

**A warm turn reuses the start of the prompt it already read.** An agent client sends the whole conversation
again on every turn, and the reply it shows may not be the tokens this engine generated. Until now a request
reused the session's state only when the new prompt began with *every* held token, prompt and reply byte for
byte; a re-rendered reply failed that test and the whole 8K prompt was read again (every request in the log
said `0 reused`). The session now keeps the common start and rewinds to it (the K / V cells past it are never
read, since the attention stops at the cell it writes), so only the new tokens are read. Measured on this
service with a 1,435-token prompt at temperature 0: the first turn read 1,435 tokens in 190.3 s (7.5 tok/s);
the next, `1,444 tokens = 1,431 reused + 13 read`, took 2.2 s. The same 37-token prompt sent twice gave
`37 = 4 reused + 33 read` then `37 = 36 reused + 1 read`, with the same 8 generated tokens both times. There
is one session (no second cache in 32 GB of VRAM at 128K): a request whose prompt shares no start with the
held one still begins from zero and leaves its own state there.

### Prompt streaming

The hybrid mode keeps some experts in VRAM (the slots) and pools the rest on the CPU. On a prompt the pool used
to compute every pooled entry on the CPU, which bound the whole read: 1,828 tokens took 140 s (13.0 tok/s)
whatever the cards did, with the cards mostly idle. The prompt path now repacks the pooled experts of each layer
into a pinned host buffer (1.95 GiB, two buffers, no VRAM at all) and runs them on the card with the native
grouped kernels, the slots' experts read straight from their VRAM blob. The GPU reads the buffer over PCIe while
the CPU fills the next layer's in the background.

Measured on this service (32K, `--low-ram`, `--pool-workers 15`, `--layer-split 31`): 1,828 tokens read in 11.1 s
(165.6 tok/s) against 140.1 s (13.0 tok/s) before, and a 5,428-token prompt in 39.8 s (4,516 read at 113.5 tok/s,
912 reused). The decode is untouched (24.7 tok/s on a 200-token answer, against 22-31 before): the staging takes
no VRAM, so the slots stay at 90 and 102 experts a layer. A tool call and a 1,089-token prompt answer the same as
before, and the same prompt twice gives the same tokens.

`STRATA_M2_NO_STREAM_NATIVE=1` puts the prompt back on the CPU pool. `STRATA_M2_STREAM_VRAM=1` keeps the staging
in VRAM instead (two buffers a card, 2 GiB each): it read the same 1,828 tokens at 144.0 tok/s and cost 16 slots a
card (90 and 102 down to 70 and 84), so it buys nothing and stays off.

The stream needs the card a layer runs on: a kernel of card 1 cannot read card 0's memory, and the first version
faulted with `Memory access fault ... Page not present` on the card 1 layers exactly (the staging pair, its copy
stream and its events were all card 0's). The staging is one pair of host buffers, read by both cards; the
"buffer is free" events are one pair per card, since an event belongs to the card that creates it.

The fill of that buffer was one thread's work: a staging buffer holds all 256 experts of a layer (1.32 GiB, so
~5.3 MiB an expert) and `stream_kick` copied the pool's share of it — 194 experts, ~1 GiB — before the layer could
start, 1.6 s per layer. It now goes to a pool of 16 threads, each taking its own stretch of the layer's experts: the
experts are disjoint slices of the buffer and a copy writes nothing else, so no lock and no ordering is needed. The
launches stay on the calling thread, so the card is queued while the copies run.

Measured on this machine (RX 9070 + RX 6800, 16-core Ryzen 9 5950X, 32K, `--low-ram`, `--layer-split 31`), a
512-token prompt on the native streamed path, one thread against the pool of 16: 96.6 s (5.3 tok/s) against 19.7 s
(26.0 tok/s) with the file in the page cache, and 91.7 s (5.6 tok/s) against 30.4 s (16.8 tok/s) with the cache
dropped. The curve bends at 16 threads: 6 gave 28.9 s, 12 gave 23.9 s, 24 gave 20.6 s. `STRATA_M2_FILL_THREADS=N`
sets the count (0 puts the fill back on one thread); the default is 16, at most one less than the machine has cores.

The same bytes land in the same places, so the answer does not move: a 512-token prompt gave byte-identical full
logits with one thread and with 16, and the reference prompt still answers `8261`.

### Which prompt path a layer takes

Two prompt paths exist. `stream_native` repacks a layer's pooled experts into the staging buffer and runs them with
the native grouped kernels, reading each resident expert from its VRAM blob. The MMQ path runs llama.cpp's int8
matrix kernels over one base pointer per layer, stepping expert by expert with `L.stride`.

MMQ can therefore only read a layer whose experts all sit in one block: every expert resident, so the base is the
layer's VRAM blob. A hybrid layer streams only the experts WITHOUT a slot into the staging and leaves the residents'
places untouched, so MMQ used to read those experts from whatever the buffer held. The prompt's logits came out as
garbage, `k_argmax` left its `INT32_MAX` sentinel in `pick`, and the next decode step handed that id to the
embedding lookup, which computes `table + 2^31 * row_bytes`: the card aborted with
`HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION` inside `embed_rows_kernel`. It took a prompt above `--hybrid-rows`
(the default 128) and at least one resident expert, and it also hit the 27 October binary. Hybrid layers now take
the native streamed path; MMQ is left to the all-resident layers.

On this machine the native path is also the faster one at an equal placement: a 512-token prompt with
`--expert-vram 0` (RX 9070 + RX 6800, IQ2_XXS) read at 10.5 tok/s against MMQ's 8.8.

### A RAM budget for the experts

Without `--low-ram` the engine copies the pooled experts of every layer into pinned host RAM: 49.10 GiB for M2
(measured) when nothing is in VRAM. That is what the normal mode buys: the pool never reads the file again.
`--resident-budget-gib N` caps the copy at N GiB.

The copy holds the experts the cards do NOT hold (the pool's): the ones with a VRAM slot are in VRAM already, so
pinning them was wasted. The rest are read where they lie in the file, as in the low-RAM mode. Each layer's
arena is sized on that layer's POOL and allocated once the cards' slots are placed: a budget that covers the
whole pool pins all of it, a smaller one pins the first layers' pools and leaves the later layers to the file.
It used to be sized on a per-layer share of the budget (`budget / layers / stride`), which on card 0's layers
came to 158 experts where the pool has 166: the last 8 were read out of the file at every decode step, and the
arena still took 30.42 GiB where 22.05 GiB hold the whole pool.

```sh
./build-hip2/strata-m2 --model ... --resident-budget-gib 30
```

Measured with `--resident-budget-gib 30.5`: the engine reports `22.05 of 49.10 GiB in pinned host memory (64-166
of 256 per layer, the experts 90-255: the pool's)` — 90-255 for card 0's layers (that card holds the slots 0-89)
and 192-255 for card 1's layers (they hold 90-191). The budget covers the whole pool, so a decode step reads
nothing of the pool out of the file. The flag does not change the VRAM slots (those come from the free VRAM).

**It does not make the decode faster, and on this machine it makes it a little slower.** The threads that
compute the pooled experts read the same bytes either way — from the pinned copy or from the page cache; a
read-bandwidth run on this box put the two at 45.7 and 44.4 GB/s (16 threads, 4 GiB each). What pinning changes is
where the RAM goes: 22.05 GiB pinned leaves the model file 40 GiB of page cache on a 62 GiB box, so the dense
weights the layers still read from the file are evicted and come back from disk — 914 MiB read during one
4,895-token prompt, 259 to 605 MiB per 200-token answer. One session, the same 4,895-token prompt and a 200-token
answer:

| mode | prompt | decode | disk reads |
|---|---|---|---|
| `--low-ram` | 133-135 tok/s | 24.3-24.8 tok/s | 0 MiB |
| `--resident-budget-gib 30.5` | 121-125 tok/s | 22.5-24.0 tok/s | 914 MiB a prompt, 259-605 MiB an answer |

The budget earns its place when the file does NOT fit in RAM — a 16 or 32 GiB box, or a file on a slow disk:
there the page cache cannot hold the pool and the pinned copy is what keeps those bytes off the disk. On 62 GiB
with a 51 GiB file the page cache does that job for free, so the service here keeps `--low-ram`.

`--low-ram` wins over the budget: it prints `--low-ram keeps no host copy: --resident-budget-gib 30 ignored`
and keeps no copy at all.

Sizing it: the copy, the page cache and the OS share the RAM. On this machine (62.7 GiB, a 6 GiB virtual
machine running) a budget of 34 GiB put 6 of the 7 GiB of swap to use. Leave 4 to 8 GiB of headroom below what
is free, or stop the other programs. **Setup does that for you:** `m2_resident_budget_gib` takes the free RAM
when setup runs less 8 GiB (`M2_RESIDENT_MARGIN_GIB`), also caps it at the PC's RAM less 32 GiB
(`M2_RESIDENT_OTHER_GIB`), rounds down to a quarter GiB and caps it at 49.10 GiB; below 4 GiB of copy it picks
the low-RAM mode instead. It prints the number it wrote (`RAM budget: 30.5 GiB of the 49.1 GiB of experts in
RAM`), so the config records what the machine looked like at that moment.

`--resident-budget-gib N` on setup's command line sets the copy yourself (the engine takes 0.25 GiB or more):
`./setup.sh --family minimax2 --resident-budget-gib 40`. It keeps the normal mode even where `auto` would pick
the low-RAM one; more than the 49.10 GiB of experts keeps all of them. `--m2-low-ram on` still wins: the engine
prints `--low-ram keeps no host copy: --resident-budget-gib 40 ignored`.

### VRAM kept free

The expert slots are sized on each card's free VRAM **less a reserve**. `--vram-reserve-mib N` sets that reserve
on the first card (default 768 MiB); `--vram-reserve-later-mib N` sets it on the split's card, and on the peer's
card when `--peer-device` puts it on the second card. The later card follows the first card's value unless it is
given, so `--vram-reserve-mib 300 --vram-reserve-later-mib 1800` gives the first card 300 MiB and the second
1800 MiB. Negative values are read as 0.

Measured here at `--max-context 2048`, `--low-ram`, `STRATA_Q36_FIXED_SLOTS=1` (the slots do not rotate), the
reference prompt:

| Setting | Card 0 slots | Card 1 slots |
| --- | ---: | ---: |
| default (768 MiB), no split | 61 (11.70 GiB) | — |
| `--vram-reserve-mib 4096`, no split | 44 (8.44 GiB) | — |
| default (768 MiB), `--layer-split 31` | 124 (11.89 GiB) | 132 (12.66 GiB) |
| `--vram-reserve-later-mib 4096`, `--layer-split 31` | 124 (11.89 GiB) | 103 (9.88 GiB) |

The reserve is what the engine keeps beside the slots for the prompt's staging pair, the CPU part's rows and the
driver; a larger reserve simply leaves more experts to the CPU pool. It does not change a run's answer: the
reference prompt gave 8261 with 768 and with 4096 MiB. When the reserve was given explicitly, the placement does
not fall back to the 256 MiB reserve of the MTP-only path.

Setup passes both through: `./setup.sh --family minimax2 --gpus 1,0 --vram-reserve-mib 600
--vram-reserve-later-mib 2048` writes them in the config's `"args"`, and a later `./setup.sh --vram-reserve-later-mib
1024` (a plain start) updates the value the model keeps. It wrote `--expert-vram` before these options existed:
the experts' total in GiB, computed from the first card's VRAM (`vram - 3.0 - reserve`), so the second card could
not have its own reserve, and the Flash-Next path got the same flag although its engine has no such option. A
config that still carries `--expert-vram` works: the engine reads it as the experts' budget, which the reserve
does not apply to.

### The start script: `run_minimax_m2.sh`

```sh
./run_minimax_m2.sh                    # the mode setup chose
M2_LOW_RAM=1 ./run_minimax_m2.sh       # low-RAM mode for this run (same as: ./run_minimax_m2.sh --low-ram)
M2_LOW_RAM=0 ./run_minimax_m2.sh       # normal mode for this run (same as: ./run_minimax_m2.sh --no-low-ram)
M2_DRY_RUN=1 ./run_minimax_m2.sh       # print the command, start nothing
```

It starts the Strata server (`serve/server.py`) with MiniMax-M2's config, on the card or the two cards the
config names. A mode other than
setup's runs from a copy of the config (`.run_minimax_m2.override.json`, next to it); setup's config is not
changed. The script always unsets `STRATA_M2_LOW_RAM`: the engine turns the low-RAM mode on when that variable
exists at all, even set to `0`, so only `--low-ram` decides. On Windows setup writes no such script.

## Tool calls

M2 has its own wording for tools. Its template takes the definitions in OpenAI's wrapper (`tool.function`) and
writes a call as:

```
<minimax:tool_call>
<invoke name="read_file">
<parameter name="path">notes.txt</parameter>
</invoke>
</minimax:tool_call>
```

The server reads the template at start-up and probes it with both tool shapes, so the request's tools are
wrapped only for a template that expects it. `OutputParser` translates this wording to the form the rest of the
server shares (`<tool_call><function=...>`); each `<invoke>` becomes one call. The streaming events
(`tool_start`, then `tool_args` as the arguments arrive, then `tool_call`) and the OpenAI and Anthropic answers
are therefore the same as for the Qwen template. A forced `tool_choice` is opened in this wording too:
`<minimax:tool_call>` followed by `<invoke name="NAME">`, or `<invoke name="` when any tool may be called.

What the frontend handles: several `<invoke>` blocks in one envelope (parallel calls), string and JSON parameter
values (following the tool's schema, as in the Qwen format), a call inside the thinking when tools are declared,
and both APIs (`/v1/chat/completions`, `/v1/messages`). Tests:
`python -m unittest serve.test_minimax_tools -v`. The Qwen template's output is unchanged.

Limits: the template has no escaping, so a parameter value that itself contains the exact
`</parameter></invoke>` sequence ends there (the Qwen format has the same limit). M2 has no image input, and the
model still decides whether to call a tool. The tests render and parse the format and drive both HTTP APIs with
a mock engine; they do not measure how often the model calls tools.

## Build by hand

```sh
cmake -S . -B build-hip -G Ninja \
  -DSTRATA_ENABLE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES="gfx1201;gfx1030" -DGPU_TARGETS="gfx1201;gfx1030" \
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

### Two cards with setup

```sh
./setup.sh --setup --family minimax2 --gpus 1,0
./run_minimax_m2.sh
```

- The cards are used in the order you give. The first card gets layers 0-30, the embedding, the output head
  and a copy of all the dense weights. The second card gets layers 31-61.
- `1,0` puts the RX 9070 first. It is the only order measured (the run that matched llama.cpp). The repository
  does not record why that order was chosen. The first card carries a bit more (the embedding, the output head
  and the dense weights). `--gpus 0,1` also works; it is not measured. Setup sizes the context on the VRAM free
  at the time, so stop a running model before you run setup.
- On NVIDIA (Linux), `--gpus 0,1` writes the same `"gpu": [0, 1]` and `"layer_split": "31"`; the engine runs the
  split through CUDA. This path has not been measured here.
- `--layer-split N` moves the split (2 to 48; 31 by default, the measured one). Above 48 the engine would not
  copy the second card's weights, so setup refuses it.
- With three or more cards in `--gpus`, setup uses the first two and says so.
- Setup writes `"gpu": [1, 0]` and `"layer_split": "31"` in `strata-minimax2-iq2_xxs.json`. It writes no
  `--layer-split` in `"args"`: the server adds `--layer-split 31` and sets `HIP_VISIBLE_DEVICES=1,0`.
- Setup writes the cache's mode too: `--kv int8` by default, or the one you ask (`--m2-kv f32|f16|int8`; see
  [the cache's precision](#the-caches-precision) for what each one costs), and the experts' RAM budget
  (`--resident-budget-gib`, [A RAM budget for the experts](#a-ram-budget-for-the-experts)).
- The same for the VRAM each card keeps free: `--vram-reserve-mib N` and `--vram-reserve-later-mib N` go into
  `"args"` as given ([VRAM kept free](#vram-kept-free)); neither is written when you leave them out.
- Each card keeps its own expert slots, sized on its own free VRAM after its weights, its half of the cache
  and its buffers. The engine prints `the second card takes 73 of 256 experts per layer (7.8 GiB free)` and
  then both counts (the service here: 61 on card 0 and 73 on card 1); the other experts are read from the file
  (the low-RAM mode, or a RAM budget).
- The server turns `"layer_split": "auto"` (or no `"layer_split"`) into 31 for this model on two cards. Its
  engine has no automatic split.
- Limits: the parallel-requests advice is still sized on the first card only. The VRAM reserve is per card:
  `--vram-reserve-mib` on the first, `--vram-reserve-later-mib` on the split's ([VRAM kept free](#vram-kept-free)).

### Two cards by hand (the engine alone)

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
| `STRATA_M2_NO_STREAM_NATIVE` | Set to any value to keep the prompt on the CPU pool (no streaming) |
| `STRATA_M2_STREAM_VRAM` | Set to any value to stage the streamed experts in VRAM instead of pinned host RAM |
| `STRATA_M2_FILL_THREADS` | Threads that fill the prompt's staging buffer (default 16, one less than the machine's cores at most); `0` puts the fill back on one thread |

## Peer tier (experimental)

A second mechanism keeps the experts on the second card instead of splitting layers. It is behind
`STRATA_M2_PEER_EXPERIMENTAL=1` until it is validated. Its rows come from the same kernels as the first card's
slots; the CPU pool's rows come from its own code, and the two are not bit-identical: on layer 0 of a 4-token
prompt, a peer row and the CPU pool's row differ by about 3% in L2 over the first twelve values (up to 33% on one
of them). An entry the peer computes therefore gets a slightly different answer than the same entry on the CPU
pool, while both give the reference **8261** for `200034,517,5505,300,5969,355`.

`--peer-device 1 --expert-slots 24` gives the first card 24 experts of every layer and the second card the ones
that follow them (73 of 256 a layer, 14.00 GiB on the RX 9070 here); the rest go to the CPU pool. The second
card's share is grouped with its own slot table, and its rows are written into the same pinned host rows the CPU
pool writes.

The peer and a layer split can share the second card: `--layer-split 31 --peer-device 1` puts the layers from 31
on card 1 as well as the peer's tier, so the two come out of the same VRAM and the placement measures both
together. Here that gives 24 slots for the split's layers (2.30 GiB) and 51 experts a layer for the peer
(9.78 GiB), with 8.96 GiB still free. It used to charge the peer alone against the card's free memory: it took
73 experts (14.00 GiB) beside the split's 2.30 GiB, more than the card has, the driver served the excess from host
memory, and a kernel read a page that was not there (`HSA_STATUS_ERROR_MEMORY_FAULT` in `k_route_group_m2`, GPU 1).

`moe_hybrid_peer` also assumed the layer lived on card 0: it copied the ids and the activations with source device 0
and left device 0 current. With a split layer (its buffers on card 1) the peer copy named the wrong source, and
every later launch on card 1's stream — the merge included — ran with card 0 current: the next native kernel
launch failed with "native MMVQ launch: invalid resource handle", at the first split layer. Fixed: the copy comes
from the layer's own card (`card_of`), is a plain copy when that is the peer's card, and the layer's card is
current again afterwards. Measured with fixed slots: exit 0, no fault, **8261** on
`200034,517,5505,300,5969,355`, and a 64-token run with slot replacements (1294) completed. Without the split the
counts are unchanged (542 of 5456 entries on card 0, 3378 on the pool, 1536 on the peer).

Fixed as well: with the split, the peer's range started at the split card's slot count (`S`, 24) instead of after
its slots (`S + S_split`, 48). The first 24 experts of the peer's range were already in the split card's slots:
loaded twice (2.30 GiB) and computed twice, the peer's rows being ignored by the merge (`h_miss` is false). The
peer now starts after the slots on its own card: 48 here (the log says `from expert 48`), 24 without the split,
and 24 for the first card's layers when the peer shares the split card. The run above exits 0 with its 51 peer
experts in [48, 99) and still answers **8261** on the reference prompt. The random 4-token prompt's first tokens
depend on which experts the peer holds (`25568 46 13724 16306 459 18347 305 2393` here;
`10 45 32 49 46 32 49 46` without the split).

A layer can find the second card with less room than the budget said (another program, the display): the
placement then keeps the experts that fit and sends the others to the CPU pool. For a layer where none fit, the
code used to `continue` past the rest of the layer's setup, which is the layer's own work, not the peer's: the
routing scores stayed an empty vector (the CPU pool's `H.score[x] += 1.0` then wrote past it), the slot table
stayed null (the grouping kernels read it), and the format table stayed zeroed (`native_fmt`, called just below,
builds it; the pool and the stream's offsets read it). The layer is then computed with tables that were never
made, and a run with one such layer dies in `Engine::layer` with a segmentation fault (measured: SIGSEGV, the
backtrace stops in `Engine::layer`; not a wrong answer, a crash).

Fixed: a layer without a peer slot keeps an all `-1` table, a null device table and its usual setup; the peer
half returns before touching the card, and the layer's experts go to the CPU pool. Measured on the same box
(Ryzen 9 5950X, RX 9070 + RX 6800, 73 of 256 experts a layer on the second card, `--pool-workers 15`,
`--low-ram`, a 4-token prompt): no fault, and a forced no-room layer (a temporary hook in the placement) ran to
the end with the same tokens as the run without the hook (measured before the pool fix below).

Fixed as well: the CPU pool no longer computes the entries the peer holds. `moe_hybrid_cpu` read
`peer_hy[li].slot_of` into `peer_slots` and never used it, so the peer computed its share and the pool computed
it again into the same host rows, with nothing ordering the two writers. The pool now skips an entry whose expert
is in the peer's table; the entry stays in the merge, which takes the row the peer wrote. Measured with fixed
slots and the same 4-token prompt: the pool's entries fall from 1781 to 1218 of 1984, the peer's 569, no fault.
The first decoded token changes (`44` -> `10` on that random prompt, since the peer's rows are now the ones
merged); the reference prompt gives 8261 before and after.

## Performance (measured on Ryzen 9 5950X, 64 GB DDR4, RX 9070 + RX 6800)

Decode, 32 tokens, one session, `--resident-budget-gib 0.25` (so the VRAM slots alone decide), the reference
prompt:

| Context | Mode | Decode | Experts in VRAM per layer (card 0 / card 1) |
| ---: | --- | ---: | ---: |
| 4096 | FP32 | 31.7 tok/s | 115 / 128 |
| 4096 | int8 | 32.1 tok/s | 123 / 133 |
| 65536 | f16 | 25.2 tok/s | 35 / 47 |
| 65536 | int8 | 28.5 tok/s | 72 / 85 |
| 131072 | int8 | 23.0 tok/s | 19 / 31 |

A 42-token prompt (past `kGemmT`, so the FP16 GEMM path runs as well as the cache's dequantisation) gave the
same 8 tokens in all three modes, with the prompt read at about 20 tok/s.

With `--prefill auto` (the service config: a chunk of 2048) and the budget of 30 GiB, the API read a 34-token
prompt at 18.1 then 26.9 tok/s and decoded two 64-token requests at 21.7 and 34.2 tok/s; the first request
after the start read 26 prompt tokens at 4.7 tok/s and decoded 45 at 5.1.

- The layer split runs on both cards; the residual and the normed hidden cross once per verify window
  (a few hundred KB through pinned RAM), not twice per layer.
- The layer split runs without the decode graph (from the engine code). One card's decode at the same
  context and mode is not measured yet.

## Notes

- The build's CMake cache may have a stale `CMAKE_PROJECT_INCLUDE` that must be removed before reconfiguring:
  `sed -i '/CMAKE_PROJECT_INCLUDE/d' build-hip/CMakeCache.txt`
- The machine crashed twice during early testing (VRAM overcommit + two concurrent engine instances). The test
  runner (`tools/run_m2_test.sh`) serializes runs with a timeout.
- M2.5 (same architecture, 62 layers, 256 experts, 8 active) reuses the same engine path; only weights and
  3 config entries change. Smallest GGUF quant is Q2_K (~78-83 GB).
