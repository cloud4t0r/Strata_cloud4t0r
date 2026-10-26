"""Tests for setup.py's MiniMax-M2 install (OWN_PATH_FAMILIES "minimax2"): the two pinned shards, the engine build
(strata-m2 compiled only when MiniMax-M2 is chosen), the low-RAM mode and the RAM budget (--m2-low-ram auto|on|off;
auto: a host copy capped at what is free now, --resident-budget-gib) and the start script run_minimax_m2.sh with its
M2_LOW_RAM switch.  Mocked engine, downloads
and processes - nothing is built, fetched or started; the generated script runs in its dry-run mode (M2_DRY_RUN=1).

    python tools/test_setup_minimax.py
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup  # noqa: E402

REPO = "bartowski/MiniMaxAI_MiniMax-M2-GGUF"
SHARDS = ["MiniMaxAI_MiniMax-M2-IQ2_XXS-00001-of-00002.gguf", "MiniMaxAI_MiniMax-M2-IQ2_XXS-00002-of-00002.gguf"]
# the user's two cards as amd_gpus() lists them (sysfs mem_info_vram_total)
R6800 = {"index": 0, "name": "AMD Radeon RX 6800", "vram_gb": 17163091968 / 2 ** 30, "arch": "gfx1030",
         "vendor": "amd"}
R9070 = {"index": 1, "name": "AMD Radeon RX 9070", "vram_gb": 17095983104 / 2 ** 30, "arch": "gfx1201",
         "vendor": "amd"}


def args(**kw):
    base = dict(family="minimax2", model=None, check=False, yes=True, vision=None, models_dir=None, gguf_dir=None,
                context=None, rope_scaling=None, rope_scale=None, kv=None, resident_budget_gib=None, low_ram="auto",
                vram_reserve_mib=None, vram_reserve_later_mib=None, host=None, api_key=None, browser=None,
                no_start=True, data_dir=None, gpu=None, build=False, prebuilt=None, m2_low_ram="auto", m2_kv="auto",
                layer_split=None)
    base.update(kw)
    return argparse.Namespace(**base)


def run(fn, *a, **kw):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        try:
            return fn(*a, **kw), out.getvalue()
        except SystemExit as e:
            return e.code, out.getvalue()


class Entries(unittest.TestCase):
    def test_both_shards_are_pinned(self):
        fam = setup.OWN_PATH_FAMILIES["minimax2"]
        self.assertEqual(setup.own_path_files(fam, "IQ2_XXS"), SHARDS)
        self.assertEqual(setup.own_path_bytes(fam, "IQ2_XXS"), 39922661440 + 14803701088)
        for name in SHARDS:
            self.assertIsNotNone(setup.file_pin(fam["hf"].format(q="IQ2_XXS") + name))
        self.assertIn(setup.HF_REVISIONS[REPO], fam["hf"])
        self.assertTrue(fam["hf"].format(q="IQ2_XXS").endswith("/MiniMaxAI_MiniMax-M2-IQ2_XXS/"))

    def test_qwen36_keeps_its_single_file(self):
        fam = setup.OWN_PATH_FAMILIES["qwen36"]
        self.assertEqual(setup.own_path_files(fam, "Q8_0"), ["Qwen3.6-35B-A3B-Q8_0.gguf"])

    def test_the_flag_exists(self):
        src = (ROOT / "setup.py").read_text(encoding="utf-8")
        self.assertIn('"--m2-low-ram", choices=["auto", "on", "off"], default="auto"', src)


class LowRamMode(unittest.TestCase):
    def test_auto_follows_the_free_ram(self):
        # auto: a host copy of the experts capped at what is free now less the margin; low-RAM when that is too little
        self.assertFalse(setup.m2_low_ram_mode("auto", 64.0, 48.0)[0])
        self.assertFalse(setup.m2_low_ram_mode("auto", 128.0, 80.0)[0])
        self.assertTrue(setup.m2_low_ram_mode("auto", 64.0, 10.0)[0])     # 2 GiB: below the floor
        self.assertTrue(setup.m2_low_ram_mode("auto", 64.0, 0.0)[0])      # unreadable: no budget, no guess
        self.assertIn("32 GiB", setup.m2_low_ram_mode("auto", 64.0, 48.0)[1])   # 48-8, below 64-32

    def test_the_budget_is_free_ram_less_the_margin(self):
        self.assertEqual(setup.M2_EXPERTS_GIB, 49.10)                    # MEASURED: the engine's "49.10 GiB"
        self.assertEqual(setup.M2_RESIDENT_MARGIN_GIB, 8)
        self.assertEqual(setup.M2_RESIDENT_MIN_GIB, 4)
        self.assertEqual(setup.M2_RESIDENT_OTHER_GIB, 32)
        self.assertEqual(setup.m2_resident_budget_gib(30.0, total=62.7), 22.0)    # free less 8
        self.assertEqual(setup.m2_resident_budget_gib(59.0, total=62.7), 30.5)    # capped at 62.7-32
        self.assertEqual(setup.m2_resident_budget_gib(59.0, total=96.0), 49.10)   # 96-32=64: all of them
        self.assertEqual(setup.m2_resident_budget_gib(57.0, total=128.0), 49.0)   # under all of them: as computed
        self.assertEqual(setup.m2_resident_budget_gib(80.0, total=128.0), 49.10)  # at most all of them
        self.assertEqual(setup.m2_resident_budget_gib(20.0, total=64.0), 12.0)    # quarter-GiB steps
        self.assertEqual(setup.m2_resident_budget_gib(11.9, total=64.0), 0.0)     # below the floor: no budget
        self.assertEqual(setup.m2_resident_budget_gib(12.0, total=64.0), 4.0)
        self.assertEqual(setup.m2_resident_budget_gib(59.0, total=24.0), 0.0)     # too little RAM for a copy
        self.assertEqual(setup.m2_resident_budget_gib(0.0, total=64.0), 0.0)      # unreadable

    def test_an_asked_budget_keeps_the_normal_mode(self):
        # --resident-budget-gib N: the normal mode, even where auto would pick the low-RAM one (--m2-low-ram on wins)
        on, why = setup.m2_low_ram_mode("auto", 64.0, 10.0, 20.0)
        self.assertFalse(on)
        self.assertIn("as you asked", why)
        self.assertTrue(setup.m2_low_ram_mode("on", 64.0, 10.0, 20.0)[0])

    def test_on_and_off_override_the_detection(self):
        self.assertTrue(setup.m2_low_ram_mode("on", 256.0, 200.0)[0])
        self.assertFalse(setup.m2_low_ram_mode("off", 32.0, 2.0)[0])

    def test_avail_ram_gb_reads_the_free_ram(self):
        if setup.WIN or sys.platform == "darwin":
            self.skipTest("Linux's /proc/meminfo")
        avail = setup.avail_ram_gb()
        self.assertGreater(avail, 0.0)
        self.assertLessEqual(avail, setup.ram_gb())

    def test_avail_ram_gb_unreadable_is_zero(self):
        with mock.patch.object(setup, "WIN", False), mock.patch.object(setup.sys, "platform", "linux"), \
                mock.patch("builtins.open", side_effect=OSError):
            self.assertEqual(setup.avail_ram_gb(), 0.0)

    def test_ram_need_follows_the_mode(self):
        fam = setup.OWN_PATH_FAMILIES["minimax2"]
        self.assertEqual(setup.own_path_ram_need(fam, "IQ2_XXS", True, 48.0, 64.0), 24)
        self.assertEqual(setup.own_path_ram_need(fam, "IQ2_XXS", False, 48.0, 64.0), 24)   # the copy is capped
        self.assertEqual(setup.own_path_ram_need(fam, "IQ2_XXS", False, 0.0, 64.0),
                         setup.M2_LOW_RAM_BELOW_GB)                                     # full copy
        # a budget asked for (--resident-budget-gib N): the low-RAM footprint plus the copy
        self.assertEqual(setup.own_path_ram_need(fam, "IQ2_XXS", False, 48.0, 64.0, 20.0), 44)
        self.assertEqual(setup.own_path_ram_need(fam, "IQ2_XXS", False, 48.0, 64.0, 49.1), setup.M2_LOW_RAM_BELOW_GB)
        self.assertEqual(setup.own_path_ram_need(fam, "IQ2_XXS", True, 48.0, 64.0, 20.0), 24)  # low-RAM wins
        q = setup.OWN_PATH_FAMILIES["qwen36"]                   # no low-RAM choice: its own need either way
        self.assertEqual(setup.own_path_ram_need(q, "Q8_0", False), q["models"]["Q8_0"]["ram_gb"])


class Context(unittest.TestCase):
    """The recommended context follows MiniMax-M2's K / V cache (int8 by default, --m2-kv) on the one card the engine
    runs on; Qwen3.6's rule is unchanged."""
    M2 = setup.OWN_PATH_FAMILIES["minimax2"]
    Q36 = setup.OWN_PATH_FAMILIES["qwen36"]

    def test_the_kv_cache_matches_the_gguf(self):
        # 62 layers, attention.head_count_kv = 8, key/value_length 128, K and V: fp32 15.5 GiB per 32K tokens, f16
        # half of it, int8 a quarter plus one fp16 scale per cell, per KV head and per block of 32 dims (K and V)
        self.assertEqual(self.M2["kv_modes"]["f32"], 62 * 8 * 128 * 2 * 4)
        self.assertEqual(self.M2["kv_modes"]["f16"], 62 * 8 * 128 * 2 * 2)
        self.assertEqual(self.M2["kv_modes"]["int8"], 62 * 8 * 128 * 2 * 1 + 62 * 8 * (128 // 32) * 2 * 2)
        self.assertAlmostEqual(self.M2["kv_modes"]["f32"] * 32768 / 2 ** 30, 15.5)
        self.assertAlmostEqual(self.M2["kv_modes"]["f16"] * 32768 / 2 ** 30, 7.75)
        self.assertAlmostEqual(self.M2["kv_modes"]["int8"] * 32768 / 2 ** 30, 4.1, places=1)
        # int8 is setup's choice: a quarter of fp32's VRAM, and as close to its logits as f16 is (docs/MINIMAX_M2.md)
        self.assertEqual(self.M2["kv_default"], "int8")
        self.assertEqual(self.M2["kv_bytes_per_token"], self.M2["kv_modes"]["int8"])
        for mode in ("f32", "f16", "int8"):
            self.assertEqual(setup.own_path_kv_family(self.M2, mode)["kv_bytes_per_token"],
                             self.M2["kv_modes"][mode])
            self.assertEqual(setup.own_path_kv_family(self.M2, mode)["kv_used"], mode)
        self.assertEqual(setup.own_path_kv_family(self.M2)["kv_used"], "int8")     # auto / unknown: the default
        self.assertEqual(setup.own_path_kv_family(self.M2, "q4_0")["kv_used"], "int8")
        self.assertEqual(setup.own_path_kv_modes(self.Q36), [])                    # unchanged: no kv_modes

    def test_a_16_gb_card_gets_a_context_that_fits(self):
        for vram in (16.0, 17095983104 / 2 ** 30):            # 16 GB, and the RX 6800's sysfs figure (15.92 GiB)
            rec = setup.own_path_context(self.M2, vram)
            self.assertNotEqual(rec, 131072)
            self.assertEqual(rec, 65536, vram)                   # int8: 64K fits a 16 GB card (32K with f16)
            b = self.M2["vram_budget"]
            self.assertLessEqual(rec * self.M2["kv_bytes_per_token"] / 2 ** 30 + b["dense_gib"] + b["fixed_gib"], vram)
            self.assertLessEqual(setup.own_path_ctx_need_gib(self.M2, rec) + b["experts_gib"], vram)
            self.assertGreater(setup.own_path_ctx_need_gib(self.M2, 131072), vram)  # 128K cannot start there

    def test_a_bigger_card_gets_a_longer_context(self):
        self.assertEqual(setup.own_path_context(self.M2, 24.0), 65536)
        self.assertEqual(setup.own_path_context(self.M2, 48.0), 262144)
        self.assertGreater(setup.own_path_context(self.M2, 48.0), setup.own_path_context(self.M2, 16.0))
        self.assertEqual(setup.own_path_context(self.M2, 8.0), 8192)    # nothing fits: the smallest
        # the modes move those contexts: f16 doubles the cache, so a 16 GB card stops at 32K with it
        self.assertEqual(setup.own_path_context(setup.own_path_kv_family(self.M2, "f32"), 16.0), 16384)
        self.assertEqual(setup.own_path_context(setup.own_path_kv_family(self.M2, "f16"), 16.0), 32768)
        self.assertEqual(setup.own_path_context(setup.own_path_kv_family(self.M2, "int8"), 16.0), 65536)

    def test_auto_picks_the_mode_that_reaches_a_usable_context(self):
        # auto: int8 (the default, the smallest cache), on any card that can start M2
        self.assertEqual(setup.own_path_kv_auto(self.M2, lambda f: setup.own_path_context(f, 15.92)), "int8")
        self.assertEqual(setup.own_path_kv_auto(self.M2, lambda f: setup.own_path_context(f, 8.0)), "int8")
        self.assertEqual(setup.own_path_kv_auto(self.Q36, lambda f: 32768), "")

    def test_qwen36_is_unchanged(self):
        self.assertNotIn("vram_budget", self.Q36)
        for vram, ctx in ((12.0, 32768), (16.0, 32768), (24.0, 65536), (32.0, 131072)):
            self.assertEqual(setup.own_path_context(self.Q36, vram), ctx, vram)
        self.assertEqual(setup.own_path_contexts(self.Q36), [c for c in setup.CONTEXTS if c <= 262144])
        self.assertIn(16384, setup.own_path_contexts(self.M2))


class SplitContext(unittest.TestCase):
    """The two-card split's sizing (DERIVED from the vram_budget constants): each card on its own VRAM, the tighter
    card decides."""
    M2 = setup.OWN_PATH_FAMILIES["minimax2"]

    def test_the_family_keys(self):
        self.assertEqual(self.M2["layer_split"], 31)
        self.assertEqual(self.M2["n_layers"], 62)
        self.assertNotIn("layer_split", setup.OWN_PATH_FAMILIES["qwen36"])
        self.assertEqual(setup.own_path_split_layers(self.M2, 31), [31, 31])
        self.assertEqual(setup.own_path_split_layers(self.M2, 29), [29, 33])

    def test_the_need_per_card(self):
        # int8 (setup's default): 32K -> 7.21 GiB on the first card (all the dense weights) and 6.28 on the second
        self.assertAlmostEqual(setup.own_path_split_need_gib(self.M2, 32768, 31, True), 7.212, places=2)
        self.assertAlmostEqual(setup.own_path_split_need_gib(self.M2, 32768, 31, False), 6.282, places=2)
        self.assertAlmostEqual(setup.own_path_split_need_gib(self.M2, 65536, 31, True), 9.776, places=2)
        self.assertAlmostEqual(setup.own_path_split_room_gib(self.M2, 31), 1.0)
        # all 62 layers on one card: the one-card rule
        self.assertAlmostEqual(setup.own_path_split_need_gib(self.M2, 32768, 62, True),
                               setup.own_path_ctx_need_gib(self.M2, 32768))

    def test_the_tighter_card_decides(self):
        ctx = lambda avail: setup.own_path_split_context(self.M2, avail, 31)
        self.assertEqual(ctx([15.922, 15.984]), 131072)
        self.assertEqual(ctx([15.922 - 1.155, 15.968]), 65536)    # the RX 9070 with the desktop's share in use
        self.assertEqual(ctx([16.0, 16.0]), 131072)
        self.assertEqual(ctx([24.0, 24.0]), 204800)
        self.assertEqual(ctx([12.0, 12.0]), 65536)
        self.assertEqual(ctx([16.0, 10.0]), 65536)                # the 10 GB card decides
        self.assertEqual(ctx([10.0, 16.0]), 32768)                # first: it also holds all the dense weights
        self.assertEqual(ctx([7.0, 8.0]), 16384)                  # the first card cannot hold 32K: 16K
        self.assertEqual(ctx([6.0, 6.0]), 8192)                   # nothing fits: the smallest
        self.assertEqual(ctx([0.0, 16.0]), 8192)                  # an unknown card is not counted as roomy
        # f16: twice int8's cache, so the same cards reach half the context
        f16 = setup.own_path_kv_family(self.M2, "f16")
        self.assertEqual(setup.own_path_split_context(f16, [15.922, 15.984], 31), 65536)
        self.assertEqual(setup.own_path_split_context(f16, [12.0, 12.0], 31), 32768)

    def test_misfit(self):
        self.assertEqual(setup.own_path_split_misfit(self.M2, 262144, [15.9, 15.9], 31), 0)
        self.assertIsNone(setup.own_path_split_misfit(self.M2, 131072, [15.9, 15.9], 31))
        self.assertEqual(setup.own_path_split_misfit(self.M2, 131072, [15.9, 12.0], 31), 1)
        self.assertIsNone(setup.own_path_split_misfit(self.M2, 131072, [0.0, 0.0], 31))   # unknown: not judged

    def test_one_card_sizing_never_sums_the_cards(self):
        # the one-card rule never sums the cards: 16 GB -> 64K (int8), however many cards the PC has
        self.assertEqual(setup.own_path_context(self.M2, R6800["vram_gb"]), 65536)
        self.assertEqual(setup.own_path_context(self.M2, R9070["vram_gb"]), 65536)
        self.assertEqual(setup.own_path_context(self.M2, R6800["vram_gb"] + R9070["vram_gb"]), 131072)   # a sum would


class BuildDir(unittest.TestCase):
    def test_where_strata_m2_is_built(self):
        with mock.patch.object(setup, "WIN", False), mock.patch.object(setup, "CONTROL", None):
            self.assertEqual(setup.m2_build_dir(False, {"source": "local"}), setup.ROOT / "build")
            self.assertEqual(setup.m2_build_dir(False, {"source": "local", "toolkit": 12}), setup.ROOT / "build-cuda12")
            self.assertEqual(setup.m2_build_dir(True, {"source": "local-hip", "backend": "hip"}), setup.ROOT / "build-hip")
            self.assertIsNone(setup.m2_build_dir(False, {"source": "github"}))        # a ready-made engine
            self.assertIsNone(setup.m2_build_dir(True, {"source": "local"}))          # the Intel Arc (SYCL) stub
        with mock.patch.object(setup, "WIN", True):
            self.assertIsNone(setup.m2_build_dir(False, {"source": "local"}))


class Install(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.calls = []
        self.families = {"qwen4exp", "qwen35moe", "minimax-m2"}
        self.build_ok = True
        self.eng = self.root / "engine"
        self.bdir = self.root / "build-hip"
        self.cfg = {}

        def engine_step(a, gpu, hip, amd, cuda_tk, vision):
            self.calls.append(("engine", vision, bool(getattr(a, "build", False))))
            self.eng.mkdir(parents=True, exist_ok=True)
            self.bdir.mkdir(parents=True, exist_ok=True)
            (self.bdir / "CMakeCache.txt").write_text("x")
            return self.root / "llama", self.eng, {"version": "0.1.47", "source": "local-hip", "backend": "hip"}, \
                ["/libs"], gpu, vision

        def fake_run(cmd, **k):
            self.calls.append(("run", [str(x) for x in cmd]))
            if "--target" in cmd and self.build_ok:
                (self.bdir / setup.M2_EXE).write_text("m2")
            return mock.Mock(returncode=0 if self.build_ok else 1)

        def write_cfg(path, cfg, source=None, explicit=None):
            self.cfg = cfg
            path.write_text(json.dumps(cfg), encoding="utf-8")
            self.calls.append(("config", path.name))

        patches = [mock.patch.object(setup, "download", lambda url, dst, what=None: self.calls.append(("get", url, dst))),
                   mock.patch.object(setup, "verify_sha256", lambda s, size, sha, own=False: self.calls.append(("sha", s.name))),
                   mock.patch.object(setup, "free_gb", lambda p: 500.0),
                   mock.patch.object(setup, "pip_install", lambda *a, **k: None),
                   mock.patch.object(setup, "engine_step", engine_step),
                   mock.patch.object(setup, "engine_families", lambda e: self.families),
                   mock.patch.object(setup, "ensure_build_tools", lambda: ("cmake", "ninja")),
                   mock.patch.object(setup, "run", fake_run),
                   mock.patch.object(setup, "data_folder", lambda d, *a, **k: (self.root / "data", [])),
                   mock.patch.object(setup, "write_setup_config", write_cfg),
                   mock.patch.object(setup, "write_run_script", lambda tag, p, port, o=True: self.root / f"run-{tag}.sh"),
                   mock.patch.object(setup, "recommend_pool_workers", lambda a: a),
                   mock.patch.object(setup, "WIN", False),
                   mock.patch.object(setup, "CONTROL", None),
                   mock.patch.object(setup, "ROOT", self.root)]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def install(self, family="minimax2", ram=64.0, gpu=None, multi=None, hip=True, amd=None, used=None, avail=48.0,
                **kw):
        gpu = gpu or {"index": 0, "count": 1, "vram_gb": 16.0, "name": "Test GPU"}
        with mock.patch.object(setup, "avail_ram_gb", lambda: avail), \
                mock.patch.object(setup, "amd_vram_used_gib", lambda *a, **k: dict(used or {})):
            return run(setup.install_own_path, args(models_dir=str(self.root / "models"), family=family, **kw), family,
                       gpu, hip, amd or [], None, multi or [], 8080, ram, None)

    def ctx(self):
        a = self.cfg["args"]
        return a[a.index("--max-context") + 1]

    def test_a_16_gb_card_writes_a_context_that_fits(self):
        rc, out = self.install()
        self.assertEqual(rc, 0, out)
        self.assertEqual(self.ctx(), "65536")                     # int8: 64K fits a 16 GB card (32K with f16)
        self.assertNotIn("context does not fit", out)             # no warning for the recommended context
        self.assertIn("K / V cache: int8", out)
        self.assertEqual(self.cfg["args"][self.cfg["args"].index("--kv") + 1], "int8")

    def test_two_16_gb_cards_get_the_one_cards_context(self):
        # two AMD cards present, no --gpus: one card, sized on ITS OWN VRAM (65536 on 16 GB), never on the sum
        gpu = {**R6800, "count": 2, "archs": ["gfx1030", "gfx1201"]}
        rc, out = self.install(gpu=gpu, amd=[R6800, R9070], used={0: 0.016, 1: 0.767})
        self.assertEqual(rc, 0, out)
        self.assertNotIn("runs on one GPU so far", out)
        self.assertEqual(self.ctx(), "65536")
        self.assertEqual(self.cfg["gpu"], 0)
        self.assertNotIn("layer_split", self.cfg)
        self.assertNotIn("layer split", out)

    def test_m2_kv_choices(self):
        # --m2-kv: an explicit mode is kept and written, and the context follows it
        rc, out = self.install(m2_kv="f32")
        self.assertEqual(rc, 0, out)
        self.assertIn("K / V cache: f32", out)
        self.assertEqual(self.cfg["args"][self.cfg["args"].index("--kv") + 1], "f32")
        self.assertEqual(self.ctx(), "16384")                     # the engine's exact path: 16K on a 16 GB card
        gpu = {"index": 0, "count": 1, "vram_gb": 12.0, "name": "Test GPU"}
        rc, out = self.install(gpu=gpu, m2_kv="auto")             # auto: int8, the default
        self.assertEqual(rc, 0, out)
        self.assertIn("K / V cache: int8", out)
        self.assertEqual(self.cfg["args"][self.cfg["args"].index("--kv") + 1], "int8")
        self.assertEqual(self.ctx(), "32768")
        rc, out = self.install(m2_kv="f16", gpu=gpu)              # asked, kept even below 32K
        self.assertEqual(rc, 0, out)
        self.assertEqual(self.cfg["args"][self.cfg["args"].index("--kv") + 1], "f16")
        rc, out = self.install(family="qwen36", vision="no", m2_kv="int8")
        self.assertEqual(rc, 0, out)
        self.assertIn("--m2-kv is MiniMax-M2's setting", out)     # said, not applied
        self.assertNotIn("--kv", self.cfg["args"])

    def test_the_kv_mode_is_a_flag(self):
        src = (ROOT / "setup.py").read_text(encoding="utf-8")
        self.assertIn('"--m2-kv", choices=["auto", "f32", "f16", "int8"], default="auto"', src)

    # ---- the two-card layer split (--gpus with two AMD cards, opt-in)
    def split(self, main=None, multi=(1, 0), used=None, **kw):
        main = main or R9070
        amd = kw.pop("amd", [R6800, R9070])
        gpu = {**main, "count": len(amd), "archs": ["gfx1030", "gfx1201"]}
        return self.install(gpu=gpu, amd=amd, multi=list(multi),
                            used={0: 0.016, 1: 0.767} if used is None else used, **kw)

    def test_gpus_1_0_writes_a_layer_split(self):
        rc, out = self.split()
        self.assertEqual(rc, 0, out)
        self.assertEqual(self.cfg["gpu"], [1, 0])
        self.assertEqual(self.cfg["layer_split"], "31")
        self.assertIs(self.cfg["gpus_asked"], True)
        self.assertEqual(self.ctx(), "65536")                     # int8: the tighter card (the RX 9070, 0.77 GB used)
        self.assertNotIn("--layer-split", self.cfg["args"])       # the server adds it from "layer_split"
        self.assertNotIn("--peer-device", self.cfg["args"])
        self.assertNotIn("--low-ram", self.cfg["args"])
        self.assertEqual(self.cfg["args"][self.cfg["args"].index("--resident-budget-gib") + 1], "32")
        self.assertNotIn("runs on one GPU so far", out)
        self.assertIn("layers 0-30 on GPU 1, 31-61 on GPU 0", out)
        self.assertIn("GPU 1 (AMD Radeon RX 9070): layers 0-30, the embedding", out)
        self.assertIn("GPU 0 (AMD Radeon RX 6800): layers 31-61", out)

    def test_gpus_0_1_keeps_the_order(self):
        rc, out = self.split(main=R6800, multi=(0, 1))
        self.assertEqual(rc, 0, out)
        self.assertEqual(self.cfg["gpu"], [0, 1])
        self.assertEqual(self.cfg["layer_split"], "31")
        self.assertEqual(self.ctx(), "131072")
        self.assertIn("layers 0-30 on GPU 0, 31-61 on GPU 1", out)

    def test_layer_split_flag(self):
        rc, out = self.split(layer_split="29")
        self.assertEqual(rc, 0, out)
        self.assertEqual(self.cfg["layer_split"], "29")
        rc, out = self.split(layer_split="auto")
        self.assertEqual(rc, 0, out)
        self.assertEqual(self.cfg["layer_split"], "31")
        for bad in ("16,32", "0", "1", "49", "62", "x"):
            self.cfg = {}
            rc, out = self.split(layer_split=bad)
            self.assertNotEqual(rc, 0, (bad, out))
            self.assertIn("one layer", out, bad)
            self.assertEqual(self.cfg, {}, bad)

    def test_split_context_that_does_not_fit_is_said_and_kept(self):
        rc, out = self.split(context=262144)
        self.assertEqual(rc, 0, out)
        self.assertIn("a 256K context does not fit MiniMax-M2 on GPU 1", out)   # the first card: all the dense weights
        self.assertIn("its share of the KV cache (31 of 62 layers) is 16.5 GB (int8)", out)
        self.assertIn("kept as you chose", out)
        self.assertEqual(self.ctx(), "262144")

    def test_split_yes_alone_never_writes_a_context_that_cannot_start(self):
        small = [{**R6800, "vram_gb": 4.0}, {**R9070, "vram_gb": 4.0}]
        rc, out = self.split(main=small[1], amd=small)
        self.assertNotEqual(rc, 0, out)
        self.assertIn("does not fit GPU 1", out)
        self.assertEqual(self.cfg, {})

    def test_unreadable_used_vram_is_said(self):
        rc, out = self.split(used={})
        self.assertEqual(rc, 0, out)
        self.assertIn("could not be read", out)
        self.assertEqual(self.ctx(), "131072")

    def test_three_cards_use_the_first_two(self):
        third = {"index": 2, "name": "AMD Radeon RX 9070", "vram_gb": 16.0, "arch": "gfx1201", "vendor": "amd"}
        rc, out = self.split(multi=(1, 0, 2), amd=[R6800, R9070, third])
        self.assertEqual(rc, 0, out)
        self.assertEqual(self.cfg["gpu"], [1, 0])
        self.assertIn("two cards", out)

    def test_split_context_menu(self):
        with mock.patch.object(setup, "ask", lambda q, choices, default, yes=False: default):
            rc, out = self.split(yes=False)
        self.assertEqual(rc, 0, out)
        self.assertIn("4) 64K tokens   (KV cache 8.2 GB, about half on each card)   (recommended for your cards)", out)
        self.assertIn("5) 128K tokens   (KV cache 16.5 GB, about half on each card)", out)
        self.assertIn("6) 200K tokens   (KV cache 25.7 GB, about half on each card: does not fit your cards)", out)
        self.assertEqual(self.ctx(), "65536")

    def test_qwen36_with_gpus_stays_on_one_card(self):
        rc, out = self.split(family="qwen36", vision="no")
        self.assertEqual(rc, 0, out)
        self.assertIn("runs on one GPU so far: using GPU 1", out)
        self.assertEqual(self.cfg["gpu"], 1)
        self.assertNotIn("layer_split", self.cfg)

    def test_nvidia_minimax_with_gpus_stays_on_one_card(self):
        rc, out = self.install(gpu={"index": 0, "count": 2, "vram_gb": 16.0, "name": "Test GPU"}, multi=[0, 1],
                               hip=False)
        self.assertEqual(rc, 0, out)
        self.assertIn("runs on one GPU so far", out)
        self.assertEqual(self.cfg["gpu"], 0)
        self.assertNotIn("layer_split", self.cfg)
        self.assertEqual(self.ctx(), "65536")

    def test_vram_reserves_go_through(self):
        # --vram-reserve-mib is the engine's own per-card option (each card's slots less that reserve), not the derived
        # --expert-vram setup wrote before: that one was the experts' total in GiB, from the first card's VRAM only
        rc, out = self.install(vram_reserve_mib=1024)
        self.assertEqual(rc, 0, out)
        a = self.cfg["args"]
        self.assertEqual(a[a.index("--vram-reserve-mib") + 1], "1024")
        self.assertNotIn("--expert-vram", a)
        self.assertNotIn("--vram-reserve-later-mib", a)
        self.assertIn("VRAM kept free for other programs: 1024 MiB", out)

    def test_the_split_card_gets_its_own_reserve(self):
        rc, out = self.split(vram_reserve_mib=600, vram_reserve_later_mib=2048)
        self.assertEqual(rc, 0, out)
        a = self.cfg["args"]
        self.assertEqual(a[a.index("--vram-reserve-mib") + 1], "600")
        self.assertEqual(a[a.index("--vram-reserve-later-mib") + 1], "2048")
        self.assertIn("VRAM kept free on the second card: 2048 MiB", out)

    def test_the_later_reserve_is_written_as_given(self):
        # one card: the engine parses it and uses it only on a split's card or the peer's (harmless here), and the
        # config keeps what was asked, as for --vram-reserve-mib
        rc, out = self.install(vram_reserve_later_mib=2048)
        self.assertEqual(rc, 0, out)
        a = self.cfg["args"]
        self.assertEqual(a[a.index("--vram-reserve-later-mib") + 1], "2048")
        self.assertIn("VRAM kept free on the second card: 2048 MiB", out)

    def test_one_card_config_is_unchanged(self):
        folder = self.root / "models" / "minimax2-IQ2_XXS"
        want = ["--native", str(folder / SHARDS[0]), "--max-context", "65536", "--prefill", "auto", "--mtp", "0",
                "--kv", "int8", "--resident-budget-gib", "32"]
        keys = {"exe", "args", "cwd", "tokenizer", "model_name", "arch", "log", "lib_dirs", "port", "backend"}
        rc, out = self.install()
        self.assertEqual(rc, 0, out)
        self.assertEqual(self.cfg["args"], want)
        self.assertEqual(set(self.cfg), keys)
        rc, out = self.install(gpu={**R6800, "count": 2, "archs": ["gfx1030", "gfx1201"]}, amd=[R6800, R9070])
        self.assertEqual(rc, 0, out)
        self.assertEqual(self.cfg["args"], want)
        self.assertEqual(set(self.cfg), keys | {"gpu", "gpus_asked"})
        self.assertEqual(self.cfg["gpu"], 0)

    def test_choices_keep_the_split(self):
        p = self.root / "strata-minimax2-iq2_xxs.json"
        p.write_text(json.dumps({"args": ["--native", "x.gguf", "--max-context", "32768"], "gpu": [1, 0],
                                 "layer_split": "31", "arch": "minimax-m2"}), encoding="utf-8")
        ch = setup.choices_from_config(p)
        self.assertEqual((ch["family"], ch["gpu"], ch["layer_split"], ch["context"]), ("minimax2", [1, 0], "31", 32768))
        q = self.root / "strata-qwen36-ud-q4_k_xl.json"
        q.write_text(json.dumps({"args": ["--native", "x.gguf", "--max-context", "32768"], "gpu": 1,
                                 "arch": "qwen35moe"}), encoding="utf-8")
        self.assertIsNone(setup.choices_from_config(q)["layer_split"])

    def test_a_48_gb_card_writes_a_longer_context(self):
        rc, out = self.install(gpu={"index": 0, "count": 1, "vram_gb": 48.0, "name": "Test GPU"})
        self.assertEqual(rc, 0, out)
        self.assertEqual(self.ctx(), "262144")

    def test_an_explicit_context_that_does_not_fit_is_said_and_kept(self):
        rc, out = self.install(context=131072)                    # --context 131072 --yes: the user's choice
        self.assertEqual(rc, 0, out)
        self.assertIn("a 128K context does not fit MiniMax-M2", out)
        self.assertIn("its KV cache alone is 16.5 GB (int8)", out)
        self.assertIn("kept as you chose", out)
        self.assertEqual(self.ctx(), "131072")

    def test_yes_alone_never_writes_a_context_that_cannot_start(self):
        rc, out = self.install(gpu={"index": 0, "count": 1, "vram_gb": 4.0, "name": "Test GPU"})
        self.assertNotEqual(rc, 0, out)
        self.assertIn("does not fit the 4 GB card", out)
        self.assertEqual(self.cfg, {})                            # no config written

    def test_the_context_menu_shows_the_kv_cache(self):
        with mock.patch.object(setup, "ask", lambda q, choices, default, yes=False: default):
            rc, out = self.install(yes=False)
        self.assertEqual(rc, 0, out)
        self.assertIn("4) 64K tokens   (KV cache 8.2 GB)   (recommended for your GPU)", out)
        self.assertIn("5) 128K tokens   (KV cache 16.5 GB: does not fit your 16 GB card)", out)
        self.assertEqual(self.ctx(), "65536")

    def builds(self):
        return [c[1] for c in self.calls if c[0] == "run" and "--target" in c[1]]

    def script(self):
        return self.root / setup.M2_RUN_SCRIPT

    def dry(self, *argv, **env):
        e = {k: v for k, v in os.environ.items() if k not in ("M2_LOW_RAM",)}
        e.update({"M2_DRY_RUN": "1", **env})
        return subprocess.run(["sh", str(self.script()), *argv], capture_output=True, text=True, env=e, timeout=60,
                              stdin=subprocess.DEVNULL)

    def test_minimax2_compiles_strata_m2_and_downloads_both_shards(self):
        rc, out = self.install()
        self.assertEqual(rc, 0, out)
        self.assertIn(("engine", "none", True), self.calls)           # compiled here, no images
        b = self.builds()
        self.assertEqual(len(b), 1, out)
        self.assertEqual(b[0][b[0].index("--target") + 1], "strata-m2")
        self.assertEqual(b[0][b[0].index("--build") + 1], str(self.bdir))
        self.assertTrue((self.eng / setup.M2_EXE).is_file())
        folder = self.root / "models" / "minimax2-IQ2_XXS"
        get = [c for c in self.calls if c[0] == "get"]
        self.assertEqual([g[2] for g in get], [folder / s for s in SHARDS])
        self.assertEqual(get[0][1], setup.hf(REPO) + "MiniMaxAI_MiniMax-M2-IQ2_XXS/" + SHARDS[0])
        a = self.cfg["args"]
        self.assertEqual(a[a.index("--native") + 1], str(folder / SHARDS[0]))   # the first shard
        self.assertEqual(self.cfg["arch"], "minimax-m2")
        self.assertNotIn("vision", self.cfg)

    def test_qwen36_builds_no_strata_m2(self):
        rc, out = self.install(family="qwen36", vision="no", m2_low_ram="auto")
        self.assertEqual(rc, 0, out)
        self.assertEqual(self.builds(), [])
        self.assertIn(("engine", "none", False), self.calls)          # the default path: no forced build
        self.assertFalse(self.script().exists())
        self.assertNotIn("--low-ram", self.cfg["args"])

    def test_a_failed_strata_m2_build_does_not_stop_setup(self):
        self.build_ok = False
        rc, out = self.install()
        self.assertEqual(rc, 0, out)
        self.assertIn("strata-m2 build failed", out)
        self.assertTrue(self.script().exists())

    def test_auto_writes_the_ram_budget(self):
        rc, out = self.install(ram=64.0, avail=48.0)             # 48 GB free now: 32 GiB of experts in RAM (64-32)
        self.assertEqual(rc, 0, out)
        self.assertIn("low-RAM mode: off (normal mode)", out)
        self.assertIn("RAM budget: 32 GiB of the 49.1 GiB of experts", out)   # 48-8, capped at 64-32
        self.assertNotIn("--low-ram", self.cfg["args"])
        self.assertEqual(self.cfg["args"][self.cfg["args"].index("--resident-budget-gib") + 1], "32")
        text = self.script().read_text(encoding="utf-8")
        self.assertIn("DEFAULT=0", text)
        self.assertTrue(os.access(self.script(), os.X_OK))
        r = self.dry()
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("normal mode", r.stdout)
        self.assertIn(str(self.root / "strata-minimax2-iq2_xxs.json"), r.stdout)   # the config as setup wrote it

    def test_the_budget_never_goes_past_all_the_experts(self):
        rc, out = self.install(ram=128.0, avail=80.0)             # 72 GiB under the margin: all of them fit
        self.assertEqual(rc, 0, out)
        self.assertEqual(self.cfg["args"][self.cfg["args"].index("--resident-budget-gib") + 1], "49.1")
        self.assertIn("RAM budget: 49.1 GiB of the 49.1 GiB of experts", out)

    def test_the_budget_can_be_asked_for(self):
        rc, out = self.install(ram=64.0, avail=48.0, resident_budget_gib=20.0)
        self.assertEqual(rc, 0, out)
        self.assertIn("low-RAM mode: off (normal mode) - 20 GiB of the 49.1 GiB of experts in RAM "
                      "(--resident-budget-gib, as you asked)", out)
        self.assertNotIn("--low-ram", self.cfg["args"])
        self.assertEqual(self.cfg["args"][self.cfg["args"].index("--resident-budget-gib") + 1], "20")
        self.assertIn("RAM budget: 20 GiB of the 49.1 GiB of experts in RAM (--resident-budget-gib, as you asked)", out)

    def test_an_asked_budget_wins_over_a_small_free_ram(self):
        rc, out = self.install(ram=64.0, avail=10.0, resident_budget_gib=20.0)   # auto alone would pick low-RAM
        self.assertEqual(rc, 0, out)
        self.assertNotIn("--low-ram", self.cfg["args"])
        self.assertEqual(self.cfg["args"][self.cfg["args"].index("--resident-budget-gib") + 1], "20")

    def test_an_asked_budget_above_all_the_experts_is_capped(self):
        rc, out = self.install(ram=128.0, avail=80.0, resident_budget_gib=80.0)
        self.assertEqual(rc, 0, out)
        self.assertIn("more than the 49.1 GiB of experts", out)
        self.assertEqual(self.cfg["args"][self.cfg["args"].index("--resident-budget-gib") + 1], "49.1")

    def test_an_asked_budget_below_the_engine_minimum_stops(self):
        rc, out = self.install(resident_budget_gib=0.1)
        self.assertEqual(rc, 1, out)
        self.assertIn("below what the MiniMax-M2 engine takes", out)

    def test_low_ram_on_ignores_the_asked_budget(self):
        rc, out = self.install(resident_budget_gib=20.0, m2_low_ram="on")
        self.assertEqual(rc, 0, out)
        self.assertIn("--resident-budget-gib 20 is ignored", out)
        self.assertIn("--low-ram", self.cfg["args"])
        self.assertNotIn("--resident-budget-gib", self.cfg["args"])

    def test_auto_falls_back_to_low_ram_when_the_free_ram_is_short(self):
        rc, out = self.install(ram=64.0, avail=10.0)             # 2 GiB under the margin: no budget worth writing
        self.assertEqual(rc, 0, out)
        self.assertIn("low-RAM mode: on (--low-ram)", out)
        self.assertIn("--low-ram", self.cfg["args"])
        self.assertNotIn("--resident-budget-gib", self.cfg["args"])
        self.assertIn("DEFAULT=1", self.script().read_text(encoding="utf-8"))

    def test_128_gb_auto_writes_the_normal_mode(self):
        rc, out = self.install(ram=128.0)
        self.assertEqual(rc, 0, out)
        self.assertIn("low-RAM mode: off (normal mode)", out)
        self.assertNotIn("--low-ram", self.cfg["args"])
        self.assertIn("--resident-budget-gib", self.cfg["args"])
        self.assertIn("DEFAULT=0", self.script().read_text(encoding="utf-8"))
        self.assertIn("normal mode", self.dry().stdout)

    def test_forced_modes_override_the_detection(self):
        rc, out = self.install(ram=128.0, m2_low_ram="on")
        self.assertEqual(rc, 0, out)
        self.assertIn("--low-ram", self.cfg["args"])
        self.assertNotIn("--resident-budget-gib", self.cfg["args"])
        self.assertIn("DEFAULT=1", self.script().read_text(encoding="utf-8"))
        rc, out = self.install(ram=64.0, m2_low_ram="off")       # explicit: the normal mode, with the budget
        self.assertEqual(rc, 0, out)
        self.assertNotIn("--low-ram", self.cfg["args"])
        self.assertIn("--resident-budget-gib", self.cfg["args"])
        self.assertIn("DEFAULT=0", self.script().read_text(encoding="utf-8"))

    def test_off_without_a_readable_free_ram_says_the_full_copy(self):
        rc, out = self.install(ram=64.0, m2_low_ram="off", avail=0.0)
        self.assertEqual(rc, 0, out)
        self.assertIn("needs about 96 GB of RAM in the normal mode", out)
        self.assertIn("free RAM could not be read", out)
        self.assertNotIn("--low-ram", self.cfg["args"])
        self.assertNotIn("--resident-budget-gib", self.cfg["args"])

    def test_m2_low_ram_switches_the_mode_for_one_run(self):
        rc, out = self.install(ram=64.0, avail=10.0)             # free RAM too low: default on
        self.assertEqual(rc, 0, out)
        text = self.script().read_text(encoding="utf-8")
        self.assertIn("unset STRATA_M2_LOW_RAM", text)
        override = self.root / ".run_minimax_m2.override.json"
        for how in ({"env": {"M2_LOW_RAM": "0"}, "argv": ()}, {"env": {}, "argv": ("--no-low-ram",)}):
            if override.exists():
                override.unlink()
            r = self.dry(*how["argv"], **how["env"])
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("normal mode", r.stdout)
            self.assertIn(str(override), r.stdout)
            self.assertNotIn("--low-ram", json.loads(override.read_text(encoding="utf-8"))["args"])
        rc, out = self.install(ram=128.0)                        # default off
        override.unlink()
        r = self.dry(M2_LOW_RAM="1")
        self.assertIn("low-RAM mode on", r.stdout)
        self.assertEqual(json.loads(override.read_text(encoding="utf-8"))["args"].count("--low-ram"), 1)
        self.assertNotEqual(self.dry(M2_LOW_RAM="maybe").returncode, 0)
        self.assertNotEqual(self.dry("--bogus").returncode, 0)

    def test_strata_m2_low_ram_is_unset_before_the_start(self):
        rc, out = self.install(ram=128.0)                        # normal mode
        self.assertEqual(rc, 0, out)
        text = self.script().read_text(encoding="utf-8")
        # the dry run's command, with STRATA_M2_LOW_RAM=0 inherited: the script must drop it (the engine reads its
        # mere presence as "on"); checked by printing the environment where the server would start
        probe = text.replace('if [ -n "$M2_DRY_RUN" ]; then\n',
                             'if [ -n "$M2_DRY_RUN" ]; then\n  env | grep -c "^STRATA_M2_LOW_RAM="\n', 1)
        self.assertNotEqual(probe, text)
        self.script().write_text(probe, encoding="utf-8")
        r = self.dry(STRATA_M2_LOW_RAM="0")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue(r.stdout.splitlines()[1].startswith("0"), r.stdout)
        self.assertIn("normal mode", r.stdout)


@unittest.skipUnless(shutil.which("sh"), "needs a POSIX shell")
class ScriptSyntax(unittest.TestCase):
    def test_the_script_parses(self):
        with tempfile.TemporaryDirectory() as t, mock.patch.object(setup, "ROOT", Path(t)), \
                mock.patch.object(setup, "CONTROL", None):
            s = setup.write_m2_run_script(Path(t) / "strata-minimax2-iq2_xxs.json", 8080, True, True, "test")
            self.assertEqual(subprocess.run(["sh", "-n", str(s)], timeout=30).returncode, 0)


if __name__ == "__main__":
    unittest.main()
