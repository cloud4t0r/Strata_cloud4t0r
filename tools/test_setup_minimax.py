"""Tests for setup.py's MiniMax-M2 install (OWN_PATH_FAMILIES "minimax2"): the two pinned shards, the engine build
(strata-m2 compiled only when MiniMax-M2 is chosen), the low-RAM mode (--m2-low-ram auto|on|off; auto by the RAM,
M2_LOW_RAM_BELOW_GB) and the start script run_minimax_m2.sh with its M2_LOW_RAM switch.  Mocked engine, downloads
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
                vram_reserve_mib=None, host=None, api_key=None, browser=None, no_start=True, data_dir=None, gpu=None,
                build=False, prebuilt=None, m2_low_ram="auto", layer_split=None)
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
    def test_auto_follows_the_ram(self):
        self.assertTrue(setup.m2_low_ram_mode("auto", 64.0)[0])
        self.assertFalse(setup.m2_low_ram_mode("auto", 128.0)[0])
        self.assertFalse(setup.m2_low_ram_mode("auto", float(setup.M2_LOW_RAM_BELOW_GB))[0])   # at the threshold: normal
        self.assertTrue(setup.m2_low_ram_mode("auto", setup.M2_LOW_RAM_BELOW_GB - 0.5)[0])
        self.assertEqual(setup.M2_LOW_RAM_BELOW_GB, 96)

    def test_on_and_off_override_the_detection(self):
        self.assertTrue(setup.m2_low_ram_mode("on", 256.0)[0])
        self.assertFalse(setup.m2_low_ram_mode("off", 32.0)[0])

    def test_ram_need_follows_the_mode(self):
        fam = setup.OWN_PATH_FAMILIES["minimax2"]
        self.assertEqual(setup.own_path_ram_need(fam, "IQ2_XXS", True), 24)
        self.assertEqual(setup.own_path_ram_need(fam, "IQ2_XXS", False), setup.M2_LOW_RAM_BELOW_GB)
        q = setup.OWN_PATH_FAMILIES["qwen36"]                   # no low-RAM choice: its own need either way
        self.assertEqual(setup.own_path_ram_need(q, "Q8_0", False), q["models"]["Q8_0"]["ram_gb"])


class Context(unittest.TestCase):
    """The recommended context follows MiniMax-M2's fp32 KV cache on the one card the engine runs on; Qwen3.6's rule
    is unchanged."""
    M2 = setup.OWN_PATH_FAMILIES["minimax2"]
    Q36 = setup.OWN_PATH_FAMILIES["qwen36"]

    def test_the_kv_cache_matches_the_gguf(self):
        # 62 layers, attention.head_count_kv = 8, key/value_length 128, fp32 K and V: 15.5 GiB per 32K tokens
        self.assertEqual(self.M2["kv_bytes_per_token"], 62 * 8 * 128 * 2 * 4)
        self.assertAlmostEqual(self.M2["kv_bytes_per_token"] * 32768 / 2 ** 30, 15.5)

    def test_a_16_gb_card_gets_a_context_that_fits(self):
        for vram in (16.0, 17095983104 / 2 ** 30):            # 16 GB, and the RX 6800's sysfs figure (15.92 GiB)
            rec = setup.own_path_context(self.M2, vram)
            self.assertNotEqual(rec, 131072)
            self.assertEqual(rec, 16384, vram)
            b = self.M2["vram_budget"]
            self.assertLessEqual(rec * self.M2["kv_bytes_per_token"] / 2 ** 30 + b["dense_gib"] + b["fixed_gib"], vram)
            self.assertLessEqual(setup.own_path_ctx_need_gib(self.M2, rec) + b["experts_gib"], vram)
            self.assertGreater(setup.own_path_ctx_need_gib(self.M2, 32768), vram)   # 32K cannot start there

    def test_a_bigger_card_gets_a_longer_context(self):
        self.assertEqual(setup.own_path_context(self.M2, 24.0), 32768)
        self.assertEqual(setup.own_path_context(self.M2, 48.0), 65536)
        self.assertGreater(setup.own_path_context(self.M2, 48.0), setup.own_path_context(self.M2, 16.0))
        self.assertEqual(setup.own_path_context(self.M2, 8.0), 8192)    # nothing fits: the smallest

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
        self.assertAlmostEqual(setup.own_path_split_need_gib(self.M2, 32768, 31, True), 13.174, places=2)
        self.assertAlmostEqual(setup.own_path_split_need_gib(self.M2, 32768, 31, False), 12.244, places=2)
        self.assertAlmostEqual(setup.own_path_split_need_gib(self.M2, 65536, 31, True), 21.430, places=2)
        self.assertAlmostEqual(setup.own_path_split_room_gib(self.M2, 31), 1.0)
        # all 62 layers on one card: the one-card rule
        self.assertAlmostEqual(setup.own_path_split_need_gib(self.M2, 32768, 62, True),
                               setup.own_path_ctx_need_gib(self.M2, 32768))

    def test_the_tighter_card_decides(self):
        ctx = lambda avail: setup.own_path_split_context(self.M2, avail, 31)
        self.assertEqual(ctx([15.922, 15.984]), 32768)
        self.assertEqual(ctx([15.922 - 1.155, 15.968]), 32768)    # the RX 9070 with the desktop's share in use
        self.assertEqual(ctx([16.0, 16.0]), 32768)
        self.assertEqual(ctx([24.0, 24.0]), 65536)
        self.assertEqual(ctx([12.0, 12.0]), 16384)
        self.assertEqual(ctx([16.0, 10.0]), 16384)                # the 10 GB card decides
        self.assertEqual(ctx([10.0, 16.0]), 8192)                 # first: it also holds all the dense weights
        self.assertEqual(ctx([6.0, 6.0]), 8192)                   # nothing fits: the smallest
        self.assertEqual(ctx([0.0, 16.0]), 8192)                  # an unknown card is not counted as roomy

    def test_misfit(self):
        self.assertEqual(setup.own_path_split_misfit(self.M2, 65536, [15.9, 15.9], 31), 0)
        self.assertIsNone(setup.own_path_split_misfit(self.M2, 32768, [15.9, 15.9], 31))
        self.assertEqual(setup.own_path_split_misfit(self.M2, 32768, [15.9, 12.0], 31), 1)
        self.assertIsNone(setup.own_path_split_misfit(self.M2, 65536, [0.0, 0.0], 31))   # unknown: not judged

    def test_one_card_sizing_is_unchanged(self):
        # the one-card rule never sums the cards: 16 GB -> 16K, however many cards the PC has
        self.assertEqual(setup.own_path_context(self.M2, R6800["vram_gb"]), 16384)
        self.assertEqual(setup.own_path_context(self.M2, R9070["vram_gb"]), 16384)
        self.assertEqual(setup.own_path_context(self.M2, R6800["vram_gb"] + R9070["vram_gb"]), 32768)   # a sum would


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

    def install(self, family="minimax2", ram=64.0, gpu=None, multi=None, hip=True, amd=None, used=None, **kw):
        gpu = gpu or {"index": 0, "count": 1, "vram_gb": 16.0, "name": "Test GPU"}
        with mock.patch.object(setup, "amd_vram_used_gib", lambda *a, **k: dict(used or {})):
            return run(setup.install_own_path, args(models_dir=str(self.root / "models"), family=family, **kw), family,
                       gpu, hip, amd or [], None, multi or [], 8080, ram, None)

    def ctx(self):
        a = self.cfg["args"]
        return a[a.index("--max-context") + 1]

    def test_a_16_gb_card_writes_a_context_that_fits(self):
        rc, out = self.install()
        self.assertEqual(rc, 0, out)
        self.assertEqual(self.ctx(), "16384")
        self.assertNotIn("context does not fit", out)             # no warning for the recommended context

    def test_two_16_gb_cards_get_the_one_cards_context(self):
        # two AMD cards present, no --gpus: one card, sized on ITS OWN VRAM (16384 on 16 GB), never on the sum
        gpu = {**R6800, "count": 2, "archs": ["gfx1030", "gfx1201"]}
        rc, out = self.install(gpu=gpu, amd=[R6800, R9070], used={0: 0.016, 1: 0.767})
        self.assertEqual(rc, 0, out)
        self.assertNotIn("runs on one GPU so far", out)
        self.assertEqual(self.ctx(), "16384")
        self.assertEqual(self.cfg["gpu"], 0)
        self.assertNotIn("layer_split", self.cfg)
        self.assertNotIn("layer split", out)

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
        self.assertEqual(self.ctx(), "32768")                     # the tighter card (the RX 9070, 0.77 GB used) fits 32K
        self.assertNotIn("--layer-split", self.cfg["args"])       # the server adds it from "layer_split"
        self.assertNotIn("--peer-device", self.cfg["args"])
        self.assertIn("--low-ram", self.cfg["args"])
        self.assertNotIn("runs on one GPU so far", out)
        self.assertIn("layers 0-30 on GPU 1, 31-61 on GPU 0", out)
        self.assertIn("GPU 1 (AMD Radeon RX 9070): layers 0-30, the embedding", out)
        self.assertIn("GPU 0 (AMD Radeon RX 6800): layers 31-61", out)

    def test_gpus_0_1_keeps_the_order(self):
        rc, out = self.split(main=R6800, multi=(0, 1))
        self.assertEqual(rc, 0, out)
        self.assertEqual(self.cfg["gpu"], [0, 1])
        self.assertEqual(self.cfg["layer_split"], "31")
        self.assertEqual(self.ctx(), "32768")
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
        rc, out = self.split(context=65536)
        self.assertEqual(rc, 0, out)
        self.assertIn("a 64K context does not fit MiniMax-M2 on GPU 1", out)   # the first card: all the dense weights
        self.assertIn("its share of the KV cache (31 of 62 layers) is 15.5 GB", out)
        self.assertIn("kept as you chose", out)
        self.assertEqual(self.ctx(), "65536")

    def test_split_yes_alone_never_writes_a_context_that_cannot_start(self):
        small = [{**R6800, "vram_gb": 6.0}, {**R9070, "vram_gb": 6.0}]
        rc, out = self.split(main=small[1], amd=small)
        self.assertNotEqual(rc, 0, out)
        self.assertIn("does not fit GPU 1", out)
        self.assertEqual(self.cfg, {})

    def test_unreadable_used_vram_is_said(self):
        rc, out = self.split(used={})
        self.assertEqual(rc, 0, out)
        self.assertIn("could not be read", out)
        self.assertEqual(self.ctx(), "32768")

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
        self.assertIn("3) 32K tokens   (KV cache 15.5 GB, about half on each card)   (recommended for your cards)", out)
        self.assertIn("4) 64K tokens   (KV cache 31.0 GB, about half on each card: does not fit your cards)", out)
        self.assertEqual(self.ctx(), "32768")

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
        self.assertEqual(self.ctx(), "16384")

    def test_one_card_config_is_unchanged(self):
        folder = self.root / "models" / "minimax2-IQ2_XXS"
        want = ["--native", str(folder / SHARDS[0]), "--max-context", "16384", "--prefill", "auto", "--mtp", "0",
                "--low-ram"]
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
        self.assertEqual(self.ctx(), "65536")

    def test_an_explicit_context_that_does_not_fit_is_said_and_kept(self):
        rc, out = self.install(context=131072)                    # --context 131072 --yes: the user's choice
        self.assertEqual(rc, 0, out)
        self.assertIn("a 128K context does not fit MiniMax-M2", out)
        self.assertIn("its KV cache alone is 62.0 GB", out)
        self.assertIn("kept as you chose", out)
        self.assertEqual(self.ctx(), "131072")

    def test_yes_alone_never_writes_a_context_that_cannot_start(self):
        rc, out = self.install(gpu={"index": 0, "count": 1, "vram_gb": 6.0, "name": "Test GPU"})
        self.assertNotEqual(rc, 0, out)
        self.assertIn("does not fit the 6 GB card", out)
        self.assertEqual(self.cfg, {})                            # no config written

    def test_the_context_menu_shows_the_kv_cache(self):
        with mock.patch.object(setup, "ask", lambda q, choices, default, yes=False: default):
            rc, out = self.install(yes=False)
        self.assertEqual(rc, 0, out)
        self.assertIn("2) 16K tokens   (KV cache 7.8 GB)   (recommended for your GPU)", out)
        self.assertIn("3) 32K tokens   (KV cache 15.5 GB: does not fit your 16 GB card)", out)
        self.assertEqual(self.ctx(), "16384")

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

    def test_64_gb_auto_writes_low_ram(self):
        rc, out = self.install(ram=64.0)
        self.assertEqual(rc, 0, out)
        self.assertIn("low-RAM mode: on (--low-ram)", out)
        self.assertIn("--low-ram", self.cfg["args"])
        text = self.script().read_text(encoding="utf-8")
        self.assertIn("DEFAULT=1", text)
        self.assertTrue(os.access(self.script(), os.X_OK))
        r = self.dry()
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("low-RAM mode on", r.stdout)
        self.assertIn(str(self.root / "strata-minimax2-iq2_xxs.json"), r.stdout)   # the config as setup wrote it

    def test_128_gb_auto_writes_the_normal_mode(self):
        rc, out = self.install(ram=128.0)
        self.assertEqual(rc, 0, out)
        self.assertIn("low-RAM mode: off (normal mode)", out)
        self.assertNotIn("--low-ram", self.cfg["args"])
        self.assertIn("DEFAULT=0", self.script().read_text(encoding="utf-8"))
        self.assertIn("normal mode", self.dry().stdout)

    def test_forced_modes_override_the_detection(self):
        rc, out = self.install(ram=128.0, m2_low_ram="on")
        self.assertEqual(rc, 0, out)
        self.assertIn("--low-ram", self.cfg["args"])
        self.assertIn("DEFAULT=1", self.script().read_text(encoding="utf-8"))
        rc, out = self.install(ram=64.0, m2_low_ram="off")       # explicit: a warning about the RAM, not a stop
        self.assertEqual(rc, 0, out)
        self.assertIn("in the normal mode", out)
        self.assertNotIn("--low-ram", self.cfg["args"])
        self.assertIn("DEFAULT=0", self.script().read_text(encoding="utf-8"))

    def test_m2_low_ram_switches_the_mode_for_one_run(self):
        rc, out = self.install(ram=64.0)                         # default on
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
