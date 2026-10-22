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


def args(**kw):
    base = dict(family="minimax2", model=None, check=False, yes=True, vision=None, models_dir=None, gguf_dir=None,
                context=None, rope_scaling=None, rope_scale=None, kv=None, resident_budget_gib=None, low_ram="auto",
                vram_reserve_mib=None, host=None, api_key=None, browser=None, no_start=True, data_dir=None, gpu=None,
                build=False, prebuilt=None, m2_low_ram="auto")
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

    def install(self, family="minimax2", ram=64.0, **kw):
        gpu = {"index": 0, "count": 1, "vram_gb": 16.0, "name": "Test GPU"}
        return run(setup.install_own_path, args(models_dir=str(self.root / "models"), family=family, **kw), family, gpu,
                   True, [], None, [], 8080, ram, None)

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
