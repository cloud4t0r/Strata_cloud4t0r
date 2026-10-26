"""A build folder configured in another source folder: a repository moved or renamed leaves the old absolute path in
build-hip/CMakeCache.txt, and cmake then refuses to configure it ("The current CMakeCache.txt directory ... is
different than the directory ... where CMakeCache.txt was created").  setup drops that cache and configures the folder
again in place, reusing the objects it already holds - and compiles an engine it would otherwise call already built,
because the folder is also where strata-m2 is compiled.

    python tools/test_setup_builddir.py
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup  # noqa: E402


class MovedCache(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.bdir = self.root / "build-hip"
        self.bdir.mkdir()

    def test_this_folders_cache_is_kept(self):
        (self.bdir / "CMakeCache.txt").write_text(f"CMAKE_HOME_DIRECTORY:INTERNAL={self.root}\n", encoding="utf-8")
        with mock.patch.object(setup, "ROOT", self.root):
            self.assertIsNone(setup.moved_build_cache(self.bdir))
            self.assertIsNone(setup.drop_moved_build_cache(self.bdir))
        self.assertTrue((self.bdir / "CMakeCache.txt").exists())

    def test_another_folders_cache_is_dropped(self):
        (self.bdir / "CMakeCache.txt").write_text("CMAKE_HOME_DIRECTORY:INTERNAL=/home/x/Strata_cloudat0r\n",
                                                  encoding="utf-8")
        out = io.StringIO()
        with mock.patch.object(setup, "ROOT", self.root), contextlib.redirect_stdout(out):
            self.assertEqual(setup.moved_build_cache(self.bdir), "/home/x/Strata_cloudat0r")
            self.assertEqual(setup.drop_moved_build_cache(self.bdir), "/home/x/Strata_cloudat0r")
        self.assertFalse((self.bdir / "CMakeCache.txt").exists())
        self.assertIn("The build folder build-hip was made in /home/x/Strata_cloudat0r", out.getvalue())

    def test_no_cache_is_nothing(self):
        with mock.patch.object(setup, "ROOT", self.root):
            self.assertIsNone(setup.moved_build_cache(self.bdir))
            self.assertIsNone(setup.drop_moved_build_cache(self.bdir))

    def test_a_cache_without_the_source_folder_is_left_alone(self):
        (self.bdir / "CMakeCache.txt").write_text("CMAKE_BUILD_TYPE:STRING=Release\n", encoding="utf-8")
        with mock.patch.object(setup, "ROOT", self.root):
            self.assertIsNone(setup.moved_build_cache(self.bdir))
        self.assertTrue((self.bdir / "CMakeCache.txt").exists())


class Build(unittest.TestCase):
    def test_cmake_build_drops_the_cache_then_configures(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            bdir = root / "build-hip"
            bdir.mkdir()
            (bdir / "CMakeCache.txt").write_text("CMAKE_HOME_DIRECTORY:INTERNAL=/old/Strata_cloudat0r\n",
                                                 encoding="utf-8")
            ran = []
            out = io.StringIO()
            with mock.patch.object(setup, "ROOT", root), \
                    mock.patch.object(setup, "ensure_build_tools", lambda: ("cmake", "ninja")), \
                    mock.patch.object(setup, "run", lambda cmd, **k: ran.append(cmd) or mock.Mock(returncode=0)), \
                    contextlib.redirect_stdout(out):
                setup.cmake_build(root, bdir, "strata", [], None, "")
            self.assertFalse((bdir / "CMakeCache.txt").exists())
            self.assertEqual(ran[0][:2], ["cmake", "-G"])
            self.assertEqual(ran[0][ran[0].index("-S") + 1], str(root))
            self.assertEqual(ran[0][ran[0].index("-B") + 1], str(bdir))
            self.assertIn("was made in /old/Strata_cloudat0r", out.getvalue())


class EngineBuild(unittest.TestCase):
    def test_a_moved_build_folder_makes_the_engine_compile_again(self):
        """Without the check, engine_ok (the source hash and the archs match) returns before cmake_build: the stale
        cache is never refreshed, and the strata-m2 build in that folder then dies on it."""
        calls = {}

        class Sh:
            which = staticmethod(lambda name: "/usr/bin/" + name)
            copy2 = staticmethod(lambda a, b: Path(b).write_text(Path(a).read_text()))

        with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ):
            root = Path(d)
            bdir = root / "build-hip"
            bdir.mkdir()
            (bdir / "CMakeCache.txt").write_text(
                "CMAKE_HOME_DIRECTORY:INTERNAL=/old/Strata_cloudat0r\n"
                "CMAKE_HIP_ARCHITECTURES:UNINITIALIZED=gfx1201\n", encoding="utf-8")
            (root / "engine").mkdir()
            (root / "engine" / setup.EXE).write_text("old")
            (root / "engine" / "BUILD.json").write_text(json.dumps(
                {"backend": "hip", "source": "local-hip", "src": "h", "archs": ["gfx1201"], "version": "0.1.47"}),
                encoding="utf-8")

            def fake_build(src, b, target, defs, vcvars, bat):
                calls["defs"] = defs
                (b / setup.EXE).write_text("engine")
                (b / "CMakeCache.txt").write_text(f"CMAKE_HOME_DIRECTORY:INTERNAL={root}\n", encoding="utf-8")

            saved = {k: getattr(setup, k) for k in ("ROOT", "rocm_root", "cmake_build", "source_hash", "source_version",
                                                    "ok", "shutil", "stale_binary", "cpu_info", "cpu_floor")}
            out = io.StringIO()
            setup.ROOT, setup.cmake_build, setup.ok, setup.shutil = root, fake_build, lambda *a: None, Sh
            setup.source_hash, setup.source_version = (lambda *a: "h"), (lambda: "0.1.47")
            setup.rocm_root = lambda archs: (calls.setdefault("archs", archs) and root, [str(root / "lib")])
            setup.stale_binary = lambda p: False
            setup.cpu_info = lambda: ("CPU", "avx2")
            setup.cpu_floor = lambda f: ""
            try:
                with contextlib.redirect_stdout(out):
                    setup.build_engine_hip({"arch": "gfx1201", "archs": ["gfx1201"]}, root)
            finally:
                for k, v in saved.items():
                    setattr(setup, k, v)
            self.assertIn("defs", calls)                              # compiled, not "already built"
            self.assertIn("was made in /old/Strata_cloudat0r", out.getvalue())
            self.assertEqual((root / "engine" / setup.EXE).read_text(), "engine")


if __name__ == "__main__":
    unittest.main()
