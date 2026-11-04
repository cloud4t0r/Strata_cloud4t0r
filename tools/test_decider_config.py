"""tools/test_decider_config.py - the decider.json writer, without a GPU or the 5 GB file.

Builds a tiny synthetic GGUF with the protocol's metadata, runs the tool on it, and checks that
serve.systemone.load_config reads back what it wrote.

    python tools/test_decider_config.py -v
"""
from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
from decider_config import config_from_metadata, main  # noqa: E402
from gguf_writer import GGUFWriter  # noqa: E402
from serve.systemone import load_config  # noqa: E402


def metadata(**changes):
    base = {
        "general.architecture": "qwen35",
        "qwen35.attention.causal": False,
        "qwen35.context_length": 262144,
        "pplx_decider.protocol_version": 3,
        "pplx_decider.template": "joint-question-inline-readout-v1",
        "pplx_decider.marker": "<|fim_suffix|>",
        "pplx_decider.temperature": 1.5539092221462472,
        "pplx_decider.codes": ["A", "B", "C"],
        "pplx_decider.token_ids": [32, 33, 34],
        "pplx_decider.escape_tokens": ["<|im_start|>", "<|im_end|>"],
        "pplx_decider.max_pixels": 1638400,
        "pplx_decider.base_model": "Qwen/Qwen3.5-4B",
    }
    base.update(changes)
    return base


def write_gguf(path: Path, **changes) -> Path:
    writer = GGUFWriter()
    for key, value in metadata(**changes).items():
        writer.add(key, value)
    return writer.write(path)


class ConfigFromMetadata(unittest.TestCase):
    def test_the_real_metadata_produces_the_protocol(self):
        got = config_from_metadata(metadata())
        self.assertEqual(got["protocol_version"], 3)
        self.assertEqual(got["marker"], "<|fim_suffix|>")
        self.assertAlmostEqual(got["temperature"], 1.5539092221462472)
        self.assertEqual(got["codes"], ["A", "B", "C"])
        self.assertEqual(got["token_ids"], [32, 33, 34])
        self.assertEqual(got["max_length"], 262144)
        self.assertEqual(got["attention_mode"], "noncausal_full_attention")
        self.assertEqual(got["pooling"], "last")

    def test_the_context_is_the_limit(self):
        self.assertEqual(config_from_metadata(metadata(**{"qwen35.context_length": 32768}))["max_length"], 32768)

    def test_a_causal_file_is_refused(self):
        with self.assertRaisesRegex(ValueError, "causal"):
            config_from_metadata(metadata(**{"qwen35.attention.causal": True}))

    def test_a_missing_causal_flag_is_refused(self):
        meta = metadata()
        meta.pop("qwen35.attention.causal")
        with self.assertRaisesRegex(ValueError, "causal"):
            config_from_metadata(meta)

    def test_another_architecture_is_refused(self):
        with self.assertRaisesRegex(ValueError, "qwen35"):
            config_from_metadata(metadata(**{"general.architecture": "qwen35moe"}))

    def test_protocol_v1_is_refused(self):
        with self.assertRaisesRegex(ValueError, "protocol_version"):
            config_from_metadata(metadata(**{"pplx_decider.protocol_version": 1}))

    def test_another_template_is_refused(self):
        with self.assertRaisesRegex(ValueError, "template"):
            config_from_metadata(metadata(**{"pplx_decider.template": "single-question-v1"}))

    def test_another_marker_is_refused(self):
        with self.assertRaisesRegex(ValueError, "marker"):
            config_from_metadata(metadata(**{"pplx_decider.marker": "<|endoftext|>"}))

    def test_temperature_must_be_positive_and_finite(self):
        for value in (0, -1, float("inf"), "1.5"):
            with self.assertRaises(ValueError):
                config_from_metadata(metadata(**{"pplx_decider.temperature": value}))

    def test_codes_and_token_ids_must_agree(self):
        with self.assertRaisesRegex(ValueError, "same length"):
            config_from_metadata(metadata(**{"pplx_decider.token_ids": [32, 33]}))

    def test_codes_must_be_nonempty_strings(self):
        with self.assertRaises(ValueError):
            config_from_metadata(metadata(**{"pplx_decider.codes": ["A", ""]}))

    def test_token_ids_must_be_nonnegative_integers(self):
        with self.assertRaises(ValueError):
            config_from_metadata(metadata(**{"pplx_decider.token_ids": [32, -1]}))

    def test_no_more_than_255_codes(self):
        with self.assertRaises(ValueError):
            config_from_metadata(metadata(**{"pplx_decider.codes": [f"c{i}" for i in range(256)],
                                            "pplx_decider.token_ids": list(range(256))}))

    def test_escape_tokens_are_optional(self):
        meta = metadata()
        meta.pop("pplx_decider.escape_tokens")
        self.assertEqual(config_from_metadata(meta)["escape_tokens"], [])


class CommandLine(unittest.TestCase):
    def test_gguf_to_stdout(self):
        with tempfile.TemporaryDirectory() as folder:
            path = write_gguf(Path(folder) / "tiny.gguf")
            out = io.StringIO()
            with redirect_stdout(out):
                code = main(["--gguf", str(path), "--stdout"])
            self.assertEqual(code, 0)
            got = json.loads(out.getvalue())
            self.assertEqual(got["codes"], ["A", "B", "C"])
            self.assertEqual(got["token_ids"], [32, 33, 34])

    def test_gguf_to_a_tokenizer_folder_and_back_through_the_server_loader(self):
        with tempfile.TemporaryDirectory() as folder:
            path = write_gguf(Path(folder) / "tiny.gguf")
            tokenizer = Path(folder) / "tokenizer"
            tokenizer.mkdir()
            with redirect_stdout(io.StringIO()):
                code = main(["--gguf", str(path), "--out", str(tokenizer)])
            self.assertEqual(code, 0)
            data = load_config(tokenizer)
            self.assertEqual(data["temperature"], 1.5539092221462472)
            self.assertEqual(data["codes"], ["A", "B", "C"])

    def test_check_accepts_what_the_tool_wrote(self):
        with tempfile.TemporaryDirectory() as folder:
            path = write_gguf(Path(folder) / "tiny.gguf")
            tokenizer = Path(folder) / "tokenizer"
            tokenizer.mkdir()
            with redirect_stdout(io.StringIO()):
                main(["--gguf", str(path), "--out", str(tokenizer)])
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--check", str(tokenizer)]), 0)

    def test_check_refuses_an_empty_folder(self):
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--check", folder]), 1)

    def test_a_refused_file_writes_nothing(self):
        with tempfile.TemporaryDirectory() as folder:
            path = write_gguf(Path(folder) / "bad.gguf", **{"pplx_decider.protocol_version": 1})
            tokenizer = Path(folder) / "tokenizer"
            tokenizer.mkdir()
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--gguf", str(path), "--out", str(tokenizer)]), 2)
            self.assertFalse((tokenizer / "decider.json").exists())


if __name__ == "__main__":
    unittest.main()
