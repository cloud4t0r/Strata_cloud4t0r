"""serve/test_systemone_goldens.py - serve/systemone.py against Perplexity's own numbers, without torch.

    python -m unittest serve.test_systemone_goldens -v

`serve/decider-goldens.json` is written by `tools/decider_goldens.py`, which runs the checkpoint's reference
package (`pplx_decider`) and records, per request, the token ids of every sequence, the marker positions, the raw
readout logits per question and the answers its own `answer()` produced.  It was generated on CPU (float32) from
`perplexity-ai/pplx-decider-flash-v1.2`, revision 9ff7cf98, with `--max-length 32768`:

    python tools/decider_goldens.py --checkpoint <checkpoint dir> --device cpu --max-length 32768 \\
        --out serve/decider-goldens.json

This file pins the arithmetic this server owns: the temperature applied once, the softmax over the *valid*
options only, and the `noul` / `choice` / `score` shapes.  The engine's own logits are not involved, so the test
runs anywhere.  The reference's float32 softmax and `math.exp` over the same values differ by rounding only; the
tolerance below is far above that (measured maximum: 1.7e-7) and far below any real drift.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import systemone  # noqa: E402

GOLDENS = Path(__file__).resolve().parent / "decider-goldens.json"
TOLERANCE = 1e-6


class GoldenProtocol(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.goldens = json.loads(GOLDENS.read_text(encoding="utf-8"))

    def test_the_file_describes_protocol_v3(self) -> None:
        goldens = self.goldens
        self.assertEqual(goldens["format"], 1)
        self.assertEqual(goldens["attention_mode"], "noncausal_full_attention")
        self.assertEqual(len(goldens["codes"]), systemone.MAX_OPTIONS)
        self.assertEqual(len(goldens["token_ids"]), systemone.MAX_OPTIONS)
        self.assertGreater(goldens["temperature"], 0)

    def test_every_sequence_has_one_marker_per_question(self) -> None:
        for case in self.goldens["cases"]:
            questions = len(case["request"]["questions"])
            markers = 0
            for prompt in case["prompts"]:
                self.assertGreaterEqual(len(prompt["markers"]), 1, case["name"])
                markers += len(prompt["markers"])
                self.assertEqual(prompt["length"], len(prompt["input_ids"]), case["name"])
                self.assertTrue(all(0 <= m < prompt["length"] for m in prompt["markers"]), case["name"])
                self.assertEqual(prompt["markers"], sorted(prompt["markers"]), case["name"])
            self.assertEqual(markers, questions, case["name"])

    def test_answers_match_the_reference(self) -> None:
        for case in self.goldens["cases"]:
            for key, question in case["request"]["questions"].items():
                with self.subTest(case=case["name"], question=key):
                    raw = case["logits"][key]
                    probabilities = systemone.softmax([value / self.goldens["temperature"] for value in raw])
                    got = systemone.answer(question, probabilities)
                    want = case["answers"][key]
                    self.assertEqual(got["type"], want["type"])
                    self._compare(got, want, case["name"], key)

    def _compare(self, got: dict, want: dict, case: str, key: str) -> None:
        for field in ("noul", "score", "confidence"):
            if field in want:
                self.assertAlmostEqual(got[field], want[field], delta=TOLERANCE,
                                       msg=f"{case}/{key}: {field}")
        if "choice" in want:
            self.assertEqual(got["choice"], want["choice"], f"{case}/{key}: choice")
        for option, value in want.get("probabilities", {}).items():
            self.assertAlmostEqual(got["probabilities"][option], value, delta=TOLERANCE,
                                   msg=f"{case}/{key}: {option}")
        if "legend" in want:
            self.assertEqual(got["legend"], want["legend"], f"{case}/{key}: legend")


if __name__ == "__main__":
    unittest.main()
