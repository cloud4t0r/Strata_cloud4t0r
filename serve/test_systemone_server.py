"""serve/test_systemone_server.py - POST /v1/systemone over HTTP, with a fake decision engine.

The protocol itself is pinned by serve/test_systemone.py (no server, no Jinja).  This file pins the route: the
request reaches the service, a v1 body is a 400 that names the version, a model that is not a decider is a 404,
and an engine with no decision path is a 501 - never a dropped connection.

    python -m unittest serve.test_systemone_server -v
"""
from __future__ import annotations

import json
import math
import string
import sys
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, Service, serve  # noqa: E402
from serve import systemone  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CODES = (list(string.ascii_uppercase)
         + [f"{a}{b}" for a in string.ascii_uppercase for b in string.ascii_uppercase])[:systemone.MAX_OPTIONS]


class DecisionTokenizer(ByteTokenizer):
    """ByteTokenizer plus the readout marker as one special token."""

    SPECIALS = ByteTokenizer.SPECIALS + ["<|fim_suffix|>"]


class DecisionEngine:
    """No generate(): the route must not ask a decision model for text."""

    max_context = 32768

    def __init__(self) -> None:
        self.calls = []

    def decide(self, ids, markers):
        self.calls.append((list(ids), list(markers)))
        # 1:3 in log space, so a two-option question answers 0.75 after the softmax at temperature 1.
        return [[0.0, math.log(3.0)] + [0.0] * 253 for _ in markers]


def body(**overrides):
    base = {
        "model": "pplx-decider-flash-v1.2",
        "state": "My integration keeps failing with a 401.",
        "questions": {"urgent": {"type": "noul", "instructions": "Is the customer blocked?"}},
    }
    base.update(overrides)
    return base


class SystemOneRoute(unittest.TestCase):
    @classmethod
    def build(cls):
        """The startup path: decider.json in the tokenizer folder -> service_from_config -> DecideService."""
        return systemone.service_from_config(
            {"tokenizer": str(cls.tokenizer), "arch": "qwen35",
             "model_name": "pplx-decider-flash-v1.2-q8_0"},
            cls.tok, cls.template, cls.engine, max_context=32768)

    @classmethod
    def setUpClass(cls):
        cls.tok = DecisionTokenizer()
        cls.engine = DecisionEngine()
        cls.template = ChatTemplate(ROOT / "serve/chat_template.jinja")
        cls.svc = Service(cls.engine, cls.tok, cls.template, model_name="pplx-decider-flash-v1.2-q8_0")
        cls.tmp = tempfile.TemporaryDirectory()
        cls.tokenizer = Path(cls.tmp.name)
        (cls.tokenizer / "decider.json").write_text(json.dumps({
            "protocol_version": 3,
            "template": "joint-question-inline-readout-v1",
            "marker": "<|fim_suffix|>",
            "temperature": 1.0,
            "codes": CODES,
            "token_ids": list(range(len(CODES))),
            "escape_tokens": [],
            "max_pixels": 1638400,
            "max_length": 262144,
        }), encoding="utf-8")
        cls.svc.systemone = cls.build()
        cls.httpd = serve(cls.svc, port=0)
        cls.url = f"http://127.0.0.1:{cls.httpd.server_address[1]}/v1/systemone"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.tmp.cleanup()

    def setUp(self):
        self.engine.calls.clear()
        self.svc.systemone = self.build()

    def post(self, payload):
        request = urllib.request.Request(self.url, data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as exc:
            with exc:
                return exc.code, json.loads(exc.read() or b"{}")

    def test_a_decision_is_answered(self):
        status, got = self.post(body())
        self.assertEqual(status, 200)
        self.assertEqual(got["model"], "pplx-decider-flash-v1.2")
        self.assertEqual(got["answers"]["urgent"]["type"], "noul")
        self.assertAlmostEqual(got["answers"]["urgent"]["noul"], 0.75)
        self.assertEqual(got["usage"], {"input_tokens": got["usage"]["input_tokens"], "output_tokens": 1,
                                        "sequences": 1})
        self.assertEqual(len(self.engine.calls), 1)

    def test_choice_and_score_keep_their_shapes(self):
        status, got = self.post(body(questions={
            "team": {"type": "choice", "criteria": {"billing": None, "support": None}},
            "grade": {"type": "score", "criteria": ["low", "high"]},
        }))
        self.assertEqual(status, 200)
        self.assertEqual(got["answers"]["team"]["type"], "choice")
        self.assertEqual(got["answers"]["team"]["choice"], "support")
        self.assertAlmostEqual(got["answers"]["team"]["probabilities"]["support"], 0.75)
        self.assertEqual(got["answers"]["grade"]["type"], "score")
        self.assertEqual(got["answers"]["grade"]["legend"], {"0": "low", "1": "high"})

    def test_a_v1_body_is_a_400_that_names_the_version(self):
        status, got = self.post({"system": "be helpful", "question": "is it urgent?"})
        self.assertEqual(status, 400)
        self.assertIn("protocol v3", got["error"]["message"])

    def test_unknown_fields_are_a_400(self):
        status, got = self.post(body(extra=1))
        self.assertEqual(status, 400)
        self.assertIn("unknown field", got["error"]["message"])

    def test_images_are_a_400_not_a_silent_drop(self):
        status, got = self.post(body(images=[{"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}]))
        self.assertEqual(status, 400)
        self.assertIn("text only", got["error"]["message"])

    def test_unknown_model_is_a_400(self):
        status, got = self.post(body(model="some-other-model"))
        self.assertEqual(status, 400)
        self.assertIn("unknown model", got["error"]["message"])

    def test_a_model_that_is_not_a_decider_is_a_404(self):
        self.svc.systemone = None
        status, got = self.post(body())
        self.assertEqual(status, 404)
        self.assertIn("not a decision model", got["error"]["message"])

    def test_an_engine_that_died_is_a_503_not_a_500(self):
        from serve.server import EngineDied

        def dead(ids, markers):
            raise EngineDied("the engine stopped unexpectedly")

        self.svc.systemone = systemone.DecideService(
            systemone.Encoder(self.tok, self.template, CODES), dead,
            temperature=1.0, model_name="pplx-decider-flash-v1.2-q8_0")
        status, got = self.post(body())
        self.assertEqual(status, 503)
        self.assertIn("stopped unexpectedly", got["error"]["message"])

    def test_an_engine_without_a_decision_path_is_a_501(self):
        def missing(ids, markers):
            raise systemone.EngineUnavailable("this engine build has no decision path yet")

        self.svc.systemone = systemone.DecideService(
            systemone.Encoder(self.tok, self.template, CODES), missing,
            temperature=1.0, model_name="pplx-decider-flash-v1.2-q8_0")
        status, got = self.post(body())
        self.assertEqual(status, 501)
        self.assertIn("no decision path", got["error"]["message"])


if __name__ == "__main__":
    unittest.main()
