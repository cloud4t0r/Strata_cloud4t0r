"""tools/decider_parity.py - hold the decider's CPU reference to the reference package's own logits.

WHAT IT DOES.  For every case in `serve/decider-goldens.json` (written by tools/decider_goldens.py from
perplexity-ai/pplx-decider-flash-v1.2 itself), it runs `strata-decider --ref` on the same token ids and marker
positions, then compares the raw logits of each question - the first `count` codes, which are the valid options -
and the probabilities the calibrated temperature gives.  This is the check that says whether the GGUF reader, the
Gated DeltaNet recurrence, the non-causal attention and the readout are wired the way the checkpoint is.

WHAT A DIFFERENCE MEANS.  The reference runs in float32 on the BF16 checkpoint; `strata-decider --ref` reads the
Q8_0 GGUF.  The vendor measured that quantisation at a maximum probability difference of 0.016 over its own golden
questions, so a probability difference of that order is expected and is not a bug.  A structural mistake (the wrong
v/k head pairing, a causal mask, the +1 counted twice in a norm) is orders of magnitude larger, and usually changes
which option wins.

    python tools/decider_parity.py --gguf <decider-flash-v1.2-q8_0.gguf> --binary build/strata-decider
    python tools/decider_parity.py --gguf ... --cases noul,choice     # a subset
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GOLDENS = ROOT / "serve" / "decider-goldens.json"
DEFAULT_BINARY = ROOT / "build-tests" / "strata-decider"
#: Probability.  The vendor measured its own Q8_0 against the float32 reference at 0.016 maximum over 13 questions;
#: over our 13 the worst is 0.026 (hostile-marker), which is the same order.  A structural mistake is far larger and
#: normally changes which option wins, so 0.03 separates the two without hiding a real one.
DEFAULT_TOLERANCE = 0.03


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gguf", required=True, help="the Q8_0 GGUF the reference reads")
    parser.add_argument("--binary", default=str(DEFAULT_BINARY), help=f"strata-decider (default: {DEFAULT_BINARY})")
    parser.add_argument("--goldens", default=str(DEFAULT_GOLDENS), help=f"default: {DEFAULT_GOLDENS}")
    parser.add_argument("--cases", default=None, help="comma separated case names (default: all)")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE,
                        help=f"maximum probability difference (default: {DEFAULT_TOLERANCE})")
    parser.add_argument("--logit-report", type=float, default=0.25,
                        help="print the worst logit difference above this (default: 0.25)")
    return parser.parse_args(argv)


def softmax(values, temperature: float) -> list[float]:
    scaled = [value / temperature for value in values]
    top = max(scaled)
    exps = [math.exp(value - top) for value in scaled]
    total = sum(exps)
    return [value / total for value in exps]


def run_reference(binary: str, gguf: str, ids: list[int], markers: list[int]) -> list[list[float]]:
    command = [binary, "--ref", gguf, "--tokens", ",".join(map(str, ids)), "--markers", ",".join(map(str, markers))]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"strata-decider --ref failed ({result.returncode}):\n{result.stderr[-2000:]}")
    return json.loads(result.stdout)["logits"]


def main(argv=None) -> int:
    args = parse_args(argv)
    goldens = json.loads(Path(args.goldens).read_text(encoding="utf-8"))
    temperature = goldens["temperature"]
    wanted = set(args.cases.split(",")) if args.cases else None
    cases = [case for case in goldens["cases"] if wanted is None or case["name"] in wanted]
    if not cases:
        raise SystemExit("no case selected")

    worst_probability = 0.0
    worst_logit = 0.0
    failures = 0
    for case in cases:
        request = case["request"]
        keys = list(request["questions"])
        # one question per sequence (per_question) or all of them in one (joint): the goldens carry both
        if case["grouping"] == "per_question" and len(keys) > 1:
            runs = []
            for index, key in enumerate(keys):
                one = {**request, "questions": {key: request["questions"][key]}}
                runs.append((key, one, case["prompts"][index]))
        else:
            runs = [(None, request, case["prompts"][0])]

        logits_by_key: dict[str, list[float]] = {}
        for key, one, prompt in runs:
            logits = run_reference(args.binary, args.gguf, prompt["input_ids"], prompt["markers"])
            for index, question_key in enumerate(one["questions"]):
                logits_by_key[question_key] = logits[index]

        case_failures = []
        for key in keys:
            want = case["logits"][key]
            got = logits_by_key[key][:len(want)]
            count = len(want)
            want_probabilities = softmax(want, temperature)
            got_probabilities = softmax(got, temperature)
            logit_difference = max(abs(a - b) for a, b in zip(got, want))
            probability_difference = max(abs(a - b) for a, b in zip(got_probabilities, want_probabilities))
            worst_logit = max(worst_logit, logit_difference)
            worst_probability = max(worst_probability, probability_difference)
            best_want = max(range(count), key=want_probabilities.__getitem__)
            best_got = max(range(count), key=got_probabilities.__getitem__)
            # The winning option is only a criterion when the reference itself separates it from the runner-up by
            # more than the tolerance: with 255 near-uniform options the top two sit 3e-6 apart, and any rounding
            # flips the argmax without anything being wrong (the max-options case does exactly that).
            runner_up = sorted(want_probabilities, reverse=True)[1] if count > 1 else 0.0
            margin = want_probabilities[best_want] - runner_up
            tie = margin <= args.tolerance
            line = (f"  {case['name']:20s} {key:10s} logits {logit_difference:7.4f}  "
                    f"p {probability_difference:7.4f}  choice {best_got} vs {best_want}"
                    + (f"  (tie, margin {margin:.2g})" if tie and best_got != best_want else ""))
            print(line, flush=True)
            if logit_difference > args.logit_report:
                print(f"      worst logit: got {max(got):.4f} / want {max(want):.4f}", flush=True)
            if probability_difference > args.tolerance or (best_got != best_want and not tie):
                case_failures.append(key)
        if case_failures:
            failures += 1
            print(f"FAIL {case['name']}: {', '.join(case_failures)}", flush=True)

    print(f"\ndecider_parity: {len(cases) - failures}/{len(cases)} cases, worst logit {worst_logit:.4f}, "
          f"worst probability {worst_probability:.4f} (tolerance {args.tolerance})")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
