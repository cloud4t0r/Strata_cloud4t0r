"""tools/decider_block_parity.py - hold one GPU kernel to the decider's CPU reference, stage by stage.

WHAT IT DOES.  Runs the reference over a golden case and dumps one block's stages as raw f32 (`strata-decider
--ref --dump-dir DIR --dump-block N`: "x_post" before the MLP's norm, "xn_post" after it, "mlp" the MLP's output),
then runs `strata-decider-gpu` on the same input and compares the two outputs element by element.

    python tools/decider_block_parity.py --gguf <q8_0.gguf> --case noul --block 0
    python tools/decider_block_parity.py --gguf ... --dump-dir /tmp/d --block 0   # reuse an earlier dump

WHAT A DIFFERENCE MEANS.  The reference dequantizes the weights and does every dot in float32; the GPU reads the
same Q8_0 rows but `iq_mmvq` quantizes the ACTIVATIONS to q8_1 first, as the engine does.  So a projection cannot
be bitwise equal to the reference: the expected difference is the activation quantization, about 1% relative, and
the cosine stays above 0.9999.  The norm has no such excuse - it is elementwise - so it is held to float rounding
(1e-6 relative), and a wrong epsilon, a wrong weight or a wrong +1 shows up there immediately.

    exit 0  every stage within tolerance
    exit 1  a stage outside it (the message names the stage and the measure)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GOLDENS = ROOT / "serve" / "decider-goldens.json"
DEFAULT_BINARY = ROOT / "build-tests" / "strata-decider"
DEFAULT_GPU_BINARY = ROOT / "build-hip" / "strata-decider-gpu"

#: The norm is elementwise: only float rounding separates the two.
NORM_TOLERANCE = 1e-6
#: The projections quantize their activations to q8_1, as the engine does; measured at 0.9% on block 0.
PROJECTION_TOLERANCE = 0.02


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gguf", required=True, help="the Q8_0 GGUF both sides read")
    parser.add_argument("--case", default="noul", help="a golden case name (default: noul)")
    parser.add_argument("--block", type=int, default=0, help="the block to compare (default: 0)")
    parser.add_argument("--dump-dir", default=None, help="where the stages are written; reused when they exist")
    parser.add_argument("--goldens", default=str(DEFAULT_GOLDENS))
    parser.add_argument("--binary", default=str(DEFAULT_BINARY), help=f"strata-decider (default: {DEFAULT_BINARY})")
    parser.add_argument("--gpu-binary", default=str(DEFAULT_GPU_BINARY),
                        help=f"strata-decider-gpu (default: {DEFAULT_GPU_BINARY})")
    return parser.parse_args(argv)


def load_case(path: str, name: str) -> dict:
    goldens = json.loads(Path(path).read_text(encoding="utf-8"))
    for case in goldens["cases"]:
        if case["name"] == name:
            if case["grouping"] != "joint" or len(case["prompts"]) != 1:
                raise SystemExit(f"case {name!r} is not a single joint sequence; pick another")
            return case["prompts"][0]
    raise SystemExit(f"no case named {name!r}")


def ensure_dump(args, prompt: dict) -> Path:
    directory = Path(args.dump_dir or f"/tmp/decider-block-{args.case}-{args.block}")
    directory.mkdir(parents=True, exist_ok=True)
    needed = [directory / f"x_post_{args.block}.f32", directory / f"mlp_{args.block}.f32"]
    if all(path.exists() for path in needed):
        print(f"reusing {directory}", flush=True)
        return directory
    print(f"running the reference for {args.case}, block {args.block} (a few minutes on CPU) ...", flush=True)
    command = [args.binary, "--ref", args.gguf,
               "--tokens", ",".join(map(str, prompt["input_ids"])),
               "--markers", ",".join(map(str, prompt["markers"])),
               "--dump-dir", str(directory), "--dump-block", str(args.block)]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"strata-decider --ref failed:\n{result.stderr[-2000:]}")
    for path in needed:
        if not path.exists():
            raise SystemExit(f"the reference wrote no {path}")
    return directory


def compare(name: str, got: np.ndarray, want: np.ndarray, tolerance: float) -> bool:
    cosine = float(got @ want / (np.linalg.norm(got) * np.linalg.norm(want)))
    relative = float(np.linalg.norm(got - want) / np.linalg.norm(want))
    worst = float(np.abs(got - want).max())
    ok = relative <= tolerance and cosine >= 0.999
    print(f"  {'ok  ' if ok else 'FAIL'} {name:10s} cos {cosine:.8f}  relL2 {relative:.6f}  max {worst:.3e}"
          f"  (tolerance {tolerance})", flush=True)
    return ok


def main(argv=None) -> int:
    args = parse_args(argv)
    prompt = load_case(args.goldens, args.case)
    directory = ensure_dump(args, prompt)
    out = directory / f"mlp_gpu_{args.block}.f32"
    xn_out = directory / f"xn_gpu_{args.block}.f32"
    command = [args.gpu_binary, "--gguf", args.gguf, "--block", str(args.block),
               "--in", str(directory / f"x_post_{args.block}.f32"), "--out", str(out), "--xn-out", str(xn_out)]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"strata-decider-gpu failed:\n{result.stderr[-2000:]}")

    def load(path: Path) -> np.ndarray:
        return np.fromfile(path, dtype=np.float32)

    ok = compare("xn_post", load(xn_out), load(directory / f"xn_post_{args.block}.f32"), NORM_TOLERANCE)
    ok = compare("mlp", load(out), load(directory / f"mlp_{args.block}.f32"), PROJECTION_TOLERANCE) and ok
    print(f"\ndecider_block_parity: block {args.block} of {args.case}: {'ok' if ok else 'FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
