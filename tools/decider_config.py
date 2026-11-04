"""tools/decider_config.py - write the decider.json that /v1/systemone reads, from the model's own GGUF.

The GGUF carries the whole protocol in its metadata (`pplx_decider.*`): protocol version, template, marker,
temperature, the 255 codes and their token ids, the escape list, the pixel cap.  The server needs them next to
the tokenizer, in the pack's ``tokenizer/`` folder, because that is the folder a model install copies around.

    python tools/decider_config.py --gguf <first shard>.gguf --out <pack>/tokenizer

It refuses a file that is not this protocol (v1, another marker, a causal attention flag, a missing key) rather
than writing a decider.json the server would then serve with wrong answers.

    python tools/decider_config.py --gguf ... --stdout     # print, write nothing
    python tools/decider_config.py --check <dir or file>   # validate an existing decider.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gguf_reader import GGUFFile  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serve.systemone import (MARKER, MAX_OPTIONS, PROTOCOL_TEMPLATE, PROTOCOL_VERSION,  # noqa: E402
                             ConfigError, load_config)

MAX_LEVELS_NOTE = "the hosted API caps a score at 10 levels; the head has 255 rows"


def _number(meta: dict, key: str, *, default=None):
    value = meta.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"missing or not a number: {key}")
    if not math.isfinite(float(value)):
        raise ValueError(f"not finite: {key}")
    return value


def config_from_metadata(meta: dict, source: str = "") -> dict:
    """The decider.json body from a GGUF's metadata.  ValueError names the key that disqualifies the file."""
    arch = meta.get("general.architecture")
    if arch != "qwen35":
        raise ValueError(f"general.architecture is {arch!r}, this tool reads 'qwen35' decision checkpoints")
    if meta.get("qwen35.attention.causal") is not False:
        raise ValueError("qwen35.attention.causal is not false: this checkpoint must attend non-causally")
    version = meta.get("pplx_decider.protocol_version", meta.get("pplx_decider.version"))
    if version != PROTOCOL_VERSION:
        raise ValueError(f"pplx_decider.protocol_version is {version!r}, the server serves {PROTOCOL_VERSION}")
    template = meta.get("pplx_decider.template")
    if template != PROTOCOL_TEMPLATE:
        raise ValueError(f"pplx_decider.template is {template!r}, the server serves {PROTOCOL_TEMPLATE!r}")
    marker = meta.get("pplx_decider.marker")
    if marker != MARKER:
        raise ValueError(f"pplx_decider.marker is {marker!r}, the server serves {MARKER!r}")
    temperature = float(_number(meta, "pplx_decider.temperature"))
    if temperature <= 0:
        raise ValueError("pplx_decider.temperature must be positive")
    codes = meta.get("pplx_decider.codes")
    token_ids = meta.get("pplx_decider.token_ids")
    if not isinstance(codes, list) or not isinstance(token_ids, list):
        raise ValueError("pplx_decider.codes and pplx_decider.token_ids must be arrays")
    if len(codes) != len(token_ids) or not 1 <= len(codes) <= MAX_OPTIONS:
        raise ValueError(f"codes and token_ids must have the same length, 1 to {MAX_OPTIONS}")
    if any(not isinstance(code, str) or not code for code in codes):
        raise ValueError("every pplx_decider.codes entry must be a nonempty string")
    if any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in token_ids):
        raise ValueError("every pplx_decider.token_ids entry must be a nonnegative integer")
    escapes = meta.get("pplx_decider.escape_tokens") or []
    if not isinstance(escapes, list) or any(not isinstance(token, str) for token in escapes):
        raise ValueError("pplx_decider.escape_tokens must be an array of strings")
    max_pixels = meta.get("pplx_decider.max_pixels", 1_638_400)
    if isinstance(max_pixels, bool) or not isinstance(max_pixels, int) or max_pixels <= 0:
        raise ValueError("pplx_decider.max_pixels must be a positive integer")
    context = meta.get("qwen35.context_length", 262_144)
    if isinstance(context, bool) or not isinstance(context, int) or context <= 0:
        raise ValueError("qwen35.context_length must be a positive integer")
    return {
        "format": 1,
        "protocol_version": PROTOCOL_VERSION,
        "template": PROTOCOL_TEMPLATE,
        "marker": MARKER,
        "temperature": temperature,
        "codes": list(codes),
        "token_ids": list(token_ids),
        "escape_tokens": list(escapes),
        "max_pixels": int(max_pixels),
        "max_length": int(context),
        "attention_mode": "noncausal_full_attention",
        "pooling": "last",
        "base_model": meta.get("pplx_decider.base_model") or "Qwen/Qwen3.5-4B",
        "levels_note": MAX_LEVELS_NOTE,
        **({"source": source} if source else {}),
    }


def _out_path(out: Path) -> Path:
    if out.exists() and out.is_dir():
        return out / "decider.json"
    if out.suffix.lower() == ".json":
        return out
    return out / "decider.json"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gguf", help="the first shard (the header carries the protocol)")
    parser.add_argument("--out", help="the pack's tokenizer/ folder, or a .json file")
    parser.add_argument("--stdout", action="store_true", help="print the JSON, write nothing")
    parser.add_argument("--check", metavar="PATH", help="validate an existing decider.json or the folder holding it")
    args = parser.parse_args(argv)
    if args.check:
        path = Path(args.check)
        try:
            data = load_config(path.parent if path.is_file() else path)
        except (ConfigError, OSError) as exc:
            print(f"decider_config: {exc}", file=sys.stderr)
            return 1
        if data is None:
            print(f"decider_config: no decider.json in {path}", file=sys.stderr)
            return 1
        print(f"decider_config: ok - {len(data['codes'])} codes, temperature {data['temperature']:.6f}, "
              f"max_length {data['max_length']}")
        return 0
    if not args.gguf:
        parser.error("--gguf is needed (or --check)")
    try:
        config = config_from_metadata(GGUFFile(args.gguf).metadata, source=str(Path(args.gguf).resolve()))
    except (ValueError, OSError) as exc:
        print(f"decider_config: {exc}", file=sys.stderr)
        return 2
    text = json.dumps(config, ensure_ascii=False, indent=1) + "\n"
    if args.stdout:
        sys.stdout.write(text)
        return 0
    if not args.out:
        parser.error("--out is needed (or --stdout)")
    out = _out_path(Path(args.out))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8", newline="\n")
    # Read it back through the same function the server uses: a file this tool writes must load there.
    try:
        load_config(out.parent)
    except ConfigError as exc:
        out.unlink(missing_ok=True)
        print(f"decider_config: wrote {out} but it does not load: {exc}", file=sys.stderr)
        return 2
    print(f"decider_config: {out} - {len(config['codes'])} codes, temperature {config['temperature']:.6f}, "
          f"max_length {config['max_length']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
