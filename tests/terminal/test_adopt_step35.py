"""Step-3.7-Flash files already on disk: `strata_cli/launch/adopt.py` writes the run config for them.

No model is needed and nothing is downloaded: the folder is made of empty files with the release's own names, and what
is checked is the config - the engine pointed at the first shard, the context, the expert tier sized out of what the
card has left, the folder's tokenizer, and the two ways it must refuse to write one.
"""

from pathlib import Path

from strata_cli.launch import adopt


def _folder(tmp_path: Path, with_tokenizer: bool = True) -> Path:
    quant = tmp_path / "IQ3_XXS"
    quant.mkdir()
    for i in (1, 2):
        (quant / f"Step-3.7-flash-IQ3_XXS-0000{i}-of-00002.gguf").write_bytes(b"")
    if with_tokenizer:
        tok = tmp_path / "tokenizer"
        tok.mkdir()
        (tok / "tokenizer.json").write_text("{}", encoding="utf-8")
    return quant


def _plan(tmp_path: Path, quant: Path, **over):
    kwargs = dict(family="step35", model="IQ3_XXS", model_dir=quant, data_dir=tmp_path, src=tmp_path, context=8192,
                  kv=None, model_name="step-3.7-flash-iq3_xxs", family_def={}, log=tmp_path / "run.log",
                  backend="hip", vram_gb=16.0, notes=[])
    kwargs.update(over)
    return adopt.plan(**kwargs)


def test_writes_a_config_for_local_files(tmp_path):
    quant = _folder(tmp_path)
    cfg, problems = _plan(tmp_path, quant)
    assert problems == []
    assert cfg["exe"] == "component:strata-engine"
    assert cfg["arch"] == "step35"
    assert cfg["args"][0] == "--model"
    assert cfg["args"][1] == str(quant / "Step-3.7-flash-IQ3_XXS-00001-of-00002.gguf")   # the first shard opens the set
    assert cfg["args"][cfg["args"].index("--max-context") + 1] == "8192"
    assert cfg["tokenizer"] == str(tmp_path / "tokenizer")
    assert Path(cfg["log"]) == tmp_path / "run.log"
    # 16 GB card, 0.315 GB of K/V cache at 8192 and 1 GB of working buffers -> about 13 GB of experts, 90% of it
    tier = int(cfg["args"][cfg["args"].index("--expert-cache") + 1])
    assert 12_000 <= tier <= 14_000, tier


def test_the_cache_grows_with_the_context_and_the_tier_shrinks(tmp_path):
    quant = _folder(tmp_path)
    small, _ = _plan(tmp_path, quant)
    big, _ = _plan(tmp_path, quant, context=32768)
    assert int(big["args"][big["args"].index("--max-context") + 1]) == 32768
    assert int(big["args"][big["args"].index("--expert-cache") + 1]) < \
           int(small["args"][small["args"].index("--expert-cache") + 1])


def test_pictures_are_on_when_the_mmproj_is_beside_the_model(tmp_path):
    quant = _folder(tmp_path)
    cfg, problems = _plan(tmp_path, quant)
    assert problems == []
    assert cfg.get("vision") if "vision" in cfg else "vision" not in cfg   # no mmproj here: off, and no key
    assert "mmproj" not in cfg

    (tmp_path / "mmproj-step3.7-flash-f16.gguf").write_bytes(b"")
    cfg, problems = _plan(tmp_path, quant, notes=[])
    assert problems == []
    assert cfg["vision"] is True
    assert cfg["mmproj"] == str(tmp_path / "mmproj-step3.7-flash-f16.gguf")


def test_a_missing_mmproj_is_said_not_silent(tmp_path):
    quant = _folder(tmp_path)
    notes = []
    cfg, _ = _plan(tmp_path, quant, notes=notes)
    assert "vision" not in cfg
    assert any("pictures stay off" in n for n in notes)


def test_a_missing_shard_is_named(tmp_path):
    quant = _folder(tmp_path)
    (quant / "Step-3.7-flash-IQ3_XXS-00002-of-00002.gguf").unlink()
    cfg, problems = _plan(tmp_path, quant)
    assert cfg is None
    assert any("shard" in p for p in problems)


def test_a_missing_tokenizer_is_named(tmp_path):
    quant = _folder(tmp_path, with_tokenizer=False)
    cfg, problems = _plan(tmp_path, quant)
    assert cfg is None
    assert any("tokenizer" in p for p in problems)


def test_the_kv_figures_are_the_measured_ones():
    # docs/STEP37.md: 0.315 GB at 8192, 0.938 at 32768, 3.431 at 131072 (the last includes the page tables)
    assert abs(adopt.step35_kv_gb(8192) - 0.315) < 0.002
    assert abs(adopt.step35_kv_gb(32768) - 0.938) < 0.002
    assert abs(adopt.step35_kv_gb(131072) - 3.431) < 0.005
