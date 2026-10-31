# SPDX-License-Identifier: Apache-2.0
"""Local model folders retain catalog MTP behavior and fail closed."""

import json
from types import SimpleNamespace

import pytest

import rapid_mlx.cli as cli

ALIAS = "qwen3.8-27b-4bit"
REPO = "rapid-mlx/Qwen3.8-27B-4bit-MTP-MLX"
CONFIG = {"model_type": "qwen3_5", "text_config": {"mtp_num_hidden_layers": 1}}


def _args(**overrides):
    values = dict(
        model=None,
        speculative_config=None,
        enable_ddtree=False,
        enable_dflash=False,
        spec_decode="none",
        dflash_drafter_path="",
        enable_mtp=False,
        mtp_num_draft_tokens=1,
        mtp_optimistic=False,
        mtp_sidecar=None,
        mtp_max_k=None,
        mtp_disable_auto_k=False,
        force_spec_decode=False,
        suffix_decoding=False,
        suffix_max_draft=None,
        suffix_max_suffix_len=None,
        suffix_min_confidence=None,
        suffix_min_draft_len=None,
        no_spec_decode=False,
        mllm=False,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.fixture
def model(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text(json.dumps(CONFIG))
    gather = cli._gather_kv_cache_dtype_inputs
    monkeypatch.setattr(
        cli,
        "_gather_kv_cache_dtype_inputs",
        lambda name: (CONFIG, None) if name == REPO else gather(name),
    )
    monkeypatch.setattr(cli, "_cached_mtp_sidecar", lambda _: str(tmp_path))
    return tmp_path


def _head(model):
    path = model / "mtp" / "model.safetensors"
    path.parent.mkdir()
    path.touch()


def test_automatic_local_and_repo_parity(model):
    _head(model)
    local = _args(model=str(model), _original_alias=ALIAS)
    repo = _args(model=REPO, _original_alias=ALIAS)
    for args in (local, repo):
        cli._normalize_speculative_config_or_exit(args)
        assert args.spec_decode == "mtp"
        assert args.mtp_sidecar == str(model)
        assert args._speculative_config_explicit is False


def test_explicit_local_sidecar(model):
    _head(model)
    args = _args(
        model=str(model), _original_alias=ALIAS, speculative_config='{"method":"mtp"}'
    )
    cli._normalize_speculative_config_or_exit(args)
    assert args.mtp_sidecar == str(model)
    assert args._speculative_config_explicit is True


def test_automatic_missing_local_head(model, capsys):
    args = _args(model=str(model), _original_alias=ALIAS)
    cli._normalize_speculative_config_or_exit(args)
    assert args.spec_decode == "none"
    assert "this local model folder has no MTP head" in capsys.readouterr().err
    assert args._speculative_config_explicit is False


@pytest.mark.parametrize("sidecar_kind", ["folder", "none", "head"])
def test_drop_unavailable_head(model, capsys, sidecar_kind):
    if sidecar_kind == "head":
        _head(model)
    args = _args(
        model=str(model),
        spec_decode="mtp",
        mtp_backend=None,
        mtp_sidecar=None if sidecar_kind == "none" else str(model),
        _speculative_config_explicit=True,
    )
    assert cli._drop_unavailable_mtp_head(args) is (sidecar_kind != "head")
    err = capsys.readouterr().err
    if sidecar_kind == "head":
        assert args.spec_decode == "mtp"
        assert not err
    else:
        assert args.spec_decode == "none"
        assert len(err.splitlines()) == 1
        assert err.startswith("MTP unavailable:")
        assert (
            "no MTP sidecar was given"
            if sidecar_kind == "none"
            else "this local model folder has no MTP head"
        ) in err


def test_missing_sidecar_copy(model, monkeypatch, capsys):
    alias = "qwen3.5-9b-4bit"
    sidecar, _ = cli._alias_mtp_declaration(alias)
    assert sidecar
    monkeypatch.setattr(cli, "_cached_mtp_sidecar", lambda _: None)
    args = _args(model=str(model), _original_alias=alias)
    cli._normalize_speculative_config_or_exit(args)
    err = capsys.readouterr().err
    assert f"rapid-mlx pull {sidecar}" in err
    assert f"rapid-mlx pull {model}" not in err
