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
    path.write_bytes(b"x")


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
    assert "this local model folder has no usable MTP head" in capsys.readouterr().err
    assert args._speculative_config_explicit is False


def test_automatic_empty_local_head(model, capsys):
    _head(model)
    (model / "mtp" / "model.safetensors").write_bytes(b"")
    args = _args(model=str(model), _original_alias=ALIAS)
    cli._normalize_speculative_config_or_exit(args)
    assert args.spec_decode == "none"
    assert "no usable MTP head" in capsys.readouterr().err


def test_automatic_cached_sidecar_kept(model, capsys):
    args = _args(
        model=str(model),
        spec_decode="mtp",
        enable_mtp=True,
        mtp_sidecar="/cached/mtp",
        _speculative_config_explicit=False,
    )
    assert cli._drop_unavailable_mtp_head(args) is False
    assert args.spec_decode == "mtp"
    assert args.enable_mtp is True
    assert args.mtp_sidecar == "/cached/mtp"
    assert not capsys.readouterr().err


@pytest.mark.parametrize("served_folder", [False, True])
def test_shard_fallback_only_for_separate_sidecar(model, served_folder):
    (model / "a.safetensors").write_bytes(b"")
    first = model / "b.safetensors"
    first.write_bytes(b"x")
    (model / "c.safetensors").write_bytes(b"x")
    assert cli._local_mtp_head_file(model, served_folder=served_folder) == (
        None if served_folder else str(first)
    )


def test_empty_fixed_candidate_prevents_shard_fallback(model):
    (model / "mtp.safetensors").write_bytes(b"")
    (model / "model.safetensors").write_bytes(b"x")
    (model / "model-00001-of-00002.safetensors").write_bytes(b"x")
    assert cli._local_mtp_head_file(model, served_folder=False) is None


@pytest.mark.parametrize("sidecar_kind", ["folder", "none", "head", "empty_head"])
def test_drop_unavailable_head(model, capsys, sidecar_kind):
    if sidecar_kind in ("head", "empty_head"):
        _head(model)
        if sidecar_kind == "empty_head":
            (model / "mtp" / "model.safetensors").write_bytes(b"")
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
            else "this local model folder has no usable MTP head"
        ) in err


@pytest.mark.parametrize(
    ("sidecar_kind", "unavailable"),
    [
        ("empty_dir", True),
        ("empty_dir_weights", True),
        ("dir_weights", False),
        ("sharded_weights", False),
        ("empty_shard", True),
        ("file_weights", False),
        ("empty_file", True),
    ],
)
def test_explicit_separate_local_sidecar(model, capsys, sidecar_kind, unavailable):
    _head(model)
    sidecar = model / "separate_sidecar"
    if sidecar_kind in ("file_weights", "empty_file"):
        sidecar.write_bytes(b"" if sidecar_kind == "empty_file" else b"x")
    else:
        sidecar.mkdir()
        if sidecar_kind != "empty_dir":
            filename = {
                "sharded_weights": "model-00001-of-00002.safetensors",
                "empty_shard": "shard.safetensors",
            }.get(sidecar_kind, "model.safetensors")
            (sidecar / filename).write_bytes(
                b"" if sidecar_kind in ("empty_dir_weights", "empty_shard") else b"x"
            )
    args = _args(
        model=str(model),
        _original_alias=ALIAS,
        speculative_config=json.dumps({"method": "mtp", "model": str(sidecar)}),
    )
    cli._normalize_speculative_config_or_exit(args)
    assert args.spec_decode == "mtp"
    assert args.mtp_sidecar == str(sidecar)
    assert args._speculative_config_explicit is True
    assert cli._drop_unavailable_mtp_head(args) is unavailable
    err = capsys.readouterr().err
    if unavailable:
        assert args.spec_decode == "none"
        assert len(err.splitlines()) == 1
        assert err.startswith("MTP unavailable:")
        assert f"the MTP sidecar at {sidecar} has no usable MTP weights" in err
    else:
        assert args.spec_decode == "mtp"
        assert not err


@pytest.mark.parametrize(
    ("sidecar_kind", "unavailable"),
    [("dangling_symlink", True), ("missing_absolute", True), ("repo", False)],
)
def test_explicit_missing_sidecar_path(model, capsys, sidecar_kind, unavailable):
    _head(model)
    sidecar = model / "missing_sidecar"
    if sidecar_kind == "dangling_symlink":
        sidecar.symlink_to(model / "missing_target")
    elif sidecar_kind == "repo":
        sidecar = "mlx-community/Qwen3.5-9B-MTP-4bit"
    args = _args(
        model=str(model),
        spec_decode="mtp",
        mtp_backend=None,
        mtp_sidecar=str(sidecar),
        _speculative_config_explicit=True,
    )
    assert cli._drop_unavailable_mtp_head(args) is unavailable
    err = capsys.readouterr().err
    if unavailable:
        assert args.spec_decode == "none"
        assert len(err.splitlines()) == 1
        assert err.startswith("MTP unavailable:")
        assert f"the MTP sidecar at {sidecar} has no usable MTP weights" in err
    else:
        assert args.spec_decode == "mtp"
        assert args.mtp_sidecar == str(sidecar)
        assert not err


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
