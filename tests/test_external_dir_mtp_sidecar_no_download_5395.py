# SPDX-License-Identifier: Apache-2.0
"""External model folders supply their embedded MTP heads (#5395)."""

import fnmatch
from types import SimpleNamespace

import huggingface_hub

from rapid_mlx.cli import (
    _local_mtp_sidecar_for_served_model,
    _normalize_speculative_config_or_exit,
)
from rapid_mlx.spec_decode.mtp.qwen3_5_inject import _resolve_sidecar_file

ALIAS = "qwen3.8-27b-4bit"
REPO = "rapid-mlx/Qwen3.8-27B-4bit-MTP-MLX"


def _args(**overrides):
    data = {
        "model": None,
        "speculative_config": None,
        "enable_ddtree": False,
        "enable_dflash": False,
        "spec_decode": "none",
        "dflash_drafter_path": "",
        "mtp_num_draft_tokens": 1,
        "mtp_optimistic": False,
        "mtp_sidecar": None,
        "mtp_max_k": None,
        "mtp_disable_auto_k": False,
        "force_spec_decode": False,
        "suffix_decoding": False,
        "suffix_max_draft": None,
        "suffix_max_suffix_len": None,
        "suffix_min_confidence": None,
        "suffix_min_draft_len": None,
    }
    data.update(overrides)
    return SimpleNamespace(**data)


def test_local_sidecar_identity(tmp_path):
    args = _args(model=str(tmp_path), _original_alias=ALIAS)
    assert _local_mtp_sidecar_for_served_model(args, REPO) == str(tmp_path)
    assert _local_mtp_sidecar_for_served_model(args, f"  {REPO.swapcase()}  ") == str(
        tmp_path
    )
    other = "mlx-community/Qwen3.6-27B-MTP-4bit"
    assert _local_mtp_sidecar_for_served_model(args, other) == other
    assert _local_mtp_sidecar_for_served_model(args, None) is None
    args.model = REPO
    assert _local_mtp_sidecar_for_served_model(args, REPO) == REPO
    args.model = str(tmp_path)
    args._original_alias = REPO
    assert _local_mtp_sidecar_for_served_model(args, REPO) == str(tmp_path)


def test_local_sidecar_identity_when_profile_resolution_fails(tmp_path, monkeypatch):
    import rapid_mlx.model_aliases as model_aliases

    def fail(_original):
        raise RuntimeError("profile resolution failed")

    monkeypatch.setattr(model_aliases, "resolve_profile", fail)
    args = _args(model=str(tmp_path), _original_alias=REPO)
    assert _local_mtp_sidecar_for_served_model(args, REPO) == str(tmp_path)
    args._original_alias = ALIAS
    assert _local_mtp_sidecar_for_served_model(args, REPO) == REPO


def test_normalize_uses_external_model_folder(tmp_path):
    args = _args(
        model=str(tmp_path),
        _original_alias=ALIAS,
        speculative_config=(
            '{"method":"mtp","model":"rapid-mlx/Qwen3.8-27B-4bit-MTP-MLX",'
            '"num_speculative_tokens":3}'
        ),
    )
    _normalize_speculative_config_or_exit(args)
    assert args.mtp_sidecar == str(tmp_path)


def test_local_injector_resolution_never_contacts_hub(tmp_path, monkeypatch):
    import rapid_mlx.model_downloads as downloads

    head = tmp_path / "mtp" / "model.safetensors"
    head.parent.mkdir()
    head.touch()

    def fail(**kwargs):
        raise AssertionError("hub contacted")

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(downloads, "snapshot_download", fail)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fail)
    assert _resolve_sidecar_file(str(tmp_path)) == head


def test_repo_resolution_limits_downloads(tmp_path, monkeypatch):
    import rapid_mlx.model_downloads as downloads

    head = tmp_path / "mtp" / "model.safetensors"
    head.parent.mkdir()
    head.touch()
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        return str(tmp_path)

    monkeypatch.setattr(downloads, "snapshot_download", fake_download)
    assert _resolve_sidecar_file(REPO) == head
    assert len(calls) == 1
    assert calls[0]["repo_id"] == REPO
    patterns = calls[0]["allow_patterns"]
    assert patterns == [
        "mtp.safetensors",
        "model-mtp.safetensors",
        "mtp/*",
        "mtplx_runtime.json",
    ]
    for target in ("model.safetensors", "model-00001-of-00003.safetensors"):
        assert not any(fnmatch.fnmatch(target, p) for p in patterns)


def test_repo_resolution_falls_back_to_standalone_sidecar(tmp_path, monkeypatch):
    import rapid_mlx.model_downloads as downloads

    first = tmp_path / "first"
    first.mkdir()
    second = tmp_path / "second"
    second.mkdir()
    head = second / "model.safetensors"
    head.touch()
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        return str((first, second)[len(calls) - 1])

    monkeypatch.setattr(downloads, "snapshot_download", fake_download)
    assert _resolve_sidecar_file(REPO) == head
    assert len(calls) == 2
    assert calls[1]["repo_id"] == REPO
    assert calls[1]["allow_patterns"] == ["model.safetensors", "mtplx_runtime.json"]
