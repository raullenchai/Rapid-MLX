# SPDX-License-Identifier: Apache-2.0
"""Explicit pulls verify selected bytes before publishing completion metadata."""

import hashlib
from types import SimpleNamespace

import pytest

from rapid_mlx import cli
from rapid_mlx.model_integrity import ModelIntegrityError


@pytest.mark.parametrize("mirror", [False, True])
@pytest.mark.parametrize("tampered", [False, True])
@pytest.mark.parametrize("subfolder", [None, "4bit"])
def test_pull_completion_requires_verified_content(
    tmp_path, monkeypatch, mirror, tampered, subfolder
):
    commit = "a" * 40
    root = tmp_path / "models--test--integrity"
    snapshot = root / "snapshots" / commit
    selected = snapshot / subfolder if subfolder else snapshot
    selected.mkdir(parents=True)
    data = b"tensor data"
    (selected / "model.safetensors").write_bytes(b"bad content" if tampered else data)
    name = f"{subfolder}/model.safetensors" if subfolder else "model.safetensors"
    metadata = SimpleNamespace(
        sha=commit,
        siblings=[
            SimpleNamespace(
                rfilename=name,
                size=len(data),
                lfs={"sha256": hashlib.sha256(data).hexdigest()},
            )
        ],
    )
    monkeypatch.setattr("huggingface_hub.HfApi.model_info", lambda *a, **k: metadata)
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download", lambda *a, **k: str(snapshot)
    )
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path))
    monkeypatch.setattr(
        "rapid_mlx._version_check.print_staleness_warning_if_any", lambda: None
    )
    monkeypatch.setattr(cli, "_resolve_variant_allow_patterns", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_try_mirror_prefetch", lambda *a, **k: mirror)
    monkeypatch.setattr(cli, "_offline_hub_mode_active", lambda: False)
    monkeypatch.setattr(
        "rapid_mlx.model_aliases.resolve_subfolder", lambda *a: subfolder
    )
    completed = []
    monkeypatch.setattr(
        cli, "_sync_pulled_variant_marker", lambda *a: completed.append("marker")
    )
    monkeypatch.setattr(
        cli, "_print_pull_summary", lambda *a, **k: completed.append("summary")
    )
    args = SimpleNamespace(model="test/integrity", bits=None, format=None)
    if tampered:
        with pytest.raises(ModelIntegrityError):
            cli._pull_repository(
                args, revision_override=commit, emit_lifecycle_event=False
            )
        assert completed == []
    else:
        cli._pull_repository(args, revision_override=commit, emit_lifecycle_event=False)
        assert completed == ["marker", "summary"]
        monkeypatch.setattr(cli, "_offline_hub_mode_active", lambda: True)
        monkeypatch.setattr(
            "huggingface_hub.HfApi.model_info",
            lambda *a, **k: pytest.fail("verified offline pull queried metadata"),
        )
        cli._pull_repository(args, revision_override=commit, emit_lifecycle_event=False)
        assert completed == ["marker", "summary", "marker", "summary"]
