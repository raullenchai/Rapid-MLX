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
@pytest.mark.parametrize("runtime_asset", [False, True])
def test_pull_completion_requires_verified_content(
    tmp_path, monkeypatch, mirror, tampered, subfolder, runtime_asset
):
    catalog_subfolder = subfolder
    if runtime_asset:
        subfolder = "runtime"
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
    if subfolder:
        metadata.siblings.append(
            SimpleNamespace(
                rfilename="unselected/model.safetensors",
                size=len(data),
                lfs={"sha256": hashlib.sha256(data).hexdigest()},
            )
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
    expected_allow = [f"{subfolder}/*"] if subfolder else None

    def transfer(*args, **kwargs):
        assert kwargs["allow_patterns"] == expected_allow
        return mirror

    monkeypatch.setattr("rapid_mlx._mirror.download_with_mirror_fallback", transfer)
    monkeypatch.setattr(cli, "_offline_hub_mode_active", lambda: False)
    monkeypatch.setattr(
        "rapid_mlx.model_aliases.resolve_subfolder", lambda *a: catalog_subfolder
    )
    completed = []
    monkeypatch.setattr(
        cli, "_sync_pulled_variant_marker", lambda *a: completed.append("marker")
    )
    monkeypatch.setattr(
        cli, "_print_pull_summary", lambda *a, **k: completed.append("summary")
    )
    args = SimpleNamespace(model="test/integrity", bits=None, format=None)
    pull_kwargs = {
        "revision_override": commit,
        "emit_lifecycle_event": False,
        "allow_patterns_override": expected_allow if runtime_asset else None,
    }
    completion = ["summary"] if runtime_asset else ["marker", "summary"]
    if tampered:
        with pytest.raises(ModelIntegrityError):
            cli._pull_repository(args, **pull_kwargs)
        assert completed == []
    else:
        cli._pull_repository(args, **pull_kwargs)
        assert completed == completion
        monkeypatch.setattr(cli, "_offline_hub_mode_active", lambda: True)
        monkeypatch.setattr(
            "huggingface_hub.HfApi.model_info",
            lambda *a, **k: pytest.fail("verified offline pull queried metadata"),
        )
        cli._pull_repository(args, **pull_kwargs)
        assert completed == completion * 2


@pytest.mark.parametrize("missing_revision", [False, True])
def test_unpinned_mirror_pull_recovers_metadata_and_reuses_receipt_offline(
    tmp_path, monkeypatch, missing_revision
):
    import httpx
    from huggingface_hub.errors import RevisionNotFoundError

    from tests.pull_integrity_helpers import snapshot_tree

    commit = "a" * 40
    root = tmp_path / "models--test--integrity"
    snapshot = snapshot_tree(root, commit)
    (root / "refs").mkdir()
    (root / "refs" / "main").write_text(commit)
    data = (snapshot / "model.safetensors").read_bytes()
    manifest = {
        "repo_id": "test/integrity",
        "revision": commit,
        "file_inventory": ["model.safetensors"],
        "files": [
            {
                "path": "model.safetensors",
                "size": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        ],
    }

    def unavailable(*args, **kwargs):
        if missing_revision:
            raise RevisionNotFoundError(
                "missing",
                response=httpx.Response(
                    404, request=httpx.Request("GET", "https://hub.invalid")
                ),
            )
        raise httpx.ConnectError("metadata unavailable")

    monkeypatch.setattr("huggingface_hub.HfApi.model_info", unavailable)
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path))
    monkeypatch.setattr(
        "rapid_mlx._mirror._fetch_json_with_status", lambda _url: (manifest, 200)
    )
    monkeypatch.setattr(
        "rapid_mlx._version_check.print_staleness_warning_if_any", lambda: None
    )
    monkeypatch.setattr("rapid_mlx.model_aliases.resolve_subfolder", lambda _repo: None)
    monkeypatch.setattr(
        cli, "_resolve_variant_allow_patterns", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(cli, "_try_mirror_prefetch", lambda *args, **kwargs: True)
    monkeypatch.setattr(cli, "_offline_hub_mode_active", lambda: False)
    monkeypatch.setattr(cli, "_sync_pulled_variant_marker", lambda *args: None)
    completed = []
    monkeypatch.setattr(
        cli, "_print_pull_summary", lambda *args, **kwargs: completed.append(True)
    )
    args = SimpleNamespace(model="test/integrity", bits=None, format=None)
    cli._pull_repository(args, emit_lifecycle_event=False)
    assert completed == [True]

    monkeypatch.setattr(cli, "_try_mirror_prefetch", lambda *args, **kwargs: False)
    monkeypatch.setattr(cli, "_offline_hub_mode_active", lambda: True)
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download", lambda *args, **kwargs: str(snapshot)
    )
    monkeypatch.setattr(
        "huggingface_hub.HfApi.model_info",
        lambda *args, **kwargs: pytest.fail(
            "offline verified snapshot queried metadata"
        ),
    )
    cli._pull_repository(args, emit_lifecycle_event=False)
    assert completed == [True, True]
