# SPDX-License-Identifier: Apache-2.0
"""The auxiliary VAD never hands unverified data to its native loader."""

import hashlib
import json
from pathlib import Path

import pytest

from rapid_mlx import model_downloads, model_integrity
from rapid_mlx.audio import vad_artifacts as vad


@pytest.fixture
def artifacts(tmp_path, monkeypatch):
    hub = tmp_path / "hub"
    snapshot = (
        hub / "models--mlx-community--silero-vad" / "snapshots" / vad.VAD_REVISION
    )
    payloads = {
        "config.json": b'{"model_type":"silero_vad"}',
        "model.safetensors": b"test tensor payload",
        "README.md": b"test model card",
        ".gitattributes": b"test lfs configuration",
    }
    record = {"revision": vad.VAD_REVISION, "git_files": {}, "lfs_files": {}}
    for name, data in payloads.items():
        if name.endswith(".safetensors"):
            record["lfs_files"][name] = {
                "sha256": hashlib.sha256(data).hexdigest(),
                "size": len(data),
            }
        else:
            record["git_files"][name] = {
                "blob_id": hashlib.sha1(
                    f"blob {len(data)}\0".encode() + data
                ).hexdigest(),
                "size": len(data),
            }
    (tmp_path / "model_catalog_lock.json").write_text(
        json.dumps({"repositories": {vad.VAD_REPO: record}})
    )
    monkeypatch.setattr(
        model_integrity, "__file__", str(tmp_path / "model_integrity.py")
    )
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(hub))
    monkeypatch.setattr(
        "huggingface_hub.HfApi.model_info",
        lambda *a, **kw: pytest.fail(
            "packaged VAD verification must not query Hub metadata"
        ),
    )
    monkeypatch.delenv(model_downloads.ENV_VAR, raising=False)
    monkeypatch.setattr(model_downloads, "_cli_disabled", False)

    def materialize(names=("config.json", "model.safetensors")):
        snapshot.mkdir(parents=True, exist_ok=True)
        for name in names:
            (snapshot / name).write_bytes(payloads[name])
        return snapshot

    return snapshot, materialize, payloads


def test_warm_snapshot_verifies_offline_without_ref_or_receipt(artifacts, monkeypatch):
    snapshot, materialize, _ = artifacts
    materialize()
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda *a, **kw: pytest.fail("warm cache must not download"),
    )
    monkeypatch.setenv(model_downloads.ENV_VAR, "1")
    assert vad.resolve_vad_snapshot() == snapshot
    assert vad.resolve_vad_snapshot() == snapshot
    assert not (snapshot.parent.parent / "refs").exists()


@pytest.mark.parametrize("name", ["config.json", "model.safetensors"])
@pytest.mark.parametrize("previously_verified", [False, True])
def test_corruption_and_repeated_warm_retry_never_return_snapshot(
    artifacts, monkeypatch, name, previously_verified
):
    snapshot, materialize, payloads = artifacts
    materialize()
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda *a, **kw: pytest.fail(
            "complete corrupt cache must not trigger a download"
        ),
    )
    if previously_verified:
        vad.resolve_vad_snapshot()
    (snapshot / name).write_bytes(b"!" * len(payloads[name]))
    for _ in range(2):
        with pytest.raises(model_integrity.ModelIntegrityError, match="hash mismatch"):
            vad.resolve_vad_snapshot()
    assert (snapshot / name).read_bytes() == b"!" * len(payloads[name])


@pytest.mark.parametrize("extra", ["extra.safetensors", "extra.npz"])
def test_native_weight_glob_cannot_consume_unverified_extra_files(artifacts, extra):
    snapshot, materialize, _ = artifacts
    materialize()
    vad.resolve_vad_snapshot()
    (snapshot / extra).write_bytes(b"unexpected tensor data")
    with pytest.raises(model_integrity.ModelIntegrityError, match="inventory mismatch"):
        vad.resolve_vad_snapshot()


@pytest.mark.parametrize("partial", [False, True])
def test_cold_or_partial_snapshot_download_is_exact_and_verified(
    artifacts, monkeypatch, partial
):
    snapshot, materialize, _ = artifacts
    if partial:
        materialize(["config.json"])
    calls = []

    def download(repo, **kwargs):
        calls.append((repo, kwargs))
        return str(materialize())

    monkeypatch.setattr("huggingface_hub.snapshot_download", download)
    assert vad.resolve_vad_snapshot() == snapshot
    assert calls == [
        (
            vad.VAD_REPO,
            {
                "revision": vad.VAD_REVISION,
                "allow_patterns": ["config.json", "model.safetensors"],
            },
        )
    ]


def test_disabled_cold_download_uses_only_local_hub_resolution(artifacts, monkeypatch):
    from huggingface_hub.errors import LocalEntryNotFoundError

    monkeypatch.setenv(model_downloads.ENV_VAR, "1")

    def local_only(repo, **kwargs):
        assert kwargs["local_files_only"] is True
        assert kwargs["revision"] == vad.VAD_REVISION
        raise LocalEntryNotFoundError("not cached")

    monkeypatch.setattr("huggingface_hub.snapshot_download", local_only)
    with pytest.raises(model_downloads.ModelDownloadsDisabledError):
        vad.resolve_vad_snapshot()


def test_downloaded_snapshot_must_match_exact_pin(artifacts, monkeypatch):
    snapshot, _, _ = artifacts
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda *a, **kw: str(snapshot.with_name("b" * 40)),
    )
    with pytest.raises(
        model_integrity.ModelIntegrityError, match="requested immutable snapshot"
    ):
        vad.resolve_vad_snapshot()


def test_full_explicit_pull_can_verify_repository_documentation(artifacts):
    snapshot, materialize, payloads = artifacts
    materialize(payloads)
    model_integrity.verify_snapshot(
        vad.VAD_REPO, snapshot, revision=vad.VAD_REVISION, local_files_only=True
    )
    assert vad.resolve_vad_snapshot() == snapshot


def test_packaged_manifest_records_full_exact_repository_inventory():
    manifest = json.loads(
        Path(model_integrity.__file__).with_name("model_catalog_lock.json").read_text()
    )
    assert set(manifest["repositories"]) == {vad.VAD_REPO}
    record = manifest["repositories"][vad.VAD_REPO]
    assert record["revision"] == vad.VAD_REVISION
    assert set(record["git_files"]) == {".gitattributes", "README.md", "config.json"}
    assert record["lfs_files"] == {
        "model.safetensors": {
            "size": 2179454,
            "sha256": "185e0bc3ee2c48ce425a37209fe917a1aca22ab6b85799430dd1b4894087a8b8",
        }
    }
    assert record["git_files"]["config.json"] == {
        "size": 549,
        "blob_id": "af83a8c743562a75af3edad522a6c94d1f885115",
    }


def test_cold_transfer_corruption_is_rejected_before_return(artifacts, monkeypatch):
    snapshot, materialize, payloads = artifacts

    def corrupt_transfer(*args, **kwargs):
        materialize()
        (snapshot / "model.safetensors").write_bytes(
            b"!" * len(payloads["model.safetensors"])
        )
        return str(snapshot)

    monkeypatch.setattr("huggingface_hub.snapshot_download", corrupt_transfer)
    with pytest.raises(model_integrity.ModelIntegrityError, match="hash mismatch"):
        vad.resolve_vad_snapshot()
