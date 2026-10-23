import hashlib
from types import SimpleNamespace

import pytest

from rapid_mlx.model_integrity import ModelIntegrityError, verify_snapshot


@pytest.fixture
def snapshot(tmp_path, monkeypatch):
    commit = "a" * 40
    path = tmp_path / "snapshots" / commit
    path.mkdir(parents=True)
    data = b"tensor data"
    (path / "model.safetensors").write_bytes(data)
    (path / "config.json").write_bytes(b"{}")
    info = SimpleNamespace(
        sha=commit,
        siblings=[
            SimpleNamespace(
                rfilename="model.safetensors",
                size=len(data),
                lfs={"sha256": hashlib.sha256(data).hexdigest()},
            ),
            SimpleNamespace(
                rfilename="config.json",
                size=2,
                lfs=None,
                blob_id=hashlib.sha1(b"blob 2\0{}").hexdigest(),
            ),
        ],
    )
    calls = []

    def metadata(*args, **kwargs):
        calls.append(kwargs)
        return info

    monkeypatch.setattr("huggingface_hub.HfApi.model_info", metadata)
    return path, info, calls


def test_hashes_and_warm_receipt(snapshot):
    path, info, calls = snapshot
    verify_snapshot("test/integrity", path, revision=path.name)
    verify_snapshot("test/integrity", path, revision=path.name)
    assert len(calls) == 1
    assert calls[0]["revision"] == path.name
    (path / "model.safetensors").write_bytes(b"tamper data")
    with pytest.raises(ModelIntegrityError, match="mismatch"):
        verify_snapshot("test/integrity", path)


def test_commit_mismatch(snapshot):
    path, info, calls = snapshot
    with pytest.raises(ModelIntegrityError):
        verify_snapshot("test/integrity", path, revision="b" * 40)
    info.sha = "b" * 40
    with pytest.raises(ModelIntegrityError):
        verify_snapshot("test/integrity", path)


def test_selection_and_git_blob_hash(snapshot):
    path, info, calls = snapshot
    (path / "model.safetensors").unlink()
    verify_snapshot("test/integrity", path, allow_patterns=["*.json"])
    (path / "config.json").write_bytes(b"[]")
    with pytest.raises(ModelIntegrityError, match="hash mismatch"):
        verify_snapshot("test/integrity", path, allow_patterns=["*.json"])


def test_no_hash_fails_closed(snapshot):
    path, info, calls = snapshot
    info.siblings[0].lfs = {}
    with pytest.raises(ModelIntegrityError, match="Missing content hash"):
        verify_snapshot("test/integrity", path)


@pytest.mark.parametrize("mirror", [False, True])
def test_pull_never_marks_or_reports_corrupt_snapshot(snapshot, monkeypatch, mirror):
    from rapid_mlx import cli

    path, info, calls = snapshot
    (path / "model.safetensors").write_bytes(b"corruption!")
    monkeypatch.setattr(
        "rapid_mlx._version_check.print_staleness_warning_if_any", lambda: None
    )
    monkeypatch.setattr(cli, "_resolve_variant_allow_patterns", lambda *a, **kw: None)
    monkeypatch.setattr(cli, "_try_mirror_prefetch", lambda *a, **kw: mirror)
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda *a, **kw: str(path))
    monkeypatch.setattr("rapid_mlx.model_aliases.resolve_subfolder", lambda *a: None)
    monkeypatch.setattr(
        cli,
        "_sync_pulled_variant_marker",
        lambda *a: pytest.fail("corrupt pull marked complete"),
    )
    monkeypatch.setattr(
        cli,
        "_print_pull_summary",
        lambda *a, **kw: pytest.fail("corrupt pull reported complete"),
    )
    if mirror:
        # Recreate mirror's repository layout without weights or network access.
        root = path.parent.parent
        target = root / "models--test--integrity" / "snapshots" / path.name
        target.parent.mkdir(parents=True)
        path.rename(target)
        monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(root))
    with pytest.raises(ModelIntegrityError):
        cli._pull_repository(
            SimpleNamespace(model="test/integrity", bits=None, format=None),
            revision_override=path.name,
            emit_lifecycle_event=False,
        )


def test_packaged_manifest_requires_selected_weights_offline(snapshot, monkeypatch):
    import json

    from rapid_mlx import model_integrity

    path, info, calls = snapshot
    real_read = model_integrity.Path.read_text
    entry = {
        "revision": path.name,
        "lfs_files": {
            "model.safetensors": {
                "sha256": info.siblings[0].lfs["sha256"],
                "size": info.siblings[0].size,
            },
            "missing.safetensors": {"sha256": "0" * 64, "size": 2},
        },
    }

    def read(file, *args, **kwargs):
        if file.name == "model_catalog_lock.json":
            return json.dumps({"repositories": {"test/integrity": entry}})
        return real_read(file, *args, **kwargs)

    monkeypatch.setattr(model_integrity.Path, "read_text", read)
    with pytest.raises(ModelIntegrityError, match="Cannot verify.*missing"):
        verify_snapshot("test/integrity", path)
    assert calls == []
    del entry["lfs_files"]["missing.safetensors"]
    verify_snapshot("test/integrity", path)
    assert calls == []
