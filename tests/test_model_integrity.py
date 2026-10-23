# SPDX-License-Identifier: Apache-2.0
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


def test_packaged_manifest_requires_selected_weights_offline(
    snapshot, monkeypatch, tmp_path
):
    import json

    from rapid_mlx import model_integrity

    path, info, calls = snapshot
    module = tmp_path / "module"
    module.mkdir()
    monkeypatch.setattr(model_integrity, "__file__", str(module / "model_integrity.py"))
    lock = module / "model_catalog_lock.json"
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

    lock.write_text(json.dumps({"repositories": {"test/integrity": entry}}))
    with pytest.raises(ModelIntegrityError, match="Cannot verify.*missing"):
        verify_snapshot("test/integrity", path)
    assert calls == []
    del entry["lfs_files"]["missing.safetensors"]
    lock.write_text(json.dumps({"repositories": {"test/integrity": entry}}))
    verify_snapshot("test/integrity", path)
    assert calls == []


def test_ignore_and_exact_filename_selection(snapshot):
    path, info, calls = snapshot
    (path / "model.safetensors").unlink()
    verify_snapshot("test/integrity", path, ignore_patterns="*.safetensors")
    verify_snapshot("test/integrity", path, filenames=["config.json"])


def test_local_dir_verification_uses_resolved_commit(snapshot, tmp_path):
    path, info, calls = snapshot
    destination = tmp_path / "local-model"
    path.rename(destination)
    verify_snapshot("test/integrity", destination, resolved_revision=info.sha)
    (destination / "config.json").write_bytes(b"[]")
    with pytest.raises(ModelIntegrityError):
        verify_snapshot("test/integrity", destination, resolved_revision=info.sha)


def test_directory_patterns_and_empty_allow_list(snapshot):
    path, info, calls = snapshot
    (path / "folder").mkdir()
    (path / "config.json").rename(path / "folder" / "config.json")
    info.siblings[1].rfilename = "folder/config.json"
    (path / "model.safetensors").unlink()
    verify_snapshot("test/integrity", path, allow_patterns=["folder/"])
    (path / "folder" / "config.json").write_bytes(b"[]")
    with pytest.raises(ModelIntegrityError):
        verify_snapshot("test/integrity", path, allow_patterns=["folder/"])
    verify_snapshot("test/integrity", path, allow_patterns=[])


def test_offline_unknown_snapshot_never_queries_metadata(snapshot):
    path, info, calls = snapshot
    with pytest.raises(ModelIntegrityError, match="Offline integrity"):
        verify_snapshot("test/integrity", path, local_files_only=True)
    assert calls == []
    verify_snapshot("test/integrity", path)
    verify_snapshot("test/integrity", path, local_files_only=True)
    assert len(calls) == 1


def test_missing_explicit_pin_uses_validated_mirror_manifest(snapshot, monkeypatch):
    from huggingface_hub.errors import RevisionNotFoundError

    from rapid_mlx import _mirror

    path, info, calls = snapshot

    def missing(*args, **kwargs):
        import httpx

        raise RevisionNotFoundError(
            "removed pin",
            response=httpx.Response(
                404, request=httpx.Request("GET", "https://hub.invalid")
            ),
        )

    monkeypatch.setattr("huggingface_hub.HfApi.model_info", missing)
    manifest = {
        "repo_id": "test/integrity",
        "revision": path.name,
        "file_inventory": ["model.safetensors", "config.json"],
        "files": [
            {
                "path": "model.safetensors",
                "size": info.siblings[0].size,
                "sha256": info.siblings[0].lfs["sha256"],
            },
            {"path": "config.json", "size": 2, "git_oid": info.siblings[1].blob_id},
        ],
    }
    requests = []
    monkeypatch.setattr(_mirror, "_mirror_base", lambda: "https://mirror.invalid")
    monkeypatch.setattr(
        _mirror,
        "_fetch_json_with_status",
        lambda url: requests.append(url) or (manifest, 200),
    )
    verify_snapshot(
        "test/integrity", path, revision=path.name, allow_patterns="*.safetensors"
    )
    assert requests == [
        f"https://mirror.invalid/test/integrity/.rapid-mlx/revisions/{path.name}.json"
    ]
    # The untouched selected receipt remains usable offline without metadata.
    verify_snapshot(
        "test/integrity",
        path,
        revision=path.name,
        allow_patterns="*.safetensors",
        local_files_only=True,
    )
    (path / "model.safetensors").write_bytes(b"corruption!")
    with pytest.raises(ModelIntegrityError):
        verify_snapshot(
            "test/integrity", path, revision=path.name, allow_patterns="*.safetensors"
        )
    count = len(requests)
    with pytest.raises(ModelIntegrityError, match="Offline integrity"):
        verify_snapshot(
            "test/integrity", path, revision=path.name, local_files_only=True
        )
    assert len(requests) == count
    # Missing inventory or wrong revision cannot authorize recovered content.
    manifest["file_inventory"].append("missing")
    with pytest.raises(ModelIntegrityError, match="no valid exact-revision"):
        verify_snapshot("test/integrity", path, revision=path.name)


@pytest.mark.parametrize("name", [None, "../outside", "/absolute", "folder\\file"])
def test_invalid_manifest_paths_are_refused(snapshot, name):
    path, info, _ = snapshot
    info.siblings[0].rfilename = name
    with pytest.raises(ModelIntegrityError, match="manifest filename"):
        verify_snapshot("test/integrity", path)


def test_missing_requested_file_and_empty_exact_selection(snapshot):
    path, _, _ = snapshot
    with pytest.raises(ModelIntegrityError, match="absent"):
        verify_snapshot("test/integrity", path, filenames=["missing"])
    with pytest.raises(ModelIntegrityError, match="no selected files"):
        verify_snapshot("test/integrity", path, filenames=[])


def test_size_mismatch_preserves_resume_files(snapshot):
    path, _, _ = snapshot
    partial = path / "weights.incomplete"
    partial.write_bytes(b"partial")
    (path / "config.json").write_bytes(b"truncated")
    with pytest.raises(ModelIntegrityError, match="size mismatch"):
        verify_snapshot("test/integrity", path)
    assert partial.read_bytes() == b"partial"


def test_mismatched_receipt_is_reverified(snapshot):
    import json

    path, _, calls = snapshot
    verify_snapshot("test/integrity", path)
    receipt = next((path.parent.parent / ".rapid-mlx-integrity").glob("*.json"))
    data = json.loads(receipt.read_text())
    data["repo"] = "other/repo"
    receipt.write_text(json.dumps(data))
    verify_snapshot("test/integrity", path)
    assert len(calls) == 2


def test_wrong_packaged_revision_is_refused(snapshot, tmp_path, monkeypatch):
    import json

    from rapid_mlx import model_integrity

    path, _, calls = snapshot
    monkeypatch.setattr(
        model_integrity, "__file__", str(tmp_path / "model_integrity.py")
    )
    (tmp_path / "model_catalog_lock.json").write_text(
        json.dumps({"repositories": {"test/integrity": {"revision": "b" * 40}}})
    )
    with pytest.raises(ModelIntegrityError, match="catalog integrity manifest"):
        verify_snapshot("test/integrity", path)
    assert calls == []


def test_unpinned_missing_revision_does_not_use_mirror(snapshot, monkeypatch):
    from huggingface_hub.errors import RevisionNotFoundError

    path, _, _ = snapshot

    def missing(*args, **kwargs):
        import httpx

        raise RevisionNotFoundError(
            "missing",
            response=httpx.Response(
                404, request=httpx.Request("GET", "https://hub.invalid")
            ),
        )

    monkeypatch.setattr("huggingface_hub.HfApi.model_info", missing)
    with pytest.raises(RevisionNotFoundError):
        verify_snapshot("test/integrity", path)
