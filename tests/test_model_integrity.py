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


@pytest.mark.parametrize("single_file", [False, True])
def test_shared_download_wrappers_enforce_hash(snapshot, monkeypatch, single_file):
    from rapid_mlx import model_downloads

    path, info, calls = snapshot
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda *a, **kw: str(path))
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download",
        lambda *a, **kw: str(path / "model.safetensors"),
    )
    monkeypatch.setattr(model_downloads, "disabled", lambda: False)
    (path / "model.safetensors").write_bytes(b"corruption!")
    with pytest.raises(ModelIntegrityError):
        if single_file:
            model_downloads.hf_hub_download("test/integrity", "model.safetensors")
        else:
            model_downloads.snapshot_download("test/integrity")


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


def test_automatic_mirror_download_propagates_integrity_failure(monkeypatch):
    from rapid_mlx import cli

    monkeypatch.setattr(cli, "_cache_runnability", lambda *a, **kw: False)
    monkeypatch.setattr(cli, "_offline_hub_mode_active", lambda: False)
    monkeypatch.setattr(cli, "_check_disk_space", lambda *a, **kw: None)
    monkeypatch.setattr(cli, "_try_mirror_prefetch", lambda *a, **kw: True)
    monkeypatch.setattr(
        cli, "_active_hf_snapshot_path", lambda *a: "/tmp/snapshots/" + "a" * 40
    )
    monkeypatch.setattr("rapid_mlx.model_downloads.disabled", lambda: False)

    def reject(*args, **kwargs):
        raise ModelIntegrityError("tampered weights")

    monkeypatch.setattr("rapid_mlx.model_integrity.verify_snapshot", reject)
    with pytest.raises(ModelIntegrityError, match="tampered weights"):
        cli._ensure_model_downloaded("test/integrity")


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


@pytest.mark.parametrize("subfolder", ["", "4bit"])
def test_cached_catalog_load_reverifies_rejected_bytes_and_allows_partial_offline(
    tmp_path, monkeypatch, subfolder
):
    import json
    from pathlib import Path

    from rapid_mlx import model_aliases, model_catalog
    from rapid_mlx.utils.tokenizer import _resolve_subfolder_checkpoint

    repo, commit = "test/cached-integrity", "c" * 40
    snapshot = tmp_path / "models--test--cached-integrity" / "snapshots" / commit
    checkpoint = snapshot / subfolder
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text("{}")
    weight = checkpoint / "model.safetensors"
    weight.write_bytes(b"valid tensors")
    records = {
        str(file.relative_to(snapshot)): {
            "sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
            "size": file.stat().st_size,
        }
        for file in checkpoint.iterdir()
    }
    # Optional repository data and other variants were never downloaded.
    records["optional/other.safetensors"] = {"sha256": "0" * 64, "size": 10}
    lock = {"repositories": {repo: {"revision": commit, "lfs_files": records}}}
    read_text = Path.read_text
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda path, *a, **kw: (
            json.dumps(lock)
            if path.name == "model_catalog_lock.json"
            else read_text(path, *a, **kw)
        ),
    )
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path))
    monkeypatch.setattr(model_aliases, "resolve_model", lambda name: repo)
    monkeypatch.setattr(model_aliases, "resolve_hub_model", lambda name: repo)
    monkeypatch.setattr(model_aliases, "resolve_revision", lambda name: commit)
    monkeypatch.setattr(
        model_aliases, "resolve_subfolder", lambda name: subfolder or None
    )
    monkeypatch.setattr(model_aliases, "checkpoint_prefix", lambda name: subfolder)
    monkeypatch.setattr(
        model_catalog, "resolve_repository_identity", lambda name: (repo, commit)
    )
    monkeypatch.setattr(
        "rapid_mlx._mirror.pinned_snapshot_download",
        lambda *a, **kw: pytest.fail("complete partial cache must not download"),
    )
    assert _resolve_subfolder_checkpoint("cached-alias") == str(checkpoint)
    weight.write_bytes(b"wrong tensors")
    names = [str(file.relative_to(snapshot)) for file in checkpoint.iterdir()]
    with pytest.raises(ModelIntegrityError, match="hash mismatch"):
        verify_snapshot(repo, snapshot, revision=commit, filenames=names)
    # A retry remains structurally complete, but cannot reach the actual loader.
    with pytest.raises(ModelIntegrityError, match="hash mismatch"):
        _resolve_subfolder_checkpoint("cached-alias")
    assert weight.read_bytes() == b"wrong tensors"


def test_cached_custom_pin_reuses_full_pull_proof_offline(snapshot, monkeypatch):
    from rapid_mlx.model_integrity import verify_cached_checkpoint

    path, info, calls = snapshot
    verify_snapshot("test/integrity", path, revision=path.name)
    monkeypatch.setattr(
        "huggingface_hub.HfApi.model_info",
        lambda *a, **kw: pytest.fail("warm verification must be offline"),
    )
    verify_cached_checkpoint("test/integrity", path, revision=path.name)
    (path / "model.safetensors").write_bytes(b"wrong data!")
    with pytest.raises(ModelIntegrityError, match="hash mismatch"):
        verify_cached_checkpoint("test/integrity", path, revision=path.name)


def test_warm_image_loader_rejects_previously_failed_weights(snapshot, monkeypatch):
    from rapid_mlx.image.engine import ImageGenerationEngine

    path, info, calls = snapshot
    verify_snapshot("test/integrity", path, revision=path.name)
    monkeypatch.setattr(
        "rapid_mlx.model_catalog.resolve_repository_identity",
        lambda name: (name, path.name),
    )
    monkeypatch.setattr(
        "rapid_mlx._download_gate.mflux_local_snapshot", lambda name: str(path)
    )
    engine = object.__new__(ImageGenerationEngine)
    engine._packaged_checkpoint = True
    engine.family = "sdxl-base"
    engine.model_name = "test/integrity"
    assert engine._model_path_for_mflux() == str(path)
    (path / "model.safetensors").write_bytes(b"wrong data!")
    with pytest.raises(ModelIntegrityError):
        verify_snapshot("test/integrity", path, revision=path.name)
    with pytest.raises(ModelIntegrityError):
        engine._model_path_for_mflux()
