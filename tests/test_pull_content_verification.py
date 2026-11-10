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

        def corrupt_again(repo_id, filename, **kwargs):
            (snapshot / filename).write_bytes(b"bad content")

        monkeypatch.setattr("huggingface_hub.hf_hub_download", corrupt_again)
        with pytest.raises(SystemExit) as caught:
            cli._pull_repository(args, **pull_kwargs)
        assert caught.value.code == 1
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
def test_unpinned_mirror_pull_requires_trusted_hub_metadata(
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
    with pytest.raises(ModelIntegrityError, match="Hub integrity metadata unavailable"):
        cli._pull_repository(args, emit_lifecycle_event=False)
    assert completed == []


@pytest.mark.parametrize("mirror", [False, True])
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("persistent", [False, True])
def test_pull_repairs_only_corrupt_blobs(
    tmp_path, monkeypatch, capsys, mirror, shared, persistent
):
    import os

    commit = "a" * 40
    root = tmp_path / "models--test--integrity"
    snapshot = root / "snapshots" / commit
    snapshot.mkdir(parents=True)
    blobs = root / "blobs"
    blobs.mkdir()
    good = b"good content"
    names = ["config.json", "nested/model.safetensors", "model-2.safetensors"]
    targets = {}
    for name in names:
        digest = hashlib.sha256(name.encode() + good).hexdigest()
        blob = blobs / digest
        if shared:
            target = tmp_path / "blobs" / digest[:2] / digest
            target.parent.mkdir(parents=True, exist_ok=True)
            blob.symlink_to(target)
        else:
            target = blob
        target.write_bytes(good if name == names[0] else b"bad! content")
        leaf = snapshot / name
        leaf.parent.mkdir(parents=True, exist_ok=True)
        leaf.symlink_to(os.path.relpath(blob, leaf.parent))
        targets[name] = target
    untouched = targets[names[0]].stat()
    unrelated = tmp_path / "models--other--repo" / "blobs" / "unrelated"
    unrelated.parent.mkdir(parents=True)
    unrelated.write_bytes(b"unrelated")
    metadata = SimpleNamespace(
        sha=commit,
        siblings=[
            SimpleNamespace(
                rfilename=name,
                size=len(good),
                lfs={"sha256": hashlib.sha256(good).hexdigest()},
            )
            for name in names
        ],
    )
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path))
    monkeypatch.setattr("huggingface_hub.HfApi.model_info", lambda *a, **k: metadata)
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download", lambda *a, **k: str(snapshot)
    )
    monkeypatch.setattr(cli, "_try_mirror_prefetch", lambda *a, **k: mirror)
    monkeypatch.setattr(cli, "_resolve_variant_allow_patterns", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_offline_hub_mode_active", lambda: False)
    monkeypatch.setattr("rapid_mlx.model_aliases.resolve_subfolder", lambda *a: None)
    monkeypatch.setattr(
        "rapid_mlx._version_check.print_staleness_warning_if_any", lambda: None
    )
    completed = []
    monkeypatch.setattr(
        cli, "_sync_pulled_variant_marker", lambda *a: completed.append("marker")
    )
    monkeypatch.setattr(
        cli, "_print_pull_summary", lambda *a, **k: completed.append("summary")
    )
    calls = []

    def download(repo_id, filename, **kwargs):
        assert repo_id == "test/integrity"
        assert kwargs == {"revision": commit, "force_download": True}
        assert filename != names[0]
        assert not targets[filename].exists()
        assert not (snapshot / filename).is_symlink()
        targets[filename].write_bytes(b"bad! content" if persistent else good)
        blob = blobs / hashlib.sha256(filename.encode() + good).hexdigest()
        (snapshot / filename).symlink_to(
            os.path.relpath(blob, (snapshot / filename).parent)
        )
        calls.append(filename)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    args = SimpleNamespace(model="test/integrity", bits=None, format=None)
    if persistent:
        with pytest.raises(SystemExit) as caught:
            cli._pull_repository(
                args, revision_override=commit, emit_lifecycle_event=False
            )
        assert caught.value.code == 1
        stderr = capsys.readouterr().err
        assert len(stderr.strip().splitlines()) == 1
        assert "download is corrupt" in stderr.lower()
        assert names[1] in stderr and names[2] in stderr
        assert "Traceback" not in stderr
        assert not completed
    else:
        cli._pull_repository(args, revision_override=commit, emit_lifecycle_event=False)
        assert completed == ["marker", "summary"]
        assert all(target.read_bytes() == good for target in targets.values())
    assert calls == names[1:]
    assert targets[names[0]].stat() == untouched
    assert unrelated.read_bytes() == b"unrelated"


@pytest.mark.parametrize(
    "case", ["other_repo", "directory_link", "offline", "unlink_error"]
)
def test_corruption_repair_refuses_unsafe_or_unavailable_eviction(
    tmp_path, monkeypatch, capsys, case
):
    from pathlib import Path

    commit = "a" * 40
    root = tmp_path / "models--test--integrity"
    snapshot = root / "snapshots" / commit
    snapshot.mkdir(parents=True)
    name = "nested/model.safetensors"
    leaf = snapshot / name
    blob = root / "blobs" / ("b" * 64)
    if case == "other_repo":
        blob = tmp_path / "models--other--repo" / "blobs" / blob.name
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"bad")
    if case == "directory_link":
        outside = tmp_path / "outside"
        outside.mkdir()
        (snapshot / "nested").symlink_to(outside, target_is_directory=True)
    else:
        leaf.parent.mkdir()
    leaf.symlink_to(blob)
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path))
    monkeypatch.setattr(
        "huggingface_hub.HfApi.model_info",
        lambda *a, **k: SimpleNamespace(
            sha=commit,
            siblings=[
                SimpleNamespace(rfilename=name, size=3, lfs={"sha256": "b" * 64})
            ],
        ),
    )
    if case == "unlink_error":
        original_unlink = Path.unlink

        def deny_unlink(path, *args, **kwargs):
            if path == blob:
                raise PermissionError("Cannot remove corrupt blob")
            return original_unlink(path, *args, **kwargs)

        monkeypatch.setattr(Path, "unlink", deny_unlink)
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download",
        lambda *a, **k: pytest.fail("unsafe or offline repair attempted a download"),
    )
    # Supply an offline manifest so verification can identify corruption without
    # network access; only the repair must be refused in offline mode.
    if case == "offline":
        from rapid_mlx import model_integrity

        module = tmp_path / "module"
        module.mkdir()
        monkeypatch.setattr(
            model_integrity, "__file__", str(module / "model_integrity.py")
        )
        import json

        (module / "model_catalog_lock.json").write_text(
            json.dumps(
                {
                    "repositories": {
                        "test/integrity": {
                            "revision": commit,
                            "lfs_files": {name: {"sha256": "b" * 64, "size": 3}},
                        }
                    }
                }
            )
        )
    with pytest.raises(SystemExit) as caught:
        cli._verify_pulled_snapshot(
            "test/integrity",
            snapshot,
            retry_hint="retry",
            local_files_only=case == "offline",
        )
    assert caught.value.code == 1
    stderr = capsys.readouterr().err
    assert "Download is corrupt" in stderr and name in stderr
    assert len(stderr.strip().splitlines()) == 1
    assert leaf.is_symlink() and blob.read_bytes() == b"bad"
