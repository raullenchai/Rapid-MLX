# SPDX-License-Identifier: Apache-2.0
"""Tiny real-file Hub metadata fixtures for pull orchestration tests."""

import hashlib
from types import SimpleNamespace

import pytest


def snapshot_tree(root, revision="a" * 40, filename="model.safetensors"):
    snapshot = root / "snapshots" / revision
    file = snapshot / filename
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_bytes(b"fixture weights")
    return snapshot


@pytest.fixture(autouse=True)
def hub_integrity_metadata(tmp_path, monkeypatch):
    # These orchestration tests build tiny synthetic snapshots. Their mocked
    # Hub metadata, rather than the release's full packaged inventory, is the
    # integrity contract for those fixtures.
    monkeypatch.setattr("rapid_mlx._mirror._packaged_digest_info", lambda *_args: None)

    def metadata(_self, _repo, *, revision, files_metadata, token=None):
        assert files_metadata is True
        snapshots = [path for path in tmp_path.rglob(revision) if path.is_dir()]
        assert len(snapshots) == 1, "mock downloader must create an exact snapshot"
        files = [path for path in snapshots[0].rglob("*") if path.is_file()]
        assert files, "mock downloader must create selected content"
        return SimpleNamespace(
            sha=revision,
            siblings=[
                SimpleNamespace(
                    rfilename=path.relative_to(snapshots[0]).as_posix(),
                    size=path.stat().st_size,
                    lfs={"sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
                )
                for path in files
            ],
        )

    monkeypatch.setattr("huggingface_hub.HfApi.model_info", metadata)
