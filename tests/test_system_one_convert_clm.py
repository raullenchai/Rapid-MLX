# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os

import pytest

from rapid_mlx.system_one.convert_clm import _publish_artifact


def _artifact(path, marker: str):
    path.mkdir()
    (path / "config.json").write_text(marker, encoding="utf-8")
    (path / "model.safetensors").write_text(marker, encoding="utf-8")


def test_publish_artifact_replaces_both_files_as_one_generation(tmp_path):
    destination = tmp_path / "head"
    staging = tmp_path / ".staging"
    _artifact(destination, "old")
    _artifact(staging, "new")
    _publish_artifact(staging, destination)
    assert (destination / "config.json").read_text() == "new"
    assert (destination / "model.safetensors").read_text() == "new"
    assert not staging.exists()
    assert not list(tmp_path.glob(".head.backup-*"))


def test_publish_artifact_restores_old_generation_when_publish_fails(
    monkeypatch, tmp_path
):
    destination = tmp_path / "head"
    staging = tmp_path / ".staging"
    _artifact(destination, "old")
    _artifact(staging, "new")
    real_replace = os.replace

    def fail_staging(source, target):
        if source == staging and target == destination:
            raise OSError("simulated publish failure")
        return real_replace(source, target)

    monkeypatch.setattr(os, "replace", fail_staging)
    with pytest.raises(OSError, match="simulated"):
        _publish_artifact(staging, destination)
    assert (destination / "config.json").read_text() == "old"
    assert (destination / "model.safetensors").read_text() == "old"
    assert staging.exists()
    assert not list(tmp_path.glob(".head.backup-*"))


def test_publish_artifact_refuses_to_delete_unrelated_files(tmp_path):
    destination = tmp_path / "head"
    staging = tmp_path / ".staging"
    _artifact(destination, "old")
    (destination / "notes.txt").write_text("keep", encoding="utf-8")
    _artifact(staging, "new")
    with pytest.raises(ValueError, match="unrelated"):
        _publish_artifact(staging, destination)
    assert (destination / "notes.txt").read_text() == "keep"
    assert staging.exists()


def test_publish_artifact_preserves_backup_when_restore_fails(monkeypatch, tmp_path):
    destination = tmp_path / "head"
    staging = tmp_path / ".staging"
    _artifact(destination, "old")
    _artifact(staging, "new")
    real_replace = os.replace

    def fail_publish_and_restore(source, target):
        if target == destination and source != destination:
            raise OSError("simulated replace failure")
        return real_replace(source, target)

    monkeypatch.setattr(os, "replace", fail_publish_and_restore)
    with pytest.raises(OSError, match="simulated replace failure"):
        _publish_artifact(staging, destination)
    backups = list(tmp_path.glob(".head.backup-*"))
    assert len(backups) == 1
    assert (backups[0] / "config.json").read_text() == "old"
    assert (backups[0] / "model.safetensors").read_text() == "old"
