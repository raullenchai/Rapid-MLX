# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import os
import sys
import threading
import time
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from rapid_mlx.system_one.convert_clm import _artifact_lock, _publish_artifact, convert


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


def test_artifact_lock_serializes_readers_and_publishers(tmp_path):
    destination = tmp_path / "head"
    staging = tmp_path / ".staging"
    _artifact(destination, "old")
    _artifact(staging, "new")
    started = threading.Event()
    finished = threading.Event()

    def publish():
        started.set()
        _publish_artifact(staging, destination)
        finished.set()

    with _artifact_lock(destination, exclusive=False):
        thread = threading.Thread(target=publish)
        thread.start()
        assert started.wait(timeout=1)
        time.sleep(0.05)
        assert not finished.is_set()
    thread.join(timeout=1)
    assert finished.is_set()
    assert (destination / "config.json").read_text() == "new"


class _FakeTensor:
    def __init__(self, value):
        self.value = np.asarray(value, dtype=np.float32)

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value

    def item(self):
        return self.value.item()

    @property
    def shape(self):
        return self.value.shape


def _fake_torch(monkeypatch, checkpoint):
    module = SimpleNamespace(
        Tensor=_FakeTensor,
        load=lambda *args, **kwargs: checkpoint,
        as_tensor=lambda value: _FakeTensor(value),
    )
    monkeypatch.setitem(sys.modules, "torch", module)


def test_convert_writes_validated_safetensors_generation(monkeypatch, tmp_path):
    head = {
        "inp.weight": _FakeTensor([[1.0]]),
        "inp.bias": _FakeTensor([0.0]),
        "out.weight": _FakeTensor([[1.0]]),
        "out.bias": _FakeTensor([0.0]),
    }
    checkpoint = {
        "state_head": head,
        "action_head": {**head, "out.weight": _FakeTensor([[2.0]])},
        "logit_scale": 1.5,
        "cfg": {"hidden_size": 1, "width": 1, "depth": 2, "projection_dim": 1},
    }
    _fake_torch(monkeypatch, checkpoint)
    destination = convert(tmp_path / "head.pt", tmp_path / "converted")
    config = json.loads((destination / "config.json").read_text())
    assert config["logit_scale"] == 1.5
    weights = mx.load(str(destination / "model.safetensors"))
    assert len(weights) == 8
    assert weights["action_head.out.weight"].item() == 2.0


@pytest.mark.parametrize(
    ("checkpoint", "message"),
    [
        ([], "root must be a mapping"),
        (
            {
                "state_head": [],
                "action_head": {},
                "logit_scale": 1,
                "cfg": {"width": 1, "depth": 2},
            },
            "state_head must be a mapping",
        ),
    ],
)
def test_convert_rejects_non_mapping_checkpoint_parts(
    monkeypatch, tmp_path, checkpoint, message
):
    _fake_torch(monkeypatch, checkpoint)
    with pytest.raises(ValueError, match=message):
        convert(tmp_path / "head.pt", tmp_path / "converted")


def test_convert_rejects_tensor_shape_mismatch(monkeypatch, tmp_path):
    head = {
        "inp.weight": _FakeTensor([[1.0, 2.0]]),
        "inp.bias": _FakeTensor([0.0]),
        "out.weight": _FakeTensor([[1.0]]),
        "out.bias": _FakeTensor([0.0]),
    }
    checkpoint = {
        "state_head": head,
        "action_head": head,
        "logit_scale": 1,
        "cfg": {"hidden_size": 1, "width": 1, "depth": 2, "projection_dim": 1},
    }
    _fake_torch(monkeypatch, checkpoint)
    with pytest.raises(ValueError, match="has shape .* expected"):
        convert(tmp_path / "head.pt", tmp_path / "converted")


def test_convert_applies_top_level_projection_dimension_before_validation(
    monkeypatch, tmp_path
):
    head = {
        "inp.weight": _FakeTensor([[1.0]]),
        "inp.bias": _FakeTensor([0.0]),
        "out.weight": _FakeTensor([[1.0], [2.0]]),
        "out.bias": _FakeTensor([0.0, 0.0]),
    }
    checkpoint = {
        "state_head": head,
        "action_head": head,
        "logit_scale": 1,
        "projection_dim": 2,
        "cfg": {"hidden_size": 1, "width": 1, "depth": 2, "projection_dim": 1},
    }
    _fake_torch(monkeypatch, checkpoint)
    destination = convert(tmp_path / "head.pt", tmp_path / "converted")
    config = json.loads((destination / "config.json").read_text())
    assert config["projection_dim"] == 2


@pytest.mark.parametrize("logit_scale", [float("nan"), float("inf"), 1000.0])
def test_convert_rejects_unsafe_logit_scale(monkeypatch, tmp_path, logit_scale):
    head = {
        "inp.weight": _FakeTensor([[1.0]]),
        "inp.bias": _FakeTensor([0.0]),
        "out.weight": _FakeTensor([[1.0]]),
        "out.bias": _FakeTensor([0.0]),
    }
    checkpoint = {
        "state_head": head,
        "action_head": head,
        "logit_scale": logit_scale,
        "cfg": {"hidden_size": 1, "width": 1, "depth": 2, "projection_dim": 1},
    }
    _fake_torch(monkeypatch, checkpoint)
    with pytest.raises(ValueError, match="finite exponential range"):
        convert(tmp_path / "head.pt", tmp_path / "converted")


def test_convert_rejects_non_json_config(monkeypatch, tmp_path):
    head = {
        "inp.weight": _FakeTensor([[1.0]]),
        "inp.bias": _FakeTensor([0.0]),
        "out.weight": _FakeTensor([[1.0]]),
        "out.bias": _FakeTensor([0.0]),
    }
    checkpoint = {
        "state_head": head,
        "action_head": head,
        "logit_scale": 1,
        "cfg": {
            "hidden_size": 1,
            "width": 1,
            "depth": 2,
            "projection_dim": 1,
            "unsupported": object(),
        },
    }
    _fake_torch(monkeypatch, checkpoint)
    with pytest.raises(ValueError, match="JSON-compatible"):
        convert(tmp_path / "head.pt", tmp_path / "converted")
