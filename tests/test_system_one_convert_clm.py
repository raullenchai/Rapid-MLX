# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib.util
import json
import os
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from rapid_mlx.system_one.convert_clm import (
    _artifact_lock,
    _expected_head_shapes,
    _publish_artifact,
    convert,
    main,
)

requires_mlx = pytest.mark.skipif(
    importlib.util.find_spec("mlx") is None, reason="conversion requires MLX"
)


def _artifact(path, marker: str):
    path.mkdir()
    (path / "config.json").write_text(marker, encoding="utf-8")
    (path / "model.safetensors").write_text(marker, encoding="utf-8")


def test_expected_shapes_validate_dimensions_and_deep_layernorm():
    with pytest.raises(ValueError, match="missing"):
        _expected_head_shapes({})
    with pytest.raises(ValueError, match="positive"):
        _expected_head_shapes({"width": 0, "depth": 2})
    shapes = _expected_head_shapes(
        {
            "hidden_size": 3,
            "width": 2,
            "depth": 3,
            "projection_dim": 4,
            "layernorm": True,
        }
    )
    assert shapes["hidden.0.weight"] == (2, 2)
    assert shapes["norms.0.bias"] == (2,)


def test_publish_artifact_rejects_incomplete_staging_and_file_destination(tmp_path):
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete"):
        _publish_artifact(staging, tmp_path / "head")

    (staging / "model.safetensors").write_bytes(b"weights")
    destination = tmp_path / "head"
    destination.write_text("occupied", encoding="utf-8")
    with pytest.raises(ValueError, match="not a directory"):
        _publish_artifact(staging, destination)


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


def test_publish_artifact_reports_success_when_backup_cleanup_fails(
    monkeypatch, tmp_path
):
    destination = tmp_path / "head"
    staging = tmp_path / ".staging"
    _artifact(destination, "old")
    _artifact(staging, "new")
    monkeypatch.setattr(
        "rapid_mlx.system_one.convert_clm.shutil.rmtree",
        lambda path: (_ for _ in ()).throw(OSError("read-only backup")),
    )
    with pytest.warns(RuntimeWarning, match="retained backup"):
        _publish_artifact(staging, destination)
    assert (destination / "config.json").read_text() == "new"


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
    assert not (tmp_path / ".head.lock").exists()


def test_artifact_read_lock_does_not_write_to_model_parent(tmp_path):
    destination = tmp_path / "read-only-model" / "head"
    destination.parent.mkdir()
    with _artifact_lock(destination, exclusive=False):
        assert not list(destination.parent.iterdir())


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


@requires_mlx
@pytest.mark.parametrize(
    ("checkpoint", "message"),
    [
        ({}, "checkpoint is missing"),
        (
            {"state_head": {}, "action_head": {}, "logit_scale": 0, "cfg": []},
            "cfg must be a mapping",
        ),
        (
            {
                "state_head": {1: _FakeTensor([1])},
                "action_head": {},
                "logit_scale": 0,
                "cfg": {"width": 1, "depth": 2, "hidden_size": 1},
            },
            "tensor names must be strings",
        ),
        (
            {
                "state_head": {},
                "action_head": {},
                "logit_scale": 0,
                "cfg": {"width": 1, "depth": 2, "hidden_size": 1},
            },
            "tensors do not match config",
        ),
        (
            {
                "state_head": {
                    "inp.weight": "bad",
                    "inp.bias": _FakeTensor([0]),
                    "out.weight": _FakeTensor([[1]]),
                    "out.bias": _FakeTensor([0]),
                },
                "action_head": {},
                "logit_scale": 0,
                "cfg": {"width": 1, "depth": 2, "hidden_size": 1},
            },
            "is not a tensor",
        ),
    ],
)
def test_convert_rejects_malformed_checkpoint_contracts(
    monkeypatch, tmp_path, checkpoint, message
):
    _fake_torch(monkeypatch, checkpoint)
    with pytest.raises(ValueError, match=message):
        convert(tmp_path / "head.pt", tmp_path / "converted")


def test_convert_requires_torch(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(RuntimeError, match="requires PyTorch"):
        convert(tmp_path / "head.pt", tmp_path / "converted")


def test_converter_main_prints_output_and_reports_errors(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(sys, "argv", ["convert", "in.pt", "out"])
    monkeypatch.setitem(main.__globals__, "convert", lambda *args: tmp_path / "done")
    main()
    assert capsys.readouterr().out.strip() == str(tmp_path / "done")

    monkeypatch.setitem(
        main.__globals__,
        "convert",
        lambda *args: (_ for _ in ()).throw(ValueError("invalid head")),
    )
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    assert "invalid head" in capsys.readouterr().err


@requires_mlx
def test_convert_writes_validated_safetensors_generation(monkeypatch, tmp_path):
    import mlx.core as mx

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


@requires_mlx
def test_convert_cleans_staging_when_publication_fails(monkeypatch, tmp_path):
    head = {
        "inp.weight": _FakeTensor([[1.0]]),
        "inp.bias": _FakeTensor([0.0]),
        "out.weight": _FakeTensor([[1.0]]),
        "out.bias": _FakeTensor([0.0]),
    }
    checkpoint = {
        "state_head": head,
        "action_head": head,
        "logit_scale": 0.0,
        "cfg": {"hidden_size": 1, "width": 1, "depth": 2, "projection_dim": 1},
    }
    _fake_torch(monkeypatch, checkpoint)
    monkeypatch.setitem(
        convert.__globals__,
        "_publish_artifact",
        lambda *args: (_ for _ in ()).throw(OSError("publish failed")),
    )
    with pytest.raises(OSError, match="publish failed"):
        convert(tmp_path / "head.pt", tmp_path / "converted")
    assert not list(tmp_path.glob(".converted.staging-*"))


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
@requires_mlx
def test_convert_rejects_non_mapping_checkpoint_parts(
    monkeypatch, tmp_path, checkpoint, message
):
    _fake_torch(monkeypatch, checkpoint)
    with pytest.raises(ValueError, match=message):
        convert(tmp_path / "head.pt", tmp_path / "converted")


@requires_mlx
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


@requires_mlx
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
@requires_mlx
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


@requires_mlx
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


@requires_mlx
def test_convert_rejects_depth_below_two(monkeypatch, tmp_path):
    checkpoint = {
        "state_head": {},
        "action_head": {},
        "logit_scale": 1,
        "cfg": {"hidden_size": 1, "width": 1, "depth": 1, "projection_dim": 1},
    }
    _fake_torch(monkeypatch, checkpoint)
    with pytest.raises(ValueError, match="depth must be at least 2"):
        convert(tmp_path / "head.pt", tmp_path / "converted")
