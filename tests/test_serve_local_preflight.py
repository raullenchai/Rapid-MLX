# SPDX-License-Identifier: Apache-2.0
"""Serve preflight rejects incomplete local paths before creating an engine."""

from __future__ import annotations

import errno
import json
import os
from types import SimpleNamespace

import pytest

from rapid_mlx.local_model_path import (
    local_serve_preflight_lines,
    validate_local_model_for_serve,
)
from rapid_mlx.local_serve_diagnostics import startup_error_lines
from rapid_mlx.telemetry.model_events import serve_error_class


@pytest.mark.parametrize("case", ["missing", "file", "config", "weights"])
def test_local_model_preflight(tmp_path, case):
    root = tmp_path / "model"
    if case == "file":
        root.write_text("not a directory")
    elif case != "missing":
        root.mkdir()
        if case == "weights":
            (root / "config.json").write_text('{"model_type":"qwen3"}')
    with pytest.raises(OSError) as captured:
        validate_local_model_for_serve(str(root))
    assert (
        serve_error_class(captured.value, model_ref=str(root)) == "local_path_missing"
    )
    cause, step = local_serve_preflight_lines(str(root), captured.value)
    assert {
        "missing": "does not exist",
        "file": "not a model directory",
        "config": "config.json",
        "weights": "model weights",
    }[case] in cause
    assert "rapid-mlx" in step


def test_local_model_preflight_ignores_unselected_nested_weights(tmp_path):
    root = tmp_path / "model"
    (root / "weights").mkdir(parents=True)
    (root / "config.json").write_text('{"model_type":"qwen3"}')
    (root / "weights" / "model.safetensors").write_bytes(b"data")
    with pytest.raises(FileNotFoundError, match="Missing model weights"):
        validate_local_model_for_serve(str(root))


def test_unrelated_nested_index_does_not_block_root_checkpoint(tmp_path):
    root = tmp_path / "model"
    (root / "variant").mkdir(parents=True)
    (root / "config.json").write_text('{"model_type":"qwen3"}')
    (root / "model.safetensors").write_bytes(b"data")
    (root / "variant" / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"x": "missing.safetensors"}})
    )
    validate_local_model_for_serve(str(root))


def test_programmatic_server_load_rejects_missing_config_before_engine(tmp_path):
    from rapid_mlx.server import load_model

    root = tmp_path / "model"
    root.mkdir()
    with pytest.raises(FileNotFoundError, match="Missing model config"):
        load_model(str(root))


@pytest.mark.parametrize("layout", ["whisper", "image", "video"])
def test_supported_nontext_layouts_reach_their_loader(tmp_path, layout):
    root = tmp_path / layout
    root.mkdir()
    if layout == "whisper":
        (root / "config.json").write_text('{"model_type":"whisper"}')
        (root / "weights.npz").write_bytes(b"data")
    else:
        (
            root / ("model_index.json" if layout == "image" else "split_model.json")
        ).write_text("{}")
    validate_local_model_for_serve(str(root))


def test_empty_local_alias_directory_keeps_catalog_fallback(tmp_path, monkeypatch):
    from rapid_mlx.model_aliases import resolve_model

    monkeypatch.chdir(tmp_path)
    (tmp_path / "qwen3-0.6b-4bit").mkdir()
    resolved = resolve_model("qwen3-0.6b-4bit")
    assert resolved != "qwen3-0.6b-4bit"
    validate_local_model_for_serve(resolved)


def test_missing_index_shard_is_rejected_before_engine(tmp_path):
    root = tmp_path / "model"
    root.mkdir()
    (root / "config.json").write_text('{"model_type":"qwen3"}')
    (root / "model-00001.safetensors").write_bytes(b"data")
    (root / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "a": "model-00001.safetensors",
                    "b": "model-00002.safetensors",
                }
            }
        )
    )
    with pytest.raises(
        FileNotFoundError, match="Missing indexed model shard"
    ) as raised:
        validate_local_model_for_serve(str(root))
    assert serve_error_class(raised.value, model_ref=str(root)) == "local_path_missing"


@pytest.mark.parametrize("denied", ["root", "config", "weights"])
def test_unreadable_local_file_reports_permissions(tmp_path, monkeypatch, denied):
    root = tmp_path / "model"
    root.mkdir()
    config = root / "config.json"
    config.write_text('{"model_type":"qwen3"}')
    weight = root / "model.safetensors"
    weight.write_bytes(b"weights")
    blocked = {"root": root, "config": config, "weights": weight}[denied]
    access = os.access
    monkeypatch.setattr(
        "rapid_mlx.local_model_path.os.access",
        lambda path, mode: False if path == blocked else access(path, mode),
    )
    with pytest.raises(PermissionError) as raised:
        validate_local_model_for_serve(str(root))
    cause, step = local_serve_preflight_lines(str(root), raised.value)
    assert "cannot be read" in cause
    assert "chmod -R u+rX" in step


def test_cache_path_suggests_catalog_alias(tmp_path):
    root = (
        tmp_path / "models--mlx-community--Qwen3-0.6B-4bit" / "snapshots" / "revision"
    )
    failure = FileNotFoundError(errno.ENOENT, "missing", str(root))
    _, step = local_serve_preflight_lines(str(root), failure)
    assert "rapid-mlx serve qwen3-0.6b-4bit" in step


def test_truncated_read_needs_local_weight_evidence(tmp_path):
    root = tmp_path / "model"
    root.mkdir()
    (root / "config.json").write_text('{"model_type":"qwen3"}')
    error = RuntimeError("[read] Unable to read 5 bytes from file.")
    assert serve_error_class(error, model_ref=str(root)) == "other"
    (root / "model.safetensors").write_bytes(b"bad")
    assert serve_error_class(error, model_ref=str(root)) == "corrupt_weights"


def test_bad_numeric_config_is_actionable(tmp_path):
    root = tmp_path / "model"
    root.mkdir()
    (root / "config.json").write_text('{"model_type":"qwen3","hidden_size":"wrong"}')
    (root / "model.safetensors").write_bytes(b"weights")
    error = TypeError("unsupported operand type(s) for /: 'int' and 'str'")
    assert serve_error_class(error, model_ref=str(root)) == "invalid_config"
    cause, step = startup_error_lines(error, str(root))
    assert "hidden_size" in cause
    assert "config.json" in step


def test_sync_cli_prints_bad_numeric_config_remedy(tmp_path, capsys):
    from rapid_mlx.cli import _print_model_load_error

    root = tmp_path / "model"
    root.mkdir()
    (root / "config.json").write_text('{"model_type":"qwen3","hidden_size":"wrong"}')
    (root / "model.safetensors").write_bytes(b"weights")
    _print_model_load_error(
        SimpleNamespace(model=str(root)),
        TypeError("unsupported operand type(s) for /: 'int' and 'str'"),
    )
    rendered = capsys.readouterr().err
    assert "Cause: config.json has a nonnumeric hidden_size value." in rendered
    assert "Next step: Correct hidden_size" in rendered


@pytest.mark.parametrize(
    ("error", "failure_class", "cause", "command"),
    [
        (
            RuntimeError("Metal command buffer failed while loading"),
            "backend_load_failed",
            "Metal backend",
            "rapid-mlx doctor",
        ),
        (
            MemoryError("out of memory"),
            "insufficient_memory",
            "out of memory",
            "rapid-mlx serve qwen3-0.6b-4bit",
        ),
    ],
)
def test_runtime_failure_has_class_and_command(error, failure_class, cause, command):
    assert serve_error_class(error) == failure_class
    lines = startup_error_lines(error, None)
    assert cause in lines[0]
    assert command in lines[1]
