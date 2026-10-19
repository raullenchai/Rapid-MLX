# SPDX-License-Identifier: Apache-2.0
"""Serve preflight rejects incomplete local paths before creating an engine."""

from __future__ import annotations

import errno

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


def test_local_model_preflight_allows_nested_weights(tmp_path):
    root = tmp_path / "model"
    (root / "weights").mkdir(parents=True)
    (root / "config.json").write_text('{"model_type":"qwen3"}')
    (root / "weights" / "model.safetensors").write_bytes(b"data")
    validate_local_model_for_serve(str(root))


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
