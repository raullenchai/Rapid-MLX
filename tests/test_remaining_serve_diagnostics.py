# SPDX-License-Identifier: Apache-2.0
"""User-facing startup remedies for remaining local load failures."""

from __future__ import annotations

import errno
import json
from types import SimpleNamespace

import pytest

from rapid_mlx.local_serve_diagnostics import startup_error_lines
from rapid_mlx.telemetry.model_events import serve_error_class


@pytest.mark.parametrize(
    ("case", "failure", "cause", "step"),
    [
        (
            "disk",
            OSError(errno.ENOSPC, "No space left on device"),
            "disk is full",
            "Free disk space",
        ),
        (
            "weights",
            FileNotFoundError("No safetensors found in model"),
            "no readable model safetensors",
            "Complete the model download",
        ),
        (
            "truncated",
            RuntimeError("[read] Unable to read 5 bytes from file."),
            "incomplete or unreadable",
            "Re-download or re-export",
        ),
    ],
)
def test_local_startup_remedies_without_reclassifying(
    tmp_path, case, failure, cause, step
):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    if case == "truncated":
        (model / "model.safetensors").write_bytes(b"bad")
    if case == "weights":
        failure = FileNotFoundError(f"No safetensors found in {model}")
    assert serve_error_class(failure, model_ref=str(model)) == (
        "local_path_missing" if case == "weights" else "other"
    )
    lines = startup_error_lines(failure, str(model))
    assert lines is not None
    assert cause in lines[0]
    assert step in lines[1]


def test_missing_config_uses_matching_checkpoint_remedy(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    failure = FileNotFoundError(errno.ENOENT, "missing", str(model / "config.json"))
    assert serve_error_class(failure, model_ref=str(model)) == "local_path_missing"
    assert startup_error_lines(failure, str(model)) == (
        "The local model directory is missing config.json.",
        "Copy config.json from the same checkpoint as the model weights.",
    )


def test_unreadable_model_file_has_permissions_remedy(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    weight = model / "model.safetensors"
    weight.write_bytes(b"valid")
    failure = PermissionError(errno.EACCES, "denied", str(weight))
    assert serve_error_class(failure, model_ref=str(model)) == "corrupt_weights"
    assert startup_error_lines(failure, str(model)) == (
        "Rapid-MLX cannot read a file in the local model directory.",
        "Grant your user read access to the model files, then retry.",
    )


def test_loader_reports_unreadable_safetensors_file(tmp_path, monkeypatch):
    model = tmp_path / "model"
    model.mkdir()
    weight = model / "model.safetensors"
    weight.write_bytes(b"valid")
    monkeypatch.setattr("rapid_mlx.local_serve_diagnostics.os.access", lambda *_: False)
    failure = RuntimeError(f"[load_safetensors] Failed to open file {weight}")
    assert "read access" in startup_error_lines(failure, str(model))[1]


def test_dangling_weight_symlink_is_not_a_permission_failure(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"qwen3"}')
    weight = model / "model.safetensors"
    weight.symlink_to("missing.safetensors")
    failure = RuntimeError(f"[load_safetensors] Failed to open file {weight}")
    assert startup_error_lines(failure, str(model)) is None


def test_relative_model_paths_keep_file_remedies(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"qwen3"}')
    assert (
        "no readable model safetensors"
        in startup_error_lines(
            FileNotFoundError("No safetensors found in model"), "model"
        )[0]
    )
    (model / "config.json").unlink()
    assert (
        "missing config.json"
        in startup_error_lines(
            FileNotFoundError(errno.ENOENT, "missing", "model/config.json"), "model"
        )[0]
    )
    weight = tmp_path / "model.safetensors"
    weight.write_bytes(b"bad")
    assert (
        "model directory"
        in startup_error_lines(
            NotADirectoryError(
                errno.ENOTDIR, "not a directory", "model.safetensors/config.json"
            ),
            "model.safetensors",
        )[0]
    )


def test_single_weight_file_points_to_parent_directory(tmp_path):
    weight = tmp_path / "model.safetensors"
    weight.write_bytes(b"bad")
    failure = NotADirectoryError(
        errno.ENOTDIR, "not a directory", str(weight / "config.json")
    )
    assert serve_error_class(failure, model_ref=str(weight)) == "corrupt_weights"
    assert startup_error_lines(failure, str(weight)) == (
        "A safetensors file was supplied where a model directory is required.",
        "Serve its parent model directory containing config.json and the matching weights.",
    )


def test_generic_prepare_error_prints_remedy(tmp_path, capsys):
    from rapid_mlx.cli import _print_model_load_error

    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"qwen3"}')
    (model / "model.safetensors").write_bytes(b"bad")
    failure = RuntimeError("[read] Unable to read 5 bytes from file.")
    _print_model_load_error(SimpleNamespace(model=str(model)), failure)
    rendered = capsys.readouterr().err
    assert "Cause: A model safetensors file is incomplete or unreadable." in rendered
    assert "Next step: Re-download or re-export" in rendered


def test_diagnostics_reject_similar_unrelated_failures(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"qwen3"}')
    (model / "model.safetensors").write_bytes(b"valid")
    assert (
        startup_error_lines(RuntimeError("Unable to read model cache"), str(model))
        is None
    )
    assert (
        startup_error_lines(
            RuntimeError("[read] Unable to read 5 bytes from file."), None
        )
        is None
    )
    assert (
        startup_error_lines(
            NotADirectoryError(errno.ENOTDIR, "unrelated", str(model / "other")),
            str(model / "model.safetensors"),
        )
        is None
    )
    assert (
        startup_error_lines(
            PermissionError(errno.EACCES, "denied", str(tmp_path / "outside")),
            str(model),
        )
        is None
    )
    assert (
        startup_error_lines(
            PermissionError(
                errno.EACCES, "denied", str(model / ".." / "outside" / "file")
            ),
            str(model),
        )
        is None
    )
    assert (
        startup_error_lines(
            FileNotFoundError("No safetensors found in another model"), str(model)
        )
        is None
    )

    from rapid_mlx.model_load_errors import TokenizerLoadFailed

    tokenizer_error = TokenizerLoadFailed("Tokenizer loading failed")
    tokenizer_error.__cause__ = RuntimeError("[read] Unable to read 5 bytes from file.")
    assert "safetensors file" not in startup_error_lines(tokenizer_error, str(model))[0]

    class HostileError(RuntimeError):
        def __str__(self):
            raise KeyboardInterrupt

    assert startup_error_lines(HostileError(), str(model)) is None
