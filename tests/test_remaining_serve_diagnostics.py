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
    assert (
        serve_error_class(failure, model_ref=str(model))
        == {
            "weights": "local_path_missing",
            "truncated": "corrupt_weights",
            "disk": "other",
        }[case]
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


def _write_tiny_weights(path, header=None, data=b"\0" * 4):
    import struct

    if header is None:
        header = {"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + data)


@pytest.mark.parametrize(
    "case",
    ["complete", "symlink", "nested", "missing", "absent", "truncated", "bad_entry"],
)
def test_checkpoint_manifest_integrity(tmp_path, case):
    from rapid_mlx.checkpoint_integrity import incomplete_checkpoint_reason

    root = tmp_path / "snapshot"
    root.mkdir()
    shard = root / "model.safetensors"
    _write_tiny_weights(shard)
    name = shard.name
    if case == "symlink":
        blob = tmp_path / "blob"
        shard.rename(blob)
        shard.symlink_to("../blob")
    if case == "nested":
        nested = root / "weights"
        nested.mkdir()
        shard.rename(nested / shard.name)
        name = "weights/" + shard.name
    weights = {"weight": name}
    if case == "missing":
        weights["other"] = "missing.safetensors"
    if case == "absent":
        weights["other"] = name
    if case == "truncated":
        _write_tiny_weights(shard, data=b"")
    if case == "bad_entry":
        _write_tiny_weights(shard, header={"weight": None})
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weights})
    )
    reason = incomplete_checkpoint_reason(root)
    if case in ("complete", "symlink", "nested"):
        assert reason is None
    else:
        assert reason is not None
        assert {
            "missing": "which is missing",
            "absent": "missing 1 of the 2",
            "truncated": "truncated",
            "bad_entry": "unreadable",
        }[case] in reason


def test_checkpoint_reports_truncated_unindexed_shard(tmp_path):
    """The loader reads every model*.safetensors file, even outside weight_map."""
    from rapid_mlx.checkpoint_integrity import incomplete_checkpoint_reason

    shard = tmp_path / "model-00001-of-00001.safetensors"
    _write_tiny_weights(shard)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": shard.name}})
    )
    _write_tiny_weights(tmp_path / "model-stale.safetensors", data=b"")

    assert incomplete_checkpoint_reason(tmp_path) == (
        "model-stale.safetensors is truncated or unreadable"
    )


@pytest.mark.parametrize(
    "header",
    [
        [],
        {"weight": None},
        {"weight": {"data_offsets": [0]}},
        {"weight": {"data_offsets": [0, "4"]}},
    ],
)
def test_checkpoint_invalid_headers(tmp_path, header):
    from rapid_mlx.checkpoint_integrity import incomplete_checkpoint_reason

    _write_tiny_weights(tmp_path / "model.safetensors", header)
    assert "truncated or unreadable" in incomplete_checkpoint_reason(tmp_path)


@pytest.mark.parametrize("data", [b"", b"\xff" * 8, b"\x01" + b"\0" * 7 + b"{"])
def test_checkpoint_unreadable_headers(tmp_path, data):
    from rapid_mlx.checkpoint_integrity import incomplete_checkpoint_reason

    (tmp_path / "model.safetensors").write_bytes(data)
    assert "truncated or unreadable" in incomplete_checkpoint_reason(tmp_path)


@pytest.mark.parametrize(
    "index",
    [
        [],
        {"weight_map": []},
        {
            "weight_map": {
                "bad": "../escape",
                "empty": "",
                "absolute": "/weights",
                "invalid": None,
            }
        },
    ],
)
def test_checkpoint_invalid_index_is_left_to_loader(tmp_path, index):
    from rapid_mlx.checkpoint_integrity import incomplete_checkpoint_reason

    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    _write_tiny_weights(
        tmp_path / "model.safetensors",
        {"__metadata__": {"format": "fixture"}, "weight": {"data_offsets": [0, 4]}},
    )
    assert incomplete_checkpoint_reason(tmp_path) is None
    assert incomplete_checkpoint_reason(tmp_path / "absent") is None


@pytest.mark.parametrize(
    "case",
    [
        "incomplete",
        "truncated",
        "missing_params",
        "hub",
        "missing_local",
        "unparsed",
        "complete_corrupt",
    ],
)
def test_checkpoint_startup_guidance(tmp_path, capsys, case):
    from rapid_mlx.cli import _print_model_load_error
    from rapid_mlx.model_load_errors import IncompatibleWeights

    ref = str(tmp_path)
    failure = None
    expected = "corrupt_weights"
    if case == "truncated":
        _write_tiny_weights(tmp_path / "model.safetensors", data=b"")
        failure = RuntimeError("[load_safetensors] Invalid file size")
    elif case in ("missing_params", "unparsed"):
        failure = IncompatibleWeights(
            "Missing 1 parameters: model.layers.0.weight."
            if case == "missing_params"
            else "mismatched weights"
        )
        expected = "incompatible_weights"
    elif case == "hub":
        ref = "owner/checkpoint"
    elif case == "missing_local":
        ref = str(tmp_path / "missing")
    elif case == "complete_corrupt":
        _write_tiny_weights(tmp_path / "model.safetensors")
        failure = RuntimeError("[load_safetensors] Invalid file size")
    if failure is None:
        from rapid_mlx.model_load_errors import IncompleteCheckpoint

        failure = IncompleteCheckpoint(
            "The model checkpoint is incomplete: shard is missing."
        )
    assert serve_error_class(failure, model_ref=ref) == expected
    lines = startup_error_lines(failure, ref)
    if case in ("unparsed", "complete_corrupt"):
        assert lines is None
        return
    assert lines is not None
    assert ("missing 1 weights" if case == "missing_params" else "incomplete") in lines[
        0
    ]
    if case == "hub":
        assert "rapid-mlx rm owner/checkpoint" in lines[1]
        assert "all revisions" in lines[1]
        assert "rapid-mlx pull owner/checkpoint" in lines[1]
    elif case != "missing_params":
        assert "this directory" in lines[1]
        assert "rapid-mlx rm" not in lines[1]
    _print_model_load_error(SimpleNamespace(model=ref), failure)
    rendered = capsys.readouterr().err
    assert f"Cause: {lines[0]}" in rendered
    assert f"Next step: {lines[1]}" in rendered
    assert "Traceback" not in rendered


@pytest.mark.parametrize("operation", ["directory", "glob", "header"])
def test_checkpoint_integrity_io_failures(tmp_path, monkeypatch, operation):
    from pathlib import Path

    from rapid_mlx.checkpoint_integrity import incomplete_checkpoint_reason

    _write_tiny_weights(tmp_path / "model.safetensors")
    if operation == "directory":

        def unavailable(_self):
            raise OSError("directory unavailable")

        monkeypatch.setattr(Path, "is_dir", unavailable)
        assert incomplete_checkpoint_reason(tmp_path) is None
    elif operation == "glob":

        def unavailable(_self, _pattern):
            raise OSError("directory unavailable")

        monkeypatch.setattr(Path, "glob", unavailable)
        assert incomplete_checkpoint_reason(tmp_path) is None
    else:
        original = Path.open

        def unavailable(self, *args, **kwargs):
            if self.suffix == ".safetensors":
                raise OSError("file unavailable")
            return original(self, *args, **kwargs)

        monkeypatch.setattr(Path, "open", unavailable)
        assert "unreadable" in incomplete_checkpoint_reason(tmp_path)
