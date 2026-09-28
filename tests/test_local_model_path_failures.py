# SPDX-License-Identifier: Apache-2.0
"""Missing local checkpoints stay distinct from Hub download failures."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError

from rapid_mlx import cli
from rapid_mlx import local_model_path as local_paths
from rapid_mlx.local_model_path import (
    MAX_MISSING_LOCAL_MODEL_FILES,
    local_model_failure_message,
    missing_local_model_files,
)
from rapid_mlx.model_load_errors import InvalidModelConfig, TokenizerLoadFailed
from rapid_mlx.request import MODEL_LOAD_FAILED_CODE, model_load_error_payload
from rapid_mlx.telemetry.model_events import serve_error_class


def test_nonexistent_local_path_has_honest_class_and_messages(tmp_path):
    model_dir = tmp_path / "private" / "missing-model"
    exc = FileNotFoundError(2, "missing", str(model_dir))

    assert serve_error_class(exc, model_ref=str(model_dir)) == "local_path_missing"
    assert local_model_failure_message(str(model_dir), exc) == (
        "The local model path does not exist."
    )
    cli_message = local_model_failure_message(
        str(model_dir), exc, include_supplied_path=True
    )
    assert str(model_dir) in cli_message


def test_nonexistent_local_path_without_attributable_file_failure_is_generic(tmp_path):
    model_dir = tmp_path / "private" / "missing-model"

    assert local_model_failure_message(str(model_dir), ValueError("bad config")) is None


def test_existing_directory_without_weights_lists_relative_name(tmp_path):
    model_dir = tmp_path / "private-model"
    model_dir.mkdir()
    missing_weight = model_dir / "model.safetensors"
    exc = FileNotFoundError(2, "missing weights", str(missing_weight))

    assert serve_error_class(exc, model_ref=str(model_dir)) == "local_path_missing"
    assert missing_local_model_files(str(model_dir), exc) == ("model.safetensors",)
    payload = model_load_error_payload(exc, model_ref=str(model_dir))
    assert payload["code"] == MODEL_LOAD_FAILED_CODE
    assert payload["message"] == (
        "The local model directory is missing required files: model.safetensors."
    )
    assert str(model_dir) not in payload["message"]


def test_partial_shards_are_relative_sorted_and_bounded(tmp_path):
    model_dir = tmp_path / "secret-model"
    model_dir.mkdir()
    names = [f"model-{index:05d}-of-00008.safetensors" for index in range(1, 9)]
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {str(i): name for i, name in enumerate(names)}}),
        encoding="utf-8",
    )
    (model_dir / names[0]).write_bytes(b"present")
    exc = FileNotFoundError(2, "private", str(model_dir / names[1]))

    missing = missing_local_model_files(str(model_dir), exc)
    assert len(missing) == MAX_MISSING_LOCAL_MODEL_FILES
    assert missing == tuple(names[1 : 1 + MAX_MISSING_LOCAL_MODEL_FILES])
    message = local_model_failure_message(str(model_dir), exc)
    assert all(name in message for name in missing)
    assert names[-1] not in message
    assert str(model_dir) not in message


def test_remote_file_not_found_and_hub_http_error_remain_download_failed(tmp_path):
    missing = FileNotFoundError("missing remote shard")
    assert serve_error_class(missing, model_ref="owner/model") == "download_failed"

    response = httpx.Response(
        503, request=httpx.Request("GET", "https://huggingface.co/owner/model")
    )
    hub_error = HfHubHTTPError("unavailable", response=response)
    assert (
        serve_error_class(hub_error, model_ref=str(tmp_path / "local-looking"))
        == "download_failed"
    )
    local_wrapper = FileNotFoundError("snapshot shard missing")
    local_wrapper.__cause__ = hub_error
    assert (
        serve_error_class(local_wrapper, model_ref=str(tmp_path / "local-looking"))
        == "download_failed"
    )


def test_local_path_helpers_fail_closed_on_hostile_inputs(monkeypatch, tmp_path):
    assert local_paths._safe_relative_name(None) is None
    assert local_paths._safe_relative_name("../secret.safetensors") is None
    assert missing_local_model_files("owner/model") == ()
    assert missing_local_model_files(str(tmp_path / "missing")) == ()

    monkeypatch.setattr(
        local_paths.os.path, "exists", lambda _value: (_ for _ in ()).throw(OSError())
    )
    assert local_paths.is_local_model_ref("plain-name") is True


def test_index_diagnostics_ignore_invalid_and_unreadable_entries(monkeypatch, tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    invalid = model_dir / "a.safetensors.index.json"
    invalid.write_text("not-json", encoding="utf-8")
    assert local_paths._missing_index_shards(model_dir) == []

    invalid.write_text(json.dumps(["not", "a", "mapping"]), encoding="utf-8")
    assert local_paths._missing_index_shards(model_dir) == []

    invalid.write_text(
        json.dumps({"weight_map": {"bad": "../outside.safetensors"}}),
        encoding="utf-8",
    )
    assert local_paths._missing_index_shards(model_dir) == []

    invalid.write_text(
        json.dumps({"weight_map": {"weight": "missing.safetensors"}}),
        encoding="utf-8",
    )
    original_is_file = Path.is_file

    def unreliable_is_file(path):
        if path.name == "missing.safetensors":
            raise OSError
        return original_is_file(path)

    monkeypatch.setattr(Path, "is_file", unreliable_is_file)
    assert local_paths._missing_index_shards(model_dir) == ["missing.safetensors"]


def test_local_diagnostics_handle_filename_causes_and_filesystem_errors(
    monkeypatch, tmp_path
):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "present.safetensors").write_bytes(b"weights")

    relative = FileNotFoundError(2, "missing", "nested/shard.safetensors")
    assert missing_local_model_files(str(model_dir), relative) == (
        "nested/shard.safetensors",
    )
    outside = FileNotFoundError(2, "missing", str(tmp_path / "outside.safetensors"))
    assert missing_local_model_files(str(model_dir), outside) == ()
    assert local_model_failure_message(str(model_dir)) is None

    monkeypatch.setattr(local_paths, "is_local_model_ref", lambda _value: True)
    original_exists = Path.exists

    def unreliable_exists(path):
        if path == model_dir.absolute():
            raise OSError
        return original_exists(path)

    monkeypatch.setattr(Path, "exists", unreliable_exists)
    assert local_model_failure_message(str(model_dir)) is None
    with pytest.raises(FileNotFoundError):
        local_paths.raise_if_missing_local_model(str(model_dir))


def test_index_scan_oserror_is_an_empty_diagnostic(monkeypatch, tmp_path):
    original_rglob = Path.rglob

    def unreliable_rglob(path, pattern):
        if path == tmp_path:
            raise OSError
        return original_rglob(path, pattern)

    monkeypatch.setattr(Path, "rglob", unreliable_rglob)
    assert local_paths._missing_index_shards(tmp_path) == []
    assert missing_local_model_files(str(tmp_path)) == ()


@pytest.mark.parametrize(
    ("failure", "error_class"),
    [
        (InvalidModelConfig("invalid rope_scaling"), "invalid_config"),
        (TokenizerLoadFailed("tokenizer.json is malformed"), "tokenizer_load_failed"),
    ],
)
def test_valid_local_checkpoint_keeps_typed_load_failure(
    tmp_path, capsys, failure, error_class
):
    model_dir = tmp_path / "valid-checkpoint"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"model_type":"test"}')
    (model_dir / "model.safetensors").write_bytes(b"weights")

    assert local_model_failure_message(str(model_dir), failure) is None
    assert serve_error_class(failure, model_ref=str(model_dir)) == error_class
    payload = model_load_error_payload(failure, model_ref=str(model_dir))
    assert payload["message"] == (
        "The model failed to load. Check the model files or choose another model."
    )

    cli._print_model_load_error(SimpleNamespace(model=str(model_dir)), failure)
    captured = capsys.readouterr()
    assert str(failure) in captured.out
    assert "missing required" not in captured.out + captured.err


def test_local_checkpoint_auxiliary_hub_404_keeps_typed_failure(tmp_path, capsys):
    model_dir = tmp_path / "valid-checkpoint"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"model_type":"test"}')
    (model_dir / "model.safetensors").write_bytes(b"weights")
    response = httpx.Response(
        404,
        request=httpx.Request("GET", "https://huggingface.co/owner/auxiliary"),
    )
    failure = RepositoryNotFoundError(
        "auxiliary adapter is unavailable", response=response
    )

    assert local_model_failure_message(str(model_dir), failure) is None
    assert serve_error_class(failure, model_ref=str(model_dir)) == "download_failed"
    cli._print_model_load_error(SimpleNamespace(model=str(model_dir)), failure)
    captured = capsys.readouterr()
    assert "auxiliary adapter is unavailable" in captured.out
    assert "missing required" not in captured.out + captured.err


@pytest.mark.parametrize(
    ("exc", "stream", "expected"),
    [
        (ValueError("404"), "out", "not found on HuggingFace"),
        (ValueError("decoder failed"), "out", "Error loading model: decoder failed"),
    ],
)
def test_cli_model_load_error_remote_messages(exc, stream, expected, capsys):
    cli._print_model_load_error(SimpleNamespace(model="owner/model"), exc)
    captured = capsys.readouterr()
    assert expected in getattr(captured, stream)


def test_cli_model_load_error_local_message_uses_typed_path(tmp_path, capsys):
    missing = tmp_path / "missing-model"
    cli._print_model_load_error(
        SimpleNamespace(model="resolved", _original_alias=str(missing)),
        FileNotFoundError(2, "missing", str(missing)),
    )
    captured = capsys.readouterr()
    assert str(missing) in captured.err
    assert captured.out == ""


def test_cli_main_missing_local_serve_path_exits_one_and_emits_failure(
    monkeypatch, tmp_path, capsys
):
    missing = tmp_path / "missing-model"
    emitted = []
    monkeypatch.setattr(sys, "argv", ["rapid-mlx", "serve", str(missing)])
    monkeypatch.setattr(
        "rapid_mlx.telemetry.model_events.emit_model_serve_failed",
        lambda exc, *, alias_or_path: emitted.append((exc, alias_or_path)),
    )

    with pytest.raises(SystemExit) as caught:
        cli.main()

    assert caught.value.code == 1
    assert emitted and emitted[0][1] == str(missing)
    assert str(missing) in capsys.readouterr().err
