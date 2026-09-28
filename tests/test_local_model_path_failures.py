# SPDX-License-Identifier: Apache-2.0
"""Missing local checkpoints stay distinct from Hub download failures."""

from __future__ import annotations

import json

import httpx
from huggingface_hub.errors import HfHubHTTPError

from rapid_mlx.local_model_path import (
    MAX_MISSING_LOCAL_MODEL_FILES,
    local_model_failure_message,
    missing_local_model_files,
)
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


def test_existing_directory_without_weights_lists_relative_name(tmp_path):
    model_dir = tmp_path / "private-model"
    model_dir.mkdir()
    exc = FileNotFoundError("missing weights")

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
