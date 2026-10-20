# SPDX-License-Identifier: Apache-2.0
"""Serve preflight rejects incomplete local paths before creating an engine."""

from __future__ import annotations

import errno
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from rapid_mlx.local_model_path import (
    local_serve_preflight_lines,
    validate_local_model_for_serve,
)
from rapid_mlx.local_serve_diagnostics import startup_error_lines
from rapid_mlx.model_load_errors import InvalidModelConfig
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


def test_model_file_argument_is_a_missing_local_directory(tmp_path):
    file = tmp_path / "model.safetensors"
    file.write_bytes(b"weights")
    with pytest.raises(NotADirectoryError) as failure:
        validate_local_model_for_serve(str(file))
    assert serve_error_class(failure.value, model_ref=str(file)) == "local_path_missing"


@pytest.mark.parametrize(
    "filename", ["pytorch_model.bin", "weights.npz", "adapter.safetensors"]
)
def test_text_model_requires_loader_visible_safetensors(tmp_path, filename):
    root = tmp_path / "model"
    root.mkdir()
    (root / "config.json").write_text('{"model_type":"qwen3"}')
    (root / filename).write_bytes(b"data")
    with pytest.raises(FileNotFoundError, match="Missing model weights"):
        validate_local_model_for_serve(str(root))


@pytest.mark.parametrize(
    "model_type",
    ["gemma4", "gemma4_assistant", "gemma4_unified", "embedding_gemma2"],
)
def test_native_loaders_accept_nonmodel_safetensors_name(tmp_path, model_type):
    root = tmp_path / "model"
    root.mkdir()
    (root / "config.json").write_text(json.dumps({"model_type": model_type}))
    (root / "weights.safetensors").write_bytes(b"data")
    validate_local_model_for_serve(str(root))


def test_indexed_native_embedding_accepts_arbitrary_shard_name(tmp_path):
    root = tmp_path / "embedding"
    root.mkdir()
    (root / "config.json").write_text('{"model_type":"embedding_gemma2"}')
    (root / "weights.safetensors").write_bytes(b"data")
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": "weights.safetensors"}})
    )
    validate_local_model_for_serve(str(root))


def test_indexed_native_embedding_accepts_nested_shard(tmp_path):
    root = tmp_path / "embedding"
    (root / "shards").mkdir(parents=True)
    (root / "config.json").write_text('{"model_type":"embedding_gemma2"}')
    (root / "shards" / "weights.safetensors").write_bytes(b"data")
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": "shards/weights.safetensors"}})
    )
    validate_local_model_for_serve(str(root))


def test_vlm_accepts_nonmodel_safetensors_name(tmp_path):
    root = tmp_path / "vision"
    root.mkdir()
    (root / "config.json").write_text('{"model_type":"qwen2_vl","vision_config":{}}')
    (root / "weights.safetensors").write_bytes(b"data")
    validate_local_model_for_serve(str(root))


def test_indexed_text_model_rejects_unloaded_shard_name(tmp_path):
    root = tmp_path / "text"
    root.mkdir()
    (root / "config.json").write_text('{"model_type":"qwen3"}')
    (root / "weights.safetensors").write_bytes(b"data")
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": "weights.safetensors"}})
    )
    with pytest.raises(FileNotFoundError, match="Missing model weights"):
        validate_local_model_for_serve(str(root))


@pytest.mark.parametrize("model_type", [[], {}])
def test_malformed_model_type_reaches_loader_diagnostic(tmp_path, model_type):
    root = tmp_path / "model"
    root.mkdir()
    (root / "config.json").write_text(json.dumps({"model_type": model_type}))
    (root / "model.safetensors").write_bytes(b"data")
    with pytest.raises(InvalidModelConfig, match="model_type"):
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


def test_unrelated_root_adapter_index_does_not_block_text_checkpoint(tmp_path):
    root = tmp_path / "model"
    root.mkdir()
    (root / "config.json").write_text('{"model_type":"qwen3"}')
    (root / "model.safetensors").write_bytes(b"data")
    (root / "adapter.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"adapter": "missing.safetensors"}})
    )
    validate_local_model_for_serve(str(root))


def test_programmatic_server_load_rejects_missing_config_before_engine(tmp_path):
    from rapid_mlx.server import load_model

    root = tmp_path / "model"
    root.mkdir()
    with pytest.raises(FileNotFoundError, match="Missing model config"):
        load_model(str(root))


@pytest.mark.parametrize(
    "layout", ["whisper", "whisper-safetensors", "kokoro", "image", "video"]
)
def test_supported_nontext_layouts_reach_their_loader(tmp_path, layout):
    root = tmp_path / layout
    root.mkdir()
    if layout.startswith("whisper"):
        (root / "config.json").write_text('{"model_type":"whisper"}')
        (
            root / ("weights.npz" if layout == "whisper" else "weights.safetensors")
        ).write_bytes(b"data")
    elif layout == "kokoro":
        (root / "config.json").write_text('{"model_type":"kokoro"}')
        (root / "kokoro-v1_0.safetensors").write_bytes(b"data")
    else:
        if layout == "image":
            (root / "model_index.json").write_text(
                '{"_class_name":"Pipeline","transformer":[]}'
            )
        else:
            (root / "split_model.json").write_text('{"components":["transformer"]}')
        (root / "transformer.safetensors").write_bytes(b"data")
    validate_local_model_for_serve(str(root))


def test_disabled_optional_image_component_needs_no_weights(tmp_path):
    root = tmp_path / "image"
    root.mkdir()
    (root / "model_index.json").write_text(
        '{"_class_name":"Pipeline","transformer":[],"text_encoder_2":null,'
        '"image_encoder":[null,null]}'
    )
    (root / "transformer.safetensors").write_bytes(b"data")
    validate_local_model_for_serve(str(root))


def test_weightless_third_tokenizer_does_not_need_weights(tmp_path):
    root = tmp_path / "image"
    root.mkdir()
    (root / "model_index.json").write_text(
        '{"_class_name":"Pipeline","transformer":[],"tokenizer_3":["transformers","T5TokenizerFast"]}'
    )
    (root / "transformer.safetensors").write_bytes(b"data")
    validate_local_model_for_serve(str(root))


def test_custom_image_component_and_indexed_shards(tmp_path):
    root = tmp_path / "image"
    component = root / "custom_encoder"
    component.mkdir(parents=True)
    (root / "model_index.json").write_text(
        '{"_class_name":"CustomPipeline","custom_encoder":[],"scheduler":[]}'
    )
    (component / "one.safetensors").write_bytes(b"data")
    (component / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"one": "one.safetensors", "two": "two.safetensors"}})
    )
    with pytest.raises(FileNotFoundError, match="Missing model component"):
        validate_local_model_for_serve(str(root))
    (component / "two.safetensors").write_bytes(b"data")
    validate_local_model_for_serve(str(root))


def test_bonsai_manifest_uses_packed_component_layout(tmp_path):
    root = tmp_path / "bonsai"
    root.mkdir()
    (root / "model_index.json").write_text(
        '{"_class_name":"Flux2KleinPipeline",'
        '"transformer-packed-mflux":[],"text_encoder-mlx-4bit":[],"vae":[]}'
    )
    weights = (
        "transformer-packed-mflux/diffusion_pytorch_model.safetensors",
        "text_encoder-mlx-4bit/model.safetensors",
        "vae/diffusion_pytorch_model.safetensors",
    )
    for relative in weights:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"data")
    validate_local_model_for_serve(str(root))
    (root / weights[0]).unlink()
    with pytest.raises(FileNotFoundError, match="Missing model component"):
        validate_local_model_for_serve(str(root))


def test_standard_flux2_klein_manifest_uses_declared_components(tmp_path):
    root = tmp_path / "flux2-klein"
    (root / "transformer").mkdir(parents=True)
    (root / "text_encoder").mkdir()
    (root / "transformer" / "diffusion_pytorch_model.safetensors").write_bytes(b"data")
    (root / "text_encoder" / "model.safetensors").write_bytes(b"data")
    (root / "model_index.json").write_text(
        '{"_class_name":"Flux2KleinPipeline",'
        '"transformer":["mflux","Flux2Transformer"],'
        '"text_encoder":["transformers","Qwen3Model"]}'
    )
    validate_local_model_for_serve(str(root))


def test_image_manifest_settings_are_not_components(tmp_path):
    root = tmp_path / "sdxl"
    (root / "unet").mkdir(parents=True)
    (root / "unet" / "diffusion_pytorch_model.safetensors").write_bytes(b"data")
    (root / "model_index.json").write_text(
        '{"_class_name":"StableDiffusionXLPipeline",'
        '"force_zeros_for_empty_prompt":true,'
        '"unet":["diffusers","UNet2DConditionModel"]}'
    )
    validate_local_model_for_serve(str(root))


@pytest.mark.parametrize("layout", ["image", "video"])
def test_incomplete_nontext_manifest_is_rejected(tmp_path, layout):
    root = tmp_path / layout
    root.mkdir()
    name = "model_index.json" if layout == "image" else "split_model.json"
    (root / name).write_text(
        '{"_class_name":"Pipeline","transformer":[]}'
        if layout == "image"
        else '{"components":["transformer"]}'
    )
    with pytest.raises(FileNotFoundError) as failure:
        validate_local_model_for_serve(str(root))
    cause, step = local_serve_preflight_lines(str(root), failure.value)
    assert "component" in cause
    assert "rapid-mlx serve" in step


def test_invalid_nontext_manifest_has_specific_remedy(tmp_path):
    root = tmp_path / "image"
    root.mkdir()
    (root / "model_index.json").write_text("{}")
    with pytest.raises(InvalidModelConfig) as failure:
        validate_local_model_for_serve(str(root))
    assert serve_error_class(failure.value, model_ref=str(root)) == "invalid_config"
    result = subprocess.run(
        [sys.executable, "-m", "rapid_mlx.cli", "serve", str(root)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 1
    assert "Local model_index.json is invalid" in result.stderr
    assert "Restore model_index.json" in result.stderr


def test_manifest_read_io_error_keeps_filesystem_cause(tmp_path, monkeypatch):
    root = tmp_path / "image"
    root.mkdir()
    manifest = root / "model_index.json"
    manifest.write_text('{"_class_name":"Pipeline"}')
    read_text = Path.read_text

    def fail_manifest_read(path, *args, **kwargs):
        if path == manifest:
            raise OSError(errno.EIO, "Input/output error", str(path))
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fail_manifest_read)
    with pytest.raises(OSError, match="Input/output error") as failure:
        validate_local_model_for_serve(str(root))
    assert serve_error_class(failure.value, model_ref=str(root)) != "local_path_missing"
    cause, step = local_serve_preflight_lines(str(root), failure.value)
    assert "Input/output error" in cause
    assert "Check the disk" in step


def test_auxiliary_params_does_not_hide_incomplete_text_model(tmp_path):
    root = tmp_path / "text"
    root.mkdir()
    (root / "params.json").write_text('{"layers":1}')
    (root / "config.json").write_text('{"model_type":"qwen3"}')
    (root / "sidecar" / "weights.safetensors").parent.mkdir()
    (root / "sidecar" / "weights.safetensors").write_bytes(b"data")
    with pytest.raises(FileNotFoundError, match="Missing model weights"):
        validate_local_model_for_serve(str(root))


def test_incidental_pipeline_manifest_does_not_hide_incomplete_text_model(tmp_path):
    root = tmp_path / "text"
    root.mkdir()
    (root / "model_index.json").write_text(
        '{"_class_name":"Pipeline","transformer":[]}'
    )
    (root / "config.json").write_text('{"model_type":"qwen3"}')
    (root / "transformer.safetensors").write_bytes(b"data")
    with pytest.raises(FileNotFoundError, match="Missing model weights"):
        validate_local_model_for_serve(str(root))


def test_unreadable_filesystem_has_recovery_command(tmp_path):
    root = tmp_path / "model"
    root.mkdir()
    cause, step = local_serve_preflight_lines(
        str(root), OSError(errno.EIO, "Input/output error", str(root))
    )
    assert "Input/output error" in cause
    assert "Check the disk" in step


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


def test_gemma4_ignores_stale_primary_index(tmp_path):
    root = tmp_path / "gemma"
    root.mkdir()
    (root / "config.json").write_text('{"model_type":"gemma4"}')
    (root / "model.safetensors").write_bytes(b"data")
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"unused": "old-shard.safetensors"}})
    )
    validate_local_model_for_serve(str(root))


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


def test_truncated_native_weight_is_corrupt_weights(tmp_path):
    root = tmp_path / "gemma"
    root.mkdir()
    (root / "config.json").write_text('{"model_type":"gemma4"}')
    (root / "weights.safetensors").write_bytes(b"bad")
    error = RuntimeError("[read] Unable to read 5 bytes from file.")
    assert serve_error_class(error, model_ref=str(root)) == "corrupt_weights"


def test_weight_scan_io_error_does_not_hide_original_error(tmp_path, monkeypatch):
    root = tmp_path / "model"
    root.mkdir()
    (root / "config.json").write_text('{"model_type":"qwen3"}')
    (root / "model.safetensors").write_bytes(b"data")
    real_rglob = type(root).rglob

    def broken_rglob(path, pattern):
        if path == root:
            raise OSError(errno.EIO, "disk read failed")
        return real_rglob(path, pattern)

    monkeypatch.setattr(type(root), "rglob", broken_rglob)
    error = RuntimeError("[read] Unable to read 5 bytes from file.")
    assert serve_error_class(error, model_ref=str(root)) == "other"


def test_bad_numeric_config_is_actionable(tmp_path):
    root = tmp_path / "model"
    root.mkdir()
    (root / "config.json").write_text('{"model_type":"qwen3","hidden_size":"wrong"}')
    (root / "model.safetensors").write_bytes(b"weights")
    with pytest.raises(InvalidModelConfig, match="hidden_size") as failure:
        validate_local_model_for_serve(str(root))
    error = failure.value
    assert serve_error_class(error, model_ref=str(root)) == "invalid_config"
    cause, step = startup_error_lines(error, str(root))
    assert "hidden_size" in cause
    assert "config.json" in step


def test_unrelated_type_error_does_not_claim_bad_numeric_config(tmp_path):
    root = tmp_path / "model"
    root.mkdir()
    (root / "config.json").write_text('{"model_type":"qwen3","hidden_size":"wrong"}')
    (root / "model.safetensors").write_bytes(b"weights")
    error = TypeError("unrelated loader failure")
    assert serve_error_class(error, model_ref=str(root)) != "invalid_config"


def test_unrelated_mlx_core_error_does_not_claim_metal_failure():
    assert serve_error_class(RuntimeError("mlx.core shape mismatch")) == "other"


def test_metal_out_of_memory_uses_memory_remedy():
    error = RuntimeError("Metal allocation failed: out of memory")
    assert serve_error_class(error) == "insufficient_memory"


def test_sync_cli_prints_bad_numeric_config_remedy(tmp_path, capsys):
    from rapid_mlx.cli import _print_model_load_error

    root = tmp_path / "model"
    root.mkdir()
    (root / "config.json").write_text('{"model_type":"qwen3","hidden_size":"wrong"}')
    (root / "model.safetensors").write_bytes(b"weights")
    with pytest.raises(InvalidModelConfig) as failure:
        validate_local_model_for_serve(str(root))
    _print_model_load_error(
        SimpleNamespace(model=str(root)),
        failure.value,
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


@pytest.mark.parametrize(
    ("manifest_name", "contents", "error", "remedy"),
    [
        (
            "split_model.json",
            '{"components":[]}',
            "Missing model components",
            "split_model.json",
        ),
        (
            "split_model.json",
            '{"components":["../escape"]}',
            "Invalid model component",
            "split_model.json",
        ),
        (
            "model_index.json",
            '{"transformer":[]}',
            "Invalid model manifest",
            "model_index.json",
        ),
        (
            "model_index.json",
            '{"_class_name":"Pipeline"}',
            "Missing model components",
            "model_index.json",
        ),
        (
            "model_index.json",
            '{"_class_name":"Pipeline","bad.name":[]}',
            "Invalid model component",
            "model_index.json",
        ),
        (
            "model_index.json",
            '{"_class_name":"Pipeline","transformer":[]}',
            "Missing model component",
            "transformer",
        ),
        ("params.json", '{"layers":1}', "Missing model weights", "model weights"),
    ],
)
def test_incomplete_layouts_identify_cause_and_repair(
    tmp_path, manifest_name, contents, error, remedy
):
    root = tmp_path / "model"
    root.mkdir()
    (root / manifest_name).write_text(contents)
    with pytest.raises(
        FileNotFoundError if error != "Invalid model manifest" else InvalidModelConfig,
        match=error,
    ) as failure:
        validate_local_model_for_serve(str(root))
    if isinstance(failure.value, FileNotFoundError):
        cause, step = local_serve_preflight_lines(str(root), failure.value)
        assert remedy in cause or remedy in step
        assert "rapid-mlx serve" in step


def test_manifest_permissions_and_invalid_json(tmp_path, monkeypatch):
    root = tmp_path / "model"
    root.mkdir()
    manifest = root / "split_model.json"
    manifest.write_text("{broken")
    with pytest.raises(InvalidModelConfig, match="split_model.json"):
        validate_local_model_for_serve(str(root))
    monkeypatch.setattr(
        "rapid_mlx.local_model_path.os.access", lambda path, mode: path != manifest
    )
    with pytest.raises(PermissionError, match="manifest") as failure:
        validate_local_model_for_serve(str(root))
    assert "chmod -R u+rX" in local_serve_preflight_lines(str(root), failure.value)[1]


def test_pipeline_params_needs_root_weights(tmp_path):
    root = tmp_path / "model"
    root.mkdir()
    (root / "params.json").write_text('{"layers":1}')
    (root / "model.safetensors").write_bytes(b"weights")
    validate_local_model_for_serve(str(root))


def test_config_read_race_still_checks_weights(tmp_path, monkeypatch):
    root = tmp_path / "model"
    root.mkdir()
    config = root / "config.json"
    config.write_text('{"model_type":"qwen3"}')
    original = Path.read_text

    def read_once(path, *args, **kwargs):
        if path == config:
            raise OSError(errno.EIO, "config disappeared")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_once)
    with pytest.raises(FileNotFoundError, match="Missing model weights"):
        validate_local_model_for_serve(str(root))


def test_model_file_config_without_model_type_needs_weights(tmp_path):
    root = tmp_path / "model"
    root.mkdir()
    (root / "config.json").write_text('{"model_file":"model.py"}')
    with pytest.raises(FileNotFoundError, match="Missing model weights"):
        validate_local_model_for_serve(str(root))


def test_embedding_index_without_weight_map_is_incomplete(tmp_path):
    root = tmp_path / "model"
    root.mkdir()
    (root / "config.json").write_text('{"model_type":"embedding_gemma2"}')
    (root / "model.safetensors.index.json").write_text("{}")
    with pytest.raises(FileNotFoundError, match="Missing model weights"):
        validate_local_model_for_serve(str(root))


def test_numeric_diagnostic_requires_local_config(tmp_path):
    from rapid_mlx.local_serve_diagnostics import malformed_numeric_config

    assert (
        malformed_numeric_config(
            InvalidModelConfig("hidden_size must be an integer"),
            str(tmp_path / "absent"),
        )
        is None
    )


def test_runtime_insufficient_memory_class():
    assert (
        serve_error_class(RuntimeError("insufficient memory to load model"))
        == "insufficient_memory"
    )


@pytest.mark.parametrize(
    ("files", "cause", "step"),
    [
        ({}, "missing config.json", "Restore config.json"),
        (
            {"config.json": '{"model_type":"qwen3"}'},
            "missing model weights",
            "Restore the checkpoint weights",
        ),
        (
            {"split_model.json": '{"components":[]}'},
            "split_model.json",
            "Restore split_model.json",
        ),
        (
            {"model_index.json": '{"_class_name":"Pipeline"}'},
            "model_index.json",
            "Restore model_index.json",
        ),
        ({"params.json": "{broken"}, "params.json", "Restore params.json"),
        (
            {"config.json": '{"model_type":"qwen3","hidden_size":"bad"}'},
            "nonnumeric hidden_size",
            "Correct hidden_size",
        ),
    ],
)
def test_cli_incomplete_local_layout_exits_with_repair(
    tmp_path, monkeypatch, capsys, files, cause, step
):
    from rapid_mlx import cli

    monkeypatch.delenv("RAPID_MLX_LOG_FILE", raising=False)
    root = tmp_path / "model"
    root.mkdir()
    for name, contents in files.items():
        (root / name).write_text(contents)
    monkeypatch.setattr(sys, "argv", ["rapid-mlx", "serve", str(root)])
    with pytest.raises(SystemExit) as exited:
        cli.main()
    assert exited.value.code == 1
    rendered = capsys.readouterr().err
    assert "Cause: " in rendered and cause in rendered
    assert "Next step: " in rendered and step in rendered


def test_cli_resolution_io_error_exits_with_resolution_error(
    tmp_path, monkeypatch, capsys
):
    from rapid_mlx import cli, model_aliases

    monkeypatch.delenv("RAPID_MLX_LOG_FILE", raising=False)
    root = tmp_path / "model"
    root.mkdir()

    def fail_resolution(_ref):
        raise OSError(errno.EIO, "Input/output error", str(root))

    monkeypatch.setattr(model_aliases, "resolve_model", fail_resolution)
    monkeypatch.setattr(sys, "argv", ["rapid-mlx", "serve", str(root)])
    with pytest.raises(SystemExit) as exited:
        cli.main()
    assert exited.value.code == 1
    rendered = capsys.readouterr().err
    assert "Error: Model resolution failed" in rendered
    assert "Input/output error" in rendered


def test_invalid_config_after_pipeline_manifest_reaches_typed_error(tmp_path):
    root = tmp_path / "model"
    root.mkdir()
    (root / "model_index.json").write_text('{"_class_name":"Pipeline"}')
    (root / "config.json").write_text("{broken")
    with pytest.raises(InvalidModelConfig, match="Invalid model config"):
        validate_local_model_for_serve(str(root))


def test_cli_invalid_config_has_fallback_repair(tmp_path, monkeypatch, capsys):
    from rapid_mlx import cli

    monkeypatch.delenv("RAPID_MLX_LOG_FILE", raising=False)
    root = tmp_path / "model"
    root.mkdir()
    (root / "config.json").write_text("{broken")
    monkeypatch.setattr(sys, "argv", ["rapid-mlx", "serve", str(root)])
    with pytest.raises(SystemExit) as exited:
        cli.main()
    assert exited.value.code == 1
    rendered = capsys.readouterr().err
    assert "Cause: Local config.json is invalid." in rendered
    assert "Next step: Correct config.json" in rendered


@pytest.mark.parametrize("valid_target", [True, False])
def test_cli_preflight_log_target_handling(tmp_path, monkeypatch, capsys, valid_target):
    from rapid_mlx import cli, log_file

    monkeypatch.delenv("RAPID_MLX_LOG_FILE", raising=False)
    root = tmp_path / "model"
    root.mkdir()
    target = (
        tmp_path / "serve.log" if valid_target else tmp_path / "missing" / "serve.log"
    )
    applied = []
    monkeypatch.setattr(
        log_file, "apply", lambda path, source: applied.append((path, source))
    )
    monkeypatch.setattr(
        sys, "argv", ["rapid-mlx", "serve", str(root), "--log-file", str(target)]
    )
    with pytest.raises(SystemExit) as exited:
        cli.main()
    assert exited.value.code == (1 if valid_target else 2)
    if valid_target:
        assert applied == [(str(target), "--log-file")]
        assert "Cause: " in capsys.readouterr().err
    else:
        assert not applied
        assert "error: --log-file" in capsys.readouterr().err


def test_cli_manifest_io_error_has_disk_repair(tmp_path, monkeypatch, capsys):
    from rapid_mlx import cli

    monkeypatch.delenv("RAPID_MLX_LOG_FILE", raising=False)
    root = tmp_path / "model"
    root.mkdir()
    manifest = root / "split_model.json"
    manifest.write_text('{"components":["transformer"]}')
    read_text = Path.read_text

    def fail_manifest_read(path, *args, **kwargs):
        if path == manifest:
            raise OSError(errno.EIO, "Input/output error", str(path))
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fail_manifest_read)
    monkeypatch.setattr(sys, "argv", ["rapid-mlx", "serve", str(root)])
    with pytest.raises(SystemExit) as exited:
        cli.main()
    assert exited.value.code == 1
    rendered = capsys.readouterr().err
    assert "Cause: The filesystem could not read" in rendered
    assert "Next step: Check the disk" in rendered


def test_cli_chat_missing_local_model_keeps_nonserve_error(
    tmp_path, monkeypatch, capsys
):
    from rapid_mlx import cli

    monkeypatch.delenv("RAPID_MLX_LOG_FILE", raising=False)
    missing = tmp_path / "absent"
    monkeypatch.setattr(sys, "argv", ["rapid-mlx", "chat", str(missing)])
    with pytest.raises(SystemExit) as exited:
        cli.main()
    assert exited.value.code == 1
    rendered = capsys.readouterr().err
    assert "Error: " in rendered and str(missing) in rendered
