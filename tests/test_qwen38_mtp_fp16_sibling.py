from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

mx = pytest.importorskip("mlx.core")

from scripts.qwen38_mtp_fp16_sibling import (  # noqa: E402
    SUCCESS_MANIFEST,
    checkpoint_shards,
    convert_snapshot,
)


def _write_checkpoint(root: Path) -> None:
    root.mkdir()
    target = {
        "model.layers.0.weight": mx.array([[1.0, -2.0]], dtype=mx.uint32),
        "model.layers.0.scales": mx.array([[0.5, 1.5]], dtype=mx.bfloat16),
        "model.norm.weight": mx.array([0.25, -0.75], dtype=mx.bfloat16),
    }
    mtp = {
        "mtp.layers.0.weight": mx.array([[3, 4]], dtype=mx.uint32),
        "mtp.layers.0.scales": mx.array([[0.125, 0.25]], dtype=mx.bfloat16),
    }
    mx.save_safetensors(
        root / "model-00001-of-00001.safetensors", target, {"format": "mlx"}
    )
    (root / "mtp").mkdir()
    mx.save_safetensors(root / "mtp/model.safetensors", mtp, {"format": "mlx"})
    (root / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "total_size": sum(value.nbytes for value in target.values())
                },
                "weight_map": {
                    name: "model-00001-of-00001.safetensors" for name in target
                },
            }
        )
    )
    (root / "config.json").write_text(
        json.dumps(
            {
                "text_config": {
                    "dtype": "bfloat16",
                    "mamba_ssm_dtype": "float32",
                    "mtp_num_hidden_layers": 1,
                },
                "quantization": {"bits": 4, "group_size": 64},
            }
        )
    )
    (root / "tokenizer_config.json").write_text('{"kind":"synthetic"}\n')


def test_converts_bf16_and_preserves_quantized_and_mtp_tensors(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    _write_checkpoint(source)

    manifest = convert_snapshot(source, output)

    assert manifest["converted_bf16_tensors"] == 3
    assert manifest["preserved_non_bf16_tensors"] == 2
    assert manifest["shards"] == [
        "model-00001-of-00001.safetensors",
        "mtp/model.safetensors",
    ]
    assert json.loads((output / SUCCESS_MANIFEST).read_text()) == manifest

    source_target = mx.load(source / "model-00001-of-00001.safetensors")
    target = mx.load(output / "model-00001-of-00001.safetensors")
    assert target["model.layers.0.scales"].dtype == mx.float16
    assert target["model.norm.weight"].dtype == mx.float16
    assert target["model.layers.0.weight"].dtype == mx.uint32
    assert mx.array_equal(
        source_target["model.layers.0.weight"], target["model.layers.0.weight"]
    ).item()

    source_mtp = mx.load(source / "mtp/model.safetensors")
    target_mtp = mx.load(output / "mtp/model.safetensors")
    assert target_mtp["mtp.layers.0.scales"].dtype == mx.float16
    assert target_mtp["mtp.layers.0.weight"].dtype == mx.uint32
    assert mx.array_equal(
        source_mtp["mtp.layers.0.weight"], target_mtp["mtp.layers.0.weight"]
    ).item()
    digest = hashlib.sha256((output / "mtp/model.safetensors").read_bytes()).hexdigest()
    assert (output / "mtp/model.safetensors.sha256").read_text() == (
        f"{digest}  model.safetensors\n"
    )

    config = json.loads((output / "config.json").read_text())
    assert config["text_config"]["dtype"] == "float16"
    assert config["text_config"]["mamba_ssm_dtype"] == "float32"
    assert config["text_config"]["mtp_num_hidden_layers"] == 1
    assert (output / "tokenizer_config.json").read_text() == '{"kind":"synthetic"}\n'


def test_requires_colocated_mtp_sidecar(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_checkpoint(source)
    (source / "mtp/model.safetensors").unlink()

    with pytest.raises(ValueError, match="MTP sidecar"):
        checkpoint_shards(source)


def test_refuses_existing_or_nested_output(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_checkpoint(source)
    existing = tmp_path / "existing"
    existing.mkdir()

    with pytest.raises(ValueError, match="already exists"):
        convert_snapshot(source, existing)
    with pytest.raises(ValueError, match="must not contain"):
        convert_snapshot(source, source / "output")


def test_rejects_index_path_escape(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_checkpoint(source)
    index_path = source / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    index["weight_map"]["model.norm.weight"] = "../outside.safetensors"
    index_path.write_text(json.dumps(index))

    with pytest.raises(ValueError, match="unsafe checkpoint path"):
        checkpoint_shards(source)


def test_reads_extensionless_hf_cache_blob_symlink(tmp_path: Path) -> None:
    repository = tmp_path / "models--rapid-mlx--synthetic"
    source = repository / "snapshots/revision"
    source.parent.mkdir(parents=True)
    _write_checkpoint(source)
    blobs = repository / "blobs"
    blobs.mkdir()
    shard = source / "model-00001-of-00001.safetensors"
    blob = blobs / "0123456789abcdef"
    shard.replace(blob)
    shard.symlink_to(Path("../../blobs") / blob.name)

    output = tmp_path / "output"
    convert_snapshot(source, output)

    converted = mx.load(output / shard.name)
    assert converted["model.norm.weight"].dtype == mx.float16
