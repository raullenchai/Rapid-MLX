# SPDX-License-Identifier: Apache-2.0
"""Build a validated file-backed Qwen4 PLE sidecar from a local checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import time
from pathlib import Path

import numpy as np

from rapid_mlx.models.qwen4_ple_sidecar import _source_refs, validate_artifact


def _source_tensor_rows(model_path: Path, weight_map: dict, tensor: str) -> int:
    source = (model_path / weight_map[tensor]).resolve()
    contained = source.is_relative_to(model_path)
    if not contained and model_path.parent.name == "snapshots":
        contained = source.is_relative_to(
            (model_path.parent.parent / "blobs").resolve()
        )
    if not contained:
        raise ValueError("PLE source tensor path escapes model repository")
    with source.open("rb") as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise ValueError("truncated safetensors header length")
        size = struct.unpack("<Q", raw)[0]
        if size > min(source.stat().st_size - 8, 64 * 1024**2):
            raise ValueError("invalid safetensors header length")
        info = json.loads(stream.read(size)).get(tensor, {})
    shape = info.get("shape", [])
    if len(shape) != 2 or isinstance(shape[0], bool) or not isinstance(shape[0], int):
        raise ValueError("invalid PLE source tensor shape")
    return shape[0]


def build_sidecar(
    model_path,
    output_path,
    *,
    chunk_rows: int = 131_072,
    overwrite: bool = False,
    validation_rows: int = 256,
) -> dict:
    """Stream source tensors into an interleaved, validated PLE sidecar."""
    if (
        isinstance(chunk_rows, bool)
        or not isinstance(chunk_rows, int)
        or chunk_rows < 1
    ):
        raise ValueError("chunk_rows must be a positive integer")
    model_path = Path(model_path).resolve()
    output_path = Path(output_path).resolve()
    manifest_path = Path(str(output_path) + ".manifest.json")
    partial_path = Path(str(output_path) + ".partial")
    partial_manifest = Path(str(partial_path) + ".manifest.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    candidates = (output_path, manifest_path, partial_path, partial_manifest)
    existing = [path for path in candidates if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"refusing to replace existing path: {existing[0]}")
    if overwrite:
        partial_path.unlink(missing_ok=True)
        partial_manifest.unlink(missing_ok=True)

    config = json.loads((model_path / "config.json").read_text())
    text = config.get("text_config", config)
    layer_ids = text.get("ple_layer_ids", [])
    if config.get("model_type") != "qwen4_exp" or len(layer_ids) != 1:
        raise ValueError("builder requires exactly one Qwen4 PLE layer")
    layer = layer_ids[0] - 1
    heads = (text.get("ngram_size", 3) - 1) * text.get("heads_per_ngram", 8)
    embed_dim = text.get("ple_embed_dim")
    shards = text.get("split_ngram_parts", 128)
    if (
        not isinstance(embed_dim, int)
        or isinstance(shards, bool)
        or not isinstance(shards, int)
        or shards < 1
        or heads <= 0
        or embed_dim % heads
    ):
        raise ValueError("invalid PLE configuration")
    dims = embed_dim // heads
    if dims <= 0 or dims % 32:
        raise ValueError("PLE dimensions must be a positive multiple of32")
    source_prefix = (
        f"model.language_model.layers.{layer}.ple.ple_embedding.ngram_embedding"
    )
    runtime_prefix = (
        f"language_model.model.layers.{layer}.ple.ple_embedding.ngram_embedding"
    )
    index_bytes = (model_path / "model.safetensors.index.json").read_bytes()
    weight_map = json.loads(index_bytes).get("weight_map", {})
    first_tensor = f"{source_prefix}.shard_0.weight"
    if first_tensor not in weight_map:
        raise ValueError("PLE source index is missing shard_0.weight")
    rows = _source_tensor_rows(model_path, weight_map, first_tensor)
    row_bytes = dims // 2 + dims // 8
    expected_bytes = rows * shards * row_bytes
    if shutil.disk_usage(output_path.parent).free < expected_bytes:
        raise OSError(f"insufficient free space for {expected_bytes} sidecar bytes")

    manifest = {
        "format": "qwen4-ple-rows",
        "version": 1,
        "tensor_prefix": runtime_prefix,
        "dims": dims,
        "group_size": 32,
        "bits": 4,
        "mode": "affine",
        "weight_bytes": dims // 2,
        "scales_bytes": dims // 16,
        "biases_bytes": dims // 16,
        "row_bytes": row_bytes,
        "num_shards": shards,
        "rows_per_shard": rows,
        "total_rows": rows * shards,
        "data_offset": 0,
        "source_index_sha256": hashlib.sha256(index_bytes).hexdigest(),
        "shard_sha256": ["0" * 64] * shards,
    }
    refs = _source_refs(model_path, manifest, weight_map, source_prefix)
    handles: dict[Path, int] = {}
    started = time.monotonic()
    try:
        with partial_path.open("wb", buffering=16 * 1024**2) as destination:
            for shard in range(shards):
                digest = hashlib.sha256()
                for begin in range(0, rows, chunk_rows):
                    count = min(chunk_rows, rows - begin)
                    packed = np.empty((count, row_bytes), dtype=np.uint8)
                    cursor = 0
                    for part in ("weight", "scales", "biases"):
                        path, start, width = refs[shard, part]
                        if path not in handles:
                            handles[path] = os.open(path, os.O_RDONLY)
                        raw = os.pread(
                            handles[path], count * width, start + begin * width
                        )
                        if len(raw) != count * width:
                            raise OSError(
                                f"short PLE source read: shard={shard} part={part}"
                            )
                        packed[:, cursor : cursor + width] = np.frombuffer(
                            raw, dtype=np.uint8
                        ).reshape(count, width)
                        cursor += width
                    blob = packed.tobytes()
                    destination.write(blob)
                    digest.update(blob)
                manifest["shard_sha256"][shard] = digest.hexdigest()
            destination.flush()
            os.fsync(destination.fileno())
        if partial_path.stat().st_size != expected_bytes:
            raise OSError("PLE sidecar output size mismatch")
        with partial_manifest.open("w") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        receipt = validate_artifact(
            model_path, partial_path, random_rows=validation_rows
        )
        os.replace(partial_path, output_path)
        os.replace(partial_manifest, manifest_path)
        receipt["sidecar"] = str(output_path)
        receipt["elapsed_seconds"] = time.monotonic() - started
        receipt["bytes_written"] = expected_bytes
        return receipt
    except BaseException:
        partial_path.unlink(missing_ok=True)
        partial_manifest.unlink(missing_ok=True)
        raise
    finally:
        for descriptor in handles.values():
            os.close(descriptor)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="local Qwen4 checkpoint path")
    parser.add_argument("--output", required=True, help="output sidecar path")
    parser.add_argument("--chunk-rows", type=int, default=131_072)
    parser.add_argument("--validation-rows", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    receipt = build_sidecar(
        args.model,
        args.output,
        chunk_rows=args.chunk_rows,
        overwrite=args.overwrite,
        validation_rows=args.validation_rows,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    raise SystemExit(main())
