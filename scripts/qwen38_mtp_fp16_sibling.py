#!/usr/bin/env python3
"""Create and verify an FP16 sibling of an MLX safetensors checkpoint.

The converter is intentionally narrow: every BF16 tensor becomes FP16 and
every other tensor is copied value-for-value.  Tensor names, shapes, shard
membership, safetensors metadata, and the model's colocated MTP sidecar are
preserved.  Publication is deliberately a separate operator action.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

SOURCE_REPOSITORY = "rapid-mlx/Qwen3.8-27B-4bit-MTP-MLX"
SOURCE_REVISION = "aa985c29ff5b334cbfdcbbc787d47e66e9d9e456"
TARGET_REPOSITORY = "rapid-mlx/Qwen3.8-27B-4bit-MTP-fp16-MLX"
SUCCESS_MANIFEST = "fp16-conversion.json"


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _validate_paths(source: Path, output: Path) -> tuple[Path, Path]:
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    if not source.is_dir():
        raise ValueError(f"source is not a directory: {source}")
    if output.exists():
        raise ValueError(f"output already exists: {output}")
    if _is_relative_to(output, source) or _is_relative_to(source, output):
        raise ValueError("source and output must not contain one another")
    return source, output


def _load_index(source: Path) -> dict[str, Any]:
    path = source / "model.safetensors.index.json"
    try:
        index = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid or missing safetensors index: {path}") from exc
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("safetensors index has no non-empty weight_map")
    return index


def _safe_relative_file(source: Path, relative: Path) -> Path:
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe checkpoint path: {relative}")
    candidate = source / relative
    if not candidate.is_file():
        raise ValueError(f"checkpoint file is missing: {relative}")
    resolved = candidate.resolve()
    allowed = [source]
    if source.parent.name == "snapshots":
        blobs = source.parent.parent / "blobs"
        if blobs.is_dir():
            allowed.append(blobs.resolve())
    if not any(_is_relative_to(resolved, root) for root in allowed):
        raise ValueError(f"checkpoint file escapes its repository cache: {relative}")
    return resolved


def checkpoint_shards(source: Path) -> list[Path]:
    """Return every indexed target shard plus the required MTP sidecar."""

    index = _load_index(source)
    raw_shards = set(index["weight_map"].values())
    if not all(isinstance(name, str) and name for name in raw_shards):
        raise ValueError("safetensors index contains an invalid shard name")
    relative = {Path(name) for name in raw_shards}
    mtp = Path("mtp/model.safetensors")
    if not (source / mtp).is_file():
        raise ValueError("checkpoint is missing the colocated MTP sidecar")
    relative.add(mtp)
    for path in relative:
        _safe_relative_file(source, path)
    return sorted(relative)


def _iter_auxiliary_files(source: Path) -> Iterator[Path]:
    for path in sorted(source.rglob("*")):
        if not path.is_file() or path.suffix == ".safetensors":
            continue
        relative = path.relative_to(source)
        _safe_relative_file(source, relative)
        yield relative


def _copy_auxiliary_files(source: Path, output: Path) -> None:
    for relative in _iter_auxiliary_files(source):
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(_safe_relative_file(source, relative), destination)


def _rewrite_declared_dtype(config: Any) -> int:
    changed = 0
    if isinstance(config, dict):
        for key, value in config.items():
            if key in {"dtype", "torch_dtype"} and value == "bfloat16":
                config[key] = "float16"
                changed += 1
            else:
                changed += _rewrite_declared_dtype(value)
    elif isinstance(config, list):
        for value in config:
            changed += _rewrite_declared_dtype(value)
    return changed


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_sidecar_checksum(output: Path) -> None:
    sidecar = output / "mtp/model.safetensors"
    digest = hashlib.sha256()
    with sidecar.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    checksum = output / "mtp/model.safetensors.sha256"
    checksum.write_text(f"{digest.hexdigest()}  model.safetensors\n")


def _free_bytes(path: Path) -> int:
    existing = path
    while not existing.exists():
        if existing.parent == existing:
            raise ValueError(f"cannot find an existing output parent for {path}")
        existing = existing.parent
    return shutil.disk_usage(existing).free


def _required_output_bytes(source: Path, shards: list[Path]) -> int:
    source_bytes = sum(
        _safe_relative_file(source, shard).stat().st_size for shard in shards
    )
    # Safetensors headers and copied metadata are small, but retain a useful
    # safety margin so a run fails before producing a partial checkpoint.
    return max(source_bytes + (1 << 30), int(source_bytes * 1.10))


def _array_equal(left: Any, right: Any) -> bool:
    import mlx.core as mx

    return bool(mx.array_equal(left, right).item())


def _convert_shard(source_file: Path, output_file: Path) -> tuple[int, int]:
    import mlx.core as mx

    arrays, metadata = mx.load(source_file, format="safetensors", return_metadata=True)
    converted: dict[str, Any] = {}
    bf16_count = 0
    preserved_count = 0
    for name, value in arrays.items():
        if value.dtype == mx.bfloat16:
            converted[name] = value.astype(mx.float16)
            bf16_count += 1
        else:
            converted[name] = value
            preserved_count += 1

    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_file.with_name(f".{output_file.stem}.partial.safetensors")
    mx.save_safetensors(temporary, converted, metadata=metadata)
    os.replace(temporary, output_file)

    written, written_metadata = mx.load(output_file, return_metadata=True)
    if metadata != written_metadata or set(arrays) != set(written):
        raise RuntimeError(f"metadata or tensor-name mismatch in {output_file.name}")
    for name, source_value in arrays.items():
        target_value = written[name]
        if source_value.shape != target_value.shape:
            raise RuntimeError(f"shape mismatch for {name}")
        if source_value.dtype == mx.bfloat16:
            if target_value.dtype != mx.float16 or not _array_equal(
                source_value.astype(mx.float16), target_value
            ):
                raise RuntimeError(f"BF16 to FP16 mismatch for {name}")
        elif source_value.dtype != target_value.dtype or not _array_equal(
            source_value, target_value
        ):
            raise RuntimeError(f"preserved tensor mismatch for {name}")
    return bf16_count, preserved_count


def convert_snapshot(
    source: Path,
    output: Path,
    *,
    source_repository: str = SOURCE_REPOSITORY,
    source_revision: str = SOURCE_REVISION,
) -> dict[str, Any]:
    """Convert a self-contained Qwen3.8 MTP snapshot and verify every tensor."""

    source, output = _validate_paths(source, output)
    shards = checkpoint_shards(source)
    required = _required_output_bytes(source, shards)
    available = _free_bytes(output)
    if available < required:
        raise ValueError(
            f"insufficient output space: need {required} bytes, have {available} bytes"
        )

    output.mkdir(parents=True)
    _copy_auxiliary_files(source, output)

    config_path = output / "config.json"
    config = json.loads(config_path.read_text())
    changed_dtype_fields = _rewrite_declared_dtype(config)
    if changed_dtype_fields == 0:
        raise RuntimeError("config declares no bfloat16 model dtype")
    _write_json_atomic(config_path, config)

    bf16_count = 0
    preserved_count = 0
    for relative in shards:
        converted, preserved = _convert_shard(
            _safe_relative_file(source, relative), output / relative
        )
        bf16_count += converted
        preserved_count += preserved

    _write_sidecar_checksum(output)

    if bf16_count == 0 or preserved_count == 0:
        raise RuntimeError("checkpoint did not contain both BF16 and preserved tensors")

    manifest: dict[str, Any] = {
        "format": "rapid-mlx-fp16-sibling-v1",
        "source_repository": source_repository,
        "source_revision": source_revision,
        "target_repository": TARGET_REPOSITORY,
        "shards": [path.as_posix() for path in shards],
        "converted_bf16_tensors": bf16_count,
        "preserved_non_bf16_tensors": preserved_count,
        "changed_config_dtype_fields": changed_dtype_fields,
        "verification": "all tensor names, shapes, dtypes, metadata, and values",
    }
    _write_json_atomic(output / SUCCESS_MANIFEST, manifest)
    return manifest


def configure_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-repository", default=SOURCE_REPOSITORY)
    parser.add_argument("--source-revision", default=SOURCE_REVISION)
    return parser


def main() -> None:
    args = configure_parser().parse_args()
    manifest = convert_snapshot(
        args.source,
        args.output,
        source_repository=args.source_repository,
        source_revision=args.source_revision,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
