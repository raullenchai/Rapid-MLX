#!/usr/bin/env python3
"""Stream a source GLM-5.3 checkpoint into the RMQ MLX layout.

The converter is fail-closed and processes one tensor at a time. It accepts
BF16/FP16/FP32 or the official E4M3 128x128 block-FP8 layout, but refuses an
already-repacked integer checkpoint because a second quantization pass cannot
restore lost information. RMQ's policy and fusion-domain constraints live in
``vllm_mlx.quantization.glm53_rmq`` and are independently unit tested.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import os
import resource
import shutil
import sys
import tempfile
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from vllm_mlx.quantization.glm53_rmq import (
    TensorDescriptor,
    module_paths,
    mtp_module_paths,
    plan_tensors,
    summarize_plan,
)

DEFAULT_MAX_SHARD_BYTES = 4 * 1024**3
_NP_DTYPES = {
    "F16": np.float16,
    "F32": np.float32,
    "F64": np.float64,
    "I8": np.int8,
    "I16": np.int16,
    "I32": np.int32,
    "I64": np.int64,
    "U8": np.uint8,
    "U16": np.uint16,
    "U32": np.uint32,
    "U64": np.uint64,
    "BOOL": np.bool_,
}
_DTYPE_BYTES = {
    "BOOL": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I8": 1,
    "U8": 1,
    "BF16": 2,
    "F16": 2,
    "I16": 2,
    "U16": 2,
    "F32": 4,
    "I32": 4,
    "U32": 4,
    "F64": 8,
    "I64": 8,
    "U64": 8,
}
_AUX_NAMES = (
    "chat_template.jinja",
    "generation_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
_FP8_BLOCK_SIZE = 128
_FLOAT_SOURCE_DTYPES = frozenset({"BF16", "F16", "F32", "F64"})


def _is_supported_block_fp8(config: dict) -> bool:
    quantization = config.get("quantization_config") or {}
    return (
        isinstance(quantization, dict)
        and quantization.get("quant_method") == "fp8"
        and quantization.get("fmt", "e4m3") == "e4m3"
        and quantization.get("weight_block_size") == [128, 128]
    )


def peak_rss_bytes() -> int:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss if sys.platform == "darwin" else rss * 1024


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _safe_source_file(snapshot: Path, relative: Path) -> Path:
    if relative.is_absolute() or len(relative.parts) != 1:
        raise RuntimeError(f"source contains a non-flat shard path: {relative}")
    candidate = snapshot / relative
    if not candidate.is_file():
        raise RuntimeError(f"missing source file: {relative}")
    resolved = candidate.resolve()
    allowed = [snapshot.resolve()]
    if snapshot.parent.name == "snapshots":
        blobs = snapshot.parent.parent / "blobs"
        if blobs.is_dir():
            allowed.append(blobs.resolve())
    if not any(_inside(resolved, root) or resolved == root for root in allowed):
        raise RuntimeError(f"source file escapes its model cache root: {relative}")
    return resolved


def _read_layout(path: Path) -> tuple[int, dict]:
    with (
        path.open("rb") as handle,
        mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mapped,
    ):
        if len(mapped) < 8:
            raise RuntimeError(f"truncated safetensors file: {path}")
        header_len = int(np.frombuffer(mapped[:8], dtype=np.uint64)[0])
        header = json.loads(mapped[8 : 8 + header_len].decode("utf-8"))
    return header_len, header


def _source_manifest(
    source: Path,
) -> tuple[dict[str, str], dict[str, tuple[int, dict]]]:
    index_path = source / "model.safetensors.index.json"
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text()).get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise RuntimeError("source index has an empty weight_map")
    else:
        files = sorted(source.glob("*.safetensors"))
        if len(files) != 1:
            raise RuntimeError("source needs one safetensors file or a canonical index")
        _, header = _read_layout(_safe_source_file(source, Path(files[0].name)))
        weight_map = {name: files[0].name for name in header if name != "__metadata__"}
        if not weight_map:
            raise RuntimeError("source safetensors file contains no weights")

    layouts: dict[str, tuple[int, dict]] = {}
    resolved_shards: dict[str, Path] = {}
    for name, shard_name in weight_map.items():
        if not isinstance(name, str) or not isinstance(shard_name, str):
            raise RuntimeError("source weight_map must contain string keys and values")
        shard = resolved_shards.get(shard_name)
        if shard is None:
            shard = _safe_source_file(source, Path(shard_name))
            resolved_shards[shard_name] = shard
        if str(shard) not in layouts:
            layouts[str(shard)] = _read_layout(shard)
        if name not in layouts[str(shard)][1]:
            raise RuntimeError(
                f"weight {name!r} is absent from declared shard {shard_name}"
            )
    return dict(weight_map), layouts


def _descriptors(
    source: Path,
) -> tuple[list[TensorDescriptor], dict[str, str], dict[str, tuple[int, dict]]]:
    weight_map, layouts = _source_manifest(source)
    descriptors = []
    resolved_shards: dict[str, Path] = {}
    for name in sorted(weight_map):
        shard_name = weight_map[name]
        shard = resolved_shards.get(shard_name)
        if shard is None:
            shard = _safe_source_file(source, Path(shard_name))
            resolved_shards[shard_name] = shard
        info = layouts[str(shard)][1][name]
        descriptors.append(
            TensorDescriptor(name, tuple(int(v) for v in info["shape"]), info["dtype"])
        )
    return descriptors, weight_map, layouts


def _tensor_bytes(path: Path, header_len: int, header: dict, name: str) -> bytes:
    info = header[name]
    begin = 8 + header_len + int(info["data_offsets"][0])
    end = 8 + header_len + int(info["data_offsets"][1])
    with (
        path.open("rb") as handle,
        mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mapped,
    ):
        return bytes(mapped[begin:end])


def _mlx_from_bytes(data: bytes, dtype: str, shape: tuple[int, ...]) -> mx.array:
    dtype = dtype.upper()
    if dtype == "F8_E4M3":
        return mx.array(np.frombuffer(data, dtype=np.uint8).reshape(shape))
    if dtype == "BF16":
        return mx.array(np.frombuffer(data, dtype=np.uint16).reshape(shape)).view(
            mx.bfloat16
        )
    np_dtype = _NP_DTYPES.get(dtype)
    if np_dtype is None:
        raise RuntimeError(f"unsupported source dtype: {dtype}")
    return mx.array(np.frombuffer(data, dtype=np_dtype).reshape(shape))


def _load_tensor(
    source: Path,
    weight_map: dict[str, str],
    layouts: dict[str, tuple[int, dict]],
    name: str,
) -> mx.array:
    shard = _safe_source_file(source, Path(weight_map[name]))
    header_len, header = layouts[str(shard)]
    info = header[name]
    raw = _tensor_bytes(shard, header_len, header, name)
    try:
        return _mlx_from_bytes(
            raw,
            str(info["dtype"]),
            tuple(int(value) for value in info["shape"]),
        )
    finally:
        del raw


def _logical_source_descriptors(
    packed: list[TensorDescriptor], config: dict
) -> tuple[list[TensorDescriptor], dict[str, str]]:
    """Pair official block-FP8 weights with their inverse-scale grids."""
    if not _is_supported_block_fp8(config):
        return packed, {}

    by_name = {item.name: item for item in packed}
    fp8_scales: dict[str, str] = {}
    logical = []
    for item in packed:
        if item.name.endswith(".weight_scale_inv"):
            weight = item.name[: -len("_scale_inv")]
            if weight not in by_name:
                raise RuntimeError(f"orphan FP8 scale tensor: {item.name}")
            continue
        if item.dtype.upper() == "F8_E4M3":
            if not item.name.endswith(".weight"):
                raise RuntimeError(f"unsupported non-weight FP8 tensor: {item.name}")
            scale = item.name + "_scale_inv"
            scale_desc = by_name.get(scale)
            if scale_desc is None:
                raise RuntimeError(f"missing FP8 inverse scale for {item.name}")
            expected = tuple(
                (value + _FP8_BLOCK_SIZE - 1) // _FP8_BLOCK_SIZE for value in item.shape
            )
            if (
                len(item.shape) != 2
                or scale_desc.shape != expected
                or scale_desc.dtype.upper() not in _FLOAT_SOURCE_DTYPES
            ):
                raise RuntimeError(
                    f"invalid FP8 scale grid for {item.name}: got shape "
                    f"{scale_desc.shape} dtype {scale_desc.dtype}, expected "
                    f"{expected} with a floating dtype"
                )
            logical.append(TensorDescriptor(item.name, item.shape, "BF16"))
            fp8_scales[item.name] = scale
        else:
            logical.append(item)
    return logical, fp8_scales


def _restore_block_fp8(weight: mx.array, scale_inv: mx.array) -> mx.array:
    if weight.dtype != mx.uint8 or weight.ndim != 2:
        raise RuntimeError("block-FP8 weights must be 2D E4M3 byte arrays")
    rows, columns = weight.shape
    expected = (
        (rows + _FP8_BLOCK_SIZE - 1) // _FP8_BLOCK_SIZE,
        (columns + _FP8_BLOCK_SIZE - 1) // _FP8_BLOCK_SIZE,
    )
    if scale_inv.shape != expected:
        raise RuntimeError(
            f"block-FP8 scale shape mismatch: got {scale_inv.shape}, expected {expected}"
        )
    pad_rows = (-rows) % _FP8_BLOCK_SIZE
    pad_columns = (-columns) % _FP8_BLOCK_SIZE
    decoded = mx.from_fp8(weight, dtype=mx.bfloat16)
    if pad_rows or pad_columns:
        decoded = mx.pad(decoded, ((0, pad_rows), (0, pad_columns)))
    decoded = decoded.reshape(
        (rows + pad_rows) // _FP8_BLOCK_SIZE,
        _FP8_BLOCK_SIZE,
        (columns + pad_columns) // _FP8_BLOCK_SIZE,
        _FP8_BLOCK_SIZE,
    )
    decoded = (decoded * scale_inv[:, None, :, None]).reshape(
        rows + pad_rows, columns + pad_columns
    )
    return decoded[:rows, :columns]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _ShardWriter:
    def __init__(self, output: Path, max_bytes: int):
        self.output = output
        self.max_bytes = max_bytes
        self.buffers: list[dict[str, mx.array]] = []
        self.buffer_bytes = 0
        self.files: list[Path] = []
        self.weight_map: dict[str, str] = {}
        self.payload_bytes = 0

    def add(self, tensors: dict[str, mx.array]) -> None:
        size = sum(int(value.nbytes) for value in tensors.values())
        if self.buffers and self.buffer_bytes + size > self.max_bytes:
            self._flush()
        self.buffers.append(tensors)
        self.buffer_bytes += size
        self.payload_bytes += size

    def _flush(self) -> None:
        if not self.buffers:
            return
        path = self.output / f"model-{len(self.files) + 1:05d}-of-00000.safetensors"
        payload: dict[str, mx.array] = {}
        for tensors in self.buffers:
            payload.update(tensors)
        mx.save_safetensors(str(path), payload, metadata={"format": "mlx"})
        self.files.append(path)
        for name in payload:
            self.weight_map[name] = path.name
        self.buffers = []
        self.buffer_bytes = 0

    def finish(self) -> None:
        self._flush()
        count = len(self.files)
        old_to_new = {}
        for i, old in enumerate(self.files, 1):
            new = self.output / f"model-{i:05d}-of-{count:05d}.safetensors"
            old.replace(new)
            old_to_new[old.name] = new.name
        self.weight_map = {
            key: old_to_new[value] for key, value in self.weight_map.items()
        }


def _load_sensitivity(path: Path | None) -> dict[str, float]:
    if path is None:
        return {}
    payload = json.loads(path.read_text())
    scores = payload.get("scores", payload) if isinstance(payload, dict) else None
    if not isinstance(scores, dict):
        raise RuntimeError(
            "sensitivity JSON must be an object or contain a scores object"
        )
    return {str(key): float(value) for key, value in scores.items()}


def inspect(source: Path, sensitivity_path: Path | None = None) -> dict[str, object]:
    source = source.expanduser().resolve()
    config = json.loads((source / "config.json").read_text())
    if config.get("model_type") != "glm5_next":
        raise RuntimeError("RMQ v1 accepts only model_type=glm5_next")
    if (config.get("quantization") or config.get("quantization_config")) and not (
        _is_supported_block_fp8(config)
    ):
        raise RuntimeError("RMQ requires BF16 or supported block-FP8 source weights")
    packed, _, _ = _descriptors(source)
    descriptors, _ = _logical_source_descriptors(packed, config)
    plan = plan_tensors(descriptors, config, _load_sensitivity(sensitivity_path))
    summary = summarize_plan(plan)
    summary.update(
        {
            "source": str(source),
            "source_tensors": len(packed),
            "logical_tensors": len(descriptors),
            "source_payload_bytes": sum(
                item.parameters * _DTYPE_BYTES[item.dtype.upper()] for item in packed
            ),
            "source_format": (
                "block-fp8-e4m3-128x128" if _is_supported_block_fp8(config) else "float"
            ),
        }
    )
    return summary


def inspect_quantized_shapes(
    source: Path, sensitivity_path: Path | None = None
) -> dict[str, object]:
    """Project RMQ storage from packed headers without requantizing payloads."""
    source = source.expanduser().resolve()
    config = json.loads((source / "config.json").read_text())
    quantization = config.get("quantization") or config.get("quantization_config")
    if config.get("model_type") != "glm5_next" or not isinstance(quantization, dict):
        raise RuntimeError("shape projection requires a quantized glm5_next checkpoint")
    group_size = int(quantization.get("group_size") or 0)
    if group_size != 64:
        raise RuntimeError("RMQ shape projection currently requires group_size=64")

    packed, _, _ = _descriptors(source)
    by_name = {item.name: item for item in packed}
    logical = []
    inferred_bits: dict[int, int] = {}
    for item in packed:
        if item.name.endswith((".scales", ".biases")):
            base = item.name.rsplit(".", 1)[0] + ".weight"
            if base in by_name:
                continue
        if item.name.endswith(".weight") and item.dtype.upper() == "U32":
            base = item.name.removesuffix(".weight")
            scales = by_name.get(f"{base}.scales")
            if scales is None or len(scales.shape) != len(item.shape):
                raise RuntimeError(
                    f"packed weight has no compatible scales: {item.name}"
                )
            original_width = scales.shape[-1] * group_size
            bits_numerator = item.shape[-1] * 32
            if not original_width or bits_numerator % original_width:
                raise RuntimeError(f"cannot infer packed width for {item.name}")
            bits = bits_numerator // original_width
            if bits not in (2, 3, 4, 5, 6, 8):
                raise RuntimeError(f"unsupported inferred Q{bits} for {item.name}")
            inferred_bits[bits] = inferred_bits.get(bits, 0) + 1
            shape = (*item.shape[:-1], original_width)
            logical.append(TensorDescriptor(item.name, shape, scales.dtype))
        else:
            logical.append(item)

    planning_config = dict(config)
    planning_config.pop("quantization", None)
    planning_config.pop("quantization_config", None)
    plan = plan_tensors(logical, planning_config, _load_sensitivity(sensitivity_path))
    summary = summarize_plan(plan)
    source_payload_bytes = sum(
        item.parameters * _DTYPE_BYTES[item.dtype.upper()] for item in packed
    )
    summary.update(
        {
            "source": str(source),
            "source_tensors": len(packed),
            "logical_tensors": len(logical),
            "source_payload_bytes": source_payload_bytes,
            "projected_delta_bytes": int(summary["projected_storage_bytes"])
            - source_payload_bytes,
            "inferred_source_quantized_weights": dict(sorted(inferred_bits.items())),
            "projection_only": True,
        }
    )
    return summary


def _write_metadata(
    source: Path,
    output: Path,
    plan,
    writer: _ShardWriter,
    summary: dict[str, object],
) -> None:
    config = json.loads((source / "config.json").read_text())
    quantization: dict[str, object] = {
        "group_size": 64,
        "bits": 4,
        "mode": "affine",
    }
    for item in plan:
        if item.spec.quantized and item.spec.bits != 4:
            paths = (
                *module_paths(item.tensor.name),
                *mtp_module_paths(item.tensor.name, config),
            )
            for path in paths:
                quantization[path] = item.spec.as_config()
    config["quantization"] = quantization
    config["quantization_config"] = quantization
    config["rapid_quantization"] = {
        "method": "rmq-v1",
        "apple_native_formats": ["q4-g64-affine", "q8-g64-affine"],
        "fusion_domain_locked": True,
        "source_format": "bf16-or-block-fp8",
        "requantized_source_rejected": True,
        "plan": summary,
    }
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    for name in _AUX_NAMES:
        candidate = source / name
        if candidate.is_file():
            _safe_source_file(source, Path(name))
            shutil.copyfile(candidate, output / name)
    index = {
        "metadata": {"total_size": writer.payload_bytes},
        "weight_map": dict(sorted(writer.weight_map.items())),
    }
    (output / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2) + "\n"
    )
    sums = []
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            sums.append(f"{_sha256(path)}  {path.name}\n")
    (output / "SHA256SUMS.txt").write_text("".join(sums))


def convert(
    source: Path,
    output: Path,
    *,
    sensitivity_path: Path | None = None,
    max_shard_bytes: int = DEFAULT_MAX_SHARD_BYTES,
    max_rss_bytes: int = 220 * 1024**3,
) -> dict[str, object]:
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    if output.exists():
        raise RuntimeError(f"output already exists: {output}")
    if not (source / "config.json").is_file():
        raise RuntimeError("source has no config.json")
    if max_shard_bytes <= 0 or max_rss_bytes <= 0:
        raise ValueError("shard and RSS limits must be positive")

    config = json.loads((source / "config.json").read_text())
    if config.get("model_type") != "glm5_next":
        raise RuntimeError("RMQ v1 accepts only model_type=glm5_next")
    if (config.get("quantization") or config.get("quantization_config")) and not (
        _is_supported_block_fp8(config)
    ):
        raise RuntimeError(
            "RMQ refuses requantization; provide BF16 or official block-FP8 weights"
        )

    packed, weight_map, layouts = _descriptors(source)
    descriptors, fp8_scales = _logical_source_descriptors(packed, config)
    plan = plan_tensors(descriptors, config, _load_sensitivity(sensitivity_path))
    summary = summarize_plan(plan)
    needed = int(summary["projected_storage_bytes"] * 1.10) + 1024**3
    output.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(output.parent).free
    if free < needed:
        raise RuntimeError(
            f"insufficient output space: {free / 1024**3:.1f} GiB free, "
            f"{needed / 1024**3:.1f} GiB required"
        )

    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent)
    )
    started = time.monotonic()
    writer = _ShardWriter(staging, max_shard_bytes)
    try:
        by_name = {item.tensor.name: item for item in plan}
        for name in sorted(by_name):
            if peak_rss_bytes() > max_rss_bytes:
                raise RuntimeError("RMQ conversion exceeded the configured RSS guard")
            item = by_name[name]
            array = _load_tensor(source, weight_map, layouts, name)
            if name in fp8_scales:
                scale_inv = _load_tensor(source, weight_map, layouts, fp8_scales[name])
                array = _restore_block_fp8(array, scale_inv)
                del scale_inv
            if item.spec.quantized:
                qweight, scales, biases = mx.quantize(
                    array,
                    group_size=int(item.spec.group_size),
                    bits=int(item.spec.bits),
                    mode=str(item.spec.mode),
                )
                mx.eval(qweight, scales, biases)
                base = name.removesuffix(".weight")
                writer.add(
                    {
                        name: qweight,
                        f"{base}.scales": scales,
                        f"{base}.biases": biases,
                    }
                )
            else:
                mx.eval(array)
                writer.add({name: array})
            del array
        writer.finish()
        _write_metadata(source, staging, plan, writer, summary)
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        **summary,
        "source": str(source),
        "output": str(output),
        "output_payload_bytes": writer.payload_bytes,
        "peak_rss_bytes": peak_rss_bytes(),
        "wall_seconds": round(time.monotonic() - started, 3),
        "status": "ok",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--sensitivity", type=Path)
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--inspect-quantized-shapes", action="store_true")
    parser.add_argument("--max-shard-bytes", type=int, default=DEFAULT_MAX_SHARD_BYTES)
    parser.add_argument("--max-rss-gib", type=float, default=220.0)
    args = parser.parse_args()
    if args.inspect_only and args.inspect_quantized_shapes:
        parser.error("choose only one inspection mode")
    if args.inspect_quantized_shapes:
        result = inspect_quantized_shapes(args.source, args.sensitivity)
    elif args.inspect_only:
        result = inspect(args.source, args.sensitivity)
    else:
        if args.output is None:
            parser.error("--output is required unless --inspect-only is used")
        result = convert(
            args.source,
            args.output,
            sensitivity_path=args.sensitivity,
            max_shard_bytes=args.max_shard_bytes,
            max_rss_bytes=int(args.max_rss_gib * 1024**3),
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
