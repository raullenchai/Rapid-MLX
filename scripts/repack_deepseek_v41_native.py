#!/usr/bin/env python3
"""Stream a split-expert DeepSeek V4.1 MLX checkpoint into native layout.

The source checkpoint already contains MLX affine 2-bit tensors. Repacking is
byte-preserving: routed-expert tensors are stacked along a new leading expert
axis and all other quantized tensors retain their stored bytes. This avoids a
second quantization pass and its associated quality loss.

The destination must not exist. Large generated builds belong on the Studio
warm tier, not in or beside the Hugging Face cache.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import struct
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np

_EXPERT_RE = re.compile(
    r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])\."
    r"(weight|scales|biases)$"
)
_LAYER_RE = re.compile(r"^layers\.(\d+)\.")
_ROUTER_RE = re.compile(r"^layers\.(\d+)\.ffn\.gate\.(weight|bias|bias_vl)$")
_DROP_PREFIXES = ("mtp.", "vision.", "aligner.", "image_")
_PROJECTION_NAMES = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}
_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "U16": 2,
    "I16": 2,
    "F16": 2,
    "BF16": 2,
    "U32": 4,
    "I32": 4,
    "F32": 4,
    "U64": 8,
    "I64": 8,
    "F64": 8,
}
_COPY_CHUNK = 64 * 1024 * 1024


@dataclass(frozen=True)
class SourceTensor:
    name: str
    path: Path
    offset: int
    nbytes: int
    dtype: str
    shape: tuple[int, ...]

    @property
    def row_bytes(self) -> int:
        if not self.shape or self.shape[0] <= 0:
            raise ValueError(f"{self.name} cannot be sliced by row")
        if self.nbytes % self.shape[0]:
            raise ValueError(f"{self.name} has non-integral row storage")
        return self.nbytes // self.shape[0]


@dataclass(frozen=True)
class TensorPlan:
    name: str
    dtype: str
    shape: tuple[int, ...]
    sources: tuple[SourceTensor, ...]
    rows: tuple[int, ...] | None = None

    @property
    def nbytes(self) -> int:
        return math.prod(self.shape) * _DTYPE_BYTES[self.dtype]


class CheckpointIndex:
    def __init__(self, root: Path):
        self.root = root
        index_path = root / "model.safetensors.index.json"
        weight_map = json.loads(index_path.read_text())["weight_map"]
        by_shard: dict[str, list[str]] = defaultdict(list)
        for name, shard in weight_map.items():
            by_shard[shard].append(name)

        self.tensors: dict[str, SourceTensor] = {}
        for shard, expected_names in sorted(by_shard.items()):
            path = root / shard
            with path.open("rb") as handle:
                header_len = struct.unpack("<Q", handle.read(8))[0]
                header = json.loads(handle.read(header_len))
            data_start = 8 + header_len
            for name in expected_names:
                meta = header[name]
                begin, end = meta["data_offsets"]
                tensor = SourceTensor(
                    name=name,
                    path=path,
                    offset=data_start + begin,
                    nbytes=end - begin,
                    dtype=meta["dtype"],
                    shape=tuple(meta["shape"]),
                )
                expected = math.prod(tensor.shape) * _DTYPE_BYTES[tensor.dtype]
                if expected != tensor.nbytes:
                    raise ValueError(
                        f"{name}: header stores {tensor.nbytes} bytes, expected {expected}"
                    )
                self.tensors[name] = tensor
        if set(self.tensors) != set(weight_map):
            raise ValueError("checkpoint index and safetensors headers disagree")


def _copy_range(source: SourceTensor, output: BinaryIO) -> None:
    remaining = source.nbytes
    with source.path.open("rb", buffering=0) as handle:
        handle.seek(source.offset)
        while remaining:
            block = handle.read(min(remaining, _COPY_CHUNK))
            if not block:
                raise EOFError(f"short read from {source.path} for {source.name}")
            output.write(block)
            remaining -= len(block)


def _copy_rows(source: SourceTensor, rows: Sequence[int], output: BinaryIO) -> None:
    row_bytes = source.row_bytes
    with source.path.open("rb", buffering=0) as handle:
        for row in rows:
            if row < 0 or row >= source.shape[0]:
                raise IndexError(f"row {row} outside {source.name} {source.shape}")
            handle.seek(source.offset + row * row_bytes)
            remaining = row_bytes
            while remaining:
                block = handle.read(min(remaining, _COPY_CHUNK))
                if not block:
                    raise EOFError(f"short row read from {source.name}")
                output.write(block)
                remaining -= len(block)


def write_safetensors(path: Path, plans: Sequence[TensorPlan]) -> None:
    offset = 0
    header: dict[str, object] = {"__metadata__": {"format": "mlx"}}
    for plan in plans:
        header[plan.name] = {
            "dtype": plan.dtype,
            "shape": list(plan.shape),
            "data_offsets": [offset, offset + plan.nbytes],
        }
        offset += plan.nbytes

    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * ((-len(encoded)) % 8)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("xb") as output:
        output.write(struct.pack("<Q", len(encoded)))
        output.write(encoded)
        for plan in plans:
            before = output.tell()
            if plan.rows is not None:
                if len(plan.sources) != 1:
                    raise ValueError(f"row plan {plan.name} must have one source")
                _copy_rows(plan.sources[0], plan.rows, output)
            else:
                for source in plan.sources:
                    _copy_range(source, output)
            written = output.tell() - before
            if written != plan.nbytes:
                raise ValueError(
                    f"{plan.name}: wrote {written} bytes, expected {plan.nbytes}"
                )
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


def _selection_from_saliency(
    path: Path | None, layers: int, experts: int, keep: int
) -> dict[int, tuple[int, ...]]:
    if keep == experts:
        return {layer: tuple(range(experts)) for layer in range(layers)}
    if path is None:
        raise ValueError("--saliency is required when pruning experts")
    data = np.load(path)
    score_key = "robust_saliency" if "robust_saliency" in data else "saliency"
    saliency = np.asarray(data[score_key])
    if saliency.shape != (layers, experts):
        raise ValueError(
            f"saliency shape {saliency.shape} != expected {(layers, experts)}"
        )
    return {
        layer: tuple(sorted(np.argsort(-saliency[layer])[:keep].tolist()))
        for layer in range(layers)
    }


def build_plans(
    checkpoint: CheckpointIndex,
    *,
    layer_count: int,
    expert_count: int,
    selection: dict[int, tuple[int, ...]],
    selected_layers: set[int] | None = None,
) -> dict[str, list[TensorPlan]]:
    ordinary: dict[str, SourceTensor] = {}
    expert_parts: dict[tuple[int, str, str], dict[int, SourceTensor]] = defaultdict(
        dict
    )

    for name, source in checkpoint.tensors.items():
        if name.startswith(_DROP_PREFIXES):
            continue
        layer_match = _LAYER_RE.match(name)
        if selected_layers is not None:
            if layer_match is None or int(layer_match.group(1)) not in selected_layers:
                continue
        match = _EXPERT_RE.match(name)
        if match:
            layer, expert, projection, suffix = match.groups()
            expert_parts[(int(layer), projection, suffix)][int(expert)] = source
        else:
            ordinary[name] = source

    groups: dict[str, list[TensorPlan]] = defaultdict(list)
    for name, source in ordinary.items():
        layer_match = _LAYER_RE.match(name)
        group = f"layer-{int(layer_match.group(1)):02d}" if layer_match else "top"
        router_match = _ROUTER_RE.match(name)
        rows = None
        shape = source.shape
        if router_match:
            layer = int(router_match.group(1))
            rows = selection[layer]
            shape = (len(rows), *shape[1:])
        groups[group].append(
            TensorPlan(name, source.dtype, shape, (source,), rows=rows)
        )

    for (layer, projection, suffix), parts in expert_parts.items():
        if set(parts) != set(range(expert_count)):
            missing = sorted(set(range(expert_count)) - set(parts))
            raise ValueError(
                f"layer {layer} {projection}.{suffix} missing experts {missing[:8]}"
            )
        chosen = selection[layer]
        sources = tuple(parts[expert] for expert in chosen)
        exemplar = sources[0]
        if any(
            (item.dtype, item.shape) != (exemplar.dtype, exemplar.shape)
            for item in sources
        ):
            raise ValueError(
                f"inconsistent expert tensors for layer {layer} {projection}"
            )
        name = f"layers.{layer}.ffn.experts.{_PROJECTION_NAMES[projection]}.{suffix}"
        groups[f"layer-{layer:02d}"].append(
            TensorPlan(
                name,
                exemplar.dtype,
                (len(sources), *exemplar.shape),
                sources,
            )
        )

    expected_layers: Iterable[int]
    if selected_layers is None:
        expected_layers = range(layer_count)
    else:
        expected_layers = selected_layers
    for layer in expected_layers:
        if f"layer-{layer:02d}" not in groups:
            raise ValueError(f"no tensors planned for layer {layer}")
    for plans in groups.values():
        plans.sort(key=lambda plan: plan.name)
    return dict(groups)


def _module_map(groups: dict[str, list[TensorPlan]], bits: int, group_size: int):
    names = {plan.name for plans in groups.values() for plan in plans}
    modules = {}
    for name in sorted(names):
        if not name.endswith(".scales") or ".engram.embed." in name:
            continue
        base = name.removesuffix(".scales")
        if f"{base}.weight" in names:
            modules[base] = {"group_size": group_size, "bits": bits}
    return modules


def _copy_metadata_files(source: Path, destination: Path) -> None:
    allowed = {
        "LICENSE",
        "chat_template.jinja",
        "generation_config.json",
        "preprocessor_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    }
    for name in allowed:
        src = source / name
        if src.is_file():
            shutil.copy2(src, destination / name)


def _write_model_card(destination: Path, metadata: dict, tensor_bytes: int) -> None:
    """Prevent a derived artifact from inheriting incorrect source claims."""
    original = metadata["original_experts"]
    kept = metadata["kept_experts"]
    (destination / "README.md").write_text(
        "# Experimental DeepSeek V4.1 native MLX repack\n\n"
        "This local text-only artifact is a byte-preserving MLX affine 2-bit "
        "repack. It is **not product-qualified and must not be published or "
        "cataloged without separate quality, performance, and license review**.\n\n"
        f"- Tensor bytes: {tensor_bytes:,}\n"
        f"- Routed experts kept: {kept} of {original}\n"
        "- Vision: omitted\n"
        "- MTP: omitted\n"
        "- Runtime: Rapid-MLX experimental native V4.1 loader\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--keep-experts", type=int, default=None)
    parser.add_argument("--saliency", type=Path)
    parser.add_argument(
        "--layers",
        help="comma-separated layer ids for a small parity fixture; default is all",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    destination = args.destination.resolve()
    if not source.is_dir():
        raise SystemExit(f"source checkpoint does not exist: {source}")
    if destination.exists():
        raise SystemExit(f"destination already exists: {destination}")

    config = json.loads((source / "config.json").read_text())
    text_config = config.get("text_config", config)
    layer_count = int(text_config["num_hidden_layers"])
    expert_count = int(text_config["n_routed_experts"])
    keep = args.keep_experts or expert_count
    if not 1 <= keep <= expert_count:
        raise SystemExit(f"--keep-experts must be in [1, {expert_count}]")
    selected_layers = (
        {int(item) for item in args.layers.split(",")} if args.layers else None
    )
    if selected_layers and not selected_layers <= set(range(layer_count)):
        raise SystemExit("--layers contains an out-of-range layer")

    q = config.get("quantization", config.get("quantization_config", {}))
    bits = int(q.get("bits", 0))
    group_size = int(q.get("group_size", 0))
    if (bits, group_size, q.get("mode", "affine")) != (2, 64, "affine"):
        raise SystemExit(
            "source must be the validated affine 2-bit/group-64 checkpoint"
        )

    selection = _selection_from_saliency(args.saliency, layer_count, expert_count, keep)
    checkpoint = CheckpointIndex(source)
    groups = build_plans(
        checkpoint,
        layer_count=layer_count,
        expert_count=expert_count,
        selection=selection,
        selected_layers=selected_layers,
    )
    total_bytes = sum(plan.nbytes for plans in groups.values() for plan in plans)
    print(
        json.dumps(
            {
                "source": str(source),
                "destination": str(destination),
                "layers": sorted(selected_layers) if selected_layers else "all",
                "experts": {"source": expert_count, "kept": keep},
                "tensor_bytes": total_bytes,
                "tensor_gb": round(total_bytes / 1e9, 3),
                "shards": len(groups),
                "dry_run": args.dry_run,
            },
            indent=2,
        )
    )
    if args.dry_run:
        return

    destination.mkdir(parents=True)
    shard_names: dict[str, str] = {}
    weight_map: dict[str, str] = {}
    ordered_groups = sorted(groups, key=lambda item: (item == "top", item))
    for index, group in enumerate(ordered_groups, 1):
        filename = f"model-{index:05d}-of-{len(groups):05d}.safetensors"
        shard_names[group] = filename
        print(f"[{index}/{len(groups)}] {group} -> {filename}", flush=True)
        write_safetensors(destination / filename, groups[group])
        weight_map.update({plan.name: filename for plan in groups[group]})

    output_config = json.loads(json.dumps(config))
    output_config["model_type"] = "deepseek_v41"
    output_config["n_routed_experts"] = keep
    if "text_config" in output_config:
        output_config["text_config"]["n_routed_experts"] = keep
    output_config["quantization"] = {
        "group_size": group_size,
        "bits": bits,
        "mode": "affine",
        "expert_bits": bits,
        "engram_bits": bits,
        "modules": _module_map(groups, bits, group_size),
    }
    output_config["rapid_quantization"] = {
        "format": "native-packed-affine",
        "source_bits": bits,
        "group_size": group_size,
        "byte_preserving_repack": True,
        "original_experts": expert_count,
        "kept_experts": keep,
        "vision": False,
        "mtp": False,
    }
    if args.saliency is not None:
        calibration = np.load(args.saliency)
        output_config["rapid_quantization"]["pruning"] = {
            "saliency_file": args.saliency.name,
            "calibration_tokens": int(calibration.get("tokens", 0)),
            "selection_policy": str(
                calibration.get("selection_policy", "combined_saliency")
            ),
        }
    if selected_layers is not None:
        output_config["rapid_quantization"]["partial_layers"] = sorted(selected_layers)
    (destination / "config.json").write_text(json.dumps(output_config, indent=2) + "\n")
    (destination / "model.safetensors.index.json").write_text(
        json.dumps(
            {"metadata": {"total_size": total_bytes}, "weight_map": weight_map},
            indent=2,
        )
        + "\n"
    )
    _copy_metadata_files(source, destination)
    _write_model_card(destination, output_config["rapid_quantization"], total_bytes)
    print(f"wrote {total_bytes / 1e9:.1f} GB to {destination}", flush=True)


if __name__ == "__main__":
    main()
