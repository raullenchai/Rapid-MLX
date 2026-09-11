#!/usr/bin/env python3
"""Convert official V4.1 DSpark shards to an MLX affine sidecar.

The official release stores routed experts as packed FP4 and the remaining
matmuls as FP8.  MLX cannot load the release's E8M0 scale dtype directly, so
this tool reads one source shard through safetensors/PyTorch, dequantizes each
weight with Rapid's reviewed release-format implementation, and requantizes the
draft-only linears.  Processing one shard at a time keeps the Hub-cache and RAM
working sets bounded.

This tool never downloads files and has no cache-location option.  Operators
must supply already-cached, revision-pinned release files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import sys
import tempfile
from pathlib import Path

import mlx.core as mx
import torch
from safetensors import safe_open

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from repack_deepseek_v41_native import (  # noqa: E402
    CheckpointIndex,
    TensorPlan,
    write_safetensors,
)

from vllm_mlx.models.deepseek_v41_native.convert import sanitize_group  # noqa: E402

_QUANTIZED_SUFFIXES = (
    ".attn.wq_a",
    ".attn.wq_b",
    ".attn.wkv",
    ".attn.wo_a",
    ".attn.wo_b",
    ".main_proj",
    ".shared_experts.w1",
    ".shared_experts.w2",
    ".shared_experts.w3",
    ".confidence_head.proj",
    ".markov_head.embed",
    ".markov_head.head",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _weight_map(path: Path) -> dict[str, str]:
    weight_map = _read_json(path).get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"index has no weight_map: {path}")
    if not all(
        isinstance(name, str) and isinstance(shard, str)
        for name, shard in weight_map.items()
    ):
        raise ValueError(f"index weight_map must map strings to strings: {path}")
    return weight_map


def _source_array(handle, name: str) -> mx.array:
    tensor_slice = handle.get_slice(name)
    dtype = tensor_slice.get_dtype()
    tensor = tensor_slice[:]
    if dtype in ("F8_E4M3", "F8_E8M0"):
        return mx.array(tensor.view(torch.uint8).numpy())
    if dtype == "I8":
        return mx.array(tensor.numpy())
    if dtype == "BF16":
        return mx.array(tensor.view(torch.uint16).numpy()).view(mx.bfloat16)
    if dtype == "F32":
        return mx.array(tensor.numpy(), dtype=mx.float32)
    raise ValueError(f"unsupported official DSpark dtype {dtype}: {name}")


def _quantized_base(name: str, array: mx.array, group_size: int) -> str | None:
    if not name.startswith("mtp.") or not name.endswith(".weight"):
        return None
    if array.ndim != 2 or int(array.shape[-1]) % group_size:
        return None
    base = name[: -len(".weight")]
    if base.endswith(_QUANTIZED_SUFFIXES):
        return base
    if ".experts." in base and base.endswith((".w1", ".w2", ".w3")):
        return base
    return None


def _convert_shard(
    source: Path,
    source_index: Path,
    destination: Path,
    *,
    bits: int,
    group_size: int,
) -> dict:
    if bits not in (2, 3, 4, 6, 8):
        raise ValueError("MLX affine bits must be one of 2, 3, 4, 6, or 8")
    index = _weight_map(source_index)
    expected = sorted(
        name
        for name, shard in index.items()
        if name.startswith("mtp.") and shard == source.name
    )
    if not expected:
        raise ValueError(f"source index maps no MTP tensors to {source.name}")
    if destination.exists():
        raise FileExistsError(
            f"refusing to infer precision for an untracked output: {destination}"
        )

    handle = safe_open(str(source), framework="pt")
    actual = sorted(name for name in handle.keys() if name.startswith("mtp."))  # noqa: SIM118 - safetensors handle is not iterable
    if actual != expected:
        raise ValueError("source index and shard MTP tensors disagree")
    raw = {name: _source_array(handle, name) for name in expected}
    converted = sanitize_group(expected, raw, dtype=mx.bfloat16)
    del raw

    output: dict[str, mx.array] = {}
    modules: dict[str, dict] = {}
    for name, array in converted.items():
        base = _quantized_base(name, array, group_size)
        if base is None:
            output[name] = array
            continue
        weight, scales, biases = mx.quantize(array, group_size=group_size, bits=bits)
        mx.eval(weight, scales, biases)
        output[base + ".weight"] = weight
        output[base + ".scales"] = scales
        output[base + ".biases"] = biases
        modules[base] = {
            "bits": bits,
            "group_size": group_size,
            "mode": "affine",
        }
    mx.eval(*output.values())
    temporary = destination.with_name(
        destination.name.removesuffix(".safetensors") + ".partial.safetensors"
    )
    mx.save_safetensors(str(temporary), output, metadata={"format": "mlx"})
    os.replace(temporary, destination)
    return {
        "source_file": source.name,
        "source_bytes": source.stat().st_size,
        "source_sha256": _sha256(source),
        "output_file": destination.name,
        "output_bytes": destination.stat().st_size,
        "output_sha256": _sha256(destination),
        "tensor_count": len(output),
        "quantized_modules": modules,
    }


def _header(path: Path) -> dict:
    with path.open("rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
        value = json.loads(handle.read(length))
    value.pop("__metadata__", None)
    return value


def _finalize(
    destination: Path,
    source_config: Path,
    source_index: Path,
    source_revision: str,
) -> dict:
    state_files = sorted(destination.glob("*.conversion.json"))
    if not state_files:
        raise ValueError("destination has no converted shard state")
    states = [_read_json(path) for path in state_files]
    weight_map: dict[str, str] = {}
    modules: dict[str, dict] = {}
    for state in states:
        output_file = state.get("output_file")
        if not isinstance(output_file, str) or Path(output_file).name != output_file:
            raise ValueError("converted shard state has an unsafe output filename")
        output = destination / output_file
        if output.stat().st_size != state["output_bytes"]:
            raise ValueError(f"converted shard size changed: {output.name}")
        if _sha256(output) != state["output_sha256"]:
            raise ValueError(f"converted shard digest changed: {output.name}")
        for name in _header(output):
            if name in weight_map:
                raise ValueError(f"duplicate converted tensor: {name}")
            weight_map[name] = output.name
        modules.update(state["quantized_modules"])

    expected_map = _weight_map(source_index)
    expected = {name for name in expected_map if name.startswith("mtp.")}

    def logical_name(name: str) -> str:
        for suffix in (".scales", ".biases"):
            if name.endswith(suffix):
                return name[: -len(suffix)] + ".weight"
        return name

    logical = {logical_name(name) for name in weight_map}
    # Source .scale tensors are consumed into each affine weight triple.
    expected_logical = {
        name[: -len(".scale")] + ".weight" if name.endswith(".scale") else name
        for name in expected
    }
    if logical != expected_logical:
        missing = sorted(expected_logical - logical)
        extra = sorted(logical - expected_logical)
        raise ValueError(
            f"converted MTP contract mismatch (missing={missing[:1]}, extra={extra[:1]})"
        )

    config = _read_json(source_config)
    bits = {entry["bits"] for entry in modules.values()}
    groups = {entry["group_size"] for entry in modules.values()}
    if len(bits) != 1 or len(groups) != 1:
        raise ValueError("converted shards do not share one affine configuration")
    config["quantization"] = {
        "bits": bits.pop(),
        "group_size": groups.pop(),
        "mode": "affine",
        "modules": modules,
    }
    config["rapid_quantization"] = {
        "mtp": True,
        "mtp_source": "official-source-requantized",
        "source_revision": source_revision,
    }
    (destination / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (destination / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map}, indent=2) + "\n"
    )
    manifest = {
        "format": 1,
        "source_repo": "deepseek-ai/DeepSeek-V4.1-Flash",
        "source_revision": source_revision,
        "source_index_sha256": _sha256(source_index),
        "config_sha256": _sha256(destination / "config.json"),
        "index_sha256": _sha256(destination / "model.safetensors.index.json"),
        "shards": states,
        "tensor_count": len(weight_map),
    }
    (destination / "rapid-dspark-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return manifest


def _use_low_precision(name: str) -> bool:
    """Keep bandwidth-dominant routed experts at the proven 2-bit precision."""
    return name.startswith("mtp.") and ".ffn.experts." in name


def _mixed_quantization(high_config: dict, low_config: dict) -> dict:
    high = high_config.get("quantization")
    low = low_config.get("quantization")
    if not isinstance(high, dict) or not isinstance(low, dict):
        raise ValueError("both inputs require MLX quantization metadata")
    low_bits = int(low["bits"])
    high_bits = int(high["bits"])
    if (low_bits, high_bits) != (2, 4):
        raise ValueError(
            "mixed DSpark artifact requires a 2-bit low input and 4-bit high input"
        )
    modules = high.get("modules")
    if not isinstance(modules, dict) or not modules:
        raise ValueError("high-precision input requires an explicit module map")
    result = {
        "bits": int(high["bits"]),
        "group_size": int(high["group_size"]),
        "mode": high.get("mode", "affine"),
        "modules": dict(modules),
    }
    low_entry = {
        "bits": int(low["bits"]),
        "group_size": int(low["group_size"]),
        "mode": low.get("mode", "affine"),
    }
    for base in list(result["modules"]):
        if _use_low_precision(base):
            result["modules"][base] = low_entry
    # The drafter shares these two matrices from the 2-bit target at runtime;
    # they are not present in the standalone high-precision head itself.
    result["modules"]["embed"] = low_entry
    result["modules"]["head"] = low_entry
    return result


def _compose_mixed(
    low_path: Path,
    high_path: Path,
    destination: Path,
    *,
    low_revision: str,
    high_revision: str,
) -> dict:
    if destination.exists():
        raise FileExistsError(destination)
    low = CheckpointIndex(low_path)
    high = CheckpointIndex(high_path)
    low_names = {name for name in low.tensors if name.startswith("mtp.")}
    high_names = {name for name in high.tensors if name.startswith("mtp.")}
    if low_names != high_names:
        raise ValueError("low/high DSpark tensor contracts differ")
    low_config = _read_json(low_path / "config.json")
    high_config = _read_json(high_path / "config.json")
    quantization = _mixed_quantization(high_config, low_config)

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent)
    )
    try:
        weight_map: dict[str, str] = {}
        shard_rows = []
        stages = sorted({int(name.split(".")[1]) for name in high_names})
        for stage in stages:
            names = sorted(
                name for name in high_names if name.startswith(f"mtp.{stage}.")
            )
            plans = []
            low_count = 0
            for name in names:
                source = (
                    low.tensors[name]
                    if _use_low_precision(name)
                    else high.tensors[name]
                )
                low_count += int(_use_low_precision(name))
                plans.append(
                    TensorPlan(name, source.dtype, source.shape, sources=(source,))
                )
            filename = f"dspark-mixed-stage-{stage}.safetensors"
            output = staging / filename
            write_safetensors(output, plans)
            for name in _header(output):
                weight_map[name] = filename
            shard_rows.append(
                {
                    "file": filename,
                    "bytes": output.stat().st_size,
                    "sha256": _sha256(output),
                    "tensor_count": len(plans),
                    "low_precision_tensors": low_count,
                }
            )

        config = dict(high_config)
        config["quantization"] = quantization
        config["rapid_quantization"] = {
            "mtp": True,
            "mtp_source": "mixed-official-source-and-affine2",
            "dense_bits": quantization["bits"],
            "expert_bits": int(low_config["quantization"]["bits"]),
            "source_revision": high_revision,
            "low_revision": low_revision,
        }
        (staging / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        (staging / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {}, "weight_map": weight_map}, indent=2) + "\n"
        )
        manifest = {
            "format": 1,
            "policy": "routed experts 2-bit; remaining DSpark linears 4-bit",
            "low_repo": "Vontra/DeepSeek-V4.1-Flash-MLX-2bit-MTP",
            "low_revision": low_revision,
            "high_source_repo": "deepseek-ai/DeepSeek-V4.1-Flash",
            "high_source_revision": high_revision,
            "tensor_count": len(weight_map),
            "config_sha256": _sha256(staging / "config.json"),
            "index_sha256": _sha256(staging / "model.safetensors.index.json"),
            "shards": shard_rows,
        }
        (staging / "rapid-dspark-manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        staging.rename(destination)
        return manifest
    except BaseException:
        shutil.rmtree(staging)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    shard = sub.add_parser("convert-shard")
    shard.add_argument("--source", type=Path, required=True)
    shard.add_argument("--source-index", type=Path, required=True)
    shard.add_argument("--destination", type=Path, required=True)
    shard.add_argument("--bits", type=int, default=4)
    shard.add_argument("--group-size", type=int, default=64)
    final = sub.add_parser("finalize")
    final.add_argument("--destination", type=Path, required=True)
    final.add_argument("--source-config", type=Path, required=True)
    final.add_argument("--source-index", type=Path, required=True)
    final.add_argument("--source-revision", required=True)
    mixed = sub.add_parser("compose-mixed")
    mixed.add_argument("--low", type=Path, required=True)
    mixed.add_argument("--high", type=Path, required=True)
    mixed.add_argument("--destination", type=Path, required=True)
    mixed.add_argument("--low-revision", required=True)
    mixed.add_argument("--high-revision", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "convert-shard":
        args.destination.mkdir(parents=True, exist_ok=True)
        output = args.destination / args.source.name.replace("model-", "dspark-")
        state_path = output.with_suffix(".conversion.json")
        if state_path.exists():
            raise FileExistsError(state_path)
        result = _convert_shard(
            args.source.absolute(),
            args.source_index.resolve(),
            output,
            bits=args.bits,
            group_size=args.group_size,
        )
        state_path.write_text(json.dumps(result, indent=2) + "\n")
    elif args.command == "finalize":
        result = _finalize(
            args.destination.resolve(),
            args.source_config.resolve(),
            args.source_index.resolve(),
            args.source_revision,
        )
    else:
        result = _compose_mixed(
            args.low.resolve(),
            args.high.resolve(),
            args.destination.absolute(),
            low_revision=args.low_revision,
            high_revision=args.high_revision,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
