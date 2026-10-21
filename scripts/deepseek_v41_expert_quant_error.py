#!/usr/bin/env python3
"""CPU-only, one-expert-at-a-time data-free output distortion and byte inventory.

No checkpoint Python executes. Reuses the native dequantizers/conversion policy
and the existing repacker's safetensors index instead of loading whole shards.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.deepseek_v41_rd_common import (
    affine_bytes,
    file_hash,
    require_all_moe_layers,
)
from scripts.repack_deepseek_v41_native import _DTYPE_BYTES, CheckpointIndex


def read_tensor(tensor):
    import mlx.core as mx

    dtype = {
        "BF16": np.uint16,
        "F16": np.float16,
        "F32": np.float32,
        "U8": np.uint8,
        # Source FP4 experts are stored as I8 but hold packed e2m1 nibble
        # pairs; dequant_fp4 unpacks raw bytes, so they must stay unsigned.
        "I8": np.uint8,
        "F8_E4M3": np.uint8,
        "F8_E4M3FN": np.uint8,
        "F8_E8M0": np.uint8,
    }.get(tensor.dtype)
    if dtype is None:
        raise ValueError(f"unsupported source dtype {tensor.dtype}: {tensor.name}")
    # Copy only this tensor, then close its mapping before the next expert.
    mapped = np.memmap(
        tensor.path, mode="r", dtype=dtype, offset=tensor.offset, shape=tensor.shape
    )
    raw = np.array(mapped)
    del mapped
    value = mx.array(raw)
    return value.view(mx.bfloat16) if tensor.dtype == "BF16" else value


def load_expert(index, layer, expert, expert_dtype="fp4"):
    import mlx.core as mx

    from rapid_mlx.models.deepseek_v41_native.dequant import dequant_fp4, dequant_fp8

    result = []
    for proj in ("w1", "w3", "w2"):
        base = f"layers.{layer}.ffn.experts.{expert}.{proj}"
        tensor = index.tensors[base + ".weight"]
        w = read_tensor(tensor)
        scale = index.tensors.get(base + ".scale")
        if scale is not None:
            # Storage dtype alone does not identify packed FP4. Source
            # quantization metadata determines the expert format.
            if expert_dtype not in ("fp4", "fp8"):
                raise ValueError(f"unsupported source expert format: {expert_dtype}")
            dequant = dequant_fp4 if expert_dtype == "fp4" else dequant_fp8
            w = dequant(w, read_tensor(scale), mx.bfloat16)
        elif tensor.dtype not in ("BF16", "F16", "F32"):
            raise ValueError(f"missing source scale: {base}")
        mx.eval(w)
        result.append(w)
    return result


def expert_output(weights, probes, limit=10.0):
    import mlx.core as mx
    import mlx.nn as nn

    gate, up, down = weights
    g = probes @ gate.astype(mx.float32).T
    u = probes @ up.astype(mx.float32).T
    if limit > 0:
        g, u = mx.minimum(g, limit), mx.clip(u, -limit, limit)
    return (nn.silu(g) * u) @ down.astype(mx.float32).T


def quant_error(weights, probes, limit=10.0):
    import mlx.core as mx

    reference = expert_output(weights, probes, limit)
    denominator = mx.sum(mx.square(reference))
    mx.eval(reference, denominator)
    if not np.isfinite(denominator.item()) or denominator.item() <= 0:
        raise ValueError("zero/nonfinite expert reference energy; use more probes")
    errors = [1.0]
    costs = [0]
    for bits in (2, 3, 4):
        restored = []
        size = 0
        for weight in weights:
            packed = mx.quantize(weight.astype(mx.bfloat16), group_size=64, bits=bits)
            mx.eval(*packed)
            size += sum(t.nbytes for t in packed)
            expected = affine_bytes(weight.shape, bits)
            if sum(t.nbytes for t in packed) != expected:
                raise ValueError("MLX quantized payload differs from byte contract")
            restored.append(mx.dequantize(*packed, group_size=64, bits=bits))
        output = expert_output(restored, probes, limit)
        error = mx.sum(mx.square(reference - output)) / denominator
        mx.eval(error)
        value = float(error.item())
        if not np.isfinite(value):
            raise ValueError("nonfinite quantization error")
        errors.append(value)
        costs.append(size)
        del restored, output
    return np.array(errors), np.array(costs, np.int64)


def projection_quant_error(weights, probes, limit=10.0):
    """Isolated output distortion q[p,b]; additive use is a disclosed surrogate."""
    import mlx.core as mx

    ref = expert_output(weights, probes, limit)
    denominator = mx.sum(mx.square(ref))
    mx.eval(ref, denominator)
    if not np.isfinite(denominator.item()) or denominator.item() <= 0:
        raise ValueError("zero/nonfinite expert reference energy")
    result = np.zeros((3, 3))
    for p in range(3):
        for j, b in enumerate((2, 3, 4)):
            restored = list(weights)
            triple = mx.quantize(weights[p].astype(mx.bfloat16), group_size=64, bits=b)
            restored[p] = mx.dequantize(*triple, group_size=64, bits=b)
            error = (
                mx.sum(mx.square(ref - expert_output(restored, probes, limit)))
                / denominator
            )
            mx.eval(error)
            result[p, j] = error.item()
    if not np.isfinite(result).all():
        raise ValueError("nonfinite projection distortion")
    return result


def inventory(index, config, nonexpert_bits=4):
    """Exact planned text tensor bytes, with sliced router rows charged to experts.

    Quantization follows convert.py, scale/bias storage BF16; vision/MTP and
    SSD Engram payloads excluded. Header/file container bytes are not resident.
    """
    from rapid_mlx.models.deepseek_v41_native.convert import is_quant_target
    from rapid_mlx.models.deepseek_v41_native.dequant import is_fp4_expert

    c = config.get("text_config", config)
    layers, experts = c["num_hidden_layers"], c["n_routed_experts"]
    costs = np.zeros((layers, experts, 4), np.int64)
    fixed, engram, router_rows = 0, 0, {}
    layers_with_experts = set()
    for name, t in index.tensors.items():
        if name.startswith(("mtp.", "vision.", "aligner.", "image_")):
            continue
        if ".engram.embed." in name:
            engram += t.nbytes
            continue
        if name.endswith(".scale"):
            continue
        shape = list(t.shape)
        has_scale = name.removesuffix(
            ".weight"
        ) + ".scale" in index.tensors and name.endswith(".weight")
        if (
            has_scale
            and is_fp4_expert(name)
            and config.get("quantization_config", {}).get("expert_dtype", "fp4")
            == "fp4"
        ):
            shape[-1] *= 2
        match = re.fullmatch(
            r"layers\.(\d+)\.ffn\.experts\.(\d+)\.w[123]\.weight", name
        )
        if match:
            layer_id, e = map(int, match.groups())
            layers_with_experts.add(layer_id)
            for j, b in enumerate((2, 3, 4), 1):
                costs[layer_id, e, j] += affine_bytes(shape, b)
            continue
        # Budget hc_*/attn_sink at source F32 width as a conservative upper bound:
        # packed non-reference builds use BF16 (~79 MB less on the 1017 checkpoint).
        # Keep this budget so existing allocations stay reproducible.
        fp32 = re.search(r"(hc_\w+|attn_sink|gate\.bias(_vl)?)$", name)
        size = (
            int(np.prod(shape)) * (4 if fp32 else 2) if has_scale or fp32 else t.nbytes
        )
        if len(shape) >= 2 and is_quant_target(name, shape[-1], 64):
            # Dequantized source matrices become BF16; unscaled passthrough
            # matrices retain their dtype in quantize_tree, including scales.
            scale_bytes = 2 if has_scale else _DTYPE_BYTES[t.dtype]
            size = affine_bytes(shape, nonexpert_bits, scale_bytes=scale_bytes)
        router = re.fullmatch(r"layers\.(\d+)\.ffn\.gate\.(weight|bias|bias_vl)", name)
        if router:
            layer_id = int(router.group(1))
            if shape[0] != experts or size % experts:
                raise ValueError("router axis does not match expert count")
            router_rows[layer_id] = router_rows.get(layer_id, 0) + size // experts
        else:
            fixed += size
    require_all_moe_layers(config, layers_with_experts)
    if np.any(costs[..., 1:] <= 0) or set(router_rows) != set(range(layers)):
        raise ValueError("source inventory is missing routed experts or router rows")
    for layer_id, row_bytes in router_rows.items():
        costs[layer_id, :, 1:] += row_bytes
    return {
        "expert_bytes": costs.tolist(),
        "fixed_non_expert_bytes": fixed,
        "excluded_engram_bytes": engram,
        "router_row_bytes_by_layer": router_rows,
        "nonexpert_bits": nonexpert_bits,
        "group_size": 64,
        "policy": "text-only; shared experts fixed; compact router rows charged per kept expert",
    }


def save_result(path, q, costs, completed, metadata, projection_q=None):
    temp = path.with_suffix(".tmp.npz")
    np.savez(
        temp,
        q=q,
        expert_weight_bytes=costs,
        completed=completed,
        metadata=np.array(json.dumps(metadata)),
        **({"projection_q": projection_q} if projection_q is not None else {}),
    )
    os.replace(temp, path)


def save_inventory(path, account):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".tmp-")
    try:
        with os.fdopen(fd, "w") as output:
            output.write(json.dumps(account, indent=2) + "\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path)
    p.add_argument("--inventory", type=Path, required=True)
    p.add_argument("--inventory-only", action="store_true")
    p.add_argument(
        "--release-policy",
        action="store_true",
        help="fixed weights 8-bit g64 / BF16; Engram 4-bit g32 SSD",
    )
    p.add_argument(
        "--projection-errors",
        action="store_true",
        help="also measure each projection quantized in isolation",
    )
    p.add_argument("--nonexpert-bits", type=int, choices=(2, 3, 4), default=4)
    p.add_argument("--model-revision")
    p.add_argument("--probes", type=int, default=16)
    p.add_argument("--seed", type=int, default=1017)
    p.add_argument("--start", type=int, default=0, help="flattened layer/expert index")
    p.add_argument("--stop", type=int, help="exclusive flattened index")
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    if a.probes <= 0 or a.start < 0:
        raise ValueError("probes must be positive and start nonnegative")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "2")
    import mlx.core as mx

    mx.set_default_device(mx.cpu)
    source = a.source.resolve()
    config = json.loads((source / "config.json").read_text())
    c = config.get("text_config", config)
    # CheckpointIndex reads headers only; tensor bodies are sliced per expert.
    index = CheckpointIndex(source)
    revision = a.model_revision or source.name
    account = inventory(index, config, a.nonexpert_bits)
    if a.release_policy:
        from scripts.deepseek_v41_rd_build import mtp_byte_costs, plan

        allocation = {
            "layers": {
                str(i): {
                    "kept_expert_ids": list(range(c["n_routed_experts"])),
                    "bits": 2,
                }
                for i in range(c["num_hidden_layers"])
            }
        }
        _, _, report = plan(index, config, allocation)
        account["mtp_bytes_by_expert_bits"] = mtp_byte_costs(index, config)
        account["mtp_expert_bits"] = 4
        account["mtp_bytes"] = account["mtp_bytes_by_expert_bits"]["4"]
        account["fixed_non_expert_bytes"] = sum(
            v
            for k, v in report["components"].items()
            if k not in ("router", "routed_experts", "engram_ssd")
        )
        account["components"] = report["components"]
        account["nonexpert_bits"] = 8
        account["policy"] = (
            "release: nonexperts 8-bit g64 / BF16; Engram 4-bit g32 SSD; MTP omitted"
        )
    account["model_revision"] = revision
    if a.inventory_only:
        save_inventory(a.inventory, account)
        return
    if a.output is None:
        raise ValueError("--output required unless --inventory-only")
    shape = (c["num_hidden_layers"], c["n_routed_experts"])
    q = np.full((*shape, 4), np.nan)
    costs = np.zeros((*shape, 4), np.int64)
    completed = np.zeros(shape, bool)
    projection_q = np.full((*shape, 3, 3), np.nan) if a.projection_errors else None
    metadata = {
        "schema_version": 1,
        "projection_errors": a.projection_errors,
        "model_revision": revision,
        "config_sha256": file_hash(source / "config.json"),
        "index_sha256": file_hash(source / "model.safetensors.index.json"),
        "probes": a.probes,
        "seed": a.seed,
        "group_size": 64,
        "bits_order": [0, 2, 3, 4],
        "probe_distribution": "iid N(0,1), common seed per expert",
        "metric": "sum ||E(x)-Q_b(E)(x)||^2 / sum ||E(x)||^2",
        "source_dtype": "BF16 dequantization matching converter",
    }
    if a.output.exists():
        if not a.resume:
            raise FileExistsError("output exists; use --resume")
        with np.load(a.output, allow_pickle=False) as old:
            if (
                json.loads(str(old["metadata"])) != metadata
                or old["q"].shape != q.shape
            ):
                raise ValueError("resume provenance/shape mismatch")
            if a.projection_errors:
                projection_q = old["projection_q"].copy()
            q, costs, completed = (
                old["q"].copy(),
                old["expert_weight_bytes"].copy(),
                old["completed"].copy(),
            )
    stop = np.prod(shape) if a.stop is None else a.stop
    if not a.start < stop <= np.prod(shape):
        raise ValueError("invalid expert range")
    save_inventory(a.inventory, account)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    # Same probes at each expert make bit-width comparisons reproducible.
    probes = mx.array(
        np.random.default_rng(a.seed)
        .standard_normal((a.probes, c["hidden_size"]))
        .astype(np.float32)
    )
    for flat in range(a.start, int(stop)):
        layer_id, e = divmod(flat, shape[1])
        if completed[layer_id, e]:
            continue
        weights = load_expert(
            index,
            layer_id,
            e,
            config.get("quantization_config", {}).get("expert_dtype", "fp4"),
        )
        q[layer_id, e], costs[layer_id, e] = quant_error(
            weights, probes, c.get("swiglu_limit", 10.0)
        )
        if a.projection_errors:
            projection_q[layer_id, e] = projection_quant_error(
                weights, probes, c.get("swiglu_limit", 10.0)
            )
        completed[layer_id, e] = True
        del weights
        mx.clear_cache()
        save_result(a.output, q, costs, completed, metadata, projection_q)
        print(
            json.dumps({"layer": layer_id, "expert": e, "q": q[layer_id, e].tolist()}),
            flush=True,
        )


if __name__ == "__main__":
    main()
