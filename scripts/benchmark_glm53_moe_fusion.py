#!/usr/bin/env python3
"""Benchmark Rapid's rewrite on one production-shape GLM-5.3 MoE layer.

This weight-free microbenchmark uses the official 4096 -> 2048, 288-expert,
top-8 contract and the Rapid alias's affine q4-g64 quantization. It is a layer
benchmark, not a substitute for an end-to-end official-checkpoint campaign.
Run each sample in a fresh process to avoid comparing against a class that was
already patched by an earlier sample.
"""

from __future__ import annotations

import hashlib
import json
import platform
import statistics
import time

import mlx.core as mx
import mlx.nn as nn
from mlx_vlm.models.switch_layers import SwitchGLU

from vllm_mlx.moe_fusion import fuse_gate_up

HIDDEN = 4096
INTERMEDIATE = 2048
EXPERTS = 288
TOP_K = 8
WARMUP = 8
RUNS = 31


class OneLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.moe = SwitchGLU(HIDDEN, INTERMEDIATE, EXPERTS)


def measure(
    model: OneLayer, x: mx.array, indices: mx.array
) -> tuple[list[float], mx.array]:
    out = model.moe(x, indices)
    mx.eval(out)
    for _ in range(WARMUP):
        out = model.moe(x, indices)
        mx.eval(out)
    samples = []
    for _ in range(RUNS):
        started = time.perf_counter()
        out = model.moe(x, indices)
        mx.eval(out)
        samples.append((time.perf_counter() - started) * 1_000)
    return samples, out


def digest(array: mx.array) -> str:
    return hashlib.sha256(bytes(array.astype(mx.float16))).hexdigest()


def main() -> None:
    mx.random.seed(53)
    model = OneLayer()
    nn.quantize(model, group_size=64, bits=4)
    x = mx.random.normal((1, 1, HIDDEN)).astype(mx.bfloat16)
    indices = mx.array([[[3, 17, 41, 89, 144, 201, 233, 287]]], dtype=mx.uint32)
    mx.eval(model.parameters(), x, indices)

    stock_samples, stock = measure(model, x, indices)
    fused_layers = fuse_gate_up(model)
    fused_samples, fused = measure(model, x, indices)
    equal = bool(mx.array_equal(stock, fused))
    stock_ms = statistics.median(stock_samples)
    fused_ms = statistics.median(fused_samples)
    result = {
        "hardware": platform.machine(),
        "mlx_version": mx.__version__,
        "shape": {
            "hidden": HIDDEN,
            "intermediate": INTERMEDIATE,
            "experts": EXPERTS,
            "top_k": TOP_K,
            "bits": 4,
            "group_size": 64,
        },
        "warmup": WARMUP,
        "runs": RUNS,
        "stock_median_ms": stock_ms,
        "fused_median_ms": fused_ms,
        "speedup": stock_ms / fused_ms,
        "fused_layers": fused_layers,
        "byte_equal": equal,
        "stock_sha256": digest(stock),
        "fused_sha256": digest(fused),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if fused_layers != 1 or not equal:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
