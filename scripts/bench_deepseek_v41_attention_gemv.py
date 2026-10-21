#!/usr/bin/env python3
"""Synchronized synthetic affine attention projection benchmark (no model load).

Run on an idle GPU: python scripts/bench_deepseek_v41_attention_gemv.py
Each JSON line reports stock, raw candidate and qualified production dispatch.
Timing excludes lazy qualification/JIT. Bandwidth counts weight+scale+bias
bytes once per call, including all groups, regardless of input row count.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rapid_mlx.models.deepseek_v41_native import attention_gemv as gemv
from rapid_mlx.models.deepseek_v41_native.attention import GroupedOutputLinear

# N, K, groups in the shipped target. Mixed precision uses both 4 and 8 bits.
LAYOUTS = {
    "wq_a": (1280, 5120, 1),
    "wq_b": (32768, 1280, 1),
    "wkv": (512, 5120, 1),
    "wo_a": (1024, 4096, 8),
    "wo_b": (5120, 8192, 1),
}


def synchronized_ms(call, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        mx.eval(call())
    mx.synchronize()
    samples = []
    for _ in range(repeats):
        mx.synchronize()
        start = time.perf_counter_ns()
        result = call()
        mx.eval(result)
        mx.synchronize()
        samples.append((time.perf_counter_ns() - start) / 1e6)
    return float(statistics.median(samples))


def benchmark(
    name: str, bits: int, rows: list[int], dtype, warmup: int, repeats: int
) -> list[dict]:
    n, k, groups = LAYOUTS[name]
    linear = (
        GroupedOutputLinear(k, n, groups) if groups > 1 else nn.Linear(k, n, bias=False)
    )
    linear.weight = linear.weight.astype(mx.bfloat16)
    layer = linear.to_quantized(group_size=64, bits=bits)
    mx.eval(layer.parameters())
    weight_bytes = sum(a.nbytes for a in (layer.weight, layer.scales, layer.biases))
    results = []
    for count in rows:
        shape = (1, count, groups, 1, k) if groups > 1 else (1, count, k)
        x = mx.random.normal(shape).astype(mx.float32 if groups > 1 else dtype)
        mx.eval(x)
        qualified = gemv._qualified(layer, x, name)
        candidate, stock = gemv._launch(layer, x), layer(x)
        exact = bool(
            mx.array_equal(candidate.view(mx.uint8), stock.view(mx.uint8)).item()
        )
        stock_ms = synchronized_ms(lambda x=x: layer(x), warmup, repeats)
        raw_ms = synchronized_ms(lambda x=x: gemv._launch(layer, x), warmup, repeats)
        dispatch_ms = synchronized_ms(
            lambda x=x: gemv.project(layer, x, name), warmup, repeats
        )
        results.append(
            {
                "projection": name,
                "bits": bits,
                "group_size": 64,
                "N": n,
                "K": k,
                "groups": groups,
                "rows": count,
                "dtype": str(x.dtype),
                "qualified": qualified,
                "sample_exact": exact,
                "stock_ms": stock_ms,
                "candidate_ms": raw_ms,
                "dispatch_ms": dispatch_ms,
                "candidate_speedup": stock_ms / raw_ms,
                "weight_bytes": weight_bytes,
                "candidate_GB_s": weight_bytes / (raw_ms * 1e6),
            }
        )
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--projection", nargs="+", choices=list(LAYOUTS), default=list(LAYOUTS)
    )
    parser.add_argument("--bits", nargs="+", type=int, choices=[4, 8], default=[4, 8])
    parser.add_argument(
        "--rows", nargs="+", type=int, choices=range(1, 9), default=list(range(1, 9))
    )
    parser.add_argument(
        "--dtype", choices=["float32", "bfloat16", "float16"], default="float32"
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.repeats < 1:
        parser.error("warmup must be nonnegative and repeats positive")
    if not mx.metal.is_available():
        parser.error("benchmark requires Metal")
    if not gemv.ENABLED or not gemv.custom_kernels_enabled():
        parser.error("enable custom kernels before benchmarking")
    mx.set_default_device(mx.gpu)
    print(
        json.dumps(
            {
                "host": platform.node(),
                "platform": platform.platform(),
                "mlx": getattr(mx, "__version__", "unknown"),
                "device": mx.metal.device_info(),
                "synthetic_weights": True,
                "warmup": args.warmup,
                "repeats": args.repeats,
            }
        )
    )
    for name in args.projection:
        for bits in args.bits:
            for result in benchmark(
                name,
                bits,
                args.rows,
                getattr(mx, args.dtype),
                args.warmup,
                args.repeats,
            ):
                print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
