#!/usr/bin/env python3
"""Measure MLX affine Q4/Q6/Q8 cost on production GLM-5.3 matrix shapes."""

from __future__ import annotations

import argparse
import json
import statistics
import time

import mlx.core as mx

SHAPES = {
    "kda_fused_input": (24896, 4096),
    "kda_recurrent_out": (8192, 128),
    "sparse_kv_b": (32768, 512),
    "routed_expert_down": (4096, 2048),
    "lm_head": (154880, 4096),
}


def _measure(out_features: int, in_features: int, bits: int, tokens: int, repeats: int):
    packed = mx.zeros((out_features, in_features * bits // 32), dtype=mx.uint32)
    scales = mx.ones((out_features, in_features // 64), dtype=mx.bfloat16)
    biases = mx.zeros_like(scales)
    inputs = mx.ones((tokens, in_features), dtype=mx.bfloat16)
    mx.eval(packed, scales, biases, inputs)

    def run():
        value = mx.quantized_matmul(
            inputs,
            packed,
            scales,
            biases,
            transpose=True,
            group_size=64,
            bits=bits,
            mode="affine",
        )
        mx.eval(value)

    for _ in range(3):
        run()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        run()
        samples.append((time.perf_counter_ns() - started) / 1e6)
    result = {
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, action="append", default=[])
    parser.add_argument("--repeats", type=int, default=11)
    args = parser.parse_args()
    token_counts = args.tokens or [1, 128]
    report = {
        "device": mx.device_info(),
        "group_size": 64,
        "repeats": args.repeats,
        "results": {},
    }
    for label, (out_features, in_features) in SHAPES.items():
        report["results"][label] = {}
        for tokens in token_counts:
            rows = {}
            for bits in (4, 6, 8):
                rows[f"q{bits}"] = _measure(
                    out_features, in_features, bits, tokens, args.repeats
                )
                mx.clear_cache()
            baseline = rows["q4"]["median_ms"]
            for row in rows.values():
                row["relative_to_q4"] = row["median_ms"] / baseline
            report["results"][label][str(tokens)] = rows
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
