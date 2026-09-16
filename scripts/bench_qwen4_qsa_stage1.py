#!/usr/bin/env python3
"""Benchmark the opt-in Qwen3.8/Qwen4 QSA stage-one selector.

Example:
  python scripts/bench_qwen4_qsa_stage1.py --tokens 16384 65536 98304
"""

from __future__ import annotations

import argparse
import math
import platform
import statistics
import time
from importlib.metadata import version

import mlx.core as mx
import numpy as np

from rapid_mlx.kernels.qsa_stage1 import qsa_stage1_select

try:
    from scripts.bench_metadata import format_bench_json
except ImportError:  # direct `python scripts/bench_*.py` execution
    from bench_metadata import format_bench_json


def eager_select(q, pooled, positions, *, topk: int, ratio: int):
    blocks = int(pooled.shape[1])
    scores = mx.matmul(q.transpose(0, 2, 1, 3), pooled.swapaxes(-1, -2)[:, None])
    scores = mx.sum(mx.maximum(scores.astype(mx.float32), 0), axis=1) / math.sqrt(
        q.shape[-1]
    )
    starts = mx.arange(blocks) * ratio
    scores = mx.where(
        (starts + ratio - 1)[None, None, :] <= positions[..., None],
        scores,
        -mx.inf,
    )
    return mx.argpartition(scores, kth=blocks - topk, axis=-1)[..., -topk:]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, nargs="+", default=[16384, 65536, 98304])
    parser.add_argument("--query-length", type=int, default=512)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--runs", type=int, default=9)
    parser.add_argument("--seed", type=int, default=19)
    args = parser.parse_args()

    mx.random.seed(args.seed)
    results = []
    for tokens in args.tokens:
        blocks = tokens // 4
        q = mx.random.normal((1, args.query_length, 4, 128)).astype(mx.bfloat16)
        pooled = mx.random.normal((1, blocks, 128)).astype(mx.bfloat16)
        positions = mx.full((1, args.query_length), tokens - 1, dtype=mx.int32)
        eager = lambda q=q, pooled=pooled, positions=positions: eager_select(
            q, pooled, positions, topk=512, ratio=4
        )
        fused = lambda q=q, pooled=pooled, positions=positions: qsa_stage1_select(
            q, pooled, positions, block_topk=512, compress_ratio=4
        )

        eager_ids = eager()
        fused_ids = fused()
        mx.eval(eager_ids, fused_ids)
        equal = np.array_equal(
            np.sort(np.asarray(eager_ids), axis=-1),
            np.sort(np.asarray(fused_ids), axis=-1),
        )
        for _ in range(args.warmups):
            mx.eval(eager(), fused())
        samples = {"eager": [], "fused": []}
        functions = {"eager": eager, "fused": fused}
        for repeat in range(args.runs):
            order = ("eager", "fused") if repeat % 2 == 0 else ("fused", "eager")
            for name in order:
                started = time.perf_counter()
                mx.eval(functions[name]())
                samples[name].append((time.perf_counter() - started) * 1000)
        peaks = {}
        for name in ("eager", "fused"):
            mx.reset_peak_memory()
            mx.eval(functions[name]())
            peaks[name] = mx.get_peak_memory()
        eager_result = {
            "median_ms": statistics.median(samples["eager"]),
            "samples_ms": samples["eager"],
            "peak_bytes": peaks["eager"],
        }
        fused_result = {
            "median_ms": statistics.median(samples["fused"]),
            "samples_ms": samples["fused"],
            "peak_bytes": peaks["fused"],
        }
        results.append(
            {
                "tokens": tokens,
                "query_length": args.query_length,
                "selection_equal": bool(equal),
                "eager": eager_result,
                "fused": fused_result,
                "speedup": eager_result["median_ms"] / fused_result["median_ms"],
            }
        )

    print(
        format_bench_json(
            {
                "machine": platform.machine(),
                "device": mx.device_info(),
                "macos": platform.mac_ver()[0],
                "mlx": version("mlx"),
                "seed": args.seed,
                "results": results,
            },
            __file__,
        )
    )


if __name__ == "__main__":
    main()
