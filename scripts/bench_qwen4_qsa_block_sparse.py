#!/usr/bin/env python3
"""Qualify Qwen4 block-sparse QSA latency, dispatch floor, and numerics."""

from __future__ import annotations

import argparse
import os
import statistics
import time
from pathlib import Path

import numpy as np

try:
    from scripts.bench_metadata import format_bench_json, write_bench_json
except ImportError:  # direct `python scripts/bench_*.py` execution
    from bench_metadata import format_bench_json, write_bench_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-length", type=int, default=64)
    parser.add_argument("--key-length", type=int, default=16_384)
    parser.add_argument("--block-topk", type=int, default=512)
    parser.add_argument("--block-size", type=int, default=4)
    parser.add_argument("--query-heads", type=int, default=24)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--dispatch-repeats", type=int, default=200)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _validate_args(args) -> None:
    positive = {
        "query-length": args.query_length,
        "key-length": args.key_length,
        "block-topk": args.block_topk,
        "block-size": args.block_size,
        "query-heads": args.query_heads,
        "kv-heads": args.kv_heads,
        "head-dim": args.head_dim,
        "repeats": args.repeats,
        "dispatch-repeats": args.dispatch_repeats,
    }
    for name, value in positive.items():
        if value < 1:
            raise SystemExit(f"--{name} must be positive")
    if args.warmup < 0:
        raise SystemExit("--warmup must be non-negative")
    if args.query_heads % args.kv_heads:
        raise SystemExit("--query-heads must be divisible by --kv-heads")
    if args.head_dim % 32:
        raise SystemExit("--head-dim must be divisible by 32")
    if args.block_topk * args.block_size > args.key_length:
        raise SystemExit("selected blocks must fit inside --key-length")


def _measure(call, evaluate, *, warmup: int, repeats: int) -> list[float]:
    for _ in range(warmup):
        evaluate(call())
    samples = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        evaluate(call())
        samples.append((time.perf_counter_ns() - started) / 1_000_000)
    return samples


def _fp64_reference(queries, keys, values, selected_by_query):
    output = np.zeros(queries.shape, dtype=np.float64)
    head_dim = queries.shape[-1]
    query_heads = queries.shape[1]
    kv_heads = keys.shape[1]
    heads_per_kv = query_heads // kv_heads
    for query_index, selected in enumerate(selected_by_query):
        for query_head in range(query_heads):
            kv_head = query_head // heads_per_kv
            scores = (
                keys[0, kv_head, selected].astype(np.float64)
                @ queries[0, query_head, query_index].astype(np.float64)
            ) / np.sqrt(head_dim)
            weights = np.exp(scores - scores.max())
            weights /= weights.sum()
            output[0, query_head, query_index] = (
                weights[:, None] * values[0, kv_head, selected].astype(np.float64)
            ).sum(axis=0)
    return output


def _numerics_probe(mx, block_sparse_attention):
    rng = np.random.default_rng(29)
    query_heads, kv_heads, query_length, key_length, head_dim = 2, 1, 2, 10, 32
    block_size = 2
    queries_np = rng.normal(size=(1, query_heads, query_length, head_dim)).astype(
        np.float16
    )
    keys_np = rng.normal(size=(1, kv_heads, key_length, head_dim)).astype(np.float16)
    values_np = rng.normal(size=(1, kv_heads, key_length, head_dim)).astype(np.float16)
    block_starts_np = np.array([[[0, 4], [2, 6]]], dtype=np.int32)
    block_counts_np = np.array([[2, 2]], dtype=np.int32)
    tail_indices_np = np.array([[[8, 0], [8, 9]]], dtype=np.int32)
    tail_counts_np = np.array([[1, 2]], dtype=np.int32)
    selected = [np.array([0, 1, 4, 5, 8]), np.array([2, 3, 6, 7, 8, 9])]

    queries = mx.array(queries_np)
    keys = mx.array(keys_np)
    values = mx.array(values_np)
    sparse = block_sparse_attention(
        queries,
        keys,
        values,
        mx.array(block_starts_np),
        mx.array(block_counts_np),
        mx.array(tail_indices_np),
        mx.array(tail_counts_np),
        block_size=block_size,
    )
    dense_mask_np = np.zeros((1, 1, query_length, key_length), dtype=np.bool_)
    for query_index, indices in enumerate(selected):
        dense_mask_np[0, 0, query_index, indices] = True
    dense_mask = mx.where(
        mx.array(dense_mask_np),
        mx.array(0.0, dtype=queries.dtype),
        mx.array(-1e9, dtype=queries.dtype),
    )
    dense = mx.fast.scaled_dot_product_attention(
        queries,
        keys,
        values,
        scale=head_dim**-0.5,
        mask=dense_mask,
    )
    mx.eval(sparse, dense)
    reference = _fp64_reference(queries_np, keys_np, values_np, selected)
    sparse_error = np.array(sparse).astype(np.float64) - reference
    dense_error = np.array(dense).astype(np.float64) - reference
    return {
        "shape": [1, query_heads, query_length, head_dim],
        "sparse_max_abs_error_vs_fp64": float(np.max(np.abs(sparse_error))),
        "dense_max_abs_error_vs_fp64": float(np.max(np.abs(dense_error))),
        "sparse_rms_error_vs_fp64": float(np.sqrt(np.mean(sparse_error**2))),
        "dense_rms_error_vs_fp64": float(np.sqrt(np.mean(dense_error**2))),
        "sparse_max_abs_delta_vs_dense": float(
            np.max(np.abs(np.array(sparse) - np.array(dense)))
        ),
    }


def run(args):
    _validate_args(args)
    if os.environ.get("MLX_ENABLE_TF32") != "0":
        raise SystemExit(
            "set MLX_ENABLE_TF32=0 before launch so the fp32 qualification "
            "oracle is not silently reduced to TF32"
        )

    import mlx.core as mx

    from vllm_mlx.kernels.qsa_block_sparse import (
        block_sparse_attention,
        block_sparse_layout_supported,
    )

    if not mx.metal.is_available():
        raise SystemExit("QSA qualification requires an Apple Metal device")
    if not block_sparse_layout_supported(
        query_heads=args.query_heads,
        kv_heads=args.kv_heads,
        head_dim=args.head_dim,
        block_size=args.block_size,
        dtype=mx.bfloat16,
    ):
        raise SystemExit("requested QSA geometry is unsupported")

    mx.random.seed(args.seed)
    queries = mx.random.normal(
        (1, args.query_heads, args.query_length, args.head_dim)
    ).astype(mx.bfloat16)
    keys = mx.random.normal((1, args.kv_heads, args.key_length, args.head_dim)).astype(
        mx.bfloat16
    )
    values = mx.random.normal(
        (1, args.kv_heads, args.key_length, args.head_dim)
    ).astype(mx.bfloat16)
    max_start = args.key_length - args.block_size
    starts = np.linspace(
        0,
        max_start,
        num=args.block_topk,
        endpoint=True,
        dtype=np.int32,
    )
    starts -= starts % args.block_size
    starts = np.broadcast_to(
        starts[None, None, :],
        (1, args.query_length, args.block_topk),
    ).copy()
    block_starts = mx.array(starts)
    block_counts = mx.full((1, args.query_length), args.block_topk, dtype=mx.int32)
    tail_indices = mx.zeros((1, args.query_length, args.block_size), dtype=mx.int32)
    tail_counts = mx.zeros((1, args.query_length), dtype=mx.int32)
    token_indices = (
        block_starts[..., None] + mx.arange(args.block_size)[None, None, None, :]
    ).reshape(1, args.query_length, -1)
    dense_mask = mx.zeros((1, args.query_length, args.key_length), dtype=mx.bool_)
    dense_mask = mx.put_along_axis(
        dense_mask,
        token_indices,
        mx.ones_like(token_indices, dtype=mx.bool_),
        axis=-1,
    )[:, None]
    additive_mask = mx.where(
        dense_mask,
        mx.array(0.0, dtype=queries.dtype),
        mx.array(-1e9, dtype=queries.dtype),
    )
    mx.eval(queries, keys, values, block_starts, additive_mask)

    def sparse_call():
        return block_sparse_attention(
            queries,
            keys,
            values,
            block_starts,
            block_counts,
            tail_indices,
            tail_counts,
            block_size=args.block_size,
        )

    def dense_call():
        return mx.fast.scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale=args.head_dim**-0.5,
            mask=additive_mask,
        )

    timings = {"sparse_ms": [], "dense_ms": []}
    orders = (("dense_ms", dense_call), ("sparse_ms", sparse_call))
    for repeat in range(args.repeats):
        ordered = orders if repeat % 2 == 0 else tuple(reversed(orders))
        for name, call in ordered:
            timings[name].extend(_measure(call, mx.eval, warmup=args.warmup, repeats=1))

    # Measure the launch floor of this exact compiled binary with its smallest
    # supported useful geometry, synchronizing every call.
    floor_q = mx.zeros((1, 1, 1, 32), dtype=mx.float16)
    floor_kv = mx.zeros((1, 1, 2, 32), dtype=mx.float16)
    floor_starts = mx.zeros((1, 1, 1), dtype=mx.int32)
    floor_counts = mx.ones((1, 1), dtype=mx.int32)
    floor_tail = mx.zeros((1, 1, 2), dtype=mx.int32)
    floor_tail_counts = mx.zeros((1, 1), dtype=mx.int32)

    def floor_call():
        return block_sparse_attention(
            floor_q,
            floor_kv,
            floor_kv,
            floor_starts,
            floor_counts,
            floor_tail,
            floor_tail_counts,
            block_size=2,
        )

    floor_ms = _measure(
        floor_call,
        mx.eval,
        warmup=max(args.warmup, 3),
        repeats=args.dispatch_repeats,
    )
    medians = {name: statistics.median(values) for name, values in timings.items()}
    return {
        "geometry": {
            "query_length": args.query_length,
            "key_length": args.key_length,
            "selected_tokens": args.block_topk * args.block_size,
            "query_heads": args.query_heads,
            "kv_heads": args.kv_heads,
            "head_dim": args.head_dim,
            "dtype": "bfloat16",
        },
        "timings_ms": timings,
        "median_sparse_ms": medians["sparse_ms"],
        "median_dense_ms": medians["dense_ms"],
        "median_speedup": medians["dense_ms"] / medians["sparse_ms"],
        "dispatch_floor": {
            "binary": "rapid_qsa_block_sparse",
            "synchronized_samples_ms": floor_ms,
            "median_microseconds": statistics.median(floor_ms) * 1_000,
        },
        "numerics": _numerics_probe(mx, block_sparse_attention),
    }


def main() -> int:
    args = parse_args()
    result = run(args)
    payload = format_bench_json(result, __file__, indent=2, sort_keys=True)
    print(payload)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_bench_json(args.output, result, __file__, indent=2, sort_keys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
