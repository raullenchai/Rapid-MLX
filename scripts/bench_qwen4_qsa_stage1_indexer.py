#!/usr/bin/env python3
"""A/B the full production-geometry QSA indexer around the stage-one route."""

from __future__ import annotations

import argparse
import os
import platform
import statistics
import time
from importlib.metadata import version

import mlx.core as mx
import numpy as np

from rapid_mlx.models.qwen4_exp import QSAIndexer, TextModelArgs
from rapid_mlx.models.qwen4_exp_cache import QSAIndexCache

try:
    from scripts.bench_metadata import format_bench_json
except ImportError:  # direct `python scripts/bench_*.py` execution
    from bench_metadata import format_bench_json


def model_args() -> TextModelArgs:
    return TextModelArgs(
        hidden_size=256,
        num_hidden_layers=1,
        vocab_size=32,
        num_attention_heads=24,
        num_key_value_heads=2,
        head_dim=256,
        linear_num_key_heads=1,
        linear_num_value_heads=3,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_conv_kernel_dim=3,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=4,
        shared_expert_intermediate_size=4,
        hc_count=4,
        hc_lowrank=3,
        layer_types=["full_attention"],
        indexer_n_heads=4,
        indexer_kv_heads=1,
        indexer_head_dim=128,
        indexer_budget=2048,
        indexer_compress_ratio=4,
        ple_layer_ids=[],
        eos_token_id=31,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=int, default=65_024)
    parser.add_argument("--query-length", type=int, default=512)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--runs", type=int, default=9)
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()
    if args.context % 4 or args.query_length % 4:
        parser.error("context and query length must be divisible by four")
    old_length = args.context - args.query_length
    if old_length <= 0 or old_length % 4:
        parser.error(
            "context minus query length must be positive and divisible by four"
        )

    mx.random.seed(args.seed)
    indexer = QSAIndexer(model_args())
    indexer.eval()
    hidden = mx.random.normal((1, args.query_length, 256)).astype(mx.bfloat16)
    capacity = ((args.context // 4 + 255) // 256) * 256
    base_keys = mx.random.normal((1, capacity, 128)).astype(mx.bfloat16)
    base_ring = mx.random.normal((1, 4, 128)).astype(mx.bfloat16)
    mx.eval(hidden, base_keys, base_ring, indexer.parameters())

    def run(enabled: bool):
        os.environ["RAPID_MLX_QSA_STAGE1"] = "1" if enabled else "0"
        cache = QSAIndexCache(4)
        cache._offsets = [old_length]
        cache._compressed_counts = [old_length // 4]
        cache._pending_left_padding = [0]
        cache.compressed_keys = mx.array(base_keys)
        cache.raw_ring = mx.array(base_ring)
        selection = indexer(
            hidden,
            cache,
            physical_kv_length=args.context,
        )
        block_starts = selection.token_indices[..., :2048:4]
        return selection, block_starts

    for enabled in (False, True):
        for _ in range(args.warmups):
            selection, starts = run(enabled)
            mx.eval(selection.valid, starts)

    samples = {False: [], True: []}
    selected = {}
    for repeat in range(args.runs):
        order = (False, True) if repeat % 2 == 0 else (True, False)
        for enabled in order:
            started = time.perf_counter()
            selection, starts = run(enabled)
            mx.eval(selection.valid, starts)
            samples[enabled].append((time.perf_counter() - started) * 1000)
            selected[enabled] = np.sort(np.asarray(starts), axis=-1)

    medians = {
        enabled: statistics.median(values) for enabled, values in samples.items()
    }
    print(
        format_bench_json(
            {
                "machine": platform.machine(),
                "device": mx.device_info(),
                "macos": platform.mac_ver()[0],
                "mlx": version("mlx"),
                "seed": args.seed,
                "context": args.context,
                "query_length": args.query_length,
                "selected_block_sets_equal": bool(
                    np.array_equal(selected[False], selected[True])
                ),
                "off": {"median_ms": medians[False], "samples_ms": samples[False]},
                "on": {"median_ms": medians[True], "samples_ms": samples[True]},
                "speedup": medians[False] / medians[True],
                "route_constructions": indexer.stage1_route_constructions,
                "declines": indexer.stage1_declines,
                "decline_reasons": indexer.stage1_decline_reasons,
            },
            __file__,
        )
    )


if __name__ == "__main__":
    main()
