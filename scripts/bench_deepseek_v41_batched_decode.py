#!/usr/bin/env python3
"""Compare independent-position and lockstep native decode step latency.

Synthetic smoke run (no checkpoint, CPU supported)::

    python scripts/bench_deepseek_v41_batched_decode.py --tiny --steps 4

Explicit local checkpoint run::

    python scripts/bench_deepseek_v41_batched_decode.py --model /path/to/model

Prefill, cache merging and warmup are outside the timed region. Each timed
step includes forward, greedy selection and device evaluation. Results include
hardware/runtime/configuration metadata; synthetic timings are not model claims.
"""

from __future__ import annotations

import argparse
import platform
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx
import numpy as np

from rapid_mlx.models.deepseek_v41_native.cache import BatchModelCache
from rapid_mlx.models.deepseek_v41_native.config import ModelArgs
from rapid_mlx.models.deepseek_v41_native.engram import build_layout
from rapid_mlx.models.deepseek_v41_native.model import Model
from scripts.bench_metadata import format_bench_json


def make_tiny_model(seed=18):
    """Small float32 stack spanning window, pooling, sharing, hashing and HC.

    Uses the 64-wide synthetic layout from the native module tests, extended
    with both compression ratios and source/consumer layers.
    """
    args = ModelArgs(
        vocab_size=32,
        dim=64,
        n_layers=6,
        moe_inter_dim=64,
        n_heads=2,
        head_dim=32,
        rope_head_dim=8,
        q_lora_rank=32,
        o_lora_rank=16,
        o_groups=2,
        window_size=4,
        norm_eps=1e-6,
        n_routed_experts=8,
        n_activated_experts=6,
        compress_ratios=(0, 2, 2, 1, 1, 1),
        kv_source_layers=(1, 3),
        index_source_layers=(1, 3, 4),
        index_n_heads=2,
        index_head_dim=32,
        index_topk=3,
        candidate_source_layer=3,
        candidate_topk_blocks=2,
        candidate_block_size=2,
        original_seq_len=16,
        max_seq_len=256,
        hc_mult=2,
        hc_sinkhorn_iters=3,
        engram_layer_ids=(0, 4),
        engram_vocab_size=7,
        engram_n_heads=1,
        engram_head_dim=32,
        engram_max_ngram_size=4,
        engram_compressed_vocab_size=32,
    )
    primes, _ = build_layout(args)
    args = replace(args, engram_num_embeddings=tuple(map(int, primes.sum(axis=(1, 2)))))
    mx.random.seed(seed)
    model = Model(args, token_map=np.arange(args.vocab_size))
    # Exercise learned paths that initialize to zero in an unloaded model.
    for layer in model.layers:
        for name in ("hc_attn_fn", "hc_ffn_fn", "hc_attn_base", "hc_ffn_base"):
            value = getattr(layer, name)
            setattr(layer, name, mx.random.normal(value.shape) * 0.03)
        layer.hc_attn_scale = mx.full((3,), 0.1)
        layer.hc_ffn_scale = mx.full((3,), 0.1)
        layer.ffn.gate.weight = mx.random.normal(layer.ffn.gate.weight.shape) * 0.03
        layer.ffn.gate.bias = mx.arange(args.n_routed_experts) * 0.001
        if layer.engram is not None:
            embed = layer.engram.embed
            embed.weight = mx.random.normal(embed.weight.shape) * 0.05
            embed.scale = mx.ones(embed.scale.shape)
    model.eval()
    mx.eval(model.parameters())
    return model


def prefill_rows(model, lengths, capacity):
    caches, next_tokens = [], []
    for row, length in enumerate(lengths):
        cache = model.make_cache(max_seq_len=capacity)
        tokens = ((mx.arange(length) + row * 7) % model.args.vocab_size)[None].astype(
            mx.int32
        )
        logits = model(tokens, cache, last_logit_only=True)
        next_tokens.append(mx.argmax(logits[:, -1], axis=-1).reshape(1, 1))
        caches.append(cache)
    tokens = mx.concatenate(next_tokens)
    mx.eval(tokens)
    return caches, tokens


def measure(model, cache, tokens, warmup, steps):
    samples = []
    for step in range(warmup + steps):
        start = time.perf_counter()
        logits = model(tokens, cache, last_logit_only=True)
        tokens = mx.argmax(logits.reshape(tokens.shape[0], -1), axis=-1)[:, None]
        mx.eval(tokens)
        elapsed = time.perf_counter() - start
        if step >= warmup:
            samples.append(elapsed * 1000)
    return {
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.mean(samples),
        "min_ms": min(samples),
        "steps": steps,
    }


def benchmark(model, rows, base_position, steps, warmup):
    lengths = [base_position + row for row in range(rows)]
    capacity = max(lengths) + warmup + steps
    mixed, mixed_tokens = prefill_rows(model, lengths, capacity)
    lock_rows, lock_tokens = prefill_rows(model, [max(lengths)] * rows, capacity)
    lockstep = model.make_cache(bsz=rows, max_seq_len=capacity)
    # A merged cache has exactly the same row buffers as lockstep when all
    # offsets agree. Restore those buffers into the existing scalar cache type.
    merged = BatchModelCache.from_caches(lock_rows)
    lockstep.layers, lockstep.engram_ids = merged.layers, merged.engram_ids
    lockstep.offset = max(lengths)
    return {
        "rows": rows,
        "mixed_positions": lengths,
        "lockstep_position": max(lengths),
        "mixed": measure(
            model, BatchModelCache.from_caches(mixed), mixed_tokens, warmup, steps
        ),
        "lockstep": measure(model, lockstep, lock_tokens, warmup, steps),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--model", type=Path, help="existing local native checkpoint directory"
    )
    source.add_argument(
        "--tiny",
        action="store_true",
        help="synthetic weights; never reads a checkpoint",
    )
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--position", type=int, default=16)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--eval-interval", type=int, default=None)
    args = parser.parse_args(argv)
    if args.steps < 1 or args.warmup < 0 or args.position < 1 or min(args.rows) < 1:
        parser.error(
            "steps, position and rows must be positive; warmup must be nonnegative"
        )
    if args.tiny:
        model = make_tiny_model()
    else:
        from rapid_mlx.models.deepseek_v41_native.load import load
        from rapid_mlx.models.deepseek_v41_native.serving import install_target_qmv

        model, _ = load(str(args.model))
        install_target_qmv(model)
    if args.eval_interval is not None:
        model.eval_interval = args.eval_interval
    print(
        format_bench_json(
            {
                "synthetic": args.tiny,
                "machine": platform.machine(),
                "platform": platform.platform(),
                "mlx": mx.__version__,
                "device": str(mx.default_device()),
                "eval_interval": model.eval_interval,
                "warmup": args.warmup,
                "results": [
                    benchmark(model, rows, args.position, args.steps, args.warmup)
                    for rows in args.rows
                ],
            },
            __file__,
        )
    )


if __name__ == "__main__":
    main()
