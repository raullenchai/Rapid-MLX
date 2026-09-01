#!/usr/bin/env python3
"""Real-weight gate for Rapid's default-off Qwen4 fused GDN decode path."""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

try:
    from scripts.bench_metadata import format_bench_json, write_bench_json
except ImportError:  # direct `python scripts/bench_*.py` execution
    from bench_metadata import format_bench_json, write_bench_json

PLAN = {
    "scope": "one production Qwen4 GDN layer with resident real weights",
    "correctness": "32 sequential steps; exact output and both cache arrays",
    "timing": "interleaved stock/fused observations without model reload",
    "excluded": ["prefill", "batch", "mask", "speculative rollback"],
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path)
    parser.add_argument("--execute-metal", action="store_true")
    parser.add_argument("--correctness-steps", type=int, default=32)
    parser.add_argument("--timing-steps", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def clone_cache(cache, cache_type, mx):
    clone = cache_type(len(cache.cache))
    clone.cache = [mx.array(value) for value in cache.cache]
    return clone


def cache_equal(left, right, mx):
    return all(
        bool(mx.array_equal(a, b).item())
        for a, b in zip(left.cache, right.cache, strict=True)
    )


def cache_diagnostics(left, right, mx):
    return [
        {
            "equal": bool(mx.array_equal(a, b).item()),
            "max_abs": float(
                mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item()
            ),
        }
        for a, b in zip(left.cache, right.cache, strict=True)
    ]


def run(args):
    if args.model is None or not args.model.is_dir():
        raise SystemExit("--model must name an existing local checkpoint")
    if args.correctness_steps < 32:
        raise SystemExit("--correctness-steps may not be below 32")

    import mlx.core as mx
    from mlx_lm.utils import load_model

    from vllm_mlx.models.qwen4_exp import (
        GatedDeltaNet,
        Qwen4ExpStateCache,
        qwen4_fused_gdn_stats,
        set_qwen4_fused_gdn_mode,
    )
    from vllm_mlx.utils.tokenizer import _register_vendored_archs

    mx.set_default_device(mx.gpu)
    _register_vendored_archs()
    model, _ = load_model(args.model.resolve(), strict=True)
    model.eval()
    layers = [
        module
        for _, module in model.named_modules()
        if isinstance(module, GatedDeltaNet)
    ]
    if not layers:
        raise SystemExit("checkpoint has no Rapid Qwen4 GDN layers")
    layer = layers[0]

    key = mx.random.key(2105)
    hidden = mx.random.normal(
        (args.correctness_steps, 1, 1, layer.hidden_size), key=key
    ).astype(layer.dt_bias.dtype)
    warm = Qwen4ExpStateCache(2)
    set_qwen4_fused_gdn_mode(layer, "stock")
    warm_output = layer(hidden[0], cache=warm)
    mx.eval(warm_output, *warm.cache)
    stock_cache = clone_cache(warm, Qwen4ExpStateCache, mx)
    fused_cache = clone_cache(warm, Qwen4ExpStateCache, mx)

    mismatch = None
    before = qwen4_fused_gdn_stats(layer)
    for step in range(args.correctness_steps):
        set_qwen4_fused_gdn_mode(layer, "stock")
        stock = layer(hidden[step], cache=stock_cache)
        mx.eval(stock, *stock_cache.cache)
        set_qwen4_fused_gdn_mode(layer, "fused")
        fused = layer(hidden[step], cache=fused_cache)
        mx.eval(fused, *fused_cache.cache)
        output_equal = bool(mx.array_equal(stock, fused).item())
        states_equal = cache_equal(stock_cache, fused_cache, mx)
        if not output_equal or not states_equal:
            mismatch = {
                "step": step,
                "output_equal": output_equal,
                "states_equal": states_equal,
                "max_output_abs": float(mx.max(mx.abs(stock - fused)).item()),
                "cache_slots": cache_diagnostics(stock_cache, fused_cache, mx),
            }
            break
    after = qwen4_fused_gdn_stats(layer)
    fused_calls = after["fused_calls"] - before["fused_calls"]
    fallbacks = after["fallbacks"] - before["fallbacks"]
    correctness = {
        "passed": mismatch is None
        and fused_calls == args.correctness_steps
        and fallbacks == 0,
        "steps": args.correctness_steps,
        "mismatch": mismatch,
        "fused_calls": fused_calls,
        "fallbacks": fallbacks,
    }
    if not correctness["passed"]:
        return {"plan": PLAN, "correctness": correctness, "timing": None}

    timings = {"stock": [], "fused": []}
    for repeat in range(args.repeats):
        order = ("stock", "fused") if repeat % 2 == 0 else ("fused", "stock")
        for mode in order:
            cache = clone_cache(warm, Qwen4ExpStateCache, mx)
            mx.eval(*cache.cache)
            set_qwen4_fused_gdn_mode(layer, mode)
            started = time.perf_counter()
            output = None
            for step in range(args.timing_steps):
                output = layer(hidden[step % len(hidden)], cache=cache)
            mx.eval(output, *cache.cache)
            timings[mode].append(time.perf_counter() - started)
    medians = {name: statistics.median(values) for name, values in timings.items()}
    timing = {
        "raw_seconds": timings,
        "median_seconds": medians,
        "median_speedup_percent": 100.0 * (medians["stock"] / medians["fused"] - 1.0),
    }
    return {"plan": PLAN, "correctness": correctness, "timing": timing}


def main() -> int:
    args = parse_args()
    if not args.execute_metal:
        print(format_bench_json({"plan_only": True, "plan": PLAN}, __file__))
        return 0
    result = run(args)
    payload = format_bench_json(result, __file__, indent=2, sort_keys=True)
    print(payload)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_bench_json(args.output, result, __file__, indent=2, sort_keys=True)
    return 0 if result["correctness"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
