#!/usr/bin/env python3
"""Interleaved real-model gate for Qwen4 fp32-input fast RMSNorm."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path

try:
    from scripts.bench_metadata import format_bench_json, write_bench_json
except ImportError:  # direct `python scripts/bench_*.py` execution
    from bench_metadata import format_bench_json, write_bench_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path)
    parser.add_argument("--micro-only", action="store_true")
    parser.add_argument("--prompt-tokens", type=int, nargs="+", default=[1024, 16384])
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--micro-repeats", type=int, default=7)
    parser.add_argument("--micro-iterations", type=int, default=400)
    parser.add_argument("--error-cases", type=int, default=128)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def counter_delta(after: dict[str, int], before: dict[str, int]) -> dict[str, int]:
    reasons = set(before) | set(after)
    return {
        reason: delta
        for reason in sorted(reasons)
        if (delta := after.get(reason, 0) - before.get(reason, 0))
    }


def build_prompt(tokenizer, target_tokens: int):
    seed = tokenizer.encode(
        "Summarize the performance engineering evidence carefully. ",
        add_special_tokens=False,
    )
    if not seed:
        raise RuntimeError("tokenizer produced an empty benchmark seed")
    repeated = (seed * ((target_tokens + len(seed) - 1) // len(seed)))[:target_tokens]
    return repeated


def run_micro(args):
    if args.micro_repeats < 1 or args.micro_iterations < 1 or args.error_cases < 1:
        raise SystemExit("micro repeat, iteration, and error counts must be positive")

    import mlx.core as mx
    import numpy as np

    mx.set_default_device(mx.gpu)
    rng = np.random.default_rng(3058)

    def stock(x, weight, eps=1e-6):
        normalized = x.astype(mx.float32)
        normalized *= mx.rsqrt(
            mx.mean(mx.square(normalized), axis=-1, keepdims=True) + eps
        )
        return (normalized * (1 + weight.astype(mx.float32))).astype(x.dtype)

    def fast_fp32(x, weight, eps=1e-6):
        normalized = mx.fast.rms_norm(x.astype(mx.float32), None, eps)
        return (normalized * (1 + weight.astype(mx.float32))).astype(x.dtype)

    def bad_bf16(x, weight, eps=1e-6):
        normalized = mx.fast.rms_norm(x, None, eps).astype(mx.float32)
        return (normalized * (1 + weight.astype(mx.float32))).astype(x.dtype)

    def time_shape(shape, weight_shape):
        inputs = mx.array(rng.normal(size=shape).astype(np.float32), dtype=mx.bfloat16)
        weight = mx.array(
            rng.normal(0, 0.05, size=weight_shape).astype(np.float32),
            dtype=mx.bfloat16,
        )
        for function in (stock, fast_fp32):
            for _ in range(30):
                mx.eval(function(inputs, weight))
        samples = {"stock": [], "fast_fp32": []}
        orders = (
            (("stock", stock), ("fast_fp32", fast_fp32)),
            (("fast_fp32", fast_fp32), ("stock", stock)),
        )
        for repeat in range(args.micro_repeats):
            for name, function in orders[repeat % len(orders)]:
                started = time.perf_counter()
                for _ in range(args.micro_iterations):
                    mx.eval(function(inputs, weight))
                samples[name].append(
                    (time.perf_counter() - started) * 1e6 / args.micro_iterations
                )
        rows = {
            name: {
                "samples_microseconds": values,
                "median_microseconds": statistics.median(values),
            }
            for name, values in samples.items()
        }
        rows["speedup"] = (
            rows["stock"]["median_microseconds"]
            / rows["fast_fp32"]["median_microseconds"]
        )
        return rows

    squared_errors = {name: 0.0 for name in ("stock", "fast_fp32", "bad_bf16")}
    differences = {name: 0 for name in ("fast_fp32", "bad_bf16")}
    elements = 0
    for _ in range(args.error_cases):
        inputs = mx.array(
            rng.normal(size=(4, 2560)).astype(np.float32), dtype=mx.bfloat16
        )
        weight = mx.array(
            rng.normal(0, 0.05, size=(4, 2560)).astype(np.float32),
            dtype=mx.bfloat16,
        )
        input64 = np.asarray(inputs.astype(mx.float32)).astype(np.float64)
        weight64 = np.asarray(weight.astype(mx.float32)).astype(np.float64)
        reference = input64 / np.sqrt(
            np.mean(input64**2, axis=-1, keepdims=True) + 1e-6
        )
        reference *= 1 + weight64
        outputs = {
            name: np.asarray(function(inputs, weight).astype(mx.float32))
            for name, function in (
                ("stock", stock),
                ("fast_fp32", fast_fp32),
                ("bad_bf16", bad_bf16),
            )
        }
        for name, output in outputs.items():
            squared_errors[name] += float(
                np.sum((output.astype(np.float64) - reference) ** 2)
            )
        for name in differences:
            differences[name] += int(
                np.count_nonzero(outputs[name] != outputs["stock"])
            )
        elements += outputs["stock"].size

    rms_errors = {
        name: (error / elements) ** 0.5 for name, error in squared_errors.items()
    }
    return {
        "mode": "micro",
        "environment": {
            "host": platform.node(),
            "platform": platform.platform(),
            "mlx": mx.__version__,
        },
        "method": {
            "warmup_iterations": 30,
            "timed_repeats": args.micro_repeats,
            "iterations_per_repeat": args.micro_iterations,
            "error_cases": args.error_cases,
            "seed": 3058,
        },
        "timings": {
            "hc_grouped_4x2560": time_shape((1, 1, 4, 2560), (4, 2560)),
            "hidden_2560": time_shape((1, 1, 2560), (2560,)),
            "qsa_24x256": time_shape((1, 1, 24, 256), (256,)),
        },
        "fp64": {
            "elements": elements,
            "rms_errors": rms_errors,
            "ratio_fast_fp32_to_stock": rms_errors["fast_fp32"] / rms_errors["stock"],
            "ratio_bad_bf16_to_stock": rms_errors["bad_bf16"] / rms_errors["stock"],
            "different_fraction": {
                name: count / elements for name, count in differences.items()
            },
        },
        "correctness": {
            "passed": rms_errors["fast_fp32"] / rms_errors["stock"] < 1.01
            and rms_errors["bad_bf16"] / rms_errors["stock"] > 1.3
        },
    }


def run(args):
    if args.max_tokens < 2:
        raise SystemExit("--max-tokens must be at least 2")
    if args.repeats < 1:
        raise SystemExit("--repeats must be at least 1")
    if any(length <= 8 for length in args.prompt_tokens):
        raise SystemExit("--prompt-tokens values must exceed the decode-width gate")

    import mlx.core as mx
    from mlx_lm.generate import generate_step
    from mlx_lm.sample_utils import make_sampler
    from mlx_lm.utils import load

    from vllm_mlx.models.qwen4_exp import (
        qwen4_fast_rmsnorm_stats,
        set_qwen4_fast_rmsnorm_mode,
    )
    from vllm_mlx.utils.tokenizer import _register_vendored_archs

    _register_vendored_archs()
    mx.set_default_device(mx.gpu)
    if args.model is None:
        raise SystemExit("--model is required unless --micro-only is set")
    model, tokenizer = load(str(args.model.resolve()))
    model.eval()
    sampler = make_sampler(temp=0.0)

    observations = []
    orders = (("stock", "fast_fp32"), ("fast_fp32", "stock"))
    for target_tokens in args.prompt_tokens:
        prompt = mx.array(build_prompt(tokenizer, target_tokens))
        for repeat in range(args.repeats):
            for mode in orders[repeat % len(orders)]:
                resident_norms = set_qwen4_fast_rmsnorm_mode(model, mode)
                before = qwen4_fast_rmsnorm_stats(model)
                tokens = []
                started = time.perf_counter()
                first_token_at = None
                for token, _ in generate_step(
                    prompt,
                    model,
                    max_tokens=args.max_tokens,
                    sampler=sampler,
                ):
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    tokens.append(int(token))
                ended = time.perf_counter()
                after = qwen4_fast_rmsnorm_stats(model)
                if len(tokens) < 2 or first_token_at is None:
                    raise RuntimeError(
                        f"generation produced {len(tokens)} tokens; need at least 2"
                    )
                fast_calls = after["fast_calls"] - before["fast_calls"]
                declines = after["declines"] - before["declines"]
                decline_reasons = counter_delta(
                    after["decline_reasons"], before["decline_reasons"]
                )
                path_engaged = (
                    fast_calls == 0 and declines == 0
                    if mode == "stock"
                    else fast_calls > 0
                    and declines > 0
                    and decline_reasons == {"sequence_too_wide": declines}
                )
                decode_seconds = ended - first_token_at
                observations.append(
                    {
                        "mode": mode,
                        "prompt_tokens": target_tokens,
                        "repeat": repeat + 1,
                        "generated_tokens": len(tokens),
                        "token_sha256": hashlib.sha256(
                            json.dumps(tokens, separators=(",", ":")).encode()
                        ).hexdigest(),
                        "ttft_seconds": first_token_at - started,
                        "decode_seconds": decode_seconds,
                        "decode_tokens_per_second": (len(tokens) - 1) / decode_seconds,
                        "resident_norms": resident_norms,
                        "fast_calls": fast_calls,
                        "declines": declines,
                        "decline_reasons": decline_reasons,
                        "path_engaged": path_engaged,
                    }
                )
                mx.clear_cache()

    summaries = {}
    correctness = True
    for target_tokens in args.prompt_tokens:
        rows = [item for item in observations if item["prompt_tokens"] == target_tokens]
        hashes = {item["token_sha256"] for item in rows}
        path_engaged = all(item["path_engaged"] for item in rows)
        medians = {
            mode: statistics.median(
                item["decode_tokens_per_second"]
                for item in rows
                if item["mode"] == mode
            )
            for mode in ("stock", "fast_fp32")
        }
        summaries[str(target_tokens)] = {
            "token_exact": len(hashes) == 1,
            "path_engaged": path_engaged,
            "token_hashes": sorted(hashes),
            "median_decode_tokens_per_second": medians,
            "median_speedup_percent": 100.0
            * (medians["fast_fp32"] / medians["stock"] - 1.0),
        }
        correctness &= len(hashes) == 1 and path_engaged

    return {
        "correctness": {"passed": correctness},
        "summaries": summaries,
        "observations": observations,
    }


def main() -> int:
    args = parse_args()
    result = run_micro(args) if args.micro_only else run(args)
    payload = format_bench_json(result, __file__, indent=2, sort_keys=True)
    print(payload)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_bench_json(args.output, result, __file__, indent=2, sort_keys=True)
    return 0 if result["correctness"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
