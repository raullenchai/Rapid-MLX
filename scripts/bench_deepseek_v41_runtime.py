#!/usr/bin/env python3
"""Profile the standalone DeepSeek V4.1 Flash MLX runtime with real weights."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

try:
    from scripts.bench_metadata import format_bench_json, write_bench_json
except ImportError:  # direct `python scripts/profile_*.py` execution
    from bench_metadata import format_bench_json, write_bench_json


PLAN = {
    "scope": "real-weight standalone DeepSeek V4.1 Flash text-runtime profile",
    "baseline": "unchanged checkpoint-bundled runtime",
    "measurements": [
        "load and sequential context-build wall time",
        "warm decode token latency and throughput",
        "existing mx.eval barrier wait time by Python call site",
        "runtime disk-read and Engram-cache counters",
        "MLX active, cache, and peak memory",
    ],
    "guardrails": [
        "never downloads or relocates model files",
        "requires explicit opt-in before executing checkpoint-bundled Python",
        "barrier wait is not kernel attribution",
        "text-only; vision and MTP are excluded",
    ],
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path)
    parser.add_argument("--execute-real-weights", action="store_true")
    parser.add_argument("--trust-checkpoint-runtime", action="store_true")
    parser.add_argument("--resident-backbone", action="store_true")
    parser.add_argument("--engram-cache-rows", type=int, default=16384)
    parser.add_argument("--execution-mode", choices=("reference", "deferred", "compiled"), default="compiled")
    parser.add_argument("--prompt", default="Explain why local inference latency matters.")
    parser.add_argument("--context-tokens", type=int, default=0)
    parser.add_argument("--warmup-tokens", type=int, default=16)
    parser.add_argument("--measure-tokens", type=int, default=128)
    parser.add_argument("--diagnostic-tokens", type=int, default=8)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _percentile(values, percentile):
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _checkpoint_total_size(model_path):
    index_path = model_path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    size = (index.get("metadata") or {}).get("total_size")
    return int(size) if size is not None else None


def _resolve_runtime(model_path):
    candidates = (model_path / "runtime" / "runtime.py", model_path / "runtime.py")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "checkpoint has no runtime/runtime.py or runtime.py; no fallback download is allowed"
    )


def _load_runtime(runtime_path):
    spec = importlib.util.spec_from_file_location(
        "rapid_deepseek_v41_checkpoint_runtime", runtime_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import checkpoint runtime: {runtime_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class EvalBarrierProfiler:
    """Measure waits at existing eval barriers without adding new barriers."""

    def __init__(self, original):
        self.original = original
        self.reset()

    def reset(self):
        self.calls = 0
        self.seconds = 0.0
        self.by_caller = defaultdict(lambda: {"calls": 0, "seconds": 0.0})

    def __call__(self, *args, **kwargs):
        frame = sys._getframe(1)
        caller = f"{frame.f_code.co_name}:{frame.f_lineno}"
        started = time.perf_counter()
        result = self.original(*args, **kwargs)
        elapsed = time.perf_counter() - started
        self.calls += 1
        self.seconds += elapsed
        bucket = self.by_caller[caller]
        bucket["calls"] += 1
        bucket["seconds"] += elapsed
        return result

    def snapshot(self):
        return {
            "calls": self.calls,
            "seconds": self.seconds,
            "by_caller": dict(
                sorted(
                    self.by_caller.items(),
                    key=lambda item: item[1]["seconds"],
                    reverse=True,
                )
            ),
            "interpretation": (
                "Time is charged to the existing barrier that waited for queued work; "
                "it is not exclusive time for the caller."
            ),
        }


def _weight_counters(runtime):
    weights = runtime.w
    return {
        "disk_bytes_read": int(getattr(weights, "disk_bytes_read", 0)),
        "disk_read_calls": int(getattr(weights, "disk_read_calls", 0)),
        "engram_cache_hits": int(getattr(weights, "engram_cache_hits", 0)),
        "engram_cache_misses": int(getattr(weights, "engram_cache_misses", 0)),
        "resident_bytes": int(getattr(weights, "resident_bytes", 0)),
    }


def _counter_delta(after, before):
    return {key: after[key] - before[key] for key in after}


def _memory_snapshot(mx):
    return {
        "active_bytes": int(mx.get_active_memory()),
        "cache_bytes": int(mx.get_cache_memory()),
        "peak_bytes": int(mx.get_peak_memory()),
    }


def _run_tokens(
    runtime,
    count,
    barrier,
    mx,
    *,
    initial_token,
    teacher_tokens=None,
):
    latencies = []
    counters_before = _weight_counters(runtime)
    if barrier is not None:
        barrier.reset()
    next_token = int(initial_token)
    started = time.perf_counter()
    for index in range(count):
        token = teacher_tokens[index] if teacher_tokens is not None else next_token
        token_started = time.perf_counter()
        logits, _ = runtime.step(int(token))
        next_token = int(mx.argmax(logits).item())
        latencies.append(time.perf_counter() - token_started)
    elapsed = time.perf_counter() - started
    return {
        "tokens": count,
        "seconds": elapsed,
        "tokens_per_second": count / elapsed if elapsed else None,
        "latency_seconds": {
            "median": statistics.median(latencies) if latencies else None,
            "p95": _percentile(latencies, 0.95),
            "max": max(latencies) if latencies else None,
            "raw": latencies,
        },
        "eval_barriers": barrier.snapshot() if barrier is not None else None,
        "weight_counters": _counter_delta(_weight_counters(runtime), counters_before),
        "memory": _memory_snapshot(mx),
        "last_token_id": next_token,
    }


def _validate_args(args):
    if args.model is None or not args.model.is_dir():
        raise SystemExit("--model must name an existing local checkpoint directory")
    if not args.trust_checkpoint_runtime:
        raise SystemExit(
            "refusing to execute checkpoint-bundled Python without "
            "--trust-checkpoint-runtime"
        )
    if min(
        args.context_tokens,
        args.warmup_tokens,
        args.measure_tokens,
        args.diagnostic_tokens,
    ) < 0:
        raise SystemExit("token counts may not be negative")
    if args.measure_tokens < 1:
        raise SystemExit("--measure-tokens must be at least 1")
    if args.diagnostic_tokens < 1:
        raise SystemExit("--diagnostic-tokens must be at least 1")
    if not 0 <= args.engram_cache_rows <= 16384:
        raise SystemExit("--engram-cache-rows must be between 0 and 16384")


def run(args):
    _validate_args(args)
    model_path = args.model.expanduser().resolve()
    runtime_path = _resolve_runtime(model_path)
    total_size = _checkpoint_total_size(model_path)

    started = time.perf_counter()
    module = _load_runtime(runtime_path)
    import mlx.core as mx

    mx.set_default_device(mx.gpu)
    mx.reset_peak_memory()
    runtime = module.TextRuntime(
        model_path,
        max_tokens=(
            args.context_tokens
            + args.warmup_tokens
            + args.measure_tokens
            + args.diagnostic_tokens
            + 8
        ),
        resident_backbone=args.resident_backbone,
        execution_mode=args.execution_mode,
    )
    runtime.w.engram_cache_rows = args.engram_cache_rows
    load_seconds = time.perf_counter() - started
    encoded = runtime.tokenizer.encode(args.prompt).ids
    if not encoded:
        raise RuntimeError("prompt encoded to zero tokens")
    original_eval = module.mx.eval
    barrier = EvalBarrierProfiler(original_eval)
    load_memory = _memory_snapshot(mx)
    mx.reset_peak_memory()

    context = [encoded[index % len(encoded)] for index in range(args.context_tokens)]
    try:
        context_result = _run_tokens(
            runtime,
            len(context),
            None,
            mx,
            initial_token=encoded[-1],
            teacher_tokens=context,
        )
        warmup_result = _run_tokens(
            runtime,
            args.warmup_tokens,
            None,
            mx,
            initial_token=context_result["last_token_id"],
        )
        measured = _run_tokens(
            runtime,
            args.measure_tokens,
            None,
            mx,
            initial_token=warmup_result["last_token_id"],
        )
        module.mx.eval = barrier
        diagnostic = _run_tokens(
            runtime,
            args.diagnostic_tokens,
            barrier,
            mx,
            initial_token=measured["last_token_id"],
        )
        module.mx.eval = original_eval
    finally:
        module.mx.eval = original_eval

    return {
        "plan": PLAN,
        "checkpoint": {
            "path": str(model_path),
            "runtime_path": str(runtime_path),
            "indexed_total_size_bytes": total_size,
            "runtime_sha256": hashlib.sha256(runtime_path.read_bytes()).hexdigest(),
        },
        "parameters": {
            "execution_mode": args.execution_mode,
            "resident_backbone": args.resident_backbone,
            "engram_cache_rows": args.engram_cache_rows,
            "context_tokens": args.context_tokens,
            "warmup_tokens": args.warmup_tokens,
            "measure_tokens": args.measure_tokens,
            "diagnostic_tokens": args.diagnostic_tokens,
        },
        "device": mx.device_info(),
        "load": {"seconds": load_seconds, "memory": load_memory},
        "context_build": context_result,
        "warmup": warmup_result,
        "measurement": measured,
        "barrier_diagnostic": diagnostic,
    }


def main() -> int:
    args = parse_args()
    if not args.execute_real_weights:
        print(format_bench_json({"plan_only": True, "plan": PLAN}, __file__))
        return 0
    result = run(args)
    print(format_bench_json(result, __file__, indent=2, sort_keys=True))
    if args.output is not None:
        write_bench_json(args.output, result, __file__, indent=2, sort_keys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
