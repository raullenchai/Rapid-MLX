#!/usr/bin/env python3
"""Measure stock versus absorbed MLA for a warm multi-token forward.

The two arms share one loaded model and start each timed call from cloned
mutable cache containers that share the same immutable MLX arrays. Only the supported attention
classes' ``__call__`` methods are switched between arms.

Example:
    python3.12 scripts/bench_mla_absorbed_verify.py \
      --model mlx-community/GLM-4.7-Flash-4bit \
      --contexts 1024 4096 16384 --width 3 --repeats 6 \
      --json /tmp/mla-verify.json
"""

from __future__ import annotations

import argparse
import copy
import os
import platform
import random
import statistics
import sys
import time
from pathlib import Path

os.environ["RAPID_MLX_MLA_ABSORBED_VERIFY"] = "1"
os.environ["RAPID_MLX_MLA_ABSORBED_VERIFY_STATS"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models import cache as mlx_cache

from scripts.bench_metadata import format_bench_json, write_bench_json
from vllm_mlx.patches import mla_absorbed_verify as patch

LONG_CODE_EDIT_PROMPT = """You are a code refactoring assistant. Re-emit the
complete Python module below with one change: rename every local variable
`total` to `accumulator`. Preserve all other tokens and emit only Python.

""" + "\n\n".join(
    f"""def compute_score_{index}(records, weights):
    if not records:
        return 0.0
    total = 0.0
    for record, weight in zip(records, weights):
        total += float(record.value) * float(weight)
    if total < 0:
        return 0.0
    return total"""
    for index in range(64)
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--contexts", nargs="+", type=int, default=[1024, 4096])
    parser.add_argument("--width", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--oracle-context", type=int, default=0)
    parser.add_argument("--oracle-cases", type=int, default=12)
    parser.add_argument("--suffix-repeats", type=int, default=0)
    parser.add_argument("--suffix-max-tokens", type=int, default=128)
    parser.add_argument("--json", type=Path)
    return parser.parse_args()


def _set_arm(targets: dict, originals: dict, arm: str) -> None:
    for key, cls in targets.items():
        cls.__call__ = originals[key] if arm == "stock" else targets[key].rapid_call


def _clone_cache(cache):
    """Clone mutable cache containers while sharing immutable MLX arrays."""
    if hasattr(cache, "shape") and hasattr(cache, "dtype"):
        return cache
    if isinstance(cache, list):
        return [_clone_cache(value) for value in cache]
    if isinstance(cache, tuple):
        return tuple(_clone_cache(value) for value in cache)
    if isinstance(cache, dict):
        return {key: _clone_cache(value) for key, value in cache.items()}

    cloned = copy.copy(cache)
    # CacheList owns nested cache objects; ArraysCache owns a mutable list.
    for attribute in ("caches", "cache"):
        if hasattr(cache, attribute):
            setattr(cloned, attribute, _clone_cache(getattr(cache, attribute)))
    return cloned


def _timed_forward(model, tokens, cache) -> tuple[float, object]:
    started = time.perf_counter()
    logits = model(tokens, cache=cache)
    mx.eval(logits)
    return (time.perf_counter() - started) * 1000, logits


def _prefill(model, token_ids: list[int], chunk_size: int):
    cache = mlx_cache.make_prompt_cache(model)
    for start in range(0, len(token_ids), chunk_size):
        logits = model(
            mx.array(token_ids[start : start + chunk_size], mx.uint32)[None],
            cache=cache,
        )
        mx.eval(logits)
    return cache


def _logit_distance(value, oracle) -> dict:
    delta = value[:, -1].astype(mx.float32) - oracle[:, -1].astype(mx.float32)
    mx.eval(delta)
    return {
        "max_abs": float(mx.max(mx.abs(delta))),
        "rms": float(mx.sqrt(mx.mean(mx.square(delta)))),
        "argmax_equal": bool(
            mx.array_equal(
                mx.argmax(value[:, -1], -1), mx.argmax(oracle[:, -1], -1)
            ).item()
        ),
    }


def _token_differences(left: list[int], right: list[int]) -> int:
    return abs(len(left) - len(right)) + sum(a != b for a, b in zip(left, right))


def _require_absorbed(before: int, label: str) -> None:
    after = patch.mla_absorbed_verify_stats()["absorbed"]
    if after <= before:
        raise RuntimeError(f"Rapid arm did not use absorbed MLA during {label}")


def _validate_absorbed_point(context: int, width: int, modules: list) -> None:
    cache_len = context + width
    if cache_len < patch.MIN_CACHE_LENGTH:
        raise SystemExit(
            f"context={context}, width={width} has post-update cache {cache_len}; "
            f"absorbed MLA requires at least {patch.MIN_CACHE_LENGTH}"
        )
    for module in modules:
        limit = patch.max_absorbed_queries(
            int(module.kv_lora_rank),
            int(module.qk_nope_head_dim),
            int(module.v_head_dim),
            cache_len,
        )
        if width > limit:
            raise SystemExit(
                f"context={context}, width={width} exceeds absorbed MLA "
                f"crossover {limit} for {type(module).__name__}"
            )


def main() -> None:
    args = _parse_args()
    invalid = []
    if args.width < 2:
        invalid.append("width must be >= 2")
    if args.repeats < 2:
        invalid.append("repeats must be >= 2")
    if any(context < 1 for context in args.contexts):
        invalid.append("contexts must be positive")
    if args.chunk_size < 1:
        invalid.append("chunk-size must be positive")
    if args.oracle_context < 0:
        invalid.append("oracle-context must be non-negative")
    if args.oracle_context and args.oracle_cases < 1:
        invalid.append("oracle-cases must be positive when oracle is enabled")
    if args.suffix_repeats < 0:
        invalid.append("suffix-repeats must be non-negative")
    if args.suffix_repeats and args.suffix_max_tokens < 1:
        invalid.append("suffix-max-tokens must be positive when suffix is enabled")
    if invalid:
        raise SystemExit("; ".join(invalid))

    patch.install_mla_absorbed_verify()
    stats = patch.mla_absorbed_verify_stats()
    if stats["provider"] != "rapid":
        raise SystemExit(f"Rapid MLA provider is not active: {stats}")

    from mlx_lm.models import mla

    originals = mla._RAPID_MLX_MLA_ABSORBED_ORIGINALS
    targets = {}
    for module_name, class_name in originals:
        module = __import__(f"mlx_lm.models.{module_name}", fromlist=[class_name])
        cls = getattr(module, class_name)
        cls.rapid_call = cls.__call__
        targets[(module_name, class_name)] = cls

    mx.random.seed(args.seed)
    model, tokenizer = load(args.model, revision=args.revision)
    active_classes = tuple(targets.values())
    active_modules = [
        module for module in model.modules() if isinstance(module, active_classes)
    ]
    if not active_modules:
        supported = ", ".join(f"{mod}.{cls}" for mod, cls in sorted(targets))
        raise SystemExit(
            f"loaded model does not contain a patched MLA class; supported: {supported}"
        )
    for context in args.contexts:
        _validate_absorbed_point(context, args.width, active_modules)
    if args.oracle_context:
        _validate_absorbed_point(args.oracle_context, args.width, active_modules)
    seed_ids = tokenizer.encode(" reproducible MLA benchmark")
    if not seed_ids:
        raise SystemExit("tokenizer produced no input IDs")
    verify = mx.array(
        [seed_ids[index % len(seed_ids)] for index in range(args.width)], mx.uint32
    )[None]

    rows = []
    for context in args.contexts:
        repeated = [seed_ids[index % len(seed_ids)] for index in range(context)]
        _set_arm(targets, originals, "stock")
        base_cache = _prefill(model, repeated, args.chunk_size)

        # Compile both shapes before sampling. ABBA order counters slow drift.
        arm_logits = {}
        for arm in ("stock", "rapid"):
            _set_arm(targets, originals, arm)
            absorbed_before = patch.mla_absorbed_verify_stats()["absorbed"]
            _, arm_logits[arm] = _timed_forward(model, verify, _clone_cache(base_cache))
            if arm == "rapid":
                _require_absorbed(absorbed_before, f"{context}-token warmup")
        samples = {"stock": [], "rapid": []}
        order = ("stock", "rapid", "rapid", "stock")
        while len(samples["stock"]) < args.repeats:
            for arm in order:
                if len(samples[arm]) >= args.repeats:
                    continue
                _set_arm(targets, originals, arm)
                absorbed_before = patch.mla_absorbed_verify_stats()["absorbed"]
                elapsed, arm_logits[arm] = _timed_forward(
                    model, verify, _clone_cache(base_cache)
                )
                if arm == "rapid":
                    _require_absorbed(absorbed_before, f"{context}-token sample")
                samples[arm].append(round(elapsed, 3))

        stock = arm_logits["stock"][:, -1].astype(mx.float32)
        rapid = arm_logits["rapid"][:, -1].astype(mx.float32)
        delta = stock - rapid
        mx.eval(delta)
        stock_median = statistics.median(samples["stock"])
        rapid_median = statistics.median(samples["rapid"])
        rows.append(
            {
                "context": context,
                "width": args.width,
                "stock_ms": samples["stock"],
                "rapid_ms": samples["rapid"],
                "stock_median_ms": stock_median,
                "rapid_median_ms": rapid_median,
                "speedup": stock_median / rapid_median,
                "max_abs_logit": float(mx.max(mx.abs(delta))),
                "rms_logit": float(mx.sqrt(mx.mean(mx.square(delta)))),
                "argmax_equal": bool(
                    mx.array_equal(mx.argmax(stock, -1), mx.argmax(rapid, -1)).item()
                ),
            }
        )

    oracle_result = None
    if args.oracle_context:
        _set_arm(targets, originals, "stock")
        repeated = [
            seed_ids[index % len(seed_ids)] for index in range(args.oracle_context)
        ]
        base_cache = _prefill(model, repeated, args.chunk_size)
        rng = random.Random(args.seed)
        vocab_size = int(getattr(tokenizer, "vocab_size", 65536))
        oracle_rows = []
        for case in range(args.oracle_cases):
            ids = [rng.randrange(vocab_size) for _ in range(args.width)]
            tokens = mx.array(ids, mx.uint32)[None]
            _set_arm(targets, originals, "stock")
            stock = model(tokens, cache=_clone_cache(base_cache))
            oracle_cache = _clone_cache(base_cache)
            oracle = None
            for token_id in ids:
                oracle = model(mx.array([[token_id]], mx.uint32), cache=oracle_cache)
            _set_arm(targets, originals, "rapid")
            absorbed_before = patch.mla_absorbed_verify_stats()["absorbed"]
            rapid = model(tokens, cache=_clone_cache(base_cache))
            mx.eval(stock, rapid, oracle)
            _require_absorbed(absorbed_before, f"oracle case {case}")
            oracle_rows.append(
                {
                    "case": case,
                    "tokens": ids,
                    "stock": _logit_distance(stock, oracle),
                    "rapid": _logit_distance(rapid, oracle),
                }
            )
        oracle_result = {
            "context": args.oracle_context,
            "cases": args.oracle_cases,
            "stock_argmax_matches": sum(
                r["stock"]["argmax_equal"] for r in oracle_rows
            ),
            "rapid_argmax_matches": sum(
                r["rapid"]["argmax_equal"] for r in oracle_rows
            ),
            "stock_mean_rms": statistics.mean(r["stock"]["rms"] for r in oracle_rows),
            "rapid_mean_rms": statistics.mean(r["rapid"]["rms"] for r in oracle_rows),
            "rows": oracle_rows,
        }

    suffix_result = None
    if args.suffix_repeats:
        from scripts.bench_suffix_decoding import _run_suffix, _run_vanilla

        _set_arm(targets, originals, "stock")
        vanilla = _run_vanilla(
            model, tokenizer, LONG_CODE_EDIT_PROMPT, args.suffix_max_tokens
        )
        runs = {"stock": [], "rapid": []}
        outputs = {"stock": [], "rapid": []}
        order = ("stock", "rapid", "rapid", "stock")
        while len(runs["stock"]) < args.suffix_repeats:
            for arm in order:
                if len(runs[arm]) >= args.suffix_repeats:
                    continue
                _set_arm(targets, originals, arm)
                absorbed_before = patch.mla_absorbed_verify_stats()["absorbed"]
                run = _run_suffix(
                    model,
                    tokenizer,
                    LONG_CODE_EDIT_PROMPT,
                    args.suffix_max_tokens,
                    max_draft=8,
                    max_suffix=4,
                    min_conf=0.3,
                )
                runs[arm].append(run.tps)
                outputs[arm].append(run.out_tokens)
                if arm == "rapid":
                    _require_absorbed(absorbed_before, "suffix repetition")
        for arm, repetitions in outputs.items():
            if any(tokens != repetitions[0] for tokens in repetitions[1:]):
                raise RuntimeError(f"{arm} suffix output changed between repetitions")
        paired_diffs = [
            _token_differences(stock, rapid)
            for stock, rapid in zip(outputs["stock"], outputs["rapid"])
        ]
        stock_output = outputs["stock"][0]
        rapid_output = outputs["rapid"][0]
        stock_median = statistics.median(runs["stock"])
        rapid_median = statistics.median(runs["rapid"])
        suffix_result = {
            "prompt_tokens": len(tokenizer.encode(LONG_CODE_EDIT_PROMPT)),
            "max_completion_tokens": args.suffix_max_tokens,
            "vanilla_tps": vanilla.tps,
            "stock_tps": runs["stock"],
            "rapid_tps": runs["rapid"],
            "stock_median_tps": stock_median,
            "rapid_median_tps": rapid_median,
            "speedup": rapid_median / stock_median,
            "vanilla_output_tokens": len(vanilla.out_tokens),
            "stock_output_tokens": len(stock_output),
            "rapid_output_tokens": len(rapid_output),
            "stock_vs_vanilla_diffs": _token_differences(
                stock_output, vanilla.out_tokens
            ),
            "rapid_vs_vanilla_diffs": _token_differences(
                rapid_output, vanilla.out_tokens
            ),
            "stock_vs_rapid_diffs": paired_diffs[0],
            "stock_vs_rapid_diffs_by_repeat": paired_diffs,
        }

    result = {
        "model": args.model,
        "revision": args.revision,
        "mlx": mx.__version__,
        "platform": platform.platform(),
        "seed": args.seed,
        "rows": rows,
        "oracle": oracle_result,
        "suffix": suffix_result,
        "stats": patch.mla_absorbed_verify_stats(),
    }
    rendered = format_bench_json(result, __file__)
    print(rendered)
    if args.json:
        write_bench_json(args.json, result, __file__)


if __name__ == "__main__":
    main()
