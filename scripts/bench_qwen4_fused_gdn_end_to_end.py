#!/usr/bin/env python3
"""Interleaved real-model gate for Qwen4 fused GDN decode."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

try:
    from scripts.bench_metadata import format_bench_json, write_bench_json
except ImportError:  # direct `python scripts/bench_*.py` execution
    from bench_metadata import format_bench_json, write_bench_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--prompt",
        default="Write the integers 1 through 200, one per line.",
    )
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def expected_path_counts(mode: str, fused_layers: int, generated_tokens: int):
    """Return the exact path counts for this gate's one-chunk prefill."""
    if mode == "stock":
        return 0, 0
    if mode != "fused":
        raise ValueError(f"unknown mode {mode!r}")
    # The first prefill call initializes every GDN cache on stock. The final
    # prompt token and each generated/look-ahead token then use fused decode.
    return fused_layers * (generated_tokens + 1), fused_layers


def run(args):
    if args.max_tokens < 2:
        raise SystemExit("--max-tokens must be at least 2")
    if args.repeats < 1:
        raise SystemExit("--repeats must be at least 1")

    import mlx.core as mx
    from mlx_lm.generate import generate_step
    from mlx_lm.sample_utils import make_sampler
    from mlx_lm.utils import load

    from vllm_mlx.models.qwen4_exp import (
        qwen4_fused_gdn_stats,
        set_qwen4_fused_gdn_mode,
    )
    from vllm_mlx.utils.tokenizer import _register_vendored_archs

    _register_vendored_archs()
    mx.set_default_device(mx.gpu)
    model, tokenizer = load(str(args.model.resolve()))
    model.eval()
    prompt = mx.array(tokenizer.encode(args.prompt))
    if not 2 <= prompt.size <= 2049:
        raise SystemExit(
            "--prompt must encode to 2..2049 tokens so the decode gate has "
            "one expected stock prefill and an exact fused-call contract"
        )
    sampler = make_sampler(temp=0.0)

    observations = []
    orders = (("stock", "fused"), ("fused", "stock"))
    for repeat in range(args.repeats):
        for mode in orders[repeat % len(orders)]:
            fused_layers = set_qwen4_fused_gdn_mode(model, mode)
            before = qwen4_fused_gdn_stats(model)
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
            after = qwen4_fused_gdn_stats(model)
            if len(tokens) < 2 or first_token_at is None:
                raise RuntimeError(
                    f"generation produced {len(tokens)} tokens; need at least 2"
                )
            ttft = first_token_at - started
            decode_seconds = ended - first_token_at
            if decode_seconds <= 0:
                raise RuntimeError("decode timer did not advance")
            fused_calls = after["fused_calls"] - before["fused_calls"]
            fallbacks = after["fallbacks"] - before["fallbacks"]
            expected_fused_calls, expected_fallbacks = expected_path_counts(
                mode, fused_layers, len(tokens)
            )
            observations.append(
                {
                    "mode": mode,
                    "repeat": repeat + 1,
                    "tokens": len(tokens),
                    "token_sha256": hashlib.sha256(
                        json.dumps(tokens, separators=(",", ":")).encode()
                    ).hexdigest(),
                    "ttft_seconds": ttft,
                    "decode_seconds": decode_seconds,
                    "decode_tokens_per_second": (len(tokens) - 1) / decode_seconds,
                    "fused_calls": fused_calls,
                    "expected_fused_calls": expected_fused_calls,
                    "fallbacks": fallbacks,
                    "expected_fallbacks": expected_fallbacks,
                    "path_counts_exact": fused_calls == expected_fused_calls
                    and fallbacks == expected_fallbacks,
                    "last_fallbacks": after["last_fallbacks"],
                }
            )
            mx.clear_cache()

    hashes = {item["token_sha256"] for item in observations}
    path_counts_exact = all(item["path_counts_exact"] for item in observations)
    medians = {
        mode: statistics.median(
            item["decode_tokens_per_second"]
            for item in observations
            if item["mode"] == mode
        )
        for mode in ("stock", "fused")
    }
    return {
        "correctness": {
            "passed": len(hashes) == 1 and path_counts_exact,
            "token_exact": len(hashes) == 1,
            "path_counts_exact": path_counts_exact,
            "hashes": sorted(hashes),
        },
        "observations": observations,
        "median_decode_tokens_per_second": medians,
        "median_speedup_percent": 100.0 * (medians["fused"] / medians["stock"] - 1.0),
    }


def main() -> int:
    args = parse_args()
    result = run(args)
    payload = format_bench_json(result, __file__, indent=2, sort_keys=True)
    print(payload)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_bench_json(args.output, result, __file__, indent=2, sort_keys=True)
    return 0 if result["correctness"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
