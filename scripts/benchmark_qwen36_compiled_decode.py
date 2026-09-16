#!/usr/bin/env python3
"""Compare Qwen3.6-35B ordinary decode with and without compiled replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

from rapid_mlx.compiled_decode import CompiledDecodeStep, convert_cache
from rapid_mlx.compiled_precision import install_qwen35_attention_gate_precision
from rapid_mlx.gdn_in_proj_fusion import fuse_gdn_in_proj
from rapid_mlx.moe_fusion import fuse_gate_up
from rapid_mlx.patches.qwen3_5_eager_dispatch import install_qwen3_5_eager_dispatch
from rapid_mlx.qwen35_fused_gdn_decode import install_qwen35_fused_gdn_decode
from rapid_mlx.qwen35_moe_router import install_qwen35_moe_router

PROMPTS = {
    "coding": (
        "Implement an async Python worker pool with bounded concurrency, "
        "ordered results, cancellation cleanup, type hints, and three tests."
    ),
    "reasoning": (
        "A price is discounted 20% and then taxed 8%. The final price is "
        "$86.40. Find the original price and show the calculation succinctly."
    ),
    "creative": (
        "Write a vivid scene about a lighthouse keeper discovering that fog "
        "carries whispered memories. Avoid cliches."
    ),
    "json": (
        'Return JSON only with keys "risk", "mitigations", and "owner". '
        '"mitigations" must contain exactly three short strings.'
    ),
    "tool": (
        "A weather tool accepts city and unit. Return only JSON arguments for "
        "Tokyo in Celsius."
    ),
}


def _tokens(tokenizer, prompt: str) -> mx.array:
    return mx.array(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        ),
        dtype=mx.int32,
    )


def _run(model, prompt: mx.array, *, max_tokens: int, compiled: bool) -> dict:
    cache = make_prompt_cache(model)
    logits = model(prompt[None], cache=cache)[:, -1, :]
    token = mx.argmax(logits, axis=-1)
    mx.eval(token, [layer.state for layer in cache])
    step = None
    if compiled:
        cache[:] = convert_cache(cache)
        step = CompiledDecodeStep(model, cache)

    output = []
    previous = None
    started = time.perf_counter()
    for _ in range(max_tokens):
        logits = (
            step(token[:, None])
            if step is not None
            else model(token[:, None], cache=cache)
        )
        next_token = mx.argmax(logits[:, -1, :], axis=-1)
        mx.async_eval(next_token)
        if step is not None and previous is not None:
            step.confirm_oldest(token)
        output.append(int(token.item()))
        token = next_token
        previous = logits
    if step is not None:
        step.drain_pending()
    mx.synchronize()
    elapsed = time.perf_counter() - started
    digest = hashlib.sha256(
        b"".join(value.to_bytes(4, "little") for value in output)
    ).hexdigest()
    return {
        "tokens": len(output),
        "token_sha256": digest,
        "tps": len(output) / elapsed,
        "receipt": step.receipt() if step is not None else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()
    if not args.model.is_dir():
        parser.error("--model must be an already-cached local snapshot directory")
    if args.rounds < 1 or args.max_tokens < 1:
        parser.error("--rounds and --max-tokens must be positive")

    model, tokenizer = load(str(args.model))
    install_qwen3_5_eager_dispatch()
    fusion = {
        "gate_up": fuse_gate_up(model),
        "router": install_qwen35_moe_router(model),
        "gdn_input": fuse_gdn_in_proj(model),
        "gdn_decode": install_qwen35_fused_gdn_decode(model),
        "attention_gate": install_qwen35_attention_gate_precision(model),
    }
    print(json.dumps({"event": "model_ready", "fusion": fusion}), flush=True)

    # Exclude compilation and Metal warmup from the paired campaign.
    warmup = _tokens(tokenizer, PROMPTS["coding"])
    _run(model, warmup, max_tokens=min(args.max_tokens, 24), compiled=False)
    _run(model, warmup, max_tokens=min(args.max_tokens, 24), compiled=True)

    rows = []
    for case, prompt_text in PROMPTS.items():
        prompt = _tokens(tokenizer, prompt_text)
        for round_index in range(args.rounds):
            modes = (False, True) if round_index % 2 == 0 else (True, False)
            pair = {}
            for compiled in modes:
                mx.clear_cache()
                pair[compiled] = _run(
                    model,
                    prompt,
                    max_tokens=args.max_tokens,
                    compiled=compiled,
                )
            exact = pair[False]["token_sha256"] == pair[True]["token_sha256"]
            row = {
                "case": case,
                "round": round_index,
                "stock_tps": pair[False]["tps"],
                "compiled_tps": pair[True]["tps"],
                "speedup": pair[True]["tps"] / pair[False]["tps"],
                "exact": exact,
                "receipt": pair[True]["receipt"],
            }
            print(json.dumps(row, sort_keys=True), flush=True)
            rows.append(row)
    summary = {
        "event": "summary",
        "pairs": len(rows),
        "all_exact": all(row["exact"] for row in rows),
        "stock_median_tps": statistics.median(row["stock_tps"] for row in rows),
        "compiled_median_tps": statistics.median(row["compiled_tps"] for row in rows),
        "paired_speedup_median": statistics.median(row["speedup"] for row in rows),
        "positive_pairs": sum(row["speedup"] > 1.0 for row in rows),
        "peak_gib": mx.get_peak_memory() / 2**30,
    }
    print(json.dumps(summary, sort_keys=True), flush=True)
    if not summary["all_exact"] or summary["positive_pairs"] < len(rows) - 1:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
