#!/usr/bin/env python3
"""Reproduce the Qwen3.5/3.6 eager layer-dispatch qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics

import mlx.core as mx
from mlx_vlm import generate, load
from mlx_vlm.utils import get_model_path

from rapid_mlx.patches import qwen3_5_eager_dispatch as eager
from rapid_mlx.speculative.native_mtp.eligibility import QWEN36_35B_4BIT

PROMPTS = {
    "coding": (
        "Write a Python function merge_intervals(intervals) that returns sorted, "
        "non-overlapping intervals. Include type hints, handle empty input, and "
        "explain time complexity briefly."
    ),
    "reasoning": (
        "A shop discounts an item by 20%, then adds 8% sales tax. The final price "
        "is $86.40. Find the original price and show the calculation succinctly."
    ),
    "creative": (
        "Write a vivid 120-word scene about a lighthouse keeper discovering that "
        "the fog is carrying whispered memories. Avoid cliches."
    ),
    "json": (
        'Return JSON only with keys "risk", "mitigations", and "owner". Assess '
        "deploying a database migration without a rollback plan. "
        '"mitigations" must contain exactly three short strings.'
    ),
    "tool": (
        "You have a weather tool with schema weather(city: string, unit: "
        "'celsius'|'fahrenheit'). Produce only the JSON arguments needed to check "
        "the weather in Tokyo in Celsius."
    ),
}


def _render(processor, prompt: str) -> str:
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def _run(model, processor, prompt: str, *, max_tokens: int) -> dict:
    result = generate(
        model,
        processor,
        prompt,
        max_tokens=max_tokens,
        temperature=0.0,
        verbose=False,
    )
    mx.synchronize()
    return {
        "sha256": hashlib.sha256(result.text.encode()).hexdigest(),
        "tokens": result.generation_tokens,
        "tps": result.generation_tps,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--max-tokens", type=int, default=192)
    args = parser.parse_args()
    if args.rounds < 1 or args.max_tokens < 1:
        parser.error("--rounds and --max-tokens must be positive")

    pair = QWEN36_35B_4BIT
    model_path = get_model_path(pair.target_repo, revision=pair.target_revision)
    eager.install_qwen3_5_eager_dispatch()
    model, processor = load(str(model_path))

    # Keep warmup outside the measured rows and warm both scheduling modes.
    warmup = _render(processor, next(iter(PROMPTS.values())))
    for enabled in (False, True):
        eager._ENABLED = enabled
        _run(model, processor, warmup, max_tokens=min(args.max_tokens, 32))

    rows = []
    paired_ratios = []
    for case, raw_prompt in PROMPTS.items():
        prompt = _render(processor, raw_prompt)
        for round_index in range(args.rounds):
            modes = ("off", "on") if round_index % 2 == 0 else ("on", "off")
            pair_rows = {}
            for mode in modes:
                eager._ENABLED = mode == "on"
                mx.clear_cache()
                pair_rows[mode] = _run(
                    model,
                    processor,
                    prompt,
                    max_tokens=args.max_tokens,
                )
            if pair_rows["off"]["sha256"] != pair_rows["on"]["sha256"]:
                raise RuntimeError(
                    f"greedy output mismatch for {case} round {round_index}"
                )
            ratio = pair_rows["on"]["tps"] / pair_rows["off"]["tps"]
            row = {
                "case": case,
                "round": round_index,
                "off_tps": pair_rows["off"]["tps"],
                "on_tps": pair_rows["on"]["tps"],
                "speedup": ratio,
                "tokens": pair_rows["on"]["tokens"],
            }
            rows.append(row)
            paired_ratios.append(ratio)
            print(json.dumps(row, sort_keys=True))

    print(
        json.dumps(
            {
                "event": "summary",
                "pairs": len(rows),
                "all_exact": True,
                "off_median_tps": statistics.median(row["off_tps"] for row in rows),
                "on_median_tps": statistics.median(row["on_tps"] for row in rows),
                "paired_speedup_median": statistics.median(paired_ratios),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
