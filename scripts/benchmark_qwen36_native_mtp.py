#!/usr/bin/env python3
"""Reproduce the Qwen3.6-35B-A3B greedy native-MTP qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics

import mlx.core as mx
from mlx_vlm import generate, load
from mlx_vlm.speculative.drafters import load_drafter
from mlx_vlm.utils import get_model_path

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
        'Return JSON only with keys "risk", "mitigations", and "owner". Assess the '
        "risk of deploying a database migration without a rollback plan. "
        '"mitigations" must be an array of exactly three short strings.'
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


def _run(model, processor, prompt: str, drafter, *, max_tokens: int) -> dict:
    kwargs = {"max_tokens": max_tokens, "temperature": 0.0, "verbose": False}
    if drafter is not None:
        for name in ("accept_lens", "draft_lens"):
            values = getattr(drafter, name, None)
            if isinstance(values, list):
                values.clear()
        kwargs.update(
            draft_model=drafter,
            draft_kind="mtp",
            draft_block_size=QWEN36_35B_4BIT.block_size,
        )
    result = generate(model, processor, prompt, **kwargs)
    mx.synchronize()
    speculative = None
    if drafter is not None:
        accepted = list(getattr(drafter, "accept_lens", None) or [])
        drafted = list(getattr(drafter, "draft_lens", None) or [])
        accepted_total = sum(accepted)
        drafted_total = sum(drafted)
        speculative = {
            "rounds": len(accepted),
            "accepted_drafts": accepted_total,
            "drafted_tokens": drafted_total,
            "acceptance": (accepted_total / drafted_total if drafted_total else None),
        }
    return {
        "sha256": hashlib.sha256(result.text.encode()).hexdigest(),
        "tokens": result.generation_tokens,
        "tps": result.generation_tps,
        "peak_gb": result.peak_memory,
        "speculative": speculative,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--max-tokens", type=int, default=192)
    args = parser.parse_args()
    if args.rounds < 1 or args.max_tokens < 1:
        parser.error("--rounds and --max-tokens must be positive")

    pair = QWEN36_35B_4BIT
    target_path = get_model_path(pair.target_repo, revision=pair.target_revision)
    drafter_path = get_model_path(pair.drafter_repo, revision=pair.drafter_revision)
    model, processor = load(str(target_path))
    drafter, kind = load_drafter(str(drafter_path), kind="mtp")
    if kind != "mtp":
        raise RuntimeError(f"expected MTP drafter, got {kind!r}")
    drafter.bind(model)

    rows = []
    for case, raw_prompt in PROMPTS.items():
        prompt = _render(processor, raw_prompt)
        for round_index in range(args.rounds):
            modes = ["ar", "mtp"] if round_index % 2 == 0 else ["mtp", "ar"]
            pair_rows = []
            for mode in modes:
                mx.clear_cache()
                row = _run(
                    model,
                    processor,
                    prompt,
                    drafter if mode == "mtp" else None,
                    max_tokens=args.max_tokens,
                )
                row.update(
                    case=case,
                    round=round_index,
                    mode=mode,
                )
                rows.append(row)
                pair_rows.append(row)
                print(json.dumps(row, sort_keys=True))
            if len({row["sha256"] for row in pair_rows}) != 1:
                raise RuntimeError(
                    f"greedy output mismatch for {case} round {round_index}"
                )

    medians = {
        mode: statistics.median(row["tps"] for row in rows if row["mode"] == mode)
        for mode in ("ar", "mtp")
    }
    print(
        json.dumps(
            {
                "event": "summary",
                "pairs": len(rows) // 2,
                "all_exact": True,
                "ar_median_tps": medians["ar"],
                "mtp_median_tps": medians["mtp"],
                "speedup": medians["mtp"] / medians["ar"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
