#!/usr/bin/env python3
"""Run the exact K4 V4.1 DSpark candidate across a fixed prompt suite."""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
from transformers import PreTrainedTokenizerFast

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from benchmark_deepseek_v41_dspark import (  # noqa: E402
    _install_direct_down_qmv,
    _prompt,
    _run_ar,
    _run_batched_dspark,
)

from vllm_mlx.models.deepseek_v41_native import dspark as rapid_dspark  # noqa: E402
from vllm_mlx.models.deepseek_v41_native.load import load  # noqa: E402

PROMPTS = (
    (
        "code",
        "Write a Python function that merges overlapping integer intervals. "
        "Return only the function and keep it concise.",
    ),
    (
        "reasoning",
        "A train travels 180 km at 60 km/h, waits 35 minutes, then travels "
        "120 km at 80 km/h. Explain the total elapsed time step by step.",
    ),
    (
        "structured",
        'Return only JSON with keys "risk", "cause", and "next_action". A checkout '
        "API returns HTTP 503 after its inventory dependency times out.",
    ),
    (
        "chinese",
        "请用简洁的中文解释为什么数据库事务需要原子性，并给出一个转账失败的例子。",
    ),
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _at_least_two(value: str) -> int:
    parsed = int(value)
    if parsed < 2:
        raise argparse.ArgumentTypeError("must be at least 2")
    return parsed


def _summary(rows: list[dict]) -> dict:
    transitions = sum(int(row["decode_transitions"]) for row in rows)
    seconds = sum(float(row["decode_seconds"]) for row in rows)
    return {
        "prompts": len(rows),
        "ar_compared_runs": sum(row["greedy_matches_ar"] is not None for row in rows),
        "exact_prompts": sum(bool(row["greedy_matches_ar"]) for row in rows),
        "decode_transitions": transitions,
        "decode_seconds": seconds,
        "weighted_tok_s": transitions / seconds if seconds else None,
        "mean_accepted_per_block": (
            sum(float(row["accepted_per_block"]) for row in rows) / len(rows)
            if rows
            else None
        ),
    }


def _first_mismatch(left: list[int], right: list[int]) -> int | None:
    for index, (left_token, right_token) in enumerate(zip(left, right)):
        if left_token != right_token:
            return index
    return None if len(left) == len(right) else min(len(left), len(right))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--tokens", type=_positive_int, default=128)
    parser.add_argument("--repeats", type=_at_least_two, default=2)
    parser.add_argument("--eval-interval", type=_positive_int, default=40)
    parser.add_argument("--collect-target-margins", action="store_true")
    parser.add_argument("--skip-ar", action="store_true")
    return parser.parse_args()


def _validate_inputs(args) -> None:
    for label in ("target", "overlay"):
        path = getattr(args, label)
        if not path.is_dir():
            raise SystemExit(f"--{label.replace('_', '-')} must be a directory: {path}")


def _prepare_target(target: Path, overlay: Path, eval_interval: int):
    model, _ = load(str(target.resolve()), lazy=False)
    model.eval_interval = eval_interval
    model._dspark_overlay_path = str(overlay.resolve())
    # AR and speculative decoding must use the same target kernels. Installing
    # direct down-QMV after the reference run would inflate the measured K4
    # speedup and compare outputs from different numerical implementations.
    replaced = _install_direct_down_qmv(model)
    return model, replaced


def main() -> None:
    args = parse_args()
    _validate_inputs(args)

    started = time.perf_counter()
    model, replaced = _prepare_target(args.target, args.overlay, args.eval_interval)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(args.target.resolve() / "tokenizer.json")
    )
    encoded = {
        prompt_id: tokenizer.encode(_prompt(text), add_special_tokens=False)
        for prompt_id, text in PROMPTS
    }
    print(
        json.dumps(
            {
                "event": "suite_loaded",
                "seconds": time.perf_counter() - started,
                "prompts": len(encoded),
                "tokens_per_prompt": args.tokens,
                "active_gb": mx.get_active_memory() / 1e9,
                "peak_gb": mx.get_peak_memory() / 1e9,
            }
        ),
        flush=True,
    )
    references = {}
    ar_rows = []
    if not args.skip_ar:
        for prompt_id, input_ids in encoded.items():
            output, metrics = _run_ar(model, input_ids, args.tokens, eos_id=1)
            references[prompt_id] = output
            ar_rows.append(metrics)
            print(
                json.dumps(
                    {
                        "event": "suite_ar",
                        "prompt_id": prompt_id,
                        **metrics,
                        "output_tokens": output,
                    }
                ),
                flush=True,
            )

    rows = []
    first_outputs = {}
    stable_prompts = set(encoded)
    for repeat in range(args.repeats):
        for prompt_id, input_ids in encoded.items():
            with contextlib.redirect_stdout(sys.stderr):
                output, metrics = _run_batched_dspark(
                    model,
                    input_ids,
                    args.tokens,
                    1,
                    rapid_dspark.Weights,
                    rapid_dspark.DSpark,
                    verify_k=4,
                    packed_mtp=True,
                    collect_target_margins=args.collect_target_margins,
                )
            reference = references.get(prompt_id)
            row = {
                "event": "suite_k4",
                "repeat": repeat,
                "prompt_id": prompt_id,
                **metrics,
                "greedy_matches_ar": (
                    output == reference if reference is not None else None
                ),
                "first_mismatch_index": (
                    _first_mismatch(output, reference)
                    if reference is not None
                    else None
                ),
                "output_tokens": output,
                "active_gb": mx.get_active_memory() / 1e9,
                "peak_gb": mx.get_peak_memory() / 1e9,
            }
            rows.append(row)
            if repeat == 0:
                first_outputs[prompt_id] = output
            elif output != first_outputs[prompt_id]:
                stable_prompts.discard(prompt_id)
            print(json.dumps(row), flush=True)

    ar_transitions = sum(int(row["decode_transitions"]) for row in ar_rows)
    ar_seconds = sum(float(row["decode_seconds"]) for row in ar_rows)
    ar_weighted_tok_s = ar_transitions / ar_seconds if ar_seconds else None
    summary = _summary(rows)
    speedup = (
        summary["weighted_tok_s"] / ar_weighted_tok_s
        if summary["weighted_tok_s"] is not None and ar_weighted_tok_s
        else None
    )
    print(
        json.dumps(
            {
                "event": "suite_summary",
                "repeats": args.repeats,
                "repeat_stable_prompts": len(stable_prompts),
                "replaced_layers": replaced,
                **summary,
                "ar_weighted_tok_s": ar_weighted_tok_s,
                "speedup_vs_ar": speedup,
                "peak_gb": mx.get_peak_memory() / 1e9,
            }
        ),
        flush=True,
    )
    if len(stable_prompts) != len(encoded):
        raise SystemExit(
            "K4 repeat-stability gate failed: "
            f"{len(stable_prompts)}/{len(encoded)} prompts were stable"
        )


if __name__ == "__main__":
    main()
