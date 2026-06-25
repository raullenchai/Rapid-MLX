#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""DFlash speculative-decode decode-tok/s bench (R15-P1 #313).

Compares ``--spec-decode dflash`` against ``--spec-decode none`` on a
Qwen3.5 / Qwen3.6 checkpoint with the matching block-diffusion drafter
bound in :mod:`vllm_mlx.spec_decode.dflash.drafter_registry` (or via
``--dflash-drafter-path``). Mirrors :file:`bench_spec_decode_mtp.py`'s
8-prompt × 3-run protocol so the two backends are directly comparable
on the same workload.

DEFERRED: at landing time the GPU is contended with the Stage B
PonyExl3 Viterbi conversion (PID 56486). The script is committed but
NOT executed in the same agent run that opens the PR. Operators run::

    python bench/bench_spec_decode_dflash.py \\
        --model qwen3.5-9b-4bit \\
        --runs 3 \\
        --max-tokens 256

Outputs JSON (default) or a markdown table for the follow-up PR
comment::

    python bench/bench_spec_decode_dflash.py --format markdown

Expected numbers (paper 2410.04097, M5 Max projection): baseline
30.95 tok/s, DFlash temp=0 135.34 tok/s (4.37×, accept ratio depends
on workload). The bench script reports the same triplet.

Dry-run mode
------------

``--dry-run`` skips the model load + generation, runs through
argument parsing + condition setup only, and prints the planned
matrix. Useful for CI smoke (we run it on every PR via the existing
bench-script-validates job) and for in-agent validation that the
script wires up cleanly without burning GPU cycles.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any

# Same 8 diverse prompts as ``bench_spec_decode_mtp.py`` (PR #990's
# bench script set). Kept verbatim so the MTP-vs-DFlash speedup
# numbers are directly comparable on identical inputs.
_BENCH_PROMPTS: tuple[str, ...] = (
    "Write a Python function that computes the n-th Fibonacci number "
    "using memoization. Include type hints, a docstring, and a small "
    "example call. Keep it under 30 lines.",
    "Explain how a Bloom filter works, including its false-positive "
    "guarantees, when you'd choose it over a hash set, and a brief "
    "complexity analysis.",
    "Return a JSON object describing the top 3 NoSQL databases by "
    "popularity, with fields name, category, primary_use_case, and "
    "year_released. No prose, just the JSON.",
    "Write a 200-word reflection on the role of perseverance in "
    "scientific discovery, citing a specific historical example.",
    "Write a short dialogue (10 lines) between a junior engineer and a "
    "senior engineer reviewing a pull request that introduces a race "
    "condition.",
    "Summarize the plot of Moby-Dick in 3 paragraphs. Focus on Ahab's "
    "obsession and how it drives the crew's fate.",
    "A train leaves station A at 9:00 traveling at 60 km/h. Another "
    "train leaves station B at 9:30 traveling at 80 km/h on the same "
    "track in the opposite direction. The stations are 350 km apart. "
    "At what time do they meet? Show work.",
    "Translate the following sentence into formal French, then into "
    "Brazilian Portuguese, then into Japanese. Sentence: 'The early "
    "bird catches the worm.' Output: three labeled lines.",
)


@dataclass(frozen=True)
class RunResult:
    """One ``(condition, run_idx, prompt_idx)`` measurement.

    The extra ``tokens_saved`` field (vs the MTP RunResult) captures
    DFlash's block-bonus: a fully accepted block of size B saves up to
    ``B - 1`` extra tokens per attempt, not just 1.
    """

    condition: str
    run_idx: int
    prompt_idx: int
    decode_tok_per_sec: float
    n_tokens: int
    accept_attempts: int
    accept_count: int
    tokens_saved: int
    elapsed_seconds: float


@dataclass(frozen=True)
class ConditionSummary:
    """Pooled tok/s + accept ratio for one condition."""

    condition: str
    n_runs: int
    pooled_tok_per_sec: float
    p50_tok_per_sec: float
    p90_tok_per_sec: float
    accept_ratio: float
    mean_tokens_per_attempt: float
    speedup_vs_baseline: float | None
    notes: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default="qwen3.5-9b-4bit",
        help=(
            "Model alias or HF path (default: qwen3.5-9b-4bit). The "
            "matching DFlash drafter must be bound in the side-registry "
            "or passed via --dflash-drafter-path."
        ),
    )
    parser.add_argument(
        "--dflash-drafter-path",
        default="",
        help=(
            "Drafter HF path override (default: empty = use registry). "
            "Mirrors the rapid-mlx serve --dflash-drafter-path flag."
        ),
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=16,
        help="DFlash block size (default: 16, paper bench value).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Runs per condition (default: 3).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Decode budget per prompt (default: 256).",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.0,
        help=(
            "Sampling temperature (default: 0.0 = greedy = lossless "
            "contract enforced). DFlash temp>0 is not yet supported; "
            "the script raises if a non-zero value is passed."
        ),
    )
    parser.add_argument(
        "--format",
        choices=["json", "markdown"],
        default="json",
        help="Output format (default: json).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Skip the actual generation; print the planned bench "
            "matrix and exit. Used by CI to validate the script "
            "wires up without burning GPU cycles."
        ),
    )
    parser.add_argument(
        "--prompts",
        type=int,
        default=len(_BENCH_PROMPTS),
        help=(
            f"Number of prompts to run (default: {len(_BENCH_PROMPTS)} "
            "= full PR #990 mix)."
        ),
    )
    return parser.parse_args()


def _planned_matrix(args: argparse.Namespace) -> dict[str, Any]:
    """Return the bench plan as a JSON-serializable dict (dry-run mode)."""
    n_prompts = min(args.prompts, len(_BENCH_PROMPTS))
    return {
        "model": args.model,
        "drafter_override": args.dflash_drafter_path or None,
        "block_size": args.block_size,
        "runs_per_condition": args.runs,
        "max_tokens": args.max_tokens,
        "temp": args.temp,
        "conditions": ["none", "dflash"],
        "prompts": list(_BENCH_PROMPTS[:n_prompts]),
        "total_generations": 2 * args.runs * n_prompts,
        "estimated_wall_time_seconds_at_30_tok_per_sec": (
            2 * args.runs * n_prompts * args.max_tokens / 30.0
        ),
        "expected_speedup_paper": 4.37,
        "expected_baseline_tok_per_sec_m5_max": 30.95,
        "expected_dflash_tok_per_sec_m5_max": 135.34,
    }


def _run_once(
    *,
    model_alias: str,
    drafter_override: str,
    condition: str,
    prompt: str,
    block_size: int,
    max_tokens: int,
    temp: float,
) -> RunResult:
    """Run one generation under the requested condition.

    Imports ``mlx_lm`` and the rapid-mlx DFlash modules lazily so
    ``--dry-run`` doesn't pay the import cost.
    """
    import mlx.core as mx
    from mlx_lm import load

    from vllm_mlx.spec_decode.dflash import (
        DFlashAcceptCounter,
        get_dflash_drafter_path,
        get_global_counter,
    )

    model, tokenizer = load(model_alias)

    if condition == "dflash":
        from vllm_mlx.spec_decode.dflash.drafter import (
            MlxVlmBlockDiffusionDrafter,
        )
        from vllm_mlx.spec_decode.dflash.generator import dflash_generate_step

        drafter_path = drafter_override or get_dflash_drafter_path(model_alias)
        if not drafter_path:
            raise RuntimeError(
                f"No DFlash drafter bound for {model_alias!r}. Pass "
                "--dflash-drafter-path or register via "
                "vllm_mlx.spec_decode.dflash.register_dflash_drafter()."
            )
        drafter = MlxVlmBlockDiffusionDrafter(drafter_path, block_size=block_size)

    prompt_ids = mx.array(tokenizer.encode(prompt), mx.uint32)
    counter = DFlashAcceptCounter()
    prior_attempts = get_global_counter().snapshot().attempts
    prior_accepts = get_global_counter().snapshot().accepts

    t0 = time.perf_counter()
    n = 0
    if condition == "dflash":
        gen = dflash_generate_step(
            prompt_ids,
            model,
            drafter,
            block_size=block_size,
            max_tokens=max_tokens,
            temperature=temp,
            accept_counter=counter,
        )
        for _ in gen:
            n += 1
    else:
        from mlx_lm.generate import stream_generate

        for _resp in stream_generate(
            model,
            tokenizer,
            prompt,
            max_tokens=max_tokens,
        ):
            n += 1
            if n >= max_tokens:
                break

    elapsed = time.perf_counter() - t0

    if condition == "dflash":
        snap = counter.snapshot()
        snap_attempts, snap_accepts, tokens_saved = (
            snap.attempts,
            snap.accepts,
            snap.tokens_saved,
        )
    else:
        snap_attempts, snap_accepts, tokens_saved = 0, 0, 0
    # Sanity-check: global counter shouldn't have moved.
    assert get_global_counter().snapshot().attempts == prior_attempts
    assert get_global_counter().snapshot().accepts == prior_accepts

    tok_per_sec = n / elapsed if elapsed > 0 else 0.0
    return RunResult(
        condition=condition,
        run_idx=-1,
        prompt_idx=-1,
        decode_tok_per_sec=tok_per_sec,
        n_tokens=n,
        accept_attempts=snap_attempts,
        accept_count=snap_accepts,
        tokens_saved=tokens_saved,
        elapsed_seconds=elapsed,
    )


def _summarize(
    condition: str,
    results: list[RunResult],
    baseline_tok_per_sec: float | None,
) -> ConditionSummary:
    """Pool tok/s + accept ratio across all runs for a condition."""
    if not results:
        return ConditionSummary(
            condition=condition,
            n_runs=0,
            pooled_tok_per_sec=0.0,
            p50_tok_per_sec=0.0,
            p90_tok_per_sec=0.0,
            accept_ratio=0.0,
            mean_tokens_per_attempt=0.0,
            speedup_vs_baseline=None,
            notes="no runs recorded",
        )
    total_tokens = sum(r.n_tokens for r in results)
    total_elapsed = sum(r.elapsed_seconds for r in results)
    pooled = total_tokens / total_elapsed if total_elapsed > 0 else 0.0
    per_run = sorted(r.decode_tok_per_sec for r in results)
    p50 = statistics.median(per_run)
    p90 = per_run[int(0.9 * (len(per_run) - 1))] if per_run else 0.0
    attempts = sum(r.accept_attempts for r in results)
    accepts = sum(r.accept_count for r in results)
    tokens_saved = sum(r.tokens_saved for r in results)
    accept_ratio = accepts / attempts if attempts > 0 else 0.0
    mean_tps = tokens_saved / attempts if attempts > 0 else 0.0
    speedup = (
        pooled / baseline_tok_per_sec
        if baseline_tok_per_sec and baseline_tok_per_sec > 0
        else None
    )
    return ConditionSummary(
        condition=condition,
        n_runs=len(results),
        pooled_tok_per_sec=round(pooled, 2),
        p50_tok_per_sec=round(p50, 2),
        p90_tok_per_sec=round(p90, 2),
        accept_ratio=round(accept_ratio, 4),
        mean_tokens_per_attempt=round(mean_tps, 4),
        speedup_vs_baseline=round(speedup, 3) if speedup else None,
        notes="",
    )


def main() -> int:
    args = _parse_args()

    if args.temp != 0.0 and not args.dry_run:
        print(
            "error: DFlash bench currently only supports temp=0.0 "
            "(greedy / lossless). See verifier docstring for the "
            "speculative-sampling extension plan.",
            file=sys.stderr,
        )
        return 2

    if args.dry_run:
        plan = _planned_matrix(args)
        if args.format == "markdown":
            print("# DFlash bench plan (dry-run)\n")
            for k, v in plan.items():
                if k == "prompts":
                    print(f"\n## Prompts ({len(v)})\n")
                    for i, p in enumerate(v, 1):
                        print(f"{i}. {p[:80]}{'…' if len(p) > 80 else ''}")
                else:
                    print(f"- **{k}**: {v}")
        else:
            print(json.dumps(plan, indent=2))
        return 0

    n_prompts = min(args.prompts, len(_BENCH_PROMPTS))
    prompts = list(_BENCH_PROMPTS[:n_prompts])

    print(
        f"[bench_spec_decode_dflash] model={args.model} "
        f"drafter={args.dflash_drafter_path or '<registry>'} "
        f"block_size={args.block_size} runs={args.runs} "
        f"prompts={n_prompts} max_tokens={args.max_tokens} temp={args.temp}",
        file=sys.stderr,
    )

    all_results: dict[str, list[RunResult]] = {"none": [], "dflash": []}
    # Interleave conditions per run to avoid thermal drift.
    for run_idx in range(args.runs):
        for prompt_idx, prompt in enumerate(prompts):
            for condition in ("none", "dflash"):
                try:
                    res = _run_once(
                        model_alias=args.model,
                        drafter_override=args.dflash_drafter_path,
                        condition=condition,
                        prompt=prompt,
                        block_size=args.block_size,
                        max_tokens=args.max_tokens,
                        temp=args.temp,
                    )
                except Exception as exc:  # pragma: no cover — bench
                    print(
                        f"[bench_spec_decode_dflash] {condition} "
                        f"run={run_idx} prompt={prompt_idx} FAILED: {exc}",
                        file=sys.stderr,
                    )
                    continue
                res = RunResult(
                    condition=condition,
                    run_idx=run_idx,
                    prompt_idx=prompt_idx,
                    decode_tok_per_sec=res.decode_tok_per_sec,
                    n_tokens=res.n_tokens,
                    accept_attempts=res.accept_attempts,
                    accept_count=res.accept_count,
                    tokens_saved=res.tokens_saved,
                    elapsed_seconds=res.elapsed_seconds,
                )
                all_results[condition].append(res)
                print(
                    f"[bench_spec_decode_dflash] {condition} "
                    f"run={run_idx} prompt={prompt_idx} "
                    f"{res.decode_tok_per_sec:.1f} tok/s "
                    f"({res.n_tokens} tokens in {res.elapsed_seconds:.1f}s)",
                    file=sys.stderr,
                )

    baseline_summary = _summarize("none", all_results["none"], None)
    dflash_summary = _summarize(
        "dflash", all_results["dflash"], baseline_summary.pooled_tok_per_sec
    )

    out = {
        "model": args.model,
        "drafter_override": args.dflash_drafter_path or None,
        "block_size": args.block_size,
        "max_tokens": args.max_tokens,
        "temp": args.temp,
        "summaries": [asdict(baseline_summary), asdict(dflash_summary)],
        "raw_runs": [asdict(r) for c in all_results.values() for r in c],
    }
    if args.format == "markdown":
        print("# DFlash spec-decode bench\n")
        print(
            f"Model: `{args.model}`  block_size: {args.block_size}  "
            f"max_tokens: {args.max_tokens}  temp: {args.temp}\n"
        )
        print(
            "| Condition | Tok/s pooled | Speedup | Accept (A/V) | Mean toks/attempt |"
        )
        print("|---|---|---|---|---|")
        for s in (baseline_summary, dflash_summary):
            speedup = f"{s.speedup_vs_baseline:.2f}×" if s.speedup_vs_baseline else "—"
            accept = f"{s.accept_ratio:.1%}" if s.accept_ratio else "—"
            mean_tps = (
                f"{s.mean_tokens_per_attempt:.2f}" if s.mean_tokens_per_attempt else "—"
            )
            print(
                f"| {s.condition} | {s.pooled_tok_per_sec:.1f} | {speedup} | "
                f"{accept} | {mean_tps} |"
            )
    else:
        print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover — script entry
    sys.exit(main())
