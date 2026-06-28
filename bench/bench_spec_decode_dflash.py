#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""DFlash speculative-decode decode-tok/s bench (R15-P1 #313, #343 fix).

Compares ``--spec-decode dflash`` against ``--spec-decode none`` on a
Qwen3.5 / Qwen3.6 checkpoint with the matching block-diffusion drafter
bound in :mod:`vllm_mlx.spec_decode.dflash.drafter_registry` (or via
``--dflash-drafter-path``). Mirrors :file:`bench_spec_decode_mtp.py`'s
8-prompt × 3-run protocol so the two backends are directly comparable
on the same workload.

Architecture note (#343 fix)
----------------------------

The bench drives the WORKING DFlash path: both target and drafter load
through ``mlx_vlm`` (0.6.3) and generation flows through
``mlx_vlm.stream_generate`` with the drafter bound. Previously this
script tried to call rapid-mlx's own
:func:`vllm_mlx.spec_decode.dflash.generator.dflash_generate_step`
loop, which in turn called a ``draft_block(prefix_tokens,
current_position)`` adapter signature that mlx-vlm 0.6.3's
``DFlashDraftModel`` does NOT implement (the actual signature is
``draft_block(last_bonus, hidden, cache, block_size, sampler,
token_dtype)``). As a result the bench could not run end-to-end — the
2.34× math / 2.22× adaptive numbers in the 0.9 release dashboard came
from a different harness, not this script. With #343 this script now
runs cleanly and re-benches are trustworthy.

Usage::

    python bench/bench_spec_decode_dflash.py \\
        --model qwen3.5-9b-4bit \\
        --runs 3 \\
        --max-tokens 256

Outputs JSON (default) or a markdown table::

    python bench/bench_spec_decode_dflash.py --format markdown

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

    Per-run accept stats reflect DFlash's block-bonus contract: each
    attempt drafts ``draft_block_size - 1`` candidate positions and
    accepts some prefix [0..draft_block_size-1] of them, plus the
    always-emitted verify bonus. We track both ``accept_count``
    (positions accepted by the verifier) and ``drafted_tokens``
    (positions drafted total) so the summary can report a true
    per-position ``accept_ratio = accept_count / drafted_tokens`` and
    a per-attempt mean ``accept_count / attempts``.
    """

    condition: str
    run_idx: int
    prompt_idx: int
    decode_tok_per_sec: float
    n_tokens: int
    accept_attempts: int
    accept_count: int
    drafted_tokens: int
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
        default="mlx-community/Qwen3.5-9B-4bit",
        help=(
            "Target model HF path (default: mlx-community/Qwen3.5-9B-4bit). "
            "Both the baseline and DFlash conditions load this via mlx-vlm. "
            "The matching DFlash drafter must be bound in the side-registry "
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
        default=0,
        help=(
            "DFlash block size override (default: 0 = use drafter's "
            "trained value, typically 16). When set, mlx-vlm treats this "
            "as the ceiling; the adaptive scaler still backs off on poor "
            "acceptance."
        ),
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
            "Sampling temperature (default: 0.0 = greedy). DFlash temp>0 "
            "is supported by mlx-vlm 0.6.3 but the lossless contract is "
            "only validated at temp=0."
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
            "= full PR #990 mix). Pass a small N for a fast smoke run."
        ),
    )
    parser.add_argument(
        "--prompt-indices",
        default="",
        help=(
            "Comma-separated list of prompt indices into the canonical "
            "8-prompt set (e.g. '0,4,6' for code/dialogue/math). When "
            "set, overrides --prompts and runs only the selected prompts."
        ),
    )
    return parser.parse_args()


def _resolve_prompt_indices(args: argparse.Namespace) -> list[int]:
    if args.prompt_indices.strip():
        # Reject malformed lists like "0,,1" or "0, ,1" — an empty
        # segment is a typo, not a "skip this index" idiom. Silently
        # dropping it would hide the operator's slip.
        raw_segments = args.prompt_indices.split(",")
        empties = [seg for seg in raw_segments if not seg.strip()]
        if empties:
            raise ValueError(
                f"--prompt-indices contains empty segment(s) in "
                f"{args.prompt_indices!r}; remove the stray comma(s)."
            )
        try:
            ix = [int(x) for x in raw_segments]
        except ValueError as exc:
            # Surface a clean argparse-style error rather than a raw
            # traceback for "--prompt-indices 0,foo,1" — the operator
            # likely typo'd.
            raise ValueError(
                f"--prompt-indices entries must be integers; "
                f"got {args.prompt_indices!r} ({exc})"
            ) from None
        if not ix:
            # ``--prompt-indices ","`` or ``--prompt-indices " "`` parses
            # to empty after strip-filtering. Without this guard the
            # bench would run zero generations and emit an empty
            # summary, which looks identical to a success and silently
            # invalidates any pooled-speedup interpretation.
            raise ValueError(
                f"--prompt-indices is empty after parsing {args.prompt_indices!r}; "
                "pass at least one valid index."
            )
        # Reject duplicates — pooling the same prompt twice silently
        # weights it 2x in the summary and reports misleading
        # speedup. The operator likely meant to type two different
        # indices; surface the typo loudly instead.
        seen: set[int] = set()
        dupes: list[int] = []
        for i in ix:
            if i in seen:
                dupes.append(i)
            else:
                seen.add(i)
        if dupes:
            raise ValueError(
                f"--prompt-indices contains duplicate entries "
                f"{sorted(set(dupes))}; each prompt should appear at most "
                "once or the pooled speedup over-weights the repeated prompt."
            )
        for i in ix:
            if i < 0 or i >= len(_BENCH_PROMPTS):
                raise ValueError(
                    f"--prompt-indices entry {i} out of range "
                    f"[0, {len(_BENCH_PROMPTS) - 1}]"
                )
        return ix
    n = min(args.prompts, len(_BENCH_PROMPTS))
    return list(range(n))


def _planned_matrix(
    args: argparse.Namespace, indices: list[int] | None = None
) -> dict[str, Any]:
    """Return the bench plan as a JSON-serializable dict (dry-run mode).

    ``indices`` is optional so callers that already validated the
    prompt-indices CLI can pass them through (avoids double validation
    and ensures the dry-run plan reflects the same resolved set the
    real run would).
    """
    if indices is None:
        indices = _resolve_prompt_indices(args)
    prompts = [_BENCH_PROMPTS[i] for i in indices]
    return {
        "model": args.model,
        "drafter_override": args.dflash_drafter_path or None,
        "block_size_override": args.block_size or None,
        "runs_per_condition": args.runs,
        "max_tokens": args.max_tokens,
        "temp": args.temp,
        "conditions": ["none", "dflash"],
        "prompt_indices": indices,
        "prompts": prompts,
        "total_generations": 2 * args.runs * len(prompts),
        "estimated_wall_time_seconds_at_30_tok_per_sec": (
            2 * args.runs * len(prompts) * args.max_tokens / 30.0
        ),
        "expected_speedup_paper": 4.37,
        "expected_baseline_tok_per_sec_m5_max": 30.95,
        "expected_dflash_tok_per_sec_m5_max": 135.34,
    }


def _load_baseline_target(model_alias: str) -> tuple[Any, Any]:
    """Load the target + processor via mlx-vlm (for the baseline run).

    Imported here so ``--dry-run`` doesn't pay the mlx-vlm import +
    weight-load cost.
    """
    from mlx_vlm import load as _mlx_vlm_load

    return _mlx_vlm_load(model_alias)


def _resolve_drafter_path(
    args: argparse.Namespace,
    model_alias: str,
) -> str:
    """Resolve the drafter HF path from the explicit flag or the registry."""
    if args.dflash_drafter_path:
        return args.dflash_drafter_path
    from vllm_mlx.spec_decode.dflash import get_dflash_drafter_path

    return get_dflash_drafter_path(model_alias) or ""


def _run_baseline_once(
    *,
    target: Any,
    processor: Any,
    prompt: str,
    max_tokens: int,
    temp: float,
) -> RunResult:
    """Run one baseline (no spec-decode) generation via mlx-vlm."""
    from mlx_vlm import stream_generate

    t0 = time.perf_counter()
    n = 0
    for chunk in stream_generate(
        target,
        processor,
        prompt,
        max_tokens=max_tokens,
        temperature=temp,
    ):
        n = chunk.generation_tokens
        if n >= max_tokens:
            break
    elapsed = time.perf_counter() - t0
    tok_per_sec = n / elapsed if elapsed > 0 else 0.0
    return RunResult(
        condition="none",
        run_idx=-1,
        prompt_idx=-1,
        decode_tok_per_sec=tok_per_sec,
        n_tokens=n,
        accept_attempts=0,
        accept_count=0,
        drafted_tokens=0,
        elapsed_seconds=elapsed,
    )


def _run_dflash_once(
    *,
    driver: Any,
    prompt: str,
    max_tokens: int,
    temp: float,
) -> RunResult:
    """Run one DFlash generation via the rapid-mlx driver wrapper."""
    t0 = time.perf_counter()
    n = 0
    for chunk in driver.generate(
        prompt,
        max_tokens=max_tokens,
        temperature=temp,
    ):
        n = chunk.generation_tokens
        if n >= max_tokens:
            break
    elapsed = time.perf_counter() - t0
    stats = driver.accept_stats()
    tok_per_sec = n / elapsed if elapsed > 0 else 0.0
    return RunResult(
        condition="dflash",
        run_idx=-1,
        prompt_idx=-1,
        decode_tok_per_sec=tok_per_sec,
        n_tokens=n,
        accept_attempts=int(stats["attempts"]),
        accept_count=int(stats["accepted_tokens"]),
        drafted_tokens=int(stats["drafted_tokens"]),
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
    drafted = sum(r.drafted_tokens for r in results)
    # accept_ratio = fraction of DRAFTED positions accepted by the
    # verifier (paper's standard metric). mean_tps = accepted positions
    # per attempt, which can be > 1 (per the block-bonus contract).
    accept_ratio = accepts / drafted if drafted > 0 else 0.0
    mean_tps = accepts / attempts if attempts > 0 else 0.0
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

    # Validate CLI inputs up front so both --dry-run and the real run
    # share the same operator-facing error path. Any ValueError raised
    # by _resolve_prompt_indices (bad int, out-of-range, duplicate,
    # empty) is caught here and surfaced as "error: ..." + exit 2,
    # the same convention argparse uses.
    if args.block_size < 0:
        print(
            f"error: --block-size must be >= 0; got {args.block_size}. "
            "Use 0 (the default) to defer to the drafter's trained value.",
            file=sys.stderr,
        )
        return 2
    if args.prompts < 1 and not args.prompt_indices.strip():
        # ``--prompts 0`` would resolve to an empty index list and the
        # bench would emit a zero-run summary that looks identical to
        # success. Reject so the operator notices the typo. (Skipped
        # when --prompt-indices is set — that path supplies its own
        # validation via _resolve_prompt_indices.)
        print(
            f"error: --prompts must be >= 1; got {args.prompts}.",
            file=sys.stderr,
        )
        return 2
    if args.runs < 1:
        print(
            f"error: --runs must be >= 1; got {args.runs}.",
            file=sys.stderr,
        )
        return 2
    if args.max_tokens < 1:
        print(
            f"error: --max-tokens must be >= 1; got {args.max_tokens}.",
            file=sys.stderr,
        )
        return 2
    try:
        indices = _resolve_prompt_indices(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        plan = _planned_matrix(args, indices=indices)
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

    prompts = [_BENCH_PROMPTS[i] for i in indices]

    drafter_path = _resolve_drafter_path(args, args.model)
    if not drafter_path:
        print(
            f"error: no DFlash drafter bound for model={args.model!r}. "
            f"Pass --dflash-drafter-path or register via "
            f"vllm_mlx.spec_decode.dflash.register_dflash_drafter().",
            file=sys.stderr,
        )
        return 2

    print(
        f"[bench_spec_decode_dflash] model={args.model} "
        f"drafter={drafter_path} "
        f"block_size={args.block_size or '<drafter-default>'} "
        f"runs={args.runs} prompts={len(prompts)} "
        f"prompt_indices={indices} "
        f"max_tokens={args.max_tokens} temp={args.temp}",
        file=sys.stderr,
    )

    # Load target + drafter ONCE, reuse across runs. mlx-vlm 0.6.3's
    # ``stream_generate`` is safe to call repeatedly against the same
    # model + drafter pair.
    from vllm_mlx.spec_decode.dflash.drafter import MlxVlmDFlashDriver

    t_load = time.perf_counter()
    print("[bench_spec_decode_dflash] loading target via mlx-vlm…", file=sys.stderr)
    target, processor = _load_baseline_target(args.model)
    print(
        f"[bench_spec_decode_dflash] target loaded in {time.perf_counter() - t_load:.1f}s; "
        "loading drafter…",
        file=sys.stderr,
    )
    block_size_arg: int | None = args.block_size or None
    driver = MlxVlmDFlashDriver(
        target_repo=args.model,
        drafter_repo=drafter_path,
        block_size=block_size_arg,
    )
    # Share the already-loaded target rather than letting the driver
    # re-load it: ``MlxVlmDFlashDriver.load()`` re-calls
    # ``mlx_vlm.load`` and we'd pay double the weight-load latency
    # (the 27B-8bit target is 28+ GB). Adopt the already-loaded
    # objects via the public injection API; the bench then exercises
    # the same generate() code path production does.
    from vllm_mlx.speculative.dflash import load_runtime

    driver.adopt(
        target=target,
        processor=processor,
        runtime=load_runtime(drafter_path),
    )
    print(
        f"[bench_spec_decode_dflash] all loaded in {time.perf_counter() - t_load:.1f}s",
        file=sys.stderr,
    )

    all_results: dict[str, list[RunResult]] = {"none": [], "dflash": []}
    # Interleave conditions per run to avoid thermal drift.
    for run_idx in range(args.runs):
        for prompt_local_idx, prompt in enumerate(prompts):
            prompt_idx = indices[prompt_local_idx]
            for condition in ("none", "dflash"):
                try:
                    if condition == "none":
                        res = _run_baseline_once(
                            target=target,
                            processor=processor,
                            prompt=prompt,
                            max_tokens=args.max_tokens,
                            temp=args.temp,
                        )
                    else:
                        res = _run_dflash_once(
                            driver=driver,
                            prompt=prompt,
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
                    drafted_tokens=res.drafted_tokens,
                    elapsed_seconds=res.elapsed_seconds,
                )
                all_results[condition].append(res)
                if condition == "dflash":
                    rate = (
                        100.0 * res.accept_count / res.drafted_tokens
                        if res.drafted_tokens > 0
                        else 0.0
                    )
                    mean_per_attempt = (
                        res.accept_count / res.accept_attempts
                        if res.accept_attempts > 0
                        else 0.0
                    )
                    accept_suffix = (
                        f" accept={res.accept_count}/{res.drafted_tokens} "
                        f"({rate:.1f}% per slot, {mean_per_attempt:.2f}/attempt, "
                        f"{res.accept_attempts} attempts)"
                    )
                else:
                    accept_suffix = ""
                print(
                    f"[bench_spec_decode_dflash] {condition} "
                    f"run={run_idx} prompt={prompt_idx} "
                    f"{res.decode_tok_per_sec:.1f} tok/s "
                    f"({res.n_tokens} tokens in {res.elapsed_seconds:.1f}s)"
                    + accept_suffix,
                    file=sys.stderr,
                )

    baseline_summary = _summarize("none", all_results["none"], None)
    dflash_summary = _summarize(
        "dflash", all_results["dflash"], baseline_summary.pooled_tok_per_sec
    )

    out = {
        "model": args.model,
        "drafter": drafter_path,
        "block_size_override": args.block_size or None,
        "max_tokens": args.max_tokens,
        "temp": args.temp,
        "prompt_indices": indices,
        "summaries": [asdict(baseline_summary), asdict(dflash_summary)],
        "raw_runs": [asdict(r) for c in all_results.values() for r in c],
    }
    if args.format == "markdown":
        print("# DFlash spec-decode bench\n")
        print(
            f"Model: `{args.model}`  drafter: `{drafter_path}`  "
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
