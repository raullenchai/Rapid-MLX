#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""MTP speculative-decode decode-tok/s bench (R15-P1 #302).

Compares ``--spec-decode mtp`` against ``--spec-decode none`` on a
Qwen3.5 / Qwen3.6 checkpoint that has been converted with the
upstream PR #990 ``sanitize()`` path (preserving ``mtp.*`` weights).
Mirrors the upstream bench script
(`gist <https://gist.github.com/AirRunner/e3aafd4de78c2cba4f4e233261cd64f2>`_)
referenced in PR #990's results table — 8 diverse prompts, 3 runs per
condition, conditions interleaved to avoid thermal drift.

**Deferred execution** — at vendoring time the GPU is contended with
the Stage B PonyExl3 Viterbi conversion (PID 56486). Running this
bench in parallel would thrash, so the script is written but NOT
executed in the same agent run that opens the PR. Operators (or the
follow-up agent on a quiet box) run::

    python bench/bench_spec_decode_mtp.py \\
        --model /path/to/qwen3.5-9b-mtp-4bit \\
        --runs 3 \\
        --max-tokens 256

Outputs JSON (default) or a markdown table for the PR follow-up
comment::

    python bench/bench_spec_decode_mtp.py --format markdown

Expected numbers (from PR #990 update comment, Qwen3.5-27B-4bit on
M4 Pro): baseline 15.3 tok/s, MTP temp=0 24.0 tok/s (1.57×, 85.2%
accept). The bench script reports the same triplet.

**Dry-run mode** — ``--dry-run`` skips the actual model load and
generation, runs through argument parsing + condition setup only,
and prints the planned bench matrix. Useful for CI validation that
the script wires up cleanly without burning GPU cycles.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any

# The 8 diverse prompts from PR #990's bench script. Kept verbatim so
# the numbers we report are directly comparable to the upstream table
# (and to any third-party that re-runs the same prompts). Mix covers
# coding, prose, dialogue, structured output, summarization, and
# multi-turn — the same coverage gradient PR #990 uses.
_BENCH_PROMPTS: tuple[str, ...] = (
    # 1. Code generation
    "Write a Python function that computes the n-th Fibonacci number "
    "using memoization. Include type hints, a docstring, and a small "
    "example call. Keep it under 30 lines.",
    # 2. Code explanation
    "Explain how a Bloom filter works, including its false-positive "
    "guarantees, when you'd choose it over a hash set, and a brief "
    "complexity analysis.",
    # 3. Structured JSON output
    "Return a JSON object describing the top 3 NoSQL databases by "
    "popularity, with fields name, category, primary_use_case, and "
    "year_released. No prose, just the JSON.",
    # 4. Prose
    "Write a 200-word reflection on the role of perseverance in "
    "scientific discovery, citing a specific historical example.",
    # 5. Dialogue
    "Write a short dialogue (10 lines) between a junior engineer and a "
    "senior engineer reviewing a pull request that introduces a race "
    "condition.",
    # 6. Summarization
    "Summarize the plot of Moby-Dick in 3 paragraphs. Focus on Ahab's "
    "obsession and how it drives the crew's fate.",
    # 7. Reasoning
    "A train leaves station A at 9:00 traveling at 60 km/h. Another "
    "train leaves station B at 9:30 traveling at 80 km/h on the same "
    "track in the opposite direction. The stations are 350 km apart. "
    "At what time do they meet? Show work.",
    # 8. Translation / adaptation
    "Translate the following sentence into formal French, then into "
    "Brazilian Portuguese, then into Japanese. Sentence: 'The early "
    "bird catches the worm.' Output: three labeled lines.",
)


@dataclass(frozen=True)
class RunResult:
    """One ``(condition, run_idx, prompt_idx)`` measurement."""

    condition: str
    run_idx: int
    prompt_idx: int
    decode_tok_per_sec: float
    n_tokens: int
    accept_attempts: int
    accept_count: int
    elapsed_seconds: float


@dataclass(frozen=True)
class ConditionSummary:
    """Pooled tok/s + accept ratio for one condition.

    Pooled means ``sum(tokens) / sum(seconds)`` rather than
    ``mean(per-prompt tok/s)`` — matches PR #990's reporting
    convention so the numbers are directly comparable.
    """

    condition: str
    n_runs: int
    pooled_tok_per_sec: float
    p50_tok_per_sec: float
    p90_tok_per_sec: float
    accept_ratio: float
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
            "Model alias or HF path (default: qwen3.5-9b-4bit). MUST be "
            "a checkpoint converted with mlx-lm PR #990's sanitize() "
            "path that preserves mtp.* weights — otherwise --spec-decode "
            "mtp will refuse at boot."
        ),
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Runs per condition (default: 3, matches PR #990 reporting).",
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
            "contract enforced). PR #990 reports speedup tables at "
            "temp=0 / 0.6 / 1.0."
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
            "matrix and exit. Useful for CI smoke / argparse "
            "validation without GPU consumption."
        ),
    )
    parser.add_argument(
        "--prompts",
        type=int,
        default=len(_BENCH_PROMPTS),
        help=(
            f"Number of prompts to run (default: {len(_BENCH_PROMPTS)} "
            "= full PR #990 mix). Lower values cut wall time at the "
            "cost of reduced workload coverage."
        ),
    )
    parser.add_argument(
        "--mtp-sidecar",
        default=None,
        help=(
            "MTP head sidecar (HF repo id or local path). The mlx-lm "
            "0.31.3 ``qwen3_5.py::sanitize`` unconditionally strips "
            "``mtp.*`` weights during the main load, so MTP weights "
            "must be loaded from a separate sidecar safetensors blob "
            "after ``mlx_lm.load(...)`` completes. Default: auto-pick "
            "based on the base model alias (Qwen3.5-9B-4bit → "
            "mlx-community/Qwen3.5-9B-MTP-4bit). Pass an explicit "
            "value to override."
        ),
    )
    parser.add_argument(
        "--mtp-only",
        action="store_true",
        help=(
            "Skip the baseline (--spec-decode none) condition and run "
            "only the MTP condition. Useful when comparing against a "
            "previously-captured baseline number."
        ),
    )
    return parser.parse_args()


# Map from common base aliases / HF paths to their matching MTP sidecar.
# The sidecar repo holds the MTP head only (no embed_tokens, no
# backbone layers) — see ``mlx-community/Qwen3.5-9B-MTP-4bit`` README.
# Keys are normalized to lowercase; ``_resolve_mtp_sidecar`` lowers
# the incoming alias before lookup so case variants
# (``Qwen3.5-9B-4bit`` vs ``qwen3.5-9b-4bit`` vs full
# ``mlx-community/Qwen3.5-9B-4bit``) all hit the same default.
_DEFAULT_MTP_SIDECAR: dict[str, str] = {
    "qwen3.5-9b-4bit": "mlx-community/Qwen3.5-9B-MTP-4bit",
    "mlx-community/qwen3.5-9b-4bit": "mlx-community/Qwen3.5-9B-MTP-4bit",
    "mlx-community/qwen3.5-9b-mlx-4bit": "mlx-community/Qwen3.5-9B-MTP-4bit",
}


def _resolve_mtp_sidecar(model_alias: str, explicit: str | None) -> str | None:
    """Pick the MTP sidecar for ``model_alias``.

    Explicit ``--mtp-sidecar`` always wins. Otherwise look up the
    alias in ``_DEFAULT_MTP_SIDECAR`` after lowercasing — codex
    flagged on PR #954 that without normalization,
    ``Qwen3.5-9B-4bit`` (case variant of the dict key) missed the
    documented default. Return ``None`` if no default is known —
    the inject will then fall back to a no-op load and log a warning.
    """
    if explicit is not None:
        return explicit
    return _DEFAULT_MTP_SIDECAR.get(model_alias.lower())


def _planned_matrix(args: argparse.Namespace) -> dict[str, Any]:
    """Return the bench plan as a JSON-serializable dict (dry-run mode)."""
    n_prompts = min(args.prompts, len(_BENCH_PROMPTS))
    return {
        "model": args.model,
        "runs_per_condition": args.runs,
        "max_tokens": args.max_tokens,
        "temp": args.temp,
        "conditions": ["none", "mtp"],
        "prompts": list(_BENCH_PROMPTS[:n_prompts]),
        "total_generations": 2 * args.runs * n_prompts,
        "estimated_wall_time_seconds_at_15_tok_per_sec": (
            2 * args.runs * n_prompts * args.max_tokens / 15.0
        ),
    }


def _run_once(
    *,
    model_alias: str,
    condition: str,
    prompt: str,
    max_tokens: int,
    temp: float,
    mtp_sidecar: str | None = None,
) -> RunResult:
    """Run one generation under the requested condition.

    Imports ``mlx_lm`` lazily so ``--dry-run`` doesn't pay the
    import cost. Imports the rapid-mlx MTP injection + generator
    only when ``condition == "mtp"``.

    Returns a :class:`RunResult` carrying the decode tok/s and accept
    counters.
    """
    import mlx.core as mx
    from mlx_lm import load

    from vllm_mlx.spec_decode.mtp import (
        MTPAcceptCounter,
        get_global_counter,
    )

    model, tokenizer = load(model_alias)

    if condition == "mtp":
        from vllm_mlx.spec_decode.mtp.dispatch import (
            dispatch_mtp_inject,
            dispatch_mtp_validate,
        )
        from vllm_mlx.spec_decode.mtp.generator import mtp_generate_step

        # Route through the family dispatcher so Gemma 4 aliases
        # (``gemma4`` / ``gemma4_unified``) can reach the bench too
        # once the follow-up AssistantModel PR lands. Model_type is
        # read off the outer wrapper's args — mlx-lm sets this on
        # every loaded model.
        model_type = (
            getattr(getattr(model, "args", None), "model_type", None)
            or getattr(model, "model_type", None)
            or "qwen3_5"  # bench historical default
        )
        if not dispatch_mtp_inject(
            model, model_type=model_type, mtp_sidecar=mtp_sidecar
        ):
            raise RuntimeError(
                f"MTP injection failed on model {model_alias!r} — "
                f"sidecar {mtp_sidecar!r}. Confirm the sidecar repo or "
                "local path exists, holds model.safetensors with the "
                "expected MTP head schema, and the base model's "
                "config carries mtp_num_hidden_layers >= 1."
            )
        # Validate through the SAME family dispatcher — codex round-2
        # flagged that mixing families here (inject via gemma4, validate
        # via qwen3_5) would falsely fail Gemma 4 patches.
        assert dispatch_mtp_validate(model, model_type=model_type)
        # The patch lands on the inner TextModel (the VLM wrapper's
        # ``language_model`` field). The generator drives the model
        # directly, so re-bind ``model`` to the patched inner for the
        # ``mtp_generate_step`` call. The mlx-lm 0.31.3 Qwen3.5
        # arch ships only the VLM wrapper; the inner TextModel
        # carries embed_tokens, lm_head, layers — everything the
        # generator and ``mtp_forward`` reference.
        if hasattr(model, "language_model"):
            model = model.language_model

    prompt_ids = mx.array(tokenizer.encode(prompt), mx.uint32)
    counter = MTPAcceptCounter()
    # Replace global counter for the duration of THIS run so the
    # ``accept_attempts`` / ``accept_count`` we report only reflects
    # the prompt under measurement (not whatever happened before).
    prior_attempts = get_global_counter().snapshot().attempts
    prior_accepts = get_global_counter().snapshot().accepts

    t0 = time.perf_counter()
    n = 0
    if condition == "mtp":
        gen = mtp_generate_step(
            prompt_ids,
            model,
            max_tokens=max_tokens,
            temp=temp,
            accept_counter=counter,
        )
        for _ in gen:
            n += 1
    else:
        from mlx_lm.generate import stream_generate

        for resp in stream_generate(
            model,
            tokenizer,
            prompt,
            max_tokens=max_tokens,
        ):
            n += 1
            if n >= max_tokens:
                break

    elapsed = time.perf_counter() - t0

    snap = counter.snapshot()
    if condition != "mtp":
        # ``none`` path doesn't touch the counter; report 0/0.
        snap_attempts, snap_accepts = 0, 0
    else:
        snap_attempts, snap_accepts = snap.attempts, snap.accepts
    # Sanity-check: global counter shouldn't have moved (per-run
    # counter is what mtp_generate_step bumps via the
    # ``accept_counter=`` kwarg).
    assert get_global_counter().snapshot().attempts == prior_attempts
    assert get_global_counter().snapshot().accepts == prior_accepts

    tok_per_sec = n / elapsed if elapsed > 0 else 0.0
    return RunResult(
        condition=condition,
        run_idx=-1,  # patched up by the caller
        prompt_idx=-1,  # patched up by the caller
        decode_tok_per_sec=tok_per_sec,
        n_tokens=n,
        accept_attempts=snap_attempts,
        accept_count=snap_accepts,
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
    accept_ratio = accepts / attempts if attempts > 0 else 0.0
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
        speedup_vs_baseline=round(speedup, 3) if speedup else None,
        notes="",
    )


def main() -> int:
    args = _parse_args()

    if args.dry_run:
        plan = _planned_matrix(args)
        if args.format == "markdown":
            print("# MTP bench plan (dry-run)\n")
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

    mtp_sidecar = _resolve_mtp_sidecar(args.model, args.mtp_sidecar)
    conditions: tuple[str, ...] = ("mtp",) if args.mtp_only else ("none", "mtp")

    print(
        f"[bench_spec_decode_mtp] model={args.model} runs={args.runs} "
        f"prompts={n_prompts} max_tokens={args.max_tokens} temp={args.temp} "
        f"mtp_sidecar={mtp_sidecar!r} conditions={conditions}",
        file=sys.stderr,
    )

    all_results: dict[str, list[RunResult]] = {"none": [], "mtp": []}
    # Interleave conditions per run to avoid thermal drift bias (PR
    # #990 follows the same protocol).
    for run_idx in range(args.runs):
        for prompt_idx, prompt in enumerate(prompts):
            for condition in conditions:
                try:
                    res = _run_once(
                        model_alias=args.model,
                        condition=condition,
                        prompt=prompt,
                        max_tokens=args.max_tokens,
                        temp=args.temp,
                        mtp_sidecar=mtp_sidecar,
                    )
                except Exception as exc:  # pragma: no cover — bench
                    print(
                        f"[bench_spec_decode_mtp] {condition} run={run_idx} "
                        f"prompt={prompt_idx} FAILED: {exc}",
                        file=sys.stderr,
                    )
                    continue
                # Patch indices for the JSON output.
                res = RunResult(
                    condition=condition,
                    run_idx=run_idx,
                    prompt_idx=prompt_idx,
                    decode_tok_per_sec=res.decode_tok_per_sec,
                    n_tokens=res.n_tokens,
                    accept_attempts=res.accept_attempts,
                    accept_count=res.accept_count,
                    elapsed_seconds=res.elapsed_seconds,
                )
                all_results[condition].append(res)
                print(
                    f"[bench_spec_decode_mtp] {condition} run={run_idx} "
                    f"prompt={prompt_idx} {res.decode_tok_per_sec:.1f} tok/s "
                    f"({res.n_tokens} tokens in {res.elapsed_seconds:.1f}s)",
                    file=sys.stderr,
                )

    baseline_summary = _summarize("none", all_results["none"], None)
    mtp_summary = _summarize(
        "mtp", all_results["mtp"], baseline_summary.pooled_tok_per_sec
    )

    out = {
        "model": args.model,
        "max_tokens": args.max_tokens,
        "temp": args.temp,
        "summaries": [asdict(baseline_summary), asdict(mtp_summary)],
        "raw_runs": [asdict(r) for c in all_results.values() for r in c],
    }
    if args.format == "markdown":
        print("# MTP spec-decode bench\n")
        print(
            f"Model: `{args.model}`  max_tokens: {args.max_tokens}  temp: {args.temp}\n"
        )
        print("| Condition | Tok/s pooled | Speedup | Accept (A/V) |")
        print("|---|---|---|---|")
        for s in (baseline_summary, mtp_summary):
            speedup = f"{s.speedup_vs_baseline:.2f}×" if s.speedup_vs_baseline else "—"
            accept = f"{s.accept_ratio:.1%}" if s.accept_ratio else "—"
            print(
                f"| {s.condition} | {s.pooled_tok_per_sec:.1f} | {speedup} | {accept} |"
            )
    else:
        print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover — script entry
    sys.exit(main())
