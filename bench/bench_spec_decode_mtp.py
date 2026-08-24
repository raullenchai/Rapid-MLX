#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""MTP speculative-decode decode-tok/s bench (R15-P1 #302).

Runs three arms over a Qwen3.5 / Qwen3.6 checkpoint converted with the
upstream PR #990 ``sanitize()`` path (preserving ``mtp.*`` weights):

``none``
    plain ``mlx_lm.generate_step`` — the upstream PR #990 baseline.
``ar``
    this generator with ``max_k=0`` and auto-K disabled, i.e. the exact
    MTP code path with speculation parked. Differs from ``mtp`` only in
    whether speculation runs, so the ratio between them attributes the
    delta to speculation and to nothing else.
``mtp``
    this generator with speculation live.

Prompt suite and shape mirror the upstream bench script
(`gist <https://gist.github.com/AirRunner/e3aafd4de78c2cba4f4e233261cd64f2>`_)
referenced in PR #990's results table — 8 diverse prompts, 3 runs per
condition.

**Estimator** — conditions are interleaved in the innermost loop, so for
every ``(run_idx, prompt_idx)`` the arms execute back to back. The
headline speedup is therefore reported *paired*: the per-cell
``mtp/ar`` ratios are collected and summarised as median + IQR. Pairing
matters because Apple-silicon thermal drift moves whole runs together —
on an M5 Max the identical greedy ``ar`` cell has read 20% apart across
back-to-back invocations, which is larger than the speculation effect
being measured. Drift that moves both arms together cancels in the
ratio; it does not cancel in a pooled mean, and it does not cancel at
all when two separate invocations of this script are compared to each
other. Never A/B sampler configurations across invocations — vary them
inside one process, or the drift will dominate the result.

The pooled ``speedup_vs_baseline`` is retained alongside for continuity
with PR #990's reporting convention.

**Running it** — the bench needs a quiet box; anything else contending
for the GPU shows up as drift, which is the one thing this measurement
cannot absorb. Greedy reference::

    python bench/bench_spec_decode_mtp.py \\
        --model mlx-community/Qwen3.5-9B-4bit \\
        --mtp-sidecar mlx-community/Qwen3.5-9B-MTP-4bit \\
        --runs 3 --max-tokens 256 --format markdown

and the same command with ``--temp 0.6 --top-p 0.95 --top-k 20`` for the
non-greedy arm. ``--runs 3`` over the 8-prompt suite yields 24 pairs per
condition, which is enough for the IQR to be meaningful; fewer than ~12
pairs and the median starts moving around.

``--mtp-sidecar`` is required for checkpoints whose MTP weights live in
a separate repo (mlx-lm's conversion strips ``mtp.*`` from the base
model). Pass a single fused checkpoint to ``--model`` alone if you have
one. ``--skip-mlx-lm-arm`` drops the ``none`` arm when only the
speculation delta is of interest, and roughly halves wall time.

Outputs JSON (default) or a markdown table for PR comments.

Expected numbers for the ``none`` arm (from PR #990's update comment,
Qwen3.5-27B-4bit on M4 Pro): baseline 15.3 tok/s, MTP temp=0 24.0 tok/s
(1.57×, 85.2% accept). That comparison predates the ``ar`` arm, so it
is the ``mtp``-vs-``none`` pair; the ``ar`` arm reports separately.

**Dry-run mode** — ``--dry-run`` skips the actual model load and
generation, runs through argument parsing + condition setup only,
and prints the planned bench matrix. Useful for CI validation that
the script wires up cleanly without burning GPU cycles.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field, replace
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


def _tokenizer_stop_tokens(tokenizer: Any) -> set[int]:
    """Return the same EOS token set mlx-lm uses for stream termination."""
    plural = getattr(tokenizer, "eos_token_ids", None)
    if plural is not None:
        return {int(token) for token in plural if token is not None}
    singular = getattr(tokenizer, "eos_token_id", None)
    return {int(singular)} if singular is not None else set()


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
    decode_elapsed_seconds: float
    prompt_eval_seconds: float
    end_to_end_tok_per_sec: float
    token_sha256: str = ""
    verify_kernel_calls: int = 0
    verify_kernel_fallbacks: int = 0
    verify_sync_seconds: float = 0.0
    draft_seconds: float = 0.0
    residual_sync_seconds: float = 0.0
    verify_calls: int = 0
    prompt_lookup_proposals: int = 0
    prompt_lookup_drafted_tokens: int = 0
    prompt_lookup_accepted_tokens: int = 0
    prompt_lookup_rejections: int = 0
    prompt_lookup_mtp_sync_seconds: float = 0.0
    k_histogram: dict[int, int] = field(default_factory=dict)


@dataclass(frozen=True)
class ConditionSummary:
    """Pooled tok/s + accept ratio for one condition.

    Pooled means ``sum(tokens) / sum(seconds)`` rather than
    ``mean(per-prompt tok/s)`` — matches PR #990's reporting
    convention so the numbers are directly comparable.

    ``speedup_vs_baseline`` is that pooled ratio. The ``paired_*``
    fields carry the drift-resistant estimator described in the module
    docstring: per-``(run_idx, prompt_idx)`` ratios against the same
    cell of the reference arm, summarised as median + IQR. Prefer the
    paired median when reading results off a thermally noisy machine.
    """

    condition: str
    n_runs: int
    pooled_tok_per_sec: float
    p50_tok_per_sec: float
    p90_tok_per_sec: float
    accept_ratio: float
    speedup_vs_baseline: float | None
    notes: str
    paired_speedup_median: float | None = None
    paired_speedup_p25: float | None = None
    paired_speedup_p75: float | None = None
    paired_speedup_min: float | None = None
    paired_pairs: int = 0
    paired_pairs_slower: int = 0


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
        "--seed",
        type=int,
        default=0,
        help="MLX sampling seed reset before every measured generation (default: 0).",
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
    parser.add_argument("--top-p", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=0)
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
        "--prompt-text",
        default=None,
        help="Run one explicit prompt instead of the built-in suite.",
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
    parser.add_argument(
        "--skip-mlx-lm-arm",
        action="store_true",
        help=(
            "Skip the mlx_lm ``stream_generate`` baseline and run only the "
            "same-path arms (--spec-decode off via K=0, and MTP). Halves "
            "wall time when the question is purely 'does speculation pay "
            "off', since that is answered by the ar-vs-mtp ratio alone."
        ),
    )
    parser.add_argument(
        "--mtp-max-k",
        type=int,
        default=3,
        help="Maximum chained MTP drafts per verify round (default: 3).",
    )
    parser.add_argument(
        "--mtp-disable-auto-k",
        action="store_true",
        help="Benchmark fixed --mtp-max-k instead of the adaptive controller.",
    )
    parser.add_argument(
        "--require-lossless",
        action="store_true",
        help=(
            "Fail unless every greedy MTP run has the exact same emitted-token "
            "hash as its paired autoregressive run. Requires --temp 0 and a "
            "baseline condition."
        ),
    )
    parser.add_argument(
        "--min-speedup",
        type=float,
        default=None,
        help=(
            "Fail unless pooled MTP throughput is at least this multiple of "
            "the paired autoregressive baseline (for example 1.10)."
        ),
    )
    return parser.parse_args()


def _evaluate_landing_gates(
    results: dict[str, list[RunResult]],
    *,
    expected_pairs: int,
    require_lossless: bool,
    min_speedup: float | None,
    speedup: float | None,
) -> dict[str, Any]:
    """Evaluate correctness/performance gates without hiding missing runs.

    The greedy lossless contract is checked against every baseline arm
    that actually ran, and each is a distinct claim:

    * vs ``none`` — MTP reproduces *mlx_lm's* reference decode. This is
      the externally meaningful statement.
    * vs ``ar``   — MTP reproduces the same generator with speculation
      switched off. This isolates the speculation logic from the rest
      of the vendored harness, so a failure here points at the verify
      path rather than at any sampler/harness difference.

    Both must hold when both arms are present; a missing arm is skipped
    rather than silently treated as passing.
    """

    mtp = {(r.run_idx, r.prompt_idx): r for r in results["mtp"]}

    per_baseline: dict[str, Any] = {}
    lossless_passed = True
    all_paired: set[tuple[int, int]] = set()
    for name in ("none", "ar"):
        arm = {(r.run_idx, r.prompt_idx): r for r in results.get(name, [])}
        if not arm:
            continue
        paired = sorted(arm.keys() & mtp.keys())
        all_paired |= set(paired)
        mismatches = [
            {"run_idx": key[0], "prompt_idx": key[1]}
            for key in paired
            if arm[key].token_sha256 != mtp[key].token_sha256
        ]
        arm_complete = (
            len(arm) == expected_pairs
            and len(mtp) == expected_pairs
            and len(paired) == expected_pairs
        )
        arm_passed = (not require_lossless) or (arm_complete and not mismatches)
        lossless_passed = lossless_passed and arm_passed
        per_baseline[name] = {
            "passed": arm_passed,
            "complete": arm_complete,
            "complete_pairs": len(paired),
            "mismatches": mismatches,
        }

    # With no baseline arm at all there is nothing to compare against,
    # so a required lossless gate cannot be satisfied.
    if require_lossless and not per_baseline:
        lossless_passed = False

    complete = (
        len(mtp) == expected_pairs
        and bool(per_baseline)
        and all(b["complete"] for b in per_baseline.values())
    )
    speedup_passed = (not min_speedup) or (
        complete and speedup is not None and speedup >= min_speedup
    )
    return {
        "passed": lossless_passed and speedup_passed,
        "complete_pairs": len(all_paired),
        "expected_pairs": expected_pairs,
        "lossless": {
            "required": require_lossless,
            "passed": lossless_passed,
            "per_baseline": per_baseline,
        },
        "performance": {
            "minimum_speedup": min_speedup,
            "observed_speedup": speedup,
            # Which arm ``observed_speedup`` is measured against. "ar"
            # is the speculation-isolating comparison and is preferred
            # whenever that arm ran.
            "reference_arm": "ar" if results.get("ar") else "none",
            "passed": speedup_passed,
        },
    }


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
    if args.mtp_only:
        conditions = ["mtp"]
    elif args.skip_mlx_lm_arm:
        conditions = ["ar", "mtp"]
    else:
        conditions = ["none", "ar", "mtp"]
    n_conditions = len(conditions)
    return {
        "model": args.model,
        "runs_per_condition": args.runs,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "temp": args.temp,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "mtp_max_k": args.mtp_max_k,
        "mtp_disable_auto_k": args.mtp_disable_auto_k,
        "conditions": conditions,
        "prompts": list(_BENCH_PROMPTS[:n_prompts]),
        "total_generations": n_conditions * args.runs * n_prompts,
        "estimated_wall_time_seconds_at_15_tok_per_sec": (
            n_conditions * args.runs * n_prompts * args.max_tokens / 15.0
        ),
    }


def _run_once(
    *,
    model_alias: str,
    condition: str,
    prompt: str,
    max_tokens: int,
    temp: float,
    top_p: float = 0.0,
    top_k: int = 0,
    mtp_sidecar: str | None = None,
    mtp_max_k: int = 3,
    mtp_disable_auto_k: bool = False,
    seed: int = 0,
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
    from vllm_mlx.spec_decode.mtp.draft_k_controller_v2 import (
        sum_across_controllers,
    )

    model, tokenizer = load(model_alias)

    last_response = None
    # Two of the three arms drive the SAME generator:
    #
    #   "mtp"  — speculation on, depth K in [0, mtp_max_k]
    #   "ar"   — speculation off, depth pinned to K=0 (the controller
    #            "parks"), i.e. plain autoregressive decode through the
    #            identical code path, sampler chain, stop-token handling
    #            and (absence of) detokenization
    #   "none" — mlx_lm ``stream_generate``, kept as an absolute floor
    #
    # "mtp" vs "ar" is the only comparison that isolates speculation:
    # it holds the harness fixed and varies K alone. "mtp" vs "none"
    # additionally folds in every difference between the two harnesses
    # (entry point, sampler implementation, per-token detokenization),
    # so a regression there does not by itself implicate speculation.
    uses_generator = condition in ("mtp", "ar")
    if uses_generator:
        from vllm_mlx.spec_decode.mtp.qwen3_5_inject import (
            inject_mtp_support,
            validate_mtp_support,
        )

        if not inject_mtp_support(model, mtp_sidecar=mtp_sidecar):
            raise RuntimeError(
                f"MTP injection failed on model {model_alias!r} — "
                f"sidecar {mtp_sidecar!r}. Confirm the sidecar repo or "
                "local path exists, holds model.safetensors with the "
                "expected MTP head schema, and the base model's "
                "config carries mtp_num_hidden_layers >= 1."
            )
        assert validate_mtp_support(model)
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
    stop_tokens = _tokenizer_stop_tokens(tokenizer)
    mx.random.seed(seed)
    counter = MTPAcceptCounter()
    # Replace global counter for the duration of THIS run so the
    # ``accept_attempts`` / ``accept_count`` we report only reflects
    # the prompt under measurement (not whatever happened before).
    prior_attempts = get_global_counter().snapshot().attempts
    prior_accepts = get_global_counter().snapshot().accepts

    # Controllers intentionally persist across requests. Capture a delta so
    # each result exposes the K values this prompt actually exercised.
    # This benchmark runs generations serially, so no other request can
    # contaminate the process-global counters between these snapshots.
    _rounds_before, _parks_before, k_hist_before = sum_across_controllers()
    t0 = time.perf_counter()
    n = 0
    emitted_token_ids: list[int] = []
    if uses_generator:
        from vllm_mlx.spec_decode.mtp.generator import mtp_generate_step

        timing_stats: dict[str, float] = {}
        # K=0 with the adaptive controller disabled is a hard park: the
        # generator never drafts, so this is plain AR decode down the
        # identical path the "mtp" arm uses.
        effective_max_k = 0 if condition == "ar" else mtp_max_k
        effective_disable_auto_k = True if condition == "ar" else mtp_disable_auto_k
        gen = mtp_generate_step(
            prompt_ids,
            model,
            max_tokens=max_tokens,
            temp=temp,
            top_p=top_p,
            top_k=top_k,
            accept_counter=counter,
            max_k=effective_max_k,
            disable_auto_k=effective_disable_auto_k,
            stop_tokens=stop_tokens,
            timing_stats=timing_stats,
        )
        for token, _logprobs, _from_draft in gen:
            n += 1
            token_id = int(token)
            emitted_token_ids.append(token_id)
            # Production consumers stop pulling the generator at EOS. The
            # standalone bench must mirror that contract, including when EOS
            # is a target bonus/residual rather than an accepted draft.
            if token_id in stop_tokens:
                break
    else:
        timing_stats = {}
        from mlx_lm.generate import stream_generate
        from mlx_lm.sample_utils import make_sampler

        for resp in stream_generate(
            model,
            tokenizer,
            prompt,
            max_tokens=max_tokens,
            sampler=make_sampler(temp=temp, top_p=top_p, top_k=top_k),
        ):
            n += 1
            last_response = resp
            emitted_token_ids.append(int(resp.token))
            if n >= max_tokens:
                break

    elapsed = time.perf_counter() - t0
    _rounds_after, _parks_after, k_hist_after = sum_across_controllers()
    k_histogram = {
        k: k_hist_after.get(k, 0) - k_hist_before.get(k, 0)
        for k in k_hist_after.keys() | k_hist_before.keys()
        if k_hist_after.get(k, 0) != k_hist_before.get(k, 0)
    }

    snap = counter.snapshot()
    if not uses_generator:
        # ``none`` path doesn't touch the counter; report 0/0.
        snap_attempts, snap_accepts = 0, 0
    else:
        # Reported verbatim for both generator arms. The "ar" arm must
        # come back 0/0 — a nonzero attempt count there would mean the
        # K=0 park did not hold and the arm is not a clean AR baseline.
        snap_attempts, snap_accepts = snap.attempts, snap.accepts
        if condition == "ar" and snap_attempts != 0:
            raise RuntimeError(
                f"ar arm drafted {snap_attempts} positions despite max_k=0; "
                "the K=0 park did not hold, so this arm is not a valid "
                "same-path AR baseline"
            )
    # Sanity-check: global counter shouldn't have moved (per-run
    # counter is what mtp_generate_step bumps via the
    # ``accept_counter=`` kwarg).
    assert get_global_counter().snapshot().attempts == prior_attempts
    assert get_global_counter().snapshot().accepts == prior_accepts

    if uses_generator:
        prompt_eval_seconds = timing_stats.get("prompt_eval_seconds", 0.0)
        decode_elapsed = max(0.0, elapsed - prompt_eval_seconds)
        tok_per_sec = n / decode_elapsed if decode_elapsed > 0 else 0.0
    else:
        prompt_eval_seconds = (
            last_response.prompt_tokens / last_response.prompt_tps
            if last_response is not None and last_response.prompt_tps > 0
            else 0.0
        )
        decode_elapsed = (
            n / last_response.generation_tps
            if last_response is not None and last_response.generation_tps > 0
            else max(0.0, elapsed - prompt_eval_seconds)
        )
        tok_per_sec = n / decode_elapsed if decode_elapsed > 0 else 0.0
    end_to_end_tok_per_sec = n / elapsed if elapsed > 0 else 0.0
    return RunResult(
        condition=condition,
        run_idx=-1,  # patched up by the caller
        prompt_idx=-1,  # patched up by the caller
        decode_tok_per_sec=tok_per_sec,
        n_tokens=n,
        accept_attempts=snap_attempts,
        accept_count=snap_accepts,
        elapsed_seconds=elapsed,
        decode_elapsed_seconds=decode_elapsed,
        prompt_eval_seconds=prompt_eval_seconds,
        end_to_end_tok_per_sec=end_to_end_tok_per_sec,
        token_sha256=hashlib.sha256(
            ",".join(str(token) for token in emitted_token_ids).encode("ascii")
        ).hexdigest(),
        verify_kernel_calls=0,
        verify_kernel_fallbacks=0,
        verify_sync_seconds=timing_stats.get("verify_sync_seconds", 0.0),
        draft_seconds=timing_stats.get("draft_seconds", 0.0),
        residual_sync_seconds=timing_stats.get("residual_sync_seconds", 0.0),
        verify_calls=int(timing_stats.get("verify_calls", 0.0)),
        prompt_lookup_proposals=int(timing_stats.get("prompt_lookup_proposals", 0.0)),
        prompt_lookup_drafted_tokens=int(
            timing_stats.get("prompt_lookup_drafted_tokens", 0.0)
        ),
        prompt_lookup_accepted_tokens=int(
            timing_stats.get("prompt_lookup_accepted_tokens", 0.0)
        ),
        prompt_lookup_rejections=int(timing_stats.get("prompt_lookup_rejections", 0.0)),
        prompt_lookup_mtp_sync_seconds=timing_stats.get(
            "prompt_lookup_mtp_sync_seconds", 0.0
        ),
        k_histogram=k_histogram if uses_generator else {},
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
    total_elapsed = sum(r.decode_elapsed_seconds for r in results)
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


def _quantile(ordered: list[float], frac: float) -> float:
    """Linear-interpolated quantile over an already-sorted list."""
    if not ordered:
        return 0.0
    pos = frac * (len(ordered) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def _paired_ratios(
    results: list[RunResult],
    reference: list[RunResult],
) -> list[float]:
    """Per-cell tok/s ratios against the same ``(run, prompt)`` reference.

    Conditions run interleaved in the innermost loop, so the two arms of
    a cell execute back to back and share whatever thermal state the
    machine was in. Taking the ratio inside the cell cancels drift that
    a pooled mean would keep.
    """
    ref = {
        (r.run_idx, r.prompt_idx): r.decode_tok_per_sec
        for r in reference
        if r.decode_tok_per_sec > 0
    }
    ratios = []
    for r in results:
        base = ref.get((r.run_idx, r.prompt_idx))
        if base and r.decode_tok_per_sec > 0:
            ratios.append(r.decode_tok_per_sec / base)
    return sorted(ratios)


def _with_paired_speedup(
    summary: ConditionSummary,
    results: list[RunResult],
    reference: list[RunResult],
) -> ConditionSummary:
    """Attach the paired-ratio statistics to an already-pooled summary."""
    ratios = _paired_ratios(results, reference)
    if not ratios:
        return summary
    return replace(
        summary,
        paired_speedup_median=round(statistics.median(ratios), 3),
        paired_speedup_p25=round(_quantile(ratios, 0.25), 3),
        paired_speedup_p75=round(_quantile(ratios, 0.75), 3),
        paired_speedup_min=round(ratios[0], 3),
        paired_pairs=len(ratios),
        paired_pairs_slower=sum(1 for x in ratios if x < 1.0),
    )


def main() -> int:
    args = _parse_args()

    if args.require_lossless and args.temp != 0:
        raise SystemExit("--require-lossless requires --temp 0")
    if (args.require_lossless or args.min_speedup is not None) and args.mtp_only:
        raise SystemExit(
            "landing gates require the paired AR baseline; remove --mtp-only"
        )
    if args.min_speedup is not None and args.min_speedup <= 0:
        raise SystemExit("--min-speedup must be greater than zero")

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

    if args.prompt_text is not None:
        prompts = [args.prompt_text]
    else:
        n_selected = min(args.prompts, len(_BENCH_PROMPTS))
        prompts = list(_BENCH_PROMPTS[:n_selected])
    n_prompts = len(prompts)

    mtp_sidecar = _resolve_mtp_sidecar(args.model, args.mtp_sidecar)
    if args.mtp_only:
        conditions: tuple[str, ...] = ("mtp",)
    elif args.skip_mlx_lm_arm:
        conditions = ("ar", "mtp")
    else:
        conditions = ("none", "ar", "mtp")

    print(
        f"[bench_spec_decode_mtp] model={args.model} runs={args.runs} "
        f"prompts={n_prompts} max_tokens={args.max_tokens} temp={args.temp} "
        f"top_p={args.top_p} top_k={args.top_k} "
        f"seed={args.seed} "
        f"mtp_max_k={args.mtp_max_k} fixed_k={args.mtp_disable_auto_k} "
        f"mtp_sidecar={mtp_sidecar!r} conditions={conditions}",
        file=sys.stderr,
    )

    all_results: dict[str, list[RunResult]] = {"none": [], "ar": [], "mtp": []}
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
                        top_p=args.top_p,
                        top_k=args.top_k,
                        mtp_sidecar=mtp_sidecar,
                        mtp_max_k=args.mtp_max_k,
                        mtp_disable_auto_k=args.mtp_disable_auto_k,
                        seed=args.seed,
                    )
                except Exception as exc:  # pragma: no cover — bench
                    print(
                        f"[bench_spec_decode_mtp] {condition} run={run_idx} "
                        f"prompt={prompt_idx} FAILED: {exc}",
                        file=sys.stderr,
                    )
                    continue
                # Patch indices without manually copying every measurement
                # field. A hand-maintained reconstruction previously made new
                # diagnostics disappear silently from the final JSON.
                res = replace(res, run_idx=run_idx, prompt_idx=prompt_idx)
                all_results[condition].append(res)
                print(
                    f"[bench_spec_decode_mtp] {condition} run={run_idx} "
                    f"prompt={prompt_idx} {res.decode_tok_per_sec:.1f} tok/s "
                    f"({res.n_tokens} tokens in {res.decode_elapsed_seconds:.1f}s "
                    f"decode, {res.elapsed_seconds:.1f}s end-to-end; "
                    f"verify-kernel calls={res.verify_kernel_calls} "
                    f"fallbacks={res.verify_kernel_fallbacks}; "
                    f"verify-sync={res.verify_sync_seconds:.3f}s "
                    f"draft={res.draft_seconds:.3f}s "
                    f"residual={res.residual_sync_seconds:.3f}s; "
                    f"lookup={res.prompt_lookup_proposals}/"
                    f"{res.prompt_lookup_drafted_tokens}/"
                    f"{res.prompt_lookup_accepted_tokens} "
                    f"sync={res.prompt_lookup_mtp_sync_seconds:.3f}s; "
                    f"K={res.k_histogram})",
                    file=sys.stderr,
                )

    baseline_summary = _summarize("none", all_results["none"], None)
    ar_summary = _summarize(
        "ar", all_results["ar"], baseline_summary.pooled_tok_per_sec
    )

    # Headline speedup is measured against the same-path AR arm when it
    # ran: that arm differs from "mtp" only in whether speculation is
    # enabled, so the ratio attributes the delta to speculation and to
    # nothing else. Falling back to the mlx_lm arm keeps the old
    # behaviour when only that baseline is available, but the ratio then
    # also carries the harness difference between the two entry points.
    reference_tok_per_sec = (
        ar_summary.pooled_tok_per_sec
        if all_results["ar"]
        else baseline_summary.pooled_tok_per_sec
    )
    mtp_summary = _summarize("mtp", all_results["mtp"], reference_tok_per_sec)

    reference_runs = all_results["ar"] or all_results["none"]
    ar_summary = _with_paired_speedup(
        ar_summary, all_results["ar"], all_results["none"]
    )
    mtp_summary = _with_paired_speedup(mtp_summary, all_results["mtp"], reference_runs)

    gates = _evaluate_landing_gates(
        all_results,
        expected_pairs=args.runs * n_prompts,
        require_lossless=args.require_lossless,
        min_speedup=args.min_speedup,
        speedup=mtp_summary.speedup_vs_baseline,
    )

    summaries = [baseline_summary, ar_summary, mtp_summary]
    out = {
        "model": args.model,
        "max_tokens": args.max_tokens,
        "temp": args.temp,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "seed": args.seed,
        "speedup_reference_arm": "ar" if all_results["ar"] else "none",
        "summaries": [asdict(s) for s in summaries],
        "landing_gates": gates,
        "raw_runs": [asdict(r) for c in all_results.values() for r in c],
    }
    if args.format == "markdown":
        print("# MTP spec-decode bench\n")
        print(
            f"Model: `{args.model}`  max_tokens: {args.max_tokens}  temp: {args.temp}\n"
        )
        ref_arm = "ar" if all_results["ar"] else "none"
        print(
            f"Speedup column is measured against the `{ref_arm}` arm."
            + (
                "  `ar` = this same generator with K pinned to 0, so the "
                "`mtp` ratio isolates speculation.\n"
                if ref_arm == "ar"
                else "  No same-path arm ran, so the ratio also carries the "
                "harness difference between entry points.\n"
            )
        )
        print(
            "Paired speedup is the median per-`(run, prompt)` ratio against "
            "that same cell of the reference arm, with the interquartile "
            "range — it is the drift-resistant number; the pooled column is "
            "kept for continuity with PR #990.\n"
        )
        print(
            "| Condition | Tok/s pooled | Speedup pooled | "
            "Speedup paired (IQR) | Accept (A/V) |"
        )
        print("|---|---|---|---|---|")
        for s in summaries:
            if s.n_runs == 0:
                continue
            speedup = f"{s.speedup_vs_baseline:.2f}×" if s.speedup_vs_baseline else "—"
            if s.paired_speedup_median:
                paired = (
                    f"{s.paired_speedup_median:.3f}× "
                    f"({s.paired_speedup_p25:.3f}–{s.paired_speedup_p75:.3f}, "
                    f"n={s.paired_pairs}, {s.paired_pairs_slower} slower)"
                )
            else:
                paired = "—"
            accept = f"{s.accept_ratio:.1%}" if s.accept_ratio else "—"
            print(
                f"| {s.condition} | {s.pooled_tok_per_sec:.1f} | {speedup} "
                f"| {paired} | {accept} |"
            )
    else:
        print(json.dumps(out, indent=2))
    if not gates["passed"]:
        print(
            "[bench_spec_decode_mtp] landing gate FAILED: " + json.dumps(gates),
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover — script entry
    sys.exit(main())
