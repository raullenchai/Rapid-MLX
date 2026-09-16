#!/usr/bin/env python3
"""Qualify the serialized-MLLM singleton no-rebatch fast path against the
legacy merge/rebatch path.

Paired A/B on one machine, one model revision, one process:

* baseline phase — ``SchedulerConfig(mllm_singleton_fastpath="off")``;
* candidate phase — ``"auto"`` (the shipped default).

Cases come from the tracked manifest
``evals/prompts/qwen36_mllm_runtime.json`` (20 media cases + 2 text-only
fallback cases: one warm-prefix case and one routing-boundary case). The
primary deterministic gate is per-case output equality
(SHA-256) between the two phases on the same exact build; the manifest's
machine-readable checkers guard against two equally wrong outputs passing.
``--lifecycle`` additionally exercises cancellation, recovery, queued
concurrency, and randomized abort/recovery iterations. ``--apc on`` runs
both phases with ``enable_prefix_cache=True`` and sends every case twice
per measured pass so warm exact-prefix resumes feed the singleton batch
(the cold default run cannot qualify that interaction); it also asserts
the candidate phase actually engaged the fast path via the generator's
``singleton_batches`` counter and that the APC actually served warm hits
in both phases (the manifest carries a long text-only case because
mlx-vlm's exact APC ignores boundaries below ``APC_EXACT_MIN_TOKENS``,
default 16).

Usage (Studio qualification, 256 GB, offline model resolution):

    python -m scripts.bench_qwen36_mllm_singleton \
        --model <snapshot-path> --pairs 3 --output /tmp/singleton.json

    python -m scripts.bench_qwen36_mllm_singleton \
        --model <snapshot-path> --pairs 2 --apc on --output /tmp/singleton-apc.json

    python -m scripts.bench_qwen36_mllm_singleton \
        --model <snapshot-path> --lifecycle --abort-iterations 50
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import re
import statistics
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = ROOT / "evals/prompts/qwen36_mllm_runtime.json"

try:
    from scripts.bench_metadata import write_bench_json
except ImportError:  # direct-script execution fallback
    from bench_metadata import write_bench_json


def _messages(case: dict[str, Any], repo_root: Path) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "text", "text": case["prompt"]}]
    for image in case.get("images", []):
        path = (repo_root / image).resolve()
        if not path.exists():
            raise FileNotFoundError(f"manifest image missing: {path}")
        content.append({"type": "image_url", "image_url": {"url": str(path)}})
    return [{"role": "user", "content": content}]


def _checker_pass(checker: dict[str, Any], text: str) -> bool:
    lowered = text.casefold()
    kind = checker.get("type", "any")
    if kind == "terms":
        if len(text.split()) < int(checker.get("min_words", 0)):
            return False
        if not all(term.casefold() in lowered for term in checker.get("required", [])):
            return False
        return not any(
            term.casefold() in lowered for term in checker.get("forbidden", [])
        )
    if kind == "json_shape":
        # Strip a trailing code fence the model may wrap the payload in.
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = stripped.split("\n", 1)[-1]
            if stripped.endswith("```"):
                stripped = stripped[:-3].rstrip()
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            return False
        if not isinstance(payload, dict) or not all(
            key in payload and payload[key] is not None
            for key in checker.get("keys", [])
        ):
            return False
        expected_types = {"str": str, "list": list, "dict": dict}
        for key, type_name in checker.get("types", {}).items():
            expected = expected_types.get(type_name)
            if expected is None:
                raise ValueError(f"unknown json checker type name: {type_name}")
            if key not in payload or not isinstance(payload[key], expected):
                return False
        for key, terms in checker.get("field_terms", {}).items():
            if key not in payload:
                return False
            rendered = json.dumps(payload[key], ensure_ascii=False).casefold()
            if not all(str(term).casefold() in rendered for term in terms):
                return False
        return True
    if kind == "regex":
        pattern = checker.get("pattern")
        if not isinstance(pattern, str) or not pattern:
            raise ValueError("regex checker requires a non-empty pattern")
        return re.fullmatch(pattern, text.strip()) is not None
    if kind == "any":
        return bool(text.strip())
    raise ValueError(f"unknown checker type: {kind}")


def _median(samples: list[dict[str, Any]], key: str) -> float:
    return statistics.median(float(sample[key]) for sample in samples)


def _delta(after: dict[str, Any], before: dict[str, Any], key: str) -> float:
    return float(after.get(key, 0.0)) - float(before.get(key, 0.0))


def _percent_change(after: float | None, before: float | None) -> float | None:
    """Return a percentage delta, or ``None`` for a zero/missing baseline."""

    if after is None or before is None or before == 0:
        return None
    return 100.0 * (after / before - 1.0)


def _lifecycle_passes(lifecycle: dict[str, Any], abort_iterations: int) -> bool:
    """Return whether every requested lifecycle qualification succeeded."""

    cancellation = lifecycle.get("cancellation", {})
    recovery = lifecycle.get("recovery", {})
    queued = lifecycle.get("queued_concurrency", {})
    soak = lifecycle.get("abort_soak", {})
    return bool(
        cancellation.get("request_id_published")
        and cancellation.get("accepted")
        and cancellation.get("tokens_before_abort", 0) > 0
        and recovery.get("exact")
        and recovery.get("completion_tokens", 0) > 0
        and queued.get("both_nonempty")
        and queued.get("serialized")
        and soak.get("pass")
        and not soak.get("failures")
        and soak.get("iterations") == abort_iterations
    )


def _hash_streams_exact(
    off_samples: list[dict[str, Any]], auto_samples: list[dict[str, Any]]
) -> bool:
    """Require one-to-one deterministic output for every measured pair/send."""

    def keyed(samples: list[dict[str, Any]]) -> dict[tuple[int, int], str] | None:
        mapped = {
            (int(sample["pair"]), int(sample["send"])): str(sample["sha256"])
            for sample in samples
        }
        return mapped if len(mapped) == len(samples) else None

    off_by_key = keyed(off_samples)
    auto_by_key = keyed(auto_samples)
    if not off_by_key or not auto_by_key or off_by_key.keys() != auto_by_key.keys():
        return False
    send_indexes = {send for _, send in off_by_key}
    deterministic = all(
        len({value for (_, stream), value in phase.items() if stream == send}) == 1
        for phase in (off_by_key, auto_by_key)
        for send in send_indexes
    )
    return deterministic and all(
        off_by_key[key] == auto_by_key[key] for key in off_by_key
    )


def _warm_phase_qualified(phase: dict[str, Any], require_singleton: bool) -> bool:
    """Require each measured warm send to hit APC on the designated case."""

    samples = phase.get("per_case", {}).get("text-warm-01", [])
    cold_samples = [sample for sample in samples if int(sample.get("send", 0)) == 1]
    warm_samples = [sample for sample in samples if int(sample.get("send", 0)) > 1]
    if not cold_samples or not warm_samples:
        return False
    cold_is_cold = all(
        int(sample.get("prefix_cache_hit_delta", 0)) == 0 for sample in cold_samples
    )
    warm_is_warm = all(
        int(sample.get("prefix_cache_hit_delta", 0)) > 0
        and (not require_singleton or int(sample.get("singleton_batch_delta", 0)) > 0)
        for sample in warm_samples
    )
    return cold_is_cold and warm_is_warm


def _measured_fastpath_engaged(phase: dict[str, Any]) -> bool:
    """Return whether a measured sample, rather than warmup, used the fast path."""
    return any(
        int(sample.get("singleton_batch_delta", 0)) > 0
        for samples in phase.get("per_case", {}).values()
        for sample in samples
    )


async def _run_case(
    engine: Any,
    case: dict[str, Any],
    repo_root: Path,
    *,
    max_tokens_override: int | None,
) -> dict[str, Any]:
    max_tokens = (
        int(case["max_tokens"]) if max_tokens_override is None else max_tokens_override
    )
    sampling = case.get("sampling", {})
    stats_before = engine.get_stats()
    before = dict(stats_before.get("batch_generator", {}))
    prefix_hits_before = int((stats_before.get("prefix_cache") or {}).get("hits", 0))
    started = time.perf_counter()
    first_token_at: float | None = None
    final = None
    async for output in engine.stream_chat(
        messages=_messages(case, repo_root),
        max_tokens=max_tokens,
        temperature=float(sampling.get("temperature", 0.0)),
        top_p=float(sampling.get("top_p", 1.0)),
        enable_thinking=False,
    ):
        if first_token_at is None and (output.new_text or output.completion_tokens > 0):
            first_token_at = time.perf_counter()
        final = output
    ended = time.perf_counter()
    stats_after = engine.get_stats()
    after = dict(stats_after["batch_generator"])
    prefix_hits_after = int((stats_after.get("prefix_cache") or {}).get("hits", 0))
    if final is None:
        raise RuntimeError(f"case {case['id']} produced no output")
    text = final.raw_text or final.text or ""
    generation_time = _delta(after, before, "generation_time")
    generation_tokens = _delta(after, before, "generation_tokens")
    return {
        "id": case["id"],
        "category": case["category"],
        "text": text,
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
        "checker_pass": _checker_pass(case.get("checker", {}), text),
        "prompt_tokens": int(final.prompt_tokens),
        "completion_tokens": int(final.completion_tokens),
        "elapsed_s": ended - started,
        "ttft_s": None if first_token_at is None else first_token_at - started,
        "generation_time_s": generation_time,
        "generation_tps": (generation_tokens / generation_time)
        if generation_time > 0
        else 0.0,
        "prefix_cache_hit_delta": prefix_hits_after - prefix_hits_before,
        "singleton_batch_delta": int(after.get("singleton_batches", 0))
        - int(before.get("singleton_batches", 0)),
    }


async def _memory_snapshot(engine: Any) -> dict[str, int]:
    def capture() -> dict[str, int]:
        import mlx.core as mx

        return {
            "active_bytes": int(mx.get_active_memory()),
            "cache_bytes": int(mx.get_cache_memory()),
            "peak_bytes": int(mx.get_peak_memory()),
        }

    return await engine.execute_on_model_worker(capture)


async def _run_phase(
    engine: Any,
    cases: list[dict[str, Any]],
    repo_root: Path,
    pairs: int,
    max_tokens_override: int | None,
    sends: int = 1,
) -> dict[str, Any]:
    """Warmup pass, then ``pairs`` measured passes over every case.

    With ``sends > 1`` (warm-hit qualification under ``--apc on``) every
    measured pass sends each case back-to-back ``sends`` times; send 2+ is
    expected to resume from warm APC state. Samples carry their ``send``
    index so exactness and medians can group cold versus warm.
    """
    for case in cases:
        await _run_case(
            engine, case, repo_root, max_tokens_override=max_tokens_override
        )

    by_case: dict[str, list[dict[str, Any]]] = {case["id"]: [] for case in cases}
    for pair in range(1, pairs + 1):
        # Warmup primes model/kernel state, not APC. Reset prompt state before
        # every pair so send 1 is cold and only send 2 can qualify a warm hit.
        engine.clear_prefix_cache(reset_stats=True)
        for case in cases:
            for send in range(1, sends + 1):
                sample = await _run_case(
                    engine, case, repo_root, max_tokens_override=max_tokens_override
                )
                sample["pair"] = pair
                sample["send"] = send
                by_case[case["id"]].append(sample)
    memory = await _memory_snapshot(engine)
    stats = engine.get_stats().get("batch_generator", {})
    cold = {
        case_id: [s for s in samples if s["send"] == 1]
        for case_id, samples in by_case.items()
    }
    warm = {
        case_id: [s for s in samples if s["send"] > 1]
        for case_id, samples in by_case.items()
    }
    return {
        "per_case": by_case,
        "memory": memory,
        "singleton_batches": int(stats.get("singleton_batches", 0)),
        "prefix_cache": engine.get_stats().get("prefix_cache"),
        # pairs == 0 runs warmup only (lifecycle-only mode): no medians.
        # Medians stay on the cold (send-1) stream; warm TTFT measures the
        # APC resume and is reported separately for evidence.
        "summary": {
            case_id: {
                "median_ttft_s": _median(samples, "ttft_s") if samples else None,
                "median_elapsed_s": _median(samples, "elapsed_s") if samples else None,
                "median_generation_tps": _median(samples, "generation_tps")
                if samples
                else None,
                "median_completion_tokens": _median(samples, "completion_tokens")
                if samples
                else None,
                "median_ttft_s_warm": _median(warm[case_id], "ttft_s")
                if warm[case_id]
                else None,
                "checker_passes": sum(bool(s["checker_pass"]) for s in samples),
            }
            for case_id, samples in cold.items()
        },
    }


async def _run_lifecycle(
    engine: Any,
    cases: list[dict[str, Any]],
    repo_root: Path,
    abort_iterations: int,
    seed: int,
) -> dict[str, Any]:
    """Cancellation, recovery, queued concurrency, randomized abort soak."""
    by_id = {case["id"]: case for case in cases}
    media_case = next(case for case in cases if case.get("images"))
    long_prompt = (
        "Describe every visible element in this screenshot in detail, then "
        "provide a long accessibility review of at least 300 words."
    )
    result: dict[str, Any] = {}

    # Capture the deterministic reference before cancellation. Comparing two
    # post-abort requests could let persistent corruption make both outputs
    # identically wrong.
    recovery_case = by_id.get("ocr-01", cases[0])
    recovery_messages = _messages(recovery_case, repo_root)
    reference = await engine.chat(
        messages=recovery_messages,
        max_tokens=int(recovery_case["max_tokens"]),
        temperature=0.0,
        top_p=1.0,
        enable_thinking=False,
    )
    reference_text = reference.raw_text or reference.text or ""

    # 1. Cancellation mid-generation.
    holder: list[str | None] = [None]
    stream = engine.stream_chat(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": long_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": str((repo_root / media_case["images"][0]).resolve())
                        },
                    },
                ],
            }
        ],
        max_tokens=512,
        temperature=0.0,
        top_p=1.0,
        enable_thinking=False,
        request_id_holder=holder,
    )
    seen = 0
    started = time.perf_counter()
    async for output in stream:
        seen = output.completion_tokens
        if seen >= 12:
            break
    request_id = holder[0]
    accepted = bool(request_id) and await engine.abort_request(request_id)
    await stream.aclose()
    await asyncio.sleep(0)
    result["cancellation"] = {
        "request_id_published": bool(request_id),
        "accepted": accepted,
        "tokens_before_abort": seen,
        "elapsed_s": time.perf_counter() - started,
    }

    # 2. Recovery after cancellation, byte-compared against the clean
    # pre-cancellation reference above.
    recovered = await engine.chat(
        messages=recovery_messages,
        max_tokens=int(recovery_case["max_tokens"]),
        temperature=0.0,
        top_p=1.0,
        enable_thinking=False,
    )
    recovered_text = recovered.raw_text or recovered.text or ""
    result["recovery"] = {
        "exact": recovered_text == reference_text,
        "completion_tokens": recovered.completion_tokens,
    }

    # 3. Queued concurrency: two simultaneous media requests serialize.
    async def queued(case: dict[str, Any]) -> str:
        output = await engine.chat(
            messages=_messages(case, repo_root),
            max_tokens=int(case["max_tokens"]),
            temperature=0.0,
            top_p=1.0,
            enable_thinking=False,
        )
        return output.raw_text or output.text or ""

    media_cases = [case for case in cases if case.get("images")][:2]
    queued_started = time.perf_counter()
    queued_tasks = [
        asyncio.create_task(queued(media_cases[0])),
        asyncio.create_task(queued(media_cases[1])),
    ]
    first, second = await asyncio.gather(*queued_tasks)
    queue_stats = engine.get_stats()
    max_running_observed = int(queue_stats.get("max_num_running_observed", 0))
    max_waiting_observed = int(queue_stats.get("max_num_waiting_observed", 0))
    observed_running_with_waiter = bool(
        queue_stats.get("observed_running_with_waiter", False)
    )
    result["queued_concurrency"] = {
        "elapsed_s": time.perf_counter() - queued_started,
        "num_requests_processed": engine.get_stats().get("num_requests_processed"),
        "both_nonempty": bool(first) and bool(second),
        "max_running_observed": max_running_observed,
        "max_waiting_observed": max_waiting_observed,
        "observed_running_with_waiter": observed_running_with_waiter,
        "serialized": max_running_observed == 1 and observed_running_with_waiter,
    }

    # 4. Randomized abort/recovery soak: abort at a random token count, then
    # require the next full request to succeed and match its own repeat.
    rng = random.Random(seed)
    soak_failures: list[str] = []
    for iteration in range(abort_iterations):
        holder = [None]
        stream = engine.stream_chat(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": long_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": str(
                                    (repo_root / media_case["images"][0]).resolve()
                                )
                            },
                        },
                    ],
                }
            ],
            max_tokens=512,
            temperature=0.0,
            top_p=1.0,
            enable_thinking=False,
            request_id_holder=holder,
        )
        abort_at = rng.randint(4, 40)
        seen = 0
        async for output in stream:
            seen = output.completion_tokens
            if seen >= abort_at:
                break
        abort_id = holder[0]
        abort_accepted = bool(abort_id) and await engine.abort_request(abort_id)
        await stream.aclose()
        await asyncio.sleep(0)

        if not abort_id:
            soak_failures.append(f"iteration-{iteration}:missing-request-id")
        elif not abort_accepted:
            soak_failures.append(f"iteration-{iteration}:abort-rejected")

        probe = await engine.chat(
            messages=recovery_messages,
            max_tokens=int(recovery_case["max_tokens"]),
            temperature=0.0,
            top_p=1.0,
            enable_thinking=False,
        )
        probe_text = probe.raw_text or probe.text or ""
        if probe_text != reference_text:
            soak_failures.append(f"iteration-{iteration}")
    result["abort_soak"] = {
        "iterations": abort_iterations,
        "seed": seed,
        "failures": soak_failures,
        "pass": not soak_failures,
    }
    return result


async def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Model snapshot path")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--cases",
        action="append",
        default=[],
        help="Case id filter (repeatable); default runs the whole manifest",
    )
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument(
        "--apc",
        choices=["off", "on"],
        default="off",
        help=(
            "Run both phases with enable_prefix_cache=True so the A/B "
            "exercises warm exact-prefix resumes feeding the singleton "
            "batch; each measured pass sends every case twice (send 2 is "
            "the warm hit). Default off keeps the cold-path primary gate."
        ),
    )
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--lifecycle", action="store_true")
    parser.add_argument("--abort-iterations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--vision-max-pixels", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    if args.pairs < 0:
        parser.error("--pairs must be non-negative")
    if args.pairs == 0 and not args.lifecycle:
        parser.error("--pairs 0 is valid only with --lifecycle")
    if args.max_tokens is not None and args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")
    if args.abort_iterations <= 0:
        parser.error("--abort-iterations must be positive")

    manifest = json.loads(args.manifest.read_text())
    # Image paths resolve against <manifest-dir>/<images_root> so a manifest
    # nested under evals/prompts/ points at repo-tracked fixtures portably.
    repo_root = (
        args.manifest.resolve().parent / manifest.get("images_root", ".")
    ).resolve()
    manifest_ids = {case["id"] for case in manifest["cases"]}
    unknown_case_ids = sorted(set(args.cases) - manifest_ids)
    if unknown_case_ids:
        parser.error(f"unknown --cases id(s): {', '.join(unknown_case_ids)}")
    cases = [
        case for case in manifest["cases"] if not args.cases or case["id"] in args.cases
    ]
    if not cases:
        raise SystemExit("no manifest cases selected")
    if args.lifecycle and sum(bool(case.get("images")) for case in cases) < 2:
        parser.error(
            "--lifecycle requires at least two selected cases with image fixtures"
        )

    # Keep validation usable on non-MLX hosts (including Linux CI) by loading
    # the runtime only after every argument and manifest contract has passed.
    from rapid_mlx.engine.batched import BatchedEngine
    from rapid_mlx.scheduler import SchedulerConfig

    result: dict[str, Any] = {
        "model": str(Path(args.model).expanduser().resolve()),
        "manifest": str(args.manifest),
        "pairs": args.pairs,
        "apc": args.apc,
        "phases": {},
    }
    sends = 2 if args.apc == "on" else 1

    async def engine_for(fastpath: str) -> BatchedEngine:
        engine = BatchedEngine(
            str(Path(args.model).expanduser().resolve()),
            force_mllm=True,
            # ``no_hybrid`` keeps the qualified native text engine unstarted
            # so the manifest's text-only fallback case exercises the MLLM
            # scheduler — exactly the A4 in-scope fallback scenario. With
            # the native text engine started, text-only requests route away
            # from this lane and are out of the singleton PR's scope.
            no_hybrid=True,
            scheduler_config=SchedulerConfig(
                # ``--apc on`` matches the production default so the A/B
                # covers the riskiest interaction: warm exact-prefix
                # snapshots (lookup/snap/rewind clones) admitted as live
                # singleton decode leaves.
                enable_prefix_cache=args.apc == "on",
                vision_max_pixels=args.vision_max_pixels,
                mllm_singleton_fastpath=fastpath,
            ),
        )
        await engine.start()
        return engine

    try:
        # Baseline: legacy merge/rebatch path.
        baseline = await engine_for("off")
        try:
            result["phases"]["off"] = await _run_phase(
                baseline, cases, repo_root, args.pairs, args.max_tokens, sends=sends
            )
        finally:
            await baseline.stop()

        # Candidate: shipped default fast path.
        candidate = await engine_for("auto")
        try:
            result["phases"]["auto"] = await _run_phase(
                candidate, cases, repo_root, args.pairs, args.max_tokens, sends=sends
            )
            if args.lifecycle:
                result["lifecycle"] = await _run_lifecycle(
                    candidate, cases, repo_root, args.abort_iterations, args.seed
                )
        finally:
            await candidate.stop()

        per_case_exact: dict[str, bool] = {}
        for case_id, off_samples in result["phases"]["off"]["per_case"].items():
            auto_samples = result["phases"]["auto"]["per_case"][case_id]
            if not off_samples and not auto_samples:
                continue
            # Compare cold and warm send streams separately so a warm-hit
            # divergence cannot hide behind the cold stream's hashes.
            per_case_exact[case_id] = _hash_streams_exact(off_samples, auto_samples)
        result["exact_by_case"] = per_case_exact
        result["exact_cases"] = sum(per_case_exact.values())
        result["total_cases"] = len(per_case_exact)
        result["checker_failures"] = [
            {
                "phase": phase,
                "case": case_id,
                "send": sample["send"],
                "pair": sample["pair"],
            }
            for phase in ("off", "auto")
            for case_id, samples in result["phases"][phase]["per_case"].items()
            for sample in samples
            if not sample["checker_pass"]
        ]

        # The candidate phase must actually have taken the fast path, and the
        # baseline must never have: an eligibility regression would otherwise
        # compare off vs off and record a vacuous pass as qualification.
        result["fastpath_engaged"] = _measured_fastpath_engaged(
            result["phases"]["auto"]
        ) and not _measured_fastpath_engaged(result["phases"]["off"])
        # Warm-hit qualification requires the APC to actually serve hits in
        # both phases. mlx-vlm's exact APC stores a turn boundary only when
        # it clears ``APC_EXACT_MIN_TOKENS`` (default 16), so the manifest
        # must carry a long text-only case; if no hit lands the warm
        # interaction went unqualified and the run must not count.
        result["apc_hits"] = {
            phase: (result["phases"][phase].get("prefix_cache") or {}).get("hits", 0)
            for phase in ("off", "auto")
        }
        result["warm_qualified"] = args.apc == "off" or (
            _warm_phase_qualified(result["phases"]["off"], False)
            and _warm_phase_qualified(result["phases"]["auto"], True)
        )

        result["summary_change_pct"] = {
            case_id: {
                "ttft": _percent_change(
                    result["phases"]["auto"]["summary"][case_id]["median_ttft_s"],
                    result["phases"]["off"]["summary"][case_id]["median_ttft_s"],
                ),
                "elapsed": _percent_change(
                    result["phases"]["auto"]["summary"][case_id]["median_elapsed_s"],
                    result["phases"]["off"]["summary"][case_id]["median_elapsed_s"],
                ),
                "generation_tps": _percent_change(
                    result["phases"]["auto"]["summary"][case_id][
                        "median_generation_tps"
                    ],
                    result["phases"]["off"]["summary"][case_id][
                        "median_generation_tps"
                    ],
                ),
            }
            for case_id in result["phases"]["auto"]["summary"]
        }
    finally:
        pass

    if args.output:
        write_bench_json(args.output, result, Path(__file__))
    payload = {
        key: result[key]
        for key in (
            "exact_by_case",
            "exact_cases",
            "total_cases",
            "checker_failures",
            "fastpath_engaged",
            "apc_hits",
            "warm_qualified",
            "summary_change_pct",
            "lifecycle",
            "phases",
        )
        if key in result
    }
    if args.summary_only:
        payload.pop("phases", None)
    try:
        from scripts.bench_metadata import format_bench_json
    except ImportError:
        from bench_metadata import format_bench_json

    print(format_bench_json(payload, Path(__file__)))
    if result.get("exact_cases", 0) != result.get("total_cases", 0):
        raise SystemExit(1)
    if result.get("checker_failures"):
        raise SystemExit(4)
    if args.pairs > 0 and not result.get("fastpath_engaged", False):
        raise SystemExit(2)
    if args.pairs > 0 and not result.get("warm_qualified", True):
        raise SystemExit(3)
    if args.lifecycle and not _lifecycle_passes(
        result.get("lifecycle", {}), args.abort_iterations
    ):
        raise SystemExit(5)


if __name__ == "__main__":
    asyncio.run(_main())
