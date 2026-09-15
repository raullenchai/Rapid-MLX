#!/usr/bin/env python3
"""Qualify the serialized-MLLM singleton no-rebatch fast path against the
legacy merge/rebatch path.

Paired A/B on one machine, one model revision, one process:

* baseline phase — ``SchedulerConfig(mllm_singleton_fastpath="off")``;
* candidate phase — ``"auto"`` (the shipped default).

Cases come from the tracked manifest
``evals/prompts/qwen36_mllm_runtime.json`` (20 media cases + 1 text-only
fallback case). The primary deterministic gate is per-case output equality
(SHA-256) between the two phases on the same exact build; the manifest's
machine-readable checkers guard against two equally wrong outputs passing.
``--lifecycle`` additionally exercises cancellation, recovery, queued
concurrency, and randomized abort/recovery iterations.

Usage (Studio qualification, 256 GB, offline model resolution):

    python -m scripts.bench_qwen36_mllm_singleton \
        --model <snapshot-path> --pairs 3 --output /tmp/singleton.json

    python -m scripts.bench_qwen36_mllm_singleton \
        --model <snapshot-path> --lifecycle --abort-iterations 50
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import statistics
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = ROOT / "evals/prompts/qwen36_mllm_runtime.json"


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
                stripped = stripped[:-3]
        try:
            start = stripped.index("{")
            payload = json.loads(stripped[start:])
        except (ValueError, json.JSONDecodeError):
            return False
        return isinstance(payload, dict) and all(
            key in payload for key in checker.get("keys", [])
        )
    if kind == "any":
        return bool(text.strip())
    raise ValueError(f"unknown checker type: {kind}")


def _median(samples: list[dict[str, Any]], key: str) -> float:
    return statistics.median(float(sample[key]) for sample in samples)


def _delta(after: dict[str, Any], before: dict[str, Any], key: str) -> float:
    return float(after.get(key, 0.0)) - float(before.get(key, 0.0))


async def _run_case(
    engine: Any,
    case: dict[str, Any],
    repo_root: Path,
    *,
    max_tokens_override: int | None,
) -> dict[str, Any]:
    max_tokens = max_tokens_override or int(case["max_tokens"])
    sampling = case.get("sampling", {})
    before = dict(engine.get_stats().get("batch_generator", {}))
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
    after = dict(engine.get_stats()["batch_generator"])
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
) -> dict[str, Any]:
    """Warmup pass, then ``pairs`` measured passes over every case."""
    for case in cases:
        await _run_case(
            engine, case, repo_root, max_tokens_override=max_tokens_override
        )

    by_case: dict[str, list[dict[str, Any]]] = {case["id"]: [] for case in cases}
    for _ in range(pairs):
        for case in cases:
            by_case[case["id"]].append(
                await _run_case(
                    engine, case, repo_root, max_tokens_override=max_tokens_override
                )
            )
    memory = await _memory_snapshot(engine)
    return {
        "per_case": by_case,
        "memory": memory,
        # pairs == 0 runs warmup only (lifecycle-only mode): no medians.
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
                "checker_passes": sum(bool(s["checker_pass"]) for s in samples),
            }
            for case_id, samples in by_case.items()
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

    # 2. Recovery after cancellation, byte-compared against a fresh run.
    recovery_case = by_id.get("ocr-01", cases[0])
    recovery_messages = _messages(recovery_case, repo_root)
    recovered = await engine.chat(
        messages=recovery_messages,
        max_tokens=int(recovery_case["max_tokens"]),
        temperature=0.0,
        top_p=1.0,
        enable_thinking=False,
    )
    reference = await engine.chat(
        messages=recovery_messages,
        max_tokens=int(recovery_case["max_tokens"]),
        temperature=0.0,
        top_p=1.0,
        enable_thinking=False,
    )
    recovered_text = recovered.raw_text or recovered.text or ""
    reference_text = reference.raw_text or reference.text or ""
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
    first, second = await asyncio.gather(queued(media_cases[0]), queued(media_cases[1]))
    result["queued_concurrency"] = {
        "elapsed_s": time.perf_counter() - queued_started,
        "num_requests_processed": engine.get_stats().get("num_requests_processed"),
        "both_nonempty": bool(first) and bool(second),
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
        if holder[0]:
            await engine.abort_request(holder[0])
        await stream.aclose()
        await asyncio.sleep(0)

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
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--lifecycle", action="store_true")
    parser.add_argument("--abort-iterations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--vision-max-pixels", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    from vllm_mlx.engine.batched import BatchedEngine
    from vllm_mlx.scheduler import SchedulerConfig

    manifest = json.loads(args.manifest.read_text())
    # Image paths resolve against <manifest-dir>/<images_root> so a manifest
    # nested under evals/prompts/ points at repo-tracked fixtures portably.
    repo_root = (
        args.manifest.resolve().parent / manifest.get("images_root", ".")
    ).resolve()
    cases = [
        case for case in manifest["cases"] if not args.cases or case["id"] in args.cases
    ]
    if not cases:
        raise SystemExit("no manifest cases selected")

    result: dict[str, Any] = {
        "model": str(Path(args.model).expanduser().resolve()),
        "manifest": str(args.manifest),
        "pairs": args.pairs,
        "phases": {},
    }

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
                enable_prefix_cache=False,
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
                baseline, cases, repo_root, args.pairs, args.max_tokens
            )
        finally:
            await baseline.stop()

        # Candidate: shipped default fast path.
        candidate = await engine_for("auto")
        try:
            result["phases"]["auto"] = await _run_phase(
                candidate, cases, repo_root, args.pairs, args.max_tokens
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
            off_hashes = {sample["sha256"] for sample in off_samples}
            auto_hashes = {sample["sha256"] for sample in auto_samples}
            per_case_exact[case_id] = off_hashes == auto_hashes
        result["exact_by_case"] = per_case_exact
        result["exact_cases"] = sum(per_case_exact.values())
        result["total_cases"] = len(per_case_exact)

        result["summary_change_pct"] = {
            case_id: {
                "ttft": 100.0
                * (
                    result["phases"]["auto"]["summary"][case_id]["median_ttft_s"]
                    / result["phases"]["off"]["summary"][case_id]["median_ttft_s"]
                    - 1.0
                ),
                "elapsed": 100.0
                * (
                    result["phases"]["auto"]["summary"][case_id]["median_elapsed_s"]
                    / result["phases"]["off"]["summary"][case_id]["median_elapsed_s"]
                    - 1.0
                ),
                "generation_tps": 100.0
                * (
                    result["phases"]["auto"]["summary"][case_id][
                        "median_generation_tps"
                    ]
                    / result["phases"]["off"]["summary"][case_id][
                        "median_generation_tps"
                    ]
                    - 1.0
                ),
            }
            for case_id in result["phases"]["auto"]["summary"]
            if result["phases"]["off"]["summary"][case_id]["median_ttft_s"]
        }
    finally:
        pass

    if args.output:
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True))
    payload = {
        key: result[key]
        for key in (
            "exact_by_case",
            "exact_cases",
            "total_cases",
            "summary_change_pct",
            "lifecycle",
            "phases",
        )
        if key in result
    }
    if args.summary_only:
        payload.pop("phases", None)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if result.get("exact_cases", 0) != result.get("total_cases", 0):
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(_main())
