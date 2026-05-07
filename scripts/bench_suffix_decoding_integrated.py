#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Integrated SuffixDecoding eligibility bench (issue #269).

Purpose: classify a model into a SuffixDecoding *tier* by measuring the
speedup (or regression) it gets from ``--suffix-decoding`` on four
representative workloads:

    chat        — open-ended assistant turn (low repetition)
    json_array  — structured output (medium repetition)
    tool_loop   — agentic tool-call loop (high repetition)
    code_edit   — code editing diff (high repetition)

The classifier in ``vllm_mlx.model_auto_config.classify_suffix_decoding_tier``
turns the resulting ``{workload: speedup}`` dict into one of:

    agent / structured / neutral / avoid / unknown

Workflow:
    1. Start a server with the model and ``--suffix-decoding`` OFF.
    2. Run all four workloads × N=3 runs × ``--max-tokens`` tokens.
    3. Stop the server; restart with ``--suffix-decoding`` ON.
    4. Repeat workload runs.
    5. Compute median TPS per (workload, mode); ratio = ON / OFF.
    6. Classify; print summary; optionally write tier + speedup_dict
       into the corresponding ``ModelConfig`` entry in
       ``vllm_mlx/model_auto_config.py`` via ``--update-profile``.

Sequential server lifecycles avoid GPU contention; total wall-clock is
~10-20 min per model on M3 Ultra.

Usage:
    python3.12 scripts/bench_suffix_decoding_integrated.py \\
        --model mlx-community/Qwen3-0.6B-8bit \\
        --max-tokens 256 --runs 3

    # Update the model_auto_config.py entry in-place after a run:
    python3.12 scripts/bench_suffix_decoding_integrated.py \\
        --model mlx-community/Qwen3-0.6B-8bit \\
        --update-profile

This script is intentionally *opinionated* about the four workloads: the
tier table in #269 was tuned for these exact prompts. Editing them
silently shifts every model's classification. If you want to extend with
a fifth workload, update ``WORKLOADS`` here AND the boundaries in
``classify_suffix_decoding_tier``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from vllm_mlx.model_auto_config import (  # noqa: E402
    classify_suffix_decoding_tier,
)

logger = logging.getLogger("bench_suffix")
logging.basicConfig(level=logging.INFO, format="%(message)s")


# ---- Workload definitions -------------------------------------------------
#
# Each workload is a chat-completions request body sans the model field.
# Token budget per workload is bounded by ``--max-tokens``; the *prompt*
# size below is fixed so changing budget doesn't shift acceptance rates.

CHAT_WORKLOAD = {
    "messages": [
        {
            "role": "user",
            "content": (
                "Write a short, friendly explanation of why the sky appears "
                "blue. Two paragraphs, no bullet points."
            ),
        }
    ],
}

JSON_ARRAY_WORKLOAD = {
    "messages": [
        {
            "role": "user",
            "content": (
                "Return a JSON array of exactly 12 objects, each with the "
                "fields ``id`` (integer 1..12), ``name`` (a short fruit "
                "name) and ``color`` (a CSS color name). Output JSON only, "
                "no commentary."
            ),
        }
    ],
}

TOOL_LOOP_WORKLOAD = {
    "messages": [
        {
            "role": "system",
            "content": (
                "You are a data-pipeline assistant. Use the tools "
                "exactly when needed. Always end with a tool call to "
                "``submit_summary`` once data has been fetched, parsed, "
                "validated and written."
            ),
        },
        {
            "role": "user",
            "content": (
                "Build a fresh export of the ``invoices`` table for "
                "Q1 2026. Read the latest snapshot, parse it, validate "
                "row counts, write the result to S3, and finalize."
            ),
        },
    ],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "read_snapshot",
                "description": "Read latest data snapshot",
                "parameters": {
                    "type": "object",
                    "properties": {"table": {"type": "string"}},
                    "required": ["table"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "parse_rows",
                "description": "Parse rows",
                "parameters": {
                    "type": "object",
                    "properties": {"format": {"type": "string"}},
                    "required": ["format"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "validate_counts",
                "description": "Validate row counts vs expected",
                "parameters": {
                    "type": "object",
                    "properties": {"expected": {"type": "integer"}},
                    "required": ["expected"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "write_s3",
                "description": "Write export to S3",
                "parameters": {
                    "type": "object",
                    "properties": {"key": {"type": "string"}},
                    "required": ["key"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "submit_summary",
                "description": "Final summary, terminates the loop",
                "parameters": {
                    "type": "object",
                    "properties": {"status": {"type": "string"}},
                    "required": ["status"],
                },
            },
        },
    ],
    "tool_choice": "auto",
}

CODE_EDIT_WORKLOAD = {
    "messages": [
        {
            "role": "user",
            "content": (
                "Below is a Python function. Replace the bare ``except:`` "
                "with ``except Exception as e:`` and log the error using "
                "``logger.exception``. Return the entire updated function "
                "as a fenced code block.\n\n"
                "```python\n"
                "def fetch(url):\n"
                "    try:\n"
                "        return httpx.get(url, timeout=5).json()\n"
                "    except:\n"
                "        return None\n"
                "```"
            ),
        }
    ],
}

WORKLOADS: dict[str, dict] = {
    "chat": CHAT_WORKLOAD,
    "json_array": JSON_ARRAY_WORKLOAD,
    "tool_loop": TOOL_LOOP_WORKLOAD,
    "code_edit": CODE_EDIT_WORKLOAD,
}


# ---- Server lifecycle -----------------------------------------------------


@dataclass
class ServerHandle:
    proc: subprocess.Popen
    base_url: str
    model: str

    def stop(self) -> None:
        try:
            self.proc.send_signal(signal.SIGINT)
            self.proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()


def start_server(model: str, port: int, suffix_decoding: bool) -> ServerHandle:
    """Spin up ``rapid-mlx serve`` and wait for /v1/models to answer.

    ``suffix_decoding=True`` adds the ``--suffix-decoding`` flag. Logs go
    to a temp file so the parent process isn't drowned in startup noise.
    """
    log_path = f"/tmp/bench_suffix_{port}.log"
    cmd = [
        sys.executable,
        "-m",
        "vllm_mlx.cli",
        "serve",
        model,
        "--port",
        str(port),
        "--log-level",
        "WARNING",
        # Determinism: disable prefix cache so the second/third run of the
        # same workload doesn't replay cached generation. Without this,
        # disk-persisted entries from prior sessions can pin TPS to bogus
        # 1000+ tok/s outliers (decode_time goes to ~0). The bench is
        # measuring the suffix-decoding optimization, not cache reuse.
        "--disable-prefix-cache",
    ]
    if suffix_decoding:
        cmd.append("--suffix-decoding")

    logger.info("  starting server: port=%d suffix_decoding=%s", port, suffix_decoding)
    logf = open(log_path, "w")
    proc = subprocess.Popen(
        cmd,
        stdout=logf,
        stderr=subprocess.STDOUT,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )

    base_url = f"http://127.0.0.1:{port}/v1"
    deadline = time.time() + 600  # 10 min for cold cache + Metal compile
    while time.time() < deadline:
        if proc.poll() is not None:
            logf.close()
            logger.error("  server exited; tail of %s:", log_path)
            # Read the last ~30 lines directly instead of os.system("tail …")
            # — no shell, no injection surface, and the server-died code path
            # benefits from being silent on Windows where ``tail`` is absent.
            try:
                with open(log_path) as f:
                    tail_lines = f.readlines()[-30:]
                for line in tail_lines:
                    logger.error("    %s", line.rstrip())
            except OSError as e:
                logger.error("    (could not read log: %s)", e)
            raise RuntimeError("server died during startup")
        try:
            r = httpx.get(f"{base_url}/models", timeout=2.0)
            if r.status_code == 200:
                logger.info("  server up at %s", base_url)
                return ServerHandle(proc=proc, base_url=base_url, model=model)
        except Exception:
            pass
        time.sleep(2)

    proc.kill()
    raise RuntimeError(f"server did not become ready within deadline (port={port})")


# ---- Workload execution ---------------------------------------------------


# Reliability thresholds — see `tests/test_suffix_bench_methodology.py`.
#
# MIN_DECODE_TIME — runs whose decode-time slice is shorter than this can't
# be measured reliably. Per-request overhead (TLS handshake, framework JSON
# round-trip, last-chunk flush) dominates; TPS becomes 1000-2700 tok/s
# nonsense for short responses. 0.5s gives at least ~50-150 generated
# tokens at typical M-series rates, enough that wall-clock noise is small.
MIN_DECODE_TIME = 0.5

# TPS_CEILING — physical sanity check. No single-request mlx-lm decode on
# any M-series Mac (M2 → M3 Ultra) has been observed above this for any
# model size we ship as an alias (largest sustained: SmolLM3-3B-4bit ~280
# tok/s, Llama-3.2-1B-4bit ~330 tok/s). A reading above the ceiling means
# the timing window collapsed (cache hit, prefill-counted-as-decode, etc.)
# rather than the model genuinely going that fast.
TPS_CEILING = 500.0


@dataclass
class WorkloadRun:
    """One end-to-end stream measurement.

    ``tps`` is ``None`` when the run was rejected (too short, ceiling
    breach, or zero-token response). The raw fields stay populated so a
    later post-mortem can see *why* it was rejected without re-running.
    """

    tps: float | None
    completion_tokens: int
    decode_time: float
    total_time: float
    rejected_reason: str | None = None


def run_workload(
    handle: ServerHandle,
    workload_body: dict,
    max_tokens: int,
) -> WorkloadRun:
    """Run one request and return a ``WorkloadRun`` with TPS + raw timing.

    Reliability rules (set ``tps=None`` and a ``rejected_reason``):

    * Zero/missing ``completion_tokens`` — server didn't report usage.
    * ``decode_time < MIN_DECODE_TIME`` — gen too short to measure.
    * ``tps > TPS_CEILING`` — implausibly fast, indicates cache leak or
      collapsed timing window. Don't poison median() with the outlier.

    Returning a structured value (instead of just a float) lets the
    caller persist raw data for forensic debugging without re-running.
    """
    payload = {
        "model": handle.model,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        **workload_body,
    }
    t0 = time.perf_counter()
    ttft: float | None = None
    completion_tokens: int = 0

    with httpx.stream(
        "POST",
        f"{handle.base_url}/chat/completions",
        json=payload,
        timeout=300.0,
    ) as r:
        for line in r.iter_lines():
            if not line or not line.startswith("data: "):
                continue
            blob = line[len("data: ") :]
            if blob.strip() == "[DONE]":
                break
            try:
                obj = json.loads(blob)
            except json.JSONDecodeError:
                continue
            # First content delta sets TTFT.
            if ttft is None:
                choices = obj.get("choices") or []
                if choices and (
                    choices[0].get("delta", {}).get("content")
                    or choices[0].get("delta", {}).get("tool_calls")
                ):
                    ttft = time.perf_counter() - t0
            usage = obj.get("usage")
            if usage and usage.get("completion_tokens") is not None:
                completion_tokens = int(usage["completion_tokens"])

    total = time.perf_counter() - t0
    decode_time = max(total - (ttft or 0.0), 0.0)

    return _classify_run(completion_tokens, decode_time, total)


def _classify_run(
    completion_tokens: int, decode_time: float, total_time: float
) -> WorkloadRun:
    """Apply the reliability gates. Pulled out so it's unit-testable
    without spinning up an actual server."""
    if completion_tokens <= 0:
        return WorkloadRun(
            None, completion_tokens, decode_time, total_time, "zero_completion_tokens"
        )
    if decode_time < MIN_DECODE_TIME:
        return WorkloadRun(
            None,
            completion_tokens,
            decode_time,
            total_time,
            f"decode_time<{MIN_DECODE_TIME}s",
        )
    tps = completion_tokens / decode_time
    if tps > TPS_CEILING:
        return WorkloadRun(
            None,
            completion_tokens,
            decode_time,
            total_time,
            f"tps>{TPS_CEILING}_ceiling",
        )
    return WorkloadRun(tps, completion_tokens, decode_time, total_time, None)


# ---- Driver ---------------------------------------------------------------


@dataclass
class ModeResult:
    """Outcome of one (vanilla|suffix) bench pass.

    ``median_tps[name]`` is ``None`` when *every* run for that workload
    was rejected (so the caller can mark it 'skipped' rather than treat
    the missing data as 0.0 → false 'avoid' classification).
    """

    median_tps: dict[str, float | None]
    raw_runs: dict[str, list[WorkloadRun]]


def bench_one_mode(
    model: str,
    port: int,
    suffix_decoding: bool,
    runs: int,
    max_tokens: int,
) -> ModeResult:
    """Run every workload ``runs`` times in a single server lifetime.

    Returns a ``ModeResult`` with both the per-workload median (over
    valid runs only) and the full raw run list for downstream
    persistence.
    """
    handle = start_server(model, port, suffix_decoding)
    try:
        # 1 warmup run that we discard — covers Metal kernel JIT, cache
        # population, and tokenizer load.
        run_workload(handle, WORKLOADS["chat"], max_tokens=32)
        median_tps: dict[str, float | None] = {}
        raw: dict[str, list[WorkloadRun]] = {}
        for name, body in WORKLOADS.items():
            workload_runs: list[WorkloadRun] = []
            for i in range(runs):
                wr = run_workload(handle, body, max_tokens=max_tokens)
                tag = (
                    f"{wr.tps:.1f} tok/s"
                    if wr.tps is not None
                    else f"REJECTED ({wr.rejected_reason}, {wr.completion_tokens}t/{wr.decode_time:.2f}s decode)"
                )
                logger.info(
                    "  [%s] %s run %d/%d: %s",
                    "ON " if suffix_decoding else "OFF",
                    name,
                    i + 1,
                    runs,
                    tag,
                )
                workload_runs.append(wr)
            valid = [r.tps for r in workload_runs if r.tps is not None]
            median_tps[name] = median(valid) if valid else None
            raw[name] = workload_runs
        return ModeResult(median_tps=median_tps, raw_runs=raw)
    finally:
        handle.stop()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Bench SuffixDecoding eligibility for one model."
    )
    parser.add_argument("--model", required=True, help="HF repo or alias")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--runs", type=int, default=3, help="runs per workload")
    parser.add_argument("--port", type=int, default=8765, help="ephemeral port")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write JSON result file (defaults to evals/results/suffix_<model>.json)",
    )
    parser.add_argument(
        "--update-profile",
        action="store_true",
        help=(
            "After classifying, print the patch that would update the "
            "model_auto_config.py entry. Does NOT auto-edit the source — "
            "shows the tier + dict so the user can paste it in."
        ),
    )
    args = parser.parse_args(argv)

    logger.info("Bench: %s", args.model)

    logger.info("--- vanilla (suffix_decoding=OFF) ---")
    vanilla = bench_one_mode(
        args.model,
        args.port,
        suffix_decoding=False,
        runs=args.runs,
        max_tokens=args.max_tokens,
    )

    logger.info("--- suffix decoding ON ---")
    suffix = bench_one_mode(
        args.model,
        args.port,
        suffix_decoding=True,
        runs=args.runs,
        max_tokens=args.max_tokens,
    )

    # Compute per-workload speedup, skipping workloads where either mode
    # failed to produce *any* valid run. A workload with no valid data in
    # either pass shouldn't poison the tier — we drop it from the dict
    # the classifier sees and report it as 'skipped' in the JSON output.
    speedup: dict[str, float] = {}
    skipped: dict[str, str] = {}
    for name in WORKLOADS:
        v = vanilla.median_tps.get(name)
        s = suffix.median_tps.get(name)
        if v is None or s is None or v <= 0:
            reason = (
                "vanilla all runs rejected"
                if v is None or v <= 0
                else "suffix all runs rejected"
            )
            skipped[name] = reason
            continue
        speedup[name] = round(s / v, 3)

    tier = classify_suffix_decoding_tier(speedup) if speedup else "unknown"

    def _serialize_runs(raw: dict[str, list[WorkloadRun]]) -> dict[str, list[dict]]:
        return {
            name: [
                {
                    "tps": round(r.tps, 1) if r.tps is not None else None,
                    "completion_tokens": r.completion_tokens,
                    "decode_time": round(r.decode_time, 3),
                    "total_time": round(r.total_time, 3),
                    "rejected_reason": r.rejected_reason,
                }
                for r in runs
            ]
            for name, runs in raw.items()
        }

    summary = {
        "model": args.model,
        "max_tokens": args.max_tokens,
        "runs": args.runs,
        "vanilla_tps": {
            k: (round(v, 1) if v is not None else None)
            for k, v in vanilla.median_tps.items()
        },
        "suffix_tps": {
            k: (round(v, 1) if v is not None else None)
            for k, v in suffix.median_tps.items()
        },
        "speedup": speedup,
        "skipped": skipped,
        "tier": tier,
        "raw_runs": {
            "vanilla": _serialize_runs(vanilla.raw_runs),
            "suffix": _serialize_runs(suffix.raw_runs),
        },
    }

    output = args.output or REPO_ROOT / "evals/results" / (
        f"suffix_{args.model.replace('/', '_').lower()}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n")
    logger.info("--- summary ---")
    for k in WORKLOADS:
        if k in skipped:
            logger.info("  %-12s SKIPPED (%s)", k, skipped[k])
        else:
            v = speedup[k]
            marker = "↑" if v >= 1.05 else ("↓" if v <= 0.95 else "·")
            logger.info("  %-12s %5.2fx %s", k, v, marker)
    logger.info("  tier: %s", tier)
    logger.info("  written: %s", output)

    if args.update_profile:
        # The SSOT lives in aliases.json (PR #283 / #281). Print a JSON
        # snippet the contributor can paste into the alias entry.
        snippet = json.dumps(
            {"suffix_decoding_tier": tier, "suffix_bench_speedup": speedup},
            indent=2,
        )
        logger.info("\nPatch for vllm_mlx/aliases.json (alias %s):", args.model)
        for line in snippet.splitlines():
            logger.info("  %s", line)

    return 0


if __name__ == "__main__":
    sys.exit(main())
