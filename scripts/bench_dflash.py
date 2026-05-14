#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""DFlash speculative-decoding speedup bench (Model Onboarding SOP §6).

Measures the per-workload TPS speedup of ``--enable-dflash`` vs the
baseline (autoregressive) decode on a fixed prompt set. Mirrors the
sequential two-server pattern of ``bench_suffix_decoding_integrated.py``
and reuses the same reliability gates (decode-time floor, TPS ceiling,
raw-runs persistence).

Workloads are code-generation-heavy (Fibonacci, Quicksort, HashTable,
SortedList) — DFlash drafters were trained on code distributions and
this is the regime where users see the headline speedup. Mixed
math/text workloads accept less, but those are not what the ≥1.3× SOP
gate is designed to validate.

Decision rule (SOP §6 / Model Onboarding SOP):
    median(speedup over workloads) ≥ 1.30  →  ship supports_dflash=true
    otherwise                               →  keep supports_dflash=false

Usage:
    python3.12 scripts/bench_dflash.py \\
        --model qwen3.5-4b-8bit --runs 3 --max-tokens 256

The script does NOT auto-edit aliases.json. It prints the patch the
contributor can paste, and persists the raw bench data to
``evals/results/dflash_<model>.json`` for the PR record.

Special: ``RAPID_MLX_DFLASH_BYPASS_MOE_GATE=1`` env var is honored so a
PoC can re-verify a previously-rejected MoE alias without first having
to land a code change.
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

logger = logging.getLogger("bench_dflash")
logging.basicConfig(level=logging.INFO, format="%(message)s")


# ---- Workloads ------------------------------------------------------------
#
# Code-gen prompts pulled from the DFlash family of papers / z-lab's
# canonical demo set. Each prompt is sized so a ~256-token budget hits
# the decode-time floor on M-series hardware. Adding a fifth workload
# is fine; removing one shifts the median speedup and should be
# documented in the PR.

FIBONACCI_WORKLOAD = {
    "messages": [
        {
            "role": "user",
            "content": (
                "Write a Python function ``fibonacci(n)`` that returns the "
                "n-th Fibonacci number using memoized recursion. Include "
                "a docstring describing the function, time complexity, and "
                "one usage example in a comment. Do not include tests."
            ),
        }
    ],
}

QUICKSORT_WORKLOAD = {
    "messages": [
        {
            "role": "user",
            "content": (
                "Implement an in-place ``quicksort(arr, lo, hi)`` in Python "
                "using the Lomuto partition scheme. Include a docstring "
                "describing the algorithm, average-case and worst-case time "
                "complexity, and one usage example in a comment. Do not "
                "include tests."
            ),
        }
    ],
}

HASHTABLE_WORKLOAD = {
    "messages": [
        {
            "role": "user",
            "content": (
                "Implement a Python ``HashTable`` class with ``put``, "
                "``get`` and ``remove`` methods using separate-chaining for "
                "collision resolution. Include a docstring on the class "
                "and one usage example in a comment. Do not include tests."
            ),
        }
    ],
}

SORTEDLIST_WORKLOAD = {
    "messages": [
        {
            "role": "user",
            "content": (
                "Implement a Python ``SortedList`` class with ``add(x)``, "
                "``remove(x)``, and ``pop_min()`` methods using a sorted "
                "internal array and ``bisect``. Include a docstring on the "
                "class and one usage example in a comment. Do not include "
                "tests."
            ),
        }
    ],
}

# Chat workload — required since prior bench rounds discovered that
# DFlash speedup on Qwen3.5/3.6 family is heavily code-gen-biased; the
# same drafter that hits 1.3-1.7× on code can regress to <1.0× on
# open-ended chat (lower token-tree overlap). Adding chat to the SHIP
# gate keeps us honest on the assistant-style users see in practice.
CHAT_WORKLOAD = {
    "messages": [
        {
            "role": "user",
            "content": (
                "Write a short, friendly explanation of why the sky "
                "appears blue. Two paragraphs, no bullet points."
            ),
        }
    ],
}

WORKLOADS: dict[str, dict] = {
    "fibonacci": FIBONACCI_WORKLOAD,
    "quicksort": QUICKSORT_WORKLOAD,
    "hashtable": HASHTABLE_WORKLOAD,
    "sortedlist": SORTEDLIST_WORKLOAD,
    "chat": CHAT_WORKLOAD,
}

# Workloads counted as "code" for the gate; the rest go in the "non-code"
# bucket and must individually not regress (chat ≥1.00x). Keeps the
# gate honest: code median up, chat at least flat.
_CODE_WORKLOADS: frozenset[str] = frozenset(
    {"fibonacci", "quicksort", "hashtable", "sortedlist"}
)


# ---- Server lifecycle -----------------------------------------------------


@dataclass
class ServerHandle:
    proc: subprocess.Popen
    base_url: str
    model: str
    # File the child's stdout/stderr is piped into. Held for the
    # lifetime of the server so the OS keeps the descriptor open for
    # writes from the child; closed in stop() once the process exits.
    logf: object = None

    def stop(self) -> None:
        try:
            self.proc.send_signal(signal.SIGINT)
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
        finally:
            if self.logf is not None:
                try:
                    self.logf.close()
                except Exception:
                    pass


def start_server(model: str, port: int, dflash: bool) -> ServerHandle:
    """Spin up ``rapid-mlx serve`` and wait for /v1/models to answer.

    Sets ``--disable-prefix-cache`` to prevent disk-persisted cache
    entries from a prior session pinning TPS to bogus 1000+ tok/s.
    DFlash speedup measurement requires real decode work on every run.
    """
    log_path = f"/tmp/bench_dflash_{port}.log"
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
        "--disable-prefix-cache",
    ]
    if dflash:
        cmd.append("--enable-dflash")

    logger.info("  starting server: port=%d dflash=%s", port, dflash)
    logf = open(log_path, "w")
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=logf,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
    except Exception:
        logf.close()
        raise

    base_url = f"http://127.0.0.1:{port}/v1"
    # 15 min: cold-cache model load + Metal kernel compile + drafter load
    deadline = time.time() + 900
    try:
        while time.time() < deadline:
            if proc.poll() is not None:
                logger.error("  server exited; tail of %s:", log_path)
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
                    return ServerHandle(
                        proc=proc, base_url=base_url, model=model, logf=logf
                    )
            except Exception:
                pass
            time.sleep(2)

        proc.kill()
        raise RuntimeError(f"server did not become ready within deadline (port={port})")
    except BaseException:
        logf.close()
        raise


# ---- Workload execution ---------------------------------------------------

# Same reliability gates as bench_suffix_decoding_integrated.py.
MIN_DECODE_TIME = 0.5
TPS_CEILING = 500.0


@dataclass
class WorkloadRun:
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
    payload = {
        "model": handle.model,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        **workload_body,
        # Greedy: DFlash drafter targets are exact-match; sampling temp
        # would silently lower acceptance and the bench would understate
        # speedup. Same contract as the suffix-decoding bench.
        "temperature": 0.0,
        # Disable thinking mode (Qwen3/Gemma-4 reasoning-capable). With
        # thinking on, the model emits 1500+ reasoning_content tokens
        # before the first content delta — TTFT below would land at
        # end-of-reasoning, decode_time computes to ~0.2s, and the
        # MIN_DECODE_TIME gate rejects every reasoning-heavy workload.
        # z-lab's canonical demo (scripts/demo_dflash.py) uses /no_think
        # in the prompt for the same reason. The kwarg path is more
        # robust since not every model honors the /no_think sentinel.
        "enable_thinking": False,
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
        # Surface 4xx/5xx as a proper error rather than treating the
        # error body as a stream of decoded events (would silently
        # produce a 0-token run otherwise).
        r.raise_for_status()
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
            if ttft is None:
                choices = obj.get("choices") or []
                # Defense in depth: count first reasoning_content as the
                # decode start too, in case a thinking-disabled-by-kwarg
                # request still emits some <think>…</think> tokens.
                # Without this, decode_time would only count the visible
                # content phase, understating real decode work.
                if choices and (
                    choices[0].get("delta", {}).get("content")
                    or choices[0].get("delta", {}).get("reasoning_content")
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
    median_tps: dict[str, float | None]
    raw_runs: dict[str, list[WorkloadRun]]


def bench_one_mode(
    model: str,
    port: int,
    dflash: bool,
    runs: int,
    max_tokens: int,
) -> ModeResult:
    handle = start_server(model, port, dflash)
    try:
        # Discard warmup: Metal JIT + cache population + drafter load.
        run_workload(handle, WORKLOADS["fibonacci"], max_tokens=32)
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
                    "DFlash" if dflash else "base  ",
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
        description="Bench DFlash speedup for one alias (Model Onboarding SOP §6)."
    )
    parser.add_argument("--model", required=True, help="HF repo or alias")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--runs", type=int, default=3, help="runs per workload")
    parser.add_argument("--port", type=int, default=8765, help="ephemeral port")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write JSON result file (defaults to evals/results/dflash_<model>.json)",
    )
    parser.add_argument(
        "--gate",
        type=float,
        default=1.30,
        help="Code-median speedup gate for ship/no-ship (default 1.30)",
    )
    parser.add_argument(
        "--non-code-floor",
        type=float,
        default=1.00,
        help=(
            "Per-workload floor for non-code workloads (chat etc.) — any "
            "regression below this fails the SHIP gate (default 1.00)"
        ),
    )
    args = parser.parse_args(argv)

    logger.info("Bench DFlash: %s", args.model)
    if os.environ.get("RAPID_MLX_DFLASH_BYPASS_MOE_GATE") == "1":
        logger.warning(
            "  RAPID_MLX_DFLASH_BYPASS_MOE_GATE=1 — eligibility MoE gate "
            "bypassed (PoC mode)"
        )

    logger.info("--- baseline (autoregressive) ---")
    base = bench_one_mode(
        args.model,
        args.port,
        dflash=False,
        runs=args.runs,
        max_tokens=args.max_tokens,
    )

    logger.info("--- DFlash ---")
    dflash = bench_one_mode(
        args.model,
        args.port,
        dflash=True,
        runs=args.runs,
        max_tokens=args.max_tokens,
    )

    speedup: dict[str, float] = {}
    skipped: dict[str, str] = {}
    for name in WORKLOADS:
        v = base.median_tps.get(name)
        s = dflash.median_tps.get(name)
        if v is None or s is None or v <= 0:
            reason = (
                "baseline all runs rejected"
                if v is None or v <= 0
                else "dflash all runs rejected"
            )
            skipped[name] = reason
            continue
        speedup[name] = round(s / v, 3)

    # Code median = the four code-gen workloads only. Chat is graded
    # separately as a non-regression floor — prior bench rounds showed
    # DFlash speedup is heavily code-biased and SHIPping a 1.3x code
    # model that regresses to 0.8x on chat would be a bad user trade.
    code_speedups = [v for k, v in speedup.items() if k in _CODE_WORKLOADS]
    code_median = median(code_speedups) if code_speedups else None
    non_code_speedups = {k: v for k, v in speedup.items() if k not in _CODE_WORKLOADS}
    non_code_floor = args.non_code_floor
    non_code_regress = {
        k: v for k, v in non_code_speedups.items() if v < non_code_floor
    }

    median_speedup = median(speedup.values()) if speedup else None

    # Boolean ship/no-ship is the load-bearing signal; the human-readable
    # ``decision`` string is just for the scorecard. Keeping them separate
    # avoids string-shape coupling in callers / exit codes.
    if code_median is None:
        ship = False
        decision = "DO NOT SHIP (no valid code workloads)"
    elif code_median < args.gate:
        ship = False
        decision = "DO NOT SHIP"
    elif non_code_regress:
        ship = False
        regressed = ", ".join(f"{k} {v:.2f}x" for k, v in non_code_regress.items())
        decision = f"DO NOT SHIP (non-code regression: {regressed})"
    else:
        ship = True
        decision = "SHIP (supports_dflash=true)"

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
        "gate": args.gate,
        "non_code_floor": non_code_floor,
        "base_tps": {
            k: (round(v, 1) if v is not None else None)
            for k, v in base.median_tps.items()
        },
        "dflash_tps": {
            k: (round(v, 1) if v is not None else None)
            for k, v in dflash.median_tps.items()
        },
        "speedup": speedup,
        "code_median": round(code_median, 3) if code_median is not None else None,
        "non_code_speedups": non_code_speedups,
        "median_speedup": round(median_speedup, 3)
        if median_speedup is not None
        else None,
        "skipped": skipped,
        "decision": decision,
        "bypass_moe_gate": os.environ.get("RAPID_MLX_DFLASH_BYPASS_MOE_GATE") == "1",
        "raw_runs": {
            "base": _serialize_runs(base.raw_runs),
            "dflash": _serialize_runs(dflash.raw_runs),
        },
    }

    output = args.output or REPO_ROOT / "evals/results" / (
        f"dflash_{args.model.replace('/', '_').lower()}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n")

    logger.info("--- summary ---")
    for k in WORKLOADS:
        if k in skipped:
            logger.info("  %-12s SKIPPED (%s)", k, skipped[k])
        else:
            v = speedup[k]
            bucket = "code" if k in _CODE_WORKLOADS else "non-code"
            threshold = args.gate if bucket == "code" else non_code_floor
            marker = "↑" if v >= threshold else ("↓" if v < non_code_floor else "·")
            logger.info("  %-12s %5.2fx %s  [%s]", k, v, marker, bucket)
    if code_median is not None:
        logger.info("  code median: %.2fx (gate %.2fx)", code_median, args.gate)
    if non_code_speedups:
        non_code_summary = ", ".join(
            f"{k} {v:.2f}x" for k, v in non_code_speedups.items()
        )
        logger.info("  non-code: %s (floor %.2fx)", non_code_summary, non_code_floor)
    logger.info("  decision: %s", decision)
    logger.info("  written: %s", output)

    return 0 if ship else 1


if __name__ == "__main__":
    sys.exit(main())
