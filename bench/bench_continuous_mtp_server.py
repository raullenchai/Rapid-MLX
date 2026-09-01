#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Measure one already-running legacy or continuous MTP server.

Start the target server with prefix caching disabled, then run this client once
for the legacy MTP configuration and once for continuous MTP.  The emitted JSON
contains per-request output hashes so the two files can be compared directly.
Hash differences are evidence to inspect with the task-level correctness
battery, not an automatic failure: greedy batching can choose a different but
equally valid continuation while preserving every task contract.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import platform
import statistics
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any

_CONTEXT_BLOCK = (
    "We are designing a Python job scheduler. Each Job has id, priority, "
    "created_at, payload, and optional deadline. Workers poll for jobs. The "
    "scheduler must be thread-safe, preserve FIFO order within equal priority, "
    "reject duplicate ids, support cancellation before dispatch, and never "
    "lose a job when a worker raises. A lease lasts 30 seconds and can be "
    "renewed. Expired leases return to the queue with their original ordering "
    "metadata. State persistence uses a transaction interface with begin, "
    "commit, and rollback. Metrics include queued, leased, completed, "
    "cancelled, retry_count, and lease_expirations. Shutdown stops new "
    "submissions but permits active leases to finish for up to 10 seconds. All "
    "public methods require type hints. Errors use DuplicateJobError, "
    "UnknownJobError, InvalidStateError, and SchedulerClosedError. Tests use a "
    "fake clock and deterministic ids. Do not use global state. "
)
_PROMPT = (_CONTEXT_BLOCK * 3) + (
    "\nWrite the core Python implementation. Return code only, around 120 lines."
)


def _prompt_for_lane(lane: int) -> str:
    """Return a deterministic, lane-distinct routing oracle prompt."""
    return (
        f"{_PROMPT}\nThe first output line must be exactly "
        f"# QUALIFICATION_LANE_{lane}. Then name the public scheduler class "
        f"Lane{lane}Scheduler and include a class constant LANE_ID = {lane}."
    )


@dataclass(frozen=True)
class RequestResult:
    run: int
    lane: int
    elapsed_seconds: float
    prompt_tokens: int
    completion_tokens: int
    output_sha256: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True, help="Result label, e.g. legacy")
    parser.add_argument("--model", required=True, help="Exact served model id/path")
    parser.add_argument("--base-url", default="http://127.0.0.1:8475/v1")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--baseline-json",
        type=Path,
        help="Ordinary-MTP report to compare against lane by lane",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for name in ("runs", "concurrency", "max_tokens"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    return args


def _request(args: argparse.Namespace, run: int, lane: int) -> RequestResult:
    prompt = _prompt_for_lane(lane)
    payload: dict[str, Any] = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        args.base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=args.timeout) as response:
        body = json.load(response)
    elapsed = time.perf_counter() - started
    content = body["choices"][0]["message"]["content"]
    usage = body["usage"]
    return RequestResult(
        run=run,
        lane=lane,
        elapsed_seconds=elapsed,
        prompt_tokens=int(usage["prompt_tokens"]),
        completion_tokens=int(usage["completion_tokens"]),
        output_sha256=hashlib.sha256(content.encode()).hexdigest(),
    )


def main() -> int:
    args = _parse_args()
    if args.dry_run:
        print(
            json.dumps(
                {
                    "label": args.label,
                    "model": args.model,
                    "base_url": args.base_url,
                    "runs": args.runs,
                    "concurrency": args.concurrency,
                    "max_tokens": args.max_tokens,
                    "planned_requests": args.runs * args.concurrency,
                    "lane_prompt_sha256": {
                        str(lane): hashlib.sha256(
                            _prompt_for_lane(lane).encode()
                        ).hexdigest()
                        for lane in range(args.concurrency)
                    },
                },
                indent=2,
            )
        )
        return 0

    results: list[RequestResult] = []
    cohorts: list[dict[str, float | int]] = []
    for run in range(1, args.runs + 1):
        started = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.concurrency
        ) as executor:
            rows = list(
                executor.map(
                    partial(_request, args, run),
                    range(args.concurrency),
                )
            )
        wall = time.perf_counter() - started
        results.extend(rows)
        tokens = sum(row.completion_tokens for row in rows)
        cohorts.append(
            {
                "run": run,
                "wall_seconds": wall,
                "aggregate_decode_tokens_per_second": tokens / wall,
            }
        )

    expected = args.runs * args.concurrency
    complete = len(results) == expected and all(
        row.completion_tokens == args.max_tokens for row in results
    )
    hashes_by_lane = {
        str(lane): sorted({row.output_sha256 for row in results if row.lane == lane})
        for lane in range(args.concurrency)
    }
    deterministic = all(len(hashes) == 1 for hashes in hashes_by_lane.values())
    output_sha256_by_lane = {
        lane: hashes[0] for lane, hashes in hashes_by_lane.items() if len(hashes) == 1
    }
    paired_output_identical: bool | None = None
    baseline_sha256_by_lane: dict[str, str] | None = None
    if args.baseline_json is not None:
        with args.baseline_json.open(encoding="utf-8") as handle:
            baseline = json.load(handle)
        baseline_sha256_by_lane = baseline.get("output_sha256_by_lane")
        if not isinstance(baseline_sha256_by_lane, dict):
            raise ValueError(
                f"{args.baseline_json} has no output_sha256_by_lane mapping"
            )
        paired_output_identical = (
            deterministic and output_sha256_by_lane == baseline_sha256_by_lane
        )
    report = {
        "label": args.label,
        "model": args.model,
        "base_url": args.base_url,
        "methodology": {
            "runs": args.runs,
            "concurrency": args.concurrency,
            "max_tokens": args.max_tokens,
            "temperature": 0,
            "thinking": False,
            "prefix_cache_expected": False,
        },
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "complete": complete,
        "output_sha256_by_lane": output_sha256_by_lane,
        "deterministic_within_lane": deterministic,
        "baseline_sha256_by_lane": baseline_sha256_by_lane,
        "paired_output_identical": paired_output_identical,
        "median_wall_seconds": statistics.median(
            row["wall_seconds"] for row in cohorts
        ),
        "median_aggregate_decode_tokens_per_second": statistics.median(
            row["aggregate_decode_tokens_per_second"] for row in cohorts
        ),
        "cohorts": cohorts,
        "requests": [asdict(row) for row in results],
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if complete and deterministic else 1


if __name__ == "__main__":
    sys.exit(main())
