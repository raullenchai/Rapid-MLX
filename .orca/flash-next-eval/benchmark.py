#!/usr/bin/env python3
"""Reproducible B=1 Flash-Next launch benchmark against a running server.

Prompt construction and SSE accounting reuse ``scripts/bench_service_prefill.py``:
exact tokenizer-counted prompts, first-visible-token TTFT, and server-reported
token counts. Each context length is cold-cache, 256 decode tokens, three runs.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import statistics
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))

import httpx  # noqa: E402
import psutil  # noqa: E402

from scripts.bench_service_prefill import (  # noqa: E402
    clear_prefix_cache,
    make_messages,
    stream_request,
)


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def process_tree_rss(pid: int) -> int:
    root = psutil.Process(pid)
    processes = [root, *root.children(recursive=True)]
    total = 0
    for process in processes:
        try:
            total += process.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return total


class RssSampler:
    def __init__(self, pid: int, interval: float = 0.05):
        self.pid = pid
        self.interval = interval
        self.samples: list[int] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.samples.append(process_tree_rss(self.pid))
            self._stop.wait(self.interval)

    def __enter__(self):
        self.samples.append(process_tree_rss(self.pid))
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        self.samples.append(process_tree_rss(self.pid))

    @property
    def peak(self) -> int:
        return max(self.samples)


def median(values: list[float]) -> float:
    return round(float(statistics.median(values)), 3)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "prompt_tokens": int(statistics.median(row["prompt_tokens"] for row in rows)),
        "completion_tokens": int(
            statistics.median(row["completion_tokens"] for row in rows)
        ),
        "ttft_ms_median": median([row["ttft_ms"] for row in rows]),
        "prefill_tokens_per_second_median": median(
            [row["prefill_tokens_per_second"] for row in rows]
        ),
        "decode_tokens_per_second_median": median(
            [row["decode_tokens_per_second"] for row in rows]
        ),
        "peak_rss_gib_max": round(
            max(row["peak_rss_bytes"] for row in rows) / 2**30, 3
        ),
        "steady_rss_gib_median": round(
            statistics.median(row["steady_rss_bytes"] for row in rows) / 2**30,
            3,
        ),
        "runs": rows,
    }


def detect_model(client: httpx.Client, api_url: str) -> str:
    response = client.get(f"{api_url}/models")
    response.raise_for_status()
    models = response.json().get("data") or []
    if not models:
        raise RuntimeError("/v1/models returned no models")
    return str(models[0]["id"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model")
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--server-pid", type=int, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--decode-tokens", type=int, default=256)
    parser.add_argument("--prompt-tokens", default="128,2048,8192,32768")
    parser.add_argument("--load-time-seconds", type=float)
    parser.add_argument("--artifact-revision")
    parser.add_argument("--rapid-sha", required=True)
    parser.add_argument("--output", type=Path, default=HERE / "benchmark-results.json")
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    lengths = [int(value) for value in args.prompt_tokens.split(",")]
    if lengths != [128, 2048, 8192, 32768]:
        raise ValueError(
            "launch methodology requires prompt lengths 128,2048,8192,32768"
        )
    if args.runs != 3 or args.decode_tokens != 256:
        raise ValueError("launch methodology requires 3 runs and 256 decode tokens")
    if args.dry_run:
        print("READY: 4 prompt lengths x 3 cold-cache runs x 256 decode tokens")
        return 0

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path, local_files_only=True, trust_remote_code=True
    )
    api_url = args.url.rstrip("/")
    root_url = api_url.removesuffix("/v1")
    client = httpx.Client(timeout=args.timeout)
    model = args.model or detect_model(client, api_url)
    before_rss = process_tree_rss(args.server_pid)

    results: dict[str, Any] = {}
    for target in lengths:
        messages = make_messages(
            tokenizer,
            target,
            suffix="\nReply with a detailed but direct explanation.",
        )
        rows = []
        for run_index in range(1, args.runs + 1):
            clear_prefix_cache(client, root_url)
            time.sleep(0.25)
            steady = process_tree_rss(args.server_pid)
            with RssSampler(args.server_pid) as rss:
                row = stream_request(
                    client,
                    api_url,
                    model,
                    messages,
                    args.decode_tokens,
                )
            ttft_seconds = row["ttft_ms"] / 1000.0
            decode_seconds = max((row["total_ms"] - row["ttft_ms"]) / 1000.0, 0.0)
            row.update(
                {
                    "run": run_index,
                    "target_prompt_tokens": target,
                    "prefill_tokens_per_second": round(
                        row["prompt_tokens"] / ttft_seconds, 3
                    ),
                    "decode_tokens_per_second": round(
                        row["completion_tokens"] / decode_seconds, 3
                    )
                    if decode_seconds and row["completion_tokens"]
                    else 0.0,
                    "steady_rss_bytes": steady,
                    "peak_rss_bytes": rss.peak,
                }
            )
            rows.append(row)
            print(
                f"{target:5d} run {run_index}: TTFT={row['ttft_ms']:.1f} ms "
                f"prefill={row['prefill_tokens_per_second']:.2f} tok/s "
                f"decode={row['decode_tokens_per_second']:.2f} tok/s "
                f"peak={row['peak_rss_bytes'] / 2**30:.2f} GiB"
            )
        results[str(target)] = summarize(rows)

    payload = {
        "schema_version": 1,
        "methodology_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "label": args.label,
        "model": model,
        "environment": {
            "chip": platform.processor(),
            "machine": platform.machine(),
            "physical_memory_gib": round(psutil.virtual_memory().total / 2**30, 2),
            "macos": platform.mac_ver()[0],
            "python": platform.python_version(),
            "rapid_mlx_sha": args.rapid_sha,
            "rapid_mlx_version": package_version("rapid-mlx"),
            "mlx_version": package_version("mlx"),
            "mlx_lm_version": package_version("mlx-lm"),
            "mlx_vlm_version": package_version("mlx-vlm"),
            "artifact_revision": args.artifact_revision,
            "quantization": "PLE q4-g32; routing gates q8-g64; remainder q4-g64",
        },
        "method": {
            "batch_size": 1,
            "prompt_tokens": lengths,
            "decode_tokens": args.decode_tokens,
            "runs": args.runs,
            "aggregate": "median except peak RSS=max",
            "cache": "cleared before every timed run",
            "ttft": "first visible SSE content/reasoning/tool delta",
            "token_counts": "server-reported usage",
            "prefill_rate": "prompt_tokens / TTFT; includes request and first-token overhead",
            "rss": "server process plus recursive children, sampled every 50 ms",
        },
        "load_time_seconds": args.load_time_seconds,
        "initial_steady_rss_gib": round(before_rss / 2**30, 3),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"RESULTS {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
