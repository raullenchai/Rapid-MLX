#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare single-process and supervised serve on the same Metal machine.

Run manually: python tools/bench_control_plane.py --model qwen3.5-4b-4bit --port 8123
Weights must already be cached. No invocation of this script is part of unit tests.
The long prompt is warmed once per mode, then repeated to measure cache hits.
"""

import argparse
import concurrent.futures
import json
import os
import signal
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

import psutil


def http_json(url, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    return urllib.request.urlopen(req, timeout=600)


def generate(url, model, prompt, tokens):
    start = time.perf_counter()
    first = None
    completion_tokens = None
    with http_json(
        url + "/v1/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": tokens,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    ) as response:
        for line in response:
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            event = json.loads(line[6:])
            for choice in event.get("choices", []):
                delta = choice.get("delta", {})
                if first is None and (
                    delta.get("content") or delta.get("reasoning_content")
                ):
                    first = time.perf_counter()
            if event.get("usage"):
                completion_tokens = event["usage"]["completion_tokens"]
    end = time.perf_counter()
    if first is None or completion_tokens is None:
        raise RuntimeError(
            "Server did not report streamed content and completion usage"
        )
    return {
        "ttft_ms": (first - start) * 1000,
        "tokens": completion_tokens,
        "decode_tps": max(0, completion_tokens - 1) / max(end - first, 1e-9),
    }


def summary(samples):
    ordered = sorted(s["ttft_ms"] for s in samples)
    return {
        "ttft_median_ms": statistics.median(ordered),
        "ttft_p95_ms": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
        "decode_tps": statistics.median(s["decode_tps"] for s in samples),
    }


def measure(args, supervised):
    env = os.environ.copy()
    for key in (
        "RAPID_MLX_CONTROL_RUNNER",
        "RAPID_MLX_CONTROL_SOCKET",
        "RAPID_MLX_CONTROL_SECRET",
        "RAPID_MLX_WATCHDOG_PPID",
        "RAPID_MLX_SINGLE_PROCESS",
    ):
        env.pop(key, None)
    if not supervised:
        env["RAPID_MLX_SINGLE_PROCESS"] = "1"
    env["RAPID_MLX_DISABLE_VERSION_CHECK"] = "1"
    env["RAPID_MLX_DISABLE_MODEL_DOWNLOADS"] = "1"
    url = f"http://127.0.0.1:{args.port}"
    with (
        tempfile.TemporaryDirectory(
            prefix="rapid-mlx-serve-cp-benchmark-", dir="/private/tmp"
        ) as scratch,
        open(Path(scratch) / "serve.log", "wb") as log,
    ):
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "rapid_mlx.cli",
                "--no-telemetry",
                "--no-banner",
                "serve",
                args.model,
                "--port",
                str(args.port),
                "--disable-model-downloads",
            ],
            env=env,
            stdout=log,
            stderr=log,
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + args.startup_timeout
            while True:
                if proc.poll() is not None:
                    raise RuntimeError(
                        f"serve exited: {(Path(scratch) / 'serve.log').read_text(errors='replace')[-4000:]}"
                    )
                try:
                    with http_json(url + "/healthz") as response:
                        if response.status == 200:
                            break
                except (OSError, urllib.error.URLError):
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError("serve readiness deadline exceeded")
                time.sleep(0.25)
            if supervised:
                with http_json(url + "/rapid/v1/status") as response:
                    status = json.load(response)
                if status["runner"]["pid"] == proc.pid:
                    raise RuntimeError("Expected a supervised runner")
            model = args.model
            with http_json(url + "/v1/models") as response:
                model = json.load(response)["data"][0]["id"]
            short = "Explain why the sky is blue in a few sentences."
            long = (
                "Context:\n"
                + "The river flows past the quiet village. " * args.context_repeats
                + "\nSummarize this context."
            )
            workloads = {}
            for label, prompt in (("short", short), ("long_cache_hit", long)):
                generate(url, model, prompt, args.tokens)  # warmup/cache seed
                samples = [
                    generate(url, model, prompt, args.tokens)
                    for _ in range(args.repetitions)
                ]
                workloads[label] = {**summary(samples), "samples": samples}
            start = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                samples = list(
                    pool.map(
                        lambda _: generate(url, model, short, args.tokens), range(8)
                    )
                )
            throughput = sum(s["tokens"] for s in samples) / (
                time.perf_counter() - start
            )
            # The parent is idle while the runner does inference. Measure it
            # after workloads; single-process numbers describe the server.
            process = psutil.Process(proc.pid)
            cpu = process.cpu_percent(interval=2)
            rss = process.memory_info().rss / (1024 * 1024)
            return {
                "workloads": workloads,
                "concurrency8_tps": throughput,
                "idle_cpu_percent": cpu,
                "idle_rss_mb": rss,
                "idle_process": "control" if supervised else "single-process server",
            }
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=6)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--port", type=int, default=8123)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--context-repeats", type=int, default=1024)
    parser.add_argument("--startup-timeout", type=float, default=600)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    if args.repetitions < 1 or args.tokens < 2 or args.context_repeats < 1:
        parser.error("repetitions/context-repeats must be positive and tokens >= 2")
    results = {
        "model": args.model,
        "parameters": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "single_process": measure(args, False),
        "supervised": measure(args, True),
    }
    print(f"{'Metric':32} {'single':>14} {'supervised':>14}")
    rows = []
    for workload in ("short", "long_cache_hit"):
        for metric in ("ttft_median_ms", "ttft_p95_ms", "decode_tps"):
            rows.append(
                (
                    f"{workload}/{metric}",
                    *(
                        results[mode]["workloads"][workload][metric]
                        for mode in ("single_process", "supervised")
                    ),
                )
            )
    for metric in ("concurrency8_tps", "idle_cpu_percent", "idle_rss_mb"):
        rows.append(
            (
                metric,
                *(results[mode][metric] for mode in ("single_process", "supervised")),
            )
        )
    for label, single, supervised in rows:
        print(f"{label:32} {single:14.3f} {supervised:14.3f}")
    output = json.dumps(results, indent=2)
    print(output)
    if args.json_output:
        args.json_output.write_text(output + "\n")


if __name__ == "__main__":
    main()
