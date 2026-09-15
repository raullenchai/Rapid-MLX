#!/usr/bin/env python3
"""DeepSeek V4 prefix-cache persistence benchmark across server processes.

Starts Rapid-MLX twice with an isolated HOME.  Process A serves a deterministic
long prompt and saves its prefix cache during graceful shutdown.  Process B
loads that cache, serves the identical prompt, and verifies a real cache hit,
byte-identical visible/reasoning output, and warm-vs-cold TTFT.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

try:
    from scripts.bench_metadata import format_bench_json
except ModuleNotFoundError:  # Direct execution outside the repository root.
    from bench_metadata import format_bench_json


def _get_json(url: str, timeout: float = 2.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def _assert_port_available(port: int) -> None:
    """Reject a benchmark port already owned by an unrelated server."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", port))
        except OSError as exc:
            raise RuntimeError(f"benchmark port {port} is already in use") from exc


def _wait_ready(base_url: str, process: subprocess.Popen, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"server exited before readiness (rc={process.returncode})"
            )
        try:
            models = _get_json(f"{base_url}/v1/models").get("data") or []
            if any(model.get("id") == "deepseek-restart-bench" for model in models):
                return
        except (OSError, urllib.error.URLError, json.JSONDecodeError):
            pass
        time.sleep(1)
    raise TimeoutError(f"server was not ready after {timeout:.0f}s")


def _stream_request(base_url: str, context: str, max_tokens: int) -> dict:
    payload = json.dumps(
        {
            "model": "deepseek-restart-bench",
            "messages": [
                {
                    "role": "system",
                    "content": context,
                },
                {"role": "user", "content": "Return exactly: CACHE_RESTART_OK"},
            ],
            "temperature": 0,
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False},
        }
    ).encode()
    request = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    first_token = None
    content: list[str] = []
    reasoning: list[str] = []
    usage: dict = {}
    with urllib.request.urlopen(request, timeout=600) as response:
        for raw in response:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            usage = event.get("usage") or usage
            choices = event.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            visible = delta.get("content") or ""
            thought = delta.get("reasoning") or delta.get("reasoning_content") or ""
            if (visible or thought) and first_token is None:
                first_token = time.perf_counter()
            content.append(visible)
            reasoning.append(thought)
    finished = time.perf_counter()
    details = usage.get("prompt_tokens_details") or {}
    return {
        "ttft_s": (first_token - started) if first_token is not None else None,
        "elapsed_s": finished - started,
        "content": "".join(content),
        "reasoning": "".join(reasoning),
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "cached_tokens": int(details.get("cached_tokens") or 0),
    }


def _start_server(args, isolated_home: str, log_path: Path) -> subprocess.Popen:
    env = dict(os.environ)
    env["HOME"] = isolated_home
    env["RAPID_MLX_PREFIX_CACHE_SHUTDOWN_BUDGET"] = str(args.shutdown_budget)
    command = [
        sys.executable,
        "-m",
        "rapid_mlx.cli",
        "serve",
        args.model,
        "--served-model-name",
        "deepseek-restart-bench",
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--enable-prefix-cache",
        "--hybrid-cache-entries",
        str(args.hybrid_cache_entries),
        "--reasoning-parser",
        "deepseek_v4",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "deepseek_v4_0731",
        "--no-spec-decode",
        "--max-num-seqs",
        "1",
        "--log-level",
        "INFO",
    ]
    log = log_path.open("ab", buffering=0)
    process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    log.close()
    return process


def _stop_server(process: subprocess.Popen, timeout: float) -> None:
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=30)
        raise TimeoutError("server exceeded graceful-shutdown/cache-save timeout")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Local DeepSeek V4 MLX model directory")
    parser.add_argument("--port", type=int, default=18081)
    parser.add_argument("--ready-timeout", type=float, default=600)
    parser.add_argument("--shutdown-budget", type=float, default=120)
    parser.add_argument("--hybrid-cache-entries", type=int, default=8)
    parser.add_argument("--prompt-repetitions", type=int, default=1200)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--min-cache-ratio", type=float, default=0.9)
    parser.add_argument("--min-ttft-speedup", type=float, default=1.1)
    parser.add_argument("--work-dir", help="Keep logs/cache here instead of a temp dir")
    args = parser.parse_args()

    model = Path(args.model).expanduser().resolve()
    if not model.is_dir():
        parser.error(f"model directory does not exist: {model}")
    args.model = str(model)
    auto_work_dir = args.work_dir is None
    work = Path(args.work_dir).resolve() if args.work_dir else Path(tempfile.mkdtemp())
    # A unique HOME guarantees the cold cycle cannot inherit a prefix cache
    # from an earlier run, even when the caller intentionally reuses work-dir.
    home = work / f"home-{uuid.uuid4().hex}"
    home.mkdir(parents=True, exist_ok=True)
    base_url = f"http://127.0.0.1:{args.port}"
    context = (
        "Review this invariant carefully: cached state must match the exact token "
        "prefix, survive restart, and never alter the generated answer. "
        * args.prompt_repetitions
    )

    results = []
    succeeded = False
    try:
        for cycle in ("cold", "restart"):
            log_path = work / f"server-{cycle}.log"
            _assert_port_available(args.port)
            process = _start_server(args, str(home), log_path)
            try:
                _wait_ready(base_url, process, args.ready_timeout)
                result = _stream_request(base_url, context, args.max_tokens)
                result["cycle"] = cycle
                results.append(result)
            finally:
                _stop_server(process, args.shutdown_budget + 30)
        cold, restart = results
        identical = (cold["content"], cold["reasoning"]) == (
            restart["content"],
            restart["reasoning"],
        )
        cache_ratio = restart["cached_tokens"] / max(restart["prompt_tokens"], 1)
        hit = cache_ratio >= args.min_cache_ratio
        speedup = (
            cold["ttft_s"] / restart["ttft_s"]
            if cold["ttft_s"] and restart["ttft_s"]
            else None
        )
        expected_output = (
            cold["content"].strip() == "CACHE_RESTART_OK"
            and restart["content"].strip() == "CACHE_RESTART_OK"
        )
        no_reasoning_leak = not cold["reasoning"] and not restart["reasoning"]
        ttft_verified = speedup is not None and speedup >= args.min_ttft_speedup
        succeeded = (
            identical
            and expected_output
            and no_reasoning_leak
            and hit
            and ttft_verified
        )
        summary = {
            "cold": cold,
            "restart": restart,
            "output_identical": identical,
            "expected_output": expected_output,
            "no_reasoning_leak": no_reasoning_leak,
            "restart_cache_hit": hit,
            "restart_cache_ratio": cache_ratio,
            "ttft_speedup": speedup,
            "ttft_verified": ttft_verified,
            "work_dir": None if auto_work_dir else str(work),
        }
        print(format_bench_json(summary, __file__, sort_keys=True))
        return 0 if succeeded else 1
    finally:
        if auto_work_dir:
            shutil.rmtree(work, ignore_errors=True)
        elif not succeeded:
            print(f"benchmark artifacts retained at {work}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
