#!/usr/bin/env python3.12
# SPDX-License-Identifier: Apache-2.0
"""B=1 engine_loop timing probe harness.

Boots a rapid-mlx server with RAPID_MLX_PROBE_ENGINE_LOOP=1 set, fires
one chat completion sized to amortize warmup, then parses the
[engine_loop_probe] log lines emitted from engine_core._engine_loop
and prints a single summary line.

Goal: confirm where the rapid-mlx HTTP B=1 ~33% gap to mlx-vlm lives.
Hypothesis: ``step_ms_avg`` matches the stock-mlx-lm BatchGenerator
budget and ``dispatch_ms_avg`` is the missing ~5 ms — i.e. the gap is
in the asyncio dispatch around scheduler.step(), not in the engine.

Usage::

    python3.12 scripts/probe_engine_loop.py --alias qwen3.5-4b
    python3.12 scripts/probe_engine_loop.py --alias qwen3.6-27b-8bit --max-tokens 256
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

PROBE_LINE_RE = re.compile(
    r"\[engine_loop_probe\] "
    r"steps=(?P<steps>\d+) tokens=(?P<tokens>\d+) "
    r"step_inside_ms_avg=(?P<inside_avg>[\d.]+) "
    r"step_inside_ms_min=(?P<inside_min>[\d.]+) "
    r"step_inside_ms_max=(?P<inside_max>[\d.]+) "
    r"step_await_ms_avg=(?P<await_avg>[\d.]+) "
    r"step_await_ms_min=(?P<await_min>[\d.]+) "
    r"step_await_ms_max=(?P<await_max>[\d.]+) "
    r"dispatch_ms_avg=(?P<disp_avg>[\d.]+) "
    r"dispatch_ms_min=(?P<disp_min>[\d.]+) "
    r"dispatch_ms_max=(?P<disp_max>[\d.]+) "
    r"tok_per_s=(?P<tps>[\d.]+)"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--alias", default="qwen3.5-4b")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument(
        "--prompt",
        default="Write a long, detailed story about a cheetah running across the savannah at sunset. Describe the scenery, the wind, the prey, every step in detail.",
    )
    p.add_argument("--max-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--ready-timeout", type=float, default=120.0)
    p.add_argument("--log-every", type=int, default=32)
    return p.parse_args()


def wait_for_ready(port: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/v1/models", timeout=1.0
            ) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(0.5)
    return False


def fire_chat(port: int, alias: str, prompt: str, max_tokens: int, temp: float) -> dict:
    body = json.dumps(
        {
            "model": alias,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temp,
            "stream": True,
        }
    ).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=body,
        headers={"content-type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    chunks = 0
    with urllib.request.urlopen(req, timeout=300) as r:
        for line in r:
            if line.startswith(b"data: "):
                chunks += 1
    return {"chunks": chunks, "wall_s": time.perf_counter() - t0}


def main() -> int:
    args = parse_args()
    env = dict(os.environ)
    env["RAPID_MLX_PROBE_ENGINE_LOOP"] = "1"
    env["RAPID_MLX_PROBE_LOG_EVERY"] = str(args.log_every)
    # Force editable-install resolution so we exercise THIS branch's
    # engine_core, not the brew-installed rapid-mlx.
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [
        sys.executable,
        "-m",
        "vllm_mlx.cli",
        "serve",
        args.alias,
        "--port",
        str(args.port),
        "--host",
        "127.0.0.1",
    ]
    print(f"[probe] launching: {' '.join(cmd)}", flush=True)
    log_path = f"/tmp/probe_engine_loop_{args.alias}.log"
    log_fp = open(log_path, "w")
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=log_fp,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    try:
        print(f"[probe] waiting for /v1/models on :{args.port} ...", flush=True)
        if not wait_for_ready(args.port, args.ready_timeout):
            print(f"[probe] server did not become ready within {args.ready_timeout}s")
            print(f"[probe] last 40 lines of {log_path}:")
            log_fp.flush()
            with open(log_path) as f:
                lines = f.readlines()
            for line in lines[-40:]:
                print("  " + line.rstrip())
            return 2
        print("[probe] server ready, firing request", flush=True)
        # Tiny warmup so the very first request's prefill cost doesn't
        # dominate the per-step measurements.
        fire_chat(args.port, args.alias, "Say hi.", 16, args.temperature)
        time.sleep(0.5)
        result = fire_chat(
            args.port, args.alias, args.prompt, args.max_tokens, args.temperature
        )
        print(
            f"[probe] request done: chunks={result['chunks']} wall={result['wall_s']:.2f}s "
            f"end_to_end_tok_per_s={result['chunks'] / result['wall_s']:.2f}",
            flush=True,
        )
        # Give the engine a moment to flush its last probe log line.
        time.sleep(0.5)
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        log_fp.close()

    # Parse probe lines
    samples: list[dict[str, float | int]] = []
    with open(log_path) as f:
        for raw in f:
            m = PROBE_LINE_RE.search(raw)
            if not m:
                continue
            samples.append({k: float(v) for k, v in m.groupdict().items()})
    if not samples:
        print(f"[probe] no [engine_loop_probe] lines found in {log_path}")
        return 3

    # Drop the first sample window — it usually captures prefill, not decode.
    decode_samples = samples[1:] if len(samples) > 1 else samples
    inside_avgs = [s["inside_avg"] for s in decode_samples]
    await_avgs = [s["await_avg"] for s in decode_samples]
    disp_avgs = [s["disp_avg"] for s in decode_samples]
    total_tokens = sum(int(s["tokens"]) for s in decode_samples)
    total_steps = sum(int(s["steps"]) for s in decode_samples)
    weighted_inside_ms = sum(
        s["inside_avg"] * s["steps"] for s in decode_samples
    ) / max(1, total_steps)
    weighted_await_ms = sum(s["await_avg"] * s["steps"] for s in decode_samples) / max(
        1, total_steps
    )
    weighted_disp_ms = sum(s["disp_avg"] * s["steps"] for s in decode_samples) / max(
        1, total_steps
    )
    rtt_ms = max(0.0, weighted_await_ms - weighted_inside_ms)

    print()
    print("=" * 72)
    print(
        f"PROBE SUMMARY ({args.alias}, B=1, {total_tokens} tokens over {total_steps} steps)"
    )
    print("=" * 72)
    print(
        f"  step_inside_ms   : avg={weighted_inside_ms:.3f}  "
        f"min={min(inside_avgs):.3f}  max={max(inside_avgs):.3f}   "
        f"<- pure scheduler.step on the executor thread"
    )
    print(
        f"  step_await_ms    : avg={weighted_await_ms:.3f}  "
        f"min={min(await_avgs):.3f}  max={max(await_avgs):.3f}   "
        f"<- run_in_executor wall on the asyncio loop"
    )
    print(
        f"  executor RTT     : avg={rtt_ms:.3f}   "
        f"<- step_await - step_inside (asyncio + GIL + thread switch)"
    )
    print(
        f"  loop_dispatch_ms : avg={weighted_disp_ms:.3f}  "
        f"min={min(disp_avgs):.3f}  max={max(disp_avgs):.3f}   "
        f"<- collector puts + mx.clear_cache + asyncio.sleep(0)"
    )
    per_token_ms = weighted_await_ms + weighted_disp_ms
    print(
        f"  per_token_ms     : {per_token_ms:.3f}  -> {1000.0 / per_token_ms:.2f} tok/s"
    )
    if per_token_ms > 0:
        print(
            f"  breakdown        : inside={100.0 * weighted_inside_ms / per_token_ms:.1f}%  "
            f"rtt={100.0 * rtt_ms / per_token_ms:.1f}%  "
            f"dispatch={100.0 * weighted_disp_ms / per_token_ms:.1f}%"
        )
    print(f"  raw log          : {log_path}")
    print(f"  windows captured : {len(samples)} (decode={len(decode_samples)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
