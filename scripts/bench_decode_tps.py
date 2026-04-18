#!/usr/bin/env python3
"""
Decode TPS Investigation — SimpleEngine vs BatchedEngine
=========================================================
Tests with and without thinking to isolate the decode speed difference.

Usage:
    # Start one engine at a time on :8000
    python3 scripts/bench_decode_tps.py --label simple
    python3 scripts/bench_decode_tps.py --label batched
"""

import argparse
import json
import os
import time
from datetime import datetime

import httpx

BASE_URL = "http://localhost:8000/v1"


def detect_model(base_url: str) -> str:
    return httpx.get(f"{base_url}/models", timeout=10).json()["data"][0]["id"]


def detect_engine(base_url: str) -> str:
    try:
        return httpx.get(base_url.replace("/v1", "") + "/health", timeout=5).json().get("engine_type", "?")
    except Exception:
        return "?"


def measure_streaming(base_url: str, model: str, messages: list, max_tokens: int,
                      enable_thinking: bool | None = None, label: str = "") -> dict:
    """Stream a request and measure TTFT + decode TPS precisely.

    Uses server-reported usage.completion_tokens for TPS, NOT SSE chunk count.
    SSE chunk count varies between engines (BatchedEngine merges ~2 tokens
    per chunk via IncrementalDecoder) so counting chunks gives wrong TPS.
    """
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0.0,  # deterministic
    }
    if enable_thinking is not None:
        payload["enable_thinking"] = enable_thinking

    t0 = time.perf_counter()
    first_token_time = None
    sse_content_chunks = 0
    sse_reasoning_chunks = 0
    finish_reason = None
    usage = {}

    with httpx.stream("POST", f"{base_url}/chat/completions", json=payload, timeout=300) as resp:
        for line in resp.iter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            data = json.loads(line[6:])
            choice = data["choices"][0]
            delta = choice.get("delta", {})

            if delta.get("content"):
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                sse_content_chunks += 1

            if delta.get("reasoning_content"):
                sse_reasoning_chunks += 1

            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]

            if "usage" in data and data["usage"]:
                usage = data["usage"]

    elapsed = time.perf_counter() - t0
    ttft = (first_token_time - t0) if first_token_time else elapsed

    # Use server-reported token counts (accurate across both engines)
    completion_tokens = usage.get("completion_tokens", 0)
    reasoning_tokens = 0
    details = usage.get("completion_tokens_details", {})
    if details:
        reasoning_tokens = details.get("reasoning_tokens", 0)
    content_tokens = completion_tokens - reasoning_tokens

    total_tps = completion_tokens / elapsed if elapsed > 0 and completion_tokens > 0 else 0

    result = {
        "label": label,
        "ttft_ms": round(ttft * 1000, 1),
        "elapsed_ms": round(elapsed * 1000, 1),
        "completion_tokens": completion_tokens,
        "content_tokens": content_tokens,
        "reasoning_tokens": reasoning_tokens,
        "sse_chunks": sse_content_chunks + sse_reasoning_chunks,
        "total_tps": round(total_tps, 1),
        "finish_reason": finish_reason,
        "usage": usage,
    }
    return result


def run_test(base_url: str, model: str, name: str, messages: list, max_tokens: int,
             enable_thinking: bool | None, runs: int = 3) -> dict:
    """Run a test multiple times and average."""
    print(f"\n  [{name}] thinking={'on' if enable_thinking else 'off' if enable_thinking is False else 'default'}, "
          f"max_tokens={max_tokens}, runs={runs}")

    results = []
    for i in range(runs):
        r = measure_streaming(base_url, model, messages, max_tokens,
                              enable_thinking=enable_thinking, label=f"{name}_run{i}")
        results.append(r)
        print(f"    run {i+1}: completion={r['completion_tokens']} tok, "
              f"chunks={r['sse_chunks']}, tps={r['total_tps']}, ttft={r['ttft_ms']}ms")

    avg = {
        "name": name,
        "enable_thinking": enable_thinking,
        "max_tokens": max_tokens,
        "avg_completion_tokens": round(sum(r["completion_tokens"] for r in results) / len(results), 1),
        "avg_sse_chunks": round(sum(r["sse_chunks"] for r in results) / len(results), 1),
        "avg_total_tps": round(sum(r["total_tps"] for r in results) / len(results), 1),
        "avg_ttft_ms": round(sum(r["ttft_ms"] for r in results) / len(results), 1),
        "runs": results,
    }
    print(f"    AVG: tps={avg['avg_total_tps']}, tokens={avg['avg_completion_tokens']}, "
          f"chunks={avg['avg_sse_chunks']}")
    return avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True, help="simple or batched")
    parser.add_argument("--url", default=BASE_URL)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    model = detect_model(args.url)
    engine = detect_engine(args.url)

    print(f"\n{'=' * 60}")
    print(f"  Decode TPS Investigation: {args.label} ({engine})")
    print(f"  Model: {model}")
    print(f"{'=' * 60}")

    short_msg = [{"role": "user", "content": "Count from 1 to 50, one number per line."}]
    long_msg = [{"role": "user", "content": "Write the numbers 1 through 200, one per line. Nothing else."}]

    tests = {}

    # --- Test 1: No thinking, short ---
    tests["no_think_short"] = run_test(
        args.url, model, "no_think_short", short_msg,
        max_tokens=200, enable_thinking=False, runs=args.runs,
    )

    # --- Test 2: No thinking, long ---
    tests["no_think_long"] = run_test(
        args.url, model, "no_think_long", long_msg,
        max_tokens=512, enable_thinking=False, runs=args.runs,
    )

    # --- Test 3: With thinking, short ---
    tests["think_short"] = run_test(
        args.url, model, "think_short", short_msg,
        max_tokens=200, enable_thinking=True, runs=args.runs,
    )

    # --- Test 4: With thinking, long ---
    tests["think_long"] = run_test(
        args.url, model, "think_long", long_msg,
        max_tokens=512, enable_thinking=True, runs=args.runs,
    )

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  SUMMARY: {args.label} ({engine})")
    print(f"{'=' * 60}")
    print(f"\n  {'Test':<20s} {'TPS':>8s} {'Tokens':>8s} {'Chunks':>8s} {'TTFT':>8s}")
    print(f"  {'─' * 56}")
    for t in tests.values():
        print(f"  {t['name']:<20s} {t['avg_total_tps']:>6.1f}   {t['avg_completion_tokens']:>6.0f}   "
              f"{t['avg_sse_chunks']:>6.0f}   {t['avg_ttft_ms']:>6.0f}ms")

    # Save
    output = {
        "label": args.label,
        "engine": engine,
        "model": model,
        "timestamp": datetime.now().isoformat(),
        "tests": tests,
    }
    out_path = f"reports/decode_tps_{args.label}.json"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
