#!/usr/bin/env python3
"""
Engine Parity Benchmark — SimpleEngine vs BatchedEngine
========================================================
Measures single-user performance on both engines with identical prompts.

Metrics: TTFT (cold/cached), decode tok/s, multi-turn latency, tool call latency.

Usage:
    # Start server with SimpleEngine (default)
    rapid-mlx serve qwen3.5-4b --port 8000

    # Start server with BatchedEngine
    rapid-mlx serve qwen3.5-4b --port 8001 --continuous-batching

    # Run benchmark
    python3 scripts/bench_engine_parity.py
"""

import json
import sys
import time
from datetime import datetime

import httpx

SIMPLE_URL = "http://localhost:8000/v1"
BATCHED_URL = "http://localhost:8001/v1"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search files by pattern",
            "parameters": {
                "type": "object",
                "properties": {"pattern": {"type": "string"}},
                "required": ["pattern"],
            },
        },
    },
]


def detect_model(base_url: str) -> str:
    r = httpx.get(f"{base_url}/models", timeout=10)
    return r.json()["data"][0]["id"]


def bench_ttft_cold(base_url: str, model: str) -> dict:
    """Cold TTFT — first request after server start (no cache)."""
    t0 = time.perf_counter()
    first_token_time = None
    total_tokens = 0

    with httpx.stream(
        "POST",
        f"{base_url}/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": "Write a haiku about the ocean."}],
            "max_tokens": 100,
            "stream": True,
        },
        timeout=120,
    ) as resp:
        for line in resp.iter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            data = json.loads(line[6:])
            delta = data["choices"][0].get("delta", {})
            if delta.get("content") and first_token_time is None:
                first_token_time = time.perf_counter()
            if delta.get("content"):
                total_tokens += 1

    elapsed = time.perf_counter() - t0
    ttft = (first_token_time - t0) if first_token_time else elapsed
    tps = total_tokens / (elapsed - ttft) if elapsed > ttft and total_tokens > 0 else 0

    return {"ttft_ms": round(ttft * 1000, 1), "decode_tps": round(tps, 1), "tokens": total_tokens}


def bench_ttft_cached(base_url: str, model: str) -> dict:
    """Cached TTFT — repeat the same system prompt, measure cache hit."""
    system = "You are a helpful assistant specialized in Python programming. You provide concise, accurate answers."
    user_msgs = [
        "What is a list comprehension?",
        "How do decorators work?",
        "Explain generators vs iterators.",
    ]

    results = []
    for msg in user_msgs:
        t0 = time.perf_counter()
        first_token_time = None
        total_tokens = 0

        with httpx.stream(
            "POST",
            f"{base_url}/chat/completions",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": msg},
                ],
                "max_tokens": 80,
                "stream": True,
            },
            timeout=120,
        ) as resp:
            for line in resp.iter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                data = json.loads(line[6:])
                delta = data["choices"][0].get("delta", {})
                if delta.get("content") and first_token_time is None:
                    first_token_time = time.perf_counter()
                if delta.get("content"):
                    total_tokens += 1

        elapsed = time.perf_counter() - t0
        ttft = (first_token_time - t0) if first_token_time else elapsed
        tps = total_tokens / (elapsed - ttft) if elapsed > ttft and total_tokens > 0 else 0
        results.append({"ttft_ms": round(ttft * 1000, 1), "decode_tps": round(tps, 1)})

    # First is cold, rest are cached
    return {
        "cold_ttft_ms": results[0]["ttft_ms"],
        "cached_ttft_ms": round(sum(r["ttft_ms"] for r in results[1:]) / len(results[1:]), 1),
        "avg_tps": round(sum(r["decode_tps"] for r in results) / len(results), 1),
    }


def bench_multi_turn(base_url: str, model: str) -> dict:
    """Multi-turn conversation — measure per-turn latency."""
    messages = [
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "What is 2+2?"},
    ]

    turn_times = []
    for turn in range(4):
        t0 = time.perf_counter()
        r = httpx.post(
            f"{base_url}/chat/completions",
            json={"model": model, "messages": messages, "max_tokens": 60},
            timeout=120,
        )
        elapsed = time.perf_counter() - t0
        turn_times.append(round(elapsed * 1000, 1))

        content = r.json()["choices"][0]["message"]["content"]
        messages.append({"role": "assistant", "content": content})

        follow_ups = [
            "What is that times 3?",
            "And divided by 2?",
            "Express that as a fraction.",
        ]
        if turn < len(follow_ups):
            messages.append({"role": "user", "content": follow_ups[turn]})

    return {
        "turn_latencies_ms": turn_times,
        "avg_turn_ms": round(sum(turn_times) / len(turn_times), 1),
    }


def bench_tool_call(base_url: str, model: str) -> dict:
    """Tool call latency — single tool call request."""
    times = []
    for prompt in [
        "What's the weather in Paris?",
        "Search for *.py files",
        "What's the weather in Tokyo?",
    ]:
        t0 = time.perf_counter()
        r = httpx.post(
            f"{base_url}/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "tools": TOOLS,
                "max_tokens": 200,
            },
            timeout=120,
        )
        elapsed = time.perf_counter() - t0
        msg = r.json()["choices"][0]["message"]
        has_tc = bool(msg.get("tool_calls"))
        times.append({"latency_ms": round(elapsed * 1000, 1), "tool_called": has_tc})

    return {
        "avg_latency_ms": round(sum(t["latency_ms"] for t in times) / len(times), 1),
        "success_rate": sum(1 for t in times if t["tool_called"]) / len(times),
        "details": times,
    }


def bench_decode_long(base_url: str, model: str) -> dict:
    """Long decode — measure sustained tok/s over 256 tokens."""
    t0 = time.perf_counter()
    first_token_time = None
    total_tokens = 0

    with httpx.stream(
        "POST",
        f"{base_url}/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": "Write a detailed essay about the history of computing. Be thorough."}],
            "max_tokens": 256,
            "stream": True,
        },
        timeout=120,
    ) as resp:
        for line in resp.iter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            data = json.loads(line[6:])
            delta = data["choices"][0].get("delta", {})
            if delta.get("content") and first_token_time is None:
                first_token_time = time.perf_counter()
            if delta.get("content"):
                total_tokens += 1

    elapsed = time.perf_counter() - t0
    ttft = (first_token_time - t0) if first_token_time else elapsed
    decode_time = elapsed - ttft
    tps = total_tokens / decode_time if decode_time > 0 and total_tokens > 0 else 0

    return {"ttft_ms": round(ttft * 1000, 1), "decode_tps": round(tps, 1), "tokens": total_tokens}


def run_suite(name: str, base_url: str, model: str) -> dict:
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"  Model: {model}")
    print(f"  URL: {base_url}")
    print(f"{'=' * 60}")

    results = {}

    print("\n  [1/5] Cold TTFT...")
    results["cold"] = bench_ttft_cold(base_url, model)
    print(f"        TTFT: {results['cold']['ttft_ms']}ms, {results['cold']['decode_tps']} tok/s")

    print("  [2/5] Cached TTFT (3 turns, same system prompt)...")
    results["cached"] = bench_ttft_cached(base_url, model)
    print(f"        Cold: {results['cached']['cold_ttft_ms']}ms, Cached: {results['cached']['cached_ttft_ms']}ms, {results['cached']['avg_tps']} tok/s")

    print("  [3/5] Multi-turn (4 turns)...")
    results["multi_turn"] = bench_multi_turn(base_url, model)
    print(f"        Avg: {results['multi_turn']['avg_turn_ms']}ms per turn")

    print("  [4/5] Tool call (3 calls)...")
    results["tool_call"] = bench_tool_call(base_url, model)
    print(f"        Avg: {results['tool_call']['avg_latency_ms']}ms, {results['tool_call']['success_rate']:.0%} success")

    print("  [5/5] Long decode (256 tokens)...")
    results["long_decode"] = bench_decode_long(base_url, model)
    print(f"        {results['long_decode']['decode_tps']} tok/s, {results['long_decode']['tokens']} tokens")

    return results


def main():
    # Check both servers are up
    for name, url in [("SimpleEngine", SIMPLE_URL), ("BatchedEngine", BATCHED_URL)]:
        try:
            r = httpx.get(f"{url}/models", timeout=5)
            r.raise_for_status()
            print(f"  {name} ({url}): OK")
        except Exception as e:
            print(f"  {name} ({url}): NOT AVAILABLE — {e}")
            print(f"\nPlease start both servers:")
            print(f"  Terminal 1: rapid-mlx serve qwen3.5-4b --port 8000")
            print(f"  Terminal 2: rapid-mlx serve qwen3.5-4b --port 8001 --continuous-batching")
            sys.exit(1)

    model_simple = detect_model(SIMPLE_URL)
    model_batched = detect_model(BATCHED_URL)

    # Run benchmarks
    simple_results = run_suite("SimpleEngine", SIMPLE_URL, model_simple)
    batched_results = run_suite("BatchedEngine", BATCHED_URL, model_batched)

    # Compare
    print(f"\n{'=' * 60}")
    print(f"  COMPARISON: BatchedEngine vs SimpleEngine")
    print(f"{'=' * 60}")

    comparisons = [
        ("Cold TTFT", simple_results["cold"]["ttft_ms"], batched_results["cold"]["ttft_ms"], "ms", True),
        ("Cold decode", simple_results["cold"]["decode_tps"], batched_results["cold"]["decode_tps"], "tok/s", False),
        ("Cached TTFT", simple_results["cached"]["cached_ttft_ms"], batched_results["cached"]["cached_ttft_ms"], "ms", True),
        ("Cached decode", simple_results["cached"]["avg_tps"], batched_results["cached"]["avg_tps"], "tok/s", False),
        ("Multi-turn avg", simple_results["multi_turn"]["avg_turn_ms"], batched_results["multi_turn"]["avg_turn_ms"], "ms", True),
        ("Tool call avg", simple_results["tool_call"]["avg_latency_ms"], batched_results["tool_call"]["avg_latency_ms"], "ms", True),
        ("Long decode", simple_results["long_decode"]["decode_tps"], batched_results["long_decode"]["decode_tps"], "tok/s", False),
    ]

    print(f"\n  {'Metric':<20s} {'Simple':>10s} {'Batched':>10s} {'Diff':>10s} {'Verdict':>10s}")
    print(f"  {'─' * 62}")

    all_pass = True
    for name, simple_val, batched_val, unit, lower_is_better in comparisons:
        if lower_is_better:
            diff_pct = ((batched_val - simple_val) / simple_val * 100) if simple_val > 0 else 0
            verdict = "OK" if diff_pct < 5 else "WARN" if diff_pct < 10 else "FAIL"
        else:
            diff_pct = ((simple_val - batched_val) / simple_val * 100) if simple_val > 0 else 0
            verdict = "OK" if diff_pct < 5 else "WARN" if diff_pct < 10 else "FAIL"

        if verdict != "OK":
            all_pass = False

        sign = "+" if diff_pct > 0 else ""
        print(f"  {name:<20s} {simple_val:>8.1f}{unit:>3s} {batched_val:>8.1f}{unit:>3s} {sign}{diff_pct:>+7.1f}% {verdict:>8s}")

    print(f"\n  Overall: {'PASS — BatchedEngine within 5%' if all_pass else 'REVIEW NEEDED'}")

    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "model_simple": model_simple,
        "model_batched": model_batched,
        "simple": simple_results,
        "batched": batched_results,
        "verdict": "pass" if all_pass else "review",
    }

    out_path = "reports/engine_parity_benchmark.json"
    import os
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
