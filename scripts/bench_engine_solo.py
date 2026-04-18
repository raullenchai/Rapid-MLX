#!/usr/bin/env python3
"""
Engine Solo Benchmark — test one engine at a time to avoid GPU contention.

Usage:
    python3 scripts/bench_engine_solo.py --url http://localhost:8000/v1 --label simple
    python3 scripts/bench_engine_solo.py --url http://localhost:8001/v1 --label batched
"""

import argparse
import json
import os
import time
from datetime import datetime

import httpx

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


def stream_request(base_url: str, model: str, messages: list, max_tokens: int = 100, tools=None) -> dict:
    """Stream a request and measure TTFT + decode TPS.

    Uses server-reported usage.completion_tokens for TPS calculation,
    NOT SSE chunk count (which varies between engines due to token merging).
    """
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if tools:
        payload["tools"] = tools

    t0 = time.perf_counter()
    first_token_time = None
    content = ""
    usage = {}

    with httpx.stream("POST", f"{base_url}/chat/completions", json=payload, timeout=120) as resp:
        for line in resp.iter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            data = json.loads(line[6:])
            delta = data["choices"][0].get("delta", {})
            if delta.get("content"):
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                content += delta["content"]
            if "usage" in data and data["usage"]:
                usage = data["usage"]

    elapsed = time.perf_counter() - t0
    ttft = (first_token_time - t0) if first_token_time else elapsed
    decode_time = elapsed - ttft
    # Use server-reported completion_tokens (accurate across both engines)
    completion_tokens = usage.get("completion_tokens", 0)
    tps = completion_tokens / decode_time if decode_time > 0 and completion_tokens > 0 else 0

    return {
        "ttft_ms": round(ttft * 1000, 1),
        "decode_tps": round(tps, 1),
        "tokens": completion_tokens,
        "elapsed_ms": round(elapsed * 1000, 1),
    }


def non_stream_request(base_url: str, model: str, messages: list, max_tokens: int = 100, tools=None) -> dict:
    """Non-streaming request, measure total latency."""
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if tools:
        payload["tools"] = tools

    t0 = time.perf_counter()
    r = httpx.post(f"{base_url}/chat/completions", json=payload, timeout=120)
    elapsed = time.perf_counter() - t0
    data = r.json()
    msg = data["choices"][0]["message"]
    tokens = data.get("usage", {}).get("completion_tokens", 0)

    return {
        "latency_ms": round(elapsed * 1000, 1),
        "tokens": tokens,
        "has_tool_calls": bool(msg.get("tool_calls")),
        "content_len": len(msg.get("content") or ""),
    }


def run_suite(base_url: str, model: str) -> dict:
    results = {}

    # --- Warmup ---
    print("  [0/6] Warmup...")
    stream_request(base_url, model, [{"role": "user", "content": "Hi"}], max_tokens=10)

    # --- 1. Short decode (streaming) ---
    print("  [1/6] Short decode (100 tokens, streaming)...")
    runs = []
    for prompt in ["Write a haiku about the ocean.", "Explain what a variable is.", "List 5 fruits."]:
        r = stream_request(base_url, model, [{"role": "user", "content": prompt}], max_tokens=100)
        runs.append(r)
    results["short_decode"] = {
        "avg_ttft_ms": round(sum(r["ttft_ms"] for r in runs) / len(runs), 1),
        "avg_tps": round(sum(r["decode_tps"] for r in runs) / len(runs), 1),
        "runs": runs,
    }
    print(f"        TTFT: {results['short_decode']['avg_ttft_ms']}ms, {results['short_decode']['avg_tps']} tok/s")

    # --- 2. Long decode (streaming) ---
    print("  [2/6] Long decode (512 tokens, streaming)...")
    r = stream_request(
        base_url, model,
        [{"role": "user", "content": "Write a detailed essay about the history of computing from Babbage to modern GPUs."}],
        max_tokens=512,
    )
    results["long_decode"] = r
    print(f"        TTFT: {r['ttft_ms']}ms, {r['decode_tps']} tok/s, {r['tokens']} tokens")

    # --- 3. Cached TTFT (same system prompt, 3 requests) ---
    print("  [3/6] Cached TTFT (same system prompt, 3 turns)...")
    system = "You are a Python expert. Give concise answers."
    runs = []
    for q in ["What is a list comprehension?", "Explain decorators.", "What are generators?"]:
        r = stream_request(base_url, model, [
            {"role": "system", "content": system},
            {"role": "user", "content": q},
        ], max_tokens=80)
        runs.append(r)
    results["cached_ttft"] = {
        "first_ttft_ms": runs[0]["ttft_ms"],
        "cached_ttft_ms": round(sum(r["ttft_ms"] for r in runs[1:]) / len(runs[1:]), 1),
        "avg_tps": round(sum(r["decode_tps"] for r in runs) / len(runs), 1),
        "runs": runs,
    }
    print(f"        First: {results['cached_ttft']['first_ttft_ms']}ms, Cached: {results['cached_ttft']['cached_ttft_ms']}ms")

    # --- 4. Multi-turn (4 turns, non-streaming) ---
    print("  [4/6] Multi-turn (4 turns)...")
    messages = [
        {"role": "system", "content": "You are concise."},
        {"role": "user", "content": "What is 2+2?"},
    ]
    turn_times = []
    for i in range(4):
        r = non_stream_request(base_url, model, messages, max_tokens=60)
        turn_times.append(r["latency_ms"])
        messages.append({"role": "assistant", "content": f"Response {i}"})
        follow = ["Times 3?", "Divided by 2?", "As a fraction?", "Done"]
        messages.append({"role": "user", "content": follow[min(i, 3)]})
    results["multi_turn"] = {
        "turn_latencies_ms": turn_times,
        "avg_turn_ms": round(sum(turn_times) / len(turn_times), 1),
    }
    print(f"        Avg: {results['multi_turn']['avg_turn_ms']}ms per turn")

    # --- 5. Tool call (3 calls, non-streaming) ---
    print("  [5/6] Tool call (3 calls)...")
    runs = []
    for prompt in ["Weather in Paris?", "Search for *.py", "Weather in Tokyo?"]:
        r = non_stream_request(base_url, model, [{"role": "user", "content": prompt}], max_tokens=200, tools=TOOLS)
        runs.append(r)
    results["tool_call"] = {
        "avg_latency_ms": round(sum(r["latency_ms"] for r in runs) / len(runs), 1),
        "success_rate": sum(1 for r in runs if r["has_tool_calls"]) / len(runs),
        "runs": runs,
    }
    print(f"        Avg: {results['tool_call']['avg_latency_ms']}ms, {results['tool_call']['success_rate']:.0%} success")

    # --- 6. Streaming tool call ---
    print("  [6/6] Streaming tool call...")
    t0 = time.perf_counter()
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "What's the weather in London?"}],
        "tools": TOOLS,
        "max_tokens": 200,
        "stream": True,
    }
    tool_chunks = 0
    with httpx.stream("POST", f"{base_url}/chat/completions", json=payload, timeout=60) as resp:
        for line in resp.iter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            data = json.loads(line[6:])
            delta = data["choices"][0].get("delta", {})
            if "tool_calls" in delta:
                tool_chunks += 1
    elapsed = time.perf_counter() - t0
    results["streaming_tool"] = {
        "latency_ms": round(elapsed * 1000, 1),
        "tool_chunks": tool_chunks,
    }
    print(f"        {results['streaming_tool']['latency_ms']}ms, {tool_chunks} chunks")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000/v1")
    parser.add_argument("--label", default="engine")
    args = parser.parse_args()

    model = detect_model(args.url)
    engine_type = "unknown"
    try:
        h = httpx.get(f"{args.url.replace('/v1', '')}/health", timeout=5).json()
        engine_type = h.get("engine_type", "unknown")
    except Exception:
        pass

    print(f"\n{'=' * 60}")
    print(f"  Engine Solo Benchmark: {args.label} ({engine_type})")
    print(f"  Model: {model}")
    print(f"  URL: {args.url}")
    print(f"{'=' * 60}")

    results = run_suite(args.url, model)

    output = {
        "label": args.label,
        "engine_type": engine_type,
        "model": model,
        "url": args.url,
        "timestamp": datetime.now().isoformat(),
        "results": results,
    }

    out_path = f"reports/engine_parity_{args.label}.json"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
