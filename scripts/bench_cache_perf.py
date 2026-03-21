#!/usr/bin/env python3
"""Benchmark cache performance: measures deepcopy overhead on cache hits.

Sends identical requests to measure:
  1. Cold TTFT (no cache)
  2. Cached TTFT (cache hit — includes deepcopy cost)
  3. Multi-turn cached TTFT (prefix match)
  4. Decode TPS

Outputs TSV-compatible metrics for perfup-results.tsv.
"""
import json
import statistics
import time
import urllib.request

BASE_URL = "http://127.0.0.1:8000/v1/chat/completions"
HEADERS = {"Content-Type": "application/json"}

SYSTEM_MSG = "You are a helpful assistant. Answer concisely. Do not use any thinking or reasoning tags."
USER_MSG = "/no_think Explain how a CPU works in detail. Write at least 500 words."
MULTI_TURN = [
    {"role": "system", "content": SYSTEM_MSG},
    {"role": "user", "content": "/no_think What is a binary search tree?"},
    {"role": "assistant", "content": "A binary search tree (BST) is a data structure where each node has at most two children."},
    {"role": "user", "content": "/no_think Now implement one in Python with insert and search."},
]


def stream_request(messages, max_tokens=500):
    """Send streaming request, return (ttft_ms, decode_tps, total_tokens)."""
    body = json.dumps({
        "model": "any",
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0.0,
    }).encode()

    req = urllib.request.Request(BASE_URL, data=body, headers=HEADERS, method="POST")
    start = time.perf_counter()
    first_token_time = None
    token_count = 0

    with urllib.request.urlopen(req) as resp:
        for line in resp:
            line = line.decode().strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
                delta = chunk["choices"][0]["delta"]
                content = delta.get("content")
                reasoning = delta.get("reasoning")
                if content or reasoning:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    token_count += 1
            except (json.JSONDecodeError, KeyError, IndexError):
                continue

    end = time.perf_counter()
    ttft = (first_token_time - start) * 1000 if first_token_time else 0
    decode_time = end - (first_token_time or start)
    tps = token_count / decode_time if decode_time > 0 and token_count > 1 else 0
    return ttft, tps, token_count


def run_benchmark(n_runs=3):
    messages_simple = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": USER_MSG},
    ]

    print("=" * 60)
    print("Cache Performance Benchmark")
    print("=" * 60)

    # 1. Cold TTFT
    print("\n[1/4] Cold TTFT (first request, no cache)...")
    ttft, tps, tokens = stream_request(messages_simple)
    print(f"  TTFT: {ttft:.0f}ms | Decode: {tps:.1f} tok/s | Tokens: {tokens}")
    cold_ttft = ttft
    baseline_tps = tps

    # 2. Cached TTFT (same prompt = exact cache hit)
    print(f"\n[2/4] Cached TTFT (x{n_runs} identical requests)...")
    cached_ttfts = []
    cached_tps_list = []
    for i in range(n_runs):
        ttft, tps, tokens = stream_request(messages_simple)
        cached_ttfts.append(ttft)
        cached_tps_list.append(tps)
        print(f"  Run {i+1}: TTFT={ttft:.0f}ms | Decode={tps:.1f} tok/s")

    # 3. Multi-turn cached TTFT (prefix match)
    print(f"\n[3/4] Multi-turn TTFT (prefix cache hit, x{n_runs})...")
    # First call to populate cache
    stream_request(MULTI_TURN, max_tokens=100)
    mt_ttfts = []
    for i in range(n_runs):
        ttft, tps, tokens = stream_request(MULTI_TURN, max_tokens=100)
        mt_ttfts.append(ttft)
        print(f"  Run {i+1}: TTFT={ttft:.0f}ms")

    # 4. Summary
    avg_cached = statistics.mean(cached_ttfts)
    avg_tps = statistics.mean(cached_tps_list)
    avg_mt = statistics.mean(mt_ttfts)

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Cold TTFT:       {cold_ttft:.0f} ms")
    print(f"  Cached TTFT:     {avg_cached:.0f} ms (avg of {n_runs})")
    print(f"  Multi-turn TTFT: {avg_mt:.0f} ms (avg of {n_runs})")
    print(f"  Cache speedup:   {cold_ttft/avg_cached:.1f}x")
    print(f"  Decode TPS:      {avg_tps:.1f} tok/s")
    print(f"  Baseline TPS:    {baseline_tps:.1f} tok/s")

    # TSV output for perfup-results.tsv
    print(f"\n# TSV: decode_tps\tcached_ttft_ms\tcold_ttft_ms\tmt_ttft_ms")
    print(f"METRIC\t{avg_tps:.1f}\t{avg_cached:.0f}\t{cold_ttft:.0f}\t{avg_mt:.0f}")

    return {
        "cold_ttft_ms": cold_ttft,
        "cached_ttft_ms": avg_cached,
        "mt_ttft_ms": avg_mt,
        "decode_tps": avg_tps,
        "cache_speedup": cold_ttft / avg_cached if avg_cached > 0 else 0,
    }


if __name__ == "__main__":
    run_benchmark()
