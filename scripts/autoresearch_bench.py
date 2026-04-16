#!/usr/bin/env python3
"""AutoResearch comprehensive benchmark for Rapid-MLX.

Measures 6 metrics across multiple scenarios to detect both improvements
and regressions. Returns a composite score for automated comparison.

Metrics:
  1. Decode TPS (tok/s) — main throughput
  2. Cold TTFT (ms) — first request latency
  3. Cached TTFT (ms) — repeat request latency (prompt cache)
  4. Multi-turn TTFT (ms) — prefix cache hit
  5. Tool call latency (ms) — time to first tool call token
  6. Long prompt TTFT (ms) — large context handling

Usage:
  python3.12 scripts/autoresearch_bench.py [--runs N] [--json]
"""
import argparse
import json
import os
import statistics
import sys
import time
import urllib.request

# Allow the doctor harness to point this at a non-default port.
BASE_URL = os.environ.get(
    "AUTORESEARCH_BASE", "http://127.0.0.1:8000/v1/chat/completions"
)
HEADERS = {"Content-Type": "application/json"}

SYSTEM_MSG = "You are a helpful assistant. Answer concisely."
USER_MSG = "/no_think Explain how a hash table works. Cover collisions, load factor, and resizing. Write at least 300 words."

MULTI_TURN = [
    {"role": "system", "content": SYSTEM_MSG},
    {"role": "user", "content": "/no_think What is a binary search tree?"},
    {"role": "assistant", "content": "A binary search tree (BST) is a data structure where each node has at most two children. The left child's value is less than the parent, and the right child's value is greater. This ordering property enables O(log n) search, insert, and delete operations in balanced trees."},
    {"role": "user", "content": "/no_think Now implement one in Python with insert and search methods."},
]

TOOL_MESSAGES = [
    {"role": "system", "content": SYSTEM_MSG},
    {"role": "user", "content": "/no_think What's the weather in Tokyo?"},
]

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    }
]

# Long prompt: system + 20 Q&A pairs to stress prefill
LONG_PROMPT_MESSAGES = [
    {"role": "system", "content": "You are a knowledgeable assistant. " * 50},
]
for i in range(20):
    LONG_PROMPT_MESSAGES.append({"role": "user", "content": f"/no_think Question {i+1}: Explain concept number {i+1} in computer science briefly."})
    LONG_PROMPT_MESSAGES.append({"role": "assistant", "content": f"Concept {i+1} is an important topic in computer science that involves understanding fundamental principles of computation, data organization, and algorithm design. It builds on prior concepts and enables more advanced techniques."})
LONG_PROMPT_MESSAGES.append({"role": "user", "content": "/no_think Summarize all concepts in one paragraph."})


def stream_request(messages, max_tokens=300, tools=None, timeout=60):
    """Send streaming request, return metrics dict."""
    body = {
        "model": "default",
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0.0,
    }
    if tools:
        body["tools"] = tools

    data = json.dumps(body).encode()
    req = urllib.request.Request(BASE_URL, data=data, headers=HEADERS, method="POST")
    start = time.perf_counter()
    first_token_time = None
    token_count = 0
    has_tool_calls = False
    tool_call_time = None

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for line in resp:
                line = line.decode().strip()
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                    delta = chunk["choices"][0]["delta"]
                    content = delta.get("content")
                    reasoning = delta.get("reasoning")
                    tc = delta.get("tool_calls")
                    if content or reasoning:
                        if first_token_time is None:
                            first_token_time = time.perf_counter()
                        token_count += 1
                    if tc:
                        has_tool_calls = True
                        if tool_call_time is None:
                            tool_call_time = time.perf_counter()
                        token_count += 1
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
    except Exception as e:
        return {"error": str(e), "ttft_ms": 0, "tps": 0, "tokens": 0}

    end = time.perf_counter()
    ttft = (first_token_time - start) * 1000 if first_token_time else 0
    if tool_call_time and not first_token_time:
        ttft = (tool_call_time - start) * 1000
    decode_time = end - (first_token_time or tool_call_time or start)
    tps = (token_count - 1) / decode_time if decode_time > 0 and token_count > 1 else 0
    tc_latency = (tool_call_time - start) * 1000 if tool_call_time else None

    return {
        "ttft_ms": ttft,
        "tps": tps,
        "tokens": token_count,
        "has_tool_calls": has_tool_calls,
        "tc_latency_ms": tc_latency,
        "total_ms": (end - start) * 1000,
    }


def run_suite(n_runs=3, verbose=True):
    """Run complete benchmark suite. Returns dict of metrics."""
    results = {}

    def log(msg):
        if verbose:
            print(msg, flush=True)

    # 1. Cold TTFT + Decode TPS (first request, no cache)
    log("\n[1/6] Cold TTFT + Decode TPS...")
    msgs = [{"role": "system", "content": SYSTEM_MSG}, {"role": "user", "content": USER_MSG}]
    r = stream_request(msgs, max_tokens=400)
    results["cold_ttft_ms"] = r["ttft_ms"]
    results["cold_tps"] = r["tps"]
    log(f"  TTFT: {r['ttft_ms']:.0f}ms | Decode: {r['tps']:.1f} tok/s | Tokens: {r['tokens']}")

    # 2. Cached TTFT + steady-state Decode TPS
    log(f"\n[2/6] Cached TTFT + Decode TPS (x{n_runs})...")
    cached_ttfts = []
    cached_tps = []
    for i in range(n_runs):
        r = stream_request(msgs, max_tokens=400)
        cached_ttfts.append(r["ttft_ms"])
        cached_tps.append(r["tps"])
        log(f"  Run {i+1}: TTFT={r['ttft_ms']:.0f}ms | Decode={r['tps']:.1f} tok/s")
    results["cached_ttft_ms"] = statistics.mean(cached_ttfts)
    results["decode_tps"] = statistics.mean(cached_tps)
    results["decode_tps_stdev"] = statistics.stdev(cached_tps) if len(cached_tps) > 1 else 0

    # 3. Multi-turn TTFT
    log(f"\n[3/6] Multi-turn TTFT (x{n_runs})...")
    stream_request(MULTI_TURN, max_tokens=150)  # warm cache
    mt_ttfts = []
    mt_tps = []
    for i in range(n_runs):
        r = stream_request(MULTI_TURN, max_tokens=150)
        mt_ttfts.append(r["ttft_ms"])
        mt_tps.append(r["tps"])
        log(f"  Run {i+1}: TTFT={r['ttft_ms']:.0f}ms | Decode={r['tps']:.1f} tok/s")
    results["mt_ttft_ms"] = statistics.mean(mt_ttfts)
    results["mt_tps"] = statistics.mean(mt_tps)

    # 4. Tool call latency
    log(f"\n[4/6] Tool call latency (x{n_runs})...")
    stream_request(TOOL_MESSAGES, max_tokens=200, tools=TOOLS)  # warm
    tc_latencies = []
    tc_success = 0
    for i in range(n_runs):
        r = stream_request(TOOL_MESSAGES, max_tokens=200, tools=TOOLS)
        if r.get("has_tool_calls"):
            tc_success += 1
            if r.get("tc_latency_ms"):
                tc_latencies.append(r["tc_latency_ms"])
        log(f"  Run {i+1}: TC={'YES' if r.get('has_tool_calls') else 'NO'} | Latency={r.get('tc_latency_ms') or 0:.0f}ms")
    results["tc_latency_ms"] = statistics.mean(tc_latencies) if tc_latencies else 0
    results["tc_success_rate"] = tc_success / n_runs

    # 5. Long prompt TTFT
    log(f"\n[5/6] Long prompt TTFT...")
    r = stream_request(LONG_PROMPT_MESSAGES, max_tokens=200)
    results["long_ttft_ms"] = r["ttft_ms"]
    results["long_tps"] = r["tps"]
    log(f"  TTFT: {r['ttft_ms']:.0f}ms | Decode: {r['tps']:.1f} tok/s | Tokens: {r['tokens']}")

    # 6. Long prompt cached TTFT
    log(f"\n[6/6] Long prompt cached TTFT (x{n_runs})...")
    long_cached = []
    for i in range(n_runs):
        r = stream_request(LONG_PROMPT_MESSAGES, max_tokens=200)
        long_cached.append(r["ttft_ms"])
        log(f"  Run {i+1}: TTFT={r['ttft_ms']:.0f}ms")
    results["long_cached_ttft_ms"] = statistics.mean(long_cached)

    # Composite score: weighted combination (higher = better)
    # Weights reflect importance for user experience
    score = (
        results["decode_tps"] * 1.0          # Primary: throughput
        + (1000 / max(results["cached_ttft_ms"], 1)) * 10  # Responsiveness
        + (1000 / max(results["mt_ttft_ms"], 1)) * 5       # Multi-turn speed
        + results["tc_success_rate"] * 20     # Tool reliability
        + (1000 / max(results["long_cached_ttft_ms"], 1)) * 3  # Long context
    )
    results["composite_score"] = round(score, 1)

    return results


def print_summary(results, label=""):
    """Print human-readable summary."""
    print("\n" + "=" * 65)
    print(f"BENCHMARK RESULTS {label}")
    print("=" * 65)
    print(f"  Decode TPS:         {results['decode_tps']:.1f} tok/s (±{results.get('decode_tps_stdev', 0):.1f})")
    print(f"  Cold TTFT:          {results['cold_ttft_ms']:.0f} ms")
    print(f"  Cached TTFT:        {results['cached_ttft_ms']:.0f} ms")
    print(f"  Multi-turn TTFT:    {results['mt_ttft_ms']:.0f} ms")
    print(f"  Tool call latency:  {results['tc_latency_ms']:.0f} ms")
    print(f"  Tool success rate:  {results['tc_success_rate']*100:.0f}%")
    print(f"  Long prompt TTFT:   {results['long_ttft_ms']:.0f} ms (cold)")
    print(f"  Long cached TTFT:   {results['long_cached_ttft_ms']:.0f} ms")
    print(f"  Cache speedup:      {results['cold_ttft_ms']/max(results['cached_ttft_ms'],1):.1f}x")
    print(f"  Composite score:    {results['composite_score']:.1f}")
    print("=" * 65)


def compare_results(baseline, experiment, label=""):
    """Compare experiment to baseline, return (improved, regression_detected)."""
    print(f"\n{'─' * 65}")
    print(f"COMPARISON: {label}")
    print(f"{'─' * 65}")

    metrics = [
        ("decode_tps", "Decode TPS", "higher", "tok/s"),
        ("cached_ttft_ms", "Cached TTFT", "lower", "ms"),
        ("mt_ttft_ms", "Multi-turn TTFT", "lower", "ms"),
        ("tc_latency_ms", "Tool latency", "lower", "ms"),
        ("tc_success_rate", "Tool success", "higher", "%"),
        ("long_cached_ttft_ms", "Long cached TTFT", "lower", "ms"),
        ("composite_score", "Composite", "higher", ""),
    ]

    improved = False
    regression = False

    for key, name, direction, unit in metrics:
        b = baseline.get(key, 0)
        e = experiment.get(key, 0)
        if b == 0:
            delta_pct = 0
        else:
            delta_pct = ((e - b) / b) * 100

        if key == "tc_success_rate":
            b_str = f"{b*100:.0f}%"
            e_str = f"{e*100:.0f}%"
        else:
            b_str = f"{b:.1f}{unit}"
            e_str = f"{e:.1f}{unit}"

        # Determine if change is good/bad
        if direction == "higher":
            is_better = delta_pct > 1
            is_worse = delta_pct < -2
        else:
            is_better = delta_pct < -1
            is_worse = delta_pct > 2

        indicator = "  "
        if is_better:
            indicator = "UP"
            improved = True
        elif is_worse:
            indicator = "DN"
            regression = True

        print(f"  {indicator} {name:20s}: {b_str:>10s} -> {e_str:>10s} ({delta_pct:+.1f}%)")

    verdict = "KEEP" if improved and not regression else "REVERT" if regression else "NEUTRAL"
    print(f"\n  Verdict: {verdict}")
    return improved, regression


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoResearch benchmark")
    parser.add_argument("--runs", type=int, default=3, help="Runs per metric")
    parser.add_argument("--json", action="store_true", help="Output JSON only")
    parser.add_argument("--label", type=str, default="", help="Label for this run")
    args = parser.parse_args()

    # Quick server health check — derive /v1/models from BASE_URL so the
    # doctor harness's port override applies here too.
    _models_url = BASE_URL.replace("/chat/completions", "/models")
    try:
        urllib.request.urlopen(_models_url, timeout=5)
    except Exception as e:
        print(f"ERROR: Server not reachable at {_models_url}: {e}", file=sys.stderr)
        sys.exit(1)

    results = run_suite(n_runs=args.runs, verbose=not args.json)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print_summary(results, label=args.label)
