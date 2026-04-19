#!/usr/bin/env python3
"""
Stress test for BatchedEngine with Qwen 3.6.

Tests:
1. Sustained throughput — 20 sequential requests
2. Concurrent load — 4 parallel streaming requests
3. Long generation — 1024 token output
4. Rapid fire — 10 requests as fast as possible (non-streaming)
5. Tool call storm — 10 sequential tool call requests
6. Mixed workload — concurrent chat + tools + streaming
7. Disconnect resilience — start streaming then abort mid-stream
8. Memory stability — 5 rounds, check no OOM
"""

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx

# Default port; overridable via --port CLI arg
_PORT = 8000
BASE_URL = f"http://localhost:{_PORT}/v1"
TIMEOUT = 120

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    },
]


def detect_model():
    return httpx.get(f"{BASE_URL}/models", timeout=10).json()["data"][0]["id"]


def chat(msg, max_tokens=100, stream=False, tools=None, enable_thinking=False):
    """Send a chat request and return (latency_ms, tokens, content/error)."""
    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": msg}],
        "max_tokens": max_tokens,
        "stream": stream,
        "enable_thinking": enable_thinking,
    }
    if tools:
        payload["tools"] = tools

    t0 = time.perf_counter()
    try:
        if stream:
            tokens = 0
            content = ""
            with httpx.stream("POST", f"{BASE_URL}/chat/completions",
                              json=payload, timeout=TIMEOUT) as resp:
                for line in resp.iter_lines():
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    data = json.loads(line[6:])
                    if not data.get("choices"):
                        continue
                    delta = data["choices"][0].get("delta", {})
                    # Count both content and reasoning_content (OutputRouter models
                    # like Gemma 4 route output to reasoning_content channel)
                    text = delta.get("content") or delta.get("reasoning_content") or ""
                    if text:
                        tokens += 1
                        content += text
            elapsed = time.perf_counter() - t0
            return round(elapsed * 1000, 1), tokens, content[:100]
        else:
            r = httpx.post(f"{BASE_URL}/chat/completions",
                           json=payload, timeout=TIMEOUT)
            elapsed = time.perf_counter() - t0
            data = r.json()
            ct = data.get("usage", {}).get("completion_tokens", 0)
            msg_data = data["choices"][0]["message"]
            tc = len(msg_data.get("tool_calls") or [])
            content = msg_data.get("content") or msg_data.get("reasoning_content") or ""
            content = content[:80]
            return round(elapsed * 1000, 1), ct, f"tc={tc} {content}" if tc else content
    except Exception as e:
        elapsed = time.perf_counter() - t0
        return round(elapsed * 1000, 1), 0, f"ERROR: {e}"


def test_sustained_throughput():
    """20 sequential requests — check consistency."""
    print("\n[1/8] Sustained throughput (20 requests)...")
    latencies = []
    errors = 0
    for i in range(20):
        ms, tokens, content = chat(f"What is {i}+{i}?", max_tokens=50, enable_thinking=False)
        latencies.append(ms)
        if "ERROR" in str(content):
            errors += 1
        sys.stdout.write(f"  {i+1}/20 {ms:.0f}ms ")
        sys.stdout.flush()
    print()
    avg = sum(latencies) / len(latencies)
    p99 = sorted(latencies)[int(len(latencies) * 0.99)]
    print(f"  Avg: {avg:.0f}ms, P99: {p99:.0f}ms, Errors: {errors}/20")
    return errors == 0


def test_concurrent_load():
    """4 parallel streaming requests."""
    print("\n[2/8] Concurrent load (4 parallel streams)...")
    prompts = [
        "Explain quantum computing in 3 sentences.",
        "What are the planets in our solar system?",
        "Write a haiku about mountains.",
        "List 5 programming languages and their uses.",
    ]

    results = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(chat, p, 100, True, None, False): p for p in prompts}
        for f in as_completed(futures):
            ms, tokens, content = f.result()
            results.append((ms, tokens, content))
            print(f"  {ms:.0f}ms, {tokens} chunks")

    errors = sum(1 for _, _, c in results if "ERROR" in str(c))
    print(f"  All completed. Errors: {errors}/4")
    return errors == 0


def test_long_generation():
    """Single 1024-token generation."""
    print("\n[3/8] Long generation (1024 tokens)...")
    ms, tokens, content = chat(
        "Write a detailed essay about the history of mathematics from ancient Egypt to modern times.",
        max_tokens=1024, stream=True, enable_thinking=False,
    )
    tps = tokens / (ms / 1000) if ms > 0 else 0
    print(f"  {ms:.0f}ms, {tokens} chunks, ~{tps:.1f} chunks/s")
    return "ERROR" not in str(content) and tokens > 50


def test_rapid_fire():
    """10 requests as fast as possible (non-streaming)."""
    print("\n[4/8] Rapid fire (10 non-streaming)...")
    t0 = time.perf_counter()
    errors = 0
    for i in range(10):
        ms, tokens, content = chat(f"Say '{i}'", max_tokens=20, enable_thinking=False)
        if "ERROR" in str(content):
            errors += 1
            print(f"  {i}: ERROR — {content}")
    elapsed = time.perf_counter() - t0
    print(f"  10 requests in {elapsed:.1f}s ({10/elapsed:.1f} req/s), Errors: {errors}")
    return errors == 0


def test_tool_call_storm():
    """10 sequential tool call requests."""
    print("\n[5/8] Tool call storm (10 requests)...")
    errors = 0
    tool_calls = 0
    for i in range(10):
        ms, tokens, content = chat(
            f"What's the weather in city_{i}?",
            max_tokens=100, tools=TOOLS, enable_thinking=False,
        )
        if "ERROR" in str(content):
            errors += 1
            print(f"  {i}: {content}")
        elif "tc=" in str(content) and "tc=0" not in str(content):
            # Structured tool_calls detected by parser
            tool_calls += 1
        elif any(kw in str(content) for kw in ["get_weather", "tool_call", "tool_use"]):
            # Model emitted tool call in content text (e.g. OutputRouter models)
            tool_calls += 1
    print(f"  Tool calls: {tool_calls}/10, Errors: {errors}")
    # Accept >= 5 (some models may not always produce tool calls for simple prompts)
    return errors == 0 and tool_calls >= 5


def test_mixed_workload():
    """Concurrent: 2 chat + 1 tool + 1 streaming."""
    print("\n[6/8] Mixed workload (4 concurrent, different types)...")

    def chat_req():
        return chat("What is 2+2?", 50, False, None, False)

    def tool_req():
        return chat("Weather in Paris?", 100, False, TOOLS, False)

    def stream_req():
        return chat("Write a poem.", 100, True, None, False)

    results = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
            pool.submit(chat_req): "chat1",
            pool.submit(chat_req): "chat2",
            pool.submit(tool_req): "tool",
            pool.submit(stream_req): "stream",
        }
        for f in as_completed(futures):
            name = futures[f]
            ms, tokens, content = f.result()
            results[name] = (ms, tokens, content)
            ok = "ERROR" not in str(content)
            print(f"  {name}: {ms:.0f}ms {'OK' if ok else 'FAIL'}")

    errors = sum(1 for _, _, c in results.values() if "ERROR" in str(c))
    print(f"  Errors: {errors}/4")
    return errors == 0


def test_disconnect_resilience():
    """Start streaming then abort after 5 chunks — server should not crash."""
    print("\n[7/8] Disconnect resilience (abort mid-stream)...")
    try:
        payload = {
            "model": "default",
            "messages": [{"role": "user", "content": "Write a very long story about a dragon."}],
            "max_tokens": 500,
            "stream": True,
            "enable_thinking": False,
        }
        chunks = 0
        with httpx.stream("POST", f"{BASE_URL}/chat/completions",
                          json=payload, timeout=30) as resp:
            for line in resp.iter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    chunks += 1
                    if chunks >= 5:
                        break  # Abort mid-stream

        # Verify server still works after disconnect
        ms, tokens, content = chat("Say hello", 20, False, None, False)
        ok = "ERROR" not in str(content)
        print(f"  Aborted after {chunks} chunks, server OK: {ok}")
        return ok
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def test_memory_stability():
    """5 rounds of mixed requests, check server stays healthy."""
    print("\n[8/8] Memory stability (5 rounds)...")
    for round_num in range(5):
        # Mix of request types
        chat("Hello", 30, False, None, False)
        chat("Weather?", 50, False, TOOLS, False)
        chat("Write something.", 80, True, None, False)

        # Health check
        h = httpx.get(f"http://localhost:{_PORT}/health", timeout=5).json()
        ok = h.get("status") == "healthy"
        print(f"  Round {round_num + 1}/5: {'OK' if ok else 'FAIL'}")
        if not ok:
            return False
    return True


def main():
    import argparse
    global _PORT, BASE_URL
    parser = argparse.ArgumentParser(description="Stress test for Rapid-MLX server")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    args = parser.parse_args()
    _PORT = args.port
    BASE_URL = f"http://localhost:{_PORT}/v1"

    model = detect_model()
    engine = httpx.get(f"http://localhost:{_PORT}/health", timeout=5).json().get("engine_type")

    print(f"{'=' * 60}")
    print(f"  Stress Test — {model}")
    print(f"  Engine: {engine}")
    print(f"{'=' * 60}")

    tests = [
        ("Sustained throughput", test_sustained_throughput),
        ("Concurrent load", test_concurrent_load),
        ("Long generation", test_long_generation),
        ("Rapid fire", test_rapid_fire),
        ("Tool call storm", test_tool_call_storm),
        ("Mixed workload", test_mixed_workload),
        ("Disconnect resilience", test_disconnect_resilience),
        ("Memory stability", test_memory_stability),
    ]

    results = {}
    for name, fn in tests:
        try:
            results[name] = fn()
        except Exception as e:
            print(f"  CRASH: {e}")
            results[name] = False

    print(f"\n{'=' * 60}")
    print("  RESULTS")
    print(f"{'=' * 60}")
    passed = 0
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  {status}  {name}")
        if ok:
            passed += 1
    print(f"\n  {passed}/{len(tests)} passed")
    print(f"{'=' * 60}")

    sys.exit(0 if passed == len(tests) else 1)


if __name__ == "__main__":
    main()
