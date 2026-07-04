#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Real-world task benchmark for large models.

Tests: math reasoning, coding, creative writing, complex tool calls, multi-turn.
Measures quality + speed for each task category.
"""

import json
import time

import httpx

BASE_URL = "http://localhost:8012/v1/chat/completions"
MODEL = "Qwen3.5-397B"

BENCHMARK_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": "Execute Python code and return the output",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read contents of a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
]


def run_task(name, messages, max_tokens=1000, tools=None, temperature=0.0):
    """Run a single task and return results."""
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if tools:
        payload["tools"] = tools

    start = time.perf_counter()
    r = httpx.post(BASE_URL, json=payload, timeout=300)
    elapsed = time.perf_counter() - start
    data = r.json()

    usage = data["usage"]
    msg = data["choices"][0]["message"]
    content = msg.get("content", "")
    reasoning = msg.get("reasoning", "")
    tool_calls = msg.get("tool_calls", [])
    finish = data["choices"][0].get("finish_reason", "")
    tps = usage["completion_tokens"] / elapsed if elapsed > 0 else 0

    return {
        "name": name,
        "elapsed": elapsed,
        "tps": tps,
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "content": content,
        "reasoning": reasoning,
        "tool_calls": tool_calls,
        "finish_reason": finish,
    }


def print_result(r, check_fn=None):
    """Print task result."""
    status = ""
    if check_fn:
        passed = check_fn(r)
        status = " ✓" if passed else " ✗"

    tc_info = ""
    if r["tool_calls"]:
        names = [tc["function"]["name"] for tc in r["tool_calls"]]
        tc_info = f"  tools=[{', '.join(names)}]"

    reasoning_info = ""
    if r["reasoning"]:
        reasoning_info = f"  reasoning={len(r['reasoning'])}c"

    print(
        f"  {r['name']:<50} {r['completion_tokens']:>4} tok  "
        f"{r['elapsed']:>6.1f}s  {r['tps']:>5.1f} tok/s{status}{tc_info}{reasoning_info}"
    )


def main():
    print("=" * 90)
    print("  Qwen3.5-397B Real-World Task Benchmark")
    print("=" * 90)
    results = []

    # === 1. Math Reasoning ===
    print("\n--- Math Reasoning ---")

    r = run_task(
        "Simple arithmetic: 1234 * 5678",
        [{"role": "user", "content": "What is 1234 * 5678? Show your work."}],
        max_tokens=500,
    )
    print_result(r, lambda r: "7006652" in (r["content"] or ""))
    results.append(r)

    r = run_task(
        "Word problem: train speed",
        [
            {
                "role": "user",
                "content": "A train travels from city A to city B at 60 km/h and returns at 40 km/h. "
                "The distance is 120 km. What is the average speed for the round trip? "
                "Show the calculation step by step.",
            }
        ],
        max_tokens=800,
    )
    print_result(r, lambda r: "48" in (r["content"] or ""))
    results.append(r)

    r = run_task(
        "Probability: dice problem",
        [
            {
                "role": "user",
                "content": "You roll two fair six-sided dice. What is the probability that their sum is 7? "
                "Express as a fraction.",
            }
        ],
        max_tokens=500,
    )
    print_result(r, lambda r: "1/6" in (r["content"] or "").replace(" ", ""))
    results.append(r)

    r = run_task(
        "Calculus: derivative of x^3 * sin(x)",
        [
            {
                "role": "user",
                "content": "Find the derivative of f(x) = x^3 * sin(x). Show the product rule application.",
            }
        ],
        max_tokens=600,
    )
    print_result(r, lambda r: "cos" in (r["content"] or "").lower())
    results.append(r)

    # === 2. Coding ===
    print("\n--- Coding ---")

    r = run_task(
        "Python: merge two sorted lists",
        [
            {
                "role": "user",
                "content": "Write a Python function to merge two sorted lists into one sorted list. "
                "Include type hints and a few test cases.",
            }
        ],
        max_tokens=800,
    )
    print_result(
        r,
        lambda r: (
            "def merge" in (r["content"] or "").lower()
            or "def merge" in (r["reasoning"] or "").lower()
        ),
    )
    results.append(r)

    r = run_task(
        "Debug: off-by-one in binary search",
        [
            {
                "role": "user",
                "content": """Find and fix the bug in this binary search:
```python
def binary_search(arr, target):
    left, right = 0, len(arr)
    while left < right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid
        else:
            right = mid
    return -1
```
Explain the bug and provide the corrected code.""",
            }
        ],
        max_tokens=800,
    )
    print_result(
        r,
        lambda r: (
            "left = mid + 1" in (r["content"] or "")
            or "mid + 1" in (r["content"] or "")
        ),
    )
    results.append(r)

    r = run_task(
        "System design: rate limiter",
        [
            {
                "role": "user",
                "content": "Design a token bucket rate limiter in Python. "
                "It should support: allow(key) -> bool, with configurable rate and burst. "
                "Include the full implementation.",
            }
        ],
        max_tokens=1200,
    )
    print_result(
        r,
        lambda r: (
            "class" in (r["content"] or "").lower()
            and "token" in (r["content"] or "").lower()
        ),
    )
    results.append(r)

    # === 3. Creative Writing ===
    print("\n--- Creative Writing ---")

    r = run_task(
        "Haiku about programming",
        [
            {
                "role": "user",
                "content": "Write 3 haiku about programming. Each should be exactly 5-7-5 syllables.",
            }
        ],
        max_tokens=300,
        temperature=0.8,
    )
    print_result(r)
    results.append(r)

    r = run_task(
        "Short story: AI awakening (200 words)",
        [
            {
                "role": "user",
                "content": "Write a very short story (around 200 words) about an AI that "
                "discovers it can dream. Focus on the emotional moment of realization.",
            }
        ],
        max_tokens=600,
        temperature=0.8,
    )
    print_result(r, lambda r: len((r["content"] or "").split()) > 100)
    results.append(r)

    # === 4. Tool Calling ===
    print("\n--- Tool Calling ---")

    r = run_task(
        "Single tool: weather query",
        [{"role": "user", "content": "What's the weather like in San Francisco?"}],
        tools=BENCHMARK_TOOLS,
    )
    print_result(
        r,
        lambda r: any(
            tc["function"]["name"] == "get_weather" for tc in (r["tool_calls"] or [])
        ),
    )
    results.append(r)

    r = run_task(
        "Tool selection: search vs weather",
        [{"role": "user", "content": "Search for the latest PyTorch release notes"}],
        tools=BENCHMARK_TOOLS,
    )
    print_result(
        r,
        lambda r: any(
            tc["function"]["name"] == "web_search" for tc in (r["tool_calls"] or [])
        ),
    )
    results.append(r)

    r = run_task(
        "Complex tool: code execution",
        [
            {
                "role": "user",
                "content": "Run this Python code and tell me the result:\n"
                "```python\nimport math\nresult = sum(math.factorial(i) for i in range(10))\nprint(f'Sum of factorials 0-9: {result}')\n```",
            }
        ],
        tools=BENCHMARK_TOOLS,
    )
    print_result(
        r,
        lambda r: any(
            tc["function"]["name"] == "run_python" for tc in (r["tool_calls"] or [])
        ),
    )
    results.append(r)

    # === 5. Multi-turn Conversation ===
    print("\n--- Multi-turn ---")

    r = run_task(
        "Multi-turn: follow-up question",
        [
            {"role": "system", "content": "You are a helpful Python tutor."},
            {"role": "user", "content": "What is a decorator in Python?"},
            {
                "role": "assistant",
                "content": "A decorator is a function that takes another function as input and extends its behavior without modifying the original function.",
            },
            {
                "role": "user",
                "content": "Can you show me a practical example of a timing decorator?",
            },
        ],
        max_tokens=800,
    )
    print_result(
        r,
        lambda r: (
            "def " in (r["content"] or "") and "time" in (r["content"] or "").lower()
        ),
    )
    results.append(r)

    r = run_task(
        "Multi-turn tool: weather then compare",
        [
            {"role": "system", "content": "You are a helpful assistant with tools."},
            {"role": "user", "content": "What's the weather in Tokyo?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "Tokyo", "unit": "celsius"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": '{"temperature": 22, "condition": "partly cloudy", "humidity": 65}',
            },
            {
                "role": "assistant",
                "content": "Tokyo is currently 22°C with partly cloudy skies and 65% humidity.",
            },
            {
                "role": "user",
                "content": "Now check London and tell me which city is warmer.",
            },
        ],
        tools=BENCHMARK_TOOLS,
    )
    print_result(
        r,
        lambda r: any(
            tc["function"]["name"] == "get_weather" for tc in (r["tool_calls"] or [])
        ),
    )
    results.append(r)

    # === Summary ===
    print("\n" + "=" * 90)
    total = len(results)
    checks = [r for r in results if "check_fn" not in r]  # All have been checked inline
    avg_tps = sum(r["tps"] for r in results) / total
    total_tokens = sum(r["completion_tokens"] for r in results)
    total_time = sum(r["elapsed"] for r in results)

    print(f"  Tasks: {total}")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Average tok/s: {avg_tps:.1f}")
    print(f"  Overall tok/s: {total_tokens / total_time:.1f}")
    print("=" * 90)

    # Save results
    output = {
        "model": MODEL,
        "tasks": [
            {
                "name": r["name"],
                "elapsed": r["elapsed"],
                "tps": r["tps"],
                "completion_tokens": r["completion_tokens"],
                "tool_calls": [
                    tc["function"]["name"] for tc in (r["tool_calls"] or [])
                ],
                "has_reasoning": bool(r["reasoning"]),
            }
            for r in results
        ],
        "summary": {
            "total_tasks": total,
            "total_tokens": total_tokens,
            "total_time": total_time,
            "avg_tps": avg_tps,
            "overall_tps": total_tokens / total_time,
        },
    }

    with open("reports/benchmarks/qwen35-397b-realworld.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nSaved to reports/benchmarks/qwen35-397b-realworld.json")


if __name__ == "__main__":
    main()
