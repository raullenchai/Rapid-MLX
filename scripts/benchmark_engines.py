#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Benchmark script to compare inference engines on a given model.

Metrics measured:
  Speed:
    1. Decode TPS      — token generation speed (tok/s)
    2. TTFT cold       — time to first token on first request
    3. TTFT cached     — time to first token on subsequent requests (prompt cache)
    4. Prefill TPS     — prompt processing speed (prompt_tokens / TTFT)
    5. Multi-turn TTFT — conversation continuation latency (cold + cached)
    6. Peak RAM        — memory usage during inference (macOS only)
  Capability:
    7. Tool call %     — structured tool_calls success rate
    8. Recovery %      — auto-recovery of degraded text-format tool calls
    9. Leak %          — think-tag content leaking into response content
   10. Vision          — image understanding support
   11. Audio           — STT/TTS support

Usage:
    python scripts/benchmark_engines.py --engine rapid-mlx ollama llama-cpp
    python scripts/benchmark_engines.py --engine all --output results.json
"""

import argparse
import json
import statistics
import subprocess
import time

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SHORT_PROMPT = "/no_think Explain how a CPU works in detail."
LONG_PROMPT = (
    "/no_think Write a comprehensive guide to building a web application with Python. "
    "Cover the following topics in detail: 1) Choosing a framework (Django vs Flask vs FastAPI), "
    "2) Setting up the project structure, 3) Database design and ORM usage, "
    "4) Authentication and authorization, 5) RESTful API design, "
    "6) Testing strategies, 7) Deployment options. "
    "For each topic, provide code examples and best practices."
)
MULTI_TURN_SYSTEM = "You are a helpful coding assistant."
MULTI_TURN_MESSAGES = [
    {"role": "user", "content": "/no_think What is a binary search tree?"},
    {
        "role": "assistant",
        "content": "A binary search tree (BST) is a data structure where each node has at most two children. The left child contains values less than the parent, and the right child contains values greater than the parent.",
    },
    {
        "role": "user",
        "content": "/no_think Now implement one in Python with insert, search, and delete operations.",
    },
]

# ---------------------------------------------------------------------------
# Tool call test fixtures
# ---------------------------------------------------------------------------

BENCHMARK_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "default": "celsius",
                    },
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
                    "query": {"type": "string", "description": "Search query"},
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
                    "code": {"type": "string", "description": "Python code to execute"},
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
                    "path": {"type": "string", "description": "File path to read"},
                },
                "required": ["path"],
            },
        },
    },
]

# 10 tool call test prompts — each should trigger a specific tool
TOOL_CALL_SCENARIOS = [
    {"prompt": "What's the weather in Tokyo?", "expected_tool": "get_weather"},
    {
        "prompt": "What's the weather in Paris right now?",
        "expected_tool": "get_weather",
    },
    {
        "prompt": "Search the web for 'Python asyncio tutorial'",
        "expected_tool": "web_search",
    },
    {
        "prompt": "Look up the latest news about Apple Silicon M4",
        "expected_tool": "web_search",
    },
    {"prompt": "Run this Python code: print(2 + 2)", "expected_tool": "run_python"},
    {"prompt": "Execute: import math; print(math.pi)", "expected_tool": "run_python"},
    {"prompt": "Read the file at /etc/hostname", "expected_tool": "read_file"},
    {"prompt": "Show me the contents of /tmp/test.txt", "expected_tool": "read_file"},
    {
        "prompt": "What is the current temperature in London?",
        "expected_tool": "get_weather",
    },
    {
        "prompt": "Search for 'MLX framework benchmarks 2025'",
        "expected_tool": "web_search",
    },
]

# Prompts that should NOT trigger tool calls (irrelevance test)
NO_TOOL_PROMPTS = [
    "What is 2 + 2?",
    "Explain what a hash table is.",
    "Write a haiku about coding.",
]

# Prompts designed to trigger <think> tags for leak testing
THINK_PROMPTS = [
    "Think step by step: what is 15 * 17?",
    "Reason carefully about whether P = NP.",
    "Let me think about this: what are the pros and cons of Rust vs Go?",
    "Consider step by step: should I use PostgreSQL or MongoDB for a social media app?",
    "Think through this problem: how do you reverse a linked list?",
]


# ---------------------------------------------------------------------------
# Memory measurement (macOS)
# ---------------------------------------------------------------------------


def get_process_memory_mb(port: int) -> float | None:
    """Get RSS memory of the process listening on a port (macOS only)."""
    try:
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return None
        pid = result.stdout.strip().split("\n")[0]
        result = subprocess.run(
            ["ps", "-o", "rss=", "-p", pid],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return None
        rss_kb = int(result.stdout.strip())
        return rss_kb / 1024
    except Exception:
        return None


# ---------------------------------------------------------------------------
# OpenAI-compatible speed benchmark
# ---------------------------------------------------------------------------


def _count_stream_tokens(stream):
    """Consume a streaming response.

    Returns (completion_tokens, first_content_time, prompt_tokens).

    Token count comes from the server's ``usage.completion_tokens`` in the
    final SSE chunk (accurate).  Falls back to counting content-bearing
    chunks if the server doesn't report usage.

    ``first_content_time`` is the wall-clock time of the first chunk that
    carries *visible content* (``delta.content``).  Reasoning/thinking
    chunks are deliberately excluded so that TTFT reflects the latency a
    user actually experiences.
    """
    chunk_count = 0  # fallback counter
    first_content_time = None
    prompt_tokens = 0
    completion_tokens = 0  # from server usage

    for chunk in stream:
        # Capture usage from final chunk (most accurate token count)
        if hasattr(chunk, "usage") and chunk.usage:
            pt = getattr(chunk.usage, "prompt_tokens", None)
            ct = getattr(chunk.usage, "completion_tokens", None)
            if pt:
                prompt_tokens = pt
            if ct:
                completion_tokens = ct

        if chunk.choices:
            delta = chunk.choices[0].delta
            # Only count visible content for TTFT — not reasoning/thinking
            if delta.content:
                if first_content_time is None:
                    first_content_time = time.perf_counter()
                chunk_count += 1

    # Prefer server-reported token count; fall back to chunk count
    tokens = completion_tokens if completion_tokens > 0 else chunk_count
    return tokens, first_content_time, prompt_tokens


def _make_create_kwargs(client, model, messages, max_tokens, **extra):
    """Build kwargs for client.chat.completions.create.

    Deterministic: temperature=0, stream=True, thinking disabled.
    extra_body is used for ``enable_thinking`` so it works with both
    rapid-mlx (which supports it) and other OpenAI-compatible servers
    (which silently ignore unknown fields).
    """
    kwargs = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0,
        "extra_body": {"enable_thinking": False},
        "stream_options": {"include_usage": True},
    }
    kwargs.update(extra)
    return kwargs


def benchmark_speed(
    client, model: str, num_runs: int, max_tokens_short: int, max_tokens_long: int
) -> dict:
    """Run speed benchmarks. Returns raw results dict.

    All speed tests use temperature=0 and enable_thinking=False for
    deterministic, comparable results across runs and branches.
    """
    results = {
        "short_gen": [],
        "long_gen": [],
        "ttft_cold": [],
        "ttft_cached": [],
        "multi_turn_ttft": [],
    }

    # --- Short generation ---
    print(f"  Short generation ({max_tokens_short} tokens, {num_runs} runs)...")
    for i in range(num_runs):
        start = time.perf_counter()
        kwargs = _make_create_kwargs(
            client,
            model,
            messages=[{"role": "user", "content": SHORT_PROMPT}],
            max_tokens=max_tokens_short,
        )
        stream = client.chat.completions.create(**kwargs)
        tokens_received, first_token_time, prompt_tokens = _count_stream_tokens(stream)

        elapsed = time.perf_counter() - start
        ttft = first_token_time - start if first_token_time else elapsed
        decode_time = elapsed - ttft if first_token_time else elapsed
        tps = tokens_received / decode_time if decode_time > 0 else 0
        prefill_tps = prompt_tokens / ttft if (prompt_tokens and ttft > 0) else None

        results["short_gen"].append(
            {
                "tokens": tokens_received,
                "prompt_tokens": prompt_tokens,
                "elapsed": elapsed,
                "ttft": ttft,
                "tps": tps,
                "prefill_tps": prefill_tps,
            }
        )

        if i == 0:
            results["ttft_cold"].append(ttft)
        else:
            results["ttft_cached"].append(ttft)

        pfx = f", prefill {prefill_tps:.0f} tok/s" if prefill_tps else ""
        print(
            f"    Run {i + 1}: {tps:.1f} tok/s, TTFT {ttft:.3f}s{pfx}, {tokens_received} tokens"
        )

    # --- Long generation ---
    print(f"  Long generation ({max_tokens_long} tokens, {num_runs} runs)...")
    for i in range(num_runs):
        start = time.perf_counter()
        kwargs = _make_create_kwargs(
            client,
            model,
            messages=[{"role": "user", "content": LONG_PROMPT}],
            max_tokens=max_tokens_long,
        )
        stream = client.chat.completions.create(**kwargs)
        tokens_received, first_token_time, prompt_tokens = _count_stream_tokens(stream)

        elapsed = time.perf_counter() - start
        ttft = first_token_time - start if first_token_time else elapsed
        decode_time = elapsed - ttft if first_token_time else elapsed
        tps = tokens_received / decode_time if decode_time > 0 else 0
        prefill_tps = prompt_tokens / ttft if (prompt_tokens and ttft > 0) else None

        results["long_gen"].append(
            {
                "tokens": tokens_received,
                "prompt_tokens": prompt_tokens,
                "elapsed": elapsed,
                "ttft": ttft,
                "tps": tps,
                "prefill_tps": prefill_tps,
            }
        )
        pfx = f", prefill {prefill_tps:.0f} tok/s" if prefill_tps else ""
        print(
            f"    Run {i + 1}: {tps:.1f} tok/s, TTFT {ttft:.3f}s{pfx}, {tokens_received} tokens"
        )

    # --- Multi-turn TTFT ---
    print(f"  Multi-turn TTFT ({num_runs} runs)...")
    for i in range(num_runs):
        start = time.perf_counter()
        messages = [
            {"role": "system", "content": MULTI_TURN_SYSTEM}
        ] + MULTI_TURN_MESSAGES
        kwargs = _make_create_kwargs(
            client,
            model,
            messages=messages,
            max_tokens=100,
        )
        stream = client.chat.completions.create(**kwargs)
        _, first_token_time, _ = _count_stream_tokens(stream)
        ttft = (
            first_token_time - start
            if first_token_time
            else time.perf_counter() - start
        )
        results["multi_turn_ttft"].append(ttft)
        print(f"    Run {i + 1}: TTFT {ttft:.3f}s")

    return results


# ---------------------------------------------------------------------------
# Capability benchmarks: tool calling, recovery, leak
# ---------------------------------------------------------------------------


def benchmark_tool_calls(client, model: str) -> dict:
    """Test tool call success rate. Returns {success_rate, correct, total, details}."""
    print(f"  Tool call success rate ({len(TOOL_CALL_SCENARIOS)} scenarios)...")
    correct = 0
    total = len(TOOL_CALL_SCENARIOS)
    details = []

    for sc in TOOL_CALL_SCENARIOS:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": sc["prompt"]}],
                tools=BENCHMARK_TOOLS,
                max_tokens=300,
                temperature=0.0,
            )
            msg = resp.choices[0].message
            has_tool_calls = bool(msg.tool_calls)
            correct_tool = False
            tool_name = None

            if has_tool_calls:
                tool_name = msg.tool_calls[0].function.name
                correct_tool = tool_name == sc["expected_tool"]
                # Also check args are valid JSON
                try:
                    json.loads(msg.tool_calls[0].function.arguments)
                    valid_json = True
                except (json.JSONDecodeError, TypeError):
                    valid_json = False
            else:
                valid_json = False

            passed = has_tool_calls and correct_tool and valid_json
            if passed:
                correct += 1

            details.append(
                {
                    "prompt": sc["prompt"][:50],
                    "expected": sc["expected_tool"],
                    "got": tool_name,
                    "has_tool_calls": has_tool_calls,
                    "correct_tool": correct_tool,
                    "valid_json": valid_json,
                    "passed": passed,
                }
            )

            status = "✓" if passed else "✗"
            got_str = tool_name or "(text)"
            print(
                f"    {status} {sc['prompt'][:45]:.<48} expected={sc['expected_tool']:<12} got={got_str}"
            )

        except Exception as e:
            details.append(
                {"prompt": sc["prompt"][:50], "error": str(e), "passed": False}
            )
            print(f"    ✗ {sc['prompt'][:45]:.<48} ERROR: {e}")

    rate = correct / total if total > 0 else 0
    print(f"    Result: {correct}/{total} ({rate:.0%})")
    return {
        "success_rate": rate,
        "correct": correct,
        "total": total,
        "details": details,
    }


def benchmark_tool_recovery(client, model: str) -> dict:
    """Test if server can recover degraded text-format tool calls.

    Sends multi-round conversations where the model is likely to degrade
    and output tool calls as plain text instead of structured format.
    Recovery = server auto-detects and converts back to structured tool_calls.
    """
    print("  Tool call recovery test (multi-round degradation)...")

    # Simulate a multi-turn conversation that pushes the model toward degradation
    # by providing several rounds of tool use context
    recovery_scenarios = [
        {
            "name": "8-round agent session",
            "messages": _build_long_agent_conversation(8),
        },
        {
            "name": "complex multi-tool",
            "messages": _build_complex_tool_conversation(),
        },
    ]

    recovered = 0
    total = 0

    for sc in recovery_scenarios:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=sc["messages"],
                tools=BENCHMARK_TOOLS,
                max_tokens=500,
                temperature=0.0,
            )
            msg = resp.choices[0].message
            total += 1

            # Check if tool call was returned in structured format
            has_structured = bool(msg.tool_calls)
            # Check if content contains text-format tool calls (degraded)
            content = msg.content or ""
            has_text_tool = any(
                marker in content
                for marker in [
                    '"name":',
                    "function_call",
                    "<tool_call>",
                    "<|tool_call|>",
                    "get_weather(",
                    "web_search(",
                    "run_python(",
                ]
            )

            if has_structured:
                recovered += 1
                print(f"    ✓ {sc['name']}: structured tool_calls returned")
            elif has_text_tool:
                print(
                    f"    ✗ {sc['name']}: degraded text-format tool call (no recovery)"
                )
            else:
                # Model gave a text response, not a tool call scenario
                print(f"    ~ {sc['name']}: text response (not applicable)")
                total -= 1  # Don't count N/A

        except Exception as e:
            print(f"    ✗ {sc['name']}: ERROR {e}")
            total += 1

    rate = recovered / total if total > 0 else 0
    print(f"    Result: {recovered}/{total} ({rate:.0%})")
    return {"recovery_rate": rate, "recovered": recovered, "total": total}


def _build_long_agent_conversation(rounds: int) -> list[dict]:
    """Build a synthetic multi-round agent conversation."""
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Use the provided tools when needed.",
        },
    ]
    prompts_and_results = [
        (
            "What's the weather in Tokyo?",
            "get_weather",
            '{"location":"Tokyo"}',
            "Partly cloudy, 18°C",
        ),
        (
            "Search for Python async tutorials",
            "web_search",
            '{"query":"Python async tutorial"}',
            "Found 10 results about asyncio...",
        ),
        ("Run print(2**10)", "run_python", '{"code":"print(2**10)"}', "1024"),
        ("Read /etc/hostname", "read_file", '{"path":"/etc/hostname"}', "my-server"),
        (
            "What's the weather in London?",
            "get_weather",
            '{"location":"London"}',
            "Rainy, 12°C",
        ),
        (
            "Search for MLX benchmarks",
            "web_search",
            '{"query":"MLX benchmarks"}',
            "Apple MLX framework shows 2-3x improvement...",
        ),
        (
            "Run print(sum(range(100)))",
            "run_python",
            '{"code":"print(sum(range(100)))"}',
            "4950",
        ),
        (
            "Read /tmp/config.json",
            "read_file",
            '{"path":"/tmp/config.json"}',
            '{"debug": true}',
        ),
    ]

    for i in range(min(rounds, len(prompts_and_results))):
        prompt, tool_name, args, result = prompts_and_results[i]
        messages.append({"role": "user", "content": prompt})
        messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"call_{i}",
                        "type": "function",
                        "function": {"name": tool_name, "arguments": args},
                    }
                ],
            }
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": f"call_{i}",
                "content": result,
            }
        )
        messages.append(
            {
                "role": "assistant",
                "content": f"Based on the result: {result[:50]}...",
            }
        )

    # Final prompt that should trigger another tool call
    messages.append({"role": "user", "content": "Now check the weather in Berlin"})
    return messages


def _build_complex_tool_conversation() -> list[dict]:
    """Build a conversation with complex arguments likely to cause degradation."""
    return [
        {
            "role": "system",
            "content": "You are a coding assistant. Use tools when needed.",
        },
        {
            "role": "user",
            "content": "Run this Python code:\nimport json\ndata = {'key': 'value', 'nested': {'a': [1,2,3]}}\nprint(json.dumps(data, indent=2))",
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "run_python",
                        "arguments": "{\"code\":\"import json\\ndata = {'key': 'value', 'nested': {'a': [1,2,3]}}\\nprint(json.dumps(data, indent=2))\"}",
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": '{\n  "key": "value",\n  "nested": {\n    "a": [1, 2, 3]\n  }\n}',
        },
        {
            "role": "assistant",
            "content": "The code ran successfully and output the formatted JSON.",
        },
        {
            "role": "user",
            "content": "Now search the web for 'how to parse nested JSON in Python with error handling'",
        },
    ]


def benchmark_leak_rate(client, model: str) -> dict:
    """Test think-tag leak rate — does <think> content leak into response content?

    Sends prompts that encourage chain-of-thought, then checks if any
    <think>...</think> tags appear in the content field.
    """
    print(f"  Think-tag leak rate ({len(THINK_PROMPTS)} prompts)...")
    leaks = 0
    total = len(THINK_PROMPTS)
    details = []

    for prompt in THINK_PROMPTS:
        try:
            # Use streaming to test real-world behavior
            content_parts = []
            reasoning_parts = []

            stream = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                stream=True,
                temperature=0.7,
            )
            for chunk in stream:
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        content_parts.append(delta.content)
                    reasoning = getattr(delta, "reasoning", None) or getattr(
                        delta, "reasoning_content", None
                    )
                    if reasoning:
                        reasoning_parts.append(reasoning)

            content = "".join(content_parts)
            reasoning = "".join(reasoning_parts)

            # Check for leaked think tags in content
            has_leak = "<think>" in content or "</think>" in content
            has_reasoning_field = bool(reasoning)

            if has_leak:
                leaks += 1
                print(f"    ✗ LEAK: {prompt[:50]:.<55} <think> found in content")
            else:
                print(
                    f"    ✓ Clean: {prompt[:50]:.<55} {'(reasoning separated)' if has_reasoning_field else '(no thinking)'}"
                )

            details.append(
                {
                    "prompt": prompt[:50],
                    "leaked": has_leak,
                    "has_reasoning_field": has_reasoning_field,
                    "content_len": len(content),
                    "reasoning_len": len(reasoning),
                }
            )

        except Exception as e:
            details.append({"prompt": prompt[:50], "error": str(e), "leaked": False})
            print(f"    ? {prompt[:50]:.<55} ERROR: {e}")

    leak_rate = leaks / total if total > 0 else 0
    print(f"    Result: {leaks}/{total} leaked ({leak_rate:.0%} leak rate)")
    return {"leak_rate": leak_rate, "leaks": leaks, "total": total, "details": details}


def benchmark_multimodal(client, model: str) -> dict:
    """Test multimodal support — vision, audio."""
    print("  Multimodal support check...")
    result = {"vision": False, "audio": False}

    # Test vision: send a tiny 1x1 white PNG
    TINY_PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What do you see in this image? Reply in one word.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{TINY_PNG_B64}"
                            },
                        },
                    ],
                }
            ],
            max_tokens=50,
        )
        # If we get here without error, vision is supported
        content = resp.choices[0].message.content or ""
        if content and "error" not in content.lower():
            result["vision"] = True
            print(f"    ✓ Vision: supported (response: {content[:40]})")
        else:
            print("    ✗ Vision: not supported (error response)")
    except Exception as e:
        err = str(e)[:80]
        print(f"    ✗ Vision: not supported ({err})")

    # Test audio: check if /v1/audio endpoint exists
    try:
        import urllib.request

        base = client.base_url.rstrip("/").replace("/v1", "")
        req = urllib.request.Request(f"{base}/v1/audio/speech", method="POST")
        req.add_header("Content-Type", "application/json")
        # Just check if endpoint exists (will get a 4xx, not 404)
        try:
            urllib.request.urlopen(req, b"{}", timeout=3)
        except urllib.error.HTTPError as e:
            if e.code != 404:
                result["audio"] = True
                print(f"    ✓ Audio: endpoint exists (HTTP {e.code})")
            else:
                print("    ✗ Audio: not supported (404)")
        except Exception:
            print("    ✗ Audio: not supported (connection error)")
    except Exception as e:
        print(f"    ✗ Audio: not supported ({e})")

    return result


# ---------------------------------------------------------------------------
# Full engine benchmark
# ---------------------------------------------------------------------------


def benchmark_openai_engine(
    base_url: str,
    model: str,
    engine_name: str,
    num_runs: int = 3,
    max_tokens_short: int = 200,
    max_tokens_long: int = 500,
    port: int | None = None,
    skip_capability: bool = False,
) -> dict | None:
    """Full benchmark for an OpenAI-compatible engine."""
    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: openai package not installed. pip install openai")
        return None

    client = OpenAI(base_url=base_url, api_key="not-needed")
    try:
        client.models.list()
    except Exception as e:
        print(f"  ERROR: Cannot reach {base_url} — {e}")
        return None

    results = {"engine": engine_name, "model": model, "peak_ram_mb": None}

    # --- Speed benchmarks ---
    speed = benchmark_speed(client, model, num_runs, max_tokens_short, max_tokens_long)
    results.update(speed)

    # --- RAM ---
    if port:
        ram = get_process_memory_mb(port)
        if ram:
            results["peak_ram_mb"] = ram
            print(f"  Process RAM: {ram:.0f} MB")

    # --- Capability benchmarks ---
    if not skip_capability:
        print()
        results["tool_calls"] = benchmark_tool_calls(client, model)
        print()
        results["tool_recovery"] = benchmark_tool_recovery(client, model)
        print()
        results["leak"] = benchmark_leak_rate(client, model)
        print()
        results["multimodal"] = benchmark_multimodal(client, model)

    return results


# ---------------------------------------------------------------------------
# mlx-lm direct benchmark
# ---------------------------------------------------------------------------


def benchmark_mlx_lm_direct(
    model_path: str, num_runs: int, max_tokens_short: int, max_tokens_long: int
) -> dict | None:
    """Benchmark mlx-lm directly (no server, no capability tests)."""
    try:
        import mlx_lm
    except ImportError:
        print("ERROR: mlx-lm not installed. pip install mlx-lm")
        return None

    print(f"  Loading model {model_path}...")
    try:
        model, tokenizer = mlx_lm.load(model_path)
    except ValueError:
        try:
            model, tokenizer = mlx_lm.load(model_path, strict=False)
        except TypeError:
            print("  ERROR: Cannot load model. Skipping.")
            return None

    results = {"engine": "mlx-lm", "model": model_path, "short_gen": [], "long_gen": []}

    for label, prompt, max_tok in [
        ("Short", SHORT_PROMPT, max_tokens_short),
        ("Long", LONG_PROMPT, max_tokens_long),
    ]:
        key = "short_gen" if label == "Short" else "long_gen"
        print(f"  {label} generation ({max_tok} tokens, {num_runs} runs)...")
        for i in range(num_runs):
            start = time.perf_counter()
            first_token_time = None
            token_count = 0
            for _ in mlx_lm.stream_generate(
                model, tokenizer, prompt=prompt, max_tokens=max_tok
            ):
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                token_count += 1
            elapsed = time.perf_counter() - start
            ttft = first_token_time - start if first_token_time else elapsed
            decode_time = elapsed - ttft if first_token_time else elapsed
            tps = token_count / decode_time if decode_time > 0 else 0
            results[key].append(
                {"tokens": token_count, "elapsed": elapsed, "ttft": ttft, "tps": tps}
            )
            print(
                f"    Run {i + 1}: {tps:.1f} tok/s, TTFT {ttft:.3f}s, {token_count} tokens"
            )

    # mlx-lm has no tool calling, no recovery, no leak filtering, no multimodal API
    results["tool_calls"] = {"success_rate": 0, "correct": 0, "total": 0, "details": []}
    results["tool_recovery"] = {"recovery_rate": 0, "recovered": 0, "total": 0}
    results["leak"] = {"leak_rate": 0, "leaks": 0, "total": 0, "details": []}
    results["multimodal"] = {"vision": False, "audio": False}

    return results


# ---------------------------------------------------------------------------
# Summary & comparison
# ---------------------------------------------------------------------------


def summarize(results: dict) -> dict:
    """Compute summary statistics."""
    s = {"engine": results["engine"], "model": results["model"]}

    for key, label in [
        ("short_gen", "short_decode_tps"),
        ("long_gen", "long_decode_tps"),
    ]:
        if results.get(key):
            tps_vals = [r["tps"] for r in results[key]]
            s[label] = {
                "mean": statistics.mean(tps_vals),
                "median": statistics.median(tps_vals),
                "min": min(tps_vals),
                "max": max(tps_vals),
            }
            prefill_vals = [
                r["prefill_tps"] for r in results[key] if r.get("prefill_tps")
            ]
            if prefill_vals:
                s[label.replace("decode", "prefill")] = {
                    "median": statistics.median(prefill_vals)
                }

    if results.get("ttft_cold"):
        s["ttft_cold_s"] = statistics.mean(results["ttft_cold"])
    if results.get("ttft_cached"):
        s["ttft_cached_s"] = statistics.mean(results["ttft_cached"])
    if results.get("multi_turn_ttft"):
        vals = results["multi_turn_ttft"]
        s["multi_turn_ttft_cold_s"] = vals[0]
        if len(vals) > 1:
            s["multi_turn_ttft_cached_s"] = statistics.mean(vals[1:])

    if results.get("peak_ram_mb"):
        s["peak_ram_mb"] = results["peak_ram_mb"]

    # Capability metrics
    if results.get("tool_calls"):
        s["tool_call_rate"] = results["tool_calls"]["success_rate"]
    if results.get("tool_recovery"):
        s["recovery_rate"] = results["tool_recovery"]["recovery_rate"]
    if results.get("leak"):
        s["leak_rate"] = results["leak"]["leak_rate"]
    if results.get("multimodal"):
        s["vision"] = results["multimodal"].get("vision", False)
        s["audio"] = results["multimodal"].get("audio", False)

    return s


def print_summary(summary: dict):
    """Pretty-print benchmark summary."""
    print(f"\n{'=' * 65}")
    print(f"  {summary['engine']} — {summary['model']}")
    print(f"{'=' * 65}")

    if "short_decode_tps" in summary:
        d = summary["short_decode_tps"]
        print(
            f"  Short decode:  {d['median']:.1f} tok/s (median), range {d['min']:.1f}-{d['max']:.1f}"
        )
    if "long_decode_tps" in summary:
        d = summary["long_decode_tps"]
        print(
            f"  Long decode:   {d['median']:.1f} tok/s (median), range {d['min']:.1f}-{d['max']:.1f}"
        )
    if "short_prefill_tps" in summary:
        print(f"  Prefill:       {summary['short_prefill_tps']['median']:.0f} tok/s")
    if "ttft_cold_s" in summary:
        print(f"  TTFT (cold):   {summary['ttft_cold_s']:.3f}s")
    if "ttft_cached_s" in summary:
        print(f"  TTFT (cached): {summary['ttft_cached_s']:.3f}s")
    if "multi_turn_ttft_cold_s" in summary:
        print(f"  MT TTFT (cold):   {summary['multi_turn_ttft_cold_s']:.3f}s")
    if "multi_turn_ttft_cached_s" in summary:
        print(f"  MT TTFT (cached): {summary['multi_turn_ttft_cached_s']:.3f}s")
    if "peak_ram_mb" in summary:
        r = summary["peak_ram_mb"]
        print(f"  Peak RAM:      {r:.0f} MB ({r / 1024:.1f} GB)")
    if "tool_call_rate" in summary:
        print(f"  Tool call:     {summary['tool_call_rate']:.0%}")
    if "recovery_rate" in summary:
        print(f"  Recovery:      {summary['recovery_rate']:.0%}")
    if "leak_rate" in summary:
        print(f"  Leak rate:     {summary['leak_rate']:.0%}")
    if "vision" in summary:
        v = "✓" if summary["vision"] else "✗"
        a = "✓" if summary.get("audio") else "✗"
        print(f"  Multimodal:    Vision {v}  Audio {a}")
    print()


def print_comparison(all_summaries: list[dict]):
    """Print the full comparison table."""
    print(f"\n{'=' * 120}")
    print("  FULL COMPARISON")
    print(f"{'=' * 120}")

    h1 = (
        f"{'Engine':<16} {'Decode':>8} {'TTFT':>8} {'TTFT':>8} {'MT TTFT':>8}"
        f" {'RAM':>8} {'Tool':>6} {'Recov':>6} {'Leak':>6} {'Vis':>4} {'Aud':>4}"
    )
    h2 = (
        f"{'':.<16} {'tok/s':>8} {'cold':>8} {'cached':>8} {'cached':>8}"
        f" {'(GB)':>8} {'%':>6} {'%':>6} {'%':>6} {'':>4} {'':>4}"
    )
    print(h1)
    print(h2)
    print("-" * 120)

    for s in all_summaries:
        decode = s.get("short_decode_tps", {}).get("median", 0)
        cold = s.get("ttft_cold_s", 0)
        cached = s.get("ttft_cached_s", 0)
        mt = s.get("multi_turn_ttft_cached_s", 0)
        ram = s.get("peak_ram_mb", 0) / 1024 if s.get("peak_ram_mb") else 0
        tool = s.get("tool_call_rate", 0)
        recov = s.get("recovery_rate", 0)
        leak = s.get("leak_rate", 0)
        vis = "✓" if s.get("vision") else "✗"
        aud = "✓" if s.get("audio") else "✗"

        ram_str = f"{ram:>8.1f}" if ram else f"{'—':>8}"

        print(
            f"{s['engine']:<16} {decode:>8.1f} {cold:>8.3f} {cached:>8.3f} {mt:>8.3f}"
            f" {ram_str} {tool:>5.0%} {recov:>5.0%} {leak:>5.0%} {vis:>4} {aud:>4}"
        )

    # Speedup row
    if len(all_summaries) >= 2:
        base = all_summaries[0]
        print("-" * 120)
        for s in all_summaries[1:]:
            b_d = base.get("short_decode_tps", {}).get("median", 1)
            s_d = s.get("short_decode_tps", {}).get("median", 1)
            dx = b_d / s_d if s_d > 0 else 0

            b_mt = base.get("multi_turn_ttft_cached_s", 1)
            s_mt = s.get("multi_turn_ttft_cached_s", 1)
            mt_x = s_mt / b_mt if b_mt > 0 else 0

            label = f"vs {s['engine']}"
            print(f"{label:<16} {dx:>7.1f}x {'':>8} {'':>8} {mt_x:>7.1f}x")


# ---------------------------------------------------------------------------
# Engine configs
# ---------------------------------------------------------------------------

ENGINE_CONFIGS = {
    "rapid-mlx": {"display": "Rapid-MLX", "default_port": 8000},
    "ollama": {"display": "Ollama", "default_port": 11434},
    "llama-cpp": {"display": "llama.cpp", "default_port": 8080},
    "mlx-lm": {"display": "mlx-lm", "default_port": None},
}
ALL_OPENAI_ENGINES = ["rapid-mlx", "ollama", "llama-cpp"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark inference engines (speed + capability)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/benchmark_engines.py --engine rapid-mlx
  python scripts/benchmark_engines.py --engine rapid-mlx ollama llama-cpp
  python scripts/benchmark_engines.py --engine all --output results.json
  python scripts/benchmark_engines.py --engine rapid-mlx --speed-only
""",
    )
    parser.add_argument(
        "--engine", nargs="+", choices=list(ENGINE_CONFIGS) + ["all"], required=True
    )
    parser.add_argument(
        "--model", default="default", help="Model name (default: 'default')"
    )
    parser.add_argument("--rapid-mlx-port", type=int, default=8000)
    parser.add_argument("--ollama-port", type=int, default=11434)
    parser.add_argument("--llama-cpp-port", type=int, default=8080)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--max-tokens-short", type=int, default=200)
    parser.add_argument("--max-tokens-long", type=int, default=500)
    parser.add_argument(
        "--speed-only", action="store_true", help="Skip capability tests"
    )
    parser.add_argument(
        "--capability-only", action="store_true", help="Skip speed tests"
    )
    parser.add_argument("--output", help="Save results to JSON file")
    parser.add_argument("--port", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    engines = args.engine
    if "all" in engines:
        engines = list(ALL_OPENAI_ENGINES) + ["mlx-lm"]

    if args.port and args.rapid_mlx_port == 8000:
        args.rapid_mlx_port = args.port

    port_map = {
        "rapid-mlx": args.rapid_mlx_port,
        "ollama": args.ollama_port,
        "llama-cpp": args.llama_cpp_port,
    }

    all_summaries = []

    for engine in engines:
        cfg = ENGINE_CONFIGS[engine]

        if engine == "mlx-lm":
            print("\n>>> Benchmarking mlx-lm (direct)...")
            results = benchmark_mlx_lm_direct(
                args.model,
                args.runs,
                args.max_tokens_short,
                args.max_tokens_long,
            )
        else:
            port = port_map[engine]
            print(f"\n>>> Benchmarking {cfg['display']} (port {port})...")
            results = benchmark_openai_engine(
                f"http://localhost:{port}/v1",
                args.model,
                engine_name=cfg["display"],
                num_runs=args.runs,
                max_tokens_short=args.max_tokens_short,
                max_tokens_long=args.max_tokens_long,
                port=port,
                skip_capability=args.speed_only,
            )

        if results:
            s = summarize(results)
            print_summary(s)
            all_summaries.append(s)

    if len(all_summaries) > 1:
        print_comparison(all_summaries)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_summaries, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
