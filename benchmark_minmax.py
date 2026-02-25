#!/usr/bin/env python3
"""
Comprehensive benchmark for vllm-mlx + MiniMax-M2.5 on M3 Ultra.

Measures 6 dimensions that map to our optimization tiers:
  1. TTFT (Time To First Token) — prefill speed
  2. Decode throughput (tok/s) — generation speed
  3. Prefix cache effectiveness — multi-turn speedup
  4. Tool calling latency + correctness — agentic reliability
  5. Reasoning separation — thinking vs content split
  6. Long context stability — sustained generation without crash

Usage:
  # Start server first:
  python -m vllm_mlx.cli serve <model> --port 8000 ...

  # Run all benchmarks:
  python benchmark_minmax.py

  # Run specific test:
  python benchmark_minmax.py --test ttft
  python benchmark_minmax.py --test decode
  python benchmark_minmax.py --test prefix_cache
  python benchmark_minmax.py --test tool_call
  python benchmark_minmax.py --test reasoning
  python benchmark_minmax.py --test long_gen

  # Save results to JSON:
  python benchmark_minmax.py --output results.json
"""

import argparse
import json
import statistics
import time

from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
MODEL = None  # auto-detected


def detect_model():
    """Auto-detect the model name from the server."""
    global MODEL
    try:
        models = client.models.list()
        MODEL = models.data[0].id
        print(f"Detected model: {MODEL}")
    except Exception:
        MODEL = "MiniMax-M2.5-MLX-4bit"
        print(f"Could not detect model, using default: {MODEL}")


def stream_and_measure(messages, max_tokens=512, temperature=0.7, tools=None):
    """Core measurement function. Returns detailed metrics."""
    kwargs = dict(
        model=MODEL,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=True,
        stream_options={"include_usage": True},
    )
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    t_start = time.perf_counter()
    response = client.chat.completions.create(**kwargs)

    t_first_token = None
    chunks = 0
    content_parts = []
    reasoning_parts = []
    tool_calls_received = []
    usage = None
    finish_reason = None

    for chunk in response:
        if chunk.usage:
            usage = chunk.usage

        if not chunk.choices:
            continue

        delta = chunk.choices[0].delta
        fr = chunk.choices[0].finish_reason
        if fr:
            finish_reason = fr

        has_content = delta.content is not None and len(delta.content) > 0
        has_reasoning = hasattr(delta, "reasoning") and delta.reasoning
        has_tool = hasattr(delta, "tool_calls") and delta.tool_calls

        if has_content or has_reasoning or has_tool:
            if t_first_token is None:
                t_first_token = time.perf_counter()
            chunks += 1

        if has_content:
            content_parts.append(delta.content)
        if has_reasoning:
            reasoning_parts.append(delta.reasoning)
        if has_tool:
            tool_calls_received.extend(delta.tool_calls)

    t_end = time.perf_counter()

    total_time = t_end - t_start
    ttft = (t_first_token - t_start) if t_first_token else total_time
    decode_time = (t_end - t_first_token) if t_first_token else 0

    prompt_tokens = usage.prompt_tokens if usage else 0
    completion_tokens = usage.completion_tokens if usage else chunks

    decode_tps = completion_tokens / decode_time if decode_time > 0 else 0

    return {
        "ttft": round(ttft, 3),
        "total_time": round(total_time, 3),
        "decode_time": round(decode_time, 3),
        "decode_tps": round(decode_tps, 1),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "content": "".join(content_parts),
        "reasoning": "".join(reasoning_parts),
        "tool_calls": tool_calls_received,
        "finish_reason": finish_reason,
        "chunks": chunks,
    }


# ---------------------------------------------------------------------------
# Test 1: TTFT — Time To First Token (prefill speed)
# ---------------------------------------------------------------------------
def test_ttft():
    """Measure TTFT across different prompt sizes."""
    print("\n" + "=" * 70)
    print(" TEST 1: TTFT (Time To First Token)")
    print("=" * 70)

    prompts = [
        ("Short (50 tok)", "What is 2+2? Answer in one sentence."),
        (
            "Medium (500 tok)",
            "You are a helpful AI assistant. "
            + "Please explain the complete history of computing, "
            + "from Charles Babbage to modern GPUs. " * 15
            + "\nSummarize in 3 bullet points.",
        ),
        (
            "Long (2K tok)",
            "You are a senior software architect. "
            + "Review this code and explain every design pattern used:\n"
            + '```python\nclass AbstractFactory:\n    def create(self): pass\n```\n' * 40
            + "\nList all patterns found.",
        ),
    ]

    results = []
    for label, prompt in prompts:
        # Warm-up run (fills cache)
        _ = stream_and_measure(
            [{"role": "user", "content": prompt}], max_tokens=32
        )
        # Measured run
        m = stream_and_measure(
            [{"role": "user", "content": prompt}], max_tokens=32
        )
        results.append({"label": label, **m})
        print(
            f"  {label:20s}  TTFT={m['ttft']:.3f}s  "
            f"prompt_tok={m['prompt_tokens']}  "
            f"decode={m['decode_tps']:.1f} tok/s"
        )

    return {"test": "ttft", "results": results}


# ---------------------------------------------------------------------------
# Test 2: Decode throughput (sustained generation speed)
# ---------------------------------------------------------------------------
def test_decode():
    """Measure sustained decode speed at different output lengths."""
    print("\n" + "=" * 70)
    print(" TEST 2: Decode Throughput")
    print("=" * 70)

    targets = [128, 512, 2048]
    results = []

    for max_tok in targets:
        m = stream_and_measure(
            [
                {
                    "role": "user",
                    "content": f"Write exactly {max_tok} words about the history of artificial intelligence. Be detailed.",
                }
            ],
            max_tokens=max_tok,
            temperature=0.7,
        )
        results.append({"max_tokens": max_tok, **m})
        print(
            f"  max_tokens={max_tok:5d}  "
            f"generated={m['completion_tokens']:5d} tok  "
            f"decode={m['decode_tps']:.1f} tok/s  "
            f"total={m['total_time']:.1f}s"
        )

    return {"test": "decode_throughput", "results": results}


# ---------------------------------------------------------------------------
# Test 3: Prefix cache effectiveness (multi-turn simulation)
# ---------------------------------------------------------------------------
def test_prefix_cache():
    """Simulate multi-turn conversation to measure cache hit benefits."""
    print("\n" + "=" * 70)
    print(" TEST 3: Prefix Cache (Multi-Turn)")
    print("=" * 70)

    system_prompt = (
        "You are an expert AI coding assistant with deep knowledge of Python, "
        "TypeScript, Rust, and system design. You always provide working code "
        "examples and explain your reasoning step by step. You have access to "
        "tools for file manipulation, web search, and code execution. "
        "Always consider edge cases, error handling, and performance implications."
    )

    # Build up conversation turns
    turns = [
        "How do I implement a thread-safe LRU cache in Python?",
        "Can you add TTL (time-to-live) expiration to that cache?",
        "Now make it work with async/await. Show the full implementation.",
        "What are the performance characteristics? Add benchmarks.",
    ]

    messages = [{"role": "system", "content": system_prompt}]
    results = []

    for i, user_msg in enumerate(turns):
        messages.append({"role": "user", "content": user_msg})

        m = stream_and_measure(messages, max_tokens=256, temperature=0.7)

        results.append({"turn": i + 1, "user_msg": user_msg[:60], **m})
        print(
            f"  Turn {i+1}: TTFT={m['ttft']:.3f}s  "
            f"prompt={m['prompt_tokens']} tok  "
            f"decode={m['decode_tps']:.1f} tok/s"
        )

        # Add assistant response to history for next turn
        messages.append({"role": "assistant", "content": m["content"][:500]})

    # Key metric: TTFT should decrease or stay flat despite growing context
    ttfts = [r["ttft"] for r in results]
    print(f"\n  TTFT trend: {' → '.join(f'{t:.3f}s' for t in ttfts)}")
    if len(ttfts) >= 2:
        ratio = ttfts[-1] / ttfts[0] if ttfts[0] > 0 else 0
        print(f"  Turn 4 / Turn 1 TTFT ratio: {ratio:.2f}x")

    return {"test": "prefix_cache", "results": results}


# ---------------------------------------------------------------------------
# Test 4: Tool calling latency + correctness
# ---------------------------------------------------------------------------
def test_tool_call():
    """Measure tool call generation speed and JSON validity."""
    print("\n" + "=" * 70)
    print(" TEST 4: Tool Calling (Latency + Correctness)")
    print("=" * 70)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City name, e.g. San Francisco",
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                        },
                    },
                    "required": ["location"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_web",
                "description": "Search the web for information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "num_results": {
                            "type": "integer",
                            "description": "Number of results",
                        },
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
                        "code": {
                            "type": "string",
                            "description": "Python code to execute",
                        },
                    },
                    "required": ["code"],
                },
            },
        },
    ]

    test_prompts = [
        ("Simple tool call", "What's the weather in Tokyo?"),
        ("Multi-arg tool call", "Search for 'vLLM MLX benchmarks' and show me 5 results"),
        (
            "Code execution",
            "Calculate the fibonacci sequence up to n=20 using Python",
        ),
        (
            "Tool selection",
            "I need to know the weather in Paris and also search for restaurants there",
        ),
    ]

    results = []
    correct = 0
    total = 0

    for label, prompt in test_prompts:
        total += 1
        m = stream_and_measure(
            [{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.3,
            tools=tools,
        )

        # Check correctness
        has_tool = m["finish_reason"] == "tool_calls" or len(m["tool_calls"]) > 0
        json_valid = True

        if has_tool and m["tool_calls"]:
            for tc in m["tool_calls"]:
                if hasattr(tc, "function") and tc.function:
                    try:
                        args = tc.function.arguments
                        if isinstance(args, str):
                            json.loads(args)
                    except (json.JSONDecodeError, AttributeError):
                        json_valid = False
                        break

        is_correct = has_tool and json_valid
        if is_correct:
            correct += 1

        tool_names = []
        if m["tool_calls"]:
            for tc in m["tool_calls"]:
                if hasattr(tc, "function") and tc.function:
                    tool_names.append(tc.function.name)

        results.append(
            {
                "label": label,
                "has_tool_call": has_tool,
                "json_valid": json_valid,
                "correct": is_correct,
                "tool_names": tool_names,
                **m,
            }
        )

        status = "OK" if is_correct else "FAIL"
        tools_str = ", ".join(tool_names) if tool_names else "none"
        print(
            f"  [{status}] {label:25s}  TTFT={m['ttft']:.3f}s  "
            f"total={m['total_time']:.2f}s  tools=[{tools_str}]"
        )

    accuracy = correct / total * 100 if total > 0 else 0
    print(f"\n  Tool call accuracy: {correct}/{total} ({accuracy:.0f}%)")

    return {
        "test": "tool_call",
        "accuracy_pct": accuracy,
        "correct": correct,
        "total": total,
        "results": results,
    }


# ---------------------------------------------------------------------------
# Test 5: Reasoning separation (thinking vs content)
# ---------------------------------------------------------------------------
def test_reasoning():
    """Test that reasoning content is properly separated from final answer."""
    print("\n" + "=" * 70)
    print(" TEST 5: Reasoning Separation")
    print("=" * 70)

    prompts = [
        ("Math", "What is 17 * 23? Think step by step."),
        ("Logic", "If all roses are flowers and some flowers fade quickly, can we conclude all roses fade quickly? Explain your reasoning."),
        ("Code", "What's wrong with this code: `x = [1,2,3]; x[5]`? Think carefully."),
    ]

    results = []
    for label, prompt in prompts:
        m = stream_and_measure(
            [{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.3,
        )

        has_reasoning = len(m["reasoning"]) > 0
        has_content = len(m["content"]) > 0
        separated = has_reasoning and has_content

        results.append(
            {
                "label": label,
                "has_reasoning": has_reasoning,
                "has_content": has_content,
                "separated": separated,
                "reasoning_chars": len(m["reasoning"]),
                "content_chars": len(m["content"]),
                **m,
            }
        )

        status = "OK" if separated else ("PARTIAL" if has_reasoning or has_content else "FAIL")
        print(
            f"  [{status:7s}] {label:10s}  "
            f"reasoning={len(m['reasoning']):5d} chars  "
            f"content={len(m['content']):5d} chars  "
            f"TTFT={m['ttft']:.3f}s"
        )

    separated_count = sum(1 for r in results if r["separated"])
    print(f"\n  Properly separated: {separated_count}/{len(results)}")

    return {"test": "reasoning", "results": results}


# ---------------------------------------------------------------------------
# Test 6: Long generation stability
# ---------------------------------------------------------------------------
def test_long_gen():
    """Test sustained long generation without crash."""
    print("\n" + "=" * 70)
    print(" TEST 6: Long Generation Stability")
    print("=" * 70)

    m = stream_and_measure(
        [
            {
                "role": "user",
                "content": (
                    "Write a comprehensive 5000-word technical guide about building "
                    "a distributed system on Apple Silicon. Cover: architecture design, "
                    "networking, consensus algorithms, fault tolerance, and monitoring. "
                    "Use code examples and diagrams described in text. "
                    "Start writing immediately."
                ),
            }
        ],
        max_tokens=8192,
        temperature=0.7,
    )

    completed = m["finish_reason"] in ("stop", "length")
    print(f"  Completed: {completed}  finish_reason={m['finish_reason']}")
    print(
        f"  Generated: {m['completion_tokens']} tokens in {m['total_time']:.1f}s"
    )
    print(f"  Decode speed: {m['decode_tps']:.1f} tok/s")
    print(f"  Output length: {len(m['content'])} chars")

    return {
        "test": "long_generation",
        "completed": completed,
        **m,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def print_summary(all_results):
    """Print a compact summary table."""
    print("\n" + "=" * 70)
    print(" SUMMARY")
    print("=" * 70)

    for r in all_results:
        test = r["test"]

        if test == "ttft":
            ttfts = [x["ttft"] for x in r["results"]]
            print(f"  TTFT:           {' / '.join(f'{t:.3f}s' for t in ttfts)}  (short/med/long)")

        elif test == "decode_throughput":
            tps_list = [x["decode_tps"] for x in r["results"]]
            print(f"  Decode tok/s:   {' / '.join(f'{t:.1f}' for t in tps_list)}  (128/512/2048 tokens)")

        elif test == "prefix_cache":
            ttfts = [x["ttft"] for x in r["results"]]
            ratio = ttfts[-1] / ttfts[0] if ttfts[0] > 0 else 0
            print(f"  Prefix cache:   Turn1={ttfts[0]:.3f}s → Turn4={ttfts[-1]:.3f}s  (ratio={ratio:.2f}x)")

        elif test == "tool_call":
            print(f"  Tool calling:   {r['correct']}/{r['total']} correct ({r['accuracy_pct']:.0f}%)")
            avg_time = statistics.mean(x["total_time"] for x in r["results"])
            print(f"                  avg latency={avg_time:.2f}s")

        elif test == "reasoning":
            sep = sum(1 for x in r["results"] if x["separated"])
            print(f"  Reasoning:      {sep}/{len(r['results'])} properly separated")

        elif test == "long_generation":
            print(
                f"  Long gen:       {'PASS' if r['completed'] else 'FAIL'}  "
                f"{r['completion_tokens']} tok @ {r['decode_tps']:.1f} tok/s"
            )

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="vllm-mlx MiniMax benchmark")
    parser.add_argument(
        "--test",
        choices=["ttft", "decode", "prefix_cache", "tool_call", "reasoning", "long_gen", "all"],
        default="all",
        help="Which test to run (default: all)",
    )
    parser.add_argument("--output", type=str, help="Save results to JSON file")
    args = parser.parse_args()

    detect_model()

    test_map = {
        "ttft": test_ttft,
        "decode": test_decode,
        "prefix_cache": test_prefix_cache,
        "tool_call": test_tool_call,
        "reasoning": test_reasoning,
        "long_gen": test_long_gen,
    }

    if args.test == "all":
        tests_to_run = list(test_map.keys())
    else:
        tests_to_run = [args.test]

    all_results = []
    for test_name in tests_to_run:
        try:
            result = test_map[test_name]()
            all_results.append(result)
        except Exception as e:
            print(f"\n  ERROR in {test_name}: {e}")
            all_results.append({"test": test_name, "error": str(e)})

    print_summary(all_results)

    # Save results
    output_file = args.output or f"benchmark_results_{int(time.time())}.json"

    def _serialize(obj):
        """Handle non-serializable objects."""
        if hasattr(obj, "__dict__"):
            return str(obj)
        return repr(obj)

    with open(output_file, "w") as f:
        json.dump(
            {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "model": MODEL,
                "results": all_results,
            },
            f,
            indent=2,
            default=_serialize,
            ensure_ascii=False,
        )
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
