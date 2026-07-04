#!/usr/bin/env python3
"""Benchmark DeltaNet state snapshot TTFT improvement.

Sends repeated requests with the same system prompt to measure how much
the RNN state snapshot reduces time-to-first-token on hybrid cache models.

Usage:
    # Start server first:
    python3.12 -m vllm_mlx.server --model <qwen3.5-model-path> --port 8000
    # Then run:
    python3.12 scripts/bench_snapshot.py [--port 8000] [--rounds 5]
"""

import argparse
import json
import time

import requests

SYSTEM_PROMPT_SHORT = (
    "You are a highly knowledgeable AI assistant specializing in software engineering. "
    "You provide clear, concise, and accurate answers about programming, algorithms, "
    "system design, and best practices. When asked about code, you include well-formatted "
    "examples. You think step by step when solving complex problems."
)

SYSTEM_PROMPT_LONG = (
    "You are a highly knowledgeable AI assistant specializing in software engineering. "
    "You provide clear, concise, and accurate answers about programming, algorithms, "
    "system design, and best practices. When asked about code, you include well-formatted "
    "examples. You think step by step when solving complex problems.\n\n"
    "## Guidelines\n"
    "- Always respond in English unless the user writes in another language\n"
    "- Use markdown formatting for code blocks\n"
    "- When showing code, include comments explaining key parts\n"
    "- If the question is ambiguous, ask for clarification\n"
    "- Provide examples whenever possible\n"
    "- Consider edge cases in your solutions\n"
    "- Mention time and space complexity for algorithms\n"
    "- Follow best practices for the language being discussed\n\n"
    "## Available Tools\n"
    "You have access to the following tools:\n\n"
    "### get_weather\n"
    "Get the current weather for a given location. Parameters: location (string, required), "
    "unit (string, optional, one of 'celsius' or 'fahrenheit', default 'celsius').\n\n"
    "### search_web\n"
    "Search the web for information. Parameters: query (string, required), "
    "num_results (integer, optional, default 5).\n\n"
    "### execute_code\n"
    "Execute Python code in a sandboxed environment. Parameters: code (string, required), "
    "timeout_seconds (integer, optional, default 30).\n\n"
    "### read_file\n"
    "Read the contents of a file. Parameters: path (string, required), "
    "encoding (string, optional, default 'utf-8').\n\n"
    "### write_file\n"
    "Write content to a file. Parameters: path (string, required), "
    "content (string, required), mode (string, optional, 'w' or 'a', default 'w').\n\n"
    "### list_directory\n"
    "List the contents of a directory. Parameters: path (string, required), "
    "recursive (boolean, optional, default false).\n\n"
    "### run_tests\n"
    "Run test suite for a project. Parameters: test_path (string, required), "
    "framework (string, optional, one of 'pytest', 'unittest', 'jest'), "
    "verbose (boolean, optional, default false).\n\n"
    "### git_operations\n"
    "Perform git operations. Parameters: operation (string, required, one of "
    "'status', 'diff', 'log', 'commit', 'push', 'pull'), "
    "args (string, optional, additional arguments).\n\n"
    "When using tools, format your response as a JSON object with 'name' and 'arguments' fields. "
    "You may call multiple tools in sequence if needed to answer the user's question."
)

USER_PROMPTS = [
    "What is 2+2?",
    "What is the capital of France?",
    "Name a prime number.",
    "What color is the sky?",
    "Say hello.",
]


def send_chat(
    port: int,
    user_msg: str,
    max_tokens: int = 5,
    system_prompt: str = SYSTEM_PROMPT_SHORT,
) -> dict:
    """Send a chat request and return timing info."""
    url = f"http://localhost:{port}/v1/chat/completions"
    payload = {
        "model": "qwen",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"/no_think {user_msg}"},
        ],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
    }

    t0 = time.perf_counter()
    ttft = None
    tokens = 0
    text = ""

    resp = requests.post(url, json=payload, stream=True, timeout=120)
    resp.raise_for_status()

    for line in resp.iter_lines():
        line = line.decode("utf-8", errors="ignore")
        if not line.startswith("data: "):
            continue
        data_str = line[6:].strip()
        if data_str == "[DONE]":
            break
        try:
            data = json.loads(data_str)
        except json.JSONDecodeError:
            continue
        delta = data.get("choices", [{}])[0].get("delta", {})
        content = delta.get("content", "")
        if content and ttft is None:
            ttft = time.perf_counter() - t0
        if content:
            tokens += 1
            text += content

    total = time.perf_counter() - t0
    return {
        "ttft": ttft or total,
        "total": total,
        "tokens": tokens,
        "text": text.strip(),
    }


def run_benchmark(port: int, rounds: int, system_prompt: str, label: str):
    """Run a benchmark for a given system prompt."""
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")

    print("\n--- Cold request (no snapshot) ---")
    cold = send_chat(port, USER_PROMPTS[0], system_prompt=system_prompt)
    print(f"  TTFT: {cold['ttft']:.3f}s  |  Response: {cold['text'][:50]}")

    # Subsequent requests (should use snapshot)
    print(f"\n--- Warm requests ({rounds} rounds) ---")
    warm_ttfts = []
    for i in range(rounds):
        prompt = USER_PROMPTS[(i + 1) % len(USER_PROMPTS)]
        result = send_chat(port, prompt, system_prompt=system_prompt)
        warm_ttfts.append(result["ttft"])
        print(
            f"  Round {i + 1}: TTFT={result['ttft']:.3f}s  |  Response: {result['text'][:50]}"
        )

    # Rounds 3+ are snapshot-restored (rounds 1-2 build the snapshot)
    restored_ttfts = warm_ttfts[2:] if len(warm_ttfts) > 2 else warm_ttfts
    avg_warm = sum(warm_ttfts) / len(warm_ttfts)
    avg_restored = (
        sum(restored_ttfts) / len(restored_ttfts) if restored_ttfts else avg_warm
    )

    print("\n  --- Results ---")
    print(f"  Cold TTFT (no snapshot):       {cold['ttft']:.3f}s")
    print(f"  Avg warm TTFT (all rounds):    {avg_warm:.3f}s")
    print(f"  Avg restored TTFT (rounds 3+): {avg_restored:.3f}s")
    if cold["ttft"] > 0 and avg_restored > 0:
        speedup = cold["ttft"] / avg_restored
        saved_pct = (1 - avg_restored / cold["ttft"]) * 100
        print(f"  Speedup (restored vs cold):    {speedup:.2f}x")
        print(f"  TTFT reduction:                {saved_pct:.1f}%")

    return {
        "label": label,
        "cold_ttft": cold["ttft"],
        "avg_warm_ttft": avg_warm,
        "avg_restored_ttft": avg_restored,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark DeltaNet snapshot TTFT")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--rounds", type=int, default=5, help="Number of repeated requests"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("DeltaNet State Snapshot Benchmark")
    print("=" * 60)

    results = []
    results.append(
        run_benchmark(
            args.port,
            args.rounds,
            SYSTEM_PROMPT_SHORT,
            "Short system prompt (~60 tokens)",
        )
    )

    # Reset snapshot by sending a completely different prompt first
    send_chat(args.port, "reset", system_prompt="reset")

    results.append(
        run_benchmark(
            args.port,
            args.rounds,
            SYSTEM_PROMPT_LONG,
            "Long system prompt (~500 tokens, with tools)",
        )
    )

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    for r in results:
        speedup = (
            r["cold_ttft"] / r["avg_restored_ttft"] if r["avg_restored_ttft"] > 0 else 0
        )
        print(
            f"  {r['label']}: {r['cold_ttft']:.3f}s -> {r['avg_restored_ttft']:.3f}s ({speedup:.2f}x)"
        )
    print()


if __name__ == "__main__":
    main()
