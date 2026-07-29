#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Test script for measuring streaming latency improvements.

This script measures:
- Time-to-first-token (TTFT)
- Inter-token latency (ITL)
- Total generation time

Usage:
    # Start server first:
    rapid-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit

    # Run test:
    python tests/test_streaming_latency.py
"""

import asyncio
import json
import statistics
import time

import httpx
import pytest


async def measure_streaming_latency(
    prompt: str,
    server_url: str = "http://localhost:8000",
    max_tokens: int = 50,
) -> tuple[float, list[float], float, int]:
    """
    Measure streaming latency metrics.

    Returns:
        Tuple of (ttft, inter_token_latencies, total_time, token_count)
    """
    start_time = time.perf_counter()
    first_token_time = None
    token_times: list[float] = []
    token_count = 0

    payload = {
        "model": "test",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": True,
    }

    async with (
        httpx.AsyncClient(timeout=60.0) as client,
        client.stream(
            "POST",
            f"{server_url}/v1/chat/completions",
            json=payload,
        ) as response,
    ):
        async for line in response.aiter_lines():
            if not line:
                continue
            if line.startswith("data: "):
                data = line[6:]
                if data == "[DONE]":
                    break

                current_time = time.perf_counter()

                try:
                    chunk = json.loads(data)
                    content = (
                        chunk.get("choices", [{}])[0]
                        .get("delta", {})
                        .get("content", "")
                    )
                    if content:
                        token_count += 1
                        if first_token_time is None:
                            first_token_time = current_time
                        token_times.append(current_time)
                except json.JSONDecodeError:
                    continue

    end_time = time.perf_counter()

    # Calculate metrics
    ttft = (first_token_time - start_time) * 1000 if first_token_time else 0
    total_time = (end_time - start_time) * 1000

    # Calculate inter-token latencies
    inter_token_latencies = []
    for i in range(1, len(token_times)):
        itl = (token_times[i] - token_times[i - 1]) * 1000
        inter_token_latencies.append(itl)

    return ttft, inter_token_latencies, total_time, token_count


async def run_benchmark(
    num_iterations: int = 5,
    server_url: str = "http://localhost:8000",
    prompts: list[str] = None,
) -> None:
    """Run the streaming latency benchmark."""
    if prompts is None:
        prompts = [
            "What is the capital of France?",
            "Explain quantum computing in simple terms.",
            "Write a haiku about programming.",
        ]

    print("=" * 60)
    print("Streaming Latency Benchmark")
    print("=" * 60)
    print(f"Server: {server_url}")
    print(f"Iterations per prompt: {num_iterations}")
    print()

    all_ttft: list[float] = []
    all_itl: list[float] = []
    all_total: list[float] = []
    all_tokens: list[int] = []

    for prompt in prompts:
        print(
            f'Prompt: "{prompt[:50]}..."' if len(prompt) > 50 else f'Prompt: "{prompt}"'
        )
        print("-" * 40)

        prompt_ttft = []
        prompt_itl = []
        prompt_total = []
        prompt_tokens = []

        for i in range(num_iterations):
            try:
                ttft, itls, total, tokens = await measure_streaming_latency(
                    prompt=prompt,
                    server_url=server_url,
                    max_tokens=50,
                )

                prompt_ttft.append(ttft)
                prompt_itl.extend(itls)
                prompt_total.append(total)
                prompt_tokens.append(tokens)

                print(
                    f"  Run {i + 1}: TTFT={ttft:.1f}ms, Tokens={tokens}, Total={total:.1f}ms"
                )

            except Exception as e:
                print(f"  Run {i + 1}: ERROR - {e}")

        if prompt_ttft:
            avg_ttft = statistics.mean(prompt_ttft)
            avg_total = statistics.mean(prompt_total)
            avg_tokens = statistics.mean(prompt_tokens)
            avg_itl = statistics.mean(prompt_itl) if prompt_itl else 0

            print(f"  Avg TTFT: {avg_ttft:.1f}ms")
            print(f"  Avg ITL:  {avg_itl:.1f}ms")
            print(f"  Avg Total: {avg_total:.1f}ms")
            print(f"  Avg Tokens: {avg_tokens:.0f}")

            all_ttft.extend(prompt_ttft)
            all_itl.extend(prompt_itl)
            all_total.extend(prompt_total)
            all_tokens.extend(prompt_tokens)

        print()

    # Overall summary
    if all_ttft:
        print("=" * 60)
        print("OVERALL SUMMARY")
        print("=" * 60)
        print("Time-to-First-Token (TTFT):")
        print(f"  Mean:   {statistics.mean(all_ttft):.1f}ms")
        print(f"  Median: {statistics.median(all_ttft):.1f}ms")
        print(f"  Min:    {min(all_ttft):.1f}ms")
        print(f"  Max:    {max(all_ttft):.1f}ms")
        if len(all_ttft) > 1:
            print(f"  StdDev: {statistics.stdev(all_ttft):.1f}ms")
        print()

        if all_itl:
            print("Inter-Token Latency (ITL):")
            print(f"  Mean:   {statistics.mean(all_itl):.1f}ms")
            print(f"  Median: {statistics.median(all_itl):.1f}ms")
            print(f"  Min:    {min(all_itl):.1f}ms")
            print(f"  Max:    {max(all_itl):.1f}ms")
            if len(all_itl) > 1:
                print(f"  StdDev: {statistics.stdev(all_itl):.1f}ms")
            print()

        print("Total Generation Time:")
        print(f"  Mean:   {statistics.mean(all_total):.1f}ms")
        print()

        # Throughput
        total_tokens = sum(all_tokens)
        total_time_sec = sum(all_total) / 1000
        if total_time_sec > 0:
            throughput = total_tokens / total_time_sec
            print(f"Throughput: {throughput:.1f} tokens/sec")


@pytest.mark.asyncio
async def test_output_collector():
    """Unit test for RequestOutputCollector."""
    import sys

    sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

    from vllm_mlx.output_collector import RequestOutputCollector, RequestStreamState
    from vllm_mlx.request import RequestOutput

    print("Testing RequestOutputCollector...")

    # Test basic put/get
    collector = RequestOutputCollector(aggregate=False)
    output1 = RequestOutput(
        request_id="test1",
        new_token_ids=[1],
        new_text="Hello",
        output_token_ids=[1],
        output_text="Hello",
        finished=False,
    )

    collector.put(output1)
    assert collector.get_nowait() == output1
    assert collector.get_nowait() is None
    print("  [PASS] Basic put/get_nowait")

    # Test aggregation
    collector = RequestOutputCollector(aggregate=True)
    collector.put(
        RequestOutput(
            request_id="test2",
            new_token_ids=[1],
            new_text="Hello",
            output_token_ids=[1],
            output_text="Hello",
            finished=False,
        )
    )
    collector.put(
        RequestOutput(
            request_id="test2",
            new_token_ids=[2],
            new_text=" World",
            output_token_ids=[1, 2],
            output_text="Hello World",
            finished=False,
        )
    )

    merged = collector.get_nowait()
    assert merged is not None
    assert merged.new_text == "Hello World"
    assert merged.new_token_ids == [1, 2]
    print("  [PASS] Output aggregation")

    # Test async get
    async def test_async():
        collector = RequestOutputCollector()

        async def producer():
            await asyncio.sleep(0.01)
            collector.put(
                RequestOutput(
                    request_id="test3",
                    new_token_ids=[1],
                    new_text="Async",
                    output_token_ids=[1],
                    output_text="Async",
                    finished=False,
                )
            )

        asyncio.create_task(producer())
        output = await collector.get()
        assert output.new_text == "Async"

    await test_async()
    print("  [PASS] Async get")

    # Test RequestStreamState
    state = RequestStreamState(stream_interval=3)
    # First token always sends (sent_tokens == 0)
    assert state.should_send(1, False) is True  # First token
    state.mark_sent(1)
    assert state.should_send(2, False) is False  # Only 1 token since last send
    assert state.should_send(3, False) is False  # Only 2 tokens
    assert state.should_send(4, False) is True  # 3 tokens - should send
    state.mark_sent(4)
    assert state.should_send(5, False) is False  # Only 1 token since last send
    assert state.should_send(5, True) is True  # Finished always sends
    print("  [PASS] RequestStreamState")

    print("\nAll unit tests passed!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Streaming latency benchmark")
    parser.add_argument(
        "--server-url",
        type=str,
        default="http://localhost:8000",
        help="Server URL",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of iterations per prompt",
    )
    parser.add_argument(
        "--unit-test",
        action="store_true",
        help="Run unit tests instead of benchmark",
    )

    args = parser.parse_args()

    if args.unit_test:
        asyncio.run(test_output_collector())
    else:
        asyncio.run(
            run_benchmark(
                num_iterations=args.iterations,
                server_url=args.server_url,
            )
        )
