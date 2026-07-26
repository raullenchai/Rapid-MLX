#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Example: Test continuous batching with rapid-mlx.

This script demonstrates the continuous batching capability by sending
multiple concurrent requests and measuring throughput.

Usage:
    python examples/test_batching.py
    python examples/test_batching.py --model mlx-community/Qwen2.5-3B-Instruct-4bit
    python examples/test_batching.py --num-requests 10 --max-tokens 50
"""

import argparse
import asyncio

# Add parent to path for development
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from vllm_mlx import AsyncEngineCore, EngineConfig, SamplingParams, SchedulerConfig


async def run_single_request(
    engine: AsyncEngineCore,
    request_id: str,
    prompt: str,
    sampling_params: SamplingParams,
) -> dict:
    """Run a single request and collect timing."""
    start = time.perf_counter()

    await engine.add_request(
        prompt=prompt,
        sampling_params=sampling_params,
        request_id=request_id,
    )

    tokens = []
    ttft = None  # Time to first token

    async for output in engine.stream_outputs(request_id):
        if ttft is None:
            ttft = time.perf_counter() - start
        tokens.extend(output.new_token_ids)

        if output.finished:
            break

    total_time = time.perf_counter() - start

    return {
        "request_id": request_id,
        "prompt_length": len(prompt.split()),
        "num_tokens": len(tokens),
        "ttft": ttft,
        "total_time": total_time,
        "tokens_per_second": len(tokens) / total_time if total_time > 0 else 0,
        "output_text": output.output_text if output else "",
    }


async def run_concurrent_requests(
    engine: AsyncEngineCore,
    prompts: list[str],
    sampling_params: SamplingParams,
) -> list[dict]:
    """Run multiple requests concurrently."""
    tasks = []
    for i, prompt in enumerate(prompts):
        task = run_single_request(engine, f"req-{i}", prompt, sampling_params)
        tasks.append(task)

    return await asyncio.gather(*tasks)


def print_results(results: list[dict], total_time: float):
    """Print benchmark results."""
    print("\n" + "=" * 60)
    print("CONTINUOUS BATCHING BENCHMARK RESULTS")
    print("=" * 60)

    total_tokens = sum(r["num_tokens"] for r in results)
    avg_ttft = sum(r["ttft"] for r in results if r["ttft"]) / len(results)
    avg_time = sum(r["total_time"] for r in results) / len(results)

    print(f"\nRequests: {len(results)}")
    print(f"Total tokens generated: {total_tokens}")
    print(f"Total wall time: {total_time:.2f}s")
    print(f"Throughput: {total_tokens / total_time:.1f} tokens/s")
    print(f"Avg time to first token: {avg_ttft * 1000:.1f}ms")
    print(f"Avg request time: {avg_time:.2f}s")

    print("\n" + "-" * 60)
    print("Per-request details:")
    print("-" * 60)
    for r in results:
        print(
            f"  {r['request_id']}: {r['num_tokens']} tokens, "
            f"{r['total_time']:.2f}s, {r['tokens_per_second']:.1f} tok/s"
        )

    print("\n" + "-" * 60)
    print("Sample outputs:")
    print("-" * 60)
    for r in results[:3]:  # Show first 3
        text = (
            r["output_text"][:100] + "..."
            if len(r["output_text"]) > 100
            else r["output_text"]
        )
        print(f"  {r['request_id']}: {text}")


async def main():
    parser = argparse.ArgumentParser(description="Test continuous batching")
    parser.add_argument(
        "--model",
        type=str,
        default="mlx-community/Llama-3.2-1B-Instruct-4bit",
        help="Model to use",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=5,
        help="Number of concurrent requests",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=30,
        help="Max tokens per request",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    from mlx_lm import load

    model, tokenizer = load(args.model)

    # Configure scheduler for batching
    scheduler_config = SchedulerConfig(
        max_num_seqs=32,
        prefill_batch_size=8,
        completion_batch_size=16,
    )

    engine_config = EngineConfig(
        model_name=args.model,
        scheduler_config=scheduler_config,
    )

    # Sample prompts
    prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a haiku about programming.",
        "What are the benefits of exercise?",
        "How do airplanes fly?",
        "What is machine learning?",
        "Describe the water cycle.",
        "What is photosynthesis?",
        "Explain gravity simply.",
        "What causes rain?",
    ][: args.num_requests]

    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    print(f"\nRunning {len(prompts)} concurrent requests...")
    print(f"Max tokens: {args.max_tokens}, Temperature: {args.temperature}")

    async with AsyncEngineCore(model, tokenizer, engine_config) as engine:
        # Let engine warm up
        await asyncio.sleep(0.1)

        start = time.perf_counter()
        results = await run_concurrent_requests(engine, prompts, sampling_params)
        total_time = time.perf_counter() - start

    print_results(results, total_time)


if __name__ == "__main__":
    asyncio.run(main())
