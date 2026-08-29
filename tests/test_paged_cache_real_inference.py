# SPDX-License-Identifier: Apache-2.0
"""
Real inference test for Paged KV Cache.

Runs 20 concurrent requests with actual model inference in 2 rounds
to demonstrate cache reuse.

Usage:
    python tests/test_paged_cache_real_inference.py
"""

import asyncio
import platform
import sys
import time

import pytest

pytest.importorskip("mlx")
pytestmark = pytest.mark.requires_mlx

# Skip if not on Apple Silicon
if sys.platform != "darwin" or platform.machine() != "arm64":
    pytest.skip("This test requires Apple Silicon", allow_module_level=True)


async def run_concurrent_inference():
    """Run 20 concurrent requests with real inference in 2 rounds."""
    from mlx_lm import load

    from vllm_mlx.engine import AsyncEngineCore, EngineConfig
    from vllm_mlx.request import SamplingParams
    from vllm_mlx.scheduler import SchedulerConfig

    model_name = "mlx-community/Qwen3-0.6B-8bit"

    print("=" * 70)
    print("  PAGED KV CACHE - REAL INFERENCE TEST")
    print("  (20 requests in 2 rounds - cache reuse on 2nd round)")
    print("=" * 70)

    print(f"\nLoading model: {model_name}")
    model, tokenizer = load(model_name)
    print("Model loaded!\n")

    # Shared system prompt (~286 tokens)
    system_prompt = """You are an expert coding assistant with deep knowledge of software engineering.
Your expertise spans Python, JavaScript, TypeScript, Rust, Go, C++, Java, and Kotlin.
You follow best practices for clean code, testing, documentation, and architecture.

Core Principles:
1. Code Quality: Write clean, readable, maintainable code with meaningful names.
2. Testing: Always consider testability. Suggest unit tests and edge cases.
3. Documentation: Include docstrings and comments for complex logic.
4. Error Handling: Implement proper exception handling and validation.
5. Security: Follow security best practices, avoid common vulnerabilities.
6. Performance: Optimize for readability first, be aware of complexity.
7. Design Patterns: Apply appropriate patterns like Factory, Observer, Strategy.

When helping with code:
- Understand the problem completely before suggesting solutions
- Provide working code examples with explanations
- Suggest multiple approaches when applicable
- Include error handling and edge cases
- Recommend relevant libraries and tools
- Point out potential performance issues

Technical Stack:
- Frontend: React, Vue, Angular, Next.js, Tailwind CSS
- Backend: FastAPI, Django, Express, Spring Boot
- Databases: PostgreSQL, MongoDB, Redis, Elasticsearch
- DevOps: Docker, Kubernetes, GitHub Actions, AWS, GCP
- Testing: pytest, Jest, Cypress, Selenium

Always explain your reasoning and provide learning resources."""

    # 20 different user questions
    user_questions = [
        "How do I implement a REST API in Python with FastAPI?",
        "What's the difference between SQL and NoSQL databases?",
        "Explain async/await in JavaScript with an example.",
        "How do I optimize a slow database query?",
        "What are microservices and when should I use them?",
        "How do I set up CI/CD for a Python project?",
        "What's the best way to handle authentication?",
        "How do I debug memory leaks in Node.js?",
        "Explain Docker containers vs virtual machines.",
        "What are the SOLID principles in OOP?",
        "How do I implement caching in a web application?",
        "What's the difference between REST and GraphQL?",
        "How do I write unit tests for async code?",
        "Explain dependency injection pattern.",
        "How do I handle errors in a distributed system?",
        "What are design patterns for scalability?",
        "How do I implement rate limiting?",
        "Explain event-driven architecture.",
        "How do I secure an API endpoint?",
        "What's the best way to log in production?",
    ]

    # Create prompts
    prompts = [f"{system_prompt}\n\nUser: {q}\nAssistant:" for q in user_questions]

    # Tokenize to show prompt sizes
    prompt_tokens = [len(tokenizer.encode(p)) for p in prompts]
    print(f"Number of requests: {len(prompts)}")
    print(f"System prompt tokens: ~{len(tokenizer.encode(system_prompt))}")
    print(f"Full prompt tokens: {min(prompt_tokens)}-{max(prompt_tokens)}")

    # Sampling params
    params = SamplingParams(
        max_tokens=50,  # Short responses for speed
        temperature=0.7,
    )

    async def get_output(rid, engine):
        async for out in engine.stream_outputs(rid, timeout=120):
            if out.finished:
                return out
        return None

    # Split into 2 rounds of 10 requests each
    round1_prompts = prompts[:10]
    round2_prompts = prompts[10:]

    # Test WITHOUT paged cache (2 rounds)
    print("\n" + "-" * 50)
    print("Test 1: WITHOUT Paged Cache (2 rounds of 10)")
    print("-" * 50)

    scheduler_config = SchedulerConfig(
        max_num_seqs=32,
        prefill_batch_size=8,
        completion_batch_size=16,
        enable_prefix_cache=True,
        use_paged_cache=False,
    )
    engine_config = EngineConfig(
        model_name=model_name,
        scheduler_config=scheduler_config,
    )

    start_time = time.perf_counter()
    total_tokens_no_paged = 0

    async with AsyncEngineCore(model, tokenizer, engine_config) as engine:
        # Round 1: First 10 requests (populates cache)
        print("  Round 1: Processing first 10 requests...")
        request_ids = []
        for prompt in round1_prompts:
            rid = await engine.add_request(prompt, params)
            request_ids.append(rid)
        results1 = await asyncio.gather(*[get_output(r, engine) for r in request_ids])

        # Small pause to ensure cache is stored
        await asyncio.sleep(0.1)

        # Round 2: Next 10 requests (should hit cache)
        print("  Round 2: Processing next 10 requests (cache reuse)...")
        request_ids = []
        for prompt in round2_prompts:
            rid = await engine.add_request(prompt, params)
            request_ids.append(rid)
        results2 = await asyncio.gather(*[get_output(r, engine) for r in request_ids])

        stats_no_paged = engine.engine.scheduler.get_stats()

    time_no_paged = time.perf_counter() - start_time

    for r in results1 + results2:
        if r:
            total_tokens_no_paged += r.completion_tokens

    print(f"  Time: {time_no_paged:.2f}s")
    print(f"  Total completion tokens: {total_tokens_no_paged}")
    print(f"  Throughput: {total_tokens_no_paged / time_no_paged:.1f} tok/s")
    if "prefix_cache" in stats_no_paged:
        pc = stats_no_paged["prefix_cache"]
        print(f"  Cache hits: {pc.get('hits', 0)}")
        print(f"  Tokens saved: {pc.get('tokens_saved', 0)}")

    # Test WITH paged cache (2 rounds)
    print("\n" + "-" * 50)
    print("Test 2: WITH Paged Cache (2 rounds of 10)")
    print("-" * 50)

    scheduler_config_paged = SchedulerConfig(
        max_num_seqs=32,
        prefill_batch_size=8,
        completion_batch_size=16,
        enable_prefix_cache=True,
        use_paged_cache=True,
        paged_cache_block_size=64,
        max_cache_blocks=500,
    )
    engine_config_paged = EngineConfig(
        model_name=model_name,
        scheduler_config=scheduler_config_paged,
    )

    start_time = time.perf_counter()
    total_tokens_paged = 0

    async with AsyncEngineCore(model, tokenizer, engine_config_paged) as engine:
        # Round 1: First 10 requests (populates cache)
        print("  Round 1: Processing first 10 requests...")
        request_ids = []
        for prompt in round1_prompts:
            rid = await engine.add_request(prompt, params)
            request_ids.append(rid)
        results1 = await asyncio.gather(*[get_output(r, engine) for r in request_ids])

        # Small pause to ensure cache is stored
        await asyncio.sleep(0.1)

        # Round 2: Next 10 requests (should hit cache)
        print("  Round 2: Processing next 10 requests (cache reuse)...")
        request_ids = []
        for prompt in round2_prompts:
            rid = await engine.add_request(prompt, params)
            request_ids.append(rid)
        results2 = await asyncio.gather(*[get_output(r, engine) for r in request_ids])

        # Get stats
        stats = engine.engine.scheduler.get_stats()

    time_paged = time.perf_counter() - start_time

    for r in results1 + results2:
        if r:
            total_tokens_paged += r.completion_tokens

    print(f"  Time: {time_paged:.2f}s")
    print(f"  Total completion tokens: {total_tokens_paged}")
    print(f"  Throughput: {total_tokens_paged / time_paged:.1f} tok/s")

    if "paged_cache" in stats:
        pc = stats["paged_cache"]
        print("\n  Paged Cache Stats:")
        print(f"    Blocks allocated: {pc.get('allocated_blocks', 'N/A')}")
        print(f"    Shared blocks: {pc.get('shared_blocks', 'N/A')}")
        print(f"    Cache hits: {pc.get('hits', 0)}")
        print(f"    Tokens saved: {pc.get('tokens_saved', 0)}")

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print("  Requests: 20 (2 rounds of 10)")
    print(f"  System prompt: ~{len(tokenizer.encode(system_prompt))} tokens (shared)")
    print("\n  Without paged cache:")
    print(f"    Time: {time_no_paged:.2f}s")
    print(f"    Throughput: {total_tokens_no_paged / time_no_paged:.1f} tok/s")
    print("\n  With paged cache:")
    print(f"    Time: {time_paged:.2f}s")
    print(f"    Throughput: {total_tokens_paged / time_paged:.1f} tok/s")

    speedup = time_no_paged / time_paged if time_paged > 0 else 0
    print(f"\n  Speedup: {speedup:.2f}x")

    # Show sample outputs
    print("\n" + "-" * 50)
    print("Sample outputs (first 3):")
    print("-" * 50)
    all_results = results1 + results2
    for i, r in enumerate(all_results[:3]):
        if r:
            print(f"\nQ{i + 1}: {user_questions[i][:50]}...")
            print(f"A{i + 1}: {r.output_text[:100]}...")


if __name__ == "__main__":
    asyncio.run(run_concurrent_inference())
