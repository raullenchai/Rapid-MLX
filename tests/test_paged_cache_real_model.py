# SPDX-License-Identifier: Apache-2.0
"""
Real model test for Paged KV Cache benefits.

This test loads a real model and demonstrates memory efficiency
with actual inference.

Usage:
    python tests/test_paged_cache_real_model.py
    python tests/test_paged_cache_real_model.py --model mlx-community/Qwen3-0.6B-8bit
"""

import pytest

pytest.importorskip("mlx")
pytestmark = pytest.mark.requires_mlx


import argparse
import platform
import sys
import time

# Skip if not on Apple Silicon
if sys.platform != "darwin" or platform.machine() != "arm64":
    pytest.skip("This test requires Apple Silicon", allow_module_level=True)

import mlx.core as mx


def get_memory_usage_mb() -> float:
    """Get current MLX memory usage in MB."""
    mx.eval(mx.zeros(1))  # Sync
    # MLX doesn't have direct memory query, estimate from cache
    return 0  # Placeholder - we'll measure block counts instead


def print_header(title: str) -> None:
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_table(
    headers: list[str], rows: list[list[str]], col_widths: list[int] = None
) -> None:
    """Print a formatted table."""
    if col_widths is None:
        col_widths = [
            max(len(str(row[i])) for row in [headers] + rows) + 2
            for i in range(len(headers))
        ]

    header_line = "|".join(h.center(w) for h, w in zip(headers, col_widths))
    separator = "+".join("-" * w for w in col_widths)

    print(f"+{separator}+")
    print(f"|{header_line}|")
    print(f"+{separator}+")

    for row in rows:
        row_line = "|".join(str(cell).center(w) for cell, w in zip(row, col_widths))
        print(f"|{row_line}|")

    print(f"+{separator}+")


def run_shared_system_prompt_real(model_name: str):
    """
    Test with real model: Multiple requests sharing same system prompt.
    """
    print_header("Real Model Test: Shared System Prompts")

    from mlx_lm import load

    print(f"\nLoading model: {model_name}")
    model, tokenizer = load(model_name)
    print("Model loaded!\n")

    # Long system prompt (~512 tokens)
    system_prompt = """You are a highly advanced AI assistant with expertise in multiple domains including:
- Software engineering and programming languages (Python, JavaScript, Rust, Go, C++, Java, Kotlin, Swift)
- Machine learning and artificial intelligence (deep learning, NLP, computer vision, reinforcement learning)
- Data science and analytics (statistics, data visualization, big data, ETL pipelines)
- Cloud computing and DevOps (AWS, GCP, Azure, Kubernetes, Docker, Terraform, Ansible)
- Cybersecurity and cryptography (encryption, authentication, penetration testing, vulnerability assessment)
- Mathematics and algorithms (linear algebra, calculus, graph theory, optimization)
- Database systems (PostgreSQL, MySQL, MongoDB, Redis, Elasticsearch, Cassandra)
- Web development (React, Vue, Angular, Node.js, Django, FastAPI, Spring Boot)

Your responses should be:
1. Accurate and well-researched with citations when appropriate
2. Clear and easy to understand for developers at all skill levels
3. Practical with real-world examples and working code snippets
4. Considerate of best practices, security implications, and performance
5. Thoughtful about edge cases and potential failure modes
6. Structured with proper formatting using markdown

When answering questions:
- First understand the context and requirements completely before responding
- Provide step-by-step explanations when appropriate for complex topics
- Include code examples when relevant with proper syntax highlighting
- Mention potential pitfalls and how to avoid them proactively
- Suggest further resources for learning and official documentation
- Consider backward compatibility and migration paths
- Discuss trade-offs between different approaches

Remember to be helpful, harmless, and honest in all interactions.
You have access to the latest information and can help with a wide range of tasks.
Always prioritize user safety and provide ethical guidance.

Additional context for this session:
This is a technical support session where users may ask about various programming topics.
Be prepared to help with debugging, code review, architecture decisions, and best practices.
You can use markdown formatting for better readability.
Code blocks should include the programming language for syntax highlighting.
Tables can be used for comparing options or presenting structured data.

Common topics you may encounter:
- API design and REST/GraphQL best practices
- Database optimization and query performance tuning
- Frontend frameworks and state management patterns
- Backend services and microservices architecture
- Testing strategies and test-driven development
- CI/CD pipelines and deployment automation
- Performance optimization and profiling techniques
- Security best practices and vulnerability prevention
- Container orchestration and cloud-native patterns
- Monitoring, logging, and observability
- Code review and refactoring strategies

Let's begin the session. I'm ready to help with any technical questions you have."""

    # Tokenize system prompt
    system_tokens = tokenizer.encode(system_prompt)
    print(f"System prompt: {len(system_tokens)} tokens")

    # User queries (different questions)
    user_queries = [
        "How do I implement a REST API in Python?",
        "What's the difference between SQL and NoSQL databases?",
        "Explain async/await in JavaScript.",
        "How do I optimize a slow database query?",
        "What are microservices and when should I use them?",
        "How do I set up CI/CD for a Python project?",
        "What's the best way to handle authentication?",
        "How do I debug memory leaks in Node.js?",
        "Explain Docker containers vs virtual machines.",
        "What are the SOLID principles in OOP?",
    ]

    num_users = len(user_queries)

    # Test WITHOUT paged cache (standard approach)
    print("\n--- Test WITHOUT Paged Cache ---")
    from vllm_mlx.prefix_cache import PrefixCacheManager

    standard_cache = PrefixCacheManager(model=model, max_entries=100)

    standard_results = []
    start_time = time.perf_counter()

    for i, query in enumerate(user_queries):
        full_prompt = system_prompt + f"\n\nUser: {query}\nAssistant:"
        tokens = tokenizer.encode(full_prompt)

        # Check cache
        cache, remaining = standard_cache.fetch_cache(tokens)
        cached_tokens = len(tokens) - len(remaining) if cache else 0

        standard_results.append(
            {
                "user": i + 1,
                "total_tokens": len(tokens),
                "cached": cached_tokens,
                "remaining": len(remaining) if remaining else len(tokens),
            }
        )

        # Store in cache (simulating after generation)
        standard_cache.store_cache(tokens, [f"cache_{i}"])

    standard_time = time.perf_counter() - start_time
    standard_stats = standard_cache.get_stats()

    print(f"  Users processed: {num_users}")
    print(f"  Cache hits: {standard_stats['hits']}")
    print(f"  Tokens saved: {standard_stats['tokens_saved']}")
    print(f"  Time: {standard_time * 1000:.1f}ms")

    # Test WITH paged cache
    print("\n--- Test WITH Paged Cache ---")
    from vllm_mlx.paged_cache import PagedCacheManager
    from vllm_mlx.prefix_cache import BlockAwarePrefixCache

    paged_manager = PagedCacheManager(block_size=64, max_blocks=500)
    paged_cache = BlockAwarePrefixCache(model=model, paged_cache_manager=paged_manager)

    paged_results = []
    start_time = time.perf_counter()

    for i, query in enumerate(user_queries):
        full_prompt = system_prompt + f"\n\nUser: {query}\nAssistant:"
        tokens = tokenizer.encode(full_prompt)

        # Check cache
        block_table, remaining = paged_cache.fetch_cache(f"user-{i}", tokens)
        cached_tokens = block_table.num_tokens if block_table else 0

        paged_results.append(
            {
                "user": i + 1,
                "total_tokens": len(tokens),
                "cached": cached_tokens,
                "shared_blocks": len(block_table.block_ids) if block_table else 0,
                "remaining": len(remaining),
            }
        )

        # Store in cache
        paged_cache.store_cache(f"user-{i}", tokens, [f"cache_{i}"])

    paged_time = time.perf_counter() - start_time
    paged_stats = paged_cache.get_stats()

    print(f"  Users processed: {num_users}")
    print(f"  Cache hits: {paged_stats['hits']}")
    print(f"  Tokens saved: {paged_stats['tokens_saved']}")
    print(f"  Blocks allocated: {paged_stats['allocated_blocks']}")
    print(f"  Shared blocks: {paged_stats['shared_blocks']}")
    print(f"  Time: {paged_time * 1000:.1f}ms")

    # Calculate theoretical blocks without sharing
    avg_tokens_per_request = sum(r["total_tokens"] for r in paged_results) / num_users
    blocks_per_request = (avg_tokens_per_request + 63) // 64
    theoretical_blocks = int(blocks_per_request * num_users)

    # Summary comparison
    print("\n" + "=" * 50)
    print("COMPARISON SUMMARY")
    print("=" * 50)

    print_table(
        ["Metric", "Standard Cache", "Paged Cache"],
        [
            ["Cache hits", str(standard_stats["hits"]), str(paged_stats["hits"])],
            [
                "Tokens saved",
                str(standard_stats["tokens_saved"]),
                str(paged_stats["tokens_saved"]),
            ],
            [
                "Blocks (theoretical)",
                str(theoretical_blocks),
                str(paged_stats["allocated_blocks"]),
            ],
            [
                "Memory savings",
                "0%",
                f"{(1 - paged_stats['allocated_blocks'] / theoretical_blocks) * 100:.1f}%",
            ],
        ],
        [20, 18, 18],
    )

    # Show per-user results
    print("\nPer-user breakdown (Paged Cache):")
    print_table(
        ["User", "Total Tokens", "Cached", "Shared Blocks", "New Tokens"],
        [
            [
                str(r["user"]),
                str(r["total_tokens"]),
                str(r["cached"]),
                str(r["shared_blocks"]),
                str(r["remaining"]),
            ]
            for r in paged_results[:5]
        ],
        [8, 14, 10, 15, 12],
    )
    if num_users > 5:
        print(f"... and {num_users - 5} more users with similar sharing ...")

    return paged_stats


async def run_real_concurrent_inference(model_name: str):
    """
    Run REAL concurrent inference with 20 requests.
    This uses actual model generation, not simulation.
    """
    import asyncio

    from mlx_lm import load

    from vllm_mlx.engine import AsyncEngineCore, EngineConfig
    from vllm_mlx.request import SamplingParams
    from vllm_mlx.scheduler import SchedulerConfig

    print_header("Real Concurrent Inference (20 requests)")

    print(f"\nLoading model: {model_name}")
    model, tokenizer = load(model_name)
    print("Model loaded!\n")

    # Long system prompt (~512 tokens) shared by all users
    system_prompt = """You are an expert coding assistant with deep knowledge of software engineering and computer science.
Your expertise spans multiple programming languages including Python, JavaScript, TypeScript, Rust, Go, C++, Java, Kotlin, Swift, and Scala.
You follow best practices for clean code, testing, documentation, and software architecture across all major paradigms.

Core Principles:
1. Code Quality: Write clean, readable, and maintainable code. Use meaningful variable and function names. Follow consistent formatting and style guides.
2. Testing: Always consider testability. Suggest unit tests, integration tests, end-to-end tests, and edge cases to cover. Aim for high test coverage.
3. Documentation: Include docstrings, comments for complex logic, README documentation, and architecture decision records (ADRs).
4. Error Handling: Implement proper exception handling, input validation, graceful degradation, and informative error messages for debugging.
5. Security: Follow security best practices, avoid common vulnerabilities like SQL injection, XSS, CSRF, and always validate untrusted input.
6. Performance: Optimize for readability first, but be aware of time and space complexity. Profile before optimizing and avoid premature optimization.
7. Design Patterns: Apply appropriate design patterns like Factory, Observer, Strategy, Dependency Injection, Repository, and CQRS when beneficial.
8. SOLID Principles: Single Responsibility, Open/Closed, Liskov Substitution, Interface Segregation, Dependency Inversion.

When helping with code:
- First understand the problem completely before suggesting solutions
- Provide working code examples with detailed explanations of each part
- Suggest multiple approaches when applicable, discussing trade-offs in terms of performance, maintainability, and complexity
- Include error handling and edge case management in all code examples
- Recommend relevant libraries, frameworks, and tools with justification
- Point out potential performance bottlenecks and scalability concerns
- Suggest improvements for code organization, architecture, and separation of concerns
- Consider backward compatibility when modifying existing code
- Discuss testing strategies appropriate for the code being written

Technical Stack Knowledge:
- Frontend: React, Vue, Angular, Next.js, Nuxt.js, Svelte, Tailwind CSS, Material UI
- Backend: FastAPI, Django, Flask, Express, NestJS, Spring Boot, ASP.NET Core
- Databases: PostgreSQL, MySQL, MongoDB, Redis, Elasticsearch, Cassandra, DynamoDB
- DevOps: Docker, Kubernetes, GitHub Actions, GitLab CI, Jenkins, AWS, GCP, Azure, Terraform
- Testing: pytest, Jest, Cypress, Selenium, Playwright, k6, Locust
- Observability: Prometheus, Grafana, Jaeger, OpenTelemetry, ELK Stack

Always explain your reasoning thoroughly and provide learning resources when helpful."""

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

    # Split into 2 rounds of 10 requests each to demonstrate cache reuse
    round1_prompts = prompts[:10]
    round2_prompts = prompts[10:]

    # Test WITHOUT paged cache
    print("\n" + "-" * 50)
    print("Test 1: WITHOUT Paged Cache (REAL INFERENCE)")
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
        # Round 1: First 10 requests
        print("  Round 1: Submitting first 10 requests...")
        request_ids = []
        for prompt in round1_prompts:
            rid = await engine.add_request(prompt, params)
            request_ids.append(rid)
        results1 = await asyncio.gather(*[get_output(r, engine) for r in request_ids])

        # Small pause to ensure cache is stored
        await asyncio.sleep(0.1)

        # Round 2: Next 10 requests (should benefit from cache)
        print("  Round 2: Submitting next 10 requests (cache reuse)...")
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

    # Test WITH paged cache
    print("\n" + "-" * 50)
    print("Test 2: WITH Paged Cache (REAL INFERENCE)")
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
        # Round 1: First 10 requests
        print("  Round 1: Submitting first 10 requests...")
        request_ids = []
        for prompt in round1_prompts:
            rid = await engine.add_request(prompt, params)
            request_ids.append(rid)
        results1 = await asyncio.gather(*[get_output(r, engine) for r in request_ids])

        # Small pause to ensure cache is stored
        await asyncio.sleep(0.1)

        # Round 2: Next 10 requests (should benefit from cache)
        print("  Round 2: Submitting next 10 requests (cache reuse)...")
        request_ids = []
        for prompt in round2_prompts:
            rid = await engine.add_request(prompt, params)
            request_ids.append(rid)
        results2 = await asyncio.gather(*[get_output(r, engine) for r in request_ids])

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
    print("REAL INFERENCE SUMMARY")
    print("=" * 50)

    speedup = time_no_paged / time_paged if time_paged > 0 else 0

    print_table(
        ["Metric", "Without Paged", "With Paged"],
        [
            ["Time", f"{time_no_paged:.2f}s", f"{time_paged:.2f}s"],
            [
                "Throughput",
                f"{total_tokens_no_paged / time_no_paged:.1f} tok/s",
                f"{total_tokens_paged / time_paged:.1f} tok/s",
            ],
            [
                "Blocks allocated",
                "-",
                str(stats.get("paged_cache", {}).get("allocated_blocks", 0)),
            ],
            [
                "Shared blocks",
                "-",
                str(stats.get("paged_cache", {}).get("shared_blocks", 0)),
            ],
            ["Cache hits", "0", str(stats.get("paged_cache", {}).get("hits", 0))],
            [
                "Tokens saved",
                "0",
                str(stats.get("paged_cache", {}).get("tokens_saved", 0)),
            ],
        ],
        [18, 15, 15],
    )

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

    return stats


def main():
    import asyncio

    parser = argparse.ArgumentParser(description="Test Paged KV Cache with real model")
    parser.add_argument(
        "--model",
        type=str,
        default="mlx-community/Qwen3-0.6B-8bit",
        help="Model to use for testing",
    )
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("     PAGED KV CACHE - REAL MODEL TEST")
    print("=" * 70)
    print(f"\nModel: {args.model}")

    # Run tests
    run_shared_system_prompt_real(args.model)
    asyncio.run(run_real_concurrent_inference(args.model))

    print("\n" + "=" * 70)
    print("     TEST COMPLETE")
    print("=" * 70)
    print("\nTo enable paged cache in production:")
    print("  rapid-mlx serve <model> --use-paged-cache")
    print()


if __name__ == "__main__":
    main()
