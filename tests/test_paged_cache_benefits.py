# SPDX-License-Identifier: Apache-2.0
"""
Test script demonstrating the benefits of Paged KV Cache.

This script shows three key benefits:
1. Shared system prompts - Multiple requests with same prefix share cache blocks
2. Memory efficiency - Reduced memory usage with many concurrent requests
3. Prefix sharing - Similar conversations reuse cached prefixes

Usage:
    python tests/test_paged_cache_benefits.py
"""

import platform
import sys

import pytest

pytest.importorskip("mlx")
pytestmark = pytest.mark.requires_mlx

# Skip if not on Apple Silicon
if sys.platform != "darwin" or platform.machine() != "arm64":
    pytest.skip("This test requires Apple Silicon", allow_module_level=True)


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

    # Header
    header_line = "|".join(h.center(w) for h, w in zip(headers, col_widths))
    separator = "+".join("-" * w for w in col_widths)

    print(f"+{separator}+")
    print(f"|{header_line}|")
    print(f"+{separator}+")

    # Rows
    for row in rows:
        row_line = "|".join(str(cell).center(w) for cell, w in zip(row, col_widths))
        print(f"|{row_line}|")

    print(f"+{separator}+")


def test_benefit_1_shared_system_prompts():
    """
    Benefit 1: Multiple users sharing the same system prompt.

    When multiple requests use the same system prompt, paged cache
    allows them to share the same cache blocks instead of duplicating.
    """
    print_header("Benefit 1: Shared System Prompts")

    from vllm_mlx.paged_cache import PagedCacheManager
    from vllm_mlx.prefix_cache import BlockAwarePrefixCache

    # Common system prompt (simulating multiple users with same instructions)
    system_prompt_tokens = list(range(256))  # 256 tokens = 4 blocks of 64

    # 20 User-specific queries with varying lengths
    num_users = 20
    user_queries = []
    for i in range(num_users):
        # Each user has different query length: 20-115 tokens
        query_len = 20 + (i * 5)
        user_queries.append(list(range(256 + i * 200, 256 + i * 200 + query_len)))

    print(
        f"\nScenario: {num_users} users with SAME system prompt (256 tokens) + different queries"
    )
    print("System prompt: 256 tokens = 4 blocks")
    print("User queries: 20-115 additional tokens each\n")

    # Initialize paged cache
    paged_manager = PagedCacheManager(block_size=64, max_blocks=500)
    cache = BlockAwarePrefixCache(model=None, paged_cache_manager=paged_manager)

    # Simulate storing cache for first request (establishes the prefix)
    first_request_tokens = system_prompt_tokens + user_queries[0]
    cache.store_cache("req-0", first_request_tokens, ["kv_cache_data"])

    initial_blocks = paged_manager.stats.allocated_blocks
    print(f"After 1st request: {initial_blocks} blocks allocated")

    # Now simulate remaining requests with same system prompt
    results = []
    total_shared_tokens = 0
    for i, query in enumerate(user_queries[1:], 1):
        full_tokens = system_prompt_tokens + query

        # Fetch will find shared prefix
        block_table, remaining = cache.fetch_cache(f"req-{i}", full_tokens)

        shared_tokens = len(full_tokens) - len(remaining) if block_table else 0
        shared_blocks = len(block_table.block_ids) if block_table else 0
        total_shared_tokens += shared_tokens

        # Store the new request
        cache.store_cache(f"req-{i}", full_tokens, [f"kv_cache_data_{i}"])

        results.append(
            [
                f"User {i + 1}",
                str(len(full_tokens)),
                str(shared_tokens),
                str(shared_blocks),
                str(len(remaining)),
            ]
        )

    final_blocks = paged_manager.stats.allocated_blocks
    stats = cache.get_stats()

    # Show summary for first 5 and last 5 users
    print("\nResults (first 5 users):")
    print_table(
        ["User", "Total Tokens", "Shared", "Shared Blocks", "New Tokens"],
        results[:5],
        [10, 15, 10, 15, 12],
    )
    print("\n... (10 more users) ...")
    print("\nResults (last 5 users):")
    print_table(
        ["User", "Total Tokens", "Shared", "Shared Blocks", "New Tokens"],
        results[-5:],
        [10, 15, 10, 15, 12],
    )

    # Calculate memory savings
    # Without sharing: each user would need ~5-6 blocks (256+query tokens)
    avg_blocks_per_user = 6  # ~320-370 tokens average = 6 blocks
    blocks_without_sharing = avg_blocks_per_user * num_users
    blocks_with_sharing = final_blocks
    savings = (1 - blocks_with_sharing / blocks_without_sharing) * 100

    print("\nMemory Analysis:")
    print(
        f"  Blocks without sharing: ~{blocks_without_sharing} ({num_users} users x {avg_blocks_per_user} blocks)"
    )
    print(f"  Blocks with sharing:    {blocks_with_sharing}")
    print(f"  Memory saved:           {savings:.1f}%")
    print(f"  Cache hits:             {stats['hits']}")
    print(f"  Tokens saved:           {stats['tokens_saved']}")

    return savings


def test_benefit_2_memory_efficiency():
    """
    Benefit 2: Memory efficiency with many concurrent requests.

    Shows how paged cache tracks memory usage efficiently
    with reference counting and block-level management.
    """
    print_header("Benefit 2: Memory Efficiency with Concurrent Requests")

    from vllm_mlx.paged_cache import PagedCacheManager
    from vllm_mlx.prefix_cache import BlockAwarePrefixCache

    num_requests = 50
    tokens_per_request = 256  # 4 blocks per request
    print(f"\nScenario: Simulating {num_requests} concurrent requests")
    print(
        f"Each request: {tokens_per_request} tokens ({tokens_per_request // 64} blocks)\n"
    )

    # Compare standard allocation vs paged allocation

    # Standard approach: each request gets independent allocation
    standard_blocks_per_request = tokens_per_request // 64
    standard_total = standard_blocks_per_request * num_requests

    # Paged approach with some prefix sharing
    paged_manager = PagedCacheManager(block_size=64, max_blocks=1000)
    cache = BlockAwarePrefixCache(model=None, paged_cache_manager=paged_manager)

    # Simulate requests with varying prefix overlap
    # Group 1: 20 requests with same 128-token prefix (2 shared blocks)
    # Group 2: 15 requests with same 128-token prefix (2 shared blocks)
    # Group 3: 15 requests with same 128-token prefix (2 shared blocks)

    common_prefix_1 = list(range(128))  # 128 tokens = 2 blocks
    common_prefix_2 = list(range(1000, 1128))
    common_prefix_3 = list(range(2000, 2128))

    for i in range(20):
        # Each request: 128 shared + 128 unique = 256 total
        tokens = common_prefix_1 + list(range(5000 + i * 200, 5128 + i * 200))
        cache.store_cache(f"group1-req-{i}", tokens, [f"cache_{i}"])

    for i in range(15):
        tokens = common_prefix_2 + list(range(10000 + i * 200, 10128 + i * 200))
        cache.store_cache(f"group2-req-{i}", tokens, [f"cache_{i}"])

    for i in range(15):
        tokens = common_prefix_3 + list(range(15000 + i * 200, 15128 + i * 200))
        cache.store_cache(f"group3-req-{i}", tokens, [f"cache_{i}"])

    paged_total = paged_manager.stats.allocated_blocks
    shared_blocks = paged_manager.stats.shared_blocks

    usage = paged_manager.get_memory_usage()

    print("Comparison:")
    print_table(
        ["Metric", "Standard", "Paged Cache"],
        [
            ["Requests", str(num_requests), str(num_requests)],
            ["Tokens/request", str(tokens_per_request), str(tokens_per_request)],
            [
                "Total tokens",
                str(num_requests * tokens_per_request),
                str(num_requests * tokens_per_request),
            ],
            ["Blocks allocated", str(standard_total), str(paged_total)],
            ["Shared blocks", "0", str(shared_blocks)],
        ],
        [20, 12, 15],
    )

    savings = (1 - paged_total / standard_total) * 100
    print(f"\nMemory saved: {savings:.1f}%")
    print(f"Cache hit rate: {usage['cache_hit_rate'] * 100:.1f}%")

    # Show reference counting in action
    print("\nReference Counting Demo:")
    print("  Releasing 10 requests from group 1...")

    for i in range(10):
        cache.release_cache(f"group1-req-{i}")

    after_release = paged_manager.stats.allocated_blocks
    freed = paged_total - after_release
    print(f"  Blocks before: {paged_total}")
    print(f"  Blocks after:  {after_release}")
    print(f"  Blocks freed:  {freed}")
    if freed == 0:
        print("  (Shared prefix blocks still referenced by other requests)")

    # Release all remaining group1 requests to show full cleanup
    print("\n  Releasing remaining 10 requests from group 1...")
    for i in range(10, 20):
        cache.release_cache(f"group1-req-{i}")

    after_full_release = paged_manager.stats.allocated_blocks
    print(f"  Blocks after full group release: {after_full_release}")
    print(f"  Total blocks freed: {paged_total - after_full_release}")

    return savings


def test_benefit_3_prefix_sharing():
    """
    Benefit 3: Prefix sharing between similar conversations.

    Shows how conversations with similar beginnings can share
    cached prefixes, reducing computation and memory.
    """
    print_header("Benefit 3: Prefix Sharing for Similar Conversations")

    from vllm_mlx.paged_cache import PagedCacheManager
    from vllm_mlx.prefix_cache import BlockAwarePrefixCache

    print("\nScenario: Chat conversations with branching responses")
    print("         Similar to tree of possible continuations\n")

    paged_manager = PagedCacheManager(block_size=64, max_blocks=100)
    cache = BlockAwarePrefixCache(model=None, paged_cache_manager=paged_manager)

    # Simulate a conversation tree:
    # Root: "You are a helpful assistant..." (64 tokens)
    # Branch 1: "Tell me about Python" -> 3 follow-ups
    # Branch 2: "Tell me about Rust" -> 2 follow-ups

    root_tokens = list(range(64))  # System prompt

    # First conversation: Python discussion
    python_intro = root_tokens + list(range(100, 140))  # "Tell me about Python"
    cache.store_cache("conv-python", python_intro, ["python_cache"])

    # Follow-ups share the python_intro prefix
    python_followups = [
        python_intro + list(range(200, 230)),  # "How about async?"
        python_intro + list(range(300, 360)),  # "What about types?"
        python_intro + list(range(400, 450)),  # "Show me decorators"
    ]

    print("Python conversation tree:")
    print("  Root (64 tokens) -> Python intro (+40) -> 3 follow-ups")

    for i, tokens in enumerate(python_followups):
        block_table, remaining = cache.fetch_cache(f"python-followup-{i}", tokens)
        shared = len(tokens) - len(remaining) if block_table else 0
        print(
            f"    Follow-up {i + 1}: {len(tokens)} tokens, {shared} shared ({shared * 100 // len(tokens)}%)"
        )
        cache.store_cache(f"python-followup-{i}", tokens, [f"followup_{i}"])

    # Second conversation: Rust discussion
    rust_intro = root_tokens + list(range(500, 540))  # "Tell me about Rust"

    # Check if root is shared
    block_table, remaining = cache.fetch_cache("conv-rust", rust_intro)
    root_shared = len(rust_intro) - len(remaining) if block_table else 0

    print("\nRust conversation:")
    print(f"  Shares root with Python: {root_shared} tokens (system prompt)")

    cache.store_cache("conv-rust", rust_intro, ["rust_cache"])

    rust_followups = [
        rust_intro + list(range(600, 650)),  # "Ownership model?"
        rust_intro + list(range(700, 780)),  # "Compare to C++?"
    ]

    for i, tokens in enumerate(rust_followups):
        block_table, remaining = cache.fetch_cache(f"rust-followup-{i}", tokens)
        shared = len(tokens) - len(remaining) if block_table else 0
        print(
            f"    Follow-up {i + 1}: {len(tokens)} tokens, {shared} shared ({shared * 100 // len(tokens)}%)"
        )
        cache.store_cache(f"rust-followup-{i}", tokens, [f"rust_followup_{i}"])

    # Summary
    stats = cache.get_stats()
    usage = paged_manager.get_memory_usage()

    total_conversations = 7  # 1 python + 3 followups + 1 rust + 2 followups

    # Calculate tokens without sharing
    all_token_counts = [
        len(python_intro),
        *[len(t) for t in python_followups],
        len(rust_intro),
        *[len(t) for t in rust_followups],
    ]
    total_tokens_without_sharing = sum(all_token_counts)
    tokens_saved = stats["tokens_saved"]

    print("\nPrefix Sharing Summary:")
    print_table(
        ["Metric", "Value"],
        [
            ["Total conversations", str(total_conversations)],
            ["Tokens without sharing", str(total_tokens_without_sharing)],
            ["Tokens saved by sharing", str(tokens_saved)],
            ["Cache hits", str(stats["hits"])],
            ["Blocks allocated", str(usage["allocated_blocks"])],
            ["Shared blocks", str(usage["shared_blocks"])],
        ],
        [25, 15],
    )

    efficiency = (
        tokens_saved / total_tokens_without_sharing * 100
        if total_tokens_without_sharing > 0
        else 0
    )
    print(f"\nCompute saved by prefix sharing: {efficiency:.1f}%")

    return efficiency


def test_copy_on_write_demo():
    """
    Bonus: Demonstrate Copy-on-Write behavior.
    """
    print_header("Bonus: Copy-on-Write (COW) Demonstration")

    from vllm_mlx.paged_cache import PagedCacheManager
    from vllm_mlx.prefix_cache import BlockAwarePrefixCache

    print("\nScenario: Fork a conversation and modify independently")
    print("COW ensures we only copy when actually modifying\n")

    paged_manager = PagedCacheManager(block_size=64, max_blocks=100)
    cache = BlockAwarePrefixCache(model=None, paged_cache_manager=paged_manager)

    # Original conversation
    original_tokens = list(range(128))  # 2 blocks
    cache.store_cache("original", original_tokens, ["original_kv_cache"])

    initial_blocks = paged_manager.stats.allocated_blocks
    print(f"Original conversation: 128 tokens, {initial_blocks} blocks")

    # Fork to new conversation (COW - no copy yet)
    cache.fork_cache("original", "forked")

    blocks_after_fork = paged_manager.stats.allocated_blocks
    shared_after_fork = paged_manager.stats.shared_blocks

    print("\nAfter fork (before modification):")
    print(f"  Blocks allocated: {blocks_after_fork} (same as before)")
    print(f"  Shared blocks: {shared_after_fork} (both point to same data)")

    # Get cache for generation - triggers COW if shared
    cache_data, was_copied = cache.get_cache_for_generation("forked")

    blocks_after_cow = paged_manager.stats.allocated_blocks
    cow_copies = paged_manager.stats.cow_copies

    print("\nAfter getting cache for generation (COW triggered):")
    print(f"  Was copied: {was_copied}")
    print(f"  Blocks allocated: {blocks_after_cow}")
    print(f"  COW copies made: {cow_copies}")
    print(f"  New blocks created: {blocks_after_cow - blocks_after_fork}")

    print("\nCOW ensures memory is only used when modifications occur!")


def main():
    """Run all benefit demonstrations."""
    print("\n" + "=" * 70)
    print("     PAGED KV CACHE BENEFITS DEMONSTRATION")
    print("=" * 70)

    results = {}

    # Run each test
    results["shared_prompts"] = test_benefit_1_shared_system_prompts()
    results["memory_efficiency"] = test_benefit_2_memory_efficiency()
    results["prefix_sharing"] = test_benefit_3_prefix_sharing()
    test_copy_on_write_demo()

    # Final summary
    print_header("FINAL SUMMARY")

    print("\nPaged KV Cache provides significant benefits:")
    print_table(
        ["Benefit", "Memory Savings"],
        [
            ["1. Shared System Prompts", f"{results['shared_prompts']:.1f}%"],
            ["2. Memory Efficiency", f"{results['memory_efficiency']:.1f}%"],
            ["3. Prefix Sharing", f"{results['prefix_sharing']:.1f}%"],
        ],
        [30, 20],
    )

    print("\nKey Features:")
    print("  - Block-based allocation (64 tokens/block)")
    print("  - Reference counting for safe sharing")
    print("  - Copy-on-Write for efficient forking")
    print("  - LRU eviction under memory pressure")
    print("  - Hash-based deduplication")

    print("\nUsage:")
    print("  rapid-mlx serve <model> --use-paged-cache")
    print()


if __name__ == "__main__":
    main()
