# SPDX-License-Identifier: Apache-2.0
"""
Tests for prefix cache functionality.

These tests verify the PrefixCacheManager for KV cache reuse
to speed up inference with repeated prompts.
"""

from unittest.mock import MagicMock

import pytest

from vllm_mlx.prefix_cache import (
    CacheEntry,
    PrefixCacheManager,
    PrefixCacheStats,
)


class TestPrefixCacheStats:
    """Tests for PrefixCacheStats class."""

    def test_initial_stats(self):
        """Test initial statistics are zero."""
        stats = PrefixCacheStats()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.tokens_saved == 0
        assert stats.total_queries == 0
        assert stats.evictions == 0
        assert stats.hit_rate == 0.0

    def test_hit_rate_calculation(self):
        """Test hit rate calculation."""
        stats = PrefixCacheStats(hits=3, misses=7, total_queries=10)
        assert stats.hit_rate == 0.3

    def test_hit_rate_zero_queries(self):
        """Test hit rate with zero queries."""
        stats = PrefixCacheStats()
        assert stats.hit_rate == 0.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        stats = PrefixCacheStats(hits=5, misses=5, tokens_saved=100, total_queries=10)
        d = stats.to_dict()
        assert d["hits"] == 5
        assert d["misses"] == 5
        assert d["hit_rate"] == 0.5
        assert d["tokens_saved"] == 100
        assert d["total_queries"] == 10


class TestCacheEntry:
    """Tests for CacheEntry class."""

    def test_cache_entry_creation(self):
        """Test creating a cache entry."""
        cache = ["mock_kv_cache"]
        entry = CacheEntry(prompt_cache=cache, count=1)
        assert entry.prompt_cache == ["mock_kv_cache"]
        assert entry.count == 1

    def test_cache_entry_count_increment(self):
        """Test incrementing reference count."""
        entry = CacheEntry(prompt_cache=["cache"], count=1)
        entry.count += 1
        assert entry.count == 2


class TestPrefixCacheManager:
    """Tests for PrefixCacheManager class."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model."""
        return MagicMock()

    @pytest.fixture
    def cache_manager(self, mock_model):
        """Create a cache manager with default settings."""
        return PrefixCacheManager(mock_model, max_entries=10)

    def test_initialization(self, mock_model):
        """Test cache manager initialization."""
        manager = PrefixCacheManager(mock_model, max_entries=50)
        assert manager.max_size == 50
        assert manager.model_key == id(mock_model)
        assert len(manager) == 0

    def test_fetch_empty_cache(self, cache_manager):
        """Test fetching from empty cache returns miss."""
        tokens = [1, 2, 3, 4, 5]
        cache, remaining = cache_manager.fetch_cache(tokens)

        assert cache is None
        assert remaining == tokens
        assert cache_manager.stats.misses == 1
        assert cache_manager.stats.hits == 0

    def test_store_and_fetch_exact_match(self, cache_manager):
        """Test storing and fetching exact match."""
        tokens = [1, 2, 3, 4, 5]
        mock_cache = ["kv_layer_1", "kv_layer_2"]

        # Store cache
        cache_manager.store_cache(tokens, mock_cache)
        assert len(cache_manager) == 1

        # Fetch exact match
        cache, remaining = cache_manager.fetch_cache(tokens)

        assert cache is not None
        assert remaining == []
        assert cache_manager.stats.hits == 1
        assert cache_manager.stats.tokens_saved == len(tokens)

    def test_fetch_shorter_prefix(self, cache_manager):
        """Test fetching when a shorter prefix is cached."""
        # Store short prefix
        short_tokens = [1, 2, 3]
        mock_cache = ["short_cache"]
        cache_manager.store_cache(short_tokens, mock_cache)

        # Fetch longer sequence
        long_tokens = [1, 2, 3, 4, 5, 6]
        cache, remaining = cache_manager.fetch_cache(long_tokens)

        assert cache is not None
        assert remaining == [4, 5, 6]
        assert cache_manager.stats.hits == 1
        assert cache_manager.stats.tokens_saved == len(short_tokens)

    def test_lru_eviction(self, mock_model):
        """Test LRU eviction when cache is full."""
        manager = PrefixCacheManager(mock_model, max_entries=3)

        # Fill cache
        manager.store_cache([1], ["cache1"])
        manager.store_cache([2], ["cache2"])
        manager.store_cache([3], ["cache3"])
        assert len(manager) == 3

        # Add one more - should evict oldest
        manager.store_cache([4], ["cache4"])
        assert len(manager) == 3
        assert manager.stats.evictions == 1

        # Token [1] should be evicted
        cache, _ = manager.fetch_cache([1])
        assert cache is None

    def test_lru_touch_on_access(self, mock_model):
        """Test that accessing a cache updates LRU order."""
        manager = PrefixCacheManager(mock_model, max_entries=3)

        # Fill cache
        manager.store_cache([1], ["cache1"])
        manager.store_cache([2], ["cache2"])
        manager.store_cache([3], ["cache3"])

        # Access [1] to make it most recently used
        manager.fetch_cache([1])

        # Add new entry - should evict [2] (oldest untouched)
        manager.store_cache([4], ["cache4"])

        # [1] should still be there
        cache, _ = manager.fetch_cache([1])
        assert cache is not None

        # [2] should be evicted
        cache, _ = manager.fetch_cache([2])
        assert cache is None

    def test_store_empty_tokens(self, cache_manager):
        """Test that empty tokens are not stored."""
        cache_manager.store_cache([], ["empty_cache"])
        assert len(cache_manager) == 0

    def test_get_stats(self, cache_manager):
        """Test getting statistics."""
        # Generate some activity
        cache_manager.store_cache([1, 2, 3], ["cache1"])
        cache_manager.fetch_cache([1, 2, 3])  # Hit
        cache_manager.fetch_cache([4, 5, 6])  # Miss

        stats = cache_manager.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5
        assert stats["total_queries"] == 2

    def test_reset_stats(self, cache_manager):
        """Test resetting statistics."""
        cache_manager.stats.hits = 10
        cache_manager.stats.misses = 5
        cache_manager.reset_stats()

        assert cache_manager.stats.hits == 0
        assert cache_manager.stats.misses == 0

    def test_clear(self, cache_manager):
        """Test clearing the cache."""
        cache_manager.store_cache([1, 2], ["cache1"])
        cache_manager.store_cache([3, 4], ["cache2"])
        assert len(cache_manager) == 2

        cache_manager.clear()
        assert len(cache_manager) == 0

        # Stats should also be reset
        assert cache_manager.stats.hits == 0

    def test_fetch_returns_deep_copy(self, cache_manager):
        """Test that fetched cache is a deep copy — mutations must not corrupt stored entry."""

        class MutableLayer:
            def __init__(self, offset):
                self.offset = offset

        original = [MutableLayer(10)]
        cache_manager.store_cache([1, 2], original)

        # Fetch and mutate the returned copy
        cache, _ = cache_manager.fetch_cache([1, 2])
        assert cache is not original
        cache[0].offset = 999

        # Fetch again — stored entry must be unchanged
        cache2, _ = cache_manager.fetch_cache([1, 2])
        assert cache2[0].offset == 10, "Stored cache entry was corrupted by mutation"

    def test_fetch_shorter_prefix_returns_deep_copy(self, cache_manager):
        """Test that shorter prefix fetch also returns a deep copy."""

        class MutableLayer:
            def __init__(self, offset):
                self.offset = offset

        original = [MutableLayer(20)]
        cache_manager.store_cache([1, 2, 3], original)

        # Fetch shorter prefix
        cache, remaining = cache_manager.fetch_cache([1, 2, 3, 4, 5])
        assert remaining == [4, 5]
        assert cache is not original
        cache[0].offset = 999

        # Fetch again — stored entry must be unchanged
        cache2, _ = cache_manager.fetch_cache([1, 2, 3])
        assert cache2[0].offset == 20, "Stored cache entry was corrupted by mutation"

    def test_multiple_prefixes(self, cache_manager):
        """Test multiple different prefixes."""
        cache_manager.store_cache([1, 2], ["cache_a"])
        cache_manager.store_cache([3, 4], ["cache_b"])
        cache_manager.store_cache([1, 2, 3], ["cache_c"])

        # Fetch each
        cache_a, _ = cache_manager.fetch_cache([1, 2])
        cache_b, _ = cache_manager.fetch_cache([3, 4])
        cache_c, _ = cache_manager.fetch_cache([1, 2, 3])

        assert cache_a == ["cache_a"]
        assert cache_b == ["cache_b"]
        assert cache_c == ["cache_c"]

    def test_trie_structure(self, cache_manager):
        """Test trie correctly handles branching prefixes."""
        # Store two prefixes with common start
        cache_manager.store_cache([1, 2, 3], ["cache_123"])
        cache_manager.store_cache([1, 2, 4], ["cache_124"])

        # Fetch the common prefix should return shorter match
        cache, remaining = cache_manager.fetch_cache([1, 2])
        # No exact match for [1,2], but [1,2,3] is longer - behavior depends on implementation
        # In our implementation, we find shorter prefix if available, otherwise return miss

        # Fetch exact matches
        cache_123, _ = cache_manager.fetch_cache([1, 2, 3])
        cache_124, _ = cache_manager.fetch_cache([1, 2, 4])

        assert cache_123 == ["cache_123"]
        assert cache_124 == ["cache_124"]


class TestPrefixCachePinning:
    """Tests for pin/unpin prefix functionality."""

    @pytest.fixture
    def mock_model(self):
        return MagicMock()

    def test_pin_survives_store(self, mock_model):
        """Pinned entry stays pinned after store_cache re-stores same key (regression)."""
        mgr = PrefixCacheManager(mock_model, max_entries=3)
        mgr.store_cache([1, 2], ["cache_a"])
        assert mgr.pin_prefix([1, 2]) is True

        # Re-store same tokens — pin must not be silently undone
        mgr.store_cache([1, 2], ["cache_b"])

        # Fill remaining capacity and overflow — pinned entry must survive
        mgr.store_cache([3], ["cache_c"])
        mgr.store_cache([4], ["cache_d"])
        mgr.store_cache([5], ["cache_e"])  # would evict [1,2] if it were in LRU

        cache, _ = mgr.fetch_cache([1, 2])
        assert cache is not None, "Pinned entry was evicted after store_cache"

    def test_pin_survives_touch(self, mock_model):
        """Pinned entry stays pinned after fetch_cache touches it (regression)."""
        mgr = PrefixCacheManager(mock_model, max_entries=3)
        mgr.store_cache([1, 2], ["cache_a"])
        mgr.pin_prefix([1, 2])

        # Access pinned entry — must not re-add to LRU
        mgr.fetch_cache([1, 2])

        # Overflow — pinned entry must survive
        mgr.store_cache([3], ["c"])
        mgr.store_cache([4], ["d"])
        mgr.store_cache([5], ["e"])
        mgr.store_cache([6], ["f"])

        cache, _ = mgr.fetch_cache([1, 2])
        assert cache is not None, "Pinned entry was evicted after fetch_cache touch"

    def test_pin_capacity_guard(self, mock_model):
        """pin_prefix rejects when pinned count reaches max_size (regression)."""
        mgr = PrefixCacheManager(mock_model, max_entries=2)
        mgr.store_cache([1], ["a"])
        mgr.store_cache([2], ["b"])
        assert mgr.pin_prefix([1]) is True
        assert mgr.pin_prefix([2]) is True

        # Now at capacity — next pin must fail
        mgr.store_cache([3], ["c"])  # won't fit in LRU, but trie entry exists
        # Actually [3] can't be stored because LRU+pinned > max_size and LRU is empty
        # So test with existing entries:
        # Unpin one, store a new entry, try to pin 3 total
        mgr.unpin_prefix([2])
        mgr.store_cache([3], ["c"])
        assert mgr.pin_prefix([3]) is True  # now 2 pinned (max_size=2)

        mgr.store_cache([4], ["d"])
        assert mgr.pin_prefix([4]) is False, "Pin should fail when at capacity"

    def test_unpin_restores_lru(self, mock_model):
        """Unpinned entry becomes evictable again."""
        mgr = PrefixCacheManager(mock_model, max_entries=2)
        mgr.store_cache([1], ["a"])
        mgr.store_cache([2], ["b"])
        mgr.pin_prefix([1])

        mgr.unpin_prefix([1])

        # Now [1] is back in LRU and can be evicted
        mgr.store_cache([3], ["c"])
        mgr.store_cache([4], ["d"])

        cache, _ = mgr.fetch_cache([1])
        assert cache is None, "Unpinned entry should be evictable"

    def test_clear_resets_pinned(self, mock_model):
        """clear() removes pinned entries too."""
        mgr = PrefixCacheManager(mock_model, max_entries=5)
        mgr.store_cache([1], ["a"])
        mgr.pin_prefix([1])
        mgr.clear()
        assert len(mgr) == 0


class TestSchedulerIntegration:
    """Test integration with scheduler."""

    def test_request_cache_fields(self):
        """Test that Request has cache fields."""
        from vllm_mlx.request import Request, SamplingParams

        request = Request(
            request_id="test-1",
            prompt="Hello",
            sampling_params=SamplingParams(),
        )

        # Check cache fields exist
        assert hasattr(request, "prompt_cache")
        assert hasattr(request, "cached_tokens")
        assert hasattr(request, "remaining_tokens")

        # Check defaults
        assert request.prompt_cache is None
        assert request.cached_tokens == 0
        assert request.remaining_tokens is None

    def test_scheduler_config_cache_options(self):
        """Test scheduler config has cache options."""
        from vllm_mlx.scheduler import SchedulerConfig

        config = SchedulerConfig(
            enable_prefix_cache=True,
            prefix_cache_size=200,
        )

        assert config.enable_prefix_cache is True
        assert config.prefix_cache_size == 200

        # Test defaults
        default_config = SchedulerConfig()
        assert default_config.enable_prefix_cache is True
        assert default_config.prefix_cache_size == 100


if __name__ == "__main__":
    # Verbose standalone test with real model
    import argparse
    import asyncio
    import os
    import time

    parser = argparse.ArgumentParser(description="LLM prefix cache benchmark")
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("VLLM_MLX_TEST_MODEL", "mlx-community/Qwen3-0.6B-8bit"),
        help="Model to benchmark",
    )
    args = parser.parse_args()

    MODEL_NAME = args.model

    def print_header(title):
        print("\n" + "=" * 70)
        print(f"  {title}")
        print("=" * 70)

    def print_subheader(title):
        print("\n" + "-" * 70)
        print(f"  {title}")
        print("-" * 70)

    def print_table(headers, rows):
        """Print a formatted table."""
        # Calculate column widths
        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(str(cell)))

        # Print header
        header_line = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
        separator = "-+-".join("-" * w for w in col_widths)
        print(f"    {header_line}")
        print(f"    {separator}")

        # Print rows
        for row in rows:
            row_line = " | ".join(
                str(cell).ljust(col_widths[i]) for i, cell in enumerate(row)
            )
            print(f"    {row_line}")

    def print_stats_table(stats, title="Cache Statistics"):
        """Print cache stats as a table."""
        print(f"\n    {title}:")
        hits = stats.get("hits", 0)
        misses = stats.get("misses", 0)
        total_queries = stats.get("total_queries", hits + misses)
        hit_rate = stats.get("hit_rate")
        if hit_rate is None:
            hit_rate = (hits / total_queries) if total_queries > 0 else 0.0

        headers = ["Metric", "Value"]
        rows = [
            ["Hits", hits],
            ["Misses", misses],
            ["Hit Rate", f"{hit_rate * 100:.1f}%"],
            ["Tokens Saved", stats.get("tokens_saved", 0)],
            ["Total Queries", total_queries],
        ]
        print_table(headers, rows)

    async def run_cache_test():
        from mlx_lm import load

        from vllm_mlx import (
            AsyncEngineCore,
            EngineConfig,
            SamplingParams,
            SchedulerConfig,
        )

        print_header("LLM PREFIX CACHE TEST")
        print(f"\n  Model: {MODEL_NAME}")
        print("  Test: Verify KV cache reuse for repeated prompts")
        print("  Expected behavior:")
        print("    - Same prompt → cache HIT (skip prompt processing)")
        print(
            "    - Different prompt → cache MISS or PREFIX_HIT (shared template tokens)"
        )

        print_subheader("Loading Model")
        load_start = time.perf_counter()
        model, tokenizer = load(MODEL_NAME)
        load_time = time.perf_counter() - load_start
        print(f"    Model loaded in {load_time:.2f}s")

        config = EngineConfig(
            model_name="test",
            scheduler_config=SchedulerConfig(
                enable_prefix_cache=True,
                prefix_cache_size=100,
            ),
        )

        async with AsyncEngineCore(model, tokenizer, config) as engine:
            await asyncio.sleep(0.1)

            # Test prompts
            prompt1 = "What is 2+2?"
            prompt2 = "What is the capital of France?"

            formatted1 = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt1}],
                tokenize=False,
                add_generation_prompt=True,
            )
            formatted2 = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt2}],
                tokenize=False,
                add_generation_prompt=True,
            )

            tokens1 = len(tokenizer.encode(formatted1))
            tokens2 = len(tokenizer.encode(formatted2))

            params = SamplingParams(max_tokens=20, temperature=0.0)

            # Collect test results for final table
            test_results = []

            # ============================================================
            # TEST 1: First request - should be cache MISS
            # ============================================================
            print_subheader("TEST 1: First Request (Cache Miss Expected)")
            print(f'    Prompt: "{prompt1}"')
            print(f"    Tokens: {tokens1}")

            start = time.perf_counter()
            rid1 = await engine.add_request(formatted1, params)
            response1 = ""
            async for out in engine.stream_outputs(rid1, timeout=30):
                if out.output_text:
                    response1 = out.output_text
                if out.finished:
                    break
            t1 = time.perf_counter() - start

            stats1 = engine.get_cache_stats()
            test1_pass = stats1["misses"] == 1 and stats1["hits"] == 0
            test_results.append(
                [
                    "TEST 1",
                    "First request",
                    "MISS",
                    "MISS" if stats1["hits"] == 0 else "HIT",
                    f"{t1 * 1000:.1f}ms",
                    "PASS" if test1_pass else "FAIL",
                ]
            )

            print(f'    Response: "{response1.strip()[:50]}..."')
            print_stats_table(stats1)

            # ============================================================
            # TEST 2: Same prompt again - should be cache HIT
            # ============================================================
            print_subheader("TEST 2: Same Prompt Again (Cache Hit Expected)")
            print(f'    Prompt: "{prompt1}" (same as TEST 1)')
            print(f"    Tokens: {tokens1}")

            start = time.perf_counter()
            rid2 = await engine.add_request(formatted1, params)
            response2 = ""
            async for out in engine.stream_outputs(rid2, timeout=30):
                if out.output_text:
                    response2 = out.output_text
                if out.finished:
                    break
            t2 = time.perf_counter() - start

            stats2 = engine.get_cache_stats()
            test2_pass = stats2["hits"] == 1
            test_results.append(
                [
                    "TEST 2",
                    "Same prompt (cached)",
                    "HIT",
                    "HIT" if stats2["hits"] > stats1["hits"] else "MISS",
                    f"{t2 * 1000:.1f}ms",
                    "PASS" if test2_pass else "FAIL",
                ]
            )

            print(f'    Response: "{response2.strip()[:50]}..."')
            speedup = t1 / t2 if t2 > 0 else 0
            print(f"    Speedup: {speedup:.2f}x faster")
            print_stats_table(stats2)

            # ============================================================
            # TEST 3: Different prompt - should be cache MISS or PREFIX_HIT
            # ============================================================
            print_subheader(
                "TEST 3: Different Prompt (Cache Miss or Prefix Hit Expected)"
            )
            print(f'    Prompt: "{prompt2}" (different from TEST 1)')
            print(f"    Tokens: {tokens2}")

            start = time.perf_counter()
            rid3 = await engine.add_request(formatted2, params)
            response3 = ""
            async for out in engine.stream_outputs(rid3, timeout=30):
                if out.output_text:
                    response3 = out.output_text
                if out.finished:
                    break
            t3 = time.perf_counter() - start

            stats3 = engine.get_cache_stats()
            hits_delta = stats3["hits"] - stats2["hits"]
            misses_delta = stats3["misses"] - stats2["misses"]
            tokens_saved_delta = stats3.get("tokens_saved", 0) - stats2.get(
                "tokens_saved", 0
            )

            # Different prompts may still share template/system prefix tokens.
            # Treat either a true miss OR any prefix-hit reuse as valid behavior.
            actual3 = "HIT"
            if misses_delta > 0:
                actual3 = "MISS"
                test3_pass = True
            elif hits_delta > 0:
                actual3 = (
                    f"PREFIX_HIT({tokens_saved_delta} tok)"
                    if tokens_saved_delta > 0
                    else "PREFIX_HIT"
                )
                test3_pass = True
            else:
                test3_pass = False

            test_results.append(
                [
                    "TEST 3",
                    "Different prompt",
                    "MISS or PREFIX_HIT",
                    actual3,
                    f"{t3 * 1000:.1f}ms",
                    "PASS" if test3_pass else "FAIL",
                ]
            )

            print(f'    Response: "{response3.strip()[:50]}..."')
            print_stats_table(stats3)

            # ============================================================
            # SUMMARY TABLE
            # ============================================================
            print_header("TEST RESULTS SUMMARY")

            # Test results table
            print("\n    Test Results:")
            print_table(
                ["Test", "Description", "Expected", "Actual", "Time", "Status"],
                test_results,
            )

            # Final stats table
            final_stats = engine.get_cache_stats()
            print("\n    Final Cache Statistics:")
            print_table(
                ["Metric", "Value"],
                [
                    ["Total Requests", 3],
                    ["Cache Hits", final_stats["hits"]],
                    ["Cache Misses", final_stats["misses"]],
                    ["Hit Rate", f"{final_stats['hit_rate'] * 100:.1f}%"],
                    ["Tokens Saved", final_stats["tokens_saved"]],
                    ["Speedup (cached)", f"{speedup:.2f}x"],
                ],
            )

            all_passed = test1_pass and test2_pass and test3_pass

            print("\n" + "=" * 70)
            if all_passed:
                print("  [OK] ALL TESTS PASSED - Prefix cache working correctly")
            else:
                print("  [FAILED] SOME TESTS FAILED - Check results above")
            print("=" * 70)

    asyncio.run(run_cache_test())
