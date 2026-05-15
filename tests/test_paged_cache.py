# SPDX-License-Identifier: Apache-2.0
"""Tests for Paged KV Cache Manager."""

import platform
import sys
import time

import pytest

# Skip all tests if not on Apple Silicon
pytestmark = pytest.mark.skipif(
    sys.platform != "darwin" or platform.machine() != "arm64",
    reason="Requires Apple Silicon",
)


class TestCacheBlock:
    """Test CacheBlock dataclass."""

    def test_cache_block_creation(self):
        """Test creating a CacheBlock."""
        from vllm_mlx.paged_cache import CacheBlock

        block = CacheBlock(block_id=0)
        assert block.block_id == 0
        assert block.token_count == 0
        assert block.ref_count == 0  # vLLM style: starts at 0, set to 1 when allocated
        assert block.hash_value is None
        assert block.cache_data is None

    def test_cache_block_is_full(self):
        """Test is_full method."""
        from vllm_mlx.paged_cache import CacheBlock

        block = CacheBlock(block_id=0, token_count=64)
        assert block.is_full(64) is True
        assert block.is_full(128) is False

        block.token_count = 32
        assert block.is_full(64) is False

    def test_cache_block_is_shared(self):
        """Test is_shared method."""
        from vllm_mlx.paged_cache import CacheBlock

        block = CacheBlock(block_id=0, ref_count=1)
        assert block.is_shared() is False

        block.ref_count = 2
        assert block.is_shared() is True

    def test_cache_block_touch(self):
        """Test touch updates last_access."""
        from vllm_mlx.paged_cache import CacheBlock

        block = CacheBlock(block_id=0)
        old_time = block.last_access
        time.sleep(0.01)
        block.touch()
        assert block.last_access > old_time


class TestBlockTable:
    """Test BlockTable dataclass."""

    def test_block_table_creation(self):
        """Test creating a BlockTable."""
        from vllm_mlx.paged_cache import BlockTable

        table = BlockTable(request_id="req-1")
        assert table.request_id == "req-1"
        assert table.block_ids == []
        assert table.num_tokens == 0
        assert len(table) == 0

    def test_block_table_copy(self):
        """Test copying a BlockTable."""
        from vllm_mlx.paged_cache import BlockTable

        table = BlockTable(
            request_id="req-1",
            block_ids=[0, 1, 2],
            num_tokens=192,
        )

        copied = table.copy("req-2")
        assert copied.request_id == "req-2"
        assert copied.block_ids == [0, 1, 2]
        assert copied.num_tokens == 192

        # Verify independence
        copied.block_ids.append(3)
        assert table.block_ids == [0, 1, 2]


class TestPagedCacheManager:
    """Test PagedCacheManager class."""

    def test_initialization(self):
        """Test manager initialization."""
        from vllm_mlx.paged_cache import PagedCacheManager

        manager = PagedCacheManager(block_size=64, max_blocks=100)

        assert manager.block_size == 64
        assert manager.max_blocks == 100
        # vLLM style: free_blocks is an int property, and null block takes 1 slot
        assert manager.free_blocks == 99  # 100 - 1 (null block)
        assert len(manager.allocated_blocks) == 1  # null block is allocated

        stats = manager.get_stats()
        assert stats.total_blocks == 100
        assert stats.free_blocks == 99
        assert stats.allocated_blocks == 1  # null block

    def test_allocate_block(self):
        """Test block allocation."""
        from vllm_mlx.paged_cache import PagedCacheManager

        manager = PagedCacheManager(block_size=64, max_blocks=10)
        # Initial: 10 blocks, 1 null block, so 9 free

        block = manager.allocate_block()
        assert block is not None
        assert block.block_id in manager.allocated_blocks
        assert manager.free_blocks == 8  # 9 - 1

        stats = manager.get_stats()
        assert stats.allocated_blocks == 2  # null block + 1 allocated
        assert stats.free_blocks == 8

    def test_allocate_all_blocks(self):
        """Test allocating all available blocks."""
        from vllm_mlx.paged_cache import PagedCacheManager

        manager = PagedCacheManager(block_size=64, max_blocks=5)
        # With null block taking 1 slot, we have 4 allocatable blocks

        blocks = []
        for _ in range(4):  # Can only allocate 4 (5 - 1 null block)
            block = manager.allocate_block()
            assert block is not None
            blocks.append(block)

        # Should return None when out of blocks
        assert manager.allocate_block() is None
        assert manager.free_blocks == 0

    def test_free_block(self):
        """Test block deallocation."""
        from vllm_mlx.paged_cache import PagedCacheManager

        manager = PagedCacheManager(block_size=64, max_blocks=10)
        initial_free = manager.free_blocks  # 9 (10 - 1 null block)

        block = manager.allocate_block()
        block_id = block.block_id
        assert manager.free_blocks == initial_free - 1

        result = manager.free_block(block_id)
        assert result is True
        assert block_id not in manager.allocated_blocks
        # Block should be back in free queue
        assert manager.free_blocks == initial_free

    def test_reference_counting(self):
        """Test reference counting."""
        from vllm_mlx.paged_cache import PagedCacheManager

        manager = PagedCacheManager(block_size=64, max_blocks=10)

        block = manager.allocate_block()
        block_id = block.block_id
        assert block.ref_count == 1

        # Increment ref
        manager.increment_ref(block_id)
        assert block.ref_count == 2

        # Free should decrement, not remove
        result = manager.free_block(block_id)
        assert result is False  # Still referenced
        assert block.ref_count == 1
        assert block_id in manager.allocated_blocks

        # Free again should remove
        result = manager.free_block(block_id)
        assert result is True
        assert block_id not in manager.allocated_blocks

    def test_allocate_blocks_for_tokens(self):
        """Test allocating blocks for a token count."""
        from vllm_mlx.paged_cache import PagedCacheManager

        manager = PagedCacheManager(block_size=64, max_blocks=100)

        # 100 tokens should need 2 blocks (ceil(100/64) = 2)
        blocks = manager.allocate_blocks_for_tokens(100)
        assert len(blocks) == 2

        # 64 tokens should need 1 block
        blocks = manager.allocate_blocks_for_tokens(64)
        assert len(blocks) == 1

        # 65 tokens should need 2 blocks
        blocks = manager.allocate_blocks_for_tokens(65)
        assert len(blocks) == 2

    def test_allocate_blocks_for_tokens_rollback(self):
        """Test rollback when allocation fails."""
        from vllm_mlx.paged_cache import PagedCacheManager

        manager = PagedCacheManager(block_size=64, max_blocks=3)
        # With null block, we have 2 allocatable blocks
        initial_free = manager.free_blocks  # 2

        # Try to allocate more than available (300 tokens needs 5 blocks)
        # vLLM style: raises ValueError instead of returning empty list
        try:
            blocks = manager.allocate_blocks_for_tokens(300)
            assert False, "Expected ValueError"
        except ValueError:
            pass

        # All blocks should be unchanged (no rollback needed since allocation failed)
        assert manager.free_blocks == initial_free


class TestHashBasedDeduplication:
    """Test hash-based deduplication."""

    def test_compute_block_hash(self):
        """Test hash computation."""
        from vllm_mlx.paged_cache import PagedCacheManager

        tokens1 = [1, 2, 3, 4, 5]
        tokens2 = [1, 2, 3, 4, 5]
        tokens3 = [1, 2, 3, 4, 6]

        hash1 = PagedCacheManager.compute_block_hash(tokens1)
        hash2 = PagedCacheManager.compute_block_hash(tokens2)
        hash3 = PagedCacheManager.compute_block_hash(tokens3)

        assert hash1 == hash2  # Same tokens = same hash
        assert hash1 != hash3  # Different tokens = different hash
        assert len(hash1) == 16  # 16 char hex string

    def test_find_cached_block(self):
        """Test finding cached block by tokens."""
        from vllm_mlx.paged_cache import PagedCacheManager

        manager = PagedCacheManager(block_size=64, max_blocks=10)

        tokens = list(range(64))

        # Initially not found
        result = manager.find_cached_block(tokens)
        assert result is None

        # Register a block
        block = manager.allocate_block()
        manager.register_block_hash(block, tokens)

        # Now should find it
        result = manager.find_cached_block(tokens)
        assert result is not None
        assert result.block_id == block.block_id


class TestBlockTableManagement:
    """Test block table management."""

    def test_create_block_table(self):
        """Test creating a block table."""
        from vllm_mlx.paged_cache import PagedCacheManager

        manager = PagedCacheManager(block_size=64, max_blocks=10)

        table = manager.create_block_table("req-1")
        assert table.request_id == "req-1"
        assert "req-1" in manager.request_tables

    def test_get_block_table(self):
        """Test getting a block table."""
        from vllm_mlx.paged_cache import PagedCacheManager

        manager = PagedCacheManager(block_size=64, max_blocks=10)

        manager.create_block_table("req-1")

        table = manager.get_block_table("req-1")
        assert table is not None
        assert table.request_id == "req-1"

        # Non-existent table
        assert manager.get_block_table("req-999") is None

    def test_delete_block_table(self):
        """Test deleting a block table frees blocks."""
        from vllm_mlx.paged_cache import PagedCacheManager

        manager = PagedCacheManager(block_size=64, max_blocks=10)
        # Initial: 9 free (10 - 1 null block), 1 allocated (null block)

        table = manager.create_block_table("req-1")
        block1 = manager.allocate_block()
        block2 = manager.allocate_block()
        manager.add_block_to_table(table, block1, 64)
        manager.add_block_to_table(table, block2, 64)

        assert len(manager.allocated_blocks) == 3  # null block + 2

        manager.delete_block_table("req-1")

        assert "req-1" not in manager.request_tables
        assert len(manager.allocated_blocks) == 1  # only null block remains
        assert manager.free_blocks == 9  # all non-null blocks free


class TestPrefixSharing:
    """Test prefix sharing functionality."""

    def test_find_shared_prefix_no_cache(self):
        """Test finding shared prefix with empty cache."""
        from vllm_mlx.paged_cache import PagedCacheManager

        manager = PagedCacheManager(block_size=64, max_blocks=10)

        tokens = list(range(200))
        shared_blocks, remaining = manager.find_shared_prefix(tokens)

        assert len(shared_blocks) == 0
        assert remaining == tokens

    def test_find_shared_prefix_with_cache(self):
        """Test finding shared prefix with cached blocks."""
        from vllm_mlx.paged_cache import PagedCacheManager

        manager = PagedCacheManager(block_size=64, max_blocks=10)

        # Cache the first block
        first_block_tokens = list(range(64))
        block = manager.allocate_block()
        block.token_count = 64
        manager.register_block_hash(block, first_block_tokens)

        # Search with tokens that start with cached prefix
        tokens = list(range(128))  # 64 cached + 64 new
        shared_blocks, remaining = manager.find_shared_prefix(tokens)

        assert len(shared_blocks) == 1
        assert shared_blocks[0] == block.block_id
        assert remaining == list(range(64, 128))

    def test_fork_block_table(self):
        """Test forking a block table (COW)."""
        from vllm_mlx.paged_cache import PagedCacheManager

        manager = PagedCacheManager(block_size=64, max_blocks=10)

        # Create source table with blocks
        source_table = manager.create_block_table("req-1")
        block1 = manager.allocate_block()
        block2 = manager.allocate_block()
        manager.add_block_to_table(source_table, block1, 64)
        manager.add_block_to_table(source_table, block2, 64)

        # Fork to new request
        forked_table = manager.fork_block_table(source_table, "req-2")

        assert forked_table.request_id == "req-2"
        assert forked_table.block_ids == source_table.block_ids
        assert forked_table.num_tokens == source_table.num_tokens

        # Blocks should now have ref_count = 2
        assert block1.ref_count == 2
        assert block2.ref_count == 2


class TestCopyOnWrite:
    """Test Copy-on-Write functionality."""

    def test_get_blocks_no_cow_needed(self):
        """Test getting blocks when no COW is needed."""
        from vllm_mlx.paged_cache import PagedCacheManager

        manager = PagedCacheManager(block_size=64, max_blocks=10)

        table = manager.create_block_table("req-1")
        block = manager.allocate_block()
        manager.add_block_to_table(table, block, 64)

        blocks, was_copied = manager.get_blocks_for_generation(table)

        assert len(blocks) == 1
        assert was_copied is False
        assert blocks[0].block_id == block.block_id

    def test_get_blocks_with_cow(self):
        """Test getting blocks triggers COW for shared blocks."""
        from vllm_mlx.paged_cache import PagedCacheManager

        manager = PagedCacheManager(block_size=64, max_blocks=10)

        # Create and fork table
        source_table = manager.create_block_table("req-1")
        block = manager.allocate_block()
        manager.add_block_to_table(source_table, block, 64)

        forked_table = manager.fork_block_table(source_table, "req-2")
        assert block.ref_count == 2

        # Get blocks for forked table - should trigger COW
        blocks, was_copied = manager.get_blocks_for_generation(forked_table)

        assert len(blocks) == 1
        assert was_copied is True
        assert blocks[0].block_id != block.block_id  # New block created
        assert block.ref_count == 1  # Original block ref decreased

        stats = manager.get_stats()
        assert stats.cow_copies == 1


class TestEviction:
    """Test LRU eviction."""

    def test_evict_lru_blocks(self):
        """Test LRU eviction."""
        from vllm_mlx.paged_cache import PagedCacheManager

        manager = PagedCacheManager(block_size=64, max_blocks=5)
        # With null block, we have 4 allocatable blocks

        # Allocate all blocks
        blocks = []
        for _ in range(4):  # 4 allocatable (5 - 1 null block)
            block = manager.allocate_block()
            block.token_count = 64
            blocks.append(block)
            time.sleep(0.01)  # Ensure different timestamps

        assert manager.free_blocks == 0

        # Free 2 blocks first (they go to free queue)
        manager.free_block(blocks[0].block_id)
        manager.free_block(blocks[1].block_id)
        assert manager.free_blocks == 2

        # Now evict_lru_blocks rotates them to clear cache data
        evicted = manager.evict_lru_blocks(2)

        assert evicted == 2
        assert manager.free_blocks == 2
        assert len(manager.allocated_blocks) == 3  # null block + 2 remaining

    def test_handle_memory_pressure(self):
        """Test handling memory pressure."""
        from vllm_mlx.paged_cache import PagedCacheManager

        manager = PagedCacheManager(block_size=64, max_blocks=5)
        # With null block, we have 4 allocatable blocks

        # Allocate 3 blocks
        allocated = []
        for _ in range(3):
            block = manager.allocate_block()
            block.token_count = 64
            allocated.append(block)

        assert manager.free_blocks == 1  # 4 - 3 = 1

        # Free 2 blocks to put them in free queue (they can be evicted from cache)
        manager.free_block(allocated[0].block_id)
        manager.free_block(allocated[1].block_id)
        assert manager.free_blocks == 3

        # Request 3 blocks - should already have enough
        result = manager.handle_memory_pressure(3)

        assert result is True
        assert manager.free_blocks >= 3


class TestStatistics:
    """Test statistics and monitoring."""

    def test_get_memory_usage(self):
        """Test memory usage reporting."""
        from vllm_mlx.paged_cache import PagedCacheManager

        manager = PagedCacheManager(block_size=64, max_blocks=100)
        # Initial: 99 free (100 - 1 null block), 1 allocated (null block)

        # Allocate 25 blocks
        for _ in range(25):
            block = manager.allocate_block()
            block.token_count = 64

        usage = manager.get_memory_usage()

        assert usage["block_size"] == 64
        assert usage["max_blocks"] == 100
        assert usage["allocated_blocks"] == 26  # null block + 25
        assert usage["free_blocks"] == 74  # 99 - 25
        assert usage["utilization"] == 0.26  # 26/100
        assert usage["total_tokens_cached"] == 0  # Not added via add_block_to_table

    def test_reset_stats(self):
        """Test resetting statistics."""
        from vllm_mlx.paged_cache import PagedCacheManager

        manager = PagedCacheManager(block_size=64, max_blocks=10)

        # Generate some stats
        manager.find_cached_block([1, 2, 3])  # Cache miss
        manager.stats.cow_copies = 5

        manager.reset_stats()

        assert manager.stats.cache_hits == 0
        assert manager.stats.cache_misses == 0
        assert manager.stats.cow_copies == 0

    def test_clear(self):
        """Test clearing all cache."""
        from vllm_mlx.paged_cache import PagedCacheManager

        manager = PagedCacheManager(block_size=64, max_blocks=10)

        # Allocate and populate
        table = manager.create_block_table("req-1")
        block = manager.allocate_block()
        manager.add_block_to_table(table, block, 64)

        manager.clear()

        # After clear, null block is re-reserved
        assert manager.free_blocks == 9  # 10 - 1 null block
        assert len(manager.allocated_blocks) == 1  # only null block
        assert len(manager.request_tables) == 0
        assert len(manager.hash_to_block) == 0


class TestThreadSafety:
    """Test thread safety."""

    def test_concurrent_allocation(self):
        """Test concurrent block allocation."""
        import threading

        from vllm_mlx.paged_cache import PagedCacheManager

        # Use 101 blocks so we have 100 allocatable (after null block)
        manager = PagedCacheManager(block_size=64, max_blocks=101)
        results = []
        errors = []

        def allocate_blocks():
            try:
                for _ in range(10):
                    block = manager.allocate_block()
                    if block:
                        results.append(block.block_id)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=allocate_blocks) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 50
        assert len(set(results)) == 50  # All unique block IDs


# =============================================================================
# BlockAwarePrefixCache Tests
# =============================================================================


class TestBlockAwarePrefixCache:
    """Test BlockAwarePrefixCache class."""

    def test_initialization(self):
        """Test cache initialization."""
        from vllm_mlx.paged_cache import PagedCacheManager
        from vllm_mlx.prefix_cache import BlockAwarePrefixCache

        paged_manager = PagedCacheManager(block_size=64, max_blocks=100)
        cache = BlockAwarePrefixCache(model=None, paged_cache_manager=paged_manager)

        assert cache.block_size == 64
        assert len(cache) == 0

    def test_store_and_fetch_cache(self):
        """Test storing and fetching cache."""
        from vllm_mlx.paged_cache import PagedCacheManager
        from vllm_mlx.prefix_cache import BlockAwarePrefixCache

        paged_manager = PagedCacheManager(block_size=64, max_blocks=100)
        cache = BlockAwarePrefixCache(model=None, paged_cache_manager=paged_manager)

        # Store cache for first request
        tokens1 = list(range(128))  # 2 blocks worth
        cache_data1 = ["cache_data_1"]
        block_table = cache.store_cache("req-1", tokens1, cache_data1)

        assert block_table is not None
        assert block_table.num_tokens == 128
        assert len(block_table.block_ids) == 2

        # Fetch cache for second request with same prefix
        block_table2, remaining = cache.fetch_cache("req-2", tokens1 + [999, 1000])

        # Should hit the prefix
        assert remaining == [999, 1000]

    def test_release_cache(self):
        """Test releasing cache."""
        from vllm_mlx.paged_cache import PagedCacheManager
        from vllm_mlx.prefix_cache import BlockAwarePrefixCache

        paged_manager = PagedCacheManager(block_size=64, max_blocks=100)
        cache = BlockAwarePrefixCache(model=None, paged_cache_manager=paged_manager)

        tokens = list(range(64))
        cache.store_cache("req-1", tokens, ["data"])

        assert len(cache) == 1

        cache.release_cache("req-1")

        assert len(cache) == 0

    def test_fork_cache(self):
        """Test forking cache (COW)."""
        from vllm_mlx.paged_cache import PagedCacheManager
        from vllm_mlx.prefix_cache import BlockAwarePrefixCache

        paged_manager = PagedCacheManager(block_size=64, max_blocks=100)
        cache = BlockAwarePrefixCache(model=None, paged_cache_manager=paged_manager)

        tokens = list(range(128))
        cache.store_cache("req-1", tokens, ["shared_data"])

        # Fork to new request
        forked_table = cache.fork_cache("req-1", "req-2")

        assert forked_table is not None
        assert len(cache) == 2

        # Both should share the same blocks
        stats = cache.get_stats()
        assert stats["shared_blocks"] > 0

    def test_get_cache_for_generation(self):
        """Test getting cache for generation with COW."""
        from vllm_mlx.paged_cache import PagedCacheManager
        from vllm_mlx.prefix_cache import BlockAwarePrefixCache

        paged_manager = PagedCacheManager(block_size=64, max_blocks=100)
        cache = BlockAwarePrefixCache(model=None, paged_cache_manager=paged_manager)

        tokens = list(range(64))
        cache.store_cache("req-1", tokens, ["data"])

        # Get cache for generation (no COW needed)
        cache_data, was_copied = cache.get_cache_for_generation("req-1")

        assert cache_data == ["data"]
        assert was_copied is False

    def test_get_cache_for_generation_with_cow(self):
        """Test COW is triggered for shared blocks."""
        from vllm_mlx.paged_cache import PagedCacheManager
        from vllm_mlx.prefix_cache import BlockAwarePrefixCache

        paged_manager = PagedCacheManager(block_size=64, max_blocks=100)
        cache = BlockAwarePrefixCache(model=None, paged_cache_manager=paged_manager)

        tokens = list(range(64))
        cache.store_cache("req-1", tokens, ["shared_data"])
        cache.fork_cache("req-1", "req-2")

        # Get cache for forked request - should trigger COW
        cache_data, was_copied = cache.get_cache_for_generation("req-2")

        assert cache_data is not None
        assert was_copied is True

    def test_stats(self):
        """Test statistics."""
        from vllm_mlx.paged_cache import PagedCacheManager
        from vllm_mlx.prefix_cache import BlockAwarePrefixCache

        paged_manager = PagedCacheManager(block_size=64, max_blocks=100)
        cache = BlockAwarePrefixCache(model=None, paged_cache_manager=paged_manager)

        # Miss
        cache.fetch_cache("req-1", [1, 2, 3])

        stats = cache.get_stats()
        assert stats["misses"] == 1
        assert stats["hits"] == 0

    def test_clear(self):
        """Test clearing cache."""
        from vllm_mlx.paged_cache import PagedCacheManager
        from vllm_mlx.prefix_cache import BlockAwarePrefixCache

        paged_manager = PagedCacheManager(block_size=64, max_blocks=100)
        cache = BlockAwarePrefixCache(model=None, paged_cache_manager=paged_manager)

        tokens = list(range(128))
        cache.store_cache("req-1", tokens, ["data"])
        cache.store_cache("req-2", tokens, ["data2"])

        assert len(cache) == 2

        cache.clear()

        assert len(cache) == 0
        stats = cache.get_stats()
        # After clear, null block is still allocated (vLLM style)
        assert stats["allocated_blocks"] == 1  # only null block

    def test_slices_3d_kv_along_seq_axis(self):
        """3D KV state (Qwen3.5-style ``(n_kv_heads, seq, head_dim)``) must
        be sliced along seq (axis 1), not axis 2 (which is ``head_dim``).
        Regression for upstream waybarrios#286 — pre-fix, the 4D-hardcoded
        slice ``keys[:, :, start:end, :]`` crashed on 3D state and dropped
        the whole entry, masking the issue. ``reconstruct_cache`` still
        refuses 3D state because mlx_lm's ``KVCache`` accessors hard-code
        ``shape[2]`` for seq; hosting 3D state there would silently corrupt
        downstream generation."""
        import mlx.core as mx
        from mlx_lm.models.cache import KVCache

        from vllm_mlx.paged_cache import PagedCacheManager
        from vllm_mlx.prefix_cache import BlockAwarePrefixCache

        paged_manager = PagedCacheManager(block_size=4, max_blocks=10)
        cache = BlockAwarePrefixCache(model=None, paged_cache_manager=paged_manager)

        kv_keys = mx.arange(2 * 8 * 3).reshape(2, 8, 3)
        kv_values = mx.arange(1000, 1000 + (2 * 8 * 3)).reshape(2, 8, 3)
        layer_state = {
            "state": (kv_keys, kv_values),
            "meta_state": "",
            "class_ref": KVCache,
            "class_name": "KVCache",
        }

        sliced = cache._extract_block_tensor_slice([layer_state], 0, 4)
        assert sliced is not None and len(sliced) == 1
        sliced_keys, sliced_values = sliced[0]
        assert sliced_keys.tolist() == kv_keys[:, :4, :].tolist()
        assert sliced_values.tolist() == kv_values[:, :4, :].tolist()
        assert cache._cache_state_seq_axis((kv_keys, kv_values)) == 1
        four_d = mx.zeros((1, 2, 8, 3))
        assert cache._cache_state_seq_axis((four_d, four_d)) == 2
        assert cache._cache_state_seq_axis((four_d,)) is None
        assert cache._cache_state_seq_axis((four_d, mx.zeros((2, 8)))) is None
