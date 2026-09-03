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


def _kv_layer_states(num_tokens, layers=1, heads=2, head_dim=4, base=0.0):
    """Build extracted layer-state dicts backed by real mlx-lm KVCache state.

    ``store_cache`` only accepts the extracted-tensor-state layout (list of
    dicts with 'state' + 'class_name' where every layer is a plain 4D
    KVCache); tests that exercise the supported path build inputs here.
    """
    import mlx.core as mx
    from mlx_lm.models.cache import KVCache

    states = []
    for layer in range(layers):
        cache = KVCache()
        n = num_tokens * heads * head_dim
        keys = (
            mx.arange(n, dtype=mx.float32).reshape(1, heads, num_tokens, head_dim)
            + base
            + layer
        )
        values = keys + 0.5
        cache.update_and_fetch(keys, values)
        states.append(
            {
                "state": cache.state,
                "meta_state": cache.meta_state,
                "class_name": "KVCache",
                "class_ref": KVCache,
            }
        )
    return states


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
            manager.allocate_blocks_for_tokens(300)
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
        cache_data1 = _kv_layer_states(128)
        block_table = cache.store_cache("req-1", tokens1, cache_data1)

        assert block_table is not None
        assert block_table.num_tokens == 128
        assert len(block_table.block_ids) == 2
        # Every stored block must carry reconstructable tensor data.
        for bid in block_table.block_ids:
            assert paged_manager.allocated_blocks[bid].cache_data is not None

        # Fetch cache for second request with same prefix
        block_table2, remaining = cache.fetch_cache("req-2", tokens1 + [999, 1000])

        # Should hit the prefix
        assert block_table2 is not None
        assert remaining == [999, 1000]

    def test_release_cache(self):
        """Test releasing cache."""
        from vllm_mlx.paged_cache import PagedCacheManager
        from vllm_mlx.prefix_cache import BlockAwarePrefixCache

        paged_manager = PagedCacheManager(block_size=64, max_blocks=100)
        cache = BlockAwarePrefixCache(model=None, paged_cache_manager=paged_manager)

        tokens = list(range(64))
        cache.store_cache("req-1", tokens, _kv_layer_states(64))

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
        cache.store_cache("req-1", tokens, _kv_layer_states(128))

        # Fork to new request
        forked_table = cache.fork_cache("req-1", "req-2")

        assert forked_table is not None
        assert len(cache) == 2

        # Both should share the same blocks
        stats = cache.get_stats()
        assert stats["shared_blocks"] > 0

        # Blocks are the source of truth: the forked table reconstructs the
        # same KV state the source stored.
        reconstructed = cache.reconstruct_cache(forked_table)
        assert reconstructed is not None
        assert reconstructed[0].offset == 128

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
        cache.store_cache("req-1", tokens, _kv_layer_states(128))
        cache.store_cache("req-2", tokens, _kv_layer_states(128, base=100.0))

        assert len(cache) == 2

        cache.clear()

        assert len(cache) == 0
        stats = cache.get_stats()
        # After clear, null block is still allocated (vLLM style)
        assert stats["allocated_blocks"] == 1  # only null block

    def test_rejects_3d_kv_state_everywhere(self):
        """3D KV state (``(n_kv_heads, seq, head_dim)``) is refused at
        store AND reconstruct. Historically (upstream waybarrios#286) the
        store side sliced 3D state along axis 1 while ``reconstruct_cache``
        refused it — mlx_lm's ``KVCache`` accessors hard-code ``shape[2]``
        for seq — so 3D blocks were stored but never reusable: dead weight
        that read as a healthy cache with zero reuse (#2955). Store and
        reconstruct must agree, so both now fail closed on anything but
        4D."""
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

        # 3D state is not blockizable: no slice, no store, no blocks.
        assert cache._extract_block_tensor_slice([layer_state], 0, 4) is None
        assert cache.store_cache("req-3d", list(range(8)), [layer_state]) is None
        assert paged_manager.stats.allocated_blocks == 1  # null block only
        assert cache._cache_state_seq_axis((kv_keys, kv_values)) is None

        four_d = mx.zeros((1, 2, 8, 3))
        assert cache._cache_state_seq_axis((four_d, four_d)) == 2
        assert cache._cache_state_seq_axis((four_d,)) is None
        assert cache._cache_state_seq_axis((four_d, mx.zeros((2, 8)))) is None
        # Partial-None state must reject up-front rather than relying on
        # downstream ``.shape`` access to crash (caught by outer try/except,
        # but fragile). Regression for codex round-2 finding on PR #392.
        assert cache._cache_state_seq_axis((four_d, None)) is None
        assert cache._cache_state_seq_axis((None, four_d)) is None
        assert cache._cache_state_seq_axis((None, None)) is None
        # Class-name gate: a Mamba/DeltaNet ``ArraysCache`` may happen to
        # hold two same-shape tensors, but its tensors are NOT seq-
        # indexed. The gate must reject any class outside the allowlist.
        # Regression for codex round-3 finding on PR #392.
        assert (
            cache._cache_state_seq_axis((four_d, four_d), class_name="ArraysCache")
            is None
        )
        assert (
            cache._cache_state_seq_axis((four_d, four_d), class_name="RotatingKVCache")
            is None
        )
        assert cache._cache_state_seq_axis((four_d, four_d), class_name="KVCache") == 2
        # When class_name is omitted, fall back to the shape-only check.
        assert cache._cache_state_seq_axis((four_d, four_d)) == 2
        # Extract path itself rejects layers whose class_name is not KVCache,
        # even when their tensors look slicable.
        non_kv_layer = {
            "state": (four_d, four_d),
            "meta_state": "",
            "class_ref": None,
            "class_name": "ArraysCache",
        }
        assert cache._extract_block_tensor_slice([non_kv_layer], 0, 4) is None

    def test_seq_axis_allowlist_anchored_to_real_mlx_lm_classes(self):
        """Anchor ``_SEQ_AXIS_KV_CLASSES`` to GROUND TRUTH — the real mlx-lm
        class *objects*, not synthetic string literals.

        The block-aware path hosts a stored layer only when its ``class_name``
        is in this fail-closed allowlist. The sibling tests assert against
        hardcoded strings ("KVCache", "RotatingKVCache", ...), so they can't
        notice if the allowlist drifts out of sync with the actual mlx-lm
        classes. This test protects two concrete regressions by keying off the
        real ``__name__``s:
          * allowlist drift — anyone adding ``RotatingKVCache`` /
            ``ChunkedKVCache`` / ``QuantizedKVCache`` / ``ArraysCache`` to the
            set (all trim-unsafe or non-seq-indexed) turns this red;
          * an mlx-lm rename of the one hostable class — if ``KVCache`` is
            renamed, ``KVCache.__name__`` no longer equals the allowlisted
            string and this turns red.
        It does NOT (and cannot) detect a hypothetical *future distinct* cache
        class that also happens to be named ``KVCache`` — a same-name collision
        is out of reach of a name-based allowlist; guarding that would need
        per-family ``make_cache`` construction coverage (weights/network), which
        is intentionally out of scope here.
        """
        from mlx_lm.models.cache import (
            ArraysCache,
            ChunkedKVCache,
            KVCache,
            QuantizedKVCache,
            RotatingKVCache,
        )

        from vllm_mlx.prefix_cache import BlockAwarePrefixCache

        allow = BlockAwarePrefixCache._SEQ_AXIS_KV_CLASSES

        # The one and only block-hostable class, keyed by its REAL __name__.
        assert KVCache.__name__ in allow

        # Every trim-unsafe / non-seq-indexed real class stays OUT.
        for cls in (
            RotatingKVCache,
            ChunkedKVCache,
            QuantizedKVCache,
            ArraysCache,
        ):
            assert cls.__name__ not in allow, (
                f"{cls.__name__} must not be block-hostable"
            )

    def test_reconstruct_refuses_non_kvcache_class_name(self):
        """``reconstruct_cache`` must refuse to host anything other than a
        vanilla ``KVCache`` even if ``block.cache_data`` somehow contains
        4D tensors. Defense in depth against a future writer that bypasses
        ``_extract_block_tensor_slice``. Regression for codex pr_validate
        finding on PR #392."""
        import mlx.core as mx

        from vllm_mlx.paged_cache import BlockTable, PagedCacheManager
        from vllm_mlx.prefix_cache import BlockAwarePrefixCache

        paged_manager = PagedCacheManager(block_size=4, max_blocks=10)
        cache = BlockAwarePrefixCache(model=None, paged_cache_manager=paged_manager)

        # Manually plant a block as if some non-KV writer had populated it.
        block = paged_manager.allocate_block()
        four_d = mx.zeros((1, 2, 4, 8))
        block.cache_data = [(four_d, four_d)]
        block.cache_class_name = "RotatingKVCache"

        table = BlockTable(request_id="req", block_ids=[block.block_id], num_tokens=4)
        assert cache.reconstruct_cache(table) is None

        # And the happy path still works once class_name is correct.
        block.cache_class_name = "KVCache"
        out = cache.reconstruct_cache(table)
        assert out is not None and len(out) == 1

    def test_extract_rejects_layer_without_class_name(self):
        """If a layer dict lacks an explicit ``class_name``, ``_extract``
        must refuse to slice. Otherwise the block would be stored with
        ``cache_class_name=None`` and silently rejected at reconstruct,
        wasting a paged-cache slot. Regression for codex pr_validate
        round-2 finding on PR #392."""
        import mlx.core as mx

        from vllm_mlx.paged_cache import PagedCacheManager
        from vllm_mlx.prefix_cache import BlockAwarePrefixCache

        paged_manager = PagedCacheManager(block_size=4, max_blocks=10)
        cache = BlockAwarePrefixCache(model=None, paged_cache_manager=paged_manager)
        four_d = mx.zeros((1, 2, 8, 3))

        # Missing class_name key entirely.
        no_class_layer = {"state": (four_d, four_d), "meta_state": ""}
        assert cache._extract_block_tensor_slice([no_class_layer], 0, 4) is None
        # Explicit None.
        explicit_none = dict(no_class_layer, class_name=None)
        assert cache._extract_block_tensor_slice([explicit_none], 0, 4) is None
        # Any non-allowlisted class.
        for forbidden in ("ArraysCache", "RotatingKVCache", "QuantizedKVCache"):
            layer = dict(no_class_layer, class_name=forbidden)
            assert cache._extract_block_tensor_slice([layer], 0, 4) is None

    def test_cow_copy_propagates_cache_class_name(self):
        """``PagedCacheManager.copy_block`` must propagate
        ``cache_class_name`` so the COW destination satisfies the
        (cache_data, cache_class_name) invariant. Regression for codex
        pr_validate round-2 finding on PR #392."""
        import mlx.core as mx

        from vllm_mlx.paged_cache import PagedCacheManager

        manager = PagedCacheManager(block_size=4, max_blocks=10)
        src = manager.allocate_block()
        four_d = mx.zeros((1, 2, 4, 8))
        src.cache_data = [(four_d, four_d)]
        src.cache_class_name = "KVCache"
        # Bump the source ref so copy_block treats it as shared (its COW
        # contract decrements the source from a shared state).
        manager.increment_ref(src.block_id)
        dst = manager._cow_copy_block(src)
        assert dst is not None
        assert dst.cache_data == src.cache_data
        assert dst.cache_class_name == "KVCache"


# =============================================================================
# #2955 — capability validation, transactional fetch, snapshot-free stores.
# Folded into this (CI-enrolled) module so the Apple lane runs them.
#
# Two invariants of ``--use-paged-cache``:
#
# 1. Capability is structural and fails closed pre-ready: the loaded model's
#    prompt-cache factory must produce a layout the block serializer can
#    losslessly slice and reconstruct (plain full-attention ``KVCache`` on
#    every layer). Rotating/sliding, Arrays/hybrid, recurrent, quantized and
#    unknown layouts abort startup with a typed, actionable error — never a
#    healthy server with silent zero reuse. No model/architecture names are
#    consulted.
#
# 2. Cache hit acquisition is a transaction: candidate lookup tentatively
#    holds block refs, reconstruction validates the result, and only then do
#    hit/tokens-saved counters commit. Any failure aborts — refs and table
#    state released, exactly one miss counted — and release is idempotent
#    after both commit and abort.
# =============================================================================


def _make_cache(block_size=64, max_blocks=100):
    from vllm_mlx.paged_cache import PagedCacheManager
    from vllm_mlx.prefix_cache import BlockAwarePrefixCache

    manager = PagedCacheManager(block_size=block_size, max_blocks=max_blocks)
    return BlockAwarePrefixCache(model=None, paged_cache_manager=manager), manager


class _FactoryModel:
    """Model stub whose prompt-cache factory returns the given layers."""

    def __init__(self, factory):
        self._factory = factory

    def make_cache(self):
        return self._factory()


class TestStructuralCapabilityMatrix:
    """Layout family matrix for ``validate_paged_cache_capability``."""

    def test_plain_kv_accepted(self):
        from mlx_lm.models.cache import KVCache

        from vllm_mlx.prefix_cache import validate_paged_cache_capability

        validate_paged_cache_capability(_FactoryModel(lambda: [KVCache(), KVCache()]))

    def test_default_factory_plain_kv_accepted(self):
        """A model without ``make_cache`` gets mlx-lm's default plain-KV
        factory — accepted."""
        from vllm_mlx.prefix_cache import validate_paged_cache_capability

        class NoFactory:
            layers = [object(), object()]

        validate_paged_cache_capability(NoFactory())

    def test_sliding_capable_arch_with_inactive_sliding_accepted(self):
        """The gate is structural: an architecture that CAN slide but whose
        factory returns plain KV (sliding disabled in its config) is
        accepted — no architecture-name allowlist involved."""
        from mlx_lm.models.cache import KVCache

        from vllm_mlx.prefix_cache import validate_paged_cache_capability

        class SlidingCapable(_FactoryModel):
            sliding_window = None  # inactive

        validate_paged_cache_capability(SlidingCapable(lambda: [KVCache(), KVCache()]))

    @pytest.mark.parametrize(
        "layer_factory, expected",
        [
            # Rotating/sliding-window layout.
            (
                lambda cachemod: [
                    cachemod.KVCache(),
                    cachemod.RotatingKVCache(max_size=512),
                ],
                "RotatingKVCache",
            ),
            # Recurrent/Mamba-style hybrid (ArraysCache holds conv/recurrent
            # state; not seq-indexed).
            (
                lambda cachemod: [cachemod.KVCache(), cachemod.ArraysCache(2)],
                "ArraysCache",
            ),
            # Quantized tuple layout.
            (
                lambda cachemod: [cachemod.QuantizedKVCache()],
                "QuantizedKVCache",
            ),
        ],
    )
    def test_unsupported_families_rejected(self, layer_factory, expected):
        from mlx_lm.models import cache as cachemod

        from vllm_mlx.errors import PagedCacheUnsupportedLayoutError
        from vllm_mlx.prefix_cache import validate_paged_cache_capability

        with pytest.raises(PagedCacheUnsupportedLayoutError) as excinfo:
            validate_paged_cache_capability(
                _FactoryModel(lambda: layer_factory(cachemod))
            )
        assert expected in excinfo.value.incompatible_layers
        # Actionable: names the flag and the way out.
        assert "--use-paged-cache" in str(excinfo.value)
        assert "Remove --use-paged-cache" in str(excinfo.value)

    def test_unknown_layout_rejected(self):
        """Fail closed on classes the serializer has never seen — including
        a same-named class that is not mlx-lm's ``KVCache``."""
        from vllm_mlx.errors import PagedCacheUnsupportedLayoutError
        from vllm_mlx.prefix_cache import validate_paged_cache_capability

        class KVCache:  # same name, unknown type — must NOT pass
            pass

        with pytest.raises(PagedCacheUnsupportedLayoutError) as excinfo:
            validate_paged_cache_capability(_FactoryModel(lambda: [KVCache()]))
        assert "KVCache" in excinfo.value.incompatible_layers

    def test_unverifiable_probe_rejected(self):
        from vllm_mlx.errors import PagedCacheUnsupportedLayoutError
        from vllm_mlx.prefix_cache import validate_paged_cache_capability

        def boom():
            raise RuntimeError("no cache for you")

        with pytest.raises(PagedCacheUnsupportedLayoutError):
            validate_paged_cache_capability(_FactoryModel(boom))
        # Non-list factory output is just as unverifiable.
        with pytest.raises(PagedCacheUnsupportedLayoutError):
            validate_paged_cache_capability(_FactoryModel(lambda: object()))

    def test_kv_transform_request_rejected(self):
        """An explicit KV-cache transform request (quantization or
        TurboQuant) is rejected even on a plain-KV factory: the paged store
        implements neither, so it must not silently serve plain blocks."""
        from mlx_lm.models.cache import KVCache

        from vllm_mlx.errors import PagedCacheUnsupportedLayoutError
        from vllm_mlx.prefix_cache import validate_paged_cache_capability

        with pytest.raises(PagedCacheUnsupportedLayoutError) as excinfo:
            validate_paged_cache_capability(
                _FactoryModel(lambda: [KVCache()]),
                kv_cache_transform_requested=True,
            )
        assert "--use-paged-cache" in str(excinfo.value)
        assert "TurboQuant" in str(excinfo.value)


class TestSchedulerStartupGate:
    """Explicit --use-paged-cache on an unsupported layout must fail at
    scheduler construction — before readiness or any request service."""

    def _config(self):
        from vllm_mlx.scheduler import SchedulerConfig

        return SchedulerConfig(
            max_num_seqs=4,
            enable_prefix_cache=True,
            use_memory_aware_cache=False,
            use_paged_cache=True,
            paged_cache_block_size=16,
            max_cache_blocks=32,
        )

    def test_rotating_layout_fails_before_ready(self):
        from unittest.mock import MagicMock

        from mlx_lm.models.cache import KVCache, RotatingKVCache

        from vllm_mlx.errors import PagedCacheUnsupportedLayoutError
        from vllm_mlx.scheduler import Scheduler

        model = MagicMock()
        model.make_cache = lambda: [KVCache(), RotatingKVCache(max_size=512)]
        tokenizer = MagicMock()
        tokenizer.encode = lambda s: list(range(len(s)))

        with pytest.raises(PagedCacheUnsupportedLayoutError) as excinfo:
            Scheduler(model=model, tokenizer=tokenizer, config=self._config())
        assert "RotatingKVCache" in str(excinfo.value)
        assert "--use-paged-cache" in str(excinfo.value)

    def test_plain_layout_boots_with_paged_cache(self):
        from unittest.mock import MagicMock

        from mlx_lm.models.cache import KVCache

        from vllm_mlx.scheduler import Scheduler

        model = MagicMock()
        model.make_cache = lambda: [KVCache(), KVCache()]
        tokenizer = MagicMock()
        tokenizer.encode = lambda s: list(range(len(s)))

        sched = Scheduler(model=model, tokenizer=tokenizer, config=self._config())
        assert sched.block_aware_cache is not None

    @pytest.mark.parametrize(
        "config_overrides",
        [
            # Ordinary live KV quantization toggles — rejected even though
            # the runtime head-dim probe might later fall back and disable
            # the transform: an explicit flag pair that cannot be honored
            # must fail closed, not silently serve plain BF16 blocks.
            {"kv_cache_quantization": True, "kv_cache_quantization_bits": 4},
            {"kv_cache_quantization": True, "kv_cache_quantization_bits": 8},
            # TurboQuant request — equally unimplemented by the paged store.
            {"kv_cache_turboquant": True},
        ],
    )
    def test_explicit_kv_transform_plus_paged_fails_before_ready(
        self, config_overrides
    ):
        """Real Scheduler configuration: --use-paged-cache combined with an
        explicit KV quantization/TurboQuant request fails at construction
        even on a plain full-attention KV layout."""
        from unittest.mock import MagicMock

        from mlx_lm.models.cache import KVCache

        from vllm_mlx.errors import PagedCacheUnsupportedLayoutError
        from vllm_mlx.scheduler import Scheduler

        config = self._config()
        for key, value in config_overrides.items():
            setattr(config, key, value)

        model = MagicMock()
        model.make_cache = lambda: [KVCache(), KVCache()]
        tokenizer = MagicMock()
        tokenizer.encode = lambda s: list(range(len(s)))

        with pytest.raises(PagedCacheUnsupportedLayoutError) as excinfo:
            Scheduler(model=model, tokenizer=tokenizer, config=config)
        message = str(excinfo.value)
        assert "--use-paged-cache" in message
        assert "TurboQuant" in message

    @pytest.mark.parametrize(
        "layout, config_overrides",
        [
            # Plain supported layout: the contradiction alone must reject.
            ("plain", {}),
            # Unsupported layout: still the same fail-closed rejection —
            # never a silent no-cache boot.
            ("rotating", {}),
            # Explicit transforms on top of the contradiction: rejected.
            ("plain", {"kv_cache_quantization": True, "kv_cache_quantization_bits": 4}),
            ("plain", {"kv_cache_turboquant": True}),
        ],
    )
    def test_paged_without_prefix_cache_rejected(self, layout, config_overrides):
        """use_paged_cache=True with enable_prefix_cache=False is a
        contradictory config: pre-fix the capability gate was nested under
        the enablement branch, so the explicit paged request was silently
        ignored. It must fail closed at construction with the typed
        actionable error, regardless of layout or transform flags."""
        from unittest.mock import MagicMock

        from mlx_lm.models.cache import KVCache, RotatingKVCache

        from vllm_mlx.errors import PagedCacheUnsupportedLayoutError
        from vllm_mlx.scheduler import Scheduler

        config = self._config()
        config.enable_prefix_cache = False
        for key, value in config_overrides.items():
            setattr(config, key, value)

        model = MagicMock()
        if layout == "plain":
            model.make_cache = lambda: [KVCache(), KVCache()]
        else:
            model.make_cache = lambda: [KVCache(), RotatingKVCache(max_size=512)]
        tokenizer = MagicMock()
        tokenizer.encode = lambda s: list(range(len(s)))

        with pytest.raises(PagedCacheUnsupportedLayoutError) as excinfo:
            Scheduler(model=model, tokenizer=tokenizer, config=config)
        message = str(excinfo.value)
        assert "--use-paged-cache" in message
        assert "prefix cache" in message

    def test_prefix_cache_disabled_without_paged_boots(self):
        """Negative control: disabling the prefix cache WITHOUT an explicit
        paged request is a valid configuration and boots with no caches."""
        from unittest.mock import MagicMock

        from vllm_mlx.scheduler import Scheduler

        config = self._config()
        config.enable_prefix_cache = False
        config.use_paged_cache = False

        model = MagicMock()
        tokenizer = MagicMock()
        tokenizer.encode = lambda s: list(range(len(s)))

        sched = Scheduler(model=model, tokenizer=tokenizer, config=config)
        assert sched.block_aware_cache is None
        assert sched.paged_cache_manager is None
        assert sched.prefix_cache is None
        assert sched.memory_aware_cache is None


class TestStoreMaterializationFailure:
    """A store whose block materialization fails must report failure (None),
    never the fetch-held table as success — and the scheduler's cleanup
    branch must then release the fetch-held refs/table/stash."""

    def test_extension_store_failure_returns_none_and_release_restores(self):
        from unittest.mock import patch

        cache, manager = _make_cache(block_size=4)
        tokens4 = list(range(4))
        tokens8 = list(range(8))

        seed_table = cache.store_cache("seed", tokens4, _kv_layer_states(4))
        assert seed_table is not None and len(seed_table.block_ids) == 1
        bid = seed_table.block_ids[0]

        # Successful fetch: one shared ref held by req-x's table.
        table, remaining = cache.fetch_cache("req-x", tokens8)
        assert table is not None
        assert remaining == tokens8[4:]
        assert manager.allocated_blocks[bid].ref_count == 2

        allocated_before = manager.stats.allocated_blocks
        with patch("mlx.core.eval", side_effect=RuntimeError("injected")):
            stored = cache.store_cache("req-x", tokens8, _kv_layer_states(8))

        # Failure is reported as failure — not the fetch-held table.
        assert stored is None
        # Materialization fails before any table/block mutation: nothing
        # new allocated, no entry claims ownership of the fetched blocks.
        assert manager.stats.allocated_blocks == allocated_before
        assert "req-x" not in cache._request_tables

        # The scheduler acts on None by releasing; mirror that here and
        # prove it restores every counter and leaves no ownerless state.
        cache.release_cache("req-x")
        assert manager.get_block_table("req-x") is None
        assert "req-x" not in cache._pending_reconstructed
        assert manager.allocated_blocks[bid].ref_count == 1
        assert manager.allocated_blocks[bid].cache_data is not None

        # The seed entry is unharmed and still serves hits.
        table2, remaining2 = cache.fetch_cache("req-y", tokens4)
        assert table2 is not None and remaining2 == []

    def test_scheduler_cleanup_releases_after_store_failure(self):
        """Production path: ``_cleanup_finished`` sees ``stored_table is
        None`` from the failed store and releases the fetch-held
        table/refs/stash — counters stay coherent, no ownerless table."""
        from unittest.mock import MagicMock, patch

        import mlx.core as mx
        from mlx_lm.models.cache import KVCache

        from vllm_mlx.scheduler import Scheduler, SchedulerConfig

        config = SchedulerConfig(
            max_num_seqs=4,
            enable_prefix_cache=True,
            use_memory_aware_cache=False,
            use_paged_cache=True,
            paged_cache_block_size=4,
            max_cache_blocks=32,
        )
        model = MagicMock()
        model.make_cache = lambda: [KVCache(), KVCache()]
        tokenizer = MagicMock()
        tokenizer.encode = lambda s: list(range(len(s)))
        sched = Scheduler(model=model, tokenizer=tokenizer, config=config)
        cache = sched.block_aware_cache
        assert cache is not None
        manager = cache.paged_cache

        tokens4 = list(range(4))
        tokens8 = list(range(8))
        assert cache.store_cache("seed", tokens4, _kv_layer_states(4)) is not None
        bid = cache._request_tables["seed"].block_table.block_ids[0]
        allocated_after_seed = manager.stats.allocated_blocks

        table, _ = cache.fetch_cache("req-x", tokens8)
        assert table is not None
        assert manager.allocated_blocks[bid].ref_count == 2

        request = MagicMock()
        request.prompt_token_ids = tokens8
        request.output_token_ids = []
        request.pflash_metadata = None
        request._extracted_cache = _kv_layer_states(8)
        sched.running["req-x"] = request

        # Spy on the store's return value; fail ONLY its materialization
        # eval (first mx.eval inside _cleanup_finished), then pass through
        # so the later incremental per-layer eval loop runs for real.
        store_returns = []
        real_store = cache.store_cache

        def spying_store(*args, **kwargs):
            result = real_store(*args, **kwargs)
            store_returns.append(result)
            return result

        real_eval = mx.eval
        eval_calls = {"n": 0}

        def flaky_eval(*args, **kwargs):
            eval_calls["n"] += 1
            if eval_calls["n"] == 1:
                raise RuntimeError("injected materialization failure")
            return real_eval(*args, **kwargs)

        with (
            patch.object(cache, "store_cache", side_effect=spying_store),
            patch("mlx.core.eval", side_effect=flaky_eval),
        ):
            sched._cleanup_finished({"req-x"})

        assert eval_calls["n"] >= 1  # the injection actually fired
        assert store_returns == [None]  # store reported failure

        # Production cleanup released everything the fetch held: no
        # ownerless table, no stash, no entry — and the shared ref is back
        # to the seed entry's single ref with its KV intact.
        assert manager.get_block_table("req-x") is None
        assert "req-x" not in cache._request_tables
        assert "req-x" not in cache._pending_reconstructed
        assert manager.allocated_blocks[bid].ref_count == 1
        assert manager.allocated_blocks[bid].cache_data is not None
        assert "seed" in cache._request_tables
        # Counters coherent: no allocation survived beyond the seed store.
        assert manager.stats.allocated_blocks == allocated_after_seed
        assert "req-x" not in sched.running
        assert "req-x" in sched.finished_req_ids


class TestTransactionalFetch:
    """Fetch commits counters only after successful reconstruction; every
    failure aborts cleanly."""

    def _seed(self, cache, tokens):
        table = cache.store_cache("seed", tokens, _kv_layer_states(len(tokens)))
        assert table is not None and len(table.block_ids) == 2
        return table

    def _assert_aborted(self, cache, manager, seed_table, request_id):
        assert cache._hits == 0
        assert cache._tokens_saved == 0
        assert cache._misses == 1
        # Tentative refs released: back to the store entry's single ref.
        for bid in seed_table.block_ids:
            assert manager.allocated_blocks[bid].ref_count == 1
        # No table or stash retained for the failed request.
        assert manager.get_block_table(request_id) is None
        assert request_id not in cache._request_tables
        assert request_id not in cache._pending_reconstructed

    def test_commit_on_success(self):
        cache, manager = _make_cache()
        tokens = list(range(128))
        seed = self._seed(cache, tokens)

        table, remaining = cache.fetch_cache("req-a", tokens + [7, 8])
        assert table is not None
        assert remaining == [7, 8]
        assert cache._hits == 1
        assert cache._misses == 0
        assert cache._tokens_saved == 128
        for bid in seed.block_ids:
            assert manager.allocated_blocks[bid].ref_count == 2

        reconstructed = cache.reconstruct_cache(table)
        assert reconstructed is not None and len(reconstructed) == 1
        assert reconstructed[0].offset == 128

    def test_missing_tensor_data_aborts(self):
        cache, manager = _make_cache()
        seed = self._seed(cache, list(range(128)))
        manager.allocated_blocks[seed.block_ids[1]].cache_data = None

        table, remaining = cache.fetch_cache("req-b", list(range(128)))
        assert table is None and remaining == list(range(128))
        self._assert_aborted(cache, manager, seed, "req-b")

    def test_wrong_cache_class_aborts(self):
        cache, manager = _make_cache()
        seed = self._seed(cache, list(range(128)))
        manager.allocated_blocks[seed.block_ids[0]].cache_class_name = "RotatingKVCache"

        table, _ = cache.fetch_cache("req-c", list(range(128)))
        assert table is None
        self._assert_aborted(cache, manager, seed, "req-c")

    def test_wrong_shape_aborts(self):
        import mlx.core as mx

        cache, manager = _make_cache()
        seed = self._seed(cache, list(range(128)))
        bad = mx.zeros((2, 64, 4))  # 3D — not hostable by mlx-lm KVCache
        manager.allocated_blocks[seed.block_ids[0]].cache_data = [(bad, bad)]

        table, _ = cache.fetch_cache("req-d", list(range(128)))
        assert table is None
        self._assert_aborted(cache, manager, seed, "req-d")

    def test_reconstruction_exception_aborts(self):
        import mlx.core as mx

        cache, manager = _make_cache()
        seed = self._seed(cache, list(range(128)))
        # Head-count mismatch between blocks: mx.concatenate raises.
        bad = mx.zeros((1, 3, 64, 4))
        manager.allocated_blocks[seed.block_ids[1]].cache_data = [(bad, bad)]

        table, _ = cache.fetch_cache("req-e", list(range(128)))
        assert table is None
        self._assert_aborted(cache, manager, seed, "req-e")

    def test_release_idempotent_after_commit(self):
        cache, manager = _make_cache()
        seed = self._seed(cache, list(range(128)))
        table, _ = cache.fetch_cache("req-f", list(range(128)))
        assert table is not None
        for _ in range(3):
            cache.release_cache("req-f")
            for bid in seed.block_ids:
                assert manager.allocated_blocks[bid].ref_count == 1
        assert manager.get_block_table("req-f") is None

    def test_release_idempotent_after_abort(self):
        cache, manager = _make_cache()
        seed = self._seed(cache, list(range(128)))
        manager.allocated_blocks[seed.block_ids[1]].cache_data = None
        table, _ = cache.fetch_cache("req-g", list(range(128)))
        assert table is None
        for _ in range(3):
            cache.release_cache("req-g")
            for bid in seed.block_ids:
                assert manager.allocated_blocks[bid].ref_count == 1

    def test_repeated_prefix_reuse_returns_identical_kv(self):
        """Supported full-attention store/fetch/reconstruct over a repeated
        prefix keeps working and returns the exact stored KV state."""
        import mlx.core as mx

        cache, _ = _make_cache()
        tokens = list(range(128))
        states = _kv_layer_states(128)
        cache.store_cache("seed", tokens, states)

        for i, request_id in enumerate(("req-h", "req-i")):
            table, remaining = cache.fetch_cache(request_id, tokens + [500 + i])
            assert table is not None
            assert remaining == [500 + i]
            reconstructed = cache.reconstruct_cache(table)
            assert reconstructed is not None
            assert reconstructed[0].offset == 128
            orig_keys, orig_values = states[0]["state"]
            assert mx.array_equal(reconstructed[0].keys, orig_keys).item()
            assert mx.array_equal(reconstructed[0].values, orig_values).item()
            cache.release_cache(request_id)

        assert cache._hits == 2
        assert cache._tokens_saved == 256


class TestStoreRetention:
    """Blocks are the source of truth; stores never retain snapshots."""

    def test_unsupported_store_retains_nothing(self):
        """A rotating-layout store must allocate zero blocks and retain
        zero snapshots — the exact baseline failure of #2955 (2 stored
        blocks, 0 with tensor data, full snapshot retained)."""
        import mlx.core as mx
        from mlx_lm.models.cache import RotatingKVCache

        cache, manager = _make_cache(block_size=64, max_blocks=16)
        rot = RotatingKVCache(max_size=256)
        rot.update_and_fetch(mx.zeros((1, 2, 128, 8)), mx.zeros((1, 2, 128, 8)))
        cache_data = [
            {
                "state": rot.state,
                "meta_state": rot.meta_state,
                "class_name": "RotatingKVCache",
                "class_ref": RotatingKVCache,
            }
        ]

        assert cache.store_cache("req-rot", list(range(128)), cache_data) is None
        assert manager.stats.allocated_blocks == 1  # null block only
        assert len(cache._request_tables) == 0
        assert len(cache._prefix_index) == 0
        assert manager.get_block_table("req-rot") is None

    def test_supported_store_does_not_retain_original_by_identity(self):
        """The stored entry and blocks must not hold the caller's cache
        objects: no snapshot on the entry, and block tensors are
        materialized copies, not the caller's arrays."""
        cache, manager = _make_cache()
        tokens = list(range(128))
        states = _kv_layer_states(128)
        table = cache.store_cache("req-1", tokens, states)
        assert table is not None

        entry = cache._request_tables["req-1"]
        assert not hasattr(entry, "cache_data")

        original_tensors = {id(t) for layer in states for t in layer["state"]}
        original_tensors.add(id(states))
        for bid in table.block_ids:
            block = manager.allocated_blocks[bid]
            assert block.cache_data is not None
            for keys, values in block.cache_data:
                assert id(keys) not in original_tensors
                assert id(values) not in original_tensors


class TestContextualBlockIdentity:
    """Block identity must cover the whole preceding history (#2955).

    KV for block N depends on blocks 0..N-1, so equal token chunks with
    divergent histories must never share storage — while truly identical
    prefixes must still deduplicate.
    """

    def test_divergent_histories_do_not_share_equal_chunks(self):
        import mlx.core as mx

        cache, _ = _make_cache(block_size=4, max_blocks=16)
        a_tokens = [1, 1, 1, 1, 7, 7, 7, 7]
        b_tokens = [2, 2, 2, 2, 7, 7, 7, 7]
        a_states = _kv_layer_states(8, base=10.0)
        b_states = _kv_layer_states(8, base=20.0)
        table_a = cache.store_cache("req-a", a_tokens, a_states)
        table_b = cache.store_cache("req-b", b_tokens, b_states)
        assert table_a is not None and len(table_a.block_ids) == 2
        assert table_b is not None and len(table_b.block_ids) == 2
        # The equal trailing chunk must NOT dedup across divergent
        # histories: its KV was computed under different context.
        assert set(table_a.block_ids).isdisjoint(table_b.block_ids)

        # Each fetch must reconstruct its OWN stored KV, bit for bit.
        for request_id, tokens, states in (
            ("read-a", a_tokens, a_states),
            ("read-b", b_tokens, b_states),
        ):
            table, remaining = cache.fetch_cache(request_id, tokens)
            assert table is not None and remaining == []
            reconstructed = cache.reconstruct_cache(table)
            keys, values = states[0]["state"]
            assert mx.array_equal(reconstructed[0].keys, keys).item()
            assert mx.array_equal(reconstructed[0].values, values).item()
            cache.release_cache(request_id)

    def test_identical_prefixes_still_dedup(self):
        cache, manager = _make_cache(block_size=4, max_blocks=16)
        tokens = [1, 1, 1, 1, 7, 7, 7, 7]
        table_1 = cache.store_cache("req-1", tokens, _kv_layer_states(8, base=1.0))
        table_2 = cache.store_cache("req-2", tokens, _kv_layer_states(8, base=1.0))
        assert table_1.block_ids == table_2.block_ids
        for bid in table_1.block_ids:
            assert manager.allocated_blocks[bid].ref_count == 2

    def test_suffix_chunk_never_matches_as_prefix(self):
        """A stored SECOND block's chunk offered at position 0 must miss:
        under chunk-local hashing it would fetch mid-sequence KV as if it
        were a sequence head."""
        cache, _ = _make_cache(block_size=4, max_blocks=16)
        cache.store_cache("req-a", [1, 1, 1, 1, 7, 7, 7, 7], _kv_layer_states(8))

        table, remaining = cache.fetch_cache("read-c", [7, 7, 7, 7, 9, 9])
        assert table is None
        assert remaining == [7, 7, 7, 7, 9, 9]


class TestEngineStartupBoundary:
    """The typed capability error must surface through the real engine
    startup boundary: ``EngineCore`` constructs the ``Scheduler`` during
    ``__init__`` — before readiness or any request service — and must not
    swallow or downgrade the exception."""

    def test_engine_core_surfaces_unsupported_layout(self):
        from unittest.mock import MagicMock

        from mlx_lm.models.cache import KVCache, RotatingKVCache

        from vllm_mlx.engine_core import EngineConfig, EngineCore
        from vllm_mlx.errors import PagedCacheUnsupportedLayoutError
        from vllm_mlx.scheduler import SchedulerConfig

        model = MagicMock()
        model.make_cache = lambda: [KVCache(), RotatingKVCache(max_size=512)]
        tokenizer = MagicMock()
        tokenizer.encode = lambda s: list(range(len(s)))

        scheduler_config = SchedulerConfig(
            max_num_seqs=4,
            enable_prefix_cache=True,
            use_memory_aware_cache=False,
            use_paged_cache=True,
            paged_cache_block_size=16,
            max_cache_blocks=32,
        )
        with pytest.raises(PagedCacheUnsupportedLayoutError) as excinfo:
            EngineCore(
                model=model,
                tokenizer=tokenizer,
                config=EngineConfig(scheduler_config=scheduler_config),
            )
        assert "RotatingKVCache" in str(excinfo.value)
        assert "--use-paged-cache" in str(excinfo.value)


class TestReusedRequestIdInvariants:
    """Reusing a request id must release prior state instead of leaking
    block references."""

    def test_fetch_with_reused_id_releases_prior_state(self):
        cache, manager = _make_cache(block_size=4, max_blocks=16)
        tokens = list(range(8))
        table = cache.store_cache("req-1", tokens, _kv_layer_states(8))
        old_ids = list(table.block_ids)
        free_before = manager.free_blocks

        result_table, remaining = cache.fetch_cache("req-1", tokens)

        # The stored entry held the only reference: releasing it frees the
        # blocks, so the fetch under the reused id is a clean miss with no
        # leaked table and the slots back in the pool.
        assert result_table is None and remaining == tokens
        assert manager.get_block_table("req-1") is None
        assert "req-1" not in cache._request_tables
        for bid in old_ids:
            assert bid not in manager.allocated_blocks
        assert manager.free_blocks == free_before + len(old_ids)

    def test_create_block_table_releases_overwritten_table(self):
        from vllm_mlx.paged_cache import PagedCacheManager

        manager = PagedCacheManager(block_size=4, max_blocks=8)
        table = manager.create_block_table("req-1")
        block = manager.allocate_block()
        manager.add_block_to_table(table, block, 4)
        assert manager.allocated_blocks[block.block_id].ref_count == 1

        manager.create_block_table("req-1")  # reused id

        # The old table's reference was released, not leaked.
        assert block.block_id not in manager.allocated_blocks
        assert manager.get_block_table("req-1").block_ids == []

    def test_fork_onto_existing_id_does_not_leak(self):
        cache, manager = _make_cache(block_size=4, max_blocks=16)
        table_a = cache.store_cache("req-a", [1, 1, 1, 1], _kv_layer_states(4))
        table_b = cache.store_cache("req-b", [2, 2, 2, 2], _kv_layer_states(4))
        old_b_ids = list(table_b.block_ids)

        forked = cache.fork_cache("req-a", "req-b")

        assert forked is not None
        # req-b's old blocks were released (their only ref was its entry).
        for bid in old_b_ids:
            assert bid not in manager.allocated_blocks
        # req-a's blocks are now shared by its entry and the fork.
        for bid in table_a.block_ids:
            assert manager.allocated_blocks[bid].ref_count == 2

    def test_self_fork_is_noop_and_does_not_leak(self):
        """fork_cache(id, id) must not bump refs and then overwrite the
        only owning entry — release afterwards must free everything."""
        cache, manager = _make_cache(block_size=4, max_blocks=16)
        table = cache.store_cache("same", [1, 2, 3, 4], _kv_layer_states(4))
        assert table is not None
        bid = table.block_ids[0]

        forked = cache.fork_cache("same", "same")

        # No-op: the existing table IS the fork; no reference was added.
        assert forked is table
        assert manager.allocated_blocks[bid].ref_count == 1

        cache.release_cache("same")
        assert bid not in manager.allocated_blocks
        assert manager.get_block_table("same") is None

    def test_paged_self_fork_block_table_is_noop(self):
        """fork_block_table onto the id the table is registered under must
        be a no-op at the paged layer too."""
        from vllm_mlx.paged_cache import PagedCacheManager

        manager = PagedCacheManager(block_size=4, max_blocks=8)
        table = manager.create_block_table("same")
        block = manager.allocate_block()
        manager.add_block_to_table(table, block, 4)

        forked = manager.fork_block_table(table, "same")

        assert forked is table
        assert manager.request_tables["same"] is table
        assert manager.allocated_blocks[block.block_id].ref_count == 1

        manager.delete_block_table("same")
        assert block.block_id not in manager.allocated_blocks


class TestStoredBlockBufferIndependence:
    """Stored blocks must own their KV memory: dropping the caller's
    extracted state must actually free its buffers. Guards against MLX
    slice aliasing — a seq-axis slice with a single KV head is already
    row-contiguous, so a no-copy materialization shortcut would silently
    retain the caller's entire KV buffer for every block."""

    def test_store_releases_caller_buffers_single_kv_head(self):
        import gc

        import mlx.core as mx

        cache, _ = _make_cache(block_size=64, max_blocks=80)
        num_tokens = 4096
        states = _kv_layer_states(num_tokens, heads=1, head_dim=64)
        keys, values = states[0]["state"]
        state_bytes = keys.nbytes + values.nbytes

        table = cache.store_cache("req-mem", list(range(num_tokens)), states)
        assert table is not None
        assert len(table.block_ids) == num_tokens // 64

        del keys, values
        gc.collect()
        before = mx.get_active_memory()
        del states
        gc.collect()
        after = mx.get_active_memory()

        # If any stored block aliased the caller's tensors, their full
        # backing buffers would stay resident past this point.
        assert before - after >= int(0.9 * state_bytes)


# ---------------------------------------------------------------------------
# Paged-cache lifecycle invariants (#2955): cumulative identity, reallocated
# and stale metadata, transactional acquisition/materialization/rollback,
# pin/unpin, and the scheduler's hit / fallback / cancel / pressure paths.
# ---------------------------------------------------------------------------


def _make_paged_scheduler(block_size=4, max_blocks=32):
    """Real ``Scheduler`` with a paged ``BlockAwarePrefixCache`` behind a
    plain full-attention cache factory (passes the #2955 structural gate)."""
    from unittest.mock import MagicMock

    from mlx_lm.models.cache import KVCache

    from vllm_mlx.scheduler import Scheduler, SchedulerConfig

    config = SchedulerConfig(
        max_num_seqs=4,
        enable_prefix_cache=True,
        use_memory_aware_cache=False,
        use_paged_cache=True,
        paged_cache_block_size=block_size,
        max_cache_blocks=max_blocks,
    )
    model = MagicMock()
    model.make_cache = lambda: [KVCache(), KVCache()]
    tokenizer = MagicMock()
    tokenizer.encode = lambda s: list(range(len(s)))
    return Scheduler(model=model, tokenizer=tokenizer, config=config)


def _make_request(request_id, tokens):
    from vllm_mlx.request import Request, SamplingParams

    return Request(
        request_id=request_id,
        prompt=list(tokens),
        sampling_params=SamplingParams(max_tokens=4),
        prompt_token_ids=list(tokens),
    )


def _pressure_tick(sched, max_evict=10):
    """One pressure-eviction call with Metal reported far above the cap."""
    from unittest.mock import patch

    with (
        patch.object(sched, "_resolve_metal_cap_bytes", return_value=100 * 10**9),
        patch.object(sched, "_current_metal_active_bytes", return_value=200 * 10**9),
    ):
        return sched.evict_prefix_cache_under_pressure(max_evict=max_evict)


class TestPrefixHasherCumulativeIdentity:
    """``PrefixHasher`` is the one identity that index keys, block
    registrations and ownership guards all agree on: it hashes the whole
    token history through a block, never the block's own chunk."""

    def test_seeded_constructor_continues_the_same_chain(self):
        from vllm_mlx.paged_cache import PrefixHasher

        chained = PrefixHasher()
        chained.update([1, 2])
        chained.update([3, 4])
        assert PrefixHasher([1, 2, 3, 4]).hexdigest() == chained.hexdigest()

        # store_cache seeds the hasher with the tokens a fetched table
        # already covers and extends it per new block: same chain.
        seeded = PrefixHasher([1, 2])
        seeded.update([3, 4])
        assert seeded.hexdigest() == chained.hexdigest()

        # No seed tokens, however spelled, is the empty chain.
        assert PrefixHasher([]).hexdigest() == PrefixHasher().hexdigest()
        assert PrefixHasher(None).hexdigest() == PrefixHasher().hexdigest()

    def test_identity_depends_on_history_not_on_chunk(self):
        from vllm_mlx.paged_cache import PrefixHasher

        full = PrefixHasher([1, 2, 3, 4]).hexdigest()
        same_chunk_other_history = PrefixHasher([9, 9, 3, 4]).hexdigest()
        chunk_alone = PrefixHasher([3, 4]).hexdigest()
        assert len({full, same_chunk_other_history, chunk_alone}) == 3


class TestReallocatedSlotIdentity:
    """A slot refilled with new KV must never resurrect as a hit for the
    tokens it used to hold, and the legacy hash map is never authoritative
    over the identity a block actually claims."""

    def test_reallocation_retires_legacy_only_registration(self):
        from vllm_mlx.paged_cache import PagedCacheManager

        # One usable slot (block 0 is the reserved null block).
        manager = PagedCacheManager(block_size=4, max_blocks=2)
        block = manager.allocate_block()
        assert block is not None
        old_hash = manager.compute_block_hash([1, 2, 3, 4])
        block.token_count = 4
        block.cache_data = [object()]  # stands in for a resident KV slice
        block.cache_class_name = "KVCache"
        # store_cache registers full blocks exactly like this: identity +
        # legacy map only, no chain ``block_hash``.
        manager.register_block_hash_value(block, old_hash)
        assert block.block_hash is None
        assert manager.find_cached_block_by_hash(old_hash, record_stats=False) is block

        assert manager.free_block(block.block_id) is True
        # The freed slab keeps its registration while parked...
        assert manager.hash_to_block.get(old_hash) == block.block_id

        reused = manager.allocate_block()
        assert reused is block
        # ...and reuse retires it BEFORE the slot is handed to a new owner.
        assert old_hash not in manager.hash_to_block
        assert reused.hash_value is None
        assert reused.cache_data is None
        assert reused.cache_class_name is None
        misses_before = manager.stats.cache_misses
        assert manager.find_cached_block_by_hash(old_hash) is None
        assert manager.stats.cache_misses == misses_before + 1

    def test_lookup_prunes_mapping_superseded_by_a_new_identity(self):
        from vllm_mlx.paged_cache import PagedCacheManager

        manager = PagedCacheManager(block_size=4, max_blocks=4)
        block = manager.allocate_block()
        first = manager.compute_block_hash([1, 2, 3, 4])
        second = manager.compute_block_hash([5, 6, 7, 8])
        manager.register_block_hash_value(block, first)
        manager.register_block_hash_value(block, second)
        assert manager.hash_to_block[first] == block.block_id  # left behind

        # The block no longer claims ``first``: the lookup must not serve
        # the slot for the old tokens — it prunes the mapping, counts a
        # miss, and stays a plain miss afterwards.
        misses_before = manager.stats.cache_misses
        assert manager.find_cached_block_by_hash(first) is None
        assert first not in manager.hash_to_block
        assert manager.stats.cache_misses == misses_before + 1
        assert manager.find_cached_block_by_hash(first) is None
        assert manager.stats.cache_misses == misses_before + 2
        # The identity the block does claim still resolves as a hit.
        hits_before = manager.stats.cache_hits
        assert manager.find_cached_block_by_hash(second) is block
        assert manager.stats.cache_hits == hits_before + 1

    def test_paged_fork_onto_reused_id_releases_old_table(self):
        from vllm_mlx.paged_cache import PagedCacheManager

        manager = PagedCacheManager(block_size=4, max_blocks=8)
        source = manager.create_block_table("src")
        src_block = manager.allocate_block()
        manager.add_block_to_table(source, src_block, 4)
        stale = manager.create_block_table("dst")
        stale_block = manager.allocate_block()
        manager.add_block_to_table(stale, stale_block, 4)
        free_before = manager.free_blocks

        forked = manager.fork_block_table(source, "dst")

        assert forked is not stale and forked.request_id == "dst"
        assert manager.request_tables["dst"] is forked
        # The overwritten table's only reference was released, not leaked.
        assert stale_block.block_id not in manager.allocated_blocks
        assert manager.free_blocks == free_before + 1
        # The source block is shared by both tables — exactly once.
        assert manager.allocated_blocks[src_block.block_id].ref_count == 2
        manager.delete_block_table("dst")
        assert manager.allocated_blocks[src_block.block_id].ref_count == 1


class TestStoreLayoutFailClosed:
    """Every unblockizable layout is refused before any block, table or
    index entry is touched; the slice helper refuses what it cannot back."""

    def test_store_refuses_unblockizable_layouts_without_side_effects(self):
        states = _kv_layer_states(4)
        cases = {
            "empty-list": [],
            "tuple": (),
            "string": "state",
            "layer-without-state": [{"class_name": "KVCache"}],
            "layer-not-a-dict": [["not", "a", "dict"]],
            "layer-without-class": [{"state": states[0]["state"], "class_name": None}],
            "rotating-class": [
                {"state": states[0]["state"], "class_name": "RotatingKVCache"}
            ],
            "second-layer-bad": states + [{"class_name": "KVCache"}],
        }
        for label, cache_data in cases.items():
            cache, manager = _make_cache(block_size=4)
            allocated_before = manager.stats.allocated_blocks
            assert cache.store_cache("req", list(range(8)), cache_data) is None, label
            assert manager.stats.allocated_blocks == allocated_before, label
            assert manager.get_block_table("req") is None, label
            assert "req" not in cache._request_tables, label
            assert cache._prefix_index == {}, label

    def test_slice_helper_refuses_layer_without_state(self):
        cache, _ = _make_cache(block_size=4)
        assert (
            cache._extract_block_tensor_slice([{"class_name": "KVCache"}], 0, 4) is None
        )

    def test_slice_helper_refuses_block_beyond_backed_rows(self):
        cache, _ = _make_cache(block_size=4)
        states = _kv_layer_states(6)
        assert cache._extract_block_tensor_slice(states, 0, 4) is not None
        # Rows 4..8 are only partially backed (6 rows): the whole block is
        # refused rather than a short slice being stored.
        assert cache._extract_block_tensor_slice(states, 4, 8) is None


class TestStoreHonesty:
    """A store claims exactly the tokens it materialized blocks for."""

    def test_state_with_no_rows_stores_nothing(self):
        import mlx.core as mx

        cache, manager = _make_cache(block_size=4)
        empty = mx.zeros((1, 2, 0, 4), dtype=mx.float32)
        states = [{"state": (empty, empty), "class_name": "KVCache"}]
        allocated_before = manager.stats.allocated_blocks

        assert cache.store_cache("req", list(range(8)), states) is None

        assert manager.stats.allocated_blocks == allocated_before
        assert manager.get_block_table("req") is None
        assert "req" not in cache._request_tables
        assert cache._prefix_index == {}

    def test_short_state_stores_backed_tokens_with_partial_block_identity(self):
        from vllm_mlx.paged_cache import PrefixHasher

        cache, manager = _make_cache(block_size=4)
        tokens8 = list(range(8))
        # Only 6 of the 8 prompt tokens are backed by KV rows.
        table = cache.store_cache("req", tokens8, _kv_layer_states(6))
        assert table is not None
        assert table.num_tokens == 6 and len(table.block_ids) == 2
        full, partial = (manager.allocated_blocks[b] for b in table.block_ids)
        assert full.token_count == 4 and partial.token_count == 2

        # Both blocks carry the cumulative identity through their end...
        assert full.hash_value == PrefixHasher(tokens8[:4]).hexdigest()
        assert partial.hash_value == PrefixHasher(tokens8[:6]).hexdigest()
        # ...but only the full block is registered for sharing.
        assert manager.hash_to_block[full.hash_value] == full.block_id
        assert partial.hash_value not in manager.hash_to_block

        # The index covers exactly the backed prefix and verifies live.
        assert set(cache._prefix_index) == {full.hash_value, partial.hash_value}
        assert cache._prefix_index[partial.hash_value] == (
            tokens8[:6],
            list(table.block_ids),
        )
        assert cache.index_entry_is_stale(tokens8[:6], list(table.block_ids)) is False

        # A later request reuses the full block only — a partial block is
        # never shared — and is told to compute the rest.
        fetched, remaining = cache.fetch_cache("other", tokens8)
        assert fetched is not None
        assert list(fetched.block_ids) == [full.block_id]
        assert remaining == tokens8[4:]

    def test_slice_failure_truncates_store_at_last_whole_block(self):
        from unittest.mock import patch

        import mlx.core as mx

        cache, manager = _make_cache(block_size=4)
        tokens8 = list(range(8))
        real_contiguous = mx.contiguous
        calls = {"n": 0}

        def flaky_contiguous(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 3:  # first tensor of the SECOND block
                raise RuntimeError("injected slice failure")
            return real_contiguous(*args, **kwargs)

        with patch("mlx.core.contiguous", side_effect=flaky_contiguous):
            table = cache.store_cache("req", tokens8, _kv_layer_states(8))

        assert calls["n"] >= 3
        assert table is not None
        assert len(table.block_ids) == 1 and table.num_tokens == 4
        assert manager.stats.allocated_blocks == 2  # null block + one block
        # Index and identity cover only the materialized prefix.
        (entry,) = cache._prefix_index.values()
        assert entry == (tokens8[:4], list(table.block_ids))
        fetched, remaining = cache.fetch_cache("other", tokens8)
        assert fetched is not None
        assert fetched.num_tokens == 4 and remaining == tokens8[4:]


class TestAcquisitionRollback:
    """Candidate acquisition is transactional: a block that vanishes
    between lookup and acquisition truncates the candidate at the last
    contiguous block, a candidate with nothing acquired leaves no table
    behind, and a reference taken on a vanished block is given back."""

    def test_owner_release_after_lookup_truncates_candidate(self):
        from unittest.mock import patch

        cache, manager = _make_cache(block_size=4)
        tokens8 = list(range(8))
        # "short" owns block A; "long" extends a fetch hit on A with B.
        short = cache.store_cache("short", tokens8[:4], _kv_layer_states(4))
        (a,) = short.block_ids
        held, _ = cache.fetch_cache("long", tokens8)
        assert held is not None and list(held.block_ids) == [a]
        long = cache.store_cache("long", tokens8, _kv_layer_states(8))
        assert list(long.block_ids)[0] == a
        b = long.block_ids[1]
        hits_before, saved_before = cache._hits, cache._tokens_saved
        real_lookup = manager.find_shared_prefix

        def lookup_then_owner_releases(tokens, **kwargs):
            ids, rest = real_lookup(tokens, **kwargs)
            assert ids == [a, b]
            cache.release_cache("long")  # B's only owner goes away
            return ids, rest

        with patch.object(
            manager, "find_shared_prefix", side_effect=lookup_then_owner_releases
        ):
            table, remaining = cache.fetch_cache("req", tokens8)

        # Truncated at the vanished block: only A's tokens are claimed.
        assert table is not None and list(table.block_ids) == [a]
        assert table.num_tokens == 4 and remaining == tokens8[4:]
        assert b not in manager.allocated_blocks
        assert manager.allocated_blocks[a].ref_count == 2
        assert cache._hits == hits_before + 1
        assert cache._tokens_saved == saved_before + 4
        assert cache.reconstruct_cache(table)[0].offset == 4

    def test_owner_release_after_lookup_aborts_with_no_table(self):
        from unittest.mock import patch

        cache, manager = _make_cache(block_size=4)
        tokens = list(range(8))
        seed = cache.store_cache("seed", tokens, _kv_layer_states(8))
        seed_ids = list(seed.block_ids)
        real_lookup = manager.find_shared_prefix

        def lookup_then_owner_releases(tokens_, **kwargs):
            ids, rest = real_lookup(tokens_, **kwargs)
            assert ids == seed_ids
            cache.release_cache("seed")
            return ids, rest

        with patch.object(
            manager, "find_shared_prefix", side_effect=lookup_then_owner_releases
        ):
            table, remaining = cache.fetch_cache("req", tokens)

        assert table is None and remaining == tokens
        assert manager.get_block_table("req") is None
        assert "req" not in cache._pending_reconstructed
        assert cache._misses == 1 and cache._hits == 0
        for bid in seed_ids:
            assert bid not in manager.allocated_blocks
        # Every slot is back in the pool: nothing leaked.
        assert manager.free_blocks == manager.max_blocks - 1

    def test_block_vanishing_after_ref_bump_is_rolled_back(self):
        cache, manager = _make_cache(block_size=4)
        tokens = list(range(8))
        seed = cache.store_cache("seed", tokens, _kv_layer_states(8))
        a, victim = seed.block_ids

        class _VanishOnce(dict):
            """Reports the victim gone exactly once — the lookup that
            follows a successful ref bump, as if another owner had freed
            it in between (the manager lock is not held across the two)."""

            armed = True

            def get(self, key, default=None):
                if self.armed and key == victim:
                    self.armed = False
                    return None
                return super().get(key, default)

        manager.allocated_blocks = _VanishOnce(manager.allocated_blocks)

        table, remaining = cache.fetch_cache("req", tokens)

        assert table is not None and list(table.block_ids) == [a]
        assert remaining == tokens[4:]
        # The tentative reference on the vanished block was given back.
        assert manager.allocated_blocks[victim].ref_count == 1
        assert manager.allocated_blocks[a].ref_count == 2
        assert cache._hits == 1 and cache._tokens_saved == 4


class TestPrefixIndexOwnershipGuard:
    """Index metadata is served only while every referenced block still
    owns the recorded cumulative prefix; anything else is a miss for
    fetch and pin, and prunable metadata for pressure."""

    def test_reallocated_slot_is_rejected_and_flagged_stale(self):
        # One usable slot, so the second store reuses the first one's.
        cache, manager = _make_cache(block_size=4, max_blocks=2)
        old = [1, 2, 3, 4]
        new = [5, 6, 7, 8]
        first = cache.store_cache("first", old, _kv_layer_states(4))
        (slot,) = first.block_ids
        (old_key,) = cache._prefix_index
        assert cache._prefix_index[old_key] == (old, [slot])

        # The owner releases, then the slot is refilled for other tokens.
        cache.release_cache("first")
        second = cache.store_cache("second", new, _kv_layer_states(4, base=50.0))
        assert list(second.block_ids) == [slot]
        assert manager.allocated_blocks[slot].cache_data is not None
        # The old entry survives as metadata pointing at the reused slot.
        assert cache._prefix_index[old_key] == (old, [slot])

        # Identity mismatch: the resident KV belongs to ``new``.
        assert cache.index_entry_is_stale(old, [slot]) is True
        assert cache._find_best_prefix_match(old) is None
        table, remaining = cache.fetch_cache("probe", old)
        assert table is None and remaining == old
        assert cache._misses == 1 and cache._hits == 0
        assert manager.allocated_blocks[slot].ref_count == 1

        # The new owner's prefix is intact and still served.
        assert cache.index_entry_is_stale(new, [slot]) is False
        table, remaining = cache.fetch_cache("probe-new", new)
        assert table is not None and remaining == []
        assert manager.allocated_blocks[slot].ref_count == 2

    def test_entry_claiming_more_tokens_than_its_blocks_verify_is_skipped(self):
        from vllm_mlx.paged_cache import PrefixHasher

        cache, _ = _make_cache(block_size=4)
        tokens8 = list(range(8))
        table = cache.store_cache("seed", tokens8[:4], _kv_layer_states(4))
        (slot,) = table.block_ids
        # Metadata that claims 8 tokens but names only the 4-token block.
        cache._prefix_index[PrefixHasher(tokens8).hexdigest()] = (tokens8, [slot])

        # The over-claiming entry is skipped; the longest VERIFIED prefix
        # wins instead.
        assert cache._find_best_prefix_match(tokens8) == (tokens8[:4], [slot])

    def test_entry_whose_blocks_exceed_its_tokens_is_not_served(self):
        from vllm_mlx.paged_cache import PrefixHasher

        cache, _ = _make_cache(block_size=4)
        tokens8 = list(range(8))
        table = cache.store_cache("seed", tokens8, _kv_layer_states(8))
        a, b = table.block_ids
        # Metadata for 4 tokens that names two 4-token blocks.
        key4 = PrefixHasher(tokens8[:4]).hexdigest()
        cache._prefix_index[key4] = (tokens8[:4], [a, b])

        assert cache._find_best_prefix_match(tokens8[:4]) is None
        assert cache.index_entry_is_stale(tokens8[:4], [a, b]) is True

    def test_unhashed_blocks_extend_the_chain_without_being_stale(self):
        from vllm_mlx.paged_cache import PrefixHasher

        cache, manager = _make_cache(block_size=4)
        tokens8 = list(range(8))
        table = cache.store_cache("seed", tokens8, _kv_layer_states(8))
        a, b = table.block_ids

        # A block that claims no identity (legacy/unhashed) is tolerated:
        # the chain continues through its span so a later hashed block is
        # still verified against the SAME cumulative identity.
        manager.allocated_blocks[a].hash_value = None
        assert cache.index_entry_is_stale(tokens8, [a, b]) is False
        # ...and a later hashed block that disagrees still flags the entry.
        manager.allocated_blocks[b].hash_value = PrefixHasher([9] * 8).hexdigest()
        assert cache.index_entry_is_stale(tokens8, [a, b]) is True
        # When the entry's tokens run out before an unhashed block, the
        # rest is unverifiable, not stale.
        manager.allocated_blocks[b].hash_value = None
        assert cache.index_entry_is_stale(tokens8[:4], [a, b]) is False

    def test_index_never_records_a_prefix_past_an_unverifiable_block(self):
        from vllm_mlx.paged_cache import PrefixHasher

        cache, _ = _make_cache(block_size=4)
        tokens8 = list(range(8))
        table = cache.store_cache("seed", tokens8[:4], _kv_layer_states(4))
        (a,) = table.block_ids
        cache._prefix_index.clear()

        cache._update_prefix_index(tokens8, [a, 999])  # 999: no such block

        assert cache._prefix_index == {
            PrefixHasher(tokens8[:4]).hexdigest(): (tokens8[:4], [a]),
        }


class TestPrefixIndexFallbackAndPinning:
    """The block-hash map is a dedup accelerator; the prefix index plus
    per-block identities are the ownership record. Fetch and pin keep
    working through the index when the map has no mapping, and pinning
    sticks to the verified blocks."""

    def test_fetch_falls_back_to_verified_index_entry(self):
        cache, manager = _make_cache(block_size=4)
        tokens = list(range(8))
        seed = cache.store_cache("seed", tokens, _kv_layer_states(8))
        manager.hash_to_block.clear()
        assert manager.find_shared_prefix(tokens, record_stats=False) == ([], tokens)

        table, remaining = cache.fetch_cache("req", tokens + [42])

        assert table is not None
        assert list(table.block_ids) == list(seed.block_ids)
        assert table.num_tokens == 8 and remaining == [42]
        for bid in seed.block_ids:
            assert manager.allocated_blocks[bid].ref_count == 2
        assert cache._hits == 1 and cache._tokens_saved == 8
        assert cache.reconstruct_cache(table)[0].offset == 8

    def test_pin_and_unpin_through_shared_prefix(self):
        cache, manager = _make_cache(block_size=4)
        tokens = list(range(8))
        seed = cache.store_cache("seed", tokens, _kv_layer_states(8))

        assert cache.pin_prefix(tokens) is True
        assert sorted(manager.get_pinned_block_ids()) == sorted(seed.block_ids)
        assert cache.unpin_prefix(tokens) is True
        assert manager.get_pinned_block_ids() == []
        # Nothing pinned any more: a repeat unpin reports no work done.
        assert cache.unpin_prefix(tokens) is False

    def test_pin_and_unpin_through_index_fallback(self):
        cache, manager = _make_cache(block_size=4)
        tokens = list(range(8))
        seed = cache.store_cache("seed", tokens, _kv_layer_states(8))
        manager.hash_to_block.clear()

        assert cache.pin_prefix(tokens) is True
        assert sorted(manager.get_pinned_block_ids()) == sorted(seed.block_ids)
        assert cache.unpin_prefix(tokens) is True
        assert manager.get_pinned_block_ids() == []

    def test_pin_and_unpin_unknown_prefix_report_false(self):
        cache, manager = _make_cache(block_size=4)
        cache.store_cache("seed", list(range(8)), _kv_layer_states(8))
        unknown = [77, 78, 79, 80]

        assert cache.pin_prefix(unknown) is False
        assert cache.unpin_prefix(unknown) is False
        assert manager.get_pinned_block_ids() == []


class TestSchedulerPagedRequestLifecycle:
    """The scheduler's paged-cache lifecycle end to end: an admitted
    request commits a fetch hit and owns its refs, a cancelled request
    releases what its hit acquired, and a committed fetch that cannot be
    rehosted is released and served as a miss — never a phantom hit."""

    def _seed(self, sched, tokens):
        table = sched.block_aware_cache.store_cache(
            "seed", tokens, _kv_layer_states(len(tokens))
        )
        assert table is not None
        return table

    def test_admitted_request_commits_hit_and_owns_refs(self):
        sched = _make_paged_scheduler()
        cache = sched.block_aware_cache
        manager = cache.paged_cache
        tokens = list(range(8))
        seed = self._seed(sched, tokens)

        request = _make_request("req-hit", tokens + [100, 101])
        sched.add_request(request)

        assert request.cache_hit_type == "hit"
        assert request.cached_tokens == 8
        assert request.shared_prefix_blocks == 2
        assert request.remaining_tokens == [100, 101]
        assert request.block_table is not None
        assert request.block_table.request_id == "req-hit"
        assert list(request.block_table.block_ids) == list(seed.block_ids)
        # The caches the fetch transaction built are handed to the request
        # (stash consumed), rehosted at the cached offset.
        assert request.prompt_cache is not None
        assert request.prompt_cache[0].offset == 8
        assert "req-hit" not in cache._pending_reconstructed
        for bid in seed.block_ids:
            assert manager.allocated_blocks[bid].ref_count == 2
        assert cache._hits == 1 and cache._misses == 0
        assert "req-hit" in sched.requests

    def test_unrelated_prompt_is_a_miss_holding_nothing(self):
        sched = _make_paged_scheduler()
        cache = sched.block_aware_cache
        manager = cache.paged_cache
        self._seed(sched, list(range(8)))

        request = _make_request("req-miss", [500, 501, 502, 503, 504])
        sched.add_request(request)

        assert request.cache_hit_type == "miss"
        assert request.cached_tokens == 0
        assert request.prompt_cache is None
        assert request.block_table is None
        assert request.remaining_tokens == request.prompt_token_ids
        assert manager.get_block_table("req-miss") is None
        assert cache._misses == 1 and cache._hits == 0

    def test_cancel_releases_refs_acquired_by_the_hit(self):
        from vllm_mlx.request import RequestStatus

        sched = _make_paged_scheduler()
        cache = sched.block_aware_cache
        manager = cache.paged_cache
        tokens = list(range(8))
        seed = self._seed(sched, tokens)
        request = _make_request("req-cancel", tokens)
        sched.add_request(request)
        for bid in seed.block_ids:
            assert manager.allocated_blocks[bid].ref_count == 2

        assert sched.abort_request("req-cancel") is True
        sched._process_pending_aborts()

        assert request.status == RequestStatus.FINISHED_CANCELLED
        assert request.prompt_cache is None
        assert "req-cancel" in sched.finished_req_ids
        assert manager.get_block_table("req-cancel") is None
        assert "req-cancel" not in cache._pending_reconstructed
        for bid in seed.block_ids:
            assert manager.allocated_blocks[bid].ref_count == 1
            assert manager.allocated_blocks[bid].cache_data is not None
        # Idempotent: the later finished-cleanup release finds nothing held.
        cache.release_cache("req-cancel")
        for bid in seed.block_ids:
            assert manager.allocated_blocks[bid].ref_count == 1
        # The seed entry still serves hits.
        table, remaining = cache.fetch_cache("req-after", tokens)
        assert table is not None and remaining == []

    def test_unrehostable_committed_fetch_is_released_and_served_as_miss(self, caplog):
        import logging
        from unittest.mock import patch

        sched = _make_paged_scheduler()
        cache = sched.block_aware_cache
        manager = cache.paged_cache
        tokens = list(range(8))
        seed = self._seed(sched, tokens)
        request = _make_request("req-fallback", tokens + [7, 7])

        with (
            patch.object(cache, "reconstruct_cache", return_value=None),
            caplog.at_level(logging.WARNING),
        ):
            sched.add_request(request)

        assert request.cache_hit_type == "miss"
        assert request.prompt_cache is None
        assert request.block_table is None
        assert request.cached_tokens == 0
        assert request.remaining_tokens == request.prompt_token_ids
        # The refs the committed fetch held were released: nothing is
        # owned by a request that will now prefill from scratch.
        assert manager.get_block_table("req-fallback") is None
        assert "req-fallback" not in cache._pending_reconstructed
        for bid in seed.block_ids:
            assert manager.allocated_blocks[bid].ref_count == 1
        assert any(
            "reconstruction failed after committed fetch" in record.getMessage()
            for record in caplog.records
        )
        # Admission still completed.
        assert "req-fallback" in sched.requests


class TestSchedulerPagedPressureCleanup:
    """Pressure ticks drain stored owners first, then prune only index
    metadata that no longer owns its prefix — never a live block."""

    def test_owner_eviction_then_stale_metadata_prune_then_quiescence(self):
        sched = _make_paged_scheduler()
        cache = sched.block_aware_cache
        manager = cache.paged_cache
        tokens = list(range(8))
        seed = cache.store_cache("seed", tokens, _kv_layer_states(8))
        seed_ids = list(seed.block_ids)
        index_keys = set(cache._prefix_index)
        assert len(index_keys) == 2

        # Tick 1: the stored owner is evicted (blocks cleared + released).
        assert _pressure_tick(sched, max_evict=1) == 1
        assert "seed" not in cache._request_tables
        for bid in seed_ids:
            assert bid not in manager.allocated_blocks
        # The index outlived its blocks: dead metadata, still present.
        assert set(cache._prefix_index) == index_keys

        # Tick 2: with no owner left, the index path prunes entries whose
        # blocks are gone — one per eviction — without touching any slot.
        free_before = manager.free_blocks
        assert _pressure_tick(sched, max_evict=10) == 2
        assert cache._prefix_index == {}
        assert manager.free_blocks == free_before
        assert sched.num_prefix_cache_pressure_evictions == 3

        # Tick 3: nothing left to reclaim.
        assert _pressure_tick(sched, max_evict=10) == 0
        # A later request is a clean miss — no phantom hit on freed slots.
        table, remaining = cache.fetch_cache("later", tokens)
        assert table is None and remaining == tokens

    def test_index_prune_leaves_active_fetch_blocks_untouched(self):
        sched = _make_paged_scheduler()
        cache = sched.block_aware_cache
        manager = cache.paged_cache
        tokens = list(range(8))
        seed = cache.store_cache("seed", tokens, _kv_layer_states(8))
        seed_ids = list(seed.block_ids)
        # An active request holds the seed's blocks through a fetch hit.
        held, _ = cache.fetch_cache("active", tokens)
        assert held is not None
        for bid in seed_ids:
            assert manager.allocated_blocks[bid].ref_count == 2

        # Owner eviction: shared blocks keep their KV (ref > 1) and the
        # active fetch becomes the sole owner.
        assert _pressure_tick(sched, max_evict=1) == 1
        for bid in seed_ids:
            blk = manager.allocated_blocks[bid]
            assert blk.ref_count == 1 and blk.cache_data is not None

        # Index-only pass: the entries still verify against the live,
        # identity-intact blocks, so nothing is pruned or cleared.
        assert _pressure_tick(sched, max_evict=10) == 0
        assert len(cache._prefix_index) == 2
        for bid in seed_ids:
            blk = manager.allocated_blocks[bid]
            assert blk.ref_count == 1 and blk.cache_data is not None
        # The active request can still be rehosted from its table.
        assert cache.reconstruct_cache(held) is not None

        # Once the active request releases, the entries are dead and the
        # next tick prunes them.
        cache.release_cache("active")
        assert _pressure_tick(sched, max_evict=10) == 2
        assert cache._prefix_index == {}
