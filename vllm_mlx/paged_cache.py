# SPDX-License-Identifier: Apache-2.0
"""
Paged KV Cache Manager for vllm-mlx.

This module implements block-based paged KV cache management following vLLM's
architecture (vllm/v1/core/block_pool.py), adapted for MLX on Apple Silicon.

Key components:
- KVCacheBlock: Metadata for each cache block with doubly linked list pointers
- FreeKVCacheBlockQueue: O(1) doubly linked list for LRU block allocation
- BlockHashToBlockMap: Hash-to-block cache for prefix caching
- PagedCacheManager: Main manager with block allocation, prefix caching, and COW

Features:
- Block-based allocation (configurable tokens per block)
- Reference counting for shared blocks
- Copy-on-Write (COW) for efficient prefix sharing
- LRU eviction using doubly linked list (O(1) operations)
- Chain hashing for prefix caching (hash depends on parent block)

Reference: vLLM v1 - vllm/v1/core/block_pool.py, vllm/v1/core/kv_cache_utils.py
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Dict, List, NewType, Optional, Tuple

logger = logging.getLogger(__name__)

# Type alias for block hash (content-based hash for prefix caching)
BlockHash = NewType("BlockHash", bytes)


def compute_block_hash(
    parent_hash: Optional[BlockHash],
    token_ids: List[int],
    extra_keys: Optional[Tuple[Any, ...]] = None,
) -> BlockHash:
    """
    Compute hash for a block based on its content and parent block.

    This enables prefix caching by creating a chain of hashes where
    each block's hash depends on all previous blocks (similar to vLLM).

    Args:
        parent_hash: Hash of the previous block, or None for first block
        token_ids: Token IDs in this block
        extra_keys: Additional keys (e.g., LoRA, multimodal)

    Returns:
        Content-based hash for this block
    """
    hasher = hashlib.sha256()

    # Include parent hash for chain
    if parent_hash:
        hasher.update(parent_hash)
    else:
        # Use fixed seed for reproducibility
        hasher.update(b"vllm-mlx-root")

    # Include token content
    hasher.update(bytes(str(tuple(token_ids)), "utf-8"))

    # Include extra keys if present
    if extra_keys:
        hasher.update(bytes(str(extra_keys), "utf-8"))

    return BlockHash(hasher.digest())


# =============================================================================
# KVCacheBlock - Following vLLM's design
# =============================================================================


@dataclass
class CacheBlock:
    """
    KV cache block metadata following vLLM's design.

    Each block represents a fixed number of tokens (block_size) worth
    of KV cache data. Blocks can be shared across requests via
    reference counting for prefix caching.

    Attributes:
        block_id: Physical block index (0 to num_blocks - 1)
        ref_count: Reference count for sharing (0 = can be evicted)
        block_hash: Content hash for prefix caching (None if not cached)
        prev_free_block: Previous block in free list (doubly linked)
        next_free_block: Next block in free list (doubly linked)
        is_null: True if this is the null/placeholder block
        cache_data: Actual KV tensor data stored in this block
        token_count: Number of tokens stored in this block
    """

    block_id: int
    ref_count: int = 0
    block_hash: Optional[BlockHash] = None

    # Doubly linked list pointers for FreeKVCacheBlockQueue
    prev_free_block: Optional["CacheBlock"] = None
    next_free_block: Optional["CacheBlock"] = None

    # Special flags
    is_null: bool = False
    is_pinned: bool = False  # Pinned blocks are never evicted (e.g., system prompt)

    # Actual tensor data for this block
    # List of (keys, values) per layer, shape: (1, n_kv_heads, block_tokens, head_dim)
    cache_data: Optional[List[Tuple[Any, Any]]] = None

    # Metadata
    token_count: int = 0
    hash_value: Optional[str] = None  # Legacy string hash for compatibility
    last_access: float = field(default_factory=time.time)
    access_count: int = 0  # Total times this block was touched/reused

    def is_full(self, block_size: int) -> bool:
        """Check if block is at capacity."""
        return self.token_count >= block_size

    def is_shared(self) -> bool:
        """Check if block is shared (ref_count > 1)."""
        return self.ref_count > 1

    def reset_hash(self) -> None:
        """Reset block hash when evicted from cache."""
        self.block_hash = None
        self.hash_value = None

    def touch(self) -> None:
        """Update last access time and increment access count."""
        self.last_access = time.time()
        self.access_count += 1

    def __repr__(self) -> str:
        prev_id = self.prev_free_block.block_id if self.prev_free_block else None
        next_id = self.next_free_block.block_id if self.next_free_block else None
        return (
            f"CacheBlock(id={self.block_id}, ref={self.ref_count}, "
            f"tokens={self.token_count}, prev={prev_id}, next={next_id})"
        )


# Alias for backwards compatibility
KVCacheBlock = CacheBlock


# =============================================================================
# FreeKVCacheBlockQueue - O(1) Doubly Linked List (vLLM style)
# =============================================================================


class FreeKVCacheBlockQueue:
    """
    Doubly linked list of free blocks following vLLM's design.

    Provides O(1) operations for:
    - popleft(): Allocate block from front (LRU order)
    - remove(): Remove block from middle (when touched by cache hit)
    - append(): Return block to end (when freed)

    The queue maintains LRU eviction order:
    - Front = least recently used (evict first)
    - Back = most recently used (evict last)

    Uses fake head/tail sentinels to simplify edge cases.
    """

    def __init__(self, blocks: List[CacheBlock]) -> None:
        """
        Initialize queue with all blocks as free.

        Args:
            blocks: List of all CacheBlock objects
        """
        self.num_free_blocks = len(blocks)

        # Initialize doubly linked list
        for i in range(len(blocks)):
            if i > 0:
                blocks[i].prev_free_block = blocks[i - 1]
            if i < len(blocks) - 1:
                blocks[i].next_free_block = blocks[i + 1]

        # Create sentinel nodes (never popped)
        self.fake_head = CacheBlock(block_id=-1)
        self.fake_tail = CacheBlock(block_id=-2)

        if blocks:
            self.fake_head.next_free_block = blocks[0]
            blocks[0].prev_free_block = self.fake_head
            self.fake_tail.prev_free_block = blocks[-1]
            blocks[-1].next_free_block = self.fake_tail
        else:
            self.fake_head.next_free_block = self.fake_tail
            self.fake_tail.prev_free_block = self.fake_head

    def popleft(self) -> CacheBlock:
        """
        Pop and return the first (LRU) non-pinned free block.

        Pinned blocks are skipped to prevent eviction of important
        prefix cache entries (e.g., system prompts).

        Raises:
            ValueError: If no free blocks available
        """
        if self.fake_head.next_free_block is self.fake_tail:
            raise ValueError("No free blocks available")

        # Skip pinned blocks
        block = self.fake_head.next_free_block
        while block is not self.fake_tail and block.is_pinned:
            block = block.next_free_block

        if block is self.fake_tail:
            raise ValueError("No free blocks available (all free blocks are pinned)")

        assert block is not None

        # Remove from list (may not be head if pinned blocks were skipped)
        if block.prev_free_block:
            block.prev_free_block.next_free_block = block.next_free_block
        if block.next_free_block:
            block.next_free_block.prev_free_block = block.prev_free_block

        block.prev_free_block = None
        block.next_free_block = None
        self.num_free_blocks -= 1

        return block

    def popleft_lfu(self, window: int = 8) -> CacheBlock:
        """Pop block with lowest frequency from an LRU window.

        Examines the first ``window`` non-pinned blocks in LRU order and
        evicts the one with the lowest ``access_count``.  Ties are broken
        by LRU position (earlier in list = older).

        Args:
            window: Number of candidates to consider from the LRU end.

        Raises:
            ValueError: If no free blocks available.
        """
        candidates: List[CacheBlock] = []
        block = self.fake_head.next_free_block
        while block is not self.fake_tail and len(candidates) < window:
            if not block.is_pinned:
                candidates.append(block)
            block = block.next_free_block

        if not candidates:
            raise ValueError("No free blocks available")

        # Pick lowest access_count; ties broken by list position (first = LRU)
        victim = min(candidates, key=lambda b: b.access_count)
        self.remove(victim)
        return victim

    def popleft_n(self, n: int) -> List[CacheBlock]:
        """
        Pop n non-pinned blocks from the front.

        Pinned blocks are skipped to prevent eviction of important
        prefix cache entries (e.g., system prompts).

        Args:
            n: Number of blocks to allocate

        Returns:
            List of n free blocks

        Raises:
            AssertionError: If not enough free blocks
        """
        if n == 0:
            return []

        assert (
            self.num_free_blocks >= n
        ), f"Need {n} blocks, have {self.num_free_blocks}"

        result = []
        curr = self.fake_head.next_free_block

        while len(result) < n:
            assert curr is not None and curr is not self.fake_tail, (
                f"Need {n} blocks, only found {len(result)} non-pinned"
            )

            if curr.is_pinned:
                curr = curr.next_free_block
                continue

            # Remove this block from the list
            block = curr
            curr = curr.next_free_block

            if block.prev_free_block:
                block.prev_free_block.next_free_block = block.next_free_block
            if block.next_free_block:
                block.next_free_block.prev_free_block = block.prev_free_block

            block.prev_free_block = None
            block.next_free_block = None
            result.append(block)

        self.num_free_blocks -= n
        return result

    def remove(self, block: CacheBlock) -> None:
        """
        Remove a block from the middle of the queue.

        Used when a free block is "touched" (reused by prefix cache hit).

        Args:
            block: Block to remove

        Raises:
            RuntimeError: If block not in queue
        """
        if block.prev_free_block is None or block.next_free_block is None:
            raise RuntimeError(f"Block {block.block_id} not in free queue")

        # Unlink
        block.prev_free_block.next_free_block = block.next_free_block
        block.next_free_block.prev_free_block = block.prev_free_block
        block.prev_free_block = None
        block.next_free_block = None

        self.num_free_blocks -= 1

    def append(self, block: CacheBlock) -> None:
        """
        Append a block to the end (MRU position).

        Args:
            block: Block to append
        """
        last = self.fake_tail.prev_free_block
        assert last is not None

        last.next_free_block = block
        block.prev_free_block = last
        block.next_free_block = self.fake_tail
        self.fake_tail.prev_free_block = block

        self.num_free_blocks += 1

    def append_n(self, blocks: List[CacheBlock]) -> None:
        """
        Append multiple blocks to the end.

        Args:
            blocks: Blocks to append (in order)
        """
        if not blocks:
            return

        last = self.fake_tail.prev_free_block
        assert last is not None

        for block in blocks:
            block.prev_free_block = last
            last.next_free_block = block
            last = block

        last.next_free_block = self.fake_tail
        self.fake_tail.prev_free_block = last

        self.num_free_blocks += len(blocks)

    def get_all_free_blocks(self) -> List[CacheBlock]:
        """Get all free blocks (for testing)."""
        result = []
        curr = self.fake_head.next_free_block
        while curr and curr is not self.fake_tail:
            result.append(curr)
            curr = curr.next_free_block
        return result


# =============================================================================
# BlockHashToBlockMap - Hash-based prefix cache (vLLM style)
# =============================================================================


class BlockHashToBlockMap:
    """
    Cache mapping block hashes to blocks for prefix caching.

    Follows vLLM's design where the same hash can map to multiple
    blocks (for different KV cache groups in hybrid models).
    """

    def __init__(self) -> None:
        self._cache: Dict[BlockHash, CacheBlock | Dict[int, CacheBlock]] = {}

    def get_block(self, block_hash: BlockHash) -> Optional[CacheBlock]:
        """Get any block with the given hash."""
        blocks = self._cache.get(block_hash)
        if blocks is None:
            return None
        if isinstance(blocks, CacheBlock):
            return blocks
        if isinstance(blocks, dict):
            return next(iter(blocks.values()))
        return None

    def get_best_block(self, block_hash: BlockHash) -> Optional[CacheBlock]:
        """Get block with highest access count for given hash.

        When multiple blocks share the same hash (e.g., in hybrid models),
        prefer the one with the highest reuse count for better cache
        locality.
        """
        blocks = self._cache.get(block_hash)
        if blocks is None:
            return None
        if isinstance(blocks, CacheBlock):
            return blocks
        if isinstance(blocks, dict):
            return max(blocks.values(), key=lambda b: b.access_count)
        return None

    def insert(self, block_hash: BlockHash, block: CacheBlock) -> None:
        """Insert a block into the cache."""
        existing = self._cache.get(block_hash)
        if existing is None:
            self._cache[block_hash] = block
        elif isinstance(existing, CacheBlock):
            self._cache[block_hash] = {
                existing.block_id: existing,
                block.block_id: block,
            }
        elif isinstance(existing, dict):
            existing[block.block_id] = block

    def pop(self, block_hash: BlockHash, block_id: int) -> Optional[CacheBlock]:
        """Remove and return a specific block from the cache."""
        blocks = self._cache.pop(block_hash, None)
        if blocks is None:
            return None

        if isinstance(blocks, CacheBlock):
            if blocks.block_id == block_id:
                return blocks
            # Wrong block ID, put it back
            self._cache[block_hash] = blocks
            return None

        if isinstance(blocks, dict):
            block = blocks.pop(block_id, None)
            if blocks:  # Still has other blocks
                self._cache[block_hash] = blocks
            return block

        return None

    def __len__(self) -> int:
        return len(self._cache)

    def clear(self) -> None:
        self._cache.clear()


# =============================================================================
# BlockTable - Per-request block mapping
# =============================================================================


@dataclass
class BlockTable:
    """
    Per-request block table mapping logical to physical blocks.

    Similar to vLLM's block table, this maps a request's token positions
    to physical cache blocks.

    Attributes:
        request_id: Unique request identifier
        block_ids: List of physical block IDs
        num_tokens: Total number of cached tokens
    """

    request_id: str
    block_ids: List[int] = field(default_factory=list)
    num_tokens: int = 0

    def add_block(self, block_id: int, num_tokens: int) -> None:
        """Add a block to the table."""
        self.block_ids.append(block_id)
        self.num_tokens += num_tokens

    def __len__(self) -> int:
        return len(self.block_ids)

    def copy(self, new_request_id: str) -> "BlockTable":
        """Create a copy with new request ID."""
        return BlockTable(
            request_id=new_request_id,
            block_ids=self.block_ids.copy(),
            num_tokens=self.num_tokens,
        )


# =============================================================================
# CacheStats - Statistics for monitoring
# =============================================================================


@dataclass
class CacheStats:
    """Statistics for cache monitoring."""

    total_blocks: int = 0
    allocated_blocks: int = 0
    free_blocks: int = 0
    shared_blocks: int = 0  # Blocks with ref_count > 1
    total_tokens_cached: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    cow_copies: int = 0
    evictions: int = 0


# =============================================================================
# PagedCacheManager - Main manager (vLLM BlockPool style)
# =============================================================================


class PagedCacheManager:
    """
    Paged KV cache manager following vLLM's BlockPool architecture.

    Features:
    - Block allocation/deallocation with reference counting
    - Prefix sharing via chain-based hash deduplication
    - Copy-on-Write for efficient forking
    - O(1) LRU eviction using doubly linked list

    Args:
        block_size: Number of tokens per block (default: 64)
        max_blocks: Maximum number of blocks to allocate (default: 1000)
        enable_caching: Whether to enable prefix caching (default: True)
    """

    def __init__(
        self,
        block_size: int = 64,
        max_blocks: int = 1000,
        enable_caching: bool = True,
    ):
        self.block_size = block_size
        self.max_blocks = max_blocks
        self.enable_caching = enable_caching

        # Create all blocks
        self.blocks: List[CacheBlock] = [
            CacheBlock(block_id=i) for i in range(max_blocks)
        ]

        # Free block queue (doubly linked list for O(1) LRU)
        self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)

        # Hash-to-block cache for prefix caching
        self.cached_block_hash_to_block = BlockHashToBlockMap()

        # Legacy hash index for compatibility
        self.hash_to_block: Dict[str, int] = {}

        # Request to block table mapping
        self.request_tables: Dict[str, BlockTable] = {}

        # Allocated blocks (for fast lookup)
        self.allocated_blocks: Dict[int, CacheBlock] = {}

        # Reserve null block (block 0) - never freed
        self.null_block = self.free_block_queue.popleft()
        self.null_block.is_null = True
        self.null_block.ref_count = 1
        self.allocated_blocks[self.null_block.block_id] = self.null_block

        # Statistics
        self.stats = CacheStats(
            total_blocks=max_blocks,
            allocated_blocks=1,  # null block
            free_blocks=max_blocks - 1,
        )

        # Thread safety
        self._lock = threading.RLock()

        logger.info(
            f"PagedCacheManager initialized: block_size={block_size}, "
            f"max_blocks={max_blocks}, max_tokens={block_size * max_blocks}"
        )

    # =========================================================================
    # Block Allocation (vLLM style)
    # =========================================================================

    def allocate_block(self) -> Optional[CacheBlock]:
        """
        Allocate a new cache block.

        When prefix caching is enabled, uses frequency-aware eviction
        (LRU-LFU hybrid) to prefer evicting low-frequency blocks.

        Returns:
            CacheBlock if available, None if out of memory.
        """
        with self._lock:
            if self.free_block_queue.num_free_blocks == 0:
                logger.warning("Out of cache blocks")
                return None

            if self.enable_caching:
                block = self.free_block_queue.popleft_lfu()
            else:
                block = self.free_block_queue.popleft()

            # Evict from hash cache if needed
            if self.enable_caching:
                self._maybe_evict_cached_block(block)

            block.ref_count = 1
            block.touch()
            self.allocated_blocks[block.block_id] = block

            self.stats.allocated_blocks += 1
            self.stats.free_blocks -= 1

            return block

    def get_new_blocks(self, num_blocks: int) -> List[CacheBlock]:
        """
        Allocate multiple blocks at once (vLLM style).

        Args:
            num_blocks: Number of blocks to allocate

        Returns:
            List of allocated blocks

        Raises:
            ValueError: If not enough free blocks
        """
        with self._lock:
            if num_blocks > self.free_block_queue.num_free_blocks:
                raise ValueError(
                    f"Cannot allocate {num_blocks} blocks, "
                    f"only {self.free_block_queue.num_free_blocks} available"
                )

            blocks = self.free_block_queue.popleft_n(num_blocks)

            for block in blocks:
                if self.enable_caching:
                    self._maybe_evict_cached_block(block)

                block.ref_count = 1
                block.touch()
                self.allocated_blocks[block.block_id] = block

            self.stats.allocated_blocks += num_blocks
            self.stats.free_blocks -= num_blocks

            return blocks

    def _maybe_evict_cached_block(self, block: CacheBlock) -> bool:
        """
        Evict a block from the hash cache if present.

        Args:
            block: Block to evict

        Returns:
            True if block was evicted from cache
        """
        if block.is_pinned:
            return False

        if block.block_hash is None:
            return False

        evicted = self.cached_block_hash_to_block.pop(block.block_hash, block.block_id)

        if evicted:
            # Also remove from legacy hash index
            if block.hash_value and block.hash_value in self.hash_to_block:
                if self.hash_to_block[block.hash_value] == block.block_id:
                    del self.hash_to_block[block.hash_value]

            block.reset_hash()
            block.cache_data = None  # Free tensor memory
            self.stats.evictions += 1
            return True

        return False

    def free_block(self, block_id: int) -> bool:
        """
        Free a cache block (decrements ref_count, frees if 0).

        Returns:
            True if block was freed, False if still referenced.
        """
        with self._lock:
            if block_id not in self.allocated_blocks:
                logger.warning(f"Attempted to free unknown block: {block_id}")
                return False

            block = self.allocated_blocks[block_id]
            if block.is_null:
                return False  # Never free null block

            block.ref_count -= 1

            if block.ref_count <= 0:
                # Remove from allocated
                del self.allocated_blocks[block_id]

                # Add to free queue (back = MRU)
                self.free_block_queue.append(block)

                self.stats.allocated_blocks -= 1
                self.stats.free_blocks += 1
                self.stats.total_tokens_cached -= block.token_count

                return True

            return False

    def free_blocks(self, blocks: Iterable[CacheBlock]) -> None:
        """
        Free multiple blocks (vLLM style).

        Blocks with ref_count=0 are added to the free queue.

        Args:
            blocks: Blocks to free (in eviction order)
        """
        with self._lock:
            blocks_list = list(blocks)
            to_free = []

            for block in blocks_list:
                if block.is_null:
                    continue

                block.ref_count -= 1

                if block.ref_count <= 0:
                    del self.allocated_blocks[block.block_id]
                    to_free.append(block)
                    self.stats.allocated_blocks -= 1
                    self.stats.free_blocks += 1
                    self.stats.total_tokens_cached -= block.token_count

            # Add to free queue (back = MRU, evicted last)
            self.free_block_queue.append_n(to_free)

    def touch(self, blocks: Iterable[CacheBlock]) -> None:
        """
        Touch blocks to prevent eviction (cache hit, vLLM style).

        Increments ref_count and removes from free queue if needed.

        Args:
            blocks: Blocks to touch
        """
        with self._lock:
            for block in blocks:
                if block.ref_count == 0 and not block.is_null:
                    # Block is in free queue, remove it
                    try:
                        self.free_block_queue.remove(block)
                        self.stats.free_blocks -= 1
                        self.stats.allocated_blocks += 1
                        self.allocated_blocks[block.block_id] = block
                    except RuntimeError:
                        pass  # Block not in queue

                block.ref_count += 1
                block.touch()

    # =========================================================================
    # Reference Counting
    # =========================================================================

    def increment_ref(self, block_id: int) -> bool:
        """Increment reference count for a block."""
        with self._lock:
            if block_id not in self.allocated_blocks:
                return False

            block = self.allocated_blocks[block_id]
            block.ref_count += 1
            block.touch()

            if block.ref_count == 2:
                self.stats.shared_blocks += 1

            return True

    def decrement_ref(self, block_id: int) -> bool:
        """Decrement reference count (alias for free_block)."""
        return self.free_block(block_id)

    # =========================================================================
    # Prefix Caching (vLLM chain-hash style)
    # =========================================================================

    def get_cached_block(self, block_hash: BlockHash) -> Optional[CacheBlock]:
        """
        Get a cached block by its hash (vLLM style).

        Args:
            block_hash: Content hash of the block

        Returns:
            Cached block if found, None otherwise
        """
        if not self.enable_caching:
            return None

        with self._lock:
            block = self.cached_block_hash_to_block.get_block(block_hash)
            if block:
                self.stats.cache_hits += 1
            else:
                self.stats.cache_misses += 1
            return block

    def cache_full_blocks(
        self,
        blocks: List[CacheBlock],
        token_ids: List[int],
        num_cached_blocks: int,
        num_full_blocks: int,
    ) -> None:
        """
        Cache full blocks for prefix caching (vLLM style).

        Computes chain hashes and adds blocks to the cache.

        Args:
            blocks: All blocks for the request
            token_ids: All token IDs for the request
            num_cached_blocks: Number of blocks already cached
            num_full_blocks: Number of full blocks to cache
        """
        if not self.enable_caching:
            return

        if num_cached_blocks >= num_full_blocks:
            return

        with self._lock:
            # Get parent hash from last cached block
            parent_hash = None
            if num_cached_blocks > 0:
                parent_hash = blocks[num_cached_blocks - 1].block_hash

            for i in range(num_cached_blocks, num_full_blocks):
                block = blocks[i]
                if block.block_hash is not None:
                    parent_hash = block.block_hash
                    continue  # Already cached

                # Get tokens for this block
                start = i * self.block_size
                end = start + self.block_size
                block_tokens = token_ids[start:end]

                # Compute chain hash
                block_hash = compute_block_hash(parent_hash, block_tokens)
                block.block_hash = block_hash
                block.token_count = len(block_tokens)

                # Add to cache
                self.cached_block_hash_to_block.insert(block_hash, block)

                # Also maintain legacy hash for compatibility
                legacy_hash = self.compute_block_hash(block_tokens)
                block.hash_value = legacy_hash
                self.hash_to_block[legacy_hash] = block.block_id

                parent_hash = block_hash

    def get_computed_blocks(
        self,
        token_ids: List[int],
    ) -> Tuple[List[CacheBlock], int]:
        """
        Find cached blocks for a token prefix (vLLM style).

        Args:
            token_ids: Token IDs to look up

        Returns:
            Tuple of (cached_blocks, num_cached_tokens)
        """
        if not self.enable_caching:
            return [], 0

        with self._lock:
            cached_blocks = []
            parent_hash = None
            num_cached_tokens = 0

            num_full_blocks = len(token_ids) // self.block_size

            for i in range(num_full_blocks):
                start = i * self.block_size
                end = start + self.block_size
                block_tokens = token_ids[start:end]

                # Compute expected hash
                block_hash = compute_block_hash(parent_hash, block_tokens)

                # Look up in cache (prefer highest-frequency block)
                cached_block = self.cached_block_hash_to_block.get_best_block(block_hash)
                if cached_block is None:
                    self.stats.cache_misses += 1
                    break  # Cache miss, stop here

                cached_blocks.append(cached_block)
                parent_hash = block_hash
                num_cached_tokens += self.block_size
                self.stats.cache_hits += 1

            return cached_blocks, num_cached_tokens

    # =========================================================================
    # Legacy hash methods (for backwards compatibility)
    # =========================================================================

    @staticmethod
    def compute_block_hash(tokens: List[int]) -> str:
        """Compute legacy string hash for a sequence of tokens."""
        token_bytes = bytes(t & 0xFF for t in tokens)
        return hashlib.sha256(token_bytes).hexdigest()[:16]

    def find_cached_block(self, tokens: List[int]) -> Optional[CacheBlock]:
        """
        Find a cached block matching the given tokens (legacy method).
        """
        with self._lock:
            hash_value = self.compute_block_hash(tokens)

            if hash_value in self.hash_to_block:
                block_id = self.hash_to_block[hash_value]
                if block_id in self.allocated_blocks:
                    block = self.allocated_blocks[block_id]
                    block.touch()
                    self.stats.cache_hits += 1
                    return block

            self.stats.cache_misses += 1
            return None

    def register_block_hash(self, block: CacheBlock, tokens: List[int]) -> None:
        """Register a block's hash for deduplication (legacy method)."""
        with self._lock:
            hash_value = self.compute_block_hash(tokens)
            block.hash_value = hash_value
            self.hash_to_block[hash_value] = block.block_id

    # =========================================================================
    # Block Table Management
    # =========================================================================

    def create_block_table(self, request_id: str) -> BlockTable:
        """Create a new block table for a request."""
        with self._lock:
            table = BlockTable(request_id=request_id)
            self.request_tables[request_id] = table
            return table

    def get_block_table(self, request_id: str) -> Optional[BlockTable]:
        """Get block table for a request."""
        with self._lock:
            return self.request_tables.get(request_id)

    def get_or_create_block_table(self, request_id: str) -> BlockTable:
        """Get or create block table for a request."""
        with self._lock:
            if request_id not in self.request_tables:
                self.request_tables[request_id] = BlockTable(request_id=request_id)
            return self.request_tables[request_id]

    def delete_block_table(self, request_id: str) -> None:
        """Delete block table and free associated blocks."""
        with self._lock:
            table = self.request_tables.pop(request_id, None)
            if table:
                for block_id in table.block_ids:
                    self.free_block(block_id)

    def add_block_to_table(
        self,
        table: BlockTable,
        block: CacheBlock,
        tokens_in_block: int,
    ) -> None:
        """Add a block to a block table."""
        with self._lock:
            table.block_ids.append(block.block_id)
            block.token_count = tokens_in_block
            table.num_tokens += tokens_in_block
            self.stats.total_tokens_cached += tokens_in_block

    # =========================================================================
    # Prefix Sharing & COW
    # =========================================================================

    def find_shared_prefix(
        self,
        tokens: List[int],
    ) -> Tuple[List[int], List[int]]:
        """
        Find shared prefix blocks for a token sequence.
        """
        with self._lock:
            shared_blocks = []
            remaining_tokens = tokens.copy()

            while len(remaining_tokens) >= self.block_size:
                chunk = remaining_tokens[: self.block_size]
                cached_block = self.find_cached_block(chunk)

                if cached_block:
                    shared_blocks.append(cached_block.block_id)
                    remaining_tokens = remaining_tokens[self.block_size :]
                else:
                    break

            return shared_blocks, remaining_tokens

    def fork_block_table(
        self,
        source_table: BlockTable,
        new_request_id: str,
    ) -> BlockTable:
        """
        Fork a block table for a new request (COW).
        """
        with self._lock:
            new_table = source_table.copy(new_request_id)

            for block_id in new_table.block_ids:
                self.increment_ref(block_id)

            self.request_tables[new_request_id] = new_table

            logger.debug(
                f"Forked block table: {source_table.request_id} -> {new_request_id}, "
                f"blocks={len(new_table.block_ids)}"
            )

            return new_table

    def get_blocks_for_generation(
        self,
        table: BlockTable,
    ) -> Tuple[List[CacheBlock], bool]:
        """
        Get blocks for generation, applying COW if needed.
        """
        with self._lock:
            blocks = []
            was_copied = False

            for i, block_id in enumerate(table.block_ids):
                block = self.allocated_blocks.get(block_id)
                if not block:
                    continue

                if block.is_shared():
                    new_block = self._cow_copy_block(block)
                    if new_block:
                        table.block_ids[i] = new_block.block_id
                        blocks.append(new_block)
                        was_copied = True
                        self.stats.cow_copies += 1
                    else:
                        blocks.append(block)
                else:
                    blocks.append(block)

                block.touch()

            return blocks, was_copied

    def _cow_copy_block(self, source_block: CacheBlock) -> Optional[CacheBlock]:
        """Create a copy of a block for COW."""
        new_block = self.allocate_block()
        if not new_block:
            return None

        new_block.token_count = source_block.token_count
        new_block.cache_data = source_block.cache_data

        source_block.ref_count -= 1
        if source_block.ref_count == 1:
            self.stats.shared_blocks -= 1

        logger.debug(f"COW copy: block {source_block.block_id} -> {new_block.block_id}")

        return new_block

    # =========================================================================
    # Legacy allocation methods (for backwards compatibility)
    # =========================================================================

    def allocate_blocks_for_tokens(self, num_tokens: int) -> List[CacheBlock]:
        """Allocate enough blocks to hold num_tokens."""
        num_blocks_needed = (num_tokens + self.block_size - 1) // self.block_size
        return self.get_new_blocks(num_blocks_needed)

    # =========================================================================
    # Eviction
    # =========================================================================

    def evict_lru_blocks(self, num_blocks: int) -> int:
        """
        Evict blocks using frequency-aware LRU-LFU hybrid.

        When caching is enabled, examines a small window of LRU candidates
        and picks the one with the lowest access_count (frequency).  This
        keeps high-frequency blocks (e.g. system prompt) cached longer.
        """
        with self._lock:
            evicted = 0

            for _ in range(min(num_blocks, self.free_block_queue.num_free_blocks)):
                try:
                    if self.enable_caching:
                        block = self.free_block_queue.popleft_lfu()
                    else:
                        block = self.free_block_queue.popleft()
                    self._maybe_evict_cached_block(block)
                    # Put back at end (now available for allocation)
                    self.free_block_queue.append(block)
                    evicted += 1
                except ValueError:
                    break

            if evicted > 0:
                logger.info(f"Evicted {evicted} LRU blocks from cache")

            return evicted

    def handle_memory_pressure(self, requested_blocks: int) -> bool:
        """Handle memory pressure by evicting blocks."""
        with self._lock:
            if self.free_block_queue.num_free_blocks >= requested_blocks:
                return True

            needed = requested_blocks - self.free_block_queue.num_free_blocks
            self.evict_lru_blocks(needed)

            return self.free_block_queue.num_free_blocks >= requested_blocks

    # =========================================================================
    # Statistics and Properties
    # =========================================================================

    @property
    def free_blocks(self) -> int:
        """Number of free blocks available."""
        return self.free_block_queue.num_free_blocks

    @property
    def usage(self) -> float:
        """Cache usage ratio (0.0 to 1.0)."""
        total = self.max_blocks - 1  # Exclude null block
        if total == 0:
            return 0.0
        return 1.0 - (self.free_blocks / total)

    def get_stats(self) -> CacheStats:
        """Get current cache statistics."""
        with self._lock:
            self.stats.shared_blocks = sum(
                1 for b in self.allocated_blocks.values() if b.ref_count > 1
            )
            self.stats.free_blocks = self.free_block_queue.num_free_blocks
            return self.stats

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage information."""
        with self._lock:
            stats = self.get_stats()
            return {
                "block_size": self.block_size,
                "max_blocks": self.max_blocks,
                "allocated_blocks": stats.allocated_blocks,
                "free_blocks": stats.free_blocks,
                "shared_blocks": stats.shared_blocks,
                "total_tokens_cached": stats.total_tokens_cached,
                "utilization": stats.allocated_blocks / self.max_blocks,
                "cache_hit_rate": (
                    stats.cache_hits / (stats.cache_hits + stats.cache_misses)
                    if (stats.cache_hits + stats.cache_misses) > 0
                    else 0
                ),
            }

    # =========================================================================
    # Block Pinning (prevent eviction of important prefixes)
    # =========================================================================

    def pin_blocks(self, block_ids: List[int]) -> int:
        """
        Pin blocks to prevent eviction (e.g., system prompt cache).

        Args:
            block_ids: Block IDs to pin

        Returns:
            Number of blocks actually pinned
        """
        with self._lock:
            pinned = 0
            for block_id in block_ids:
                block = self.allocated_blocks.get(block_id)
                if block and not block.is_null and not block.is_pinned:
                    block.is_pinned = True
                    pinned += 1
            if pinned:
                logger.info(f"Pinned {pinned} cache blocks: {block_ids}")
            return pinned

    def unpin_blocks(self, block_ids: List[int]) -> int:
        """
        Unpin blocks to allow eviction.

        Args:
            block_ids: Block IDs to unpin

        Returns:
            Number of blocks actually unpinned
        """
        with self._lock:
            unpinned = 0
            for block_id in block_ids:
                block = self.allocated_blocks.get(block_id)
                if not block:
                    # Also check blocks in the free queue
                    if block_id < len(self.blocks):
                        block = self.blocks[block_id]
                if block and block.is_pinned:
                    block.is_pinned = False
                    unpinned += 1
            if unpinned:
                logger.info(f"Unpinned {unpinned} cache blocks: {block_ids}")
            return unpinned

    def get_pinned_block_ids(self) -> List[int]:
        """Get list of all pinned block IDs."""
        with self._lock:
            return [
                b.block_id for b in self.blocks
                if b.is_pinned
            ]

    def reset_stats(self) -> None:
        """Reset statistics counters."""
        with self._lock:
            self.stats.cache_hits = 0
            self.stats.cache_misses = 0
            self.stats.cow_copies = 0
            self.stats.evictions = 0

    def reset_prefix_cache(self) -> bool:
        """Reset the prefix cache."""
        with self._lock:
            num_used = self.max_blocks - self.free_block_queue.num_free_blocks
            if num_used > 1:  # null_block is always "used"
                logger.warning(f"Cannot reset cache: {num_used - 1} blocks in use")
                return False

            self.cached_block_hash_to_block.clear()
            self.hash_to_block.clear()

            for block in self.blocks:
                block.reset_hash()
                block.cache_data = None

            self.stats.evictions = 0
            self.stats.cache_hits = 0
            self.stats.cache_misses = 0

            logger.info("Prefix cache reset successfully")
            return True

    def clear(self) -> None:
        """Clear all cached data."""
        with self._lock:
            # Recreate blocks and queue
            self.blocks = [CacheBlock(block_id=i) for i in range(self.max_blocks)]
            self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)

            self.cached_block_hash_to_block.clear()
            self.hash_to_block.clear()
            self.request_tables.clear()
            self.allocated_blocks.clear()

            # Reserve null block
            self.null_block = self.free_block_queue.popleft()
            self.null_block.is_null = True
            self.null_block.ref_count = 1
            self.allocated_blocks[self.null_block.block_id] = self.null_block

            self.stats = CacheStats(
                total_blocks=self.max_blocks,
                allocated_blocks=1,
                free_blocks=self.max_blocks - 1,
            )

            logger.info("PagedCacheManager cleared")
