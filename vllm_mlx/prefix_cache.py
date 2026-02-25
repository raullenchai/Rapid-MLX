# SPDX-License-Identifier: Apache-2.0
"""
Prefix Cache Manager for vllm-mlx.

Wraps mlx-lm's LRUPromptCache to provide prefix caching functionality,
allowing reuse of computed KV cache for common prompt prefixes.

This module provides two implementations:
- PrefixCacheManager: Original trie-based LRU cache (for backward compatibility)
- BlockAwarePrefixCache: Block-based cache with PagedCacheManager integration
"""

import copy
import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

from .paged_cache import BlockTable, PagedCacheManager

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Entry in the prefix cache."""

    prompt_cache: List[Any]  # The cached KV state
    count: int  # Reference count for sharing


@dataclass
class PrefixCacheStats:
    """Statistics for prefix cache performance."""

    hits: int = 0
    misses: int = 0
    tokens_saved: int = 0
    total_queries: int = 0
    evictions: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        if self.total_queries == 0:
            return 0.0
        return self.hits / self.total_queries

    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hit_rate,
            "tokens_saved": self.tokens_saved,
            "total_queries": self.total_queries,
            "evictions": self.evictions,
        }


class PrefixCacheManager:
    """
    Manages prefix caching for vllm-mlx using a trie-based LRU cache.

    This implementation is inspired by mlx-lm's LRUPromptCache but adapted
    for vllm-mlx's batching architecture.

    The cache stores KV states keyed by token sequences, allowing:
    - Exact match: Full prompt found in cache
    - Shorter match: Partial prefix found, process remaining tokens
    - Longer match: Cached prefix longer than request, trim excess

    Example:
        cache_manager = PrefixCacheManager(model, max_entries=100)

        # Check for cached prefix
        cache, remaining_tokens = cache_manager.fetch_cache(tokens)
        if cache:
            # Use cached KV, only process remaining_tokens
            pass

        # After generation, store cache for reuse
        cache_manager.store_cache(full_tokens, prompt_cache)
    """

    def __init__(self, model: Any, max_entries: int = 100):
        """
        Initialize the prefix cache manager.

        Args:
            model: The MLX model (used for cache key identification)
            max_entries: Maximum number of cached entries before LRU eviction
        """
        self.model = model
        self.model_key = id(model)
        self.max_size = max_entries

        # Trie-based cache: nested dicts with token keys
        # Structure: {model_key: {token1: {token2: {..., "cache": CacheEntry}}}}
        self._cache: Dict[Any, Dict] = {}

        # LRU tracking: (model_key, tuple(tokens)) ordered by access time
        self._lru: deque = deque()

        # Statistics
        self.stats = PrefixCacheStats()

    def _search(
        self, tokens: List[int]
    ) -> Tuple[Optional[List[int]], Optional[List[int]], Optional[List[int]], int]:
        """
        Search for cached prefix matching tokens.

        Returns:
            Tuple of (exact, shorter, longer, common_prefix_len)
            - exact: Tokens if exact match found
            - shorter: Tokens of shorter cached prefix
            - longer: Tokens of longer cached prefix
            - common_prefix_len: Length of common prefix with longer match
        """
        if self.model_key not in self._cache:
            return None, None, None, 0

        current = self._cache[self.model_key]
        path = []

        # Traverse trie following token sequence
        for i, tok in enumerate(tokens):
            if tok not in current:
                # No match for this token
                # Check if we have a shorter prefix with cache
                if "cache" in current:
                    return None, list(path), None, 0
                return None, None, None, 0

            path.append(tok)
            current = current[tok]

        # Reached end of tokens
        if "cache" in current:
            # Exact match
            return list(tokens), None, None, 0

        # Check for longer cached prefix
        # DFS to find shortest extension with cache
        stack = [(current, list(path))]
        while stack:
            node, node_path = stack.pop()
            if "cache" in node:
                return None, None, node_path, len(tokens)
            for tok, child in node.items():
                if tok != "cache":
                    stack.append((child, node_path + [tok]))

        return None, None, None, 0

    def fetch_cache(self, tokens: List[int]) -> Tuple[Optional[List[Any]], List[int]]:
        """
        Find cached prefix for the given tokens.

        Args:
            tokens: Input token sequence

        Returns:
            Tuple of (cache, remaining_tokens)
            - cache: Cached KV state if found, None otherwise
            - remaining_tokens: Tokens that still need processing
        """
        self.stats.total_queries += 1
        tokens_tuple = tuple(tokens)

        exact, shorter, longer, common_len = self._search(tokens)

        if exact:
            # Exact match - return full cache
            cache_entry = self._get_cache_entry(exact)
            if cache_entry:
                self.stats.hits += 1
                self.stats.tokens_saved += len(tokens)
                self._touch_lru(tokens_tuple)
                # No copy needed - MLX arrays are immutable
                return cache_entry.prompt_cache, []

        if shorter:
            # Shorter prefix cached - return cache and remaining tokens
            cache_entry = self._get_cache_entry(shorter)
            if cache_entry:
                self.stats.hits += 1
                self.stats.tokens_saved += len(shorter)
                self._touch_lru(tuple(shorter))
                remaining = tokens[len(shorter) :]
                # No copy needed - MLX arrays are immutable
                return cache_entry.prompt_cache, remaining

        if longer:
            # Longer prefix cached - trim to match and return
            cache_entry = self._get_cache_entry(longer)
            if cache_entry:
                # Check if cache supports trimming
                prompt_cache = cache_entry.prompt_cache
                if self._can_trim_cache(prompt_cache):
                    trim_amount = len(longer) - len(tokens)
                    trimmed_cache = self._trim_cache(
                        copy.deepcopy(prompt_cache), trim_amount
                    )
                    self.stats.hits += 1
                    self.stats.tokens_saved += len(tokens)
                    return trimmed_cache, []

        # No cache hit
        self.stats.misses += 1
        return None, tokens

    def store_cache(self, tokens: List[int], prompt_cache: List[Any]) -> None:
        """
        Store computed cache for future reuse.

        Args:
            tokens: Token sequence that was processed
            prompt_cache: The computed KV cache to store
        """
        if not tokens:
            return

        tokens_tuple = tuple(tokens)

        # Build trie path
        if self.model_key not in self._cache:
            self._cache[self.model_key] = {}

        current = self._cache[self.model_key]
        for tok in tokens:
            if tok not in current:
                current[tok] = {}
            current = current[tok]

        # Store or update cache entry
        if "cache" in current:
            current["cache"].count += 1
            # Update LRU position
            try:
                self._lru.remove((self.model_key, tokens_tuple))
            except ValueError:
                pass
        else:
            current["cache"] = CacheEntry(prompt_cache, 1)

        self._lru.append((self.model_key, tokens_tuple))

        # Evict if over capacity
        while len(self._lru) > self.max_size:
            self._evict_lru()

    def _get_cache_entry(self, tokens: List[int]) -> Optional[CacheEntry]:
        """Get cache entry for given tokens."""
        if self.model_key not in self._cache:
            return None

        current = self._cache[self.model_key]
        for tok in tokens:
            if tok not in current:
                return None
            current = current[tok]

        return current.get("cache")

    def _touch_lru(self, tokens_tuple: tuple) -> None:
        """Move entry to end of LRU queue (most recently used)."""
        key = (self.model_key, tokens_tuple)
        try:
            self._lru.remove(key)
        except ValueError:
            pass
        self._lru.append(key)

    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self._lru:
            return

        model_key, tokens_tuple = self._lru.popleft()
        self._delete_cache(model_key, list(tokens_tuple))
        self.stats.evictions += 1

    def _delete_cache(self, model_key: Any, tokens: List[int]) -> None:
        """Delete cache entry and clean up empty trie branches."""
        if model_key not in self._cache:
            return

        # Navigate to entry
        path = [(self._cache[model_key], None)]
        current = self._cache[model_key]

        for tok in tokens:
            if tok not in current:
                return
            path.append((current[tok], tok))
            current = current[tok]

        # Delete cache entry
        if "cache" in current:
            del current["cache"]

        # Clean up empty branches (bottom-up)
        for i in range(len(path) - 1, 0, -1):
            node, tok = path[i]
            parent, _ = path[i - 1]
            if not node:  # Empty dict
                del parent[tok]

    def _can_trim_cache(self, prompt_cache: List[Any]) -> bool:
        """Check if cache can be trimmed."""
        if not prompt_cache:
            return False
        # Check if first cache layer has is_trimmable method
        first_cache = prompt_cache[0]
        if hasattr(first_cache, "is_trimmable"):
            return first_cache.is_trimmable()
        return hasattr(first_cache, "trim")

    def _trim_cache(self, prompt_cache: List[Any], num_tokens: int) -> List[Any]:
        """Trim cache by removing num_tokens from the end."""
        for cache in prompt_cache:
            if hasattr(cache, "trim"):
                cache.trim(num_tokens)
        return prompt_cache

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.stats.to_dict()

    def reset_stats(self) -> None:
        """Reset statistics."""
        self.stats = PrefixCacheStats()

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
        self._lru.clear()
        self.reset_stats()

    def pin_prefix(self, tokens: List[int]) -> bool:
        """
        Pin a prefix in the cache to prevent eviction.

        For the trie-based cache, this removes the entry from the LRU queue
        so it is never evicted. The entry remains accessible for lookups.

        Args:
            tokens: Token sequence of the prefix to pin

        Returns:
            True if prefix was found and pinned
        """
        tokens_tuple = tuple(tokens)
        key = (self.model_key, tokens_tuple)
        try:
            self._lru.remove(key)
            logger.info(f"Pinned prefix ({len(tokens)} tokens) - removed from LRU")
            return True
        except ValueError:
            logger.warning(f"Cannot pin prefix: not found in cache")
            return False

    def unpin_prefix(self, tokens: List[int]) -> bool:
        """
        Unpin a prefix, making it eligible for LRU eviction again.

        Args:
            tokens: Token sequence of the prefix to unpin

        Returns:
            True if prefix was found and unpinned
        """
        tokens_tuple = tuple(tokens)
        key = (self.model_key, tokens_tuple)
        # Check the entry exists in the trie
        entry = self._get_cache_entry(tokens)
        if entry is None:
            return False
        # Re-add to LRU (at MRU end)
        if key not in self._lru:
            self._lru.append(key)
            logger.info(f"Unpinned prefix ({len(tokens)} tokens) - added back to LRU")
            return True
        return False

    def __len__(self) -> int:
        """Return number of cached entries."""
        return len(self._lru)


# =============================================================================
# Block-Aware Prefix Cache (uses PagedCacheManager)
# =============================================================================


@dataclass
class BlockCacheEntry:
    """Entry mapping a token sequence to cache blocks."""

    block_table: BlockTable
    cache_data: List[Any]  # Actual KV cache data per block
    last_access: float


class BlockAwarePrefixCache:
    """
    Prefix cache that uses PagedCacheManager for block-based storage.

    Features:
    - Block-level prefix sharing (64 tokens per block)
    - Copy-on-Write for efficient forking
    - Hash-based deduplication across requests
    - Reference counting for memory efficiency

    This is the recommended cache for production use when memory
    efficiency for concurrent requests is important.

    Example:
        paged_manager = PagedCacheManager(block_size=64, max_blocks=1000)
        cache = BlockAwarePrefixCache(model, paged_manager)

        # Check for cached prefix
        block_table, remaining_tokens = cache.fetch_cache(request_id, tokens)

        # After generation, store cache
        cache.store_cache(request_id, tokens, kv_cache_data)

        # Clean up when request completes
        cache.release_cache(request_id)
    """

    def __init__(
        self,
        model: Any,
        paged_cache_manager: PagedCacheManager,
    ):
        """
        Initialize block-aware prefix cache.

        Args:
            model: The MLX model (used for identification)
            paged_cache_manager: The PagedCacheManager instance for block management
        """
        self.model = model
        self.model_key = id(model)
        self.paged_cache = paged_cache_manager
        self.block_size = paged_cache_manager.block_size

        # Hash table for quick prefix lookup
        # Maps hash(tokens[:block_size*n]) -> (tokens, block_ids)
        self._prefix_index: Dict[str, Tuple[List[int], List[int]]] = {}

        # Request to block table mapping
        self._request_tables: Dict[str, BlockCacheEntry] = {}

        # Statistics
        self._hits = 0
        self._misses = 0
        self._tokens_saved = 0

    def fetch_cache(
        self,
        request_id: str,
        tokens: List[int],
    ) -> Tuple[Optional[BlockTable], List[int]]:
        """
        Find cached prefix blocks for the given tokens.

        Args:
            request_id: Unique request identifier
            tokens: Input token sequence

        Returns:
            Tuple of (block_table, remaining_tokens)
            - block_table: BlockTable if prefix found, None otherwise
            - remaining_tokens: Tokens that need processing
        """
        if not tokens:
            return None, tokens

        # Try to find shared prefix blocks
        shared_block_ids, remaining = self.paged_cache.find_shared_prefix(tokens)

        if shared_block_ids:
            # Create block table for this request with shared blocks
            block_table = self.paged_cache.create_block_table(request_id)

            for block_id in shared_block_ids:
                # Increment ref count for sharing
                self.paged_cache.increment_ref(block_id)
                block = self.paged_cache.allocated_blocks.get(block_id)
                if block:
                    block_table.block_ids.append(block_id)
                    block_table.num_tokens += block.token_count

            num_prefix_tokens = len(tokens) - len(remaining)
            self._hits += 1
            self._tokens_saved += num_prefix_tokens

            logger.debug(
                f"Cache hit for {request_id}: "
                f"{len(shared_block_ids)} blocks, {num_prefix_tokens} tokens"
            )

            return block_table, remaining

        # Try prefix index for longer matches
        best_match = self._find_best_prefix_match(tokens)
        if best_match:
            matched_tokens, matched_block_ids = best_match

            # Fork the matched blocks
            block_table = self.paged_cache.create_block_table(request_id)
            for block_id in matched_block_ids:
                self.paged_cache.increment_ref(block_id)
                block = self.paged_cache.allocated_blocks.get(block_id)
                if block:
                    block_table.block_ids.append(block_id)
                    block_table.num_tokens += block.token_count

            remaining = tokens[len(matched_tokens) :]
            self._hits += 1
            self._tokens_saved += len(matched_tokens)

            logger.debug(
                f"Prefix index hit for {request_id}: "
                f"{len(matched_tokens)} tokens matched"
            )

            return block_table, remaining

        # No cache hit
        self._misses += 1
        logger.debug(f"Cache miss for {request_id}")
        return None, tokens

    def store_cache(
        self,
        request_id: str,
        tokens: List[int],
        cache_data: List[Any],
    ) -> Optional[BlockTable]:
        """
        Store computed cache for future reuse.

        This method stores actual tensor data (not references) when cache_data
        contains extracted states from mlx-lm's KVCache.state property.

        Args:
            request_id: Unique request identifier
            tokens: Token sequence that was processed
            cache_data: The computed KV cache to store. Can be:
                - List of KVCache objects (legacy, stores references)
                - List of dicts with 'state': (keys, values) tensors (new, stores slices)

        Returns:
            BlockTable for the stored cache, or None on failure
        """
        if not tokens:
            return None

        # Check if cache_data contains extracted tensor states
        is_tensor_data = (
            cache_data
            and isinstance(cache_data, list)
            and len(cache_data) > 0
            and isinstance(cache_data[0], dict)
            and "state" in cache_data[0]
        )

        # Get or create block table
        block_table = self.paged_cache.get_block_table(request_id)
        if not block_table:
            block_table = self.paged_cache.create_block_table(request_id)

        # Determine tokens we need to cache (not already in block_table)
        existing_tokens = block_table.num_tokens
        new_tokens = tokens[existing_tokens:]

        if not new_tokens:
            # All tokens already cached
            return block_table

        # Allocate blocks for new tokens
        num_new_blocks = (len(new_tokens) + self.block_size - 1) // self.block_size

        for i in range(num_new_blocks):
            start_idx = i * self.block_size
            end_idx = min(start_idx + self.block_size, len(new_tokens))
            block_tokens = new_tokens[start_idx:end_idx]

            # Token range in the original sequence (accounting for existing tokens)
            global_start = existing_tokens + start_idx
            global_end = existing_tokens + end_idx

            # Check if this block already exists (deduplication)
            if len(block_tokens) == self.block_size:
                existing_block = self.paged_cache.find_cached_block(block_tokens)
                if existing_block:
                    # Reuse existing block
                    self.paged_cache.increment_ref(existing_block.block_id)
                    block_table.block_ids.append(existing_block.block_id)
                    block_table.num_tokens += len(block_tokens)
                    continue

            # Allocate new block
            block = self.paged_cache.allocate_block()
            if not block:
                # Handle memory pressure
                if not self.paged_cache.handle_memory_pressure(1):
                    logger.warning(f"Cannot allocate block for {request_id}")
                    break
                block = self.paged_cache.allocate_block()
                if not block:
                    break

            # Store block data
            block.token_count = len(block_tokens)
            block_table.block_ids.append(block.block_id)
            block_table.num_tokens += len(block_tokens)

            # Extract and store actual tensor slices for this block
            if is_tensor_data and HAS_MLX:
                block_kv_data = self._extract_block_tensor_slice(
                    cache_data, global_start, global_end
                )
                if block_kv_data:
                    block.cache_data = block_kv_data
                    logger.debug(
                        f"Stored tensor slice for block {block.block_id}: "
                        f"tokens [{global_start}:{global_end}], {len(block_kv_data)} layers"
                    )

            # Register hash for full blocks (for deduplication)
            if len(block_tokens) == self.block_size:
                self.paged_cache.register_block_hash(block, block_tokens)

        # Update prefix index
        self._update_prefix_index(tokens, block_table.block_ids)

        # Store entry for request (for legacy compatibility)
        self._request_tables[request_id] = BlockCacheEntry(
            block_table=block_table,
            cache_data=cache_data,
            last_access=time.time(),
        )

        blocks_with_data = sum(
            1
            for bid in block_table.block_ids
            if self.paged_cache.allocated_blocks.get(bid)
            and self.paged_cache.allocated_blocks[bid].cache_data is not None
        )

        logger.debug(
            f"Stored cache for {request_id}: "
            f"{len(block_table.block_ids)} blocks ({blocks_with_data} with tensor data), "
            f"{block_table.num_tokens} tokens"
        )

        return block_table

    def _extract_block_tensor_slice(
        self,
        cache_data: List[Dict[str, Any]],
        start_idx: int,
        end_idx: int,
    ) -> Optional[List[Tuple[Any, Any]]]:
        """
        Extract tensor slices for a single block from cache data.

        Args:
            cache_data: List of layer states, each containing 'state': (keys, values)
            start_idx: Start token index in the sequence
            end_idx: End token index in the sequence

        Returns:
            List of (keys_slice, values_slice) for each layer, or None on failure
        """
        if not HAS_MLX or not cache_data:
            return None

        try:
            block_slices = []
            for layer_state in cache_data:
                if "state" not in layer_state:
                    continue

                keys, values = layer_state["state"]

                # KV cache shape: (batch, n_kv_heads, seq_len, head_dim)
                # Slice along seq_len dimension (axis 2)
                seq_len = keys.shape[2] if hasattr(keys, "shape") else 0

                if end_idx > seq_len:
                    # Requested range extends beyond available data
                    logger.debug(
                        f"Block slice [{start_idx}:{end_idx}] exceeds seq_len {seq_len}"
                    )
                    # Use whatever is available
                    actual_end = min(end_idx, seq_len)
                    if start_idx >= actual_end:
                        continue
                    keys_slice = keys[:, :, start_idx:actual_end, :]
                    values_slice = values[:, :, start_idx:actual_end, :]
                else:
                    keys_slice = keys[:, :, start_idx:end_idx, :]
                    values_slice = values[:, :, start_idx:end_idx, :]

                block_slices.append((keys_slice, values_slice))

            return block_slices if block_slices else None

        except Exception as e:
            logger.warning(f"Failed to extract block tensor slice: {e}")
            return None

    def get_cache_for_generation(
        self,
        request_id: str,
    ) -> Tuple[Optional[List[Any]], bool]:
        """
        Get cache data for generation, applying COW if needed.

        Args:
            request_id: Request identifier

        Returns:
            Tuple of (cache_data, was_copied)
        """
        entry = self._request_tables.get(request_id)
        if not entry:
            return None, False

        # Get blocks with COW
        blocks, was_copied = self.paged_cache.get_blocks_for_generation(
            entry.block_table
        )

        if was_copied:
            # Deep copy cache data for modified blocks
            cache_data = copy.deepcopy(entry.cache_data)
        else:
            cache_data = entry.cache_data

        entry.last_access = time.time()
        return cache_data, was_copied

    def release_cache(self, request_id: str) -> None:
        """
        Release cache blocks for a completed request.

        Args:
            request_id: Request identifier
        """
        entry = self._request_tables.pop(request_id, None)
        if entry:
            self.paged_cache.delete_block_table(request_id)
            logger.debug(f"Released cache for {request_id}")

    def fork_cache(
        self,
        source_request_id: str,
        new_request_id: str,
    ) -> Optional[BlockTable]:
        """
        Fork cache from one request to another (COW).

        Args:
            source_request_id: Source request ID
            new_request_id: New request ID

        Returns:
            Forked BlockTable, or None if source not found
        """
        source_entry = self._request_tables.get(source_request_id)
        if not source_entry:
            return None

        # Fork block table (increments ref counts)
        forked_table = self.paged_cache.fork_block_table(
            source_entry.block_table,
            new_request_id,
        )

        # Create new entry with reference to same cache data
        self._request_tables[new_request_id] = BlockCacheEntry(
            block_table=forked_table,
            cache_data=source_entry.cache_data,  # Shared reference
            last_access=time.time(),
        )

        logger.debug(f"Forked cache: {source_request_id} -> {new_request_id}")

        return forked_table

    def reconstruct_cache(
        self,
        block_table: BlockTable,
    ) -> Optional[List[Any]]:
        """
        Reconstruct KVCache objects from stored block tensor data.

        This method concatenates tensor slices from all blocks and
        creates new KVCache objects that can be used for inference.

        Args:
            block_table: BlockTable containing block IDs to reconstruct from

        Returns:
            List of reconstructed KVCache objects (one per layer),
            or None if reconstruction fails
        """
        if not block_table or not block_table.block_ids:
            return None

        if not HAS_MLX:
            logger.warning("Cannot reconstruct cache: MLX not available")
            return None

        try:
            # Collect cache data from all blocks
            all_block_data = []
            for block_id in block_table.block_ids:
                block = self.paged_cache.allocated_blocks.get(block_id)
                if not block:
                    logger.warning(f"Block {block_id} not found in allocated blocks")
                    return None

                if block.cache_data is None:
                    logger.debug(f"Block {block_id} has no tensor data stored")
                    return None

                all_block_data.append(block.cache_data)

            if not all_block_data:
                return None

            # Get number of layers from first block
            num_layers = len(all_block_data[0])
            if num_layers == 0:
                return None

            # Concatenate tensors for each layer
            reconstructed_caches = []

            for layer_idx in range(num_layers):
                layer_keys = []
                layer_values = []

                for block_data in all_block_data:
                    if layer_idx < len(block_data):
                        keys_slice, values_slice = block_data[layer_idx]
                        layer_keys.append(keys_slice)
                        layer_values.append(values_slice)

                if not layer_keys:
                    continue

                # Concatenate along sequence dimension (axis 2)
                # Shape: (batch, n_kv_heads, seq_len, head_dim)
                concat_keys = mx.concatenate(layer_keys, axis=2)
                concat_values = mx.concatenate(layer_values, axis=2)

                # Create KVCache object
                # Try to use mlx_lm's KVCache.from_state if available
                try:
                    from mlx_lm.models.cache import KVCache

                    # Create new cache and set its state
                    cache = KVCache()
                    seq_len = concat_keys.shape[2]

                    # Set internal state directly
                    # KVCache stores keys/values and offset
                    cache.keys = concat_keys
                    cache.values = concat_values
                    cache.offset = seq_len

                    reconstructed_caches.append(cache)

                except ImportError:
                    # Fallback: create a simple cache-like object
                    class SimpleKVCache:
                        def __init__(self, keys, values):
                            self.keys = keys
                            self.values = values
                            self.offset = keys.shape[2]

                        @property
                        def state(self):
                            return (self.keys, self.values)

                        @property
                        def meta_state(self):
                            return (str(self.offset),)

                    cache = SimpleKVCache(concat_keys, concat_values)
                    reconstructed_caches.append(cache)

            if not reconstructed_caches:
                return None

            logger.debug(
                f"Reconstructed cache: {len(reconstructed_caches)} layers, "
                f"{block_table.num_tokens} tokens from {len(block_table.block_ids)} blocks"
            )

            return reconstructed_caches

        except Exception as e:
            logger.warning(f"Failed to reconstruct cache: {e}")
            import traceback

            logger.debug(traceback.format_exc())
            return None

    def _find_best_prefix_match(
        self,
        tokens: List[int],
    ) -> Optional[Tuple[List[int], List[int]]]:
        """Find best matching prefix in the index."""
        best_match = None
        best_len = 0

        # Try progressively longer prefixes
        for num_blocks in range(1, len(tokens) // self.block_size + 1):
            prefix_len = num_blocks * self.block_size
            if prefix_len > len(tokens):
                break

            prefix_tokens = tokens[:prefix_len]
            prefix_hash = self.paged_cache.compute_block_hash(prefix_tokens)

            if prefix_hash in self._prefix_index:
                cached_tokens, block_ids = self._prefix_index[prefix_hash]
                if cached_tokens == prefix_tokens and len(cached_tokens) > best_len:
                    best_match = (cached_tokens, block_ids)
                    best_len = len(cached_tokens)

        return best_match

    def _update_prefix_index(
        self,
        tokens: List[int],
        block_ids: List[int],
    ) -> None:
        """Update prefix index with new token sequence."""
        # Index block-aligned prefixes
        for i in range(1, len(block_ids) + 1):
            prefix_len = min(i * self.block_size, len(tokens))
            prefix_tokens = tokens[:prefix_len]
            prefix_hash = self.paged_cache.compute_block_hash(prefix_tokens)
            self._prefix_index[prefix_hash] = (prefix_tokens, block_ids[:i])

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        paged_stats = self.paged_cache.get_memory_usage()
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": (
                self._hits / (self._hits + self._misses)
                if (self._hits + self._misses) > 0
                else 0
            ),
            "tokens_saved": self._tokens_saved,
            "active_requests": len(self._request_tables),
            **paged_stats,
        }

    def reset_stats(self) -> None:
        """Reset statistics."""
        self._hits = 0
        self._misses = 0
        self._tokens_saved = 0
        self.paged_cache.reset_stats()

    def clear(self) -> None:
        """Clear all cached data."""
        self._request_tables.clear()
        self._prefix_index.clear()
        self.paged_cache.clear()
        self.reset_stats()

    def pin_prefix(self, tokens: List[int]) -> bool:
        """
        Pin blocks covering a token prefix to prevent eviction.

        Args:
            tokens: Token sequence of the prefix to pin

        Returns:
            True if blocks were found and pinned
        """
        # Find blocks covering this prefix
        shared_block_ids, _ = self.paged_cache.find_shared_prefix(tokens)
        if shared_block_ids:
            pinned = self.paged_cache.pin_blocks(shared_block_ids)
            if pinned > 0:
                logger.info(
                    f"Pinned prefix: {pinned} blocks, "
                    f"{len(tokens)} tokens"
                )
                return True

        # Try prefix index
        best_match = self._find_best_prefix_match(tokens)
        if best_match:
            _, block_ids = best_match
            pinned = self.paged_cache.pin_blocks(block_ids)
            if pinned > 0:
                logger.info(
                    f"Pinned prefix via index: {pinned} blocks, "
                    f"{len(tokens)} tokens"
                )
                return True

        logger.warning(f"Cannot pin prefix: no cached blocks found for {len(tokens)} tokens")
        return False

    def unpin_prefix(self, tokens: List[int]) -> bool:
        """
        Unpin blocks covering a token prefix.

        Args:
            tokens: Token sequence of the prefix to unpin

        Returns:
            True if blocks were found and unpinned
        """
        shared_block_ids, _ = self.paged_cache.find_shared_prefix(tokens)
        if shared_block_ids:
            unpinned = self.paged_cache.unpin_blocks(shared_block_ids)
            return unpinned > 0

        best_match = self._find_best_prefix_match(tokens)
        if best_match:
            _, block_ids = best_match
            unpinned = self.paged_cache.unpin_blocks(block_ids)
            return unpinned > 0

        return False

    def __len__(self) -> int:
        """Return number of active request entries."""
        return len(self._request_tables)
