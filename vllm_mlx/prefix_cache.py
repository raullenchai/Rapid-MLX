# SPDX-License-Identifier: Apache-2.0
"""
Prefix Cache Manager for rapid-mlx.

Wraps mlx-lm's LRUPromptCache to provide prefix caching functionality,
allowing reuse of computed KV cache for common prompt prefixes.

This module provides two implementations:
- PrefixCacheManager: Original trie-based LRU cache (for backward compatibility)
- BlockAwarePrefixCache: Block-based cache with PagedCacheManager integration
"""

import copy
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

from .errors import PagedCacheUnsupportedLayoutError
from .paged_cache import BlockTable, PagedCacheManager, PrefixHasher

logger = logging.getLogger(__name__)


# Cache classes whose ``is_trimmable()`` returns True unconditionally but which
# corrupt on a trim-then-continue (``ChunkedKVCache`` drops front history via
# ``maybe_trim_front`` + ``start_position``; ``ConcatenateKVCache.trim`` never
# slices its buffers). The canonical denylist + rationale live in
# ``memory_cache._TRIM_UNSAFE_CACHE_CLASSES``; this legacy trie cache keeps a
# self-contained mirror so ``_can_trim_cache`` refuses them too (over-classify
# = safe, only ever skips reuse). Neither is reachable by a supported family
# (llama4 / afm7, not in ``aliases.json``) — defense-in-depth for a latent gap.
_TRIM_UNSAFE_CACHE_CLASSES = frozenset({"ChunkedKVCache", "ConcatenateKVCache"})


@dataclass
class CacheEntry:
    """Entry in the prefix cache."""

    prompt_cache: list[Any]  # The cached KV state
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

    def to_dict(self) -> dict[str, Any]:
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
    Manages prefix caching for rapid-mlx using a trie-based LRU cache.

    This implementation is inspired by mlx-lm's LRUPromptCache but adapted
    for rapid-mlx's batching architecture.

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
        self._cache: dict[Any, dict] = {}

        # LRU tracking: OrderedDict keyed by (model_key, tuple(tokens)), insertion
        # order = least-recently-used first. move_to_end() and popitem() are O(1).
        self._lru: OrderedDict = OrderedDict()

        # Pinned entries: keys excluded from LRU eviction
        self._pinned: set = set()

        # Statistics
        self.stats = PrefixCacheStats()

    def _search(
        self, tokens: list[int]
    ) -> tuple[list[int] | None, list[int] | None, list[int] | None, int]:
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

    def fetch_cache(self, tokens: list[int]) -> tuple[list[Any] | None, list[int]]:
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
                # Deep copy: cache objects have mutable offset/state that
                # generation will modify in-place, corrupting the stored entry.
                return copy.deepcopy(cache_entry.prompt_cache), []

        if shorter:
            # Shorter prefix cached - return cache and remaining tokens
            cache_entry = self._get_cache_entry(shorter)
            if cache_entry:
                self.stats.hits += 1
                self.stats.tokens_saved += len(shorter)
                self._touch_lru(tuple(shorter))
                remaining = tokens[len(shorter) :]
                # Deep copy: same reason as exact match above.
                return copy.deepcopy(cache_entry.prompt_cache), remaining

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

    def store_cache(self, tokens: list[int], prompt_cache: list[Any]) -> None:
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
        key = (self.model_key, tokens_tuple)
        if "cache" in current:
            current["cache"].count += 1
        else:
            current["cache"] = CacheEntry(prompt_cache, 1)

        # Only track in LRU if not pinned (move_to_end is O(1) for OrderedDict)
        if key not in self._pinned:
            if key in self._lru:
                self._lru.move_to_end(key)
            else:
                self._lru[key] = None

        # Evict if over capacity (count pinned entries toward total)
        while len(self._lru) + len(self._pinned) > self.max_size and len(self._lru) > 0:
            self._evict_lru()

    def _get_cache_entry(self, tokens: list[int]) -> CacheEntry | None:
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
        """Move entry to most-recently-used position — O(1) with OrderedDict."""
        key = (self.model_key, tokens_tuple)
        if key in self._pinned:
            return  # Pinned entries stay out of LRU
        if key in self._lru:
            self._lru.move_to_end(key)
        else:
            self._lru[key] = None

    def _evict_lru(self) -> None:
        """Evict least recently used entry — O(1) popitem from OrderedDict."""
        if not self._lru:
            return

        (model_key, tokens_tuple), _ = self._lru.popitem(last=False)
        self._delete_cache(model_key, list(tokens_tuple))
        self.stats.evictions += 1

    def _delete_cache(self, model_key: Any, tokens: list[int]) -> None:
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

    def _can_trim_cache(self, prompt_cache: list[Any]) -> bool:
        """Check if all cache layers can be trimmed.

        A trim-unsafe "trimmable liar" layer (``ChunkedKVCache`` /
        ``ConcatenateKVCache``) reports ``is_trimmable()==True`` but would
        corrupt on ``_trim_cache`` -> ``cache.trim()``, so it is treated as
        NOT trimmable here (falls through to a full prefill).
        """
        if not prompt_cache:
            return False
        for c in prompt_cache:
            name = type(c).__name__
            if any(marker in name for marker in _TRIM_UNSAFE_CACHE_CLASSES):
                return False
            trimmable = (
                c.is_trimmable() if hasattr(c, "is_trimmable") else hasattr(c, "trim")
            )
            if not trimmable:
                return False
        return True

    def _trim_cache(self, prompt_cache: list[Any], num_tokens: int) -> list[Any]:
        """Trim cache by removing num_tokens from the end."""
        for cache in prompt_cache:
            if hasattr(cache, "trim"):
                cache.trim(num_tokens)
        return prompt_cache

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return self.stats.to_dict()

    def reset_stats(self) -> None:
        """Reset statistics."""
        self.stats = PrefixCacheStats()

    def clear(self, *, reset_stats: bool = True) -> None:
        """Clear all cached entries."""
        self._cache.clear()
        self._lru.clear()
        self._pinned.clear()
        if reset_stats:
            self.reset_stats()

    def pin_prefix(self, tokens: list[int]) -> bool:
        """
        Pin a prefix in the cache to prevent eviction.

        For the trie-based cache, this removes the entry from the LRU queue
        so it is never evicted. The entry remains accessible for lookups.

        Note: Pinned entries count toward max_size capacity. If the number of
        pinned entries already equals max_size, this method returns False to
        prevent capacity from becoming unenforceable. Unpin existing entries
        first to make room.

        Args:
            tokens: Token sequence of the prefix to pin

        Returns:
            True if prefix was found and pinned
        """
        tokens_tuple = tuple(tokens)
        key = (self.model_key, tokens_tuple)
        # Verify entry exists in trie
        entry = self._get_cache_entry(tokens)
        if entry is None:
            logger.warning("Cannot pin prefix: not found in cache")
            return False
        # Reject if pinning would make capacity unenforceable
        if key not in self._pinned and len(self._pinned) >= self.max_size:
            logger.warning(
                f"Cannot pin prefix: pinned count ({len(self._pinned)}) "
                f"already at capacity ({self.max_size})"
            )
            return False
        self._lru.pop(key, None)
        self._pinned.add(key)
        logger.info(f"Pinned prefix ({len(tokens)} tokens)")
        return True

    def unpin_prefix(self, tokens: list[int]) -> bool:
        """
        Unpin a prefix, making it eligible for LRU eviction again.

        Args:
            tokens: Token sequence of the prefix to unpin

        Returns:
            True if prefix was found and unpinned
        """
        tokens_tuple = tuple(tokens)
        key = (self.model_key, tokens_tuple)
        if key not in self._pinned:
            return False
        self._pinned.discard(key)
        # Re-add to LRU (at MRU end)
        if key not in self._lru:
            self._lru[key] = None
        logger.info(f"Unpinned prefix ({len(tokens)} tokens) - added back to LRU")
        return True

    def __len__(self) -> int:
        """Return number of cached entries (including pinned)."""
        return len(self._lru) + len(self._pinned)


# =============================================================================
# Block-Aware Prefix Cache (uses PagedCacheManager)
# =============================================================================


def find_paged_incompatible_layers(caches: list[Any]) -> list[str]:
    """Return cache class names the paged block serializer cannot host.

    Structural check: a layer is block-compatible only when it is exactly
    mlx-lm's plain full-attention ``KVCache`` — the one layout the serializer
    can losslessly slice into blocks and reconstruct. Rotating/sliding,
    Arrays/hybrid, recurrent, quantized, subclassed, and unknown cache
    classes all fail closed. Returns a sorted, de-duplicated list of the
    offending class names (empty = fully compatible).
    """
    from mlx_lm.models.cache import KVCache

    return sorted({type(c).__name__ for c in caches if type(c) is not KVCache})


def validate_paged_cache_capability(
    model: Any, *, kv_cache_transform_requested: bool = False
) -> None:
    """Fail closed when the paged prefix cache cannot serve ``model``.

    Probes the model's actual prompt-cache factory
    (``mlx_lm.models.cache.make_prompt_cache``) and raises
    :class:`PagedCacheUnsupportedLayoutError` unless every layer is a plain
    full-attention ``KVCache``. The decision is purely structural — no model
    or architecture names — so an architecture whose sliding mode is inactive
    and whose factory returns plain KV layers is accepted.

    ``kv_cache_transform_requested`` covers every EXPLICIT KV-cache
    transform request (ordinary live quantization or TurboQuant): the paged
    block store implements neither, so the combination is rejected even
    when the transform itself would later fall back or disable at runtime —
    honoring the flag pair by silently serving untransformed plain blocks
    would be exactly the #2955 failure mode.

    This necessarily runs after the model is loaded (the cache factory does
    not exist earlier); callers must invoke it before serving any request so
    an explicit ``--use-paged-cache`` aborts pre-ready instead of silently
    providing zero reuse.
    """
    action = "Remove --use-paged-cache to use the default prefix cache for this model."
    if kv_cache_transform_requested:
        raise PagedCacheUnsupportedLayoutError(
            "--use-paged-cache is incompatible with KV cache "
            "quantization/TurboQuant: the paged block store keeps plain "
            "untransformed KV tensors and implements neither transform, so "
            "an explicit request for both cannot be honored (this fails "
            "closed even when the transform would fall back at runtime). "
            f"Remove the KV quantization/TurboQuant flags, or: {action}"
        )
    try:
        from mlx_lm.models.cache import make_prompt_cache

        probe = make_prompt_cache(model)
    except Exception as exc:
        raise PagedCacheUnsupportedLayoutError(
            "--use-paged-cache could not verify this model's prompt-cache "
            f"layout (cache factory probe failed: {exc!r}); refusing to "
            f"enable an unverifiable paged cache. {action}"
        ) from exc
    if not isinstance(probe, list) or not probe:
        raise PagedCacheUnsupportedLayoutError(
            "--use-paged-cache could not verify this model's prompt-cache "
            f"layout (cache factory returned {type(probe).__name__}); "
            f"refusing to enable an unverifiable paged cache. {action}",
        )
    incompatible = find_paged_incompatible_layers(probe)
    if incompatible:
        raise PagedCacheUnsupportedLayoutError(
            "--use-paged-cache only supports models whose prompt cache is "
            "plain full-attention KVCache on every layer; this model's cache "
            f"layout contains: {', '.join(incompatible)}. {action}",
            incompatible_layers=tuple(incompatible),
        )


@dataclass
class BlockCacheEntry:
    """LRU bookkeeping for a stored request's block table.

    Blocks (``PagedCacheManager.allocated_blocks[*].cache_data``) are the
    source of truth for KV state; this entry deliberately does NOT retain
    the request's original full cache snapshot.
    """

    block_table: BlockTable
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
        self._prefix_index: dict[str, tuple[list[int], list[int]]] = {}

        # Request to block table mapping
        self._request_tables: dict[str, BlockCacheEntry] = {}

        # Reconstructed caches produced inside a committed fetch
        # transaction, keyed by request_id. ``reconstruct_cache`` pops from
        # here so the scheduler's fetch -> reconstruct call pair does not
        # rebuild the tensors a second time.
        self._pending_reconstructed: dict[str, list[Any]] = {}

        # Statistics
        self._hits = 0
        self._misses = 0
        self._tokens_saved = 0

    def fetch_cache(
        self,
        request_id: str,
        tokens: list[int],
    ) -> tuple[BlockTable | None, list[int]]:
        """
        Find cached prefix blocks for the given tokens — transactionally.

        The whole acquisition is one transaction: candidate lookup
        tentatively holds block references, reconstruction validates that
        every candidate block can actually be rebuilt into usable KV state,
        and only then do the hit/tokens-saved counters commit. Any missing
        block, wrong cache class/shape, or reconstruction failure aborts:
        all tentative references and table state are released, exactly one
        miss is counted, and no hit or saved tokens are reported.

        On commit the reconstructed caches are stashed so the follow-up
        ``reconstruct_cache(block_table)`` call returns them without
        rebuilding.

        Args:
            request_id: Unique request identifier
            tokens: Input token sequence

        Returns:
            Tuple of (block_table, remaining_tokens)
            - block_table: BlockTable if prefix found AND reconstructable,
              None otherwise
            - remaining_tokens: Tokens that need processing
        """
        if not tokens:
            return None, tokens

        # A reused request id starts a new lifecycle: release any prior
        # state held under it (stored entry, stale table, pending stash)
        # BEFORE acquiring, so the old table's block references cannot leak
        # when this transaction registers its own table under the same id.
        if (
            request_id in self._request_tables
            or request_id in self._pending_reconstructed
            or self.paged_cache.get_block_table(request_id) is not None
        ):
            self.release_cache(request_id)

        candidate = self._acquire_candidate_blocks(request_id, tokens)
        if candidate is None:
            self._misses += 1
            self.paged_cache.stats.cache_misses += 1
            logger.debug(f"Cache miss for {request_id}")
            return None, tokens

        block_table, remaining = candidate
        reconstructed = self._reconstruct_from_table(block_table)
        if reconstructed is None:
            # Abort: release the tentative refs and table state, count a
            # miss. ``delete_block_table`` decrements exactly the reference
            # this transaction added per block, restoring prior refcounts.
            self.paged_cache.delete_block_table(request_id)
            self._misses += 1
            self.paged_cache.stats.cache_misses += 1
            logger.debug(
                f"Cache candidate for {request_id} failed reconstruction; "
                "aborted fetch transaction (counted as miss)"
            )
            return None, tokens

        # Commit: counters reflect a hit only now that usable KV state
        # exists for the caller.
        self._hits += 1
        self._tokens_saved += block_table.num_tokens
        self.paged_cache.stats.cache_hits += len(block_table.block_ids)
        self._pending_reconstructed[request_id] = reconstructed

        logger.debug(
            f"Cache hit for {request_id}: {len(block_table.block_ids)} "
            f"blocks, {block_table.num_tokens} tokens"
        )
        return block_table, remaining

    def _acquire_candidate_blocks(
        self,
        request_id: str,
        tokens: list[int],
    ) -> tuple[BlockTable, list[int]] | None:
        """Tentatively acquire candidate prefix blocks for ``tokens``.

        Holds one reference per candidate block and registers a block table
        for ``request_id``. Counts nothing — the caller commits or aborts.
        Returns None (with no state held) when there is no candidate.
        """
        shared_block_ids, _ = self.paged_cache.find_shared_prefix(
            tokens, record_stats=False
        )

        matched_block_ids: list[int] = []
        if shared_block_ids:
            matched_block_ids = shared_block_ids
        else:
            best_match = self._find_best_prefix_match(tokens)
            if best_match:
                _, matched_block_ids = best_match

        if not matched_block_ids:
            return None

        block_table = self.paged_cache.create_block_table(request_id)
        for block_id in matched_block_ids:
            # A block that vanished mid-lookup truncates the candidate at
            # the last contiguous block — a gap would silently misalign the
            # reconstructed prefix against the token sequence.
            if not self.paged_cache.increment_ref(block_id):
                break
            block = self.paged_cache.allocated_blocks.get(block_id)
            if block is None:
                self.paged_cache.decrement_ref(block_id)
                break
            block_table.block_ids.append(block_id)
            block_table.num_tokens += block.token_count

        if not block_table.block_ids:
            self.paged_cache.delete_block_table(request_id)
            return None

        # Derive remaining from the tokens actually covered by acquired
        # blocks so a truncated candidate never over-claims coverage.
        return block_table, tokens[block_table.num_tokens :]

    def store_cache(
        self,
        request_id: str,
        tokens: list[int],
        cache_data: list[Any],
    ) -> BlockTable | None:
        """
        Store computed cache for future reuse.

        Blocks are the source of truth: tensor slices are extracted per
        block and materialized as independent arrays, and the original
        ``cache_data`` object is never retained (neither by identity nor via
        MLX slice views into its backing buffers).

        The layout is validated up front. Unsupported layouts — anything
        other than extracted per-layer state dicts whose every layer is a
        plain 4D full-attention ``KVCache`` state — are refused without
        allocating or retaining anything, so an incompatible request leaves
        zero blocks and zero snapshots behind. Every stored block carries
        reconstructable tensor data; a block that cannot be fully backed is
        never added.

        Args:
            request_id: Unique request identifier
            tokens: Token sequence that was processed
            cache_data: List of extracted layer-state dicts with
                'state': (keys, values) tensors and 'class_name'.

        Returns:
            BlockTable for the stored cache, or None when the layout is
            unsupported (nothing stored).
        """
        if not tokens:
            return None

        layout = self._validate_blockizable_layout(cache_data)
        if layout is None:
            logger.debug(
                f"Refusing paged store for {request_id}: cache layout is not "
                "blockizable (only plain 4D KVCache state is supported)"
            )
            return None
        class_name, usable_tokens = layout

        # Only tokens fully backed by tensor data are storable — a block
        # whose token span exceeds the available KV rows would reconstruct
        # into a cache shorter than the prefix it claims.
        storable_tokens = tokens[: min(len(tokens), usable_tokens)]

        existing_tokens = 0
        existing_table = self.paged_cache.get_block_table(request_id)
        if existing_table:
            existing_tokens = existing_table.num_tokens
        new_tokens = storable_tokens[existing_tokens:]

        if not new_tokens and existing_table is None:
            return None

        # Extract (lazily) and then materialize every block slice BEFORE
        # touching any block/table state, so a failure leaves nothing to
        # roll back. ``mx.eval`` forces the contiguous copies so the stored
        # blocks own their buffers and drop every reference to the caller's
        # full KV tensors.
        block_slices: list[list[tuple[Any, Any]]] = []
        num_new_blocks = (len(new_tokens) + self.block_size - 1) // self.block_size
        for i in range(num_new_blocks):
            global_start = existing_tokens + i * self.block_size
            global_end = min(
                global_start + self.block_size, existing_tokens + len(new_tokens)
            )
            block_kv_data = self._extract_block_tensor_slice(
                cache_data, global_start, global_end
            )
            if not block_kv_data:
                break
            block_slices.append(block_kv_data)
        try:
            if block_slices and HAS_MLX:
                mx.eval([t for layers in block_slices for kv in layers for t in kv])
        except Exception as exc:
            # Report failure, never the fetch-held table as a "successful"
            # store: a non-None return tells the scheduler an entry owns the
            # blocks, so it would skip the release path and the fetch-held
            # refs/pending state would leak with no owning entry.
            logger.warning(
                f"Failed to materialize block tensors for {request_id}: {exc}"
            )
            return None

        # All tensor data is ready; now mutate table/block state.
        block_table = existing_table or self.paged_cache.create_block_table(request_id)

        # Cumulative identity: each block is keyed by the hash of the ENTIRE
        # token prefix through its end, seeded with the tokens already
        # covered by a fetched table. KV for a block depends on all
        # preceding blocks, so two sequences that share a chunk but diverge
        # earlier must get distinct identities — chunk-local hashing would
        # dedup them onto one block and silently serve wrong KV.
        identity = PrefixHasher(tokens[:existing_tokens])

        for i, block_kv_data in enumerate(block_slices):
            start_idx = i * self.block_size
            end_idx = min(start_idx + self.block_size, len(new_tokens))
            block_tokens = new_tokens[start_idx:end_idx]
            identity.update(block_tokens)
            block_identity = identity.hexdigest()

            # Check if this block already exists (deduplication). A hash
            # match pins both the full preceding prefix and the block's end
            # position; registered blocks are always full-size, so a match
            # is exactly the same token span under the same history.
            if len(block_tokens) == self.block_size:
                existing_block = self.paged_cache.find_cached_block_by_hash(
                    block_identity, record_stats=False
                )
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
            block.cache_data = block_kv_data
            block.cache_class_name = class_name

            # Record the block's cumulative identity hash so the fetch-side
            # ownership guard can detect a freed-then-reallocated block for
            # EVERY block, including the trailing partial one. Only FULL
            # blocks are additionally registered in ``hash_to_block`` for
            # dedup — partial blocks must never be shared, but they still
            # need a hash_value the guard can re-derive and compare.
            if len(block_tokens) == self.block_size:
                self.paged_cache.register_block_hash_value(block, block_identity)
            else:
                block.hash_value = block_identity

        # Index only the prefix actually backed by the table's blocks.
        self._update_prefix_index(
            tokens[: block_table.num_tokens], block_table.block_ids
        )

        self._request_tables[request_id] = BlockCacheEntry(
            block_table=block_table,
            last_access=time.time(),
        )

        logger.debug(
            f"Stored cache for {request_id}: "
            f"{len(block_table.block_ids)} blocks, "
            f"{block_table.num_tokens} tokens"
        )

        return block_table

    def _validate_blockizable_layout(
        self, cache_data: list[Any]
    ) -> tuple[str, int] | None:
        """Validate that ``cache_data`` can be losslessly blockized.

        Accepts only a non-empty list of extracted layer-state dicts whose
        every layer has an allowlisted ``class_name`` and a 4D
        ``(batch, n_kv_heads, seq, head_dim)`` KV state — the one layout
        ``reconstruct_cache`` can rebuild. Returns ``(class_name,
        usable_tokens)`` where ``usable_tokens`` is the minimum sequence
        length across layers, or None when the layout is unsupported.
        """
        if not HAS_MLX or not isinstance(cache_data, list) or not cache_data:
            return None

        layouts: list[tuple[str, int]] = []
        for layer_state in cache_data:
            if not isinstance(layer_state, dict) or "state" not in layer_state:
                return None
            name = layer_state.get("class_name")
            # Explicit gate: ``_cache_state_seq_axis`` skips the class check
            # when ``class_name`` is None, but a layer without a recorded
            # class must fail closed here.
            if name not in self._SEQ_AXIS_KV_CLASSES:
                return None
            state = layer_state["state"]
            seq_axis = self._cache_state_seq_axis(state, class_name=name)
            if seq_axis is None:
                return None
            keys, values = state
            seq_len = min(keys.shape[seq_axis], values.shape[seq_axis])
            layouts.append((name, seq_len))

        # ``cache_data`` is a non-empty list (checked above) and every
        # iteration either refused the layout or recorded one, so at least
        # one layout is present here.
        class_name = layouts[-1][0]
        usable_tokens = min(seq_len for _, seq_len in layouts)
        return class_name, usable_tokens

    # Cache classes whose ``state`` is ``(keys, values)`` and whose seq
    # axis is well-defined by ndim alone. Other classes (Mamba/DeltaNet
    # ``ArraysCache``, ``QuantizedKVCache`` with tri-tuple values, rotating
    # caches with non-monotonic seq positions) are explicitly rejected
    # even when their tensors happen to look 3D/4D and same-shape.
    _SEQ_AXIS_KV_CLASSES = frozenset({"KVCache"})

    def _cache_state_seq_axis(
        self, state: Any, *, class_name: str | None = None
    ) -> int | None:
        """Return the sequence axis for cache states that support block concat.

        Only the 4D ``(batch, n_kv_heads, seq, head_dim)`` layout is
        supported (seq_axis = 2): mlx-lm's ``KVCache`` accessors hard-code
        ``shape[2]`` for seq, so any state the serializer stores but
        ``reconstruct_cache`` cannot host (e.g. 3D layouts) would be dead
        weight that silently yields zero reuse. Store and reconstruct must
        agree, so both fail closed on anything but 4D.

        ``class_name``, when supplied, must be in ``_SEQ_AXIS_KV_CLASSES`` —
        a Mamba/DeltaNet ``ArraysCache`` may incidentally hold two same-
        shape tensors but is NOT seq-indexed, and slicing it along a
        guessed axis would silently corrupt the cache. When ``class_name``
        is omitted the shape-only check is used (reconstruct-side rehost
        check on already-gated block data).

        Returns ``None`` for unsupported shapes or class names.
        """
        if not isinstance(state, (list, tuple)) or not state:
            return None

        # KV-cache state is always (keys, values); anything else is some
        # other cache class (Mamba conv state + recurrent state, etc.).
        # Reject up-front when either side is missing or non-tensorlike so
        # downstream ``keys.shape`` / ``values.shape`` access is safe.
        if len(state) != 2:
            return None
        if any(t is None or not hasattr(t, "shape") for t in state):
            return None

        if class_name is not None and class_name not in self._SEQ_AXIS_KV_CLASSES:
            return None

        if all(len(tensor.shape) == 4 for tensor in state):
            return 2
        return None

    def _extract_block_tensor_slice(
        self,
        cache_data: list[dict[str, Any]],
        start_idx: int,
        end_idx: int,
    ) -> list[tuple[Any, Any]] | None:
        """
        Extract tensor slices for a single block from cache data.

        Args:
            cache_data: List of layer states, each containing 'state': (keys, values)
            start_idx: Start token index in the sequence
            end_idx: End token index in the sequence

        Returns:
            List of (keys_slice, values_slice) for each layer, or None on
            failure. Slices are wrapped in ``mx.contiguous`` and later
            force-evaluated by ``store_cache`` so the stored copies do not
            retain the caller's full KV buffers. Verified by active-memory
            accounting in ``TestStoredBlockBufferIndependence`` — including
            the single-KV-head case, where a seq-axis slice is already
            row-contiguous and a no-copy shortcut would silently alias the
            whole buffer.
        """
        if not HAS_MLX or not cache_data:
            return None

        try:
            block_slices = []
            for layer_state in cache_data:
                if "state" not in layer_state:
                    return None

                keys, values = layer_state["state"]

                # Reject layers without an explicit allowlisted class_name
                # *before* slicing — otherwise we'd store the block but
                # ``reconstruct_cache`` would later refuse to host it,
                # silently wasting a paged-cache slot. Mamba/DeltaNet
                # ``ArraysCache`` and any future variant are bounced here.
                class_name = layer_state.get("class_name")
                if class_name not in self._SEQ_AXIS_KV_CLASSES:
                    return None

                seq_axis = self._cache_state_seq_axis(
                    (keys, values), class_name=class_name
                )
                if seq_axis is None:
                    # Tensor shape didn't match the supported KV layout even
                    # though the class name was right (e.g. corrupted
                    # state). Bail out entirely.
                    return None

                # Take the min over both tensors. ``keys`` and ``values``
                # share a seq axis by contract, but a paranoid floor avoids
                # an IndexError if one is shorter than the other (e.g. a
                # partially-written cache during a torn shutdown).
                seq_len = min(keys.shape[seq_axis], values.shape[seq_axis])

                if end_idx > seq_len:
                    # A partially-backed block would reconstruct into fewer
                    # KV rows than the tokens it claims — refuse the whole
                    # block rather than store a short slice.
                    logger.debug(
                        f"Block slice [{start_idx}:{end_idx}] exceeds seq_len "
                        f"{seq_len}; refusing partial block"
                    )
                    return None

                block_slices.append(
                    (
                        mx.contiguous(keys[:, :, start_idx:end_idx, :]),
                        mx.contiguous(values[:, :, start_idx:end_idx, :]),
                    )
                )

            return block_slices if block_slices else None

        except Exception as e:
            logger.warning(f"Failed to extract block tensor slice: {e}")
            return None

    def release_cache(self, request_id: str) -> None:
        """
        Release cache/block state held for a request — idempotent.

        Covers every acquisition path: a committed fetch transaction (block
        table + pending reconstructed caches), an aborted one (nothing
        left), and a stored entry. Safe to call repeatedly and after either
        commit or abort.

        Args:
            request_id: Request identifier
        """
        self._pending_reconstructed.pop(request_id, None)
        self._request_tables.pop(request_id, None)
        self.paged_cache.delete_block_table(request_id)
        logger.debug(f"Released cache for {request_id}")

    def fork_cache(
        self,
        source_request_id: str,
        new_request_id: str,
    ) -> BlockTable | None:
        """
        Fork cache from one request to another (COW).

        Only the block table is forked (with reference counts bumped) —
        blocks carry the KV state, so there is no snapshot to share.
        Forking a request onto its own id is a no-op returning the
        existing table (no reference counts change).

        Args:
            source_request_id: Source request ID
            new_request_id: New request ID

        Returns:
            Forked BlockTable (the existing table for a self-fork), or
            None if source not found
        """
        source_entry = self._request_tables.get(source_request_id)
        if not source_entry:
            return None

        # Self-fork is a no-op returning the existing table: the entry
        # already owns it, and a real fork would bump every block ref and
        # then overwrite the only owning entry — leaking one reference per
        # block forever.
        if new_request_id == source_request_id:
            source_entry.last_access = time.time()
            return source_entry.block_table

        # A reused target id must release its prior entry/refs first, or
        # the overwrite below would leak them.
        if (
            new_request_id in self._request_tables
            or self.paged_cache.get_block_table(new_request_id) is not None
        ):
            self.release_cache(new_request_id)

        # Fork block table (increments ref counts)
        forked_table = self.paged_cache.fork_block_table(
            source_entry.block_table,
            new_request_id,
        )

        self._request_tables[new_request_id] = BlockCacheEntry(
            block_table=forked_table,
            last_access=time.time(),
        )

        logger.debug(f"Forked cache: {source_request_id} -> {new_request_id}")

        return forked_table

    def reconstruct_cache(
        self,
        block_table: BlockTable,
    ) -> list[Any] | None:
        """
        Reconstruct KVCache objects from stored block tensor data.

        When ``block_table`` came from a committed ``fetch_cache``
        transaction, this returns the caches already built and validated by
        that transaction (popping the stash). Otherwise it concatenates the
        block tensor slices and builds new KVCache objects.

        Args:
            block_table: BlockTable containing block IDs to reconstruct from

        Returns:
            List of reconstructed KVCache objects (one per layer),
            or None if reconstruction fails
        """
        if block_table is not None and block_table.request_id:
            pending = self._pending_reconstructed.pop(block_table.request_id, None)
            if pending is not None:
                return pending
        return self._reconstruct_from_table(block_table)

    def _reconstruct_from_table(
        self,
        block_table: BlockTable,
    ) -> list[Any] | None:
        """Concatenate block tensor slices into fresh KVCache objects."""
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

                # Belt-and-suspenders: even though the store path gates on
                # ``class_name``, refuse to host anything but a vanilla
                # ``KVCache`` here. mlx_lm's ``KVCache`` accessors hard-code
                # ``shape[2]`` for seq; a rotating/chunked cache with the
                # same 4D shape would be silently misinterpreted.
                if block.cache_class_name not in self._SEQ_AXIS_KV_CLASSES:
                    logger.debug(
                        f"Block {block_id} cache_class_name="
                        f"{block.cache_class_name!r} not in "
                        f"{sorted(self._SEQ_AXIS_KV_CLASSES)}; refusing to "
                        "reconstruct as KVCache."
                    )
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

                # Only 4D ``(batch, n_kv_heads, seq, head_dim)`` states can
                # be hosted by mlx_lm's ``KVCache`` — its accessors are hard-
                # coded to ``shape[2]`` for seq. 3D states (Qwen3.5-style
                # ``(n_kv_heads, seq, head_dim)``) come from non-standard
                # caches and would be silently misinterpreted as 4D here,
                # corrupting any subsequent generation off this prefix.
                seq_axis = self._cache_state_seq_axis((layer_keys[0], layer_values[0]))
                if seq_axis != 2:
                    logger.warning(
                        "Cache layer has non-4D KV shape "
                        f"({getattr(layer_keys[0], 'shape', '?')}); skipping "
                        "reconstruction — mlx_lm.KVCache requires 4D layout."
                    )
                    return None

                concat_keys = mx.concatenate(layer_keys, axis=seq_axis)
                concat_values = mx.concatenate(layer_values, axis=seq_axis)

                # Create KVCache object
                # Try to use mlx_lm's KVCache.from_state if available
                try:
                    from mlx_lm.models.cache import KVCache

                    # Create new cache and set its state
                    cache = KVCache()

                    # Set internal state directly
                    # KVCache stores keys/values and offset
                    cache.keys = concat_keys
                    cache.values = concat_values
                    cache.offset = concat_keys.shape[seq_axis]

                    reconstructed_caches.append(cache)

                except ImportError:
                    # Fallback: create a simple cache-like object
                    class SimpleKVCache:
                        def __init__(self, keys, values, offset: int):
                            self.keys = keys
                            self.values = values
                            self.offset = offset

                        @property
                        def state(self):
                            return (self.keys, self.values)

                        @property
                        def meta_state(self):
                            return (str(self.offset),)

                    cache = SimpleKVCache(
                        concat_keys, concat_values, concat_keys.shape[seq_axis]
                    )
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
        tokens: list[int],
    ) -> tuple[list[int], list[int]] | None:
        """Find best matching prefix in the index."""
        best_match = None
        best_len = 0

        # Try progressively longer block-aligned prefixes; the index is
        # keyed by cumulative-identity hashes, computed incrementally here.
        probe = PrefixHasher()
        for num_blocks in range(1, len(tokens) // self.block_size + 1):
            prefix_len = num_blocks * self.block_size
            probe.update(tokens[prefix_len - self.block_size : prefix_len])
            prefix_hash = probe.hexdigest()

            if prefix_hash in self._prefix_index:
                cached_tokens, block_ids = self._prefix_index[prefix_hash]
                prefix_tokens = tokens[:prefix_len]
                if cached_tokens == prefix_tokens and len(cached_tokens) > best_len:
                    # Hermes patch: stale-block guard. The prefix index
                    # is never pruned when a block is released to the
                    # free queue (ref_count -> 0) or when the paged
                    # manager force-releases KV tensor memory under
                    # Metal pressure (release_pressure_blocks). A stale
                    # hit would truncate ``remaining`` past tokens whose
                    # blocks no longer hold cache_data — dropping the
                    # prefix from the decode is a correctness bomb.
                    # Require every referenced block to still be live
                    # with resident tensor data AND still own the tokens
                    # this prefix expects before trusting the hit.
                    #
                    # Ownership check. cache_data != None is necessary
                    # but NOT sufficient: a block freed under pressure
                    # and then REALLOCATED for different tokens carries
                    # fresh cache_data yet the wrong KV. store_cache
                    # records EVERY block's cumulative identity hash —
                    # the hash of the whole prefix through that block,
                    # full and trailing partial alike — and reallocation
                    # replaces it. Recompute the expected identities over
                    # the ACTUAL stored block spans and require an exact
                    # match, so a reallocated block is rejected even
                    # though its cache_data is non-None.
                    live = True
                    guard = PrefixHasher()
                    end = 0
                    for bid in block_ids:
                        blk = self.paged_cache.allocated_blocks.get(bid)
                        if blk is None or blk.cache_data is None:
                            live = False
                            break
                        span = blk.token_count
                        if span <= 0 or end + span > len(cached_tokens):
                            live = False
                            break
                        guard.update(cached_tokens[end : end + span])
                        end += span
                        if blk.hash_value != guard.hexdigest():
                            live = False
                            break
                    if live and end != len(cached_tokens):
                        live = False
                    if not live:
                        continue
                    best_match = (cached_tokens, block_ids)
                    best_len = len(cached_tokens)

        return best_match

    def _update_prefix_index(
        self,
        tokens: list[int],
        block_ids: list[int],
    ) -> None:
        """Update prefix index with new token sequence.

        Index keys are cumulative-identity hashes over the ACTUAL stored
        block spans (``token_count``), matching the identities recorded on
        the blocks themselves — so index probes, block registrations, and
        ownership guards all agree on what a prefix hash means.
        """
        hasher = PrefixHasher()
        end = 0
        for i, block_id in enumerate(block_ids):
            block = self.paged_cache.allocated_blocks.get(block_id)
            if (
                block is None
                or block.token_count <= 0
                or end + block.token_count > len(tokens)
            ):
                break
            hasher.update(tokens[end : end + block.token_count])
            end += block.token_count
            self._prefix_index[hasher.hexdigest()] = (
                tokens[:end],
                block_ids[: i + 1],
            )

    def index_entry_is_stale(
        self,
        cached_tokens: list[int],
        block_ids: list[int],
    ) -> bool:
        """Return True when this index entry can no longer serve the
        cumulative token prefix it records — safe to prune as metadata.

        Used by pressure eviction to prune index entries without touching
        any physical block. Two prunable conditions:

        * A referenced block is ABSENT from ``allocated_blocks`` (its
          owner released it): the entry can never be acquired again
          (``increment_ref`` fails on absent slots), so keeping it would
          leak dead metadata forever after owner release.
        * A block was reallocated for different tokens (identity
          mismatch), or claims an identity WITHOUT a consistent span —
          unverifiable, fail closed: prune the entry, touch no block.

        Blocks that claim no identity (``hash_value`` None —
        legacy/unhashed fixtures) are tolerated as live.
        """
        hasher = PrefixHasher()
        end = 0
        for block_id in block_ids:
            block = self.paged_cache.allocated_blocks.get(block_id)
            if block is None:
                return True
            if block.hash_value is None:
                # Nothing to verify; extend the chain on a best-effort span
                # so later hashed blocks can still be checked. When the
                # entry's tokens run out the rest is unverifiable, not
                # stale.
                span = block.token_count if block.token_count > 0 else self.block_size
                span = min(span, len(cached_tokens) - end)
                if span <= 0:
                    return False
                hasher.update(cached_tokens[end : end + span])
                end += span
                continue
            if block.token_count <= 0 or end + block.token_count > len(cached_tokens):
                return True
            hasher.update(cached_tokens[end : end + block.token_count])
            end += block.token_count
            if block.hash_value != hasher.hexdigest():
                return True
        return False

    def get_stats(self) -> dict[str, Any]:
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

    def clear(self, *, reset_stats: bool = True) -> None:
        """Clear all cached data."""
        self._request_tables.clear()
        self._prefix_index.clear()
        self._pending_reconstructed.clear()
        self.paged_cache.clear(reset_stats=reset_stats)
        if reset_stats:
            self.reset_stats()

    def pin_prefix(self, tokens: list[int]) -> bool:
        """
        Pin blocks covering a token prefix to prevent eviction.

        Args:
            tokens: Token sequence of the prefix to pin

        Returns:
            True if blocks were found and pinned
        """
        # Find blocks covering this prefix
        shared_block_ids, _ = self.paged_cache.find_shared_prefix(
            tokens, record_stats=False
        )
        if shared_block_ids:
            pinned = self.paged_cache.pin_blocks(shared_block_ids)
            if pinned > 0:
                logger.info(f"Pinned prefix: {pinned} blocks, {len(tokens)} tokens")
                return True

        # Try prefix index
        best_match = self._find_best_prefix_match(tokens)
        if best_match:
            _, block_ids = best_match
            pinned = self.paged_cache.pin_blocks(block_ids)
            if pinned > 0:
                logger.info(
                    f"Pinned prefix via index: {pinned} blocks, {len(tokens)} tokens"
                )
                return True

        logger.warning(
            f"Cannot pin prefix: no cached blocks found for {len(tokens)} tokens"
        )
        return False

    def unpin_prefix(self, tokens: list[int]) -> bool:
        """
        Unpin blocks covering a token prefix.

        Args:
            tokens: Token sequence of the prefix to unpin

        Returns:
            True if blocks were found and unpinned
        """
        shared_block_ids, _ = self.paged_cache.find_shared_prefix(
            tokens, record_stats=False
        )
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
