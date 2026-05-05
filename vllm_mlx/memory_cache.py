# SPDX-License-Identifier: Apache-2.0
"""
Memory-aware prefix cache for vllm-mlx.

This module provides a prefix cache implementation that tracks memory usage
and evicts entries based on memory pressure rather than entry count.

Key features:
- Automatic memory limit detection based on available system RAM
- Accurate memory tracking for MLX array caches
- LRU eviction triggered by memory thresholds
- Deep copies on fetch to prevent mutation of stored cache entries

Example:
    config = MemoryCacheConfig(max_memory_percent=0.25)
    cache = MemoryAwarePrefixCache(model, config)

    # Fetch returns reference (no copy) - safe because MLX arrays are immutable
    kv_cache, remaining = cache.fetch(tokens)

    # Store tracks memory automatically
    cache.store(tokens, kv_cache)
"""

from __future__ import annotations

import bisect
import copy
import json
import logging
import math
import os
import struct
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Constants
_BYTES_PER_MB = 1024 * 1024
_DEFAULT_MEMORY_PERCENT = 0.20  # 20% of available RAM
_MIN_MEMORY_BYTES = 100 * _BYTES_PER_MB  # Minimum 100MB
_MAX_ENTRIES_FALLBACK = 50  # Fallback if memory detection fails


def _safetensors_is_complete(path: str) -> bool:
    """Validate a safetensors file is at least as long as its header claims.

    Catches the body-truncated case that ``mx.load`` happily mmaps over —
    a partial KV file that returns zeros at the missing positions and only
    blows up much later with a wrong-output bug. Cheap: reads ≤ a few KB.

    File layout (per safetensors spec):
        [8 bytes LE uint64: header_len]
        [header_len bytes: JSON header with data_offsets per tensor]
        [tensor data]

    Returns False on any structural problem (caller should drop the entry).
    """
    parsed = _read_safetensors_header(path)
    if parsed is None:
        return False
    header, header_len = parsed
    try:
        size = os.path.getsize(path)
        max_end = 0
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            offsets = meta.get("data_offsets") if isinstance(meta, dict) else None
            if (
                not isinstance(offsets, list)
                or len(offsets) != 2
                or not all(isinstance(x, int) for x in offsets)
                or offsets[0] < 0
                or offsets[1] < offsets[0]
            ):
                return False
            if offsets[1] > max_end:
                max_end = offsets[1]
        return size >= 8 + header_len + max_end
    except (OSError, ValueError, struct.error, AttributeError, TypeError):
        return False


def _read_safetensors_header(path: str) -> tuple[dict, int] | None:
    """Parse a safetensors header without loading any tensor data.

    Returns ``(header_dict, header_len_bytes)`` on success, or ``None`` if
    the file is structurally invalid. Both values are needed by
    :func:`_safetensors_is_complete` to compute the absolute end-of-data
    offset; :func:`_safetensors_cache_classes` ignores ``header_len``.
    Returning both from one read avoids opening the file twice.
    """
    try:
        size = os.path.getsize(path)
        if size < 8:
            return None
        with open(path, "rb") as f:
            header_len_bytes = f.read(8)
            if len(header_len_bytes) != 8:
                return None
            header_len = struct.unpack("<Q", header_len_bytes)[0]
            if header_len <= 0 or 8 + header_len > size:
                return None
            header_bytes = f.read(header_len)
            if len(header_bytes) != header_len:
                return None
        header = json.loads(header_bytes)
        if not isinstance(header, dict):
            return None
        return header, header_len
    except (OSError, ValueError, struct.error, AttributeError, TypeError):
        return None


def _safetensors_cache_classes(path: str) -> list[str]:
    """Read mlx-lm cache class names from a safetensors prompt-cache file.

    ``mlx_lm.models.cache.save_prompt_cache`` writes per-layer class names
    under metadata keys of the form ``"2.{layer_idx}"``. This reads them
    back without instantiating the cache — needed to gate disk-cache
    loading on cache-type compatibility (see Bug B in #198).

    Returns ``[]`` if the file is unreadable, has no metadata, or has no
    ``2.*`` keys. The caller treats ``[]`` as "permissive — assume
    ``KVCache``" for backward compat with files saved before the
    in-index ``cache_types`` field existed; that's safe today because
    every mlx-lm version we depend on writes the ``2.*`` metadata, so
    an actually-quantized file always yields a non-empty list. If a
    future mlx-lm changes the metadata key layout this will silently
    misclassify quantized files as KVCache and re-expose Bug A — emit
    a one-time WARNING when the metadata block exists but yields no
    ``2.*`` keys, so format drift is at least visible in logs.
    """
    parsed = _read_safetensors_header(path)
    if parsed is None:
        return []
    header, _ = parsed
    meta = header.get("__metadata__")
    if not isinstance(meta, dict):
        return []
    layer_classes: dict[int, str] = {}
    for k, v in meta.items():
        if not isinstance(k, str) or not k.startswith("2."):
            continue
        try:
            idx = int(k.split(".", 1)[1])
        except (IndexError, ValueError):
            continue
        if isinstance(v, str) and v:
            layer_classes[idx] = v
    if not layer_classes and meta:
        # Header has metadata but no recognizable per-layer class keys —
        # likely a future mlx-lm format change. Surface it so the cause
        # is diagnosable without parsing the safetensors by hand.
        logger.warning(
            f"[cache_persist] {path}: safetensors __metadata__ present "
            f"but no '2.*' cache-class keys found "
            f"(meta_keys={sorted(meta)[:8]}...) — assuming plain KVCache; "
            f"if this file was actually quantized, the entry may crash "
            f"the scheduler at fetch (#198 BUG A)"
        )
    return [layer_classes[i] for i in sorted(layer_classes)]


def _cache_classes_compatible(
    class_names: list[str], config: MemoryCacheConfig
) -> tuple[bool, str]:
    """Check whether a persisted cache is loadable under the current config.

    Reasoning (see #198 BUG B):

    * ``KVCache`` / ``MambaCache`` / etc. — always loadable. Under
      ``kv_quantize`` or ``kv_turboquant`` the next ``store()`` call
      will recompress; until then they pass through fetch unchanged.
    * ``QuantizedKVCache`` — only loadable when ``kv_quantize=True``.
      The dequantize path in ``_decompress_cache`` is guarded on the
      flag; under any other config the tuple-form ``keys`` reach the
      scheduler and crash (#198 BUG A's downstream symptom).
    * ``TurboQuantKVCache`` — only loadable when ``kv_turboquant=True``.
      In practice never persisted (no ``state`` attribute), so this
      branch is defensive.

    Returns ``(is_compatible, reason)``. ``reason`` is empty when ok.
    """
    if not class_names:
        # Backward compat: pre-cache_type files have no class info. Assume
        # KVCache (the only thing all earlier rapid-mlx versions wrote).
        # Always compatible.
        return True, ""
    for cn in class_names:
        if cn == "QuantizedKVCache" and not config.kv_quantize:
            return (
                False,
                f"persisted {cn} requires --kv-cache-quantization "
                "(current config does not enable it)",
            )
        if cn == "TurboQuantKVCache" and not config.kv_turboquant:
            return (
                False,
                f"persisted {cn} requires --kv-cache-turboquant "
                "(current config does not enable it)",
            )
    return True, ""


def _get_available_memory() -> int:
    """
    Get available system memory in bytes.

    Returns:
        Available memory in bytes, or 0 if detection fails.
    """
    try:
        import psutil

        return psutil.virtual_memory().available
    except ImportError:
        logger.warning("psutil not installed, using fallback memory limit")
        return 0
    except Exception as e:
        logger.warning(f"Failed to detect available memory: {e}")
        return 0


def _array_memory(arr) -> int:
    """
    Estimate array memory from shape+dtype without triggering lazy eval.

    Accessing .nbytes on a lazy MLX array forces evaluation of the entire
    computation graph, causing a VRAM spike. This function uses shape and
    dtype metadata (which are always available without eval) to compute
    the same value.

    Args:
        arr: An MLX array or similar object.

    Returns:
        Estimated memory in bytes.
    """
    if arr is None:
        return 0
    if hasattr(arr, "shape") and hasattr(arr, "dtype"):
        dtype = arr.dtype
        if hasattr(dtype, "size"):
            return math.prod(arr.shape) * dtype.size
    # Fallback for non-MLX arrays or objects without shape/dtype
    if hasattr(arr, "nbytes"):
        return arr.nbytes
    return 0


def estimate_kv_cache_memory(cache: list[Any]) -> int:
    """
    Estimate memory usage of a KV cache in bytes.

    This function inspects MLX arrays in the cache and calculates their
    total memory footprint using shape+dtype metadata to avoid triggering
    lazy evaluation (which would cause a VRAM spike).

    Args:
        cache: List of layer cache objects, each containing keys/values tensors.

    Returns:
        Estimated memory usage in bytes.
    """
    if not cache:
        return 0

    total_bytes = 0

    for layer_cache in cache:
        if layer_cache is None:
            continue
        # TurboQuantKVCache: has values_compressed instead of values
        from .turboquant import TurboQuantKVCache

        if isinstance(layer_cache, TurboQuantKVCache):
            total_bytes += layer_cache.memory_bytes
            continue
        # Handle different cache object types
        # Check dict first since dicts have .keys() method that would match below
        if isinstance(layer_cache, dict) and "state" in layer_cache:
            # Extracted state dict
            keys, values = layer_cache["state"]
            total_bytes += _array_memory(keys)
            total_bytes += _array_memory(values)
        # Handle QuantizedKVCache: keys/values are tuples of (data, scales, biases)
        elif hasattr(layer_cache, "keys") and isinstance(
            getattr(layer_cache, "keys", None), (list, tuple)
        ):
            for arr in layer_cache.keys:
                total_bytes += _array_memory(arr)
            for arr in layer_cache.values:
                total_bytes += _array_memory(arr)
            continue
        elif hasattr(layer_cache, "state") and not isinstance(layer_cache, dict):
            # Cache with state property returning (keys, values)
            try:
                keys, values = layer_cache.state
                total_bytes += _array_memory(keys)
                total_bytes += _array_memory(values)
            except (TypeError, ValueError):
                pass
        elif hasattr(layer_cache, "keys") and hasattr(layer_cache, "values"):
            # Standard KVCache with keys/values attributes (not dict)
            keys_attr = layer_cache.keys
            values_attr = layer_cache.values
            # Ensure these are arrays, not methods
            if not callable(keys_attr):
                total_bytes += _array_memory(keys_attr)
            if not callable(values_attr):
                total_bytes += _array_memory(values_attr)

    return total_bytes


@dataclass(frozen=True)
class MemoryCacheConfig:
    """
    Configuration for memory-aware prefix cache.

    Attributes:
        max_memory_mb: Maximum memory in MB. If None, auto-detects.
        max_memory_percent: Fraction of available RAM to use (0.0-1.0).
        max_entries: Hard limit on number of entries (safety net).
        enable_memory_tracking: Whether to track per-entry memory.
        kv_quantize: Whether to quantize KV cache layers for reduced memory.
        kv_bits: Number of bits for KV cache quantization.
        kv_group_size: Group size for KV cache quantization.
        kv_min_quantize_tokens: Minimum sequence length for quantization to apply.
    """

    max_memory_mb: int | None = None
    max_memory_percent: float = _DEFAULT_MEMORY_PERCENT
    max_entries: int = 1000  # Safety limit
    enable_memory_tracking: bool = True
    kv_quantize: bool = False
    kv_bits: int = 8
    kv_group_size: int = 64
    kv_min_quantize_tokens: int = 256
    # TurboQuant V-only compression (asymmetric: K=FP16, V=3-4bit)
    kv_turboquant: bool = False
    kv_turboquant_bits: int | None = None  # None = auto-select by head_dim
    kv_turboquant_group_size: int = 32

    def __post_init__(self) -> None:
        if not 0.0 < self.max_memory_percent <= 1.0:
            raise ValueError(
                f"max_memory_percent must be in (0, 1], got {self.max_memory_percent}"
            )
        if self.max_entries < 1:
            raise ValueError(f"max_entries must be >= 1, got {self.max_entries}")
        if self.kv_min_quantize_tokens < 0:
            raise ValueError(
                f"kv_min_quantize_tokens must be >= 0, got {self.kv_min_quantize_tokens}"
            )

    def compute_memory_limit(self) -> int:
        """
        Compute the memory limit in bytes.

        Returns:
            Memory limit in bytes.
        """
        if self.max_memory_mb is not None:
            return self.max_memory_mb * _BYTES_PER_MB

        available = _get_available_memory()
        if available > 0:
            limit = int(available * self.max_memory_percent)
            return max(limit, _MIN_MEMORY_BYTES)

        # Fallback: assume 8GB system, use configured percent
        fallback_total = 8 * 1024 * _BYTES_PER_MB
        return int(fallback_total * self.max_memory_percent)


@dataclass
class CacheStats:
    """Statistics for cache performance monitoring."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    tokens_saved: int = 0
    current_memory_bytes: int = 0
    max_memory_bytes: int = 0
    entry_count: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def memory_utilization(self) -> float:
        if self.max_memory_bytes == 0:
            return 0.0
        return self.current_memory_bytes / self.max_memory_bytes

    def to_dict(self) -> dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hit_rate, 4),
            "evictions": self.evictions,
            "tokens_saved": self.tokens_saved,
            "current_memory_mb": round(self.current_memory_bytes / _BYTES_PER_MB, 2),
            "max_memory_mb": round(self.max_memory_bytes / _BYTES_PER_MB, 2),
            "memory_utilization": round(self.memory_utilization, 4),
            "entry_count": self.entry_count,
        }


@dataclass
class _CacheEntry:
    """Internal cache entry with memory tracking."""

    tokens: tuple[int, ...]
    cache: list[Any]
    memory_bytes: int

    @classmethod
    def create(cls, tokens: list[int], cache: list[Any]) -> _CacheEntry:
        """Create a cache entry with memory estimation."""
        memory = estimate_kv_cache_memory(cache)
        return cls(
            tokens=tuple(tokens),
            cache=cache,
            memory_bytes=memory,
        )


def _trim_cache_offset(cache: list[Any], trim_by: int) -> list[Any]:
    """Create shallow copies of KVCache/QuantizedKVCache layers with offset reduced.

    This is used when returning a cached KV state to the scheduler so that
    the last N positions are "freed" and the model will recompute them on the
    next forward pass (preventing duplicate KV entries).

    Supports both KVCache (keys/values are arrays) and QuantizedKVCache
    (keys/values are 3-tuples of arrays).
    """
    from mlx_lm.models.cache import KVCache

    try:
        from mlx_lm.models.cache import QuantizedKVCache
    except ImportError:
        QuantizedKVCache = None  # noqa: N806

    trimmed: list[Any] = []
    for layer_cache in cache:
        if layer_cache is None:
            trimmed.append(layer_cache)
            continue
        if QuantizedKVCache is not None and isinstance(layer_cache, QuantizedKVCache):
            tc = QuantizedKVCache.__new__(QuantizedKVCache)
            tc.keys = layer_cache.keys
            tc.values = layer_cache.values
            tc.offset = max(layer_cache.offset - trim_by, 0)
            tc.group_size = layer_cache.group_size
            tc.bits = layer_cache.bits
            trimmed.append(tc)
        elif hasattr(layer_cache, "values_compressed"):
            # TurboQuantKVCache — use its trim method on a copy
            tc = copy.copy(layer_cache)
            tc.trim(trim_by)
            trimmed.append(tc)
        elif (
            hasattr(layer_cache, "offset")
            and hasattr(layer_cache, "keys")
            and not isinstance(layer_cache.keys, (list, tuple))
        ):
            tc = KVCache.__new__(KVCache)
            tc.keys = layer_cache.keys
            tc.values = layer_cache.values
            tc.offset = max(layer_cache.offset - trim_by, 0)
            trimmed.append(tc)
        else:
            # Deep copy unknown/wrapper layers (e.g. CacheList) to prevent
            # aliasing the stored cache entry — generation mutates in-place.
            trimmed.append(copy.deepcopy(layer_cache))
    return trimmed


def _needs_kv_trim(layer: Any) -> bool:
    """Check if a cache layer has oversized KV arrays (duck-typed, no MLX import)."""
    if layer is None:
        return False
    keys = getattr(layer, "keys", None)
    offset = getattr(layer, "offset", None)
    if keys is None or offset is None:
        return False
    if isinstance(keys, (list, tuple)):
        return False  # QuantizedKVCache — skip
    shape = getattr(keys, "shape", None)
    if shape is None or len(shape) < 3:
        return False
    return 0 < offset < shape[2]


def _trim_to_offset(cache: list[Any]) -> list[Any]:
    """Trim KV arrays to their actual used size (offset) before storage.

    KV arrays are often pre-allocated larger than needed (e.g. 4096 slots
    when only 100 are used).  This slices them down to ``offset`` and
    evaluates the result so the original large buffer can be freed.

    Args:
        cache: List of cache layer objects (KVCache or other types).

    Returns:
        New list with KVCache layers trimmed to their offset.
        Non-KVCache layers are passed through unchanged.
    """
    if not any(_needs_kv_trim(layer) for layer in cache):
        return cache

    import mlx.core as mx
    from mlx_lm.models.cache import KVCache

    trimmed = []
    eval_targets = []
    for layer in cache:
        if isinstance(layer, KVCache) and layer.keys is not None:
            offset = layer.offset
            if offset <= 0 or offset >= layer.keys.shape[2]:
                trimmed.append(layer)
                continue
            tc = KVCache()
            tc.keys = layer.keys[:, :, :offset, :]
            tc.values = layer.values[:, :, :offset, :]
            tc.offset = offset
            eval_targets.extend([tc.keys, tc.values])
            trimmed.append(tc)
        else:
            trimmed.append(layer)

    if eval_targets:
        mx.eval(*eval_targets)

    return trimmed


def _quantize_cache(cache: list[Any], bits: int = 8, group_size: int = 64) -> list[Any]:
    """Quantize KVCache layers to reduce memory. Non-KVCache layers are kept as-is."""
    from mlx_lm.models.cache import KVCache

    quantized = []
    for layer in cache:
        if layer is None:
            quantized.append(layer)
            continue
        if isinstance(layer, KVCache) and layer.keys is not None:
            quantized.append(layer.to_quantized(group_size=group_size, bits=bits))
        else:
            quantized.append(layer)
    return quantized


def _dequantize_cache(cache: list[Any]) -> list[Any]:
    """Dequantize QuantizedKVCache layers back to regular KVCache."""
    import mlx.core as mx
    from mlx_lm.models.cache import KVCache, QuantizedKVCache

    result = []
    for layer in cache:
        if layer is None:
            result.append(layer)
            continue
        if isinstance(layer, QuantizedKVCache) and layer.keys is not None:
            kv = KVCache()
            kv.keys = mx.dequantize(
                *layer.keys, group_size=layer.group_size, bits=layer.bits
            )
            kv.values = mx.dequantize(
                *layer.values, group_size=layer.group_size, bits=layer.bits
            )
            kv.offset = layer.offset
            result.append(kv)
        else:
            result.append(layer)
    return result


def _turboquant_compress_cache(
    cache: list[Any], bits: int | None, group_size: int
) -> list[Any]:
    """Compress KVCache V tensors using TurboQuant (K stays FP16)."""
    from mlx_lm.models.cache import KVCache

    from .turboquant import TurboQuantConfig, TurboQuantKVCache, auto_select_bits

    compressed_count = 0
    result = []
    for layer in cache:
        if layer is None:
            result.append(layer)
            continue
        if isinstance(layer, KVCache) and layer.keys is not None:
            head_dim = layer.values.shape[-1] if layer.values is not None else 128
            actual_bits = bits if bits is not None else auto_select_bits(head_dim)
            config = TurboQuantConfig(bits=actual_bits, group_size=group_size)
            result.append(TurboQuantKVCache.from_kv_cache(layer, config))
            compressed_count += 1
        else:
            result.append(layer)

    if compressed_count > 0:
        logger.debug(
            f"TurboQuant compressed {compressed_count}/{len(cache)} layers "
            f"({bits or 'auto'}-bit, group_size={group_size})"
        )
    return result


def _turboquant_decompress_cache(cache: list[Any]) -> list[Any]:
    """Decompress TurboQuantKVCache layers back to regular KVCache."""
    from .turboquant import TurboQuantKVCache

    result = []
    for layer in cache:
        if layer is None:
            result.append(layer)
            continue
        if isinstance(layer, TurboQuantKVCache) and layer.keys is not None:
            result.append(layer.to_kv_cache())
        else:
            result.append(layer)
    return result


class MemoryAwarePrefixCache:
    """
    Prefix cache with memory-based eviction.

    This cache tracks memory usage per entry and evicts based on memory
    pressure rather than entry count. It uses LRU (Least Recently Used)
    ordering for eviction decisions.

    Key design decisions:
    - No deep copies on fetch: MLX arrays are immutable, so sharing is safe
    - Memory tracking per entry: Accurate accounting for eviction
    - Auto-detection of available RAM: Adapts to different systems
    - OrderedDict for O(1) LRU operations

    Thread Safety:
        ``fetch``, ``store``, ``remove`` and ``clear`` hold an internal lock,
        so it is safe to call them from different threads (e.g. the asyncio
        event loop calling ``fetch`` while the mlx-step worker calls
        ``store``). Read-only attribute access (``__contains__``, ``__len__``,
        ``get_stats``) is single-op and relies on the GIL — no lock needed.
    """

    def __init__(
        self,
        model: Any,
        config: MemoryCacheConfig | None = None,
    ) -> None:
        """
        Initialize the memory-aware prefix cache.

        Args:
            model: The MLX model (used for identification).
            config: Cache configuration. Uses defaults if None.
        """
        self._model_id = id(model)
        self._config = config or MemoryCacheConfig()

        # OrderedDict maintains insertion order for LRU
        # Key: tuple(tokens), Value: _CacheEntry
        self._entries: OrderedDict[tuple[int, ...], _CacheEntry] = OrderedDict()

        # Sorted index of token keys for efficient prefix/supersequence lookup.
        # Tuple lexicographic ordering means a prefix key P is always < any
        # extension of P, so bisect gives O(log N) range scans instead of O(N).
        self._sorted_keys: list[tuple[int, ...]] = []

        # Memory tracking
        self._max_memory = self._config.compute_memory_limit()
        self._current_memory = 0

        # Statistics
        self._stats = CacheStats(max_memory_bytes=self._max_memory)

        # Track the match type from the last fetch() call
        self._last_match_type: str | None = None

        # Guards _entries / _sorted_keys mutations against concurrent
        # fetch/store/evict from multiple threads (asyncio loop + mlx-step).
        self._lock = threading.Lock()

        logger.info(
            f"MemoryAwarePrefixCache initialized: "
            f"max_memory={self._max_memory / _BYTES_PER_MB:.1f}MB, "
            f"max_entries={self._config.max_entries}"
        )

    def _decompress_cache(self, cache: list[Any]) -> list[Any]:
        """Decompress cache layers (TurboQuant or standard quantization)."""
        if self._config.kv_turboquant:
            return _turboquant_decompress_cache(cache)
        elif self._config.kv_quantize:
            return _dequantize_cache(cache)
        return cache

    def fetch(self, tokens: list[int]) -> tuple[list[Any] | None, list[int]]:
        """
        Find cached KV state for the given tokens.

        This method searches for exact matches, prefix matches, supersequence
        matches, and longest-common-prefix (LCP) matches.  Uses a sorted key
        index for O(log N) lookup instead of scanning all entries.

        Returns the cached KV state directly (no copy) since MLX arrays
        are immutable and safe to share.

        Args:
            tokens: Input token sequence.

        Returns:
            Tuple of (cache, remaining_tokens):
            - cache: Cached KV state if found, None otherwise
            - remaining_tokens: Tokens that still need processing
        """
        if not tokens:
            self._stats.misses += 1
            self._last_match_type = "miss"
            return None, tokens

        tokens_key = tuple(tokens)

        with self._lock:
            return self._fetch_locked(tokens, tokens_key)

    def _fetch_locked(
        self, tokens: list[int], tokens_key: tuple[int, ...]
    ) -> tuple[list[Any] | None, list[int]]:
        # --- O(1) exact match ---
        if tokens_key in self._entries:
            entry = self._entries[tokens_key]
            self._entries.move_to_end(tokens_key)
            self._stats.hits += 1
            self._stats.tokens_saved += len(tokens)
            self._last_match_type = "exact"
            # Deep copy: cache objects have mutable offset/state that
            # generation modifies in-place, corrupting the stored entry.
            cache_out = copy.deepcopy(entry.cache)
            cache_out = self._decompress_cache(cache_out)
            return cache_out, []

        # --- O(log N) prefix & supersequence match via sorted index ---
        best_match: _CacheEntry | None = None
        best_length = 0
        best_super: _CacheEntry | None = None

        sorted_keys = self._sorted_keys
        if sorted_keys:
            # Find insertion point for tokens_key in the sorted list.
            # Keys that are prefixes of tokens_key or supersequences will be
            # clustered around this position due to lexicographic ordering.
            idx = bisect.bisect_left(sorted_keys, tokens_key)

            # Scan backwards from idx to find cached keys that are PREFIXES
            # of tokens_key (shorter cached sequences).  A prefix P of T
            # satisfies P <= T lexicographically, so P is at idx-1 or earlier.
            for i in range(idx - 1, -1, -1):
                cached_key = sorted_keys[i]
                cached_len = len(cached_key)
                if cached_len >= len(tokens_key):
                    continue  # Not a prefix (same length or longer)
                # Check if cached_key is a prefix of tokens_key
                if tokens_key[:cached_len] == cached_key:
                    if cached_len > best_length:
                        best_match = self._entries[cached_key]
                        best_length = cached_len
                    # Found best prefix — shorter entries can't be longer
                    break
                # Once we go past the prefix range, stop
                if cached_key[0] != tokens_key[0]:
                    break

            # Scan forward from idx to find cached keys that are SUPERSEQUENCES
            # of tokens_key (longer cached sequences starting with tokens_key).
            for i in range(idx, len(sorted_keys)):
                cached_key = sorted_keys[i]
                cached_len = len(cached_key)
                if cached_len < len(tokens_key):
                    continue
                # Check if tokens_key is a prefix of cached_key
                if cached_key[: len(tokens_key)] == tokens_key:
                    if best_super is None or cached_len > len(best_super.tokens):
                        best_super = self._entries[cached_key]
                else:
                    # Past the supersequence range
                    break

        # --- Supersequence match handling ---
        if best_super is not None:
            n_cached = len(best_super.tokens)
            n_requested = len(tokens)
            excess = n_cached - n_requested

            has_non_trimmable = any(
                not (
                    lc.is_trimmable()
                    if hasattr(lc, "is_trimmable")
                    else hasattr(lc, "trim")
                )
                for lc in best_super.cache
            )

            if excess > 0 and has_non_trimmable:
                logger.debug(
                    "[cache_fetch] supersequence match skipped: "
                    "non-trimmable cache layers (hybrid model)"
                )
            elif excess > 0:
                trimmed_cache = _trim_cache_offset(best_super.cache, excess)
                self._entries.move_to_end(best_super.tokens)
                self._stats.hits += 1
                self._stats.tokens_saved += n_requested
                self._last_match_type = "supersequence"
                trimmed_cache = self._decompress_cache(trimmed_cache)
                return trimmed_cache, []
            else:
                self._entries.move_to_end(best_super.tokens)
                self._stats.hits += 1
                self._stats.tokens_saved += n_requested
                self._last_match_type = "supersequence"
                cache_out = copy.deepcopy(best_super.cache)
                cache_out = self._decompress_cache(cache_out)
                return cache_out, []

        # --- Prefix match ---
        if best_match is not None:
            self._entries.move_to_end(best_match.tokens)
            self._stats.hits += 1
            self._stats.tokens_saved += best_length
            remaining = tokens[best_length:]
            self._last_match_type = "prefix"
            cache_out = copy.deepcopy(best_match.cache)
            cache_out = self._decompress_cache(cache_out)
            return cache_out, remaining

        # --- LCP (Longest Common Prefix) for divergent sequences ---
        # This handles the agentic pattern: same system+context prefix
        # but different final user message.  Use the sorted index to find
        # the nearest neighbor which likely shares the longest prefix.
        best_lcp_entry: _CacheEntry | None = None
        best_lcp_length = 0

        if sorted_keys:
            idx = bisect.bisect_left(sorted_keys, tokens_key)
            # Check neighbors around insertion point (they share the most
            # common prefix due to lexicographic ordering).
            for i in (idx - 1, idx):
                if i < 0 or i >= len(sorted_keys):
                    continue
                cached_key = sorted_keys[i]
                if cached_key == tokens_key:
                    continue  # Skip exact (already handled)
                min_len = min(len(cached_key), len(tokens_key))
                if min_len <= best_lcp_length:
                    continue
                # Compute LCP length
                lcp = 0
                for j in range(min_len):
                    if cached_key[j] != tokens_key[j]:
                        break
                    lcp = j + 1
                if lcp > best_lcp_length:
                    best_lcp_entry = self._entries[cached_key]
                    best_lcp_length = lcp
                    logger.debug(
                        f"[cache_fetch] LCP scan: cached_len={len(cached_key)} "
                        f"req_len={len(tokens_key)} lcp={lcp}"
                    )

        if best_lcp_entry is not None and best_lcp_length > 0:
            excess = len(best_lcp_entry.tokens) - best_lcp_length

            has_non_trimmable = any(
                not (
                    lc.is_trimmable()
                    if hasattr(lc, "is_trimmable")
                    else hasattr(lc, "trim")
                )
                for lc in best_lcp_entry.cache
            )
            logger.debug(
                f"[cache_fetch] LCP candidate: lcp={best_lcp_length} "
                f"entry_len={len(best_lcp_entry.tokens)} excess={excess} "
                f"non_trimmable={has_non_trimmable} "
                f"cache_layers={len(best_lcp_entry.cache)} "
                f"layer_types={[type(lc).__name__ for lc in best_lcp_entry.cache[:3]]}"
            )

            if not has_non_trimmable:
                trimmed_cache = _trim_cache_offset(best_lcp_entry.cache, excess)
                self._entries.move_to_end(best_lcp_entry.tokens)
                self._stats.hits += 1
                self._stats.tokens_saved += best_lcp_length
                remaining = tokens[best_lcp_length:]
                logger.debug(
                    f"[cache_fetch] LCP hit: shared={best_lcp_length} "
                    f"trimmed={excess} remaining={len(remaining)}"
                )
                self._last_match_type = "lcp"
                trimmed_cache = self._decompress_cache(trimmed_cache)
                return trimmed_cache, remaining

        self._stats.misses += 1
        self._last_match_type = "miss"

        return None, tokens

    def store(
        self, tokens: list[int], cache: list[Any], evict_prefixes: bool = True
    ) -> bool:
        """
        Store KV cache for future reuse.

        This method stores the cache reference directly (no copy) and
        tracks memory usage. If memory limit is exceeded, LRU entries
        are evicted until there's room.

        Args:
            tokens: Token sequence that was processed.
            cache: The computed KV cache to store.
            evict_prefixes: If True, evict existing entries whose token
                sequence is a strict prefix of ``tokens``.  Set to False
                when storing prompt+output entries to preserve prompt-only
                entries created by prompt_cache_save (those are the entries
                that future requests will actually match).

        Returns:
            True if stored successfully, False if rejected.
        """
        if not tokens or not cache:
            return False

        tokens_key = tuple(tokens)

        # Fast path: already cached — bump LRU and skip expensive trim/quantize.
        # Holds the lock briefly so the bump is consistent with concurrent fetch.
        with self._lock:
            if tokens_key in self._entries:
                self._entries.move_to_end(tokens_key)
                return True

        # Trim oversized KV arrays to actual used size (pure compute, no shared
        # state — kept outside the lock so concurrent fetch isn't blocked).
        cache = _trim_to_offset(cache)

        # Compress cache for storage (TurboQuant or standard quantization)
        if (
            self._config.kv_turboquant
            and len(tokens) >= self._config.kv_min_quantize_tokens
        ):
            cache = _turboquant_compress_cache(
                cache,
                self._config.kv_turboquant_bits,
                self._config.kv_turboquant_group_size,
            )
        elif (
            self._config.kv_quantize
            and len(tokens) >= self._config.kv_min_quantize_tokens
        ):
            cache = _quantize_cache(
                cache, self._config.kv_bits, self._config.kv_group_size
            )

        # Create entry and estimate memory (pure compute, no shared state).
        entry = _CacheEntry.create(tokens, cache)

        # Check if single entry exceeds limit
        if entry.memory_bytes > self._max_memory:
            logger.warning(
                f"Cache entry too large: {entry.memory_bytes / _BYTES_PER_MB:.1f}MB "
                f"exceeds limit {self._max_memory / _BYTES_PER_MB:.1f}MB"
            )
            return False

        with self._lock:
            # Re-check exact match: a concurrent store may have inserted
            # the same key while we were trimming/compressing outside the
            # lock. Just bump LRU and bail.
            if tokens_key in self._entries:
                self._entries.move_to_end(tokens_key)
                return True

            # Prefix-subset eviction: remove entries whose token sequence
            # is a strict prefix of the new entry.  Uses sorted index for
            # O(log N + K) lookup instead of O(N) scan.
            if evict_prefixes and self._sorted_keys:
                to_remove = []
                idx = bisect.bisect_left(self._sorted_keys, tokens_key)
                # Scan backwards — prefixes of tokens_key are immediately before idx
                for i in range(idx - 1, -1, -1):
                    key = self._sorted_keys[i]
                    klen = len(key)
                    if klen >= len(tokens_key):
                        continue
                    if tokens_key[:klen] == key:
                        to_remove.append(key)
                    elif key[0] != tokens_key[0]:
                        break
                for key in to_remove:
                    # Remove from sorted index FIRST so a concurrent fetch
                    # never sees a key in the index that's missing from
                    # _entries (was the source of issue #163's KeyError
                    # under the higher store() rate from PR #165).
                    self._remove_from_sorted(key)
                    old = self._entries.pop(key)
                    self._current_memory -= old.memory_bytes
                    self._stats.evictions += 1
                    logger.debug(
                        f"[prefix_evict] removed {len(key)} tokens, "
                        f"freed {old.memory_bytes / _BYTES_PER_MB:.2f}MB, "
                        f"new_entry={len(tokens_key)} tokens"
                    )
                if to_remove:
                    self._stats.entry_count = len(self._entries)
                    self._stats.current_memory_bytes = self._current_memory

            # Evict until we have room
            while (
                self._current_memory + entry.memory_bytes > self._max_memory
                or len(self._entries) >= self._config.max_entries
            ) and self._entries:
                self._evict_lru()

            # Store entry. Insert into _entries before _sorted_keys so
            # that even if a future change drops the lock, fetch never
            # observes a key in sorted_keys that's missing from entries.
            self._entries[tokens_key] = entry
            self._current_memory += entry.memory_bytes
            bisect.insort(self._sorted_keys, tokens_key)
            self._stats.entry_count = len(self._entries)
            self._stats.current_memory_bytes = self._current_memory

        logger.debug(
            f"Stored cache: {len(tokens)} tokens, "
            f"{entry.memory_bytes / _BYTES_PER_MB:.2f}MB, "
            f"total={self._current_memory / _BYTES_PER_MB:.1f}MB"
        )

        return True

    def _remove_from_sorted(self, key: tuple[int, ...]) -> None:
        """Remove a key from the sorted index using bisect for O(log N)."""
        idx = bisect.bisect_left(self._sorted_keys, key)
        if idx < len(self._sorted_keys) and self._sorted_keys[idx] == key:
            self._sorted_keys.pop(idx)

    def _evict_lru(self) -> None:
        """Evict the least recently used entry.

        Caller must hold ``self._lock``.
        """
        if not self._entries:
            return

        # Peek the oldest key, drop sorted-index entry first so a fetch
        # without the lock can't trip the orphaned-sorted-key KeyError.
        tokens_key = next(iter(self._entries))
        self._remove_from_sorted(tokens_key)
        entry = self._entries.pop(tokens_key)
        self._current_memory -= entry.memory_bytes
        self._stats.evictions += 1
        self._stats.entry_count = len(self._entries)
        self._stats.current_memory_bytes = self._current_memory

        logger.debug(
            f"[lru_evict] removed {len(tokens_key)} tokens, "
            f"freed {entry.memory_bytes / _BYTES_PER_MB:.2f}MB"
        )

    def remove(self, tokens: list[int]) -> bool:
        """
        Remove a specific cache entry.

        Args:
            tokens: Token sequence to remove.

        Returns:
            True if entry was found and removed.
        """
        tokens_key = tuple(tokens)
        with self._lock:
            if tokens_key not in self._entries:
                return False
            self._remove_from_sorted(tokens_key)
            entry = self._entries.pop(tokens_key)
            self._current_memory -= entry.memory_bytes
            self._stats.entry_count = len(self._entries)
            self._stats.current_memory_bytes = self._current_memory
        return True

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._entries.clear()
            self._sorted_keys.clear()
            self._current_memory = 0
            self._stats = CacheStats(max_memory_bytes=self._max_memory)
        logger.debug("Cache cleared")

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return self._stats.to_dict()

    def reset_stats(self) -> None:
        """Reset statistics while preserving cache contents."""
        self._stats = CacheStats(
            max_memory_bytes=self._max_memory,
            current_memory_bytes=self._current_memory,
            entry_count=len(self._entries),
        )

    @property
    def memory_usage_mb(self) -> float:
        """Current memory usage in MB."""
        return self._current_memory / _BYTES_PER_MB

    @property
    def memory_limit_mb(self) -> float:
        """Memory limit in MB."""
        return self._max_memory / _BYTES_PER_MB

    def __len__(self) -> int:
        """Return number of cached entries."""
        return len(self._entries)

    def __contains__(self, tokens: list[int]) -> bool:
        """Check if tokens are cached."""
        return tuple(tokens) in self._entries

    # -----------------------------------------------------------------
    # Disk persistence — survives server restarts
    # -----------------------------------------------------------------

    def save_to_disk(self, cache_dir: str) -> bool:
        """Save all cache entries to disk using mlx_lm's safetensors format.

        The snapshot is committed via a directory-rename to make it
        all-or-nothing: writes go to ``<cache_dir>.new/``, then a
        three-step swap (``cache_dir → .old``, ``.new → cache_dir``,
        ``rm .old``) atomically replaces the previous snapshot. A crash
        anywhere during the writes leaves the previous snapshot intact;
        :meth:`load_from_disk` recovers from a crash mid-swap.

        Directory layout (committed)::

            cache_dir/
              index.json          # token keys + metadata per entry
              entry_0.safetensors # KV arrays for entry 0
              entry_0_tokens.bin
              entry_1.safetensors
              entry_1_tokens.bin
              ...

        Returns True if at least one entry was saved.
        """
        import shutil
        import time as _time

        if not self._entries:
            logger.info("[cache_persist] nothing to save (0 entries)")
            return False

        t0 = _time.monotonic()

        try:
            from mlx_lm.models.cache import save_prompt_cache
        except ImportError:
            logger.warning("[cache_persist] mlx_lm not available, cannot save")
            return False

        # Strip trailing separators so ``<cache_dir>.new`` is a sibling of
        # cache_dir, not a child. A child path silently breaks the swap.
        cache_dir = cache_dir.rstrip(os.sep)
        new_dir = cache_dir + ".new"
        old_dir = cache_dir + ".old"

        # Pre-clean stale staging dirs from a previous interrupted save.
        for stale in (new_dir, old_dir):
            if os.path.exists(stale):
                logger.info(f"[cache_persist] removing stale staging dir: {stale}")
                shutil.rmtree(stale, ignore_errors=True)

        os.makedirs(new_dir, exist_ok=True)

        # Single source of truth for per-entry on-disk filenames. Used
        # by both the save loop and the post-loop "did the files
        # actually survive?" filter — keep them in lockstep so a future
        # rename only has one place to edit.
        def _entry_paths(idx: int) -> tuple[str, str]:
            return (
                os.path.join(new_dir, f"entry_{idx}.safetensors"),
                os.path.join(new_dir, f"entry_{idx}_tokens.bin"),
            )

        index = {
            "version": 2,
            "num_entries": len(self._entries),
            "total_memory_bytes": self._current_memory,
            "entries": [],
        }

        saved = 0
        for i, (tokens_key, entry) in enumerate(self._entries.items()):
            entry_path, tokens_path = _entry_paths(i)
            try:
                save_prompt_cache(
                    entry_path,
                    entry.cache,
                    metadata={"num_tokens": str(len(tokens_key))},
                )
                # Save tokens separately (can be 100K+ ints → binary is smaller).
                import array as _array

                arr = _array.array("i", tokens_key)  # 32-bit signed ints
                with open(tokens_path, "wb") as f:
                    arr.tofile(f)

                # Record the per-layer cache class names so loaders can
                # gate on cache-type compatibility (#198 BUG B). Stored
                # inline in index.json so we can pre-filter incompatible
                # entries without parsing the safetensors header for each.
                # Falls back to ``[]`` if the entry was empty (defensive —
                # shouldn't happen since we reject empty caches at store).
                cache_types = [
                    type(layer).__name__ for layer in entry.cache if layer is not None
                ]

                index["entries"].append(
                    {
                        "index": i,
                        "num_tokens": len(tokens_key),
                        "memory_bytes": entry.memory_bytes,
                        "cache_types": cache_types,
                    }
                )
                saved += 1
                logger.info(
                    f"[cache_persist] saved entry {i}: "
                    f"{len(tokens_key)} tokens, "
                    f"{entry.memory_bytes / _BYTES_PER_MB:.1f}MB KV, "
                    f"file={entry_path}"
                )
            except Exception as e:
                logger.warning(f"[cache_persist] failed to save entry {i}: {e}")

        if saved == 0:
            shutil.rmtree(new_dir, ignore_errors=True)
            logger.warning("[cache_persist] no entries saved successfully, aborting")
            return False

        # Filter index to entries whose files actually survived to disk.
        # Defends against the staging dir being clobbered mid-save by an
        # external process (e.g. macOS Spotlight, purgeable-cache cleanup
        # under disk pressure) — observed in the wild during long
        # multi-GB shutdown saves where 14GB+ of cache was being written.
        # Without this filter, index.json may reference entry files that
        # no longer exist, and the open() below raises FileNotFoundError
        # because new_dir itself is gone.
        def _both_exist(e: dict) -> bool:
            sf, tk = _entry_paths(e["index"])
            return os.path.exists(sf) and os.path.exists(tk)

        verified = [e for e in index["entries"] if _both_exist(e)]
        if not verified:
            shutil.rmtree(new_dir, ignore_errors=True)
            logger.warning(
                "[cache_persist] staging dir vanished mid-save, no entries survived "
                f"(saved {saved}/{len(self._entries)} but 0 files remain on disk)"
            )
            return False
        if len(verified) < len(index["entries"]):
            logger.warning(
                f"[cache_persist] {len(index['entries']) - len(verified)} of "
                f"{len(index['entries'])} entry files vanished mid-save, "
                f"persisting {len(verified)} that survived"
            )
            index["entries"] = verified
            index["num_entries"] = len(verified)

        # Defensively recreate new_dir before the index.json write — the
        # filter above proves at least one entry's files exist, so the
        # dir must too, but a stat-cache delay or NFS-style coherence
        # window could still trip the open() below. Cheap insurance.
        os.makedirs(new_dir, exist_ok=True)

        # TOCTOU re-check: between the filter above and the index.json
        # write below, the same external process could clobber new_dir
        # again. If that happens, makedirs recreates an EMPTY dir, and
        # we'd commit an index.json pointing to entry files that no
        # longer exist (load_from_disk's _has_valid_index() would then
        # reject the snapshot — recoverable, but a wasted swap). Verify
        # the first entry still exists right before we write; if not,
        # abort cleanly.
        first_sf, first_tk = _entry_paths(index["entries"][0]["index"])
        if not (os.path.exists(first_sf) and os.path.exists(first_tk)):
            shutil.rmtree(new_dir, ignore_errors=True)
            logger.warning(
                "[cache_persist] staging dir vanished after filter — entry "
                "files gone before index.json could be written, aborting"
            )
            return False

        # Write index.json LAST inside the staging dir. Its presence is the
        # signal to load_from_disk that .new contains a complete snapshot.
        # Catch FileNotFoundError as a final guard against the recheck
        # above missing the dir-loss window — the file or dir could still
        # vanish in the microseconds between the recheck and the open().
        index_path = os.path.join(new_dir, "index.json")
        try:
            with open(index_path, "w") as f:
                json.dump(index, f, indent=2)
        except FileNotFoundError:
            shutil.rmtree(new_dir, ignore_errors=True)
            logger.warning(
                "[cache_persist] staging dir vanished while writing index.json, "
                "aborting"
            )
            return False

        # Atomic-ish directory swap. If we crash between the two renames,
        # load_from_disk's recovery path (see below) handles it.
        if os.path.exists(cache_dir):
            os.rename(cache_dir, old_dir)
        os.rename(new_dir, cache_dir)
        if os.path.exists(old_dir):
            shutil.rmtree(old_dir, ignore_errors=True)

        dt = _time.monotonic() - t0
        logger.info(
            f"[cache_persist] SAVED {saved}/{len(self._entries)} entries "
            f"to {cache_dir} in {dt:.1f}s "
            f"({self._current_memory / _BYTES_PER_MB:.0f}MB total)"
        )
        return saved > 0

    def load_from_disk(self, cache_dir: str) -> int:
        """Load cache entries from disk.

        Recovers from a save interrupted between the two directory
        renames in :meth:`save_to_disk`:

        * if ``cache_dir`` is missing but ``cache_dir.new/index.json``
          exists, the snapshot was fully written but never swapped in
          → promote ``.new`` to ``cache_dir``;
        * else if ``cache_dir.old`` is present and ``cache_dir`` is
          missing, restore ``.old``.

        Each entry is validated before insertion: the on-disk
        ``tokens.bin`` size must match ``num_tokens * 4``, the
        ``.safetensors`` file size must cover the data range declared
        in its header (``mx.load`` mmaps lazily and returns zeros past
        EOF, so a body-truncated KV would otherwise slip through), and
        ``cache.offset`` must equal ``len(tokens)``. Any entry that
        fails validation is dropped with a warning.

        Returns the number of entries successfully loaded.
        """
        import shutil
        import time as _time

        # Strip trailing separators (see save_to_disk for rationale).
        cache_dir = cache_dir.rstrip(os.sep)
        new_dir = cache_dir + ".new"
        old_dir = cache_dir + ".old"

        def _has_valid_index(d: str) -> bool:
            """Cheap sanity check: index.json exists, is valid JSON, has the
            expected version, AND at least one referenced entry file exists
            on disk. The last check defends against the pathological case
            where index.json survives but its entry files don't (manual
            deletion, fs corruption, partial restore from backup) — without
            it, recovery would promote a "valid index, no data" snapshot
            and discard the previous good `.old` snapshot for nothing."""
            p = os.path.join(d, "index.json")
            if not os.path.exists(p):
                return False
            try:
                with open(p) as f:
                    obj = json.load(f)
            except (OSError, ValueError):
                return False
            if not (isinstance(obj, dict) and obj.get("version", 0) >= 2):
                return False
            entries = obj.get("entries") or []
            if not entries:
                # An index claiming zero entries is degenerate; nothing to
                # promote. Treat as missing so recovery can fall through
                # to a real snapshot in the other staging dir.
                return False
            first_idx = entries[0].get("index")
            if first_idx is None:
                return False
            sf = os.path.join(d, f"entry_{first_idx}.safetensors")
            tk = os.path.join(d, f"entry_{first_idx}_tokens.bin")
            return os.path.exists(sf) and os.path.exists(tk)

        # Crash-recovery for an interrupted save_to_disk.
        if not os.path.exists(cache_dir):
            if _has_valid_index(new_dir):
                logger.info(
                    f"[cache_persist] recovering interrupted save: "
                    f"promoting {new_dir} → {cache_dir}"
                )
                os.rename(new_dir, cache_dir)
                if os.path.exists(old_dir):
                    shutil.rmtree(old_dir, ignore_errors=True)
            elif _has_valid_index(old_dir):
                logger.info(
                    f"[cache_persist] recovering interrupted save: "
                    f"restoring {old_dir} → {cache_dir}"
                )
                os.rename(old_dir, cache_dir)
                if os.path.exists(new_dir):
                    shutil.rmtree(new_dir, ignore_errors=True)
        else:
            # cache_dir exists — clean up any orphan staging dirs that a
            # previous interrupted save may have left behind.
            for stale in (new_dir, old_dir):
                if os.path.exists(stale):
                    logger.info(f"[cache_persist] cleaning orphan staging dir: {stale}")
                    shutil.rmtree(stale, ignore_errors=True)

        index_path = os.path.join(cache_dir, "index.json")
        if not os.path.exists(index_path):
            logger.info(f"[cache_persist] no index at {index_path}, nothing to load")
            return 0

        t0 = _time.monotonic()

        try:
            from mlx_lm.models.cache import load_prompt_cache
        except ImportError:
            logger.warning("[cache_persist] mlx_lm not available, cannot load")
            return 0

        with open(index_path) as f:
            index = json.load(f)

        version = index.get("version", 1)
        if version < 2:
            logger.warning(f"[cache_persist] unsupported version {version}, skipping")
            return 0

        loaded = 0
        corrupt_skipped = 0
        duplicate_skipped = 0
        incompatible_skipped = 0
        for entry_meta in index.get("entries", []):
            i = entry_meta["index"]
            expected_num_tokens = entry_meta["num_tokens"]
            entry_path = os.path.join(cache_dir, f"entry_{i}.safetensors")
            tokens_path = os.path.join(cache_dir, f"entry_{i}_tokens.bin")

            if not os.path.exists(entry_path) or not os.path.exists(tokens_path):
                logger.warning(f"[cache_persist] missing files for entry {i}, skipping")
                corrupt_skipped += 1
                continue

            # Cache-type compatibility check (#198 BUG B). Reject entries
            # whose persisted cache class doesn't match what the current
            # config can dequantize at fetch time — otherwise tuple-form
            # keys reach the scheduler. Done early to skip the safetensors
            # body validation work for entries we'd discard anyway.
            cache_types = entry_meta.get("cache_types") or []
            if not cache_types:
                # Backward compat with index.json from before cache_types
                # existed: peek at safetensors __metadata__.
                cache_types = _safetensors_cache_classes(entry_path)
            ok, reason = _cache_classes_compatible(cache_types, self._config)
            if not ok:
                logger.info(
                    f"[cache_persist] entry {i} skipped — {reason}; "
                    f"persisted types={cache_types}"
                )
                incompatible_skipped += 1
                continue

            # Cross-check tokens.bin size against index.json's claim.
            # Mismatch means the entry was partially rewritten by an
            # interrupted previous save (BUG A). Drop it.
            expected_bytes = expected_num_tokens * 4  # 32-bit signed ints
            actual_bytes = os.path.getsize(tokens_path)
            if actual_bytes != expected_bytes:
                logger.warning(
                    f"[cache_persist] entry {i} tokens.bin size mismatch "
                    f"(expected {expected_bytes} bytes for {expected_num_tokens} "
                    f"tokens, got {actual_bytes}) — corruption, skipping"
                )
                corrupt_skipped += 1
                continue

            # mx.load mmaps safetensors lazily and will silently return
            # zeros for positions past EOF. Verify the body is fully on
            # disk via the safetensors header before trusting the entry
            # (BUG D — body-truncated file slips through load otherwise).
            if not _safetensors_is_complete(entry_path):
                logger.warning(
                    f"[cache_persist] entry {i} safetensors body is short of "
                    f"its header's declared data range — corruption, skipping"
                )
                corrupt_skipped += 1
                continue

            try:
                # Load tokens from binary
                import array as _array

                arr = _array.array("i")
                with open(tokens_path, "rb") as f:
                    arr.fromfile(f, expected_num_tokens)
                tokens = list(arr)

                # Skip duplicates (e.g. an entry that warmup already
                # populated). Done BEFORE load_prompt_cache so a duplicate
                # entry doesn't pay the safetensors mmap cost only to be
                # discarded. Without this guard, bisect.insort would also
                # create duplicate keys in _sorted_keys and memory would
                # double-count. Benign — not a corruption signal.
                tokens_key = tuple(tokens)
                if tokens_key in self._entries:
                    logger.debug(
                        f"[cache_persist] entry {i} already present in cache "
                        f"(len={len(tokens)}), skipping disk copy"
                    )
                    duplicate_skipped += 1
                    continue

                # Load KV cache (header completeness already validated above).
                cache = load_prompt_cache(entry_path)

                # Invariant: a well-formed entry has cache.offset == len(tokens).
                # Any deviation means BUG A poisoning slipped through earlier
                # checks; drop it rather than risk corrupting fetch output.
                if cache:
                    head_offset = getattr(cache[0], "offset", None)
                    if head_offset is not None and head_offset != len(tokens):
                        logger.warning(
                            f"[cache_persist] entry {i} cache offset "
                            f"({head_offset}) != tokens length ({len(tokens)}) "
                            f"— corruption, skipping"
                        )
                        corrupt_skipped += 1
                        continue

                # Estimate memory
                memory = estimate_kv_cache_memory(cache)

                # Check if it fits
                if self._current_memory + memory > self._max_memory:
                    logger.info(
                        f"[cache_persist] entry {i} would exceed memory limit "
                        f"({(self._current_memory + memory) / _BYTES_PER_MB:.0f}MB > "
                        f"{self._max_memory / _BYTES_PER_MB:.0f}MB), stopping load"
                    )
                    break

                entry = _CacheEntry(
                    tokens=tokens_key,
                    cache=cache,
                    memory_bytes=memory,
                )
                self._entries[tokens_key] = entry
                self._current_memory += memory
                bisect.insort(self._sorted_keys, tokens_key)
                loaded += 1

                logger.info(
                    f"[cache_persist] loaded entry {i}: "
                    f"{len(tokens)} tokens, "
                    f"{memory / _BYTES_PER_MB:.1f}MB KV"
                )

            except Exception as e:
                logger.warning(f"[cache_persist] failed to load entry {i}: {e}")
                corrupt_skipped += 1

        self._stats.entry_count = len(self._entries)
        self._stats.current_memory_bytes = self._current_memory

        dt = _time.monotonic() - t0
        summary = (
            f"[cache_persist] LOADED {loaded} entries from {cache_dir} "
            f"in {dt:.1f}s ({self._current_memory / _BYTES_PER_MB:.0f}MB total)"
        )
        if duplicate_skipped:
            summary += (
                f", {duplicate_skipped} skipped as duplicates of in-memory entries"
            )
        if incompatible_skipped:
            summary += (
                f", {incompatible_skipped} skipped as incompatible with "
                f"current cache config (e.g. config changed between runs)"
            )
        if corrupt_skipped:
            logger.warning(
                f"{summary}, SKIPPED {corrupt_skipped} corrupt entries — "
                f"disk cache may need cleanup"
            )
        else:
            logger.info(summary)
        return loaded
