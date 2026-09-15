# SPDX-License-Identifier: Apache-2.0
"""
Vision Embedding Cache for MLLM continuous batching.

This module provides caching for vision embeddings to avoid redundant
computation when the same images are processed multiple times.

Cache Levels:
1. Pixel Values Cache - Caches processed image tensors (prepare_inputs output)
2. Vision Encoding Cache - Caches VLM forward pass output (logits + cache state)

Performance Impact:
- Without cache: ~2s per image for vision encoding
- With cache hit: ~0.01s (100x speedup for repeated images)
"""

import hashlib
import logging
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx

logger = logging.getLogger(__name__)


@dataclass
class VisionCacheStats:
    """Statistics for vision cache performance."""

    pixel_cache_hits: int = 0
    pixel_cache_misses: int = 0
    encoding_cache_hits: int = 0
    encoding_cache_misses: int = 0
    total_time_saved: float = 0.0
    total_images_processed: int = 0

    @property
    def pixel_hit_rate(self) -> float:
        total = self.pixel_cache_hits + self.pixel_cache_misses
        return self.pixel_cache_hits / total if total > 0 else 0.0

    @property
    def encoding_hit_rate(self) -> float:
        total = self.encoding_cache_hits + self.encoding_cache_misses
        return self.encoding_cache_hits / total if total > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "pixel_cache_hits": self.pixel_cache_hits,
            "pixel_cache_misses": self.pixel_cache_misses,
            "pixel_hit_rate": self.pixel_hit_rate,
            "encoding_cache_hits": self.encoding_cache_hits,
            "encoding_cache_misses": self.encoding_cache_misses,
            "encoding_hit_rate": self.encoding_hit_rate,
            "total_time_saved": self.total_time_saved,
            "total_images_processed": self.total_images_processed,
        }


@dataclass
class PixelCacheEntry:
    """Cached pixel values from prepare_inputs."""

    pixel_values: mx.array
    input_ids: mx.array
    attention_mask: mx.array | None
    image_grid_thw: mx.array | None
    extra_kwargs: dict[str, Any]
    processing_time: float = 0.0


@dataclass
class PixelOnlyCacheEntry:
    """Cached pixel values only (prompt-independent).

    This cache stores only the image-dependent data that doesn't
    change with different prompts. Useful when the same images
    are used with different prompts.
    """

    pixel_values: mx.array
    image_grid_thw: mx.array | None
    processing_time: float = 0.0


@dataclass
class EncodingCacheEntry:
    """Cached vision encoding output."""

    logits: mx.array
    first_token: int
    logprobs: mx.array
    encoding_time: float = 0.0


def compute_image_hash(image_path: str) -> str:
    """
    Compute hash of image content.

    For files: hash the actual content
    For URLs/base64: hash the string
    """
    try:
        path = Path(image_path)
        if path.exists() and path.is_file():
            # Hash full file content to avoid collisions between images
            # that share a prefix.
            content = path.read_bytes()
            return hashlib.sha256(content).hexdigest()[:16]
        else:
            # Hash the string (URL or base64)
            return hashlib.sha256(image_path.encode()).hexdigest()[:16]
    except Exception:
        return hashlib.sha256(str(image_path).encode()).hexdigest()[:16]


def compute_images_hash(images: list[str]) -> str:
    """Compute combined hash for multiple images.

    Order is preserved: [img_a, img_b] and [img_b, img_a] produce
    different hashes because image order affects vision embeddings
    and KV state.
    """
    if not images:
        return "no_images"
    hashes = [compute_image_hash(img) for img in images]
    return hashlib.sha256("_".join(hashes).encode()).hexdigest()[:16]


class VisionEmbeddingCache:
    """
    Two-level cache for vision processing in MLLM.

    Level 1 (Pixel Cache):
        - Caches output of prepare_inputs() (pixel_values, input_ids, etc.)
        - Key: hash(images) + hash(prompt)
        - Saves: Image loading, resizing, normalization time (~0.5-1s)

    Level 2 (Encoding Cache):
        - Caches output of VLM forward pass (logits, first token)
        - Key: hash(images) + hash(prompt)
        - Saves: Vision encoder computation time (~1-2s)

    Example:
        >>> cache = VisionEmbeddingCache(max_pixel_entries=50, max_encoding_entries=20)
        >>>
        >>> # First request - cache miss
        >>> pixel_entry = cache.get_pixel_cache(images, prompt)
        >>> if pixel_entry is None:
        ...     # Process images...
        ...     cache.set_pixel_cache(images, prompt, pixel_values, ...)
        >>>
        >>> # Second request with same image - cache hit!
        >>> pixel_entry = cache.get_pixel_cache(images, prompt)  # Returns cached data
    """

    def __init__(
        self,
        max_pixel_entries: int = 100,
        max_encoding_entries: int = 50,
        enabled: bool = True,
    ):
        """
        Initialize the vision embedding cache.

        Args:
            max_pixel_entries: Max entries in pixel cache (LRU eviction)
            max_encoding_entries: Max entries in encoding cache
            enabled: Whether caching is enabled
        """
        self.max_pixel_entries = max_pixel_entries
        self.max_encoding_entries = max_encoding_entries
        self.enabled = enabled

        # LRU caches using OrderedDict
        self._pixel_cache: OrderedDict[str, PixelCacheEntry] = OrderedDict()
        self._pixel_only_cache: OrderedDict[str, PixelOnlyCacheEntry] = OrderedDict()
        self._encoding_cache: OrderedDict[str, EncodingCacheEntry] = OrderedDict()

        self.stats = VisionCacheStats()

    def _make_key(self, images: list[str], prompt: str) -> str:
        """Create cache key from images and prompt."""
        img_hash = compute_images_hash(images)
        # Use shorter prompt hash for cache key
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:12]
        return f"{img_hash}_{prompt_hash}"

    def _make_image_only_key(self, images: list[str]) -> str:
        """Create cache key from images only (prompt-independent)."""
        return compute_images_hash(images)

    # ========== Pixel Cache ==========

    def get_pixel_cache(
        self,
        images: list[str],
        prompt: str,
    ) -> PixelCacheEntry | None:
        """
        Get cached pixel values for images+prompt.

        Returns:
            PixelCacheEntry if found, None otherwise
        """
        if not self.enabled or not images:
            return None

        key = self._make_key(images, prompt)

        if key in self._pixel_cache:
            # Move to end (most recently used)
            entry = self._pixel_cache.pop(key)
            self._pixel_cache[key] = entry

            self.stats.pixel_cache_hits += 1
            self.stats.total_time_saved += entry.processing_time
            logger.debug(
                f"Pixel cache hit: {key[:20]}... (saved {entry.processing_time:.2f}s)"
            )
            return entry

        self.stats.pixel_cache_misses += 1
        return None

    def set_pixel_cache(
        self,
        images: list[str],
        prompt: str,
        pixel_values: mx.array,
        input_ids: mx.array,
        attention_mask: mx.array | None = None,
        image_grid_thw: mx.array | None = None,
        extra_kwargs: dict[str, Any] | None = None,
        processing_time: float = 0.0,
    ) -> None:
        """Store pixel values in cache."""
        if not self.enabled or not images:
            return

        key = self._make_key(images, prompt)

        # Evict oldest if at capacity
        while len(self._pixel_cache) >= self.max_pixel_entries:
            oldest_key = next(iter(self._pixel_cache))
            del self._pixel_cache[oldest_key]
            logger.debug(f"Pixel cache evicted: {oldest_key[:20]}...")

        entry = PixelCacheEntry(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_grid_thw=image_grid_thw,
            extra_kwargs=extra_kwargs or {},
            processing_time=processing_time,
        )
        self._pixel_cache[key] = entry
        self.stats.total_images_processed += len(images)
        logger.debug(f"Pixel cache stored: {key[:20]}...")

    # ========== Pixel-Only Cache (prompt-independent) ==========

    def get_pixel_values(
        self,
        images: list[str],
    ) -> PixelOnlyCacheEntry | None:
        """
        Get cached pixel values for images (prompt-independent).

        This is useful when the same images are used with different prompts.
        Only the pixel_values and image_grid_thw are cached (no input_ids).

        Returns:
            PixelOnlyCacheEntry if found, None otherwise
        """
        if not self.enabled or not images:
            return None

        key = self._make_image_only_key(images)

        if key in self._pixel_only_cache:
            # Move to end (most recently used)
            entry = self._pixel_only_cache.pop(key)
            self._pixel_only_cache[key] = entry

            self.stats.pixel_cache_hits += 1
            self.stats.total_time_saved += entry.processing_time
            logger.debug(
                f"Pixel-only cache hit: {key[:16]}... (saved {entry.processing_time:.2f}s)"
            )
            return entry

        self.stats.pixel_cache_misses += 1
        return None

    def set_pixel_values(
        self,
        images: list[str],
        pixel_values: mx.array,
        image_grid_thw: mx.array | None = None,
        processing_time: float = 0.0,
    ) -> None:
        """Store pixel values in cache (prompt-independent)."""
        if not self.enabled or not images:
            return

        key = self._make_image_only_key(images)

        # Evict oldest if at capacity
        while len(self._pixel_only_cache) >= self.max_pixel_entries:
            oldest_key = next(iter(self._pixel_only_cache))
            del self._pixel_only_cache[oldest_key]
            logger.debug(f"Pixel-only cache evicted: {oldest_key[:16]}...")

        entry = PixelOnlyCacheEntry(
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            processing_time=processing_time,
        )
        self._pixel_only_cache[key] = entry
        logger.debug(f"Pixel-only cache stored: {key[:16]}...")

    # ========== Encoding Cache ==========

    def get_encoding_cache(
        self,
        images: list[str],
        prompt: str,
    ) -> EncodingCacheEntry | None:
        """
        Get cached vision encoding output.

        Returns:
            EncodingCacheEntry if found, None otherwise
        """
        if not self.enabled or not images:
            return None

        key = self._make_key(images, prompt)

        if key in self._encoding_cache:
            entry = self._encoding_cache.pop(key)
            self._encoding_cache[key] = entry

            self.stats.encoding_cache_hits += 1
            self.stats.total_time_saved += entry.encoding_time
            logger.debug(
                f"Encoding cache hit: {key[:20]}... (saved {entry.encoding_time:.2f}s)"
            )
            return entry

        self.stats.encoding_cache_misses += 1
        return None

    def set_encoding_cache(
        self,
        images: list[str],
        prompt: str,
        logits: mx.array,
        first_token: int,
        logprobs: mx.array,
        encoding_time: float = 0.0,
    ) -> None:
        """Store vision encoding output in cache."""
        if not self.enabled or not images:
            return

        key = self._make_key(images, prompt)

        # Evict oldest if at capacity
        while len(self._encoding_cache) >= self.max_encoding_entries:
            oldest_key = next(iter(self._encoding_cache))
            del self._encoding_cache[oldest_key]
            logger.debug(f"Encoding cache evicted: {oldest_key[:20]}...")

        entry = EncodingCacheEntry(
            logits=logits,
            first_token=first_token,
            logprobs=logprobs,
            encoding_time=encoding_time,
        )
        self._encoding_cache[key] = entry
        logger.debug(f"Encoding cache stored: {key[:20]}...")

    # ========== Utilities ==========

    def get_stats(self) -> dict:
        """Get cache statistics."""
        stats = self.stats.to_dict()
        stats["pixel_cache_size"] = len(self._pixel_cache)
        stats["pixel_only_cache_size"] = len(self._pixel_only_cache)
        stats["encoding_cache_size"] = len(self._encoding_cache)
        return stats

    def clear(self) -> None:
        """Clear all caches and reset stats."""
        self._pixel_cache.clear()
        self._pixel_only_cache.clear()
        self._encoding_cache.clear()
        self.stats = VisionCacheStats()

    def __repr__(self) -> str:
        return (
            f"<VisionEmbeddingCache "
            f"pixel={len(self._pixel_cache)}/{self.max_pixel_entries} "
            f"pixel_only={len(self._pixel_only_cache)}/{self.max_pixel_entries} "
            f"encoding={len(self._encoding_cache)}/{self.max_encoding_entries}>"
        )
