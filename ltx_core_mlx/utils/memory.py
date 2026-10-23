"""Metal memory management utilities."""

import gc

import mlx.core as mx


def aggressive_cleanup() -> None:
    """Force garbage collection and clear the MLX Metal cache.

    Must be called between every pipeline stage to prevent unbounded
    Metal memory growth.
    """
    gc.collect()
    mx.clear_cache()


def get_memory_stats() -> dict[str, float]:
    """Return current Metal memory usage in GB.

    Returns:
        Dict with keys: active_gb, peak_gb, cache_gb.
    """
    to_gb = 1 / (1024**3)
    return {
        "active_gb": mx.get_active_memory() * to_gb,
        "peak_gb": mx.get_peak_memory() * to_gb,
        "cache_gb": mx.get_cache_memory() * to_gb,
    }
