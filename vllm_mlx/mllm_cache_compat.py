# SPDX-License-Identifier: Apache-2.0
"""Cache compatibility helpers for MLLM continuous batching."""

from collections.abc import Iterable
from typing import Any


def first_incompatible_mllm_cache_type(
    caches: Iterable[Any], *, allow_arrays_cache: bool = False
) -> str | None:
    """Return the first cache type that MLLM batching cannot merge.

    mlx-vlm 0.6.4 split its cache classes from mlx-lm's parallel classes.
    Models loaded by mlx-vlm therefore return native ``KVCache`` /
    ``RotatingKVCache`` instances that fail an mlx-lm-only ``isinstance``
    check despite exposing the supported batching API. Accept both namespaces.

    Sparse-attention backbones can wrap independently mergeable cache leaves in
    mlx-vlm's ``CacheList``. Validate those leaves recursively instead of
    treating the wrapper itself as an unknown cache. ``PoolingCache`` is a
    supported leaf because mlx-vlm owns its batch merge/filter/extract
    lifecycle. ``ArraysCache`` remains restricted to the explicitly serialized
    hybrid compatibility lane. Mamba and quantized/unknown caches remain
    fail-closed.
    """
    from mlx_lm.models.cache import ArraysCache, KVCache, RotatingKVCache

    supported_types: tuple[type, ...] = (KVCache, RotatingKVCache)
    compound_type: type | None = None
    if allow_arrays_cache:
        supported_types += (ArraysCache,)
    try:
        from mlx_vlm.models import cache as vlm_cache
    except ImportError:
        # mlx-vlm is optional. Text-only installations still import the engine,
        # although they never enter the MLLM serving path.
        pass
    else:
        supported_types += (vlm_cache.KVCache, vlm_cache.RotatingKVCache)
        compound_type = getattr(vlm_cache, "CacheList", None)
        pooling_type = getattr(vlm_cache, "PoolingCache", None)
        if isinstance(pooling_type, type):
            supported_types += (pooling_type,)
        # mlx-vlm owns a distinct ArraysCache class as well. Hybrid VLM
        # backbones return this native type, so the serialized compatibility
        # lane must accept it for the same reason it accepts mlx-lm's class.
        if allow_arrays_cache and hasattr(vlm_cache, "ArraysCache"):
            supported_types += (vlm_cache.ArraysCache,)

    for cache in caches:
        if compound_type is not None and isinstance(cache, compound_type):
            incompatible = first_incompatible_mllm_cache_type(
                cache.caches, allow_arrays_cache=allow_arrays_cache
            )
            if incompatible is not None:
                return incompatible
            continue
        if not isinstance(cache, supported_types):
            return type(cache).__name__
    return None
