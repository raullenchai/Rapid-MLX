# SPDX-License-Identifier: Apache-2.0
"""Cache compatibility helpers for MLLM continuous batching."""

from collections.abc import Iterable
from typing import Any, cast


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
    from mlx_lm.models import cache as lm_cache

    from .models.mlx_vlm_vendored import cache as vendored_cache

    supported_types: tuple[type, ...] = (
        lm_cache.KVCache,
        lm_cache.RotatingKVCache,
        vendored_cache.KVCache,
        vendored_cache.RotatingKVCache,
    )
    # mlx-lm, mlx-vlm, and the vendored copy each execute their own cache
    # module, so the structurally identical wrappers are distinct class
    # objects. Keep all three namespaces symmetric and recurse into every
    # qualified CacheList rather than rejecting mlx-lm's wrapper by name.
    compound_types: tuple[type, ...] = (lm_cache.CacheList,)
    vendored_compound = getattr(vendored_cache, "CacheList", None)
    if isinstance(vendored_compound, type):
        compound_types += (vendored_compound,)
    pooling_type = getattr(vendored_cache, "PoolingCache", None)
    if isinstance(pooling_type, type):
        supported_types += (pooling_type,)
    # The pinned mlx-lm does not define PoolingCache, but keep the namespace
    # resolver symmetric if a future coherence-qualified release adds it.
    lm_pooling_type = getattr(lm_cache, "PoolingCache", None)
    if isinstance(lm_pooling_type, type):
        supported_types += (lm_pooling_type,)
    # The vendored package owns a distinct ArraysCache class as well. Hybrid
    # VLM backbones return this native type, so the serialized compatibility
    # lane must accept it for the same reason it accepts mlx-lm's class.
    if allow_arrays_cache and hasattr(vendored_cache, "ArraysCache"):
        supported_types += (vendored_cache.ArraysCache,)
    if allow_arrays_cache:
        supported_types += (lm_cache.ArraysCache,)
    try:
        from mlx_vlm.models import cache as vlm_cache
    except ImportError:
        # mlx-vlm is optional. Text-only installations still import the engine,
        # although they never enter the MLLM serving path.
        pass
    else:
        # Upstream model classes still *create* caches with their own
        # identical class objects (type unification lands with step 3's
        # model vendoring), so the upstream namespace stays recognized too.
        # VENDOR-DEVIATION(dual-namespace): upstream recognition is
        # transitional; one mechanical revert restores byte-verbatim once
        # step 3 unifies the types.
        supported_types += (vlm_cache.KVCache, vlm_cache.RotatingKVCache)
        upstream_compound = getattr(vlm_cache, "CacheList", None)
        if isinstance(upstream_compound, type):
            compound_types += (upstream_compound,)
        upstream_pooling = getattr(vlm_cache, "PoolingCache", None)
        if isinstance(upstream_pooling, type):
            supported_types += (upstream_pooling,)
        if allow_arrays_cache and hasattr(vlm_cache, "ArraysCache"):
            supported_types += (vlm_cache.ArraysCache,)

    for cache in caches:
        if compound_types and isinstance(cache, compound_types):
            incompatible = first_incompatible_mllm_cache_type(
                cast(Any, cache).caches,
                allow_arrays_cache=allow_arrays_cache,
            )
            if incompatible is not None:
                return incompatible
            continue
        if not isinstance(cache, supported_types):
            return type(cache).__name__
    return None
