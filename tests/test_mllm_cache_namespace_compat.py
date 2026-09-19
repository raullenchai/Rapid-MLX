# SPDX-License-Identifier: Apache-2.0
"""Dual-namespace cache contract for the vendored mlx-vlm cache types.

The vendored ``rapid_mlx.models.mlx_vlm_vendored.cache`` module owns its own
class objects, while upstream mlx-vlm model classes (until step 3's model
vendoring unifies the types) still create caches with *their* identical class
objects. Every lane qualification site must accept both namespaces — a
silent fast-path or prefix-resume disable is the failure mode this file
guards against. The pairing tests below pin that a cache built in either
namespace yields the same verdict at every seam.
"""

import builtins

import pytest

pytest.importorskip("mlx")
pytestmark = pytest.mark.requires_mlx

import mlx.core as mx

from rapid_mlx.mllm_cache_compat import first_incompatible_mllm_cache_type


def _kv_state(n_tokens: int):
    keys = mx.arange(n_tokens * 4, dtype=mx.float32).reshape(1, 1, n_tokens, 4)
    return keys, mx.zeros((1, 1, n_tokens, 4))


@pytest.fixture(params=["vendored", "upstream", "mlx-lm"])
def cache_ns(request):
    """The vendored, upstream mlx-vlm, and mlx-lm namespaces, paired.

    Every recognition contract in this file is asserted once per namespace:
    whichever class objects production models end up producing, the lane
    must treat the two namespaces identically. The upstream parameter is
    skipped when the optional ``mlx_vlm`` distribution is absent — the
    vendored cases must still run on that base-install configuration.
    """
    if request.param == "vendored":
        from rapid_mlx.models.mlx_vlm_vendored import cache
    elif request.param == "upstream":
        pytest.importorskip("mlx_vlm.models.cache")
        import mlx_vlm.models.cache as cache  # type: ignore[no-redef]
    else:
        import mlx_lm.models.cache as cache  # type: ignore[no-redef]
    return cache


def test_kv_cache_accepted_for_batch_merge(cache_ns):
    assert first_incompatible_mllm_cache_type([cache_ns.KVCache()]) is None


def test_rotating_kv_cache_accepted_for_batch_merge(cache_ns):
    assert first_incompatible_mllm_cache_type([cache_ns.RotatingKVCache(8)]) is None


def test_cache_list_leaves_validated_recursively(cache_ns):
    compound = cache_ns.CacheList(cache_ns.KVCache(), cache_ns.KVCache())
    assert first_incompatible_mllm_cache_type([compound]) is None
    compound.caches = (compound.caches[0], object())
    assert first_incompatible_mllm_cache_type([compound]) == "object"


def test_pooling_cache_accepted_when_namespace_defines_it(cache_ns):
    pooling_type = getattr(cache_ns, "PoolingCache", None)
    if pooling_type is None:
        pytest.skip("this cache namespace does not define PoolingCache")
    assert first_incompatible_mllm_cache_type([pooling_type(4)]) is None


def test_arrays_cache_gated_on_hybrid_lane(cache_ns):
    cache = cache_ns.ArraysCache(1)
    assert first_incompatible_mllm_cache_type([cache], allow_arrays_cache=True) is None
    assert (
        first_incompatible_mllm_cache_type([cache], allow_arrays_cache=False)
        == "ArraysCache"
    )


def test_unknown_cache_stays_fail_closed(cache_ns):
    del cache_ns
    assert first_incompatible_mllm_cache_type([object()]) == "object"


def test_singleton_leaf_qualification(cache_ns):
    from rapid_mlx.mllm_batch_generator import _singleton_regular_cache_leaves

    kv = cache_ns.KVCache()
    kv.update_and_fetch(*_kv_state(4))
    recurrent = cache_ns.ArraysCache(2)
    recurrent.cache = [mx.zeros((1, 2)), mx.zeros((1, 3))]

    # Exact-type eligibility is namespace-symmetric: the serialized lane
    # produces upstream-typed leaves in production and vendored-typed leaves
    # in tests, and both must take the singleton no-rebatch path.
    assert _singleton_regular_cache_leaves([kv], allow_arrays_cache=True)
    assert _singleton_regular_cache_leaves([kv, recurrent], allow_arrays_cache=True)
    assert not _singleton_regular_cache_leaves([kv], allow_arrays_cache=False)
    assert not _singleton_regular_cache_leaves(
        [cache_ns.RotatingKVCache(8)], allow_arrays_cache=True
    )


def test_singleton_leaf_extraction_is_namespace_symmetric(cache_ns):
    from rapid_mlx.mllm_batch_generator import _extract_detached_singleton_leaf

    kv = cache_ns.KVCache()
    kv.update_and_fetch(*_kv_state(3))
    detached = _extract_detached_singleton_leaf(kv, 0)
    assert type(detached) is type(kv)
    # The leaf is batch-row 0 of a B=1 cache, trimmed to ``offset`` tokens.
    keys, values = detached.state
    assert keys.shape == (1, 1, 3, 4)
    assert values.shape == (1, 1, 3, 4)

    recurrent = cache_ns.ArraysCache(2)
    recurrent.cache = [mx.full((1, 2), 1.0), mx.full((1, 3), 2.0)]
    detached = _extract_detached_singleton_leaf(recurrent, 0)
    assert type(detached) is type(recurrent)
    assert [state.shape for state in detached.cache] == [(1, 2), (1, 3)]


def test_singleton_leaf_extraction_survives_optional_mlx_vlm_absence(monkeypatch):
    """The vendored lane remains usable in a text-only/base installation."""
    from rapid_mlx.mllm_batch_generator import _extract_detached_singleton_leaf
    from rapid_mlx.models.mlx_vlm_vendored.cache import KVCache

    real_import = builtins.__import__

    def import_without_mlx_vlm(name, *args, **kwargs):
        if name == "mlx_vlm.models.cache":
            raise ImportError("optional mlx-vlm distribution is absent")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_mlx_vlm)
    cache = KVCache()
    cache.update_and_fetch(*_kv_state(2))

    detached = _extract_detached_singleton_leaf(cache, 0)

    assert type(detached) is KVCache
    assert detached.state[0].shape == (1, 1, 2, 4)


def test_future_mlx_lm_pooling_cache_is_accepted(monkeypatch):
    """Keep the qualified mlx-lm namespace symmetric if it adds pooling."""
    from mlx_lm.models import cache as lm_cache

    class FuturePoolingCache:
        pass

    monkeypatch.setattr(lm_cache, "PoolingCache", FuturePoolingCache, raising=False)

    assert first_incompatible_mllm_cache_type([FuturePoolingCache()]) is None


def test_recurrent_layer_recognition(cache_ns):
    from rapid_mlx.hybrid_state_checkpoints import is_recurrent_layer

    recurrent = cache_ns.ArraysCache(2)
    recurrent.cache = [mx.zeros((1, 2)), mx.zeros((1, 3))]
    assert is_recurrent_layer(recurrent)
    # A look-alike wrapper with a non-list ``cache`` stays refused — the
    # fetch path must not restore a shallow copy of state it cannot check.
    assert not is_recurrent_layer(cache_ns.KVCache())


def test_quantized_lane_supports_both_namespaces():
    from mlx_lm.models.cache import KVCache as LMKVCache
    from mlx_lm.models.cache import RotatingKVCache as LMRotatingKVCache

    from rapid_mlx.models.mlx_vlm_vendored.cache import (
        KVCache as VendoredKVCache,
    )
    from rapid_mlx.models.mlx_vlm_vendored.cache import (
        RotatingKVCache as VendoredRotatingKVCache,
    )
    from rapid_mlx.quantized_batch_cache import supported_kv_cache_types

    plain, rotating = supported_kv_cache_types()
    assert VendoredKVCache in plain
    assert VendoredRotatingKVCache in rotating
    assert LMKVCache in plain
    assert LMRotatingKVCache in rotating
    try:
        import mlx_vlm.models.cache as vlm_cache
    except ImportError:  # pragma: no cover - optional runtime
        pass
    else:
        assert vlm_cache.KVCache in plain
        assert vlm_cache.RotatingKVCache in rotating
