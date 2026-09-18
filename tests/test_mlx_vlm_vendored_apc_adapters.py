# SPDX-License-Identifier: Apache-2.0
"""Contract tests for the vendored ``apc_adapters`` module.

The vendored adapters serve both cache namespaces during the transition (the
vendored module and upstream ``mlx_vlm.models.cache``), and constructors must
keep results in the producer's namespace. Each test pins one side of that
contract; the upstream-namespace tests double as behavior-parity probes
against the pre-vendoring lane behavior.
"""

import pytest

pytest.importorskip("mlx")
pytestmark = pytest.mark.requires_mlx

import mlx.core as mx

from rapid_mlx.models.mlx_vlm_vendored import apc_adapters
from rapid_mlx.models.mlx_vlm_vendored import cache as vendored_cache
from rapid_mlx.models.mlx_vlm_vendored.apc_adapters import (
    Capability,
    clone_cache_entry,
    resolve_capability,
)


def _upstream_cache_ns():
    pytest.importorskip("mlx_vlm.models.cache")
    from mlx_vlm.models import cache as upstream_cache

    return upstream_cache


def _populated_kv(ns):
    cache = ns.KVCache()
    keys = mx.arange(2 * 4 * 6, dtype=mx.float32).reshape(1, 2, 4, 6)
    cache.update_and_fetch(keys, mx.zeros((1, 2, 4, 6)))
    return cache


def test_capability_resolution_covers_both_namespaces():
    upstream = _upstream_cache_ns()
    for ns in (vendored_cache, upstream):
        assert resolve_capability(ns.KVCache()) is Capability.PAGEABLE
        assert resolve_capability(ns.RotatingKVCache(8)) is Capability.WINDOWED
        assert resolve_capability(ns.ArraysCache(2)) is Capability.CHECKPOINT
        assert (
            resolve_capability(ns.CacheList(ns.KVCache(), ns.ArraysCache(2)))
            is Capability.COMPOSITE
        )


def test_clone_kv_entry_preserves_producer_namespace_and_content():
    upstream = _upstream_cache_ns()
    for ns in (vendored_cache, upstream):
        cache = _populated_kv(ns)
        eval_targets: list = []
        cloned = clone_cache_entry(
            cache, min_capacity_tokens=32, eval_targets=eval_targets
        )
        assert type(cloned) is ns.KVCache
        assert cloned.offset == 4
        assert mx.array_equal(cloned.keys[..., :4, :], cache.keys[..., :4, :])
        # Detachment contract: the clone must not alias the live cache's
        # arrays (identity is the observable form of aliasing in MLX).
        assert cloned.keys is not cache.keys
        assert cloned.values is not cache.values
        assert eval_targets


def test_clone_rotating_entry_preserves_window_state():
    upstream = _upstream_cache_ns()
    for ns in (vendored_cache, upstream):
        cache = ns.RotatingKVCache(8)
        cache.update_and_fetch(mx.ones((1, 1, 3, 4)), mx.zeros((1, 1, 3, 4)))
        cloned = clone_cache_entry(cache, min_capacity_tokens=0, eval_targets=[])
        assert type(cloned) is ns.RotatingKVCache
        assert cloned.max_size == 8
        assert cloned.offset == 3
        assert mx.array_equal(cloned.keys, cache.keys)


def test_clone_arrays_entry_preserves_producer_namespace():
    upstream = _upstream_cache_ns()
    for ns in (vendored_cache, upstream):
        cache = ns.ArraysCache(2)
        cache.cache = [mx.full((1, 2), 1.0), mx.full((1, 3), 2.0)]
        cloned = clone_cache_entry(cache, min_capacity_tokens=0, eval_targets=[])
        assert type(cloned) is ns.ArraysCache
        assert len(cloned.cache) == 2
        assert mx.array_equal(cloned.cache[0], cache.cache[0])
        assert mx.array_equal(cloned.cache[1], cache.cache[1])


def test_clone_cache_list_preserves_producer_namespace():
    upstream = _upstream_cache_ns()
    for ns in (vendored_cache, upstream):
        entry = ns.CacheList(_populated_kv(ns), ns.ArraysCache(1))
        cloned = clone_cache_entry(entry, min_capacity_tokens=0, eval_targets=[])
        assert type(cloned) is ns.CacheList
        assert type(cloned.caches[0]) is ns.KVCache
        assert type(cloned.caches[1]) is ns.ArraysCache


def test_clone_empty_single_row_batch_returns_producer_namespace_type():
    """The empty-batch fast path must not switch namespaces: upstream-typed
    input yields an upstream-typed fresh cache and vice versa."""
    upstream = _upstream_cache_ns()
    for ns in (vendored_cache, upstream):
        batch = ns.BatchKVCache(mx.array([0]))
        assert batch.is_single_row() and batch.empty()
        cloned = clone_cache_entry(batch, min_capacity_tokens=0, eval_targets=[])
        assert type(cloned) is ns.KVCache

        rotating = ns.BatchRotatingKVCache(8, [0])
        assert rotating.is_single_row() and rotating.empty()
        cloned = clone_cache_entry(rotating, min_capacity_tokens=0, eval_targets=[])
        assert type(cloned) is ns.RotatingKVCache
        assert cloned.max_size == 8


def test_clone_single_row_batch_extracts_into_producer_namespace():
    upstream = _upstream_cache_ns()
    for ns in (vendored_cache, upstream):
        keys = mx.arange(1 * 2 * 3 * 4, dtype=mx.float32).reshape(1, 2, 3, 4)
        batch = ns.BatchKVCache(mx.array([0]))
        batch.update_and_fetch(keys, mx.zeros((1, 2, 3, 4)))
        assert batch.is_single_row() and not batch.empty()
        cloned = clone_cache_entry(batch, min_capacity_tokens=0, eval_targets=[])
        assert type(cloned) is ns.KVCache
        assert cloned.offset == 3
        assert mx.array_equal(cloned.keys[..., :3, :], keys)


def test_merge_rows_keeps_producer_namespace():
    upstream = _upstream_cache_ns()
    for ns in (vendored_cache, upstream):
        a = ns.BatchKVCache(mx.array([0]))
        a.update_and_fetch(mx.ones((1, 1, 2, 4)), mx.zeros((1, 1, 2, 4)))
        b = ns.BatchKVCache(mx.array([0]))
        b.update_and_fetch(mx.full((1, 1, 4, 4), 2.0), mx.zeros((1, 1, 4, 4)))
        # Real merge_rows callers pass cloned rows, whose clone left
        # ``offset`` as a plain int; mirror that shape here.
        a.offset, b.offset = 2, 4

        merged = apc_adapters.KVCacheCloneAdapter().merge_rows([a, b], [2, 4])

        assert type(merged) is ns.BatchKVCache
        assert merged.batch_size == 2
        assert merged.keys.shape[0] == 2


def test_explicit_snapshot_contract_recognizes_both_base_classes():
    upstream = _upstream_cache_ns()

    class VendoredSnapshot(vendored_cache._BaseCache):
        def prefix_cache_snapshot(self):
            return {}

    class UpstreamSnapshot(upstream._BaseCache):
        def prefix_cache_snapshot(self):
            return {}

    class Bare(vendored_cache._BaseCache):
        pass

    assert apc_adapters._has_explicit_snapshot_contract(VendoredSnapshot())
    assert apc_adapters._has_explicit_snapshot_contract(UpstreamSnapshot())
    assert not apc_adapters._has_explicit_snapshot_contract(Bare())
    assert not apc_adapters._has_explicit_snapshot_contract(object())


def test_apc_exact_eligible_covers_both_namespaces():
    upstream = _upstream_cache_ns()
    for ns in (vendored_cache, upstream):
        assert apc_adapters.apc_exact_eligible(_populated_kv(ns))
        assert apc_adapters.apc_exact_eligible(ns.ArraysCache(1))
        assert apc_adapters.apc_exact_eligible(ns.CacheList(_populated_kv(ns)))
        assert not apc_adapters.apc_exact_eligible(object())


def test_adapter_tables_are_namespace_complete():
    """Both namespaces must appear in the capability and clone tables — the
    silent-failure mode this whole slice guards against is one namespace's
    types missing from an exact-type dispatch table."""
    upstream = _upstream_cache_ns()
    apc_adapters.register_default_capabilities()
    rules = apc_adapters._clone_rules()
    rule_types = {typ for typ, _ in rules}
    for ns in (vendored_cache, upstream):
        assert resolve_capability(ns.KVCache()) is Capability.PAGEABLE
        for cls in (
            ns.KVCache,
            ns.RotatingKVCache,
            ns.ChunkedKVCache,
            ns.ArraysCache,
            ns.PoolingCache,
        ):
            assert cls in rule_types
