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
    merge_cache_entries,
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
        # Some cache producers annotate their batch window with the matching
        # single-row retention policy even though the base constructor does
        # not expose it.  The empty conversion must preserve that state.
        rotating.keep = 3
        assert rotating.is_single_row() and rotating.empty()
        cloned = clone_cache_entry(rotating, min_capacity_tokens=0, eval_targets=[])
        assert type(cloned) is ns.RotatingKVCache
        assert cloned.max_size == 8
        assert cloned.keep == 3

        quantized = ns.BatchQuantizedKVCache([0], group_size=32, bits=4)
        assert quantized.is_single_row() and quantized.empty()
        cloned = clone_cache_entry(quantized, min_capacity_tokens=0, eval_targets=[])
        assert type(cloned) is ns.QuantizedKVCache
        assert cloned.group_size == 32
        assert cloned.bits == 4


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


def test_arrays_merge_rows_preserves_per_row_metadata():
    upstream = _upstream_cache_ns()
    for ns in (vendored_cache, upstream):
        first = ns.ArraysCache(1)
        first.cache = [mx.ones((1, 2))]
        first.left_padding = mx.array([2])
        second = ns.ArraysCache(1)
        second.cache = [mx.zeros((1, 2))]
        second.lengths = mx.array([5])

        merged = apc_adapters.ArraysCacheCloneAdapter().merge_rows(
            [first, second], [3, 5]
        )

        assert type(merged) is ns.ArraysCache
        assert merged.cache[0].shape[0] == 2
        assert merged.left_padding.tolist() == [2, 0]
        assert merged.lengths.tolist() == [0, 5]


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


def test_third_party_snapshot_clones_without_cache_namespace(monkeypatch):
    class ThirdPartyCache:
        offset = 1

        def __init__(self):
            self.value = mx.array([1.0])

        def prefix_cache_snapshot(self):
            return {"value": self.value}

        def prefix_cache_restore(self, payload):
            self.value = payload["value"]

    source = ThirdPartyCache()
    monkeypatch.setattr(apc_adapters, "_cache_namespace_of", lambda _cache: None)

    cloned = clone_cache_entry(source, min_capacity_tokens=0, eval_targets=[])

    assert type(cloned) is ThirdPartyCache
    assert mx.array_equal(cloned.value, source.value)
    assert cloned.value is not source.value


def test_third_party_merge_runs_without_cache_namespace(monkeypatch):
    class ThirdPartyCache:
        def __init__(self, value):
            self.value = value

        def prefix_cache_merge(self, entries, prefix_lens):
            return type(self)(sum(e.value for e in entries) + sum(prefix_lens))

    monkeypatch.setattr(apc_adapters, "_cache_namespace_of", lambda _cache: None)

    merged = merge_cache_entries([ThirdPartyCache(2), ThirdPartyCache(3)], [5, 7])

    assert type(merged) is ThirdPartyCache
    assert merged.value == 17


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


@pytest.fixture
def no_upstream_cache_namespace(monkeypatch):
    """Simulate an install whose ``mlx_vlm.models`` subpackage is absent —
    the dual-namespace resolution then has only the vendored namespace, and
    the upstream fallback inside ``_cache_namespace_of`` must fail. (The
    documented ``mlx_vlm.apc`` redirect stays live so the adapters
    themselves keep working; this isolates the namespace-resolution
    defect.)"""
    import sys

    import mlx_vlm.apc  # noqa: F401 - pin the redirect before blocking

    monkeypatch.setitem(sys.modules, "mlx_vlm.models", None)
    yield


def test_clone_tuple_entry_without_upstream_namespace(
    no_upstream_cache_namespace,
):
    """A bare tuple is namespace-agnostic, and ``apc_exact_eligible``
    declares tuples supported. Upstream resolves its single cache module
    unconditionally, so its tuple branch always works; the vendored copy
    must not drop composite caches just because the tuple itself resolves
    to no namespace (pre-fix behavior: ``clone_cache_entry`` returned
    ``None`` whenever the upstream cache namespace was unavailable)."""
    a = _populated_kv(vendored_cache)
    b = _populated_kv(vendored_cache)
    eval_targets: list = []
    cloned = clone_cache_entry(
        (a, b), min_capacity_tokens=32, eval_targets=eval_targets
    )
    assert isinstance(cloned, tuple) and len(cloned) == 2
    assert all(type(s) is vendored_cache.KVCache for s in cloned)
    assert cloned[0].offset == 4
    assert cloned[1].offset == 4


def test_merge_tuple_entries_without_upstream_namespace(
    no_upstream_cache_namespace,
):
    """``merge_cache_entries`` derives the container namespace from the
    first tuple element when the tuple itself resolves to none; pre-fix it
    bailed out with ``None`` for composite tuple entries."""
    merged = apc_adapters.merge_cache_entries(
        [
            (_populated_kv(vendored_cache), _populated_kv(vendored_cache)),
            (_populated_kv(vendored_cache), _populated_kv(vendored_cache)),
        ],
        [3, 3],
    )
    assert merged is not None
    assert type(merged) is vendored_cache.CacheList
    assert all(type(s) is vendored_cache.BatchKVCache for s in merged.caches)


def test_merge_tuple_entries_keeps_child_namespace_with_upstream_installed():
    """A bare tuple has no namespace, so its first child must select the
    result container even when both cache implementations are importable."""
    upstream = _upstream_cache_ns()
    for ns in (vendored_cache, upstream):
        merged = apc_adapters.merge_cache_entries(
            [
                (_populated_kv(ns), _populated_kv(ns)),
                (_populated_kv(ns), _populated_kv(ns)),
            ],
            [3, 3],
        )
        assert merged is not None
        assert type(merged) is ns.CacheList
        assert all(type(s) is ns.BatchKVCache for s in merged.caches)


def test_type_table_build_publishes_only_complete_tables(monkeypatch):
    """The lazy table builders must publish their globals only after every
    namespace is processed. Assigning inside the namespace loop let a
    concurrent first caller observe a vendored-only table and reject the
    upstream cache types the lane still produces."""

    class _BrokenNamespace:
        def __getattr__(self, name):
            raise RuntimeError("namespace probe failed mid-build")

    def _namespaces():
        return [vendored_cache, _BrokenNamespace()]

    monkeypatch.setattr(apc_adapters, "_cache_namespaces", _namespaces)
    monkeypatch.setattr(apc_adapters, "_APC_TYPE_TABLES", None)
    with pytest.raises(RuntimeError):
        apc_adapters._apc_type_tables()
    assert apc_adapters._APC_TYPE_TABLES is None


def test_clone_rules_build_publishes_only_complete_rules(monkeypatch):
    """Same partial-publication hazard for the clone-rule table."""

    class _BrokenNamespace:
        def __getattr__(self, name):
            raise RuntimeError("namespace probe failed mid-build")

    def _namespaces():
        return [vendored_cache, _BrokenNamespace()]

    monkeypatch.setattr(apc_adapters, "_cache_namespaces", _namespaces)
    monkeypatch.setattr(apc_adapters, "_CLONE_RULES", None)
    with pytest.raises(RuntimeError):
        apc_adapters._clone_rules()
    assert apc_adapters._CLONE_RULES is None


def test_cache_specs_capabilities_and_plan_descriptions(monkeypatch):
    pageable = apc_adapters.CacheSpec(Capability.PAGEABLE, "KV", block_eligible=True)
    unsupported = apc_adapters.CacheSpec(Capability.UNSUPPORTED, "Bad")
    composite = apc_adapters.CacheSpec(
        Capability.COMPOSITE, "Tuple", children=(pageable,)
    )
    assert pageable.pageable and pageable.restorable
    assert composite.pageable and composite.restorable
    assert pageable.group_key[0] == "pageable"

    assert (
        apc_adapters.cache_spec((vendored_cache.KVCache(),)).capability
        is Capability.COMPOSITE
    )
    assert apc_adapters.cache_spec(()).capability is Capability.UNSUPPORTED
    assert apc_adapters.cache_spec(
        vendored_cache.CacheList(vendored_cache.KVCache())
    ).children

    class _OddWindow(vendored_cache.KVCache):
        window_size = "not-an-int"

    class _Window(vendored_cache.KVCache):
        window_size = 12

    assert apc_adapters.cache_spec(_OddWindow()).window_size is None
    assert apc_adapters.cache_spec(_Window()).window_size == 12
    assert (
        resolve_capability(object(), {object: Capability.CHECKPOINT})
        is Capability.CHECKPOINT
    )

    class _PageableChild(vendored_cache.KVCache):
        pass

    class _WindowedChild(vendored_cache.KVCache):
        max_size = 8

    assert resolve_capability(_PageableChild()) is Capability.CHECKPOINT
    assert resolve_capability(_WindowedChild()) is Capability.WINDOWED

    class _CheckpointChild(vendored_cache.ArraysCache):
        pass

    assert resolve_capability(_CheckpointChild(1)) is Capability.CHECKPOINT

    class _Explicit:
        def prefix_cache_snapshot(self):
            return {}

    assert resolve_capability(_Explicit()) is Capability.CHECKPOINT
    assert resolve_capability(object()) is Capability.UNSUPPORTED

    plan = apc_adapters.build_prefix_cache_plan_from_caches(
        [vendored_cache.KVCache(), vendored_cache.ArraysCache(1)]
    )
    assert plan.restorable and plan.is_hybrid and plan.strategy == "checkpoint"
    assert plan.legacy_mode == "exact"
    assert plan.capabilities == [Capability.PAGEABLE, Capability.CHECKPOINT]
    assert "PrefixCachePlan" in plan.describe()
    empty = apc_adapters.PrefixCachePlan()
    assert not empty.restorable and empty.strategy is None

    class _BrokenModel:
        def make_cache(self):
            raise RuntimeError("broken")

    assert not apc_adapters.build_prefix_cache_plan(_BrokenModel()).restorable

    import mlx_vlm.models.cache as upstream_cache

    monkeypatch.setattr(
        upstream_cache,
        "make_prompt_cache",
        lambda _model: [vendored_cache.KVCache()],
    )
    assert apc_adapters.build_prefix_cache_plan(object()).strategy == "block"


def test_tree_checkpoint_and_capacity_helpers():
    tree = (mx.array([1]), [mx.array([2])], {"x": mx.array([3])}, "plain")
    snap = apc_adapters._snapshot_tree(tree)
    arrays = []
    apc_adapters._eval_tree(snap, arrays)
    assert len(arrays) == 3 and snap[0] is not tree[0]
    assert not apc_adapters._is_snapshotable(object())

    adapter = apc_adapters.CheckpointAdapter()
    assert adapter.capture(object(), 0) is None

    class _State:
        state = {"x": mx.array([1])}
        meta_state = {"offset": 1}

    fragment = adapter.capture(_State(), 1)
    fresh = _State()
    adapter.restore(fresh, fragment)
    assert mx.array_equal(fresh.state["x"], mx.array([1]))
    assert fresh.meta_state == {"offset": 1}

    apc_adapters.reserve_checkpoint_capacity(fresh, min_capacity_tokens=None)
    apc_adapters.reserve_checkpoint_capacity(fresh, min_capacity_tokens=4)

    class _Reservable:
        def prefix_cache_reserve(self, count):
            self.count = count
            return {"reserved": mx.array([count])}

    reservable = _Reservable()
    targets = []
    apc_adapters.reserve_checkpoint_capacity(
        reservable, min_capacity_tokens=6, eval_targets=targets
    )
    assert reservable.count == 6 and len(targets) == 1


def test_namespace_resolution_and_optional_turboquant(monkeypatch):
    monkeypatch.setattr(apc_adapters, "_cache_namespaces", lambda: [])
    assert apc_adapters._cache_namespace_of(object()).__name__ == "mlx_vlm.models.cache"

    import builtins

    real_import = builtins.__import__

    def _without_turbo(name, *args, **kwargs):
        if name == "mlx_vlm.turboquant":
            raise ImportError("optional")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _without_turbo)
    monkeypatch.setattr(apc_adapters, "_DEFAULTS_REGISTERED", False)
    monkeypatch.setattr(apc_adapters, "_cache_namespaces", lambda: [vendored_cache])
    apc_adapters.register_default_capabilities()


def test_clone_adapter_remaining_shapes(monkeypatch):
    # Chunked clone/merge.
    chunked = vendored_cache.ChunkedKVCache(chunk_size=4)
    chunked.offset = 2
    chunked.start_position = 1
    chunked.keys = mx.ones((1, 1, 2, 3))
    chunked.values = mx.zeros((1, 1, 2, 3))
    targets = []
    cloned = apc_adapters.ChunkedKVCacheCloneAdapter().clone(
        chunked, min_capacity_tokens=0, eval_targets=targets
    )
    assert cloned.offset == 2 and cloned.start_position == 1 and len(targets) == 2

    merged = apc_adapters.ChunkedKVCacheCloneAdapter().merge_rows(
        [cloned, cloned], [2, 2]
    )
    assert type(merged) is vendored_cache.BatchKVCache

    rotating = vendored_cache.RotatingKVCache(8)
    rotating.update_and_fetch(mx.ones((1, 1, 2, 3)), mx.zeros((1, 1, 2, 3)))
    fake_ns = type(
        "NS",
        (),
        {
            "BatchRotatingKVCache": type(
                "BatchRotating",
                (),
                {"merge": staticmethod(lambda caches: ("rotating", len(caches)))},
            )
        },
    )
    original_namespace = apc_adapters._cache_namespace_of
    monkeypatch.setattr(apc_adapters, "_cache_namespace_of", lambda _c: fake_ns)
    assert apc_adapters.RotatingKVCacheCloneAdapter().merge_rows(
        [rotating, rotating], [2, 2]
    ) == ("rotating", 2)
    monkeypatch.setattr(apc_adapters, "_cache_namespace_of", original_namespace)

    arrays = vendored_cache.ArraysCache(2)
    arrays.cache = [None, mx.ones((1, 2))]
    arrays.left_padding = mx.array([1])
    arrays.lengths = mx.array([2])
    targets = []
    cloned_arrays = apc_adapters.ArraysCacheCloneAdapter().clone(
        arrays, min_capacity_tokens=0, eval_targets=targets
    )
    assert cloned_arrays.cache[0] is None and len(targets) == 3

    empty_arrays = vendored_cache.ArraysCache(2)
    empty_arrays.cache = [None, None]
    merged_arrays = apc_adapters.ArraysCacheCloneAdapter().merge_rows(
        [empty_arrays, empty_arrays], [0, 0]
    )
    assert merged_arrays.cache == [None, None]

    class _Pooling:
        ratio = 2
        remainder = 1
        buf_kv = mx.ones((1, 2))
        buf_gate = None
        pooled = mx.zeros((1, 2))

        def __init__(self, ratio):
            self.ratio = ratio

        @classmethod
        def merge(cls, caches):
            return ("merged", len(caches))

    targets = []
    pooled = apc_adapters.PoolingCacheCloneAdapter().clone(
        _Pooling(2), min_capacity_tokens=0, eval_targets=targets
    )
    assert pooled.ratio == 2 and pooled.remainder == 1 and len(targets) == 2
    assert apc_adapters.PoolingCacheCloneAdapter().merge_rows([pooled], [1]) == (
        "merged",
        1,
    )


def test_clone_and_merge_fallback_contracts(monkeypatch):
    class _StateOnly:
        def __init__(self):
            self.state = {"a": mx.array([1])}
            self.meta_state = {"m": 2}

        @classmethod
        def from_state(cls, state, meta):
            out = cls()
            out.state, out.meta_state = state, meta
            return out

    monkeypatch.setattr(apc_adapters, "_cache_namespace_of", lambda _c: None)
    monkeypatch.setattr(
        apc_adapters, "_custom_state_contract", lambda c: isinstance(c, _StateOnly)
    )
    targets = []
    cloned = clone_cache_entry(
        _StateOnly(), min_capacity_tokens=0, eval_targets=targets
    )
    assert isinstance(cloned, _StateOnly) and len(targets) == 1

    class _StateWithoutFactory:
        state = {"a": mx.array([2])}
        meta_state = {"m": 3}

    monkeypatch.setattr(
        apc_adapters,
        "_custom_state_contract",
        lambda c: isinstance(c, (_StateOnly, _StateWithoutFactory)),
    )
    cloned_without_factory = clone_cache_entry(
        _StateWithoutFactory(), min_capacity_tokens=0, eval_targets=[]
    )
    assert mx.array_equal(cloned_without_factory.state["a"], mx.array([2]))
    assert clone_cache_entry(object(), min_capacity_tokens=0, eval_targets=[]) is None
    assert merge_cache_entries([], []) is None
    assert merge_cache_entries([object()], [0]) is None

    # Constructor-less snapshot caches use __new__ and the restore protocol.
    class _Snapshot:
        def __init__(self, required):
            self.value = required

        def prefix_cache_snapshot(self):
            return {"value": mx.array([self.value])}

        def prefix_cache_restore(self, payload):
            self.value = int(payload["value"].item())

    monkeypatch.setattr(
        apc_adapters,
        "_has_explicit_snapshot_contract",
        lambda c: isinstance(c, _Snapshot),
    )
    cloned = clone_cache_entry(_Snapshot(4), min_capacity_tokens=0, eval_targets=[])
    assert cloned.value == 4
    monkeypatch.setattr(
        apc_adapters.CheckpointAdapter, "capture", lambda *_a, **_kw: None
    )
    assert (
        clone_cache_entry(_Snapshot(4), min_capacity_tokens=0, eval_targets=[]) is None
    )


def test_clone_known_namespace_edge_cases(monkeypatch):
    monkeypatch.setattr(apc_adapters, "_cache_namespace_of", lambda _c: vendored_cache)

    class _MultiRow:
        def extract(self, _idx):
            return None

        def is_single_row(self):
            return False

    assert (
        clone_cache_entry(_MultiRow(), min_capacity_tokens=0, eval_targets=[]) is None
    )

    class _Dequant:
        def dequantize_for_apc(self):
            return None, None

    assert (
        type(clone_cache_entry(_Dequant(), min_capacity_tokens=0, eval_targets=[]))
        is vendored_cache.KVCache
    )

    class _ExplicitKnown:
        def prefix_cache_snapshot(self):
            return {"value": 1}

        def prefix_cache_restore(self, payload):
            self.value = payload["value"]

    monkeypatch.setattr(
        apc_adapters,
        "_has_explicit_snapshot_contract",
        lambda c: isinstance(c, _ExplicitKnown),
    )
    assert (
        clone_cache_entry(
            _ExplicitKnown(), min_capacity_tokens=0, eval_targets=[]
        ).value
        == 1
    )

    class _KnownState:
        state = {"a": mx.array([1])}
        meta_state = {}

    monkeypatch.setattr(
        apc_adapters, "_custom_state_contract", lambda c: isinstance(c, _KnownState)
    )
    assert isinstance(
        clone_cache_entry(_KnownState(), min_capacity_tokens=0, eval_targets=[]),
        _KnownState,
    )
    monkeypatch.setattr(
        apc_adapters, "_has_explicit_snapshot_contract", lambda _c: False
    )
    monkeypatch.setattr(apc_adapters, "_custom_state_contract", lambda _c: False)
    assert clone_cache_entry(object(), min_capacity_tokens=0, eval_targets=[]) is None

    class _DequantPopulated:
        def dequantize_for_apc(self):
            return mx.ones((1, 1, 2, 3)), mx.zeros((1, 1, 2, 3))

    targets = []
    out = clone_cache_entry(
        _DequantPopulated(), min_capacity_tokens=0, eval_targets=targets
    )
    assert out.offset == 2 and len(targets) == 2

    class _Merge:
        @classmethod
        def merge(cls, entries, prefix_lens):
            return (len(entries), sum(prefix_lens))

    assert merge_cache_entries([_Merge(), _Merge()], [2, 3]) == (2, 5)

    class _NoMerge:
        pass

    assert merge_cache_entries([_NoMerge()], [0]) is None


def test_cachelist_merge_and_plan_failure_descriptions():
    first = vendored_cache.CacheList(_populated_kv(vendored_cache))
    second = vendored_cache.CacheList(_populated_kv(vendored_cache))
    merged = merge_cache_entries([first, second], [4, 4])
    assert type(merged) is vendored_cache.CacheList
    assert type(merged.caches[0]) is vendored_cache.BatchKVCache

    assert apc_adapters.apc_block_eligible(
        type("D", (), {"dequantize_for_apc": lambda self: (None, None)})()
    )
    assert apc_adapters.apc_exact_eligible((vendored_cache.KVCache(),))
    assert apc_adapters.apc_mode([vendored_cache.KVCache()]) == "block"


def test_custom_state_contract_detects_declared_property():
    class _State(vendored_cache._BaseCache):
        @property
        def state(self):
            return ()

    assert apc_adapters._custom_state_contract(_State())
