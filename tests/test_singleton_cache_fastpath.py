# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the singleton KV-cache fast path (B=1 decode).

``install_singleton_cache_fastpath`` patches ``mlx_lm.generate`` so a
one-row batch keeps plain ``KVCache``/``RotatingKVCache`` layers (aligned
causal-mask attention) instead of converting to batched forms, and
promotes them via ``cls.merge([self])`` the moment a second row joins.

The batch surface (``filter``/``extract``/``extend``) is bound onto the
admitted INSTANCES only — never the classes — so ``hasattr``-based
capability detection elsewhere in the process (MLLM batch merging)
stays truthful. ``extract`` returns an independent offset-trimmed copy,
mirroring ``BatchKVCache.extract``. End-to-end correctness (mid-flight
join produces identical tokens) was verified by the bench A/B
(tune1-singleton: midflight-join counts {0:160, 1:120}).
"""

from __future__ import annotations

import pytest

pytest.importorskip("mlx")
pytestmark = pytest.mark.requires_mlx


import importlib

import mlx.core as mx
from mlx_lm.models.cache import KVCache, RotatingKVCache

from vllm_mlx.singleton_cache_fastpath import (
    _is_singleton_passthrough_layer,
    _promote_layer,
    install_singleton_cache_fastpath,
)


@pytest.fixture(scope="module", autouse=True)
def installed():
    gen = importlib.import_module("mlx_lm.generate")
    saved = (
        gen._merge_caches,
        gen._extend_cache,
        getattr(gen, "_rapid_singleton_cache_fastpath", False),
    )
    assert install_singleton_cache_fastpath() is True
    # Idempotent: second install is a no-op success.
    assert install_singleton_cache_fastpath() is True
    yield
    # Restore so unrelated test modules are not order-dependent on the
    # patch (if the process had already installed before this module,
    # this restores that same patched state — a no-op).
    gen._merge_caches, gen._extend_cache = saved[0], saved[1]
    gen._rapid_singleton_cache_fastpath = saved[2]


def _filled_kv(n_tokens=4, n_heads=2, head_dim=8):
    c = KVCache()
    k = mx.zeros((1, n_heads, n_tokens, head_dim))
    v = mx.ones((1, n_heads, n_tokens, head_dim))
    c.update_and_fetch(k, v)
    return c


def _passthrough(layer):
    """Run one layer through the patched merge seam, the way production
    admits it to the singleton lane (binds the surface; data-carrying
    layers are detached into independent copies on admission)."""
    gen = importlib.import_module("mlx_lm.generate")
    merged = gen._merge_caches([[layer]])
    assert type(merged[0]) is type(layer)
    return merged[0]


# ------------------------------------------------------- gate decisions


def test_exact_kvcache_passes_through():
    assert _is_singleton_passthrough_layer(KVCache()) is True
    assert _is_singleton_passthrough_layer(RotatingKVCache(max_size=64)) is True


def test_kvcache_subclass_excluded():
    """_QuantizableKVCache-style subclasses carry merge() semantics the
    passthrough would silently skip (#1197/#1862) — exact types only."""

    class _QuantishKVCache(KVCache):
        pass

    assert _is_singleton_passthrough_layer(_QuantishKVCache()) is False


def test_mlx_lm_module_subclass_still_excluded():
    """pr_validate codex BLOCKING: a KVCache subclass must never qualify
    via the native-batch-surface fallback — inherited (or instance-bound
    base-object) surfaces plus an ``mlx_lm.`` module path would
    otherwise re-admit it and bypass its merge() semantics."""

    class _Spoof(KVCache):
        pass

    _Spoof.__module__ = "mlx_lm.models.cache"
    # Exercise the fallback exactly: give the INSTANCE the surface too.
    obj = _Spoof()
    obj.filter = lambda keep: None
    obj.extract = lambda idx: None
    obj.extend = lambda other: None
    assert _is_singleton_passthrough_layer(obj) is False


def test_foreign_object_excluded():
    class _NotACache:
        pass

    assert _is_singleton_passthrough_layer(_NotACache()) is False


def test_composite_without_own_surface_rejected():
    """codex r2 BLOCKING: a wrapper is not admissible on child
    qualification alone — batch ops run on the WRAPPER, and one without
    its own filter/extract/extend would AttributeError at batch time."""

    class _BareWrapper:
        def __init__(self):
            self.caches = [KVCache()]

    assert _is_singleton_passthrough_layer(_BareWrapper()) is False


def test_composite_with_surface_and_qualifying_children_accepted():
    class _SurfacedWrapper:
        def __init__(self):
            self.caches = [KVCache()]

        def filter(self, keep):
            pass

        def extract(self, idx):
            pass

        def extend(self, other):
            pass

    assert _is_singleton_passthrough_layer(_SurfacedWrapper()) is True


def test_mlx_lm_native_batch_surface_accepted():
    """Hybrid layers (ArraysCache) already store batch-leading state and
    expose filter/extract/extend natively — they pass through."""
    cache_mod = importlib.import_module("mlx_lm.models.cache")
    arrays_cls = getattr(cache_mod, "ArraysCache", None)
    if arrays_cls is None:
        pytest.skip("mlx-lm build has no ArraysCache")
    obj = arrays_cls(size=2)
    if not (
        hasattr(obj, "filter") and hasattr(obj, "extract") and hasattr(obj, "extend")
    ):
        pytest.skip("this mlx-lm ArraysCache lacks native batch surface")
    assert _is_singleton_passthrough_layer(obj) is True


# ------------------------------------------- instance-scoped binding


def test_install_leaves_plain_instances_bare():
    """Manual codex MAJOR: class-level patching would flip hasattr-based
    capability gates for every cache in the process (the MLLM batch path
    guards `hasattr(c, "extend")` and would silently skip merging). A
    cache that never entered the singleton lane must stay bare."""
    c = KVCache()
    assert not hasattr(c, "filter")
    assert not hasattr(c, "extract")
    assert not hasattr(c, "extend")
    r = RotatingKVCache(max_size=64)
    assert not hasattr(r, "extend")


def test_admitted_instance_gains_surface():
    c = _passthrough(_filled_kv())
    assert callable(c.filter) and callable(c.extract) and callable(c.extend)
    # ...and only that instance — the class remains untouched.
    assert not hasattr(KVCache, "filter")


# ----------------------------------------------------------- promotion


def test_promote_kvcache_matches_merge():
    c = _filled_kv()
    promoted = _promote_layer(c)
    expected = type(c).merge([_filled_kv()])
    assert type(promoted) is type(expected)
    assert promoted.offset == expected.offset


def test_promote_is_identity_for_batched():
    c = _filled_kv()
    promoted = _promote_layer(c)
    # Promoting an already-batched cache must be a no-op.
    assert _promote_layer(promoted) is promoted


def test_promote_admitted_instance_still_merges():
    """The bound surface must not confuse promotion: an admitted
    singleton still converts through cls.merge on a join."""
    c = _passthrough(_filled_kv())
    promoted = _promote_layer(c)
    assert type(promoted) is not KVCache
    assert promoted.keys.shape[0] == 1


def test_promote_composite_preserves_wrapper_state():
    """codex r3 BLOCKING: composite promotion must clone the wrapper,
    not rebuild it via ``type(obj)(*children)`` — wrappers holding extra
    constructor state would crash on a second request's join."""

    class _StatefulWrapper:
        def __init__(self, tag, caches):
            self.tag = tag
            self.caches = caches

    w = _StatefulWrapper("keep-me", [_filled_kv()])
    promoted = _promote_layer(w)
    assert promoted is not w
    assert promoted.tag == "keep-me"
    assert type(promoted.caches[0]) is not KVCache  # child promoted


def test_extend_into_empty_batch_all_or_nothing():
    """codex r6: a mixed cache list at the extend seam must not be
    partially bound — either every layer qualifies (detach + bind all)
    or the list returns unchanged, which is stock behavior."""
    gen = importlib.import_module("mlx_lm.generate")
    raw = _filled_kv()

    class _Foreign:
        pass

    out = gen._extend_cache([], [raw, _Foreign()])
    assert out[0] is raw
    assert "filter" not in raw.__dict__  # no partial binding


def test_extend_into_empty_batch_binds_surface():
    """codex r3 BLOCKING: layers reaching the extend seam without going
    through the patched merge must still get the singleton surface (and
    the copy-on-admit detach — a loan is hazardous at any entry)."""
    gen = importlib.import_module("mlx_lm.generate")
    raw = _filled_kv()
    out = gen._extend_cache([], [raw])
    assert type(out[0]) is KVCache
    assert out[0].keys is not raw.keys  # detached copy
    assert "filter" in out[0].__dict__ and "extract" in out[0].__dict__


# ----------------------------------------------- singleton batch surface


def test_filter_keep_one_row_is_noop():
    c = _passthrough(_filled_kv())
    keys_before = c.keys
    c.filter([0])
    assert c.keys is keys_before
    assert c.offset == 4


def test_filter_zero_rows_resets():
    """Defensive branch only: mlx-lm's drain path clears the layer list
    without calling filter([]) — but if a caller ever does, reset."""
    c = _passthrough(_filled_kv())
    c.filter([])
    assert c.keys is None
    assert c.values is None
    assert c.offset == 0


def test_filter_multi_row_raises():
    c = _passthrough(_filled_kv())
    with pytest.raises(NotImplementedError):
        c.filter([0, 1])


def test_filter_wrong_row_raises():
    """codex r2 BLOCKING: filter([1]) must not silently keep row 0 —
    that would hand another request this row's KV state."""
    c = _passthrough(_filled_kv())
    with pytest.raises(IndexError):
        c.filter([1])


def test_extract_returns_independent_trimmed_copy():
    """Manual codex MAJOR: the scheduler extracts from LIVE rows
    (prompt-cache save, disk-KV checkpoints) while decode keeps writing
    into the original buffers — the payload must mirror the batched
    extract contract: independent arrays, trimmed to offset."""
    c = _passthrough(_filled_kv())
    clone = c.extract(0)
    assert type(clone) is KVCache
    assert clone.offset == 4
    assert clone.keys is not c.keys
    assert clone.values is not c.values
    assert clone.keys.shape[2] == 4  # trimmed to offset, not buffer size
    # A payload is a plain cache — no singleton surface rides along.
    assert "extract" not in clone.__dict__
    # Live row keeps decoding: in-place writes into the original buffer
    # must not show through the clone.
    c.values[..., 0:4, :] = c.values[..., 0:4, :] * 0.0 + 7.0
    assert float(clone.values[0, 0, 0, 0]) == 1.0


def test_extract_full_span_buffer_still_independent():
    """codex r4 claimed mx.contiguous may alias when the extracted slice
    spans the full contiguous buffer. Refuted empirically (MLX setitem
    is functional — prior value nodes never see it), and pinned here:
    offset == buffer width, then mutate the original after extract."""
    c = KVCache()
    k = mx.zeros((1, 2, 4, 8))
    v = mx.ones((1, 2, 4, 8))
    c.update_and_fetch(k, v)
    c.keys = mx.contiguous(c.keys[..., :4, :])  # buffer width == offset
    c.values = mx.contiguous(c.values[..., :4, :])
    mx.eval(c.keys, c.values)
    c = _passthrough(c)
    clone = c.extract(0)
    assert clone.keys is not c.keys
    assert clone.values is not c.values
    c.values[..., 0:4, :] = c.values[..., 0:4, :] * 0.0 + 7.0
    assert float(clone.values[0, 0, 0, 0]) == 1.0


def test_extract_rotating_is_independent_copy():
    r = RotatingKVCache(max_size=64)
    k = mx.zeros((1, 2, 4, 8))
    v = mx.ones((1, 2, 4, 8))
    r.update_and_fetch(k, v)
    r = _passthrough(r)
    clone = r.extract(0)
    assert type(clone) is RotatingKVCache
    assert clone.keys is not r.keys
    assert clone.keys.shape[2] == clone._idx
    assert clone.offset == r.offset


def test_extract_survives_batch_drop():
    """The extracted payload outlives whatever the batch does to the
    slot afterwards (drop or reset)."""
    c = _passthrough(_filled_kv())
    clone = c.extract(0)
    c.filter([])  # defensive reset path
    assert clone.keys is not None
    assert clone.offset == 4


def test_extract_nonzero_row_raises():
    c = _passthrough(_filled_kv())
    with pytest.raises(IndexError):
        c.extract(1)


def test_promote_composite_slots_wrapper():
    """codex r5 BLOCKING: wrapper cloning must survive __slots__ classes
    (copy.copy, not __new__ + __dict__)."""

    class _SlotsWrapper:
        __slots__ = ("tag", "caches")

        def __init__(self, tag, caches):
            self.tag = tag
            self.caches = caches

    w = _SlotsWrapper("keep-me", [_filled_kv()])
    promoted = _promote_layer(w)
    assert promoted is not w
    assert promoted.tag == "keep-me"
    assert type(promoted.caches[0]) is not KVCache


def test_detach_deepcopies_native_surface_layers():
    """codex r5 BLOCKING: native batch-surface layers must not retain
    aliases to loaned state either — admission deep-copies them."""
    gen = importlib.import_module("mlx_lm.generate")

    class _NativeLayer:
        def __init__(self):
            self.state = [mx.ones((2, 2))]

        def filter(self, keep):
            pass

        def extract(self, idx):
            pass

        def extend(self, other):
            pass

    _NativeLayer.__module__ = "mlx_lm.models.cache"
    obj = _NativeLayer()
    [admitted] = gen._merge_caches([[obj]])
    assert admitted is not obj
    assert admitted.state[0] is not obj.state[0]


def test_extend_requires_promotion():
    a = _passthrough(_filled_kv())
    b = _filled_kv()
    with pytest.raises(NotImplementedError):
        a.extend(b)


# ------------------------------------------------------ patched seams


def test_merge_caches_single_passthrough_empty_is_identity():
    """Fresh (empty) caches admit unchanged — the hot path pays nothing."""
    gen = importlib.import_module("mlx_lm.generate")
    layers = [KVCache(), KVCache()]
    merged = gen._merge_caches([layers])
    assert merged[0] is layers[0]
    assert merged[1] is layers[1]


def test_merge_caches_detaches_loaned_prefix_state():
    """The l1-smoke llama3-3b golden regression: the memory-aware prefix
    cache loans supersequence/LCP hits as shallow trims SHARING the
    stored entry's arrays (offset rewound only). Admission must copy —
    otherwise in-place decode writes corrupt the stored prefix and every
    later hit returns poisoned state ('Transparent.' ≠ 'Paris')."""
    gen = importlib.import_module("mlx_lm.generate")
    store = _filled_kv(n_tokens=8)
    # Replicate memory_cache._trim_cache_offset's loan: share arrays,
    # rewind offset to 4.
    loan = KVCache.__new__(KVCache)
    loan.keys = store.keys
    loan.values = store.values
    loan.offset = 4
    [live] = gen._merge_caches([[loan]])
    # Decode writes past the trim point (positions 4..5).
    k = mx.full((1, 2, 2, 8), 5.0)
    v = mx.full((1, 2, 2, 8), 5.0)
    live.update_and_fetch(k, v)
    # The stored entry's own region must be untouched.
    assert float(store.values[0, 0, 4, 0]) == 1.0
    assert float(store.values[0, 0, 5, 0]) == 1.0


def test_merge_caches_multi_uses_stock_merge():
    gen = importlib.import_module("mlx_lm.generate")
    merged = gen._merge_caches([[_filled_kv()], [_filled_kv()]])
    # Two rows -> batched form, not the singleton objects.
    assert type(merged[0]) is not KVCache


def test_extend_cache_promotes_then_extends():
    gen = importlib.import_module("mlx_lm.generate")
    a = [_filled_kv()]
    b = [_filled_kv()]
    out = gen._extend_cache(a, b)
    assert type(out[0]) is not KVCache  # promoted to batched form
    # Batched cache now holds two rows' worth of state.
    assert out[0].keys.shape[0] == 2


def test_extend_cache_layer_count_mismatch_raises():
    """codex r2 BLOCKING: a bare zip would silently drop trailing layers
    and serve incomplete KV state. Same-model joins always agree on
    layer count, so a mismatch is corruption — fail loudly."""
    gen = importlib.import_module("mlx_lm.generate")
    with pytest.raises(ValueError, match="layer count mismatch"):
        gen._extend_cache([_filled_kv(), _filled_kv()], [_filled_kv()])
