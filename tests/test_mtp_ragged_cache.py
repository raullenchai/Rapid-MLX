# SPDX-License-Identifier: Apache-2.0
"""Pure/mock coverage for the mlx-lm 0.31.x ragged rollback adapter."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest

from vllm_mlx.spec_decode.mtp import ragged_cache as ragged
from vllm_mlx.spec_decode.mtp.continuous_engine import SelfMTPCachePair
from vllm_mlx.spec_decode.mtp.ragged_cache import (
    RaggedCacheUnsupportedError,
    RapidRaggedCacheAdapter,
    install_ragged_cache_rollback,
    preflight_ragged_cache,
    trim_ragged_cache,
)


class MergeableLayerCache:
    def __init__(self, label, rows=None):
        self.label = label
        self.rows = list([label] if rows is None else rows)
        self.events = []

    @classmethod
    def merge(cls, caches):
        merged = cls("merged", [])
        for cache in caches:
            merged.rows.extend(cache.rows)
        return merged

    def extend(self, other):
        self.events.append(("extend", tuple(other.rows)))
        self.rows.extend(other.rows)

    def extract(self, index):
        self.events.append(("extract", index))
        return type(self)(f"{self.label}:{index}", [self.rows[index]])

    def filter(self, indices):
        self.events.append(("filter", tuple(indices)))
        self.rows = [self.rows[index] for index in indices]


class Rows:
    def __init__(self, rows):
        self.rows = list(rows)
        self.shape = (len(self.rows), 1)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return Rows(self.rows[item])
        return self.rows[item]

    def __eq__(self, other):
        return isinstance(other, Rows) and self.rows == other.rows


class Vector:
    def __init__(self, values):
        self.values = list(values)

    def tolist(self):
        return list(self.values)

    def __sub__(self, other):
        values = (
            other.values if isinstance(other, Vector) else [other] * len(self.values)
        )
        return Vector([left - right for left, right in zip(self.values, values)])

    def __add__(self, other):
        values = (
            other.values if isinstance(other, Vector) else [other] * len(self.values)
        )
        return Vector([left + right for left, right in zip(self.values, values)])


class FakeOps:
    def __init__(self):
        self.rolls = []

    @staticmethod
    def vector(values):
        return Vector(values)

    @staticmethod
    def tolist(value):
        return value.tolist()

    @staticmethod
    def concat(rows):
        result = []
        for row in rows:
            result.extend(row.rows)
        return Rows(result)

    def roll_rows(self, value, shifts, *, axis, stop):
        self.rolls.append((value, shifts.tolist(), axis, stop))
        return ("rolled", value, tuple(shifts.tolist()), axis, stop)


class FakeArraysCache:
    rollback_state = None

    def __init__(self, cache):
        self.cache = cache
        self.rollback_state = None

    @property
    def batch_size(self):
        return self.cache[0].shape[0]

    def trim(self, count):
        return count


class FakeBatchKVCache:
    def __init__(self):
        self.keys = "keys"
        self.values = "values"
        self.offset = Vector([8, 6])
        self.left_padding = Vector([0, 2])
        self._idx = 8
        self._right_padding = None

    def trim(self, count):
        self._idx -= count
        self.offset = self.offset - count
        return count

    def _invalidate_attention_groups(self):
        self.invalidations = getattr(self, "invalidations", 0) + 1


class FakeQwen4StateCache(FakeArraysCache):
    def __init__(self, cache):
        super().__init__(cache)
        self._rollback_slots = None

    def restore_rollback(self, n_to_drop, verify_size):
        self.scalar_restore = (n_to_drop, verify_size)


class FakeQSAIndexCache(FakeArraysCache):
    def __init__(self):
        super().__init__([Rows(["raw-a", "raw-b"])])
        self._offsets = [9, 8]
        self._compressed_counts = [2, 2]
        self.compress_ratio = 4
        self._right_padding = None
        self.lengths = None

    @staticmethod
    def _can_trim_row(offset, count):
        if count < 0 or count > offset:
            return False
        if count == 0:
            return True
        remainder = offset % 4
        available = remainder if remainder else min(4, offset)
        return count <= available

    def trim(self, count):
        if not all(self._can_trim_row(offset, count) for offset in self._offsets):
            return 0
        self._offsets = [offset - count for offset in self._offsets]
        return count


def _install(ops=None, **kwargs):
    arrays = kwargs.pop("arrays", type("TestArraysCache", (FakeArraysCache,), {}))
    batch_kv = kwargs.pop("batch_kv", type("TestBatchKVCache", (FakeBatchKVCache,), {}))
    module = kwargs.pop(
        "module", SimpleNamespace(ArraysCache=arrays, BatchKVCache=batch_kv)
    )
    return install_ragged_cache_rollback(
        mlx_lm_version=kwargs.pop("version", "0.31.3"),
        cache_module=module,
        qwen4_state_cls=kwargs.pop("qwen", None),
        qsa_cls=kwargs.pop("qsa", None),
        array_ops=ops or FakeOps(),
        **kwargs,
    )


@pytest.mark.parametrize("version", ["0.31.2", "0.32.0", "main"])
def test_installer_is_strictly_version_gated(version):
    with pytest.raises(RaggedCacheUnsupportedError):
        _install(version=version)


def test_install_is_idempotent_and_preserves_scalar_methods():
    class Arrays(FakeArraysCache):
        pass

    class Batch(FakeBatchKVCache):
        pass

    class Qwen(FakeQwen4StateCache, Arrays):
        pass

    module = SimpleNamespace(ArraysCache=Arrays, BatchKVCache=Batch)
    ops = FakeOps()
    scalar_kv = Batch.trim
    scalar_qwen = Qwen.restore_rollback
    first = _install(ops, module=module, qwen=Qwen)
    second = _install(ops, module=module, qwen=Qwen)

    assert first.patched
    assert not second.patched
    assert second.already_present
    assert Batch.trim is scalar_kv
    assert Qwen.restore_rollback is scalar_qwen


def test_installer_refuses_to_replace_an_unknown_existing_method():
    class ConflictingArrays(FakeArraysCache):
        def trim_ragged(self, values, **kwargs):
            return values

    with pytest.raises(RaggedCacheUnsupportedError, match="refusing to replace"):
        _install(arrays=ConflictingArrays, qwen=None, qsa=None)


def test_installer_rejects_qwen_cache_from_a_different_runtime():
    class Arrays(FakeArraysCache):
        pass

    class ForeignQwen:
        pass

    with pytest.raises(RaggedCacheUnsupportedError, match="not an ArraysCache"):
        _install(arrays=Arrays, qwen=ForeignQwen, qsa=None)


def test_batch_kv_splits_uniform_cursor_move_from_residual_row_roll():
    ops = FakeOps()

    class Arrays(FakeArraysCache):
        pass

    class Batch(FakeBatchKVCache):
        pass

    _install(ops, arrays=Arrays, batch_kv=Batch, qwen=None, qsa=None)
    cache = Batch()
    assert cache.trim_ragged([1, 3], verify_size=3) == [1, 3]

    assert cache._idx == 7
    assert cache.offset.tolist() == [7, 3]
    assert cache.left_padding.tolist() == [0, 4]
    assert len(ops.rolls) == 2
    assert ops.rolls[0][1:] == ([0, 2], 2, 7)
    assert cache.invalidations == 1


def test_unknown_batch_kv_subclass_cannot_silently_inherit_adapter():
    class Arrays(FakeArraysCache):
        pass

    class Batch(FakeBatchKVCache):
        pass

    _install(FakeOps(), arrays=Arrays, batch_kv=Batch, qwen=None, qsa=None)

    class UndeclaredLedgerCache(Batch):
        pass

    with pytest.raises(RaggedCacheUnsupportedError, match="without declaring"):
        UndeclaredLedgerCache().preflight_ragged_trim([1, 2], verify_size=2)


def test_declared_batch_kv_auxiliary_ledger_rolls_with_kv():
    ops = FakeOps()

    class Arrays(FakeArraysCache):
        pass

    class Batch(FakeBatchKVCache):
        pass

    _install(ops, arrays=Arrays, batch_kv=Batch, qwen=None, qsa=None)

    class DeclaredLedgerCache(Batch):
        _RAGGED_TRIM_AUX_ARRAYS = (("ledger", 1),)

        def __init__(self):
            super().__init__()
            self.ledger = type("Ledger", (), {"shape": (2, 8)})()

    cache = DeclaredLedgerCache()
    cache.trim_ragged([1, 2], verify_size=2)
    assert ops.rolls[-1][0] is not None
    assert ops.rolls[-1][2:] == (1, 7)


def test_batch_kv_preflight_rejects_pending_padding_and_overtrim():
    class Arrays(FakeArraysCache):
        pass

    class Batch(FakeBatchKVCache):
        pass

    _install(FakeOps(), arrays=Arrays, batch_kv=Batch, qwen=None, qsa=None)
    cache = Batch()
    cache._right_padding = Vector([0, 1])
    with pytest.raises(RaggedCacheUnsupportedError, match="finalize"):
        cache.preflight_ragged_trim([1, 1], verify_size=2)
    cache._right_padding = None
    with pytest.raises(ValueError, match="before token zero"):
        cache.preflight_ragged_trim([1, 7], verify_size=7)


def test_arrays_cache_selects_each_rows_exact_verify_boundary():
    class Arrays(FakeArraysCache):
        pass

    _install(FakeOps(), arrays=Arrays, qwen=None, qsa=None)
    cache = Arrays([Rows(["a-live", "b-live"]), Rows(["A-live", "B-live"])])
    cache.rollback_state = [
        [Rows(["a-keep1", "b-keep1"]), Rows(["A-keep1", "B-keep1"])],
        [Rows(["a-keep2", "b-keep2"]), Rows(["A-keep2", "B-keep2"])],
    ]

    assert cache.trim_ragged([2, 0], verify_size=3) == [2, 0]
    assert cache.cache == [
        Rows(["a-keep1", "b-live"]),
        Rows(["A-keep1", "B-live"]),
    ]
    assert cache.rollback_state is None


def test_qwen4_refuses_partially_staged_atomic_state():
    class Arrays(FakeArraysCache):
        pass

    class Qwen(FakeQwen4StateCache, Arrays):
        pass

    _install(FakeOps(), arrays=Arrays, qwen=Qwen, qsa=None)
    cache = Qwen([Rows(["a", "b"])])
    cache.rollback_state = [[Rows(["a0", "b0"])]]
    cache._rollback_slots = {0: [Rows(["staged-a", "staged-b"])]}
    with pytest.raises(RaggedCacheUnsupportedError, match="partially staged"):
        cache.preflight_ragged_trim([1, 1], verify_size=2)


def test_qsa_rewinds_logical_rows_only_within_retained_raw_group():
    class Arrays(FakeArraysCache):
        pass

    class QSA(FakeQSAIndexCache, Arrays):
        pass

    _install(FakeOps(), arrays=Arrays, qwen=None, qsa=QSA)
    cache = QSA()
    assert cache.trim_ragged([1, 3], verify_size=4) == [1, 3]
    assert cache._offsets == [8, 5]
    assert cache._compressed_counts == [2, 1]

    cache._offsets = [9, 8]
    with pytest.raises(RaggedCacheUnsupportedError, match="raw-ring history"):
        cache.preflight_ragged_trim([2, 1], verify_size=3)


@pytest.mark.parametrize(
    "cache_type, message",
    [("BatchQuantizedKVCache", "quantized"), ("BatchRotatingKVCache", "windowed")],
)
def test_public_preflight_fails_loud_for_unsupported_cache_families(
    cache_type, message
):
    cache = type(cache_type, (), {})()
    with pytest.raises(RaggedCacheUnsupportedError, match=message):
        preflight_ragged_cache(cache, [1, 1], verify_size=2)


def test_cache_tree_preflights_every_member_before_mutating_anything():
    class Arrays(FakeArraysCache):
        pass

    class Batch(FakeBatchKVCache):
        pass

    _install(FakeOps(), arrays=Arrays, batch_kv=Batch, qwen=None, qsa=None)
    first = Batch()
    unsupported = type("QuantizedKVCache", (), {})()
    tree = SimpleNamespace(caches=(first, unsupported))

    with pytest.raises(RaggedCacheUnsupportedError, match="quantized"):
        trim_ragged_cache(tree, [1, 1], verify_size=2)
    assert first._idx == 8
    assert first.offset.tolist() == [8, 6]


def test_supported_cache_tree_applies_after_successful_preflight():
    class Arrays(FakeArraysCache):
        pass

    class Batch(FakeBatchKVCache):
        pass

    _install(FakeOps(), arrays=Arrays, batch_kv=Batch, qwen=None, qsa=None)
    first, second = Batch(), Batch()
    tree = SimpleNamespace(caches=(first, second))
    assert trim_ragged_cache(tree, [1, 1], verify_size=2) == [1, 1]
    assert first._idx == second._idx == 7


def test_transaction_adapter_merge_rollback_extract_and_extend():
    events = []
    adapter = RapidRaggedCacheAdapter(
        preflight=lambda group, drops, **kwargs: events.append(
            ("preflight", tuple(drops), kwargs)
        ),
        trim=lambda group, drops, **kwargs: events.append(
            ("trim", tuple(drops), kwargs)
        ),
    )
    one = SelfMTPCachePair([MergeableLayerCache("t1")], [MergeableLayerCache("d1")])
    two = SelfMTPCachePair([MergeableLayerCache("t2")], [MergeableLayerCache("d2")])
    merged = adapter.attach(None, [one, two])
    assert merged.target[0].rows == ["t1", "t2"]
    assert merged.draft[0].rows == ["d1", "d2"]

    adapter.rollback(
        merged,
        target_drops=[1, 0],
        draft_drops=[2, 2],
        verify_width=3,
    )
    assert [event[0] for event in events] == [
        "preflight",
        "preflight",
        "trim",
        "trim",
    ]
    remaining, detached = adapter.detach(merged, [1], [0])
    assert remaining.target[0].rows == ["t1"]
    assert detached[0].target[0].rows == ["t2"]

    third = SelfMTPCachePair([MergeableLayerCache("t3")], [MergeableLayerCache("d3")])
    adapter.attach(remaining, [third])
    assert remaining.target[0].rows == ["t1", "t3"]


@pytest.mark.parametrize("name", ["QuantizedKVCache", "SinkWindowKVCache"])
def test_transaction_adapter_rejects_unsupported_cache_classes(name):
    unsupported = type(name, (MergeableLayerCache,), {})
    pair = SelfMTPCachePair([unsupported("target")], [MergeableLayerCache("draft")])
    adapter = RapidRaggedCacheAdapter(
        preflight=lambda *args, **kwargs: None,
        trim=lambda *args, **kwargs: None,
    )

    with pytest.raises(RaggedCacheUnsupportedError, match="no supported"):
        adapter.attach(None, [pair])


def test_transaction_adapter_attach_failure_leaves_live_pair_unchanged():
    class FailingReplacementMerge(MergeableLayerCache):
        @classmethod
        def merge(cls, caches):
            if len(caches) > 1:
                raise MemoryError("replacement allocation failed")
            return super().merge(caches)

    adapter = RapidRaggedCacheAdapter(
        preflight=lambda *args, **kwargs: None,
        trim=lambda *args, **kwargs: None,
    )
    current = SelfMTPCachePair(
        target=[
            MergeableLayerCache("t-live"),
            FailingReplacementMerge("t-fail"),
        ],
        draft=[
            MergeableLayerCache("d-live"),
            FailingReplacementMerge("d-fail"),
        ],
    )
    joining = SelfMTPCachePair(
        target=[
            MergeableLayerCache("t-new"),
            FailingReplacementMerge("t-new-fail"),
        ],
        draft=[
            MergeableLayerCache("d-new"),
            FailingReplacementMerge("d-new-fail"),
        ],
    )
    original_target = current.target
    original_draft = current.draft

    with pytest.raises(MemoryError, match="replacement allocation failed"):
        adapter.attach(current, [joining])

    assert current.target is original_target
    assert current.draft is original_draft
    assert [cache.rows for cache in current.target] == [["t-live"], ["t-fail"]]
    assert [cache.rows for cache in current.draft] == [["d-live"], ["d-fail"]]


def test_transaction_adapter_detach_failure_leaves_live_pair_unchanged():
    class FailingExtract(MergeableLayerCache):
        def extract(self, index):
            raise MemoryError(f"extract {index} failed")

    adapter = RapidRaggedCacheAdapter(
        preflight=lambda *args, **kwargs: None,
        trim=lambda *args, **kwargs: None,
    )
    current = SelfMTPCachePair(
        target=[
            MergeableLayerCache("target", ["t0", "t1"]),
            FailingExtract("target-fail", ["tf0", "tf1"]),
        ],
        draft=[
            MergeableLayerCache("draft", ["d0", "d1"]),
            MergeableLayerCache("draft-2", ["d20", "d21"]),
        ],
    )
    original_target = current.target
    original_draft = current.draft

    with pytest.raises(MemoryError, match="extract 1 failed"):
        adapter.detach(current, [1], [0])

    assert current.target is original_target
    assert current.draft is original_draft
    assert [cache.rows for cache in current.target] == [
        ["t0", "t1"],
        ["tf0", "tf1"],
    ]
    assert [cache.rows for cache in current.draft] == [
        ["d0", "d1"],
        ["d20", "d21"],
    ]


@pytest.mark.parametrize(
    "values",
    [1, object(), [1], [True, 0], [-1, 0]],
)
def test_drop_vector_rejects_non_ragged_or_invalid_counts(values):
    with pytest.raises((TypeError, ValueError)):
        ragged._drop_vector(values, 2, "test")


def test_drop_vector_accepts_array_like_values():
    assert ragged._drop_vector(Vector([1, 0]), 2, "test") == [1, 0]


def test_default_array_ops_surface_and_roll_assignment():
    class Shift:
        def reshape(self, shape):
            self.shape = shape
            return self

    class Tensor:
        def __init__(self):
            self.assigned = None

        def __getitem__(self, key):
            return ("window", key)

        def __setitem__(self, key, value):
            self.assigned = (key, value)

    class MX:
        @staticmethod
        def array(values):
            return ("vector", tuple(values))

        @staticmethod
        def concatenate(rows, axis):
            return ("concat", tuple(rows), axis)

    module = SimpleNamespace(
        mx=MX,
        dynamic_roll=lambda value, shifts, axis: ("roll", value, shifts, axis),
    )
    ops = ragged._DefaultArrayOps(module)
    assert ops.vector([1, 2]) == ("vector", (1, 2))
    assert ops.tolist(Vector([1, 2])) == [1, 2]
    assert ops.concat(["a", "b"]) == ("concat", ("a", "b"), 0)
    assert ops.roll_rows(None, Shift(), axis=1, stop=2) is None
    tensor = Tensor()
    assert ops.roll_rows(tensor, Shift(), axis=1, stop=2) is tensor
    assert tensor.assigned is not None


def test_arrays_preflight_and_snapshot_failure_contracts():
    class Arrays(FakeArraysCache):
        pass

    _install(FakeOps(), arrays=Arrays, qwen=None, qsa=None)
    cache = Arrays([Rows(["a", "b"])])
    with pytest.raises(ValueError, match="verify_size"):
        cache.preflight_ragged_trim([1, 1], verify_size=True)
    with pytest.raises(ValueError, match="verify_size"):
        cache.preflight_ragged_trim([1, 1], verify_size=0)
    with pytest.raises(RaggedCacheUnsupportedError, match="no rollback"):
        cache.preflight_ragged_trim([1, 1], verify_size=2)
    cache.rollback_state = []
    with pytest.raises(RaggedCacheUnsupportedError, match="empty"):
        cache.preflight_ragged_trim([1, 1], verify_size=2)
    cache.rollback_state = [[Rows(["a", "b"])], []]
    with pytest.raises(RaggedCacheUnsupportedError, match="diverge"):
        cache.preflight_ragged_trim([1, 1], verify_size=2)
    cache.rollback_state = [[Rows(["a", "b"])]]
    with pytest.raises(RaggedCacheUnsupportedError, match="boundary"):
        cache.preflight_ragged_trim([2, 1], verify_size=2)
    cache.rollback_state = [[None]]
    with pytest.raises(RaggedCacheUnsupportedError, match="is None"):
        cache.preflight_ragged_trim([1, 1], verify_size=2)
    cache.rollback_state = [[Rows(["only-one-row"])]]
    with pytest.raises(RaggedCacheUnsupportedError, match="does not cover"):
        cache.preflight_ragged_trim([1, 1], verify_size=2)


def test_arrays_legacy_snapshot_and_zero_drop_paths():
    class Arrays(FakeArraysCache):
        pass

    _install(FakeOps(), arrays=Arrays, qwen=None, qsa=None)
    cache = Arrays([Rows(["a", "b"])])
    cache.rollback_state = (Rows(["old-a", "old-b"]),)
    with pytest.raises(RaggedCacheUnsupportedError, match="only rewind one"):
        cache.preflight_ragged_trim([2, 1], verify_size=3)
    assert cache.trim_ragged([1, 0], verify_size=2) == [1, 0]
    assert cache.cache == [Rows(["old-a", "b"])]
    cache.rollback_state = [[Rows(["a0", "b0"])]]
    assert cache.trim_ragged([0, 0], verify_size=1) == [0, 0]
    assert cache.rollback_state is None


def test_batch_kv_additional_preflight_and_aux_hook_paths():
    ops = FakeOps()

    class Arrays(FakeArraysCache):
        pass

    class Batch(FakeBatchKVCache):
        pass

    _install(ops, arrays=Arrays, batch_kv=Batch, qwen=None, qsa=None)
    cache = Batch()
    with pytest.raises(ValueError, match="shared KV cursor"):
        cache.preflight_ragged_trim([9, 0], verify_size=9, validate=False)

    class ShortLedger(Batch):
        _RAGGED_TRIM_AUX_ARRAYS = (("ledger", 1),)

        def __init__(self):
            super().__init__()
            self.ledger = type("Ledger", (), {"shape": (2, 1)})()

    with pytest.raises(RaggedCacheUnsupportedError, match="does not reach"):
        ShortLedger().preflight_ragged_trim([1, 2], verify_size=2)

    calls = []

    class Hooked(Batch):
        def _trim_ragged_aux(self, shifts, *, stop, array_ops):
            calls.append((shifts.tolist(), stop, array_ops))

    Hooked().trim_ragged([1, 2], verify_size=2)
    assert calls == [([0, 1], 7, ops)]


def test_qsa_padding_requires_finalize():
    class Arrays(FakeArraysCache):
        pass

    class QSA(FakeQSAIndexCache, Arrays):
        pass

    _install(FakeOps(), arrays=Arrays, qwen=None, qsa=QSA)
    cache = QSA()
    cache.lengths = [9, 8]
    with pytest.raises(RaggedCacheUnsupportedError, match="finalize"):
        cache.preflight_ragged_trim([1, 1], verify_size=2)


def test_qwen4_valid_preflight_delegates_to_arrays_contract():
    class Arrays(FakeArraysCache):
        pass

    class Qwen(FakeQwen4StateCache, Arrays):
        pass

    _install(FakeOps(), arrays=Arrays, qwen=Qwen, qsa=None)
    cache = Qwen([Rows(["a", "b"])])
    cache.rollback_state = [[Rows(["a0", "b0"])]]
    assert cache.preflight_ragged_trim([1, 1], verify_size=2) == [1, 1]


def test_installer_rejects_missing_classes_and_different_ops():
    with pytest.raises(RaggedCacheUnsupportedError, match="lacks"):
        _install(module=SimpleNamespace(), qwen=None, qsa=None)

    class Arrays(FakeArraysCache):
        pass

    class Batch(FakeBatchKVCache):
        pass

    module = SimpleNamespace(ArraysCache=Arrays, BatchKVCache=Batch)
    _install(FakeOps(), module=module, qwen=None, qsa=None)
    with pytest.raises(RaggedCacheUnsupportedError, match="different"):
        _install(FakeOps(), module=module, qwen=None, qsa=None)


def test_installer_discovers_default_cache_classes(monkeypatch):
    class Arrays(FakeArraysCache):
        pass

    class Batch(FakeBatchKVCache):
        pass

    class Qwen(FakeQwen4StateCache, Arrays):
        pass

    class QSA(FakeQSAIndexCache, Arrays):
        pass

    cache_module = SimpleNamespace(ArraysCache=Arrays, BatchKVCache=Batch)
    mlx_models = types.ModuleType("mlx_lm.models")
    mlx_models.cache = cache_module
    qwen_module = types.ModuleType("vllm_mlx.models.qwen4_exp_cache")
    qwen_module.Qwen4ExpStateCache = Qwen
    qwen_module.QSAIndexCache = QSA
    monkeypatch.setitem(sys.modules, "mlx_lm.models", mlx_models)
    monkeypatch.setitem(sys.modules, "vllm_mlx.models.qwen4_exp_cache", qwen_module)

    report = install_ragged_cache_rollback(mlx_lm_version="0.31.3", array_ops=FakeOps())
    assert report.patched


def test_call_ragged_method_without_verify_and_uninspectable_signature():
    calls = []

    def without_verify(values, *, validate):
        calls.append((values, validate))
        return values

    assert ragged._call_ragged_method(
        without_verify, [1], verify_size=2, validate=True
    ) == [1]
    assert calls == [([1], True)]
    with pytest.raises(TypeError):
        ragged._call_ragged_method(object(), [1], verify_size=2, validate=True)


def test_public_preflight_rejects_unadapted_leaf_and_accepts_list_tree():
    with pytest.raises(RaggedCacheUnsupportedError, match="no ragged"):
        preflight_ragged_cache(object(), [1], verify_size=1)
    leaf = SimpleNamespace(
        preflight_ragged_trim=lambda values, *, validate: values,
        trim_ragged=lambda values, *, validate: values,
    )
    assert trim_ragged_cache([leaf], [1], verify_size=1) == [1]


def test_transaction_adapter_validation_and_empty_paths():
    with pytest.raises(TypeError, match="hooks"):
        RapidRaggedCacheAdapter(preflight=object())
    adapter = RapidRaggedCacheAdapter(
        preflight=lambda *args, **kwargs: None,
        trim=lambda *args, **kwargs: None,
    )
    with pytest.raises(ValueError, match="non-empty"):
        adapter.attach(None, [SelfMTPCachePair([], [MergeableLayerCache("d")])])
    with pytest.raises(ValueError, match="no cache rows"):
        adapter.attach(None, [])
    current = SelfMTPCachePair([MergeableLayerCache("t")], [MergeableLayerCache("d")])
    assert adapter.attach(current, []) is current
    with pytest.raises(ValueError, match="empty target"):
        adapter._merge([], "target")
    with pytest.raises(ValueError, match="equal non-zero width"):
        adapter._merge([[], []], "target")

    uneven = SelfMTPCachePair(
        [MergeableLayerCache("t1"), MergeableLayerCache("t2")],
        [MergeableLayerCache("d")],
    )
    with pytest.raises(ValueError, match="widths differ"):
        adapter.attach(current, [uneven])


def test_transaction_adapter_rejects_mixed_missing_merge_and_extract_surfaces():
    class OtherMergeable(MergeableLayerCache):
        pass

    class NoMerge:
        def extract(self, index):
            return self

    adapter = RapidRaggedCacheAdapter(
        preflight=lambda *args, **kwargs: None,
        trim=lambda *args, **kwargs: None,
    )
    mixed = [
        SelfMTPCachePair([MergeableLayerCache("a")], [MergeableLayerCache("d")]),
        SelfMTPCachePair([OtherMergeable("b")], [MergeableLayerCache("e")]),
    ]
    with pytest.raises(Exception, match="mixed target"):
        adapter.attach(None, mixed)
    no_merge = SelfMTPCachePair([NoMerge()], [NoMerge()])
    with pytest.raises(Exception, match="no merge"):
        adapter.attach(None, [no_merge, no_merge])
    no_extract = SelfMTPCachePair([object()], [object()])
    with pytest.raises(Exception, match="no extract"):
        adapter.detach(no_extract, [0], [])


def test_transaction_adapter_detach_empty_keep_and_multi_keep_missing_merge():
    adapter = RapidRaggedCacheAdapter(
        preflight=lambda *args, **kwargs: None,
        trim=lambda *args, **kwargs: None,
    )
    pair = SelfMTPCachePair(
        [MergeableLayerCache("t", ["t0", "t1"])],
        [MergeableLayerCache("d", ["d0", "d1"])],
    )
    remaining, detached = adapter.detach(pair, [1], [])
    assert remaining.target == [] and remaining.draft == []
    assert detached[0].target[0].rows == ["t1"]

    pair = SelfMTPCachePair(
        [MergeableLayerCache("t", ["t0", "t1", "t2"])],
        [MergeableLayerCache("d", ["d0", "d1", "d2"])],
    )
    remaining, _ = adapter.detach(pair, [2], [0, 1])
    assert remaining.target[0].rows == ["t0", "t1"]

    class ExtractOnly:
        def extract(self, index):
            return type(self)()

    pair = SelfMTPCachePair([ExtractOnly()], [ExtractOnly()])
    with pytest.raises(Exception, match="no merge"):
        adapter.detach(pair, [], [0, 1])
