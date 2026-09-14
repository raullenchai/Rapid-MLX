# SPDX-License-Identifier: Apache-2.0
"""Recurrent-state checkpoints for hybrid prompt caches.

Hybrid (GatedDeltaNet / Mamba) entries can only be resumed at the exact
length they were captured at, so ``MemoryAwarePrefixCache`` used to refuse
every trim-requiring match (supersequence / LCP) for them. With checkpoints
recorded at prefill chunk boundaries, fetch snaps the trim to the newest
checkpoint at or below the shared prefix instead.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

# The module positively identifies mlx-lm's ``ArraysCache``; every test here
# builds on that class, so the whole file rides the no-MLX auto-skip.
pytestmark = pytest.mark.requires_mlx
ArraysCache = pytest.importorskip("mlx_lm.models.cache").ArraysCache

from vllm_mlx.hybrid_state_checkpoints import (  # noqa: E402
    CHECKPOINT_ATTR,
    StateCheckpoints,
    achievable_position,
    attach_checkpoints,
    checkpoint_bytes,
    collect_checkpoints,
    is_recurrent_layer,
    layer_checkpoints,
    record_checkpoints,
    restore_recurrent_layer,
)


class _Array:
    """Shape/dtype-only stand-in for an MLX array (no MLX needed)."""

    def __init__(self, tag: int, shape=(1, 4, 4)):
        self.tag = tag
        self.shape = shape
        self.dtype = SimpleNamespace(size=2)


class _RecurrentLayer(ArraysCache):
    """A real ``ArraysCache`` (the only class the module accepts) holding
    lightweight fake arrays so the bookkeeping tests need no GPU work."""

    def __init__(self, tag: int):
        super().__init__(2)
        self.cache = [_Array(tag), _Array(tag, (1, 2, 2))]


class _LookAlikeLayer:
    """Duck-types ``ArraysCache`` but is not one; must be refused."""

    def __init__(self):
        self.cache = [_Array(0)]

    def is_trimmable(self) -> bool:
        return False


class _KVLayer:
    def __init__(self):
        self.offset = 0

    def is_trimmable(self) -> bool:
        return True

    def trim(self, n):
        return n


def _cache(tag: int):
    return [_KVLayer(), _RecurrentLayer(tag), _RecurrentLayer(tag)]


class TestStateCheckpoints:
    def test_records_sorted_and_respects_stride(self):
        holder = StateCheckpoints()
        holder = holder.with_checkpoint(2048, (_Array(1),), max_count=4, stride=2048)
        same = holder.with_checkpoint(2100, (_Array(2),), max_count=4, stride=2048)
        assert same is holder  # too close to the newest checkpoint
        older = holder.with_checkpoint(1000, (_Array(3),), max_count=4, stride=2048)
        assert older is holder  # never records behind the newest
        holder = holder.with_checkpoint(4096, (_Array(4),), max_count=4, stride=2048)
        assert holder.positions == (2048, 4096)

    def test_thinning_drops_smallest_gap_but_keeps_newest(self):
        holder = StateCheckpoints()
        for pos in (2048, 4096, 6144, 8192):
            holder = holder.with_checkpoint(
                pos, (_Array(pos),), max_count=3, stride=2048
            )
        assert holder.positions == (4096, 6144, 8192)
        holder = holder.with_checkpoint(
            10240, (_Array(10240),), max_count=3, stride=2048
        )
        assert holder.positions == (4096, 8192, 10240)
        assert holder.positions[-1] == 10240

    def test_max_count_zero_disables_recording(self):
        holder = StateCheckpoints().with_checkpoint(2048, (), max_count=0, stride=1)
        assert len(holder) == 0

    def test_lookup_truncate_and_bytes(self):
        holder = StateCheckpoints([(2048, (_Array(1),)), (4096, (_Array(2),))])
        assert holder.newest_at_or_below(3000) == 2048
        assert holder.newest_at_or_below(4096) == 4096
        assert holder.newest_at_or_below(100) == 0
        assert holder.arrays_at(4096)[0].tag == 2
        assert holder.arrays_at(5) is None
        assert holder.truncated(2048).positions == (2048,)
        assert holder.nbytes == 2 * (1 * 4 * 4 * 2)

    def test_copies_share_the_holder(self):
        holder = StateCheckpoints([(2048, (_Array(1),))])
        assert copy.deepcopy(holder) is holder
        assert copy.copy(holder) is holder


class TestRecordAndRestore:
    def test_is_recurrent_layer_classification(self):
        assert is_recurrent_layer(_RecurrentLayer(0))
        assert not is_recurrent_layer(_KVLayer())
        assert not is_recurrent_layer(_LookAlikeLayer())
        assert not is_recurrent_layer(None)

    def test_record_keeps_every_recurrent_layer_in_lockstep(self):
        cache = _cache(0)
        holders = collect_checkpoints(cache)
        assert holders == [None, None, None]
        assert record_checkpoints(cache, holders, 2048, max_count=4, stride=2048)
        cache = _cache(1)
        assert record_checkpoints(cache, holders, 4096, max_count=4, stride=2048)
        assert not record_checkpoints(cache, holders, 4100, max_count=4, stride=2048)
        assert holders[0] is None
        assert holders[1].positions == holders[2].positions == (2048, 4096)
        assert holders[1].arrays_at(4096)[0].tag == 1

    def test_record_refuses_unmaterialised_state(self):
        cache = _cache(0)
        cache[2].cache[0] = None
        holders = collect_checkpoints(cache)
        assert not record_checkpoints(cache, holders, 2048, max_count=4, stride=1)
        assert holders == [None, None, None]

    def test_record_drops_checkpoint_when_materialisation_fails(self, monkeypatch):
        import mlx.core as mx

        def _boom(*_args, **_kwargs):
            raise RuntimeError("Metal allocation failed")

        monkeypatch.setattr(mx, "eval", _boom)
        cache = _cache(0)
        holders = collect_checkpoints(cache)
        assert not record_checkpoints(cache, holders, 2048, max_count=4, stride=1)
        assert holders == [None, None, None]

    def test_attach_truncates_to_the_stored_length(self):
        cache = _cache(0)
        holders = collect_checkpoints(cache)
        for pos in (2048, 4096, 6144):
            assert record_checkpoints(cache, holders, pos, max_count=4, stride=1)

        boundary_entry = _cache(1)
        attach_checkpoints(boundary_entry, holders, max_position=5000)
        assert layer_checkpoints(boundary_entry[1]).positions == (2048, 4096)
        # The live holders are untouched: later chunks keep extending them.
        assert holders[1].positions == (2048, 4096, 6144)

        # A layer copied from a checkpoint-bearing cache carries the old
        # holder; attaching an empty/absent one must clear it.
        short_entry = _cache(2)
        attach_checkpoints(short_entry, holders)
        assert layer_checkpoints(short_entry[1]).positions == (2048, 4096, 6144)
        attach_checkpoints(short_entry, holders, max_position=1000)
        assert layer_checkpoints(short_entry[1]) is None
        assert not hasattr(short_entry[1], CHECKPOINT_ATTR)
        attach_checkpoints(short_entry, holders)
        attach_checkpoints(short_entry, [None, None, None])
        assert not hasattr(short_entry[1], CHECKPOINT_ATTR)

    def test_record_needs_aligned_holders_and_positive_position(self):
        cache = _cache(0)
        assert not record_checkpoints(cache, [None], 2048, max_count=4, stride=1)
        holders = collect_checkpoints(cache)
        assert not record_checkpoints(cache, holders, 0, max_count=4, stride=1)

    def test_attach_achievable_and_restore(self):
        cache = _cache(0)
        holders = collect_checkpoints(cache)
        for pos in (2048, 4096):
            record_checkpoints(_cache(pos), holders, pos, max_count=4, stride=2048)
        attach_checkpoints(cache, holders)
        assert layer_checkpoints(cache[1]) is holders[1]
        assert layer_checkpoints(cache[0]) is None
        assert achievable_position(cache, 5000) == 4096
        assert achievable_position(cache, 4096) == 4096
        assert achievable_position(cache, 2047) == 0
        assert achievable_position([_KVLayer()], 777) == 777
        assert checkpoint_bytes(cache) == 2 * holders[1].nbytes

        restored = restore_recurrent_layer(cache[1], 2048)
        assert restored is not cache[1]
        assert restored.cache[0].tag == 2048
        assert layer_checkpoints(restored).positions == (2048,)
        assert cache[1].cache[0].tag == 0  # source untouched
        assert restore_recurrent_layer(cache[1], 3000) is None

    def test_layers_with_disjoint_checkpoints_have_no_common_position(self):
        cache = _cache(0)
        setattr(cache[1], CHECKPOINT_ATTR, StateCheckpoints([(2048, (_Array(1),))]))
        setattr(cache[2], CHECKPOINT_ATTR, StateCheckpoints([(4096, (_Array(1),))]))
        assert achievable_position(cache, 9000) == 0

    def test_deepcopy_of_layer_shares_holder(self):
        cache = _cache(0)
        holders = collect_checkpoints(cache)
        record_checkpoints(cache, holders, 2048, max_count=4, stride=1)
        attach_checkpoints(cache, holders)
        clone = copy.deepcopy(cache)
        assert layer_checkpoints(clone[1]) is layer_checkpoints(cache[1])


# ---------------------------------------------------------------------------
# Fetch-side snap (real mlx-lm cache classes)
# ---------------------------------------------------------------------------


class TestGuards:
    """Every refusal branch records or restores nothing."""

    def test_env_knobs_fall_back_on_garbage(self, monkeypatch):
        from vllm_mlx import hybrid_state_checkpoints as hsc

        monkeypatch.setenv("RAPID_MLX_HYBRID_CHECKPOINT_MAX", "four")
        monkeypatch.setenv("RAPID_MLX_HYBRID_CHECKPOINT_STRIDE", "  ")
        assert hsc.checkpoint_max() == hsc._DEFAULT_MAX
        assert hsc.checkpoint_stride() == hsc._DEFAULT_STRIDE
        monkeypatch.setenv("RAPID_MLX_HYBRID_CHECKPOINT_STRIDE", "0")
        assert hsc.checkpoint_stride() == 1

    def test_array_bytes_fallbacks(self):
        from vllm_mlx.hybrid_state_checkpoints import _array_bytes

        assert _array_bytes(None) == 0
        assert _array_bytes(SimpleNamespace(nbytes=24)) == 24
        assert _array_bytes(object()) == 0
        assert _array_bytes(_Array(0, (2, 3))) == 12

    def test_without_mlx_lm_nothing_is_recurrent(self, monkeypatch):
        import sys

        from vllm_mlx import hybrid_state_checkpoints as hsc

        monkeypatch.setattr(hsc, "_RECURRENT_TYPES", None)
        monkeypatch.setitem(sys.modules, "mlx_lm.models.cache", None)
        monkeypatch.setitem(sys.modules, "mlx_vlm.models.cache", None)
        assert hsc._recurrent_cache_types() == ()
        assert not hsc.is_recurrent_layer(_RecurrentLayer(0))

    def test_mlx_vlm_arrays_cache_is_recurrent(self, monkeypatch):
        """The MLLM lane stores mlx-vlm's own ``ArraysCache`` through its
        exact APC; checkpoints must record and restore on that class too."""
        vlm_cache = pytest.importorskip("mlx_vlm.models.cache")
        from vllm_mlx import hybrid_state_checkpoints as hsc

        monkeypatch.setattr(hsc, "_RECURRENT_TYPES", None)
        layer = vlm_cache.ArraysCache(2)
        layer.cache = [_Array(0, (1, 4)), _Array(1, (1, 4))]
        assert hsc.is_recurrent_layer(layer)
        holders = [None]
        assert record_checkpoints([layer], holders, 8, max_count=2, stride=1)
        layer.cache = [_Array(2, (1, 4)), _Array(3, (1, 4))]
        attach_checkpoints([layer], holders)
        restored = restore_recurrent_layer(layer, 8)
        assert restored is not None
        assert [a.tag for a in restored.cache] == [0, 1]
        assert restored.left_padding is None
        assert layer.cache[0].tag == 2

    def test_attach_ignores_misaligned_holders(self):
        cache = _cache(0)
        holders = collect_checkpoints(cache)
        record_checkpoints(cache, holders, 2048, max_count=4, stride=1)
        target = _cache(1)
        attach_checkpoints(target, holders[:2])
        assert layer_checkpoints(target[1]) is None

    def test_record_without_recurrent_layers_is_a_noop(self):
        cache = [_KVLayer(), _KVLayer()]
        holders = collect_checkpoints(cache)
        assert not record_checkpoints(cache, holders, 2048, max_count=4, stride=1)
        assert holders == [None, None]

    def test_restore_without_checkpoints_returns_none(self):
        assert restore_recurrent_layer(_RecurrentLayer(0), 2048) is None
        assert restore_recurrent_layer(_KVLayer(), 2048) is None

    def test_snap_refusals(self, monkeypatch):
        from vllm_mlx import memory_cache
        from vllm_mlx.memory_cache import _snap_hybrid_trim

        cache = _cache(0)
        assert _snap_hybrid_trim(cache, 100, 0) is None
        # A non-trimmable layer that is not a real ArraysCache keeps the
        # pre-checkpoint refusal even when the real layers have checkpoints.
        liar = _cache(0) + [_LookAlikeLayer()]
        liar_holders = collect_checkpoints(liar)
        record_checkpoints(liar, liar_holders, 2048, max_count=4, stride=1)
        attach_checkpoints(liar, liar_holders)
        assert _snap_hybrid_trim(liar, 4096, 3000) is None
        assert _snap_hybrid_trim(cache, 0, 100) is None
        # Recurrent layers without any checkpoint: no position to resume at.
        assert _snap_hybrid_trim(cache, 4096, 3000) is None

        holders = collect_checkpoints(cache)
        record_checkpoints(cache, holders, 2048, max_count=4, stride=1)
        attach_checkpoints(cache, holders)
        assert _snap_hybrid_trim(cache, 4096, 3000)[1] == 2048
        # A recurrent layer whose checkpoint cannot be restored vetoes the snap.
        monkeypatch.setattr(memory_cache, "restore_recurrent_layer", lambda *_: None)
        assert _snap_hybrid_trim(cache, 4096, 3000) is None
        # So does a KV layer that cannot be rewound exactly.
        monkeypatch.setattr(memory_cache, "_trim_cache_offset", lambda *_: None)
        assert _snap_hybrid_trim(cache, 4096, 3000) is None


@pytest.mark.requires_mlx
class TestMemoryCacheSnap:
    @staticmethod
    def _entry(length: int, checkpoints: tuple[int, ...]):
        import mlx.core as mx
        from mlx_lm.models.cache import ArraysCache, KVCache

        kv = KVCache()
        kv.keys = mx.zeros((1, 2, length, 8))
        kv.values = mx.zeros((1, 2, length, 8))
        kv.offset = length
        rec = ArraysCache(2)
        rec.cache = [mx.full((1, 3, 4), length), mx.full((1, 2, 2), length)]
        holder = StateCheckpoints(
            (pos, (mx.full((1, 3, 4), pos), mx.full((1, 2, 2), pos)))
            for pos in checkpoints
        )
        setattr(rec, CHECKPOINT_ATTR, holder)
        return [kv, rec]

    @staticmethod
    def _cache():
        from vllm_mlx.memory_cache import MemoryAwarePrefixCache, MemoryCacheConfig

        config = MemoryCacheConfig(
            max_memory_mb=64, max_entries=16, hybrid_reuse_max_entries=4
        )
        return MemoryAwarePrefixCache(MagicMock(), config)

    def test_lcp_snaps_to_newest_checkpoint_below_divergence(self):
        cache = self._cache()
        stored = list(range(1000, 7000))
        assert cache.store(stored, self._entry(6000, (2048, 4096)))
        divergent = stored[:5000] + [1, 2, 3, 4, 5]

        result, remaining = cache.fetch(divergent)

        assert result is not None
        assert remaining == divergent[4096:]
        assert result[0].offset == 4096
        assert result[1].cache[0][0, 0, 0].item() == 4096
        assert layer_checkpoints(result[1]).positions == (2048, 4096)
        assert cache._last_match_type == "lcp"
        assert cache.get_stats()["tokens_saved"] == 4096

    def test_supersequence_snaps_when_no_longer_prefix_entry_exists(self):
        cache = self._cache()
        stored = list(range(1000, 7000))
        assert cache.store(stored, self._entry(6000, (2048, 4096)))

        result, remaining = cache.fetch(stored[:5000])

        assert result is not None
        assert remaining == stored[4096:5000]
        assert result[0].offset == 4096
        assert cache._last_match_type == "supersequence"

    def test_exact_boundary_entry_still_beats_a_snap(self):
        cache = self._cache()
        stored = list(range(1000, 7000))
        assert cache.store(
            stored, self._entry(6000, (2048, 4096)), evict_prefixes=False
        )
        assert cache.store(stored[:4500], self._entry(4500, ()), evict_prefixes=False)

        result, remaining = cache.fetch(stored[:5000])

        assert result is not None
        assert remaining == stored[4500:5000]
        assert cache._last_match_type == "prefix"

    def test_no_checkpoint_below_divergence_is_still_a_miss(self):
        cache = self._cache()
        stored = list(range(1000, 7000))
        assert cache.store(stored, self._entry(6000, (4096,)))

        result, remaining = cache.fetch(stored[:3000] + [7, 8, 9])

        assert result is None
        assert remaining == stored[:3000] + [7, 8, 9]

    def test_entry_memory_charges_checkpoints(self):
        from vllm_mlx.memory_cache import estimate_kv_cache_memory

        plain = self._entry(64, ())
        with_ckpt = self._entry(64, (16, 32))
        assert estimate_kv_cache_memory(with_ckpt) == estimate_kv_cache_memory(
            plain
        ) + checkpoint_bytes(with_ckpt)


# ---------------------------------------------------------------------------
# Scheduler glue
# ---------------------------------------------------------------------------


@pytest.mark.requires_mlx
class TestSchedulerRecording:
    @staticmethod
    def _scheduler(hybrid_entries: int = 8):
        pytest.importorskip("mlx")
        from vllm_mlx.scheduler import Scheduler, SchedulerConfig

        tokenizer = MagicMock()
        tokenizer.encode = lambda x: list(range(len(x.split())))
        config = SchedulerConfig(
            enable_prefix_cache=True,
            use_memory_aware_cache=True,
            hybrid_cache_entries=hybrid_entries,
        )
        scheduler = Scheduler(MagicMock(), tokenizer, config)
        scheduler.batch_generator = MagicMock()
        return scheduler

    @staticmethod
    def _register(scheduler, uid: int, cached_tokens: int = 0):
        from vllm_mlx.request import Request, SamplingParams

        request = Request(
            request_id=f"req-{uid}",
            prompt="ignored",
            prompt_token_ids=list(range(9000)),
            sampling_params=SamplingParams(max_tokens=4),
        )
        request.cached_tokens = cached_tokens
        scheduler.requests[request.request_id] = request
        scheduler.uid_to_request_id[uid] = request.request_id
        scheduler.request_id_to_uid[request.request_id] = uid
        return request

    @staticmethod
    def _response(uid: int, processed: int, *, end_of_prompt: bool = False):
        return SimpleNamespace(
            uid=uid,
            progress=(processed, 9000),
            end_of_segment=False,
            end_of_prompt=end_of_prompt,
        )

    def test_records_absolute_positions_and_attaches_before_store(self, monkeypatch):
        monkeypatch.setenv("RAPID_MLX_HYBRID_CHECKPOINT_STRIDE", "2048")
        monkeypatch.setenv("RAPID_MLX_HYBRID_CHECKPOINT_MAX", "4")
        scheduler = self._scheduler()
        self._register(scheduler, uid=7, cached_tokens=1000)
        scheduler.batch_generator.extract_cache.side_effect = lambda uids: {
            uids[0]: (_cache(uids[0]), [])
        }

        scheduler._record_hybrid_checkpoints([self._response(7, 2048)])
        scheduler._record_hybrid_checkpoints([self._response(7, 2100)])  # < stride
        scheduler._record_hybrid_checkpoints([self._response(7, 4096)])
        scheduler._record_hybrid_checkpoints(
            [self._response(7, 8000, end_of_prompt=True)]
        )

        holders = scheduler._hybrid_checkpoints[7]
        assert holders[0] is None
        assert holders[1].positions == (3048, 5096)
        assert scheduler.batch_generator.extract_cache.call_count == 2

        stored = _cache(99)
        scheduler._attach_hybrid_checkpoints(7, stored)
        assert layer_checkpoints(stored[1]) is holders[1]
        assert layer_checkpoints(stored[0]) is None

    def test_boundary_snapshot_drops_checkpoints_past_the_boundary(self):
        """Turn 1 stores a boundary entry at 3000 tokens while its own
        prefill continued to 8000. Turn 2 shares the boundary prefix but has
        a different tail: seeding it from that entry must not carry turn 1's
        4096/6144 checkpoints, which describe tokens turn 2 never sent."""
        scheduler = self._scheduler()
        live = _cache(0)
        holders = collect_checkpoints(live)
        for pos in (2048, 4096, 6144):
            assert record_checkpoints(live, holders, pos, max_count=4, stride=1)
        scheduler._hybrid_checkpoints[7] = holders

        boundary_entry = _cache(1)
        scheduler._attach_hybrid_checkpoints(7, boundary_entry, length=3000)
        assert layer_checkpoints(boundary_entry[1]).positions == (2048,)

        scheduler._seed_hybrid_checkpoints(8, boundary_entry)
        assert scheduler._hybrid_checkpoints[8][1].positions == (2048,)
        assert achievable_position(boundary_entry, 7000) == 2048

    def test_disabled_without_hybrid_entries_or_when_max_is_zero(self, monkeypatch):
        scheduler = self._scheduler(hybrid_entries=0)
        self._register(scheduler, uid=1)
        scheduler._record_hybrid_checkpoints([self._response(1, 2048)])
        scheduler.batch_generator.extract_cache.assert_not_called()

        monkeypatch.setenv("RAPID_MLX_HYBRID_CHECKPOINT_MAX", "0")
        scheduler = self._scheduler(hybrid_entries=8)
        self._register(scheduler, uid=1)
        scheduler._record_hybrid_checkpoints([self._response(1, 2048)])
        scheduler.batch_generator.extract_cache.assert_not_called()

    def test_seed_carries_restored_checkpoints_into_the_live_request(self):
        scheduler = self._scheduler()
        restored = _cache(0)
        holders = collect_checkpoints(restored)
        record_checkpoints(restored, holders, 2048, max_count=4, stride=1)
        attach_checkpoints(restored, holders)

        scheduler._seed_hybrid_checkpoints(5, restored)
        assert scheduler._hybrid_checkpoints[5][1].positions == (2048,)

        scheduler._seed_hybrid_checkpoints(6, _cache(0))  # nothing attached
        assert 6 not in scheduler._hybrid_checkpoints

    def test_record_skips_malformed_or_unknown_responses(self, monkeypatch):
        monkeypatch.setenv("RAPID_MLX_HYBRID_CHECKPOINT_STRIDE", "1")
        scheduler = self._scheduler()
        self._register(scheduler, uid=4)
        scheduler.batch_generator.extract_cache.return_value = {}
        scheduler._record_hybrid_checkpoints(
            [
                SimpleNamespace(
                    uid=4, progress=None, end_of_prompt=False
                ),  # no progress
                self._response(99, 2048),  # unknown uid
                self._response(4, 2048),  # extract_cache returns no payload
            ]
        )
        assert scheduler.batch_generator.extract_cache.call_count == 1
        assert 4 not in scheduler._hybrid_checkpoints

        scheduler.batch_generator.extract_cache.return_value = {4: ([], [])}
        scheduler._record_hybrid_checkpoints([self._response(4, 2048)])
        assert 4 not in scheduler._hybrid_checkpoints

        # Holders from a restored prefix that do not line up with the live
        # cache layout are left alone rather than mis-attached.
        scheduler._hybrid_checkpoints[4] = [None]
        scheduler.batch_generator.extract_cache.return_value = {4: (_cache(4), [])}
        scheduler._record_hybrid_checkpoints([self._response(4, 2048)])
        assert scheduler._hybrid_checkpoints[4] == [None]

    def test_scheduler_without_batch_generator_or_cache_records_nothing(self):
        scheduler = self._scheduler()
        self._register(scheduler, uid=2)
        scheduler.batch_generator = None
        scheduler._record_hybrid_checkpoints([self._response(2, 2048)])
        assert 2 not in scheduler._hybrid_checkpoints

        scheduler.memory_aware_cache = None
        assert not scheduler._hybrid_checkpoints_enabled()
        scheduler._seed_hybrid_checkpoints(2, _cache(0))
        assert 2 not in scheduler._hybrid_checkpoints

    def test_extract_failure_is_ignored(self):
        scheduler = self._scheduler()
        self._register(scheduler, uid=3)
        scheduler.batch_generator.extract_cache.side_effect = RuntimeError("boom")
        scheduler._record_hybrid_checkpoints([self._response(3, 2048)])
        assert 3 not in scheduler._hybrid_checkpoints
