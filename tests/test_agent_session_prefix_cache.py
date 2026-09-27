# SPDX-License-Identifier: Apache-2.0
"""Agent-session prefix reuse on 16-32 GB Macs.

Measured failure (M3 Pro 18 GB, Qwen3.5-9B-4bit, Claude-Code-like session with
a ~23k-token system+tools prefix): every turn re-prefilled the whole prompt
(~70 s TTFT) because

1. the prefix-cache budget was 20% of *currently available* RAM, measured after
   the weights were resident (~0.8 GB), below one ~1 GB session entry, so the
   entry was dropped at store ("Cache entry too large");
2. when a boot happened to measure a budget the entry fit (~1.1 GB), the
   cache-self pressure trigger (90% of that budget) evicted the only entry
   right after the store;
3. a hybrid (recurrent-state) request stored BOTH its message-boundary
   snapshot and an N-token prompt entry that no later turn can use, and the
   second store LRU-evicted the first;
4. the shutdown save predicted ~6.4 s for a ~1 GB entry from a fixed 150 MB/s
   floor and skipped it under the 3.5 s SIGTERM budget.

Each test below pins one of those policies.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

pytest.importorskip("mlx")
pytestmark = pytest.mark.requires_mlx

from rapid_mlx import memory_cache as mc
from rapid_mlx.memory_cache import (
    MemoryAwarePrefixCache,
    MemoryCacheConfig,
    session_floor_bytes,
)
from rapid_mlx.request import Request, SamplingParams
from rapid_mlx.scheduler import Scheduler, SchedulerConfig

GB = 1024**3
MB = 1024**2


class _MockArray:
    def __init__(self, nbytes: int):
        self.nbytes = nbytes


class _KVLayer:
    """Stands in for KVCache (trimmable attention layer)."""

    def __init__(self, nbytes: int):
        self.keys = _MockArray(nbytes // 2)
        self.values = _MockArray(nbytes // 2)
        self.offset = 0

    def is_trimmable(self) -> bool:
        return True

    def trim(self, n: int) -> int:
        return n


class _RecurrentLayer(_KVLayer):
    """Stands in for ArraysCache (GatedDeltaNet state, not trimmable)."""

    def is_trimmable(self) -> bool:
        return False


def _hybrid_cache(total_bytes: int) -> list:
    """Qwen3.5 layout: 1 attention layer per 3 recurrent layers."""
    per_layer = total_bytes // 4
    return [_KVLayer(per_layer)] + [_RecurrentLayer(per_layer) for _ in range(3)]


# --------------------------------------------------------------------------
# Budget: session floor from Metal headroom
# --------------------------------------------------------------------------


class TestSessionFloor:
    def test_floor_is_a_third_of_headroom_after_weights(self):
        # The measured 18 GB Mac: auto cap 11.6 GB, 5.2 GB resident weights.
        cap, resident = int(11.6 * GB), int(5.2 * GB)
        assert session_floor_bytes(cap, resident) == int((cap - resident) / 3)

    def test_floor_holds_one_23k_token_session_on_16_and_18_gb(self):
        # ~1 GB per 23k-token Qwen3.5-9B entry (958 MB measured). A 16 GB Mac
        # has a ~10.7 GB working set -> 0.9 x 10.7 = 9.6 GB auto cap.
        entry = 960 * MB
        assert session_floor_bytes(int(9.6 * GB), int(5.2 * GB)) > entry
        assert session_floor_bytes(int(11.6 * GB), int(5.2 * GB)) > entry

    def test_floor_is_capped_for_large_hosts(self):
        assert session_floor_bytes(200 * GB, 20 * GB) == 4 * GB

    @pytest.mark.parametrize(
        ("cap", "resident"), [(0, 0), (-1, 0), (8 * GB, 8 * GB), (8 * GB, 9 * GB)]
    )
    def test_no_floor_without_a_cap_or_headroom(self, cap, resident):
        assert session_floor_bytes(cap, resident) == 0


class TestComputeMemoryLimit:
    @pytest.fixture(autouse=True)
    def _no_env_override(self, monkeypatch):
        monkeypatch.delenv(mc.PREFIX_CACHE_MAX_BYTES_ENV, raising=False)

    def test_floor_raises_the_percent_of_available_budget(self, monkeypatch):
        monkeypatch.setattr(mc, "_get_available_memory", lambda: 4 * GB)
        cfg = MemoryCacheConfig(min_memory_bytes=2 * GB)
        assert cfg.compute_memory_limit() == 2 * GB

    def test_floor_never_lowers_the_percent_budget(self, monkeypatch):
        monkeypatch.setattr(mc, "_get_available_memory", lambda: 100 * GB)
        cfg = MemoryCacheConfig(min_memory_bytes=2 * GB)
        assert cfg.compute_memory_limit() == int(100 * GB * 0.20)

    def test_floor_applies_to_the_no_psutil_fallback(self, monkeypatch):
        monkeypatch.setattr(mc, "_get_available_memory", lambda: 0)
        assert MemoryCacheConfig(min_memory_bytes=3 * GB).compute_memory_limit() == (
            3 * GB
        )
        assert MemoryCacheConfig().compute_memory_limit() == int(8 * GB * 0.20)

    def test_explicit_budget_and_env_override_ignore_the_floor(self, monkeypatch):
        monkeypatch.setattr(mc, "_get_available_memory", lambda: 4 * GB)
        cfg = MemoryCacheConfig(max_memory_mb=256, min_memory_bytes=2 * GB)
        assert cfg.compute_memory_limit() == 256 * MB
        monkeypatch.setenv(mc.PREFIX_CACHE_MAX_BYTES_ENV, str(300 * MB))
        assert MemoryCacheConfig(min_memory_bytes=2 * GB).compute_memory_limit() == (
            300 * MB
        )

    def test_negative_floor_is_rejected(self):
        with pytest.raises(ValueError, match="min_memory_bytes"):
            MemoryCacheConfig(min_memory_bytes=-1)


# --------------------------------------------------------------------------
# Scheduler wiring + pressure / admission policy
# --------------------------------------------------------------------------

CAP = int(11.6 * GB)
RESIDENT = int(5.2 * GB)
FLOOR = session_floor_bytes(CAP, RESIDENT)


def _scheduler(monkeypatch, *, active: int = RESIDENT) -> Scheduler:
    """Memory-aware scheduler on the measured 18 GB shape (small free RAM)."""
    active_box = [active]
    monkeypatch.delenv(mc.PREFIX_CACHE_MAX_BYTES_ENV, raising=False)
    monkeypatch.setattr(mc, "_get_available_memory", lambda: int(4.07 * GB))
    monkeypatch.setattr(Scheduler, "_resolve_metal_cap_bytes", lambda self: CAP)
    monkeypatch.setattr(
        Scheduler, "_current_metal_active_bytes", lambda self: active_box[0]
    )
    config = SchedulerConfig(
        enable_prefix_cache=True,
        use_memory_aware_cache=True,
        use_paged_cache=False,
        gpu_memory_utilization=0.9,
        hybrid_cache_entries=8,
    )
    tokenizer = MagicMock()
    tokenizer.encode = lambda s: list(range(len(s)))
    sched = Scheduler(model=MagicMock(), tokenizer=tokenizer, config=config)
    sched._test_active = active_box  # lets a test move Metal active memory
    return sched


def test_scheduler_budget_uses_the_session_floor(monkeypatch):
    sched = _scheduler(monkeypatch)
    # 20% of 4.07 GB available is the 0.81 GB the 18 GB Mac measured; the
    # floor lifts it to a third of the 6.4 GB Metal headroom.
    assert int(4.07 * GB * 0.20) < 900 * MB
    assert sched.memory_aware_cache._max_memory == FLOOR
    assert sched._prefix_cache_session_floor_bytes() == FLOOR


def test_turn_two_extending_turn_one_hits_under_a_small_budget(monkeypatch):
    """Regression: the measured failure end to end at the cache layer."""
    sched = _scheduler(monkeypatch)
    cache = sched.memory_aware_cache
    system_and_tools = list(range(23_000))
    turn_1_boundary = system_and_tools + [1, 2, 3]
    entry_bytes = int(FLOOR * 0.95)  # above the 90% cache-self threshold

    assert cache.store(
        turn_1_boundary, _hybrid_cache(entry_bytes), message_boundary=True
    )
    # Engine-loop pressure tick right after the store (the pre-fix evictor
    # dropped the only entry here).
    assert sched.evict_prefix_cache_under_pressure() == 0
    assert tuple(turn_1_boundary) in cache._entries

    turn_2 = turn_1_boundary + [7, 8, 9, 10]
    result, remaining = cache.fetch(turn_2)
    assert result is not None
    assert remaining == [7, 8, 9, 10]


def test_cache_self_pressure_trims_older_entries_but_keeps_the_newest(monkeypatch):
    sched = _scheduler(monkeypatch)
    cache = sched.memory_aware_cache
    cache.store([1, 2, 3], _hybrid_cache(int(FLOOR * 0.04)))
    cache.store(list(range(100, 200)), _hybrid_cache(int(FLOOR * 0.9)))

    assert sched.evict_prefix_cache_under_pressure() == 1
    assert list(cache._entries) == [tuple(range(100, 200))]


def test_soft_metal_pressure_trims_older_entries_but_keeps_the_newest(monkeypatch):
    sched = _scheduler(monkeypatch)
    cache = sched.memory_aware_cache
    cache.store([1, 2, 3], _hybrid_cache(int(FLOOR * 0.04)))
    cache.store(list(range(100)), _hybrid_cache(int(FLOOR * 0.5)))
    # Between 90% of the cap and the cap: the finishing request's transient
    # state (measured 9.3 of a 9.6 GB cap on a hit turn).
    sched._test_active[0] = int(CAP * 0.97)

    assert sched.evict_prefix_cache_under_pressure() == 1
    assert list(cache._entries) == [tuple(range(100))]


def test_metal_pressure_at_the_cap_evicts_the_newest_entry(monkeypatch):
    sched = _scheduler(monkeypatch)
    cache = sched.memory_aware_cache
    cache.store(list(range(100)), _hybrid_cache(int(FLOOR * 0.95)))
    sched._test_active[0] = CAP  # real unified-memory pressure

    assert sched.evict_prefix_cache_under_pressure() >= 1
    assert len(cache._entries) == 0


def _request(rid: str, prompt_len: int) -> Request:
    return Request(
        request_id=rid,
        prompt="ignored",
        prompt_token_ids=list(range(prompt_len)),
        sampling_params=SamplingParams(max_tokens=16),
    )


def test_admission_evicts_prefix_cache_before_rejecting(monkeypatch):
    sched = _scheduler(monkeypatch)
    cache = sched.memory_aware_cache
    cache.store([1, 2], _hybrid_cache(64 * MB))
    cache.store(list(range(10, 20)), _hybrid_cache(64 * MB))
    # Over cap until the prefix cache yields; each eviction frees 1 GB here.
    sched._test_active[0] = CAP + GB // 2

    def _evict(keep_mru: bool = False, _orig=sched._evict_one_prefix_cache_entry):
        evicted = _orig(keep_mru=keep_mru)
        if evicted:
            sched._test_active[0] -= GB
        return evicted

    monkeypatch.setattr(sched, "_evict_one_prefix_cache_entry", _evict)
    monkeypatch.setattr(sched, "_estimate_request_kv_bytes", lambda _r: 0)
    before = sched.num_prefix_cache_pressure_evictions

    sched._enforce_metal_cap_at_admission(_request("admit-after-evict", 8))

    assert len(cache._entries) == 1  # only the LRU entry had to go
    assert sched.num_prefix_cache_pressure_evictions == before + 1
    assert sched.num_metal_cap_violations == 0


def test_admission_still_rejects_when_the_cache_cannot_free_enough(monkeypatch):
    from rapid_mlx.scheduler import BackpressureError

    sched = _scheduler(monkeypatch)
    cache = sched.memory_aware_cache
    cache.store([1, 2], _hybrid_cache(64 * MB))
    sched._test_active[0] = CAP * 2  # evicting 64 MB cannot fix this
    monkeypatch.setattr(sched, "_estimate_request_kv_bytes", lambda _r: 0)

    with pytest.raises(BackpressureError):
        sched._enforce_metal_cap_at_admission(_request("reject", 8))
    assert len(cache._entries) == 0
    assert sched.num_metal_cap_violations == 1


# --------------------------------------------------------------------------
# Prompt-entry policy for hybrid (non-trimmable) caches
# --------------------------------------------------------------------------


def _register(sched: Scheduler, uid: int, prompt_len: int) -> Request:
    req = _request(f"req-{uid}", prompt_len)
    sched.requests[req.request_id] = req
    sched.uid_to_request_id[uid] = req.request_id
    sched.request_id_to_uid[req.request_id] = uid
    return req


def test_hybrid_prompt_entry_skipped_after_a_stored_boundary(monkeypatch):
    sched = _scheduler(monkeypatch)
    req = _register(sched, 7, 40)
    req.prefix_boundary = 30
    req._cache_snapshot_stored = True
    sched.memory_aware_cache.store = MagicMock(return_value=True)

    sched._prompt_cache_save_cb(7, _hybrid_cache(4 * MB))

    sched.memory_aware_cache.store.assert_not_called()


def test_hybrid_prompt_entry_kept_when_no_boundary_was_stored(monkeypatch):
    sched = _scheduler(monkeypatch)
    _register(sched, 8, 40)
    sched.memory_aware_cache.store = MagicMock(return_value=True)
    layers = _hybrid_cache(4 * MB)

    sched._prompt_cache_save_cb(8, layers)

    sched.memory_aware_cache.store.assert_called_once_with(
        list(range(40)), layers, evict_prefixes=False
    )


def test_trimmable_prompt_entry_kept_after_a_stored_boundary(monkeypatch):
    sched = _scheduler(monkeypatch)
    req = _register(sched, 9, 40)
    req._cache_snapshot_stored = True
    sched.memory_aware_cache.store = MagicMock(return_value=True)
    layers = [_KVLayer(4 * MB)]

    sched._prompt_cache_save_cb(9, layers)

    sched.memory_aware_cache.store.assert_called_once_with(
        list(range(40)), layers, evict_prefixes=False
    )


# --------------------------------------------------------------------------
# Shutdown save: measured throughput instead of the fixed 150 MB/s floor
# --------------------------------------------------------------------------


def test_throughput_probe_measures_and_cleans_up(tmp_path):
    bps = mc._probe_write_bytes_per_sec(str(tmp_path))
    assert bps > 0
    assert list(tmp_path.iterdir()) == []


def test_throughput_probe_reports_zero_on_an_unwritable_dir(tmp_path):
    assert mc._probe_write_bytes_per_sec(str(tmp_path / "missing")) == 0.0


def _saved_entry_count(tmp_path, probe_bps: float, monkeypatch) -> int:
    import mlx.core as mx
    from mlx_lm.models.cache import KVCache

    monkeypatch.setattr(mc, "_probe_write_bytes_per_sec", lambda _d: probe_bps)
    cache = MemoryAwarePrefixCache(MagicMock(), MemoryCacheConfig(max_memory_mb=64))
    layer = KVCache()
    keys = mx.ones((1, 2, 64, 8), dtype=mx.float32)
    layer.update_and_fetch(keys, keys)
    cache.store(list(range(64)), [layer])
    entry_bytes = next(iter(cache._entries.values())).memory_bytes
    # Budget that fits the entry at >= 500 MB/s but not at the 150 MB/s floor.
    budget_sec = entry_bytes / (500 * MB)
    snap = tmp_path / "snap"
    cache.save_to_disk(
        str(snap), should_abort=lambda predicted_sec=0.0: predicted_sec > budget_sec
    )
    loaded = MemoryAwarePrefixCache(MagicMock(), MemoryCacheConfig(max_memory_mb=64))
    return loaded.load_from_disk(str(snap)) if snap.exists() else 0


def test_budgeted_save_uses_measured_throughput(tmp_path, monkeypatch):
    assert _saved_entry_count(tmp_path, 1024 * MB, monkeypatch) == 1


def test_budgeted_save_keeps_the_floor_on_a_slow_disk(tmp_path, monkeypatch):
    assert _saved_entry_count(tmp_path, 10 * MB, monkeypatch) == 0


def test_hybrid_prompt_entry_kept_when_the_boundary_is_far_back(monkeypatch):
    # Fallback (dummy-LCP) boundaries can sit before a long last message; the
    # N-token entry then saves more than the bounded gap, so keep it.
    sched = _scheduler(monkeypatch)
    req = _register(sched, 10, 400)
    req.prefix_boundary = 100
    req._cache_snapshot_stored = True
    sched.memory_aware_cache.store = MagicMock(return_value=True)
    layers = _hybrid_cache(4 * MB)

    sched._prompt_cache_save_cb(10, layers)

    sched.memory_aware_cache.store.assert_called_once_with(
        list(range(400)), layers, evict_prefixes=False
    )


def test_internal_snapshot_is_not_a_message_boundary(monkeypatch):
    sched = _scheduler(monkeypatch)
    req = _register(sched, 11, 40)
    req._cache_snapshot_boundary = 39
    req._cache_snapshot_is_internal = True
    req._cache_snapshot_stored = True
    layers = _hybrid_cache(4 * MB)
    assert sched._boundary_snapshot_supersedes(req, layers, 40) is False
    assert sched._protect_boundary_behind_completion(req, layers) is True
    sched._touch_boundary_entry(req)  # no-op, must not raise


# --------------------------------------------------------------------------
# Completion (prompt + output) entry never displaces the boundary entry
# --------------------------------------------------------------------------


def _finish_with_completion(sched, uid, prompt_len, boundary, cache_bytes, output):
    req = _register(sched, uid, prompt_len)
    req.prefix_boundary = boundary
    req._cache_snapshot_stored = True
    req.output_token_ids = list(output)
    sched.running[req.request_id] = req
    boundary_tokens = list(range(boundary))
    assert sched.memory_aware_cache.store(
        boundary_tokens, _hybrid_cache(cache_bytes), message_boundary=True
    )
    req._extracted_cache = _hybrid_cache(cache_bytes)
    sched._cleanup_finished({req.request_id})
    return tuple(boundary_tokens), tuple(list(range(prompt_len)) + list(output))


def test_completion_entry_skipped_when_it_would_evict_the_boundary(monkeypatch):
    # 16 GB shape: one ~entry fits the floor, two do not.
    sched = _scheduler(monkeypatch)
    entry = int(FLOOR * 0.6)
    boundary_key, completion_key = _finish_with_completion(
        sched, 20, 500, 490, entry, [900, 901, 902]
    )
    assert list(sched.memory_aware_cache._entries) == [boundary_key]
    assert completion_key not in sched.memory_aware_cache._entries


def test_completion_entry_stored_behind_the_boundary_when_both_fit(monkeypatch):
    sched = _scheduler(monkeypatch)
    entry = int(FLOOR * 0.3)
    boundary_key, completion_key = _finish_with_completion(
        sched, 21, 500, 490, entry, [900, 901, 902]
    )
    # Boundary is most-recently-used, so any trim takes the completion first.
    assert list(sched.memory_aware_cache._entries) == [completion_key, boundary_key]


def test_trimmable_completion_entry_is_stored_normally(monkeypatch):
    sched = _scheduler(monkeypatch)
    req = _register(sched, 22, 50)
    req._cache_snapshot_stored = True
    req.prefix_boundary = 45
    req.output_token_ids = [7, 8]
    sched.running[req.request_id] = req
    req._extracted_cache = [_KVLayer(int(FLOOR * 2))]  # would not fit: no gate
    assert sched._protect_boundary_behind_completion(req, req._extracted_cache)


# --------------------------------------------------------------------------
# Long prefill reclaims the cache before its transient peak
# --------------------------------------------------------------------------


def _long_request(rid: str, tokens: int) -> Request:
    req = _request(rid, tokens)
    req.num_prompt_tokens = tokens
    return req


def test_cold_prefill_reclaims_the_cache_on_a_16gb_class_cap(monkeypatch):
    cap_16 = int(9.6 * GB)
    sched = _scheduler(monkeypatch)
    monkeypatch.setattr(sched, "_resolve_metal_cap_bytes", lambda: cap_16)
    monkeypatch.setattr(sched, "_resolve_kv_bytes_per_token", lambda: 32 * 1024)
    monkeypatch.setattr(sched, "_estimate_request_kv_bytes", lambda _r: int(0.75 * GB))
    cache = sched.memory_aware_cache
    cache.store(list(range(50_000, 50_100)), _hybrid_cache(GB))  # other session
    freed_by_evict = sched._test_active

    def _evict(keep_mru: bool = False, _orig=sched._evict_one_prefix_cache_entry):
        ok = _orig(keep_mru=keep_mru)
        if ok:
            freed_by_evict[0] -= GB
        return ok

    monkeypatch.setattr(sched, "_evict_one_prefix_cache_entry", _evict)
    sched._test_active[0] = RESIDENT + GB

    req = _long_request("cold", 22_800)
    sched.add_request(req)

    assert req.remaining_tokens == req.prompt_token_ids  # cold miss
    assert len(cache._entries) == 0
    assert sched.num_metal_cap_violations == 0


def test_short_remaining_prefill_keeps_the_cache(monkeypatch):
    sched = _scheduler(monkeypatch)
    monkeypatch.setattr(sched, "_resolve_kv_bytes_per_token", lambda: 32 * 1024)
    monkeypatch.setattr(sched, "_estimate_request_kv_bytes", lambda _r: 0)
    cache = sched.memory_aware_cache
    cache.store(list(range(100)), _hybrid_cache(64 * MB))
    req = _long_request("short", 50)
    req.remaining_tokens = list(range(50))
    assert sched._reclaim_prefix_cache_for_prefill(req) == 0
    assert len(cache._entries) == 1


def test_prefill_reclaim_is_a_no_op_without_a_cap_or_entries(monkeypatch):
    sched = _scheduler(monkeypatch)
    req = _long_request("empty", 10)
    assert sched._reclaim_prefix_cache_for_prefill(req) == 0  # no entries
    sched.memory_aware_cache.store(list(range(10)), _hybrid_cache(MB))
    monkeypatch.setattr(sched, "_resolve_metal_cap_bytes", lambda: 0)
    assert sched._reclaim_prefix_cache_for_prefill(req) == 0


def test_prefill_reclaim_stops_when_nothing_is_left_to_evict(monkeypatch):
    sched = _scheduler(monkeypatch)
    monkeypatch.setattr(sched, "_resolve_kv_bytes_per_token", lambda: 32 * 1024)
    monkeypatch.setattr(sched, "_estimate_request_kv_bytes", lambda _r: 0)
    sched.memory_aware_cache.store(list(range(10)), _hybrid_cache(MB))
    sched._test_active[0] = CAP * 2  # over the cap regardless
    req = _long_request("hopeless", 10)
    req.remaining_tokens = list(range(10))
    assert sched._reclaim_prefix_cache_for_prefill(req) == 1
    assert len(sched.memory_aware_cache._entries) == 0


def test_budgeted_save_caps_an_optimistic_probe(tmp_path, monkeypatch):
    # A page-cache-speed probe (10 GB/s) is capped at 600 MB/s, so an entry
    # that needs 1 GB/s to fit the budget is still skipped.
    import mlx.core as mx
    from mlx_lm.models.cache import KVCache

    monkeypatch.setattr(mc, "_probe_write_bytes_per_sec", lambda _d: 10 * GB)
    cache = MemoryAwarePrefixCache(MagicMock(), MemoryCacheConfig(max_memory_mb=64))
    layer = KVCache()
    keys = mx.ones((1, 2, 64, 8), dtype=mx.float32)
    layer.update_and_fetch(keys, keys)
    cache.store(list(range(64)), [layer])
    entry_bytes = next(iter(cache._entries.values())).memory_bytes
    budget_sec = entry_bytes / (1000 * MB)
    assert (
        cache.save_to_disk(
            str(tmp_path / "snap"),
            should_abort=lambda predicted_sec=0.0: predicted_sec > budget_sec,
        )
        is False
    )


# --------------------------------------------------------------------------
# Hybrid checkpoints survive a restart
# --------------------------------------------------------------------------


def _hybrid_entry(length: int, checkpoints: tuple[int, ...]):
    import mlx.core as mx
    from mlx_lm.models.cache import ArraysCache, KVCache

    from rapid_mlx.hybrid_state_checkpoints import CHECKPOINT_ATTR, StateCheckpoints

    kv = KVCache()
    kv.keys = mx.zeros((1, 2, length, 8))
    kv.values = mx.zeros((1, 2, length, 8))
    kv.offset = length
    rec = ArraysCache(2)
    rec.cache = [mx.full((1, 3, 4), length), mx.full((1, 2, 2), length)]
    setattr(
        rec,
        CHECKPOINT_ATTR,
        StateCheckpoints(
            (pos, (mx.full((1, 3, 4), pos), mx.full((1, 2, 2), pos)))
            for pos in checkpoints
        ),
    )
    return [kv, rec]


def _hybrid_cache_store():
    return MemoryAwarePrefixCache(
        MagicMock(),
        MemoryCacheConfig(max_memory_mb=64, max_entries=16, hybrid_reuse_max_entries=4),
    )


def _save_and_reload(tmp_path, entry_tokens, entry):
    cache = _hybrid_cache_store()
    assert cache.store(entry_tokens, entry, message_boundary=True)
    snap = tmp_path / "snap"
    assert cache.save_to_disk(str(snap), should_abort=lambda _s=0.0: False)
    restored = _hybrid_cache_store()
    assert restored.load_from_disk(str(snap)) == 1
    return snap, restored


def test_replayed_first_turn_snaps_to_a_restored_checkpoint(tmp_path):
    """After a restart, a session replayed from turn 1 (a strict prefix of
    the saved deepest boundary) resumes from the newest persisted checkpoint
    instead of re-prefilling the whole prompt."""
    from rapid_mlx.hybrid_state_checkpoints import layer_checkpoints

    stored = list(range(1000, 7000))
    snap, restored = _save_and_reload(
        tmp_path, stored, _hybrid_entry(6000, (2048, 4096))
    )
    assert (snap / "entry_0_ckpt.safetensors").exists()
    entry = next(iter(restored._entries.values()))
    assert layer_checkpoints(entry.cache[1]).positions == (2048, 4096)

    turn_1 = stored[:5000] + [1, 2, 3]
    result, remaining = restored.fetch(turn_1)

    assert result is not None
    assert remaining == turn_1[4096:]
    assert result[1].cache[0][0, 0, 0].item() == 4096


def test_entry_without_checkpoints_writes_no_sidecar(tmp_path):
    snap, restored = _save_and_reload(tmp_path, list(range(50)), _hybrid_entry(50, ()))
    assert not (snap / "entry_0_ckpt.safetensors").exists()
    assert restored.fetch(list(range(40)) + [9])[0] is None


@pytest.mark.parametrize(
    "tamper",
    ["truncate", "wrong_shape", "outside_entry", "not_recurrent", "missing_slot"],
)
def test_bad_sidecar_drops_checkpoints_but_keeps_the_entry(tmp_path, tamper):
    import mlx.core as mx

    stored = list(range(1000, 7000))
    cache = _hybrid_cache_store()
    assert cache.store(stored, _hybrid_entry(6000, (2048, 4096)))
    snap = tmp_path / "snap"
    assert cache.save_to_disk(str(snap), should_abort=lambda _s=0.0: False)
    sidecar = snap / "entry_0_ckpt.safetensors"
    good = {
        "1.2048.0": mx.full((1, 3, 4), 2048),
        "1.2048.1": mx.full((1, 2, 2), 2048),
    }
    if tamper == "truncate":
        data = sidecar.read_bytes()
        sidecar.write_bytes(data[: len(data) - 16])
    else:
        bad = dict(good)
        if tamper == "wrong_shape":
            bad["1.2048.1"] = mx.full((1, 2, 3), 2048)
        elif tamper == "outside_entry":
            bad = {k.replace("2048", "9000"): v for k, v in good.items()}
        elif tamper == "not_recurrent":
            bad = {k.replace("1.", "0.", 1): v for k, v in good.items()}
        else:
            del bad["1.2048.1"]
        mx.save_safetensors(str(sidecar), bad)

    restored = _hybrid_cache_store()
    assert restored.load_from_disk(str(snap)) == 1
    result, _ = restored.fetch(stored[:5000] + [1, 2, 3])
    assert result is None  # no checkpoint to snap to, but no crash either
    assert restored.fetch(stored + [5])[0] is not None  # exact extension still hits


def test_checkpoint_sidecar_write_failure_keeps_the_entry(tmp_path, monkeypatch):
    import mlx.core as mx

    def _boom(*_a, **_k):
        raise OSError("disk full")

    cache = _hybrid_cache_store()
    stored = list(range(1000, 7000))
    assert cache.store(stored, _hybrid_entry(6000, (2048,)))
    monkeypatch.setattr(mx, "save_safetensors", _boom)
    assert (
        mc._save_checkpoints_sidecar(
            str(tmp_path / "x_ckpt.safetensors"),
            next(iter(cache._entries.values())).cache,
        )
        is False
    )
    assert not (tmp_path / "x_ckpt.safetensors").exists()
