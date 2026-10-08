"""A prompt that shares a long prefix with a stored entry it cannot resume
from gets a snapshot stored where the shared span ends (text scheduler lane)."""

import pytest

pytest.importorskip("mlx")

from types import SimpleNamespace
from unittest.mock import MagicMock

from rapid_mlx import scheduler as scheduler_module
from rapid_mlx.memory_cache import MemoryAwarePrefixCache
from rapid_mlx.request import Request, SamplingParams
from rapid_mlx.scheduler import Scheduler, SchedulerConfig


def _scheduler(monkeypatch, *, min_tokens=64):
    monkeypatch.setattr(
        scheduler_module, "_SHARED_PREFIX_SNAPSHOT_MIN_TOKENS", min_tokens
    )
    config = SchedulerConfig(enable_prefix_cache=True, use_memory_aware_cache=True)
    scheduler = Scheduler(MagicMock(), MagicMock(), config)
    assert scheduler.memory_aware_cache is not None
    return scheduler


def _request(request_id, tokens, *, cached=0):
    request = Request(
        request_id=request_id,
        prompt="ignored",
        prompt_token_ids=list(tokens),
        sampling_params=SamplingParams(max_tokens=4),
    )
    request.cached_tokens = cached
    return request


def _stored(scheduler, shared):
    scheduler.memory_aware_cache.shared_prefix_length = MagicMock(return_value=shared)


def test_shared_prefix_length_reads_the_longest_shared_span():
    cache = MemoryAwarePrefixCache(model=MagicMock())
    assert cache.shared_prefix_length([1, 2, 3]) == 0
    cache._sorted_keys = [(1, 2, 3, 4, 9), (1, 2, 7), (5, 6)]
    cache._sorted_keys.sort()
    assert cache.shared_prefix_length([1, 2, 3, 4, 5, 6]) == 4
    assert cache.shared_prefix_length([1, 2, 8]) == 2
    assert cache.shared_prefix_length([1, 2, 3, 4, 9]) == 5
    assert cache.shared_prefix_length([7]) == 0


def test_split_lands_on_a_whole_tile_below_the_shared_span(monkeypatch):
    scheduler = _scheduler(monkeypatch)
    scheduler._prefill_tile_rows = MagicMock(return_value=32)
    request = _request("r", range(400), cached=10)
    _stored(scheduler, 300)

    # 290 pending shared tokens round down to 288 (nine tiles).
    assert scheduler._shared_prefix_local_split(request, 390) == 288
    assert request.shared_prefix_snapshot_at == 298


def test_no_split_when_the_shared_span_is_short_or_already_restored(monkeypatch):
    scheduler = _scheduler(monkeypatch)
    scheduler._prefill_tile_rows = MagicMock(return_value=32)

    short = _request("short", range(400))
    _stored(scheduler, 50)
    assert scheduler._shared_prefix_local_split(short, 400) is None

    restored = _request("restored", range(400), cached=280)
    _stored(scheduler, 300)
    assert scheduler._shared_prefix_local_split(restored, 120) is None
    assert restored.shared_prefix_snapshot_at == 0

    # A prompt wholly inside a stored entry keeps its last token to prefill.
    inside = _request("inside", range(64))
    _stored(scheduler, 64)
    assert scheduler._shared_prefix_local_split(inside, 64) is None

    # Fewer tokens are pending than the shared span would need.
    partial = _request("partial", range(400))
    _stored(scheduler, 300)
    assert scheduler._shared_prefix_local_split(partial, 100) is None
    assert partial.shared_prefix_snapshot_at == 0

    one = _request("one", range(400))
    assert scheduler._shared_prefix_local_split(one, 1) is None

    scheduler.memory_aware_cache = None
    assert (
        scheduler._shared_prefix_local_split(_request("none", range(400)), 400) is None
    )


def test_no_split_for_a_compressed_prompt(monkeypatch):
    scheduler = _scheduler(monkeypatch)
    monkeypatch.setattr(scheduler_module, "_pflash_compressed", lambda request: True)
    _stored(scheduler, 300)
    assert scheduler._shared_prefix_local_split(_request("r", range(400)), 400) is None


def _schedule(scheduler, request):
    scheduler.waiting.append(request)
    batch_generator = MagicMock()
    batch_generator.insert_segments.return_value = [101]
    batch_generator.insert.return_value = [101]
    scheduler.batch_generator = batch_generator
    scheduler._ensure_batch_generator = MagicMock(return_value=True)
    scheduler._get_request_sampler = MagicMock(return_value=MagicMock())
    scheduler._register_uid_processors = MagicMock()
    assert scheduler._schedule_waiting() == [request]
    return batch_generator


def test_schedule_splits_at_the_shared_span_and_the_message_boundary(monkeypatch):
    scheduler = _scheduler(monkeypatch)
    scheduler._prefill_tile_rows = MagicMock(return_value=32)
    request = _request("r", range(400))
    request.prefix_boundary = 384
    _stored(scheduler, 200)

    batch_generator = _schedule(scheduler, request)

    segments = batch_generator.insert_segments.call_args.args[0]
    assert [len(part) for part in segments[0]] == [192, 192, 16]
    assert request.shared_prefix_snapshot_at == 192


def test_schedule_splits_at_the_shared_span_alone(monkeypatch):
    scheduler = _scheduler(monkeypatch)
    scheduler._prefill_tile_rows = MagicMock(return_value=32)
    request = _request("r", range(400))
    _stored(scheduler, 200)

    batch_generator = _schedule(scheduler, request)

    segments = batch_generator.insert_segments.call_args.args[0]
    assert [len(part) for part in segments[0]] == [192, 208]


def test_schedule_keeps_one_split_when_both_positions_coincide(monkeypatch):
    scheduler = _scheduler(monkeypatch)
    scheduler._prefill_tile_rows = MagicMock(return_value=32)
    request = _request("r", range(400))
    request.prefix_boundary = 192
    _stored(scheduler, 200)

    batch_generator = _schedule(scheduler, request)

    segments = batch_generator.insert_segments.call_args.args[0]
    assert [len(part) for part in segments[0]] == [192, 208]
    assert request.shared_prefix_snapshot_at == 0


def test_schedule_retry_without_cache_drops_the_shared_split(monkeypatch):
    scheduler = _scheduler(monkeypatch)
    scheduler._prefill_tile_rows = MagicMock(return_value=32)
    request = _request("r", range(400), cached=8)
    request.prompt_cache = [MagicMock()]
    request.remaining_tokens = list(range(8, 400))
    _stored(scheduler, 208)
    scheduler.waiting.append(request)
    batch_generator = MagicMock()
    batch_generator.insert_segments.side_effect = RuntimeError("bad cache")
    batch_generator.insert.return_value = [101]
    scheduler.batch_generator = batch_generator
    scheduler._ensure_batch_generator = MagicMock(return_value=True)
    scheduler._get_request_sampler = MagicMock(return_value=MagicMock())
    scheduler._register_uid_processors = MagicMock()
    scheduler._validate_cache = MagicMock(return_value=True)

    assert scheduler._schedule_waiting() == [request]
    assert batch_generator.insert_segments.call_count == 1
    assert request.shared_prefix_snapshot_at == 0
    batch_generator.insert.assert_called_once()


def _armed(scheduler, *, position=192, cached=0):
    request = _request("r", range(400), cached=cached)
    request.shared_prefix_snapshot_at = position
    scheduler.requests["r"] = request
    scheduler.uid_to_request_id[101] = "r"
    scheduler._extract_cache_states = MagicMock(return_value=[{"k": "v"}])
    scheduler._reconstruct_cache_from_states = MagicMock(return_value=["rebuilt"])
    scheduler.memory_aware_cache.store = MagicMock(return_value=True)
    scheduler.batch_generator = MagicMock()
    scheduler.batch_generator.extract_cache.return_value = {101: (["raw"], [])}
    return request


def _segment_end(progress, *, end_of_prompt=False, uid=101):
    return SimpleNamespace(
        uid=uid, progress=progress, end_of_segment=True, end_of_prompt=end_of_prompt
    )


def test_snapshot_is_stored_when_the_prefill_reaches_the_split(monkeypatch):
    scheduler = _scheduler(monkeypatch)
    request = _armed(scheduler, position=202, cached=10)

    scheduler._snapshot_shared_prefix_segments([_segment_end((192, 390))])

    scheduler.batch_generator.extract_cache.assert_called_once_with([101])
    store = scheduler.memory_aware_cache.store
    assert store.call_args.args == (list(range(202)), ["rebuilt"])
    assert store.call_args.kwargs == {"evict_prefixes": False}
    assert request.shared_prefix_snapshot_at == 0

    # One attempt per request.
    scheduler._snapshot_shared_prefix_segments([_segment_end((192, 390))])
    assert store.call_count == 1


def test_snapshot_ignores_other_segment_ends(monkeypatch):
    scheduler = _scheduler(monkeypatch)
    request = _armed(scheduler)
    extract = scheduler.batch_generator.extract_cache

    scheduler._snapshot_shared_prefix_segments([])
    scheduler._snapshot_shared_prefix_segments(
        [
            _segment_end((384, 400)),  # the message boundary, not the split
            _segment_end((192, 400), end_of_prompt=True),
            _segment_end((192, 400), uid=999),  # unknown request
            _segment_end(None),
            SimpleNamespace(
                uid=101, progress=(192, 400), end_of_segment=False, end_of_prompt=False
            ),
        ]
    )
    extract.assert_not_called()

    request.output_token_ids = [1]
    scheduler._snapshot_shared_prefix_segments([_segment_end((192, 400))])
    extract.assert_not_called()

    scheduler.memory_aware_cache = None
    scheduler._snapshot_shared_prefix_segments([_segment_end((192, 400))])
    extract.assert_not_called()


def test_snapshot_failures_leave_the_request_untouched(monkeypatch):
    scheduler = _scheduler(monkeypatch)
    request = _armed(scheduler)
    store = scheduler.memory_aware_cache.store

    scheduler.batch_generator.extract_cache.side_effect = RuntimeError("gone")
    scheduler._snapshot_shared_prefix_segments([_segment_end((192, 400))])
    assert request.shared_prefix_snapshot_at == 192

    scheduler.batch_generator.extract_cache.side_effect = None
    scheduler.batch_generator.extract_cache.return_value = {101: "removed", 7: (1, 2)}
    scheduler._snapshot_shared_prefix_segments([_segment_end((192, 400))])
    store.assert_not_called()

    scheduler.batch_generator.extract_cache.return_value = {101: (["raw"], [])}
    scheduler._reconstruct_cache_from_states.return_value = None
    scheduler._snapshot_shared_prefix_segments([_segment_end((192, 400))])
    store.assert_not_called()
    assert request.shared_prefix_snapshot_at == 0

    request.shared_prefix_snapshot_at = 192
    scheduler._reconstruct_cache_from_states.return_value = ["rebuilt"]
    store.side_effect = RuntimeError("full")
    scheduler._snapshot_shared_prefix_segments([_segment_end((192, 400))])
    assert store.call_count == 1

    request.shared_prefix_snapshot_at = 192
    store.side_effect = None
    store.return_value = False
    scheduler._snapshot_shared_prefix_segments([_segment_end((192, 400))])
    assert store.call_count == 2
