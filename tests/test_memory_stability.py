# SPDX-License-Identifier: Apache-2.0
"""
Tests for VRAM memory stability fixes.

Verifies that:
1. BatchGenerator.close() is called when replacing/discarding generators
2. Periodic mx.clear_cache() is triggered in generation loop
3. Metal memory stats are reported in get_stats()
"""

import pytest

pytest.importorskip("mlx")
pytestmark = pytest.mark.requires_mlx


import time
from unittest.mock import MagicMock, patch

from rapid_mlx.request import SamplingParams
from rapid_mlx.scheduler import Scheduler, SchedulerConfig


def _make_scheduler(
    enable_prefix_cache=False,
) -> Scheduler:
    """Create a scheduler with mocked model/tokenizer for unit tests."""
    model = MagicMock()
    tokenizer = MagicMock()
    tokenizer.encode = lambda x: list(range(len(x.split())))
    tokenizer.eos_token_id = 0

    config = SchedulerConfig(
        max_num_seqs=4,
        enable_prefix_cache=enable_prefix_cache,
    )
    return Scheduler(model, tokenizer, config)


class TestBatchGeneratorClose:
    """Tests that BatchGenerator.close() is called properly."""

    def test_close_called_on_replacement(self):
        """Verify .close() is called when BatchGenerator is replaced."""
        scheduler = _make_scheduler()

        # Create a mock BatchGenerator with close()
        old_generator = MagicMock()
        old_generator.close = MagicMock()
        scheduler.batch_generator = old_generator

        # Trigger replacement via _close_batch_generator
        scheduler._close_batch_generator()

        old_generator.close.assert_called_once()
        assert scheduler.batch_generator is None

    def test_close_called_on_reset(self):
        """Verify .close() is called during reset()."""
        scheduler = _make_scheduler()

        mock_generator = MagicMock()
        mock_generator.close = MagicMock()
        scheduler.batch_generator = mock_generator

        scheduler.reset()

        mock_generator.close.assert_called_once()
        assert scheduler.batch_generator is None

    def test_close_called_on_cache_error_recovery(self):
        """Verify .close() is called during _recover_from_cache_error()."""
        scheduler = _make_scheduler()

        mock_generator = MagicMock()
        mock_generator.close = MagicMock()
        scheduler.batch_generator = mock_generator

        scheduler._recover_from_cache_error()

        mock_generator.close.assert_called_once()
        assert scheduler.batch_generator is None

    def test_close_not_called_when_none(self):
        """Verify no error when batch_generator is already None."""
        scheduler = _make_scheduler()
        assert scheduler.batch_generator is None

        # Should not raise
        scheduler._close_batch_generator()
        assert scheduler.batch_generator is None

    def test_close_exception_is_caught(self):
        """Verify exceptions in close() are caught gracefully."""
        scheduler = _make_scheduler()

        mock_generator = MagicMock()
        mock_generator.close = MagicMock(side_effect=RuntimeError("close failed"))
        scheduler.batch_generator = mock_generator

        # Should not raise
        scheduler._close_batch_generator()
        assert scheduler.batch_generator is None

    def test_close_called_in_ensure_batch_generator(self):
        """Verify _close_batch_generator is called when _ensure_batch_generator replaces."""
        scheduler = _make_scheduler()

        mock_generator = MagicMock()
        mock_generator.close = MagicMock()
        scheduler.batch_generator = mock_generator
        scheduler._current_sampler_params = (0.5, 0.9, 0.0)

        # Patch _create_batch_generator to return a new mock
        new_generator = MagicMock()
        with patch.object(
            scheduler, "_create_batch_generator", return_value=new_generator
        ):
            # Different params forces recreation
            params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=100)
            scheduler._ensure_batch_generator(params)

        # Old generator should have been closed
        mock_generator.close.assert_called_once()
        assert scheduler.batch_generator is new_generator


class TestClearCacheInterval:
    """Tests for periodic mx.clear_cache() calls."""

    def test_clear_cache_interval_configured(self):
        """Verify default clear_cache interval is set."""
        scheduler = _make_scheduler()
        assert scheduler._step_count == 0
        assert scheduler._clear_cache_interval == 32

    @patch("rapid_mlx.scheduler.mx")
    def test_clear_cache_called_periodically(self, mock_mx):
        """Verify mx.clear_cache() is called every _clear_cache_interval steps."""
        scheduler = _make_scheduler()
        scheduler._clear_cache_interval = 4  # Small interval for testing

        # Simulate steps without actual generation (no running requests)
        for _i in range(8):
            scheduler.step()

        # Should have been called at step 4 and 8
        assert mock_mx.clear_cache.call_count >= 2

    @patch("rapid_mlx.scheduler.mx")
    def test_clear_cache_called_on_cleanup(self, mock_mx):
        """Verify mx.clear_cache() is called when requests finish."""
        scheduler = _make_scheduler()

        # Call _cleanup_finished with non-empty set
        scheduler._cleanup_finished({"req-1"})

        mock_mx.clear_cache.assert_called()

    @patch("rapid_mlx.scheduler.mx")
    def test_clear_cache_not_called_on_empty_cleanup(self, mock_mx):
        """Verify mx.clear_cache() is NOT called when no requests finish."""
        scheduler = _make_scheduler()

        scheduler._cleanup_finished(set())

        mock_mx.clear_cache.assert_not_called()


class TestIncrementalCacheEval:
    """Tests for incremental per-layer cache evaluation in _cleanup_finished()."""

    @patch("rapid_mlx.scheduler.mx")
    def test_incremental_eval_called_per_layer(self, mock_mx):
        """Verify mx.eval is called per layer during cleanup, not as one batch."""
        scheduler = _make_scheduler()

        # Create a mock request with extracted cache (dict-state format)
        mock_request = MagicMock()
        mock_request.prompt_token_ids = [1, 2, 3]
        mock_request.output_token_ids = [4, 5]
        mock_keys_1 = MagicMock()
        mock_values_1 = MagicMock()
        mock_keys_2 = MagicMock()
        mock_values_2 = MagicMock()
        mock_request._extracted_cache = [
            {"state": (mock_keys_1, mock_values_1)},
            {"state": (mock_keys_2, mock_values_2)},
        ]

        scheduler.running["req-1"] = mock_request

        scheduler._cleanup_finished({"req-1"})

        # mx.eval should have been called once per layer (2 layers)
        eval_calls = mock_mx.eval.call_args_list
        assert len(eval_calls) == 2
        # First call with layer 1 keys/values
        assert eval_calls[0] == ((mock_keys_1, mock_values_1),)
        # Second call with layer 2 keys/values
        assert eval_calls[1] == ((mock_keys_2, mock_values_2),)

    @patch("rapid_mlx.scheduler.mx")
    def test_no_eval_when_no_extracted_cache(self, mock_mx):
        """Verify mx.eval is not called when request has no extracted cache."""
        scheduler = _make_scheduler()

        mock_request = MagicMock()
        mock_request.prompt_token_ids = [1, 2, 3]
        mock_request.output_token_ids = [4, 5]
        mock_request._extracted_cache = None

        scheduler.running["req-1"] = mock_request

        scheduler._cleanup_finished({"req-1"})

        # mx.eval should NOT have been called (only mx.clear_cache for cleanup)
        mock_mx.eval.assert_not_called()

    @patch("rapid_mlx.scheduler.mx")
    def test_no_eager_eval_in_extraction_path(self, mock_mx):
        """Verify mx.eval(mx.array(0)) is NOT called during cache extraction."""
        scheduler = _make_scheduler()

        # Create a mock response with prompt_cache
        mock_response = MagicMock()
        mock_response.uid = 42
        mock_response.token = 100
        mock_response.finish_reason = "stop"
        mock_response.prompt_cache = [MagicMock()]

        # Setup request/uid mapping
        mock_request = MagicMock()
        mock_request.request_id = "req-1"
        mock_request.output_token_ids = [100]
        mock_request.num_output_tokens = 1
        mock_request.num_prompt_tokens = 3
        # Real floats for the timing fields _process_batch_responses reads:
        # it computes prefill_s / generation_s and gates each on ``> 0``,
        # which a bare auto-created MagicMock cannot satisfy (MagicMock
        # supports arithmetic magic methods but not comparison ones, so
        # ``MagicMock() > 0`` raises TypeError). Leaving these as mocks is
        # what made this test go stale; the subject under test — no eager
        # mx.eval in the extraction path — is unaffected by the values.
        # Offsets follow the real request lifecycle (arrival → prefill
        # start → first token, all in the past) so both the prefill and
        # generation durations come out positive.
        now = time.time()
        mock_request.arrival_time = now - 2.0
        mock_request._prefill_started_at = now - 1.5
        mock_request.first_token_time = now - 1.0
        # Monotonic twins read by the terminal timing snapshot.
        mono = time.monotonic()
        mock_request._arrival_monotonic = mono - 2.0
        mock_request._first_token_monotonic = mono - 1.0
        # No reasoning parser on this request. A bare MagicMock here reads
        # as "scope stop strings to the answer" and sends mock text into
        # the real stop matcher.
        mock_request.sampling_params.stop = []
        mock_request.sampling_params.reasoning_stop_scope = None
        scheduler.running["req-1"] = mock_request
        scheduler.uid_to_request_id[42] = "req-1"

        scheduler._process_batch_responses([mock_response])

        # Verify mx.eval was NOT called with mx.array(0) — the old spike pattern
        for call_args in mock_mx.eval.call_args_list:
            args = call_args[0]
            # Should not be called with a single mx.array argument
            assert not (len(args) == 1 and args[0] == mock_mx.array(0))


class TestMemoryStats:
    """Tests for Metal memory stats in get_stats()."""

    @patch("rapid_mlx.memory_peak.mx")
    @patch("rapid_mlx.scheduler.mx")
    def test_metal_stats_included(self, mock_mx, mock_peak_mx):
        """Verify Metal memory stats appear in get_stats()."""
        mock_mx.metal.is_available.return_value = True
        mock_mx.get_active_memory.return_value = 10_000_000_000  # 10GB
        mock_mx.get_peak_memory.return_value = 15_000_000_000  # 15GB
        # The cumulative peak is read through the shared peak helper.
        mock_peak_mx.get_peak_memory.return_value = 15_000_000_000
        mock_mx.get_cache_memory.return_value = 2_000_000_000  # 2GB

        scheduler = _make_scheduler()
        stats = scheduler.get_stats()

        assert stats["metal_active_memory_gb"] == 10.0
        assert stats["metal_peak_memory_gb"] == 15.0
        assert stats["metal_cache_memory_gb"] == 2.0

    @patch("rapid_mlx.scheduler.mx")
    def test_metal_stats_graceful_on_error(self, mock_mx):
        """Verify get_stats() works even if Metal stats fail."""
        mock_mx.metal.is_available.side_effect = RuntimeError("no metal")

        scheduler = _make_scheduler()
        stats = scheduler.get_stats()

        # Should still return basic stats without Metal info
        assert "num_waiting" in stats
        assert "metal_active_memory_gb" not in stats


class _RetiredCache:
    """Weak-referenceable stand-in for a cache owning a Metal graph."""


class _RetiredBatch:
    def __init__(self, cache):
        self.prompt_cache = [cache]
        self.tokens = [[1]]
        self.uids = [0]
        self.samplers = [cache]
        self.logits_processors = [[cache]]
        self.state_machines = [cache]
        self.max_tokens = [10]
        self._current_tokens = cache
        self._next_tokens = cache
        self._current_logprobs = [cache]
        self._next_logprobs = [cache]
        self._token_context = [cache]
        self._num_tokens = [1]
        self._matcher_states = [cache]

    def _step(self):
        return [], []

    def next(self):
        return []


class _RetiredGenerator:
    def __init__(self, cache, *, close_raises=False):
        from collections import deque

        self._prompt_batch = _RetiredBatch(cache)
        self._generation_batch = _RetiredBatch(cache)
        self._unprocessed_sequences = deque([(0, cache)])
        self._currently_processing = [(0, cache)]
        self.close_raises = close_raises
        self.closed = False

    def close(self):
        self.closed = True
        if self.close_raises:
            raise RuntimeError("synchronize failed")

    def next(self):
        return [], []

    def remove(self, uids, return_prompt_caches=False):
        return {}


@pytest.mark.parametrize("close_raises", [False, True])
def test_retired_generator_releases_graph_without_gc(monkeypatch, close_raises):
    import weakref

    cache = _RetiredCache()
    ref = weakref.ref(cache)
    scheduler = _make_scheduler()
    retired = _RetiredGenerator(cache, close_raises=close_raises)
    batches = (retired._prompt_batch, retired._generation_batch)
    scheduler.batch_generator = retired
    scheduler._hybrid_checkpoints[0] = [cache]
    del cache

    scheduler._close_batch_generator()

    assert retired.closed
    assert scheduler.batch_generator is None
    assert not retired._unprocessed_sequences
    assert not retired._currently_processing
    assert not scheduler._hybrid_checkpoints
    for batch in batches:
        assert batch.prompt_cache == []
        assert batch.uids == []
        assert batch._next_tokens is None
        assert batch._current_tokens is None
    assert ref() is None
    replacement = _RetiredGenerator(_RetiredCache())
    monkeypatch.setattr(scheduler, "_create_batch_generator", lambda _: replacement)
    assert scheduler._ensure_batch_generator(SamplingParams())
    assert scheduler.batch_generator.next() == ([], [])


def test_retirement_clears_slotted_generator_and_retained_batch():
    import gc
    import weakref
    from collections import deque

    class SlottedBatch:
        __slots__ = (
            "prompt_cache",
            "tokens",
            "uids",
            "_current_tokens",
            "_next_tokens",
            "_step",
        )

        def __init__(self, cache):
            self.prompt_cache = [cache]
            self.tokens = [cache]
            self.uids = [cache]
            self._current_tokens = cache
            self._next_tokens = cache
            self._step = cache

    class SlottedGenerator:
        __slots__ = (
            "_compiled_decode_teardown",
            "_unprocessed_sequences",
            "_currently_processing",
            "_prompt_batch",
            "_generation_batch",
            "active_batch",
        )

        def close(self):
            closed.append(True)

    scheduler = _make_scheduler()
    cache = _RetiredCache()
    ref = weakref.ref(cache)
    batch = SlottedBatch(cache)
    retired = SlottedGenerator()
    closed = []
    torn_down = []
    retired._compiled_decode_teardown = lambda: torn_down.append(True)
    retired._unprocessed_sequences = deque([cache])
    retired._currently_processing = [cache]
    retired._prompt_batch = batch
    retired._generation_batch = batch
    retired.active_batch = batch
    scheduler.batch_generator = retired

    scheduler._close_batch_generator()

    assert torn_down == [True]
    assert closed == [True]
    assert not retired._unprocessed_sequences
    assert not retired._currently_processing
    assert retired._prompt_batch is None
    assert retired._generation_batch is None
    assert retired.active_batch is None
    assert batch.prompt_cache == []
    assert batch.tokens == []
    assert batch.uids == []
    assert batch._current_tokens is None
    assert batch._next_tokens is None
    assert getattr(batch, "_step", None) is None
    assert scheduler.batch_generator is None
    # Retirement must preserve the class method rather than shadowing it.
    assert retired.close.__func__ is SlottedGenerator.close
    scheduler.batch_generator = retired
    scheduler._close_batch_generator()
    assert torn_down == [True]
    del cache
    gc.collect()
    assert ref() is None


def test_retirement_preserves_ordinary_class_methods():
    calls = []

    class Generator:
        def _compiled_decode_teardown(self):
            calls.append("teardown")

        def close(self):
            calls.append("close")

    retired = Generator()
    scheduler = _make_scheduler()
    scheduler.batch_generator = retired

    scheduler._close_batch_generator()

    assert calls == ["close"]
    assert "_compiled_decode_teardown" not in vars(retired)
    assert "close" not in vars(retired)
    assert (
        retired._compiled_decode_teardown.__func__
        is Generator._compiled_decode_teardown
    )
    assert retired.close.__func__ is Generator.close


def test_retirement_does_not_synthesize_magicmock_state():
    retired = MagicMock()
    batch = retired._generation_batch
    batch_state = dict(vars(batch))
    batch_children = dict(batch._mock_children)
    scheduler = _make_scheduler()
    scheduler.batch_generator = retired

    scheduler._close_batch_generator()

    for name in (
        "_continuous_mtp_teardown",
        "_mtp_vendored_teardown",
        "_dspark_teardown",
        "_suffix_teardown",
        "_compiled_decode_teardown",
    ):
        assert name not in retired._mock_children
        assert name not in vars(retired)
    assert vars(batch) == batch_state
    assert batch._mock_children == batch_children
    assert retired._generation_batch is batch


@pytest.mark.parametrize("unreadable", ["both", "queue", "hook", "batch"])
def test_retirement_continues_after_unreadable_properties(unreadable):
    accessed = []

    class Generator:
        def __init__(self, batch):
            self._prompt_batch = batch
            self._generation_batch = batch
            self._currently_processing = [object()]

        @property
        def _unprocessed_sequences(self):
            accessed.append("queue")
            if unreadable in ("both", "queue"):
                raise TypeError("queue unavailable")
            return None

        @property
        def _continuous_mtp_teardown(self):
            accessed.append("hook")
            if unreadable in ("both", "hook"):
                raise RuntimeError("hook unavailable")
            return None

        @property
        def active_batch(self):
            accessed.append("batch")
            if unreadable == "batch":
                raise RuntimeError("batch unavailable")
            return None

        def close(self):
            pass

    batch = _RetiredBatch(_RetiredCache())
    retired = Generator(batch)
    calls = []
    retired._compiled_decode_teardown = lambda: calls.append(True)
    scheduler = _make_scheduler()
    scheduler.batch_generator = retired

    scheduler._close_batch_generator()

    assert batch.prompt_cache == []
    assert batch._current_tokens is None
    assert batch._next_tokens is None
    assert retired._prompt_batch is None
    assert retired._generation_batch is None
    assert retired._currently_processing == []
    assert calls == [True]
    assert accessed == []
    assert scheduler.batch_generator is None


@pytest.mark.parametrize("assignment_fails", [False, True])
def test_retirement_continues_after_storage_mutation_fails(assignment_fails):
    class RestrictedState(dict):
        def pop(self, name, default=None):
            if name == "_compiled_decode_teardown":
                raise RuntimeError("hook cannot be deleted")
            return super().pop(name, default)

        def __setitem__(self, name, value):
            if name == "tokens" or (
                assignment_fails and name == "_compiled_decode_teardown"
            ):
                raise RuntimeError("field cannot be replaced")
            return super().__setitem__(name, value)

    retired = _RetiredGenerator(_RetiredCache())
    calls = []
    hook = lambda: calls.append(True)
    retired._compiled_decode_teardown = hook
    retired.__dict__ = RestrictedState(vars(retired))
    batch = retired._generation_batch
    batch.__dict__ = RestrictedState(vars(batch))
    scheduler = _make_scheduler()
    scheduler.batch_generator = retired

    scheduler._close_batch_generator()

    assert calls == [True]
    assert retired._compiled_decode_teardown is (hook if assignment_fails else None)
    assert retired._prompt_batch is None
    assert retired._generation_batch is None
    assert batch.tokens == [[1]]
    assert batch.prompt_cache == []
    assert batch._current_tokens is None
    assert batch._next_tokens is None


def test_retirement_clears_retained_batch_with_iterator_slots():
    class Batch:
        __slots__ = iter(("prompt_cache", "_next_tokens"))

    batch = Batch()
    batch.prompt_cache = [object()]
    batch._next_tokens = object()
    retired = _RetiredGenerator(_RetiredCache())
    retired._generation_batch = batch
    scheduler = _make_scheduler()
    scheduler.batch_generator = retired

    scheduler._close_batch_generator()

    assert batch.prompt_cache == []
    assert batch._next_tokens is None


def test_retirement_reads_inherited_string_slots_with_unreadable_dict():
    class BaseBatch:
        __slots__ = "prompt_cache"

    class Batch(BaseBatch):
        __slots__ = ("_next_tokens",)

        @property
        def __dict__(self):
            raise RuntimeError("dictionary unavailable")

    class Generator:
        __slots__ = "_generation_batch"

        @property
        def __dict__(self):
            raise RuntimeError("dictionary unavailable")

    batch = Batch()
    batch.prompt_cache = [object()]
    batch._next_tokens = object()
    retired = Generator()
    retired._generation_batch = batch
    scheduler = _make_scheduler()
    scheduler.batch_generator = retired

    scheduler._close_batch_generator()

    assert batch.prompt_cache == []
    assert batch._next_tokens is None
    assert retired._generation_batch is None


@pytest.mark.parametrize("error_type", [AttributeError, TypeError, RuntimeError])
@pytest.mark.parametrize("assignment_fails", [False, True])
def test_retirement_tolerates_restricted_attributes(
    monkeypatch, error_type, assignment_fails
):
    import weakref

    import rapid_mlx.scheduler as scheduler_module

    class RestrictedGenerator(_RetiredGenerator):
        __slots__ = ("_compiled_decode_teardown",)

        def __getattribute__(self, name):
            if name == "_suffix_teardown":
                raise error_type("unreadable hook")
            return super().__getattribute__(name)

    slot = RestrictedGenerator._compiled_decode_teardown
    attempts = []

    class RestrictedSlot:
        def __delete__(self, obj):
            attempts.append("delete")
            raise error_type("hook cannot be deleted")

        def __set__(self, obj, value):
            attempts.append("set")
            if assignment_fails:
                raise error_type("hook cannot be replaced")
            slot.__set__(obj, value)

    cache = _RetiredCache()
    ref = weakref.ref(cache)
    retired = RestrictedGenerator(cache)
    batch = _RetiredBatch(cache)
    retired._prompt_batch = batch
    calls = []
    hook = lambda: calls.append(True)
    retired._compiled_decode_teardown = hook
    owned_attr = scheduler_module._owned_attr
    restricted_slot = RestrictedSlot()

    def restricted_owned_attr(obj, name):
        owned = owned_attr(obj, name)
        # Real member descriptors cannot be patched to raise on delete/set.
        if obj is retired and name == "_compiled_decode_teardown":
            return restricted_slot, owned[1]
        return owned

    monkeypatch.setattr(scheduler_module, "_owned_attr", restricted_owned_attr)
    scheduler = _make_scheduler()
    scheduler.batch_generator = retired
    del cache

    scheduler._close_batch_generator()

    assert calls == [True]
    assert attempts == ["delete", "set"]
    assert retired.closed
    assert retired._prompt_batch is None
    assert retired._generation_batch is None
    assert batch.tokens == []
    assert batch.prompt_cache == []
    assert batch._current_tokens is None
    assert batch._next_tokens is None
    assert retired._compiled_decode_teardown is (hook if assignment_fails else None)
    assert ref() is None
    assert scheduler.batch_generator is None

    scheduler._close_batch_generator()

    assert calls == [True]
    assert attempts == ["delete", "set"]


@pytest.mark.parametrize(
    "generator_kind",
    ["regular", "slotted", "slotted_batch", "unreadable_generator", "unreadable_batch"],
)
def test_generation_recovery_releases_request_and_checkpoint_ownership(
    monkeypatch, generator_kind
):
    import weakref

    from rapid_mlx.request import Request, RequestStatus

    scheduler = _make_scheduler()
    cache = _RetiredCache()
    ref = weakref.ref(cache)
    request = Request("failed", "prompt", SamplingParams())
    request.status = RequestStatus.RUNNING
    request.batch_uid = 0
    request.prompt_cache = [cache]
    request._extracted_cache = [cache]
    request._continuous_mtp_state = cache
    scheduler.running[request.request_id] = request
    scheduler.requests[request.request_id] = request
    scheduler.request_id_to_uid[request.request_id] = 0
    scheduler.uid_to_request_id[0] = request.request_id
    scheduler._hybrid_checkpoints[0] = [cache]

    class SlottedGenerator:
        __slots__ = ()

        def close(self):
            pass

    class SlottedBatch:
        __slots__ = ("prompt_cache",)

        def __init__(self, cache):
            self.prompt_cache = [cache]

    class UnreadableGenerator(SlottedGenerator):
        @property
        def __dict__(self):
            raise RuntimeError("instance state unavailable")

    class UnreadableBatch:
        @property
        def __dict__(self):
            raise RuntimeError("batch state unavailable")

    retired = _RetiredGenerator(cache)
    if generator_kind == "slotted":
        retired = SlottedGenerator()
    elif generator_kind == "slotted_batch":
        retired._prompt_batch = SlottedBatch(cache)
    elif generator_kind == "unreadable_generator":
        retired = UnreadableGenerator()
    elif generator_kind == "unreadable_batch":
        retired._prompt_batch = UnreadableBatch()
    scheduler.batch_generator = retired
    scheduler._admission_prefill_uids.add(0)
    scheduler.spec_decode_runtime_method = "mtp"
    scheduler.spec_decode_runtime_attempted = True
    scheduler.block_aware_cache = MagicMock()
    prefix = MagicMock()
    scheduler.memory_aware_cache = prefix
    monkeypatch.setattr("rapid_mlx.scheduler.mx.clear_cache", lambda: None)
    del cache

    assert scheduler._recover_from_generation_error() == {request.request_id}

    assert request.status == RequestStatus.FINISHED_ABORTED
    assert request.prompt_cache is None
    assert request._extracted_cache is None
    assert request._continuous_mtp_state is None
    assert request.batch_uid is None
    assert not scheduler.running
    assert not scheduler._hybrid_checkpoints
    assert not scheduler.request_id_to_uid
    assert not scheduler.uid_to_request_id
    assert not scheduler._admission_prefill_uids
    assert scheduler.spec_decode_runtime_method is None
    assert scheduler.spec_decode_runtime_attempted is False
    scheduler.block_aware_cache.release_cache.assert_called_once_with(
        request.request_id
    )
    prefix.store.assert_not_called()
    prefix.clear.assert_not_called()
    assert ref() is None


@pytest.mark.parametrize("release_raises", [False, True])
def test_generation_recovery_continues_after_cache_release_failure(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    release_raises: bool,
) -> None:
    import mlx.core as mx

    from rapid_mlx.paged_cache import PagedCacheManager
    from rapid_mlx.prefix_cache import BlockAwarePrefixCache
    from rapid_mlx.request import Request, RequestStatus

    scheduler = _make_scheduler()
    manager = PagedCacheManager(max_blocks=4)
    cache = BlockAwarePrefixCache(scheduler.model, manager)
    scheduler.block_aware_cache = cache
    request_ids = ["first", "failed", "last"]
    calls: list[str] = []
    released: list[str] = []
    original_release = cache.release_cache

    def release_cache(request_id: str) -> None:
        calls.append(request_id)
        if release_raises and request_id == "failed":
            raise RuntimeError("release failed")
        original_release(request_id)
        released.append(request_id)

    def grammar(tokens: mx.array, logits: mx.array) -> mx.array:
        return logits

    for uid, request_id in enumerate(request_ids):
        request = Request(request_id, "prompt", SamplingParams())
        request.status = RequestStatus.RUNNING
        request.batch_uid = uid
        scheduler.running[request_id] = request
        scheduler.requests[request_id] = request
        scheduler.request_id_to_uid[request_id] = uid
        scheduler.uid_to_request_id[uid] = request_id
        scheduler._register_uid_processors(uid, request, [grammar], grammar)
        manager.create_block_table(request_id)

    monkeypatch.setattr(cache, "release_cache", release_cache)
    monkeypatch.setattr("rapid_mlx.scheduler.mx.clear_cache", lambda: None)

    assert scheduler._recover_from_generation_error() == set(request_ids)

    assert calls == request_ids
    assert released == (["first", "last"] if release_raises else request_ids)
    assert set(manager.request_tables) == ({"failed"} if release_raises else set())
    assert not scheduler.running
    assert scheduler.finished_req_ids == set(request_ids)
    assert not scheduler.request_id_to_uid
    assert not scheduler.uid_to_request_id
    assert not scheduler.uid_to_request_processors
    assert not scheduler._uids_with_grammar
    assert not scheduler._known_stateful_processors
    assert not scheduler._stateful_processor_objs
    assert not scheduler._stateful_tombstones
    assert all(
        request.status == RequestStatus.FINISHED_ABORTED and request.batch_uid is None
        for request in scheduler.requests.values()
    )
    assert ("release failed" in caplog.text) is release_raises


def test_retirement_detaches_externally_retained_mtp_hooks():
    import weakref
    from types import SimpleNamespace

    from rapid_mlx.scheduler import (
        _install_dense_sampler_fastpath,
        _install_mtp_vendored,
    )

    class Generation(_RetiredBatch):
        def _step(self):
            return [], []

        def next(self):
            return []

    cache = _RetiredCache()
    cache_ref = weakref.ref(cache)
    retired = _RetiredGenerator(cache)
    retired._generation_batch = Generation(cache)
    retired.completion_batch_size = 4
    scheduler = _make_scheduler()
    scheduler.batch_generator = retired
    model = SimpleNamespace(mtp_forward=object(), make_mtp_cache=object(), mtp=object())
    assert _install_mtp_vendored(retired, model)
    batch = retired._generation_batch
    _install_dense_sampler_fastpath(retired)
    scheduler._recurrent_output_chain_batch = batch
    hooks = (
        retired.next,
        retired.remove,
        batch.next,
        batch._step.__func__.__closure__[0].cell_contents,
    )
    closed = []

    def verifier(owned):
        try:
            yield owned
        finally:
            closed.append(True)

    gen = verifier(cache)
    next(gen)
    batch._mtp_vendored_state[0] = {"gen": gen, "queue": [cache]}
    generator_ref = weakref.ref(retired)
    batch_ref = weakref.ref(batch)
    del cache, gen

    scheduler._close_batch_generator()

    assert closed == [True]
    assert scheduler._recurrent_output_chain_batch is None
    assert batch._mtp_vendored_state == {}
    assert "next" not in vars(retired)
    assert "remove" not in vars(retired)
    assert "_step" not in vars(batch)
    assert "next" not in vars(batch)
    del retired, batch
    assert cache_ref() is None
    assert generator_ref() is None
    assert batch_ref() is None
    assert all(callable(hook) for hook in hooks)


@pytest.mark.parametrize("method", ["suffix", "dspark"])
def test_retirement_reaps_other_speculative_hook_state(method):
    import inspect
    import weakref
    from types import SimpleNamespace

    from rapid_mlx.scheduler import _install_dspark, _install_suffix_decoding

    cache = _RetiredCache()
    ref = weakref.ref(cache)
    retired = _RetiredGenerator(cache)
    scheduler = _make_scheduler()
    scheduler.batch_generator = retired
    if method == "suffix":
        assert _install_suffix_decoding(retired, object(), None, 8, 4, 0.3, {}, {})
        map_names = ("_pending_emits", "_drafters", "_uid_state")
    else:
        model = SimpleNamespace(
            args=SimpleNamespace(dspark_block_size=5),
            mtp=[object()],
            dspark_forward=lambda *args: None,
            make_dspark_cache=lambda: [],
            _last_dspark_hidden=None,
        )
        assert _install_dspark(retired, model, {}, {}, max_draft=5)
        map_names = ("_caches", "_pending", "_pending_replay")
    teardown = getattr(retired, f"_{method}_teardown")
    owned = inspect.getclosurevars(teardown).nonlocals
    for name in map_names:
        owned[name][0] = [cache]
    hooks = (retired._generation_batch._step, retired._generation_batch.next)
    # Keep only the exposed state maps, not our inspection's copy of gb.
    maps = [owned[name] for name in map_names]
    del owned, cache

    scheduler._close_batch_generator()

    assert all(not state for state in maps)
    assert ref() is None
    assert all(callable(hook) for hook in hooks)


def test_retirement_continues_after_hook_teardown_raises():
    import weakref

    cache = _RetiredCache()
    ref = weakref.ref(cache)
    retired = _RetiredGenerator(cache)
    scheduler = _make_scheduler()
    scheduler.batch_generator = retired
    retired._mtp_vendored_teardown = lambda: (_ for _ in ()).throw(
        RuntimeError("hook close failed")
    )
    scheduler._recurrent_output_chain_batch = retired._generation_batch
    del cache

    scheduler._close_batch_generator()

    assert scheduler._recurrent_output_chain_batch is None
    assert ref() is None


@pytest.mark.parametrize(
    "queue_name", ["_unprocessed_sequences", "_currently_processing"]
)
def test_retirement_continues_after_queue_clear_raises(queue_name):
    import weakref

    class BrokenQueue:
        def __init__(self, cache):
            self.cache = cache

        def clear(self):
            raise RuntimeError("queue clear failed")

    cache = _RetiredCache()
    ref = weakref.ref(cache)
    retired = _RetiredGenerator(cache)
    setattr(retired, queue_name, BrokenQueue(cache))
    scheduler = _make_scheduler()
    scheduler.batch_generator = retired
    del cache

    scheduler._close_batch_generator()

    assert queue_name not in vars(retired)
    other_queue = (
        "_currently_processing"
        if queue_name == "_unprocessed_sequences"
        else "_unprocessed_sequences"
    )
    assert not getattr(retired, other_queue)
    assert retired._prompt_batch is None
    assert retired._generation_batch is None
    assert ref() is None


def test_retirement_preserves_shared_prefix_layers():
    cache = _RetiredCache()
    cache.state = [object()]
    prefix_entry = [cache]
    scheduler = _make_scheduler()
    scheduler.batch_generator = _RetiredGenerator(cache)
    # The identical list can be shared too; retirement must replace ownership.
    scheduler.batch_generator._prompt_batch.prompt_cache = prefix_entry

    scheduler._close_batch_generator()

    assert prefix_entry == [cache]
    assert len(cache.state) == 1


def test_retirement_drops_recurrent_barrier_external_root():
    import weakref

    scheduler = _make_scheduler()
    retired = _RetiredGenerator(_RetiredCache())
    batch = retired._generation_batch
    ref = weakref.ref(batch)
    scheduler.batch_generator = retired
    scheduler._recurrent_output_chain_batch = batch
    scheduler._recurrent_output_chain_failures = 2
    scheduler._recurrent_chain_depth = 31
    scheduler._recurrent_prev_running = 4

    scheduler._close_batch_generator()

    assert scheduler._recurrent_output_chain_batch is None
    assert scheduler._recurrent_output_chain_failures == 0
    assert scheduler._recurrent_chain_depth == 0
    assert scheduler._recurrent_prev_running == 0
    del batch, retired
    assert ref() is None


@pytest.mark.parametrize("active", [80, 95, 110])
def test_recovery_preserves_existing_pressure_threshold_and_mru(monkeypatch, active):
    from collections import OrderedDict
    from threading import RLock
    from types import SimpleNamespace

    scheduler = _make_scheduler()
    retired = _RetiredGenerator(_RetiredCache())
    scheduler.batch_generator = retired
    entries = OrderedDict([("valid-prefix", object())])
    cache = SimpleNamespace(_lock=RLock(), _entries=entries, _max_memory=0)
    # A single valid prefix is protected below the cap by existing policy.
    cache._evict_lru = lambda: entries.popitem(last=False)
    scheduler.memory_aware_cache = cache
    monkeypatch.setattr(scheduler, "_resolve_metal_cap_bytes", lambda: 100)
    monkeypatch.setattr(
        scheduler,
        "_current_metal_active_bytes",
        lambda: active if not retired.closed else 50,
    )
    monkeypatch.setattr("rapid_mlx.scheduler.mx.clear_cache", lambda: None)

    scheduler._recover_from_generation_error()

    assert retired.closed
    assert bool(entries) is (active < 100)
    assert scheduler.num_prefix_cache_pressure_evictions == int(active >= 100)


def test_failed_pressure_reclaim_does_not_prevent_retirement(monkeypatch):
    scheduler = _make_scheduler()
    retired = _RetiredGenerator(_RetiredCache(), close_raises=True)
    scheduler.batch_generator = retired
    monkeypatch.setattr(
        scheduler,
        "evict_prefix_cache_under_pressure",
        lambda: (_ for _ in ()).throw(RuntimeError("eviction failed")),
    )

    assert scheduler._recover_from_generation_error() == set()
    assert retired.closed
    assert scheduler.batch_generator is None
    assert retired._prompt_batch is None
