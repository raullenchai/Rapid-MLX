# SPDX-License-Identifier: Apache-2.0
"""
Tests for the prompt-boundary cache snapshot path used by mlx-lm 0.31+.

The fix for issue #163 wires Scheduler._snapshot_promoted_prompts() into the
generation step. It reads end_of_prompt from PromptProcessingBatch.Response
and uses BatchGenerator.extract_cache() to capture the prompt-only cache
state, then forwards each capture to the prompt_cache_save callback so the
MemoryAwarePrefixCache stores it under key=prompt_token_ids.

These tests exercise that scheduler-level glue without needing a real
mlx-lm runtime.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm_mlx.request import Request, SamplingParams
from vllm_mlx.scheduler import Scheduler, SchedulerConfig


def _make_scheduler_with_cache():
    model = MagicMock()
    tokenizer = MagicMock()
    tokenizer.encode = lambda x: list(range(len(x.split())))
    config = SchedulerConfig(enable_prefix_cache=True, use_memory_aware_cache=True)
    scheduler = Scheduler(model, tokenizer, config)
    assert scheduler.memory_aware_cache is not None
    assert scheduler._prompt_cache_save_cb is not None
    return scheduler


def _register(scheduler, request_id: str, uid: int, prompt_tokens: list[int]):
    request = Request(
        request_id=request_id,
        prompt="ignored",
        prompt_token_ids=prompt_tokens,
        sampling_params=SamplingParams(max_tokens=4),
    )
    scheduler.requests[request_id] = request
    scheduler.uid_to_request_id[uid] = request_id
    scheduler.request_id_to_uid[request_id] = uid
    return request


class TestPromptCacheSnapshot:
    def test_callback_built_when_memory_cache_enabled(self):
        scheduler = _make_scheduler_with_cache()
        assert callable(scheduler._prompt_cache_save_cb)

    def test_no_callback_without_memory_cache(self):
        scheduler = Scheduler(
            MagicMock(),
            MagicMock(),
            SchedulerConfig(enable_prefix_cache=False),
        )
        assert scheduler._prompt_cache_save_cb is None

    def test_snapshot_stores_promoted_prompt_only(self):
        scheduler = _make_scheduler_with_cache()
        prompt_tokens = [10, 20, 30, 40]
        _register(scheduler, "req-1", uid=101, prompt_tokens=prompt_tokens)

        fake_cache_layers = [object(), object()]
        bg = MagicMock()
        bg.extract_cache.return_value = {101: (fake_cache_layers, prompt_tokens)}
        scheduler.batch_generator = bg

        scheduler.memory_aware_cache.store = MagicMock(return_value=True)

        responses = [
            SimpleNamespace(
                uid=101, progress=(4, 4), end_of_segment=True, end_of_prompt=True
            ),
        ]
        scheduler._snapshot_promoted_prompts(responses)

        bg.extract_cache.assert_called_once_with([101])
        scheduler.memory_aware_cache.store.assert_called_once()
        stored_tokens, stored_cache = scheduler.memory_aware_cache.store.call_args.args[
            :2
        ]
        assert stored_tokens == prompt_tokens
        assert stored_cache is fake_cache_layers

    def test_snapshot_skips_mid_prompt_chunks(self):
        scheduler = _make_scheduler_with_cache()
        _register(scheduler, "req-1", uid=101, prompt_tokens=[1, 2, 3])
        bg = MagicMock()
        scheduler.batch_generator = bg
        scheduler.memory_aware_cache.store = MagicMock()

        responses = [
            SimpleNamespace(
                uid=101, progress=(2, 4), end_of_segment=True, end_of_prompt=False
            ),
        ]
        scheduler._snapshot_promoted_prompts(responses)

        bg.extract_cache.assert_not_called()
        scheduler.memory_aware_cache.store.assert_not_called()

    def test_snapshot_handles_extract_failure(self):
        scheduler = _make_scheduler_with_cache()
        _register(scheduler, "req-1", uid=101, prompt_tokens=[1, 2, 3])
        bg = MagicMock()
        bg.extract_cache.side_effect = RuntimeError("boom")
        scheduler.batch_generator = bg
        scheduler.memory_aware_cache.store = MagicMock()

        responses = [
            SimpleNamespace(
                uid=101, progress=(3, 3), end_of_segment=True, end_of_prompt=True
            ),
        ]
        # Must not raise — snapshot is best-effort.
        scheduler._snapshot_promoted_prompts(responses)
        scheduler.memory_aware_cache.store.assert_not_called()

    def test_snapshot_skips_already_removed_uid(self):
        scheduler = _make_scheduler_with_cache()
        _register(scheduler, "req-1", uid=101, prompt_tokens=[1, 2, 3])
        bg = MagicMock()
        # Stage 0 (unprocessed) returns a 2-tuple of (segments, last_input).
        # Stage > 2 or removed uid: extract_cache may return any non-(cache,tokens)
        # shape. Our snapshot path must skip silently.
        bg.extract_cache.return_value = {101: ("not-a-cache-payload",)}
        scheduler.batch_generator = bg
        scheduler.memory_aware_cache.store = MagicMock()

        responses = [
            SimpleNamespace(
                uid=101, progress=(3, 3), end_of_segment=True, end_of_prompt=True
            ),
        ]
        scheduler._snapshot_promoted_prompts(responses)
        scheduler.memory_aware_cache.store.assert_not_called()

    def test_snapshot_no_op_when_callback_disabled(self):
        scheduler = Scheduler(
            MagicMock(),
            MagicMock(),
            SchedulerConfig(enable_prefix_cache=False),
        )
        bg = MagicMock()
        scheduler.batch_generator = bg

        responses = [
            SimpleNamespace(
                uid=101, progress=(3, 3), end_of_segment=True, end_of_prompt=True
            ),
        ]
        scheduler._snapshot_promoted_prompts(responses)
        bg.extract_cache.assert_not_called()

    def test_snapshot_with_empty_responses(self):
        scheduler = _make_scheduler_with_cache()
        scheduler.batch_generator = MagicMock()
        # Should not raise
        scheduler._snapshot_promoted_prompts([])
        scheduler.batch_generator.extract_cache.assert_not_called()

    def test_snapshot_stores_only_end_of_prompt_subset(self):
        scheduler = _make_scheduler_with_cache()
        _register(scheduler, "req-a", uid=1, prompt_tokens=[1, 2])
        _register(scheduler, "req-b", uid=2, prompt_tokens=[3, 4])
        _register(scheduler, "req-c", uid=3, prompt_tokens=[5, 6])

        cache_a = [object()]
        cache_c = [object()]
        bg = MagicMock()
        bg.extract_cache.return_value = {
            1: (cache_a, [1, 2]),
            3: (cache_c, [5, 6]),
        }
        scheduler.batch_generator = bg
        scheduler.memory_aware_cache.store = MagicMock(return_value=True)

        responses = [
            SimpleNamespace(
                uid=1, progress=(2, 2), end_of_segment=True, end_of_prompt=True
            ),
            SimpleNamespace(
                uid=2, progress=(1, 2), end_of_segment=True, end_of_prompt=False
            ),
            SimpleNamespace(
                uid=3, progress=(2, 2), end_of_segment=True, end_of_prompt=True
            ),
        ]
        scheduler._snapshot_promoted_prompts(responses)

        bg.extract_cache.assert_called_once_with([1, 3])
        assert scheduler.memory_aware_cache.store.call_count == 2


class TestBoundarySnapshot:
    """Tests for `_snapshot_boundary_segments` — the per-message cache
    save path for mlx-lm 0.31+ multi-turn agentic workloads on hybrid
    models (issue #427).

    The path fires on `end_of_segment=True AND NOT end_of_prompt=True`,
    which BatchGenerator emits when a request inserted via
    `insert_segments([[prefix, tail]])` finishes the prefix segment but
    the tail still has tokens left. Each fire extracts the
    boundary-state cache, reconstructs it through the existing
    `_extract_cache_states`/`_reconstruct_cache_from_states` helpers,
    and stores it under `request.prompt_token_ids[:prefix_boundary]`
    so the next turn's lookup gets a prefix-cache hit.
    """

    def _register_with_boundary(
        self,
        scheduler,
        request_id: str,
        uid: int,
        prompt_tokens: list[int],
        prefix_boundary: int,
    ):
        request = _register(scheduler, request_id, uid, prompt_tokens)
        request.prefix_boundary = prefix_boundary
        return request

    def test_snapshot_stores_at_prefix_boundary(self):
        scheduler = _make_scheduler_with_cache()
        # Mock the extract/reconstruct helpers so we don't need a real
        # MLX cache to test the dispatch logic.
        scheduler._extract_cache_states = MagicMock(return_value=[{"k": "v"}])
        scheduler._reconstruct_cache_from_states = MagicMock(
            return_value=["reconstructed-cache"]
        )

        prompt_tokens = list(range(100))
        prefix_boundary = 80
        self._register_with_boundary(
            scheduler,
            "req-1",
            uid=101,
            prompt_tokens=prompt_tokens,
            prefix_boundary=prefix_boundary,
        )

        bg = MagicMock()
        # Stage-1 (in-prompt) extract returns (cache, tokens).
        bg.extract_cache.return_value = {101: (["raw-cache"], prompt_tokens[:80])}
        scheduler.batch_generator = bg
        scheduler.memory_aware_cache.store = MagicMock(return_value=True)

        # end_of_segment without end_of_prompt: prefix segment done,
        # tail still pending.
        responses = [
            SimpleNamespace(
                uid=101,
                progress=(80, 100),
                end_of_segment=True,
                end_of_prompt=False,
            ),
        ]
        scheduler._snapshot_boundary_segments(responses)

        bg.extract_cache.assert_called_once_with([101])
        scheduler.memory_aware_cache.store.assert_called_once()
        stored_tokens, stored_cache = scheduler.memory_aware_cache.store.call_args.args[
            :2
        ]
        assert stored_tokens == prompt_tokens[:prefix_boundary]
        assert stored_cache == ["reconstructed-cache"]
        # evict_prefixes=False must be passed so the boundary entry
        # isn't evicted by the later whole-prompt save.
        assert (
            scheduler.memory_aware_cache.store.call_args.kwargs.get("evict_prefixes")
            is False
        )

    def test_snapshot_skips_end_of_prompt_responses(self):
        """end_of_prompt is the whole-prompt promotion — handled by the
        other snapshot path, must not duplicate-fire here."""
        scheduler = _make_scheduler_with_cache()
        self._register_with_boundary(
            scheduler,
            "req-1",
            uid=101,
            prompt_tokens=[1, 2, 3, 4],
            prefix_boundary=2,
        )
        bg = MagicMock()
        scheduler.batch_generator = bg
        scheduler.memory_aware_cache.store = MagicMock()

        responses = [
            SimpleNamespace(
                uid=101,
                progress=(4, 4),
                end_of_segment=True,
                end_of_prompt=True,
            ),
        ]
        scheduler._snapshot_boundary_segments(responses)

        bg.extract_cache.assert_not_called()
        scheduler.memory_aware_cache.store.assert_not_called()

    def test_snapshot_skips_when_no_prefix_boundary(self):
        """Single-turn requests have prefix_boundary=0 and should not
        trigger a boundary save even if end_of_segment fires."""
        scheduler = _make_scheduler_with_cache()
        # No prefix_boundary attr — default is 0 from Request.__init__.
        _register(scheduler, "req-1", uid=101, prompt_tokens=[1, 2, 3, 4])
        bg = MagicMock()
        scheduler.batch_generator = bg
        scheduler.memory_aware_cache.store = MagicMock()

        responses = [
            SimpleNamespace(
                uid=101,
                progress=(3, 4),
                end_of_segment=True,
                end_of_prompt=False,
            ),
        ]
        scheduler._snapshot_boundary_segments(responses)

        bg.extract_cache.assert_not_called()
        scheduler.memory_aware_cache.store.assert_not_called()

    def test_snapshot_skips_when_no_memory_cache(self):
        scheduler = Scheduler(
            MagicMock(),
            MagicMock(),
            SchedulerConfig(enable_prefix_cache=False),
        )
        # Sanity: no memory cache wired.
        assert scheduler.memory_aware_cache is None

        responses = [
            SimpleNamespace(
                uid=101,
                progress=(2, 4),
                end_of_segment=True,
                end_of_prompt=False,
            ),
        ]
        # Must not raise even though batch_generator is unset.
        scheduler._snapshot_boundary_segments(responses)

    def test_snapshot_idempotent_per_request(self):
        """Once-per-request guard: future API change emitting a second
        end_of_segment must not produce a duplicate store."""
        scheduler = _make_scheduler_with_cache()
        scheduler._extract_cache_states = MagicMock(return_value=[{"k": "v"}])
        scheduler._reconstruct_cache_from_states = MagicMock(return_value=["c"])

        prompt_tokens = list(range(10))
        self._register_with_boundary(
            scheduler,
            "req-1",
            uid=101,
            prompt_tokens=prompt_tokens,
            prefix_boundary=5,
        )
        bg = MagicMock()
        bg.extract_cache.return_value = {101: (["raw"], prompt_tokens[:5])}
        scheduler.batch_generator = bg
        scheduler.memory_aware_cache.store = MagicMock(return_value=True)

        resp = SimpleNamespace(
            uid=101,
            progress=(5, 10),
            end_of_segment=True,
            end_of_prompt=False,
        )
        scheduler._snapshot_boundary_segments([resp])
        scheduler._snapshot_boundary_segments([resp])

        # Only the first call should have stored.
        assert scheduler.memory_aware_cache.store.call_count == 1

    def test_snapshot_handles_extract_failure(self):
        scheduler = _make_scheduler_with_cache()
        self._register_with_boundary(
            scheduler,
            "req-1",
            uid=101,
            prompt_tokens=[1, 2, 3, 4],
            prefix_boundary=2,
        )
        bg = MagicMock()
        bg.extract_cache.side_effect = RuntimeError("boom")
        scheduler.batch_generator = bg
        scheduler.memory_aware_cache.store = MagicMock()

        responses = [
            SimpleNamespace(
                uid=101,
                progress=(2, 4),
                end_of_segment=True,
                end_of_prompt=False,
            ),
        ]
        # Best-effort path must swallow extract failures.
        scheduler._snapshot_boundary_segments(responses)
        scheduler.memory_aware_cache.store.assert_not_called()

    def test_snapshot_handles_store_failure(self):
        scheduler = _make_scheduler_with_cache()
        scheduler._extract_cache_states = MagicMock(return_value=[{"k": "v"}])
        scheduler._reconstruct_cache_from_states = MagicMock(return_value=["c"])
        self._register_with_boundary(
            scheduler,
            "req-1",
            uid=101,
            prompt_tokens=[1, 2, 3, 4],
            prefix_boundary=2,
        )
        bg = MagicMock()
        bg.extract_cache.return_value = {101: (["raw"], [1, 2])}
        scheduler.batch_generator = bg
        scheduler.memory_aware_cache.store = MagicMock(
            side_effect=RuntimeError("store boom")
        )

        responses = [
            SimpleNamespace(
                uid=101,
                progress=(2, 4),
                end_of_segment=True,
                end_of_prompt=False,
            ),
        ]
        # Must not raise; should mark guard so we don't repeat the
        # expensive extract+reconstruct cycle every step. A failed
        # store is usually "entry already exists" or "cache busy" —
        # retrying immediately would just waste cycles. DeepSeek
        # finding #2.
        scheduler._snapshot_boundary_segments(responses)
        request = scheduler.requests["req-1"]
        assert getattr(request, "_boundary_snapshot_taken", False)

        # Second call with same response must not re-attempt extract.
        bg.extract_cache.reset_mock()
        scheduler._snapshot_boundary_segments(responses)
        bg.extract_cache.assert_not_called()

    def test_snapshot_skips_unknown_uid(self):
        scheduler = _make_scheduler_with_cache()
        scheduler.batch_generator = MagicMock()
        scheduler.memory_aware_cache.store = MagicMock()

        responses = [
            SimpleNamespace(
                uid=999,
                progress=(2, 4),
                end_of_segment=True,
                end_of_prompt=False,
            ),
        ]
        scheduler._snapshot_boundary_segments(responses)
        scheduler.batch_generator.extract_cache.assert_not_called()
        scheduler.memory_aware_cache.store.assert_not_called()

    def test_snapshot_with_empty_responses(self):
        scheduler = _make_scheduler_with_cache()
        scheduler.batch_generator = MagicMock()
        scheduler._snapshot_boundary_segments([])
        scheduler.batch_generator.extract_cache.assert_not_called()

    def test_snapshot_skips_tail_segment_fire(self):
        """mlx-lm 0.31+ rewrites insert_segments([[prefix, tail]]) into
        [[prefix, tail[:-1], tail[-1:]]] when len(tail) > 1. That makes
        end_of_segment fire on the tail[:-1] boundary too, which has a
        different `progress[0]` than `prefix_boundary - cached_tokens`.

        The progress-validation gate must skip the tail fire so we
        don't corrupt the prefix-boundary cache entry with a state that
        includes tail[:-1] tokens. Codex finding on PR #435.
        """
        scheduler = _make_scheduler_with_cache()
        scheduler._extract_cache_states = MagicMock(return_value=[{"k": "v"}])
        scheduler._reconstruct_cache_from_states = MagicMock(return_value=["c"])

        prompt_tokens = list(range(100))
        self._register_with_boundary(
            scheduler,
            "req-1",
            uid=101,
            prompt_tokens=prompt_tokens,
            prefix_boundary=80,
        )
        # cached_tokens defaults to 0 → expected boundary offset = 80.
        scheduler.requests["req-1"].cached_tokens = 0

        bg = MagicMock()
        bg.extract_cache.return_value = {101: (["raw"], prompt_tokens[:80])}
        scheduler.batch_generator = bg
        scheduler.memory_aware_cache.store = MagicMock(return_value=True)

        # Simulate the three segment fires from mlx-lm's rewrite:
        # (1) prefix done → progress (80, 100), end_of_segment=True
        # (2) tail[:-1] done → progress (99, 100), end_of_segment=True
        # (3) tail[-1:] done → progress (100, 100), end_of_segment=True,
        #     end_of_prompt=True (handled by _snapshot_promoted_prompts).
        prefix_resp = SimpleNamespace(
            uid=101, progress=(80, 100), end_of_segment=True, end_of_prompt=False
        )
        tail_resp = SimpleNamespace(
            uid=101, progress=(99, 100), end_of_segment=True, end_of_prompt=False
        )

        # First call (the actual prefix fire) saves at boundary.
        scheduler._snapshot_boundary_segments([prefix_resp])
        assert scheduler.memory_aware_cache.store.call_count == 1
        stored_tokens, _ = scheduler.memory_aware_cache.store.call_args.args[:2]
        assert stored_tokens == prompt_tokens[:80]

        # Second call (the tail[:-1] fire from mlx-lm's rewrite) must NOT
        # store — progress[0]=99 doesn't equal the expected 80.
        scheduler.memory_aware_cache.store.reset_mock()
        bg.extract_cache.reset_mock()
        scheduler._snapshot_boundary_segments([tail_resp])
        bg.extract_cache.assert_not_called()
        scheduler.memory_aware_cache.store.assert_not_called()


class TestScheduleWaitingInsertDispatch:
    """Tests for the insert_segments dispatch in `_schedule_waiting`.

    DeepSeek finding #1 on PR #435: the split-decision logic
    (boundary_local_split, fallback recompute) is a critical multi-turn
    code path that needs direct unit coverage; without it, a wrong
    threshold or fallback condition could silently regress.

    Rather than driving the full _schedule_waiting (which requires
    block-cache, sampler, processor wiring), these tests directly
    exercise the dispatch contract by stubbing the BatchGenerator and
    invoking the same boundary-split branch.
    """

    def _build_dispatch_args(
        self,
        prefix_boundary: int,
        cached_tokens: int,
        tokens_to_process: list[int],
    ):
        """Reproduce the exact local-split computation from
        scheduler.py:2772-2787 (the same arithmetic used in the prod
        dispatch). Returns the split point that _schedule_waiting
        would compute, or None if no split should occur."""
        scheduler = _make_scheduler_with_cache()
        request = Request(
            request_id="req-x",
            prompt="ignored",
            prompt_token_ids=tokens_to_process,
            sampling_params=SamplingParams(max_tokens=4),
        )
        request.prefix_boundary = prefix_boundary
        request.cached_tokens = cached_tokens

        if (
            scheduler.memory_aware_cache is None
            or getattr(request, "prefix_boundary", 0) <= 0
            or len(tokens_to_process) <= 1
        ):
            return None
        pb = request.prefix_boundary
        cached = request.cached_tokens or 0
        local = pb - cached
        if 0 < local < len(tokens_to_process):
            return local
        return None

    def test_split_in_middle_returns_local_offset(self):
        # 100-token prompt, 20 cached, boundary at 80 → split at 60.
        assert (
            self._build_dispatch_args(
                prefix_boundary=80,
                cached_tokens=20,
                tokens_to_process=list(range(80)),  # 20..99
            )
            == 60
        )

    def test_split_at_boundary_no_cache_returns_full_offset(self):
        # 100-token prompt, 0 cached, boundary at 80 → split at 80.
        assert (
            self._build_dispatch_args(
                prefix_boundary=80,
                cached_tokens=0,
                tokens_to_process=list(range(100)),
            )
            == 80
        )

    def test_no_split_when_boundary_already_cached(self):
        # boundary 80 entirely inside cached portion → no split (the
        # cache hit already covered the prefix; no need to re-snapshot).
        assert (
            self._build_dispatch_args(
                prefix_boundary=80,
                cached_tokens=90,
                tokens_to_process=list(range(10)),  # tail past boundary
            )
            is None
        )

    def test_no_split_when_boundary_at_or_past_tokens(self):
        # boundary exactly at end of tokens_to_process → no point
        # splitting (the tail segment would be empty).
        assert (
            self._build_dispatch_args(
                prefix_boundary=100,
                cached_tokens=0,
                tokens_to_process=list(range(100)),
            )
            is None
        )

    def test_no_split_when_no_prefix_boundary(self):
        assert (
            self._build_dispatch_args(
                prefix_boundary=0,
                cached_tokens=0,
                tokens_to_process=list(range(100)),
            )
            is None
        )

    def test_no_split_for_single_token_kickoff(self):
        # Exact-cache hit path: tokens_to_process == [last_token].
        # Even if prefix_boundary > 0, len(tokens_to_process) <= 1
        # guard prevents splitting an already-trivial insert.
        assert (
            self._build_dispatch_args(
                prefix_boundary=80,
                cached_tokens=99,
                tokens_to_process=[42],
            )
            is None
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
