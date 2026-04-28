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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
