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

from vllm_mlx.request import SamplingParams
from vllm_mlx.scheduler import Scheduler, SchedulerConfig


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

    @patch("vllm_mlx.scheduler.mx")
    def test_clear_cache_called_periodically(self, mock_mx):
        """Verify mx.clear_cache() is called every _clear_cache_interval steps."""
        scheduler = _make_scheduler()
        scheduler._clear_cache_interval = 4  # Small interval for testing

        # Simulate steps without actual generation (no running requests)
        for _i in range(8):
            scheduler.step()

        # Should have been called at step 4 and 8
        assert mock_mx.clear_cache.call_count >= 2

    @patch("vllm_mlx.scheduler.mx")
    def test_clear_cache_called_on_cleanup(self, mock_mx):
        """Verify mx.clear_cache() is called when requests finish."""
        scheduler = _make_scheduler()

        # Call _cleanup_finished with non-empty set
        scheduler._cleanup_finished({"req-1"})

        mock_mx.clear_cache.assert_called()

    @patch("vllm_mlx.scheduler.mx")
    def test_clear_cache_not_called_on_empty_cleanup(self, mock_mx):
        """Verify mx.clear_cache() is NOT called when no requests finish."""
        scheduler = _make_scheduler()

        scheduler._cleanup_finished(set())

        mock_mx.clear_cache.assert_not_called()


class TestIncrementalCacheEval:
    """Tests for incremental per-layer cache evaluation in _cleanup_finished()."""

    @patch("vllm_mlx.scheduler.mx")
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

    @patch("vllm_mlx.scheduler.mx")
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

    @patch("vllm_mlx.scheduler.mx")
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

    @patch("vllm_mlx.scheduler.mx")
    def test_metal_stats_included(self, mock_mx):
        """Verify Metal memory stats appear in get_stats()."""
        mock_mx.metal.is_available.return_value = True
        mock_mx.get_active_memory.return_value = 10_000_000_000  # 10GB
        mock_mx.get_peak_memory.return_value = 15_000_000_000  # 15GB
        mock_mx.get_cache_memory.return_value = 2_000_000_000  # 2GB

        scheduler = _make_scheduler()
        stats = scheduler.get_stats()

        assert stats["metal_active_memory_gb"] == 10.0
        assert stats["metal_peak_memory_gb"] == 15.0
        assert stats["metal_cache_memory_gb"] == 2.0

    @patch("vllm_mlx.scheduler.mx")
    def test_metal_stats_graceful_on_error(self, mock_mx):
        """Verify get_stats() works even if Metal stats fail."""
        mock_mx.metal.is_available.side_effect = RuntimeError("no metal")

        scheduler = _make_scheduler()
        stats = scheduler.get_stats()

        # Should still return basic stats without Metal info
        assert "num_waiting" in stats
        assert "metal_active_memory_gb" not in stats
