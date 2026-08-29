"""Long-context adaptive prefill policy regression tests."""

import pytest

pytest.importorskip("mlx")
pytestmark = pytest.mark.requires_mlx


from types import SimpleNamespace

from vllm_mlx.scheduler import Scheduler, SchedulerConfig


def _scheduler(*, prompt_tokens=100_000, active=0, rss=0, cap=100_000):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.config = SchedulerConfig(
        prefill_step_size=2048,
        adaptive_prefill=True,
        adaptive_prefill_min_tokens=32_768,
        adaptive_prefill_min_chunk_size=256,
    )
    scheduler.batch_generator = SimpleNamespace(
        prefill_step_size=2048,
        _prompt_batch=SimpleNamespace(prefill_step_size=2048, tokens=[[]]),
        _currently_processing=[[[list(range(1))], 0, prompt_tokens]],
        _unprocessed_sequences=[],
    )
    scheduler._resolve_metal_cap_bytes = lambda: cap
    scheduler._current_metal_active_bytes = lambda: active
    scheduler._current_process_resident_bytes = lambda: rss
    scheduler._last_adaptive_prefill_size = 2048
    scheduler._adaptive_prefill_protected_chunks = 0
    scheduler._adaptive_prefill_reduced_chunks = 0
    scheduler.running = {}
    return scheduler


def test_short_prefill_keeps_configured_chunk():
    scheduler = _scheduler(prompt_tokens=8_000, active=95_000)
    assert scheduler._select_adaptive_prefill_size() == 2048


def test_long_prefill_tightens_at_pressure_bands():
    assert _scheduler(active=69_000)._select_adaptive_prefill_size() == 2048
    assert _scheduler(active=70_000)._select_adaptive_prefill_size() == 1024
    assert _scheduler(active=80_000)._select_adaptive_prefill_size() == 512
    assert _scheduler(active=88_000)._select_adaptive_prefill_size() == 256


def test_process_footprint_wins_over_mlx_active():
    scheduler = _scheduler(active=50_000, rss=85_000)
    assert scheduler._select_adaptive_prefill_size() == 512


def test_cached_offset_counts_toward_long_context_guard():
    scheduler = _scheduler(prompt_tokens=2_000, active=75_000)
    scheduler.batch_generator._prompt_batch.tokens = [[0] * 100_000]
    assert scheduler._active_prefill_token_count() == 102_000
    assert scheduler._select_adaptive_prefill_size() == 1024


def test_running_request_supplies_full_offset_when_mlx_only_exposes_suffix():
    scheduler = _scheduler(prompt_tokens=16_000, active=75_000)
    scheduler.running = {
        "request": SimpleNamespace(model_prompt_tokens=96_000, num_prompt_tokens=96_000)
    }
    assert scheduler._active_prefill_token_count() == 96_000
    assert scheduler._select_adaptive_prefill_size() == 1024


def test_processed_prompt_tokens_are_not_double_counted():
    scheduler = _scheduler(prompt_tokens=64_000, active=75_000)
    scheduler.batch_generator._prompt_batch.tokens = [[0] * 1_024]
    scheduler.batch_generator._currently_processing[0][1] = 1_024
    assert scheduler._active_prefill_token_count() == 64_000


def test_active_prefill_only_downshifts_until_prompt_completes():
    scheduler = _scheduler(active=85_000)
    assert scheduler._apply_adaptive_prefill_size() == 512
    scheduler._current_metal_active_bytes = lambda: 65_000
    scheduler._current_process_resident_bytes = lambda: 65_000
    assert scheduler._apply_adaptive_prefill_size() == 512


def test_policy_never_grows_operator_chunk_or_crosses_floor():
    scheduler = _scheduler(active=99_000)
    scheduler.config.prefill_step_size = 128
    scheduler.config.adaptive_prefill_min_chunk_size = 256
    assert scheduler._select_adaptive_prefill_size() == 128


def test_apply_updates_future_and_in_progress_prompt_batches():
    scheduler = _scheduler(active=82_000)
    selected = scheduler._apply_adaptive_prefill_size()
    assert selected == 512
    assert scheduler.batch_generator.prefill_step_size == 512
    assert scheduler.batch_generator._prompt_batch.prefill_step_size == 512
    assert scheduler._adaptive_prefill_protected_chunks == 1
    assert scheduler._adaptive_prefill_reduced_chunks == 1


def test_disabled_policy_is_exact_noop():
    scheduler = _scheduler(active=99_000)
    scheduler._last_adaptive_prefill_size = 256
    scheduler.batch_generator.prefill_step_size = 256
    scheduler.batch_generator._prompt_batch.prefill_step_size = 256
    scheduler.config.adaptive_prefill = False
    assert scheduler._select_adaptive_prefill_size() == 2048
    assert scheduler._apply_adaptive_prefill_size() == 2048
    assert scheduler.batch_generator.prefill_step_size == 2048
    assert scheduler.batch_generator._prompt_batch.prefill_step_size == 2048
    assert scheduler._adaptive_prefill_protected_chunks == 0


def test_zero_minimum_threshold_restores_on_idle():
    scheduler = _scheduler(active=82_000)
    scheduler.config.adaptive_prefill_min_tokens = 0
    assert scheduler._apply_adaptive_prefill_size() == 512
    scheduler.batch_generator._currently_processing = []
    scheduler.batch_generator._prompt_batch.tokens = []
    assert scheduler._apply_adaptive_prefill_size() == 2048
