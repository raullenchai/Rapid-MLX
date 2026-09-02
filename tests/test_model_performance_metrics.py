# SPDX-License-Identifier: Apache-2.0
"""Regression tests for per-model engine performance metrics."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from vllm_mlx.output_collector import RequestOutputCollector
from vllm_mlx.runtime.model_performance import (
    MODEL_LEDGER_REGISTRY_LIMIT,
    RETIRED_MODEL_SNAPSHOT_LIMIT,
    SEEN_REQUEST_ID_LIMIT,
    ModelPerformanceLedger,
)

if TYPE_CHECKING:
    from vllm_mlx.request import Request
    from vllm_mlx.scheduler import Scheduler


def _scheduler() -> Scheduler:
    pytest.importorskip("mlx")

    from vllm_mlx.scheduler import Scheduler, SchedulerConfig

    tokenizer = MagicMock()
    tokenizer.encode = lambda text: list(range(len(text.split())))
    tokenizer.decode = lambda tokens, **_kwargs: " ".join(map(str, tokens))
    scheduler = Scheduler(
        MagicMock(),
        tokenizer,
        SchedulerConfig(max_num_seqs=1, model_name="model-under-test"),
    )
    scheduler.batch_generator = MagicMock()
    scheduler.batch_generator.remove.return_value = {}
    return scheduler


def _running_request(scheduler: Scheduler, request_id: str) -> Request:
    from vllm_mlx.request import Request, RequestStatus, SamplingParams

    request = Request(
        request_id,
        "ignored prompt",
        SamplingParams(max_tokens=16),
    )
    request.status = RequestStatus.RUNNING
    request.num_prompt_tokens = 5
    request.arrival_time = time.time() - 0.25
    request.first_token_time = time.time() - 0.2
    for token in (11, 12):
        request.append_output_token(token)
    scheduler.running[request_id] = request
    scheduler.uid_to_request_id[1] = request_id
    scheduler.requests[request_id] = request
    scheduler.request_id_to_uid[request_id] = 1
    return request


def _terminal_response() -> MagicMock:
    response = MagicMock(
        uid=1,
        token=13,
        finish_reason="stop",
        logprobs=None,
    )
    del response.prompt_cache
    return response


def test_ledger_records_outcomes_once_and_ignores_bad_values():
    ledger = ModelPerformanceLedger("model-a")
    assert ledger.record_success(
        "success",
        prompt_tokens=8,
        completion_tokens=12,
        ttft_seconds=0.07,
        decode_tokens_per_second=42.0,
    )
    assert ledger.record_cancelled(
        "cancelled",
        prompt_tokens=4,
        completion_tokens=2,
        ttft_seconds=0.2,
        decode_tokens_per_second=18.0,
    )
    assert ledger.record_failure("failure")

    snapshot = ledger.snapshot()
    assert snapshot.total_requests == 3
    assert snapshot.requests_succeeded == 1
    assert snapshot.requests_cancelled == 1
    assert snapshot.requests_failed == 1

    assert not ledger.record_success(
        "success",
        prompt_tokens=99,
        completion_tokens=99,
        ttft_seconds=1,
        decode_tokens_per_second=1,
    )
    assert not ledger.record_failure("success")
    assert not ledger.record_cancelled(
        "cancelled",
        prompt_tokens=99,
        completion_tokens=99,
        ttft_seconds=99,
        decode_tokens_per_second=99,
    )
    assert ledger.model_name == "model-a"
    assert ledger.snapshot().prompt_tokens == 12


def test_ledger_ignores_unusable_timing_observations():
    ledger = ModelPerformanceLedger("model-b")
    ledger.record_success(
        "invalid-timings",
        prompt_tokens=1,
        completion_tokens=2,
        ttft_seconds=float("nan"),
        decode_tokens_per_second=-5,
    )

    snapshot = ledger.snapshot()
    assert snapshot.requests_succeeded == 1
    assert snapshot.prompt_tokens == 1
    assert snapshot.completion_tokens == 2
    assert snapshot.ttft_seconds_count == 0
    assert snapshot.decode_observations == 0
    assert snapshot.ttft_seconds_max is None
    assert snapshot.decode_tokens_per_second_max is None


def test_ledger_best_effort_helpers_ignore_unusable_timings_and_errors(
    monkeypatch: pytest.MonkeyPatch,
):
    from types import SimpleNamespace

    request = SimpleNamespace(
        arrival_time=time.time() - 0.2,
        first_token_time=time.time() + 5.0,
        num_output_tokens=2,
        num_prompt_tokens=3,
        request_id="future-first-token",
    )
    ledger = ModelPerformanceLedger("model-b")

    assert ledger.decode_rate_for_request(request) is None

    monkeypatch.setattr(
        ledger,
        "record_request_performance",
        MagicMock(side_effect=RuntimeError("accounting failure")),
    )

    ledger.record_finished_performance(request)
    ledger.record_cancelled_performance(request)

    snapshot = ledger.snapshot()
    assert snapshot.total_requests == 0


def test_ledger_dedupe_cache_is_bounded():
    ledger = ModelPerformanceLedger("model-b")
    for request_id in range(SEEN_REQUEST_ID_LIMIT):
        assert ledger.record_failure(str(request_id))

    assert ledger.record_failure("overflow") is True
    assert len(ledger._seen_request_ids) == SEEN_REQUEST_ID_LIMIT
    assert ledger.record_failure("0") is True
    assert ledger.record_failure("overflow") is False


def test_request_owned_idempotency_survives_low_level_cache_eviction():
    from types import SimpleNamespace

    ledger = ModelPerformanceLedger("model-b")
    request = SimpleNamespace(
        request_id="request-owned",
        status=SimpleNamespace(name="RUNNING"),
        num_prompt_tokens=3,
        num_output_tokens=2,
    )
    assert ledger.record_request_performance(request, "failed") is True
    for request_id in range(SEEN_REQUEST_ID_LIMIT + 1):
        ledger.record_failure(f"synthetic-{request_id}")

    assert ledger.record_request_performance(request, "failed") is False
    assert ledger.snapshot().requests_failed == SEEN_REQUEST_ID_LIMIT + 2


def test_request_accounting_uses_compressed_model_prompt_tokens():
    from types import SimpleNamespace

    ledger = ModelPerformanceLedger("compressed-model")
    request = SimpleNamespace(
        request_id="compressed",
        status=SimpleNamespace(name="RUNNING"),
        num_prompt_tokens=100_000,
        model_prompt_tokens=4_096,
        num_output_tokens=2,
    )

    assert ledger.record_request_performance(request, "succeeded") is True
    assert ledger.snapshot().prompt_tokens == 4_096


def test_request_accounting_rejects_unknown_outcome_without_marking_request():
    from types import SimpleNamespace

    ledger = ModelPerformanceLedger("invalid-outcome-model")
    request = SimpleNamespace(
        request_id="invalid-outcome",
        num_prompt_tokens=3,
        num_output_tokens=2,
    )

    with pytest.raises(ValueError, match="unsupported terminal outcome: unknown"):
        ledger.record_request_performance(request, "unknown")

    assert request._performance_recorded is False
    assert ledger.snapshot().total_requests == 0


def test_distinct_request_lifetimes_may_reuse_the_same_id():
    from types import SimpleNamespace

    ledger = ModelPerformanceLedger("model-b")
    first = SimpleNamespace(
        request_id="reused",
        arrival_time=1.0,
        first_token_time=None,
        num_prompt_tokens=2,
        num_output_tokens=0,
    )
    second = SimpleNamespace(
        request_id="reused",
        arrival_time=2.0,
        first_token_time=None,
        num_prompt_tokens=3,
        num_output_tokens=0,
    )

    ledger.record_finished_performance(first)
    ledger.record_finished_performance(first)
    ledger.record_cancelled_performance(second)

    snapshot = ledger.snapshot()
    assert snapshot.requests_succeeded == 1
    assert snapshot.requests_cancelled == 1
    assert snapshot.prompt_tokens == 5


def test_ledger_histograms_are_cumulative_and_memory_bounded():
    ledger = ModelPerformanceLedger("model-a")
    for ttft, decode in ((0.05, 4.0), (0.3, 25.0), (1.2, 150.0)):
        ledger.record_success(
            str(ttft),
            prompt_tokens=1,
            completion_tokens=4,
            ttft_seconds=ttft,
            decode_tokens_per_second=decode,
        )

    snapshot = ledger.snapshot()
    assert snapshot.ttft_bucket_counts == {
        "0.05": 1,
        "0.1": 1,
        "0.25": 1,
        "0.5": 2,
        "1": 2,
        "2": 3,
        "5": 3,
        "10": 3,
        "30": 3,
        "+Inf": 3,
    }
    assert snapshot.decode_bucket_counts["1"] == 0
    assert snapshot.decode_bucket_counts["5"] == 1
    assert snapshot.decode_bucket_counts["20"] == 1
    assert snapshot.decode_bucket_counts["50"] == 2
    assert snapshot.decode_bucket_counts["+Inf"] == 3
    assert snapshot.ttft_seconds_count == 3
    assert snapshot.decode_observations == 3


def test_scheduler_records_terminal_success_once():
    pytest.importorskip("mlx")

    scheduler = _scheduler()
    request = _running_request(scheduler, "success")
    response = _terminal_response()

    outputs, finished = scheduler._process_batch_responses([response])
    assert finished == {"success"}
    assert outputs[0].finished is True

    performance = scheduler.performance.snapshot()
    assert performance.model_name == "model-under-test"
    assert performance.requests_succeeded == 1
    assert performance.prompt_tokens == 5
    assert performance.completion_tokens == 3
    assert performance.ttft_seconds_count == 1
    assert performance.ttft_seconds_sum >= 0.04
    assert performance.decode_observations == 1

    # Re-delivery of the same terminal response must not double-count.
    scheduler.performance.record_finished_performance(request)
    assert scheduler.performance.snapshot().requests_succeeded == 1


def test_scheduler_does_not_commit_success_when_later_response_fails():
    scheduler = _scheduler()
    _running_request(scheduler, "first")
    second = _running_request(scheduler, "second")
    scheduler.uid_to_request_id[1] = "first"
    scheduler.uid_to_request_id[2] = "second"
    first_response = _terminal_response()
    second_response = _terminal_response()
    second_response.uid = 2
    second.append_output_token = MagicMock(side_effect=RuntimeError("later failure"))

    with pytest.raises(RuntimeError, match="later failure"):
        scheduler._process_batch_responses([first_response, second_response])

    performance = scheduler.performance.snapshot()
    assert performance.requests_succeeded == 0


def test_scheduler_records_explicit_cancellation_once():
    pytest.importorskip("mlx")

    scheduler = _scheduler()
    _running_request(scheduler, "cancelled")

    assert scheduler._do_abort_request("cancelled") is True
    performance = scheduler.performance.snapshot()
    assert performance.requests_cancelled == 1
    assert performance.prompt_tokens == 5
    assert performance.completion_tokens == 2

    scheduler.performance.record_cancelled_performance(scheduler.requests["cancelled"])
    assert scheduler.performance.snapshot().requests_cancelled == 1


def test_waiting_cancellation_records_no_unprocessed_prompt_tokens():
    from vllm_mlx.request import Request, SamplingParams

    scheduler = _scheduler()
    request = Request("waiting", "queued prompt", SamplingParams(max_tokens=8))
    request.num_prompt_tokens = 5
    scheduler.requests[request.request_id] = request
    scheduler.waiting.append(request)

    assert scheduler._do_abort_request(request.request_id) is True

    performance = scheduler.performance.snapshot()
    assert performance.requests_cancelled == 1
    assert performance.prompt_tokens == 0


def test_mllm_waiting_cancellation_records_zero_prompt_tokens():
    pytest.importorskip("mlx")

    from vllm_mlx.mllm_scheduler import MLLMScheduler, MLLMSchedulerConfig

    processor = MagicMock()
    processor.tokenizer = MagicMock()
    scheduler = MLLMScheduler(
        MagicMock(),
        processor,
        MLLMSchedulerConfig(),
        model_name="mllm-cancel-test",
    )
    request_id = scheduler.add_request("queued prompt")
    scheduler.requests[request_id].num_prompt_tokens = 5

    scheduler._do_abort_request(request_id)

    performance = scheduler.performance.snapshot()
    assert performance.requests_cancelled == 1
    assert performance.prompt_tokens == 0


def test_mllm_reset_accounts_waiting_and_running_requests():
    pytest.importorskip("mlx")

    from vllm_mlx.mllm_scheduler import MLLMScheduler, MLLMSchedulerConfig
    from vllm_mlx.request import RequestStatus

    processor = MagicMock()
    processor.tokenizer = MagicMock()
    scheduler = MLLMScheduler(
        MagicMock(),
        processor,
        MLLMSchedulerConfig(),
        model_name="mllm-reset-test",
    )
    waiting_id = scheduler.add_request("queued prompt")
    running_id = scheduler.add_request("active prompt")
    waiting = scheduler.requests[waiting_id]
    running = scheduler.requests[running_id]
    waiting.num_prompt_tokens = 7
    running.num_prompt_tokens = 5
    scheduler.waiting.remove(running)
    running.status = RequestStatus.RUNNING
    scheduler.running[running_id] = running

    scheduler.reset()

    performance = scheduler.performance.snapshot()
    assert performance.requests_cancelled == 2
    assert performance.prompt_tokens == 5
    assert scheduler.requests == {}
    assert not scheduler.waiting
    assert scheduler.running == {}


def test_scheduler_records_reconciled_orphan_cancellation():
    pytest.importorskip("mlx")

    scheduler = _scheduler()
    request = _running_request(scheduler, "disconnect-orphan")
    scheduler.remove_finished_request(request.request_id)

    scheduler.step()

    performance = scheduler.performance.snapshot()
    assert performance.requests_cancelled == 1
    assert performance.prompt_tokens == 5
    assert performance.completion_tokens == 2
    assert request.request_id not in scheduler.running


def test_scheduler_records_generation_recovery_failures():
    pytest.importorskip("mlx")

    scheduler = _scheduler()
    _running_request(scheduler, "generation-failure")
    scheduler.batch_generator.next.side_effect = RuntimeError("Metal OOM")

    output = scheduler.step()

    assert output.finished_request_ids == {"generation-failure"}
    assert output.outputs[0].error is not None
    performance = scheduler.performance.snapshot()
    assert performance.requests_failed == 1
    assert performance.total_requests == 1
    assert performance.prompt_tokens == 5
    assert performance.completion_tokens == 2
    assert performance.ttft_seconds_count == 1
    assert performance.decode_observations == 1


def test_mllm_global_failure_does_not_charge_queued_prompt_tokens():
    pytest.importorskip("mlx")

    from vllm_mlx.mllm_scheduler import (
        MLLMRequest,
        MLLMScheduler,
        MLLMSchedulerConfig,
    )
    from vllm_mlx.request import RequestStatus

    processor = MagicMock()
    processor.tokenizer = MagicMock()
    scheduler = MLLMScheduler(
        MagicMock(),
        processor,
        MLLMSchedulerConfig(),
        model_name="model-under-test",
    )
    running = MLLMRequest(request_id="running", prompt="started")
    running.status = RequestStatus.RUNNING
    running.num_prompt_tokens = 5
    waiting = MLLMRequest(request_id="waiting", prompt="not started")
    waiting.num_prompt_tokens = 7
    scheduler.requests = {running.request_id: running, waiting.request_id: waiting}
    scheduler.running = {running.request_id: running}
    scheduler.waiting.append(waiting)

    output = scheduler._fail_all_inflight(RuntimeError("Metal OOM"))

    assert output.finished_request_ids == {"running", "waiting"}
    performance = scheduler.performance.snapshot()
    assert performance.requests_failed == 2
    assert performance.prompt_tokens == 5


@pytest.mark.asyncio
async def test_engine_loop_records_pending_failures():
    pytest.importorskip("mlx")

    from vllm_mlx.engine_core import EngineConfig, EngineCore

    engine = EngineCore(
        MagicMock(), MagicMock(), EngineConfig(model_name="model-under-test")
    )

    from types import SimpleNamespace

    failed_request = SimpleNamespace(
        request_id="failure",
        arrival_time=time.time() - 0.25,
        first_token_time=time.time() - 0.2,
        num_prompt_tokens=5,
        num_output_tokens=2,
    )

    class _BoomScheduler:
        performance = ModelPerformanceLedger("model-under-test")

        def has_requests(self):
            return True

        def step(self):
            raise RuntimeError("Metal command buffer failure")

        def add_request(self, *_args, **_kwargs):
            pass

        def abort_request(self, *_args, **_kwargs):
            return True

        def remove_finished_request(self, *_args, **_kwargs):
            pass

        def get_request(self, request_id):
            return failed_request if request_id == failed_request.request_id else None

    engine.scheduler = _BoomScheduler()
    engine._output_collectors["failure"] = RequestOutputCollector(aggregate=True)
    engine._finished_events["failure"] = asyncio.Event()
    engine._running = True
    loop_task = asyncio.create_task(engine._engine_loop())
    try:
        await asyncio.wait_for(engine._finished_events["failure"].wait(), timeout=1)
    finally:
        engine._running = False
        loop_task.cancel()
        try:
            await loop_task
        except asyncio.CancelledError:
            pass

    performance = engine.scheduler.performance.snapshot()
    assert performance.requests_failed == 1
    assert performance.prompt_tokens == 5
    assert performance.completion_tokens == 2
    final = engine._output_collectors["failure"].get_nowait()
    assert final is not None and final.finished and final.error


@pytest.mark.asyncio
async def test_engine_loop_does_not_count_failure_before_scheduler_admission():
    pytest.importorskip("mlx")

    from vllm_mlx.engine_core import EngineConfig, EngineCore

    engine = EngineCore(
        MagicMock(), MagicMock(), EngineConfig(model_name="model-under-test")
    )

    class _AdmissionRaceScheduler:
        performance = ModelPerformanceLedger("model-under-test")

        def has_requests(self):
            return True

        def step(self):
            raise RuntimeError("Metal command buffer failure")

        def get_request(self, _request_id):
            return None

        def add_request(self, *_args, **_kwargs):
            pass

        def abort_request(self, *_args, **_kwargs):
            return True

        def remove_finished_request(self, *_args, **_kwargs):
            pass

    engine.scheduler = _AdmissionRaceScheduler()
    engine._output_collectors["not-admitted"] = RequestOutputCollector(aggregate=True)
    engine._finished_events["not-admitted"] = asyncio.Event()
    engine._running = True
    loop_task = asyncio.create_task(engine._engine_loop())
    try:
        await asyncio.wait_for(
            engine._finished_events["not-admitted"].wait(), timeout=1
        )
    finally:
        engine._running = False
        loop_task.cancel()
        try:
            await loop_task
        except asyncio.CancelledError:
            pass

    assert engine.scheduler.performance.snapshot().total_requests == 0


def test_metrics_renders_model_performance_series():
    from types import SimpleNamespace

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from vllm_mlx.config import reset_config
    from vllm_mlx.routes.metrics import _reset_accumulator_for_tests, router
    from vllm_mlx.runtime.model_performance import get_model_performance_ledger

    ledger = get_model_performance_ledger("gemma-4-12b")
    ledger.record_success(
        "1",
        prompt_tokens=7,
        completion_tokens=4,
        ttft_seconds=0.07,
        decode_tokens_per_second=120,
    )
    ledger.record_success(
        "2",
        prompt_tokens=5,
        completion_tokens=3,
        ttft_seconds=0.4,
        decode_tokens_per_second=80,
    )
    ledger.record_success(
        "3",
        prompt_tokens=3,
        completion_tokens=2,
        ttft_seconds=0.9,
        decode_tokens_per_second=20,
    )
    ledger.record_failure("4")

    cfg = reset_config()
    cfg.model_name = "gemma-4-12b"
    _reset_accumulator_for_tests()
    app = FastAPI()
    app.include_router(router)
    cfg.engine = SimpleNamespace(
        get_stats=lambda: {"model_performance": ledger.snapshot().__dict__}
    )
    body = TestClient(app).get("/metrics").text

    for metric_name in (
        "rapid_mlx_model_requests_total",
        "rapid_mlx_model_ttft_seconds",
        "rapid_mlx_model_decode_tokens_per_second",
    ):
        assert body.count(f"# TYPE {metric_name} ") == 1
        assert body.count(f"# HELP {metric_name} ") == 1

    assert (
        'rapid_mlx_model_requests_total{model="gemma-4-12b",outcome="succeeded"} 3'
        in body
    )
    assert (
        'rapid_mlx_model_requests_total{model="gemma-4-12b",outcome="failed"} 1' in body
    )
    assert 'outcome="total"' not in body
    outcome_samples = [
        float(line.rsplit(" ", 1)[1])
        for line in body.splitlines()
        if line.startswith("rapid_mlx_model_requests_total{")
    ]
    assert sum(outcome_samples) == 4
    assert 'rapid_mlx_model_prompt_tokens_total{model="gemma-4-12b"} 15' in body
    assert 'rapid_mlx_model_completion_tokens_total{model="gemma-4-12b"} 9' in body
    assert 'rapid_mlx_model_ttft_seconds_bucket{model="gemma-4-12b",le="0.1"} 1' in body
    assert (
        'rapid_mlx_model_ttft_seconds_bucket{model="gemma-4-12b",le="+Inf"} 3' in body
    )
    assert (
        'rapid_mlx_model_decode_tokens_per_second_bucket{model="gemma-4-12b",le="50"} 1'
        in body
    )
    assert 'rapid_mlx_model_ttft_seconds_max{model="gemma-4-12b"} 0.9' in body
    assert 'rapid_mlx_model_ttft_seconds_count{model="gemma-4-12b"} 3' in body
    assert 'rapid_mlx_model_ttft_seconds_sum{model="gemma-4-12b"} 1.37' in body
    assert (
        'rapid_mlx_model_decode_tokens_per_second_last{model="gemma-4-12b"} 20' in body
    )

    reset_config()
    _reset_accumulator_for_tests()


def test_model_performance_renderer_ignores_missing_or_malformed_payload():
    from vllm_mlx.routes.metrics import _render_model_performance

    assert _render_model_performance({}) == []
    assert _render_model_performance({"model_performance": "malformed"}) == []


def test_metrics_preserves_unseen_events_across_scheduler_reloads():
    from types import SimpleNamespace

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from vllm_mlx.config import reset_config
    from vllm_mlx.routes.metrics import _reset_accumulator_for_tests, router
    from vllm_mlx.runtime.model_performance import get_model_performance_ledger

    first = get_model_performance_ledger("reloadable-model")
    first.record_success(
        "first",
        prompt_tokens=7,
        completion_tokens=2,
        ttft_seconds=0.2,
        decode_tokens_per_second=10,
    )
    current = {"ledger": first}
    cfg = reset_config()
    _reset_accumulator_for_tests()
    cfg.engine = SimpleNamespace(
        get_stats=lambda: {"model_performance": current["ledger"].snapshot().__dict__}
    )
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    first_body = client.get("/metrics").text
    assert (
        'rapid_mlx_model_requests_total{model="reloadable-model",outcome="succeeded"} 1'
        in first_body
    )

    # This request finishes after the last scrape but before the scheduler is
    # replaced. Process-owned ledger state must retain it without a final scrape.
    first.record_success(
        "between-scrapes",
        prompt_tokens=6,
        completion_tokens=1,
        ttft_seconds=0.8,
        decode_tokens_per_second=30,
    )
    replacement = get_model_performance_ledger("reloadable-model")
    assert replacement is first
    replacement.record_success(
        "second",
        prompt_tokens=3,
        completion_tokens=4,
        ttft_seconds=0.1,
        decode_tokens_per_second=5,
    )
    current["ledger"] = replacement
    second_body = client.get("/metrics").text

    assert (
        'rapid_mlx_model_requests_total{model="reloadable-model",outcome="succeeded"} 3'
        in second_body
    )
    assert (
        'rapid_mlx_model_prompt_tokens_total{model="reloadable-model"} 16'
        in second_body
    )
    assert (
        'rapid_mlx_model_completion_tokens_total{model="reloadable-model"} 7'
        in second_body
    )
    assert (
        'rapid_mlx_model_ttft_seconds_bucket{model="reloadable-model",le="+Inf"} 3'
        in second_body
    )
    assert (
        'rapid_mlx_model_ttft_seconds_count{model="reloadable-model"} 3' in second_body
    )
    assert (
        'rapid_mlx_model_ttft_seconds_sum{model="reloadable-model"} 1.1' in second_body
    )
    assert (
        'rapid_mlx_model_decode_tokens_per_second_count{model="reloadable-model"} 3'
        in second_body
    )
    assert (
        'rapid_mlx_model_decode_tokens_per_second_sum{model="reloadable-model"} 45.0'
        in second_body
    )
    assert (
        'rapid_mlx_model_ttft_seconds_max{model="reloadable-model"} 0.8' in second_body
    )
    assert (
        'rapid_mlx_model_decode_tokens_per_second_max{model="reloadable-model"} 30.0'
        in second_body
    )

    # Unloading the active engine must not make the retained model series stale.
    cfg.engine = None
    unloaded_body = client.get("/metrics").text
    assert (
        'rapid_mlx_model_requests_total{model="reloadable-model",outcome="succeeded"} 3'
        in unloaded_body
    )

    switched = get_model_performance_ledger("second-model")
    switched.record_failure("failure-after-switch")
    switched_body = client.get("/metrics").text
    assert (
        'rapid_mlx_model_requests_total{model="reloadable-model",outcome="succeeded"} 3'
        in switched_body
    )
    assert (
        'rapid_mlx_model_requests_total{model="second-model",outcome="failed"} 1'
        in switched_body
    )
    assert switched_body.count("# HELP rapid_mlx_model_requests_total ") == 1
    assert switched_body.count("# TYPE rapid_mlx_model_requests_total ") == 1

    reset_config()
    _reset_accumulator_for_tests()


def test_process_model_registry_is_lru_bounded():
    from vllm_mlx.runtime.model_performance import (
        _MODEL_LEDGER_REGISTRY,
        _RETIRED_MODEL_SNAPSHOTS,
        get_model_performance_ledger,
        get_model_performance_snapshots,
    )

    first = get_model_performance_ledger("model-0")
    first.record_success(
        "before-retirement",
        prompt_tokens=3,
        completion_tokens=2,
        ttft_seconds=0.2,
        decode_tokens_per_second=10,
    )
    for index in range(1, MODEL_LEDGER_REGISTRY_LIMIT + 1):
        get_model_performance_ledger(f"model-{index}")

    retained = get_model_performance_snapshots()
    assert len(_MODEL_LEDGER_REGISTRY) == MODEL_LEDGER_REGISTRY_LIMIT
    assert len(retained) == MODEL_LEDGER_REGISTRY_LIMIT + 1
    retired = next(
        snapshot for snapshot in retained if snapshot.model_name == "model-0"
    )
    assert retired.requests_succeeded == 1
    assert any(
        snapshot.model_name == f"model-{MODEL_LEDGER_REGISTRY_LIMIT}"
        for snapshot in retained
    )

    from types import SimpleNamespace

    post_eviction = SimpleNamespace(
        request_id="after-retirement",
        status=SimpleNamespace(name="RUNNING"),
        num_prompt_tokens=5,
        num_output_tokens=4,
    )
    assert first.record_request_performance(post_eviction, "succeeded") is True
    revived = get_model_performance_ledger("model-0")
    assert revived is first
    continuity = first.snapshot()
    assert continuity.requests_succeeded == 2
    assert continuity.prompt_tokens == 8
    assert len(_MODEL_LEDGER_REGISTRY) == MODEL_LEDGER_REGISTRY_LIMIT

    for index in range(
        MODEL_LEDGER_REGISTRY_LIMIT + 1,
        MODEL_LEDGER_REGISTRY_LIMIT + RETIRED_MODEL_SNAPSHOT_LIMIT + 3,
    ):
        get_model_performance_ledger(f"model-{index}")
    assert len(_MODEL_LEDGER_REGISTRY) == MODEL_LEDGER_REGISTRY_LIMIT
    assert len(_RETIRED_MODEL_SNAPSHOTS) == RETIRED_MODEL_SNAPSHOT_LIMIT
    assert len(get_model_performance_snapshots()) <= (
        MODEL_LEDGER_REGISTRY_LIMIT + RETIRED_MODEL_SNAPSHOT_LIMIT
    )
