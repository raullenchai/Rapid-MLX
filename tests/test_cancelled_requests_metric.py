# SPDX-License-Identifier: Apache-2.0
"""Observability tests for the M-01 cancelled-requests counter.

Background
----------
Mei r0.8.1 and Yana r3 both flagged that ``/metrics`` reported
``rapid_mlx_requests_processed_total = 0`` after fifty client-cancelled
streaming requests. ``num_requests_processed`` deliberately excludes
aborted requests (a request that never produced an EOS-bounded response
shouldn't be billed as "completed"), so operators staring at the
counter-of-record can't distinguish "model is idle" from "every caller
is bailing out before EOS". That's the M-01 gap.

The fix exposes two new scheduler counters via ``Scheduler.get_stats()``
and ``Scheduler`` MLLM-twin, rendered as Prometheus counters in
``routes/metrics.py``:

* ``rapid_mlx_requests_cancelled_total`` — +1 per public-API abort the
  scheduler accepted (client disconnect, explicit cancel route,
  timeout, or internal abort), once per ``request_id`` irrespective of
  how many idempotent re-enqueues fire.
* ``rapid_mlx_requests_cancelled_via_disconnect_total`` — sub-counter
  attributing the subset triggered through the disconnect_guard
  ``_force_abort_request`` path, so the (total - disconnect) gap
  surfaces explicit-cancel + timeout traffic for capacity planning.

These are observability-only counters. ``Scheduler.abort_request`` and
``_force_abort_request`` semantics are unchanged — the counters bolt on
to existing returns. Defaults to zero on engines that never see an
abort so dashboards never flip to "no data" after a deploy (mirrors the
M-02 PFlash flat-line treatment).

The tests below cover four behavioural surfaces:

1. **Scheduler-side total counter** — increments once per accepted
   abort, never on rejected unknown-id aborts, idempotent against
   double-enqueue of the same id, and surfaced through ``get_stats``.
2. **Scheduler-side disconnect sub-counter** — bumps only when
   ``record_disconnect_abort`` is invoked, deduplicated by
   ``_disconnect_abort_ids`` so the three-branch helper-layer fire
   (disconnect + GeneratorExit + finally) attributes once per request.
3. **End-to-end disconnect_guard** — `_force_abort_request` calls
   `scheduler.abort_request` AND `scheduler.record_disconnect_abort`,
   driving both counters to +N for N aborted requests. Three streaming
   requests aborted via raw httpx aclose → counter = 3; two requests
   that complete normally → counter unchanged.
4. **Route-side Prometheus render** — both counters surface with
   correct HELP/TYPE lines, zero-default when the keys are absent
   (older engines), and stable monotonic values.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm_mlx.request import Request, SamplingParams
from vllm_mlx.scheduler import Scheduler, SchedulerConfig

# ---------------------------------------------------------------------------
# Helpers — real scheduler driven without the live model
# ---------------------------------------------------------------------------


class _DummyTokenizer:
    eos_token_id = None

    def encode(self, prompt):
        if isinstance(prompt, str):
            return [ord(c) for c in prompt]
        return list(prompt)

    def decode(self, token_ids):
        return "".join(chr(t) for t in token_ids)


def _make_scheduler() -> Scheduler:
    """Build a real ``Scheduler`` against an identity tokenizer.

    Mirrors the helper in ``test_pflash_metrics.py`` so the M-01
    counters are exercised by the same scheduler entry points that
    serve production traffic.
    """
    return Scheduler(
        model=object(),
        tokenizer=_DummyTokenizer(),
        config=SchedulerConfig(
            enable_prefix_cache=False,
            use_memory_aware_cache=False,
        ),
    )


def _admit(scheduler: Scheduler, request_id: str) -> None:
    """Push a request into the scheduler so ``abort_request`` will
    consider its ``request_id`` "known" via ``self.requests``.

    The scheduler's abort gate returns False for an unknown id, so
    every "abort that should count" must first appear in this dict.
    """
    request = Request(request_id, list(range(8)), SamplingParams(max_tokens=4))
    scheduler.add_request(request)


# ---------------------------------------------------------------------------
# Scheduler-side: total counter
# ---------------------------------------------------------------------------


def test_total_counter_starts_at_zero():
    """Fresh scheduler exposes both counters at zero through ``get_stats``.

    Dashboard panels treat an absent series as "no data" and trigger
    spurious alerts after a deploy; we always want a flat-line zero
    instead.
    """
    scheduler = _make_scheduler()
    stats = scheduler.get_stats()
    assert stats["num_requests_cancelled"] == 0
    assert stats["num_requests_cancelled_via_disconnect"] == 0


def test_total_counter_increments_once_per_accepted_abort():
    """Three known request_ids aborted → counter = 3.

    Sanity check that the counter advances per public-API abort, not
    per token or per batch step.
    """
    scheduler = _make_scheduler()
    for index in range(3):
        request_id = f"req-{index}"
        _admit(scheduler, request_id)
        assert scheduler.abort_request(request_id) is True
    assert scheduler.get_stats()["num_requests_cancelled"] == 3


def test_total_counter_not_bumped_for_unknown_request_id():
    """``abort_request`` returns False for unknown ids and the
    counter does NOT advance.

    F-151 hardening: an attacker who pokes random ids must not be
    able to inflate the counter (it would be a free signal that
    /metrics is observable; more importantly, it would corrupt the
    series and trigger false alarms).
    """
    scheduler = _make_scheduler()
    assert scheduler.abort_request("nonexistent-request-id") is False
    assert scheduler.get_stats()["num_requests_cancelled"] == 0


def test_total_counter_is_idempotent_against_double_enqueue():
    """Two ``abort_request`` calls for the same id → counter = 1.

    The scheduler is intentionally idempotent against double-enqueue
    (``Scheduler.abort_request`` docstring) because the disconnect
    guard fires from multiple branches per request. Without the de-dup
    gate the counter would double-count every disconnect-triggered
    abort.
    """
    scheduler = _make_scheduler()
    _admit(scheduler, "req-double")
    assert scheduler.abort_request("req-double") is True
    assert scheduler.abort_request("req-double") is True  # idempotent
    assert scheduler.get_stats()["num_requests_cancelled"] == 1


def test_total_counter_survives_reset_unchanged():
    """``reset()`` clears in-flight aborts but NOT the lifetime counter.

    Prometheus counters MUST be monotonic — a step backward would be
    interpreted as a process restart and corrupt rate() / increase()
    calculations on the scraper side.
    """
    scheduler = _make_scheduler()
    _admit(scheduler, "req-pre-reset")
    scheduler.abort_request("req-pre-reset")
    assert scheduler.get_stats()["num_requests_cancelled"] == 1

    scheduler.reset()
    assert scheduler.get_stats()["num_requests_cancelled"] == 1


# ---------------------------------------------------------------------------
# Scheduler-side: disconnect sub-counter
# ---------------------------------------------------------------------------


def test_disconnect_sub_counter_bumps_on_record_call():
    """``record_disconnect_abort`` advances the sub-counter."""
    scheduler = _make_scheduler()
    _admit(scheduler, "req-disc")
    scheduler.abort_request("req-disc")
    scheduler.record_disconnect_abort("req-disc")
    stats = scheduler.get_stats()
    assert stats["num_requests_cancelled"] == 1
    assert stats["num_requests_cancelled_via_disconnect"] == 1


def test_disconnect_sub_counter_dedupes_per_request_id():
    """Three ``record_disconnect_abort`` calls for the same id → sub = 1.

    The disconnect_guard fires ``_force_abort_request`` from the
    ``if disconnect_task in done`` branch, the ``except GeneratorExit``
    branch, AND the ``finally`` belt-and-suspenders. Without the
    per-id de-dup the sub-counter would over-count by up to 3x per
    disconnect event.
    """
    scheduler = _make_scheduler()
    _admit(scheduler, "req-multi")
    scheduler.abort_request("req-multi")
    scheduler.record_disconnect_abort("req-multi")
    scheduler.record_disconnect_abort("req-multi")
    scheduler.record_disconnect_abort("req-multi")
    assert scheduler.get_stats()["num_requests_cancelled_via_disconnect"] == 1


def test_disconnect_sub_counter_silent_on_empty_request_id():
    """Empty string / None ``request_id`` is a no-op, not a crash.

    The disconnect_guard sometimes invokes ``_force_abort_request``
    with an empty holder (engine cancelled before ``add_request``
    returned). The helper must swallow that path silently rather than
    propagate the no-op into the live disconnect flow.
    """
    scheduler = _make_scheduler()
    scheduler.record_disconnect_abort("")
    scheduler.record_disconnect_abort(None)  # type: ignore[arg-type]
    assert scheduler.get_stats()["num_requests_cancelled_via_disconnect"] == 0


def test_disconnect_sub_counter_decoupled_from_total():
    """An explicit cancel (no disconnect attribution) leaves the
    sub-counter at zero.

    The total counter ticks on every accepted abort; the sub-counter
    only ticks when ``_force_abort_request`` (disconnect path) was
    the trigger. The (total - via_disconnect) gap is the operator's
    signal for explicit-cancel + timeout traffic — if the sub-counter
    were entangled with the total this gap would always be zero.
    """
    scheduler = _make_scheduler()
    _admit(scheduler, "req-explicit")
    scheduler.abort_request("req-explicit")  # explicit cancel route
    # No ``record_disconnect_abort`` call.
    stats = scheduler.get_stats()
    assert stats["num_requests_cancelled"] == 1
    assert stats["num_requests_cancelled_via_disconnect"] == 0


# ---------------------------------------------------------------------------
# Helper-layer: _force_abort_request feeds both counters end-to-end
# ---------------------------------------------------------------------------


class _CountingScheduler:
    """Real-shaped scheduler stand-in that records both the public-API
    abort and the disconnect attribution call.

    Lets the helper-layer test pin the exact contract between
    ``_force_abort_request`` and the scheduler-side counters without
    spinning up a full scheduler + tokenizer.
    """

    def __init__(self):
        self.aborts: list[str] = []
        self.disconnect_records: list[str] = []

    def abort_request(self, request_id: str) -> bool:
        self.aborts.append(request_id)
        return True

    def record_disconnect_abort(self, request_id: str) -> None:
        self.disconnect_records.append(request_id)


class _Engine:
    """Engine stub that exposes ``scheduler`` directly (the simple
    pre-BatchedEngine shape ``_resolve_sync_scheduler_for_abort``
    matches first).
    """

    def __init__(self):
        self.scheduler = _CountingScheduler()


def test_force_abort_bumps_disconnect_subcounter():
    """``_force_abort_request`` calls ``record_disconnect_abort``
    once after a successful sync abort.

    This is the contract that ties the helper-layer disconnect path
    to the scheduler-side sub-counter; deleting the
    ``_record_disconnect_abort_on_scheduler`` call would silently
    drop the attribution and the operator's "via disconnect" series
    would stay flat through real disconnects.
    """
    from vllm_mlx.service.helpers import _force_abort_request

    engine = _Engine()
    holder = ["req-disconnect"]

    fired = _force_abort_request(engine, holder)

    assert fired is True
    assert engine.scheduler.aborts == ["req-disconnect"]
    assert engine.scheduler.disconnect_records == ["req-disconnect"]


def test_force_abort_does_not_record_when_sync_abort_rejected():
    """If ``abort_request`` returned False (unknown id), the helper
    must NOT call ``record_disconnect_abort``.

    Pre-fix candidate bug: a helper that records unconditionally
    would inflate the sub-counter every time a stale holder fires
    through ``_force_abort_request`` (e.g. when the request finished
    before the disconnect_guard tore down). The total counter is
    already protected against this by ``Scheduler.abort_request``
    returning False for unknown ids; the helper must propagate that
    gate to the sub-counter or the two series will drift.
    """
    from vllm_mlx.service.helpers import _force_abort_request

    class _RejectingScheduler:
        def __init__(self):
            self.disconnect_records: list[str] = []

        def abort_request(self, request_id: str) -> bool:
            return False  # unknown id — F-151 path

        def record_disconnect_abort(self, request_id: str) -> None:
            self.disconnect_records.append(request_id)

    class _RejectingEngine:
        def __init__(self):
            self.scheduler = _RejectingScheduler()

    engine = _RejectingEngine()
    holder = ["req-unknown"]

    fired = _force_abort_request(engine, holder)

    # The sync abort entry returned False but the helper still
    # returns True per its docstring (it DID dispatch). The
    # sub-counter, however, must NOT advance.
    assert fired is True
    assert engine.scheduler.disconnect_records == []


def test_force_abort_does_not_crash_when_record_method_absent():
    """Older schedulers without ``record_disconnect_abort`` continue
    to work — the helper swallows ``AttributeError`` silently and
    the total counter alone keeps surfacing the abort.

    Critical because the helper layer sits across a public API
    surface that downstream forks / external schedulers depend on.
    A regression here would break every non-rapid_mlx scheduler.
    """
    from vllm_mlx.service.helpers import _force_abort_request

    class _LegacyScheduler:
        def __init__(self):
            self.aborts: list[str] = []

        def abort_request(self, request_id: str) -> bool:
            self.aborts.append(request_id)
            return True

        # NB: no record_disconnect_abort.

    class _LegacyEngine:
        def __init__(self):
            self.scheduler = _LegacyScheduler()

    engine = _LegacyEngine()
    holder = ["req-legacy"]

    fired = _force_abort_request(engine, holder)

    assert fired is True
    assert engine.scheduler.aborts == ["req-legacy"]


@pytest.mark.asyncio
async def test_force_abort_attribution_walks_production_batched_engine_shape():
    """Production ``BatchedEngine`` over ``AsyncEngineCore`` hides the
    scheduler at ``engine._engine.engine.scheduler`` (one hop deeper
    than the abort resolver covers, because ``AsyncEngineCore`` wraps
    ``EngineCore`` via ``self.engine``).

    The attribution resolver MUST dig the extra hop or the
    via_disconnect sub-counter stays flat through every real
    production disconnect — which is the exact symptom Mei + Yana
    flagged the cancel-rate counter for in the first place. A static
    introspection / unit-test wins; the alternative is to ship the
    fix and discover via dashboards two weeks later that the gap is
    always 100% of total.

    The production sync abort path falls through to the async
    fallback (also covered by the helper now), but the attribution
    must still land on the right scheduler so the (total -
    via_disconnect) gap reflects reality.
    """
    from vllm_mlx.service.helpers import _force_abort_request

    class _SyncScheduler:
        def __init__(self):
            self.disconnect_records: list[str] = []
            self._async_abort_calls: list[str] = []

        def record_disconnect_abort(self, request_id: str) -> None:
            self.disconnect_records.append(request_id)

    class _EngineCoreLike:
        """Mirrors the production ``EngineCore`` shape — owns the
        scheduler directly."""

        def __init__(self):
            self.scheduler = _SyncScheduler()

    class _AsyncEngineCoreLike:
        """Production-shaped ``AsyncEngineCore`` — has ``.engine``
        (the inner ``EngineCore``) but NO direct ``.scheduler``,
        and exposes an ``async abort_request`` shim that lands on
        ``self.engine.engine.abort_request`` in production. Crucially
        the helper's sync-abort resolver returns ``None`` here, so
        the async fallback fires (matching the production text path).
        """

        def __init__(self):
            self.engine = _EngineCoreLike()

        async def abort_request(self, request_id: str) -> bool:
            # Production-fidelity: the async shim ultimately reaches
            # ``self.engine.scheduler.abort_request`` (sync).
            self.engine.scheduler._async_abort_calls.append(request_id)
            return True

    class _BatchedEngineLike:
        def __init__(self):
            self._engine = _AsyncEngineCoreLike()
            self._is_mllm = False

        async def abort_request(self, request_id: str) -> bool:
            return await self._engine.abort_request(request_id)

    engine = _BatchedEngineLike()
    holder = ["req-prod-shape"]

    fired = _force_abort_request(engine, holder)
    # Drain the fire-and-forget task the helper queued via
    # ``asyncio.ensure_future`` so the test doesn't trip the
    # "Task was destroyed but it is pending" runtime warning. The
    # production code intentionally fires-and-forgets (operators see
    # the WARNING log) — the test owns the loop, so we drain.
    await asyncio.sleep(0)

    # The sync resolver returned None (no sync abort path), so the
    # helper hits the async fallback branch and returns False per
    # the docstring contract.
    assert fired is False
    # But the attribution MUST still have landed — that's the bug
    # we're guarding against.
    assert engine._engine.engine.scheduler.disconnect_records == ["req-prod-shape"]


def test_force_abort_attribution_walks_active_backend():
    """Cancel-attribution resolver respects ``_is_mllm`` like the abort
    resolver — text-active engines record on ``_engine.scheduler``,
    MLLM-active engines on ``_mllm_scheduler``.

    The codex r2 BLOCKING #1 finding on PR #777 pinned this for the
    abort resolver; the attribution resolver must follow the same
    rule or it would bump the sub-counter on the wrong scheduler.
    """
    from vllm_mlx.service.helpers import _force_abort_request

    class _SyncSched:
        def __init__(self, name):
            self.name = name
            self.aborts: list[str] = []
            self.disconnect_records: list[str] = []

        def abort_request(self, request_id: str) -> bool:
            self.aborts.append(request_id)
            return True

        def record_disconnect_abort(self, request_id: str) -> None:
            self.disconnect_records.append(request_id)

    class _Inner:
        def __init__(self):
            self.scheduler = _SyncSched("text")

    class _DualEngine:
        def __init__(self, is_mllm: bool):
            self._is_mllm = is_mllm
            self._engine = _Inner()
            self._mllm_scheduler = _SyncSched("mllm")

    # Text path
    text_engine = _DualEngine(is_mllm=False)
    _force_abort_request(text_engine, ["req-text"])
    assert text_engine._engine.scheduler.disconnect_records == ["req-text"]
    assert text_engine._mllm_scheduler.disconnect_records == []

    # MLLM path
    mllm_engine = _DualEngine(is_mllm=True)
    _force_abort_request(mllm_engine, ["req-mllm"])
    assert mllm_engine._mllm_scheduler.disconnect_records == ["req-mllm"]
    assert mllm_engine._engine.scheduler.disconnect_records == []


# ---------------------------------------------------------------------------
# End-to-end: streaming-route abort drives both counters via TestClient
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_three_aborted_streaming_requests_advance_counters_by_three():
    """Fire 3 streaming requests through ``_disconnect_guard`` and
    abort each via ``GeneratorExit``. The total counter and the
    disconnect sub-counter both increment by 3.

    This is the headline behaviour Mei r0.8.1 / Yana r3 asked for:
    aborted streams MUST be visible in /metrics so operators see the
    cancel-rate. Pre-fix the counters didn't exist; post-fix the test
    pins the +N contract.
    """
    from vllm_mlx.service.helpers import _disconnect_guard

    engine = _Engine()

    aborted_holders: list[list[str]] = []
    for i in range(3):
        holder = [f"req-stream-{i}"]
        aborted_holders.append(holder)

        async def upstream():
            yield 'data: {"chunk":"hello"}\n\n'
            # Wait long enough that GeneratorExit reaches us before
            # the upstream exhausts naturally.
            await asyncio.sleep(60)
            yield "data: [DONE]\n\n"

        class _NeverDisconnects:
            async def is_disconnected(self) -> bool:
                return False

        guard = _disconnect_guard(
            upstream(),
            _NeverDisconnects(),
            poll_interval=0.05,
            engine=engine,
            request_id_holder=holder,
            keepalive_seconds=0,
        )

        # Pull one chunk then close — simulates Starlette tearing
        # down the StreamingResponse mid-stream (Astrid r3 fingerprint).
        agen = guard.__aiter__()
        await agen.__anext__()
        await agen.aclose()

    # All three request_ids should be in the scheduler's abort list
    # AND the disconnect-attribution list. De-dup guarantees one
    # entry per id even when the helper fires from multiple branches.
    accepted_aborts = sorted(set(engine.scheduler.aborts))
    accepted_records = sorted(set(engine.scheduler.disconnect_records))
    assert accepted_aborts == [
        "req-stream-0",
        "req-stream-1",
        "req-stream-2",
    ]
    assert accepted_records == [
        "req-stream-0",
        "req-stream-1",
        "req-stream-2",
    ]


@pytest.mark.asyncio
async def test_two_completed_streaming_requests_leave_counter_unchanged():
    """Streams that exhaust normally do NOT advance the cancellation
    counters.

    ``_disconnect_guard.finally`` has a ``finished_normally`` flag
    (codex r1 NIT #3 on PR #777) that skips the belt-and-suspenders
    force-abort when the upstream drained cleanly via
    ``StopAsyncIteration``. Without that gate the cancellation
    counter would tick once per completed request and the operator's
    "cancel rate" panel would look like "100% of traffic" — useless.
    """
    from vllm_mlx.service.helpers import _disconnect_guard

    engine = _Engine()

    for i in range(2):
        holder = [f"req-clean-{i}"]

        async def upstream():
            yield 'data: {"chunk":"a"}\n\n'
            yield 'data: {"chunk":"b"}\n\n'
            yield "data: [DONE]\n\n"

        class _NeverDisconnects:
            async def is_disconnected(self) -> bool:
                return False

        guard = _disconnect_guard(
            upstream(),
            _NeverDisconnects(),
            poll_interval=0.05,
            engine=engine,
            request_id_holder=holder,
            keepalive_seconds=0,
        )

        chunks = []
        async for chunk in guard:
            chunks.append(chunk)
        # Sanity: the consumer pulled the full stream.
        assert "[DONE]" in chunks[-1]

    # Neither counter should have moved.
    assert engine.scheduler.aborts == []
    assert engine.scheduler.disconnect_records == []


# ---------------------------------------------------------------------------
# Route-side: counters surface in the /metrics Prometheus body
# ---------------------------------------------------------------------------


@pytest.fixture
def metrics_client():
    """FastAPI TestClient mounting only the metrics router.

    Mirrors ``test_metrics_route.metrics_client`` and ``test_pflash_
    metrics.metrics_client`` so the M-01 counters are exercised
    through the same render path as every other series.
    """
    from vllm_mlx.config import reset_config
    from vllm_mlx.routes.metrics import _reset_accumulator_for_tests, router

    cfg = reset_config()
    cfg.model_name = "qwen3-0.6b"
    _reset_accumulator_for_tests()

    app = FastAPI()
    app.include_router(router)
    yield SimpleNamespace(client=TestClient(app), cfg=cfg)
    reset_config()
    _reset_accumulator_for_tests()


def _fake_engine(stats: dict[str, Any]):
    return SimpleNamespace(get_stats=lambda: stats)


_CANCEL_STATS = {
    "num_waiting": 0,
    "num_running": 0,
    "num_requests_processed": 0,
    "total_prompt_tokens": 0,
    "total_completion_tokens": 0,
    "num_requests_cancelled": 5,
    "num_requests_cancelled_via_disconnect": 3,
}


def test_metrics_route_renders_total_cancelled_counter(metrics_client):
    """``rapid_mlx_requests_cancelled_total`` HELP / TYPE / value all
    present and the value matches the scheduler stat.
    """
    metrics_client.cfg.engine = _fake_engine(_CANCEL_STATS)
    body = metrics_client.client.get("/metrics").text

    assert "# HELP rapid_mlx_requests_cancelled_total" in body
    assert "# TYPE rapid_mlx_requests_cancelled_total counter" in body
    assert "rapid_mlx_requests_cancelled_total 5" in body


def test_metrics_route_renders_disconnect_subcounter(metrics_client):
    """``rapid_mlx_requests_cancelled_via_disconnect_total`` HELP /
    TYPE / value all present and the value matches the scheduler stat.
    """
    metrics_client.cfg.engine = _fake_engine(_CANCEL_STATS)
    body = metrics_client.client.get("/metrics").text

    assert "# HELP rapid_mlx_requests_cancelled_via_disconnect_total" in body
    assert "# TYPE rapid_mlx_requests_cancelled_via_disconnect_total counter" in body
    assert "rapid_mlx_requests_cancelled_via_disconnect_total 3" in body


def test_metrics_route_renders_zero_when_cancel_keys_missing(metrics_client):
    """Engines without the M-01 keys render flat-line zero rather than
    omit the series.

    Older non-rapid_mlx schedulers (downstream forks, custom backends)
    may not populate the keys at all. Dashboards configured against
    these series must not flip to "no data".
    """
    stats_without_cancel = {
        "num_requests_processed": 1,
        "total_prompt_tokens": 100,
        "total_completion_tokens": 5,
        "num_running": 0,
        "num_waiting": 0,
    }
    metrics_client.cfg.engine = _fake_engine(stats_without_cancel)
    body = metrics_client.client.get("/metrics").text

    assert "rapid_mlx_requests_cancelled_total 0" in body
    assert "rapid_mlx_requests_cancelled_via_disconnect_total 0" in body


def test_metrics_route_renders_zero_when_both_counters_zero(metrics_client):
    """Quiet engine — both counters render at zero (not absent)."""
    quiet_stats = {
        "num_waiting": 0,
        "num_running": 0,
        "num_requests_processed": 0,
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "num_requests_cancelled": 0,
        "num_requests_cancelled_via_disconnect": 0,
    }
    metrics_client.cfg.engine = _fake_engine(quiet_stats)
    body = metrics_client.client.get("/metrics").text

    assert "rapid_mlx_requests_cancelled_total 0" in body
    assert "rapid_mlx_requests_cancelled_via_disconnect_total 0" in body
