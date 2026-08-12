# SPDX-License-Identifier: Apache-2.0
"""Tests for hybrid-throttle retirement + admission-wave coalescing
(#1861 conc investigation).

Rows admitted at different scheduler steps carry ragged per-row offsets
for the batch's whole lifetime, keeping mlx-lm's batched attention on
the array-mask slow path (Qwen3.6-35B-A3B B=8: 190 vs 267.7 agg tok/s).
Two engine-side mechanisms fixed it:

* ``_resolve_hybrid_throttle`` — the #115 200ms admission spacing is
  retired (default OFF; env re-enables for unsupported mlx-lm builds).
* ``EngineCore._await_admission_wave`` — the first step after idle
  waits tick-wise while a burst of submissions is still landing, so the
  wave prefills as ONE aligned insert.

The engine object is built via ``__new__`` (the helper only touches
``scheduler.get_num_running`` and ``_admission_seq``), and
``asyncio.sleep`` inside the engine module is monkeypatched to count
ticks and drive submissions without real waiting.
"""

from __future__ import annotations

import asyncio
import contextlib

import pytest

import vllm_mlx.engine_core as engine_core
from vllm_mlx.engine_core import EngineCore, _resolve_hybrid_throttle

# ---------------------------------------------------------------- throttle


def test_hybrid_throttle_default_off(monkeypatch):
    monkeypatch.delenv("RAPID_HYBRID_THROTTLE", raising=False)
    assert _resolve_hybrid_throttle(True) is False
    assert _resolve_hybrid_throttle(False) is False


def test_hybrid_throttle_env_reenables_for_hybrid_only(monkeypatch):
    monkeypatch.setenv("RAPID_HYBRID_THROTTLE", "1")
    assert _resolve_hybrid_throttle(True) is True
    # Non-hybrid models never throttled, even with the env set.
    assert _resolve_hybrid_throttle(False) is False


def test_hybrid_throttle_env_zero_stays_off(monkeypatch):
    monkeypatch.setenv("RAPID_HYBRID_THROTTLE", "0")
    assert _resolve_hybrid_throttle(True) is False


# ------------------------------------------------------- admission window


class _StubScheduler:
    def __init__(self, num_running=0):
        self._num_running = num_running

    def get_num_running(self):
        return self._num_running


class _NoRunningApi:
    """Duck-typed scheduler stub without get_num_running."""


def _make_engine(num_running=0, seq=0, scheduler=None):
    eng = EngineCore.__new__(EngineCore)
    eng.scheduler = scheduler if scheduler is not None else _StubScheduler(num_running)
    eng._admission_seq = seq
    return eng


@pytest.fixture
def sleep_spy(monkeypatch):
    """Counts asyncio.sleep ticks inside engine_core and lets a test
    inject new submissions per tick (simulating add_request calls that
    land while the window is open)."""
    calls: list[float] = []
    per_tick: list[int] = []
    holder: dict = {"eng": None}

    async def fake_sleep(seconds):
        calls.append(seconds)
        if per_tick:
            holder["eng"]._admission_seq += per_tick.pop(0)

    monkeypatch.setattr(engine_core.asyncio, "sleep", fake_sleep)
    return calls, per_tick, holder


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def test_lone_request_pays_one_tick(sleep_spy, monkeypatch):
    calls, _, holder = sleep_spy
    monkeypatch.delenv("RAPID_ADMISSION_WINDOW_MS", raising=False)
    eng = _make_engine()
    holder["eng"] = eng
    _run(eng._await_admission_wave())
    assert calls == [pytest.approx(0.008)]


def test_burst_extends_while_submissions_land(sleep_spy, monkeypatch):
    calls, per_tick, holder = sleep_spy
    monkeypatch.delenv("RAPID_ADMISSION_WINDOW_MS", raising=False)
    eng = _make_engine()
    holder["eng"] = eng
    # Ticks 1-3 each land new submissions; tick 4 is quiet -> stop.
    per_tick.extend([3, 2, 2, 0])
    _run(eng._await_admission_wave())
    assert len(calls) == 4
    assert eng._admission_seq == 7


def test_no_wait_when_batch_active(sleep_spy):
    calls, _, holder = sleep_spy
    eng = _make_engine(num_running=5)
    holder["eng"] = eng
    _run(eng._await_admission_wave())
    # Mid-batch joins are ragged by definition — must not be delayed.
    assert calls == []


def test_no_wait_when_scheduler_lacks_api(sleep_spy):
    calls, _, holder = sleep_spy
    eng = _make_engine(scheduler=_NoRunningApi())
    holder["eng"] = eng
    _run(eng._await_admission_wave())
    assert calls == []


def test_disabled_by_env_zero(sleep_spy, monkeypatch):
    calls, _, holder = sleep_spy
    monkeypatch.setenv("RAPID_ADMISSION_WINDOW_MS", "0")
    eng = _make_engine()
    holder["eng"] = eng
    _run(eng._await_admission_wave())
    assert calls == []


def test_env_overrides_tick(sleep_spy, monkeypatch):
    calls, _, holder = sleep_spy
    monkeypatch.setenv("RAPID_ADMISSION_WINDOW_MS", "25")
    eng = _make_engine()
    holder["eng"] = eng
    _run(eng._await_admission_wave())
    assert calls == [pytest.approx(0.025)]


def test_garbage_env_falls_back_to_default(sleep_spy, monkeypatch):
    """A typo'd tuning var must degrade to the default, not raise inside
    the engine step loop (where ValueError would abort every in-flight
    request)."""
    calls, _, holder = sleep_spy
    monkeypatch.setenv("RAPID_ADMISSION_WINDOW_MS", "fast")
    eng = _make_engine()
    holder["eng"] = eng
    _run(eng._await_admission_wave())
    assert calls == [pytest.approx(0.008)]


@pytest.mark.parametrize("bad", ["inf", "-inf", "nan"])
def test_nonfinite_env_falls_back_to_default(sleep_spy, monkeypatch, bad):
    """inf/nan parse as floats but would make asyncio.sleep never return
    — every fresh-wave first step would hang. Must fall back."""
    calls, _, holder = sleep_spy
    monkeypatch.setenv("RAPID_ADMISSION_WINDOW_MS", bad)
    eng = _make_engine()
    holder["eng"] = eng
    _run(eng._await_admission_wave())
    assert calls == [pytest.approx(0.008)]


class _WarmupEngine:
    """Stub engine for the warmup hybrid-detection contract."""

    def __init__(self, hybrid_probe=None, is_mllm=False, model=None):
        if hybrid_probe is not None:
            self._is_hybrid_model = hybrid_probe
        self._is_mllm = is_mllm
        if model is not None:
            self._model = model


def test_warmup_detection_uses_profile_probe():
    """Round-1 MAJOR regression guard, behavioral: the warmup gating
    (``_detect_hybrid_for_warmup``) must follow the engine's fail-closed
    ``_is_hybrid_model()`` probe — NOT ``_hybrid_throttle``, which no
    longer implies is-hybrid now that the #115 throttle defaults OFF."""
    from vllm_mlx.server import _detect_hybrid_for_warmup

    assert _detect_hybrid_for_warmup(_WarmupEngine(hybrid_probe=lambda: True)) is True
    assert _detect_hybrid_for_warmup(_WarmupEngine(hybrid_probe=lambda: False)) is False
    # A throttle attribute must be IGNORED — it is not a hybrid signal.
    eng = _WarmupEngine(hybrid_probe=lambda: False)
    eng._hybrid_throttle = True
    assert _detect_hybrid_for_warmup(eng) is False


def test_warmup_detection_falls_back_to_arrays_cache():
    """Wrappers without the probe keep the pre-existing structural
    detection: a model whose make_cache() yields an ArraysCache layer is
    hybrid (pr_validate codex r2 BLOCKING claimed this fallback was
    lost — it is retained, and this pins it)."""
    from mlx_lm.models.cache import ArraysCache, KVCache

    from vllm_mlx.server import _detect_hybrid_for_warmup

    class _HybridModel:
        def make_cache(self):
            return [KVCache(), ArraysCache(size=2)]

    class _PlainModel:
        def make_cache(self):
            return [KVCache(), KVCache()]

    assert _detect_hybrid_for_warmup(_WarmupEngine(model=_HybridModel())) is True
    assert _detect_hybrid_for_warmup(_WarmupEngine(model=_PlainModel())) is False
    # No probe, no model → fail closed (not hybrid → bare warmup, the
    # pre-change behavior for opaque wrappers).
    assert _detect_hybrid_for_warmup(_WarmupEngine()) is False


def test_warmup_detection_mllm_excluded():
    from vllm_mlx.server import _detect_hybrid_for_warmup

    assert _detect_hybrid_for_warmup(_WarmupEngine(is_mllm=True)) is False


def test_warmup_detection_mllm_wins_over_hybrid_probe():
    """MLLM exclusion runs BEFORE the hybrid probe (pr_validate codex r3
    BLOCKING): an MLLM engine whose probe answers True must still take
    the bare-warmup path. Unreachable today (hybrid VLMs auto-downgrade
    to the text lane, #352) but must fail closed if that ever changes."""
    from vllm_mlx.server import _detect_hybrid_for_warmup

    eng = _WarmupEngine(hybrid_probe=lambda: True, is_mllm=True)
    assert _detect_hybrid_for_warmup(eng) is False


def test_add_request_bumps_admission_seq():
    """Wiring guard: ``add_request`` must move the counter the window
    watches — the increment sits at the very top of the method, before
    any validation, so a bare stub reaches it (later attribute errors
    are irrelevant to this contract and suppressed)."""
    eng = EngineCore.__new__(EngineCore)
    eng._admission_seq = 0
    with contextlib.suppress(Exception):
        _run(eng.add_request("hi"))
    assert eng._admission_seq == 1


def test_tick_larger_than_cap_cannot_overshoot(monkeypatch):
    """A tick bigger than the cap must be clamped to the remaining
    window — otherwise a huge finite RAPID_ADMISSION_WINDOW_MS holds
    the first step far past the advertised cap (pr_validate codex
    BLOCKING)."""
    calls: list[float] = []
    clock = {"t": 200.0}
    eng = _make_engine()

    async def fake_sleep(seconds):
        calls.append(seconds)
        clock["t"] += seconds
        eng._admission_seq += 1  # never quiet — cap must still cut it

    monkeypatch.setattr(engine_core.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(engine_core.time, "monotonic", lambda: clock["t"])
    monkeypatch.setenv("RAPID_ADMISSION_WINDOW_MS", "500")
    monkeypatch.setenv("RAPID_ADMISSION_WINDOW_CAP_MS", "45")
    _run(eng._await_admission_wave())
    # One sleep, clamped to the 45ms cap; then no time remains.
    assert calls == [pytest.approx(0.045)]


def test_hard_cap_bounds_never_quiet_stream(monkeypatch):
    """A submission stream that never goes quiet must still stop at the
    cap. Fake clock: each tick advances monotonic time by one tick."""
    calls: list[float] = []
    clock = {"t": 500.0}
    eng = _make_engine()

    async def fake_sleep(seconds):
        calls.append(seconds)
        clock["t"] += seconds
        eng._admission_seq += 1  # never quiet

    monkeypatch.setattr(engine_core.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(engine_core.time, "monotonic", lambda: clock["t"])
    monkeypatch.setenv("RAPID_ADMISSION_WINDOW_MS", "10")
    # Cap sits mid-tick (45ms / 10ms) so float accumulation on the fake
    # clock cannot straddle the deadline boundary.
    monkeypatch.setenv("RAPID_ADMISSION_WINDOW_CAP_MS", "45")
    _run(eng._await_admission_wave())
    # Ticks at t=0,10,20,30,40ms all pass the <45ms deadline check; the
    # sixth check (t=50ms) fails -> exactly 5 sleeps.
    assert len(calls) == 5
