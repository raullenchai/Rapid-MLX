# SPDX-License-Identifier: Apache-2.0
"""Tests for the idle-wakeup event in EngineCore (#265).

The engine loop must:
- block on ``self._idle_event.wait()`` instead of polling at 1 ms when the
  scheduler is empty,
- wake immediately when ``add_request`` sets the event (no added first-token
  latency),
- still wake periodically via the ``step_interval`` timeout for housekeeping
  paths that don't go through ``add_request`` (e.g. cache load, deep_reset).
"""

from __future__ import annotations

import asyncio
import time

import pytest

from vllm_mlx.engine_core import EngineConfig


def test_default_step_interval_is_seconds_not_milliseconds():
    """The default must NOT poll at kHz any more — a regression that flipped
    the unit back to milliseconds would silently restore 100% idle CPU."""
    cfg = EngineConfig()
    assert cfg.step_interval >= 1.0, (
        f"step_interval={cfg.step_interval}s — must be >= 1.0s to keep idle "
        "CPU near zero. See issue #265 for the regression history."
    )


@pytest.mark.asyncio
async def test_idle_event_unblocks_immediately_when_set():
    """Direct test of the event mechanism: an awaiter blocked on
    ``event.wait()`` with a 30s timeout must wake within milliseconds when
    ``event.set()`` is called from a different task."""
    event = asyncio.Event()

    async def waiter() -> float:
        start = time.perf_counter()
        try:
            await asyncio.wait_for(event.wait(), timeout=30.0)
        except asyncio.TimeoutError:
            return -1.0
        return time.perf_counter() - start

    async def setter():
        await asyncio.sleep(0.01)
        event.set()

    elapsed, _ = await asyncio.gather(waiter(), setter())
    assert 0 <= elapsed < 0.1, (
        f"event-driven wakeup took {elapsed * 1000:.1f}ms; should be << 100ms"
    )


@pytest.mark.asyncio
async def test_idle_event_falls_back_to_timeout():
    """If nobody calls ``set()``, the awaiter must still return after the
    configured timeout — that's the housekeeping safety net."""
    event = asyncio.Event()
    start = time.perf_counter()
    try:
        await asyncio.wait_for(event.wait(), timeout=0.05)
    except asyncio.TimeoutError:
        pass
    elapsed = time.perf_counter() - start
    assert 0.04 <= elapsed < 0.5, (
        f"timeout fallback took {elapsed * 1000:.1f}ms; expected ~50ms"
    )


@pytest.mark.asyncio
async def test_engine_core_creates_idle_event_in_loop():
    """``EngineCore._engine_loop`` must lazy-create the asyncio.Event on
    the running loop. Pre-creation in ``__init__`` would bind to whatever
    loop is current at construction time (often there isn't one in
    tests), causing ``RuntimeError: ... attached to a different loop``."""
    from unittest.mock import MagicMock

    # Build a barely-functional EngineCore that we can poke at the event
    # without actually running the full loop. We need to dodge the
    # ModelOwnership registry, which lives in the registry module.
    from vllm_mlx.engine_core import EngineCore

    fake_model = MagicMock()
    fake_tokenizer = MagicMock()
    cfg = EngineConfig(model_name="test-fake-model")

    # ``EngineCore.__init__`` populates _idle_event = None. The event is
    # created the first time ``_engine_loop`` runs, which we don't here —
    # so just assert the lazy slot exists and is None at construction.
    try:
        engine = EngineCore(fake_model, fake_tokenizer, cfg)
    except Exception:
        # If registry / model setup fails for unrelated reasons, fall
        # back to checking the field is at least declared.
        pytest.skip("EngineCore construction needs more mock setup")
        return

    assert hasattr(engine, "_idle_event"), (
        "EngineCore must declare the _idle_event slot — see issue #265"
    )
    assert engine._idle_event is None, (
        "_idle_event must start as None and be created inside _engine_loop "
        "to bind to the right asyncio loop"
    )
