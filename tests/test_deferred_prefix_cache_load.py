"""Regression tests for the deferred (background) prefix-cache load (#1350).

Issue #1350: on startup the persisted prefix cache was loaded *synchronously*
inside the lifespan handler, before ``_cfg.ready = True``. A large cache
therefore held ``/health/ready`` and ``/v1/models`` at 503 for the whole
multi-second disk read — unlike the shutdown *save*, which was already
offloaded via ``asyncio.to_thread``. The fix flips readiness first, then runs
``server._deferred_load_prefix_cache`` as a background task.

These tests pin the production helper at its callsite (mirroring
``test_shutdown_save_prefix_cache_runs_off_event_loop``): if anyone replaces
the ``await asyncio.to_thread(...)`` wrap with a direct call, the loop-starves
regression fires.
"""

from __future__ import annotations

import asyncio
import logging
import time as _time


def test_deferred_load_prefix_cache_runs_off_event_loop(monkeypatch):
    """The background load must run on a worker thread, not the loop.

    Pretend the disk load takes ~600ms. Drive the production helper AND a
    50ms ticker concurrently; assert the ticker advanced multiple times.
    If ``_deferred_load_prefix_cache`` loses its ``asyncio.to_thread`` wrap,
    the sleep blocks the loop and the ticker count collapses to 1.
    """
    from vllm_mlx import server as _server_mod

    def _slow_load():
        # Block ~600ms on the worker thread — production wrap is
        # asyncio.to_thread so the loop stays responsive.
        _time.sleep(0.6)

    class _StubEngine:
        # Only the attribute's existence matters — it satisfies the
        # ``hasattr(_engine, "load_cache_from_disk")`` guard.
        def load_cache_from_disk(self, *a, **k):
            return None

    monkeypatch.setattr(_server_mod, "_engine", _StubEngine())
    monkeypatch.setattr(_server_mod, "_load_prefix_cache_from_disk", _slow_load)

    async def _drive():
        ticks: list[float] = []
        t0 = _time.monotonic()

        async def _ticker():
            while _time.monotonic() - t0 < 0.6:
                ticks.append(_time.monotonic() - t0)
                await asyncio.sleep(0.05)

        # Drive the production lifespan helper as-is. Don't wrap it in
        # asyncio.to_thread out here — that would re-introduce the exact bug
        # this test exists to catch.
        await asyncio.gather(
            _server_mod._deferred_load_prefix_cache(),
            _ticker(),
        )
        return ticks

    ticks = asyncio.run(_drive())
    assert len(ticks) >= 5, (
        f"event loop was blocked during prefix-cache load — only saw "
        f"{len(ticks)} ticks in 600ms (expected ≥5). Did "
        f"_deferred_load_prefix_cache lose its asyncio.to_thread wrap?"
    )


def test_deferred_load_prefix_cache_no_op_when_engine_missing(monkeypatch):
    """No model loaded (``_engine is None``) → helper returns silently."""
    from vllm_mlx import server as _server_mod

    monkeypatch.setattr(_server_mod, "_engine", None)
    asyncio.run(_server_mod._deferred_load_prefix_cache())  # must not raise


def test_deferred_load_prefix_cache_no_op_when_engine_lacks_loader(monkeypatch):
    """Engine without ``load_cache_from_disk`` (e.g. embedding-only) → no-op;
    the loader function must never be called."""
    from vllm_mlx import server as _server_mod

    called = {"n": 0}

    def _should_not_run():
        called["n"] += 1

    class _NoLoaderEngine:
        pass

    monkeypatch.setattr(_server_mod, "_engine", _NoLoaderEngine())
    monkeypatch.setattr(_server_mod, "_load_prefix_cache_from_disk", _should_not_run)
    asyncio.run(_server_mod._deferred_load_prefix_cache())
    assert called["n"] == 0


def test_deferred_load_prefix_cache_swallows_errors(monkeypatch, caplog):
    """A failing disk load must NOT crash the lifespan — a cold cache only
    costs a few early prefix recomputes. The helper logs a warning and
    returns instead of propagating."""
    from vllm_mlx import server as _server_mod

    def _boom():
        raise RuntimeError("corrupt cache index")

    class _StubEngine:
        def load_cache_from_disk(self, *a, **k):
            return None

    monkeypatch.setattr(_server_mod, "_engine", _StubEngine())
    monkeypatch.setattr(_server_mod, "_load_prefix_cache_from_disk", _boom)

    with caplog.at_level(logging.WARNING):
        asyncio.run(_server_mod._deferred_load_prefix_cache())  # must not raise

    assert "deferred prefix-cache load failed" in caplog.text


def test_deferred_load_prefix_cache_honors_autoload_opt_out(monkeypatch, caplog):
    """Desktop can prefer a truthful cold first turn over disk-cache restore.

    The restore runs on the engine's sole MLX step thread.  If it is slow, a
    server can report Ready while every generation request queues behind it.
    The opt-out must return before touching either the cache directory or the
    engine loader; persisted files remain available for explicit import.
    """
    from vllm_mlx import server as _server_mod

    class _ExplodingEngine:
        def load_cache_from_disk(self, *args, **kwargs):
            raise AssertionError("autoload opt-out reached the engine")

    monkeypatch.setenv("RAPID_MLX_PREFIX_CACHE_AUTOLOAD", "0")
    monkeypatch.setattr(_server_mod, "_engine", _ExplodingEngine())
    monkeypatch.setattr(
        _server_mod,
        "_load_prefix_cache_from_disk",
        lambda: (_ for _ in ()).throw(AssertionError("cache loader was called")),
    )

    with caplog.at_level(logging.INFO):
        asyncio.run(_server_mod._deferred_load_prefix_cache())

    assert "Prefix-cache auto-load disabled" in caplog.text


def test_autoload_opt_out_does_not_disable_explicit_loader(monkeypatch):
    """The Desktop startup policy must not change the general load operation."""
    from vllm_mlx.runtime import cache as _cache_mod

    calls = []

    class _Engine:
        def load_cache_from_disk(self, path, *, protected_import):
            calls.append((path, protected_import))
            return 0

    class _Config:
        engine = _Engine()

    monkeypatch.setenv("RAPID_MLX_PREFIX_CACHE_AUTOLOAD", "0")
    monkeypatch.setattr(_cache_mod, "get_config", lambda: _Config())
    monkeypatch.setattr(_cache_mod, "get_cache_dir", lambda: "/tmp/explicit-cache")

    _cache_mod.load_prefix_cache_from_disk()

    assert calls == [("/tmp/explicit-cache", False)]


def test_drain_awaits_load_to_completion(monkeypatch):
    """Shutdown drain must AWAIT the load to completion, not cancel it.

    ``Task.cancel()`` around ``asyncio.to_thread`` only stops us awaiting the
    wrapper — the worker thread keeps running ``_load_prefix_cache_from_disk``
    and would race the shutdown save / engine teardown. This test proves the
    load has fully finished by the time the drain returns: if anyone swaps the
    ``await`` for a ``cancel()``, the flag is still unset when the drain
    returns and the assert fires.
    """
    from vllm_mlx import server as _server_mod

    finished = {"done": False}

    def _slow_load():
        _time.sleep(0.3)
        finished["done"] = True

    class _StubEngine:
        def load_cache_from_disk(self, *a, **k):
            return None

    monkeypatch.setattr(_server_mod, "_engine", _StubEngine())
    monkeypatch.setattr(_server_mod, "_load_prefix_cache_from_disk", _slow_load)

    async def _scenario():
        task = asyncio.create_task(_server_mod._deferred_load_prefix_cache())
        monkeypatch.setattr(_server_mod, "_prefix_cache_load_task", task)
        # Let the load actually start on its worker thread.
        await asyncio.sleep(0.05)
        assert not finished["done"], "load finished too fast to test the await"
        await _server_mod._drain_deferred_prefix_cache_load()
        assert finished["done"], (
            "drain returned before the load finished — did it cancel instead "
            "of await? A live loader would race the shutdown save / teardown."
        )

    asyncio.run(_scenario())


def test_drain_no_op_when_no_load_task(monkeypatch):
    """No deferred load was scheduled (embedding-only server, or engine without
    a loader) → the shutdown drain is a silent no-op."""
    from vllm_mlx import server as _server_mod

    monkeypatch.setattr(_server_mod, "_prefix_cache_load_task", None)
    asyncio.run(_server_mod._drain_deferred_prefix_cache_load())  # must not raise
