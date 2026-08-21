# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the audio-lane MLX worker routing.

``text + audio in one process`` requires the STT/TTS lanes to run their MLX
inference on the SAME worker thread the primary engine owns (the module-global
``mlx_lm.generate.generation_stream`` is re-bound to that worker's stream at
startup; running audio inline on the asyncio loop thread of a TEXT server hits
the #170 / #452 ``Stream(gpu, N)`` crash).

These tests pin the resolution walker (which must descend through the nested
``BatchedEngine._engine`` -> ``AsyncEngineCore.engine`` -> ``engine_core``
topology to find ``_mlx_executor``) and the fallback-to-inline contract, using
pure stub objects — no real GPU, no real engine.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
import unittest

from vllm_mlx import routes, server


def _make_executor():
    """A real single-thread executor, faithful to the engine's MLX worker."""
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    return executor


class _WorkerCore:
    """Plain object shaped like the engine_core. Attribute presence is real.

    ``_mlx_executor`` is set in ``__init__``; ``_run_on_step_thread`` only if
    ``run_on_step`` is given, so `hasattr`/`getattr(..., None)` distinguish
    "present" from "absent" instead of MagicMock auto-creating them.
    """

    def __init__(self, run_on_step=None):
        self._mlx_executor = _make_executor()
        if run_on_step is not None:
            self._run_on_step_thread = run_on_step


def _make_worker_core(run_on_step=None):
    return _WorkerCore(run_on_step=run_on_step)


def _nested_topology(worker_core):
    """BatchedEngine(top)._engine -> AsyncEngineCore.engine -> worker_core."""
    batched = type("BatchedEngine", (), {"_engine": None})()
    async_core = type("AsyncEngineCore", (), {"engine": None})()
    async_core.engine = worker_core
    batched._engine = async_core
    return batched


def _install_engine(engine):
    server._engine = engine


class AudioMlxWorkerTests(unittest.TestCase):
    def setUp(self):
        self._old_engine = getattr(server, "_engine", None)

    def tearDown(self):
        server._engine = self._old_engine

    def test_resolve_finds_inner_executor_through_nested_topology(self):
        worker_core = _make_worker_core()
        _install_engine(_nested_topology(worker_core))
        self.assertIs(routes.audio._resolve_mlx_worker_core(), worker_core)

    def test_resolve_returns_none_without_engine(self):
        _install_engine(None)
        self.assertIsNone(routes.audio._resolve_mlx_worker_core())

    def test_resolve_uses_direct_owner(self):
        worker_core = _make_worker_core()
        _install_engine(worker_core)
        self.assertIs(routes.audio._resolve_mlx_worker_core(), worker_core)

    def test_run_audio_mlx_submits_to_worker_executor(self):
        worker_core = _make_worker_core()
        _install_engine(_nested_topology(worker_core))
        ran_on = {}

        def sample():
            import threading

            ran_on["name"] = threading.current_thread().name
            return "worked"

        async def scenario():
            return await routes.audio._run_audio_mlx(sample)

        self.assertEqual(asyncio.run(scenario()), "worked")
        # The callable must run on the worker thread, not the caller thread.
        self.assertNotEqual(ran_on["name"], threading.main_thread().name)

    def test_run_audio_mlx_falls_back_inline_without_worker(self):
        _install_engine(None)

        async def scenario():
            return await routes.audio._run_audio_mlx(lambda: "inline")

        self.assertEqual(asyncio.run(scenario()), "inline")

    def test_run_audio_mlx_sync_calls_through_worker_runner_when_present(self):
        worker_core = _make_worker_core()
        worker_core._run_on_step_thread = lambda fn, *a, **k: fn(*a, **k)
        _install_engine(_nested_topology(worker_core))
        self.assertEqual(
            routes.audio._run_audio_mlx_sync(lambda: "onworker"), "onworker"
        )

    def test_run_audio_mlx_sync_falls_back_inline_without_runner(self):
        worker_core = _make_worker_core()  # no _run_on_step_thread attached
        _install_engine(_nested_topology(worker_core))
        self.assertEqual(routes.audio._run_audio_mlx_sync(lambda: "inline"), "inline")


if __name__ == "__main__":
    unittest.main()
