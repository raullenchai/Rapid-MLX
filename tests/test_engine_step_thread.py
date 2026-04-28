# SPDX-License-Identifier: Apache-2.0
"""
Tests for EngineCore._run_on_step_thread / _mlx_executor.

Background: mlx-lm 0.31+ binds `mlx_lm.generate.generation_stream` to the
thread that creates it. Any MLX op that touches arrays tagged with that
stream from a different thread raises
``RuntimeError: There is no Stream(gpu, N) in current thread.``

The engine creates a single dedicated mlx-step worker thread so the
generation hot path stays on it. Anything else that touches KV-cache
arrays — most importantly the shutdown call to save the prefix cache —
must be routed through the same worker via _run_on_step_thread().

These tests exercise that machinery without spinning up a real model.
"""

import threading
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def engine_core(monkeypatch):
    """Build an EngineCore with mocked model/registry/scheduler."""
    from vllm_mlx import engine_core as ec

    # Avoid the real model registry (which expects a real MLX model).
    fake_registry = MagicMock()
    monkeypatch.setattr(ec, "get_registry", lambda: fake_registry)

    # Don't spin up a real Scheduler — patch it to a MagicMock so we can
    # assert how save/load_cache_to_disk reaches it.
    with patch.object(ec, "Scheduler") as scheduler_cls:
        scheduler_instance = MagicMock()
        scheduler_cls.return_value = scheduler_instance
        engine = ec.EngineCore(model=MagicMock(), tokenizer=MagicMock())
        engine.scheduler = scheduler_instance
        yield engine
        engine.close()


class TestStepThread:
    def test_executor_is_lazy(self, engine_core):
        """Executor is None until start() runs."""
        assert engine_core._mlx_executor is None

    @pytest.mark.asyncio
    async def test_start_creates_named_executor(self, engine_core, monkeypatch):
        """start() creates a single-thread executor with the mlx-step name."""
        # Stub out _init_mlx_step_thread so the test doesn't try to talk to
        # Metal — we only care about thread-naming + executor lifecycle here.
        from vllm_mlx import engine_core as ec

        monkeypatch.setattr(ec, "_init_mlx_step_thread", lambda: None)

        await engine_core.start()
        try:
            assert engine_core._mlx_executor is not None

            captured = {}

            def capture():
                captured["thread"] = threading.current_thread().name
                return "ok"

            result = engine_core._run_on_step_thread(capture)
            assert result == "ok"
            assert captured["thread"].startswith("mlx-step")
        finally:
            await engine_core.stop()
            assert engine_core._mlx_executor is None

    def test_run_on_step_thread_falls_back_when_no_executor(self, engine_core):
        """Without start(), _run_on_step_thread runs inline (and the caller
        will see whatever stream error MLX would have raised)."""
        captured = {}

        def capture():
            captured["thread"] = threading.current_thread().name

        engine_core._run_on_step_thread(capture)
        # Should have run on the *current* thread (no executor available).
        assert captured["thread"] == threading.current_thread().name

    @pytest.mark.asyncio
    async def test_save_cache_to_disk_routes_to_worker(self, engine_core, monkeypatch):
        """The shutdown save MUST execute on the mlx-step worker thread."""
        from vllm_mlx import engine_core as ec

        monkeypatch.setattr(ec, "_init_mlx_step_thread", lambda: None)

        captured = {}

        def fake_save(cache_dir):
            captured["thread"] = threading.current_thread().name
            captured["cache_dir"] = cache_dir
            return True

        engine_core.scheduler.save_cache_to_disk.side_effect = fake_save

        await engine_core.start()
        try:
            assert engine_core.save_cache_to_disk("/tmp/whatever") is True
            assert captured["cache_dir"] == "/tmp/whatever"
            assert captured["thread"].startswith("mlx-step")
        finally:
            await engine_core.stop()

    @pytest.mark.asyncio
    async def test_load_cache_from_disk_routes_to_worker(
        self, engine_core, monkeypatch
    ):
        """Loading also runs on the worker so loaded arrays are tagged with
        the stream that subsequent fetches will run on."""
        from vllm_mlx import engine_core as ec

        monkeypatch.setattr(ec, "_init_mlx_step_thread", lambda: None)

        captured = {}

        def fake_load(cache_dir):
            captured["thread"] = threading.current_thread().name
            return 17

        engine_core.scheduler.load_cache_from_disk.side_effect = fake_load

        await engine_core.start()
        try:
            assert engine_core.load_cache_from_disk("/tmp/whatever") == 17
            assert captured["thread"].startswith("mlx-step")
        finally:
            await engine_core.stop()

    @pytest.mark.asyncio
    async def test_run_on_step_thread_propagates_exceptions(
        self, engine_core, monkeypatch
    ):
        """Worker-thread exceptions must propagate to the caller — silent
        failure here would mean we save half the cache and never log why."""
        from vllm_mlx import engine_core as ec

        monkeypatch.setattr(ec, "_init_mlx_step_thread", lambda: None)

        await engine_core.start()
        try:

            def boom():
                raise RuntimeError("There is no Stream(gpu, 2) in current thread.")

            with pytest.raises(RuntimeError, match="Stream"):
                engine_core._run_on_step_thread(boom)
        finally:
            await engine_core.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
