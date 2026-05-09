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

    @pytest.mark.asyncio
    async def test_add_request_routes_to_worker(self, engine_core, monkeypatch):
        """add_request() MUST run scheduler.add_request on the mlx-step worker.

        scheduler.add_request walks the prefix cache (memory_aware_cache.fetch
        does copy.deepcopy of cached KV state, paged_cache.reconstruct_cache
        materializes block tensors). Those allocations get tagged with the
        calling thread's default stream. If add_request runs on the asyncio
        loop thread, the cached KV ends up on a stream the mlx-step worker
        cannot mx.eval against, and the next batch_generator.next() raises
        "There is no Stream(gpu, N) in current thread" inside
        `mx.eval([c.state for c in self.prompt_cache])`.

        This is the third leg of the #170 fix (after #173 warmup and #174
        model load). Without it, every text-only model with a populated
        prefix cache (e.g. --pin-system-prompt loaded entries from disk,
        or a prior request's system prompt) breaks on the next request.
        """
        from vllm_mlx import engine_core as ec

        monkeypatch.setattr(ec, "_init_mlx_step_thread", lambda: None)

        captured = {}

        def fake_add_request(request):
            captured["thread"] = threading.current_thread().name
            captured["request_id"] = request.request_id

        engine_core.scheduler.add_request.side_effect = fake_add_request

        await engine_core.start()
        try:
            request_id = await engine_core.add_request("hello")
            assert captured["request_id"] == request_id
            assert captured["thread"].startswith("mlx-step"), (
                f"add_request ran on {captured['thread']!r}, expected mlx-step worker. "
                "If add_request runs on the asyncio loop thread, prefix-cache "
                "KV deep-copies are tagged with the wrong stream and the next "
                "step() crashes with 'There is no Stream(gpu, N) in current thread'."
            )
        finally:
            await engine_core.stop()

    @pytest.mark.asyncio
    async def test_add_request_falls_back_when_executor_missing(self, engine_core):
        """Without start(), add_request runs scheduler.add_request inline.

        Sync test/CLI paths that build an EngineCore without calling start()
        must still work — the caller will see whatever stream error MLX would
        have raised, but no spurious AttributeError on `_mlx_executor`."""
        captured = {}

        def fake_add_request(request):
            captured["thread"] = threading.current_thread().name

        engine_core.scheduler.add_request.side_effect = fake_add_request

        # Engine never started → _mlx_executor is None.
        assert engine_core._mlx_executor is None
        await engine_core.add_request("hello")
        assert captured["thread"] == threading.current_thread().name


class TestBatchedEngineWarmup:
    """#170 regression: BatchedEngine.generate_warmup() must run on the
    mlx-step worker so cache arrays it touches carry the worker's stream.

    Without this, models with eager cache materialization (Gemma 4
    RotatingKVCache, sliding-window) fail every request with
    "There is no Stream(gpu, 1) in current thread" the moment
    BatchGenerator.prompt() evals the prompt cache state on the worker.
    """

    def test_warmup_runs_on_mlx_step_thread(self, monkeypatch):
        """generate_warmup() routes the model forward through _run_on_step_thread."""
        from vllm_mlx.engine.batched import BatchedEngine

        captured: dict = {}

        # Stub mx.array / mx.eval so we don't touch real Metal.
        import mlx.core as mx

        monkeypatch.setattr(mx, "array", lambda x: MagicMock())
        monkeypatch.setattr(mx, "eval", lambda *_a, **_k: None)

        engine = BatchedEngine.__new__(BatchedEngine)
        engine._loaded = True
        engine._is_mllm = False
        engine._tokenizer = MagicMock()
        engine._tokenizer.encode = MagicMock(return_value=[1, 2])

        def model_call(input_ids):
            captured["model_thread"] = threading.current_thread().name
            return MagicMock()

        engine._model = MagicMock(side_effect=model_call)

        # Build a tiny EngineCore-like wrapper exposing _run_on_step_thread
        # via a real single-thread executor named mlx-step-test.
        import concurrent.futures

        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="mlx-step-test"
        )
        try:
            inner = MagicMock()
            inner._mlx_executor = executor

            def _run_on_step_thread(func, *args, **kwargs):
                return executor.submit(func, *args, **kwargs).result()

            inner._run_on_step_thread = _run_on_step_thread

            wrapper = MagicMock()
            wrapper.engine = inner
            engine._engine = wrapper

            engine.generate_warmup()

            assert captured.get("model_thread", "").startswith("mlx-step-test"), (
                f"generate_warmup ran model on {captured.get('model_thread')!r}, "
                f"expected mlx-step-test thread"
            )
        finally:
            executor.shutdown(wait=True)

    def test_warmup_falls_back_when_executor_missing(self, monkeypatch):
        """Without a step-thread executor, warmup runs inline (legacy path)."""
        from vllm_mlx.engine.batched import BatchedEngine

        captured: dict = {}

        import mlx.core as mx

        monkeypatch.setattr(mx, "array", lambda x: MagicMock())
        monkeypatch.setattr(mx, "eval", lambda *_a, **_k: None)

        engine = BatchedEngine.__new__(BatchedEngine)
        engine._loaded = True
        engine._is_mllm = False
        engine._tokenizer = MagicMock()
        engine._tokenizer.encode = MagicMock(return_value=[1, 2])

        def model_call(input_ids):
            captured["model_thread"] = threading.current_thread().name
            return MagicMock()

        engine._model = MagicMock(side_effect=model_call)
        engine._engine = None  # no AsyncEngineCore yet

        engine.generate_warmup()

        # Should have run on caller thread.
        assert captured.get("model_thread") == threading.current_thread().name


class TestBatchedEngineGetStats:
    """get_stats() must promote MLLMScheduler keys (Metal memory + the
    batch_generator throughput dict) to the top level. /v1/status reads
    from the top level — without this forwarding, generation_tps stays
    invisible to monitoring even though the underlying counters tick.
    """

    def _make_engine(self, mllm_stats):
        from vllm_mlx.engine.batched import BatchedEngine

        engine = BatchedEngine.__new__(BatchedEngine)
        engine._model_name = "test-model"
        engine._is_mllm = False
        engine._loaded = True
        engine._stream_interval = 1
        engine._engine = None
        engine._mllm_scheduler = MagicMock()
        engine._mllm_scheduler.get_stats = MagicMock(return_value=mllm_stats)
        return engine

    def test_get_stats_forwards_batch_generator(self):
        bg = {"generation_tps": 42.0, "prompt_tps": 100.0}
        engine = self._make_engine(
            {
                "metal_active_memory_gb": 1.0,
                "batch_generator": bg,
                "other_key": "ignored",
            }
        )

        stats = engine.get_stats()

        assert stats["batch_generator"] == bg
        assert stats["metal_active_memory_gb"] == 1.0
        # Non-promoted keys must not appear at top level.
        assert "other_key" not in stats
        # Full mllm_stats remains nested for debugging.
        assert stats["mllm_scheduler"]["other_key"] == "ignored"

    def test_get_stats_omits_missing_batch_generator(self):
        engine = self._make_engine({"metal_active_memory_gb": 2.5})

        stats = engine.get_stats()

        assert "batch_generator" not in stats
        assert stats["metal_active_memory_gb"] == 2.5


class TestGuidedGenerationStepThread:
    """#170 regression: BatchedEngine.generate_with_schema must run
    _run_guided_generation on the mlx-step worker (the same thread that
    loaded the model), not asyncio's default executor.

    outlines materializes mx.array against the model weights. mlx-lm 0.31.3+
    tags every array with the calling thread's default stream. If guided
    generation runs on a different thread than the model load thread, the
    first eval crashes with "There is no Stream(gpu, N) in current thread".

    The bug was silent in production because _run_guided_generation catches
    the exception and falls back to non-guided generation — guided decoding
    has been quietly broken since #174 swapped model loading onto
    _model_load_executor.
    """

    @pytest.mark.asyncio
    async def test_guided_generation_routes_to_step_thread(self, monkeypatch):
        """generate_with_schema must dispatch via _model_load_executor."""
        import concurrent.futures

        from vllm_mlx.engine.batched import BatchedEngine

        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="mlx-step-test"
        )
        try:
            engine = BatchedEngine.__new__(BatchedEngine)
            engine._loaded = True
            engine._is_mllm = False
            engine._model = MagicMock()
            engine._tokenizer = MagicMock()
            engine._tokenizer.apply_chat_template = MagicMock(return_value="prompt")
            engine._tokenizer.encode = MagicMock(return_value=[1, 2])
            engine._model_load_executor = executor
            engine._engine = None

            # Force HAS_GUIDED True so supports_guided_generation passes.
            from vllm_mlx.engine import batched as batched_mod

            monkeypatch.setattr(batched_mod, "HAS_GUIDED", True)

            captured: dict = {}

            def fake_run_guided(prompt, json_schema, max_tokens, temperature):
                captured["thread"] = threading.current_thread().name
                return '{"ok": true}'

            engine._run_guided_generation = fake_run_guided

            result = await engine.generate_with_schema(
                messages=[{"role": "user", "content": "hi"}],
                json_schema={"type": "object"},
                max_tokens=8,
                temperature=0.0,
            )

            assert result.text == '{"ok": true}'
            assert captured["thread"].startswith("mlx-step-test"), (
                f"_run_guided_generation ran on {captured['thread']!r}, "
                "expected mlx-step worker. asyncio.to_thread() would dispatch "
                "to the default executor and crash with 'There is no Stream(gpu, N)' "
                "the first time outlines materializes against the model."
            )
        finally:
            executor.shutdown(wait=True)

    @pytest.mark.asyncio
    async def test_guided_generation_falls_back_without_executor(self, monkeypatch):
        """No executor available → fall back to asyncio.to_thread (best-effort)."""
        from vllm_mlx.engine.batched import BatchedEngine

        engine = BatchedEngine.__new__(BatchedEngine)
        engine._loaded = True
        engine._is_mllm = False
        engine._model = MagicMock()
        engine._tokenizer = MagicMock()
        engine._tokenizer.apply_chat_template = MagicMock(return_value="prompt")
        engine._tokenizer.encode = MagicMock(return_value=[1])
        engine._model_load_executor = None
        engine._engine = None

        from vllm_mlx.engine import batched as batched_mod

        monkeypatch.setattr(batched_mod, "HAS_GUIDED", True)

        called = {"n": 0}

        def fake_run_guided(prompt, json_schema, max_tokens, temperature):
            called["n"] += 1
            return '{"ok": true}'

        engine._run_guided_generation = fake_run_guided

        result = await engine.generate_with_schema(
            messages=[{"role": "user", "content": "hi"}],
            json_schema={"type": "object"},
        )

        # Should still execute (via asyncio default executor) even without
        # a mlx-step worker. Caller may then hit Stream(gpu, N), but the
        # dispatch itself must not crash.
        assert called["n"] == 1
        assert result.text == '{"ok": true}'


class TestMLLMSchedulerStepThread:
    """#170 regression: MLLMScheduler must run every step on the mllm-step
    worker, not split between worker (when waiting) and loop thread (when
    only generating).

    BatchGenerator keeps KV state across calls. Splitting prefill onto
    mllm-step and decode onto the loop thread tags freshly-allocated
    arrays with mismatched streams, so the next decode step crashes with
    "There is no Stream(gpu, N) in current thread" inside
    `mx.eval([c.state for c in self.prompt_cache])`.
    """

    @pytest.mark.asyncio
    async def test_step_runs_on_mllm_step_thread_with_and_without_waiting(self):
        """_step_no_queue must execute on mllm-step in BOTH branches.

        Drives _process_loop for two iterations and toggles ``waiting``
        between non-empty (prefill) and empty (decode-only). Pre-fix the
        decode-only iteration ran inline on the loop thread; post-fix
        every iteration must land on mllm-step.
        """
        from vllm_mlx.mllm_scheduler import MLLMScheduler

        scheduler = MLLMScheduler.__new__(MLLMScheduler)
        scheduler._running = True
        scheduler._step_executor = None
        scheduler._injected_step_executor = None  # _process_loop creates its own

        threads: list[str] = []
        waiting_seen: list[bool] = []
        call_count = {"n": 0}

        def fake_step():
            threads.append(threading.current_thread().name)
            waiting_seen.append(bool(scheduler.waiting))
            call_count["n"] += 1
            if call_count["n"] == 1:
                # Prepare iter 2 to be decode-only (empty waiting).
                scheduler.waiting = []
            elif call_count["n"] >= 2:
                scheduler._running = False
            return None  # nothing to distribute

        scheduler._step_no_queue = fake_step
        scheduler.has_requests = lambda: True
        scheduler.waiting = [object()]  # iter 1: prefill
        scheduler._distribute_outputs = lambda _o: None

        await scheduler._process_loop()

        assert len(threads) == 2, f"Expected 2 step calls, got {len(threads)}"
        assert waiting_seen == [True, False], (
            f"Expected iter 1 with waiting + iter 2 without, got {waiting_seen}"
        )
        for i, name in enumerate(threads):
            assert name.startswith("mllm-step"), (
                f"step #{i + 1} (waiting={waiting_seen[i]}) ran on {name!r}, "
                "expected mllm-step worker. Splitting steps between mllm-step "
                "and loop thread tags BatchGenerator KV arrays with mismatched "
                "streams and the next batch_generator.next() crashes with "
                "'There is no Stream(gpu, N) in current thread'."
            )

    @pytest.mark.asyncio
    async def test_step_uses_injected_executor_not_a_fresh_one(self):
        """When BatchedEngine hands in the model-load executor, MLLMScheduler
        MUST step on that same thread — the model arrays are tagged with its
        stream. Creating a fresh mllm-step worker would be a new stream and
        every batch_generator.next() would crash with Stream(gpu, N).
        """
        import concurrent.futures

        from vllm_mlx.mllm_scheduler import MLLMScheduler

        injected = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="mllm-step-injected"
        )
        try:
            scheduler = MLLMScheduler.__new__(MLLMScheduler)
            scheduler._running = True
            scheduler._step_executor = None
            scheduler._injected_step_executor = injected
            scheduler._owns_step_executor = (
                False  # set by _process_loop, but be explicit
            )

            captured: dict = {}
            call_count = {"n": 0}

            def fake_step():
                captured["thread"] = threading.current_thread().name
                call_count["n"] += 1
                if call_count["n"] >= 1:
                    scheduler._running = False
                return None

            scheduler._step_no_queue = fake_step
            scheduler.has_requests = lambda: True
            scheduler.waiting = [object()]
            scheduler._distribute_outputs = lambda _o: None

            await scheduler._process_loop()

            assert captured["thread"].startswith("mllm-step-injected"), (
                f"step ran on {captured['thread']!r}; expected the injected "
                "executor's thread. A fresh executor would crash with "
                "Stream(gpu, N) because the model arrays are tagged with the "
                "injected executor's stream."
            )
        finally:
            injected.shutdown(wait=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
