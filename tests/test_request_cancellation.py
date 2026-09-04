# SPDX-License-Identifier: Apache-2.0
"""Tests for /v1/requests/{id}/cancel and BatchedEngine.abort_request routing.

Coverage adapted from https://github.com/waybarrios/vllm-mlx/issues/426 for our
routes/ split (their endpoint lives on the FastAPI app; ours lives on the
health router).
"""

from unittest.mock import AsyncMock, MagicMock

import pytest


def _make_guided_engine(monkeypatch, run_guided, *, executor=None):
    """Build the no-weights engine shape used by cancellation lifecycle tests."""
    import threading

    from vllm_mlx.engine import batched as batched_mod
    from vllm_mlx.engine.batched import BatchedEngine

    monkeypatch.setattr(batched_mod, "HAS_GUIDED", True)
    monkeypatch.setattr(
        batched_mod, "shared_apply_chat_template", lambda *_a, **_k: "prompt"
    )
    engine = BatchedEngine.__new__(BatchedEngine)
    engine._loaded = True
    engine._is_mllm = False
    engine._model_name = "test-model"
    engine._model = MagicMock()
    engine._tokenizer = MagicMock()
    engine._tokenizer.encode = MagicMock(return_value=[1])
    engine._model_load_executor = executor
    engine._engine = None
    engine._mllm_scheduler = None
    engine._guided_requests_lock = threading.Lock()
    engine._guided_abort_events = {}
    engine._guided_owner_tasks = {}
    engine._guided_stopping = False
    engine._run_guided_generation = run_guided
    return engine


class _StubAsyncEngine:
    """Minimal async engine stub exposing ``abort_request`` as a coroutine."""

    def __init__(self, returns: bool):
        self._returns = returns
        self.calls: list[str] = []

    async def abort_request(self, request_id: str) -> bool:
        self.calls.append(request_id)
        return self._returns


class _StubSyncMllmScheduler:
    """Minimal sync MLLM scheduler stub."""

    def __init__(self, returns: bool):
        self._returns = returns
        self.calls: list[str] = []

    def abort_request(self, request_id: str) -> bool:
        self.calls.append(request_id)
        return self._returns


class TestBatchedEngineAbortRouting:
    def test_constructor_initializes_guided_lifecycle_state(self, monkeypatch):
        from vllm_mlx.engine import batched as batched_mod

        monkeypatch.setattr(batched_mod, "is_mllm_model", lambda _name: False)
        engine = batched_mod.BatchedEngine("test-model", force_text=True)

        assert engine._guided_abort_events == {}
        assert engine._guided_owner_tasks == {}
        assert engine._guided_stopping is False

    def test_guided_registry_legacy_shapes_and_direct_abort(self):
        """Lightweight embedders get lazy state without weakening cancellation."""
        import asyncio
        import threading

        from vllm_mlx.engine.batched import BatchedEngine

        engine = BatchedEngine.__new__(BatchedEngine)
        engine._guided_requests_lock = threading.Lock()
        engine._guided_abort_events = {}
        event = threading.Event()
        engine._register_guided_request("req", event)

        assert engine._guided_owner_tasks == {}
        assert engine.abort_guided_request("missing") is False
        assert engine.abort_guided_request("req") is True
        assert event.is_set()

        owner = asyncio.new_event_loop().create_task(asyncio.sleep(0))
        try:
            engine._mark_lifecycle_aborted_tasks((owner,))
            assert not hasattr(engine, "_lifecycle_aborted_tasks")

            engine._admission_lock = threading.Lock()
            engine._mark_lifecycle_aborted_tasks((owner,))
            assert owner in engine._lifecycle_aborted_tasks
        finally:
            owner.cancel()
            owner.get_loop().run_until_complete(
                asyncio.gather(owner, return_exceptions=True)
            )
            owner.get_loop().close()

    def test_guided_helpers_are_safe_before_runtime_initialization(self):
        """Abort and handoff are harmless on legacy ``__new__`` engines."""
        import threading

        from vllm_mlx.engine.batched import BatchedEngine

        engine = BatchedEngine.__new__(BatchedEngine)

        event = threading.Event()
        engine._register_guided_request("legacy", event)
        assert engine._guided_abort_events == {"legacy": event}
        assert engine._guided_owner_tasks == {}
        assert engine._guided_stopping is False
        assert engine._finish_guided_request("legacy", event) is False

        assert engine.abort_guided_request("missing") is False
        uninitialized = BatchedEngine.__new__(BatchedEngine)
        uninitialized._abort_all_guided_requests()
        outcome = uninitialized.finish_guided_handoff("missing")
        assert outcome.cancelled is False
        assert outcome.lifecycle_task is None

    def test_guided_handoff_returns_owner_and_cancel_state(self):
        import asyncio
        import threading

        from vllm_mlx.engine.batched import BatchedEngine

        loop = asyncio.new_event_loop()
        owner = loop.create_task(asyncio.sleep(0))
        try:
            engine = BatchedEngine.__new__(BatchedEngine)
            engine._guided_requests_lock = threading.Lock()
            event = threading.Event()
            event.set()
            engine._guided_abort_events = {"req": event}
            engine._guided_owner_tasks = {"req": owner}

            outcome = engine.finish_guided_handoff("req")

            assert outcome.cancelled is True
            assert outcome.lifecycle_task is owner
            assert engine._guided_abort_events == {}
            assert engine._guided_owner_tasks == {}
        finally:
            owner.cancel()
            loop.run_until_complete(asyncio.gather(owner, return_exceptions=True))
            loop.close()

    @pytest.mark.asyncio
    async def test_guided_abort_precedes_scheduler_routing(self):
        import threading

        from vllm_mlx.engine.batched import BatchedEngine

        engine = BatchedEngine.__new__(BatchedEngine)
        engine._guided_requests_lock = threading.Lock()
        event = threading.Event()
        engine._guided_abort_events = {"req-guided": event}
        engine._guided_owner_tasks = {}
        engine._mllm_scheduler = _StubSyncMllmScheduler(returns=False)
        engine._engine = _StubAsyncEngine(returns=False)

        assert await engine.abort_request("req-guided") is True
        assert event.is_set()
        assert engine._mllm_scheduler.calls == []
        assert engine._engine.calls == []

    @pytest.mark.asyncio
    async def test_start_reopens_guided_admission_and_stop_closes_it(self):
        """Restart and shutdown own guided admission with the engine lifecycle."""
        import threading

        from vllm_mlx.engine.batched import BatchedEngine

        engine = BatchedEngine.__new__(BatchedEngine)
        engine._loaded = False
        engine._is_mllm = False
        engine._model_name = "test-model"
        engine._guided_requests_lock = threading.Lock()
        engine._guided_abort_events = {}
        engine._guided_owner_tasks = {}
        engine._guided_stopping = True
        engine._validate_lane_capabilities = MagicMock()
        engine._start_llm = AsyncMock()

        await engine.start()
        assert engine._guided_stopping is False

        engine._mllm_scheduler = None
        engine._engine = None
        engine._model_load_executor = None
        engine._processor = object()
        engine._mllm_instance = object()
        engine._engine_started = True
        await engine.stop()
        assert engine._guided_stopping is True
        assert engine._loaded is False

    @pytest.mark.asyncio
    async def test_guided_admission_publication_is_best_effort(self, monkeypatch):
        """A broken compatibility holder cannot prevent worker admission."""
        import concurrent.futures

        class _BrokenHolder:
            def __setitem__(self, _index, _value):
                raise RuntimeError("read-only holder")

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            engine = _make_guided_engine(
                monkeypatch,
                lambda **_kwargs: '{"ok": true}',
                executor=executor,
            )
            admitted = MagicMock()

            result = await engine.generate_with_schema(
                messages=[{"role": "user", "content": "hi"}],
                json_schema={"type": "object"},
                request_id_holder=_BrokenHolder(),
                request_admitted_event=admitted,
            )

            assert result.text == '{"ok": true}'
            admitted.set.assert_called_once_with()
            assert engine._guided_abort_events == {}
        finally:
            executor.shutdown(wait=True)

    @pytest.mark.asyncio
    async def test_guided_submission_failure_releases_identity(self, monkeypatch):
        class _RejectingExecutor:
            def submit(self, *_args, **_kwargs):
                raise RuntimeError("executor stopped")

        engine = _make_guided_engine(
            monkeypatch,
            lambda **_kwargs: "unused",
            executor=_RejectingExecutor(),
        )

        with pytest.raises(RuntimeError, match="executor stopped"):
            await engine.generate_with_schema(
                messages=[{"role": "user", "content": "hi"}],
                json_schema={"type": "object"},
                request_id="submission-failed",
            )
        assert engine._guided_abort_events == {}

    @pytest.mark.asyncio
    async def test_guided_worker_cancellation_preserves_lifecycle(self, monkeypatch):
        from vllm_mlx.api.errors import GuidedGenerationCancelledError

        def cancelled(**_kwargs):
            raise GuidedGenerationCancelledError()

        engine = _make_guided_engine(monkeypatch, cancelled)

        with pytest.raises(GuidedGenerationCancelledError) as exc:
            await engine.generate_with_schema(
                messages=[{"role": "user", "content": "hi"}],
                json_schema={"type": "object"},
                request_id="worker-cancelled",
            )
        assert exc.value.lifecycle_task is not None
        assert engine._guided_abort_events == {}

    @pytest.mark.asyncio
    async def test_guided_worker_base_exception_releases_identity(self, monkeypatch):
        class _WorkerFailure(BaseException):
            pass

        def fail(**_kwargs):
            raise _WorkerFailure()

        engine = _make_guided_engine(monkeypatch, fail)

        with pytest.raises(_WorkerFailure):
            await engine.generate_with_schema(
                messages=[{"role": "user", "content": "hi"}],
                json_schema={"type": "object"},
                request_id="worker-failed",
            )
        assert engine._guided_abort_events == {}

    @pytest.mark.asyncio
    async def test_completed_cancelled_future_releases_identity(self, monkeypatch):
        def cancel(**_kwargs):
            raise asyncio.CancelledError()

        import asyncio

        engine = _make_guided_engine(monkeypatch, cancel)

        with pytest.raises(asyncio.CancelledError):
            await engine.generate_with_schema(
                messages=[{"role": "user", "content": "hi"}],
                json_schema={"type": "object"},
                request_id="future-cancelled",
            )
        assert engine._guided_abort_events == {}

    @pytest.mark.asyncio
    async def test_task_cancel_defers_cleanup_until_worker_finishes(self, monkeypatch):
        """A live executor future owns the registry until its callback runs."""
        import asyncio
        import concurrent.futures
        import threading
        import time

        started = threading.Event()
        abort_seen = threading.Event()
        release_worker = threading.Event()
        stopped = threading.Event()

        def cooperate(*, should_abort, **_kwargs):
            started.set()
            while not should_abort():
                time.sleep(0.001)
            abort_seen.set()
            release_worker.wait()
            stopped.set()
            return None

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        engine = None
        task = None
        try:
            engine = _make_guided_engine(monkeypatch, cooperate, executor=executor)
            task = asyncio.create_task(
                engine.generate_with_schema(
                    messages=[{"role": "user", "content": "hi"}],
                    json_schema={"type": "object"},
                    request_id="pending-worker",
                )
            )
            assert await asyncio.to_thread(started.wait, 1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert await asyncio.to_thread(abort_seen.wait, 1)
            assert "pending-worker" in engine._guided_abort_events

            release_worker.set()
            assert await asyncio.to_thread(stopped.wait, 1)
            for _ in range(100):
                if not engine._guided_abort_events:
                    break
                await asyncio.sleep(0.001)
            assert engine._guided_abort_events == {}
        finally:
            release_worker.set()
            if engine is not None:
                engine.abort_guided_request("pending-worker")
            if task is not None and not task.done():
                task.cancel()
            executor.shutdown(wait=True)
            if task is not None:
                await asyncio.gather(task, return_exceptions=True)

    @pytest.mark.asyncio
    async def test_retained_guided_failure_keeps_identity_for_handoff(
        self, monkeypatch
    ):
        engine = _make_guided_engine(monkeypatch, lambda **_kwargs: None)

        with pytest.raises(RuntimeError, match="produced no result"):
            await engine.generate_with_schema(
                messages=[{"role": "user", "content": "hi"}],
                json_schema={"type": "object"},
                request_id="retained-failure",
                raise_on_failure=True,
                retain_guided_request_on_failure=True,
            )
        assert "retained-failure" in engine._guided_abort_events
        outcome = engine.finish_guided_handoff("retained-failure")
        assert outcome.cancelled is False
        assert engine._guided_abort_events == {}

    @pytest.mark.asyncio
    async def test_abort_after_worker_result_suppresses_commit(self, monkeypatch):
        from vllm_mlx.api.errors import GuidedGenerationCancelledError

        engine = None

        def abort_then_return(**_kwargs):
            assert engine is not None
            assert engine.abort_guided_request("result-race") is True
            return '{"must_not_commit": true}'

        engine = _make_guided_engine(monkeypatch, abort_then_return)
        with pytest.raises(GuidedGenerationCancelledError):
            await engine.generate_with_schema(
                messages=[{"role": "user", "content": "hi"}],
                json_schema={"type": "object"},
                request_id="result-race",
            )
        assert engine._guided_abort_events == {}

    @pytest.mark.asyncio
    async def test_retained_guided_failure_honors_accepted_abort(self, monkeypatch):
        from vllm_mlx.api.errors import GuidedGenerationCancelledError

        engine = None

        def abort_then_fail(**_kwargs):
            assert engine is not None
            assert engine.abort_guided_request("retained") is True
            return None

        engine = _make_guided_engine(monkeypatch, abort_then_fail)

        with pytest.raises(GuidedGenerationCancelledError):
            await engine.generate_with_schema(
                messages=[{"role": "user", "content": "hi"}],
                json_schema={"type": "object"},
                request_id="retained",
                raise_on_failure=True,
                retain_guided_request_on_failure=True,
            )
        assert engine._guided_abort_events == {}

    def test_run_guided_generation_never_degrades_cancellation(self, monkeypatch):
        from vllm_mlx.api.errors import GuidedGenerationCancelledError
        from vllm_mlx.engine import batched as batched_mod
        from vllm_mlx.engine.batched import BatchedEngine

        class _CancelledGenerator:
            def __init__(self, _model, _tokenizer):
                pass

            def generate_json(self, **_kwargs):
                raise GuidedGenerationCancelledError()

        monkeypatch.setattr(batched_mod, "GuidedGenerator", _CancelledGenerator)
        engine = BatchedEngine.__new__(BatchedEngine)
        engine._model = object()
        engine._tokenizer = object()
        engine._is_mllm = False

        with pytest.raises(GuidedGenerationCancelledError):
            engine._run_guided_generation("prompt", {}, 8, 0.0)

    def test_guided_registry_cleanup_is_instance_safe_and_shutdown_signals_all(self):
        """Late cleanup cannot remove a newer job reusing the same id."""
        import threading

        from vllm_mlx.engine.batched import BatchedEngine

        engine = BatchedEngine.__new__(BatchedEngine)
        engine._guided_requests_lock = threading.Lock()
        first = threading.Event()
        replacement = threading.Event()
        engine._guided_abort_events = {"req-guided": replacement, "req-other": first}
        engine._guided_owner_tasks = {}

        engine._finish_guided_request("req-guided", first)
        assert engine._guided_abort_events["req-guided"] is replacement

        with pytest.raises(RuntimeError, match="already active"):
            engine._register_guided_request("req-guided", threading.Event())

        engine._abort_all_guided_requests()
        assert first.is_set()
        assert replacement.is_set()

        late = threading.Event()
        from vllm_mlx.api.errors import GuidedGenerationCancelledError

        with pytest.raises(GuidedGenerationCancelledError):
            engine._register_guided_request("req-late", late)
        assert late.is_set()
        assert "req-late" not in engine._guided_abort_events

    def test_stop_and_registration_serialize_on_the_guided_lock(self):
        """A registration waiting behind shutdown cannot miss its signal."""
        import threading

        from vllm_mlx.api.errors import GuidedGenerationCancelledError
        from vllm_mlx.engine.batched import BatchedEngine

        engine = BatchedEngine.__new__(BatchedEngine)
        engine._admission_lock = threading.Lock()
        engine._lifecycle_aborted_tasks = set()
        engine._guided_requests_lock = threading.Lock()
        engine._guided_abort_events = {}
        engine._guided_owner_tasks = {}
        engine._guided_stopping = False
        late = threading.Event()
        started = threading.Event()
        outcome: list[type[BaseException]] = []

        def register_late() -> None:
            started.set()
            try:
                engine._register_guided_request("req-late", late)
            except BaseException as exc:
                outcome.append(type(exc))

        with engine._guided_requests_lock:
            thread = threading.Thread(target=register_late)
            thread.start()
            assert started.wait(timeout=1)
            engine._guided_stopping = True
        thread.join(timeout=1)

        assert outcome == [GuidedGenerationCancelledError]
        assert late.is_set()
        assert engine._guided_abort_events == {}

    @pytest.mark.asyncio
    async def test_shutdown_marks_guided_owner_in_real_lifecycle_ledger(self):
        """Guided shutdown cancellation reaches the standard route 503 ledger."""
        import asyncio
        import threading

        from vllm_mlx.engine.batched import BatchedEngine

        engine = BatchedEngine.__new__(BatchedEngine)
        engine._admission_lock = threading.Lock()
        engine._lifecycle_aborted_tasks = set()
        engine._guided_requests_lock = threading.Lock()
        engine._guided_abort_events = {}
        engine._guided_owner_tasks = {}
        engine._guided_stopping = False
        owner = asyncio.current_task()
        assert owner is not None
        abort_event = threading.Event()

        engine._register_guided_request("req-guided", abort_event, owner)
        engine._abort_all_guided_requests()

        assert abort_event.is_set()
        assert engine.consume_lifecycle_task_abort(owner) is True
        assert engine.consume_lifecycle_task_abort(owner) is False

    @pytest.mark.asyncio
    async def test_routes_to_mllm_scheduler_when_present(self):
        from vllm_mlx.engine.batched import BatchedEngine

        engine = BatchedEngine.__new__(BatchedEngine)
        engine._mllm_scheduler = _StubSyncMllmScheduler(returns=True)
        engine._engine = _StubAsyncEngine(returns=False)

        result = await engine.abort_request("req-mllm")

        assert result is True
        assert engine._mllm_scheduler.calls == ["req-mllm"]
        assert engine._engine.calls == []

    @pytest.mark.asyncio
    async def test_routes_to_text_engine_when_no_mllm_scheduler(self):
        from vllm_mlx.engine.batched import BatchedEngine

        engine = BatchedEngine.__new__(BatchedEngine)
        engine._mllm_scheduler = None
        engine._engine = _StubAsyncEngine(returns=True)

        result = await engine.abort_request("req-text")

        assert result is True
        assert engine._engine.calls == ["req-text"]

    @pytest.mark.asyncio
    async def test_returns_false_when_no_engine_loaded(self):
        from vllm_mlx.engine.batched import BatchedEngine

        engine = BatchedEngine.__new__(BatchedEngine)
        engine._mllm_scheduler = None
        engine._engine = None

        result = await engine.abort_request("req-none")

        assert result is False

    @pytest.mark.asyncio
    async def test_handles_sync_text_engine_abort(self):
        """Synthetic engine returning bool directly (not a coroutine)."""
        from vllm_mlx.engine.batched import BatchedEngine

        sync_engine = MagicMock()
        sync_engine.abort_request = MagicMock(return_value=True)

        engine = BatchedEngine.__new__(BatchedEngine)
        engine._mllm_scheduler = None
        engine._engine = sync_engine

        result = await engine.abort_request("req-sync")

        assert result is True
        sync_engine.abort_request.assert_called_once_with("req-sync")


class TestBaseEngineDefaultAbort:
    @pytest.mark.asyncio
    async def test_default_returns_false(self):
        """Invoke ``BaseEngine.abort_request`` via the unbound method to dodge
        the abstract-method instantiation guard. We only care that the default
        returns False — no engine state is needed."""
        from vllm_mlx.engine.base import BaseEngine

        sentinel = object()
        result = await BaseEngine.abort_request(sentinel, "any")  # type: ignore[arg-type]
        assert result is False


class TestGuidedRouteCancellationClassification:
    """Explicit user cancellation stays distinct from model replacement."""

    @staticmethod
    def _cancelled_engine():
        from vllm_mlx.api.errors import GuidedGenerationCancelledError
        from vllm_mlx.engine.base import GenerationOutput

        class _CancelledEngine:
            preserve_native_tool_format = False
            is_mllm = False
            tokenizer = None
            supports_guided_generation = True

            def build_prompt(self, messages, tools=None, enable_thinking=None):
                return "prompt"

            async def generate_with_schema(self, *, messages, json_schema, **kwargs):
                raise GuidedGenerationCancelledError()

            async def chat(self, *, messages, **kwargs):
                return GenerationOutput(text='{"value": 42}', finished=True)

        return _CancelledEngine()

    def test_responses_guided_user_cancel_propagates_cancelled_error(self):
        import concurrent.futures

        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from vllm_mlx.config import reset_config
        from vllm_mlx.middleware.auth import rate_limiter
        from vllm_mlx.routes.responses import router as responses_router

        saved_enabled = rate_limiter.enabled
        saved_rpm = rate_limiter.requests_per_minute
        saved_requests = dict(rate_limiter._requests)
        try:
            rate_limiter.enabled = False
            cfg = reset_config()
            cfg.engine = self._cancelled_engine()
            cfg.model_name = "test-model"
            cfg.model_registry = None
            cfg.no_thinking = True
            app = FastAPI()
            app.include_router(responses_router)
            client = TestClient(app)
            with pytest.raises(concurrent.futures.CancelledError):
                client.post(
                    "/v1/responses",
                    json={
                        "model": "test-model",
                        "input": "emit json",
                        "text": {
                            "format": {
                                "type": "json_schema",
                                "name": "result",
                                "schema": {"type": "object"},
                                "strict": True,
                            }
                        },
                    },
                )
        finally:
            rate_limiter.enabled = saved_enabled
            rate_limiter.requests_per_minute = saved_rpm
            rate_limiter._requests.clear()
            rate_limiter._requests.update(saved_requests)


class TestCancelRequestEndpoint:
    # Per F-150, the cancel route now lives on ``admin_router`` and requires
    # ``X-Rapid-MLX-Internal: true``. All tests in this class pass it via
    # ``_HDRS`` — the header-only-403 path is exercised separately in
    # ``test_internal_route_auth.py``.
    _HDRS = {"X-Rapid-MLX-Internal": "true"}

    @pytest.fixture
    def client_with_engine(self):
        """Build a FastAPI test client with a stub engine wired into the
        process-wide ``ServerConfig`` singleton."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from vllm_mlx.config import get_config
        from vllm_mlx.routes.health import admin_router, router

        cfg = get_config()
        prev_engine, prev_model_name = cfg.engine, cfg.model_name

        engine = AsyncMock()
        engine.abort_request = AsyncMock(return_value=True)
        cfg.engine = engine
        cfg.model_name = "test-model"

        app = FastAPI()
        app.include_router(router)
        app.include_router(admin_router)
        # ``TestClient(app)`` defaults the scope's ``client`` to
        # ``("testclient", 50000)`` which is NOT loopback under
        # ``ipaddress.is_loopback``. ``verify_internal_admin`` (codex r1 fix)
        # rejects non-loopback callers when ``cfg.api_key`` is unset, so we
        # pin the client to ``127.0.0.1`` here — this fixture is exercising
        # the route's body/leak behaviour, not the auth gate's loopback
        # branch. The dedicated coverage lives in
        # ``test_internal_route_auth.py``.
        client = TestClient(app, client=("127.0.0.1", 50000))
        try:
            yield client, engine
        finally:
            cfg.engine = prev_engine
            cfg.model_name = prev_model_name

    def test_post_cancel_returns_200_when_engine_aborts(self, client_with_engine):
        client, engine = client_with_engine

        response = client.post(
            "/v1/requests/chatcmpl-abc123/cancel", headers=self._HDRS
        )

        assert response.status_code == 200
        body = response.json()
        assert body["object"] == "request.cancel"
        assert body["id"] == "chatcmpl-abc123"
        assert body["cancelled"] is True
        # F-151: ``model`` MUST NOT appear in the cancel envelope. Echoing
        # ``cfg.model_name`` here used to leak the HF repo id to anonymous
        # callers (which, before the F-150 gate, was every LAN client).
        assert "model" not in body
        engine.abort_request.assert_awaited_once_with("chatcmpl-abc123")

    def test_post_cancel_returns_404_when_engine_returns_false(
        self, client_with_engine
    ):
        client, engine = client_with_engine
        engine.abort_request.return_value = False

        response = client.post("/v1/requests/missing/cancel", headers=self._HDRS)

        assert response.status_code == 404
        assert "Request not found" in response.json()["detail"]
        # F-151: the 404 detail MUST NOT echo server-side state like the
        # raw model name (``cfg.model_name`` happens to be "test-model" in
        # this fixture).
        assert "test-model" not in response.text

    def test_delete_alias_returns_200(self, client_with_engine):
        client, _ = client_with_engine

        response = client.delete("/v1/requests/chatcmpl-xyz", headers=self._HDRS)

        assert response.status_code == 200
        body = response.json()
        assert body["cancelled"] is True
        # Same F-151 leak assertion as the POST path.
        assert "model" not in body

    def test_post_cancel_returns_503_when_no_engine(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from vllm_mlx.config import get_config
        from vllm_mlx.routes.health import admin_router, router

        cfg = get_config()
        prev_engine = cfg.engine
        cfg.engine = None
        try:
            app = FastAPI()
            app.include_router(router)
            app.include_router(admin_router)
            client = TestClient(app, client=("127.0.0.1", 50000))

            response = client.post("/v1/requests/any/cancel", headers=self._HDRS)

            assert response.status_code == 503
        finally:
            cfg.engine = prev_engine
