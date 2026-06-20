# SPDX-License-Identifier: Apache-2.0
"""Tests for /v1/requests/{id}/cancel and BatchedEngine.abort_request routing.

Cherry-pick coverage of upstream waybarrios/vllm-mlx#426 adapted to our
routes/ split (their endpoint lives on the FastAPI app; ours lives on the
health router).
"""

from unittest.mock import AsyncMock, MagicMock

import pytest


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
        client = TestClient(app)
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
            client = TestClient(app)

            response = client.post("/v1/requests/any/cancel", headers=self._HDRS)

            assert response.status_code == 503
        finally:
            cfg.engine = prev_engine
