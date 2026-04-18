# SPDX-License-Identifier: Apache-2.0
"""Tests for extracted route modules (health, models, embeddings, MCP, audio).

Uses FastAPI TestClient with mocked server globals to test each route
in isolation without needing a real model or server running.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_engine():
    """Mock engine with standard attributes."""
    engine = MagicMock()
    engine.is_mllm = False
    engine.get_stats.return_value = {
        "engine_type": "batched",
        "running": False,
        "uptime_seconds": 123.4,
        "steps_executed": 500,
        "num_running": 0,
        "num_waiting": 0,
        "num_requests_processed": 42,
        "total_prompt_tokens": 1000,
        "total_completion_tokens": 2000,
        "metal_active_memory_gb": 8.5,
        "metal_peak_memory_gb": 12.0,
        "metal_cache_memory_gb": 3.0,
    }
    return engine


@pytest.fixture
def mock_registry():
    """Mock model registry."""
    entry = MagicMock()
    entry.model_name = "test-model"
    entry.aliases = {"test-alias", "test-model"}
    registry = MagicMock()
    registry.list_entries.return_value = [entry]
    registry.__contains__ = lambda self, x: x in ("test-model", "test-alias")
    return registry


# ---------------------------------------------------------------------------
# Health routes
# ---------------------------------------------------------------------------


class TestHealthRoutes:
    def _make_app(self):
        from vllm_mlx.routes.health import router
        app = FastAPI()
        app.include_router(router)
        return app

    def test_health_no_engine(self, mock_engine):
        """Health endpoint works when no engine is loaded."""
        with patch("vllm_mlx.server._engine", None), \
             patch("vllm_mlx.server._mcp_manager", None), \
             patch("vllm_mlx.server._model_name", None):
            app = self._make_app()
            client = TestClient(app)
            r = client.get("/health")
            assert r.status_code == 200
            data = r.json()
            assert data["status"] == "healthy"
            assert data["model_loaded"] is False

    def test_health_with_engine(self, mock_engine):
        """Health endpoint returns engine info."""
        with patch("vllm_mlx.server._engine", mock_engine), \
             patch("vllm_mlx.server._mcp_manager", None), \
             patch("vllm_mlx.server._model_name", "test-model"):
            app = self._make_app()
            client = TestClient(app)
            r = client.get("/health")
            assert r.status_code == 200
            data = r.json()
            assert data["model_loaded"] is True
            assert data["model_name"] == "test-model"
            assert data["engine_type"] == "batched"

    def test_health_with_mcp(self, mock_engine):
        """Health endpoint includes MCP info."""
        mcp = MagicMock()
        server_status = MagicMock()
        server_status.state.value = "connected"
        mcp.get_server_status.return_value = [server_status]
        mcp.get_all_tools.return_value = [MagicMock(), MagicMock()]

        with patch("vllm_mlx.server._engine", mock_engine), \
             patch("vllm_mlx.server._mcp_manager", mcp), \
             patch("vllm_mlx.server._model_name", "test-model"):
            app = self._make_app()
            client = TestClient(app)
            r = client.get("/health")
            data = r.json()
            assert data["mcp"]["enabled"] is True
            assert data["mcp"]["servers_connected"] == 1
            assert data["mcp"]["tools_available"] == 2

    def test_status_no_engine(self):
        """Status returns not_loaded when no engine."""
        with patch("vllm_mlx.server._engine", None), \
             patch("vllm_mlx.server._model_name", None):
            app = self._make_app()
            client = TestClient(app)
            r = client.get("/v1/status")
            assert r.status_code == 200
            assert r.json()["status"] == "not_loaded"

    def test_status_with_engine(self, mock_engine):
        """Status returns engine stats."""
        with patch("vllm_mlx.server._engine", mock_engine), \
             patch("vllm_mlx.server._model_name", "test-model"):
            app = self._make_app()
            client = TestClient(app)
            r = client.get("/v1/status")
            data = r.json()
            assert data["status"] == "idle"
            assert data["model"] == "test-model"
            assert data["steps_executed"] == 500
            assert data["metal"]["active_memory_gb"] == 8.5

    def test_cache_clear_no_engine(self):
        """Cache clear returns 503 when no engine."""
        with patch("vllm_mlx.server._engine", None):
            app = self._make_app()
            client = TestClient(app)
            r = client.post("/v1/cache/clear")
            assert r.status_code == 503

    def test_cache_clear_no_prompt_cache(self, mock_engine):
        """Cache clear works when no prompt cache exists."""
        mock_engine._model = MagicMock(spec=[])  # no _prompt_cache attr
        with patch("vllm_mlx.server._engine", mock_engine):
            app = self._make_app()
            client = TestClient(app)
            r = client.post("/v1/cache/clear")
            assert r.status_code == 200
            assert "No prompt cache" in r.json()["message"]

    def test_cache_stats_no_vlm(self):
        """Cache stats returns fallback when mlx_vlm not available."""
        app = self._make_app()
        client = TestClient(app)
        r = client.get("/v1/cache/stats")
        assert r.status_code == 200
        # Either returns stats or fallback message
        data = r.json()
        assert "multimodal_kv_cache" in data or "model_type" in data

    def test_cache_delete(self):
        """Cache delete endpoint works."""
        app = self._make_app()
        client = TestClient(app)
        r = client.delete("/v1/cache")
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# Models routes
# ---------------------------------------------------------------------------


class TestModelsRoutes:
    def _make_app(self):
        from vllm_mlx.routes.models import router
        app = FastAPI()
        # Mock the verify_api_key dependency
        app.include_router(router)
        return app

    def test_list_models_single(self):
        """List models with single model loaded."""
        with patch("vllm_mlx.server._model_registry", None), \
             patch("vllm_mlx.server._model_name", "test-model"), \
             patch("vllm_mlx.server._model_alias", "test-alias"), \
             patch("vllm_mlx.server._api_key", None):
            app = self._make_app()
            client = TestClient(app)
            r = client.get("/v1/models")
            assert r.status_code == 200
            models = r.json()["data"]
            ids = [m["id"] for m in models]
            assert "test-model" in ids
            assert "test-alias" in ids

    def test_list_models_registry(self, mock_registry):
        """List models with multi-model registry."""
        with patch("vllm_mlx.server._model_registry", mock_registry), \
             patch("vllm_mlx.server._model_name", "test-model"), \
             patch("vllm_mlx.server._model_alias", None), \
             patch("vllm_mlx.server._api_key", None):
            app = self._make_app()
            client = TestClient(app)
            r = client.get("/v1/models")
            assert r.status_code == 200
            assert len(r.json()["data"]) >= 1

    def test_retrieve_model_found(self):
        """Retrieve existing model returns 200."""
        with patch("vllm_mlx.server._model_registry", None), \
             patch("vllm_mlx.server._model_name", "test-model"), \
             patch("vllm_mlx.server._model_alias", "test-alias"), \
             patch("vllm_mlx.server._api_key", None):
            app = self._make_app()
            client = TestClient(app)
            r = client.get("/v1/models/test-model")
            assert r.status_code == 200
            assert r.json()["id"] == "test-model"

    def test_retrieve_model_not_found(self):
        """Retrieve non-existent model returns 404."""
        with patch("vllm_mlx.server._model_registry", None), \
             patch("vllm_mlx.server._model_name", "test-model"), \
             patch("vllm_mlx.server._model_alias", None), \
             patch("vllm_mlx.server._api_key", None):
            app = self._make_app()
            client = TestClient(app)
            r = client.get("/v1/models/nonexistent")
            assert r.status_code == 404


# ---------------------------------------------------------------------------
# MCP routes
# ---------------------------------------------------------------------------


class TestMCPRoutes:
    def _make_app(self):
        from vllm_mlx.routes.mcp_routes import router
        app = FastAPI()
        app.include_router(router)
        return app

    def test_list_tools_no_mcp(self):
        """List tools returns empty when MCP not configured."""
        with patch("vllm_mlx.server._mcp_manager", None), \
             patch("vllm_mlx.server._api_key", None):
            app = self._make_app()
            client = TestClient(app)
            r = client.get("/v1/mcp/tools")
            assert r.status_code == 200
            assert r.json()["count"] == 0

    def test_list_tools_with_mcp(self):
        """List tools returns tools when MCP configured."""
        tool = MagicMock()
        tool.full_name = "test_tool"
        tool.description = "A test tool"
        tool.server_name = "test_server"
        tool.input_schema = {"type": "object"}

        mcp = MagicMock()
        mcp.get_all_tools.return_value = [tool]

        with patch("vllm_mlx.server._mcp_manager", mcp), \
             patch("vllm_mlx.server._api_key", None):
            app = self._make_app()
            client = TestClient(app)
            r = client.get("/v1/mcp/tools")
            data = r.json()
            assert data["count"] == 1
            assert data["tools"][0]["name"] == "test_tool"

    def test_list_servers_no_mcp(self):
        """List servers returns empty when MCP not configured."""
        with patch("vllm_mlx.server._mcp_manager", None), \
             patch("vllm_mlx.server._api_key", None):
            app = self._make_app()
            client = TestClient(app)
            r = client.get("/v1/mcp/servers")
            assert r.status_code == 200
            assert r.json()["servers"] == []

    def test_execute_no_mcp(self):
        """Execute tool returns 503 when MCP not configured."""
        with patch("vllm_mlx.server._mcp_manager", None), \
             patch("vllm_mlx.server._api_key", None):
            app = self._make_app()
            client = TestClient(app)
            r = client.post("/v1/mcp/execute", json={
                "tool_name": "test", "arguments": {}
            })
            assert r.status_code == 503

    def test_execute_with_mcp(self):
        """Execute tool works with MCP configured."""
        result = MagicMock()
        result.tool_name = "test_tool"
        result.content = "result"
        result.is_error = False
        result.error_message = None

        mcp = MagicMock()
        mcp.execute_tool = AsyncMock(return_value=result)

        with patch("vllm_mlx.server._mcp_manager", mcp), \
             patch("vllm_mlx.server._api_key", None):
            app = self._make_app()
            client = TestClient(app)
            r = client.post("/v1/mcp/execute", json={
                "tool_name": "test_tool", "arguments": {"key": "val"}
            })
            assert r.status_code == 200
            assert r.json()["tool_name"] == "test_tool"


# ---------------------------------------------------------------------------
# Embeddings routes
# ---------------------------------------------------------------------------


class TestEmbeddingsRoutes:
    def _make_app(self):
        from vllm_mlx.routes.embeddings import router
        app = FastAPI()
        app.include_router(router)
        return app

    def test_embeddings_success(self):
        """Embeddings endpoint returns vectors."""
        mock_emb_engine = MagicMock()
        mock_emb_engine.count_tokens.return_value = 5
        mock_emb_engine.embed.return_value = [[0.1, 0.2, 0.3]]

        with patch("vllm_mlx.server._embedding_engine", mock_emb_engine), \
             patch("vllm_mlx.server._embedding_model_locked", None), \
             patch("vllm_mlx.server.load_embedding_model"), \
             patch("vllm_mlx.server._api_key", None), \
             patch("vllm_mlx.server.check_rate_limit", return_value=None):
            app = self._make_app()
            client = TestClient(app)
            r = client.post("/v1/embeddings", json={
                "model": "test-embed",
                "input": "hello world",
            })
            assert r.status_code == 200
            data = r.json()
            assert len(data["data"]) == 1
            assert data["data"][0]["embedding"] == [0.1, 0.2, 0.3]
            assert data["usage"]["prompt_tokens"] == 5

    def test_embeddings_batch(self):
        """Embeddings endpoint handles batch input."""
        mock_emb_engine = MagicMock()
        mock_emb_engine.count_tokens.return_value = 10
        mock_emb_engine.embed.return_value = [[0.1], [0.2], [0.3]]

        with patch("vllm_mlx.server._embedding_engine", mock_emb_engine), \
             patch("vllm_mlx.server._embedding_model_locked", None), \
             patch("vllm_mlx.server.load_embedding_model"), \
             patch("vllm_mlx.server._api_key", None), \
             patch("vllm_mlx.server.check_rate_limit", return_value=None):
            app = self._make_app()
            client = TestClient(app)
            r = client.post("/v1/embeddings", json={
                "model": "test-embed",
                "input": ["a", "b", "c"],
            })
            assert r.status_code == 200
            assert len(r.json()["data"]) == 3

    def test_embeddings_locked_model_reject(self):
        """Embeddings rejects wrong model when locked."""
        with patch("vllm_mlx.server._embedding_engine", MagicMock()), \
             patch("vllm_mlx.server._embedding_model_locked", "locked-model"), \
             patch("vllm_mlx.server._api_key", None), \
             patch("vllm_mlx.server.check_rate_limit", return_value=None):
            app = self._make_app()
            client = TestClient(app)
            r = client.post("/v1/embeddings", json={
                "model": "wrong-model",
                "input": "test",
            })
            assert r.status_code == 400
            assert "not available" in r.json()["detail"]
