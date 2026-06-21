# SPDX-License-Identifier: Apache-2.0
"""Tests for the OpenAI-compatible Embeddings API."""

import platform
import sys
from unittest.mock import MagicMock, patch

import pytest

# Skip all tests if not on Apple Silicon
pytestmark = pytest.mark.skipif(
    sys.platform != "darwin" or platform.machine() != "arm64",
    reason="Requires Apple Silicon",
)


# =============================================================================
# Unit Tests - Pydantic Models
# =============================================================================


class TestEmbeddingModels:
    """Test embedding request/response Pydantic models."""

    def test_embedding_request_single_string(self):
        """Test EmbeddingRequest with a single input string."""
        from vllm_mlx.api.models import EmbeddingRequest

        req = EmbeddingRequest(model="test-model", input="Hello world")
        assert req.model == "test-model"
        assert req.input == "Hello world"
        assert req.encoding_format == "float"

    def test_embedding_response_serialization(self):
        """Test that EmbeddingResponse serializes to OpenAI-compatible JSON."""
        from vllm_mlx.api.models import (
            EmbeddingData,
            EmbeddingResponse,
            EmbeddingUsage,
        )

        response = EmbeddingResponse(
            data=[EmbeddingData(index=0, embedding=[1.0, 2.0, 3.0])],
            model="text-embedding-3-large",
            usage=EmbeddingUsage(prompt_tokens=5, total_tokens=5),
        )
        d = response.model_dump()
        assert d["object"] == "list"
        assert d["data"][0]["object"] == "embedding"
        assert d["data"][0]["index"] == 0
        assert d["data"][0]["embedding"] == [1.0, 2.0, 3.0]
        assert d["model"] == "text-embedding-3-large"
        assert d["usage"]["prompt_tokens"] == 5
        assert d["usage"]["total_tokens"] == 5


# =============================================================================
# Unit Tests - Embedding Engine
# =============================================================================


class TestEmbeddingEngine:
    """Test the EmbeddingEngine wrapper."""

    @patch("vllm_mlx.embedding.EmbeddingEngine.load")
    @patch(
        "vllm_mlx.embedding.EmbeddingEngine.is_loaded",
        new_callable=lambda: property(lambda self: True),
    )
    def test_embed_calls_model_directly(self, _mock_loaded, mock_load):
        """Test embed tokenizes and calls model directly (bypasses generate)."""
        import numpy as np

        from vllm_mlx.embedding import EmbeddingEngine

        engine = EmbeddingEngine("test-model")

        mock_output = MagicMock()
        mock_output.text_embeds.tolist.return_value = [[0.1, 0.2], [0.3, 0.4]]

        mock_model = MagicMock(return_value=mock_output)

        mock_inner_tokenizer = MagicMock()
        mock_inner_tokenizer.return_value = {
            "input_ids": np.array([[1, 2], [3, 4]]),
            "attention_mask": np.array([[1, 1], [1, 1]]),
        }
        mock_tokenizer = MagicMock()
        mock_tokenizer._tokenizer = mock_inner_tokenizer

        engine._model = mock_model
        engine._tokenizer = mock_tokenizer
        result = engine.embed(["hello", "world"])

        mock_model.assert_called_once()
        assert len(result) == 2
        assert result[0] == [0.1, 0.2]

    def test_embed_normalises_single_string(self):
        """Test that a single string input is wrapped into a list."""
        import numpy as np

        from vllm_mlx.embedding import EmbeddingEngine

        engine = EmbeddingEngine("test-model")

        mock_output = MagicMock()
        mock_output.text_embeds.tolist.return_value = [[0.5, 0.6]]

        mock_model = MagicMock(return_value=mock_output)

        mock_inner_tokenizer = MagicMock()
        mock_inner_tokenizer.return_value = {
            "input_ids": np.array([[1, 2]]),
            "attention_mask": np.array([[1, 1]]),
        }
        mock_tokenizer = MagicMock()
        mock_tokenizer._tokenizer = mock_inner_tokenizer

        with patch.object(engine, "_ensure_loaded"):
            engine._model = mock_model
            engine._tokenizer = mock_tokenizer
            result = engine.embed("single text")

        assert len(result) == 1

    def test_count_tokens(self):
        """Test token counting for usage reporting."""
        from vllm_mlx.embedding import EmbeddingEngine

        engine = EmbeddingEngine("test-model")
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        engine._tokenizer = mock_tokenizer
        engine._model = MagicMock()  # mark as loaded

        count = engine.count_tokens(["hello", "world"])
        assert count == 10  # 5 tokens * 2 texts


# =============================================================================
# Integration Tests - FastAPI Endpoint
# =============================================================================


class TestEmbeddingsEndpoint:
    """Test the /v1/embeddings endpoint via TestClient."""

    def _set_embedding_engine(self, engine):
        """Set embedding engine in both server globals and config."""
        import vllm_mlx.server as srv
        from vllm_mlx.config import get_config

        srv._embedding_engine = engine
        get_config().embedding_engine = engine

    @pytest.fixture()
    def client(self):
        """Create a FastAPI test client with mocked embedding engine."""
        from fastapi.testclient import TestClient

        from vllm_mlx.server import app

        return TestClient(app)

    def test_batch_input_preserves_order(self, client):
        """Test batch embedding returns vectors with correct indices."""
        import vllm_mlx.server as srv
        from vllm_mlx.config import get_config

        texts = ["first", "second", "third"]
        mock_engine = MagicMock()
        mock_engine.model_name = "test-embed"
        mock_engine.embed.return_value = [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.5],
        ]
        mock_engine.count_tokens.return_value = 9

        cfg = get_config()
        original = srv._embedding_engine
        original_locked = srv._embedding_model_locked
        cfg_original_engine = cfg.embedding_engine
        cfg_original_locked = cfg.embedding_model_locked
        srv._embedding_engine = mock_engine
        # H-09: the route now requires the embedding model to be
        # configured (i.e. the server was started with
        # --embedding-model). Set BOTH the server global AND the cfg
        # mirror so the route's bridge branch doesn't reset to a
        # stale value from a prior test that left cfg set.
        srv._embedding_model_locked = "test-embed"
        cfg.embedding_engine = mock_engine
        cfg.embedding_model_locked = "test-embed"
        try:
            resp = client.post(
                "/v1/embeddings",
                json={"model": "test-embed", "input": texts},
            )
        finally:
            srv._embedding_engine = original
            srv._embedding_model_locked = original_locked
            cfg.embedding_engine = cfg_original_engine
            cfg.embedding_model_locked = cfg_original_locked

        assert resp.status_code == 200
        body = resp.json()
        assert len(body["data"]) == 3
        for i in range(3):
            assert body["data"][i]["index"] == i
        # Verify order matches
        assert body["data"][0]["embedding"] == [1.0, 0.0]
        assert body["data"][2]["embedding"] == [0.5, 0.5]

    def test_empty_input_returns_400(self, client):
        """Test that empty input list returns 400 error."""
        import vllm_mlx.server as srv
        from vllm_mlx.config import get_config

        mock_engine = MagicMock()
        mock_engine.model_name = "test-embed"

        cfg = get_config()
        original = srv._embedding_engine
        original_locked = srv._embedding_model_locked
        cfg_original_engine = cfg.embedding_engine
        cfg_original_locked = cfg.embedding_model_locked
        srv._embedding_engine = mock_engine
        srv._embedding_model_locked = "test-embed"
        cfg.embedding_engine = mock_engine
        cfg.embedding_model_locked = "test-embed"
        try:
            resp = client.post(
                "/v1/embeddings",
                json={"model": "test-embed", "input": []},
            )
        finally:
            srv._embedding_engine = original
            srv._embedding_model_locked = original_locked
            cfg.embedding_engine = cfg_original_engine
            cfg.embedding_model_locked = cfg_original_locked

        assert resp.status_code == 400

    def test_model_hot_swap_disabled_when_unlocked(self, client):
        """H-09: hot-swap is gone — without --embedding-model, every
        ``/v1/embeddings`` request is rejected with 400.

        Pre-fix the route would call ``load_embedding_model`` on
        ``request.model`` (typically the chat model, since no embedding
        model was configured) and silently return chat-model hidden
        states as if they were embeddings. The new guard refuses to do
        that — callers must explicitly configure the embedding model at
        startup.
        """
        import vllm_mlx.server as srv
        from vllm_mlx.config import get_config

        mock_engine = MagicMock()
        mock_engine.model_name = "old-model"

        cfg = get_config()
        original = srv._embedding_engine
        original_locked = srv._embedding_model_locked
        cfg_original_engine = cfg.embedding_engine
        cfg_original_locked = cfg.embedding_model_locked
        srv._embedding_engine = mock_engine
        srv._embedding_model_locked = None  # not configured
        cfg.embedding_engine = mock_engine
        cfg.embedding_model_locked = None

        try:
            with patch("vllm_mlx.embedding.EmbeddingEngine") as mock_cls:
                resp = client.post(
                    "/v1/embeddings",
                    json={"model": "new-model", "input": "test"},
                )
                # Hot-swap path is gone; route 400s without touching
                # the engine class.
                assert resp.status_code == 400
                mock_cls.assert_not_called()
                body = resp.json()
                msg = body["error"]["message"]
                assert "embeddings model not configured" in msg
                assert "rapid-mlx[embeddings]" in msg
        finally:
            srv._embedding_engine = original
            srv._embedding_model_locked = original_locked
            cfg.embedding_engine = cfg_original_engine
            cfg.embedding_model_locked = cfg_original_locked

    def test_model_locked_rejects_different_model(self, client):
        """Test that a locked embedding model rejects requests for different models."""
        import vllm_mlx.server as srv
        from vllm_mlx.config import get_config

        mock_engine = MagicMock()
        mock_engine.model_name = "locked-model"

        cfg = get_config()
        original_engine = srv._embedding_engine
        original_locked = srv._embedding_model_locked
        cfg_original_engine = cfg.embedding_engine
        cfg_original_locked = cfg.embedding_model_locked
        srv._embedding_engine = mock_engine
        srv._embedding_model_locked = "locked-model"
        cfg.embedding_engine = mock_engine
        cfg.embedding_model_locked = "locked-model"

        try:
            resp = client.post(
                "/v1/embeddings",
                json={"model": "other-model", "input": "test"},
            )
            assert resp.status_code == 400
            body = resp.json()
            # OpenAI-style error envelope, per the server-wide
            # StarletteHTTPException handler installed by PR #491.
            message = body["error"]["message"]
            assert "locked-model" in message
            assert "other-model" in message
        finally:
            srv._embedding_engine = original_engine
            srv._embedding_model_locked = original_locked
            cfg.embedding_engine = cfg_original_engine
            cfg.embedding_model_locked = cfg_original_locked


# =============================================================================
# Slow Integration Test - Real Model
# =============================================================================


@pytest.mark.slow
class TestEmbeddingsRealModel:
    """Integration tests with a real mlx-embeddings model."""

    @pytest.fixture(scope="class")
    def engine(self):
        pytest.importorskip("mlx_embeddings")
        from vllm_mlx.embedding import EmbeddingEngine

        eng = EmbeddingEngine("mlx-community/all-MiniLM-L6-v2-4bit")
        eng.load()
        return eng

    def test_single_embedding_shape(self, engine):
        """Test that a single text produces a correctly shaped vector."""
        result = engine.embed("Hello world")
        assert len(result) == 1
        assert len(result[0]) > 0  # non-empty embedding
        assert all(isinstance(v, float) for v in result[0])
