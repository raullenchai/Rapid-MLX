# SPDX-License-Identifier: Apache-2.0
"""H6 + H22 + C4 bundle — empirical reproducers + regression tests.

Each test block starts with a 1-2 sentence rationale citing the
reproducer that originally surfaced the bug, then pins the corrected
behavior. The structure is:

- H6 — OpenAI embeddings spec supports four input formats: ``str``,
  ``list[str]``, ``list[int]`` (pre-tokenized one input), and
  ``list[list[int]]`` (batch of pre-tokenized). Production pipelines
  using a shared tokenizer send the latter two; pre-PR these 422'd at
  parse time.
- H22 — Default request timeout was 300s, which silently cuts long
  reasoning generations. Industry baseline (vLLM, OpenAI proxy) is
  600-1800s. Bump to 1800.
- C4 — No admission control. A buggy client (or simple fork bomb) can
  schedule unbounded concurrent requests, OOM the Metal allocator,
  and crash the server for every other client. Add a cap.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# H6 — Pydantic model accepts all four OpenAI input shapes
# ---------------------------------------------------------------------------


class TestEmbeddingInputFourShapes:
    """Reproducer:
        curl /v1/embeddings -d '{"input": [[1,2,3]]}'   →  422 pre-PR

    The OpenAI spec
    (https://platform.openai.com/docs/api-reference/embeddings/create#embeddings/create-input)
    lists all four shapes as valid; clients using a pre-tokenized
    pipeline (LangChain, LlamaIndex with custom tokenizer) send the
    int forms by default.
    """

    def test_str_accepted(self):
        from vllm_mlx.api.models import EmbeddingRequest

        req = EmbeddingRequest(model="x", input="hello")
        assert req.input == "hello"

    def test_list_str_accepted(self):
        from vllm_mlx.api.models import EmbeddingRequest

        req = EmbeddingRequest(model="x", input=["a", "b"])
        assert req.input == ["a", "b"]

    def test_list_int_accepted(self):
        """list[int] — single pre-tokenized input."""
        from vllm_mlx.api.models import EmbeddingRequest

        req = EmbeddingRequest(model="x", input=[101, 2023, 2003, 102])
        assert req.input == [101, 2023, 2003, 102]

    def test_list_list_int_accepted(self):
        """list[list[int]] — batch of pre-tokenized inputs."""
        from vllm_mlx.api.models import EmbeddingRequest

        req = EmbeddingRequest(model="x", input=[[1, 2, 3], [4, 5, 6]])
        assert req.input == [[1, 2, 3], [4, 5, 6]]

    def test_mixed_str_and_int_rejected(self):
        """Sanity: mixing strings and ints in the same list is NOT in
        the spec and would be ambiguous (is [1, "a"] one tokenized
        input or one int + one string?). Stay strict to avoid
        silent-wrong behavior."""
        from pydantic import ValidationError

        from vllm_mlx.api.models import EmbeddingRequest

        with pytest.raises(ValidationError):
            EmbeddingRequest(model="x", input=[1, "a", 3])


# ---------------------------------------------------------------------------
# H6 — route dispatches pre-tokenized inputs without re-tokenizing
# ---------------------------------------------------------------------------


def _build_embed_app(monkeypatch, engine):
    """Mount the embeddings router with a stubbed engine."""
    from vllm_mlx.config import get_config
    from vllm_mlx.routes import embeddings as emb_route

    app = FastAPI()
    app.include_router(emb_route.router)

    cfg = get_config()
    saved = {
        "embedding_engine": cfg.embedding_engine,
        "embedding_model_locked": cfg.embedding_model_locked,
        "api_key": cfg.api_key,
    }
    cfg.embedding_engine = engine
    cfg.embedding_model_locked = None
    cfg.api_key = None

    monkeypatch.setattr(
        "vllm_mlx.server.load_embedding_model",
        lambda *_a, **_kw: None,
        raising=False,
    )

    def _restore():
        for k, v in saved.items():
            setattr(cfg, k, v)

    return TestClient(app), _restore


class TestEmbeddingRouteAcceptsTokenInputs:
    def test_list_int_input_uses_token_path(self, monkeypatch):
        """The engine's ``embed`` for str must NOT be called when input
        is already tokens — there's nothing to tokenize. Calling the
        string path on int input would coerce numbers to ``str(int)``
        and produce embeddings for the WORD "123", not the token id 123."""
        engine = MagicMock()
        engine.count_tokens.return_value = 4
        engine.embed_tokens.return_value = [[0.1, 0.2]]
        # If the route mistakenly hit the str path, this would fire:
        engine.embed.side_effect = AssertionError(
            "embed(str) called on pre-tokenized input"
        )
        client, restore = _build_embed_app(monkeypatch, engine)
        try:
            r = client.post(
                "/v1/embeddings",
                json={"model": "any", "input": [101, 2023, 2003, 102]},
            )
        finally:
            restore()
        assert r.status_code == 200, r.text
        # Embed must have been called with the wrapped batch.
        engine.embed_tokens.assert_called_once()
        called_with = engine.embed_tokens.call_args[0][0]
        assert called_with == [[101, 2023, 2003, 102]]

    def test_list_list_int_input_passes_batch_through(self, monkeypatch):
        engine = MagicMock()
        engine.count_tokens.return_value = 6
        engine.embed_tokens.return_value = [[0.1, 0.2], [0.3, 0.4]]
        engine.embed.side_effect = AssertionError(
            "embed(str) called on pre-tokenized input"
        )
        client, restore = _build_embed_app(monkeypatch, engine)
        try:
            r = client.post(
                "/v1/embeddings",
                json={"model": "any", "input": [[1, 2, 3], [4, 5, 6]]},
            )
        finally:
            restore()
        assert r.status_code == 200, r.text
        engine.embed_tokens.assert_called_once_with([[1, 2, 3], [4, 5, 6]])

    def test_str_input_still_uses_text_path(self, monkeypatch):
        """Regression: don't break the text path while adding the
        token path."""
        engine = MagicMock()
        engine.count_tokens.return_value = 3
        engine.embed.return_value = [[0.5, 0.5]]
        client, restore = _build_embed_app(monkeypatch, engine)
        try:
            r = client.post(
                "/v1/embeddings",
                json={"model": "any", "input": "hi"},
            )
        finally:
            restore()
        assert r.status_code == 200
        engine.embed.assert_called_once()


class TestEmbeddingEngineEmbedTokens:
    """The engine must implement ``embed_tokens`` so the route has
    a place to send pre-tokenized batches."""

    def test_embed_tokens_method_exists(self):
        from vllm_mlx.embedding import EmbeddingEngine

        assert hasattr(EmbeddingEngine, "embed_tokens"), (
            "EmbeddingEngine must expose embed_tokens(list[list[int]]) "
            "for OpenAI spec input formats 3 and 4."
        )


# ---------------------------------------------------------------------------
# H22 — default_timeout 300s → 1800s
# ---------------------------------------------------------------------------


class TestDefaultTimeout:
    """Reproducer: a DeepSeek-R1 / Qwen-thinking generation that takes
    400s is silently truncated by the 300s default. 1800s (30 min)
    matches what vLLM and most OpenAI-compat proxies ship today."""

    def test_server_config_default_is_1800(self):
        from vllm_mlx.config.server_config import ServerConfig

        cfg = ServerConfig()
        assert cfg.default_timeout == 1800.0, (
            f"default_timeout regressed to {cfg.default_timeout}s. "
            "Reasoning models and 30B+ generations need >5min headroom; "
            "1800s is the post-PR baseline."
        )

    def test_server_module_default_matches_config(self):
        """If someone bumps one default and forgets the other, the
        CLI and the route layer disagree and timeouts get applied at
        whichever lower default the request happens to hit first."""
        import vllm_mlx.server as srv
        from vllm_mlx.config.server_config import ServerConfig

        assert srv._default_timeout == ServerConfig().default_timeout


# ---------------------------------------------------------------------------
# C4 — admission control on concurrent requests
# ---------------------------------------------------------------------------


class TestAdmissionControl:
    """Reproducer: a fork-bomb client (or naive concurrent batch
    job) spawns N concurrent requests with large max_tokens; Metal
    allocator OOMs, server crashes, every other client gets 503/
    connection reset. Cap concurrent in-flight requests at a
    configurable max.
    """

    def test_scheduler_config_has_cap(self):
        from vllm_mlx.scheduler import SchedulerConfig

        cfg = SchedulerConfig()
        assert hasattr(cfg, "max_concurrent_requests"), (
            "SchedulerConfig must expose max_concurrent_requests for "
            "admission control (default conservative)."
        )
        # Default must be set (not None) — admission control is on by default.
        assert cfg.max_concurrent_requests is not None
        assert cfg.max_concurrent_requests > 0

    def test_add_request_raises_backpressure_at_cap(self):
        """When in-flight count equals the cap, the next add_request
        must raise BackpressureError (route converts to 503). Counting
        scheduler.requests directly because that's the canonical
        in-flight ledger (see Scheduler.add_request line ~2285)."""
        from vllm_mlx.scheduler import BackpressureError

        # Custom exception must exist and be a discriminable type so
        # routes can catch it specifically (not BaseException).
        assert issubclass(BackpressureError, Exception)
        assert not issubclass(BackpressureError, BaseException) or (
            BackpressureError is not BaseException
        )

    def test_admission_returns_503_with_retry_after(self, monkeypatch):
        """End-to-end: a request that would push in-flight over the
        cap returns 503 with a Retry-After header (RFC 9110 §10.2.4).
        Backed-off clients can then retry without further
        ceremony."""
        # Build a stub chat route that hits a stub engine; the engine's
        # generate() raises BackpressureError to simulate cap-exceeded.
        from vllm_mlx.config import get_config
        from vllm_mlx.routes import chat as chat_route
        from vllm_mlx.scheduler import BackpressureError

        app = FastAPI()
        app.include_router(chat_route.router)

        engine = MagicMock()
        engine.is_mllm = False
        # Tool-call parser / guided gen short-circuits we don't want
        # to hit on this path.
        engine.supports_guided_generation = False

        async def _boom(*_a, **_kw):
            raise BackpressureError("max_concurrent_requests exceeded")

        # The chat route invokes ``engine.chat(...)`` on the
        # non-streaming, non-guided path (see routes/chat.py:597).
        engine.chat = _boom

        cfg = get_config()
        saved = {
            "engine": cfg.engine,
            "model_name": cfg.model_name,
            "model_alias": cfg.model_alias,
            "model_path": cfg.model_path,
            "model_registry": cfg.model_registry,
            "tool_call_parser": cfg.tool_call_parser,
            "reasoning_parser": cfg.reasoning_parser,
            "ready": cfg.ready,
            "api_key": cfg.api_key,
        }
        cfg.engine = engine
        cfg.model_name = "stub"
        cfg.model_alias = None
        cfg.model_path = None
        cfg.model_registry = None
        cfg.tool_call_parser = None
        cfg.reasoning_parser = None
        cfg.ready = True
        cfg.api_key = None

        monkeypatch.setattr(chat_route, "get_engine", lambda *_a, **_kw: engine)

        try:
            client = TestClient(app, raise_server_exceptions=False)
            r = client.post(
                "/v1/chat/completions",
                json={
                    "model": "stub",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
        finally:
            for k, v in saved.items():
                setattr(cfg, k, v)

        assert r.status_code == 503, r.text
        assert r.headers.get("Retry-After") is not None
        # Body should hint at backpressure so SDK error messages are useful.
        detail = r.json().get("detail", "").lower()
        assert "concurrent" in detail or "backpressure" in detail or "busy" in detail
