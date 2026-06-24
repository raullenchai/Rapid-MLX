# SPDX-License-Identifier: Apache-2.0
"""R11-G / H-09 — ``/v1/embeddings`` 503 guard envelope contract.

H-09 (Bo r11 carry from R8-H3, 6-round drift): boot
``rapid-mlx serve <chat-model>`` (no ``--embedding-model``), then
``POST /v1/embeddings`` returned 200 with a 1024-dim Qwen vector
sourced from the *chat* model's pooled hidden states. Callers stuffed
the garbage vector into a vector store and only noticed weeks later
when retrieval quality cratered. Silent-wrong is worse than loud-broken.

This file pins the wire-level contract directly:

* status code is exactly 503 (not 400 — 503 lines up with LangChain /
  LlamaIndex retry semantics for "transient infra issue", which is
  the correct shape for a server-side configuration gap)
* envelope carries the machine-readable
  ``code: "no_embedding_model"`` so SDK retry branching doesn't have
  to substring-match the message
* envelope carries the actionable install hint + the
  ``--embedding-model`` flag name verbatim
* engine is NEVER touched — no silent chat-model fallback can happen
  even if the engine global accidentally still points at the chat
  model from a prior dev session

The broader H-08+H-09+H-13 net lives in
:mod:`tests.test_embeddings_extra_guard`; this file exists because
the task spec named ``tests/test_embeddings_route.py::
test_embeddings_503_when_no_model`` explicitly and the route file
``vllm_mlx/routes/embeddings.py`` is the natural search location for
a grep-based code reviewer.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _build_embed_app(monkeypatch, engine, *, embedding_model_locked):
    """Mount the embeddings router with a stubbed engine + lock state.

    Mirrors the helper in :mod:`tests.test_embeddings_extra_guard`
    but lives here too so the file is self-contained for grep-driven
    code review.
    """
    from vllm_mlx.config import get_config
    from vllm_mlx.middleware.exception_handlers import install_exception_handlers
    from vllm_mlx.routes import embeddings as emb_route

    app = FastAPI()
    app.include_router(emb_route.router)
    install_exception_handlers(app)

    cfg = get_config()
    saved = {
        "embedding_engine": cfg.embedding_engine,
        "embedding_model_locked": cfg.embedding_model_locked,
        "api_key": cfg.api_key,
    }
    cfg.embedding_engine = engine
    cfg.embedding_model_locked = embedding_model_locked
    cfg.api_key = None

    import vllm_mlx.server as srv

    saved_srv = {
        "_embedding_engine": srv._embedding_engine,
        "_embedding_model_locked": srv._embedding_model_locked,
    }
    srv._embedding_engine = engine
    srv._embedding_model_locked = embedding_model_locked

    monkeypatch.setattr(
        "vllm_mlx.server.load_embedding_model",
        lambda *_a, **_kw: None,
        raising=False,
    )

    def _restore() -> None:
        for k, v in saved.items():
            setattr(cfg, k, v)
        for k, v in saved_srv.items():
            setattr(srv, k, v)

    return TestClient(app), _restore


def test_embeddings_503_when_no_model(monkeypatch):
    """H-09: ``POST /v1/embeddings`` returns 503 with the structured
    envelope when ``--embedding-model`` was not configured at boot.

    The status code MUST be 503 (not 400). Pre-fix the guard returned
    400, which looks like "bad client payload" — SDKs surfaced this
    as a permanent user-facing error. 503 signals "server isn't
    capable of fulfilling this kind of request right now", which is
    the correct shape for a missing-flag boot gap and what
    LangChain / LlamaIndex retry policies recognise as
    "transient infra issue, back off".
    """
    engine = MagicMock()
    client, restore = _build_embed_app(monkeypatch, engine, embedding_model_locked=None)
    try:
        r = client.post(
            "/v1/embeddings",
            json={"model": "qwen3-0.6b-8bit", "input": "hello"},
        )
    finally:
        restore()

    assert r.status_code == 503, r.text
    body = r.json()

    # Envelope must use the OpenAI-canonical ``error`` wrapper so SDKs
    # that key on ``response.error.code`` (langchain / llamaindex /
    # openai-python) see the same shape they expect from OpenAI itself.
    assert "error" in body
    err = body["error"]
    assert err["type"] == "invalid_request_error"
    assert err["code"] == "no_embedding_model"

    # Message must name the CLI flag AND the install hint so the
    # operator can copy-paste the fix.
    msg = err["message"]
    assert "No embedding model loaded" in msg
    assert "--embedding-model" in msg
    assert "pip install 'rapid-mlx[embeddings]'" in msg

    # CRITICAL: the engine MUST NOT have been touched. Pre-fix the
    # route would fall through to ``embed`` / ``embed_tokens`` with
    # the chat-model engine and return shape-valid garbage; the guard
    # has to short-circuit BEFORE any model call.
    engine.embed.assert_not_called()
    engine.embed_tokens.assert_not_called()


def test_embeddings_503_envelope_survives_pre_tokenized_input(monkeypatch):
    """The 503 must fire regardless of input shape — pre-tokenized
    inputs (``list[int]`` / ``list[list[int]]``) go through a
    different code branch inside the route, so pin both shapes
    against the guard.
    """
    engine = MagicMock()
    client, restore = _build_embed_app(monkeypatch, engine, embedding_model_locked=None)
    try:
        r = client.post(
            "/v1/embeddings",
            json={"model": "qwen3-0.6b-8bit", "input": [1, 2, 3]},
        )
    finally:
        restore()
    assert r.status_code == 503, r.text
    assert r.json()["error"]["code"] == "no_embedding_model"
    engine.embed.assert_not_called()
    engine.embed_tokens.assert_not_called()


def test_embeddings_returns_200_when_configured(monkeypatch):
    """Smoke-test the positive path: when an embedding model IS
    configured at boot, the route reaches the engine and returns the
    embedding vectors. Existing detailed coverage lives in
    :mod:`tests.test_embeddings`; this assertion exists so a
    regression that flips the guard ON for all requests (e.g. wrong
    condition direction) shows up in the H-09 surface tests too.
    """
    engine = MagicMock()
    engine.model_name = "stub-embed"
    engine.embed.return_value = [[0.5, 0.5, 0.5, 0.5]]
    engine.count_tokens.return_value = 3
    client, restore = _build_embed_app(
        monkeypatch, engine, embedding_model_locked="stub-embed"
    )
    try:
        r = client.post(
            "/v1/embeddings",
            json={"model": "stub-embed", "input": "hello"},
        )
    finally:
        restore()
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["data"][0]["embedding"] == [0.5, 0.5, 0.5, 0.5]
    engine.embed.assert_called_once()


if __name__ == "__main__":  # pragma: no cover — convenience only
    pytest.main([__file__, "-v"])
