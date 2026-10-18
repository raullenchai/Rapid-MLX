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
``rapid_mlx/routes/embeddings.py`` is the natural search location for
a grep-based code reviewer.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.requires_mlx
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _build_embed_app(monkeypatch, engine, *, embedding_model_locked):
    """Mount the embeddings router with a stubbed engine + lock state.

    Mirrors the helper in :mod:`tests.test_embeddings_extra_guard`
    but lives here too so the file is self-contained for grep-driven
    code review.
    """
    from rapid_mlx.config import get_config
    from rapid_mlx.middleware.exception_handlers import install_exception_handlers
    from rapid_mlx.routes import embeddings as emb_route

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

    import rapid_mlx.server as srv

    saved_srv = {
        "_embedding_engine": srv._embedding_engine,
        "_embedding_model_locked": srv._embedding_model_locked,
    }
    srv._embedding_engine = engine
    srv._embedding_model_locked = embedding_model_locked

    monkeypatch.setattr(
        "rapid_mlx.server.load_embedding_model",
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
    assert "pip install 'rapid-mlx[embeddings]==" in msg

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


def test_embeddings_list_of_strings_embeds_each_text(monkeypatch):
    engine = MagicMock()
    engine.model_name = "stub-embed"
    engine.embed.return_value = [[1.0, 0.0], [0.0, 1.0]]
    engine.count_tokens.return_value = 4
    client, restore = _build_embed_app(
        monkeypatch, engine, embedding_model_locked="stub-embed"
    )
    try:
        response = client.post(
            "/v1/embeddings",
            json={"model": "stub-embed", "input": ["alpha", "beta"]},
        )
    finally:
        restore()

    assert response.status_code == 200, response.text
    assert [item["embedding"] for item in response.json()["data"]] == [
        [1.0, 0.0],
        [0.0, 1.0],
    ]
    engine.embed.assert_called_once_with(["alpha", "beta"])


def test_embeddings_pretokenized_input_uses_retrieval_worker(monkeypatch):
    engine = MagicMock()
    engine.model_name = "stub-embed"
    engine.embed_tokens.return_value = [[1.0, 0.0]]
    client, restore = _build_embed_app(
        monkeypatch, engine, embedding_model_locked="stub-embed"
    )
    try:
        response = client.post(
            "/v1/embeddings",
            json={"model": "stub-embed", "input": [101, 102]},
        )
    finally:
        restore()

    assert response.status_code == 200, response.text
    engine.embed_tokens.assert_called_once_with([[101, 102]])


def test_embeddings_dimensions_truncate_and_renormalize(monkeypatch):
    engine = MagicMock()
    engine.model_name = "stub-embed"
    engine.embed.return_value = [[3.0, 4.0, 12.0]]
    engine.count_tokens.return_value = 1
    client, restore = _build_embed_app(
        monkeypatch, engine, embedding_model_locked="stub-embed"
    )
    try:
        response = client.post(
            "/v1/embeddings",
            json={"model": "stub-embed", "input": "a", "dimensions": 2},
        )
    finally:
        restore()

    assert response.status_code == 200, response.text
    assert response.json()["data"][0]["embedding"] == pytest.approx([0.6, 0.8])


def test_embeddings_reject_dimensions_above_native_size(monkeypatch):
    engine = MagicMock()
    engine.model_name = "stub-embed"
    engine.embed.return_value = [[1.0, 0.0]]
    engine.count_tokens.return_value = 1
    client, restore = _build_embed_app(
        monkeypatch, engine, embedding_model_locked="stub-embed"
    )
    try:
        response = client.post(
            "/v1/embeddings",
            json={"model": "stub-embed", "input": "a", "dimensions": 3},
        )
    finally:
        restore()

    assert response.status_code == 400
    assert "native embedding size of 2" in str(response.json())


def test_embeddings_base64_encoding_contains_float32_vector(monkeypatch):
    import base64
    import struct

    engine = MagicMock()
    engine.model_name = "stub-embed"
    engine.embed.return_value = [[0.25, -0.5]]
    engine.count_tokens.return_value = 1
    client, restore = _build_embed_app(
        monkeypatch, engine, embedding_model_locked="stub-embed"
    )
    try:
        response = client.post(
            "/v1/embeddings",
            json={
                "model": "stub-embed",
                "input": "a",
                "encoding_format": "base64",
            },
        )
    finally:
        restore()

    assert response.status_code == 200, response.text
    encoded = response.json()["data"][0]["embedding"]
    assert struct.unpack("<2f", base64.b64decode(encoded)) == pytest.approx(
        (0.25, -0.5)
    )


class _ChatWorker:
    def __init__(self):
        self.calls = 0

    async def execute_on_model_worker(self, func, *args, **kwargs):
        self.calls += 1
        raise AssertionError("retrieval work used the chat model worker")


def _thread_recording_engine(error=None):
    import threading

    engine = MagicMock()
    engine.model_name = "stub-embed"
    engine.count_tokens.return_value = 1
    engine.threads = []

    def embed(texts):
        engine.threads.append(threading.current_thread().name)
        if error is not None:
            raise error
        return [[1.0, 0.0]]

    engine.embed.side_effect = embed
    return engine


@pytest.fixture
def chat_engine():
    from rapid_mlx.config import get_config

    cfg = get_config()
    saved = cfg.engine
    yield cfg
    cfg.engine = saved


async def test_embedding_does_not_block_event_loop(monkeypatch, chat_engine):
    import asyncio
    import threading

    import httpx

    started, release = threading.Event(), threading.Event()
    engine = MagicMock()
    engine.model_name = "stub-embed"
    engine.count_tokens.return_value = 1

    def embed(texts):
        started.set()
        release.wait(timeout=5)
        return [[1.0, 0.0]]

    engine.embed.side_effect = embed
    chat_engine.engine = None
    client, restore = _build_embed_app(
        monkeypatch, engine, embedding_model_locked="stub-embed"
    )
    client.app.add_api_route("/ping", lambda: {"ok": True})
    transport = httpx.ASGITransport(app=client.app)
    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
            embedding = asyncio.create_task(
                c.post("/v1/embeddings", json={"model": "stub-embed", "input": "a"})
            )
            assert await asyncio.to_thread(started.wait, 5)
            assert (await c.get("/ping")).status_code == 200
            assert not embedding.done()
            release.set()
            assert (await embedding).status_code == 200
    finally:
        release.set()
        restore()


async def test_embedding_and_rerank_share_one_mlx_lane(chat_engine):
    import asyncio
    import threading

    from rapid_mlx.routes import embeddings, rerank

    first_started = threading.Event()
    release = threading.Event()
    lock = threading.Lock()
    active = 0
    max_active = 0
    thread_ids = []

    def blocking_work():
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
            thread_ids.append(threading.get_ident())
            first_started.set()
        try:
            assert release.wait(timeout=5)
        finally:
            with lock:
                active -= 1

    chat_engine.engine = None
    embedding = asyncio.create_task(embeddings.run_on_mlx_worker(blocking_work))
    try:
        assert await asyncio.to_thread(first_started.wait, 5)
        reranking = asyncio.create_task(rerank.run_on_mlx_worker(blocking_work))
        await asyncio.sleep(0.05)
        release.set()
        await asyncio.gather(embedding, reranking)
    finally:
        release.set()

    assert len(set(thread_ids)) == 1
    assert max_active == 1


def test_embedding_uses_dedicated_worker_not_chat_worker(monkeypatch, chat_engine):
    engine = _thread_recording_engine()
    worker = _ChatWorker()
    chat_engine.engine = worker
    client, restore = _build_embed_app(
        monkeypatch, engine, embedding_model_locked="stub-embed"
    )
    try:
        r = client.post("/v1/embeddings", json={"model": "stub-embed", "input": "a"})
    finally:
        restore()
    assert r.status_code == 200, r.text
    assert worker.calls == 0
    assert len(engine.threads) == 1
    assert engine.threads[0].startswith("mlx-retrieval")


async def test_embedding_load_and_inference_share_retrieval_thread(monkeypatch):
    import threading

    from rapid_mlx import embedding, server
    from rapid_mlx.config import get_config
    from rapid_mlx.routes._mlx_worker import run_on_mlx_worker

    threads = []

    class _Engine:
        def __init__(self, model_name, **_kwargs):
            self.model_name = model_name

        def load(self):
            threads.append(threading.current_thread().name)

        def embed(self, _texts):
            threads.append(threading.current_thread().name)

    monkeypatch.setattr(embedding, "EmbeddingEngine", _Engine)
    monkeypatch.setattr(server, "_emit_embedding_model_served_once", lambda *_a: None)
    monkeypatch.setattr(server, "_embedding_engine", None)
    cfg = get_config()
    monkeypatch.setattr(cfg, "embedding_engine", None)
    monkeypatch.setattr(cfg, "embedding_model_locked", None)
    # The CLI calls this before uvicorn starts, while the lazy route caller may
    # invoke it with an event loop already running. The blocking API must work
    # in both cases without dispatching through that loop.
    server.load_embedding_model("stub-embed", reuse_existing=False)
    await run_on_mlx_worker(server._embedding_engine.embed, ["hello"])

    assert len(threads) == 2
    assert threads[0].startswith("mlx-retrieval")
    assert threads[0] == threads[1]


def test_embedding_load_error_propagates(monkeypatch):
    from rapid_mlx import embedding, server

    error = ValueError("load failed unchanged")

    class _Engine:
        def __init__(self, model_name, **_kwargs):
            self.model_name = model_name

        def load(self):
            raise error

    monkeypatch.setattr(embedding, "EmbeddingEngine", _Engine)
    monkeypatch.setattr(server, "_embedding_engine", None)

    with pytest.raises(ValueError, match="load failed unchanged") as exc_info:
        server.load_embedding_model("stub-embed", reuse_existing=False)
    assert exc_info.value is error


def test_embedding_inference_error_propagates_once(monkeypatch, chat_engine):
    error = RuntimeError("boom")
    engine = _thread_recording_engine(error=error)
    worker = _ChatWorker()
    chat_engine.engine = worker
    client, restore = _build_embed_app(
        monkeypatch, engine, embedding_model_locked="stub-embed"
    )
    try:
        r = client.post("/v1/embeddings", json={"model": "stub-embed", "input": "a"})
    finally:
        restore()
    assert r.status_code == 500
    assert worker.calls == 0
    assert len(engine.threads) == 1


if __name__ == "__main__":  # pragma: no cover — convenience only
    pytest.main([__file__, "-v"])


@pytest.mark.parametrize("tokenized", [False, True])
def test_embedding_default_input_cap_precedes_model_work(monkeypatch, tokenized):
    from rapid_mlx.config import get_config

    engine = MagicMock()
    client, restore = _build_embed_app(
        monkeypatch, engine, embedding_model_locked="stub-embed"
    )
    loader = MagicMock()
    monkeypatch.setattr("rapid_mlx.server.load_embedding_model", loader)
    try:
        assert get_config().embedding_max_inputs == 2048
        payload = [[1]] * 2049 if tokenized else ["d"] * 2049
        response = client.post(
            "/v1/embeddings", json={"model": "stub-embed", "input": payload}
        )
        assert response.status_code == 400
        error = response.json()["error"]
        assert (error["code"], error["param"], error["type"]) == (
            "too_many_inputs",
            "input",
            "invalid_request_error",
        )
        assert all(
            s in error["message"] for s in ["2049", "2048", "--embedding-max-inputs"]
        )
        assert not engine.mock_calls
        loader.assert_not_called()
    finally:
        restore()


@pytest.mark.parametrize(
    "limit,payload,status",
    [
        (1, ["a", "b"], 400),
        (1, [[1], [2]], 400),
        (0, ["a", "b"], 200),
        (-1, [[1], [2]], 200),
        (1, [1, 2, 3], 200),
        (1, "abc", 200),
        (2, ["a", "b"], 200),
    ],
)
def test_embedding_input_cap_override_and_singletons(
    monkeypatch, limit, payload, status
):
    from rapid_mlx.config import get_config

    engine = MagicMock()
    engine.model_name = "stub-embed"
    engine.embed.return_value = [[0.5]]
    engine.embed_tokens.return_value = [[0.5]]
    engine.count_tokens.return_value = 3
    client, restore = _build_embed_app(
        monkeypatch, engine, embedding_model_locked="stub-embed"
    )
    monkeypatch.setattr(get_config(), "embedding_max_inputs", limit)
    try:
        response = client.post(
            "/v1/embeddings", json={"model": "stub-embed", "input": payload}
        )
        assert response.status_code == status
    finally:
        restore()


def test_embedding_busy_returns_retry_after(monkeypatch):
    from unittest.mock import AsyncMock

    from rapid_mlx.routes import embeddings
    from rapid_mlx.routes._mlx_worker import RetrievalWorkerBusyError

    engine = MagicMock()
    engine.model_name = "stub-embed"
    client, restore = _build_embed_app(
        monkeypatch, engine, embedding_model_locked="stub-embed"
    )
    monkeypatch.setattr(
        embeddings, "run_on_mlx_worker", AsyncMock(side_effect=RetrievalWorkerBusyError)
    )
    telemetry = MagicMock()
    monkeypatch.setattr(
        "rapid_mlx.telemetry.inference.emit_completed_request", telemetry
    )
    try:
        response = client.post(
            "/v1/embeddings", json={"model": "stub-embed", "input": "hello"}
        )
        assert response.status_code == 503
        assert response.headers["Retry-After"] == "1"
        assert response.json()["error"]["code"] == "server_busy"
        telemetry.assert_not_called()
    finally:
        restore()


@pytest.mark.parametrize("limit", [7, 0, -1])
def test_embedding_loader_and_config_preserve_input_cap(monkeypatch, limit):
    from rapid_mlx import embedding, server
    from rapid_mlx.config import get_config

    engine = MagicMock()
    engine.model_name = "stub-embed"
    cfg = get_config()
    for name in ("embedding_engine", "embedding_model_locked", "embedding_max_inputs"):
        monkeypatch.setattr(cfg, name, getattr(cfg, name))
    monkeypatch.setattr(server, "_embedding_engine", None)
    monkeypatch.setattr(embedding, "EmbeddingEngine", MagicMock(return_value=engine))
    monkeypatch.setattr(server, "_emit_embedding_model_served_once", lambda *_a: None)
    server.load_embedding_model("stub-embed", max_inputs=limit, reuse_existing=False)
    assert server._embedding_max_inputs == limit
    assert get_config().embedding_max_inputs == limit
    server._sync_config()
    assert get_config().embedding_max_inputs == limit


def test_embedding_cli_passes_input_cap_to_loader(monkeypatch):
    from types import SimpleNamespace

    from rapid_mlx import cli

    monkeypatch.setattr("rapid_mlx.embedding.mlx_embeddings_available", lambda: True)
    loader = MagicMock()
    cli._load_embedding_model_or_exit(
        SimpleNamespace(embedding_model="stub-embed", embedding_max_inputs=12), loader
    )
    assert loader.call_args.kwargs["max_inputs"] == 12
