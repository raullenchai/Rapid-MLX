# SPDX-License-Identifier: Apache-2.0
"""R11-G / H-13 — ``/v1/models`` visibility of the configured embedding model.

H-13 (Bo r11 carry from R8-H3): boot
``rapid-mlx serve --embedding-model mlx-community/Qwen3-Embedding-0.6B-4bit``
and the response from ``GET /v1/models`` only listed the *chat* model.
The embedding model id was missing, which broke
``client.models.list()`` auto-discovery for LangChain, LlamaIndex, and
openai-python — those clients enumerate ``/v1/models`` to find
``capabilities`` containing ``"embedding"`` before they will route a
``client.embeddings.create()`` call.

This file pins the discovery contract directly against
``vllm_mlx.routes.models``. The broader H-08+H-09+H-13 regression net
lives in :mod:`tests.test_embeddings_extra_guard` (the same
``ModelsListEmbeddingCapability`` class); this dedicated file exists
because the task spec named ``tests/test_routes_models.py::
test_embedding_model_in_models_list`` explicitly and discovery clients
typically grep for the route file name when wiring up a new transport.

A regression here is a wire-shape break, not a unit-level bug, so we
mount a real :class:`FastAPI` app with the ``vllm_mlx.routes.models``
router and inspect the JSON response — same shape rapid-desktop and
the openai client see in production.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _mount_models_app(*, embedding_model_locked: str | None):
    """Mount a TestClient on the models router with a stubbed config.

    Saves + restores both :class:`ServerConfig` fields AND the
    ``vllm_mlx.server._embedding_model_locked`` global so a test
    interleave can't bleed state across cases.
    """
    from vllm_mlx.config import get_config
    from vllm_mlx.routes import models as models_route

    app = FastAPI()
    app.include_router(models_route.router)

    cfg = get_config()
    saved = {
        k: getattr(cfg, k, None)
        for k in (
            "model_name",
            "model_alias",
            "model_registry",
            "embedding_model_locked",
            "api_key",
        )
    }
    cfg.model_name = "mlx-community/Llama-3.2-1B-Instruct-4bit"
    cfg.model_alias = "llama-3.2-1b-4bit"
    cfg.model_registry = None
    cfg.embedding_model_locked = embedding_model_locked
    cfg.api_key = None

    import vllm_mlx.server as srv

    saved_srv = {"_embedding_model_locked": srv._embedding_model_locked}
    srv._embedding_model_locked = embedding_model_locked

    def _restore() -> None:
        for k, v in saved.items():
            setattr(cfg, k, v)
        for k, v in saved_srv.items():
            setattr(srv, k, v)

    return TestClient(app), _restore


def test_embedding_model_in_models_list():
    """H-13: when an embedding model is locked at boot, it MUST appear
    in ``/v1/models`` alongside the chat model with
    ``capabilities=["embedding"]``.

    Pre-fix the listing only enumerated ``cfg.model_name`` /
    ``cfg.model_alias`` — the dedicated embedding model id was
    invisible. Discovery clients (``client.models.list()`` in
    langchain / llamaindex / openai-python) iterate this listing and
    pick the embedding id by ``"embedding" in capabilities``; without
    the entry the clients fell back to substring-matching the model
    name, which fails on every aliased id.
    """
    embed_id = "mlx-community/Qwen3-Embedding-0.6B-4bit"
    client, restore = _mount_models_app(embedding_model_locked=embed_id)
    try:
        r = client.get("/v1/models")
    finally:
        restore()

    assert r.status_code == 200, r.text
    body = r.json()

    ids = [entry["id"] for entry in body["data"]]
    # Both cards must be present — the chat model AND the embedding model.
    assert "mlx-community/Llama-3.2-1B-Instruct-4bit" in ids, (
        "Chat model went missing from /v1/models after adding the embedding "
        "entry. Listing regression — H-13 fix must not drop the chat card."
    )
    assert embed_id in ids, (
        f"Configured embedding model {embed_id!r} missing from /v1/models. "
        "client.models.list() auto-discovery is broken for "
        "langchain / llamaindex / openai-python."
    )

    # Exactly one entry carries the embedding capability — chat models
    # MUST NOT silently advertise it. The H-09 route guard would 503
    # ``/v1/embeddings`` for non-locked ids, so claiming a chat card is
    # embedding-capable on the listing would mislead the desktop client.
    embedding_entries = [
        entry for entry in body["data"] if "embedding" in entry.get("capabilities", [])
    ]
    assert len(embedding_entries) == 1, embedding_entries
    embed_card = embedding_entries[0]
    assert embed_card["id"] == embed_id
    assert embed_card["capabilities"] == ["embedding"]
    # The modality on the wire is "text" — embedding models accept
    # text input; the ``capabilities`` tag is what distinguishes the
    # lane (F-D01 cosmetic).
    assert embed_card["modality"] == "text"
    # ``object`` field is OpenAI-canonical "model" — clients that
    # validate the response shape against the OpenAI spec rely on it.
    assert embed_card["object"] == "model"


def test_no_embedding_model_no_embedding_card():
    """Sanity: without ``--embedding-model``, no card carries the
    embedding capability tag. The H-09 route guard already 503s
    ``/v1/embeddings``, so claiming the chat card is embedding-capable
    here would mislead discovery clients into routing real traffic at
    a path that will only error."""
    client, restore = _mount_models_app(embedding_model_locked=None)
    try:
        r = client.get("/v1/models")
    finally:
        restore()
    assert r.status_code == 200, r.text
    body = r.json()
    for entry in body["data"]:
        caps = entry.get("capabilities", [])
        assert "embedding" not in caps, (
            f"Model {entry['id']} advertises embedding capability but no "
            "embedding model is configured — /v1/embeddings would 503."
        )


def test_retrieve_embedding_model_by_path_id():
    """``GET /v1/models/{embed_id}`` must resolve a slash-containing
    HF id directly — every other rapid-mlx endpoint accepts the bare
    HF id, this one should too.

    desktop / rapid-desktop hydrates per-model state from this path
    (R10-D contract); a future refactor that breaks slash handling
    would silently kneecap the per-model UI without touching
    ``/v1/models``.
    """
    embed_id = "mlx-community/Qwen3-Embedding-0.6B-4bit"
    client, restore = _mount_models_app(embedding_model_locked=embed_id)
    try:
        r = client.get(f"/v1/models/{embed_id}")
    finally:
        restore()
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["id"] == embed_id
    assert "embedding" in body["capabilities"]


if __name__ == "__main__":  # pragma: no cover — convenience only
    pytest.main([__file__, "-v"])
