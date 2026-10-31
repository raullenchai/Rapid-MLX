# SPDX-License-Identifier: Apache-2.0
"""Health and update reporting for model-free serve modes."""

import json
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rapid_mlx import cli, server, update_refresh
from rapid_mlx.config import reset_config
from rapid_mlx.middleware.probe_fastpath import _build_healthz_payload
from rapid_mlx.routes.health import probe_router

EMBED = "org/embedding"
RERANK = "org/reranker"
_real_start = update_refresh.start


@pytest.fixture
def health_client(monkeypatch):
    cfg = reset_config()
    monkeypatch.setattr(server, "_embedding_model_locked", None)
    monkeypatch.setattr(server, "_rerank_model_locked", None)
    app = FastAPI()
    app.include_router(probe_router)
    yield cfg, TestClient(app)
    reset_config()


@pytest.mark.parametrize(
    "embed,rerank,expected",
    [
        (None, RERANK, [RERANK]),
        (EMBED, None, [EMBED]),
        (EMBED, RERANK, [EMBED, RERANK]),
    ],
)
def test_retrieval_models(health_client, embed, rerank, expected):
    cfg, client = health_client
    cfg.embedding_model_locked = embed
    cfg.rerank_model_locked = rerank
    cfg.embedding_engine = object() if embed else None
    cfg.rerank_engine = object() if rerank else None
    data = client.get("/health").json()
    assert data["model_name"] == expected[0]
    assert data["models"] == expected


def test_primary_shape_unchanged(health_client):
    cfg, client = health_client
    cfg.model_name = "primary"
    cfg.embedding_model_locked = EMBED
    data = client.get("/health").json()
    assert data["model_name"] == "primary"
    assert "models" not in data


@pytest.mark.parametrize("draining", [False, True])
def test_rerank_only_probes(health_client, draining):
    cfg, client = health_client
    cfg.rerank_model_locked = RERANK
    cfg.rerank_engine = object()
    cfg.ready = True
    cfg.draining = draining

    ready = client.get("/health/ready").json()
    assert ready["model"] == RERANK
    assert "models" not in ready

    response = client.get("/healthz")
    status, body = _build_healthz_payload()
    assert response.status_code == status == (503 if draining else 200)
    assert response.json() == json.loads(body)
    assert response.json()["model_name"] == RERANK
    assert "models" not in response.json()


def test_primary_probe_models_unchanged(health_client):
    cfg, client = health_client
    cfg.model_name = "primary"
    cfg.rerank_model_locked = RERANK
    cfg.rerank_engine = object()
    cfg.ready = True
    ready = client.get("/health/ready").json()
    healthz = client.get("/healthz").json()
    _, body = _build_healthz_payload()
    assert ready["model"] == "primary"
    assert healthz["model_name"] == json.loads(body)["model_name"] == "primary"
    assert all(
        "models" not in payload for payload in (ready, healthz, json.loads(body))
    )


@pytest.mark.parametrize("mode", ["retrieval", "cua"])
def test_model_free_modes_start_checker(monkeypatch, mode):
    args = cli.build_parser().parse_args(
        ["serve", "--embedding-model", EMBED, "--port", "8123"]
        if mode == "retrieval"
        else ["serve", "--cua-only", "--api-key", "secret", "--port", "8123"]
    )
    args.log_level = "info"
    calls = []
    monkeypatch.setattr(
        "rapid_mlx._version_check.print_staleness_warning_if_any",
        lambda **kw: calls.append(("warning", kw)) or "1.2.3",
    )
    monkeypatch.setattr(
        update_refresh, "start", lambda **kw: calls.append(("start", kw))
    )
    monkeypatch.setattr(cli, "_run_uvicorn", lambda *a, **kw: calls.append(("run", {})))
    monkeypatch.setattr(cli, "_hard_exit_after_serve", lambda: None)
    monkeypatch.setattr(cli, "_load_embedding_model_or_exit", lambda *a: None)
    monkeypatch.setattr(cli, "_resolve_serve_port", lambda *a, **kw: 8123)
    try:
        if mode == "retrieval":
            cli._serve_retrieval_only_mode(args)
        else:
            cli._serve_cua_only_mode(args)
        assert calls == [
            ("warning", {"allow_non_tty": True}),
            ("start", {"announced": "1.2.3"}),
            ("run", {}),
        ]
    finally:
        reset_config()


def test_latest_version_on_retrieval_server(monkeypatch, health_client):
    cfg, client = health_client
    cfg.rerank_model_locked = RERANK
    cfg.rerank_engine = object()
    # A fake thread preserves start() semantics without network or background work.
    monkeypatch.setattr(
        update_refresh.threading,
        "Thread",
        lambda **kwargs: SimpleNamespace(start=lambda: None),
    )
    monkeypatch.setattr(update_refresh.vc, "_explicitly_disabled", lambda: False)
    monkeypatch.setattr(update_refresh.vc, "get_latest_version", lambda: "999.0.0")
    monkeypatch.setattr(update_refresh.vc, "_installed_version", lambda: "1.0.0")
    update_refresh._reset_after_fork()
    try:
        assert _real_start()
        update_refresh._refresh()
        assert client.get("/health").json()["latest_version"] == "999.0.0"
    finally:
        update_refresh._reset_after_fork()
