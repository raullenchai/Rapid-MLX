# SPDX-License-Identifier: Apache-2.0
"""Lightweight coverage for speculative server probes and authentication."""

import builtins
from concurrent.futures import Future

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rapid_mlx.speculative.health import install_health_probes


@pytest.mark.parametrize(
    ("details", "message", "state"),
    [
        ({"status": "loading"}, "model loading", "loading"),
        ({"ready": False}, "model loading", "loading"),
        ({"status": "error"}, "model failed to load", "error"),
    ],
)
def test_not_ready_probe(details, message, state):
    async def healthz():
        return details

    app = FastAPI()
    install_health_probes(app, "test-model", healthz)
    client = TestClient(app)

    expected = {
        "error": {
            "message": message,
            "type": "service_unavailable",
            "code": "model_not_ready",
        },
        "ready": False,
        "state": state,
        "model_loaded": False,
    }
    for path in ("/health/ready", "/readyz"):
        response = client.get(path)
        assert response.status_code == 503
        assert response.json() == expected
    # Liveness is process-only and stays up while the model loads or fails.
    assert client.get("/livez").json() == {"status": "alive"}
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "healthy"
    assert health.json()["ready"] is False
    assert health.json()["state"] == state


def test_ready_and_basic_probes():
    async def healthz():
        return {"status": "ok", "engine": "fake"}

    app = FastAPI()
    install_health_probes(app, "test-model", healthz)
    client = TestClient(app)

    for path in ("/health/ready", "/readyz"):
        response = client.get(path)
        assert response.status_code == 200
        assert response.json() == {
            "status": "ok",
            "ready": True,
            "state": "ready",
            "model_loaded": True,
            "model": "test-model",
        }
    assert client.get("/livez").json() == {"status": "alive"}
    assert client.get("/health").json() == {
        "status": "healthy",
        "engine": "fake",
        "ready": True,
        "state": "ready",
        "model_loaded": True,
        "model_name": "test-model",
        "model_type": "llm",
        "engine_type": "fake",
    }
    assert client.get("/").json() == {"status": "ok"}
    response = client.head("/")
    assert response.status_code == 200
    assert response.content == b""


def test_ddtree_probes_bypass_api_key_without_mlx(monkeypatch):
    # Fail explicitly if the app's import or request path needs MLX on Linux.
    original_import = builtins.__import__

    def without_mlx(name, *args, **kwargs):
        if name == "mlx" or name.startswith("mlx."):
            raise ImportError("MLX deliberately unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_mlx)
    from rapid_mlx.config import get_config
    from rapid_mlx.speculative.ddtree.server import _build_app

    monkeypatch.setattr(get_config(), "api_key", None)
    app = _build_app(
        runtime_future=Future(),
        served_model_name="test-model",
        default_max_tokens=64,
        cors_origins=[],
        api_key="secret",
    )
    client = TestClient(app, base_url="http://127.0.0.1")
    assert client.get("/v1/models").status_code == 401
    assert (
        client.get("/v1/models", headers={"Authorization": "Bearer secret"}).status_code
        == 200
    )
    for path in ("/health/ready", "/readyz"):
        response = client.get(path)
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "model_not_ready"
    assert client.get("/livez").status_code == 200
