# SPDX-License-Identifier: Apache-2.0
"""No-MLX contracts for explicit configured-primary activation."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm_mlx.config import get_config
from vllm_mlx.routes.health import admin_router


@pytest.fixture(autouse=True)
def restore_server_config() -> Iterator[None]:
    cfg = get_config()
    fields = (
        "api_key",
        "draining",
        "engine",
        "model_name",
        "primary_model_lifecycle",
        "ready",
    )
    original = {field: getattr(cfg, field) for field in fields}
    yield
    for field, value in original.items():
        setattr(cfg, field, value)


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(admin_router)
    return TestClient(app)


def _configure(**updates) -> None:
    cfg = get_config()
    defaults = {
        "api_key": None,
        "draining": False,
        "engine": object(),
        "model_name": "test-model",
        "primary_model_lifecycle": None,
        "ready": True,
    }
    defaults.update(updates)
    for field, value in defaults.items():
        setattr(cfg, field, value)


class _Lifecycle:
    def __init__(self, *, state: str = "ready", loaded: bool = False) -> None:
        self.state = state
        self.loaded = loaded
        self.owners = 0

    def acquire_request(self) -> None:
        self.owners += 1

    async def ensure_loaded(self) -> None:
        self.loaded = True

    def snapshot(self) -> dict[str, object]:
        return {"state": self.state, "model_loaded": self.loaded}

    def release_request(self) -> None:
        self.owners -= 1


def test_activate_loads_lazy_primary_and_requires_authentication():
    lifecycle = _Lifecycle()
    _configure(api_key="test-secret", primary_model_lifecycle=lifecycle)
    client = _client()

    assert client.post("/v1/models/activate").status_code == 401
    response = client.post(
        "/v1/models/activate",
        headers={"Authorization": "Bearer test-secret"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "model": "test-model",
        "state": "ready",
        "model_loaded": True,
    }
    assert lifecycle.owners == 0


def test_activate_eager_primary_returns_ready_without_lifecycle():
    _configure()
    response = _client().post("/v1/models/activate")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


class _FailingLifecycle(_Lifecycle):
    async def ensure_loaded(self) -> None:
        raise RuntimeError("/private/model/path must not leak")


def test_activate_sanitizes_load_failure_and_releases_owner():
    lifecycle = _FailingLifecycle()
    _configure(primary_model_lifecycle=lifecycle)
    response = _client().post("/v1/models/activate")
    assert response.status_code == 503
    assert "/private/model/path" not in response.text
    assert lifecycle.owners == 0


@pytest.mark.parametrize(
    ("updates", "expected_detail"),
    [
        ({"ready": False}, "Service is not accepting work"),
        ({"draining": True}, "Service is not accepting work"),
        ({"engine": None}, "Primary model unavailable"),
    ],
)
def test_activate_rejects_unavailable_service(updates, expected_detail):
    _configure(**updates)
    response = _client().post("/v1/models/activate")
    assert response.status_code == 503
    assert response.json()["detail"] == expected_detail


def test_activate_requires_loaded_ready_snapshot():
    lifecycle = _Lifecycle(state="standby")
    _configure(primary_model_lifecycle=lifecycle)
    response = _client().post("/v1/models/activate")
    assert response.status_code == 503
    assert "ready state" in response.json()["detail"]
