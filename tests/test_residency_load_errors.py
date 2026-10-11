# SPDX-License-Identifier: Apache-2.0
"""Resident-load failures must reach clients as actionable, safe JSON (#4492)."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from huggingface_hub.errors import (
    GatedRepoError,
    HfHubHTTPError,
    RepositoryNotFoundError,
)

from rapid_mlx.middleware.auth import verify_api_key
from rapid_mlx.middleware.exception_handlers import install_exception_handlers
from rapid_mlx.routes import residency


@pytest.fixture
def control_plane(monkeypatch):
    manager = SimpleNamespace(load=AsyncMock(), snapshot=lambda: {"models": []})
    monkeypatch.setattr(residency, "_manager", lambda: manager)
    app = FastAPI()
    app.include_router(residency.router)
    app.dependency_overrides[verify_api_key] = lambda: None
    install_exception_handlers(app)
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client, manager


def hub_error(kind, status):
    response = httpx.Response(
        status, request=httpx.Request("GET", "https://example.com/?token=secret")
    )
    return kind("upstream secret /Users/private/model", response=response)


@pytest.mark.parametrize(
    "kind,status,expected,marker",
    [
        (RepositoryNotFoundError, 404, 404, "not found"),
        (RepositoryNotFoundError, 401, 404, "private"),
        (GatedRepoError, 403, 403, "HF_TOKEN"),
        (HfHubHTTPError, 401, 403, "HF_TOKEN"),
        (HfHubHTTPError, 403, 403, "HF_TOKEN"),
    ],
)
@pytest.mark.parametrize("wrapped", [False, True])
def test_hub_failure_json(control_plane, kind, status, expected, marker, wrapped):
    client, manager = control_plane
    error = hub_error(kind, status)
    if wrapped:
        outer = RuntimeError("loader wrapper secret")
        outer.__cause__ = error
        error = outer
    manager.load.side_effect = error
    response = client.post("/v1/models/load", json={"model": "org/missing"})
    assert response.status_code == expected
    assert response.headers["content-type"] == "application/json"
    body = response.json()["error"]
    assert body["type"] == (
        "not_found_error" if expected == 404 else "permission_error"
    )
    assert marker in body["message"]
    assert "secret" not in response.text
    assert "/Users/private" not in response.text
    assert client.get("/v1/models/residency").status_code == 200


@pytest.mark.parametrize("override", [False, True])
def test_missing_local_path(control_plane, tmp_path, override):
    client, manager = control_plane
    missing = str(tmp_path / "absent")
    manager.load.side_effect = RuntimeError(
        "model directory not found; secret internals"
    )
    body = (
        {"model": "org/alias", "model_path": missing}
        if override
        else {"model": missing}
    )
    response = client.post("/v1/models/load", json=body)
    assert response.status_code == 404
    assert "local model path" in response.json()["error"]["message"]
    assert "does not exist" in response.text
    assert missing not in response.text
    assert "secret" not in response.text


def test_missing_local_file(control_plane, tmp_path):
    client, manager = control_plane
    manager.load.side_effect = FileNotFoundError(
        2, "missing", str(tmp_path / "config.json")
    )
    response = client.post("/v1/models/load", json={"model": str(tmp_path)})
    assert response.status_code == 400
    assert "config.json" in response.json()["error"]["message"]
    assert str(tmp_path) not in response.text


def test_unknown_failure_stays_safe_json(control_plane):
    client, manager = control_plane
    manager.load.side_effect = RuntimeError("secret internals")
    response = client.post("/v1/models/load", json={"model": "org/model"})
    assert response.status_code == 500
    assert response.json()["error"]["type"] == "api_error"
    assert "secret" not in response.text


def test_resolution_failure_before_manager_is_json(control_plane, monkeypatch):
    client, manager = control_plane

    def fail(*args, **kwargs):
        raise hub_error(RepositoryNotFoundError, 401)

    monkeypatch.setattr(residency, "resolve_resident_performance", fail)
    response = client.post("/v1/models/load", json={"model": "org/missing"})
    assert response.status_code == 404
    assert "HF_TOKEN" in response.json()["error"]["message"]
    manager.load.assert_not_awaited()


def test_alias_to_missing_local_path(control_plane, monkeypatch, tmp_path):
    client, manager = control_plane
    monkeypatch.setattr(residency, "resolve_model", lambda _: str(tmp_path / "missing"))
    manager.load.side_effect = RuntimeError("secret internals")
    response = client.post("/v1/models/load", json={"model": "saved-local-model"})
    assert response.status_code == 404
    assert "local model path" in response.json()["error"]["message"]


def test_resident_reuse_does_not_require_checkpoint_on_disk(control_plane, tmp_path):
    client, manager = control_plane
    model = str(tmp_path / "moved")
    manager.load.return_value = SimpleNamespace(model_id=model)
    manager.snapshot = lambda: {"models": [{"id": model}]}
    response = client.post("/v1/models/load", json={"model": model})
    assert response.status_code == 200
    assert response.json()["id"] == model


def test_cyclic_cause_and_unrelated_context_stay_safe(control_plane):
    client, manager = control_plane
    error = RuntimeError("secret internals")
    error.__cause__ = error
    error.__context__ = hub_error(RepositoryNotFoundError, 401)
    manager.load.side_effect = error
    response = client.post("/v1/models/load", json={"model": "org/model"})
    assert response.status_code == 500
    assert response.json()["error"]["type"] == "api_error"
    assert "secret" not in response.text
