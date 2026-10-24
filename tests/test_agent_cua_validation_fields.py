# SPDX-License-Identifier: Apache-2.0
"""Agent/CUA validation identifies schema fields without reflecting request data."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rapid_mlx.config import get_config
from rapid_mlx.middleware.exception_handlers import install_exception_handlers
from rapid_mlx.routes import agents, cua


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(get_config(), "api_key", "validation-test")
    app = FastAPI()
    app.include_router(agents.router)
    app.include_router(cua.router)
    install_exception_handlers(app)
    with TestClient(app, headers={"Authorization": "Bearer validation-test"}) as client:
        yield client


@pytest.mark.parametrize(
    "path,payload,field",
    [
        ("/v1/agent/runs", {}, "goal"),
        ("/v1/agent/runs", {"input": "private-input-value"}, "goal"),
        ("/v1/cua/runs", {"goal": "open Finder"}, "app"),
        ("/v1/cua/runs", {"app": "Finder"}, "goal"),
    ],
)
def test_missing_run_field(client, path, payload, field):
    response = client.post(path, json=payload)
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["type"] == "invalid_request_error"
    assert error["code"] == "invalid_request"
    assert error["param"] == field
    assert f"{field}: Field required" in error["message"]
    assert "<field>: Field required" not in error["message"]
    assert "private-input-value" not in response.text


def test_missing_nested_cua_field(client):
    target = {"target_id": "first", "app": "pid:123", "pid": 123}
    response = client.post(
        "/v1/cua/runs",
        json={"app": "pid:123", "goal": "open Finder", "targets": [target, target]},
    )
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["param"] == "window_id"
    assert "targets.0.window_id: Field required" in error["message"]
    assert "targets.1.window_id: Field required" in error["message"]
    assert "<field>" not in error["message"]


@pytest.mark.parametrize(
    "path,payload,field",
    [
        ("/v1/agent/runs", {"goal": "hello", "tool_names": [123]}, "tool_names"),
        ("/v1/cua/runs", {"goal": "hello", "app": {"private-key": "secret"}}, "app"),
    ],
)
def test_invalid_run_field_redacts_values(client, path, payload, field):
    response = client.post(path, json=payload)
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["param"] == field
    assert field in error["message"]
    assert "<field>" not in error["message"]
    assert "private-key" not in response.text
    assert "secret" not in response.text


def test_unknown_agent_field_stays_redacted(client):
    response = client.post(
        "/v1/agent/runs", json={"goal": "hello", "private-key": "secret"}
    )
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["param"] is None
    assert "<field>: Extra inputs are not permitted" in error["message"]
    assert "private-key" not in response.text
    assert "secret" not in response.text
