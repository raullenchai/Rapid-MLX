"""Security contract for the read-only CUA visual observation endpoint."""

from __future__ import annotations

import base64

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rapid_mlx.computer_use.errors import ComputerUseError
from rapid_mlx.config import reset_config
from rapid_mlx.routes import cua as cua_routes

AUTH = {"Authorization": "Bearer secret"}
REQUEST = {"app": "Finder", "pid": 42, "window_id": "cg:123"}


def _snapshot(*, screenshot_png: bytes | None = None) -> dict:
    snapshot = {
        "snapshot_id": "42:cg:123:fresh",
        "observed_at": 123.5,
        "app": {"name": "Finder", "bundleId": "com.apple.finder", "pid": 42},
        "window_id": "cg:123",
        "window_index": 0,
        "window": {
            "window_id": "cg:123",
            "index": 0,
            "title": "Private document",
            "x": 10,
            "y": 20,
            "width": 800,
            "height": 600,
        },
        "coordinate_space": "screen",
        "elements": [
            {
                "index": 0,
                "role": "AXButton",
                "subrole": "",
                "label": "Open",
                "actions": ["AXPress"],
                "x": 12,
                "y": 24,
                "width": 40,
                "height": 20,
                "center": [32, 34],
            }
        ],
        "element_count": 1,
        "tree_text": "must never cross the API boundary",
        "truncated": False,
    }
    if screenshot_png is not None:
        snapshot["screenshot_png"] = screenshot_png
    return snapshot


@pytest.fixture()
def observation_client(monkeypatch):
    cfg = reset_config()
    cfg.api_key = "secret"
    monkeypatch.setattr(cua_routes.sys, "platform", "darwin")
    backend = cua_routes._backend()
    monkeypatch.setattr(
        backend,
        "permissions",
        lambda: {"accessibility": True, "screen_recording": True, "hints": []},
    )
    app = FastAPI()
    app.include_router(cua_routes.router)
    with TestClient(app) as client:
        yield client, backend
    reset_config()


def test_observation_is_fresh_pid_window_bound_and_has_no_screenshot_by_default(
    observation_client, monkeypatch
):
    client, backend = observation_client
    calls = []

    def observe(app, **kwargs):
        calls.append((app, kwargs))
        return _snapshot(screenshot_png=b"should not be returned")

    monkeypatch.setattr(backend, "get_app_state", observe)
    response = client.post("/v1/cua/observations", headers=AUTH, json=REQUEST)

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["pragma"] == "no-cache"
    assert calls == [
        (
            "pid:42",
            {
                "screenshot": False,
                "use_cache": False,
                "window_id": "cg:123",
                "activate": False,
            },
        )
    ]
    payload = response.json()
    assert payload["screenshot"] is None
    assert "tree_text" not in payload
    assert "screenshot_png" not in payload
    assert payload["app"]["pid"] == 42
    assert payload["window_id"] == "cg:123"


@pytest.mark.parametrize(
    ("role", "subrole"),
    [
        ("AXTextField", "AXSecureTextField"),
        ("AXSecureTextField", ""),
    ],
)
def test_observation_redacts_secure_text_labels(
    observation_client, monkeypatch, role, subrole
):
    client, backend = observation_client
    snapshot = _snapshot()
    snapshot["elements"][0].update(
        {
            "role": role,
            "subrole": subrole,
            "label": "hunter2-private",
        }
    )
    monkeypatch.setattr(backend, "get_app_state", lambda *a, **k: snapshot)

    response = client.post("/v1/cua/observations", headers=AUTH, json=REQUEST)

    assert response.status_code == 200
    assert response.json()["elements"][0]["label"] == "[secure text redacted]"
    assert "hunter2-private" not in response.text


def test_observation_ax_tree_stays_on_requested_pid_with_same_named_apps(
    observation_client, monkeypatch
):
    client, backend = observation_client
    app_info = {"name": "Finder", "bundleId": "com.apple.finder", "pid": 42}
    window = {
        "window_id": "cg:123",
        "index": 0,
        "title": "Requested window",
        "x": 10,
        "y": 20,
        "width": 800,
        "height": 600,
    }
    monkeypatch.setattr(
        backend, "_resolve_app", lambda app, *, activate: (object(), app_info)
    )
    monkeypatch.setattr(backend, "_select_window", lambda *a, **k: window)
    monkeypatch.setattr(backend, "_window_records", lambda app: [window])
    seen_pid = []

    def collect(app, **kwargs):
        seen_pid.append(kwargs.get("expected_pid"))
        label = (
            "PID 42 control"
            if kwargs.get("expected_pid") == 42
            else "PID 99 private control"
        )
        return [
            {
                "target_id": "t000",
                "role": "AXButton",
                "subrole": "",
                "text": label,
                "actions": ["AXPress"],
                "rect": [12, 24, 40, 20],
            }
        ]

    monkeypatch.setattr(backend, "_collect_with_timeout", collect)
    response = client.post("/v1/cua/observations", headers=AUTH, json=REQUEST)
    assert response.status_code == 200
    assert seen_pid == [42]
    assert response.json()["elements"][0]["label"] == "PID 42 control"
    assert "PID 99 private control" not in response.text


def test_screenshot_requires_server_opt_in(observation_client, monkeypatch):
    client, backend = observation_client
    called = False

    def observe(*args, **kwargs):
        nonlocal called
        called = True
        return _snapshot(screenshot_png=b"png")

    monkeypatch.setattr(backend, "get_app_state", observe)
    response = client.post(
        "/v1/cua/observations", headers=AUTH, json={**REQUEST, "screenshot": True}
    )
    assert response.status_code == 403
    assert response.json()["detail"]["code"] == "screenshot_disabled"
    assert response.headers["cache-control"] == "no-store"
    assert called is False


def test_screenshot_requires_screen_recording_permission(
    observation_client, monkeypatch
):
    client, backend = observation_client
    monkeypatch.setenv("RAPID_MLX_CUA_EXPOSE_SCREENSHOTS", "true")
    monkeypatch.setattr(
        backend,
        "permissions",
        lambda: {"accessibility": True, "screen_recording": False, "hints": []},
    )
    response = client.post(
        "/v1/cua/observations", headers=AUTH, json={**REQUEST, "screenshot": True}
    )
    assert response.status_code == 403
    assert response.json()["detail"]["code"] == "permission_denied"
    assert response.headers["cache-control"] == "no-store"


def test_screenshot_is_encoded_only_after_both_opt_ins(observation_client, monkeypatch):
    client, backend = observation_client
    png = b"\x89PNG\r\n\x1a\n" + b"payload"
    monkeypatch.setenv("RAPID_MLX_CUA_EXPOSE_SCREENSHOTS", "1")
    monkeypatch.setattr(
        backend, "get_app_state", lambda *a, **k: _snapshot(screenshot_png=png)
    )

    response = client.post(
        "/v1/cua/observations", headers=AUTH, json={**REQUEST, "screenshot": True}
    )
    assert response.status_code == 200
    image = response.json()["screenshot"]
    assert image["data"] == base64.b64encode(png).decode("ascii")
    assert image["byte_count"] == len(png)


def test_oversize_png_is_rejected_without_echoing_content(
    observation_client, monkeypatch
):
    client, backend = observation_client
    monkeypatch.setenv("RAPID_MLX_CUA_EXPOSE_SCREENSHOTS", "true")
    secret = b"sensitive-pixels"
    monkeypatch.setattr(cua_routes, "MAX_OBSERVATION_PNG_BYTES", len(secret) - 1)
    monkeypatch.setattr(
        backend, "get_app_state", lambda *a, **k: _snapshot(screenshot_png=secret)
    )
    response = client.post(
        "/v1/cua/observations", headers=AUTH, json={**REQUEST, "screenshot": True}
    )
    assert response.status_code == 413
    assert response.json()["detail"]["code"] == "screenshot_too_large"
    assert base64.b64encode(secret).decode("ascii") not in response.text


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        ({"app": {"name": "Other", "bundleId": "other", "pid": 42}}, "app_mismatch"),
        (
            {"app": {"name": "Finder", "bundleId": "com.apple.finder", "pid": 99}},
            "app_mismatch",
        ),
        ({"window_id": "cg:999"}, "window_stale"),
    ],
)
def test_wrong_app_pid_or_window_fails_closed(
    observation_client, monkeypatch, mutation, code
):
    client, backend = observation_client
    snapshot = _snapshot()
    snapshot.update(mutation)
    monkeypatch.setattr(backend, "get_app_state", lambda *a, **k: snapshot)
    response = client.post("/v1/cua/observations", headers=AUTH, json=REQUEST)
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == code
    assert "Private document" not in response.text
    assert "must never" not in response.text


@pytest.mark.parametrize(
    ("error", "status_code"),
    [
        (ComputerUseError("window_not_found", "window is gone"), 404),
        (ComputerUseError("window_stale", "window moved"), 409),
        (ComputerUseError("permission_denied", "Accessibility denied"), 403),
    ],
)
def test_backend_failures_remain_typed(
    observation_client, monkeypatch, error, status_code
):
    client, backend = observation_client

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(backend, "get_app_state", fail)
    response = client.post("/v1/cua/observations", headers=AUTH, json=REQUEST)
    assert response.status_code == status_code
    assert response.json()["detail"]["code"] == error.code
    assert response.headers["cache-control"] == "no-store"


def test_unsupported_platform_is_typed(observation_client, monkeypatch):
    client, _ = observation_client
    monkeypatch.setattr(cua_routes.sys, "platform", "linux")
    response = client.post("/v1/cua/observations", headers=AUTH, json=REQUEST)
    assert response.status_code == 501
    assert response.json()["detail"]["code"] == "unsupported_platform"
