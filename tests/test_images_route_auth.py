# SPDX-License-Identifier: Apache-2.0
"""Image router auth contracts, runnable without MLX or the server module."""

import base64

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rapid_mlx.config import get_config
from rapid_mlx.middleware import auth
from rapid_mlx.routes.images import (
    _IMAGE_EDIT_REQUEST_BYTES,
    ImageBodyLimitMiddleware,
    install_image_body_limit_middleware,
    router,
)

_TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII="
)


@pytest.fixture
def image_client(monkeypatch):
    cfg = get_config()
    monkeypatch.setattr(cfg, "api_key", "test-image-secret")
    monkeypatch.setattr(cfg, "engine", None)
    monkeypatch.setattr(cfg, "model_registry", None)
    # The development-path dependency logs once; restore that state as well.
    monkeypatch.setattr(auth, "_auth_warning_logged", False)
    app = FastAPI()
    app.include_router(router)
    install_image_body_limit_middleware(app)
    with TestClient(app) as client:
        yield client


def _post_image(client, endpoint, headers):
    if endpoint == "generations":
        return client.post(
            "/v1/images/generations", json={"prompt": "a cat"}, headers=headers
        )
    return client.post(
        "/v1/images/edits",
        data={"prompt": "add a hat"},
        files={"image": ("tiny.png", _TINY_PNG, "image/png")},
        headers=headers,
    )


@pytest.mark.parametrize("endpoint", ["generations", "edits"])
@pytest.mark.parametrize(
    ("token", "status", "detail"),
    [
        (None, 401, "API key required"),
        ("wrong-key", 401, "Invalid API key"),
        ("test-image-secret", 409, None),
    ],
    ids=["missing-key", "wrong-key", "valid-key"],
)
def test_image_post_auth_precedes_capability(
    image_client, endpoint, token, status, detail
):
    headers = {} if token is None else {"Authorization": f"Bearer {token}"}
    response = _post_image(image_client, endpoint, headers)
    assert response.status_code == status, response.text
    if status == 401:
        assert response.json()["detail"] == detail
    else:
        assert response.json()["detail"]["error"]["code"] == "image_model_not_loaded"


@pytest.mark.parametrize("method,endpoint", [("GET", "progress"), ("POST", "cancel")])
def test_image_control_requires_key(image_client, method, endpoint):
    response = image_client.request(method, f"/v1/images/{endpoint}")
    assert response.status_code == 401
    assert response.json()["detail"] == "API key required"


def test_image_generation_without_api_key_keeps_dev_path_open(
    image_client, monkeypatch
):
    monkeypatch.setattr(get_config(), "api_key", None)
    response = _post_image(image_client, "generations", {})
    assert response.status_code == 409, response.text
    assert response.json()["detail"]["error"]["code"] == "image_model_not_loaded"


@pytest.mark.parametrize(
    ("token", "status", "detail"),
    [
        (None, 401, "API key required"),
        ("wrong-key", 401, "Invalid API key"),
        ("test-image-secret", 413, None),
    ],
    ids=["missing-key", "wrong-key", "valid-key"],
)
def test_image_edit_auth_precedes_advertised_body_limit(
    image_client, token, status, detail
):
    headers = {"Content-Length": str(_IMAGE_EDIT_REQUEST_BYTES + 1)}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"

    response = image_client.post("/v1/images/edits", headers=headers)

    assert response.status_code == status, response.text
    if detail is not None:
        assert response.json()["detail"] == detail


@pytest.mark.parametrize(
    ("headers", "status", "detail"),
    [
        (
            [
                ("Authorization", "Bearer wrong-key"),
                ("Authorization", "Bearer test-image-secret"),
            ],
            401,
            "Invalid API key",
        ),
        (
            [
                ("Authorization", "Bearer test-image-secret"),
                ("Authorization", "Bearer wrong-key"),
            ],
            401,
            "Invalid API key",
        ),
        (
            [("Authorization", "Bearer  test-image-secret ")],
            413,
            None,
        ),
        ([("Authorization", "Basic xyz")], 401, "Invalid API key"),
    ],
    ids=[
        "duplicate-first-wrong",
        "duplicate-first-valid",
        "whitespace-padded-valid",
        "non-bearer",
    ],
)
def test_image_edit_scope_auth_matches_router_parsing(
    image_client, headers, status, detail
):
    headers.append(("Content-Length", str(_IMAGE_EDIT_REQUEST_BYTES + 1)))

    response = image_client.post("/v1/images/edits", headers=headers)

    assert response.status_code == status, response.text
    if detail is not None:
        assert response.json()["detail"] == detail


@pytest.mark.parametrize("authenticated", [False, True])
def test_image_edit_auth_precedes_multipart_parsing(image_client, authenticated):
    headers = {"Content-Type": "multipart/form-data"}
    if authenticated:
        headers["Authorization"] = "Bearer test-image-secret"

    response = image_client.post(
        "/v1/images/edits", content=b"not multipart", headers=headers
    )

    if authenticated:
        assert response.status_code in {400, 422}, response.text
        assert response.status_code != 401
    else:
        assert response.status_code == 401, response.text
        assert response.json()["detail"] == "API key required"


@pytest.mark.asyncio
async def test_image_edit_authenticates_before_reading_body(monkeypatch):
    monkeypatch.setattr(get_config(), "api_key", "test-image-secret")
    receive_calls = 0
    sent = []

    async def downstream(scope, receive, send):
        raise AssertionError("unauthenticated request reached FastAPI")

    async def receive():
        nonlocal receive_calls
        receive_calls += 1
        raise AssertionError("unauthenticated request body was read")

    async def send(message):
        sent.append(message)

    middleware = ImageBodyLimitMiddleware(downstream)
    await middleware(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/images/edits",
            "headers": [],
        },
        receive,
        send,
    )

    assert receive_calls == 0
    assert sent[0]["status"] == 401


def test_image_edit_without_api_key_keeps_body_limit(image_client, monkeypatch):
    monkeypatch.setattr(get_config(), "api_key", None)
    response = image_client.post(
        "/v1/images/edits",
        headers={"Content-Length": str(_IMAGE_EDIT_REQUEST_BYTES + 1)},
    )
    assert response.status_code == 413, response.text
