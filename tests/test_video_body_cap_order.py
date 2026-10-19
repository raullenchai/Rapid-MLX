# SPDX-License-Identifier: Apache-2.0
"""Video upload auth and body caps with the production middleware order."""

import json

import pytest
from fastapi import Depends, FastAPI, Request
from fastapi.testclient import TestClient

from rapid_mlx.config.server_config import get_config, reset_config
from rapid_mlx.middleware import auth
from rapid_mlx.middleware.auth import verify_api_key
from rapid_mlx.middleware.body_depth import install_request_body_depth_middleware
from rapid_mlx.middleware.body_size import (
    _path_is_guarded,
    install_request_body_limit_middleware,
)
from rapid_mlx.routes import video


@pytest.fixture
def video_app(monkeypatch):
    reset_config()
    cfg = get_config()
    assert cfg.max_request_bytes == 8 * 1024 * 1024
    monkeypatch.setattr(cfg, "api_key", "test-video-secret")
    monkeypatch.setattr(auth, "_auth_warning_logged", False)
    monkeypatch.setenv("RAPID_MLX_MAX_BODY_DEPTH", "64")

    app = FastAPI()

    @app.post("/v1/videos", dependencies=[Depends(verify_api_key)])
    @app.post("/v1/videos/extend", dependencies=[Depends(verify_api_key)])
    async def echo_body(request: Request):
        return {"received_bytes": len(await request.body())}

    # Match server.py: size -> depth -> video. Image middleware is unnecessary
    # here because it only gates image routes.
    video.install_video_body_limit_middleware(app)
    install_request_body_depth_middleware(app)
    install_request_body_limit_middleware(app)
    try:
        yield app
    finally:
        reset_config()


@pytest.fixture
def video_client(video_app):
    with TestClient(video_app) as client:
        yield client


@pytest.mark.parametrize("path", ["/v1/videos", "/v1/videos/extend"])
@pytest.mark.parametrize("advertised_bytes", [10_000_000, 30_000_000])
def test_video_body_auth_precedes_advertised_cap(video_client, path, advertised_bytes):
    response = video_client.post(
        path,
        content=b"x",
        headers={"Content-Length": str(advertised_bytes)},
    )

    assert response.request.headers["content-length"] == str(advertised_bytes)
    assert response.status_code == 401, response.text
    assert response.json()["detail"] == "API key required"


@pytest.mark.parametrize("path", ["/v1/videos", "/v1/videos/extend"])
def test_video_body_advertised_cap_uses_video_detail(video_client, path):
    advertised_bytes = video._VIDEO_REQUEST_BYTES + 1
    response = video_client.post(
        path,
        content=b"x",
        headers={
            "Authorization": "Bearer test-video-secret",
            "Content-Length": str(advertised_bytes),
        },
    )

    # A one-byte actual body can only trip this cap via the advertised length.
    assert response.request.headers["content-length"] == str(advertised_bytes)
    assert response.status_code == 413, response.text
    assert response.json() == {"detail": "video request body exceeds 21 MB"}


@pytest.mark.parametrize("path", ["/v1/videos", "/v1/videos/extend"])
def test_video_body_above_generic_cap_reaches_handler(video_client, path):
    body = b"x" * (10 * 1024 * 1024)
    assert get_config().max_request_bytes < len(body) < video._VIDEO_REQUEST_BYTES

    response = video_client.post(
        path, content=body, headers={"Authorization": "Bearer test-video-secret"}
    )

    assert response.status_code == 200, response.text
    assert response.json() == {"received_bytes": len(body)}


def test_video_extend_exclusion_is_exact_and_subpaths_stay_guarded():
    assert _path_is_guarded("/v1/videos/extend") is False
    assert _path_is_guarded("/v1/videos/extend/x") is True


async def _call_app(app, path, receive, *, headers=(), method="POST"):
    sent = []

    async def send(message):
        sent.append(message)

    await app(
        {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": method,
            "scheme": "http",
            "path": path,
            "query_string": b"",
            "headers": list(headers),
        },
        receive,
        send,
    )
    starts = [msg for msg in sent if msg["type"] == "http.response.start"]
    assert len(starts) == 1
    body = b"".join(
        msg.get("body", b"") for msg in sent if msg["type"] == "http.response.body"
    )
    return starts[0]["status"], json.loads(body)


@pytest.mark.parametrize("path", ["/v1/videos", "/v1/videos/extend"])
@pytest.mark.parametrize("advertise_length", [False, True])
async def test_unauthenticated_json_video_never_reads_body(
    video_app, path, advertise_length
):
    # This oversized JSON body must remain entirely unread, with or without
    # Content-Length. Raising makes any attempt to pre-buffer it fail the test.
    body = b'"' + b"x" * video._VIDEO_REQUEST_BYTES + b'"'
    receive_calls = 0

    async def receive():
        nonlocal receive_calls
        receive_calls += 1
        raise AssertionError("unauthenticated video body was read")

    headers = [(b"content-type", b"application/json")]
    if advertise_length:
        headers.append((b"content-length", str(len(body)).encode("ascii")))
    status, payload = await _call_app(video_app, path, receive, headers=headers)
    assert receive_calls == 0
    assert status == 401
    assert payload == {"detail": "API key required"}


@pytest.mark.parametrize("path", ["/v1/videos", "/v1/videos/extend"])
async def test_authenticated_json_video_stream_is_bounded(video_app, path):
    chunk = b" " * (1024 * 1024)
    receive_calls = 0

    async def receive():
        nonlocal receive_calls
        receive_calls += 1
        assert receive_calls <= 22, "video gate read beyond the first over-cap chunk"
        # No Content-Length, and the stream has not ended when the cap trips.
        return {"type": "http.request", "body": chunk, "more_body": True}

    status, payload = await _call_app(
        video_app,
        path,
        receive,
        headers=[
            (b"content-type", b"application/json"),
            (b"authorization", b"Bearer test-video-secret"),
        ],
    )
    assert receive_calls == 22
    assert status == 413
    assert payload == {"detail": "video request body exceeds 21 MB"}


@pytest.mark.parametrize("path", ["/v1/videos", "/v1/videos/extend"])
@pytest.mark.parametrize("method", ["DELETE", "PUT", "PATCH"])
def test_non_post_video_keeps_generic_advertised_cap(video_client, path, method):
    response = video_client.request(
        method,
        path,
        content=b"x",
        headers={"Content-Length": str(8 * 1024 * 1024 + 1)},
    )
    assert response.status_code == 413, response.text
    assert response.json()["error"]["code"] == "request_too_large"


@pytest.mark.parametrize("method", ["POST", "DELETE", "PUT", "PATCH", "GET"])
def test_size_guard_preserves_method_and_path_boundaries(method):
    for path in ("/v1/videos", "/v1/videos/extend", "/v1/images/edits"):
        assert _path_is_guarded(path, method) is (method != "POST")
        assert _path_is_guarded(path + "/x", method) is True
    assert _path_is_guarded("/v1/audio/transcriptions") is False
    assert _path_is_guarded("/v1/audio/transcriptions", method) is False


@pytest.mark.parametrize(
    ("method", "path"),
    [("POST", "/v1/chat/completions"), ("DELETE", "/v1/videos/extend")],
)
def test_depth_gate_still_rejects_deep_json(video_client, method, path):
    body = b"[" * 65 + b"0" + b"]" * 65
    response = video_client.request(
        method, path, content=body, headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 400, response.text
    assert response.json()["error"]["code"] == "request_body_too_deep"
