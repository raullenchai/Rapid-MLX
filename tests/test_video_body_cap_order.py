# SPDX-License-Identifier: Apache-2.0
"""Video upload auth and body caps with the production middleware order."""

import pytest
from fastapi import Depends, FastAPI, Request
from fastapi.testclient import TestClient

from rapid_mlx.config.server_config import get_config, reset_config
from rapid_mlx.middleware import auth
from rapid_mlx.middleware.auth import verify_api_key
from rapid_mlx.middleware.body_size import (
    _path_is_guarded,
    install_request_body_limit_middleware,
)
from rapid_mlx.routes import video


@pytest.fixture
def video_client(monkeypatch):
    reset_config()
    cfg = get_config()
    assert cfg.max_request_bytes == 8 * 1024 * 1024
    monkeypatch.setattr(cfg, "api_key", "test-video-secret")
    monkeypatch.setattr(auth, "_auth_warning_logged", False)

    app = FastAPI()

    @app.post("/v1/videos", dependencies=[Depends(verify_api_key)])
    @app.post("/v1/videos/extend", dependencies=[Depends(verify_api_key)])
    async def echo_body(request: Request):
        return {"received_bytes": len(await request.body())}

    # Match server.py: the generic cap is installed last and runs outermost.
    video.install_video_body_limit_middleware(app)
    install_request_body_limit_middleware(app)
    try:
        with TestClient(app) as client:
            yield client
    finally:
        reset_config()


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
