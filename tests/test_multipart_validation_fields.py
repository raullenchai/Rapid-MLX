# SPDX-License-Identifier: Apache-2.0
"""Multipart validation names trusted route fields without reflecting input."""

import pytest
from fastapi import FastAPI, Form
from fastapi.testclient import TestClient

from rapid_mlx.config import get_config
from rapid_mlx.middleware.exception_handlers import install_exception_handlers
from rapid_mlx.routes import audio, images, video


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(get_config(), "api_key", "multipart-test")
    app = FastAPI()
    for router in (audio.router, images.router, video.router):
        app.include_router(router)
    install_exception_handlers(app)
    with TestClient(app, headers={"Authorization": "Bearer multipart-test"}) as client:
        yield client


@pytest.mark.parametrize(
    "path,fields,field",
    [
        ("/v1/images/edits", {"prompt": "green"}, "image"),
        ("/v1/audio/transcriptions", {"model": "whisper"}, "file"),
        ("/v1/audio/translations", {"model": "whisper"}, "file"),
        ("/v1/videos", {"model": "unused"}, "prompt"),
        ("/v1/videos/extend", {"prompt": "green", "extend_frames": "8"}, "input_video"),
    ],
)
@pytest.mark.parametrize("multipart", [True, False])
def test_missing_required_form_field(client, path, fields, field, multipart):
    kwargs = (
        {"files": {key: (None, value) for key, value in fields.items()}}
        if multipart
        else {"data": fields}
    )
    response = client.post(path, **kwargs)
    assert response.status_code == 400, response.text
    error = response.json()["error"]
    assert error["param"] == field
    assert f"{field}: Field required" in error["message"]
    assert "<field>" not in error["message"]
    assert error["code"] == "invalid_request"


@pytest.mark.parametrize(
    "path,fields,uploads,field",
    [
        (
            "/v1/images/edits",
            {"prompt": "green", "steps": "secret-value"},
            {"image": ("a.png", b"x", "image/png")},
            "steps",
        ),
        ("/v1/videos", {"prompt": "green", "fps": "secret-value"}, {}, "fps"),
        (
            "/v1/videos/extend",
            {"prompt": "green", "extend_frames": "secret-value"},
            {"input_video": ("a.mp4", b"x", "video/mp4")},
            "extend_frames",
        ),
    ],
)
def test_invalid_form_field(client, path, fields, uploads, field):
    files = {key: (None, value) for key, value in fields.items()} | uploads
    response = client.post(path, files=files)
    assert response.status_code == 400, response.text
    error = response.json()["error"]
    assert error["param"] == field
    assert f"{field}:" in error["message"]
    assert "<field>" not in error["message"]
    assert "secret-value" not in response.text


def test_form_alias_and_mounted_route():
    app = FastAPI()
    install_exception_handlers(app)

    @app.post("/proxy/form")
    async def form(internal_name: int = Form(..., alias="public_name")):
        return internal_name

    with TestClient(app) as client:
        response = client.post(
            "/proxy/form", files={"public_name": (None, "secret-value")}
        )
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["param"] == "public_name"
    assert "public_name:" in error["message"]
    assert "secret-value" not in response.text


def test_image_json_validation(client):
    response = client.post(
        "/v1/images/generations", json={"prompt": "green", "size": "999x999"}
    )
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["param"] == "size"
    assert "size:" in error["message"]
    assert "<field>" not in error["message"]
