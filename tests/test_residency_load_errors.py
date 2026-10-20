# SPDX-License-Identifier: Apache-2.0
"""Resident-load failures must reach clients as actionable, safe JSON (#4492)."""

import asyncio
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


@pytest.mark.parametrize(
    "diagnostic", ["raise_if_missing_local_model", "local_model_failure_message"]
)
@pytest.mark.parametrize("diagnostic_error", [OSError, ValueError])
@pytest.mark.parametrize("hub_failure", [False, True])
def test_diagnostic_failure_preserves_safe_load_error(
    control_plane, monkeypatch, diagnostic, diagnostic_error, hub_failure
):
    client, manager = control_plane

    def fail_diagnostic(*args, **kwargs):
        raise diagnostic_error("diagnostic secret /Users/private/model")

    monkeypatch.setattr(residency, diagnostic, fail_diagnostic)
    manager.load.side_effect = (
        hub_error(RepositoryNotFoundError, 404)
        if hub_failure
        else RuntimeError("loader secret /Users/private/model")
    )
    response = client.post("/v1/models/load", json={"model": "org/model"})
    assert response.status_code == (404 if hub_failure else 500)
    assert response.headers["content-type"] == "application/json"
    assert response.json()["error"]["type"] == (
        "not_found_error" if hub_failure else "api_error"
    )
    if hub_failure:
        assert "HF_TOKEN" in response.json()["error"]["message"]
    assert "secret" not in response.text
    assert "/Users/private" not in response.text
    assert client.get("/v1/models/residency").status_code == 200


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


@pytest.mark.parametrize(
    ("model", "mode", "message"),
    [
        ("flux-schnell", "editing", "flux-schnell does not support image editing."),
        (
            "qwen-image-edit",
            "generation",
            "qwen-image-edit does not support image generation.",
        ),
    ],
)
def test_unsupported_image_mode_rejected_before_assets_in_dynamic_loader(
    monkeypatch, model, mode, message
):
    from rapid_mlx import server
    from rapid_mlx.image.engine import ImageRequestError

    asset_calls = []
    monkeypatch.setattr(
        "rapid_mlx.model_downloads.require_local",
        lambda _name: asset_calls.append("checkpoint"),
    )
    monkeypatch.setattr(
        "rapid_mlx.image.engine.ImageGenerationEngine._ensure_runtime_assets",
        lambda _self: asset_calls.append("runtime assets"),
    )
    monkeypatch.setattr(
        "rapid_mlx.image.engine.ImageGenerationEngine._verify_weights_complete",
        lambda _self: asset_calls.append("weight verification"),
    )

    with pytest.raises(ImageRequestError, match=message):
        asyncio.run(server._load_dynamic_resident_model(model, None, image_mode=mode))
    assert asset_calls == []


def test_supported_image_mode_reaches_dynamic_resident_load(monkeypatch):
    from rapid_mlx import server
    from rapid_mlx.runtime.image_lane import ImageEngine

    modes = []
    monkeypatch.setattr("rapid_mlx.model_downloads.require_local", lambda _name: None)
    monkeypatch.setattr(
        ImageEngine,
        "ensure_resident",
        lambda _self, *, mode=None: modes.append(mode),
    )

    entry = asyncio.run(
        server._load_dynamic_resident_model("qwen-image", None, image_mode="generation")
    )

    assert isinstance(entry.engine, ImageEngine)
    assert modes == ["generation"]


@pytest.mark.parametrize(
    ("model", "mode", "message"),
    [
        ("flux-schnell", "editing", "flux-schnell does not support image editing."),
        (
            "qwen-image-edit",
            "generation",
            "qwen-image-edit does not support image generation.",
        ),
    ],
)
def test_unsupported_image_mode_rejected_before_manager_admission(
    control_plane, model, mode, message
):
    client, manager = control_plane

    response = client.post("/v1/models/load", json={"model": model, "image_mode": mode})

    assert response.status_code == 400
    assert response.json()["error"]["message"] == message
    manager.load.assert_not_awaited()


@pytest.mark.parametrize(
    ("model", "mode"),
    [("flux-schnell", "editing"), ("qwen-image-edit", "generation")],
)
def test_unsupported_image_mode_rejected_for_resident_reuse(model, mode):
    from rapid_mlx.image.engine import ImageRequestError
    from rapid_mlx.runtime.image_lane import ImageEngine
    from rapid_mlx.runtime.model_registry import ModelEntry, ModelRegistry
    from rapid_mlx.runtime.resident_models import ResidentModelManager

    engine = ImageEngine(model)
    registry = ModelRegistry()
    entry = ModelEntry(engine=engine, model_name=model, model_path=model)
    registry.add(entry, is_default=True)
    manager = ResidentModelManager(registry, AsyncMock(), memory_reader=lambda: 0)
    manager.register_primary(entry, estimated_bytes=1)

    with pytest.raises(ImageRequestError):
        asyncio.run(manager.load(model, image_mode=mode))

    manager.loader.assert_not_awaited()


def test_wrapped_image_request_error(control_plane):
    from rapid_mlx.image.engine import ImageRequestError

    client, manager = control_plane
    error = RuntimeError("loader wrapper secret")
    error.__cause__ = ImageRequestError("qwen-image does not support image editing.")
    manager.load.side_effect = error
    response = client.post(
        "/v1/models/load",
        json={"model": "Qwen/Qwen-Image", "image_mode": "editing"},
    )
    assert response.status_code == 400
    assert response.json()["error"]["message"] == str(error.__cause__)


def test_image_request_error_precedes_local_diagnostics(
    control_plane, monkeypatch, tmp_path
):
    from rapid_mlx.image.engine import ImageRequestError

    client, manager = control_plane
    manager.load.side_effect = ImageRequestError("unsupported image mode")
    monkeypatch.setattr(
        residency,
        "raise_if_missing_local_model",
        lambda _ref: pytest.fail("image request error reached local diagnostics"),
    )
    monkeypatch.setattr(
        residency,
        "local_model_failure_message",
        lambda _ref, _exc: pytest.fail("image request error reached local diagnostics"),
    )

    response = client.post(
        "/v1/models/load",
        json={"model": str(tmp_path), "image_mode": "editing"},
    )

    assert response.status_code == 400
    assert response.json()["error"]["message"] == "unsupported image mode"


def test_operational_image_runtime_error_stays_unchanged(control_plane):
    from rapid_mlx.image.engine import ImageRuntimeError

    client, manager = control_plane
    manager.load.side_effect = ImageRuntimeError("loader secret /Users/private/model")
    response = client.post("/v1/models/load", json={"model": "Qwen/Qwen-Image"})
    assert response.status_code == 500
    assert response.json() == {
        "error": {
            "message": "Failed to load resident model: ImageRuntimeError",
            "type": "api_error",
            "code": None,
            "param": None,
        }
    }


def test_residency_import_and_failure_without_image_extra():
    import subprocess
    import sys

    script = """
import builtins
original_import = builtins.__import__
def without_image_extra(name, *args, **kwargs):
    if name.split('.')[0] in {'mlx', 'mflux'}:
        raise ModuleNotFoundError(name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = without_image_extra
from rapid_mlx.routes import residency
assert residency._load_failure(RuntimeError('failure'), 'org/model').status_code == 500
"""
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True)
