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

    (tmp_path / "config.json").write_text("{}")
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


@pytest.mark.parametrize(
    "model", ["nope-xyz", "qwen2.5-72b-4bit", "foo/bar-does-not-exist-xyz"]
)
def test_cli_prefetch_exit_is_actionable_http_error(control_plane, monkeypatch, model):
    from rapid_mlx import cli, server

    client, manager = control_plane
    error = hub_error(RepositoryNotFoundError, 404)

    def missing_metadata(*args, **kwargs):
        raise error

    monkeypatch.setattr(cli, "_cache_runnability", lambda _: False)
    monkeypatch.setattr(cli, "_offline_hub_mode_active", lambda: False)
    monkeypatch.setattr(cli, "_check_disk_space", lambda *a, **kw: None)
    monkeypatch.setattr(cli, "_try_mirror_prefetch", lambda *a, **kw: False)
    monkeypatch.setattr("rapid_mlx.model_downloads.disabled", lambda: False)
    monkeypatch.setattr("huggingface_hub.model_info", missing_metadata)

    async def load(*args, **kwargs):
        server._ensure_routing_config(model)

    manager.load.side_effect = load
    # Exercise the actual CLI prefetch failure, without downloading weights.
    # Direct invocation first makes a leaked SystemExit a normal test failure.
    with pytest.raises(residency.HTTPException) as caught:
        asyncio.run(
            residency.load_resident_model(residency.ModelLoadRequest(model=model))
        )
    assert caught.value.status_code == 404
    response = client.post("/v1/models/load", json={"model": model})
    assert response.status_code == 404
    assert response.json()["error"]["type"] == "not_found_error"
    assert "Check the model name" in response.json()["error"]["message"]
    assert "secret" not in response.text
    assert client.get("/v1/models/residency").status_code == 200


@pytest.mark.parametrize("model", ["ministral-3b-4bit", "qwopus-27b-8bit"])
def test_retired_alias_is_actionable_http_error(control_plane, model):
    client, manager = control_plane
    response = client.post("/v1/models/load", json={"model": model})
    assert response.status_code == 400
    assert "retired" in response.json()["error"]["message"]
    manager.load.assert_not_awaited()


@pytest.mark.parametrize("explicit_cause", [False, True])
def test_cli_exit_preserves_gated_access_error(control_plane, explicit_cause):
    client, manager = control_plane
    try:
        raise hub_error(GatedRepoError, 403)
    except GatedRepoError as error:
        try:
            if explicit_cause:
                raise SystemExit(1) from error
            raise SystemExit(1)
        except SystemExit as exit_error:
            manager.load.side_effect = exit_error
    response = client.post("/v1/models/load", json={"model": "org/gated"})
    assert response.status_code == 403
    assert "HF_TOKEN" in response.json()["error"]["message"]
    assert "secret" not in response.text


def test_unclassified_cli_exit_is_safe_json(control_plane):
    client, manager = control_plane
    manager.load.side_effect = SystemExit("secret /Users/private/model")
    response = client.post("/v1/models/load", json={"model": "org/model"})
    assert response.status_code == 500
    assert response.json()["error"]["type"] == "api_error"
    assert "secret" not in response.text


def test_request_cancellation_still_propagates(control_plane):
    _, manager = control_plane
    manager.load.side_effect = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            residency.load_resident_model(residency.ModelLoadRequest(model="org/model"))
        )


@pytest.mark.parametrize(
    "model", ["/", "//", "..", "../model", "org/../model", "'..'", "' .. '", " / "]
)
@pytest.mark.parametrize("field", ["model", "model_path"])
def test_directory_like_paths_fail_fast(control_plane, monkeypatch, model, field):
    import os
    import time
    from pathlib import Path

    client, manager = control_plane

    def no_walk(*args, **kwargs):
        pytest.fail("request attempted a filesystem walk")

    monkeypatch.setattr(os, "walk", no_walk)
    monkeypatch.setattr(Path, "rglob", no_walk)
    monkeypatch.setattr(residency, "resolve_model", no_walk)
    started = time.monotonic()
    response = client.post("/v1/models/load", json={"model": "org/model", field: model})
    assert response.status_code == 400
    assert time.monotonic() - started < 1
    manager.load.assert_not_awaited()
    assert client.get("/v1/models/residency").status_code == 200


@pytest.mark.parametrize("kind", ["empty", "nested", "file", "root-link"])
def test_non_model_directory_rejected(control_plane, monkeypatch, tmp_path, kind):
    client, manager = control_plane
    target = tmp_path / "target"
    if kind == "file":
        target.write_text("not a model")
    elif kind == "root-link":
        target.symlink_to("/", target_is_directory=True)
    else:
        target.mkdir()
        if kind == "nested":
            (target / "child").mkdir()
            (target / "child" / "config.json").write_text("{}")
    response = client.post("/v1/models/load", json={"model": str(target)})
    assert response.status_code == 400
    manager.load.assert_not_awaited()


@pytest.mark.parametrize(
    "marker", ["config.json", "model_index.json", "split_model.json", "params.json"]
)
@pytest.mark.parametrize("override", [False, True])
def test_local_model_manifests_accepted(control_plane, tmp_path, marker, override):
    client, manager = control_plane
    (tmp_path / marker).write_text("{}")
    manager.load.return_value = SimpleNamespace(model_id="loaded")
    manager.snapshot = lambda: {"models": [{"id": "loaded"}]}
    body = (
        {"model": "org/model", "model_path": str(tmp_path)}
        if override
        else {"model": str(tmp_path)}
    )
    assert client.post("/v1/models/load", json=body).status_code == 200
    manager.load.assert_awaited_once()


def test_resolved_alias_to_root_rejected(control_plane, monkeypatch):
    client, manager = control_plane
    monkeypatch.setattr(residency, "resolve_model", lambda _: "/")
    assert client.post("/v1/models/load", json={"model": "saved"}).status_code == 400
    manager.load.assert_not_awaited()


def test_local_failure_does_not_scan_tree(control_plane, monkeypatch, tmp_path):
    from pathlib import Path

    client, manager = control_plane
    (tmp_path / "config.json").write_text("{}")
    monkeypatch.setattr(
        Path, "rglob", lambda *a, **kw: pytest.fail("recursive diagnostic")
    )
    manager.load.side_effect = FileNotFoundError(
        2, "missing", str(tmp_path / "model.safetensors")
    )
    response = client.post("/v1/models/load", json={"model": str(tmp_path)})
    assert response.status_code == 400
    assert "model.safetensors" in response.text


async def test_path_validation_keeps_event_loop_responsive(control_plane, monkeypatch):
    import threading

    _, manager = control_plane
    entered = threading.Event()
    release = threading.Event()
    original = residency._validate_load_ref

    def slow_probe(value):
        entered.set()
        assert release.wait(2), "event loop could not release filesystem probe"
        return original(value)

    monkeypatch.setattr(residency, "_validate_load_ref", slow_probe)
    task = asyncio.create_task(
        residency.load_resident_model(residency.ModelLoadRequest(model="/"))
    )
    try:
        for _ in range(100):
            if entered.is_set():
                break
            await asyncio.sleep(0.005)
        assert entered.is_set()
        assert (await residency.model_residency())["models"] == []
    finally:
        release.set()
    with pytest.raises(residency.HTTPException) as caught:
        await task
    assert caught.value.status_code == 400
    manager.load.assert_not_awaited()


@pytest.mark.parametrize("field", ["model", "model_path"])
def test_symlink_target_is_pinned_before_resolution(
    control_plane, monkeypatch, tmp_path, field
):
    client, manager = control_plane
    target = tmp_path / "checkpoint"
    target.mkdir()
    (target / "config.json").write_text("{}")
    link = tmp_path / "selected"
    link.symlink_to(target, target_is_directory=True)

    def retarget(value):
        link.unlink()
        link.symlink_to("/", target_is_directory=True)
        return value

    monkeypatch.setattr(residency, "resolve_model", retarget)
    manager.load.return_value = SimpleNamespace(model_id="loaded")
    manager.snapshot = lambda: {"models": [{"id": "loaded"}]}
    response = client.post(
        "/v1/models/load", json={"model": "org/model", field: str(link)}
    )
    assert response.status_code == 200
    assert manager.load.call_args.kwargs["model_path"] == str(target.resolve())
    if field == "model":
        assert manager.load.call_args.args[0] == str(target.resolve())


def test_alias_directory_collision_preserves_catalog(
    control_plane, monkeypatch, tmp_path
):
    client, manager = control_plane
    monkeypatch.chdir(tmp_path)
    (tmp_path / "qwen3.5-4b-4bit").mkdir()
    manager.load.return_value = SimpleNamespace(model_id="loaded")
    manager.snapshot = lambda: {"models": [{"id": "loaded"}]}
    response = client.post("/v1/models/load", json={"model": "qwen3.5-4b-4bit"})
    assert response.status_code == 200
    assert manager.load.call_args.args[0] == "qwen3.5-4b-4bit"
    assert manager.load.call_args.kwargs["model_path"] is None


def test_existing_repo_spelling_keeps_local_precedence(
    control_plane, monkeypatch, tmp_path
):
    client, manager = control_plane
    monkeypatch.chdir(tmp_path)
    (tmp_path / "org" / "model").mkdir(parents=True)
    response = client.post("/v1/models/load", json={"model": "org/model"})
    assert response.status_code == 400
    manager.load.assert_not_awaited()


def test_resolved_alias_loads_canonical_target(control_plane, monkeypatch, tmp_path):
    client, manager = control_plane
    target = tmp_path / "checkpoint"
    target.mkdir()
    (target / "config.json").write_text("{}")
    link = tmp_path / "selected"
    link.symlink_to(target, target_is_directory=True)
    monkeypatch.setattr(residency, "resolve_model", lambda _: str(link))

    def retarget(performance, **kwargs):
        link.unlink()
        link.symlink_to("/", target_is_directory=True)
        return performance

    monkeypatch.setattr(residency, "resolve_resident_performance", retarget)
    manager.load.return_value = SimpleNamespace(model_id="loaded")
    manager.snapshot = lambda: {"models": [{"id": "loaded"}]}
    assert client.post("/v1/models/load", json={"model": "saved"}).status_code == 200
    assert manager.load.call_args.args[0] == "saved"
    assert manager.load.call_args.kwargs["model_path"] == str(target.resolve())


def test_replaced_canonical_directory_cannot_trigger_recursive_diagnostics(
    control_plane, monkeypatch, tmp_path
):
    import os
    import time
    from pathlib import Path

    client, manager = control_plane
    target = tmp_path / "checkpoint"
    target.mkdir()
    (target / "config.json").write_text("{}")

    def replace_directory(performance, **kwargs):
        target.rename(tmp_path / "moved")
        target.symlink_to("/", target_is_directory=True)
        return performance

    def no_walk(*args, **kwargs):
        pytest.fail("a replaced checkpoint triggered a recursive scan")

    monkeypatch.setattr(residency, "resolve_resident_performance", replace_directory)
    monkeypatch.setattr(os, "walk", no_walk)
    monkeypatch.setattr(Path, "rglob", no_walk)
    manager.load.side_effect = FileNotFoundError(
        2, "missing", str(target / "config.json")
    )
    started = time.monotonic()
    response = client.post("/v1/models/load", json={"model": str(target)})
    assert response.status_code == 400
    assert "config.json" in response.text
    assert time.monotonic() - started < 1
    assert client.get("/v1/models/residency").status_code == 200
