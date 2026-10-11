# SPDX-License-Identifier: Apache-2.0
"""CPU-only HTTP contracts for unavailable cache persistence."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rapid_mlx.cache.protocol import (
    PROTOCOL_VERSION,
    CachePersistenceUnsupportedError,
    EngineNotReadyError,
    Manifest,
    write_manifest,
)


@pytest.mark.parametrize("operation", ["export", "import"])
@pytest.mark.parametrize(
    "error,status",
    [(CachePersistenceUnsupportedError, 501), (EngineNotReadyError, 503)],
)
def test_cache_persistence_maps_engine_capability_errors(
    monkeypatch, tmp_path, operation, error, status
):
    from rapid_mlx.routes import cache

    monkeypatch.setenv("RAPID_MLX_CACHE_EXPORT_DIR", str(tmp_path))
    engine = SimpleNamespace(
        config=SimpleNamespace(model_name="test-model"),
        save_cache_with_outcome=Mock(side_effect=error("unavailable")),
        load_cache_with_result=Mock(side_effect=error("unavailable")),
    )
    cfg = SimpleNamespace(engine=engine, model_name="test-model")
    monkeypatch.setattr(cache, "get_config", lambda: cfg)
    app = FastAPI()
    app.include_router(cache.router)
    app.dependency_overrides[cache.verify_api_key] = lambda: None
    source = tmp_path / "source"
    source.mkdir()
    write_manifest(
        source,
        Manifest(protocol_version=PROTOCOL_VERSION, model_id="test-model", entries=0),
    )

    with TestClient(app) as client:
        response = client.post(
            f"/v1/cache/{operation}",
            json={"destination": "dest"}
            if operation == "export"
            else {"source": "source"},
        )

    assert response.status_code == status, response.text
    if status == 501:
        assert response.json()["detail"]["error"] == {
            "message": f"prompt-cache {operation} not supported for vision models",
            "type": "invalid_request_error",
            "code": "cache_persistence_unsupported",
        }
    else:
        assert response.json()["detail"] == "engine not loaded"
    called = (
        engine.save_cache_with_outcome
        if operation == "export"
        else engine.load_cache_with_result
    )
    called.assert_called_once()
    assert not (tmp_path / "dest" / "manifest.json").exists()
