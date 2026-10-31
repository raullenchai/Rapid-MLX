# SPDX-License-Identifier: Apache-2.0
"""Health classification for audio servers without a primary engine."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rapid_mlx.config import reset_config
from rapid_mlx.routes.health import probe_router


@pytest.fixture
def health_client():
    cfg = reset_config()
    app = FastAPI()
    app.include_router(probe_router)
    yield cfg, TestClient(app)
    reset_config()


@pytest.mark.parametrize(
    "alias,model_type,engine_type",
    [
        ("whisper-small", "stt", "audio"),
        ("kokoro", "tts", "audio"),
        ("not-a-model", "llm", "unknown"),
    ],
)
def test_model_type_without_engine(health_client, alias, model_type, engine_type):
    cfg, client = health_client
    cfg.model_alias = alias
    data = client.get("/health").json()
    assert data["model_type"] == model_type
    assert data["engine_type"] == engine_type
