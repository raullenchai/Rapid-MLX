# SPDX-License-Identifier: Apache-2.0
"""Live embedding discovery metadata, independent of optional GPU imports."""

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rapid_mlx.api.models import ModelInfo, ModelsResponse
from rapid_mlx.config import get_config
from rapid_mlx.routes import models

FIELDS = {
    "embedding_dimensions",
    "supported_dimensions",
    "max_input_tokens",
    "pooling",
    "normalized",
}


@pytest.fixture
def discovery(monkeypatch):
    cfg = get_config()
    for key, value in {
        "model_name": "chat-model",
        "model_alias": None,
        "model_registry": None,
        "engine": None,
        "embedding_model_locked": "embedding-model",
        "embedding_engine": None,
        "api_key": None,
    }.items():
        monkeypatch.setattr(cfg, key, value)
    monkeypatch.setattr(models, "_locked_rerank_id", lambda: None)
    monkeypatch.setattr(models, "_audio_lane_snapshot", lambda: None)
    app = FastAPI()
    app.include_router(models.router)
    with TestClient(app) as client:
        yield cfg, client


def engine(config=None, *, maximum=4096, effective=1024, setting="auto", loaded=True):
    return SimpleNamespace(
        is_loaded=loaded,
        _model=SimpleNamespace(config=config or SimpleNamespace(hidden_size=37)),
        effective_max_length=effective,
        _max_length_setting=setting,
        _discover_model_max_length=lambda: maximum,
    )


def cards(client):
    response = client.get("/v1/models")
    assert response.status_code == 200
    listing = response.json()
    ModelsResponse.model_validate(listing)
    detail = client.get("/v1/models/embedding-model")
    assert detail.status_code == 200
    ModelInfo.model_validate(detail.json())
    listed = next(card for card in listing["data"] if card["id"] == "embedding-model")
    for key in FIELDS:
        assert listed.get(key) == detail.json().get(key)
    chat = next(card for card in listing["data"] if card["id"] == "chat-model")
    assert not FIELDS.intersection(chat)
    assert chat["object"] == "model"
    assert chat["owned_by"] == "rapid-mlx"
    assert isinstance(chat["created"], int)
    return listed


def test_embedding_metadata_on_both_endpoints(discovery):
    cfg, client = discovery
    cfg.embedding_engine = engine(
        SimpleNamespace(
            hidden_size=37, pooling_config={"pooling_mode": "mean"}, normalized=True
        )
    )
    card = cards(client)
    assert card["capabilities"] == ["embedding"]
    assert card["embedding_dimensions"] == 37
    assert card["supported_dimensions"] == {"min": 1, "max": 37}
    assert card["max_input_tokens"] == 1024
    assert card["pooling"] == "mean"
    assert card["normalized"] is True


@pytest.mark.parametrize("field", ["embedding_dim", "projection_dim"])
def test_projection_precedes_hidden_size(discovery, field):
    cfg, client = discovery
    cfg.embedding_engine = engine(SimpleNamespace(hidden_size=64, **{field: 23}))
    assert cards(client)["embedding_dimensions"] == 23


def test_nested_text_config_and_args(discovery):
    cfg, client = discovery
    cfg.embedding_engine = engine()
    cfg.embedding_engine._model = SimpleNamespace(
        args=SimpleNamespace(
            text_config=SimpleNamespace(embedding_dim=19, hidden_size=64)
        )
    )
    assert cards(client)["embedding_dimensions"] == 19


@pytest.mark.parametrize("dimension", [None, 0, -1, True, "37"])
def test_unknown_dimensions_omitted(discovery, dimension):
    cfg, client = discovery
    cfg.embedding_engine = engine(SimpleNamespace(hidden_size=dimension))
    card = cards(client)
    assert "embedding_dimensions" not in card
    assert "supported_dimensions" not in card
    assert card["max_input_tokens"] == 1024
    assert "pooling" not in card
    assert "normalized" not in card


@pytest.mark.parametrize("loaded", [False, True])
def test_missing_or_unloaded_engine_omits_metadata(discovery, loaded):
    cfg, client = discovery
    cfg.embedding_engine = engine(loaded=False) if not loaded else None
    assert not FIELDS.intersection(cards(client))


@pytest.mark.parametrize("effective", [None, 0, -1, True, "512"])
def test_unknown_effective_limit_omitted(discovery, effective):
    cfg, client = discovery
    cfg.embedding_engine = engine(effective=effective)
    assert "max_input_tokens" not in cards(client)


def test_fallback_limit_is_not_model_metadata(discovery):
    cfg, client = discovery
    cfg.embedding_engine = engine(maximum=None, effective=512)
    assert "max_input_tokens" not in cards(client)
    cfg.embedding_engine._max_length_setting = 256
    cfg.embedding_engine.effective_max_length = 256
    assert cards(client)["max_input_tokens"] == 256


def test_failed_probe_does_not_break_discovery(discovery):
    cfg, client = discovery
    cfg.embedding_engine = engine()

    def fail():
        raise RuntimeError("unavailable")

    cfg.embedding_engine._discover_model_max_length = fail
    assert not FIELDS.intersection(cards(client))


def test_native_encoder_facts(discovery):
    cfg, client = discovery
    cfg.embedding_engine = engine(
        SimpleNamespace(text_config=SimpleNamespace(embedding_dim=29, hidden_size=64))
    )
    cfg.embedding_engine._model.model_type = "embedding_gemma2"
    card = cards(client)
    assert card["embedding_dimensions"] == 29
    assert card["pooling"] == "mean"
    assert card["normalized"] is True


def test_server_bridge_uses_dedicated_embedding_engine(discovery, monkeypatch):
    import rapid_mlx.server as server

    _, client = discovery
    monkeypatch.setattr(server, "_embedding_engine", engine())
    assert cards(client)["embedding_dimensions"] == 37


def test_profiled_embedding_card(discovery, monkeypatch):
    cfg, client = discovery
    profile = models.resolve_profile("embeddinggemma-2-4bit")
    assert profile is not None
    monkeypatch.setattr(models, "resolve_profile", lambda _: profile)
    cfg.embedding_engine = engine()
    assert cards(client)["embedding_dimensions"] == 37


def test_known_unnormalized_and_unknown_pooling(discovery):
    cfg, client = discovery
    cfg.embedding_engine = engine(
        SimpleNamespace(
            hidden_size=37, normalized=False, pooling_config={"pooling_mode": None}
        )
    )
    card = cards(client)
    assert card["normalized"] is False
    assert "pooling" not in card
