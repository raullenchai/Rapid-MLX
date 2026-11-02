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


@pytest.mark.parametrize("trusted", [False, True])
@pytest.mark.parametrize("metadata_file", ["config.json", "tokenizer_config.json"])
def test_consent_required_model_is_not_ready(
    health_client, tmp_path, monkeypatch, metadata_file, trusted
):
    cfg, client = health_client
    cfg.ready = True
    cfg.model_alias = "whisper-small"
    cfg.model_path = str(tmp_path)
    monkeypatch.delenv("RAPID_MLX_TRUST_REMOTE_CODE", raising=False)
    # Metadata alone must suffice; there are no weights or executable files.
    (tmp_path / metadata_file).write_text(
        '{"auto_map":{"AutoTokenizer":"private.Tokenizer"}}'
    )
    if trusted:
        monkeypatch.setenv("RAPID_MLX_TRUST_REMOTE_CODE", "1")
        assert client.get("/health").json()["ready"] is True
        assert client.get("/health/ready").status_code == 200
        return
    data = client.get("/health").json()
    assert data["ready"] is False
    assert data["reason"] == "model_code_consent_required"
    assert "RAPID_MLX_TRUST_REMOTE_CODE=1" in data["message"]
    assert str(tmp_path) not in data["message"]
    assert "private.Tokenizer" not in data["message"]
    for path in ("/health/ready", "/readyz"):
        response = client.get(path)
        assert response.status_code == 503
        assert response.json()["reason"] == data["reason"]
        assert response.json()["ready"] is False


@pytest.mark.parametrize("audio", [False, True])
def test_model_without_consent_requirement_stays_ready(
    health_client, tmp_path, monkeypatch, audio
):
    cfg, client = health_client
    cfg.ready = True
    cfg.model_alias = "whisper-small" if audio else None
    cfg.model_path = str(tmp_path)
    monkeypatch.delenv("RAPID_MLX_TRUST_REMOTE_CODE", raising=False)
    # Text loaders need no consent for metadata they do not execute.
    (tmp_path / "config.json").write_text(
        "{}" if audio else '{"auto_map":{"AutoTokenizer":"private.Tokenizer"}}'
    )
    assert client.get("/health").json()["ready"] is True
    assert client.get("/health/ready").status_code == 200


def test_cached_consent_required_model_is_not_ready(
    health_client, tmp_path, monkeypatch
):
    from rapid_mlx import model_metadata
    from rapid_mlx.audio.registry import resolve_audio_alias

    cfg, client = health_client
    cfg.ready = True
    cfg.model_alias = "whisper-small"
    cfg.model_path = resolve_audio_alias(cfg.model_alias).hf_id
    monkeypatch.delenv("RAPID_MLX_TRUST_REMOTE_CODE", raising=False)
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "tokenizer_config.json").write_text(
        '{"auto_map":{"AutoTokenizer":"private.Tokenizer"}}'
    )
    lookups = []

    def cached_file(model, filename):
        lookups.append((model, filename))
        return tmp_path / filename

    monkeypatch.setattr(model_metadata, "_cached_file", cached_file)
    monkeypatch.setattr(
        model_metadata, "resolve_offline_cached_snapshot", lambda _: None
    )
    response = client.get("/health/ready")
    assert response.status_code == 503
    assert response.json()["reason"] == "model_code_consent_required"
    assert lookups and all(model == cfg.model_path for model, _ in lookups)


def test_model_file_consent_blocks_readiness_during_startup(
    health_client, tmp_path, monkeypatch
):
    cfg, client = health_client
    cfg.model_path = str(tmp_path)
    (tmp_path / "config.json").write_text('{"model_file":"private.py"}')
    monkeypatch.delenv("RAPID_MLX_TRUST_REMOTE_CODE", raising=False)
    for started in (False, True):
        cfg.ready = started
        response = client.get("/health/ready")
        assert response.status_code == 503
        assert response.json()["reason"] == "model_code_consent_required"
        assert client.get("/health").json()["ready"] is False


@pytest.mark.parametrize("blocked", [False, True])
def test_consent_probe_caches_metadata_after_snapshot_found(
    health_client, tmp_path, monkeypatch, blocked
):
    from unittest.mock import patch

    from rapid_mlx import model_metadata
    from rapid_mlx.routes.health import _model_consent_blocker

    cfg, _ = health_client
    cfg.model_path = str(tmp_path)
    cfg.model_alias = "whisper-small"
    monkeypatch.delenv("RAPID_MLX_TRUST_REMOTE_CODE", raising=False)
    (tmp_path / "config.json").write_text(
        '{"model_file":"custom.py"}' if blocked else "{}"
    )
    with patch.object(
        model_metadata, "read_model_metadata", wraps=model_metadata.read_model_metadata
    ) as read:
        first = _model_consent_blocker(cfg)
        assert bool(first) == blocked
        assert _model_consent_blocker(cfg) == first
        assert read.call_count == 1
        cfg.model_name = "changed"
        assert _model_consent_blocker(cfg) == first
        assert read.call_count == 2


def test_consent_probe_rechecks_missing_snapshot(health_client, tmp_path, monkeypatch):
    from unittest.mock import Mock

    from rapid_mlx import model_metadata
    from rapid_mlx.routes.health import _model_consent_blocker

    cfg, _ = health_client
    cfg.model_path = str(tmp_path)
    monkeypatch.delenv("RAPID_MLX_TRUST_REMOTE_CODE", raising=False)
    read = Mock(
        side_effect=[
            None,
            model_metadata.ModelMetadata(
                config={"model_file": "custom.py"},
                chat_template=None,
                snapshot_dir=tmp_path,
            ),
        ]
    )
    monkeypatch.setattr(model_metadata, "read_model_metadata", read)
    assert _model_consent_blocker(cfg) == {}
    assert _model_consent_blocker(cfg)["reason"] == "model_code_consent_required"
    assert read.call_count == 2
