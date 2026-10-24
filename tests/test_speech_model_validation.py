# SPDX-License-Identifier: Apache-2.0
"""Speech model errors precede optional backend dependency errors (#5200)."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from rapid_mlx.middleware.exception_handlers import install_exception_handlers
from rapid_mlx.routes import audio


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr("rapid_mlx.middleware.auth.verify_api_key", lambda: None)
    app = FastAPI()
    app.include_router(audio.router)
    install_exception_handlers(app)
    with TestClient(app) as client:
        yield client


def serve(monkeypatch, alias=None, name=None):
    monkeypatch.setattr(
        "rapid_mlx.config.get_config",
        lambda: SimpleNamespace(model_alias=alias, model_name=name),
    )


@pytest.mark.parametrize("served", [None, "qwen3-tts-4bit"])
@pytest.mark.parametrize(
    "model", ["nope", "qwen3-tts-4bti", "org/unknown", "whisper-large-v3"]
)
def test_unknown_model_precedes_all_dependency_probes(
    client, monkeypatch, served, model
):
    serve(monkeypatch, served)
    assert audio._served_tts_default() == served
    probe = Mock(side_effect=AssertionError("must validate before dependencies"))
    monkeypatch.setattr("rapid_mlx.audio.probe.require_mlx_audio_tts", probe)
    monkeypatch.setattr("rapid_mlx.audio.probe.require_kokoro_runtime", probe)
    monkeypatch.setattr(audio, "_generate_speech_blocking", probe)
    response = client.post(
        "/v1/audio/speech", json={"model": model, "input": "hi", "voice": "default"}
    )
    assert response.status_code == 404
    error = response.json()["error"]
    assert error["code"] == "model_not_found"
    assert error["type"] == "model_not_found_error"
    assert error["param"] == "model"
    assert model in error["message"]
    if served:
        assert served in error["message"]
    probe.assert_not_called()


@pytest.mark.parametrize(
    "model", [None, "default", "qwen3-tts-4bit", "gateway", "KOKORO"]
)
def test_valid_models_resolve_before_generation(client, monkeypatch, model):
    qwen = audio.TTS_MODEL_ALIASES["qwen3-tts-4bit"]
    serve(monkeypatch, "gateway", qwen)
    monkeypatch.setattr("rapid_mlx.audio.probe.require_mlx_audio_tts", lambda: None)
    kokoro = Mock()
    monkeypatch.setattr("rapid_mlx.audio.probe.require_kokoro_runtime", kokoro)
    generate = Mock(return_value=(b"RIFF0", 24000, 1))
    monkeypatch.setattr(audio, "_generate_speech_blocking", generate)
    body = {"input": "hi", "voice": "default"}
    if model is not None:
        body["model"] = model
    response = client.post("/v1/audio/speech", json=body)
    assert response.status_code == 200, response.text
    expected = audio.TTS_MODEL_ALIASES["kokoro"] if model == "KOKORO" else qwen
    assert generate.call_args.args[0] == expected
    assert kokoro.call_count == (1 if model == "KOKORO" else 0)


def test_known_hf_id_retains_dependency_error(client, monkeypatch):
    serve(monkeypatch)
    monkeypatch.setattr("rapid_mlx.audio.probe.require_mlx_audio_tts", lambda: None)
    probe = Mock(side_effect=HTTPException(503, "Kokoro requires misaki"))
    monkeypatch.setattr("rapid_mlx.audio.probe.require_kokoro_runtime", probe)
    response = client.post(
        "/v1/audio/speech",
        json={"model": audio.TTS_MODEL_ALIASES["kokoro"], "input": "hi"},
    )
    assert response.status_code == 503
    assert "misaki" in response.json()["error"]["message"]
    probe.assert_called_once()
