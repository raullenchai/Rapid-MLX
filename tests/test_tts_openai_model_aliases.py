# SPDX-License-Identifier: Apache-2.0
"""Offline speech compatibility and model validation regressions."""

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from huggingface_hub.errors import RepositoryNotFoundError

from rapid_mlx.audio import tts
from rapid_mlx.routes import audio
from rapid_mlx.utils.model_file_guard import RemoteCodeConsentRequired

OPENAI_MODELS = ["tts-1", "tts-1-hd", "gpt-4o-mini-tts"]
OPENAI_VOICES = [
    "alloy",
    "echo",
    "fable",
    "onyx",
    "nova",
    "shimmer",
    "ash",
    "sage",
    "coral",
    "ballad",
    "verse",
]


# Presets with a same-named Kokoro voice keep it; the rest fall back to af_heart.
KOKORO_PRESET_VOICES = {
    "alloy": "af_alloy",
    "echo": "am_echo",
    "fable": "bm_fable",
    "onyx": "am_onyx",
    "nova": "af_nova",
}


@pytest.fixture
def speech_client(monkeypatch):
    monkeypatch.setattr(audio, "_served_tts_default", lambda: None)
    monkeypatch.setattr("rapid_mlx.audio.probe.require_mlx_audio_tts", lambda: None)
    monkeypatch.setattr(
        "rapid_mlx.audio.probe.require_kokoro_runtime", lambda *args: None
    )
    monkeypatch.setattr(tts, "_list_snapshot_voices", lambda _: [])
    observed = []

    def generate(model, text, response_format, kwargs, *args):
        observed.append((model, kwargs["voice"]))
        return b"RIFF0", 24000, 1

    monkeypatch.setattr(audio, "_generate_speech_blocking", generate)
    app = FastAPI()
    app.dependency_overrides[audio.verify_api_key] = lambda: None
    app.dependency_overrides[audio.require_audio_worker_available] = lambda: None
    app.include_router(audio.router)
    with TestClient(app) as client:
        yield client, observed


@pytest.mark.parametrize("model", OPENAI_MODELS)
@pytest.mark.parametrize("voice", OPENAI_VOICES + ["af_heart", "am_adam"])
def test_openai_speech_aliases(speech_client, model, voice):
    client, observed = speech_client
    response = client.post(
        "/v1/audio/speech", json={"model": model, "input": "hello", "voice": voice}
    )
    assert response.status_code == 200, response.text
    assert observed == [
        (
            audio.TTS_MODEL_ALIASES["kokoro"],
            KOKORO_PRESET_VOICES.get(voice, "af_heart")
            if voice in OPENAI_VOICES
            else voice,
        )
    ]


@pytest.mark.parametrize("model", OPENAI_MODELS)
def test_openai_alias_follows_served_default(monkeypatch, model):
    monkeypatch.setattr(audio, "_served_tts_default", lambda: "chatterbox")
    assert audio._resolve_tts_model(model) == audio.TTS_MODEL_ALIASES["chatterbox"]


@pytest.mark.parametrize(
    "model", ["does-not-exist", "foo/bar/baz", "////", "org/../repo", "org/repo.git"]
)
def test_unknown_model_is_actionable_404(speech_client, monkeypatch, model):
    client, observed = speech_client

    def unexpected_probe():
        pytest.fail("invalid model must be rejected before runtime probing")

    monkeypatch.setattr("rapid_mlx.audio.probe.require_mlx_audio_tts", unexpected_probe)
    response = client.post("/v1/audio/speech", json={"model": model, "input": "hi"})
    assert response.status_code == 404, response.text
    error = response.json()["detail"]["error"]
    assert error["param"] == "model"
    assert error["code"] == "model_not_found"
    for accepted in OPENAI_MODELS + ["kokoro"]:
        assert accepted in error["message"]
    assert observed == []


@pytest.mark.parametrize(
    "failure",
    [
        RepositoryNotFoundError(
            "missing",
            response=httpx.Response(
                404, request=httpx.Request("GET", "https://huggingface.co/org/missing")
            ),
        ),
        FileNotFoundError("missing checkpoint"),
        ValueError("Model type not supported"),
        RuntimeError("backend failed"),
        RemoteCodeConsentRequired("consent required"),
    ],
)
def test_model_load_errors(speech_client, monkeypatch, failure):
    client, _ = speech_client
    # Exercise the real blocking loader, failing before inference or downloads.
    monkeypatch.setattr(audio, "_generate_speech_blocking", _REAL_GENERATE)
    monkeypatch.setattr(audio, "_tts_engine", None)

    class MissingModel:
        def __init__(self, model):
            pass

        def load(self):
            raise failure

    monkeypatch.setattr(tts, "TTSEngine", MissingModel)
    monkeypatch.setattr(
        "rapid_mlx.runtime.audio_worker.run_audio_mlx_sync",
        lambda lane, model, operation, fn: fn(),
    )
    response = client.post(
        "/v1/audio/speech",
        json={"model": "org/missing", "input": "hi", "voice": "default"},
    )
    expected = (
        403
        if isinstance(failure, RemoteCodeConsentRequired)
        else 500
        if isinstance(failure, RuntimeError)
        else 404
    )
    assert response.status_code == expected, response.text
    if expected == 404:
        assert "tts-1" in response.json()["detail"]["error"]["message"]
    assert audio._tts_engine is None


_REAL_GENERATE = audio._generate_speech_blocking


@pytest.mark.parametrize("model", OPENAI_MODELS)
def test_alias_with_omitted_voice(speech_client, model):
    client, observed = speech_client
    response = client.post("/v1/audio/speech", json={"model": model, "input": "hi"})
    assert response.status_code == 200, response.text
    assert observed == [(audio.TTS_MODEL_ALIASES["kokoro"], "af_heart")]


def test_invalid_native_voice_still_rejected(speech_client):
    client, observed = speech_client
    response = client.post(
        "/v1/audio/speech",
        json={"model": "tts-1", "input": "hi", "voice": "af_hart"},
    )
    assert response.status_code == 400, response.text
    assert response.json()["detail"]["error"]["code"] == "invalid_voice"
    assert observed == []
