# SPDX-License-Identifier: Apache-2.0
"""OpenAI voice presets work with model-specific voice lists."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rapid_mlx.audio import tts
from rapid_mlx.config import get_config
from rapid_mlx.routes import audio


@pytest.mark.parametrize(
    ("model", "voice", "valid", "expected"),
    [
        ("kokoro", "alloy", ["af_alloy"], "af_alloy"),
        ("moss-tts-nano", "alloy", ["default"], "default"),
        ("kokoro", "af_hart", ["af_heart"], "af_hart"),
        ("kokoro", "alloy", ["alloy", "af_alloy"], "alloy"),
        ("kokoro", "ECHO", ["bm_echo", "am_echo", "af_echo"], "af_echo"),
        ("unknown", "alloy", ["other"], "alloy"),
    ],
)
def test_preset_mapping(model, voice, valid, expected):
    assert audio._map_openai_preset_voice(model, voice, valid) == expected


@pytest.mark.parametrize(
    ("preset", "expected"),
    [("alloy", "af_alloy"), ("echo", "am_echo"), ("fable", "bm_fable")],
)
def test_kokoro_preset_mapping_matches_cached_snapshot(monkeypatch, preset, expected):
    assert tts.KOKORO_VOICES[0] == "af_heart"
    assert len(set(tts.KOKORO_VOICES)) == 54
    model = audio._resolve_tts_model("kokoro")
    for snapshot_voices in ([], sorted(tts.KOKORO_VOICES)):
        monkeypatch.setattr(
            tts, "_list_snapshot_voices", lambda _, voices=snapshot_voices: voices
        )
        valid = audio._allowed_voices_for(model)
        assert audio._map_openai_preset_voice(model, preset, valid) == expected


def test_speech_route_maps_preset_to_single_voice(monkeypatch):
    monkeypatch.setattr(get_config(), "api_key", None)
    monkeypatch.setattr("rapid_mlx.audio.probe.require_mlx_audio_tts", lambda: None)
    monkeypatch.setattr(audio, "_allowed_voices_for", lambda _: ["default"])
    observed = []

    def generate(model_name, text, response_format, gen_kwargs, *args):
        observed.append(gen_kwargs["voice"])
        return b"RIFF0", 24000, 1

    monkeypatch.setattr(audio, "_generate_speech_blocking", generate)
    app = FastAPI()
    app.include_router(audio.router)
    with TestClient(app) as client:
        response = client.post(
            "/v1/audio/speech",
            json={"model": "moss-tts-nano", "input": "hello", "voice": "alloy"},
        )
    assert response.status_code == 200, response.text
    assert observed == ["default"]
