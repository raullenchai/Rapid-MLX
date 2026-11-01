import json
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rapid_mlx.audio import registry


def local(tmp_path, monkeypatch, name, config):
    folder = tmp_path / name
    folder.mkdir()
    if config is not None:
        (folder / "config.json").write_text(json.dumps(config))
    monkeypatch.setattr(registry, "_LOCAL_AUDIO_ENTRIES", {})
    return folder


def serve_check(folder, monkeypatch):
    from rapid_mlx import cli
    from tests.test_audio_alias_registry import _make_serve_args

    monkeypatch.setitem(
        sys.modules,
        "rapid_mlx.server",
        SimpleNamespace(_resolve_api_key=lambda key: key),
    )
    monkeypatch.setattr(cli, "_resolve_serve_port", lambda *a, **kw: 8000)
    with (
        patch.object(cli, "_serve_audio_mode") as audio,
        patch.object(cli, "_ensure_model_downloaded") as text,
        patch.object(cli, "_run_optional_runtime_guard"),
    ):
        cli.serve_command(_make_serve_args(str(folder)))
    assert audio.called and not text.called


@pytest.mark.parametrize(
    ("config", "family"),
    [
        ({"model_type": "voxcpm2", "architecture": "voxcpm2"}, "voxcpm"),
        (
            {"model_type": "chatterbox_turbo", "architecture": "chatterbox_turbo"},
            "chatterbox",
        ),
    ],
)
def test_real_metadata_and_serve(tmp_path, monkeypatch, config, family):
    folder = local(tmp_path, monkeypatch, "audio", config)
    assert registry.resolve_local_audio_folder(str(folder)).family == family
    serve_check(folder, monkeypatch)


def test_configless_f5(tmp_path, monkeypatch):
    folder = local(tmp_path, monkeypatch, "f5-tts", None)
    (folder / "model.safetensors").touch()
    assert registry.resolve_local_audio_folder(str(folder)) is None
    (folder / "vocab.txt").touch()
    assert registry.resolve_local_audio_folder(str(folder)).family == "f5"
    serve_check(folder, monkeypatch)
    text = tmp_path / "text-model"
    text.mkdir()
    assert registry.resolve_local_audio_folder(str(text)) is None


def configure(folder, monkeypatch, alias=None):
    from rapid_mlx.config import get_config

    entry = registry.resolve_local_audio_folder(str(folder))
    registry.register_served_local_audio_entry(entry, str(folder))
    cfg = get_config()
    for key, value in {
        "model_alias": alias,
        "model_name": str(folder),
        "model_path": str(folder),
        "api_key": None,
    }.items():
        monkeypatch.setattr(cfg, key, value)
    return entry


def test_collision_transcription(tmp_path, monkeypatch):
    from rapid_mlx.audio import probe
    from rapid_mlx.routes import audio

    folder = local(tmp_path, monkeypatch, "whisper-base", {"model_type": "whisper"})
    configure(folder, monkeypatch, "whisper-base")
    monkeypatch.setattr(probe, "require_mlx_audio_stt", lambda: None)
    seen = []

    async def run(*, model, **kwargs):
        seen.append(audio._resolve_stt_model(model))
        return {"text": "hi"}

    monkeypatch.setattr(audio, "_run_stt_request", run)
    app = FastAPI()
    app.include_router(audio.router)
    with TestClient(app) as client:
        for model in (None, "default", "whisper-1", "whisper-base", "whisper-small"):
            response = client.post(
                "/v1/audio/transcriptions",
                data={"model": model} if model else {},
                files={"file": ("tone.wav", b"audio", "audio/wav")},
            )
            assert response.status_code == 200, response.text
    assert seen == [str(folder.resolve())] * 4 + [
        audio.STT_MODEL_ALIASES["whisper-small"]
    ]


@pytest.mark.parametrize("served_tts", [False, True], ids=["hub-stt", "local-tts"])
def test_transcription_served_alias_requires_local_stt(
    tmp_path, monkeypatch, served_tts
):
    from rapid_mlx.audio import probe
    from rapid_mlx.config import get_config
    from rapid_mlx.routes import audio

    if served_tts:
        model = "served-speech"
        folder = local(tmp_path, monkeypatch, "speech-local", {"model_type": "kokoro"})
        entry = configure(folder, monkeypatch, model)
        assert entry.type == "tts"
    else:
        model = "whisper-base"
        repo = audio.STT_MODEL_ALIASES[model]
        monkeypatch.setattr(registry, "_LOCAL_AUDIO_ENTRIES", {})
        cfg = get_config()
        for key, value in {
            "model_alias": model,
            "model_name": repo,
            "model_path": repo,
            "api_key": None,
        }.items():
            monkeypatch.setattr(cfg, key, value)
        entry = registry.resolve_audio_alias(repo)
        assert entry.type == "stt" and entry.notes != "local folder"

    monkeypatch.setattr(probe, "require_mlx_audio_stt", lambda: None)
    seen = []

    async def run(*, model, **kwargs):
        seen.append(audio._resolve_stt_model(model))
        return {"text": "hi"}

    monkeypatch.setattr(audio, "_run_stt_request", run)
    app = FastAPI()
    app.include_router(audio.router)
    with TestClient(app) as client:
        response = client.post(
            "/v1/audio/transcriptions",
            data={"model": model},
            files={"file": ("tone.wav", b"audio", "audio/wav")},
        )
    if served_tts:
        assert response.status_code == 404, response.text
        assert response.json()["detail"]["error"]["code"] == "model_not_found"
        assert seen == []
    else:
        assert response.status_code == 200, response.text
        assert seen == [repo]


@pytest.mark.parametrize(
    ("config", "clone", "expected"),
    [
        ({"model_type": "qwen3_tts", "tts_model_type": "custom_voice"}, False, 200),
        ({"model_type": "qwen3_tts", "tts_model_type": "custom_voice"}, True, 400),
        ({"model_type": "chatterbox"}, True, 200),
    ],
)
def test_speech_local(tmp_path, monkeypatch, config, clone, expected):
    import numpy as np

    from rapid_mlx.audio import tts
    from rapid_mlx.routes import audio

    folder = local(tmp_path, monkeypatch, "speech-local", config)
    entry = configure(folder, monkeypatch)
    if config["model_type"] == "qwen3_tts":
        assert entry.default_voice == "Serena"
    seen = []

    class FakeEngine:
        def __init__(self, model_name):
            pass

        def load(self):
            pass

        def generate(self, text, **kwargs):
            seen.append(kwargs)
            return tts.AudioOutput(np.zeros(10, dtype=np.float32), 24000, 10 / 24000)

        def to_bytes(self, output, format):
            return b"RIFFfake"

    monkeypatch.setattr(tts, "TTSEngine", FakeEngine)
    monkeypatch.setattr(
        "rapid_mlx.runtime.audio_worker.run_audio_mlx_sync",
        lambda role, model, phase, fn, *a, **kw: fn(*a, **kw),
    )
    monkeypatch.setitem(
        sys.modules,
        "rapid_mlx.server",
        SimpleNamespace(_emit_audio_model_served_once=lambda *a: None),
    )
    monkeypatch.setattr("rapid_mlx.audio.probe.require_mlx_audio_tts", lambda: None)
    monkeypatch.setattr(audio, "_tts_engine", None)
    app = FastAPI()
    app.include_router(audio.router)
    body = {"input": "Hello", "response_format": "wav"}
    if clone:
        body.update(ref_audio="AAAA", ref_text="hello")
    with TestClient(app) as client:
        response = client.post("/v1/audio/speech", json=body)
    assert response.status_code == expected, response.text
    if expected == 400:
        assert "unsupported_voice_cloning" in response.text
    elif not clone:
        assert seen[0]["voice"] == "Serena"


def test_moss_local_download_policy(tmp_path, monkeypatch):
    from rapid_mlx import model_downloads
    from rapid_mlx.audio import tts

    folder = local(tmp_path, monkeypatch, "moss-local", {"model_type": "moss_tts_nano"})
    configure(folder, monkeypatch)
    calls = []
    monkeypatch.setattr(
        model_downloads,
        "snapshot_download",
        lambda *a, **kw: calls.append((a, kw)) or str(folder),
    )
    monkeypatch.setattr(tts, "_resolve_safe_tts_checkpoint", lambda name: name)
    monkeypatch.setitem(
        sys.modules,
        "mlx_audio.tts.generate",
        SimpleNamespace(load_model=lambda path: object()),
    )
    tts.TTSEngine(str(folder)).load()
    assert len(calls) == 1 and "local_files_only" not in calls[0][1]
