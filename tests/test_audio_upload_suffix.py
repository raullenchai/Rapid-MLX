# SPDX-License-Identifier: Apache-2.0
"""Upload container hints must survive the route's temporary-file boundary."""

import asyncio
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.mark.parametrize(
    "filename,content_type,suffix",
    [
        (None, None, ".wav"),
        ("", "audio/webm", ".webm"),
        ("recording", "application/ogg", ".ogg"),
        ("blob", " video/webm ; codecs=opus", ".webm"),
        ("blob", "audio/x-m4a", ".m4a"),
        ("blob", "audio/aac", ".aac"),
        ("blob", "audio/x-caf", ".caf"),
        ("C:\\recordings\\audio.OPUS", None, ".opus"),
        ("../../escape", "unknown/type", ".wav"),
        ("audio.webm/../../escape", None, ".wav"),
        ("audio.webm.exe", None, ".wav"),
        ("audio." + "x" * 300, None, ".wav"),
    ],
)
def test_suffix_metadata_fallbacks(filename, content_type, suffix):
    from rapid_mlx.routes.audio import _audio_upload_suffix

    assert (
        _audio_upload_suffix(
            SimpleNamespace(filename=filename, content_type=content_type)
        )
        == suffix
    )


@pytest.fixture
def upload_client(monkeypatch):
    from rapid_mlx.audio import probe
    from rapid_mlx.routes import audio
    from rapid_mlx.runtime import audio_worker

    paths = []
    expected = {}

    def infer(path, **kwargs):
        path = Path(path)
        paths.append(path)
        assert path.read_bytes() == expected["payload"]
        assert path.suffix == expected["suffix"]
        if expected.get("fail"):
            raise RuntimeError("could not decode audio file")
        return SimpleNamespace(text="hello", language="en", duration=0.2, segments=[])

    engine = SimpleNamespace(model_name="whisper-small", transcribe=infer)
    monkeypatch.setattr(audio, "_stt_engine", engine)
    monkeypatch.setattr(audio, "_aligner_engine", SimpleNamespace(model_name="aligner"))
    monkeypatch.setattr(audio, "_get_stt_lane_lock", asyncio.Lock)
    monkeypatch.setattr(audio, "_resolve_stt_model", lambda model: model)
    monkeypatch.setattr(audio, "_is_aligner_model", lambda model: model == "aligner")
    monkeypatch.setattr(audio, "_canonical_model_id", lambda model: model)
    monkeypatch.setattr(audio, "_note_served_stt_engine", lambda engine: None)
    monkeypatch.setattr(probe, "require_mlx_audio_stt", lambda: None)
    monkeypatch.setattr(
        audio, "_align_blocking", lambda model, path, text, language: infer(path)
    )

    async def run(_lane, _model, _operation, func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(audio_worker, "run_audio_mlx", run)
    module = types.ModuleType("rapid_mlx.audio.stt")
    module.STTEngine = object
    monkeypatch.setitem(sys.modules, "rapid_mlx.audio.stt", module)
    app = FastAPI()
    app.include_router(audio.router)
    app.dependency_overrides[audio.verify_api_key] = lambda: None
    with TestClient(app) as client:
        yield client, paths, expected


@pytest.mark.parametrize("lane", ["transcriptions", "translations", "alignment"])
@pytest.mark.parametrize(
    "filename,content_type,suffix",
    [
        ("audio.webm", "audio/webm", ".webm"),
        ("audio.M4A", "application/octet-stream", ".m4a"),
        ("audio.ogg", "audio/ogg", ".ogg"),
        ("audio.opus", "audio/opus", ".opus"),
        ("audio.m4b", "audio/mp4", ".m4b"),
        ("audio.mp4", "video/mp4", ".mp4"),
        ("audio.aac", "audio/aac", ".aac"),
        ("audio.caf", "audio/x-caf", ".caf"),
        ("audio.wav", "audio/wav", ".wav"),
        ("audio.mp3", "audio/mpeg", ".mp3"),
        ("audio.flac", "audio/flac", ".flac"),
        ("blob", "Audio/WebM; codecs=opus", ".webm"),
        ("audio.bin", "audio/mp4", ".mp4"),
        ("../../audio.WEBM", "audio/wav", ".webm"),
        ("audio.untrusted", "application/octet-stream", ".wav"),
    ],
)
def test_upload_container_reaches_engine(
    upload_client, lane, filename, content_type, suffix
):
    client, paths, expected = upload_client
    expected.update(payload=b"uploaded audio bytes", suffix=suffix)
    data = {"model": "whisper-small"}
    if lane == "alignment":
        lane = "transcriptions"
        data.update(model="aligner", text="hello")
    response = client.post(
        f"/v1/audio/{lane}",
        files={"file": (filename, expected["payload"], content_type)},
        data=data,
    )
    assert response.status_code == 200, response.text
    assert response.json()["text"] == "hello"
    assert len(paths) == 1
    assert not paths[0].exists()


@pytest.mark.parametrize("lane", ["transcriptions", "translations", "alignment"])
def test_container_decode_failure_keeps_400_and_cleans_tempfile(upload_client, lane):
    client, paths, expected = upload_client
    expected.update(payload=b"corrupted webm", suffix=".webm", fail=True)
    data = {"model": "whisper-small"}
    if lane == "alignment":
        lane = "transcriptions"
        data.update(model="aligner", text="hello")
    response = client.post(
        f"/v1/audio/{lane}",
        files={"file": ("audio.webm", expected["payload"], "audio/webm")},
        data=data,
    )
    assert response.status_code == 400, response.text
    assert response.json()["detail"]["error"]["code"] == "invalid_audio_file"
    assert len(paths) == 1
    assert not paths[0].exists()
