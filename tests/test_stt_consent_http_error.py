# SPDX-License-Identifier: Apache-2.0
"""Consent refusals remain actionable at the audio API boundary."""

import json
import sys
import types

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rapid_mlx.utils.model_file_guard import (
    RemoteCodeConsentRequired,
    validate_processor_code,
)


@pytest.mark.parametrize("failure", ["direct", "wrapped", "generic", "suppressed"])
def test_transcription_load_errors(monkeypatch, failure):
    from rapid_mlx.audio import probe
    from rapid_mlx.routes import audio
    from rapid_mlx.runtime import audio_worker

    class BrokenEngine:
        def __init__(self, model_name):
            self.model_name = model_name

        def load(self):
            if failure == "generic":
                raise RuntimeError("ordinary failure")
            if failure == "suppressed":
                try:
                    raise RemoteCodeConsentRequired("private path /model")
                except RemoteCodeConsentRequired:
                    raise RuntimeError("load failed") from None
            if failure == "wrapped":
                try:
                    raise RemoteCodeConsentRequired("private path /model")
                except RemoteCodeConsentRequired as exc:
                    raise RuntimeError("load failed") from exc
            raise RemoteCodeConsentRequired("private path /model")

    async def run_audio(_lane, _model, _operation, func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setitem(
        sys.modules,
        "rapid_mlx.audio.stt",
        types.SimpleNamespace(STTEngine=BrokenEngine),
    )
    monkeypatch.setattr(probe, "require_mlx_audio_stt", lambda: None)
    monkeypatch.setattr(audio_worker, "run_audio_mlx", run_audio)
    monkeypatch.setattr(audio, "_stt_engine", None)
    app = FastAPI()
    app.include_router(audio.router)
    app.dependency_overrides[audio.require_audio_worker_available] = lambda: None
    app.dependency_overrides[audio.verify_api_key] = lambda: None
    response = TestClient(app).post(
        "/v1/audio/transcriptions",
        data={"model": "whisper-large-v3"},
        files={"file": ("audio.wav", b"test", "audio/wav")},
    )
    error = response.json()["detail"]["error"]
    if failure in ("generic", "suppressed"):
        assert response.status_code == 500
        assert error["code"] == "transcription_failed"
    else:
        assert response.status_code == 403
        assert error["code"] == "remote_code_consent_required"
        assert error["type"] == "permission_error"
        assert error["param"] == "model"
        assert "RAPID_MLX_TRUST_REMOTE_CODE=1" in error["message"]
        assert "/model" not in error["message"].replace("auto_map/model_file", "")


def test_processor_auto_map_requires_consent(tmp_path, monkeypatch):
    monkeypatch.delenv("RAPID_MLX_TRUST_REMOTE_CODE", raising=False)
    (tmp_path / "config.json").write_text(
        json.dumps({"auto_map": {"AutoProcessor": "custom.Processor"}})
    )
    with pytest.raises(RemoteCodeConsentRequired):
        validate_processor_code(tmp_path)
