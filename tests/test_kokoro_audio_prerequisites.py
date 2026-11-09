"""Fresh-install Kokoro prerequisites, with no downloads or audio inference."""

import importlib
from unittest.mock import Mock

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from huggingface_hub import constants

from rapid_mlx import model_downloads
from rapid_mlx.audio import probe, runtime_requirements
from rapid_mlx.config import get_config
from rapid_mlx.routes.audio import router


@pytest.fixture(autouse=True)
def prerequisites(monkeypatch):
    probe._reset_g2p_model_state()
    monkeypatch.setattr(probe, "_ensure_kokoro_g2p_ready", lambda: None)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(importlib, "import_module", lambda name: object())
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    monkeypatch.delenv(model_downloads.ENV_VAR, raising=False)
    monkeypatch.setattr(model_downloads, "_cli_disabled", False)
    monkeypatch.setattr(get_config(), "api_key", None)
    monkeypatch.setattr(probe, "require_mlx_audio_tts", lambda: None)
    yield
    probe._reset_g2p_model_state()


def test_speech_request_never_installs_missing_pipeline(monkeypatch):
    # #2664: inference is check-only; preparation belongs to pull/install.
    prepare = Mock()
    monkeypatch.setattr(runtime_requirements, "prepare_runtime_requirement", prepare)
    monkeypatch.setattr(
        runtime_requirements, "spacy_pipeline_available", lambda name: False
    )
    with pytest.raises(HTTPException) as excinfo:
        probe.require_kokoro_runtime()
    assert excinfo.value.status_code == 503
    assert "rapid-mlx pull kokoro" in str(excinfo.value.detail)
    prepare.assert_not_called()


def probe_runtime_requirement():
    from rapid_mlx.audio.registry import AudioRuntimeRequirement

    return AudioRuntimeRequirement("spacy_pipeline", "en_core_web_sm")


@pytest.mark.parametrize("missing", ["misaki", "misaki.en", "spacy", "pipeline"])
def test_missing_prerequisite_returns_actionable_json_503(monkeypatch, missing):
    from fastapi import HTTPException

    def imports(name):
        if name == missing:
            raise ImportError("private path must not escape")
        return object()

    monkeypatch.setattr(importlib, "import_module", imports)
    if missing == "misaki":
        monkeypatch.setattr(importlib.util, "find_spec", lambda _: None)
    monkeypatch.setattr(
        runtime_requirements, "spacy_pipeline_available", lambda _: False
    )
    monkeypatch.setattr(
        runtime_requirements,
        "prepare_runtime_requirement",
        Mock(side_effect=runtime_requirements.AudioRuntimePreparationError("failed")),
    )
    app = FastAPI()
    app.include_router(router)
    response = TestClient(app).post(
        "/v1/audio/speech",
        json={"model": "kokoro", "input": "hello", "voice": "af_heart"},
    )
    assert response.status_code == 503
    message = response.json()["detail"]
    assert "private path" not in message
    assert "rapid-mlx pull kokoro" in message or "misaki" in message
    if missing in {"misaki", "misaki.en", "spacy"}:
        assert "misaki" in message
    with pytest.raises(HTTPException):
        probe.require_kokoro_runtime()


def test_startup_warning_is_one_line_and_never_prepares(monkeypatch, caplog):
    monkeypatch.setattr(
        importlib, "import_module", Mock(side_effect=ImportError("missing"))
    )
    prepare = Mock()
    monkeypatch.setattr(runtime_requirements, "prepare_runtime_requirement", prepare)
    probe.warn_if_kokoro_runtime_missing()
    assert len(caplog.records) == 1
    assert "misaki" in caplog.text
    assert "\n" not in caplog.records[0].getMessage()
    prepare.assert_not_called()


def test_offline_accepts_already_prepared_pipeline(monkeypatch):
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)
    monkeypatch.setattr(
        runtime_requirements, "spacy_pipeline_available", lambda _: True
    )
    installer = Mock()
    monkeypatch.setattr(runtime_requirements, "_run_installer", installer)
    runtime_requirements.prepare_runtime_requirement(probe_runtime_requirement())
    probe.require_kokoro_runtime()
    installer.assert_not_called()


def test_startup_ready_is_silent(monkeypatch, caplog):
    monkeypatch.setattr(
        runtime_requirements, "spacy_pipeline_available", lambda _: True
    )
    probe.warn_if_kokoro_runtime_missing()
    assert not caplog.records
