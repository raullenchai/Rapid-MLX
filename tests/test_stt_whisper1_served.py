# SPDX-License-Identifier: Apache-2.0
"""The OpenAI STT placeholder follows the server's primary STT model."""

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from rapid_mlx.routes import audio


def _serve(monkeypatch, alias):
    monkeypatch.setattr(
        "rapid_mlx.config.get_config",
        lambda: SimpleNamespace(model_alias=alias, model_name=None, model_path=None),
    )


def test_whisper1_uses_served_stt_and_explicit_alias_stays_explicit(monkeypatch):
    _serve(monkeypatch, "nemotron-asr")
    assert (
        audio._resolve_stt_model("whisper-1") == audio.STT_MODEL_ALIASES["nemotron-asr"]
    )
    assert (
        audio._resolve_stt_model("whisper-small")
        == audio.STT_MODEL_ALIASES["whisper-small"]
    )


def test_whisper1_on_chat_server_keeps_alias(monkeypatch):
    _serve(monkeypatch, "qwen3.5-4b-4bit")
    assert (
        audio._resolve_stt_model("whisper-1")
        == audio.STT_MODEL_ALIASES["whisper-large-v3"]
    )


def test_whisper1_guards_classify_served_model(monkeypatch):
    _serve(monkeypatch, "nemotron-asr")
    with pytest.raises(HTTPException) as translation:
        audio._reject_non_whisper_for_translation("whisper-1")
    assert translation.value.detail["error"]["code"] == "invalid_model_for_translation"
    with pytest.raises(HTTPException) as timestamps:
        audio._reject_word_timestamps_for_non_whisper("whisper-1", ["word"])
    assert timestamps.value.status_code == 400
