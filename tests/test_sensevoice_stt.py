# SPDX-License-Identifier: Apache-2.0
"""SenseVoice STT adapter and OpenAI-route contracts."""

from __future__ import annotations

import types

import pytest
from fastapi import HTTPException

from vllm_mlx.audio.registry import resolve_audio_alias, stt_aliases
from vllm_mlx.audio.stt import STTEngine
from vllm_mlx.routes.audio import (
    _reject_non_whisper_for_translation,
    _resolve_stt_model,
)


class _SenseVoiceModel:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def generate(self, audio: str, **kwargs):
        self.calls.append((audio, kwargs))
        return types.SimpleNamespace(
            text=" 你好，世界 ",
            language="zh",
            segments=[
                {
                    "text": "你好，世界",
                    "language": "zh",
                    "emotion": "neutral",
                    "event": "Speech",
                }
            ],
        )


@pytest.mark.parametrize("alias", ["sensevoice", "sensevoice-small"])
def test_sensevoice_alias_contract(alias: str) -> None:
    entry = resolve_audio_alias(alias)
    assert entry is not None
    assert entry.type == "stt"
    assert entry.family == "sensevoice"
    assert entry.hf_id == "mlx-community/SenseVoiceSmall"
    assert stt_aliases()[alias] == entry.hf_id
    assert _resolve_stt_model(alias) == entry.hf_id


def test_sensevoice_transcribe_uses_native_generate_contract() -> None:
    model = _SenseVoiceModel()
    engine = STTEngine("mlx-community/SenseVoiceSmall")
    engine.model = model
    engine._loaded = True

    result = engine.transcribe("speech.wav", language="ZH")

    assert model.calls == [("speech.wav", {"verbose": False, "language": "zh"})]
    assert result.text == "你好，世界"
    assert result.language == "zh"
    assert result.duration is None
    assert result.segments == [
        {
            "text": "你好，世界",
            "language": "zh",
            "emotion": "neutral",
            "event": "Speech",
        }
    ]


def test_sensevoice_unknown_language_degrades_to_auto() -> None:
    model = _SenseVoiceModel()
    engine = STTEngine("mlx-community/SenseVoiceSmall")
    engine.model = model
    engine._loaded = True

    engine.transcribe("speech.wav", language="es")

    assert model.calls[0][1] == {"verbose": False, "language": "auto"}


@pytest.mark.parametrize(
    "model",
    [
        "sensevoice",
        "sensevoice-small",
        "mlx-community/SenseVoiceSmall",
    ],
)
def test_translations_reject_sensevoice_before_inference(model: str) -> None:
    with pytest.raises(HTTPException) as exc_info:
        _reject_non_whisper_for_translation(model)

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail["error"]["code"] == "invalid_model_for_translation"
