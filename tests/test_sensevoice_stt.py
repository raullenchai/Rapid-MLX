# SPDX-License-Identifier: Apache-2.0
"""SenseVoice STT adapter and OpenAI-route contracts."""

from __future__ import annotations

import types

import pytest

pytestmark = pytest.mark.requires_mlx
from fastapi import HTTPException

from rapid_mlx.audio.registry import resolve_audio_alias, stt_aliases
from rapid_mlx.audio.stt import STTEngine
from rapid_mlx.routes.audio import (
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


def test_sensevoice_direct_translation_fails_instead_of_silently_transcribing() -> None:
    engine = STTEngine("mlx-community/SenseVoiceSmall")
    engine.model = _SenseVoiceModel()
    engine._loaded = True

    with pytest.raises(ValueError, match="Whisper"):
        engine.transcribe("speech.wav", task="translate")


def test_sensevoice_backend_detection_uses_loaded_model_metadata(
    monkeypatch, tmp_path
) -> None:
    model = _SenseVoiceModel()
    model.config = types.SimpleNamespace(model_type="sensevoice")
    monkeypatch.setattr(
        "mlx_audio.stt.utils.load_model", lambda _name, **_kwargs: model
    )
    monkeypatch.setattr(
        "rapid_mlx.model_downloads.snapshot_download", lambda *a, **kw: str(tmp_path)
    )

    engine = STTEngine("acme/private-asr")
    engine.load()
    engine.transcribe("speech.wav", language="zh")

    assert engine._is_sensevoice is True
    assert model.calls == [("speech.wav", {"verbose": False, "language": "zh"})]


def test_loaded_model_metadata_overrides_false_positive_name(
    monkeypatch, tmp_path
) -> None:
    model = _SenseVoiceModel()
    model.config = types.SimpleNamespace(model_type="whisper")
    monkeypatch.setattr(
        "mlx_audio.stt.utils.load_model", lambda _name, **_kwargs: model
    )
    monkeypatch.setattr(
        "rapid_mlx.model_downloads.snapshot_download", lambda *a, **kw: str(tmp_path)
    )

    engine = STTEngine("acme/not-really-sensevoice")
    engine.load()

    assert engine._is_sensevoice is False


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


@pytest.mark.parametrize("name", ["parakeet", "mlx-community/parakeet-tdt-0.6b-v2"])
def test_parakeet_native_dispatch_with_custom_snapshot_cache(
    monkeypatch, tmp_path, name
):
    """Use real installed dispatch, with no architecture hint and no weights download."""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_audio.stt.models import parakeet

    snapshot = (
        tmp_path
        / "custom-cache"
        / "models--mlx-community--parakeet-tdt-0.6b-v2"
        / "snapshots"
        / ("a" * 40)
    )
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    mx.save_safetensors(str(snapshot / "model.safetensors"), {"weight": mx.zeros((1,))})
    loaded_configs = []

    class TinyNativeModel(nn.Module):
        def __init__(self, config):
            super().__init__()
            loaded_configs.append(config)
            self.weight = mx.zeros((1,))

    monkeypatch.setattr(parakeet, "Model", TinyNativeModel)
    monkeypatch.setattr(
        parakeet, "ModelConfig", types.SimpleNamespace(from_dict=lambda config: config)
    )
    downloads = []

    def verified_snapshot(repository):
        downloads.append(repository)
        return str(snapshot)

    monkeypatch.setattr(
        "rapid_mlx.model_downloads.snapshot_download", verified_snapshot
    )
    monkeypatch.setattr(
        "mlx_audio.utils.get_model_path",
        lambda *a, **kw: pytest.fail("native loader must keep verified Path"),
    )
    engine = STTEngine(name)
    engine.load()
    assert isinstance(engine.model, TinyNativeModel)
    assert loaded_configs[0]["model_path"] == str(snapshot)
    assert downloads == [name]


def test_local_stt_directory_keeps_native_path_inference(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        "mlx_audio.stt.utils.load_model",
        lambda path, **kwargs: calls.append((path, kwargs)) or types.SimpleNamespace(),
    )
    monkeypatch.setattr(
        "rapid_mlx.model_downloads.snapshot_download",
        lambda *a, **kw: pytest.fail("local directory must not download"),
    )
    STTEngine(str(tmp_path)).load()
    assert calls == [(tmp_path, {})]
