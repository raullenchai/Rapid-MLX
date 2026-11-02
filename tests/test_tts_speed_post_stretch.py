import json
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from rapid_mlx.audio.tts import TTSEngine, detect_tts_family


@pytest.mark.parametrize(
    "model,is_dia",
    [
        ("acme/nvidia-tts", False),
        ("acme/media-voice", False),
        ("acme/indiana-voice", False),
        ("nvidia/tts-model", False),
        ("/Users/x/media/tts-model", False),
        ("/Users/x/dia/tts-model", False),
        ("dia/tts-model", False),
        ("nari-labs/Dia-1.6B", True),
        ("mlx-community/Dia-1.6B-fp16", True),
        ("acme/voice_dia_fp16", True),
        ("acme/voice.dia.fp16", True),
        ("/Users/x/models/Dia-1.6B", True),
    ],
)
def test_dia_family_token_boundaries(model, is_dia):
    assert (detect_tts_family(model) == "dia") is is_dia


@pytest.mark.parametrize(
    "model,stretched",
    [
        ("mlx-community/VoxCPM2-8bit", True),
        ("mlx-community/VibeVoice-Realtime-0.5B-4bit", True),
        ("mlx-community/Dia-1.6B-4bit", True),
        ("dia", True),
        ("local-dia", True),
        ("mlx-community/Kokoro-82M-bf16", False),
    ],
)
@pytest.mark.parametrize("stream", [False, True])
def test_speed(model, stretched, stream, monkeypatch, tmp_path):
    if model == "local-dia":
        from rapid_mlx.audio import registry

        folder = tmp_path / "nvidia-tts"
        folder.mkdir()
        (folder / "config.json").write_text(json.dumps({"model_type": "dia"}))
        model = str(folder)
        entry = registry.resolve_local_audio_folder(model)
        assert entry is not None
        assert entry.notes == "local folder"
        assert entry.family == "dia"
        monkeypatch.setattr(registry, "_LOCAL_AUDIO_ENTRIES", {})
        registry.register_served_local_audio_entry(entry, model)
        assert registry.resolve_audio_alias(model) == entry
        assert detect_tts_family(model) == "dia"

    mlx = ModuleType("mlx")
    mlx.__path__ = []
    core = ModuleType("mlx.core")
    core.array = np.ndarray
    mlx.core = core
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    monkeypatch.setattr(
        TTSEngine,
        "_generate_results",
        lambda self, kwargs: self.model.generate(**kwargs),
    )

    class Backend:
        def __init__(self):
            self.calls = []

        def generate(self, **kwargs):
            self.calls.append(kwargs)
            audio = np.sin(np.arange(48000) * 2 * np.pi * 440 / 24000).astype(
                np.float32
            )
            yield SimpleNamespace(audio=audio, sample_rate=24000)

    engine = TTSEngine(model)
    engine.model = Backend()
    engine._loaded = True
    output = (
        list(engine.stream_generate("hello", speed=2.0))
        if stream
        else [engine.generate("hello", speed=2.0)]
    )
    assert sum(item.duration for item in output) == pytest.approx(
        1 if stretched else 2, rel=0.1
    )
    assert ("speed" in engine.model.calls[0]) is not stretched
