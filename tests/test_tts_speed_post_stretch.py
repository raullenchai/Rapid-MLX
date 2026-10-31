import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from rapid_mlx.audio.tts import TTSEngine


@pytest.mark.parametrize(
    "model,stretched",
    [
        ("mlx-community/VoxCPM2-8bit", True),
        ("mlx-community/VibeVoice-Realtime-0.5B-4bit", True),
        ("mlx-community/Kokoro-82M-bf16", False),
    ],
)
@pytest.mark.parametrize("stream", [False, True])
def test_speed(model, stretched, stream, monkeypatch):
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
