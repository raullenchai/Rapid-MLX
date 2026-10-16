"""Post-generation tempo for TTS families whose model ignores ``speed``.

Uses fake models only, so it runs without MLX (Linux coverage included).
"""

from __future__ import annotations

import sys
import types

import pytest

CUSTOMVOICE_BF16 = "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16"


@pytest.mark.parametrize("family", ["qwen3_tts", "chatterbox", "indextts", "kokoro"])
@pytest.mark.parametrize("speed", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("stream", [False, True])
def test_family_speed_handling(monkeypatch, family, speed, stream):
    """Stretch assembled chunks once; preserve native speed kwargs/handling."""
    import numpy as np

    from rapid_mlx.audio.tts import TTSEngine

    sample_rate = 24000
    source = np.sin(2 * np.pi * 220 * np.arange(sample_rate) / sample_rate).astype(
        np.float32
    )
    calls = []

    class FakeModel:
        def generate(self, **kwargs):
            calls.append(kwargs)
            # Simulate native duration control for Kokoro only.
            audio = (
                np.resize(source, round(len(source) / speed))
                if family == "kokoro"
                else source
            )
            for chunk in np.array_split(audio, 3):
                yield types.SimpleNamespace(audio=chunk, sample_rate=sample_rate)

    engine = TTSEngine(CUSTOMVOICE_BF16)
    engine._model_family = family
    engine._loaded = True
    engine.model = FakeModel()
    if stream and family == "qwen3_tts" and speed == 1.0:
        engine._loaded = False
        monkeypatch.setattr(engine, "load", lambda: setattr(engine, "_loaded", True))
    # IndexTTS requires a reference in generate(); isolate decoding from tempo.
    fake_generate = types.ModuleType("mlx_audio.tts.generate")
    fake_generate.load_audio = lambda *args, **kwargs: source
    monkeypatch.setitem(sys.modules, "mlx_audio.tts.generate", fake_generate)
    # generate() imports mlx.core only for an isinstance check; stub it so
    # this runs on hosts without MLX (Linux coverage).
    fake_mlx = types.ModuleType("mlx")
    fake_mlx.core = types.ModuleType("mlx.core")
    fake_mlx.core.array = type("array", (), {})
    monkeypatch.setitem(sys.modules, "mlx", fake_mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mlx.core)
    if stream:
        outputs = list(engine.stream_generate("hello", speed=speed))
    else:
        outputs = [engine.generate("hello", speed=speed, ref_audio="reference.wav")]
    (call,) = calls
    if family == "kokoro":
        assert call["speed"] == speed
    else:
        assert "speed" not in call
    expected = len(source) / speed
    assert sum(len(out.audio) for out in outputs) == round(expected)
    assert sum(out.duration for out in outputs) == pytest.approx(expected / sample_rate)
    if stream and family != "kokoro" and speed != 1.0:
        # Post-stretched streams are buffered and stretched once.
        assert len(outputs) == 1
    if stream and (family == "kokoro" or speed == 1.0):
        assert engine._loaded
        assert len(outputs) == 3
        if family != "kokoro":
            for output, chunk in zip(outputs, np.array_split(source, 3), strict=True):
                np.testing.assert_array_equal(output.audio, chunk)


def test_post_stretch_stream_rejects_mixed_sample_rates():
    import numpy as np

    from rapid_mlx.audio.tts import TTSEngine

    class FakeModel:
        def generate(self, **kwargs):
            for rate in (24000, 16000):
                yield types.SimpleNamespace(
                    audio=np.zeros(4800, dtype=np.float32), sample_rate=rate
                )

    engine = TTSEngine(CUSTOMVOICE_BF16)
    engine._model_family = "qwen3_tts"
    engine._loaded = True
    engine.model = FakeModel()
    with pytest.raises(ValueError, match="sample rate"):
        list(engine.stream_generate("hello", speed=2.0))
