"""Audio import failures give a repair command for the running Python."""

import sys

import pytest

from rapid_mlx.connect import package_install_command


@pytest.mark.parametrize("component", ["processor", "stt", "tts"])
def test_missing_mlx_audio_uses_active_environment(monkeypatch, component):
    from rapid_mlx import model_downloads

    monkeypatch.setattr(model_downloads, "require_local", lambda _model: None)
    monkeypatch.setitem(sys.modules, "mlx_audio", None)

    if component == "processor":
        from rapid_mlx.audio.processor import AudioProcessor

        engine = AudioProcessor()
    elif component == "stt":
        from rapid_mlx.audio.stt import STTEngine

        engine = STTEngine()
    else:
        from rapid_mlx.audio.tts import TTSEngine

        engine = TTSEngine()

    with pytest.raises(ImportError) as error:
        engine.load()

    assert package_install_command("mlx-audio") in str(error.value)
