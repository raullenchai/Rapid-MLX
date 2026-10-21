"""Audio import failures give a repair command for the running Python."""

import builtins

import pytest

from rapid_mlx.connect import package_install_command


@pytest.mark.parametrize("component", ["processor", "stt", "tts"])
def test_missing_mlx_audio_uses_active_environment(monkeypatch, component):
    from rapid_mlx import model_downloads

    monkeypatch.setattr(model_downloads, "require_local", lambda _model: None)
    original_import = builtins.__import__
    blocked_imports = []

    def missing_audio(name, *args, **kwargs):
        if name.startswith("mlx_audio"):
            blocked_imports.append(name)
            raise ImportError("mlx_audio unavailable for this test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_audio)

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

    assert blocked_imports
    assert package_install_command("mlx-audio") in str(error.value)
