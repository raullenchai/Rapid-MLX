import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from rapid_mlx import model_downloads
from rapid_mlx.audio import registry, tts


@pytest.mark.parametrize("stream", [False, True])
def test_local_kokoro_synthesis_uses_local_voice(tmp_path, monkeypatch, stream):
    folder = tmp_path / "custom-speech"
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps({"model_type": "kokoro"}))
    (folder / "voices").mkdir()
    voice_file = folder / "voices" / "af_custom.safetensors"
    voice_file.write_bytes(b"custom voice tensor")
    monkeypatch.setattr(registry, "_LOCAL_AUDIO_ENTRIES", {})

    def forbid_hub(*args, **kwargs):
        raise AssertionError("Hub access forbidden")

    monkeypatch.setattr(model_downloads, "snapshot_download", forbid_hub)
    monkeypatch.setattr("huggingface_hub.snapshot_download", forbid_hub)
    pipeline = ModuleType("mlx_audio.tts.models.kokoro.pipeline")
    pipeline.snapshot_download = forbid_hub
    used = []

    class FakePipeline:
        def __init__(self, repo_id):
            self.repo_id = repo_id

        def load_voice(self, voice):
            # Mirror lazy snapshot resolution before opening the voice tensor.
            root = pipeline.snapshot_download(
                repo_id=self.repo_id,
                allow_patterns=[f"voices/{voice}.safetensors"],
                local_files_only=True,
            )
            path = Path(root) / "voices" / f"{voice}.safetensors"
            used.append((path, path.read_bytes()))

    class FakeKokoro:
        repo_id = None

        def generate(self, text, voice, speed, **kwargs):
            source = self.repo_id if self.repo_id is not None else "default/voices"
            FakePipeline(source).load_voice(voice)
            yield SimpleNamespace(audio=np.zeros(240), sample_rate=24000)

    model = FakeKokoro()

    def load_model(path):
        assert path == folder
        assert model.repo_id is None
        return model

    kokoro = ModuleType("mlx_audio.tts.models.kokoro")
    kokoro.pipeline = pipeline
    mlx = ModuleType("mlx")
    mlx.core = SimpleNamespace(array=np.ndarray)
    for name, module in {
        "mlx": mlx,
        "mlx.core": mlx.core,
        "mlx_audio": ModuleType("mlx_audio"),
        "mlx_audio.tts": ModuleType("mlx_audio.tts"),
        "mlx_audio.tts.generate": SimpleNamespace(load_model=load_model),
        "mlx_audio.tts.models": ModuleType("mlx_audio.tts.models"),
        "mlx_audio.tts.models.kokoro": kokoro,
        "mlx_audio.tts.models.kokoro.pipeline": pipeline,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    assert tts._list_snapshot_voices(str(folder)) == ["af_custom"]
    engine = tts.TTSEngine(str(folder))
    assert engine._model_family == "kokoro"
    outputs = (
        list(engine.stream_generate("hello", voice="af_custom"))
        if stream
        else [engine.generate("hello", voice="af_custom")]
    )
    assert model.repo_id == str(folder.resolve())
    assert used == [(voice_file, b"custom voice tensor")]
    assert sum(len(output.audio) for output in outputs) == 240
    assert pipeline.snapshot_download is forbid_hub
