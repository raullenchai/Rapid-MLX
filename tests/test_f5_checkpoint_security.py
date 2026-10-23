# SPDX-License-Identifier: Apache-2.0
import sys
from pathlib import Path
from types import ModuleType

from rapid_mlx.audio.f5_security import load_f5_checkpoint


def test_both_f5_and_nested_vocoder_use_pinned_resolution_without_global_patch(
    tmp_path, monkeypatch
):
    from rapid_mlx import model_downloads

    main = tmp_path / "main"
    vocoder = tmp_path / "vocoder"
    main.mkdir()
    vocoder.mkdir()
    calls = []

    def resolve(name, **kwargs):
        calls.append((name, kwargs))
        return str(vocoder if "vocos" in name else main)

    monkeypatch.setattr(model_downloads, "snapshot_download", resolve)
    module = ModuleType("f5_tts_mlx.cfm")

    class FakeF5:
        @classmethod
        def from_pretrained(cls, name):
            # These names intentionally resolve against the private function's
            # injected globals, matching pinned upstream's two dependencies.
            path = fetch_from_hub(name)  # noqa: F821
            nested = Vocos.from_pretrained("lucasnewman/vocos-mel-24khz")  # noqa: F821
            return cls, path, nested

    module.F5TTS = FakeF5
    monkeypatch.setitem(sys.modules, "f5_tts_mlx.cfm", module)
    vocos = ModuleType("vocos_mlx")

    class FakeVocos:
        @staticmethod
        def from_pretrained(path):
            assert Path(path).is_dir()
            return path

    vocos.Vocos = FakeVocos
    monkeypatch.setitem(sys.modules, "vocos_mlx", vocos)
    original = FakeF5.from_pretrained.__func__
    original_globals = dict(original.__globals__)
    result = load_f5_checkpoint("org/f5-model")
    assert result == (FakeF5, main, str(vocoder))
    assert [name for name, _ in calls] == [
        "org/f5-model",
        "lucasnewman/vocos-mel-24khz",
    ]
    assert FakeF5.from_pretrained.__func__ is original
    assert "fetch_from_hub" not in original.__globals__
    assert original.__globals__ == original_globals
    calls.clear()
    assert load_f5_checkpoint(str(main))[1] == main
    assert [name for name, _ in calls] == ["lucasnewman/vocos-mel-24khz"]
