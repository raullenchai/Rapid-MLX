"""Music weight download failures identify the running Python environment."""

import builtins
import importlib.util
import sys
import types

import pytest

from rapid_mlx.audio import music
from rapid_mlx.connect import package_install_command


def test_missing_misaki_for_kokoro(monkeypatch):
    from rapid_mlx.audio import probe

    class FakeHTTPError(Exception):
        def __init__(self, status_code, detail):
            self.status_code = status_code
            self.detail = detail

    fastapi = types.ModuleType("fastapi")
    fastapi.HTTPException = FakeHTTPError
    monkeypatch.setitem(sys.modules, "fastapi", fastapi)

    original_find_spec = importlib.util.find_spec

    def missing_misaki(name, *args, **kwargs):
        if name == "misaki":
            return None
        return original_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", missing_misaki)

    with pytest.raises(FakeHTTPError) as error:
        probe.require_kokoro_runtime()

    assert error.value.status_code == 503
    assert package_install_command("misaki") in error.value.detail


def test_missing_hub_dependency_for_music_weights(monkeypatch, tmp_path):
    monkeypatch.setattr(music, "_SA3_MLX_DIR", tmp_path)
    original_import = builtins.__import__

    def missing_hub(name, *args, **kwargs):
        if name == "huggingface_hub":
            raise ImportError("hub unavailable for this test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_hub)

    with pytest.raises(RuntimeError) as error:
        music.MusicEngine()._ensure_weights()

    assert package_install_command("huggingface_hub") in str(error.value)
