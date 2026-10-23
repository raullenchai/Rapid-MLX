# SPDX-License-Identifier: Apache-2.0
"""Reject legacy audio pickle before the downstream TTS loader is reached."""

import json
import pickle
import sys
import types
from pathlib import Path

import pytest

from rapid_mlx.audio.tts import TTSEngine


def _loader(monkeypatch, callback):
    for name in ("mlx_audio", "mlx_audio.tts"):
        module = types.ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    generate = types.ModuleType("mlx_audio.tts.generate")
    generate.load_model = callback
    monkeypatch.setitem(sys.modules, "mlx_audio.tts.generate", generate)


def _checkpoint(tmp_path, kind, marker):
    checkpoint = tmp_path / "innocent-name"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps({"model_type": kind}))

    class Sentinel:
        def __reduce__(self):
            return Path.touch, (marker,)

    (checkpoint / "audiovae.pth").write_bytes(pickle.dumps(Sentinel()))
    return checkpoint


@pytest.mark.parametrize("kind", ["voxcpm", "voxcpm2"])
@pytest.mark.parametrize("remote", [False, True])
@pytest.mark.parametrize(
    "torch_version,force_unsafe", [("2.5.0", None), ("2.10.0", "1")]
)
def test_legacy_vae_is_rejected_before_deserialization(
    tmp_path, monkeypatch, kind, remote, torch_version, force_unsafe
):
    from rapid_mlx import model_downloads

    marker = tmp_path / "pickle-executed"
    checkpoint = _checkpoint(tmp_path, kind, marker)
    monkeypatch.setitem(
        sys.modules, "torch", types.SimpleNamespace(__version__=torch_version)
    )
    if force_unsafe is None:
        monkeypatch.delenv("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", raising=False)
    else:
        monkeypatch.setenv("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", force_unsafe)
    resolutions = []

    def resolve(name, **kwargs):
        resolutions.append(name)
        return str(checkpoint)

    monkeypatch.setattr(model_downloads, "snapshot_download", resolve)
    calls = []

    def unsafe_loader(path):
        calls.append(path)
        with (Path(path) / "audiovae.pth").open("rb") as handle:
            return pickle.load(handle)

    _loader(monkeypatch, unsafe_loader)
    engine = TTSEngine("org/innocent-name" if remote else str(checkpoint))
    with pytest.raises(ValueError, match="Refusing legacy audiovae.pth"):
        engine.load()
    assert calls == []
    assert not marker.exists()
    assert engine._loaded is False
    assert resolutions == (["org/innocent-name"] if remote else [])


def test_unused_legacy_sidecar_is_still_rejected(tmp_path, monkeypatch):
    checkpoint = _checkpoint(tmp_path, "voxcpm2", tmp_path / "executed")
    (checkpoint / "audiovae.safetensors").write_bytes(b"unused in this preflight test")
    _loader(monkeypatch, lambda path: pytest.fail("loader must not run"))
    with pytest.raises(ValueError, match="Refusing legacy audiovae.pth"):
        TTSEngine(str(checkpoint)).load()


def test_safe_checkpoint_uses_same_resolved_snapshot(tmp_path, monkeypatch):
    from rapid_mlx import model_downloads

    checkpoint = tmp_path / "snapshot"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text('{"model_type":"voxcpm2"}')
    calls = []
    monkeypatch.setattr(
        model_downloads, "snapshot_download", lambda *a, **k: str(checkpoint)
    )
    model = object()

    def safe_loader(path):
        calls.append(path)
        return model

    _loader(monkeypatch, safe_loader)
    engine = TTSEngine("org/model")
    engine.load()
    engine.load()
    assert calls == [checkpoint]
    assert engine.model is model
    assert engine._loaded is True


def test_sidecar_symlink_is_rejected(tmp_path, monkeypatch):
    checkpoint = tmp_path / "snapshot"
    checkpoint.mkdir()
    (checkpoint / "audiovae.pth").symlink_to(tmp_path / "unavailable-pickle")
    _loader(monkeypatch, lambda path: pytest.fail("loader must not run"))
    with pytest.raises(ValueError, match="Refusing legacy audiovae.pth"):
        TTSEngine(str(checkpoint)).load()
