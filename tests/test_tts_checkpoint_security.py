# SPDX-License-Identifier: Apache-2.0
"""Legacy VoxCPM keeps tensor support without reaching upstream pickle loads."""

import json
import pickle
import sys
import types
from pathlib import Path

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_audio")
pytestmark = pytest.mark.requires_mlx

from rapid_mlx.audio.tts import TTSEngine


def _loader(monkeypatch, callback):
    generate = types.ModuleType("mlx_audio.tts.generate")
    generate.load_model = callback
    monkeypatch.setitem(sys.modules, "mlx_audio.tts.generate", generate)


def _native(monkeypatch, weights=None):
    import mlx_audio.utils as utils

    model = types.SimpleNamespace()
    model.args = types.SimpleNamespace(model_path=None)
    model.sanitize_calls = []

    def sanitize(value):
        assert model.args.model_path is None
        model.sanitize_calls.append(dict(value))
        return value

    model.sanitize = sanitize
    model.load_weights = lambda value, strict: setattr(model, "weights", dict(value))
    model.parameters = lambda: {}
    model.eval = lambda: None

    def construct(config):
        model.args.model_path = config["model_path"]
        return model

    module = types.SimpleNamespace(
        Model=construct, ModelConfig=types.SimpleNamespace(from_dict=lambda v: v)
    )
    monkeypatch.setattr(
        utils, "get_model_class", lambda **kwargs: (module, kwargs["model_type"])
    )
    monkeypatch.setattr(utils, "load_weights", lambda path: dict(weights or {}))
    monkeypatch.setattr(utils, "apply_quantization", lambda *args: None)
    return model


def _checkpoint(tmp_path, kind="voxcpm"):
    path = tmp_path / "innocent-name"
    path.mkdir()
    (path / "config.json").write_text(json.dumps({"model_type": kind}))
    (path / "audiovae.pth").write_bytes(b"legacy")
    return path


@pytest.mark.parametrize("kind", ["voxcpm", "voxcpm2"])
@pytest.mark.parametrize("remote", [False, True])
def test_legacy_tensor_sidecar_preserves_native_sanitization(
    tmp_path, monkeypatch, kind, remote
):
    from rapid_mlx import model_downloads
    from rapid_mlx.utils import torch_checkpoint

    path = _checkpoint(tmp_path, kind)
    model = _native(monkeypatch)
    tensor = pytest.importorskip("torch").tensor([1.0, 2.0])
    seen = []

    def safe_load(value):
        seen.append(value)
        return {"state_dict": {"module.conv.weight": tensor}}

    monkeypatch.setattr(torch_checkpoint, "load_tensor_checkpoint", safe_load)
    monkeypatch.setattr(
        model_downloads, "snapshot_download", lambda *a, **kw: str(path)
    )
    _loader(
        monkeypatch, lambda *a: pytest.fail("must not reach upstream sidecar loader")
    )
    engine = TTSEngine("org/innocent-name" if remote else str(path))
    engine.load()
    assert engine.model is model
    assert seen == [path / "audiovae.pth"]
    assert len(model.sanitize_calls) == 1
    assert model.weights["audio_vae.conv.weight"].tolist() == [1.0, 2.0]
    assert model.args.model_path == str(path)


@pytest.mark.parametrize("kind", ["voxcpm", "voxcpm2"])
def test_embedded_vae_does_not_deserialize_unused_sidecar(tmp_path, monkeypatch, kind):
    from rapid_mlx.utils import torch_checkpoint

    path = _checkpoint(tmp_path, kind)
    model = _native(monkeypatch, {"audio_vae.conv.weight": mx.array([3.0])})
    monkeypatch.setattr(
        torch_checkpoint,
        "load_tensor_checkpoint",
        lambda p: pytest.fail("unused pickle"),
    )
    _loader(monkeypatch, lambda *a: pytest.fail("upstream fallback"))
    engine = TTSEngine(str(path))
    engine.load()
    assert engine.model is model


def test_malicious_pickle_cannot_execute_with_unsafe_environment(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch", minversion="2.10.0")
    marker = tmp_path / "executed"

    class Sentinel:
        def __reduce__(self):
            return Path.touch, (marker,)

    path = _checkpoint(tmp_path)
    torch.save(Sentinel(), path / "audiovae.pth")
    _native(monkeypatch)
    _loader(monkeypatch, lambda *a: pytest.fail("upstream fallback"))
    monkeypatch.setenv("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    engine = TTSEngine(str(path))
    with pytest.raises(pickle.UnpicklingError):
        engine.load()
    assert not marker.exists()
    assert not engine._loaded


def test_old_torch_rejected_before_load(tmp_path, monkeypatch):
    path = _checkpoint(tmp_path)
    _native(monkeypatch)
    _loader(monkeypatch, lambda *a: pytest.fail("upstream fallback"))
    monkeypatch.setitem(
        sys.modules,
        "torch",
        types.SimpleNamespace(
            __version__="2.9.1", load=lambda *a, **k: pytest.fail("unsafe runtime")
        ),
    )
    with pytest.raises(RuntimeError, match="PyTorch >=2.10.0"):
        TTSEngine(str(path)).load()


def test_safe_checkpoint_uses_same_resolved_snapshot(tmp_path, monkeypatch):
    from rapid_mlx import model_downloads

    path = tmp_path / "snapshot"
    path.mkdir()
    (path / "config.json").write_text('{"model_type":"voxcpm2"}')
    monkeypatch.setattr(model_downloads, "snapshot_download", lambda *a, **k: str(path))
    calls = []
    model = object()

    def load(path):
        calls.append(path)
        return model

    _loader(monkeypatch, load)
    engine = TTSEngine("org/model")
    engine.load()
    engine.load()
    assert calls == [path]
    assert engine.model is model


def test_unknown_architecture_rejects_legacy_sidecar(tmp_path, monkeypatch):
    path = _checkpoint(tmp_path, "unknown")
    _loader(monkeypatch, lambda *a: pytest.fail("unsafe route"))
    with pytest.raises(ValueError, match="unrecognized TTS"):
        TTSEngine(str(path)).load()


@pytest.mark.parametrize("kind", ["voxcpm", "voxcpm2"])
def test_native_vae_sanitize_runs_once_on_legacy_tensor_weights(
    tmp_path, monkeypatch, kind
):
    import importlib
    from types import MethodType

    from rapid_mlx.utils import torch_checkpoint

    path = _checkpoint(tmp_path, kind)
    model = _native(monkeypatch)
    native = importlib.import_module(f"mlx_audio.tts.models.{kind}.{kind}").Model
    counts = []

    def sanitize_vae(weights):
        counts.append(weights)
        # A second invocation would transpose this back and fail the assertion.
        return {key: value.transpose() for key, value in weights.items()}

    model.audio_vae = types.SimpleNamespace(sanitize=sanitize_vae)
    model.sanitize = MethodType(native.sanitize, model)
    tensor = pytest.importorskip("torch").tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    monkeypatch.setattr(
        torch_checkpoint, "load_tensor_checkpoint", lambda p: {"conv.weight": tensor}
    )
    _loader(monkeypatch, lambda *a: pytest.fail("upstream loader"))
    TTSEngine(str(path)).load()
    assert len(counts) == 1
    assert model.weights["audio_vae.conv.weight"].shape == (3, 2)


@pytest.mark.parametrize("state", [None, {}, {"weight": "not a tensor"}])
def test_invalid_legacy_weights_refused(tmp_path, monkeypatch, state):
    from rapid_mlx.utils import torch_checkpoint

    path = _checkpoint(tmp_path)
    _native(monkeypatch)
    monkeypatch.setattr(torch_checkpoint, "load_tensor_checkpoint", lambda p: state)
    with pytest.raises(ValueError, match="tensor weights"):
        TTSEngine(str(path)).load()


def test_numeric_bfloat_checkpoint_and_post_load_hook(tmp_path, monkeypatch):
    import mlx_audio.utils as utils
    import torch

    path = _checkpoint(tmp_path)
    torch.save(
        {"weight": torch.tensor([1.0, 2.0], dtype=torch.bfloat16)},
        path / "audiovae.pth",
    )
    model = _native(monkeypatch)
    module, _ = utils.get_model_class(model_type="voxcpm")
    calls = []
    module.Model.post_load_hook = lambda loaded, root: calls.append(root) or loaded
    TTSEngine(str(path)).load()
    assert model.weights["audio_vae.weight"].tolist() == [1.0, 2.0]
    assert calls == [path]


@pytest.mark.parametrize("empty", [False, True])
def test_voxcpm2_safetensors_sidecar_preferred(tmp_path, monkeypatch, empty):
    from rapid_mlx.utils import torch_checkpoint

    path = _checkpoint(tmp_path, "voxcpm2")
    model = _native(monkeypatch)
    mx.save_safetensors(
        str(path / "audiovae.safetensors"), {} if empty else {"weight": mx.array([3.0])}
    )
    monkeypatch.setattr(
        torch_checkpoint,
        "load_tensor_checkpoint",
        lambda p: pytest.fail("unused legacy sidecar"),
    )
    if empty:
        with pytest.raises(ValueError, match="empty"):
            TTSEngine(str(path)).load()
    else:
        TTSEngine(str(path)).load()
        assert model.weights["audio_vae.weight"].tolist() == [3.0]


def test_native_asset_patterns_reach_the_resolved_model(tmp_path, monkeypatch):
    import fnmatch

    import mlx_audio.utils as utils

    from rapid_mlx import model_downloads

    native_patterns = [*utils.DEFAULT_ALLOW_PATTERNS, "*.wav", "vocabulary.bin"]
    monkeypatch.setattr(utils, "DEFAULT_ALLOW_PATTERNS", native_patterns)
    files = {"config.json": "{}", "reference.wav": "audio", "vocabulary.bin": "tokens"}
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    downloads = []

    def resolve(name, *, allow_patterns):
        downloads.append(name)
        assert allow_patterns == native_patterns
        for filename, content in files.items():
            if any(fnmatch.fnmatch(filename, pattern) for pattern in allow_patterns):
                (snapshot / filename).write_text(content)
        return str(snapshot)

    model = object()

    def native_load(path):
        assert path == snapshot
        assert (path / "reference.wav").read_text() == "audio"
        assert (path / "vocabulary.bin").read_text() == "tokens"
        return model

    monkeypatch.setattr(model_downloads, "snapshot_download", resolve)
    _loader(monkeypatch, native_load)
    engine = TTSEngine("org/model")
    engine.load()
    assert engine.model is model
    assert downloads == ["org/model"]


@pytest.mark.parametrize("missing", [False, True])
def test_native_strict_weight_validation_is_preserved(tmp_path, monkeypatch, missing):
    import mlx.nn as nn
    import torch

    path = _checkpoint(tmp_path)
    torch.save({"weight": torch.tensor([1.0, 2.0])}, path / "audiovae.pth")
    base_weights = {} if missing else {"projection": mx.ones((1,))}
    model = _native(monkeypatch, weights=base_weights)
    native = nn.Module()
    native["projection"] = mx.zeros((1,))
    native["audio_vae"] = {"weight": mx.zeros((2,))}
    model.load_weights = native.load_weights
    model.parameters = native.parameters
    model.eval = native.eval
    _loader(monkeypatch, lambda *a: pytest.fail("upstream sidecar fallback"))
    engine = TTSEngine(str(path))
    if missing:
        with pytest.raises(ValueError, match="Missing"):
            engine.load()
        assert not engine._loaded
    else:
        engine.load()
        assert engine._loaded
        assert native["projection"].tolist() == [1.0]
        assert native["audio_vae"]["weight"].tolist() == [1.0, 2.0]
