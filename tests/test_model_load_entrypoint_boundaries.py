# SPDX-License-Identifier: Apache-2.0
"""Platform-neutral coverage for typed per-model load call sites."""

from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from rapid_mlx.model_load_errors import IncompatibleWeights, QuantizationMismatch

_ROOT = Path(__file__).parents[1]


def _load_source(monkeypatch, module_name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(module_name, _ROOT / relative_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def _install_fake_mlx(monkeypatch, *, quantize, load_weights):
    mlx = ModuleType("mlx")
    mlx.__path__ = []
    mx = ModuleType("mlx.core")
    nn = ModuleType("mlx.nn")
    utils = ModuleType("mlx.utils")

    class Module:
        def load_weights(self, items, *, strict):
            return load_weights(self, items, strict=strict)

    class Linear(Module):
        pass

    def unavailable_device_info():
        raise ValueError("device information unavailable in the fake runtime")

    mx.device_info = unavailable_device_info
    mx.eval = lambda *_args, **_kwargs: None
    mx.load = lambda _path: {}
    mx.set_wired_limit = lambda _limit: None
    nn.Linear = Linear
    nn.Module = Module
    nn.quantize = quantize
    utils.tree_flatten = lambda tree: list(tree.items())
    mlx.core = mx
    mlx.nn = nn
    mlx.utils = utils

    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", mx)
    monkeypatch.setitem(sys.modules, "mlx.nn", nn)
    monkeypatch.setitem(sys.modules, "mlx.utils", utils)
    return mx, nn


def _load_fake_fp8_module(monkeypatch, *, quantize, load_weights):
    mx, nn = _install_fake_mlx(
        monkeypatch, quantize=quantize, load_weights=load_weights
    )
    module = _load_source(
        monkeypatch,
        "rapid_mlx._test_fp8_repack_boundaries",
        "rapid_mlx/fp8_repack.py",
    )
    return module, mx, nn


@pytest.mark.parametrize(
    ("failure", "error_type"),
    [
        ("checkpoint_quantize", QuantizationMismatch),
        ("load_weights", IncompatibleWeights),
        ("lm_head_quantize", QuantizationMismatch),
    ],
)
def test_fp8_entrypoint_types_real_call_site_failures(
    tmp_path, monkeypatch, failure, error_type
) -> None:
    quantize_calls = []

    def quantize(*args, **kwargs):
        quantize_calls.append((args, kwargs))
        call = len(quantize_calls)
        if failure == "checkpoint_quantize" and call == 1:
            raise ValueError("checkpoint quantization mismatch")
        if failure == "lm_head_quantize" and call == 2:
            raise ValueError("lm_head quantization mismatch")

    def load_weights(_model, _items, *, strict):
        assert strict is True
        if failure == "load_weights":
            raise ValueError("checkpoint weights do not fit")

    fp8, _mx, nn = _load_fake_fp8_module(
        monkeypatch, quantize=quantize, load_weights=load_weights
    )

    class Model(nn.Module):
        def __init__(self):
            self.lm_head = nn.Linear()

        def parameters(self):
            return {}

        def eval(self):
            return self

    model = Model()
    arch = SimpleNamespace(
        Model=lambda _args: model,
        ModelArgs=SimpleNamespace(from_dict=lambda _config: object()),
    )
    monkeypatch.setattr(fp8.importlib, "import_module", lambda _name: arch)
    monkeypatch.setattr(fp8, "_open_shards", lambda _path: ({}, {}))
    monkeypatch.setenv("RAPID_MLX_FP8_LM_HEAD_AFFINE8", "1")
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "fake_fp8",
                "quantization_config": {
                    "quant_method": "fp8",
                    "fmt": "e4m3",
                    "scale_fmt": "ue8m0",
                    "weight_block_size": [32, 32],
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(error_type) as raised:
        fp8.load_fp8_model_online(tmp_path)

    assert isinstance(raised.value.__cause__, ValueError)


def test_fp8_entrypoint_returns_after_all_checked_calls(tmp_path, monkeypatch) -> None:
    calls = []

    def quantize(*_args, **_kwargs):
        calls.append("quantize")

    def load_weights(_model, items, *, strict):
        assert items == []
        assert strict is True
        calls.append("load_weights")

    fp8, _mx, nn = _load_fake_fp8_module(
        monkeypatch, quantize=quantize, load_weights=load_weights
    )

    class Model(nn.Module):
        def __init__(self):
            self.lm_head = nn.Linear()
            self.evaluated = False

        def parameters(self):
            return {}

        def eval(self):
            self.evaluated = True
            return self

    model = Model()
    arch = SimpleNamespace(
        Model=lambda _args: model,
        ModelArgs=SimpleNamespace(from_dict=lambda _config: object()),
    )
    monkeypatch.setattr(fp8.importlib, "import_module", lambda _name: arch)
    monkeypatch.setattr(fp8, "_open_shards", lambda _path: ({}, {}))
    monkeypatch.setenv("RAPID_MLX_FP8_LM_HEAD_AFFINE8", "1")
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "fake_fp8",
                "quantization_config": {
                    "quant_method": "fp8",
                    "fmt": "e4m3",
                    "scale_fmt": "ue8m0",
                    "weight_block_size": [32, 32],
                },
            }
        ),
        encoding="utf-8",
    )

    assert fp8.load_fp8_model_online(tmp_path) is model
    assert calls == ["quantize", "load_weights", "quantize"]
    assert model.evaluated is True


def _load_fake_deepseek_module(monkeypatch, *, quantize, load_weights):
    mx, nn = _install_fake_mlx(
        monkeypatch, quantize=quantize, load_weights=load_weights
    )
    importlib.import_module("rapid_mlx.models.deepseek_v41_native")

    config = ModuleType("rapid_mlx.models.deepseek_v41_native.config")
    convert = ModuleType("rapid_mlx.models.deepseek_v41_native.convert")
    model = ModuleType("rapid_mlx.models.deepseek_v41_native.model")
    config.ModelArgs = type("ModelArgs", (), {})
    convert.VISION_PREFIXES = ("vision.", "aligner.", "image_")
    convert.bits_for = lambda _name, bits, _expert_bits: bits
    convert.is_quant_target = lambda *_args: True
    model.Model = type("Model", (), {})
    monkeypatch.setitem(sys.modules, config.__name__, config)
    monkeypatch.setitem(sys.modules, convert.__name__, convert)
    monkeypatch.setitem(sys.modules, model.__name__, model)

    module = _load_source(
        monkeypatch,
        "rapid_mlx.models.deepseek_v41_native._test_load_boundaries",
        "rapid_mlx/models/deepseek_v41_native/load.py",
    )
    return module, mx, nn


@pytest.mark.parametrize(
    ("failure", "error_type"),
    [
        ("quantize", QuantizationMismatch),
        ("load_weights", IncompatibleWeights),
    ],
)
def test_deepseek_entrypoint_types_real_call_site_failures(
    tmp_path, monkeypatch, failure, error_type
) -> None:
    def quantize(*_args, **_kwargs):
        if failure == "quantize":
            raise ValueError("native quantization mismatch")

    def load_weights(_model, _items, *, strict):
        assert strict is False
        if failure == "load_weights":
            raise ValueError("native weights do not fit")

    native_load, mx, nn = _load_fake_deepseek_module(
        monkeypatch, quantize=quantize, load_weights=load_weights
    )
    weight = object()
    mx.load = lambda _path: {"weight": weight}

    class Model(nn.Module):
        def __init__(self, _args, token_map=None):
            assert token_map is None

        def parameters(self):
            return {"weight": weight}

        def eval(self):
            return self

    args = SimpleNamespace(engram_layer_ids=[], n_layers=0, o_groups=1, o_lora_rank=1)
    native_load.ModelArgs.from_dict = lambda _config: args
    native_load.Model = Model
    (tmp_path / "config.json").write_text(
        json.dumps({"quantization": {"group_size": 32, "bits": 4}}),
        encoding="utf-8",
    )
    (tmp_path / "model.safetensors").write_bytes(b"fixture")

    with pytest.raises(error_type) as raised:
        native_load.load(str(tmp_path), lazy=True)

    assert isinstance(raised.value.__cause__, ValueError)


def test_deepseek_entrypoint_returns_after_checked_calls(tmp_path, monkeypatch) -> None:
    calls = []

    def quantize(*_args, **_kwargs):
        calls.append("quantize")

    def load_weights(_model, items, *, strict):
        assert len(items) == 1 and items[0][0] == "weight"
        assert strict is False
        calls.append("load_weights")

    native_load, mx, nn = _load_fake_deepseek_module(
        monkeypatch, quantize=quantize, load_weights=load_weights
    )
    weight = object()
    mx.load = lambda _path: {"weight": weight}

    class Model(nn.Module):
        def __init__(self, _args, token_map=None):
            assert token_map is None
            self.evaluated = False

        def parameters(self):
            return {"weight": weight}

        def eval(self):
            self.evaluated = True
            return self

    args = SimpleNamespace(engram_layer_ids=[], n_layers=0, o_groups=1, o_lora_rank=1)
    native_load.ModelArgs.from_dict = lambda _config: args
    native_load.Model = Model
    (tmp_path / "config.json").write_text(
        json.dumps({"quantization": {"group_size": 32, "bits": 4}}),
        encoding="utf-8",
    )
    (tmp_path / "model.safetensors").write_bytes(b"fixture")

    loaded, loaded_args = native_load.load(str(tmp_path), lazy=True)

    assert loaded is not None and loaded_args is args
    assert calls == ["quantize", "load_weights"]
    assert loaded.evaluated is True
