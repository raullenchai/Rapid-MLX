"""Freeze effective eager dispatch before moving its policy into the lattice."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def eager(monkeypatch):
    # A fresh module captures the switch without disturbing the installed hook.
    path = Path(__file__).parents[1] / "rapid_mlx/patches/qwen3_5_eager_dispatch.py"
    spec = importlib.util.spec_from_file_location("rapid_mlx.patches._gate_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Layer:
        input_layernorm = SimpleNamespace(weight=SimpleNamespace(shape=(2048,)))
        mlp = SimpleNamespace(num_experts=256, top_k=8)

        def __call__(self, output):
            return output

    submitted = []
    mx = ModuleType("mlx.core")
    mx.async_eval = submitted.append
    package = ModuleType("mlx")
    package.core = mx
    models = ModuleType("mlx_lm.models")
    models.qwen3_5 = SimpleNamespace(DecoderLayer=Layer)
    precision = ModuleType("rapid_mlx.compiled_precision")
    precision.in_compiled_decode = lambda: False
    for name, value in {
        "mlx": package,
        "mlx.core": mx,
        "mlx_lm.models": models,
        "rapid_mlx.compiled_precision": precision,
    }.items():
        monkeypatch.setitem(sys.modules, name, value)
    module.install_qwen3_5_eager_dispatch()
    return module, Layer, submitted


@pytest.mark.parametrize("hidden", [2047, 2048, 2049])
@pytest.mark.parametrize("experts", [255, 256, 257])
@pytest.mark.parametrize("top_k", [7, 8, 9])
def test_eager_geometry(eager, hidden, experts, top_k, monkeypatch):
    """rapid_mlx/patches/qwen3_5_eager_dispatch.py:39 pins exact geometry."""
    module, Layer, submitted = eager
    monkeypatch.setattr(module, "_ENABLED", True)
    layer = Layer()
    layer.input_layernorm = SimpleNamespace(weight=SimpleNamespace(shape=(hidden,)))
    layer.mlp = SimpleNamespace(num_experts=experts, top_k=top_k)
    output = SimpleNamespace(shape=(1, 1, 8))
    assert layer(output) is output
    assert bool(submitted) == (hidden == 2048 and experts == 256 and top_k == 8)


@pytest.mark.parametrize(
    "shape,selected",
    [
        ((1, 0, 8), True),
        ((1, 1, 8), True),
        ((1, 64, 8), True),
        ((1, 65, 8), False),
        ((8, 8, 8), True),
        ((5, 13, 8), False),
        ((8,), False),
        ((), False),
    ],
)
@pytest.mark.parametrize("compiled", [False, True])
def test_eager_rows_and_compiled_exclusion(
    eager, monkeypatch, shape, selected, compiled
):
    """rapid_mlx/patches/qwen3_5_eager_dispatch.py:80 pins flattened rows."""
    module, Layer, submitted = eager
    monkeypatch.setattr(module, "_ENABLED", True)
    monkeypatch.setattr(module, "_in_compiled_decode", lambda: compiled)
    output = SimpleNamespace(shape=shape)
    assert Layer()(output) is output
    assert bool(submitted) == (selected and not compiled)


@pytest.mark.parametrize(
    "raw,enabled",
    [
        (None, True),
        ("", True),
        ("0", False),
        (" false ", False),
        ("OFF", False),
        ("disabled", False),
        ("1", True),
        ("garbage", True),
    ],
)
def test_eager_import_time_switch(monkeypatch, raw, enabled):
    """rapid_mlx/patches/qwen3_5_eager_dispatch.py:36 captures falsey at import."""
    key = "RAPID_MLX_QWEN35_EAGER_LAYER_DISPATCH"
    if raw is None:
        monkeypatch.delenv(key, raising=False)
    else:
        monkeypatch.setenv(key, raw)
    module, Layer, submitted = eager.__wrapped__(monkeypatch)
    monkeypatch.setenv(key, "1" if not enabled else "0")
    output = SimpleNamespace(shape=(1, 1, 8))
    assert Layer()(output) is output
    assert module._ENABLED is enabled
    assert bool(submitted) is enabled


@pytest.mark.parametrize("missing", ["input_layernorm", "mlp"])
def test_eager_missing_facts_decline(eager, monkeypatch, missing):
    """rapid_mlx/patches/qwen3_5_eager_dispatch.py:39 declines absent layer facts."""
    module, Layer, submitted = eager
    monkeypatch.setattr(module, "_ENABLED", True)
    monkeypatch.delattr(Layer, missing)
    output = SimpleNamespace(shape=(1, 1, 8))
    assert Layer()(output) is output
    assert submitted == []
