"""Gemma expert layout guards and selected affine-4 decode numerics."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tests.headless_mlx.test_lattice_gate_characterization import Tensor, fake_mx

mx = fake_mx()
# Load an isolated copy so metadata-only checks run even on hosts where
# importing the native runtime itself requires an unavailable Metal device.
spec = importlib.util.spec_from_file_location(
    "rapid_mlx.patches._gemma4_contract",
    Path(__file__).parents[1] / "rapid_mlx/patches/gemma4_affine4.py",
)
fused = importlib.util.module_from_spec(spec)
# Import the real dependencies first: patch.dict drops modules first imported
# inside it, which would leave later tests with duplicate Lattice modules.
importlib.import_module("rapid_mlx._env")
importlib.import_module("rapid_mlx.lattice.fusion")
mlx = MagicMock()
mlx.core = mx
with patch.dict(sys.modules, {"mlx": mlx, "mlx.core": mx, "mlx.nn": mlx.nn}):
    spec.loader.exec_module(fused)


@pytest.fixture(autouse=True)
def runtime(monkeypatch):
    monkeypatch.setattr(fused, "mx", mx)


class Bank(dict):
    def __init__(self, shape):
        super().__init__()
        self.bits, self.group_size, self.mode = 4, 64, "affine"
        self.weight = SimpleNamespace(shape=shape, dtype=mx.uint32)
        self.scales = SimpleNamespace(
            shape=(*shape[:-1], shape[-1] // 8), dtype=mx.bfloat16
        )
        self.biases = SimpleNamespace(shape=self.scales.shape, dtype=mx.bfloat16)


def fake_block(monkeypatch, combined):
    class Activation:
        pass

    class Switch:
        pass

    class Experts:
        def __call__(self, x, ids, scores):
            return "stock"

    monkeypatch.setattr(
        fused, "_families", lambda: ((Experts, Switch, Activation, Bank),)
    )
    block = Experts()
    sm = block.switch_glu = Switch()
    sm.activation = Activation()
    sm.down_proj = Bank((128, 2816, 88))
    if combined:
        sm.gate_up_proj = Bank((128, 1408, 352))
    else:
        sm.gate_proj = Bank((128, 704, 352))
        sm.up_proj = Bank((128, 704, 352))
    return block


@pytest.mark.parametrize("combined", [False, True])
@pytest.mark.parametrize(
    "mutation",
    [
        None,
        "class",
        "activation",
        "bits",
        "group",
        "mode",
        "weight",
        "scales",
        "biases",
        "dtype",
        "bias",
        "shard",
        "switch",
        "projection",
    ],
)
def test_strict_layout(monkeypatch, combined, mutation):
    b = fake_block(monkeypatch, combined)
    sm = b.switch_glu
    p = sm.down_proj
    assert fused._compatible(b)
    if mutation == "class":
        b = SimpleNamespace(switch_glu=sm)
    elif mutation == "activation":
        sm.activation = object()
    elif mutation == "bits":
        p.bits = 8
    elif mutation == "group":
        p.group_size = 128
    elif mutation == "mode":
        p.mode = "mxfp4"
    elif mutation == "weight":
        p.weight.shape = (128, 2816, 64)
    elif mutation == "scales":
        p.scales.shape = (128, 2816, 10)
    elif mutation == "biases":
        p.biases = None
    elif mutation == "dtype":
        p.scales.dtype = mx.float16
    elif mutation == "bias":
        p["bias"] = True
    elif mutation == "shard":
        sm.sharding_group = object()
    elif mutation == "switch":
        b.switch_glu = object()
    elif mutation == "projection":
        sm.down_proj = SimpleNamespace(**p.__dict__)
    assert fused._compatible(b) == (mutation is None)


@pytest.mark.parametrize("width", [0, 1, 2, 5, 9, 10])
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("dtype", ["bf16", "fp16", "fp32"])
def test_dispatch_and_fallback(monkeypatch, width, batch, dtype):
    b = fake_block(monkeypatch, True)
    b._rapid_gemma4_affine4 = True
    fused._patch_class(type(b))
    monkeypatch.setattr(fused, "forward", lambda *_: "fused")
    x = Tensor((batch, width, 2816), dtype)
    ids = Tensor((batch, width, 8), mx.uint32)
    scores = Tensor(ids.shape)
    assert b(x, ids, scores) == (
        "fused" if batch == 1 and 1 <= width <= 9 and dtype == "bf16" else "stock"
    )
    monkeypatch.setenv("RAPID_MLX_GEMMA4_AFFINE4", "0")
    assert b(x, ids, scores) == "stock"


def test_dispatch_rechecks_layout_and_routes(monkeypatch):
    b = fake_block(monkeypatch, True)
    b._rapid_gemma4_affine4 = True
    fused._patch_class(type(b))
    monkeypatch.setattr(fused, "forward", lambda *_: "fused")
    x, ids, scores = (
        Tensor((1, 1, 2816)),
        Tensor((1, 1, 8), mx.uint32),
        Tensor((1, 1, 8)),
    )
    assert b(x, ids, scores) == "fused"
    assert b(x, Tensor((1, 1, 4), mx.uint32), scores) == "stock"
    assert b(x, ids, Tensor((1, 1, 8), "fp32")) == "stock"
    b.switch_glu.down_proj.bits = 8
    assert b(x, ids, scores) == "stock"


@pytest.mark.parametrize("probe", [False, True, "error"])
@pytest.mark.parametrize("enabled", [False, True])
def test_installer_probe_policy_and_lifetime(monkeypatch, caplog, probe, enabled):
    import gc
    import weakref
    from unittest.mock import Mock

    b = fake_block(monkeypatch, False)
    monkeypatch.setattr(fused, "install_enabled", lambda *a, **k: enabled)
    check = (
        Mock(side_effect=ValueError("decline"))
        if probe == "error"
        else Mock(return_value=probe)
    )
    monkeypatch.setattr(fused, "_probe", check)
    # Captured log tracebacks retain locals until pytest finishes the phase.
    monkeypatch.setattr(fused.logger, "warning", lambda *a, **k: None)
    model = SimpleNamespace(named_modules=lambda block=b: [("experts", block)])
    accepted = enabled and probe is True
    assert fused.install_gemma4_affine4(model) == int(accepted)
    assert getattr(b, "_rapid_gemma4_affine4", False) == accepted
    if accepted:
        assert fused.install_gemma4_affine4(model) == 0
        assert check.call_count == 1
        sibling = type(b)()
        assert not getattr(sibling, "_rapid_gemma4_affine4", False)
        assert sibling(None, None, None) == "stock"
    check.reset_mock()
    check.side_effect = None
    caplog.clear()
    ref = weakref.ref(b)
    del b, model
    gc.collect()
    assert ref() is None


def test_lattice_scope_overrides_and_kill_switch(monkeypatch):
    from rapid_mlx import lattice
    from rapid_mlx.lattice.defaults import Profile

    monkeypatch.setattr(lattice, "overlay_profile", lambda: Profile())
    monkeypatch.setattr(lattice, "local_calibration", lambda _: {})
    monkeypatch.delenv("RAPID_MLX_BENCHMARK_KNOBS", raising=False)
    assert fused.install_enabled("gemma4_affine4", mx, route="gemma4_affine4_install")
    assert not fused.install_enabled("gemma4_affine4", mx)
    monkeypatch.setenv("RAPID_MLX_BENCHMARK_KNOBS", '{"gemma4_affine4": false}')
    assert not fused.install_enabled(
        "gemma4_affine4", mx, route="gemma4_affine4_install"
    )
    b = fake_block(monkeypatch, True)
    monkeypatch.setenv("RAPID_MLX_GEMMA4_AFFINE4", "0")
    monkeypatch.setattr(fused, "_probe", lambda _: pytest.fail("disabled probe"))
    assert (
        fused.install_gemma4_affine4(SimpleNamespace(named_modules=lambda: [("", b)]))
        == 0
    )


@pytest.mark.parametrize(
    "module,switch",
    [
        ("rapid_mlx.models.gemma4_vendored.language", "mlx_lm.models.switch_layers"),
        ("mlx_lm.models.gemma4_text", "mlx_lm.models.switch_layers"),
        ("mlx_vlm.models.gemma4.language", "mlx_vlm.models.switch_layers"),
    ],
)
def test_resident_class_discovery(monkeypatch, module, switch):
    classes = tuple(
        type(name, (), {})
        for name in ("Experts", "SwitchGLU", "GeGLU", "QuantizedSwitchLinear")
    )
    monkeypatch.setitem(
        sys.modules, module, SimpleNamespace(Experts=classes[0], GeGLU=classes[2])
    )
    monkeypatch.setitem(
        sys.modules,
        switch,
        SimpleNamespace(SwitchGLU=classes[1], QuantizedSwitchLinear=classes[3]),
    )
    assert classes in fused._families()


def test_shader_is_packaged_with_shared_license():
    import tomllib

    root = Path(__file__).parents[1]
    data = tomllib.loads((root / "pyproject.toml").read_text())
    patterns = data["tool"]["setuptools"]["package-data"]["rapid_mlx"]
    assert "patches/gemma4_affine4_*.metal" in patterns
    assert "patches/QWEN35_AFFINE4_LICENSE" in patterns
