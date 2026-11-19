"""Gemma expert layout guards and selected affine-4 decode numerics."""

import functools
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, call, patch

import pytest

from tests.headless_mlx.test_lattice_gate_characterization import Tensor, fake_mx

mx = fake_mx()
# Load an isolated copy so metadata-only checks run even on hosts where
# importing the native runtime itself requires an unavailable Metal device.
# Keep its production name for module-filtered coverage, but do not register
# this copy in sys.modules or replace the real module used by Metal tests.
spec = importlib.util.spec_from_file_location(
    "rapid_mlx.patches.gemma4_affine4",
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
    monkeypatch.setattr(fused, "mx", fake_mx())
    monkeypatch.delenv("RAPID_MLX_GEMMA4_AFFINE4", raising=False)


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

    # Register installer mutations for teardown, including attributes it adds.
    monkeypatch.setattr(Experts, "__call__", Experts.__call__)
    for name in ("_rapid_gemma4_affine4_original", "_rapid_gemma4_affine4_patched"):
        monkeypatch.setattr(Experts, name, None, raising=False)
        monkeypatch.delattr(Experts, name)

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


def test_patch_class_is_idempotent(monkeypatch):
    b = fake_block(monkeypatch, True)
    fused._patch_class(type(b))
    patched = type(b).__call__
    fused._patch_class(type(b))
    assert type(b).__call__ is patched


def test_installer_skips_models_without_named_modules(monkeypatch):
    monkeypatch.setattr(fused, "install_enabled", lambda *a, **k: True)
    monkeypatch.setattr(fused, "_probe", lambda _: pytest.fail("no blocks to probe"))
    assert fused.install_gemma4_affine4(SimpleNamespace()) == 0


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


@pytest.mark.parametrize("combined", [False, True])
@pytest.mark.parametrize("tokens", [1, 2, 9])
def test_kernel_build_cache_and_launch_contract(monkeypatch, combined, tokens):
    b = fake_block(monkeypatch, combined)
    runtime = fused.mx
    # Give this isolated module a fresh cache without disturbing other suites.
    monkeypatch.setattr(
        fused, "kernels", functools.lru_cache(maxsize=1)(fused.kernels.__wrapped__)
    )
    gate_output = Tensor((tokens, 8, 704))
    down_output = Tensor((1, tokens, 2816))
    gate = Mock(return_value=[gate_output])
    down = Mock(return_value=[down_output])
    builder = Mock(side_effect=[gate, down])
    monkeypatch.setattr(runtime.fast, "metal_kernel", builder)
    gelu = Mock(return_value=Tensor((65536,)))
    monkeypatch.setattr(fused, "nn", SimpleNamespace(gelu_approx=gelu))

    x = Tensor((1, tokens, 2816))
    ids = Tensor((1, tokens, 8), runtime.uint32)
    scores = Tensor(ids.shape)
    for _ in range(2):
        assert fused.forward(b, x, ids, scores) is down_output

    root = Path(fused.__file__).parent
    header = (root / "qwen35_affine4_dot.metal").read_text()
    assert header
    assert builder.call_args_list == [
        call(
            name="rapid_gemma4_affine4_gateup",
            input_names=[
                "x",
                "wg_q",
                "g_scales",
                "g_biases",
                "wu_q",
                "u_scales",
                "u_biases",
                "inds",
                "gelutab",
            ],
            output_names=["y"],
            header=header,
            source=(root / "gemma4_affine4_gateup.metal").read_text(),
        ),
        call(
            name="rapid_gemma4_affine4_down",
            input_names=[
                "act",
                "wd_q",
                "d_scales",
                "d_biases",
                "inds",
                "scores",
                "sdown",
                "glog",
            ],
            output_names=["y"],
            header=header,
            source=(root / "qwen35_affine4_down.metal").read_text(),
        ),
    ]
    runtime.arange.assert_called_once_with(65536, dtype=runtime.uint16)
    runtime.arange.return_value.view.assert_called_once_with(runtime.bfloat16)
    gelu.assert_called_once_with(runtime.arange.return_value.view.return_value)
    runtime.eval.assert_called_once_with(gelu.return_value)
    assert fused.kernels() == (gate, down, gelu.return_value)
    assert fused.kernels.cache_info().misses == 1
    assert fused.kernels.cache_info().hits == 2

    sm = b.switch_glu
    gp = sm.gate_up_proj if combined else sm.gate_proj
    up = sm.gate_up_proj if combined else sm.up_proj
    dp = sm.down_proj
    gate_call = call(
        inputs=[
            x,
            gp.weight,
            gp.scales,
            gp.biases,
            up.weight,
            up.scales,
            up.biases,
            ids,
            gelu.return_value,
        ],
        template=[
            ("T", x.dtype),
            ("ROWS", 1),
            ("K", 2816),
            ("N", 704),
            ("GS", 64),
            ("TOPK", 8),
            ("COMBINED", int(combined)),
        ],
        grid=(32, 704, 8 * tokens),
        threadgroup=(32, 2, 1),
        output_shapes=[(tokens, 8, 704)],
        output_dtypes=[x.dtype],
    )
    down_call = call(
        inputs=[gate_output, dp.weight, dp.scales, dp.biases, ids, scores, x, x],
        template=[
            ("T", x.dtype),
            ("ROWS", 1),
            ("I", 704),
            ("H", 2816),
            ("GS", 64),
            ("TOPK", 8),
            ("SHARED", 0),
        ],
        grid=(32, 2816, tokens),
        threadgroup=(32, 2, 1),
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
    )
    assert gate.call_args_list == [gate_call, gate_call]
    assert down.call_args_list == [down_call, down_call]
    for launch, expected_call in ((gate, gate_call), (down, down_call)):
        assert all(
            actual is expected
            for actual, expected in zip(
                launch.call_args.kwargs["inputs"], expected_call.kwargs["inputs"]
            )
        )


@pytest.mark.parametrize("patched", [False, True])
@pytest.mark.parametrize(
    "expected_value,actual_value,accepted",
    [
        (1, 1, True),
        (1, 1.019, True),
        (1, 1.0201, False),
        (0, 0, True),
        (0, 0.001, False),
        (1, float("nan"), False),
    ],
)
def test_probe_deterministic_inputs_and_relative_rms(
    monkeypatch, patched, expected_value, actual_value, accepted
):
    import numpy as np

    b = fake_block(monkeypatch, False)
    runtime = fused.mx
    # NumPy evaluates the probe's arithmetic; float32 stands in for bf16 here.
    # Actual bf16 rounding and shader parity belong to the Metal suite.
    for name in ("arange", "array", "sin", "sqrt", "mean", "square", "maximum"):
        monkeypatch.setattr(runtime, name, Mock(side_effect=getattr(np, name)))
    monkeypatch.setattr(runtime, "bfloat16", np.float32)
    monkeypatch.setattr(runtime, "float32", np.float32)
    monkeypatch.setattr(runtime, "uint32", np.uint32)
    expected = np.full((1, 1, 2816), expected_value, dtype=np.float32)
    actual = np.full_like(expected, actual_value)
    original = Mock(return_value=expected)
    monkeypatch.setattr(type(b), "__call__", original)
    forward = Mock(return_value=actual)
    monkeypatch.setattr(fused, "forward", forward)
    if patched:
        fused._patch_class(type(b))
        b._rapid_gemma4_affine4 = True

    assert fused._probe(b) == accepted
    original.assert_called_once()
    forward.assert_called_once()
    block, x, ids, scores = original.call_args.args
    assert block is b
    assert all(a is e for a, e in zip(forward.call_args.args, (b, x, ids, scores)))
    assert x.shape == (1, 1, 2816)
    assert x.dtype == scores.dtype == np.float32
    assert ids.dtype == np.uint32
    np.testing.assert_array_equal(
        x, np.sin(np.arange(2816, dtype=np.float32) * 0.731).reshape(1, 1, 2816)
    )
    np.testing.assert_array_equal(ids, [[[0, 17, 39, 58, 77, 92, 111, 127]]])
    np.testing.assert_allclose(
        scores, [[[0.07, 0.11, 0.23, 0.04, 0.13, 0.31, 0.19, 0.09]]]
    )
    runtime.arange.assert_called_once_with(2816, dtype=np.float32)
    runtime.random.assert_not_called()
    runtime.maximum.assert_called_once()
    assert runtime.maximum.call_args.args[1] == 1e-12
    runtime.eval.assert_called_once()
    np.testing.assert_allclose(
        runtime.eval.call_args.args[0], np.sqrt(runtime.sqrt.call_args.args[0])
    )


@pytest.mark.parametrize("reason", ["metal", "incompatible", "noncallable"])
def test_installer_declines_without_probe(monkeypatch, reason):
    b = fake_block(monkeypatch, True)
    if reason == "metal":
        monkeypatch.setattr(fused.mx.metal, "is_available", lambda: False)
    elif reason == "incompatible":
        b.switch_glu.down_proj.bits = 8
    monkeypatch.setattr(fused, "install_enabled", lambda *a, **k: True)
    probe = Mock(side_effect=AssertionError("must decline before probing"))
    monkeypatch.setattr(fused, "_probe", probe)
    model = SimpleNamespace(
        named_modules=42 if reason == "noncallable" else lambda: [("experts", b)]
    )
    assert fused.install_gemma4_affine4(model) == 0
    probe.assert_not_called()
    assert "_rapid_gemma4_affine4_patched" not in type(b).__dict__
