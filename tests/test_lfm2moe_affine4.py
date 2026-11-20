"""Portable layout/dispatch guards and Metal expert parity (2% relative RMS)."""

import sys
from functools import cache
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from rapid_mlx.patches import lfm2moe_affine4 as fused
from rapid_mlx.patches import moe_decode_fusion as router


class Module(dict):
    def __setattr__(self, name, value):
        if name == "__class__":
            object.__setattr__(self, name, value)
        else:
            self[name] = value

    __delattr__ = dict.__delitem__

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None


class Block(Module):
    def __call__(self, x):
        return "original"


class Switch(Module):
    pass


class Activation:
    pass


class Quantized(Module):
    pass


def array(shape, dtype="bf16"):
    return SimpleNamespace(shape=shape, ndim=len(shape), dtype=dtype)


def projection(e, n, k):
    return Quantized(
        weight=array((e, n, k // 8), "u32"),
        scales=array((e, n, k // 64)),
        biases=array((e, n, k // 64)),
        bits=4,
        group_size=64,
        mode="affine",
    )


@pytest.fixture
def portable(monkeypatch):
    mx = SimpleNamespace(uint32="u32", bfloat16="bf16")
    monkeypatch.setattr(
        fused, "_runtime", lambda: (mx, Block, Switch, Activation, Quantized)
    )
    monkeypatch.delenv("RAPID_MLX_LFM2MOE_AFFINE4", raising=False)
    e, h, i = 64, 2048, 1536
    sm = Switch(
        gate_proj=projection(e, i, h),
        up_proj=projection(e, i, h),
        down_proj=projection(e, h, i),
        activation=Activation(),
    )
    return Block(
        gate=Module(weight=array((e, h), "f32")), switch_mlp=sm, num_experts=e, top_k=4
    )


@pytest.mark.parametrize("combined", [False, True])
def test_layout_and_dispatch(portable, monkeypatch, combined):
    b = portable
    if combined:
        b.switch_mlp.gate_up_proj = projection(64, 3072, 2048)
        del b.switch_mlp.gate_proj
        del b.switch_mlp.up_proj
    assert fused.compatible(b)
    monkeypatch.setattr(fused, "_probe", lambda b: True)
    assert fused.enroll(b)
    b.__class__ = router._decode_type(Block, Block.__call__)
    x = array((1, 1, 2048))
    assert fused.eligible(b, x)
    monkeypatch.setattr(router, "eligible", lambda b, x: True)
    monkeypatch.setattr(router, "route", lambda b, x: ("indices", "scores"))
    monkeypatch.setattr(fused, "forward", lambda *args: args[2:])
    assert b(x) == ("indices", "scores")
    monkeypatch.setenv("RAPID_MLX_LFM2MOE_AFFINE4", "0")
    assert not fused.eligible(b, x)
    monkeypatch.setattr(Switch, "__call__", lambda *args: "experts", raising=False)
    monkeypatch.setattr(router, "combine", lambda *args: args)
    assert b(x) == ("experts", "scores")


@pytest.mark.parametrize(
    "change",
    [
        lambda b: setattr(b.switch_mlp.gate_proj, "bits", 8),
        lambda b: setattr(b.switch_mlp.up_proj, "group_size", 32),
        lambda b: setattr(b.switch_mlp.down_proj, "mode", "mxfp4"),
        lambda b: setattr(b.switch_mlp.gate_proj, "bias", array((64, 1536))),
        lambda b: setattr(b.switch_mlp.up_proj.scales, "dtype", "f32"),
        lambda b: setattr(b.switch_mlp.down_proj.biases, "dtype", "f16"),
        lambda b: setattr(b.switch_mlp.gate_proj.weight, "dtype", "bf16"),
        lambda b: setattr(b.switch_mlp.up_proj.weight, "shape", (64, 1537, 256)),
        lambda b: setattr(b.switch_mlp.down_proj.biases, "shape", (64, 2048, 25)),
        lambda b: setattr(b.switch_mlp.down_proj, "biases", None),
        lambda b: setattr(b.switch_mlp.down_proj, "sharding_group", object()),
        lambda b: setattr(b.switch_mlp, "sharding_group", object()),
        lambda b: setattr(b, "sharding_group", object()),
        lambda b: setattr(b.gate, "sharding_group", object()),
        lambda b: setattr(b.switch_mlp, "activation", object()),
        lambda b: setattr(b, "num_experts", 32),
        lambda b: setattr(b, "top_k", 9),
        lambda b: setattr(b.switch_mlp, "gate_up_proj", projection(64, 1536, 2048)),
        lambda b: setattr(b, "switch_mlp", None),
        lambda b: setattr(b.gate, "weight", None),
        lambda b: setattr(b.switch_mlp.down_proj, "scales", None),
    ],
)
def test_layout_rejected(portable, change):
    change(portable)
    assert not fused.compatible(portable)


def test_installed_layout_requires_router_subclass(portable):
    # Dispatch-time checks accept only the router installer's decode subclass.
    assert fused.compatible(portable)
    assert not fused.compatible(portable, installed=True)


@pytest.mark.parametrize("which", ["block", "switch", "projection"])
def test_subclasses_rejected(portable, which):
    target = {
        "block": portable,
        "switch": portable.switch_mlp,
        "projection": portable.switch_mlp.gate_proj,
    }[which]
    target.__class__ = type("Custom", (type(target),), {})
    assert not fused.compatible(portable)


@pytest.mark.parametrize(
    "shape,dtype",
    [
        ((2, 1, 2048), "bf16"),
        ((1, 2, 2048), "bf16"),
        ((1, 1, 2048), "f16"),
        ((1, 1, 2048), "f32"),
        ((1, 1, 1024), "bf16"),
    ],
)
def test_input_fallback(portable, monkeypatch, shape, dtype):
    monkeypatch.setattr(fused, "_probe", lambda b: True)
    assert fused.enroll(portable)
    portable.__class__ = router._decode_type(Block, Block.__call__)
    assert not fused.eligible(portable, array(shape, dtype))


@pytest.mark.parametrize("result", [False, ValueError("probe failure")])
def test_probe_declines(portable, monkeypatch, result):
    def probe(b):
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(fused, "_probe", probe)
    assert not fused.enroll(portable)
    assert not getattr(portable, "_rapid_lfm2moe_affine4", False)


def test_kill_switch_prevents_probe(portable, monkeypatch):
    monkeypatch.setenv("RAPID_MLX_LFM2MOE_AFFINE4", "0")
    monkeypatch.setattr(fused, "_probe", lambda b: pytest.fail("probe ran"))
    assert not fused.enroll(portable)


def test_runtime_imports(portable, monkeypatch):
    mx = fused._runtime()[0]
    modules = {
        name: ModuleType(name)
        for name in (
            "mlx",
            "mlx.core",
            "mlx_lm",
            "mlx_lm.models",
            "mlx_lm.models.lfm2_moe",
            "mlx_lm.models.switch_layers",
        )
    }
    modules["mlx"].core = mx
    modules["mlx.core"] = mx
    modules["mlx_lm.models.lfm2_moe"].Lfm2MoeSparseMoeBlock = Block
    switch = modules["mlx_lm.models.switch_layers"]
    switch.SwitchGLU, switch.SwiGLU, switch.QuantizedSwitchLinear = (
        Switch,
        Activation,
        Quantized,
    )
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    # Use the real loader without populating its process-wide cache with fakes.
    assert _runtime_loader() == (mx, Block, Switch, Activation, Quantized)


# Retain the undecorated loader before the portable fixture replaces _runtime.
_runtime_loader = fused._runtime.__wrapped__


@pytest.fixture
def recorded_kernels(portable, monkeypatch):
    mx = fused._runtime()[0]
    table = array((65536,))
    bits = Mock()
    viewed = bits.view.return_value
    mx.uint16 = "u16"
    mx.arange = Mock(return_value=bits)
    mx.sigmoid = Mock(return_value=table)
    mx.eval = Mock()
    builders, launches, launch_names = [], [], []

    def build(**kwargs):
        builders.append(kwargs)
        name = kwargs["name"]

        def launch(**kwargs):
            launch_names.append(name)
            output = array(kwargs["output_shapes"][0], kwargs["output_dtypes"][0])
            launches.append((kwargs, output))
            return [output]

        return launch

    mx.fast = SimpleNamespace(metal_kernel=build)
    # Give each test a fresh cache, preserving any real Metal kernels cached by
    # other tests and preventing fake kernels from escaping this fixture.
    monkeypatch.setattr(fused, "kernels", cache(fused.kernels.__wrapped__))
    return SimpleNamespace(
        mx=mx,
        table=table,
        bits=bits,
        viewed=viewed,
        builders=builders,
        launches=launches,
        launch_names=launch_names,
    )


@pytest.mark.parametrize("combined", [False, True])
def test_kernel_build_and_launch_contract(portable, recorded_kernels, combined):
    b, record = portable, recorded_kernels
    sm = b.switch_mlp
    if combined:
        sm.gate_up_proj = projection(64, 3072, 2048)
        del sm.gate_proj
        del sm.up_proj
    gp = sm.gate_up_proj if combined else sm.gate_proj
    up = sm.gate_up_proj if combined else sm.up_proj
    dp = sm.down_proj
    x, inds, scores = array((1, 1, 2048)), array((1, 1, 4), "u32"), array((1, 1, 4))
    for _ in range(2):
        result = fused.forward(b, x, inds, scores)
        assert result is record.launches[-1][1]
        assert result.shape == x.shape
        assert result.dtype == x.dtype
    root = Path(fused.__file__).parent
    header = (root / "qwen35_affine4_dot.metal").read_text()
    assert record.builders == [
        dict(
            name="rapid_lfm2moe_affine4_gateup",
            input_names=[
                "x",
                "wg_q",
                "g_scales",
                "g_biases",
                "wu_q",
                "u_scales",
                "u_biases",
                "inds",
                "sigtab",
            ],
            output_names=["y"],
            header=header,
            source=(root / "lfm2moe_affine4_gateup.metal").read_text(),
        ),
        dict(
            name="rapid_lfm2moe_affine4_down",
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
    record.mx.arange.assert_called_once_with(65536, dtype="u16")
    record.bits.view.assert_called_once_with("bf16")
    record.mx.sigmoid.assert_called_once_with(record.viewed)
    record.mx.eval.assert_called_once_with(record.table)
    assert (
        record.launch_names
        == [
            "rapid_lfm2moe_affine4_gateup",
            "rapid_lfm2moe_affine4_down",
        ]
        * 2
    )
    for offset in (0, 2):
        gate, act = record.launches[offset]
        down, _ = record.launches[offset + 1]
        # Shape-only fake banks compare equal, so also check buffer identity.
        assert all(
            actual is expected
            for actual, expected in zip(
                gate["inputs"],
                [
                    x,
                    gp.weight,
                    gp.scales,
                    gp.biases,
                    up.weight,
                    up.scales,
                    up.biases,
                    inds,
                    record.table,
                ],
                strict=True,
            )
        )
        assert all(
            actual is expected
            for actual, expected in zip(
                down["inputs"],
                [act, dp.weight, dp.scales, dp.biases, inds, scores, x, x],
                strict=True,
            )
        )
        assert gate == dict(
            inputs=[
                x,
                gp.weight,
                gp.scales,
                gp.biases,
                up.weight,
                up.scales,
                up.biases,
                inds,
                record.table,
            ],
            template=[
                ("T", "bf16"),
                ("ROWS", 1),
                ("K", 2048),
                ("N", 1536),
                ("GS", 64),
                ("TOPK", 4),
                ("COMBINED", int(combined)),
            ],
            grid=(32, 1536, 4),
            threadgroup=(32, 2, 1),
            output_shapes=[(1, 4, 1536)],
            output_dtypes=["bf16"],
        )
        assert down == dict(
            inputs=[act, dp.weight, dp.scales, dp.biases, inds, scores, x, x],
            template=[
                ("T", "bf16"),
                ("ROWS", 1),
                ("I", 1536),
                ("H", 2048),
                ("GS", 64),
                ("TOPK", 4),
                ("SHARED", 0),
            ],
            grid=(32, 2048, 1),
            threadgroup=(32, 2, 1),
            output_shapes=[x.shape],
            output_dtypes=["bf16"],
        )


@pytest.mark.parametrize("top_k", [1, 4])
@pytest.mark.parametrize("relative_error", [0.0, 0.019, 0.021, float("nan")])
def test_probe_enrollment(portable, monkeypatch, top_k, relative_error):
    b = portable
    b.top_k = top_k
    # NumPy supplies real arithmetic for the probe, without MLX or Metal. The
    # portable runtime uses float32 for the otherwise unavailable bfloat16 cast.
    mx = SimpleNamespace(
        **{
            name: getattr(np, name)
            for name in (
                "arange",
                "sin",
                "array",
                "sqrt",
                "mean",
                "square",
                "maximum",
                "float32",
                "uint32",
            )
        },
        bfloat16=np.float32,
        eval=Mock(),
    )
    runtime = fused._runtime()
    monkeypatch.setattr(fused, "_runtime", lambda: (mx, *runtime[1:]))
    # Layout guards still see the fake layout dtypes; only probe arithmetic uses
    # the NumPy dtype aliases.
    monkeypatch.setattr(fused, "compatible", lambda block: block is b)
    calls = []

    def experts(sm, x, inds):
        assert sm is b.switch_mlp
        np.testing.assert_allclose(
            x, np.sin(np.arange(2048, dtype=np.float32) * 0.731).reshape(1, 1, 2048)
        )
        expected_inds = [63] if top_k == 1 else [0, 21, 42, 63]
        np.testing.assert_array_equal(inds, [[expected_inds]])
        assert inds.dtype == np.uint32
        calls.append((x, inds))
        return np.ones((1, 1, top_k, 2048), dtype=np.float32)

    def forward(block, x, inds, scores):
        assert block is b
        assert x is calls[-1][0] and inds is calls[-1][1]
        expected_scores = np.arange(1, top_k + 1, dtype=np.float32)
        expected_scores /= expected_scores.sum()
        np.testing.assert_allclose(scores, expected_scores.reshape(1, 1, top_k))
        return np.full(x.shape, 1 + relative_error, dtype=np.float32)

    monkeypatch.setattr(Switch, "__call__", experts, raising=False)
    monkeypatch.setattr(fused, "forward", forward)
    accepted = relative_error in (0.0, 0.019)
    assert fused.enroll(b) is accepted
    assert getattr(b, "_rapid_lfm2moe_affine4", False) is accepted
    assert len(calls) == 1
    mx.eval.assert_called_once()


def test_incompatible_block_prevents_probe(portable, monkeypatch):
    portable.top_k = 9
    monkeypatch.setattr(fused, "_probe", lambda b: pytest.fail("probe ran"))
    assert not fused.enroll(portable)


def test_lattice_policy():
    from rapid_mlx.lattice import ModelProfile, resolve, runtime_context

    ctx = runtime_context(
        {"device_name": "Apple M5 Max"},
        model=ModelProfile(family="lfm2moe_affine4_install"),
    )
    assert resolve("lfm2moe_affine4", ctx, environ={}).value is True
    ctx = runtime_context({}, model=ModelProfile(family="lfm2moe_affine4_install"))
    assert resolve("lfm2moe_affine4", ctx, environ={}).value is False
    ctx = runtime_context({}, model=ModelProfile(family="unknown"))
    assert resolve("lfm2moe_affine4", ctx, environ={}).value is False


@pytest.fixture(scope="module")
def metal():
    mx = pytest.importorskip("mlx.core", exc_type=ImportError)
    if not mx.metal.is_available():
        pytest.skip("requires Metal and both GPU locks")
    return mx


@pytest.mark.parametrize("combined", [False, True])
def test_metal_full_geometry(metal, combined, monkeypatch):
    mx = metal
    # Kernel parity is hardware-independent; pin enrollment so the result does
    # not depend on whether this host's chip is in the performance cell.
    monkeypatch.setattr(router, "install_enabled", lambda *a, **k: True)
    import mlx.nn as nn
    from mlx_lm.models.lfm2_moe import Lfm2MoeSparseMoeBlock
    from mlx_lm.models.switch_layers import QuantizedSwitchLinear, SwiGLU, SwitchGLU

    # Build quantized banks directly to avoid retaining full float expert banks.
    b = Lfm2MoeSparseMoeBlock.__new__(Lfm2MoeSparseMoeBlock)
    nn.Module.__init__(b)
    b.num_experts, b.top_k = 64, 4
    b.norm_topk_prob = b.use_expert_bias = True
    b.gate = nn.Linear(2048, 64, bias=False)
    b.expert_bias = mx.zeros((64,), mx.float32)
    sm = SwitchGLU.__new__(SwitchGLU)
    nn.Module.__init__(sm)
    sm.activation = SwiGLU()
    for name, n, k in [
        ("gate_proj", 1536, 2048),
        ("up_proj", 1536, 2048),
        ("down_proj", 2048, 1536),
    ]:
        mx.random.seed(71 + n + k + len(name))
        p = QuantizedSwitchLinear(k, n, 64, bias=False, bits=4, group_size=64)
        p.set_dtype(mx.bfloat16)
        mx.eval(p.parameters())
        setattr(sm, name, p)
    b.switch_mlp = sm
    if combined:
        from rapid_mlx.moe_fusion import fuse_gate_up

        assert fuse_gate_up(nn.Sequential(b)) == 1
    assert fused.compatible(b)
    for seed in (13, 37, 91):
        x = mx.random.normal((1, 1, 2048), key=mx.random.key(seed)).astype(mx.bfloat16)
        # Route separately so expert arithmetic is compared with identical scores.
        inds, scores = router.route(b, x)
        if seed == 91:
            inds = mx.array([[[0, 17, 39, 63]]], mx.uint32)
            scores = mx.array([[[0.07, 0.23, 0.31, 0.39]]], mx.bfloat16)
        expected = (sm(x, inds) * scores[..., None]).sum(-2).astype(mx.float32)
        actual = fused.forward(b, x, inds, scores).astype(mx.float32)
        relative_rms = mx.sqrt(mx.mean((actual - expected) ** 2) / mx.mean(expected**2))
        assert relative_rms.item() <= 0.02
        assert mx.max(mx.abs(actual - expected)).item() <= 0.01
    assert fused._probe(b)
    assert router.install_moe_decode_fusion(nn.Sequential(b)) == 1
    assert fused.eligible(b, x)
    assert b(x).shape == x.shape


@pytest.mark.parametrize("allow", [False, True])
def test_installer_policy_and_existing_router(portable, monkeypatch, allow):
    import sys
    from types import ModuleType

    mx = SimpleNamespace(metal=SimpleNamespace(is_available=lambda: True))
    mlx = ModuleType("mlx")
    mlx.core = mx
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", mx)
    decisions = []

    def policy(name, runtime, **kwargs):
        decisions.append(name)
        return name == "moe_decode_fusion" or allow

    monkeypatch.setattr(router, "install_enabled", policy)
    monkeypatch.setattr(
        router,
        "_block_types",
        lambda: {Block: router._decode_type(Block, Block.__call__)},
    )
    probes = []
    monkeypatch.setattr(fused, "_probe", lambda b: probes.append(b) or True)
    model = SimpleNamespace(named_modules=lambda: [("block", portable)])
    assert router.install_moe_decode_fusion(model) == 1
    assert decisions == ["moe_decode_fusion", "lfm2moe_affine4"]
    assert len(probes) == int(allow)
    assert bool(getattr(portable, "_rapid_lfm2moe_affine4", False)) == allow
    # Policy never runs in the forward path; the enrolled marker carries it.
    monkeypatch.setattr(
        router, "install_enabled", lambda *a, **k: pytest.fail("hot policy resolution")
    )
    monkeypatch.setattr(router, "eligible", lambda b, x: True)
    monkeypatch.setattr(router, "route", lambda *args: (None, None))
    monkeypatch.setattr(fused, "forward", lambda *args: "affine4")
    monkeypatch.setattr(Switch, "__call__", lambda *args: None, raising=False)
    monkeypatch.setattr(router, "combine", lambda *args: "composed")
    assert portable(array((1, 1, 2048))) == ("affine4" if allow else "composed")


@pytest.mark.parametrize(
    "e,h,i,k", [(32, 2048, 1024, 4), (64, 1536, 512, 2), (128, 2816, 704, 8)]
)
def test_generic_geometry(portable, e, h, i, k):
    portable.gate.weight = array((e, h), "f32")
    portable.num_experts, portable.top_k = e, k
    portable.switch_mlp.gate_proj = projection(e, i, h)
    portable.switch_mlp.up_proj = projection(e, i, h)
    portable.switch_mlp.down_proj = projection(e, h, i)
    assert fused.compatible(portable)
