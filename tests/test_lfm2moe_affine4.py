"""Portable layout/dispatch guards and Metal expert parity (2% relative RMS)."""

from types import SimpleNamespace

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
    ],
)
def test_layout_rejected(portable, change):
    change(portable)
    assert not fused.compatible(portable)


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


def test_lattice_policy():
    from rapid_mlx.lattice import ModelProfile, resolve, runtime_context

    ctx = runtime_context({}, model=ModelProfile(family="lfm2moe_affine4_install"))
    assert resolve("lfm2moe_affine4", ctx, environ={}).value is True
    ctx = runtime_context({}, model=ModelProfile(family="unknown"))
    assert resolve("lfm2moe_affine4", ctx, environ={}).value is False


@pytest.fixture(scope="module")
def metal():
    mx = pytest.importorskip("mlx.core", exc_type=ImportError)
    if not mx.metal.is_available():
        pytest.skip("requires Metal and both GPU locks")
    return mx


@pytest.mark.parametrize("combined", [False, True])
def test_metal_full_geometry(metal, combined):
    mx = metal
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
