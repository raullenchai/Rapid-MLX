"""Layout and reporting contracts using tensor metadata, without an MLX runtime."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1] / "rapid_mlx/models/deepseek_v41_native"
ROUTED = [(2, 2, 2), (2, 2, 3), (3, 3, 3), (4, 4, 3), (2, 2, 4), (4, 4, 4)]


class Module(SimpleNamespace):
    def __contains__(self, name):
        return hasattr(self, name)


class Tensor:
    def __init__(self, shape, dtype="bf16"):
        self.shape = shape
        self.ndim = len(shape)
        self.dtype = dtype

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]
        return Tensor(shape, self.dtype)


@pytest.fixture
def kernels(monkeypatch):
    # Import the real implementation with only its device dependencies replaced.
    package = ModuleType("_v41_cpu_layout")
    package.__path__ = [str(ROOT)]
    mx = ModuleType("mlx.core")
    for name, value in {
        "bfloat16": "bf16",
        "float16": "fp16",
        "float32": "fp32",
        "uint32": "u32",
    }.items():
        setattr(mx, name, value)
    mlx = ModuleType("mlx")
    mlx.core = mx
    for name, value in {
        "mlx": mlx,
        "mlx.core": mx,
        package.__name__: package,
        package.__name__ + ".head_gemv": ModuleType("head_gemv"),
        package.__name__ + ".kernels": SimpleNamespace(
            custom_kernels_enabled=lambda: True
        ),
    }.items():
        monkeypatch.setitem(sys.modules, name, value)
    loaded = []
    for name in ("moe_decode", "affine_route_qmv"):
        spec = importlib.util.spec_from_file_location(
            package.__name__ + "." + name, ROOT / (name + ".py")
        )
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
        loaded.append(module)
    return loaded


def projection(bits, n, k):
    return Module(
        bits=bits,
        group_size=64,
        mode="affine",
        weight=Tensor((*n, k * bits // 32), "u32"),
        scales=Tensor((*n, k // 64)),
        biases=Tensor((*n, k // 64)),
    )


def moe(routed=(2, 2, 2), shared=8):
    return Module(
        experts=Module(
            gate_proj=projection(routed[0], (64, 256), 512),
            up_proj=projection(routed[1], (64, 256), 512),
            down_proj=projection(routed[2], (64, 512), 256),
            activation=Module(limit=10.0),
        ),
        shared_experts=Module(
            w1=projection(shared, (256,), 512),
            w3=projection(shared, (256,), 512),
            w2=projection(shared, (512,), 256),
            limit=10.0,
        ),
        gate=Module(
            score_func="sqrtsoftplus",
            bias=Tensor((64,), "fp32"),
            weight=Tensor((64, 512)),
            topk=6,
            norm_topk_prob=True,
        ),
    )


@pytest.mark.parametrize("routed", ROUTED + [(2, 3, 4)])
@pytest.mark.parametrize("shared", [2, 4, 8])
def test_layout_and_projection_dispatch(kernels, monkeypatch, routed, shared):
    decode, route = kernels
    layer = moe(routed, shared)
    layout = decode._layout(layer)
    assert layout == (512, 256, shared, *routed)
    for p in (
        layer.experts.gate_proj,
        layer.experts.up_proj,
        layer.experts.down_proj,
        layer.shared_experts.w1,
        layer.shared_experts.w2,
        layer.shared_experts.w3,
    ):
        original = p.weight
        p.weight = Tensor((*original.shape[:-1], original.shape[-1] - 1), "u32")
        assert decode._layout(layer) is None
        p.weight = original
    launches = {}

    def kernel(name):
        def run(**kw):
            launches[name] = kw
            return [
                Tensor(shape, dtype)
                for shape, dtype in zip(
                    kw["output_shapes"], kw["output_dtypes"], strict=True
                )
            ]

        return run

    monkeypatch.setattr(decode, "_up_kernel", lambda limit: kernel("up"))
    monkeypatch.setattr(decode, "_down_kernel", lambda: kernel("down"))
    decode._experts(layer, layout, Tensor((3, 512)), Tensor((3, 6)), Tensor((3, 6)))
    up, down = (dict(launches[name]["template"]) for name in ("up", "down"))
    assert (up["GBITS"], up["UBITS"], down["DBITS"], up["SBITS"]) == (*routed, shared)
    assert up["GVPT"] == decode._qmv_vpt(routed[0], 512, 256)
    assert up["UVPT"] == decode._qmv_vpt(routed[1], 512, 256)
    assert down["DVPT"] == decode._qmv_vpt(routed[2], 256, 512)
    assert route._shape(
        layer.experts.down_proj, Tensor((3, 256), "fp32"), Tensor((3, 6))
    ) == (18, 6, 256, 512)
    signature = decode._signature(layer)[1]
    layer.experts.down_proj.group_size = 32
    assert decode._layout(layer) is None
    assert decode._signature(layer)[1] != signature
    with pytest.raises(ValueError, match="unsupported exact"):
        route._shape(layer.experts.down_proj, Tensor((3, 256), "fp32"), Tensor((3, 6)))


@pytest.mark.parametrize(
    "routed,shared", [((8, 2, 2), 8), ((2, 8, 2), 8), ((2, 2, 8), 8), ((2, 2, 2), 3)]
)
def test_unsupported_widths(kernels, routed, shared):
    assert kernels[0]._layout(moe(routed, shared)) is None


def test_summary_waits_for_qualification_and_logs_once(kernels, caplog):
    decode = kernels[0]
    layers = [Module(ffn=moe()), Module(ffn=moe((4, 4, 3)))]
    model = Module(layers=layers)
    with caplog.at_level("INFO", logger=decode.__name__):
        decode.log_coverage(Module(layers=[]), "bf16")
        decode.log_coverage(model, "bf16")
        setattr(
            layers[0].ffn,
            decode._CHECK_ATTR,
            (None, None, {"bf16": decode._layout(layers[0].ffn)}),
        )
        decode.log_coverage(model, "bf16")
        assert not caplog.records
        setattr(layers[1].ffn, decode._CHECK_ATTR, (None, None, {"bf16": None}))
        decode.log_coverage(model, "bf16")
        decode.log_coverage(model, "bf16")
        assert caplog.messages == [
            "fused V4.1 MoE: 1/2 layers; fallback: layer 1 (4,4,3|8)"
        ]
        decode.log_coverage(Module(layers=layers[:1]), "bf16")
        assert caplog.messages[-1] == "fused V4.1 MoE: 1/1 layers; fallback: none"


def test_route_launch_and_unsupported_qualification(kernels, monkeypatch):
    decode, route = kernels
    mx = sys.modules["mlx.core"]
    monkeypatch.setattr(mx, "contiguous", lambda x: x, raising=False)
    monkeypatch.setattr(
        Tensor, "astype", lambda x, dtype: Tensor(x.shape, dtype), raising=False
    )
    built = []

    def run(**kw):
        return [Tensor(kw["output_shapes"][0], kw["output_dtypes"][0])]

    monkeypatch.setattr(
        mx,
        "fast",
        Module(metal_kernel=lambda **kw: built.append(kw) or run),
        raising=False,
    )
    # Tensor metadata checks the raw launch; runtime qualification is covered
    # by test_deepseek_v41_self_check with evaluated CPU arrays and mocks.
    out = route._fused(
        projection(3, (64, 512), 256), Tensor((3, 256), "fp32"), Tensor((3, 6))
    )
    assert out.shape == (3, 6, 512)
    assert built[0]["header"] == decode._HEADER
    assert decode._qualified(moe((8, 2, 2)), None, "bf16") is None


@pytest.fixture
def qualification(kernels, monkeypatch):
    decode = kernels[0]
    mx = sys.modules["mlx.core"]
    monkeypatch.setattr(Tensor, "__getitem__", lambda x, key: x, raising=False)
    monkeypatch.setattr(Tensor, "astype", lambda x, dtype: x, raising=False)
    monkeypatch.setattr(
        mx,
        "random",
        Module(key=lambda seed: seed, normal=lambda shape, **kw: Tensor(shape)),
        raising=False,
    )
    monkeypatch.setattr(
        mx, "array_equal", lambda a, b: Module(item=lambda: True), raising=False
    )
    monkeypatch.setattr(decode, "_fused", lambda moe, layout, x: x)
    monkeypatch.setattr(decode, "supports", lambda x: True)
    layer = moe((2, 2, 3), 4)
    object.__setattr__(layer, "layer_id", 7)
    return decode, layer


@pytest.mark.parametrize("failure", ["mismatch", "exception"])
def test_exactness_fallback_warning(qualification, monkeypatch, caplog, failure):
    decode, layer = qualification
    if failure == "mismatch":
        monkeypatch.setattr(
            sys.modules["mlx.core"],
            "array_equal",
            lambda a, b: Module(item=lambda: False),
        )
    else:

        def fail(*args):
            raise RuntimeError("self-check launch failed")

        monkeypatch.setattr(decode, "_fused", fail)
    with caplog.at_level("WARNING", logger=decode.__name__):
        assert decode._qualified(layer, lambda x: x, "bf16") is None
        assert decode._qualified(layer, lambda x: x, "bf16") is None
    record = caplog.records[-1]
    assert record.levelname == "WARNING"
    assert record.getMessage() == (
        "fused V4.1 MoE disabled: stock path for layer 7 (2,2,3|4)"
    )
    assert len(caplog.records) == (1 if failure == "mismatch" else 2)
    assert decode.fused_layers(Module(layers=[Module(ffn=layer)]), "bf16") == (0, 1)


def test_launch_fallback_after_coverage(qualification, monkeypatch, caplog):
    decode, layer = qualification
    model = Module(layers=[Module(ffn=layer)])
    x = Module(shape=(1, 512), size=512, dtype="bf16")
    assert decode._qualified(layer, lambda x: x, "bf16") is not None
    with caplog.at_level("INFO", logger=decode.__name__):
        decode.log_coverage(model, "bf16")
        assert decode.fused_layers(model, "bf16") == (1, 1)

        def fail(*args):
            raise RuntimeError("decode launch failed")

        monkeypatch.setattr(decode, "_fused", fail)
        assert decode.moe_forward(layer, x, None) is None
        assert decode.moe_forward(layer, x, None) is None
        decode.log_coverage(model, "bf16")
    assert caplog.messages == [
        "fused V4.1 MoE: 1/1 layers; fallback: none",
        "fused V4.1 MoE failed: stock path for layer 7 (2,2,3|4)",
    ]
    assert caplog.records[-1].levelname == "WARNING"
    assert isinstance(caplog.records[-1].exc_info[1], RuntimeError)
    assert decode.fused_layers(model, "bf16") == (0, 1)


@pytest.mark.parametrize("layers", [[], [Module(), Module()], [Module(ffn=None)]])
def test_coverage_without_ffn(kernels, caplog, layers):
    decode = kernels[0]
    model = Module(layers=layers)
    with caplog.at_level("INFO", logger=decode.__name__):
        decode.log_coverage(model, "bf16")
        decode.log_coverage(model, "bf16")
    assert decode.fused_layers(model, "bf16") == (0, 0)
    assert caplog.messages == []
    assert not getattr(model, "_rapid_moe_summary_logged", False)


def test_coverage_with_mixed_layers(kernels, caplog):
    decode = kernels[0]
    fused, stock, pending = moe(), moe(), moe()
    setattr(fused, decode._CHECK_ATTR, (None, None, {"bf16": decode._layout(fused)}))
    setattr(stock, decode._CHECK_ATTR, (None, None, {"bf16": None}))
    model = Module(
        layers=[
            Module(),
            Module(ffn=fused),
            Module(ffn=None),
            Module(ffn=stock),
            Module(ffn=pending, layer_id=7),
            Module(),
        ]
    )
    with caplog.at_level("INFO", logger=decode.__name__):
        decode.log_coverage(model, "bf16")
        assert caplog.messages == []
        assert decode.fused_layers(model, "bf16") == (1, 3)
        setattr(pending, decode._CHECK_ATTR, (None, None, {"bf16": None}))
        decode.log_coverage(model, "bf16")
        decode.log_coverage(model, "bf16")
    assert caplog.messages == [
        "fused V4.1 MoE: 1/3 layers; fallback: layer 3 (2,2,2|8), layer 7 (2,2,2|8)"
    ]
    assert decode.fused_layers(model, "bf16") == (1, 3)


def test_fused_layers_cached_verdicts(kernels):
    decode = kernels[0]
    assert decode.fused_layers(Module(layers=[]), "bf16") == (0, 0)
    layers = [Module(ffn=moe()) for _ in range(4)]
    model = Module(layers=layers)
    assert decode.fused_layers(model, "bf16") == (0, 4)
    for layer, verdicts in zip(
        layers,
        [{"bf16": decode._layout(layers[0].ffn)}, {"bf16": None}, {"fp16": (1,)}, {}],
        strict=True,
    ):
        object.__setattr__(layer.ffn, decode._CHECK_ATTR, (None, None, verdicts))
    assert decode.fused_layers(model, "bf16") == (1, 4)
    assert decode.fused_layers(model, "fp16") == (1, 4)
    assert decode.fused_layers(model, "fp32") == (0, 4)


def test_model_hook_and_target_gate_without_runtime(kernels, monkeypatch, caplog):
    """Execute the model's real forward and serving gates with inert dependencies."""
    from unittest.mock import MagicMock

    decode = kernels[0]
    package = decode.__package__
    mx = sys.modules["mlx.core"]
    nn = ModuleType("mlx.nn")
    nn.Module = Module
    monkeypatch.setitem(sys.modules, "mlx.nn", nn)
    dependencies = {
        "attention": {"Attention": Module},
        "cache": {"BatchModelCache": type("BatchCache", (), {}), "ModelCache": Module},
        "config": {"ModelArgs": Module},
        "engram": {"Engram": Module, "EngramHasher": Module},
        "head_gemv": {"head_logits": lambda head, h: h},
        "hyper_connections": {
            "hc_mixes": None,
            "hc_post": None,
            "hc_pre_norm": lambda h, *args: h,
            "make_identity_pre_mix": lambda *args: None,
        },
        "layers": {"RMSNorm": Module},
        "moe": {"MoE": Module, "ClampedSwiGLU": Module},
        "dspark": {
            name: Module
            for name in (
                "DSpark",
                "DSparkWeights",
                "draft_attention",
                "quantize_cache",
                "rotate_positions",
            )
        },
        "load": {"load": None, "supports_engram_ssd_offload": None},
    }
    for name, attrs in dependencies.items():
        module = ModuleType(package + "." + name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, module.__name__, module)
    for name, attrs in {
        "fastapi": {"HTTPException": Exception},
        "transformers": {"PreTrainedTokenizerFast": Module},
        "rapid_mlx": {"_mlx_compat": Module(install=lambda: None)},
        "mlx_lm.models.switch_layers": {"SwitchGLU": Module},
        "mlx_lm.tokenizer_utils": {"NaiveStreamingDetokenizer": Module},
    }.items():
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)

    def load(name):
        spec = importlib.util.spec_from_file_location(
            package + "." + name, ROOT / (name + ".py")
        )
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
        return module

    serving = load("serving")
    for bits in (2, 3, 4):
        experts = moe((bits,) * 3).experts
        assert serving.supports_target_qmv(experts)
        experts.down_proj.weight = Tensor((64, 512, 1), "u32")
        assert not serving.supports_target_qmv(experts)
    model_module = load("model")
    args = Module(
        norm_eps=1e-6,
        hc_mult=4,
        hc_sinkhorn_iters=20,
        hc_eps=1e-6,
        experts_per_layer=(),
        dim=512,
        engram_layer_ids=(),
    )
    for name in ("Attention", "MoE", "RMSNorm"):
        monkeypatch.setattr(model_module, name, lambda *args: Module())
    monkeypatch.setattr(mx, "zeros", lambda *args, **kw: None, raising=False)
    block = model_module.Block(7, args)
    assert block.ffn.layer_id == block.layer_id == 7
    model = model_module.Model.__new__(model_module.Model)
    ffn = moe()
    setattr(ffn, decode._CHECK_ATTR, (None, None, {"bf16": decode._layout(ffn)}))
    layer = MagicMock()
    layer.ffn, layer.engram = ffn, None
    h = MagicMock(dtype="bf16")
    layer.return_value = (h, None)
    model.__dict__.update(
        engram_hasher=None,
        args=Module(engram_layer_ids=()),
        embed=lambda ids: h,
        hc_mult=4,
        layers=[layer],
        _break_sharing=False,
        eval_interval=0,
        head=None,
        norm=Module(weight=None, eps=0),
    )
    monkeypatch.setattr(mx, "broadcast_to", lambda h, shape: h, raising=False)
    cache = Module(offset=0, rollback_start=None)
    with caplog.at_level("INFO", logger=decode.__name__):
        model(Tensor((1, 1)), cache)
        model(Tensor((1, 1)), cache)
    assert caplog.messages == ["fused V4.1 MoE: 1/1 layers; fallback: none"]
