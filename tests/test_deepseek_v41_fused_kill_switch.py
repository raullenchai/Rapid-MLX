"""RAPID_MLX_DSV41_FUSED=0 sends every native V4.1 custom kernel to stock."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mlx")
pytestmark = pytest.mark.requires_mlx

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402

from rapid_mlx.models.deepseek_v41_native import (  # noqa: E402
    affine_route_qmv,
    decode_attention,
    head_gemv,
    kernels,
    moe_decode,
    rope_quant,
    serving,
)
from rapid_mlx.models.deepseek_v41_native import (  # noqa: E402
    hyper_connections as hc,
)
from rapid_mlx.models.deepseek_v41_native.config import ModelArgs  # noqa: E402
from rapid_mlx.models.deepseek_v41_native.model import Model  # noqa: E402
from rapid_mlx.models.deepseek_v41_native.moe import MoE  # noqa: E402

needs_metal = pytest.mark.skipif(
    not mx.metal.is_available(), reason="custom kernels need Metal"
)

# every custom Metal kernel factory of the backend
FACTORIES = [
    (affine_route_qmv, "_kernel"),
    (decode_attention, "_kernels"),
    (head_gemv, "_kernel"),
    (hc, "_sinkhorn_kernel"),
    (hc, "_hc_mixes_partial_kernel"),
    (hc, "_hc_mixes_finish_kernel"),
    (hc, "_hc_post_kernel"),
    (hc, "_hc_pre_norm_kernel"),
    (moe_decode, "_route_kernel"),
    (moe_decode, "_up_kernel"),
    (moe_decode, "_down_kernel"),
    (rope_quant, "_kernel"),
]


def _forbid_kernels(monkeypatch):
    for module, name in FACTORIES:

        def fail(*_a, _name=f"{module.__name__}.{name}", **_kw):
            pytest.fail(f"{_name} built with {kernels.FUSED_ENV}=0")

        monkeypatch.setattr(module, name, fail)


def _record_kernels(monkeypatch):
    used = []
    for module, name in FACTORIES:
        real = getattr(module, name)

        def spy(*a, _real=real, _name=name, **kw):
            used.append(_name)
            return _real(*a, **kw)

        monkeypatch.setattr(module, name, spy)
    return used


def _stock(monkeypatch):
    """The stock paths, by the per-module switches the kill switch mirrors."""
    monkeypatch.setattr(decode_attention, "ENABLED", False)
    monkeypatch.setattr(head_gemv, "ENABLED", False)
    monkeypatch.setattr(moe_decode, "ENABLED", False)
    monkeypatch.setattr(rope_quant, "ENABLED", False)
    monkeypatch.setattr(hc, "HC_MIXES_ENABLED", False)
    monkeypatch.setattr(hc, "_metal_fast_path_available", lambda: False)
    monkeypatch.setattr(serving, "supports_target_qmv", lambda _experts: False)


@pytest.mark.parametrize(
    "value,enabled",
    [
        (None, True),
        ("1", True),
        ("on", True),
        ("0", False),
        ("false", False),
        (" OFF ", False),
        ("disabled", False),
    ],
)
def test_switch_values(monkeypatch, value, enabled):
    if value is None:
        monkeypatch.delenv(kernels.FUSED_ENV, raising=False)
    else:
        monkeypatch.setenv(kernels.FUSED_ENV, value)
    assert kernels.custom_kernels_enabled() is enabled


def test_every_eligibility_check_declines_when_off(monkeypatch):
    """Each lever's own check says yes on a GPU, and no once the switch is off."""
    monkeypatch.setattr(mx, "default_device", lambda: mx.gpu)
    monkeypatch.setattr(mx.metal, "is_available", lambda: True)
    monkeypatch.setattr(head_gemv, "_gpu_default", lambda: True)
    head = nn.Linear(512, 1003, bias=False)
    head.weight = head.weight.astype(mx.bfloat16)
    x = mx.zeros((1, 512))
    t1 = mx.zeros((1, 32))
    checks = {
        "decode_attention": lambda: decode_attention.supports(
            mx.zeros((1, 1, 64), dtype=mx.bfloat16), mx.zeros((1, 8, 512)), 64, 512, 64
        ),
        "head_gemv": lambda: head_gemv.supports(head, x),
        "moe_decode": lambda: moe_decode.supports(mx.zeros((1, 1, 512))),
        "rope_quant": lambda: rope_quant.supports(
            mx.zeros((1, 1, 4, 128), dtype=mx.bfloat16), 64, t1, t1
        ),
        "hyper_connections": hc._metal_fast_path_available,
        "affine_route_qmv": lambda: serving.supports_target_qmv(_experts()),
    }
    monkeypatch.delenv(kernels.FUSED_ENV, raising=False)
    assert {name: bool(check()) for name, check in checks.items()} == dict.fromkeys(
        checks, True
    )
    monkeypatch.setenv(kernels.FUSED_ENV, "0")
    assert {name: bool(check()) for name, check in checks.items()} == dict.fromkeys(
        checks, False
    )


def _experts():
    layer = _moe(seed=1)
    return layer.experts


def _moe(seed: int) -> MoE:
    mx.random.seed(seed)
    layer = MoE(
        ModelArgs(
            dim=512,
            moe_inter_dim=256,
            n_routed_experts=64,
            n_activated_experts=4,
            swiglu_limit=10.0,
        )
    )
    layer.update(
        nn.utils.tree_map(
            lambda p: (mx.random.normal(p.shape) * 0.05).astype(mx.bfloat16),
            layer.parameters(),
        )
    )
    layer.gate.bias = mx.random.normal((64,)) * 0.02
    nn.quantize(
        layer,
        group_size=64,
        bits=2,
        class_predicate=lambda _path, m: hasattr(m, "to_quantized"),
    )
    layer.experts.__class__ = serving.ExactDirectDownSwitchGLU
    mx.eval(layer.parameters())
    return layer


def _tiny_model() -> Model:
    mx.random.seed(3)
    model = Model(
        ModelArgs(
            vocab_size=64,
            dim=64,
            n_layers=3,
            moe_inter_dim=16,
            n_heads=8,
            head_dim=128,
            rope_head_dim=64,
            q_lora_rank=32,
            o_lora_rank=16,
            o_groups=2,
            window_size=8,
            n_routed_experts=4,
            n_activated_experts=2,
            compress_ratios=(0, 4, 4),
            kv_source_layers=(1,),
            index_source_layers=(1,),
            index_n_heads=2,
            index_head_dim=64,
            index_topk=2,
            hc_mult=4,
        )
    )
    model.set_dtype(mx.bfloat16)
    return model


def _decode(model, steps: int = 3) -> np.ndarray:
    cache = model.make_cache(max_seq_len=48)
    logits = model(mx.array([list(range(1, 22))]), cache, last_logit_only=True)
    outs = [logits]
    for token in range(steps):
        outs.append(model(mx.array([[token + 5]]), cache))
    mx.eval(outs)
    return np.concatenate([np.array(o.astype(mx.float32)).reshape(-1) for o in outs])


@needs_metal
def test_model_decode_off_builds_no_kernel_and_equals_stock(monkeypatch):
    model = _tiny_model()
    monkeypatch.delenv(kernels.FUSED_ENV, raising=False)
    used = _record_kernels(monkeypatch)
    _decode(model)
    assert used, "the default path should take at least one custom kernel"

    monkeypatch.setenv(kernels.FUSED_ENV, "0")
    _forbid_kernels(monkeypatch)
    off = _decode(model)
    monkeypatch.undo()

    _stock(monkeypatch)
    np.testing.assert_array_equal(off, _decode(model))


@needs_metal
def test_moe_decode_off_builds_no_kernel_and_equals_stock(monkeypatch):
    layer = _moe(seed=2)
    x = (mx.random.normal((1, 1, 512)) * 0.5).astype(mx.bfloat16)
    monkeypatch.delenv(kernels.FUSED_ENV, raising=False)
    used = _record_kernels(monkeypatch)
    mx.eval(layer(x))
    assert used

    monkeypatch.setenv(kernels.FUSED_ENV, "0")
    _forbid_kernels(monkeypatch)
    off = layer(x)
    mx.eval(off)
    monkeypatch.undo()

    _stock(monkeypatch)
    stock = layer(x)
    assert mx.array_equal(off, stock).item()
