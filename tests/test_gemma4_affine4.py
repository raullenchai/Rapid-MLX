"""Metal numerical qualification for the exact Gemma 4 26B expert geometry."""

import pytest

try:
    import mlx.core as mx
    import mlx.nn as nn
except ImportError:
    pytest.skip("Native MLX/Metal unavailable", allow_module_level=True)
if not mx.metal.is_available():
    pytest.skip("Metal required", allow_module_level=True)

from rapid_mlx.patches import gemma4_affine4 as fused  # noqa: E402


def relative_rms(actual, expected):
    a, b = actual.astype(mx.float32), expected.astype(mx.float32)
    return mx.sqrt(mx.mean((a - b) ** 2) / mx.maximum(mx.mean(b**2), 1e-12)).item()


def test_gelu_table_matches_activation_dtype():
    _, _, table = fused.kernels()
    values = mx.arange(65536, dtype=mx.uint16).view(mx.bfloat16)
    expected = nn.gelu_approx(values)
    finite = mx.isfinite(expected)
    assert mx.array_equal(
        mx.where(finite, table, 0), mx.where(finite, expected, 0)
    ).item()


@pytest.fixture(scope="module")
def quantized_banks():
    # Quantize one bank at a time and evaluate before releasing the float bank.
    # No checkpoint, downloads or model-sized allocations are involved.
    banks = []
    for seed, shape, scale in (
        (32, (128, 704, 2816), 2816**-0.5),
        (33, (128, 704, 2816), 2816**-0.5),
        (34, (128, 2816, 704), 704**-0.5),
    ):
        weight = (mx.random.normal(shape, key=mx.random.key(seed)) * scale).astype(
            mx.bfloat16
        )
        bank = mx.quantize(weight, group_size=64, bits=4, mode="affine")
        mx.eval(*bank)
        banks.append(bank)
    return banks


@pytest.mark.parametrize("width", [1, 2, 9])
@pytest.mark.parametrize("combined", [False, True])
@pytest.mark.parametrize("family", ["vendored", "lm", "vlm"])
def test_exact_geometry_against_composed(
    quantized_banks, width, combined, family, monkeypatch
):
    from types import SimpleNamespace

    from rapid_mlx.moe_fusion import fuse_gate_up

    module_name, switch_name = {
        "vendored": (
            "rapid_mlx.models.gemma4_vendored.language",
            "mlx_lm.models.switch_layers",
        ),
        "lm": ("mlx_lm.models.gemma4_text", "mlx_lm.models.switch_layers"),
        "vlm": ("mlx_vlm.models.gemma4.language", "mlx_vlm.models.switch_layers"),
    }[family]
    module = pytest.importorskip(module_name)
    switch = pytest.importorskip(switch_name)
    Experts, Router, GeGLU = module.Experts, module.Router, module.GeGLU
    SwitchGLU, QuantizedSwitchLinear = switch.SwitchGLU, switch.QuantizedSwitchLinear
    config = SimpleNamespace(
        hidden_size=2816,
        moe_intermediate_size=704,
        num_experts=128,
        top_k_experts=8,
        rms_norm_eps=1e-6,
    )
    # Build the composed modules around shared quantized banks without allocating
    # three more unquantized copies of the full expert geometry.
    block = dict.__new__(Experts)
    nn.Module.__init__(block)
    sm = dict.__new__(SwitchGLU)
    nn.Module.__init__(sm)
    sm.activation = GeGLU()
    for name, bank in zip(("gate_proj", "up_proj", "down_proj"), quantized_banks):
        p = dict.__new__(QuantizedSwitchLinear)
        nn.Module.__init__(p)
        p.bits, p.group_size, p.mode = 4, 64, "affine"
        p.weight, p.scales, p.biases = bank
        setattr(sm, name, p)
    block.switch_glu = sm
    block.eval()
    router = Router(config)
    router.per_expert_scale = mx.linspace(0.25, 1.75, 128).astype(mx.bfloat16)
    router.proj.weight = (
        mx.random.normal((128, 2816), key=mx.random.key(95)) * 0.02
    ).astype(mx.bfloat16)
    router.proj = router.proj.to_quantized(group_size=64, bits=8)
    router.scale = router.scale.astype(mx.bfloat16)
    x = mx.random.normal((1, width, 2816), key=mx.random.key(91)).astype(mx.bfloat16)
    ids, scores = router(x)
    assert scores.dtype == mx.bfloat16
    # Force boundary experts while retaining the real router's non-unit weights.
    ids = mx.concatenate(
        [
            mx.zeros_like(ids[..., :1]),
            ids[..., 1:-1],
            mx.full(ids[..., -1:].shape, 127, ids.dtype),
        ],
        axis=-1,
    )
    expected = block(x, ids, scores)
    mx.eval(expected)
    if combined:
        monkeypatch.setenv("RAPID_MLX_MOE_GATE_UP_FUSION", "1")
        monkeypatch.setattr(
            "rapid_mlx.moe_fusion.install_enabled", lambda *a, **k: True
        )
        # fuse_gate_up rewrites the SwitchGLU class; restore it after this test.
        cls = type(block.switch_glu)
        monkeypatch.setattr(cls, "__call__", cls.__call__)
        for name in ("_rapid_gate_up_fused_call", "_rapid_gate_up_original_call"):
            if name in cls.__dict__:
                monkeypatch.setattr(cls, name, cls.__dict__[name])
            else:
                monkeypatch.setattr(cls, name, None, raising=False)
                monkeypatch.delattr(cls, name)
        assert fuse_gate_up(block, extended_call_contract=True) == 1
    assert fused._compatible(block)
    actual = fused.forward(block, x, ids, scores)
    error = relative_rms(actual, expected)
    print(f"family={family} width={width} combined={combined} relative_rms={error:.6f}")
    # Float32 dot reduction order differs from gather_qmm; bf16 projections,
    # GeGLU, and weighted products each round before the float32 expert sum.
    # Bound RMS rather than per-element relative error near cancellation zeros.
    assert error <= 0.02
    assert mx.all(mx.isfinite(actual)).item()
    if width == 1:
        assert fused._probe(block)
    if width > 1:
        singles = mx.concatenate(
            [
                fused.forward(
                    block, x[:, t : t + 1], ids[:, t : t + 1], scores[:, t : t + 1]
                )
                for t in range(width)
            ],
            axis=1,
        )
        assert mx.array_equal(actual, singles).item()
