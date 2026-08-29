# SPDX-License-Identifier: Apache-2.0
"""Correctness gates for the MoE gate+up expert-projection fusion.

The fusion must be BIT-exact — affine quantization packs each output row
independently, so a concatenated gate+up gather_qmm returns the same
bytes as two separate calls — and strictly opt-out-able. These tests pin
byte equality on quantized and unquantized SwitchGLU, the structural
_can_fuse gate, in-place rewrite semantics (original buffers dropped,
stock path preserved for unfused instances), idempotency, and the env
kill-switch.
"""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")
pytestmark = pytest.mark.requires_mlx

import mlx.nn as nn
from mlx_lm.models.switch_layers import QuantizedSwitchLinear, SwitchGLU, SwitchLinear

from vllm_mlx import moe_fusion

# Small MoE geometry: E experts, k active, quantization-friendly dims.
E, HID, INTER, K = 8, 64, 128, 2


class TinyMoE(nn.Module):
    """Minimal container so fuse_gate_up's module scan finds the layers."""

    def __init__(self, n_layers=2, quantize=True):
        super().__init__()
        self.layers = [SwitchGLU(HID, INTER, E) for _ in range(n_layers)]
        if quantize:
            nn.quantize(self, group_size=32, bits=4)


def _decode_inputs(seed=3, tokens=1):
    mx.random.seed(seed)
    x = mx.random.normal((1, tokens, HID)).astype(mx.float16)
    inds = mx.random.randint(0, E, (1, tokens, K))
    mx.eval(x, inds)
    return x, inds


def _run_all(model, x, inds):
    outs = [layer(x, inds) for layer in model.layers]
    mx.eval(*outs)
    return outs


class TestBitExactness:
    @pytest.mark.parametrize("quantize", [True, False])
    def test_fused_output_is_byte_identical(self, quantize):
        model = TinyMoE(quantize=quantize)
        mx.eval(model.parameters())
        x, inds = _decode_inputs()
        before = _run_all(model, x, inds)
        n = moe_fusion.fuse_gate_up(model)
        assert n == 2
        after = _run_all(model, x, inds)
        for b, a in zip(before, after):
            assert bool(mx.array_equal(b, a)), "fusion changed output bytes"

    def test_sorted_path_byte_identical(self):
        # indices.size >= 64 triggers the gather-sort branch; the fused
        # call must keep it byte-identical too (prefill / batched decode).
        model = TinyMoE()
        mx.eval(model.parameters())
        x, inds = _decode_inputs(tokens=40)  # 40*2 = 80 >= 64
        before = _run_all(model, x, inds)
        assert moe_fusion.fuse_gate_up(model) == 2
        after = _run_all(model, x, inds)
        for b, a in zip(before, after):
            assert bool(mx.array_equal(b, a))


class TestRewriteSemantics:
    def test_originals_dropped_and_container_reused(self):
        model = TinyMoE()
        mx.eval(model.parameters())
        assert moe_fusion.fuse_gate_up(model) == 2
        for layer in model.layers:
            assert hasattr(layer, "gate_up_proj")
            assert not hasattr(layer, "gate_proj")
            assert not hasattr(layer, "up_proj")
            # fused output axis is 2*inter
            assert layer.gate_up_proj["scales"].shape[1] == 2 * INTER

    def test_idempotent(self):
        model = TinyMoE()
        mx.eval(model.parameters())
        assert moe_fusion.fuse_gate_up(model) == 2
        # second run finds nothing eligible (gate_up_proj present)
        assert moe_fusion.fuse_gate_up(model) == 0

    def test_unfused_instances_keep_stock_path(self):
        # A model with a fused layer and a fresh (unfused) layer must run
        # both correctly through the patched class __call__.
        model = TinyMoE()
        mx.eval(model.parameters())
        assert moe_fusion.fuse_gate_up(model) == 2
        fresh = SwitchGLU(HID, INTER, E)
        nn.quantize(fresh, group_size=32, bits=4)
        mx.eval(fresh.parameters())
        x, inds = _decode_inputs()
        out = fresh(x, inds)  # goes through orig_call branch
        mx.eval(out)
        assert out.shape == (1, 1, K, HID)

    def test_dense_model_is_noop(self):
        class Dense(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(HID, HID)

        assert moe_fusion.fuse_gate_up(Dense()) == 0


class TestGates:
    def test_env_opt_out(self, monkeypatch):
        monkeypatch.setenv("RAPID_MLX_MOE_GATE_UP_FUSION", "0")
        model = TinyMoE()
        mx.eval(model.parameters())
        assert moe_fusion.fuse_gate_up(model) == 0
        assert hasattr(model.layers[0], "gate_proj")

    def test_mismatched_quant_params_not_fused(self):
        # gate and up quantized with different bit widths -> ineligible
        m4 = TinyMoE(n_layers=1, quantize=False)
        m8 = TinyMoE(n_layers=1, quantize=False)
        nn.quantize(m4, group_size=32, bits=4)
        nn.quantize(m8, group_size=32, bits=8)
        layer = m4.layers[0]
        layer.up_proj = m8.layers[0].up_proj
        mx.eval(m4.parameters())
        assert isinstance(layer.gate_proj, QuantizedSwitchLinear)
        assert moe_fusion.fuse_gate_up(m4) == 0
        assert hasattr(layer, "gate_proj")

    def test_subclass_not_fused(self):
        # Exact-type gate: a subclass may override __call__ and never read
        # gate_up_proj, so it must be left alone.
        class CustomGLU(SwitchGLU):
            pass

        class Holder(nn.Module):
            def __init__(self):
                super().__init__()
                self.glu = CustomGLU(HID, INTER, E)

        model = Holder()
        nn.quantize(model, group_size=32, bits=4)
        mx.eval(model.parameters())
        assert moe_fusion.fuse_gate_up(model) == 0
        assert hasattr(model.glu, "gate_proj")


class TestPlainSwitchLinear:
    def test_unquantized_fusion_bit_exact_with_bias(self):
        class BiasMoE(nn.Module):
            def __init__(self):
                super().__init__()
                self.glu = SwitchGLU(HID, INTER, E, bias=True)

        model = BiasMoE()
        layer = model.glu
        assert isinstance(layer.gate_proj, SwitchLinear)
        # the additive-bias fusion path must actually be exercised
        assert "bias" in layer.gate_proj and "bias" in layer.up_proj
        mx.eval(model.parameters())
        x, inds = _decode_inputs()
        before = layer(x, inds)
        mx.eval(before)
        assert moe_fusion.fuse_gate_up(model) == 1
        assert "bias" in layer.gate_up_proj
        after = layer(x, inds)
        mx.eval(after)
        assert bool(mx.array_equal(before, after))
