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

import inspect
import weakref
from unittest.mock import Mock

import pytest

mx = pytest.importorskip("mlx.core")
pytestmark = pytest.mark.requires_mlx

import mlx.nn as nn
from mlx_lm.models.switch_layers import QuantizedSwitchLinear, SwitchGLU, SwitchLinear

from rapid_mlx import moe_fusion

# Small MoE geometry: E experts, k active, quantization-friendly dims.
E, HID, INTER, K = 8, 64, 128, 2


class TinyMoE(nn.Module):
    """Minimal container so fuse_gate_up's module scan finds the layers."""

    def __init__(self, n_layers=2, quantize=True):
        super().__init__()
        self.layers = [SwitchGLU(HID, INTER, E) for _ in range(n_layers)]
        if quantize:
            nn.quantize(self, group_size=32, bits=4)


class TinyVLMMoE(nn.Module):
    """mlx-vlm owns a separate, API-compatible SwitchGLU class family."""

    def __init__(self, n_layers=2, quantize=True):
        super().__init__()
        from mlx_vlm.models.switch_layers import SwitchGLU as VLMSwitchGLU

        self.layers = [VLMSwitchGLU(HID, INTER, E) for _ in range(n_layers)]
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
    def test_removed_up_projection_is_released_per_layer(self, monkeypatch):
        """The module scan must not pin old expert buffers until fusion ends."""
        model = TinyMoE(n_layers=2)
        mx.eval(model.parameters())
        old_ups = [weakref.ref(layer.up_proj) for layer in model.layers]
        released_at_clear = []

        monkeypatch.setattr(
            moe_fusion.mx,
            "clear_cache",
            lambda: released_at_clear.append(any(ref() is None for ref in old_ups)),
        )

        assert moe_fusion.fuse_gate_up(model) == 2
        assert released_at_clear[0] is True

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

    def test_mllm_wrapper_applies_fusion_after_load(self, monkeypatch):
        """The VLM lane must not leave eligible legacy projections unfused.

        GLM-5.3-Flash is forced through ``MLXMultimodalLM``. Before this
        regression guard, only the text BatchedEngine called ``fuse_gate_up``
        after loading, so all 42 GLM sparse layers missed the optimization.
        mlx-vlm 0.7.2 owns an equivalent exact gate/up fast path and expands
        the SwitchGLU call contract; Rapid must leave that class untouched.
        """
        import mlx_vlm
        import mlx_vlm.utils

        from rapid_mlx.models import glm5_shardwise_load, mllm
        from rapid_mlx.utils import tokenizer as tokenizer_utils

        model = TinyVLMMoE(n_layers=1)
        model.config = type("Config", (), {})()
        processor = type(
            "Processor",
            (),
            {"tokenizer": type("Tokenizer", (), {})()},
        )()
        monkeypatch.setattr(mllm, "_require_mlx_vlm", lambda: None)
        loader = Mock(return_value=(model, processor))
        monkeypatch.setattr(mlx_vlm, "load", loader)
        # Darwin routes GLM through the shardwise loader instead of mlx_vlm.load.
        monkeypatch.setattr(glm5_shardwise_load, "load_glm5_shardwise", loader)
        monkeypatch.setattr(
            mlx_vlm.utils,
            "load_config",
            lambda *args, **kwargs: {"model_type": "glm5_next"},
        )
        monkeypatch.setattr(
            tokenizer_utils,
            "augment_eos_token_ids_from_generation_config",
            lambda *args, **kwargs: None,
        )
        monkeypatch.setattr(
            tokenizer_utils,
            "repair_byte_level_decoder",
            lambda *args, **kwargs: None,
        )

        wrapper = mllm.MLXMultimodalLM("unit-test/glm5-next")
        wrapper.load()
        loader.assert_called_once()

        layer = model.layers[0]
        if moe_fusion._supports_fused_call_contract(type(layer)):
            assert hasattr(layer, "gate_up_proj")
            assert not hasattr(layer, "gate_proj")
            assert not hasattr(layer, "up_proj")
        else:
            assert "exact_affine_switch_gate_up" in inspect.getsource(
                type(layer).__call__
            )
            assert not hasattr(layer, "gate_up_proj")
            assert hasattr(layer, "gate_proj")
            assert hasattr(layer, "up_proj")

    def test_mlx_vlm_family_is_byte_identical(self):
        model = TinyVLMMoE(n_layers=1)
        mx.eval(model.parameters())
        x, inds = _decode_inputs()
        before = _run_all(model, x, inds)[0]

        compatible = moe_fusion._supports_fused_call_contract(type(model.layers[0]))
        assert moe_fusion.fuse_gate_up(model) == int(compatible)
        after = _run_all(model, x, inds)[0]

        assert bool(mx.array_equal(before, after))

    def test_mlx_vlm_sorted_path_is_byte_identical(self):
        model = TinyVLMMoE(n_layers=1)
        mx.eval(model.parameters())
        x, inds = _decode_inputs(tokens=40)
        before = _run_all(model, x, inds)[0]

        compatible = moe_fusion._supports_fused_call_contract(type(model.layers[0]))
        assert moe_fusion.fuse_gate_up(model) == int(compatible)
        after = _run_all(model, x, inds)[0]

        assert bool(mx.array_equal(before, after))

    def test_non_glm_mllm_does_not_enable_unqualified_fusion(self, monkeypatch):
        import mlx_vlm
        import mlx_vlm.utils

        from rapid_mlx.models import mllm
        from rapid_mlx.utils import tokenizer as tokenizer_utils

        model = TinyVLMMoE(n_layers=1)
        model.config = type("Config", (), {})()
        processor = type(
            "Processor",
            (),
            {"tokenizer": type("Tokenizer", (), {})()},
        )()
        monkeypatch.setattr(mllm, "_require_mlx_vlm", lambda: None)
        monkeypatch.setattr(mlx_vlm, "load", lambda *args, **kwargs: (model, processor))
        monkeypatch.setattr(
            mlx_vlm.utils,
            "load_config",
            lambda *args, **kwargs: {"model_type": "other_vlm"},
        )
        monkeypatch.setattr(
            tokenizer_utils,
            "augment_eos_token_ids_from_generation_config",
            lambda *args, **kwargs: None,
        )
        monkeypatch.setattr(
            tokenizer_utils,
            "repair_byte_level_decoder",
            lambda *args, **kwargs: None,
        )

        wrapper = mllm.MLXMultimodalLM("unit-test/other-vlm")
        wrapper.load()

        assert hasattr(model.layers[0], "gate_proj")
        assert hasattr(model.layers[0], "up_proj")


class TestGates:
    def test_changed_switch_glu_call_contract_fails_closed(self):
        class FutureSwitchGLU:
            def __call__(self, x, indices, routing_weights):
                return x, indices, routing_weights

        assert not moe_fusion._supports_fused_call_contract(FutureSwitchGLU)

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


class TestExtendedVLMContract:
    @pytest.fixture(autouse=True)
    def restore_class_patch(self, monkeypatch):
        from mlx_vlm.models.switch_layers import SwitchGLU as VLMSwitchGLU

        monkeypatch.setattr(VLMSwitchGLU, "__call__", VLMSwitchGLU.__call__)
        for name in ("_rapid_gate_up_fused_call", "_rapid_gate_up_original_call"):
            monkeypatch.setattr(
                VLMSwitchGLU, name, getattr(VLMSwitchGLU, name, None), raising=False
            )

    def test_qwen_mllm_wrapper_fuses_extended_family(self, monkeypatch):
        import mlx_vlm
        import mlx_vlm.utils

        from rapid_mlx import qwen35_fused_gdn_decode, qwen35_moe_router
        from rapid_mlx.models import mllm
        from rapid_mlx.utils import tokenizer as tokenizer_utils

        model = TinyVLMMoE(n_layers=1)
        model.config = type("Config", (), {})()
        processor = type("Processor", (), {"tokenizer": type("Tokenizer", (), {})()})()
        monkeypatch.setattr(mllm, "_require_mlx_vlm", lambda: None)
        monkeypatch.setattr(mlx_vlm, "load", lambda *a, **kw: (model, processor))
        monkeypatch.setattr(
            mlx_vlm.utils, "load_config", lambda *a, **kw: {"model_type": "qwen3_5_moe"}
        )
        monkeypatch.setattr(
            qwen35_moe_router, "install_qwen35_moe_router", lambda m: None
        )
        monkeypatch.setattr(
            qwen35_fused_gdn_decode, "install_qwen35_fused_gdn_decode", lambda m: None
        )
        for name in (
            "augment_eos_token_ids_from_generation_config",
            "repair_byte_level_decoder",
        ):
            monkeypatch.setattr(tokenizer_utils, name, lambda *a, **kw: None)

        mllm.MLXMultimodalLM("unit-test/qwen35-moe").load()
        layer = model.layers[0]
        assert hasattr(layer, "gate_up_proj")
        assert not hasattr(layer, "gate_proj")
        assert not hasattr(layer, "up_proj")

    @pytest.mark.parametrize("quantize", [True, False])
    @pytest.mark.parametrize(
        "batch,tokens", [(1, 1), (1, 2), (1, 5), (1, 8), (1, 40), (3, 1)]
    )
    @pytest.mark.parametrize("combine", [False, True])
    def test_byte_identity(self, quantize, batch, tokens, combine):
        model = TinyVLMMoE(n_layers=1, quantize=quantize)
        model.eval()
        x, inds = _decode_inputs(tokens=tokens)
        x = mx.repeat(x, batch, axis=0)
        inds = mx.repeat(inds, batch, axis=0)
        weights = mx.softmax(mx.random.normal(inds.shape), axis=-1) if combine else None
        shared = mx.random.normal(x.shape).astype(x.dtype) if combine else None
        layer = model.layers[0]
        before = layer(x, inds, weights, shared)
        mx.eval(before)
        assert moe_fusion.fuse_gate_up(model, extended_call_contract=True) == 1
        after = layer(x, inds, weights, shared)
        mx.eval(after)
        assert bool(mx.array_equal(before, after))
        patched_call = type(layer).__call__
        assert moe_fusion.fuse_gate_up(model, extended_call_contract=True) == 0
        assert (
            moe_fusion.fuse_gate_up(TinyVLMMoE(n_layers=1), extended_call_contract=True)
            == 1
        )
        assert type(layer).__call__ is patched_call
        # A new instance still requires opt-in even after its class is patched.
        assert moe_fusion.fuse_gate_up(TinyVLMMoE(n_layers=1)) == 0

    def test_unfused_forwards_all_arguments(self, monkeypatch):
        from mlx_vlm.models.switch_layers import SwitchGLU as VLMSwitchGLU

        spy = Mock(return_value=object())

        # Keep an inspectable extended signature for enrollment.
        def stock(self, x, indices, weights=None, shared=None, residual=None):
            return spy(self, x, indices, weights, shared, residual)

        monkeypatch.setattr(VLMSwitchGLU, "__call__", stock)
        assert (
            moe_fusion.fuse_gate_up(TinyVLMMoE(n_layers=1), extended_call_contract=True)
            == 1
        )
        fresh = TinyVLMMoE(n_layers=1).layers[0]
        x, inds = _decode_inputs()
        weights = mx.ones(inds.shape, dtype=x.dtype)
        shared = mx.zeros(x.shape, dtype=x.dtype)
        # Spy returns a sentinel so even opaque residuals must pass unchanged.
        sentinel = spy.return_value
        residual = object()
        assert fresh(x, inds, weights, shared, residual) is sentinel
        args = spy.call_args.args
        assert all(
            a is b for a, b in zip(args, (fresh, x, inds, weights, shared, residual))
        )

    @pytest.mark.parametrize(
        "missing", ["combine", "block_size", "tokenwise", "linear", "contract"]
    )
    def test_missing_helpers_or_unknown_contract_fail_closed(
        self, monkeypatch, missing
    ):
        import sys

        import mlx_vlm.models.linear as linear
        from mlx_vlm.models.switch_layers import SwitchGLU as VLMSwitchGLU

        model = TinyVLMMoE(n_layers=1)
        if missing == "combine":
            monkeypatch.delattr(VLMSwitchGLU, "_combine")
        elif missing == "block_size":
            monkeypatch.delattr(linear, "DECODE_BLOCK_SIZE")
        elif missing == "tokenwise":
            monkeypatch.setattr(linear, "tokenwise", None)
        elif missing == "linear":
            monkeypatch.delitem(sys.modules, "mlx_vlm.models.linear")
        else:
            monkeypatch.setattr(
                VLMSwitchGLU, "__call__", lambda self, x, indices, future: x
            )
        assert moe_fusion.fuse_gate_up(model, extended_call_contract=True) == 0
        assert hasattr(model.layers[0], "gate_proj")
