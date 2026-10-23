# SPDX-License-Identifier: Apache-2.0
"""Correctness and enrollment tests for the fused Qwen MoE router."""

from __future__ import annotations

import builtins
import sys
from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")
pytestmark = pytest.mark.requires_mlx

import mlx.nn as nn

import rapid_mlx.qwen35_moe_router as router


def _stock(probs: mx.array, top_k: int = 8) -> tuple[mx.array, mx.array]:
    indices = mx.argpartition(probs, kth=-top_k, axis=-1)[..., -top_k:]
    scores = mx.take_along_axis(probs, indices, axis=-1)
    return indices, scores / scores.sum(axis=-1, keepdims=True)


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("rows", [1, 4, 8])
def test_kernel_is_bit_exact_for_decode_widths(dtype, rows):
    mx.random.seed(36 + rows)
    for _ in range(50):
        logits = (mx.random.normal((1, rows, 256)) * 2).astype(dtype)
        probs = mx.softmax(logits, axis=-1, precise=True)
        expected_i, expected_s = _stock(probs)
        actual_i, actual_s = router.fused_router_topk(probs, 8)
        mx.eval(expected_i, expected_s, actual_i, actual_s)
        assert mx.array_equal(actual_i, expected_i)
        assert mx.array_equal(actual_s, expected_s)


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
def test_kernel_matches_argpartition_tie_order():
    probs = mx.ones((1, 8, 256), dtype=mx.bfloat16)
    expected_i, expected_s = _stock(probs)
    actual_i, actual_s = router.fused_router_topk(probs, 8)
    mx.eval(expected_i, expected_s, actual_i, actual_s)
    assert mx.array_equal(actual_i, expected_i)
    assert mx.array_equal(actual_s, expected_s)


def test_probe_rejects_a_numerically_different_kernel(monkeypatch):
    def wrong(probs, top_k):
        shape = (*probs.shape[:-1], top_k)
        return mx.zeros(shape, dtype=mx.uint32), mx.zeros(shape, dtype=probs.dtype)

    monkeypatch.setattr(router, "fused_router_topk", wrong)
    assert not router._probe(256, 8, mx.bfloat16)


def test_eligibility_is_model_local_and_decode_only():
    block = SimpleNamespace(
        num_experts=256,
        top_k=8,
        norm_topk_prob=True,
        sharding_group=None,
    )
    decode = mx.zeros((1, 1, 2048), dtype=mx.bfloat16)
    verify = mx.zeros((1, 8, 2048), dtype=mx.float16)
    prefill = mx.zeros((1, 9, 2048), dtype=mx.bfloat16)

    assert not router._eligible(decode, block)
    setattr(block, router._TAG, True)
    assert router._eligible(decode, block)
    assert router._eligible(verify, block)
    assert not router._eligible(prefill, block)
    assert not router._eligible(mx.zeros((2048,), dtype=mx.bfloat16), block)
    block.num_experts = 250
    assert not router._eligible(decode, block)


def test_kernel_wrapper_rejects_unqualified_shapes():
    with pytest.raises(ValueError, match="rank"):
        router.fused_router_topk(mx.zeros((256,), dtype=mx.bfloat16), 8)
    with pytest.raises(ValueError, match="shape"):
        router.fused_router_topk(mx.zeros((1, 9, 256), dtype=mx.bfloat16), 8)
    with pytest.raises(ValueError, match="shape"):
        router.fused_router_topk(mx.zeros((1, 1, 250), dtype=mx.bfloat16), 8)


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
def test_install_tags_only_blocks_from_the_loaded_model(monkeypatch):
    from mlx_lm.models.qwen3_next import Qwen3NextSparseMoeBlock

    block = Qwen3NextSparseMoeBlock.__new__(Qwen3NextSparseMoeBlock)
    nn.Module.__init__(block)
    block.num_experts = 256
    block.top_k = 8
    block.norm_topk_prob = True
    block.sharding_group = None

    model = SimpleNamespace(named_modules=lambda: iter((("mlp", block),)))
    monkeypatch.setattr(router, "_patch_class", lambda _cls: None)
    assert router.install_qwen35_moe_router(model) == 1
    assert getattr(block, router._TAG) is True


def test_install_honors_disable_switch(monkeypatch):
    monkeypatch.setenv("RAPID_MLX_QWEN35_MOE_ROUTER", "0")
    model = SimpleNamespace(named_modules=lambda: iter(()))
    assert router.install_qwen35_moe_router(model) == 0


def test_install_ignores_non_module_model_placeholder():
    assert router.install_qwen35_moe_router(object()) == 0


def test_install_fails_closed_without_metal(monkeypatch):
    monkeypatch.setattr(mx.metal, "is_available", lambda: False)
    assert router.install_qwen35_moe_router(object()) == 0


def test_install_fails_closed_when_text_backend_class_is_unavailable(monkeypatch):
    original_import = builtins.__import__

    def rejecting_import(name, *args, **kwargs):
        if name == "mlx_lm.models.qwen3_next":
            raise ImportError("backend class unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", rejecting_import)
    monkeypatch.delitem(
        sys.modules, "mlx_vlm.models.qwen3_5_moe.language", raising=False
    )
    model = SimpleNamespace(named_modules=lambda: iter(()))
    assert router.install_qwen35_moe_router(model) == 0


def test_install_enrolls_already_loaded_vision_backend_class(monkeypatch):
    class VisionBlock:
        pass

    block = VisionBlock()
    block.num_experts = 256
    block.top_k = 8
    block.norm_topk_prob = True
    module = SimpleNamespace(Qwen3_5MoeSparseMoeBlock=VisionBlock)
    monkeypatch.setitem(sys.modules, "mlx_vlm.models.qwen3_5_moe.language", module)
    monkeypatch.setattr(router, "_probe", lambda *args: True)
    monkeypatch.setattr(router, "_patch_class", lambda _cls: None)
    model = SimpleNamespace(named_modules=lambda: iter((("mlp", block),)))

    assert router.install_qwen35_moe_router(model) == 1
    assert getattr(block, router._TAG)


@pytest.mark.parametrize("probe_result", [False, RuntimeError("probe failed")])
def test_install_fails_closed_when_parity_probe_does_not_pass(
    monkeypatch, probe_result
):
    from mlx_lm.models.qwen3_next import Qwen3NextSparseMoeBlock

    block = Qwen3NextSparseMoeBlock.__new__(Qwen3NextSparseMoeBlock)
    nn.Module.__init__(block)
    block.num_experts = 256
    block.top_k = 8
    block.norm_topk_prob = True
    model = SimpleNamespace(named_modules=lambda: iter((("mlp", block),)))

    def probe(*args):
        if isinstance(probe_result, Exception):
            raise probe_result
        return probe_result

    monkeypatch.setattr(router, "_probe", probe)
    assert router.install_qwen35_moe_router(model) == 0
    assert not getattr(block, router._TAG, False)


def test_patched_class_keeps_stock_fallback_and_matches_fused_composition(monkeypatch):
    class Block:
        def __call__(self, x):
            return "stock"

    block = Block()
    block.num_experts = 256
    block.top_k = 8
    block.norm_topk_prob = True
    block.sharding_group = None
    setattr(block, router._TAG, True)
    block.gate = lambda x: mx.zeros((*x.shape[:-1], 256), dtype=x.dtype)
    block.switch_mlp = lambda x, indices: mx.ones((*x.shape[:-1], 8, 2), dtype=x.dtype)
    block.shared_expert = lambda x: mx.ones((*x.shape[:-1], 2), dtype=x.dtype)
    block.shared_expert_gate = lambda x: mx.zeros((*x.shape[:-1], 2), dtype=x.dtype)
    indices = mx.zeros((1, 1, 8), dtype=mx.uint32)
    scores = mx.full((1, 1, 8), 0.125, dtype=mx.bfloat16)
    monkeypatch.setattr(router, "fused_router_topk", lambda *_: (indices, scores))

    router._patch_class(Block)
    assert block(mx.zeros((1, 9, 2), dtype=mx.bfloat16)) == "stock"
    output = block(mx.zeros((1, 1, 2), dtype=mx.bfloat16))
    mx.eval(output)
    assert mx.array_equal(output, mx.full((1, 1, 2), 1.5, dtype=mx.bfloat16))
    router._patch_class(Block)


@pytest.mark.parametrize("spec_decode", ["none", "mtp", "draft", "unknown", None])
def test_mllm_load_enrolls_qwen_moe_optimizations(monkeypatch, spec_decode, tmp_path):
    import mlx_vlm
    import mlx_vlm.utils

    from rapid_mlx import moe_fusion
    from rapid_mlx.models import mllm
    from rapid_mlx.utils import tokenizer as tokenizer_utils

    model = SimpleNamespace(config=SimpleNamespace())
    processor = SimpleNamespace(tokenizer=SimpleNamespace())
    fused = []
    routed = []
    monkeypatch.setattr(mllm, "_require_mlx_vlm", lambda: None)
    monkeypatch.setattr(mlx_vlm, "load", lambda *args, **kwargs: (model, processor))
    monkeypatch.setattr(
        mlx_vlm.utils,
        "load_config",
        lambda *args, **kwargs: {"model_type": "qwen3_5_moe"},
    )
    monkeypatch.setattr(
        moe_fusion,
        "fuse_gate_up",
        lambda model, *, extended_call_contract=False: fused.append(
            (model, extended_call_contract)
        ),
    )
    monkeypatch.setattr(router, "install_qwen35_moe_router", routed.append)
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

    from rapid_mlx import gdn_in_proj_fusion, qwen35_fused_gdn_decode

    calls = []
    monkeypatch.setattr(
        gdn_in_proj_fusion,
        "fuse_gdn_in_proj",
        lambda m: calls.append(("projection", m)),
    )
    monkeypatch.setattr(
        qwen35_fused_gdn_decode,
        "install_qwen35_fused_gdn_decode",
        lambda m: calls.append(("decode", m)),
    )
    monkeypatch.setattr(mlx_vlm.utils, "get_model_path", lambda *a, **kw: tmp_path)
    mllm.MLXMultimodalLM("local/qwen-moe", spec_decode=spec_decode).load()
    assert calls == (
        [("projection", model), ("decode", model)]
        if spec_decode == "none"
        else [("decode", model)]
    )

    assert fused == [(model, True)]
    assert routed == [model]


# -- Extended mlx-vlm SwitchGLU contract fusion (#4823) ----------------------
# Lives here because this file runs on the Apple Silicon CI lane.

from unittest.mock import Mock  # noqa: E402

from rapid_mlx import moe_fusion  # noqa: E402

E, HID, INTER, K = 8, 64, 128, 2


class TinyVLMMoE(nn.Module):
    """mlx-vlm owns a separate SwitchGLU class family."""

    def __init__(self, n_layers=2, quantize=True, *, real_geometry=False, experts=E):
        super().__init__()
        from mlx_vlm.models.switch_layers import SwitchGLU as VLMSwitchGLU

        hidden, inter = (512, 512) if real_geometry else (HID, INTER)
        self.layers = [VLMSwitchGLU(hidden, inter, experts) for _ in range(n_layers)]
        if quantize:
            nn.quantize(self, group_size=64 if real_geometry else 32, bits=4)


def _decode_inputs(seed=3, tokens=1):
    mx.random.seed(seed)
    x = mx.random.normal((1, tokens, HID)).astype(mx.float16)
    inds = mx.random.randint(0, E, (1, tokens, K))
    mx.eval(x, inds)
    return x, inds


class TestExtendedVLMContract:
    @pytest.fixture(autouse=True)
    def restore_class_patch(self, monkeypatch):
        from mlx_vlm.models.switch_layers import SwitchGLU as VLMSwitchGLU

        # Start every test from the pristine stock method with no patch
        # markers (an earlier test in the process may have patched the
        # class); monkeypatch restores the prior state afterwards.
        pristine = VLMSwitchGLU.__dict__.get(
            "_rapid_gate_up_original_call", VLMSwitchGLU.__call__
        )
        monkeypatch.setattr(VLMSwitchGLU, "__call__", pristine)
        monkeypatch.setattr(
            VLMSwitchGLU, "_rapid_gate_up_fused_call", False, raising=False
        )
        monkeypatch.setattr(
            VLMSwitchGLU, "_rapid_gate_up_original_call", None, raising=False
        )

    def test_qwen_mllm_wrapper_fuses_extended_family(self, monkeypatch, tmp_path):
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

        monkeypatch.setattr(mlx_vlm.utils, "get_model_path", lambda *a, **kw: tmp_path)
        mllm.MLXMultimodalLM("unit-test/qwen35-moe").load()
        layer = model.layers[0]
        assert hasattr(layer, "gate_up_proj")
        assert not hasattr(layer, "gate_proj")
        assert not hasattr(layer, "up_proj")

    @pytest.mark.parametrize("quantize", [True, False])
    @pytest.mark.parametrize(
        "batch,tokens,real_geometry",
        [
            (1, 1, False),
            (1, 2, False),
            (1, 5, False),
            (1, 8, False),
            (1, 40, False),
            (3, 1, False),
            (1, 1, True),
        ],
    )
    @pytest.mark.parametrize("combine", [False, True])
    def test_byte_identity(self, quantize, batch, tokens, real_geometry, combine):
        model = TinyVLMMoE(n_layers=1, quantize=quantize, real_geometry=real_geometry)
        model.eval()
        x, inds = _decode_inputs(tokens=tokens)
        if real_geometry:
            model.set_dtype(mx.bfloat16)
            x = mx.random.normal((1, tokens, 512)).astype(mx.bfloat16)
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
        assert VLMSwitchGLU.__call__ is not stock
        assert VLMSwitchGLU._rapid_gate_up_original_call is stock
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
        "missing",
        [
            "combine",
            "block_size",
            "tokenwise",
            "linear",
            "contract",
            "_affine_switch_gate_up_kernel",
            "exact_affine_moe_down",
            "exact_quantized_moe_hc_expand",
            "exact_quantized_selected_linear",
            "supports_quantization",
            "fast_import",
            "verifier_import",
        ],
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
        elif missing == "contract":
            monkeypatch.setattr(
                VLMSwitchGLU, "__call__", lambda self, x, indices, future: x
            )
        else:
            import mlx_vlm.models.fast_ops as fast_ops
            import mlx_vlm.models.quantized_verifier as verifier

            if missing.endswith("_import"):
                name = (
                    "mlx_vlm.models.fast_ops"
                    if missing == "fast_import"
                    else "mlx_vlm.models.quantized_verifier"
                )
                monkeypatch.delitem(sys.modules, name)
                original_import = moe_fusion.importlib.import_module

                def rejecting_import(module):
                    if module == name:
                        raise ImportError("helper module unavailable")
                    return original_import(module)

                monkeypatch.setattr(
                    moe_fusion.importlib, "import_module", rejecting_import
                )
            else:
                module = fast_ops if hasattr(fast_ops, missing) else verifier
                monkeypatch.setattr(module, missing, None)
        assert moe_fusion.fuse_gate_up(model, extended_call_contract=True) == 0
        assert hasattr(model.layers[0], "gate_proj")

    @pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
    @pytest.mark.parametrize("tokens", [2, 3, 5, 8])
    @pytest.mark.parametrize("batch", [1, 2])
    @pytest.mark.parametrize("epilogue", ["selected", "combined", "residual"])
    def test_real_short_block_byte_identity(self, monkeypatch, tokens, batch, epilogue):
        import mlx_vlm.models.fast_ops as fast_ops
        import mlx_vlm.models.quantized_verifier as verifier

        mx.random.seed(4838 + tokens)
        model = TinyVLMMoE(n_layers=1, real_geometry=True, experts=4)
        model.set_dtype(mx.bfloat16)
        model.eval()
        layer = model.layers[0]
        x = mx.random.normal((batch, tokens, 512)).astype(mx.bfloat16)
        # Adjacent experts are distinct; cover both halves including expert 3.
        first = mx.arange(batch * tokens).reshape(batch, tokens) % 4
        indices = mx.stack([first, (first + 1) % 4], axis=-1).astype(mx.uint32)
        weights = shared = residual = None
        if epilogue != "selected":
            weights = mx.softmax(mx.random.normal(indices.shape), axis=-1)
            shared = mx.random.normal(x.shape).astype(x.dtype)
        if epilogue == "residual":
            residual = (
                mx.random.normal((batch, tokens, 4, 512)).astype(x.dtype),
                mx.random.normal((batch, tokens, 4)),
                mx.random.normal((batch, tokens, 4, 4)),
            )
        stock_gate_up = fast_ops.exact_affine_switch_gate_up
        stock_results = []

        def stock_spy(*args):
            result = stock_gate_up(*args)
            stock_results.append(result)
            return result

        monkeypatch.setattr(fast_ops, "exact_affine_switch_gate_up", stock_spy)
        kernel_spy = Mock(wraps=fast_ops._affine_switch_gate_up_kernel)
        monkeypatch.setattr(fast_ops, "_affine_switch_gate_up_kernel", kernel_spy)
        helper_name = {
            "selected": "exact_quantized_selected_linear",
            "combined": "exact_affine_moe_down",
            "residual": "exact_quantized_moe_hc_expand",
        }[epilogue]
        module = fast_ops if epilogue == "combined" else verifier
        helper_spy = Mock(wraps=getattr(module, helper_name))
        monkeypatch.setattr(module, helper_name, helper_spy)
        stock = layer(x, indices, weights, shared, residual)
        mx.eval(stock)
        assert len(stock_results) == 1 and stock_results[0] is not None
        assert kernel_spy.call_count == (1 if batch == 1 else 0)
        assert helper_spy.call_count == 1
        kernel_spy.reset_mock()
        helper_spy.reset_mock()
        assert moe_fusion.fuse_gate_up(model, extended_call_contract=True) == 1
        gather_spy = Mock(wraps=mx.gather_qmm)
        monkeypatch.setattr(mx, "gather_qmm", gather_spy)
        fused = layer(x, indices, weights, shared, residual)
        mx.eval(fused)
        fused_gathers = [
            call
            for call in gather_spy.call_args_list
            if call.args[1] is layer.gate_up_proj.weight
        ]
        assert len(fused_gathers) == (1 if batch > 1 else 0)
        if fused_gathers:
            assert fused_gathers[0].kwargs["sorted_indices"] is False
        assert kernel_spy.call_count == (1 if batch == 1 else 0)
        assert helper_spy.call_count == 1
        assert mx.array_equal(stock, fused)

    @pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
    def test_short_block_has_no_weight_copy_or_duplicate_residency(self):
        from mlx.utils import tree_flatten

        model = TinyVLMMoE(n_layers=1, real_geometry=True, experts=64)
        model.set_dtype(mx.bfloat16)
        model.eval()
        layer = model.layers[0]
        mx.eval(model.parameters())
        original_bytes = sum(
            array.nbytes
            for proj in (layer.gate_proj, layer.up_proj)
            for _, array in tree_flatten(proj.parameters())
        )
        half_weight_bytes = layer.gate_proj.weight.nbytes
        assert moe_fusion.fuse_gate_up(model, extended_call_contract=True) == 1
        assert not hasattr(layer, "gate_proj") and not hasattr(layer, "up_proj")
        assert (
            sum(
                array.nbytes
                for _, array in tree_flatten(layer.gate_up_proj.parameters())
            )
            == original_bytes
        )
        x = mx.random.normal((1, 8, 512)).astype(mx.bfloat16)
        indices = mx.arange(16).reshape(1, 8, 2).astype(mx.uint32)
        mx.eval(x, indices, model.parameters())
        mx.synchronize()
        mx.clear_cache()
        baseline = mx.get_active_memory()
        mx.reset_peak_memory()
        output = layer(x, indices)
        mx.eval(output)
        increase = mx.get_peak_memory() - baseline
        assert increase < half_weight_bytes // 4, (increase, half_weight_bytes)

    @pytest.mark.parametrize(
        "fallback", ["generic", "plain", "many_routes", "down_plain"]
    )
    def test_short_block_fallbacks(self, monkeypatch, fallback):
        import mlx_vlm.models.linear as linear

        model = TinyVLMMoE(n_layers=1, quantize=fallback != "plain")
        model.eval()
        layer = model.layers[0]
        if fallback == "down_plain":
            from mlx_vlm.models.switch_layers import SwitchLinear

            layer.down_proj = SwitchLinear(INTER, HID, E, bias=False)
        batch = 32 if fallback == "many_routes" else 1
        x, indices = _decode_inputs(tokens=3)
        x = mx.repeat(x, batch, axis=0)
        indices = mx.repeat(indices, batch, axis=0)
        expected = layer(x, indices)
        mx.eval(expected)
        tokenwise_spy = Mock(wraps=linear.tokenwise)
        monkeypatch.setattr(linear, "tokenwise", tokenwise_spy)
        assert moe_fusion.fuse_gate_up(model, extended_call_contract=True) == 1
        gather_spy = Mock(wraps=mx.gather_qmm)
        monkeypatch.setattr(mx, "gather_qmm", gather_spy)
        actual = layer(x, indices)
        mx.eval(actual)
        assert mx.array_equal(actual, expected)
        if fallback == "generic":
            fused_gathers = [
                call
                for call in gather_spy.call_args_list
                if call.args[1] is layer.gate_up_proj.weight
            ]
            assert len(fused_gathers) == 1
            assert fused_gathers[0].kwargs["sorted_indices"] is False
        assert tokenwise_spy.call_count == (0 if fallback == "generic" else 1)

    def test_helper_modules_resolved_when_not_resident(self, monkeypatch):
        model = TinyVLMMoE(n_layers=1)
        monkeypatch.delitem(sys.modules, "mlx_vlm.models.fast_ops")
        monkeypatch.delitem(sys.modules, "mlx_vlm.models.quantized_verifier")
        assert moe_fusion.fuse_gate_up(model, extended_call_contract=True) == 1
