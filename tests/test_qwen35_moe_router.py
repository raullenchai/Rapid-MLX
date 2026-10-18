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


def test_mllm_load_enrolls_qwen_moe_optimizations(monkeypatch):
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

    mllm.MLXMultimodalLM("local/qwen-moe").load()

    assert fused == [(model, True)]
    assert routed == [model]


# -- Extended mlx-vlm SwitchGLU contract fusion (#4823) ----------------------
# Lives here because this file runs on the Apple Silicon CI lane.

from unittest.mock import Mock  # noqa: E402

from rapid_mlx import moe_fusion  # noqa: E402

E, HID, INTER, K = 8, 64, 128, 2


class TinyVLMMoE(nn.Module):
    """mlx-vlm owns a separate SwitchGLU class family."""

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
