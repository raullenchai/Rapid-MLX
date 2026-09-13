# SPDX-License-Identifier: Apache-2.0
"""Correctness and enrollment tests for the fused Qwen MoE router."""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest

import vllm_mlx.qwen35_moe_router as router


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


def test_mllm_load_enrolls_qwen_moe_optimizations(monkeypatch):
    import mlx_vlm
    import mlx_vlm.utils

    from vllm_mlx import moe_fusion
    from vllm_mlx.models import mllm
    from vllm_mlx.utils import tokenizer as tokenizer_utils

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
    monkeypatch.setattr(moe_fusion, "fuse_gate_up", fused.append)
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

    assert fused == [model]
    assert routed == [model]
