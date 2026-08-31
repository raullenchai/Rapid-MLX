# SPDX-License-Identifier: Apache-2.0
"""Weight-free contracts for the GLM-5 Next text-runtime overlay."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_mlx.patches.glm5_next_runtime import (
    _keep_glm5_next_fp32,
    _projection_quantization_is_homogeneous,
)


class _Linear:
    pass


class _Quantized:
    def __init__(self, *, group_size=64, bits=4, mode="affine"):
        self.scales = object()
        self.group_size = group_size
        self.bits = bits
        self.mode = mode


def test_kda_projection_fusion_requires_one_quantization() -> None:
    assert _projection_quantization_is_homogeneous([_Linear(), _Linear()])
    assert _projection_quantization_is_homogeneous([_Quantized(), _Quantized()])
    assert not _projection_quantization_is_homogeneous([_Linear(), _Quantized()])
    assert not _projection_quantization_is_homogeneous(
        [_Quantized(bits=4), _Quantized(bits=8)]
    )
    assert not _projection_quantization_is_homogeneous(
        [_Quantized(mode="affine"), _Quantized(mode="mxfp4")]
    )


def test_fp32_state_allowlist_is_narrow() -> None:
    assert _keep_glm5_next_fp32("model.layers.0.attn_hc.base")
    assert _keep_glm5_next_fp32("model.layers.0.ffn_hc.scale")
    assert _keep_glm5_next_fp32("model.layers.0.self_attn.A_log")
    assert _keep_glm5_next_fp32("model.layers.0.self_attn.dt_bias")
    assert not _keep_glm5_next_fp32("model.layers.0.self_attn.q_proj.weight")


@pytest.mark.requires_mlx
def test_installer_applies_glm_math_without_changing_shared_models() -> None:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_vlm.models.deepseek_v32 import language as deepseek_language
    from mlx_vlm.models.glm5_next import language

    from vllm_mlx.patches import glm5_next_runtime as patch

    def config(*, attention="linear_attention", mlp="sparse"):
        return language.TextConfig(
            model_type="glm5_next_text",
            vocab_size=32,
            hidden_size=8,
            intermediate_size=16,
            moe_intermediate_size=4,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            n_shared_experts=1 if mlp == "sparse" else None,
            n_routed_experts=4 if mlp == "sparse" else None,
            routed_scaling_factor=1.0,
            kv_lora_rank=4,
            q_lora_rank=4,
            qk_rope_head_dim=0,
            v_head_dim=4,
            qk_nope_head_dim=4,
            num_experts_per_tok=2,
            first_k_dense_replace=0 if mlp == "sparse" else 99,
            max_position_embeddings=32,
            rms_norm_eps=1e-5,
            index_topk=2,
            index_head_dim=4,
            index_n_heads=1,
            layer_types=[attention],
            mlp_layer_types=[mlp],
            linear_attn_config={
                "num_heads": 1,
                "head_dim": 8,
                "short_conv_kernel_size": 2,
                "gate_lower_bound": -5.0,
            },
            index_kpool=2,
            hc_mult=1,
            hc_sinkhorn_iters=1,
            n_group=1,
            topk_group=1,
            topk_method="noaux_tc",
        )

    patched_names = (
        "DeepseekMLP",
        "DeepseekV32MoE",
        "Glm5NextSparseAttention",
        "Glm5NextLinearAttention",
    )
    originals = {name: getattr(language, name) for name in patched_names}
    original_sanitize = language.LanguageModel.sanitize
    original_cast_predicate = language.LanguageModel.cast_predicate
    marker = getattr(language, "_RAPID_MLX_RUNTIME_FIX_INSTALLED", None)
    marker_existed = hasattr(language, "_RAPID_MLX_RUNTIME_FIX_INSTALLED")
    patch._INSTALLED = False
    if marker_existed:
        del language._RAPID_MLX_RUNTIME_FIX_INSTALLED

    try:
        shared_mlp = deepseek_language.DeepseekV32MoE
        assert patch.install_glm5_next_runtime_fix() is True
        assert patch.install_glm5_next_runtime_fix() is False
        assert patch.is_installed() is True
        assert deepseek_language.DeepseekV32MoE is shared_mlp

        sparse_layer = language.Glm5NextDecoderLayer(config(), 0)
        assert type(sparse_layer.mlp).__name__ == "Glm5NextMoE"
        activation = sparse_layer.mlp.switch_mlp.activation
        x_up = mx.array([20.0, -20.0])
        x_gate = mx.array([20.0, 20.0])
        expected = nn.silu(mx.array([10.0, 10.0])) * mx.array([10.0, -10.0])
        assert mx.allclose(activation(x_up, x_gate), expected).item()

        _, scores = sparse_layer.mlp.gate(mx.ones((1, 8), dtype=mx.bfloat16))
        assert scores.dtype == mx.float32

        dense_layer = language.Glm5NextDecoderLayer(config(mlp="dense"), 0)
        assert type(dense_layer.mlp).__name__ == "Glm5NextMLP"
        assert dense_layer.mlp.limit == 10.0
        assert dense_layer.mlp(mx.ones((1, 1, 8))).shape == (1, 1, 8)

        attention_layer = language.Glm5NextDecoderLayer(
            config(attention="deepseek_sparse_attention", mlp="dense"), 0
        )
        assert attention_layer.self_attn.q_a_layernorm.eps == 1e-5
        assert attention_layer.self_attn.kv_a_layernorm.eps == 1e-5
        assert attention_layer.self_attn.indexer.k_norm.eps == 1e-6

        linear_attention = dense_layer.self_attn
        fused = linear_attention._fused_in_proj(mx.ones((1, 1, 8)))
        assert len(fused) == 6

        class Projection:
            def __init__(self, result, *, quantized=False):
                self.result = result
                if quantized:
                    self.scales = object()

            def __call__(self, _inputs):
                return self.result

        mixed = SimpleNamespace(
            q_proj=Projection("q"),
            k_proj=Projection("k", quantized=True),
            v_proj=Projection("v"),
            forget_gate=SimpleNamespace(f_a_proj=Projection("forget")),
            g_a_proj=Projection("gate"),
            b_proj=Projection("beta"),
        )
        assert language.Glm5NextLinearAttention._fused_in_proj(mixed, object()) == (
            "q",
            "k",
            "v",
            "forget",
            "gate",
            "beta",
        )

        model = language.LanguageModel(config(mlp="dense"))
        weights = {
            "model.layers.0.hc_attn_base": mx.ones((1,), dtype=mx.bfloat16),
            "model.layers.0.self_attn.A_log": mx.ones((1,), dtype=mx.bfloat16),
            "model.layers.0.self_attn.q_proj.weight": mx.ones((1,), dtype=mx.bfloat16),
        }
        sanitized = model.sanitize(weights)
        assert sanitized["model.layers.0.attn_hc.base"].dtype == mx.float32
        assert (
            sanitized["model.layers.0.self_attn.forget_gate.A_log"].dtype == mx.float32
        )
        assert sanitized["model.layers.0.self_attn.q_proj.weight"].dtype == mx.bfloat16
        predicate = model.cast_predicate
        assert predicate("model.layers.0.e_score_correction_bias") is False
        assert predicate("model.layers.0.attn_hc.base") is False
        assert predicate("model.layers.0.self_attn.q_proj.weight") is True
    finally:
        for name, value in originals.items():
            setattr(language, name, value)
        language.LanguageModel.sanitize = original_sanitize
        language.LanguageModel.cast_predicate = original_cast_predicate
        if marker_existed:
            language._RAPID_MLX_RUNTIME_FIX_INSTALLED = marker
        elif hasattr(language, "_RAPID_MLX_RUNTIME_FIX_INSTALLED"):
            del language._RAPID_MLX_RUNTIME_FIX_INSTALLED
        patch._INSTALLED = False


@pytest.mark.requires_mlx
def test_installer_respects_runtime_that_is_already_patched() -> None:
    from mlx_vlm.models.glm5_next import language

    from vllm_mlx.patches import glm5_next_runtime as patch

    marker = getattr(language, "_RAPID_MLX_RUNTIME_FIX_INSTALLED", None)
    marker_existed = hasattr(language, "_RAPID_MLX_RUNTIME_FIX_INSTALLED")
    patch._INSTALLED = False
    language._RAPID_MLX_RUNTIME_FIX_INSTALLED = True
    try:
        assert patch.install_glm5_next_runtime_fix() is False
        assert patch.is_installed() is True
    finally:
        if marker_existed:
            language._RAPID_MLX_RUNTIME_FIX_INSTALLED = marker
        else:
            del language._RAPID_MLX_RUNTIME_FIX_INSTALLED
        patch._INSTALLED = False
