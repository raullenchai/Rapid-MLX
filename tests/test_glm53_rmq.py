from __future__ import annotations

import pytest

import vllm_mlx.quantization.glm53_rmq as rmq
from vllm_mlx.quantization.glm53_rmq import (
    QuantSpec,
    TensorDescriptor,
    fusion_domain,
    module_path,
    module_paths,
    mtp_module_paths,
    plan_tensors,
    projected_storage_bytes,
    summarize_plan,
)

CONFIG = {"model_type": "glm5_next", "text_config": {"num_hidden_layers": 45}}


def td(name: str, shape=(64, 64), dtype="BF16") -> TensorDescriptor:
    return TensorDescriptor(name, tuple(shape), dtype)


def test_glm53_rmq_static_precision_floors():
    tensors = [
        td("lm_head.weight", (1024, 64)),
        td("model.language_model.embed_tokens.weight", (1024, 64)),
        td("model.language_model.layers.3.self_attn.indexer.wk.weight"),
        td("model.language_model.layers.4.self_attn.q_proj.weight"),
        td("model.language_model.layers.4.self_attn.f_b_proj.weight"),
        td("model.language_model.layers.4.mlp.experts.0.down_proj.weight"),
        td("model.language_model.layers.4.mlp.shared_experts.down_proj.weight"),
        td("model.language_model.layers.4.mlp.gate.weight", (288, 4096)),
        td("model.language_model.layers.45.mlp.experts.0.down_proj.weight"),
        td("model.language_model.layers.4.self_attn.dt_bias", (64,), "F32"),
    ]
    bits = [item.spec.bits for item in plan_tensors(tensors, CONFIG)]
    assert bits == [8, 8, 8, 8, 8, 4, 8, None, 4, None]


def test_mtp_precision_matches_the_corresponding_target_domains():
    tensors = [
        td("model.language_model.layers.44.mlp.experts.0.down_proj.weight"),
        td("model.language_model.layers.45.mlp.experts.0.down_proj.weight"),
        td("model.language_model.layers.44.mlp.shared_experts.down_proj.weight"),
        td("model.language_model.layers.45.mlp.shared_experts.down_proj.weight"),
        td("model.language_model.layers.45.self_attn.o_proj.weight"),
    ]
    assert [item.spec.bits for item in plan_tensors(tensors, CONFIG)] == [
        4,
        4,
        8,
        8,
        8,
    ]


def test_sensitivity_raise_is_locked_across_kda_fusion_domain():
    tensors = [
        td(f"model.language_model.layers.4.self_attn.{name}.weight")
        for name in ("q_proj", "k_proj", "v_proj", "f_a_proj", "g_a_proj", "b_proj")
    ]
    domain = "layer:4:kda-fused-input"
    plan = plan_tensors(tensors, CONFIG, {tensors[0].name: 0.9})
    assert {item.spec.bits for item in plan} == {8}
    assert {item.fusion_domain for item in plan} == {domain}


def test_routed_expert_projection_cannot_mix_bits():
    tensors = [
        td(f"model.language_model.layers.10.mlp.experts.{i}.down_proj.weight")
        for i in range(4)
    ]
    plan = plan_tensors(tensors, CONFIG, {tensors[2].name: 0.6})
    assert {item.spec.bits for item in plan} == {8}
    assert {item.fusion_domain for item in plan} == {"layer:10:routed:down_proj"}


def test_invalid_sensitivity_fails_closed():
    tensor = td("model.language_model.layers.1.self_attn.q_proj.weight")
    with pytest.raises(ValueError, match="within"):
        plan_tensors([tensor], CONFIG, {tensor.name: 1.1})


def test_invalid_quant_specs_fail_closed():
    with pytest.raises(ValueError, match="floating-point"):
        QuantSpec(bits=None).as_config()
    assert QuantSpec(bits=8, group_size=64, mode="affine").as_config() == {
        "bits": 8,
        "group_size": 64,
        "mode": "affine",
    }
    with pytest.raises(ValueError, match="Q7"):
        rmq._q(7, "invalid")


def test_non_native_width_and_vision_stay_float():
    tensors = [
        td("model.language_model.layers.1.mlp.down_proj.weight", (64, 96)),
        td("model.visual.blocks.0.attn.qkv.weight", (192, 64)),
    ]
    plan = plan_tensors(tensors, CONFIG)
    assert [item.spec.bits for item in plan] == [None, None]


def test_remaining_policy_fallbacks_and_float_storage():
    tensors = [
        td("model.language_model.layers.1.buffer.weight", dtype="U8"),
        td("model.language_model.layers.1.mlp.down_proj.weight"),
        td("model.language_model.layers.1.other.weight"),
        td("model.language_model.layers.1.float_state", (64,), "F32"),
    ]
    plan = plan_tensors(tensors, CONFIG)
    assert [item.spec.bits for item in plan] == [None, 8, 4, None]
    assert projected_storage_bytes([plan[-1]]) == 64 * 4


def test_module_path_matches_mlx_vlm_tree_and_summary_is_stable():
    name = "model.language_model.layers.2.self_attn.q_proj.weight"
    assert module_path(name) == "language_model.model.layers.2.self_attn.q_proj"
    plan = plan_tensors([td(name), td("lm_head.weight")], CONFIG)
    summary = summarize_plan(plan)
    assert summary["fusion_domains"] == 1
    assert set(summary["formats"]) == {"q8-g64"}
    assert summary["projected_storage_bytes"] > 0


def test_module_path_tracks_glm_forget_gate_sanitizer():
    name = "model.language_model.layers.2.self_attn.f_a_proj.weight"
    assert (
        module_path(name)
        == "language_model.model.layers.2.self_attn.forget_gate.f_a_proj"
    )


def test_module_paths_track_split_attention_and_fused_experts():
    kv_b = "model.language_model.layers.3.self_attn.kv_b_proj.weight"
    assert module_paths(kv_b) == (
        "language_model.model.layers.3.self_attn.embed_q",
        "language_model.model.layers.3.self_attn.unembed_out",
    )
    expert = "model.language_model.layers.3.mlp.experts.7.down_proj.weight"
    assert module_paths(expert) == (
        "language_model.model.layers.3.mlp.switch_mlp.down_proj",
    )


def test_fusion_domain_is_explicit_and_narrow():
    assert (
        fusion_domain("model.language_model.layers.7.self_attn.f_a_proj.weight")
        == "layer:7:kda-fused-input"
    )
    assert (
        fusion_domain("model.language_model.layers.7.self_attn.f_b_proj.weight") is None
    )


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("eh_proj", ("eh_proj",)),
        ("shared_head.norm", ("shared_head_norm",)),
        (
            "mlp.experts.7.down_proj",
            ("mtp_block.mlp.switch_mlp.down_proj",),
        ),
        (
            "mlp.shared_experts.gate_proj",
            ("mtp_block.mlp.shared_experts.gate_up_proj",),
        ),
        (
            "self_attn.kv_a_proj_with_mqa",
            ("mtp_block.self_attn.qkv_a_proj",),
        ),
        (
            "self_attn.kv_b_proj",
            ("mtp_block.self_attn.embed_q", "mtp_block.self_attn.unembed_out"),
        ),
        (
            "self_attn.indexer.wk",
            ("mtp_block.self_attn.indexer.wk",),
        ),
    ],
)
def test_mtp_module_paths_match_post_split_sanitizer(source, expected):
    name = f"model.language_model.layers.45.{source}.weight"
    assert mtp_module_paths(name, CONFIG) == expected
    assert mtp_module_paths(name.replace("layers.45", "layers.44"), CONFIG) == ()
