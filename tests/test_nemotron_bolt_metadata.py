"""Converted Bolt paths retain parser and recurrent-cache safety contracts."""

import json

import pytest

from rapid_mlx.model_auto_config import (
    config_declares_linear_attention,
    detect_model_config,
)


@pytest.mark.parametrize(
    "template",
    [
        None,
        "{% if tools %}<tool_call><function=example><parameter=value>x</parameter></function></tool_call>{% endif %}",
    ],
)
def test_bolt_local_metadata_selects_native_parsers(tmp_path, template):
    model = tmp_path / "renamed-text-model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"model_type": "nemotron_h_puzzle"}))
    if template:
        (model / "chat_template.jinja").write_text(template)
    profile = detect_model_config(str(model))
    assert profile.tool_call_parser == "nemotron"
    assert profile.reasoning_parser == "qwen3"
    assert profile.is_hybrid is True
    assert profile.is_hybrid_explicit is True
    assert profile.is_moe is True
    assert profile.supports_spec_decode is False


@pytest.mark.parametrize("nested", [False, True])
def test_bolt_declares_recurrence_before_loading_weights(nested):
    config = {"model_type": "nemotron_h_puzzle"}
    if nested:
        config = {"model_type": "wrapper", "text_config": config}
    assert config_declares_linear_attention(config) is True


def test_unknown_puzzle_name_does_not_claim_bolt_contract(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "other_puzzle"}))
    assert detect_model_config(str(tmp_path)) is None
    assert config_declares_linear_attention({"model_type": "other_puzzle"}) is False


@pytest.mark.parametrize(
    "alias,bits,size",
    [
        # scripts/gen_model_sizes.py payload totals, excluding README/.jinja.
        ("nemotron-3.5-bolt", 4, 43_274_333_036),
        ("nemotron-3.5-bolt-3bit", 3, 31_670_608_078),
    ],
)
def test_bolt_catalog_contract(alias, bits, size):
    from rapid_mlx.model_aliases import list_profiles
    from rapid_mlx.model_sizes import size_bytes

    profile = list_profiles()[alias]
    assert profile.hf_path == (
        f"rapid-mlx/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-MLX-{bits}bit"
    )
    assert profile.tool_call_parser == "nemotron"
    assert profile.reasoning_parser == "qwen3"
    assert profile.is_hybrid and profile.is_hybrid_explicit and profile.is_moe
    assert profile.is_text_only
    assert not profile.supports_spec_decode
    assert size_bytes(profile.hf_path) == size
