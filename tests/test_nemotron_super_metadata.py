"""Super text aliases retain parser, precision-lane and footprint contracts."""

import pytest


@pytest.mark.parametrize(
    "alias,bits,size",
    [
        ("nemotron-3.5-super-vl", 4, 69_540_654_086),
        ("nemotron-3.5-super-vl-3bit", 3, 42_764_296_516),
    ],
)
def test_super_catalog_contract(alias, bits, size):
    from rapid_mlx.model_aliases import list_profiles
    from rapid_mlx.model_sizes import size_bytes

    profile = list_profiles()[alias]
    assert profile.hf_path == (
        f"rapid-mlx/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-MLX-{bits}bit"
    )
    assert profile.tool_call_parser == "nemotron"
    assert profile.reasoning_parser == "qwen3"
    assert profile.is_hybrid and profile.is_hybrid_explicit and profile.is_moe
    assert profile.is_text_only
    assert not profile.supports_spec_decode
    # Same suffix filter as scripts/gen_model_sizes.py: includes tokenizer and
    # config JSON; excludes README, license and chat-template Jinja sidecar.
    assert size_bytes(profile.hf_path) == size
