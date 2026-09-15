# SPDX-License-Identifier: Apache-2.0
"""Qwen3-Coder-Next alias and fallback-routing contracts for #925."""

from __future__ import annotations

import pytest

from rapid_mlx.model_aliases import list_profiles
from rapid_mlx.model_auto_config import detect_model_config

_CANONICAL_ALIASES = (
    (
        "qwen3-coder-next-80b-4bit",
        "mlx-community/Qwen3-Coder-Next-4bit",
    ),
    (
        "qwen3-coder-next-80b-8bit",
        "lmstudio-community/Qwen3-Coder-Next-MLX-8bit",
    ),
)


@pytest.mark.parametrize(("alias", "hf_path"), _CANONICAL_ALIASES)
def test_qwen3_coder_next_aliases_declare_the_safe_runtime_profile(
    alias: str, hf_path: str
) -> None:
    """Both shipped precisions are hybrid MoE, XML-tool, non-thinking models.

    The upstream Qwen template requests ``<tool_call><function=...>`` XML,
    while the model card states that Coder-Next never emits ``<think>``.
    Keeping this contract in the alias profile prevents the generic Hermes
    parser from leaking XML tool calls and prevents the qwen3 reasoning parser
    from duplicating ordinary content as reasoning.
    """
    profile = list_profiles()[alias]

    assert profile.hf_path == hf_path
    assert profile.tool_call_parser == "qwen3_coder_xml"
    assert profile.reasoning_parser is None
    assert profile.is_hybrid is True
    assert profile.is_moe is True
    assert profile.supports_spec_decode is False
    assert profile.supports_dflash is False
    assert dict(profile.recommended_sampling or ()) == {
        "temperature": 1.0,
        "top_k": 40.0,
        "top_p": 0.95,
    }


def test_legacy_qwen3_coder_alias_keeps_coder_next_safe_profile() -> None:
    """The existing short alias is a Coder-Next repack, not older Coder-30B.

    Preserve its public name for compatibility while giving it the same parser
    and architecture gates as the explicit 80B aliases.
    """
    profile = list_profiles()["qwen3-coder-4bit"]

    assert profile.tool_call_parser == "qwen3_coder_xml"
    assert profile.reasoning_parser is None
    assert profile.is_hybrid is True
    assert profile.is_moe is True
    assert profile.supports_spec_decode is False
    assert profile.supports_dflash is False


@pytest.mark.parametrize(
    "model_path",
    [
        "Qwen/Qwen3-Coder-Next",
        *(hf_path for _, hf_path in _CANONICAL_ALIASES),
    ],
)
def test_qwen3_coder_next_direct_paths_use_xml_tools_without_reasoning(
    model_path: str,
) -> None:
    """Unaliased official/repacked paths stay on the same safe fallback.

    This is the boundary case: a user can serve the official HF ID directly
    instead of a Rapid-MLX alias and must still receive XML tool parsing with
    hybrid speculative-decode gating.
    """
    config = detect_model_config(model_path)

    assert config is not None
    assert config.tool_call_parser == "qwen3_coder_xml"
    assert config.reasoning_parser is None
    assert config.is_hybrid is True
    assert config.supports_spec_decode is False
    assert config.supports_dflash is False
