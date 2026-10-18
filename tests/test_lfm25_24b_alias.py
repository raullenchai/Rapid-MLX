# SPDX-License-Identifier: Apache-2.0
"""Offline contract for the official LFM2.5-24B-A2B 4-bit alias."""

from rapid_mlx import model_sizes
from rapid_mlx.model_aliases import resolve_profile


def test_lfm25_24b_alias_profile_and_size() -> None:
    profile = resolve_profile("lfm2.5-24b-a2b-4bit")

    assert profile is not None
    assert profile.hf_path == "LiquidAI/LFM2.5-24B-A2B-MLX-4bit"
    assert profile.modality == "text"
    assert profile.tool_call_parser == "lfm"
    assert profile.reasoning_parser == "qwen3"
    assert profile.is_hybrid is False
    assert profile.is_moe is True
    assert profile.supports_spec_decode is False
    assert profile.subfolder is None
    size = model_sizes.size_bytes(profile.hf_path)
    assert size == 13_585_282_965
