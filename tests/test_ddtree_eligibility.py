# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``vllm_mlx/speculative/ddtree/eligibility.py``."""

from __future__ import annotations

import pytest

from vllm_mlx.model_aliases import AliasProfile
from vllm_mlx.speculative.ddtree.eligibility import (
    DDTreeUnavailable,
    check,
    report,
)


def _good_profile() -> AliasProfile:
    return AliasProfile(
        hf_path="mlx-community/Qwen3.5-9B-8bit",
        is_moe=False,
        supports_ddtree=True,
        ddtree_draft_model="z-lab/Qwen3.5-9B-DFlash",
        ddtree_speculative_tokens=16,
        ddtree_tree_budget=24,
    )


def test_check_passes_for_good_profile() -> None:
    p = _good_profile()
    check(p, alias="qwen3.5-9b-8bit")
    assert report(p, alias="qwen3.5-9b-8bit").reasons == ()


def test_check_rejects_alias_without_supports_ddtree() -> None:
    p = AliasProfile(hf_path="mlx-community/Qwen3.5-9B-8bit")
    with pytest.raises(DDTreeUnavailable, match="not DDTree-enabled"):
        check(p, alias="qwen3.5-9b-8bit")


def test_check_rejects_moe_alias() -> None:
    p = AliasProfile(
        hf_path="mlx-community/Qwen3.6-35B-A3B-8bit",
        is_moe=True,
        supports_ddtree=True,
        ddtree_draft_model="z-lab/Qwen3.6-35B-A3B-DFlash",
        ddtree_speculative_tokens=16,
        ddtree_tree_budget=24,
    )
    with pytest.raises(DDTreeUnavailable, match="MoE"):
        check(p, alias="qwen3.6-35b-8bit")


def test_check_rejects_4bit_main_model() -> None:
    p = AliasProfile(
        hf_path="mlx-community/Qwen3.5-9B-4bit",
        supports_ddtree=True,
        ddtree_draft_model="z-lab/Qwen3.5-9B-DFlash",
        ddtree_speculative_tokens=16,
        ddtree_tree_budget=24,
    )
    with pytest.raises(DDTreeUnavailable, match="4-bit"):
        check(p, alias="qwen3.5-9b-4bit")


def test_report_collects_all_failures() -> None:
    bad = AliasProfile(
        hf_path="mlx-community/Qwen3.6-35B-A3B-4bit",
        is_moe=True,
        supports_ddtree=True,
    )
    r = report(bad, alias="qwen3.6-35b-4bit")
    joined = " ".join(r.reasons)
    assert "MoE" in joined
    assert "4-bit" in joined
    assert "ddtree_draft_model" in joined
    assert "ddtree_speculative_tokens" in joined
    assert "ddtree_tree_budget" in joined


def test_qwen3_5_9b_8bit_alias_passes_check() -> None:
    from vllm_mlx.model_aliases import resolve_profile

    profile = resolve_profile("qwen3.5-9b-8bit")
    assert profile is not None, "qwen3.5-9b-8bit alias missing"
    check(profile, alias="qwen3.5-9b-8bit")


def test_qwen3_5_9b_4bit_alias_fails_with_4bit_reason() -> None:
    from vllm_mlx.model_aliases import resolve_profile

    profile = resolve_profile("qwen3.5-9b-4bit")
    assert profile is not None
    with pytest.raises(DDTreeUnavailable) as excinfo:
        check(profile, alias="qwen3.5-9b-4bit")
    assert "4-bit" in str(excinfo.value)
