# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``vllm_mlx/speculative/dflash/eligibility.py``.

These verify each gate fires in isolation. Integration with the CLI
and engine is covered separately in ``test_dflash_integration.py``
(which is skipped when the drafter isn't cached).
"""

from __future__ import annotations

import pytest

from vllm_mlx.model_aliases import AliasProfile
from vllm_mlx.speculative.dflash.eligibility import (
    DFlashUnavailable,
    _looks_like_4bit,
    check,
    report,
)


def _good_profile() -> AliasProfile:
    """Reference 'should pass' profile: dense, 8-bit, has drafter."""
    return AliasProfile(
        hf_path="mlx-community/Qwen3.5-27B-8bit",
        is_hybrid=True,
        is_moe=False,
        supports_dflash=True,
        dflash_draft_model="z-lab/Qwen3.5-27B-DFlash",
    )


# =============================================================================
# _looks_like_4bit — quantization detection from HF path
# =============================================================================


@pytest.mark.parametrize(
    "hf_path,expected",
    [
        ("mlx-community/Qwen3.5-4B-MLX-4bit", True),
        ("mlx-community/Qwen3.5-27B-4bit", True),
        ("unsloth/Qwen3.6-27B-UD-MLX-4bit", True),
        ("nightmedia/Qwen3.5-122B-A10B-Text-mxfp4-mlx", True),
        ("RedHatAI/Qwen3.6-35B-A3B-NVFP4", True),
        ("mlx-community/Qwen3.6-35B-A3B-4bit-DWQ", True),
        # 8-bit + higher should NOT match
        ("mlx-community/Qwen3.5-27B-8bit", False),
        ("mlx-community/Qwen3.6-35B-A3B-8bit", False),
        ("mlx-community/Qwen3.5-27B-bf16", False),
        ("Qwen/Qwen3.5-4B", False),
        ("mlx-community/Qwen3.5-27B-6bit", False),
        ("mlx-community/Qwen3.5-27B-5bit", False),
    ],
)
def test_looks_like_4bit_classification(hf_path: str, expected: bool) -> None:
    assert _looks_like_4bit(hf_path) is expected


# =============================================================================
# Gate-by-gate: check() raises with actionable messages
# =============================================================================


def test_check_passes_for_good_profile() -> None:
    """Reference happy path — no exception, no reasons."""
    p = _good_profile()
    check(p, alias="qwen3.5-27b-8bit")
    r = report(p, alias="qwen3.5-27b-8bit")
    assert r.reasons == ()


def test_check_rejects_alias_without_supports_dflash() -> None:
    """Profile not marked DFlash-eligible — most common case (any
    non-validated alias). Default ``supports_dflash=False`` must trip
    the gate."""
    p = AliasProfile(hf_path="mlx-community/Qwen3.5-27B-8bit")
    with pytest.raises(DFlashUnavailable, match="not DFlash-enabled"):
        check(p, alias="qwen3.5-27b-not-validated")


def test_check_rejects_moe_alias() -> None:
    """MoE → reject even if supports_dflash=True (contradiction caught
    here as a defense-in-depth; alias-contract test rejects this at
    schema-load time too)."""
    p = AliasProfile(
        hf_path="mlx-community/Qwen3.6-35B-A3B-8bit",
        is_moe=True,
        supports_dflash=True,
        dflash_draft_model="z-lab/Qwen3.6-35B-A3B-DFlash",
    )
    with pytest.raises(DFlashUnavailable, match="MoE"):
        check(p, alias="qwen3.6-35b-8bit")


def test_check_rejects_4bit_main_model() -> None:
    """4-bit main model → reject. Note: aliases marked
    ``supports_dflash=True`` can't be 4-bit (alias-contract guard),
    so this gate primarily defends against test-only profiles."""
    p = AliasProfile(
        hf_path="mlx-community/Qwen3.5-27B-4bit",  # 4-bit!
        supports_dflash=True,
        dflash_draft_model="z-lab/Qwen3.5-27B-DFlash",
    )
    with pytest.raises(DFlashUnavailable, match="4-bit"):
        check(p, alias="qwen3.5-27b")


def test_check_message_lists_eligible_aliases() -> None:
    """Error messages must point users at a working alias — saves a
    docs round-trip."""
    p = AliasProfile(hf_path="mlx-community/Qwen3.6-35B-A3B-8bit", is_moe=True)
    try:
        check(p, alias="qwen3.6-35b-8bit")
        raise AssertionError("should have raised")
    except DFlashUnavailable as e:
        msg = str(e)
        assert "qwen3.5-27b-8bit" in msg, (
            f"error message should suggest a working alias; got:\n{msg}"
        )


# =============================================================================
# report() — structured per-gate status (used by `info` command)
# =============================================================================


def test_report_collects_all_failures() -> None:
    """``report()`` must NOT short-circuit on first failure — render
    all failing gates so the user fixes everything in one round."""
    bad = AliasProfile(
        hf_path="mlx-community/Qwen3.6-35B-A3B-4bit",  # 4-bit AND MoE
        is_moe=True,
        # supports_dflash=False (default) → 3 reasons total
    )
    r = report(bad, alias="qwen3.6-35b")
    assert len(r.reasons) == 3, f"expected 3 reasons, got: {r.reasons}"
    joined = " ".join(r.reasons)
    assert "MoE" in joined
    assert "4-bit" in joined
    assert "not DFlash-enabled" in joined


def test_report_no_alias_name_renders_cleanly() -> None:
    """Some callers (programmatic use) don't have an alias name. Header
    fallback must still produce something useful."""
    bad = AliasProfile(hf_path="mlx-community/Qwen3.5-27B-4bit", is_moe=True)
    try:
        check(bad)  # alias=None
        raise AssertionError("should have raised")
    except DFlashUnavailable as e:
        assert "DFlash unavailable" in str(e)


# =============================================================================
# AliasProfile<>aliases.json integration — currently eligible aliases
# =============================================================================


def test_qwen3_5_27b_8bit_alias_passes_check() -> None:
    """The one alias we've validated by PoC must pass eligibility — a
    regression here means we accidentally tightened a gate."""
    from vllm_mlx.model_aliases import resolve_profile

    profile = resolve_profile("qwen3.5-27b-8bit")
    assert profile is not None, "qwen3.5-27b-8bit alias missing"
    check(profile, alias="qwen3.5-27b-8bit")


def test_default_qwen3_5_27b_alias_fails_check_with_4bit_reason() -> None:
    """The default ``qwen3.5-27b`` alias points at the 4-bit variant —
    eligibility must reject it with a clear 4-bit hint, not the
    generic 'not enabled' message (since supports_dflash=False).
    Confirms users get the right pointer when they pick the wrong
    quantization."""
    from vllm_mlx.model_aliases import resolve_profile

    profile = resolve_profile("qwen3.5-27b")
    assert profile is not None
    # Match-string: capture both reasons (4-bit + not-opted-in). The
    # bare ``raises`` would pass even if the gate silently degraded to
    # the generic message, defeating the point of this regression test.
    with pytest.raises(DFlashUnavailable) as excinfo:
        check(profile, alias="qwen3.5-27b")
    msg = str(excinfo.value)
    assert "4-bit" in msg, (
        f"4-bit hint missing from DFlashUnavailable message; got:\n{msg}"
    )
