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
    is_registry_verified_pair,
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
        dflash_target_revision="a" * 40,
        dflash_draft_revision="b" * 40,
        dflash_algorithm="dflash",
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
        ("user/Qwen3.5-9B-4-bit", True),
        ("user/Qwen3.5-9B_4bit", True),
        ("user/model-4bit-instruct", True),
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


def test_explicit_4bit_main_model_is_experimental() -> None:
    p = AliasProfile(
        hf_path="mlx-community/Qwen3.5-27B-4bit",  # 4-bit!
        supports_dflash=True,
        dflash_draft_model="z-lab/Qwen3.5-27B-DFlash",
    )
    r = report(
        p,
        alias="qwen3.5-27b-4bit",
        explicit=True,
        drafter_model=p.dflash_draft_model,
    )
    assert r.reasons == ()
    assert r.recommendation == "experimental"
    assert "4-bit" in " ".join(r.warnings)
    with pytest.raises(DFlashUnavailable, match="explicit experimental opt-in"):
        check(p, alias="qwen3.5-27b-4bit")


def test_legacy_dflash_4bit_cannot_become_curated_by_registry_flag_alone() -> None:
    profile = AliasProfile(
        hf_path="user/target-4bit",
        supports_dflash=True,
        dflash_draft_model="user/legacy-dflash",
        dflash_algorithm="dflash",
    )
    result = report(profile, alias="legacy-4bit")
    assert result.recommendation == "experimental"
    assert "explicit experimental opt-in" in " ".join(result.reasons)
    with pytest.raises(DFlashUnavailable, match="explicit experimental opt-in"):
        check(profile, alias="legacy-4bit")


def test_registry_pair_receipt_requires_exact_verified_tuple(monkeypatch) -> None:
    from vllm_mlx import model_aliases

    profile = AliasProfile(
        hf_path="user/target-4bit",
        supports_dflash=True,
        dflash_draft_model="user/dflash2",
        dflash_target_revision="a" * 40,
        dflash_draft_revision="b" * 40,
        dflash_algorithm="dflash2",
    )
    monkeypatch.setattr(model_aliases, "list_profiles", lambda: {"target": profile})

    assert is_registry_verified_pair(
        "user/target-4bit", "a" * 40, "user/dflash2", "b" * 40, "dflash2"
    )
    assert not is_registry_verified_pair(
        "user/other-4bit", "a" * 40, "user/dflash2", "b" * 40, "dflash2"
    )
    assert not is_registry_verified_pair(
        "user/target-4bit", "a" * 40, "user/other-drafter", "b" * 40, "dflash2"
    )
    assert not is_registry_verified_pair(
        "user/target-4bit", "a" * 40, "user/dflash2", "b" * 40, "dflash"
    )
    assert not is_registry_verified_pair(
        "user/target-4bit", "c" * 40, "user/dflash2", "b" * 40, "dflash2"
    )


def test_registry_pair_searches_past_duplicate_experimental_alias(monkeypatch) -> None:
    from vllm_mlx import model_aliases

    experimental = AliasProfile(
        hf_path="user/shared-target-4bit",
        supports_dflash=False,
        dflash_draft_model="user/shared-dflash2",
        dflash_target_revision="a" * 40,
        dflash_draft_revision="b" * 40,
        dflash_algorithm="dflash2",
    )
    verified = AliasProfile(
        hf_path="user/shared-target-4bit",
        supports_dflash=True,
        dflash_draft_model="user/shared-dflash2",
        dflash_target_revision="a" * 40,
        dflash_draft_revision="b" * 40,
        dflash_algorithm="dflash2",
    )
    monkeypatch.setattr(
        model_aliases,
        "list_profiles",
        lambda: {"experimental-first": experimental, "verified-second": verified},
    )

    assert is_registry_verified_pair(
        "user/shared-target-4bit",
        "a" * 40,
        "user/shared-dflash2",
        "b" * 40,
        "dflash2",
    )


def test_registry_pair_lookup_fails_closed_on_registry_error(monkeypatch) -> None:
    from vllm_mlx import model_aliases

    def _raise():
        raise RuntimeError("registry unavailable")

    monkeypatch.setattr(model_aliases, "list_profiles", _raise)
    assert not is_registry_verified_pair(
        "user/target", "a" * 40, "user/drafter", "b" * 40, "dflash"
    )


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
        # supports_dflash=False (default)
    )
    r = report(bad, alias="qwen3.6-35b-4bit")
    joined = " ".join(r.reasons)
    assert "MoE" in joined
    assert "4-bit" in " ".join(r.warnings)
    assert "explicit experimental opt-in" in joined
    assert "not DFlash-enabled" in joined
    assert "drafter" in joined


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


def test_qwen3_8_27b_dflash2_pair_remains_explicit_after_negative_bench() -> None:
    """Known pairing metadata must not turn a failed qualification into support."""
    from vllm_mlx.model_aliases import resolve_profile

    profile = resolve_profile("qwen3.8-27b-4bit")
    assert profile is not None
    default_result = report(profile, alias="qwen3.8-27b-4bit")
    assert default_result.recommendation == "experimental"
    assert default_result.reasons

    explicit_result = report(
        profile,
        alias="qwen3.8-27b-4bit",
        explicit=True,
        drafter_model=profile.dflash_draft_model,
    )
    assert explicit_result.reasons == ()
    assert explicit_result.recommendation == "experimental"
    assert "performance-validated" in " ".join(explicit_result.warnings)


def test_default_qwen3_5_27b_alias_fails_check_with_4bit_reason() -> None:
    """The default ``qwen3.5-27b-4bit`` alias points at the 4-bit variant —
    eligibility must reject it with a clear 4-bit hint, not the
    generic 'not enabled' message (since supports_dflash=False).
    Confirms users get the right pointer when they pick the wrong
    quantization."""
    from vllm_mlx.model_aliases import resolve_profile

    profile = resolve_profile("qwen3.5-27b-4bit")
    assert profile is not None
    # Match-string: capture both reasons (4-bit + not-opted-in). The
    # bare ``raises`` would pass even if the gate silently degraded to
    # the generic message, defeating the point of this regression test.
    with pytest.raises(DFlashUnavailable) as excinfo:
        check(profile, alias="qwen3.5-27b-4bit")
    msg = str(excinfo.value)
    assert "not DFlash-enabled" in msg


def test_unknown_4bit_path_explicit_opt_in_is_allowed() -> None:
    profile = AliasProfile(hf_path="user/Qwen3.5-9B-abliterated-4bit")
    result = report(
        profile,
        alias=None,
        explicit=True,
        drafter_model="user/Qwen3.5-9B-abliterated-DFlash",
    )
    assert result.recommendation == "experimental"
    assert result.reasons == ()
    assert "performance-validated" in " ".join(result.warnings)


def test_check_preserves_none_success_contract() -> None:
    assert check(_good_profile(), alias="qwen3.5-27b-8bit") is None


def test_unverified_explicit_check_does_not_inherit_residual_drafter() -> None:
    profile = AliasProfile(
        hf_path="user/unverified",
        supports_dflash=False,
        dflash_draft_model="registry/residual",
    )
    with pytest.raises(DFlashUnavailable, match="explicit drafter"):
        check(profile, explicit=True, drafter_model=None)


def test_verified_target_with_overridden_drafter_is_experimental() -> None:
    result = report(
        _good_profile(),
        explicit=True,
        drafter_model="user/different-drafter",
    )
    assert result.recommendation == "experimental"
    assert "performance-validated" in " ".join(result.warnings)
