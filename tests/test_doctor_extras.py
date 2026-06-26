# SPDX-License-Identifier: Apache-2.0
"""Tests for the 0.9.7 doctor cleanup:

1. ``YouTube/HF cookies`` (Trio-project leakage) is gone from the Network
   section — rapid-mlx has nothing to do with YouTube.
2. ``anthropic SDK`` row (irrelevant for an MLX inference server) is gone
   from the Optional Tools section.
3. A dedicated ``mlx-vlm 0.5.0+ (dflash extras)`` row exists in the
   Optional Packages section, gated on the actual installed mlx-vlm
   version — the headline 0.9.x feature deserves an explicit check.

These tests pin the user-facing output. A future drive-by that brings
either ripped row back will turn these red and force a conscious
re-litigation of the decision.
"""

from __future__ import annotations

from unittest import mock

from vllm_mlx.doctor import env_health as eh

# ---------------------------------------------------------------------------
# Rip 1: YouTube cookies row removed from Network section
# ---------------------------------------------------------------------------


def test_youtube_cookie_check_is_gone_from_network_section(monkeypatch):
    """The Network section must not mention YouTube — that was Trio-project
    leakage. The HF reachability probe is the only row this section owns."""
    # Clear every YOUTUBE_COOKIES* variant so an env leak from the host
    # session can't mask a regression where the row sneaks back in.
    for name in (
        "YOUTUBE_COOKIES",
        "YOUTUBE_COOKIES_1",
        "YOUTUBE_COOKIES_2",
        "YOUTUBE_COOKIES_3",
        "YOUTUBE_COOKIES_4",
        "YOUTUBE_COOKIES_5",
    ):
        monkeypatch.delenv(name, raising=False)

    def fake_probe() -> tuple[eh.CheckStatus, str]:
        return eh.CheckStatus.OK, "HTTP 200"

    section = eh.section_network(probe=fake_probe)

    for c in section.checks:
        assert "YouTube" not in c.label, (
            f"Network section still mentions YouTube: {c.label!r}. "
            "0.9.7 ripped this vestigial Trio-project check."
        )
        assert "youtube" not in (c.detail or "").lower(), (
            f"Network section detail still mentions youtube: {c.detail!r}"
        )


def test_youtube_cookie_check_gone_even_when_env_var_is_set(monkeypatch):
    """Setting ``YOUTUBE_COOKIES_1`` must NOT resurrect the row — the env-var
    handling itself was ripped, not just the WARN branch."""
    monkeypatch.setenv("YOUTUBE_COOKIES_1", "fake-cookie-content")

    def fake_probe() -> tuple[eh.CheckStatus, str]:
        return eh.CheckStatus.OK, "HTTP 200"

    section = eh.section_network(probe=fake_probe)
    assert not any("YouTube" in c.label for c in section.checks)


# ---------------------------------------------------------------------------
# Rip 2: anthropic SDK row removed from Optional Tools section
# ---------------------------------------------------------------------------


def test_anthropic_sdk_check_is_gone_from_optional_tools():
    """Optional Tools must not probe for the anthropic SDK — irrelevant for
    an MLX inference server. The codex CLI row is the only optional-tool
    row this section owns post-0.9.7."""
    section = eh.section_optional_tools(which=lambda _name: None)

    for c in section.checks:
        assert "anthropic" not in c.label.lower(), (
            f"Optional Tools still mentions anthropic: {c.label!r}. "
            "0.9.7 ripped this — agent-harness use case is too niche."
        )


def test_anthropic_sdk_check_gone_even_when_installed():
    """Even if anthropic happens to be importable, no row should surface —
    the check itself was ripped, not just the WARN branch."""
    # _safe_version is the only path through which a value could surface;
    # forcing it to always return a string proves the call site is gone.
    with mock.patch.object(eh, "_safe_version", return_value="1.99.0"):
        section = eh.section_optional_tools(which=lambda _name: None)
    assert not any("anthropic" in c.label.lower() for c in section.checks)


# ---------------------------------------------------------------------------
# Add: DFlash extras row in Optional Packages section
# ---------------------------------------------------------------------------


def test_dflash_row_ok_when_mlx_vlm_at_min_version():
    """``mlx-vlm == 0.5.0`` exactly → ✓ DFlash row."""

    def fake_ver(dist: str) -> str | None:
        return "0.5.0" if dist == "mlx-vlm" else "1.0.0"

    with mock.patch.object(eh, "_safe_version", side_effect=fake_ver):
        section = eh.section_optional_packages()

    dflash = next((c for c in section.checks if "dflash" in c.label), None)
    assert dflash is not None, (
        "Optional Packages section is missing the DFlash row — "
        f"got rows: {[c.label for c in section.checks]}"
    )
    assert dflash.status is eh.CheckStatus.OK, (
        f"DFlash row should be OK at mlx-vlm 0.5.0; got {dflash.status!r}"
    )
    assert "0.5.0+" in dflash.label and "dflash extras" in dflash.label


def test_dflash_row_ok_when_mlx_vlm_above_min_version():
    """``mlx-vlm == 1.2.3`` → ✓ DFlash row (version comparison handles
    multi-component bumps, not just exact-match)."""

    def fake_ver(dist: str) -> str | None:
        return "1.2.3" if dist == "mlx-vlm" else "1.0.0"

    with mock.patch.object(eh, "_safe_version", side_effect=fake_ver):
        section = eh.section_optional_packages()

    dflash = next(c for c in section.checks if "dflash" in c.label)
    assert dflash.status is eh.CheckStatus.OK


def test_dflash_row_warns_when_mlx_vlm_too_old():
    """``mlx-vlm == 0.4.9`` → ⚠ DFlash row with the current version
    surfaced in the label so the user can see exactly how far they're
    behind without digging through `pip show`."""

    def fake_ver(dist: str) -> str | None:
        return "0.4.9" if dist == "mlx-vlm" else "1.0.0"

    with mock.patch.object(eh, "_safe_version", side_effect=fake_ver):
        section = eh.section_optional_packages()

    dflash = next(c for c in section.checks if "dflash" in c.label)
    assert dflash.status is eh.CheckStatus.WARN
    assert "current: 0.4.9" in dflash.label
    assert "need: 0.5.0+" in dflash.label


def test_dflash_row_warns_when_mlx_vlm_missing():
    """``mlx-vlm`` not installed at all → ⚠ DFlash row labelled
    ``current: not installed``."""

    def fake_ver(dist: str) -> str | None:
        return None if dist == "mlx-vlm" else "1.0.0"

    with mock.patch.object(eh, "_safe_version", side_effect=fake_ver):
        section = eh.section_optional_packages()

    dflash = next(c for c in section.checks if "dflash" in c.label)
    assert dflash.status is eh.CheckStatus.WARN
    assert "current: not installed" in dflash.label


# ---------------------------------------------------------------------------
# _version_at_least helper — pinpoint coverage so a future "simplification"
# can't silently break the version gate.
# ---------------------------------------------------------------------------


def test_version_at_least_accepts_exact_floor():
    assert eh._version_at_least("0.5.0", (0, 5, 0)) is True


def test_version_at_least_accepts_higher():
    assert eh._version_at_least("0.5.1", (0, 5, 0)) is True
    assert eh._version_at_least("1.0.0", (0, 5, 0)) is True
    assert eh._version_at_least("0.10.0", (0, 5, 0)) is True


def test_version_at_least_rejects_lower():
    assert eh._version_at_least("0.4.9", (0, 5, 0)) is False
    assert eh._version_at_least("0.0.1", (0, 5, 0)) is False


def test_version_at_least_handles_pep440_suffixes():
    """PEP 440 pre-release / local-version suffixes must be stripped —
    ``0.5.0rc1`` and ``0.5.0+local`` both clear the (0,5,0) floor."""
    assert eh._version_at_least("0.5.0rc1", (0, 5, 0)) is True
    assert eh._version_at_least("0.5.0+local.1", (0, 5, 0)) is True


def test_version_at_least_handles_short_version_strings():
    """``"1"`` and ``"1.0"`` must compare correctly against a 3-tuple
    floor — short versions get zero-padded."""
    assert eh._version_at_least("1", (0, 5, 0)) is True
    assert eh._version_at_least("1.0", (0, 5, 0)) is True
    assert eh._version_at_least("0.5", (0, 5, 0)) is True


def test_version_at_least_malformed_returns_false():
    """A non-numeric version string returns False — safer to nudge the
    user to upgrade than to silently pass."""
    assert eh._version_at_least("not-a-version", (0, 5, 0)) is False
    assert eh._version_at_least("", (0, 5, 0)) is False
