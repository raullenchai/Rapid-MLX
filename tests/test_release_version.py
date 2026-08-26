# SPDX-License-Identifier: Apache-2.0
"""Release-version SSOT regression tests."""

from scripts.release_version import parse_version, version_from_subject


def test_release_order_supports_rc_progression_and_stable_promotion():
    versions = [
        parse_version("0.12.18"),
        parse_version("0.13.0-rc1"),
        parse_version("0.13.0-rc2"),
        parse_version("0.13.0"),
    ]
    assert versions == sorted(versions)


def test_subject_parser_preserves_hyphenated_rc_spelling():
    assert version_from_subject("chore: bump version to 0.13.0-rc1") == "0.13.0-rc1"


def test_auto_release_parser_can_tolerate_squash_suffix():
    assert (
        version_from_subject(
            "chore: bump version to 0.13.0-rc1 (#42)", allow_pr_suffix=True
        )
        == "0.13.0-rc1"
    )


from scripts.release_version import preceding_stable_tag


def test_preceding_pretag_rc2_with_only_stable_baseline():
    # Pre-tag RC2 has no tag yet; the predecessor is the last stable release.
    assert (
        preceding_stable_tag("0.13.0-rc2", ["rapid-mac-v0.12.18"])
        == "rapid-mac-v0.12.18"
    )


def test_preceding_pretag_stable_with_only_earlier_stable():
    # A brand-new stable (0.13.0) that is not yet tagged still finds 0.12.18.
    assert (
        preceding_stable_tag("0.13.0", ["rapid-mac-v0.12.18", "rapid-mac-v0.13.0-rc1"])
        == "rapid-mac-v0.12.18"
    )


def test_preceding_tagged_rc_picks_last_stable_below():
    # RC2 is already tagged alongside a released 0.13.0 stable. 0.13.0 sorts
    # ABOVE 0.13.0-rc2 (RC before stable), so it is not strictly below rc2.
    assert (
        preceding_stable_tag(
            "0.13.0-rc2",
            ["rapid-mac-v0.12.18", "rapid-mac-v0.13.0", "rapid-mac-v0.13.0-rc2"],
        )
        == "rapid-mac-v0.12.18"
    )


def test_preceding_excludes_rcs_and_itself():
    # RCs never count as a predecessor; the exact intended tag is excluded.
    assert (
        preceding_stable_tag(
            "0.13.0-rc2",
            ["rapid-mac-v0.12.18", "rapid-mac-v0.13.0-rc1", "rapid-mac-v0.13.0-rc2"],
        )
        == "rapid-mac-v0.12.18"
    )


def test_preceding_no_baseline_returns_none():
    # No stable tag below the intended version -> no baseline (delta skips).
    assert preceding_stable_tag("0.1.0", []) is None
    assert preceding_stable_tag("0.5.0", ["rapid-mac-v0.5.0"]) is None


from scripts.release_version import preceding_release_tag


def test_preceding_release_rc2_selects_rc1_of_same_line():
    # Sparkle build monotonicity: rc2 must beat rc1, not just the last stable.
    assert (
        preceding_release_tag(
            "0.13.0-rc2", ["rapid-mac-v0.12.18", "rapid-mac-v0.13.0-rc1"]
        )
        == "rapid-mac-v0.13.0-rc1"
    )


def test_preceding_release_stable_selects_latest_rc():
    # A stable 0.13.0 must beat its latest RC (0.13.0-rc2), which sorts below it.
    assert (
        preceding_release_tag(
            "0.13.0",
            ["rapid-mac-v0.12.18", "rapid-mac-v0.13.0-rc1", "rapid-mac-v0.13.0-rc2"],
        )
        == "rapid-mac-v0.13.0-rc2"
    )


def test_preceding_release_rc2_with_stable_already_released():
    # 0.13.0 stable sorts ABOVE 0.13.0-rc2, so it is not below rc2; rc1 is the
    # greatest release predecessor for Sparkle monotonicity.
    assert (
        preceding_release_tag(
            "0.13.0-rc2",
            ["rapid-mac-v0.12.18", "rapid-mac-v0.13.0", "rapid-mac-v0.13.0-rc1"],
        )
        == "rapid-mac-v0.13.0-rc1"
    )


def test_preceding_release_no_predecessor_returns_none():
    assert preceding_release_tag("0.13.0-rc1", ["rapid-mac-v0.13.0-rc1"]) is None
    assert preceding_release_tag("0.13.0-rc1", []) is None
