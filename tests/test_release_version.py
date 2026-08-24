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
