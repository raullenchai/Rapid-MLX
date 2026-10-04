# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for the opencode profile's headless claim (#4042).

The profile used to claim opencode was "interactive-only" and set
``query_cmd: null`` — but ``opencode run '<prompt>'`` is a headless one-shot
mode, verified against opencode 1.18.34 (T1–T6 + follow-up turn pass). The
claim cost a skipped e2e gate for no reason.
"""

from rapid_mlx.agents import get_profile, load_profiles


def setup_function():
    load_profiles()


def test_opencode_has_a_headless_query_cmd():
    profile = get_profile("opencode")
    assert profile is not None
    assert profile.testing.binary == "opencode"
    assert profile.testing.query_cmd == "opencode run '{query}'"


def test_opencode_known_issues_do_not_claim_interactive_only():
    profile = get_profile("opencode")
    assert profile is not None
    for issue in profile.known_issues:
        assert "interactive-only" not in issue, issue
