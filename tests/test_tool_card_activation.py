# SPDX-License-Identifier: Apache-2.0
"""Desktop tool-card milestones stay closed and desktop-only."""

import re
from pathlib import Path

from rapid_mlx.telemetry.activation_spec import ACTIVATION_KIND_SURFACE_PAIRS


def test_tool_card_milestones_match_desktop_wire_enum():
    source = (
        Path(__file__).resolve().parents[1]
        / "apps/rapid-mac/Sources/Rapid/Telemetry/TelemetryEvent.swift"
    ).read_text()
    swift_kinds = set(re.findall(r'case toolCard\w+ = "(tool_card_\w+)"', source))
    expected = {
        "tool_card_shown",
        "tool_card_copy_url",
        "tool_card_copy_command",
        "tool_card_launch",
        "tool_card_dismissed",
    }
    assert swift_kinds == expected
    assert {
        (kind, surface)
        for kind, surface in ACTIVATION_KIND_SURFACE_PAIRS
        if kind.startswith("tool_card_")
    } == {(kind, "desktop") for kind in expected}
