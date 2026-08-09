# SPDX-License-Identifier: Apache-2.0
"""AX structural baselines must not depend on the macOS version.

A golden-flow baseline is generated on whatever machine a developer happens to
have and then enforced on a GitHub-hosted runner. If any part of the normalized
tree varies with the OS, the two can never agree: whoever regenerates the
baseline turns the other side red, and ``--update-baselines`` stops being a
tool and becomes a trap.

That is not hypothetical. PR #1721 built one commit on both, and the runner's
accessibility tree reported an ``AXTitle`` that the dev machine did not:

    AXPopUpButton id="Sidebar.Conversation.Menu.<uuid>"
        title="More" desc="Conversation actions"      # macOS 15, hosted runner
    AXPopUpButton id="Sidebar.Conversation.Menu.<uuid>"
        desc="Conversation actions"                   # macOS 26, dev machine

The fixtures beside this file are the REAL dumps from those two runs, not
hand-written approximations, so this suite fails if the normalizer ever stops
absorbing that difference.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
BASELINE_TOOL = ROOT / "apps/rapid-mac/scripts/ax-baseline.py"
FIXTURES = Path(__file__).resolve().parent / "fixtures/ax_baseline"
SNAPSHOTS = ROOT / "apps/rapid-mac/Tests/GUIGoldenFlows/__Snapshots__"
# The token the golden flows pass, so fixture normalization matches production.
SCRUB = ("fake-alias",)


def _load_tool():
    """Import ax-baseline.py, whose filename is not a valid module name."""
    spec = importlib.util.spec_from_file_location("ax_baseline", BASELINE_TOOL)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


ax_baseline = _load_tool()


def _normalize(path: Path) -> list[str]:
    return ax_baseline.normalize_dump(path, SCRUB)


def test_fixtures_actually_differ_before_normalization():
    """Guard against a vacuous cross-OS test.

    If someone regenerates both fixtures on one machine they become identical,
    and the equality test below would pass while checking nothing at all. Pin
    the difference these fixtures exist to carry.
    """
    macos15 = json.loads((FIXTURES / "chat-settled.macos15.json").read_text())
    macos26 = json.loads((FIXTURES / "chat-settled.macos26.json").read_text())

    def titled_menus(payload: dict) -> list[str]:
        return [
            element.get("title", "")
            for element in payload["data"]["ui_elements"]
            if str(element.get("identifier", "")).startswith(
                "Sidebar.Conversation.Menu."
            )
        ]

    assert titled_menus(macos15) == ["More"], (
        "the macOS 15 fixture no longer carries the AppKit-synthesised title; "
        "it is no longer evidence of cross-OS variance"
    )
    assert titled_menus(macos26) == [""], (
        "the macOS 26 fixture unexpectedly carries a title; regenerate it on "
        "macOS 26 or this suite proves nothing"
    )


def test_real_cross_os_dumps_normalize_identically():
    """The whole point: same commit, two macOS versions, one baseline."""
    macos15 = _normalize(FIXTURES / "chat-settled.macos15.json")
    macos26 = _normalize(FIXTURES / "chat-settled.macos26.json")
    assert macos15 == macos26, (
        "the normalizer lets a macOS difference through — a baseline generated "
        "on one macOS cannot be enforced on the other"
    )


def test_committed_baseline_accepts_the_other_os_dump():
    """End to end, against the baseline the golden flows actually enforce.

    Equality between the two fixtures is necessary but not sufficient: both
    could normalize to something the committed file does not match.
    """
    observed = _normalize(FIXTURES / "chat-settled.macos15.json")
    committed = (SNAPSHOTS / "chat-restore.answered.txt").read_text().splitlines()
    assert observed == committed


def _render(record: dict) -> str:
    node = ax_baseline.Node(record)
    return ax_baseline.render_node(node, SCRUB)


def test_title_is_dropped_when_a_description_is_present():
    rendered = _render(
        {
            "role": "AXPopUpButton",
            "identifier": "Sidebar.Conversation.Menu.x",
            "title": "More",
            "description": "Conversation actions",
        }
    )
    assert "More" not in rendered
    assert 'desc="Conversation actions"' in rendered


def test_title_is_kept_when_it_is_the_only_label():
    """The rule has to stay narrow or it destroys real signal.

    `Settings.ModelManagement.SortMenu` publishes a title and no description;
    without it that popup is indistinguishable from its neighbour.
    """
    rendered = _render(
        {
            "role": "AXPopUpButton",
            "identifier": "Settings.ModelManagement.SortMenu",
            "title": "Sort",
        }
    )
    assert 'title="Sort"' in rendered


@pytest.mark.parametrize(
    "baseline", sorted(SNAPSHOTS.glob("*.txt")), ids=lambda p: p.name
)
def test_no_committed_baseline_pins_a_title_beside_a_description(baseline: Path):
    """Corpus invariant, enforced on every committed baseline.

    Covers the files this change did not regenerate, and catches a baseline
    written by an older copy of the normalizer.
    """
    offenders = [
        line.strip()
        for line in baseline.read_text().splitlines()
        if " title=" in f" {line.strip()}" and " desc=" in line
    ]
    assert not offenders, (
        f"{baseline.name} pins an AppKit-synthesised title next to the app's "
        f"own description: {offenders}"
    )


def test_window_ordering_ignores_a_title_the_baseline_hides():
    """Sorting must use what is rendered, or two OSes order windows differently
    while rendering identical lines."""
    with_title = ax_baseline.Node(
        {"role": "AXWindow", "title": "More", "description": "Main"}
    )
    without_title = ax_baseline.Node({"role": "AXWindow", "description": "Main"})
    assert ax_baseline.window_sort_key(with_title) == ax_baseline.window_sort_key(
        without_title
    )
