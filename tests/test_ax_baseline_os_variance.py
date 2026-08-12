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

They are a FROZEN corpus for the normalizer, deliberately not tied to the app's
current UI. Do not add a test asserting that a fixture normalizes to a
committed baseline: the two drift apart the first time anyone changes the
chat view, and the churn teaches people to refresh fixtures reflexively — the
exact habit that let three separate UI changes ship with stale baselines. That
the committed baselines match a LIVE app is the macOS golden-flow job's
assertion, made on both operating systems, every PR.
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


def test_fixtures_carry_all_three_known_os_differences():
    """Guard against a vacuous cross-OS test.

    Regenerate both fixtures on one machine and they become identical, and the
    equality test below passes while checking nothing at all. So pin what these
    two dumps exist to carry. The same UI, from the same commit, differs in
    THREE independent ways between the two releases:

      1. the conversation menu's role spelling — AXMenuButton on macOS 26,
         AXPopUpButton on macOS 15 (absorbed by ``_ROLE_EQUIVALENTS``);
      2. an extra lazily-realized AXButton child under the "Hide Sidebar"
         control on macOS 15 (absorbed by ``is_lazy_button_wrapper``, which
         collapses that self-copy for every toolbar button by structure);
      3. an AppKit-synthesised AXTitle="More" on the conversation menu on
         macOS 15 (absorbed by the title rule in ``render_node``).

    Any of the three regressing makes ``--update-baselines`` machine-specific
    again, so the corpus has to keep covering all three.
    """
    macos15 = json.loads((FIXTURES / "chat-settled.macos15.json").read_text())["data"][
        "ui_elements"
    ]
    macos26 = json.loads((FIXTURES / "chat-settled.macos26.json").read_text())["data"][
        "ui_elements"
    ]

    def menu(elements: list[dict]) -> dict:
        return next(
            element
            for element in elements
            if str(element.get("identifier", "")).startswith(
                "Sidebar.Conversation.Menu."
            )
        )

    # 1. role spelling
    assert menu(macos15)["role"] == "AXPopUpButton"
    assert menu(macos26)["role"] == "AXMenuButton"

    # 2. the extra OS-owned sidebar child
    def hide_sidebar_nodes(elements: list[dict]) -> int:
        return sum(1 for e in elements if e.get("description") == "Hide Sidebar")

    assert hide_sidebar_nodes(macos15) == 2
    assert hide_sidebar_nodes(macos26) == 1

    # 3. the synthesised title
    assert menu(macos15).get("title") == "More"
    assert "title" not in menu(macos26)


def test_real_cross_os_dumps_normalize_identically():
    """The whole point: same commit, two macOS versions, one baseline."""
    macos15 = _normalize(FIXTURES / "chat-settled.macos15.json")
    macos26 = _normalize(FIXTURES / "chat-settled.macos26.json")
    assert macos15 == macos26, (
        "the normalizer lets a macOS difference through — a baseline generated "
        "on one macOS cannot be enforced on the other"
    )


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


def test_gpu_gauge_normalizes_identically_across_machines():
    """Apple Silicon reads a live GPU %, Intel/sandboxed says "unavailable".

    Same footer item, different role + attribute + text. A baseline generated
    on one turns the other red on the GPU line unless the normalizer collapses
    both to one token — the exact regression these flows hit.
    """
    apple_silicon = _render(
        {"role": "AXUnknown", "description": "GPU 47 percent", "enabled": True}
    )
    intel_sandboxed = _render(
        {
            "role": "AXStaticText",
            "help": (
                "GPU probe unavailable — Intel Macs and sandboxed apps don't "
                "expose AGXAccelerator utilisation."
            ),
            "value": "text",
            "enabled": True,
        }
    )
    assert apple_silicon == intel_sandboxed
    assert "percent" not in apple_silicon
    assert "AGXAccelerator" not in intel_sandboxed


def test_gpu_canonicalization_does_not_swallow_unrelated_controls():
    """The gauge is matched on its exact wording, not a bare "GPU " prefix.

    A control that merely starts with "GPU" — a settings row, a menu item —
    keeps its own structure, so a real regression there still fails the gate.
    """
    # A gauge-shaped button keeps its own structure — the role guard rejects it
    # even though its text starts like the live reading.
    rendered = _render(
        {
            "role": "AXButton",
            "identifier": "Settings.Category.gpu",
            "description": "GPU settings",
            "enabled": True,
        }
    )
    assert 'desc="GPU settings"' in rendered
    assert "<gpu>" not in rendered
    assert 'id="Settings.Category.gpu"' in rendered

    # Neither guard alone suffices: each of these clears the role check but the
    # full-match rejects the text, so a "GPU 47 percent settings" reading, a
    # "GPU probe unavailable options" note, or an incidental AGXAccelerator
    # mention is never mistaken for the gauge.
    for record in (
        {"role": "AXUnknown", "description": "GPU 47 percent settings"},
        {"role": "AXStaticText", "help": "GPU probe unavailable options"},
        {
            "role": "AXStaticText",
            "help": "Enable AGXAccelerator logging in the developer menu",
        },
    ):
        rendered = _render({**record, "enabled": True})
        assert "<gpu>" not in rendered, rendered


def test_memory_gauge_total_is_machine_independent():
    """The footer memory readout names the machine's total RAM. A 32 GB Mac and
    a 7 GB runner must still normalize to the same line."""
    big = _render(
        {
            "role": "AXUnknown",
            "description": "Memory normal: 12.3 gigabytes used out of 32 gigabytes",
            "enabled": True,
        }
    )
    small = _render(
        {
            "role": "AXUnknown",
            "description": "Memory normal: 4.1 gigabytes used out of 7 gigabytes",
            "enabled": True,
        }
    )
    assert big == small
    # Neither the used reading nor the total survives as a literal number.
    assert "32" not in big and "7" not in big


def test_recommended_header_chip_is_machine_independent():
    """The model-management header names the machine's Apple-silicon chip. An
    M2 Pro dev machine and an M1/M4 runner must normalize to the same line."""

    def header(chip: str) -> str:
        return _render(
            {
                "role": "AXHeading",
                "identifier": "Settings.ModelManagement.RecommendedHeader",
                "description": f"Recommended for your 32 GB · {chip}",
                "enabled": True,
            }
        )

    m2pro = header("M2 Pro")
    m1 = header("M1")
    m4max = header("M4 Max")
    assert m2pro == m1 == m4max
    assert "M2" not in m2pro and "M4" not in m4max
    assert "<chip>" in m2pro

    # Scoped to the header: an "M2" living in a model name or elsewhere is not
    # a chip and keeps its own value.
    elsewhere = _render(
        {
            "role": "AXStaticText",
            "identifier": "Settings.ModelManagement.Row.<model>",
            "description": "llama-M2 · 4-bit",
            "enabled": True,
        }
    )
    assert "M2" in elsewhere and "<chip>" not in elsewhere


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
