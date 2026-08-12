# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the macOS AX golden-flow normalizer."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "apps" / "rapid-mac" / "scripts" / "ax-baseline.py"


@pytest.fixture(scope="module")
def ax_baseline():
    spec = importlib.util.spec_from_file_location("rapid_ax_baseline", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_popup_roles_are_stable_across_macos_versions(ax_baseline):
    menu = ax_baseline.Node({"role": "AXMenuButton", "identifier": "sort"})
    popup = ax_baseline.Node({"role": "AXPopUpButton", "identifier": "sort"})

    assert ax_baseline.render_node(menu, ()) == ax_baseline.render_node(popup, ())
    assert ax_baseline.render_node(menu, ()).startswith("AXPopUpButton ")


def test_live_percentage_in_description_is_scrubbed(ax_baseline):
    cpu = ax_baseline.Node(
        {"role": "AXUnknown", "description": "CPU 99 percent", "enabled": True}
    )

    assert ax_baseline.render_node(cpu, ()) == (
        'AXUnknown desc="CPU <percent>" enabled=true'
    )


@pytest.mark.parametrize("pressure", ["normal", "tight", "critical"])
def test_live_memory_pressure_bucket_is_scrubbed(ax_baseline, pressure):
    memory = ax_baseline.Node(
        {
            "role": "AXUnknown",
            "description": f"Memory {pressure}: 12 gigabytes used out of 32 gigabytes",
            "enabled": True,
        }
    )

    assert ax_baseline.render_node(memory, ()) == (
        'AXUnknown desc="Memory <pressure>: <size> used out of <size>" enabled=true'
    )


@pytest.mark.parametrize("description", ["Hide Sidebar", "Show Sidebar"])
def test_system_sidebar_button_subtree_is_ignored(ax_baseline, description):
    button = ax_baseline.Node(
        {"role": "AXButton", "description": description, "enabled": True}
    )
    button.children.append(
        ax_baseline.Node(
            {
                "role": "AXButton",
                "description": description,
                "help": description,
                "enabled": True,
            }
        )
    )

    assert ax_baseline.render(button, ()) == [
        f'AXButton desc="{description}" enabled=true'
    ]


@pytest.mark.parametrize(
    ("description", "help_text"),
    [
        (
            "Rapid-MLX 0.12.10 · up to date",
            "Rapid-MLX 0.12.10 is the latest release. Click to open Settings → App.",
        ),
        (
            "Rapid-MLX 0.12.11",
            "Rapid-MLX 0.12.11. Click to open Settings → App.",
        ),
        (
            "Rapid-MLX 0.12.10 · update 0.13.0 available",
            "Rapid-MLX 0.13.0 is available (you're on 0.12.10). Click to install.",
        ),
    ],
)
def test_version_pill_collapses_every_update_verdict(
    ax_baseline, description, help_text
):
    """The footer pill states the updater's verdict, which compares the
    running build against the latest PUBLISHED release. That inverts on
    every version-bump PR — the bumped app is newer than any release
    until the one it is cutting exists — so all three states must render
    identically or the bump PR turns the golden flows red."""
    pill = ax_baseline.Node(
        {
            "role": "AXButton",
            "identifier": "Footer.DesktopVersionPill",
            "description": description,
            "help": help_text,
            "enabled": True,
        }
    )

    assert ax_baseline.render_node(pill, ()) == (
        'AXButton id="Footer.DesktopVersionPill" '
        'desc="Rapid-MLX <version> <update-state>" '
        'help="Rapid-MLX <version> <update-state>" enabled=true'
    )


def test_version_text_elsewhere_keeps_its_wording(ax_baseline):
    """Scoped by identifier, not by phrasing: Settings → App publishes the
    same sentence and the flows assert the version it names."""
    pane = ax_baseline.Node(
        {
            "role": "AXStaticText",
            "identifier": "Settings.App.UpToDate",
            "description": "Rapid-MLX 0.12.11 is the latest release.",
            "enabled": True,
        }
    )

    assert ax_baseline.render_node(pane, ()) == (
        'AXStaticText id="Settings.App.UpToDate" '
        'desc="Rapid-MLX <version> is the latest release." enabled=true'
    )
