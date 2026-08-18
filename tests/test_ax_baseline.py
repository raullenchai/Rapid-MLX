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


def test_virtual_runner_suffix_is_scoped_to_recommendation_header(ax_baseline):
    physical = ax_baseline.Node(
        {
            "role": "AXHeading",
            "identifier": "Settings.ModelManagement.RecommendedHeader",
            "description": "Recommended for your 16 GB · M2",
        }
    )
    virtual = ax_baseline.Node(
        {
            "role": "AXHeading",
            "identifier": "Settings.ModelManagement.RecommendedHeader",
            "description": "Recommended for your 16 GB · M2 (Virtual)",
        }
    )
    unrelated = ax_baseline.Node(
        {"role": "AXStaticText", "description": "Virtual model"}
    )

    assert ax_baseline.render_node(physical, ()) == ax_baseline.render_node(virtual, ())
    assert "Virtual model" in ax_baseline.render_node(unrelated, ())


def _toolbar_with(ax_baseline, button):
    """Wrap ``button`` in an AXToolbar, since the lazy-copy collapse is armed
    only for toolbar descendants (where AppKit's self-copy is observed)."""
    toolbar = ax_baseline.Node({"role": "AXToolbar", "enabled": True})
    toolbar.children.append(button)
    return toolbar


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

    assert ax_baseline.render(_toolbar_with(ax_baseline, button), ()) == [
        "AXToolbar enabled=true",
        f'  AXButton desc="{description}" enabled=true',
    ]


def test_app_toolbar_button_lazy_copy_is_ignored(ax_baseline):
    """An app-authored toolbar button carries its own identifier, so the old
    description allowlist could not reach it — every new toolbar button would
    have had to be added by hand or turn macOS 26 red against a macOS 15
    baseline. The structural rule collapses macOS 15's lazily-realized inner
    copy (same identifier and description, tooltip as AXHelp) for any of them.
    """
    button = ax_baseline.Node(
        {
            "role": "AXButton",
            "identifier": "Toolbar.SearchChats",
            "description": "Search chats",
            "enabled": True,
        }
    )
    button.children.append(
        ax_baseline.Node(
            {
                "role": "AXButton",
                "identifier": "Toolbar.SearchChats",
                "description": "Search chats",
                "help": "Search chats - Command-K",
                "enabled": True,
            }
        )
    )

    assert ax_baseline.render(_toolbar_with(ax_baseline, button), ()) == [
        "AXToolbar enabled=true",
        '  AXButton id="Toolbar.SearchChats" desc="Search chats" enabled=true',
    ]


def test_system_sidebar_button_order_is_session_independent(ax_baseline):
    sidebar = ax_baseline.Node(
        {"role": "AXButton", "description": "Hide Sidebar", "enabled": True}
    )
    search = ax_baseline.Node(
        {
            "role": "AXButton",
            "identifier": "Toolbar.SearchChats",
            "description": "Search chats",
            "enabled": True,
        }
    )
    toolbar_a = ax_baseline.Node({"role": "AXToolbar", "enabled": True})
    toolbar_a.children = [sidebar, search]
    toolbar_b = ax_baseline.Node({"role": "AXToolbar", "enabled": True})
    toolbar_b.children = [search, sidebar]
    window_a = ax_baseline.Node({"role": "AXWindow", "enabled": True})
    window_a.children = [toolbar_a]
    window_b = ax_baseline.Node({"role": "AXWindow", "enabled": True})
    window_b.children = [toolbar_b]

    assert ax_baseline.render(window_a, ()) == ax_baseline.render(window_b, ())


def test_app_authored_toolbar_order_remains_regression_sensitive(ax_baseline):
    first = ax_baseline.Node(
        {"role": "AXButton", "identifier": "Toolbar.First", "enabled": True}
    )
    second = ax_baseline.Node(
        {"role": "AXButton", "identifier": "Toolbar.Second", "enabled": True}
    )
    toolbar_a = ax_baseline.Node({"role": "AXToolbar", "enabled": True})
    toolbar_a.children = [first, second]
    toolbar_b = ax_baseline.Node({"role": "AXToolbar", "enabled": True})
    toolbar_b.children = [second, first]

    window_a = ax_baseline.Node({"role": "AXWindow", "enabled": True})
    window_a.children = [toolbar_a]
    window_b = ax_baseline.Node({"role": "AXWindow", "enabled": True})
    window_b.children = [toolbar_b]

    assert ax_baseline.render(window_a, ()) != ax_baseline.render(window_b, ())


def test_nested_button_with_its_own_identity_is_preserved(ax_baseline):
    """The collapse is narrow even inside a toolbar: a child button that is a
    DIFFERENT control keeps its line, so a real reparented button still shows up
    as a structural diff.
    """
    outer = ax_baseline.Node(
        {"role": "AXButton", "description": "Outer", "enabled": True}
    )
    outer.children.append(
        ax_baseline.Node({"role": "AXButton", "description": "Inner", "enabled": True})
    )

    assert ax_baseline.render(_toolbar_with(ax_baseline, outer), ()) == [
        "AXToolbar enabled=true",
        '  AXButton desc="Outer" enabled=true',
        '    AXButton desc="Inner" enabled=true',
    ]


def test_anonymous_nested_button_is_preserved(ax_baseline):
    """An anonymous button (no identifier, no description) has no identity to
    re-publish, so a button nested inside it is a distinct control — not
    AppKit's self-copy — even under a toolbar. Collapsing on empty==empty would
    hide that nesting from every golden diff, so the identity guard keeps both.
    """
    outer = ax_baseline.Node({"role": "AXButton", "enabled": True})
    outer.children.append(ax_baseline.Node({"role": "AXButton", "enabled": True}))

    assert ax_baseline.render(_toolbar_with(ax_baseline, outer), ()) == [
        "AXToolbar enabled=true",
        "  AXButton enabled=true",
        "    AXButton enabled=true",
    ]


def test_identical_identity_nested_button_outside_toolbar_is_preserved(ax_baseline):
    """The collapse is scoped to toolbar descendants. AppKit's lazy self-copy is
    only seen there; the identical shape ANYWHERE ELSE is a real nested control,
    so a change to it must still surface as a golden diff.
    """
    outer = ax_baseline.Node(
        {
            "role": "AXButton",
            "identifier": "Some.Button",
            "description": "Twin",
            "enabled": True,
        }
    )
    outer.children.append(
        ax_baseline.Node(
            {
                "role": "AXButton",
                "identifier": "Some.Button",
                "description": "Twin",
                "enabled": True,
            }
        )
    )

    assert ax_baseline.render(outer, ()) == [
        'AXButton id="Some.Button" desc="Twin" enabled=true',
        '  AXButton id="Some.Button" desc="Twin" enabled=true',
    ]


def test_same_identity_toolbar_button_with_a_deeper_child_is_preserved(ax_baseline):
    """Even under a toolbar and even sharing its parent's identity, a button that
    is NOT AppKit's leaf self-copy — here it owns a child of its own — is a real
    control whose structure must stay in the diff. The lazy copy is always a
    single leaf, so requiring that shape keeps this nesting visible.
    """
    outer = ax_baseline.Node(
        {"role": "AXButton", "identifier": "Real.Group", "enabled": True}
    )
    inner = ax_baseline.Node(
        {"role": "AXButton", "identifier": "Real.Group", "enabled": True}
    )
    inner.children.append(
        ax_baseline.Node(
            {"role": "AXStaticText", "description": "count", "enabled": True}
        )
    )
    outer.children.append(inner)

    assert ax_baseline.render(_toolbar_with(ax_baseline, outer), ()) == [
        "AXToolbar enabled=true",
        '  AXButton id="Real.Group" enabled=true',
        '    AXButton id="Real.Group" enabled=true',
        '      AXStaticText desc="count" enabled=true',
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
