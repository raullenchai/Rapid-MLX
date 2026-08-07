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
