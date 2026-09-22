# SPDX-License-Identifier: Apache-2.0
"""Model identity remains captured for the v2 inference emitters."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_ROUTES = ("chat.py", "completions.py", "anthropic.py")
_ROOT = Path(__file__).resolve().parents[1] / "rapid_mlx" / "routes"


@pytest.mark.parametrize("route", _ROUTES)
def test_route_captures_identity_from_the_selected_engine(route):
    tree = ast.parse((_ROOT / route).read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "engine_telemetry_id"
    ]
    assert calls, f"{route} stopped capturing the selected engine identity"
    assert all(len(call.args) == 1 for call in calls)
    assert all(isinstance(call.args[0], ast.Name) for call in calls)
    assert all(call.args[0].id == "engine" for call in calls)


@pytest.mark.parametrize("route", _ROUTES)
def test_route_forwards_the_captured_identity_to_terminal_helpers(route):
    source = (_ROOT / route).read_text(encoding="utf-8")
    assert "served_telemetry_id" in source
    assert "engine_telemetry_id(engine)" in source
