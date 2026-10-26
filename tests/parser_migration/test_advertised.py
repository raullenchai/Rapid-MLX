"""Catalog pairing coverage is distinct from unsupported Cartesian stress cases."""

import json
from pathlib import Path

import pytest

from .harness import route

ROWS = json.loads(Path(__file__).with_name("advertised.json").read_text())


def test_catalog_pairs_have_evidence():
    catalog = json.loads(
        (Path(__file__).parents[2] / "rapid_mlx" / "aliases.json").read_text()
    )
    pairs = {
        (v.get("reasoning_parser"), v["tool_call_parser"])
        for v in catalog.values()
        if v.get("tool_call_parser")
    }
    assert pairs <= {tuple(row["key"][:2]) for row in ROWS}


@pytest.mark.parametrize("row", ROWS, ids=lambda row: str(row["key"]))
def test_advertised_pair_observations(row):
    reasoning, tool, _, thinking, mode, surface, stream, _ = row["key"]
    tools = [
        {
            "type": "function",
            "function": {"name": name, "parameters": {"type": "object"}},
        }
        for name, _ in row["expected_calls"]
    ]
    actual = route(
        reasoning,
        tool,
        row["raw"],
        surface,
        stream,
        thinking=thinking,
        tools_mode=mode,
        tools=tools,
    )
    assert actual == row["result"]
