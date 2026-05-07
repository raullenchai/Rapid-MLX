# SPDX-License-Identifier: Apache-2.0
"""Tests for `rapid-mlx models` — pins the per-alias profile rendering."""

from __future__ import annotations

import io
import sys
from unittest.mock import patch

from vllm_mlx import cli
from vllm_mlx.model_aliases import list_profiles


def _capture_models_output() -> str:
    """Run `models_command` with stdout captured. Patches the version-check
    helper so the test stays hermetic (no PyPI lookup at test time)."""
    buf = io.StringIO()
    with (
        patch.object(sys, "stdout", buf),
        patch("vllm_mlx._version_check.print_staleness_warning_if_any"),
    ):
        cli.models_command(None)
    return buf.getvalue()


def test_models_command_lists_all_aliases():
    """Every alias in aliases.json must appear in the table."""
    out = _capture_models_output()
    profiles = list_profiles()
    assert len(profiles) >= 20, "expected 20+ aliases (per project goal)"
    for alias in profiles:
        assert alias in out, f"alias {alias!r} missing from `rapid-mlx models` output"
    assert f"({len(profiles)} aliases)" in out


def test_models_command_shows_capability_columns():
    """Capability columns (Tools, Reasoning, Spec-Decode, Suffix Tier) appear."""
    out = _capture_models_output()
    for header in ("Tools", "Reasoning", "Spec-Decode", "Suffix Tier"):
        assert header in out, f"column header {header!r} missing"


def test_models_command_renders_hybrid_marker_for_qwen35():
    """Hybrid models (e.g. qwen3.5-4b) must show '✗ hybrid' + tier 'n/a'.

    The point of the column is to spare users an `info` round-trip when
    deciding whether spec-decode/suffix-decode will help. Trust the gate.
    """
    out = _capture_models_output()
    profiles = list_profiles()
    qwen35_4b = profiles.get("qwen3.5-4b")
    assert qwen35_4b is not None, "qwen3.5-4b alias missing — fixture drift"
    assert qwen35_4b.is_hybrid, "qwen3.5-4b should still be is_hybrid=True"

    # Find the qwen3.5-4b row and confirm the hybrid markers.
    matches = [line for line in out.splitlines() if "qwen3.5-4b " in line]
    assert matches, "no row found for qwen3.5-4b"
    row = matches[0]
    assert "✗ hybrid" in row, f"expected '✗ hybrid' marker in row: {row!r}"
    assert "n/a" in row, f"expected suffix tier 'n/a' in row: {row!r}"


def test_models_command_renders_parser_for_hermes3_8b():
    """Non-hybrid model with both parsers populated should show them.

    hermes3-8b has tool_call_parser='hermes' and a benched suffix tier
    ('neutral'). Pin this so a tier/data regression on hermes3-8b is
    caught at CI time rather than first user report.
    """
    out = _capture_models_output()
    matches = [line for line in out.splitlines() if "hermes3-8b " in line]
    assert matches, "no row found for hermes3-8b"
    row = matches[0]
    assert "hermes" in row, f"expected 'hermes' tool parser in row: {row!r}"
    assert "neutral" in row, f"expected 'neutral' tier in row: {row!r}"


def test_models_command_renders_em_dash_for_unset_parsers():
    """An alias without tool_call_parser / reasoning_parser should show '—'."""
    out = _capture_models_output()
    profiles = list_profiles()
    # Find any alias that has neither parser populated.
    candidates = [
        a
        for a, p in profiles.items()
        if not p.tool_call_parser and not p.reasoning_parser
    ]
    if not candidates:
        # Schema may have tightened — skip cleanly.
        import pytest

        pytest.skip("no aliases without parsers (schema may have tightened)")
    alias = candidates[0]
    matches = [line for line in out.splitlines() if f"{alias} " in line]
    assert matches, f"no row found for {alias}"
    row = matches[0]
    # Both placeholders should appear.
    assert row.count("—") >= 2, f"expected two em-dashes in row: {row!r}"


def test_models_command_mentions_chat_pull_serve_in_tip():
    """The footer should advertise the four canonical actions."""
    out = _capture_models_output()
    for cmd in ("info", "pull", "chat", "serve"):
        assert f"rapid-mlx {cmd}" in out, (
            f"footer tip missing 'rapid-mlx {cmd}' suggestion"
        )


def test_models_command_subparser_smoke():
    """`rapid-mlx models --help` exits 0 (subparser still wired)."""
    import pytest

    with (
        patch.object(sys, "argv", ["rapid-mlx", "models", "--help"]),
        pytest.raises(SystemExit) as exc,
    ):
        cli.main()
    assert exc.value.code == 0
