# SPDX-License-Identifier: Apache-2.0
"""Tests for the `rapid-mlx models` table column-alignment contract.

Dogfood-driven: 0.9.5 had a hardcoded 24-char alias column. The actual
registry has names up to 31 chars (``deepseek-coder-v2-lite-16b-4bit``),
which overflowed and shifted the rest of that row's columns. 0.9.6 sizes
the column from the data with a 24-char floor.
"""

from __future__ import annotations

from types import SimpleNamespace

from vllm_mlx.cli import models_command
from vllm_mlx.model_aliases import list_profiles


def _capture(capsys, **arg_overrides):
    args = SimpleNamespace(cached=False, **arg_overrides)
    models_command(args)
    return capsys.readouterr().out


def test_every_row_aligns_with_the_header_separator(capsys):
    """Each data row must have the same number of visible columns and
    the same column positions as the header. With the old fixed 24-char
    alias column, the 31-char ``deepseek-coder-v2-lite-16b-4bit`` row
    pushed Tools / Reasoning / Spec-Decode out one full column position.
    """
    out = _capture(capsys)
    lines = [ln for ln in out.splitlines() if ln.startswith("  ")]
    # Find the header line ("  Alias ... DFlash") and the data rows
    # immediately following (between two separator lines).
    header_idx = next(
        i
        for i, ln in enumerate(lines)
        if ln.lstrip().startswith("Alias") and "DFlash" in ln and "HF id" not in ln
    )
    header = lines[header_idx]
    # The data rows start two lines after the header (separator, then rows)
    # and continue until the next separator line of box-drawing dashes.
    data_rows: list[str] = []
    for ln in lines[header_idx + 2 :]:
        if set(ln.strip()) == {"─"}:
            break
        data_rows.append(ln)
    assert len(data_rows) >= 100, "expected the full 120-alias listing"

    # Column position of "Tools" in the header — every data row must
    # have its second column starting at the same offset.
    tools_col = header.index("Tools")
    # The split-on-spaces second token starts at the first non-space
    # character after the alias. With the dynamic width that position
    # is exactly tools_col on every row.
    for row in data_rows:
        # Find the position of the first non-space after the leading
        # alias name. The alias may itself contain hyphens but not
        # spaces; the first space-delimited gap separates alias and
        # tools.
        stripped = row[2:]  # drop the leading "  " indent
        first_gap = stripped.find(" ")
        # Index of the second column (Tools) in absolute terms:
        second_col_abs = (
            2
            + len(stripped[:first_gap])
            + (len(stripped[first_gap:]) - len(stripped[first_gap:].lstrip()))
        )
        assert second_col_abs == tools_col, (
            f"Row mis-aligned: tools col at {second_col_abs}, header at "
            f"{tools_col}. Row: {row!r}"
        )


def test_alias_column_width_floor_is_24(capsys, monkeypatch):
    """If the registry only has short names, the alias column must
    still be 24 wide so short tables don't feel cramped."""
    from vllm_mlx import model_aliases
    from vllm_mlx.model_aliases import AliasProfile

    short_profile = AliasProfile(hf_path="x/y")
    monkeypatch.setattr(model_aliases, "list_profiles", lambda: {"qwen": short_profile})
    out = _capture(capsys)
    # Header has "Alias" followed by at least 19 spaces before "Tools"
    # → column starts at position 2 + 24 + 1 = 27.
    header_line = next(
        ln for ln in out.splitlines() if "Alias" in ln and "DFlash" in ln
    )
    assert header_line.index("Tools") - header_line.index("Alias") == 25, (
        "Alias-column floor regression: short registry should still pad "
        "to 24 chars (Tools header at offset 25 from Alias)."
    )


def test_longest_real_alias_does_not_overflow(capsys):
    """End-to-end: with the real registry, the longest alias still gets
    its column with at least 1 space before the next column."""
    out = _capture(capsys)
    longest_alias = max(list_profiles().keys(), key=len)
    data_line = next(
        ln for ln in out.splitlines() if ln.lstrip().startswith(longest_alias)
    )
    after_alias = data_line[2 + len(longest_alias) :]
    # The character immediately after the alias must be a space and
    # what follows must be the Tools column (not another part of the
    # alias name).
    assert after_alias.startswith(" "), (
        f"No padding between alias and Tools column for {longest_alias!r}"
    )
