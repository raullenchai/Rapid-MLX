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

    # Column position of "Size" in the header — the column immediately
    # after the alias (issue #1286). Every data row must have its second
    # column starting at the same offset.
    size_col = header.index("Size")
    # The split-on-spaces second token starts at the first non-space
    # character after the alias. With the dynamic width that position
    # is exactly size_col on every row.
    for row in data_rows:
        # Find the position of the first non-space after the leading
        # alias name. The alias may itself contain hyphens but not
        # spaces; the first space-delimited gap separates alias and
        # the Size column.
        stripped = row[2:]  # drop the leading "  " indent
        first_gap = stripped.find(" ")
        # Index of the second column (Size) in absolute terms:
        second_col_abs = (
            2
            + len(stripped[:first_gap])
            + (len(stripped[first_gap:]) - len(stripped[first_gap:].lstrip()))
        )
        assert second_col_abs == size_col, (
            f"Row mis-aligned: size col at {second_col_abs}, header at "
            f"{size_col}. Row: {row!r}"
        )


def test_spec_decode_column_aligns_despite_long_reasoning(capsys):
    """#1999: a Reasoning value wider than the old fixed 12 chars
    (``deepseek_r1_distill`` is 19) must not shift Spec-Decode and every
    column right of it out from under the header. Alias/Size/Tools/Reasoning
    are ASCII, so the Spec-Decode column START is at a fixed char offset in
    both the header and every row; the wide ✓/✗/— glyphs only appear from
    Spec-Decode onward.
    """
    out = _capture(capsys)
    lines = [ln for ln in out.splitlines() if ln.startswith("  ")]
    header_idx = next(
        i
        for i, ln in enumerate(lines)
        if ln.lstrip().startswith("Alias") and "DFlash" in ln and "HF id" not in ln
    )
    header = lines[header_idx]
    spec_col = header.index("Spec-Decode")
    data_rows: list[str] = []
    for ln in lines[header_idx + 2 :]:
        if set(ln.strip()) == {"─"}:
            break
        data_rows.append(ln)

    # At least one row carries the long reasoning value that used to overflow.
    assert any("deepseek_r1_distill" in r for r in data_rows), (
        "expected a row with the long 'deepseek_r1_distill' reasoning value"
    )
    for row in data_rows:
        assert len(row) > spec_col, f"row too short for Spec-Decode col: {row!r}"
        assert row[spec_col - 1] == " ", (
            f"no gap before Spec-Decode at col {spec_col}: {row!r}"
        )
        assert row[spec_col] != " ", (
            f"Spec-Decode value not at header col {spec_col}: {row!r}"
        )


def test_alias_column_width_floor_is_24(capsys, monkeypatch):
    """If the registry only has short names, the alias column must
    still be 24 wide so short tables don't feel cramped."""
    from vllm_mlx import model_aliases
    from vllm_mlx.model_aliases import AliasProfile

    short_profile = AliasProfile(hf_path="x/y")
    monkeypatch.setattr(model_aliases, "list_profiles", lambda: {"qwen": short_profile})
    out = _capture(capsys)
    # The alias column's floor is 24, so the next column (Size, added for
    # issue #1286) starts at position 2 + 24 + 1 = 27 → offset 25 from Alias.
    header_line = next(
        ln for ln in out.splitlines() if "Alias" in ln and "DFlash" in ln
    )
    assert header_line.index("Size") - header_line.index("Alias") == 25, (
        "Alias-column floor regression: short registry should still pad "
        "to 24 chars (Size header at offset 25 from Alias)."
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


def test_search_narrows_to_matching_aliases(capsys):
    """#2355: ``--search`` is a case-insensitive alias substring match that
    narrows the 200+-line catalog to the requested slice and retitles the
    section with the term."""
    out = _capture(capsys, search="qwen3-0.6b")
    # The section title reflects the active search.
    assert "matching 'qwen3-0.6b'" in out
    lines = [ln for ln in out.splitlines() if ln.lstrip().startswith("qwen3-0.6b")]
    # Only the qwen3-0.6b aliases (and nothing larger) survive the filter.
    ids = {ln.split()[0] for ln in lines}
    assert "qwen3-0.6b" in ids
    assert all(a.startswith("qwen3-0.6b") for a in ids)
    assert len(ids) >= 3  # 0.6b + 0.6b-4bit + 0.6b-8bit


def test_search_ignores_case(capsys):
    """#2355: the substring match is case-insensitive."""
    out = _capture(capsys, search="Qwen3")
    assert "matching 'Qwen3'" in out
    # Some qwen rows survive (the filter is casefolded).
    assert any(ln.lstrip().startswith("qwen3") for ln in out.splitlines())


def test_modality_audio_shows_only_audio_section(capsys):
    """#2355: ``--modality audio`` blanks the text chat table and shows
    only the audio section — the terminal settles on the requested slice."""
    out = _capture(capsys, modality="audio")
    assert "Models [audio]" in out
    # The audio section header is present.
    assert "[audio:" in out or "Audio models" in out
    # Text chat aliases (e.g. qwen3-0.6b) must NOT appear.
    assert not any(ln.lstrip().startswith("qwen3-0.6b") for ln in out.splitlines())


def test_modality_video_gen_shows_video_section(capsys):
    """#2355: ``--modality video-gen`` shows the video section and hides
    the text chat table + image section."""
    out = _capture(capsys, modality="video-gen")
    assert "Models [video-gen]" in out
    assert "Video models" in out
    # Image section must not be present (only the requested modality).
    assert "Image models" not in out
    # A text chat alias must not leak in.
    assert not any(ln.lstrip().startswith("qwen3-0.6b") for ln in out.splitlines())


def test_default_view_preserves_full_catalog_and_recipe_pointer(capsys):
    """#2355 regression: no filters keeps the full catalog AND points the
    user to the recipe command for RAM-fit recommendations."""
    out = _capture(capsys)
    assert "Available models" in out
    assert "recipe" in out  # discoverability pointer to recommendations
