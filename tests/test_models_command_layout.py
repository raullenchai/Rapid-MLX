# SPDX-License-Identifier: Apache-2.0
"""Tests for the `rapid-mlx models` table column-alignment contract.

Dogfood-driven: 0.9.5 had a hardcoded 24-char alias column. The actual
registry has names up to 31 chars (``deepseek-coder-v2-lite-16b-4bit``),
which overflowed and shifted the rest of that row's columns. 0.9.6 sizes
the column from the data with a 24-char floor.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

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
    # The filter narrows to the matching slice: the qwen3-0.6b rows survive,
    # and a KNOWN UNRELATED alias that exists in the real catalog must NOT
    # leak through. Scoring presence/absence of concrete aliases (rather than
    # parsing every table row) catches a filter that keeps the matches but
    # also lets non-matching rows through (codex r4).
    lines = [ln for ln in out.splitlines() if ln.lstrip().startswith("qwen3-0.6b")]
    ids = {ln.split()[0] for ln in lines}
    assert len(ids) >= 3  # 0.6b + 0.6b-4bit + 0.6b-8bit
    # ``deepseek`` / ``gemma3`` aliases exist in the full catalog but must be
    # filtered out by a ``qwen3-0.6b`` search.
    for leaked in ("deepseek", "gemma3"):
        assert not any(ln.lstrip().startswith(leaked) for ln in out.splitlines()), (
            f"search 'qwen3-0.6b' leaked a {leaked} row"
        )


def test_search_ignores_case(capsys):
    """#2355: the substring match is case-insensitive."""
    out = _capture(capsys, search="Qwen3")
    assert "matching 'Qwen3'" in out
    # Some qwen rows survive (the filter is casefolded).
    assert any(ln.lstrip().startswith("qwen3") for ln in out.splitlines())


def test_search_title_shows_stripped_term(capsys):
    """#2355 (codex r6 NIT): the title shows the stripped search term, not the
    raw ``--search`` value — ``--search " qwen "`` matches ``qwen`` and the
    title must claim ``'qwen'`` (not the literal ``' qwen '``)."""
    out = _capture(capsys, search="  qwen3-0.6b  ")
    assert "matching 'qwen3-0.6b'" in out
    assert "matching '  qwen3-0.6b  '" not in out


def test_modality_audio_shows_only_audio_section(capsys):
    """#2355: ``--modality audio`` blanks the text chat table and shows
    only the audio section — the terminal settles on the requested slice."""
    out = _capture(capsys, modality="audio")
    assert "Models [audio]" in out
    # The audio section header is present.
    assert "[audio:" in out or "Audio models" in out
    # Text chat aliases (e.g. qwen3-0.6b) must NOT appear.
    assert not any(ln.lstrip().startswith("qwen3-0.6b") for ln in out.splitlines())
    # The other tagged sections must be blanked too — audio-only really is
    # audio-only (codex r5 BLOCKING).
    assert "Video models" not in out
    assert "Image models" not in out


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


def test_modality_image_gen_shows_image_section(capsys):
    """#2355: ``--modality image-gen`` shows the image section and hides
    the text chat table + video section."""
    out = _capture(capsys, modality="image-gen")
    assert "Models [image-gen]" in out
    assert "Image models" in out
    # Video section must not be present (only the requested modality).
    assert "Video models" not in out
    # A text chat alias must not leak in.
    assert not any(ln.lstrip().startswith("qwen3-0.6b") for ln in out.splitlines())


def test_modality_text_restricts_to_text_chat(capsys):
    """#2355 (codex review #1): ``--modality text`` must show ONLY the text
    chat table — the video / image / audio sections are blanked so the view
    really is text-only (unlike the default full catalog)."""
    out = _capture(capsys, modality="text")
    assert "Models [text]" in out
    # A text chat alias appears.
    assert any(ln.lstrip().startswith("qwen3-0.6b") for ln in out.splitlines())
    # Tagged sections are suppressed.
    assert "Video models" not in out
    assert "Image models" not in out
    assert "Audio models" not in out


def test_modality_audio_count_does_not_crash(capsys):
    """#2355 (codex review #3): the modality title count must reflect the
    section actually shown without crashing on the audio registry."""
    out = _capture(capsys, modality="audio")
    assert "Models [audio]" in out


def test_broken_audio_registry_propagates_a_genuine_bug(capsys, monkeypatch):
    """#2355 (codex r5 NIT): a REAL bug in the audio registry must surface
    loudly, not be silently swallowed into a misleading empty catalog. Only
    the expectable 'registry unavailable / malformed' failures (absent module,
    missing file, bad JSON) degrade — anything else propagates."""
    import json

    import pytest

    from vllm_mlx import model_aliases as ma
    from vllm_mlx.audio import registry as audio_registry

    # Patch list_profiles so the text table is minimal and deterministic.
    monkeypatch.setattr(
        ma, "list_profiles", lambda: {"qwen3-0.6b": ma.AliasProfile(hf_path="x/q")}
    )

    # list_audio_aliases raises a GENUINE bug (RuntimeError) — not a
    # registry-format failure — so it must propagate, not degrade to [].
    monkeypatch.setattr(
        audio_registry,
        "list_audio_aliases",
        lambda: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    with pytest.raises(RuntimeError, match="boom"):
        _capture(capsys)

    # A malformed aliases.json (JSONDecodeError, a ValueError subclass) is an
    # expectable format failure and still degrades: the text table renders.
    monkeypatch.setattr(
        audio_registry,
        "list_audio_aliases",
        lambda: (_ for _ in ()).throw(json.JSONDecodeError("bad", "doc", 0)),
    )
    out = _capture(capsys)
    assert "qwen3-0.6b" in out  # text table intact despite broken audio registry


def test_modality_audio_count_tolerates_a_broken_audio_registry(capsys, monkeypatch):
    """#2355 (coverage): a broken/missing audio registry must not crash the
    modality count — the guarded fallback yields 0 aliases and the title
    still renders."""
    import sys
    import types

    broken = types.ModuleType("vllm_mlx.audio.registry")
    monkeypatch.setitem(sys.modules, "vllm_mlx.audio.registry", broken)
    out = _capture(capsys, modality="audio")
    assert "Models [audio] (0 aliases)" in out


def test_whole_catalog_search_counts_tagged_matches(capsys, monkeypatch):
    """#2355 (codex r3 BLOCKING): a ``--search`` with no ``--modality`` greps
    the whole catalog, so the title count must include matches in the tagged
    (video / image / audio) sections too. Before the fix, a search matching
    only a tagged model printed its rows under a misleading ``(0 aliases)``."""
    import sys
    import types

    from vllm_mlx import model_aliases
    from vllm_mlx.model_aliases import AliasProfile

    # A tiny controlled registry with one chat + one video-gen + one
    # image-gen alias. ``voodoo`` is deliberately unique to the video lane.
    monkeypatch.setattr(
        model_aliases,
        "list_profiles",
        lambda: {
            "qwen3-0.6b": AliasProfile(hf_path="x/qwen"),
            "voodoo-video": AliasProfile(hf_path="x/voodoo", modality="video-gen"),
            "photon-img": AliasProfile(hf_path="x/photon", modality="image-gen"),
        },
    )
    # Blank the audio registry so the aggregate count is exactly the three
    # aliases above (guarded import degrades to 0 aliases, not a crash).
    monkeypatch.setitem(
        sys.modules,
        "vllm_mlx.audio.registry",
        types.ModuleType("vllm_mlx.audio.registry"),
    )

    out = _capture(capsys, search="voodoo")
    # The single video-lane match must render as an actual ALIAS TABLE ROW
    # (first column == ``voodoo-video``) in the video section — checking any
    # line containing "voodoo" would also be satisfied by the title line
    # "matching 'voodoo'" (codex r4).
    video_rows = [
        ln for ln in out.splitlines() if ln.lstrip().startswith("voodoo-video")
    ]
    assert video_rows, "voodoo-video table row must render under the search"
    # The title counts the single video-lane match, not ``(0 aliases)``.
    assert "matching 'voodoo' (1 aliases)" in out


def test_default_view_preserves_full_catalog_and_recipe_pointer(capsys):
    """#2355 regression: no filters keeps the full catalog AND points the
    user to the recipe command for RAM-fit recommendations."""
    out = _capture(capsys)
    assert "Available models" in out
    assert "recipe" in out  # discoverability pointer to recommendations


@pytest.mark.parametrize("other_flag", ("json", "cached"))
@pytest.mark.parametrize("filter_flag", ("search", "modality"))
def test_filters_do_not_silently_noop_on_other_views(capsys, other_flag, filter_flag):
    """Accepted filter syntax must never return an unfiltered alternate view."""
    args = {"cached": False, "json": False, "search": None, "modality": None}
    args[other_flag] = True
    args[filter_flag] = "qwen" if filter_flag == "search" else "text"
    with pytest.raises(SystemExit) as exc:
        models_command(SimpleNamespace(**args))
    assert exc.value.code == 2
    assert "cannot be combined" in capsys.readouterr().err
