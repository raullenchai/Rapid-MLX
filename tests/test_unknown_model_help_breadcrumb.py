"""Regression for dogfood #2126: the unknown-model help breadcrumb must
not print a hardcoded alias total that contradicts the ``rapid-mlx
models`` command it points at.

``list_aliases()`` == ``list_profiles()`` (182), but ``models`` splits
that registry into tagged sections (chat / audio / video / image) and its
first header shows only the chat subset (172). Embedding a grand total in
the breadcrumb therefore guaranteed a mismatch with the first number a
user lands on. The fix drops the number and lets ``models`` be the single
source of truth for the per-section counts.
"""

from __future__ import annotations

import re

from vllm_mlx import cli


def _capture_unknown_model_help(capsys, name: str = "totally-unknown-xyz") -> str:
    cli._print_unknown_model_help(
        name, full_path_example="mlx-community/Qwen3.5-9B-4bit"
    )
    return capsys.readouterr().out


def test_breadcrumb_still_points_at_models(capsys):
    """The help must still route the user to ``rapid-mlx models``."""
    out = _capture_unknown_model_help(capsys)
    assert "rapid-mlx models" in out
    assert "aliases" in out


def test_breadcrumb_does_not_embed_a_hardcoded_alias_count(capsys):
    """Pre-fix this line read ``... to see all 182 aliases`` while
    ``models`` opened with ``Available models (172 aliases)`` — a visible
    contradiction. The ``models``-pointer line must carry no standalone
    integer count that can drift from what ``models`` displays."""
    out = _capture_unknown_model_help(capsys)
    models_line = next(
        (ln for ln in out.splitlines() if "rapid-mlx models" in ln), None
    )
    assert models_line is not None, f"no models breadcrumb line in: {out!r}"
    assert not re.search(r"\d", models_line), (
        f"breadcrumb must not hardcode an alias count that can contradict "
        f"`rapid-mlx models`, got: {models_line!r}"
    )
