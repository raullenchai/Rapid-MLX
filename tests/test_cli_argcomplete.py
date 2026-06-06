# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the rapid-mlx CLI shell-completion wiring.

Locks in three things that are easy to break:

1. The ``# PYTHON_ARGCOMPLETE_OK`` magic marker stays in the first 1024
   bytes of ``cli.py``. ``register-python-argcomplete`` grep-skips
   scripts without this marker for speed — losing it silently turns
   off tab completion across every install.
2. ``alias_completer`` returns aliases filtered by prefix (the actual
   contract argcomplete invokes per keystroke).
3. ``alias_csv_completer`` correctly carries the comma-separated
   prefix forward so ``--models qwen3.5-4b,gem<TAB>`` expands the
   trailing token only.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vllm_mlx._completion import (
    _ALIASES_PATH,
    alias_completer,
    alias_csv_completer,
)

_CLI_PATH = Path(__file__).parent.parent / "vllm_mlx" / "cli.py"


def test_python_argcomplete_ok_marker_present() -> None:
    """``register-python-argcomplete`` grep-scans the first ~1024 bytes
    of the script for ``PYTHON_ARGCOMPLETE_OK``. Without it, the shell
    completion handler refuses to invoke the script entirely — losing
    the marker would silently break tab completion on every install
    until a user noticed and reported it."""
    head = _CLI_PATH.read_bytes()[:1024]
    assert b"PYTHON_ARGCOMPLETE_OK" in head, (
        "PYTHON_ARGCOMPLETE_OK magic marker must be in the first 1024 "
        "bytes of cli.py — argcomplete grep-skips scripts without it."
    )


def test_alias_completer_no_prefix_returns_sorted_list() -> None:
    """Empty prefix → full sorted alias list (shell collapses to the
    longest common prefix and re-prompts on second Tab, standard UX)."""
    result = alias_completer("")
    assert len(result) > 50, (
        f"expected the full alias list, got {len(result)}; if the file "
        "moved or load_alias_names returned [], this regressed"
    )
    assert result == sorted(result), "completer must return sorted output"


def test_alias_completer_filters_by_prefix() -> None:
    """``gemma-4-<TAB>`` must surface all gemma-4-* aliases and nothing
    else. This is the user-visible contract: a startswith filter on the
    alias name."""
    result = alias_completer("gemma-4-")
    assert len(result) >= 5, "should match at least 5 gemma-4 aliases"
    assert all(n.startswith("gemma-4-") for n in result), (
        f"completer leaked non-matching aliases: "
        f"{[n for n in result if not n.startswith('gemma-4-')]}"
    )


def test_alias_completer_unknown_prefix_returns_empty() -> None:
    """Unknown prefix → []; the shell will then beep/fall back to
    filename completion. Must not raise or return stale results."""
    result = alias_completer("this-alias-does-not-exist-xyz")
    assert result == []


def test_alias_completer_handles_missing_aliases_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Tab completion must NEVER raise. A missing or corrupt
    aliases.json should degrade to ``[]`` — anything else propagates
    as a Python traceback into the user's shell, which is worse than a
    silent no-match."""
    missing = tmp_path / "no_such_aliases.json"
    monkeypatch.setattr("vllm_mlx._completion._ALIASES_PATH", missing)

    assert alias_completer("") == []
    assert alias_completer("gemma-4-") == []


def test_alias_completer_handles_corrupt_aliases_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Same robustness contract for a syntactically broken file."""
    corrupt = tmp_path / "broken.json"
    corrupt.write_text("not valid json {{")
    monkeypatch.setattr("vllm_mlx._completion._ALIASES_PATH", corrupt)

    assert alias_completer("") == []


def test_alias_csv_completer_first_token() -> None:
    """``rapid-mlx doctor --models <TAB>`` (no comma yet) behaves
    exactly like ``alias_completer``."""
    no_comma = alias_csv_completer("gemma-4-")
    plain = alias_completer("gemma-4-")
    assert no_comma == plain


def test_alias_csv_completer_appends_to_existing_csv() -> None:
    """``--models qwen3.5-4b,gem<TAB>`` should expand only the
    trailing token but emit the full re-assembled value so the shell
    inserts ``qwen3.5-4b,gemma-4-12b`` rather than dropping the head."""
    result = alias_csv_completer("qwen3.5-4b,gemma-4-")
    assert all(m.startswith("qwen3.5-4b,gemma-4-") for m in result), (
        f"csv completer dropped the head before the comma: "
        f"{[m for m in result if not m.startswith('qwen3.5-4b,')]}"
    )
    assert len(result) >= 5, "should match at least 5 gemma-4-* tokens"


def test_alias_csv_completer_multiple_commas() -> None:
    """``--models a,b,c<TAB>`` only completes ``c``; ``a,b,`` is
    carried through unchanged. Lock this in because rpartition vs
    partition is an easy-to-flip bug."""
    result = alias_csv_completer("qwen3.5-4b,gemma-4-12b,qwen3.6-")
    assert all(m.startswith("qwen3.5-4b,gemma-4-12b,qwen3.6-") for m in result), (
        "csv completer must preserve all prior csv tokens"
    )


def test_aliases_path_resolves_to_real_file() -> None:
    """Sanity check the path we ship resolves to a real file in the
    installed package — catches a path bug at the module level."""
    assert _ALIASES_PATH.exists(), (
        f"aliases.json missing at {_ALIASES_PATH}; if the file moved, "
        "_completion.py needs the new location"
    )
