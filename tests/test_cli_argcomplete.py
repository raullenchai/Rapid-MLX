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
    # Lock in the five QAT aliases shipped in v0.6.78 (PR #523) — a
    # ``len >= 5`` floor alone would have passed even before they were
    # added (pre-#523 there were already 5 gemma-4-* aliases). Explicit
    # membership catches a silent regression where the QAT entries get
    # dropped from aliases.json without the test failing.
    qat_aliases = {
        "gemma-4-12b-qat",
        "gemma-4-12b-qat-8bit",
        "gemma-4-26b-qat",
        "gemma-4-31b-qat",
        "gemma-4-31b-qat-8bit",
    }
    missing = qat_aliases - set(result)
    assert not missing, f"QAT aliases missing from completer output: {missing}"


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


def test_alias_csv_completer_handles_whitespace_after_comma() -> None:
    """``--models a, gem<TAB>`` — the runtime alias parser already
    accepts whitespace around commas (``split + strip``). The completer
    must match that contract so users who naturally type the
    human-friendly ``a, b, c`` shape get suggestions instead of an
    empty list."""
    spaced = alias_csv_completer("qwen3.5-4b, gemma-4-")
    tight = alias_csv_completer("qwen3.5-4b,gemma-4-")
    assert len(spaced) == len(tight), (
        "csv completer must produce the same number of matches whether "
        "the user typed `a,b` or `a, b`"
    )


def test_load_alias_names_rejects_oversized_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A hostile multi-megabyte ``aliases.json`` (supply-chain swap, or
    a dev-machine fat-finger) must not stall every keystroke on a
    multi-second JSON decode. Cap is hard and fail-closed."""
    from vllm_mlx import _completion

    huge = tmp_path / "huge.json"
    payload = "{" + ",".join(f'"{i}":1' for i in range(200_000)) + "}"
    huge.write_text(payload)
    assert huge.stat().st_size > _completion._MAX_ALIASES_BYTES

    monkeypatch.setattr(_completion, "_ALIASES_PATH", huge)
    monkeypatch.setattr(_completion, "_CACHE", None)
    assert _completion._load_alias_names() == []


def test_load_alias_names_strips_unsafe_keys(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A key containing newline / vertical-tab / NUL bytes would split
    argcomplete's line-oriented stdout IPC into multiple bogus
    completions or corrupt the user's terminal. The loader filters
    them out before returning. Legitimate aliases pass through."""
    from vllm_mlx import _completion

    spiked = tmp_path / "spiked.json"
    spiked.write_text(
        '{"good-alias":{}, "evil\\nname":{}, "evil\\u000bname":{}, '
        '"with space":{}, "":{}}'
    )
    monkeypatch.setattr(_completion, "_ALIASES_PATH", spiked)
    monkeypatch.setattr(_completion, "_CACHE", None)

    assert _completion._load_alias_names() == ["good-alias"]


def test_load_alias_names_caches_by_mtime(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A hot Tab burst must hit the in-memory cache, not re-decode
    JSON each keystroke. The cache key is ``(mtime, size)`` so a
    fresh write invalidates automatically."""
    import os

    from vllm_mlx import _completion

    f = tmp_path / "cached.json"
    f.write_text('{"alpha":{}}')
    monkeypatch.setattr(_completion, "_ALIASES_PATH", f)
    monkeypatch.setattr(_completion, "_CACHE", None)

    first = _completion._load_alias_names()
    assert first == ["alpha"]
    cached_id = id(_completion._CACHE)

    second = _completion._load_alias_names()
    assert second == ["alpha"]
    assert id(_completion._CACHE) == cached_id, "second call should reuse cache"

    # Edit the file; the mtime+size change must invalidate.
    f.write_text('{"alpha":{}, "beta":{}}')
    # Force a measurable mtime delta in case the FS has 1-second
    # resolution and the writes landed in the same second.
    future = f.stat().st_mtime + 5
    os.utime(f, (future, future))

    refreshed = _completion._load_alias_names()
    assert refreshed == ["alpha", "beta"]


def test_cli_lazy_imports_argcomplete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``vllm_mlx.cli`` must NOT import ``argcomplete`` at module load
    — the minimal-deps CI lane runs without it. Regression guard for
    the lazy-import fix in the round-2 of this PR."""
    import importlib
    import sys

    # Force-reload cli with argcomplete blocked at import time.
    monkeypatch.setitem(sys.modules, "argcomplete", None)
    sys.modules.pop("vllm_mlx.cli", None)
    cli = importlib.import_module("vllm_mlx.cli")

    # The module imported successfully (no ModuleNotFoundError) and
    # exports the expected entry point.
    assert hasattr(cli, "main"), "cli module missing main() entry point"


def test_autocomplete_handshake_returns_aliases_on_subprocess() -> None:
    """End-to-end shell-completion handshake. Spawn ``python -m
    vllm_mlx.cli`` with the argcomplete env vars and verify fd 8
    receives the expected alias list — proves the magic marker,
    the lazy import, and ``argcomplete.autocomplete(parser)`` all
    line up at runtime, not just in unit tests.

    Skipped when ``argcomplete`` is not importable in the current
    interpreter (minimal-deps CI lane) — the lazy-import path is
    covered by ``test_cli_lazy_imports_argcomplete`` instead."""
    import os
    import subprocess
    import sys

    pytest.importorskip("argcomplete")

    env = {
        **os.environ,
        "_ARGCOMPLETE": "1",
        "COMP_LINE": "rapid-mlx serve gemma-4-",
        "COMP_POINT": "24",
        "_ARGCOMPLETE_IFS": "\n",
    }
    # ``sys.executable`` so the child uses THIS interpreter — relying
    # on the bare ``python3`` alias would silently route to a Python
    # without our editable install or without argcomplete.
    # fd 8 is where argcomplete writes; route it to stdout via the
    # child shell so subprocess.run can capture it.
    result = subprocess.run(
        [
            "bash",
            "-c",
            f"{sys.executable} -m vllm_mlx.cli 8>&1 1>/dev/null 2>/dev/null",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"argcomplete handshake exit={result.returncode}; "
        f"stderr={result.stderr!r}; stdout={result.stdout!r}"
    )
    completions = [ln for ln in result.stdout.splitlines() if ln.strip()]
    assert any(c.startswith("gemma-4-") for c in completions), (
        f"expected gemma-4-* matches, got: {completions[:5]}"
    )
