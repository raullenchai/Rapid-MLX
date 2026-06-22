# SPDX-License-Identifier: Apache-2.0
"""R6-H1 (Eva 0.8.7 dogfood) regression lock-in for the ``[audio]`` extra.

Eva's r1 dogfood run reproduced a fresh ``pip install rapid-mlx[audio]==0.8.7``
followed by a Kokoro TTS request returning **500** with
``AttributeError: type object 'EspeakWrapper' has no attribute
'set_data_path'``. The root cause is a missing-dep + wrong-fork pair:

* ``misaki/espeak.py`` imports ``espeakng_loader`` at module top — that
  package was NOT in the ``[audio]`` extra, so the import crashed before
  the Kokoro pipeline could initialize.
* ``misaki/espeak.py`` calls ``EspeakWrapper.set_data_path(...)`` — that
  classmethod was REMOVED in vanilla ``phonemizer>=3.3``; only
  ``phonemizer-fork>=3.3`` still exposes it. The vanilla pin
  (``phonemizer>=3.2.0``) gleefully resolved to 3.3.0, which then
  failed at runtime with ``AttributeError``.

The r6-C fix adds ``espeakng-loader`` and swaps ``phonemizer`` for
``phonemizer-fork`` in the ``[audio]`` extra. These tests are LOCK-IN
guards — they fail if a future PR drops either pin (or accidentally
re-introduces vanilla ``phonemizer``), so the regression can't ship.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# pyproject.toml structural pins — guard against silent drift.
# ---------------------------------------------------------------------------

PYPROJECT_PATH = Path(__file__).resolve().parent.parent / "pyproject.toml"


def _read_audio_extra_block() -> list[str]:
    """Return the raw lines of the ``[audio]`` extra block (including
    comments and whitespace, as they appear in pyproject.toml).

    Walk the file line-by-line rather than parsing TOML — the parsed
    list loses comments / formatting, and ``tomllib`` is stdlib only
    on Python ≥3.11 (the CI matrix still runs 3.10).
    """
    text = PYPROJECT_PATH.read_text()
    lines = text.splitlines()
    out: list[str] = []
    in_block = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("audio = ["):
            in_block = True
            continue
        if in_block:
            if stripped.startswith("]"):
                break
            out.append(line)
    if not out:
        raise AssertionError(
            "R6-H1: pyproject.toml is missing the `[audio]` extra block "
            "entirely — guard catches a future rename / extras refactor."
        )
    return out


def _parsed_dependency_names() -> set[str]:
    """Return the set of bare package names declared in ``[audio]``.

    Codex r2 BLOCKING: the previous lockin used substring match
    against the entire block (comments included), so a future PR that
    deleted the actual ``"espeakng-loader>=0.2.0"`` line while leaving
    the explanatory comment intact would keep the test green. Parse
    each line: strip comments, drop blank lines, strip quotes/commas/
    whitespace, then split off the version specifier so we get just
    the package name. Returns a set of lowercased package names — the
    assertion is against the actual installed dep, not the surrounding
    documentation.
    """
    # PEP 508 separators that terminate the package name token.
    # ``re`` keeps the split cheap and language-correct (vs hand-coding
    # the precedence of >= / [ / ; / @).
    name_split_re = re.compile(r"[<>=!~\[;@\s]")
    out: set[str] = set()
    for raw in _read_audio_extra_block():
        # Strip inline comment first, then strip trailing comma and
        # surrounding whitespace + quotes. Empty or pure-comment lines
        # collapse to empty and are skipped.
        no_comment = raw.split("#", 1)[0]
        cleaned = no_comment.strip().rstrip(",").strip().strip('"').strip("'")
        if not cleaned:
            continue
        # First token before any version specifier IS the package name.
        name = name_split_re.split(cleaned, 1)[0].strip().lower()
        if name:
            out.add(name)
    return out


def test_audio_extra_pins_espeakng_loader() -> None:
    """``espeakng-loader`` must be in ``[audio]`` — misaki imports it.

    Codex r2 BLOCKING: assert against the PARSED dependency name set,
    not raw block text. Comments that mention ``espeakng-loader`` in
    English prose (e.g. the rationale comment above the actual pin)
    would otherwise let the assertion stay green even after the dep
    line is deleted.
    """
    names = _parsed_dependency_names()
    assert "espeakng-loader" in names, (
        "R6-H1: `[audio]` extra is missing `espeakng-loader`. "
        f"Parsed dep names: {sorted(names)}. "
        "Without `espeakng-loader`, `misaki/espeak.py` crashes on "
        "import with `ModuleNotFoundError: No module named "
        "'espeakng_loader'` the first time `/v1/audio/speech` is hit "
        "with a Kokoro alias. Re-add the pin to "
        "`pyproject.toml [project.optional-dependencies] audio` "
        "(Eva 0.8.7 dogfood lock-in)."
    )


def test_audio_extra_pins_phonemizer_fork() -> None:
    """``phonemizer-fork`` (not vanilla ``phonemizer``) must be pinned.

    Vanilla ``phonemizer>=3.3`` removed ``EspeakWrapper.set_data_path``,
    which is the API misaki uses. The fork keeps the classmethod
    around so Kokoro TTS works on a clean install.

    Codex r2 BLOCKING: assert against the parsed dependency name set.
    Documentation that names ``phonemizer-fork`` in prose must not
    spoof the existence of the pin.
    """
    names = _parsed_dependency_names()
    assert "phonemizer-fork" in names, (
        "R6-H1: `[audio]` extra is missing `phonemizer-fork`. "
        f"Parsed dep names: {sorted(names)}. "
        "Vanilla `phonemizer>=3.3` removed `EspeakWrapper.set_data_path`, "
        "which misaki still calls. Add `phonemizer-fork>=3.3.0` to the "
        "extra. (Eva 0.8.7 dogfood lock-in)"
    )


def test_audio_extra_does_not_pin_vanilla_phonemizer() -> None:
    """Vanilla ``phonemizer`` must NOT co-exist with the fork.

    Both packages claim the same top-level import name (``phonemizer``);
    leaving both in produces non-deterministic resolution depending on
    pip's resolver order. Drop the vanilla pin entirely — the fork
    supersedes it.

    Codex r2 BLOCKING: assert against parsed names so a comment that
    happens to mention "phonemizer" in prose doesn't trip the guard,
    AND a future PR that re-adds the vanilla pin still gets caught.
    """
    names = _parsed_dependency_names()
    if "phonemizer" in names:
        raise AssertionError(
            f"R6-H1: `[audio]` extra still pins vanilla `phonemizer` "
            f"alongside (or instead of) `phonemizer-fork`. Parsed dep "
            f"names: {sorted(names)}. Vanilla phonemizer 3.3 removed "
            "`EspeakWrapper.set_data_path` which Kokoro's misaki dep "
            "calls — pip's resolver order is not deterministic, so "
            "leaving both in means CI passes but fresh installs "
            "randomly break. Keep ONLY `phonemizer-fork`."
        )


# ---------------------------------------------------------------------------
# Runtime importability — only runs when [audio] is actually installed.
# ---------------------------------------------------------------------------


_ESPEAKNG_LOADER_AVAILABLE = importlib.util.find_spec("espeakng_loader") is not None
_PHONEMIZER_AVAILABLE = importlib.util.find_spec("phonemizer") is not None


@pytest.mark.skipif(
    not _ESPEAKNG_LOADER_AVAILABLE,
    reason="espeakng-loader not installed (base install — pyproject pin "
    "test above guards drift; runtime test only fires when [audio] is in)",
)
def test_espeakng_loader_importable_and_exposes_data_path() -> None:
    """When ``[audio]`` is installed, ``espeakng_loader`` must work.

    The misaki integration calls ``espeakng_loader.get_data_path()`` and
    ``espeakng_loader.get_library_path()`` — if either attribute goes
    away, Kokoro TTS will 500 on the first request even though the
    package is technically installed.
    """
    import espeakng_loader

    assert hasattr(espeakng_loader, "get_data_path"), (
        "espeakng_loader is installed but `get_data_path()` is missing — "
        "misaki's Kokoro integration calls this; the [audio] extra needs "
        "a version pin that still exposes it."
    )
    assert hasattr(espeakng_loader, "get_library_path"), (
        "espeakng_loader is installed but `get_library_path()` is missing — "
        "misaki's Kokoro integration calls this; the [audio] extra needs "
        "a version pin that still exposes it."
    )


@pytest.mark.skipif(
    not _PHONEMIZER_AVAILABLE,
    reason="phonemizer not installed (base install — pyproject pin "
    "test above guards drift; runtime test only fires when [audio] is in)",
)
def test_phonemizer_wrapper_exposes_set_data_path() -> None:
    """When ``phonemizer`` is installed, ``EspeakWrapper.set_data_path``
    must exist.

    Only ``phonemizer-fork`` keeps this classmethod around in
    3.3+ — vanilla phonemizer 3.3 removed it. Misaki still calls
    ``EspeakWrapper.set_data_path(espeakng_loader.get_data_path())`` at
    Kokoro init, so if this attribute disappears the first
    ``/v1/audio/speech`` request 500s with ``AttributeError``.
    """
    from phonemizer.backend.espeak.wrapper import EspeakWrapper

    assert hasattr(EspeakWrapper, "set_data_path"), (
        "R6-H1: `EspeakWrapper.set_data_path` is gone — the installed "
        "phonemizer must be `phonemizer-fork>=3.3` (vanilla phonemizer "
        "3.3 removed this classmethod, and misaki/Kokoro still call it)."
    )
