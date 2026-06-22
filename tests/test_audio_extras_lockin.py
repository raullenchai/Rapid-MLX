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
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# pyproject.toml structural pins — guard against silent drift.
# ---------------------------------------------------------------------------

PYPROJECT_PATH = Path(__file__).resolve().parent.parent / "pyproject.toml"


def _read_audio_extra_block() -> list[str]:
    """Return the raw lines of the ``[audio]`` extra block.

    Walk pyproject.toml line-by-line rather than parsing TOML — the
    parsed list loses comments / formatting, but the lines view is
    enough for substring assertions and works even on the Python 3.10
    runners that don't ship ``tomllib``.
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


def test_audio_extra_pins_espeakng_loader() -> None:
    """``espeakng-loader`` must be in ``[audio]`` — misaki imports it."""
    block = _read_audio_extra_block()
    joined = "\n".join(block)
    assert "espeakng-loader" in joined, (
        "R6-H1: `[audio]` extra is missing `espeakng-loader`. "
        "Without it, `misaki/espeak.py` crashes on import with "
        "`ModuleNotFoundError: No module named 'espeakng_loader'` "
        "the first time `/v1/audio/speech` is hit with a Kokoro alias. "
        "Re-add the pin to `pyproject.toml [project.optional-dependencies] "
        "audio` (Eva 0.8.7 dogfood lock-in)."
    )


def test_audio_extra_pins_phonemizer_fork() -> None:
    """``phonemizer-fork`` (not vanilla ``phonemizer``) must be pinned.

    Vanilla ``phonemizer>=3.3`` removed ``EspeakWrapper.set_data_path``,
    which is the API misaki uses. The fork keeps the classmethod
    around so Kokoro TTS works on a clean install.
    """
    block = _read_audio_extra_block()
    joined = "\n".join(block)
    assert "phonemizer-fork" in joined, (
        "R6-H1: `[audio]` extra is missing `phonemizer-fork`. "
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
    """
    block = _read_audio_extra_block()
    for line in block:
        stripped = (
            line.split("#", 1)[0].strip().strip(",").strip().strip('"').strip("'")
        )
        # ``phonemizer-fork`` is fine; ``phonemizer>=...`` is the bad pin.
        if stripped.startswith("phonemizer") and not stripped.startswith(
            "phonemizer-fork"
        ):
            raise AssertionError(
                "R6-H1: `[audio]` extra still pins vanilla `phonemizer` "
                f"({stripped!r}) alongside (or instead of) `phonemizer-fork`. "
                "Vanilla phonemizer 3.3 removed `EspeakWrapper.set_data_path` "
                "which Kokoro's misaki dep calls — pip's resolver order is "
                "not deterministic, so leaving both in means CI passes but "
                "fresh installs randomly break. Keep ONLY `phonemizer-fork`."
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
