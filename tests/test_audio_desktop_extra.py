# SPDX-License-Identifier: Apache-2.0
"""Lock in the bounded audio dependency group used by the macOS sidecar."""

from __future__ import annotations

import re
from pathlib import Path

try:
    import tomllib  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib  # type: ignore[import-not-found,no-redef]


PYPROJECT_PATH = Path(__file__).resolve().parents[1] / "pyproject.toml"


def _extras() -> dict[str, list[str]]:
    with PYPROJECT_PATH.open("rb") as handle:
        return tomllib.load(handle)["project"]["optional-dependencies"]


def _dependency_name(spec: str) -> str:
    return re.split(r"[<>=!~\[;@\s]", spec, maxsplit=1)[0].lower()


def test_audio_desktop_extra_stays_bounded() -> None:
    specs = _extras()["audio-desktop"]
    names = {_dependency_name(spec) for spec in specs}

    assert names == {"mlx-audio", "soundfile"}
    assert "mlx-audio>=0.2.9,<0.4.4" in specs
    assert "soundfile>=0.12.0" in specs


def test_full_audio_extra_remains_independent_and_complete() -> None:
    extras = _extras()
    desktop_names = {_dependency_name(spec) for spec in extras["audio-desktop"]}
    full_names = {_dependency_name(spec) for spec in extras["audio"]}

    assert desktop_names <= full_names
    assert {"f5-tts-mlx", "misaki", "spacy", "phonemizer-fork"} <= full_names


def test_doctor_desktop_audio_contract_matches_the_extra() -> None:
    """``rapid-mlx doctor`` grades a bundled sidecar against
    ``_AUDIO_DESKTOP_IMPORTS``. If this extra grows a dependency without the
    doctor list growing too, doctor silently stops noticing it is missing.
    """
    from vllm_mlx.doctor import env_health

    assert env_health._AUDIO_DESKTOP_IMPORTS == (
        ("mlx-audio", "mlx_audio"),
        ("soundfile", "soundfile"),
    )
    doctor_names = {dist for dist, _ in env_health._AUDIO_DESKTOP_IMPORTS}
    assert doctor_names == {
        _dependency_name(spec) for spec in _extras()["audio-desktop"]
    }
