"""Pure-data contracts for audio alias metadata."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ALIASES_PATH = (
    Path(__file__).resolve().parents[1] / "rapid_mlx" / "audio" / "aliases.json"
)
PARAKEET_V3_LANGUAGES = {
    "bg",
    "cs",
    "da",
    "de",
    "el",
    "en",
    "es",
    "et",
    "fi",
    "fr",
    "hr",
    "hu",
    "it",
    "lt",
    "lv",
    "mt",
    "nl",
    "pl",
    "pt",
    "ro",
    "ru",
    "sk",
    "sl",
    "sv",
    "uk",
}


@pytest.mark.parametrize(
    "alias,expected_languages",
    [
        ("parakeet", {"en"}),
        ("parakeet-tdt-0.6b-v2", {"en"}),
        ("parakeet-v3", PARAKEET_V3_LANGUAGES),
        ("parakeet-tdt-0.6b-v3", PARAKEET_V3_LANGUAGES),
    ],
)
def test_parakeet_language_metadata_distinguishes_v2_and_v3(
    alias: str, expected_languages: set[str]
) -> None:
    aliases = json.loads(ALIASES_PATH.read_text(encoding="utf-8"))
    languages = aliases[alias]["languages"].split(",")
    assert set(languages) == expected_languages
    assert all(len(language) == 2 and language.isalpha() for language in languages)
