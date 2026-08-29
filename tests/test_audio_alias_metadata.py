"""Pure-data contracts for audio alias metadata."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ALIASES_PATH = (
    Path(__file__).resolve().parents[1] / "vllm_mlx" / "audio" / "aliases.json"
)


@pytest.mark.parametrize(
    "alias,expected_languages",
    [
        ("parakeet", "en"),
        ("parakeet-tdt-0.6b-v2", "en"),
        ("parakeet-v3", "25 European languages"),
        ("parakeet-tdt-0.6b-v3", "25 European languages"),
    ],
)
def test_parakeet_language_metadata_distinguishes_v2_and_v3(
    alias: str, expected_languages: str
) -> None:
    aliases = json.loads(ALIASES_PATH.read_text(encoding="utf-8"))
    assert aliases[alias]["languages"] == expected_languages
