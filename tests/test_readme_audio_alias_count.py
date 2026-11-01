import re
from pathlib import Path

from rapid_mlx.audio.registry import list_audio_aliases


def test_readme_audio_alias_count_matches_registry():
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(
        encoding="utf-8"
    )
    matches = re.findall(r"All (\d+) aliases across (\d+) families", readme)
    assert len(matches) == 1, "Expected exactly one audio alias count in README.md"

    alias_count, family_count = map(int, matches[0])
    entries = list_audio_aliases()
    assert alias_count == len(entries)
    assert family_count == len({entry.family for entry in entries})
