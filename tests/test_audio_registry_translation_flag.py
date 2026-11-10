# SPDX-License-Identifier: Apache-2.0
"""Catalog ``translation`` capability flag (#5718).

The registry is pure JSON parsing, so these run without MLX.
"""

import pytest


@pytest.mark.parametrize(
    "alias,translation",
    [
        ("whisper-large-v3-turbo", False),
        ("whisper-large-v3", True),
        ("whisper-1", True),
    ],
)
def test_translation_capability(alias, translation):
    from rapid_mlx.audio.registry import resolve_audio_alias

    assert resolve_audio_alias(alias).translation is translation


@pytest.mark.parametrize("translation", [None, "false", 0, 1, [], {}])
def test_translation_requires_boolean(monkeypatch, tmp_path, translation):
    import json

    from rapid_mlx.audio import registry

    path = tmp_path / "aliases.json"
    path.write_text(
        json.dumps(
            {
                "test": {
                    "type": "stt",
                    "hf_id": "org/model",
                    "family": "whisper",
                    "translation": translation,
                }
            }
        )
    )
    monkeypatch.setattr(registry, "_registry_path", lambda: str(path))
    with pytest.raises(ValueError, match="translation.*bool"):
        registry._load_registry_uncached()
