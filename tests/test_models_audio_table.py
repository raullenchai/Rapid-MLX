# SPDX-License-Identifier: Apache-2.0
"""Audio catalog presentation regression tests."""

from types import SimpleNamespace

from rapid_mlx.cli import models_command


def test_audio_listing_alignment_and_tips(capsys):
    models_command(SimpleNamespace(cached=False, modality="audio"))
    out = capsys.readouterr().out
    assert out.count("Models [audio]") == 1
    assert "Audio models (" not in out
    assert "rapid-mlx chat" not in out
    lines = out.splitlines()
    header = next(line for line in lines if "Alias" in line and "HF id" in line)
    offset = header.index("HF id")
    for alias in ("granite-speech", "kokoro"):
        rows = [line for line in lines if line.split() and line.split()[0] == alias]
        assert rows, f"missing {alias}"
        for row in rows:
            assert row.index(row.split()[-1]) == offset


def test_default_listing_retains_audio_heading(capsys):
    models_command(SimpleNamespace(cached=False))
    assert "Audio models (" in capsys.readouterr().out
