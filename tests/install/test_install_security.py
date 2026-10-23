"""Offline trust-boundary checks for installer target validation."""

import os
import subprocess
from pathlib import Path

import pytest

INSTALLER = Path(__file__).resolve().parents[2] / "install.sh"


@pytest.mark.parametrize(
    "commit", ["", "HEAD", "main", "a" * 39, "g" * 40, "a" * 40 + ";id"]
)
def test_latest_requires_full_commit_before_install(commit):
    result = subprocess.run(
        ["bash", str(INSTALLER), "latest"],
        env={**os.environ, "RAPID_MLX_GIT_COMMIT": commit},
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "requires RAPID_MLX_GIT_COMMIT" in result.stderr
    assert "Check platform" not in result.stdout
    assert "Rapid-MLX — AI" not in result.stdout


@pytest.mark.parametrize(
    "target", ["1.2.3 @ https://untrusted.invalid", "1.2.3;id", "1.2.3junk"]
)
def test_version_target_cannot_include_requirement_suffix(target):
    result = subprocess.run(
        ["bash", str(INSTALLER), target], capture_output=True, text=True
    )
    assert result.returncode != 0
    assert "Usage:" in result.stderr
