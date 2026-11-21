"""Offline trust-boundary checks for installer target validation."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

INSTALLER = Path(__file__).resolve().parents[2] / "install.sh"


@pytest.mark.parametrize(
    "shell", [("/bin/sh",), ("dash",), ("zsh", "--emulate", "sh"), ("bash", "--posix")]
)
def test_piped_installer_requires_bash(shell):
    if shutil.which(shell[0]) is None:
        pytest.skip(f"{shell[0]} unavailable")
    result = subprocess.run(
        shell,
        input=INSTALLER.read_text(),
        env={**os.environ, "RAPID_INSTALL_LIB": "1"},
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == (
        "install.sh needs bash. Run: curl -fsSL https://rapidmlx.com/install.sh | bash\n"
    )


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
    "target", ["1.2.3 @ https://untrusted.invalid", "1.2.3;id", "1.2.3/other"]
)
def test_version_target_cannot_include_requirement_suffix(target):
    result = subprocess.run(
        ["bash", str(INSTALLER), target], capture_output=True, text=True
    )
    assert result.returncode != 0
    assert "Usage:" in result.stderr


@pytest.mark.parametrize("target", ["1.2.3.post1.dev1", "1.2.3+corp", "1.2.3rc1"])
def test_version_suffixes_reach_package_manager(target):
    result = subprocess.run(
        [
            "bash",
            "-c",
            'RAPID_INSTALL_LIB=1 source "$1"; TARGET="$2"; validate_target',
            "bash",
            str(INSTALLER),
            target,
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
