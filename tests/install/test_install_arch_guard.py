"""install.sh explains why a non-arm64 shell cannot install Rapid-MLX."""

import os
import subprocess
from pathlib import Path

import pytest

INSTALL_SH = Path(__file__).resolve().parents[2] / "install.sh"


def _run(tmp_path: Path, has_arm64: str) -> subprocess.CompletedProcess:
    stubs = tmp_path / "bin"
    stubs.mkdir()
    for name, body in {
        "uname": 'case "$1" in -s) echo Darwin;; -m) echo x86_64;; esac',
        "sysctl": f'case "$*" in *hw.optional.arm64*) echo {has_arm64};; *) echo 0;; esac',
        "curl": "exit 99",
    }.items():
        path = stubs / name
        path.write_text(f"#!/bin/sh\n{body}\n")
        path.chmod(0o755)
    env = {**os.environ, "PATH": f"{stubs}:/usr/bin:/bin", "HOME": str(tmp_path)}
    return subprocess.run(
        ["bash", str(INSTALL_SH)], env=env, capture_output=True, text=True, timeout=30
    )


@pytest.mark.parametrize(
    ("has_arm64", "expected", "unexpected"),
    [
        ("0", "Intel processor", "Rosetta"),
        ("1", "running under Rosetta", "Intel processor"),
    ],
)
def test_non_arm64_shell_gets_a_specific_reason(tmp_path, has_arm64, expected, unexpected):
    result = _run(tmp_path, has_arm64)
    output = result.stdout + result.stderr
    assert result.returncode == 1
    assert expected in output
    assert unexpected not in output
    assert "M1 or later" in output or "Apple Silicon" in output
