"""install.sh explains why a non-arm64 shell cannot install Rapid-MLX."""

import os
import subprocess
from pathlib import Path

import pytest

INSTALL_SH = Path(__file__).resolve().parents[2] / "install.sh"


def _run(tmp_path: Path, probe_body: str) -> subprocess.CompletedProcess:
    stubs = tmp_path / "bin"
    stubs.mkdir()
    for name, body in {
        "uname": 'case "$1" in -s) echo Darwin;; -m) echo x86_64;; esac',
        "sysctl": probe_body,
        "curl": "exit 99",
    }.items():
        path = stubs / name
        path.write_text(f"#!/bin/sh\n{body}\n")
        path.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{stubs}:/usr/bin:/bin",
        "HOME": str(tmp_path),
        "RAPID_INSTALL_SYSCTL": str(stubs / "sysctl"),
    }
    return subprocess.run(
        ["bash", str(INSTALL_SH)], env=env, capture_output=True, text=True, timeout=30
    )


@pytest.mark.parametrize(
    ("probe_body", "expected", "unexpected"),
    [
        (
            'case "$2" in machdep.cpu.brand_string) echo "Intel(R) Core";; *) exit 1;; esac',
            "Intel processor",
            "Rosetta",
        ),
        (
            'case "$2" in hw.optional.arm64) echo 1;; *) exit 1;; esac',
            "running under Rosetta",
            "Intel processor",
        ),
        (
            'case "$2" in sysctl.proc_translated) echo 1;; *) exit 1;; esac',
            "running under Rosetta",
            "Intel processor",
        ),
        ("exit 1", "This terminal reports x86_64", "Intel processor"),
    ],
)
def test_non_arm64_shell_gets_a_specific_reason(
    tmp_path, probe_body, expected, unexpected
):
    result = _run(tmp_path, probe_body)
    output = result.stdout + result.stderr
    assert result.returncode == 1
    assert expected in output
    assert unexpected not in output
    assert "M1 or later" in output or "Apple Silicon" in output

    if probe_body == "exit 1":
        assert "Open Terminal without Rosetta" in output
        assert "needs a Mac with Apple Silicon (M1 or later)" in output
