"""Identify the installed OpenCode major version for config and test launch."""

from __future__ import annotations

import re
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path


def resolve_executable() -> str:
    """Use the current launch PATH, then the standard per-user installation."""
    binary = shutil.which("opencode")
    return str(
        Path(binary).absolute() if binary else Path.home() / ".opencode/bin/opencode"
    )


def installed_version(
    runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    *,
    binary: str | None = None,
) -> str | None:
    binary = binary or resolve_executable()
    try:
        output = (runner or subprocess.run)(
            [binary, "--version"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.match(r"^(?:opencode\s+)?v?(\d+(?:\.\d+)+)\b", output, re.I)
    return match.group(1) if match else None
