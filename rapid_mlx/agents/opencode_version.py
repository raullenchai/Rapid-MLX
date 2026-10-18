"""Identify the installed OpenCode major version for config and test launch."""

from __future__ import annotations

import re
import shutil
import subprocess
from collections.abc import Callable


def installed_version(
    runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
) -> str | None:
    binary = shutil.which("opencode")
    if not binary:
        return None
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
