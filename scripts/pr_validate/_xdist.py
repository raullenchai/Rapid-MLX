# SPDX-License-Identifier: Apache-2.0
"""Optional parallelism shared by the full-suite validation steps."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys


def _performance_cores() -> int:
    """Prefer physical performance cores on Apple Silicon."""
    if sys.platform == "darwin":
        try:
            proc = subprocess.run(
                ["sysctl", "-n", "hw.perflevel0.physicalcpu"],
                capture_output=True,
                text=True,
                check=True,
            )
            count = int(proc.stdout.strip())
            if count > 0:
                return count
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
    return os.cpu_count() or 1


def pytest_parallel_args() -> tuple[list[str], str]:
    """Return pytest arguments and a human-readable execution note.

    Invalid overrides fail explicitly rather than silently changing concurrency.
    xdist remains optional: the validator's required-plugin preflight excludes it.
    """
    override = os.environ.get("PR_VALIDATE_UNIT_WORKERS")
    if override is not None:
        try:
            workers = int(override)
            if workers < 0:
                raise ValueError
        except ValueError as exc:
            raise ValueError(
                "PR_VALIDATE_UNIT_WORKERS must be a non-negative integer "
                "(0 or 1 runs serially)"
            ) from exc
    else:
        # 2026-10-16, M3 Ultra (20 performance cores, 256 GiB): the full suite
        # took 279/212/213/192 s at 8/12/16/20 workers, with >=90% free memory.
        # Cap at the fastest measured count; smaller hosts use their core count.
        workers = min(20, _performance_cores())
    if importlib.util.find_spec("xdist") is None:
        return [], "pytest-xdist not installed — running serially"
    if workers <= 1:
        return [], f"{workers} pytest worker — running serially"
    return [
        "-n",
        str(workers),
        "--dist",
        "loadgroup",
    ], f"pytest-xdist: {workers} workers (loadgroup)"
