# SPDX-License-Identifier: Apache-2.0
"""Small, process-frozen hardware facts for local terminal generations."""

from __future__ import annotations

import os
import platform
import subprocess
from functools import lru_cache
from typing import Any

from .api.models import ExecutionHostMetrics
from .decode_expectations import chip_name


@lru_cache(maxsize=1)
def _process_host() -> ExecutionHostMetrics | None:
    # Do not initialize a distributed group or read launcher files. Even an
    # ambiguous launcher signal makes a single-machine reference unsuitable.
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return None
    if any(
        os.environ.get(key)
        for key in (
            "MLX_HOSTFILE",
            "MLX_WORLD_SIZE",
            "OMPI_COMM_WORLD_SIZE",
            "PMI_SIZE",
        )
    ):
        return None
    try:
        values = []
        for key in ("machdep.cpu.brand_string", "hw.memsize"):
            values.append(
                subprocess.run(
                    ["/usr/sbin/sysctl", "-n", key],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=2,
                ).stdout.strip()
            )
        chip = chip_name(values[0])
        if chip is None:
            return None
        return ExecutionHostMetrics(chip=chip, memory_bytes=int(values[1]))
    except (OSError, subprocess.SubprocessError, ValueError, TypeError):
        return None


def capture_local_execution_host(output: Any) -> dict | None:
    """Called only at the supported local BatchedEngine output boundary.

    No model work, partial result or ambiguous execution means no host claim.
    Cached replay never enters this boundary and preserves its original facts.
    """
    reference = getattr(output, "execution_reference", None)
    if (
        getattr(output, "finished", True) is not True
        or not isinstance(reference, dict)
        or reference.get("status") != "known"
        or reference.get("backend") != "ordinary"
    ):
        return None
    host = _process_host()
    return None if host is None else host.model_dump()
