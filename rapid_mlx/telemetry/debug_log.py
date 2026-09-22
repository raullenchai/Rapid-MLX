# SPDX-License-Identifier: Apache-2.0
"""Opt-in stderr tracing for telemetry v2 internals."""

from __future__ import annotations

import os
import sys

DEBUG_ENV = "RAPID_MLX_TELEMETRY_DEBUG"


def debug_enabled() -> bool:
    """Whether telemetry debug tracing is enabled for this process."""
    value = os.environ.get(DEBUG_ENV, "").strip().lower()
    return value not in ("", "0", "false", "no", "off")


def _log(message: str) -> None:
    """Write one debug line to stderr, never to command stdout."""
    if debug_enabled():
        print(f"[telemetry] {message}", file=sys.stderr, flush=True)
