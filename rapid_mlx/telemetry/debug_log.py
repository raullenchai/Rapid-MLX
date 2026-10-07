# SPDX-License-Identifier: Apache-2.0
"""Opt-in stderr tracing for telemetry v2 internals."""

from __future__ import annotations

import sys

from .._env import env_truthy

DEBUG_ENV = "RAPID_MLX_TELEMETRY_DEBUG"


def debug_enabled() -> bool:
    """Whether telemetry debug tracing is enabled for this process."""
    return env_truthy(DEBUG_ENV)


def _log(message: str) -> None:
    """Write one debug line to stderr, never to command stdout."""
    if debug_enabled():
        print(f"[telemetry] {message}", file=sys.stderr, flush=True)
