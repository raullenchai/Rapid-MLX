# SPDX-License-Identifier: Apache-2.0
"""Closed startup boundary diagnostics that preserve the original failure."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Literal

StartupBoundary = Literal["weights_load", "draft_attach"]
_STARTUP_BOUNDARIES = frozenset({"weights_load", "draft_attach"})
_BOUNDARY_ATTRIBUTE = "_rapid_mlx_startup_boundary"


def startup_boundary_for(exc: BaseException) -> str | None:
    """Read only a closed token; messages and traceback frames stay local."""
    try:
        value = getattr(exc, _BOUNDARY_ATTRIBUTE, None)
        return value if type(value) is str and value in _STARTUP_BOUNDARIES else None
    except Exception:
        return None


@contextmanager
def startup_failure_boundary(boundary: StartupBoundary) -> Iterator[None]:
    """Annotate a propagated failure without wrapping it or changing its type.

    Call at the precise operation, including worker Future.result(), so the
    annotation survives the thread handoff. Diagnostics must never mask the
    original exception, including exceptions refusing attribute assignment.
    Cancellation and process interrupts deliberately bypass this handler.
    """
    if boundary not in _STARTUP_BOUNDARIES:
        raise ValueError("Unknown engine startup boundary")
    try:
        yield
    except Exception as exc:
        try:
            if startup_boundary_for(exc) is None:
                setattr(exc, _BOUNDARY_ATTRIBUTE, boundary)
        except Exception:
            pass
        raise
