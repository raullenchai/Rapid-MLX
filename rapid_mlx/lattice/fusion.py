# SPDX-License-Identifier: Apache-2.0
"""Fusion defaults resolved only by existing post-load installers."""

from __future__ import annotations

from typing import Any

from . import ModelProfile, resolve, runtime_context


def install_enabled(
    name: str, runtime: Any, *, route: str | None = None, mode: str = "unknown"
) -> bool:
    """Resolve a default without interpreting any legacy environment switch.

    The explicit installer fact is separate from checkpoint declarations.
    Existing callers retain all class/route/layout checks and probes; their
    installed markers and classes carry the decision into forward calls.
    Never call this helper from a forward, eligibility or per-layer function.
    """
    try:
        device = dict(runtime.device_info())
    except Exception:
        device = {}
    context = runtime_context(
        device, model=ModelProfile(family=route or "unknown"), mode=mode
    )
    return bool(resolve(name, context).value)
