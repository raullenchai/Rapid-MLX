# SPDX-License-Identifier: Apache-2.0
"""Cached crossovers; callers retain all enablement and kernel qualification."""

from __future__ import annotations

import os
from collections.abc import Callable
from functools import lru_cache
from typing import Any, cast

from . import resolve, runtime_context


class AttentionThresholds:
    """Freeze real runtime facts and candidate overrides on first policy use.

    Each kernel owns an adapter. Exact widths preserve arbitrary overlay cells;
    bounded cache hits never enter runtime_context, profile loading or resolve.
    Legacy environment switches remain dynamic at their existing call sites.
    """

    def __init__(self, runtime: Callable[[], Any]):
        @lru_cache(maxsize=1)
        def device_info():
            try:
                return dict(runtime().device_info())
            except Exception:  # Missing device facts stay unknown.
                return {}

        @lru_cache(maxsize=1)
        def environment():
            return dict(os.environ)

        @lru_cache(maxsize=256)
        def minimum(name: str, width: int) -> int:
            ctx = runtime_context(device_info(), width=width)
            return cast(int, resolve(name, ctx, environ=environment()).value)

        self.device_info = device_info
        self.minimum = minimum
