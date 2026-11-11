# SPDX-License-Identifier: Apache-2.0
"""Cached GDN enrollment policy; callers retain qualification and env gates."""

from __future__ import annotations

import os
from collections.abc import Callable
from functools import lru_cache
from typing import Any, cast

from . import ModelProfile, resolve, runtime_context


class GDNPolicy:
    """Read device facts once and freeze candidate overrides on first use.

    Route identity records the caller's existing allow_dense distinction.
    Missing widths cannot authorize decode, even through an overlay.
    """

    def __init__(self, runtime: Callable[[], Any]):
        @lru_cache(maxsize=1)
        def device_info():
            try:
                return dict(runtime().device_info())
            except Exception:
                return {}

        @lru_cache(maxsize=1)
        def environment():
            return dict(os.environ)

        @lru_cache(maxsize=256)
        def value(name: str, family: str, hidden: int | None = None):
            ctx = runtime_context(
                device_info(), model=ModelProfile(family=family, hidden_size=hidden)
            )
            return resolve(name, ctx, environ=environment()).value

        self.device_info = device_info
        self.value = value

    def decode(self, hidden: Any, allow_dense: bool) -> bool:
        # Legacy equality admitted e.g. 2048.0 too; preserve it before recording
        # the integer installed fact. Other widths require a positive exact int;
        # the universal cells own legacy versus dense enrollment.
        if hidden == 2048:
            hidden = 2048
        elif not (type(hidden) is int and hidden > 0):
            return False
        family = "qwen35_gdn_dense" if allow_dense else "qwen35_gdn"
        return bool(self.value("qwen35_fused_gdn_decode", family, hidden))

    def projection_enabled(self) -> bool:
        return bool(self.value("gdn_in_proj_fusion", "gdn_in_proj"))

    def max_rows(self) -> int:
        return cast(int, self.value("gdn_in_proj_max_rows", "gdn_in_proj"))
