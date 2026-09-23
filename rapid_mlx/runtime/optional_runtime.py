# SPDX-License-Identifier: Apache-2.0
"""Typed failures for optional serving runtimes."""

from __future__ import annotations

import sys
from typing import Literal

OptionalExtra = Literal["vision", "video", "audio", "image"]
OptionalRuntimeStatus = Literal["absent", "broken", "incompatible"]


class OptionalRuntimeMissing(RuntimeError):  # noqa: N818 - public API name is fixed
    """An actionable, privacy-safe optional-runtime startup failure."""

    def __init__(
        self,
        *,
        extra: OptionalExtra,
        install_hint: str,
        detail: str,
        status: OptionalRuntimeStatus,
        marker_reason: str | None = None,
    ) -> None:
        self.extra = extra
        self.install_hint = install_hint
        self.detail = detail
        self.status = status
        self._marker_reason = marker_reason
        super().__init__(self.format_user_message())

    @property
    def marker_reason(self) -> str:
        if self._marker_reason is not None:
            return self._marker_reason
        return {
            "absent": "runtime_extra_missing",
            "broken": "runtime_broken",
            "incompatible": "runtime_incompatible",
        }[self.status]

    def format_user_message(self) -> str:
        """Return actionable stderr text; ``detail`` never enters telemetry."""
        message = self.detail
        if self.install_hint and self.install_hint not in message:
            message = f"{message}\n{self.install_hint}"
        return message


def handle_optional_runtime_missing(
    exc: OptionalRuntimeMissing,
    *,
    alias_or_path=None,
    engine=None,
    auto_selected: bool = False,
) -> None:
    """Render and record the sole terminal result for an unavailable extra."""
    print(exc.format_user_message(), file=sys.stderr)
    print(
        f"RAPID-MLX-STARTUP-FAILURE: {exc.marker_reason} extra={exc.extra}",
        file=sys.stderr,
    )
    from rapid_mlx.telemetry.server_start import failed

    failed("preflight")
    from rapid_mlx.telemetry.model_events import emit_model_serve_failed

    emit_model_serve_failed(
        exc,
        engine=engine,
        alias_or_path=alias_or_path,
        auto_selected=auto_selected,
    )
    raise SystemExit(2)
