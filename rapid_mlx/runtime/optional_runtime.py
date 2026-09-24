# SPDX-License-Identifier: Apache-2.0
"""Typed failures for optional serving runtimes."""

from __future__ import annotations

import importlib.util
import os
import select
import subprocess
import sys
from pathlib import Path
from typing import Literal

OptionalExtra = Literal["vision", "video", "audio", "image"]
OptionalRuntimeStatus = Literal["absent", "broken", "incompatible"]

_EXTRA_INSTALL_SIZE_MB: dict[OptionalExtra, int] = {
    "vision": 322,
    "audio": 600,
}
_INSTALL_PROMPT_TIMEOUT_SECONDS = 30.0


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


def _running_in_desktop_sidecar() -> bool:
    """Return whether Desktop owns this interpreter and its dependencies."""
    executable_paths = (Path(sys.executable), Path(sys.executable).resolve())
    if any(
        part.endswith(".app")
        for executable in executable_paths
        for part in executable.parts
    ):
        return True

    from rapid_mlx.telemetry.consent_runtime import is_desktop_sidecar

    return is_desktop_sidecar()


def _prompt_to_install(extra: OptionalExtra) -> bool:
    """Wait at most 30 seconds for an explicit interactive opt-in."""
    size_mb = _EXTRA_INSTALL_SIZE_MB.get(extra)
    size = f" (~{size_mb} MB)" if size_mb is not None else ""
    print(
        f"Install rapid-mlx[{extra}] now?{size} [y/N] ",
        end="",
        file=sys.stderr,
        flush=True,
    )
    readable, _, _ = select.select([sys.stdin], [], [], _INSTALL_PROMPT_TIMEOUT_SECONDS)
    if not readable:
        print(file=sys.stderr)
        return False
    return sys.stdin.readline().strip().lower() in {"y", "yes"}


def _install_optional_extra(exc: OptionalRuntimeMissing) -> None:
    """Install one pinned extra, then replace this process on success."""
    from rapid_mlx import __version__

    install_argv = [
        sys.executable,
        "-m",
        "pip",
        "install",
        f"rapid-mlx[{exc.extra}]=={__version__}",
    ]
    completed = subprocess.run(install_argv, check=False)
    if completed.returncode != 0:
        print(
            f"Install failed (rc {completed.returncode}); "
            f"run {exc.install_hint} manually.",
            file=sys.stderr,
        )
        return

    # ``exec`` skips atexit callbacks. Drain the two already-enqueued failure
    # terminals before replacing this process so the new PID can be measured
    # as a separate attempted -> ready journey.
    from rapid_mlx.telemetry import posthog_sender

    posthog_sender.get_sender().flush(2.0)
    restart_argv = [sys.executable, *sys.orig_argv[1:]]
    os.execv(sys.executable, restart_argv)


def handle_optional_runtime_missing(
    exc: OptionalRuntimeMissing,
    *,
    alias_or_path=None,
    engine=None,
    auto_selected: bool = False,
    assume_yes: bool = False,
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
    if (
        exc.status == "absent"
        and not _running_in_desktop_sidecar()
        and importlib.util.find_spec("pip") is not None
        and (
            assume_yes
            or (
                sys.stdin.isatty()
                and sys.stderr.isatty()
                and _prompt_to_install(exc.extra)
            )
        )
    ):
        _install_optional_extra(exc)
    raise SystemExit(2)
