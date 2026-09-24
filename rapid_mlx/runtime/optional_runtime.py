# SPDX-License-Identifier: Apache-2.0
"""Typed failures for optional serving runtimes."""

from __future__ import annotations

import importlib.util
import os
import select
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Literal, Protocol

OptionalExtra = Literal["vision", "video", "audio", "image"]
OptionalRuntimeStatus = Literal["absent", "broken", "incompatible"]

_EXTRA_INSTALL_SIZE_MB: dict[OptionalExtra, int] = {
    "vision": 322,
    "audio": 600,
}
_INSTALL_PROMPT_TIMEOUT_SECONDS = 30.0


class _PromptInput(Protocol):
    def fileno(self) -> int: ...


_assume_yes = False


def set_assume_yes(flag: bool) -> None:
    """Set whether this process should accept optional-runtime installs."""
    global _assume_yes
    _assume_yes = bool(flag)


def _reset_assume_yes_for_tests() -> None:
    """Restore the default optional-runtime prompt policy for test isolation."""
    set_assume_yes(False)


def assume_yes() -> bool:
    """Return the process-wide optional-runtime prompt policy."""
    return _assume_yes


def format_startup_failure_marker(
    reason: str, *, extra: OptionalExtra | None = None
) -> str:
    """Format the closed desktop startup-failure wire marker."""
    marker = f"RAPID-MLX-STARTUP-FAILURE: {reason}"
    if extra is not None:
        marker += f" extra={extra}"
    return marker


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


def _prompt_to_install(
    extra: OptionalExtra,
    *,
    timeout_seconds: float | None = None,
) -> bool:
    """Wait at most 30 seconds for an explicit interactive opt-in."""
    size_mb = _EXTRA_INSTALL_SIZE_MB.get(extra)
    size = f" (~{size_mb} MB)" if size_mb is not None else ""
    print(
        f"Install rapid-mlx[{extra}] now?{size} [y/N] ",
        end="",
        file=sys.stderr,
        flush=True,
    )
    timeout = (
        _INSTALL_PROMPT_TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds
    )
    response = (
        _read_windows_prompt_response(timeout)
        if sys.platform == "win32"
        else _read_posix_prompt_response(sys.stdin, timeout)
    )
    if response is None:
        print(file=sys.stderr)
        return False
    return response.strip().lower() in {"y", "yes"}


def _read_posix_prompt_response(
    stdin: _PromptInput, timeout_seconds: float | None = None
) -> str | None:
    """Read a newline-terminated response without exceeding the deadline."""
    try:
        stdin_fd = stdin.fileno()
        timeout = (
            _INSTALL_PROMPT_TIMEOUT_SECONDS
            if timeout_seconds is None
            else timeout_seconds
        )
        deadline = time.monotonic() + timeout
        response = bytearray()
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            ready, _, _ = select.select([stdin_fd], [], [], remaining)
            if not ready:
                return None
            if time.monotonic() >= deadline:
                return None
            chunk = os.read(stdin_fd, 256)
            if not chunk:
                return None
            response.extend(chunk)
            terminators = [
                index
                for marker in (b"\n", b"\r")
                if (index := response.find(marker)) >= 0
            ]
            if terminators:
                return bytes(response[: min(terminators)]).decode(
                    "utf-8", errors="replace"
                )
    except Exception:
        return None


def _read_windows_prompt_response(
    timeout_seconds: float | None = None,
) -> str | None:
    """Poll the Windows console until Enter or the prompt deadline."""
    try:
        import msvcrt

        console: Any = msvcrt
        timeout = (
            _INSTALL_PROMPT_TIMEOUT_SECONDS
            if timeout_seconds is None
            else timeout_seconds
        )
        deadline = time.monotonic() + timeout
        response: list[str] = []
        while time.monotonic() < deadline:
            if console.kbhit():
                character = console.getwche()
                if character in {"\r", "\n"}:
                    return "".join(response)
                if character == "\b":
                    if response:
                        response.pop()
                else:
                    response.append(character)
            else:
                time.sleep(0.05)
        return None
    except Exception:
        return None


def _is_tty(stream: object | None) -> bool:
    """Treat detached streams and stream stand-ins as non-interactive."""
    try:
        isatty = getattr(stream, "isatty", None)
        return bool(isatty and isatty())
    except Exception:
        return False


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
            f"Install failed (rc {completed.returncode}).\n{exc.install_hint}",
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
        format_startup_failure_marker(exc.marker_reason, extra=exc.extra),
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
                _is_tty(sys.stdin)
                and _is_tty(sys.stderr)
                and _prompt_to_install(exc.extra)
            )
        )
    ):
        _install_optional_extra(exc)
    raise SystemExit(2)
