# SPDX-License-Identifier: Apache-2.0
"""Typed failures for optional serving runtimes."""

from __future__ import annotations

import importlib.util
import os
import select
import shlex
import subprocess
import sys
import time
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Protocol, cast

OptionalExtra = Literal["vision", "video", "audio", "image"]
OptionalRuntimeStatus = Literal["absent", "broken", "incompatible"]
ExtraRecovery = Literal[
    "accepted",
    "declined",
    "no_answer",
    "interrupted",
    "non_interactive",
    "assume_yes",
    "no_installer",
    "managed_runtime",
    "broken_runtime",
]

_EXTRA_INSTALL_SIZE_MB: dict[OptionalExtra, int] = {
    "vision": 322,
    "audio": 600,
}
_INSTALL_PROMPT_TIMEOUT_SECONDS = 30.0


class _PromptInput(Protocol):
    def fileno(self) -> int: ...


class PromptResult(str, Enum):
    """Closed result of the bounded optional-runtime consent prompt."""

    ACCEPTED = "accepted"
    DECLINED = "declined"
    NO_ANSWER = "no_answer"
    INTERRUPTED = "interrupted"


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


def optional_extra_repair_command(
    extra: str,
    *,
    version: str | None = None,
    include_paths: bool = True,
    status: OptionalRuntimeStatus = "absent",
) -> str:
    """Return the install-method-aware command for one pinned optional extra.

    The detector is shared with ``rapid-mlx upgrade`` so tool-managed installs
    are never mistaken for ordinary virtual environments.  ``include_paths``
    keeps HTTP-visible guidance free of local filesystem paths.
    """
    from rapid_mlx import __version__
    from rapid_mlx._version_check import detect_install_method

    pinned = f"rapid-mlx[{extra}]=={version or __version__}"
    try:
        install_info = detect_install_method()
        method = install_info.method
        global_pipx = method == "unknown" and getattr(
            install_info, "upgrade_command", ""
        ).startswith("sudo pipx ")
    except Exception:  # noqa: BLE001 - repair guidance must never mask the error
        method = "unknown"
        global_pipx = False
    if method == "uv":
        return f"uv tool install --force {shlex.quote(pinned)}"
    if method == "pipx":
        return f"pipx install --force {shlex.quote(pinned)}"
    if global_pipx:
        return f"sudo pipx install --global --force {shlex.quote(pinned)}"
    if method == "brew":
        return (
            "The Homebrew build cannot add Python optional extras in place. "
            "Switch to an isolated tool install with:\n"
            f"    brew uninstall rapid-mlx && uv tool install {shlex.quote(pinned)}"
        )
    python = shlex.quote(sys.executable) if include_paths else "python"
    reinstall = "--upgrade --force-reinstall " if status == "broken" else ""
    return f"{python} -m pip install {reinstall}{shlex.quote(pinned)}"


def optional_extra_install_hint(
    extra: str,
    *,
    version: str | None = None,
    include_paths: bool = True,
    status: OptionalRuntimeStatus = "absent",
) -> str:
    """Return consistent human-facing repair guidance for an optional extra."""
    return "Install the optional runtime with:\n    " + optional_extra_repair_command(
        extra, version=version, include_paths=include_paths, status=status
    )


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
) -> PromptResult:
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
    try:
        response = (
            _read_windows_prompt_response(timeout)
            if sys.platform == "win32"
            else _read_posix_prompt_response(sys.stdin, timeout)
        )
    except KeyboardInterrupt:
        print(file=sys.stderr)
        return PromptResult.INTERRUPTED
    if response is None:
        print(file=sys.stderr)
        return PromptResult.NO_ANSWER
    if response.strip().lower() in {"y", "yes"}:
        return PromptResult.ACCEPTED
    return PromptResult.DECLINED


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
    managed_runtime = _running_in_desktop_sidecar()
    install_detection_failed = False
    try:
        from rapid_mlx._version_check import detect_install_method

        install_info = detect_install_method()
        install_method = install_info.method
        globally_managed_pipx = install_method == "unknown" and getattr(
            install_info, "upgrade_command", ""
        ).startswith("sudo pipx ")
    except Exception:  # noqa: BLE001 - the original typed failure still wins
        install_detection_failed = True
        install_method = "unknown"
        globally_managed_pipx = False
    can_install = bool(
        exc.status == "absent"
        and not managed_runtime
        and not install_detection_failed
        and not globally_managed_pipx
        and install_method in {"pip", "install_sh", "unknown"}
        and importlib.util.find_spec("pip") is not None
    )
    is_interactive = (
        can_install and not assume_yes and _is_tty(sys.stdin) and _is_tty(sys.stderr)
    )
    if can_install and not assume_yes and not is_interactive:
        print(
            "Non-interactive session: rerun with --yes to install "
            f"rapid-mlx[{exc.extra}] automatically, or install it manually "
            "with the command above.",
            file=sys.stderr,
        )
    print(
        format_startup_failure_marker(exc.marker_reason, extra=exc.extra),
        file=sys.stderr,
    )
    from rapid_mlx.telemetry.server_start import failed

    failed("preflight")
    accepted = False
    if managed_runtime:
        extra_recovery: ExtraRecovery = "managed_runtime"
    elif exc.status != "absent":
        extra_recovery = "broken_runtime"
    elif not can_install:
        extra_recovery = "no_installer"
    elif assume_yes:
        accepted = True
        extra_recovery = "assume_yes"
    elif not is_interactive:
        extra_recovery = "non_interactive"
    else:
        try:
            prompt_result = _prompt_to_install(exc.extra)
        except KeyboardInterrupt:
            print(file=sys.stderr)
            prompt_result = PromptResult.INTERRUPTED
        accepted = prompt_result is PromptResult.ACCEPTED
        extra_recovery = cast(ExtraRecovery, prompt_result.value)
    from rapid_mlx.telemetry.model_events import emit_model_serve_failed

    emit_model_serve_failed(
        exc,
        engine=engine,
        alias_or_path=alias_or_path,
        auto_selected=auto_selected,
        extra_recovery=extra_recovery,
    )
    if accepted:
        _install_optional_extra(exc)
    if extra_recovery == "interrupted":
        raise KeyboardInterrupt
    raise SystemExit(2)
