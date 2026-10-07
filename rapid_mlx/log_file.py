# SPDX-License-Identifier: Apache-2.0
"""``--log-file`` / ``RAPID_MLX_LOG_FILE``: where server output goes.

``serve`` points its own stdout and stderr (file descriptors 1 and 2) at the
target, so Python logging, ``print`` output, tracebacks and native-library
messages all follow it. Commands that start a server (``chat``, ``start``,
``share``) pass the target on through the environment and let the child
inherit their terminal instead of capturing its output in their own log file.
"""

from __future__ import annotations

import os
import sys

ENV_VAR = "RAPID_MLX_LOG_FILE"
FLAG = "--log-file"
STDOUT = "-"


class LogFileError(ValueError):
    """The configured log target cannot be used."""


def resolve(flag_value: str | None) -> tuple[str | None, str | None]:
    """Return ``(target, source)``; the flag wins over the environment."""
    if flag_value is not None:
        return flag_value, FLAG
    env_value = os.environ.get(ENV_VAR, "")
    if env_value:
        return env_value, ENV_VAR
    return None, None


def validate(target: str, source: str) -> str:
    """Return the target with paths made absolute, or raise LogFileError."""
    if target == STDOUT:
        return target
    if not target.strip():
        raise LogFileError(f"{source} must not be empty")
    path = os.path.abspath(os.path.expanduser(target))
    if os.path.isdir(path):
        raise LogFileError(f"{source} {target!r} is a directory")
    parent = os.path.dirname(path)
    if not os.path.isdir(parent):
        raise LogFileError(f"{source} {target!r}: directory {parent} does not exist")
    return path


def resolve_validated(flag_value: str | None) -> tuple[str | None, str | None]:
    """:func:`resolve` followed by :func:`validate`."""
    target, source = resolve(flag_value)
    if target is None or source is None:
        return None, None
    return validate(target, source), source


def describe(target: str) -> str:
    """Human-readable destination for status lines."""
    if target == STDOUT:
        return "stdout"
    if target == os.devnull:
        return "discarded (/dev/null)"
    return target


def apply(target: str, source: str = FLAG) -> None:
    """Point this process's stdout and stderr at ``target``.

    Raises LogFileError, with stdout and stderr untouched, when the target
    cannot be opened for appending.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    try:
        if target == STDOUT:
            os.dup2(1, 2)
            return
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    except OSError as exc:
        raise LogFileError(
            f"{source} {target!r} cannot be opened for writing: {exc.strerror}"
        ) from None
    try:
        os.dup2(fd, 1)
        os.dup2(fd, 2)
    finally:
        # A process started with stdout or stderr closed gets 1 or 2 back
        # from os.open; that descriptor is now the log target itself.
        if fd > 2:
            os.close(fd)
