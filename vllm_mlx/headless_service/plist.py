# SPDX-License-Identifier: Apache-2.0
"""Deterministic LaunchDaemon plist generation for ``rapid-mlx service``.

The generated plist is a strict, parameterised version of the documented
template ``examples/launchd/com.rapidmlx.server.plist``, hardened by the
same safety rules the manual runbook and ``tests/test_headless_service_
assets.py`` already enforce:

* binds to ``127.0.0.1`` loopback by default;
* keeps secrets OUT of ``ProgramArguments`` and ``EnvironmentVariables``
  (launchd can display configured env through ``launchctl print``, and
  argv is visible to other local processes);
* sets a mandatory ``HOME`` (the system domain does not inherit the GUI
  session's environment — Hugging Face would otherwise fail to find the
  service account's cache);
* uses unconditional ``KeepAlive`` + a ``ThrottleInterval`` so a crash is
  restarted but a broken config does not hot-spin faster than once/10s;
* sends stdout and stderr to two distinct files.

Generation is a pure function of ``(label, user, executable, model,
serve_args, host, port, home, log_dir)`` — no I/O, no launchd calls — so
tests can assert byte-determinism and the safety properties without ever
touching the system.
"""

from __future__ import annotations

import plistlib
from pathlib import Path

from .common import STDERR_LOG_NAME, STDOUT_LOG_NAME


def _prepend_exec(argv: list[str], executable: str | list[str]) -> list[str]:
    """Splice the executable (path, or path + invocation tokens) onto a argv.

    ``executable`` may be either a single absolute path (the common case: the
    symlink ``~/.local/bin/rapid-mlx``) or a list of argv tokens for a
    module invocation (``[<python>, "-m", "vllm_mlx"]``). Every element is a
    separate ProgramArguments entry — launchd/posix_spawn treats argv[0] as a
    literal executable path, so we must never collapse a ``-m`` invocation
    into a single space-joined string (which would not exist as a file).
    """
    if isinstance(executable, str):
        argv.insert(0, executable)
    else:
        argv[0:0] = list(executable)
    return argv


def serve_argv(
    executable: str | list[str],
    model: str,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    serve_args: tuple[str, ...] | None = None,
) -> list[str]:
    """The daemon's ``ProgramArguments``.

    The daemon boots through the ordinary ``rapid-mlx serve`` path — the
    engine is already daemon-aware for non-interactive invocations (no
    consent prompt when not on a TTY), so we get full serve-feature parity
    for free without forking a second server entrypoint.

    ``executable`` is either an absolute path or a ``[python, "-m",
    "vllm_mlx"]``-style list (see :func:`_prepend_exec`).
    ``serve_args`` are additional non-secret ``--flag value`` tokens (e.g.
    ``--max-num-seqs 4``) passed straight through after ``--host/--port``.
    """
    argv = ["serve", model, "--host", host, "--port", str(port)]
    if serve_args:
        argv.extend(serve_args)
    return _prepend_exec(argv, executable)


def build_plist_dict(
    *,
    label: str,
    user: str,
    executable: str,
    model: str,
    home: Path,
    log_dir: Path,
    host: str = "127.0.0.1",
    port: int = 8000,
    serve_args: tuple[str, ...] | None = None,
) -> dict:
    """Build the plist dict for a system LaunchDaemon.

    ``home`` is the service account's home dir (used for the mandatory
    ``HOME`` env and the PATH prefix). ``log_dir`` is the absolute daemon
    log directory. All paths are absolute — launchd does not expand ``$HOME``
    or ``~`` in plist values.
    """
    environment: dict[str, str] = {
        # Mandatory in the system launchd domain: without HOME, Hugging
        # Face cannot resolve the service account's model cache.
        "HOME": str(home),
        "PATH": f"{home / '.local/bin'}:/usr/bin:/bin:/usr/sbin:/sbin",
    }
    return {
        "Label": label,
        "UserName": user,
        "ProgramArguments": serve_argv(
            executable, model, host=host, port=port, serve_args=serve_args
        ),
        "WorkingDirectory": str(home),
        "EnvironmentVariables": environment,
        # Unconditional KeepAlive = continuously enabled appliance. Use
        # ``service restart`` (bootout/kickstart) before planned maintenance.
        "KeepAlive": True,
        # Prevent a broken config from re-spawning faster than once/10s.
        "ThrottleInterval": 10,
        "ExitTimeOut": 30,
        # plist ints are decimal; 23 decimal == 027 octal — masks group/other
        # write on files the daemon creates.
        "Umask": 0o27,
        "StandardOutPath": str(log_dir / STDOUT_LOG_NAME),
        "StandardErrorPath": str(log_dir / STDERR_LOG_NAME),
    }


def serialize_plist(config: dict) -> bytes:
    """Serialize a plist dict to deterministic XML bytes.

    ``plistlib.dumps`` with ``fmt=FMT_XML`` is deterministic for a given
    dict, so identical ``(label, user, executable, model, ...)`` inputs
    yield byte-identical plists — the property the "no surprise diff on
    reinstall" guarantee depends on.
    """
    return plistlib.dumps(config, fmt=plistlib.FMT_XML, sort_keys=True)


def parse_plist(data: bytes) -> dict:
    """Parse plist bytes back to a dict (used by status/uninstall to inspect
    an installed plist without launching it)."""
    parsed = plistlib.loads(data)
    assert isinstance(parsed, dict)
    return parsed
