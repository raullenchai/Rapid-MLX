# SPDX-License-Identifier: Apache-2.0
"""``rapid-mlx service restart`` — kickstart the daemon and wait for health.

``launchctl kickstart -k`` stops and restarts a loaded job regardless of
KeepAlive (which would otherwise fight a manual restart). Afterward we poll
the endpoint with the same readiness window the install path and the smoke
script use, so "restart" means "ready," not merely "respawning."

A service that is registered but currently down is ``kickstart -k``-ed the
same way; a service that is not registered is a hard error telling the user
to install first. ``--dry-run`` prints the command only.
"""

from __future__ import annotations

import subprocess
import sys

from .common import DEFAULT_DOMAIN, DEFAULT_LABEL
from .install import _wait_ready


def _declared_bind(label: str) -> tuple[str, int] | None:
    """The ``(host, port)`` the installed plist declares, if readable."""
    from .install import _plist_path
    from .plist import parse_plist

    plist_path = _plist_path(label)
    if not plist_path.is_file():
        return None
    try:
        config = parse_plist(plist_path.read_bytes())
    except Exception:
        return None
    argv = config.get("ProgramArguments") or []
    host, port = None, None
    for i, tok in enumerate(argv[:-1]):
        if tok == "--host" and i + 1 < len(argv):
            host = argv[i + 1]
        if tok == "--port" and i + 1 < len(argv):
            port = argv[i + 1]
    if port is None:
        return None
    return (host or "127.0.0.1"), int(port)


def _kickstart_status(label: str) -> int:
    """``launchctl print`` exit code: 0 if the job is loaded in the domain."""
    try:
        result = subprocess.run(
            ["launchctl", "print", f"{DEFAULT_DOMAIN}/{label}"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (subprocess.SubprocessError, OSError):
        return 1
    return result.returncode


def restart_command(args) -> int:
    label = getattr(args, "label", None) or DEFAULT_LABEL
    host = getattr(args, "host", None) or "127.0.0.1"
    port = getattr(args, "port", None) or 8000
    dry_run = bool(getattr(args, "dry_run", False))

    # Resolve the actual bind from the installed plist so we wait on the right
    # port (the service may have been installed on a non-default --port).
    declared = _declared_bind(label)
    if declared:
        declared_host, declared_port = declared
        host, port = declared_host, declared_port

    if _kickstart_status(label) != 0:
        print(
            f"error: service {label} is not registered in the {DEFAULT_DOMAIN} "
            "domain. Install it first with `rapid-mlx service install` "
            "(or `--dry-run` to rehearse).",
            file=sys.stderr,
        )
        return 1

    cmd = ["launchctl", "kickstart", "-k", f"{DEFAULT_DOMAIN}/{label}"]
    if dry_run:
        print(f"[DRY-RUN] would run: {' '.join(cmd)}")
        return 0

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except (subprocess.CalledProcessError, subprocess.SubprocessError) as exc:
        print(f"error: kickstart failed: {exc}", file=sys.stderr)
        return 1

    if not _wait_ready(host, port):
        print(
            f"error: service {label} did not become ready on {host}:{port} "
            "within 120s after restart. Check `rapid-mlx service logs`.",
            file=sys.stderr,
        )
        return 1

    print(f"restarted {label}; healthy on {host}:{port}.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(restart_command(sys.argv))
