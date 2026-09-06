# SPDX-License-Identifier: Apache-2.0
"""``rapid-mlx service status`` — actionable diagnostics for the daemon.

Aggregates four independent sources of truth — the launchd registration,
the live process, the installed plist, and the API endpoint — so a problem
in any layer is surfaced rather than masked:

* is the job registered in the system domain?
* is a live PID running (and as whom)?
* what model/port does the installed plist declare?
* does the endpoint respond to /livez and /readyz?

``--json`` emits a machine-readable object; the default is a compact
human table mirroring the smoke-test vocabulary (registered / running /
owner / model / port / healthy / log paths / last launchd exit).
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

from .common import (
    DEFAULT_DOMAIN,
    DEFAULT_LABEL,
    log_dir_for,
)
from .install import _plist_path, _port_busy


def _launchctl_print(label: str) -> str | None:
    """Raw ``launchctl print <domain>/<label>`` output, or None if the job is
    not registered."""
    try:
        result = subprocess.run(
            ["launchctl", "print", f"{DEFAULT_DOMAIN}/{label}"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout


def _parse_pid(print_out: str | None) -> int | None:
    """Extract the job PID from ``launchctl print`` output (None if no live
    process). Mirrors the smoke script's awk parse."""
    if not print_out:
        return None
    for line in print_out.splitlines():
        if line.strip().startswith("pid ="):
            digits = "".join(ch for ch in line.split("=", 1)[1] if ch.isdigit())
            if digits:
                return int(digits)
    return None


def _parse_last_exit(print_out: str | None) -> int | None:
    """Last exit status from ``launchctl print`` (None when absent)."""
    if not print_out:
        return None
    for line in print_out.splitlines():
        if "last exit code" in line.lower():
            match = re.search(r"-?\d+", line.split("=", 1)[1])
            if match:
                return int(match.group())
    return None


def _parse_launchd_field(print_out: str | None, field: str) -> str | None:
    if not print_out:
        return None
    prefix = f"{field.lower()} ="
    for line in print_out.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith(prefix):
            return stripped.split("=", 1)[1].strip() or None
    return None


def _read_installed_plist(label: str) -> dict | None:
    """Parse the installed plist (None if not present)."""
    path = _plist_path(label)
    if not path.is_file():
        return None
    from .plist import parse_plist

    try:
        return parse_plist(path.read_bytes())
    except Exception:
        return None


def _endpoint_health(host: str, port: int) -> tuple[bool, bool]:
    """``(live, ready)`` for ``/livez`` and ``/readyz`` over HTTP.

    ``live`` = the process is alive (``/livez`` returns 200 — it never means
    "model ready"). ``ready`` = the model is loaded (``/readyz`` returns 200
    AND ``"ready": true``), sharing the corrected ``install._readyz_ready``
    so status, install, and restart all agree on "healthy". Best-effort via
    a raw socket GET — no external HTTP client dependency.
    """
    import socket

    def _probe_live() -> bool:
        try:
            with socket.create_connection((host, port), timeout=2.0) as sock:
                sock.settimeout(2.0)
                sock.sendall(
                    b"GET /livez HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n"
                )
                return b"200" in sock.recv(4096).split(b"\r\n", 1)[0]
        except OSError:
            return False

    from .install import _probe_host, _readyz_ready

    probe_host = _probe_host(host)
    if probe_host != host:
        host = probe_host
    return _probe_live(), _readyz_ready(host, port)


def collect_status(
    *,
    label: str = DEFAULT_LABEL,
    user: str | None = None,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> dict:
    """Aggregate full service status into a plain dict (JSON-serializable)."""
    print_out = _launchctl_print(label)
    registered = print_out is not None
    pid = _parse_pid(print_out)
    last_exit = _parse_last_exit(print_out)
    launchd_state = _parse_launchd_field(print_out, "state")
    raw_runs = _parse_launchd_field(print_out, "runs")
    runs = int(raw_runs) if raw_runs and raw_runs.isdigit() else None
    plist = _read_installed_plist(label)

    model = port_declared = host_declared = None
    config_file = config_sha256 = config_error = None
    pending_config = False
    credential_configured: bool | None = False
    if plist:
        argv = plist.get("ProgramArguments") or []
        # argv shape: [<bin>, "serve", <model>, ...]
        if len(argv) >= 3 and argv[0].endswith("rapid-mlx") and argv[1] == "serve":
            model = argv[2] if len(argv) > 2 else None
            for i, tok in enumerate(argv[:-1]):
                if tok == "--port" and i + 1 < len(argv):
                    port_declared = argv[i + 1]
                if tok == "--host" and i + 1 < len(argv):
                    host_declared = argv[i + 1]

        # New definitions use a stable config-backed launcher. Keep the argv
        # parser above for installations created by the first service release.
        from .config import (
            config_digest,
            load_config,
            pending_config_path,
            private_file_present,
        )
        from .definition import installed_identity

        identity = installed_identity(label)
        if identity is not None:
            config_file = str(identity[2])
            try:
                effective = load_config(identity[2])
                model = effective.model
                host_declared = effective.host
                port_declared = effective.port
                config_sha256 = config_digest(effective)
                pending_config = pending_config_path(identity[1], label).is_file()
                credential_configured = (
                    private_file_present(Path(effective.credential_file))
                    if effective.credential_file
                    else False
                )
            except Exception as exc:
                config_error = str(exc)

    # Probe the bind the plist declares (fall back to CLI defaults) — probing
    # the CLI default while the service actually listens elsewhere would report
    # a false "down".
    effective_host = host_declared or host
    effective_port = int(port_declared) if port_declared else port
    live, ready = _endpoint_health(effective_host, effective_port)

    owner = None
    if pid:
        try:
            out = subprocess.run(
                ["ps", "-o", "user=", "-p", str(pid)],
                capture_output=True,
                text=True,
                timeout=5,
            )
            owner = out.stdout.strip() or None
        except (subprocess.SubprocessError, OSError):
            owner = None

    declared_user = plist.get("UserName") if plist else None
    effective_user = user or (declared_user if isinstance(declared_user, str) else None)
    log_dir = log_dir_for(effective_user) if effective_user else None
    return {
        "label": label,
        "domain": DEFAULT_DOMAIN,
        "registered": registered,
        "pid": pid,
        "owner": owner,
        "last_exit": last_exit,
        "launchd_state": launchd_state,
        "runs": runs,
        "crash_loop_suspected": bool(
            registered and pid is None and last_exit not in (None, 0)
        ),
        "model": model,
        "host": effective_host,
        "port": effective_port,
        "livez": live,
        "readyz": ready,
        "port_open": _port_busy(effective_host, effective_port),
        "plist": str(_plist_path(label)),
        "log_dir": str(log_dir) if log_dir else None,
        "plist_present": Path(_plist_path(label)).is_file(),
        "config_file": config_file,
        "config_sha256": config_sha256,
        "config_error": config_error,
        "pending_config": pending_config,
        "credential_configured": credential_configured,
    }


def _render_human(s: dict) -> str:
    lines = [
        f"service {s['label']} ({s['domain']} domain)",
        f"  launcher registration: {'installed' if s['plist_present'] else 'MISSING'}",
        f"  launchd state:         {'registered' if s['registered'] else 'not registered'}",
    ]
    if s["pid"]:
        owner = f" (as {s['owner']})" if s["owner"] else ""
        lines.append(f"  pid:                   {s['pid']}{owner}")
    else:
        lines.append("  pid:                   (no live process)")
    if s["last_exit"] is not None:
        lines.append(f"  last launchd exit:     {s['last_exit']}")
    if s.get("launchd_state") or s.get("runs") is not None:
        lines.append(
            f"  launchd details:       state={s.get('launchd_state') or 'unknown'} "
            f"runs={s.get('runs') if s.get('runs') is not None else 'unknown'}"
        )
    if s.get("crash_loop_suspected"):
        lines.append(
            "  warning:               repeated startup failure suspected; inspect logs"
        )
    if s["model"]:
        lines.append(f"  model:                 {s['model']}")
    if s.get("config_file"):
        digest = (s.get("config_sha256") or "invalid")[:12]
        staged = " (PENDING changes)" if s.get("pending_config") else ""
        lines.append(f"  config:                {s['config_file']} [{digest}]{staged}")
    if s.get("config_error"):
        lines.append(f"  config error:          {s['config_error']}")
    if s.get("config_file"):
        credential_state = s.get("credential_configured")
        auth_label = (
            "credential file"
            if credential_state is True
            else "disabled"
            if credential_state is False
            else "unknown (run status with sudo)"
        )
        lines.append("  authentication:        " + auth_label)
    lines.append(f"  endpoint:              http://{s['host']}:{s['port']}")
    lines.append(
        f"  health:                livez={'ok' if s['livez'] else 'down'} "
        f"readyz={'ok' if s['readyz'] else 'down'} "
        f"port={'open' if s['port_open'] else 'closed'}"
    )
    lines.append(f"  plist:                 {s['plist']}")
    if s["log_dir"]:
        lines.append(f"  logs:                  {s['log_dir']}/server.stdout.log")
        lines.append(f"                         {s['log_dir']}/server.stderr.log")
    if not s["registered"]:
        lines.append(
            "  hint: not registered — `rapid-mlx service install --dry-run` "
            "shows the install plan."
        )
    return "\n".join(lines)


def status_command(args) -> int:
    label = getattr(args, "label", None) or DEFAULT_LABEL
    user = getattr(args, "service_user", None)
    host = getattr(args, "host", None) or "127.0.0.1"
    port = getattr(args, "port", None) or 8000
    data = collect_status(label=label, user=user, host=host, port=port)
    if getattr(args, "json", False):
        print(json.dumps(data, indent=2))
    else:
        print(_render_human(data))
    # Non-zero exit when the service is not actually up (actionable for
    # scripts that gate on health).
    if not data["registered"] or not data["pid"] or not data["readyz"]:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(status_command(sys.argv))
