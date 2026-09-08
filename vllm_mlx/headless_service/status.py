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

import contextlib
import io
import ipaddress
import json
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

from .common import (
    DEFAULT_DOMAIN,
    DEFAULT_LABEL,
    log_dir_for,
)
from .install import _plist_path, _port_busy


def _launchctl_probe(
    label: str, *, timeout_s: float = 10
) -> tuple[str | None, str | None]:
    """Return launchctl output plus a distinct execution-error description."""
    try:
        result = subprocess.run(
            ["/bin/launchctl", "print", f"{DEFAULT_DOMAIN}/{label}"],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if result.returncode == 0:
        return result.stdout, None
    stderr = str(getattr(result, "stderr", "") or "").strip()
    lowered = stderr.lower()
    if "could not find service" in lowered or "service not found" in lowered:
        return None, None
    detail = stderr or "no diagnostic output"
    return None, f"launchctl exited {result.returncode}: {detail}"


def _launchctl_print(label: str, *, timeout_s: float = 10) -> str | None:
    """Raw ``launchctl print <domain>/<label>`` output, or None if the job is
    not registered."""
    return _launchctl_probe(label, timeout_s=timeout_s)[0]


def _parse_legacy_serve(argv: object) -> tuple[str, str, str, int]:
    """Strictly parse a legacy direct-serve definition with the real CLI parser."""
    if not isinstance(argv, list) or any(not isinstance(item, str) for item in argv):
        raise ValueError("ProgramArguments must be a string array")
    if (
        len(argv) < 3
        or not Path(argv[0]).is_absolute()
        or not argv[0].endswith("rapid-mlx")
    ):
        raise ValueError("unrecognized legacy service executable")
    from vllm_mlx.cli import build_parser

    try:
        with (
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            parsed = build_parser().parse_args(argv[1:])
    except SystemExit as exc:
        raise ValueError("invalid legacy serve arguments") from exc
    if getattr(parsed, "command", None) != "serve":
        raise ValueError("legacy definition is not a serve command")
    return argv[0], parsed.model, parsed.host, parsed.port


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


def _endpoint_health(
    host: str,
    port: int,
    *,
    timeout_s: float = 2.0,
    shared_deadline: bool = False,
) -> tuple[bool | None, bool | None]:
    """Tri-state ``(live, ready)`` for ``/livez`` and ``/readyz`` over HTTP.

    ``live`` = the process is alive (``/livez`` returns 200). ``ready`` = the
    endpoint can accept work (``/readyz`` returns 200 AND ``"ready": true``).
    A lazy service can therefore be ready in ``standby`` without resident
    weights; install/apply/upgrade separately start with eager loading and
    require model-resident readiness before restoring the requested lazy policy.
    Best-effort via a raw socket GET —
    no external HTTP client dependency.
    """
    shared_probe_deadline = time.monotonic() + timeout_s

    def probe_deadline() -> float:
        return (
            shared_probe_deadline if shared_deadline else time.monotonic() + timeout_s
        )

    def remaining(deadline: float) -> float:
        return max(0.001, deadline - time.monotonic())

    def http_status(status_line: bytes) -> int | None:
        match = re.fullmatch(
            rb"HTTP/[0-9]+\.[0-9]+ ([0-9]{3})(?: [^\r\n]*)?", status_line
        )
        return int(match.group(1)) if match else None

    def chunked_message_complete(body: bytes) -> bool:
        """Return true once chunk data and the trailer section are complete."""
        remaining_body = body
        while True:
            raw_size, separator, remaining_body = remaining_body.partition(b"\r\n")
            if not separator:
                return False
            try:
                size = int(raw_size.split(b";", 1)[0], 16)
            except ValueError:
                return False
            if size == 0:
                return remaining_body == b"\r\n" or b"\r\n\r\n" in remaining_body
            if (
                len(remaining_body) < size + 2
                or remaining_body[size : size + 2] != b"\r\n"
            ):
                return False
            remaining_body = remaining_body[size + 2 :]

    def _probe_live() -> bool | None:
        deadline = probe_deadline()
        try:
            with socket.create_connection(
                (host, port), timeout=remaining(deadline)
            ) as sock:
                sock.settimeout(remaining(deadline))
                sock.sendall(
                    b"GET /livez HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n"
                )
                status_data = bytearray()
                while len(status_data) < 8192 and b"\r\n\r\n" not in status_data:
                    if time.monotonic() >= deadline:
                        return None
                    sock.settimeout(remaining(deadline))
                    chunk = sock.recv(8192 - len(status_data))
                    if not chunk:
                        break
                    status_data.extend(chunk)
                status_line = bytes(status_data).split(b"\r\n", 1)[0]
                status = http_status(status_line)
                if b"\r\n\r\n" not in status_data or status is None:
                    return None
                return status == 200
        except OSError:
            return None

    def _probe_ready() -> bool | None:
        deadline = probe_deadline()
        try:
            with socket.create_connection(
                (host, port), timeout=remaining(deadline)
            ) as sock:
                sock.settimeout(remaining(deadline))
                sock.sendall(
                    b"GET /readyz HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n"
                )
                data = bytearray()
                content_length: int | None = None
                transfer_codings: list[bytes] = []
                headers_parsed = False
                while len(data) < 64 * 1024:
                    if time.monotonic() >= deadline:
                        return None
                    sock.settimeout(remaining(deadline))
                    chunk = sock.recv(min(4096, 64 * 1024 - len(data)))
                    if not chunk:
                        break
                    data.extend(chunk)
                    head, separator, body = bytes(data).partition(b"\r\n\r\n")
                    if separator and not headers_parsed:
                        headers_parsed = True
                        for header in head.split(b"\r\n")[1:]:
                            name, colon, value = header.partition(b":")
                            header_name = name.strip().lower()
                            if colon and header_name == b"content-length":
                                content_length = int(value.strip())
                            elif colon and header_name == b"transfer-encoding":
                                transfer_codings.extend(
                                    coding.strip().lower()
                                    for coding in value.split(b",")
                                    if coding.strip()
                                )
                    chunked = (
                        bool(transfer_codings) and transfer_codings[-1] == b"chunked"
                    )
                    if (
                        not chunked
                        and content_length is not None
                        and len(body) >= content_length
                    ):
                        break
                    if separator and chunked and chunked_message_complete(body):
                        break
        except (OSError, ValueError):
            return None
        head, separator, body = bytes(data).partition(b"\r\n\r\n")
        status = http_status(head.split(b"\r\n", 1)[0])
        if not separator or status is None:
            return None
        if status != 200:
            return False
        chunked = bool(transfer_codings) and transfer_codings[-1] == b"chunked"
        if chunked:
            decoded = bytearray()
            terminal_chunk_seen = False
            try:
                while body:
                    raw_size, separator, body = body.partition(b"\r\n")
                    if not separator:
                        return None
                    size = int(raw_size.split(b";", 1)[0], 16)
                    if size == 0:
                        if body != b"\r\n":
                            trailers, trailer_end, remainder = body.partition(
                                b"\r\n\r\n"
                            )
                            if (
                                not trailer_end
                                or remainder
                                or any(
                                    not name.strip() or not colon
                                    for line in trailers.split(b"\r\n")
                                    for name, colon, _value in [line.partition(b":")]
                                )
                            ):
                                return None
                        terminal_chunk_seen = True
                        break
                    if len(body) < size + 2 or body[size : size + 2] != b"\r\n":
                        return None
                    decoded.extend(body[:size])
                    body = body[size + 2 :]
                if not terminal_chunk_seen:
                    return None
                body = bytes(decoded)
            except ValueError:
                return None
        elif content_length is not None:
            if len(body) < content_length:
                return None
            body = body[:content_length]
        try:
            payload = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError, RecursionError):
            return None
        if not isinstance(payload, dict) or "ready" not in payload:
            return None
        return payload.get("ready") is True

    from .install import _probe_host

    probe_host = _probe_host(host)
    if shared_deadline:
        if probe_host.lower() == "localhost":
            probe_host = "127.0.0.1"
        try:
            ipaddress.ip_address(probe_host.strip("[]"))
        except ValueError:
            return None, None
    if probe_host != host:
        host = probe_host
    return _probe_live(), _probe_ready()


def _listener_covers_host(
    listener: str, configured_host: str, *, resolve_hostnames: bool = False
) -> bool:
    listener = listener.strip().removeprefix("[").removesuffix("]")
    configured = configured_host.strip().removeprefix("[").removesuffix("]")
    if listener == "*":
        return True
    if configured.lower() == "localhost":
        return listener in {"127.0.0.1", "::1"}
    try:
        return ipaddress.ip_address(listener) == ipaddress.ip_address(configured)
    except ValueError:
        if not resolve_hostnames:
            return False
    try:
        addresses = {
            entry[4][0]
            for entry in socket.getaddrinfo(configured, None, type=socket.SOCK_STREAM)
        }
    except OSError:
        return False
    return listener in addresses


def _pid_listens_on_port(
    pid: int,
    host: str,
    port: int,
    *,
    timeout_s: float = 5.0,
    resolve_hostnames: bool = False,
) -> bool | None:
    """Prove that *pid* owns the configured TCP listener; None means unverified."""
    lsof = Path("/usr/sbin/lsof")
    if not lsof.is_file():
        return None
    try:
        result = subprocess.run(
            [
                str(lsof),
                "-nP",
                "-a",
                "-p",
                str(pid),
                f"-iTCP:{port}",
                "-sTCP:LISTEN",
                "-Fpn",
            ],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode == 0:
        lines = result.stdout.splitlines()
        if f"p{pid}" not in lines:
            return False
        for line in lines:
            if not line.startswith("n"):
                continue
            endpoint = line[1:].removesuffix(" (LISTEN)")
            listener, separator, listener_port = endpoint.rpartition(":")
            if (
                separator
                and listener_port == str(port)
                and _listener_covers_host(
                    listener, host, resolve_hostnames=resolve_hostnames
                )
            ):
                return True
        return False
    stderr = str(getattr(result, "stderr", "") or "").strip()
    if result.returncode == 1 and not result.stdout.strip() and not stderr:
        return False
    return None


def collect_status(
    *,
    label: str = DEFAULT_LABEL,
    user: str | None = None,
    host: str = "127.0.0.1",
    port: int = 8000,
    probe_timeout_s: float | None = None,
) -> dict:
    """Aggregate full service status into a plain dict (JSON-serializable)."""
    deadline = (
        time.monotonic() + probe_timeout_s if probe_timeout_s is not None else None
    )

    def remaining(default: float) -> float:
        return default if deadline is None else max(0.001, deadline - time.monotonic())

    print_out, launchctl_error = _launchctl_probe(
        label,
        timeout_s=10 if probe_timeout_s is None else remaining(probe_timeout_s),
    )
    registered = print_out is not None
    pid = _parse_pid(print_out)
    last_exit = _parse_last_exit(print_out)
    launchd_state = _parse_launchd_field(print_out, "state")
    raw_runs = _parse_launchd_field(print_out, "runs")
    runs = int(raw_runs) if raw_runs and raw_runs.isdigit() else None
    plist = _read_installed_plist(label)

    model = port_declared = host_declared = executable = declared_user = None
    config_file = config_sha256 = config_error = None
    config_valid = False
    endpoint_configured = False
    pending_config = False
    credential_configured: bool | None = False
    if plist:
        declared_user = plist.get("UserName")
        argv = plist.get("ProgramArguments") or []
        # argv shape: [<bin>, "serve", <model>, ...]
        try:
            legacy_executable, legacy_model, legacy_host, legacy_port = (
                _parse_legacy_serve(argv)
            )
            config_valid = True
            endpoint_configured = True
            executable, model = legacy_executable, legacy_model
            host_declared, port_declared = legacy_host, legacy_port
        except ValueError as exc:
            config_error = str(exc)

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
            config_valid = False
            config_file = str(identity[2])
            pending_config = pending_config_path(identity[1], label).is_file()
            try:
                effective = load_config(identity[2])
                executable = effective.executable
                model = effective.model
                host_declared = effective.host
                port_declared = effective.port
                config_sha256 = config_digest(effective)
                config_valid = True
                endpoint_configured = True
                config_error = None
                credential_configured = (
                    private_file_present(Path(effective.credential_file))
                    if effective.credential_file
                    else False
                )
            except Exception as exc:
                config_error = str(exc)

    owner = None
    if pid:
        try:
            out = subprocess.run(
                ["/bin/ps", "-o", "user=", "-p", str(pid)],
                capture_output=True,
                text=True,
                timeout=remaining(5),
            )
            owner = out.stdout.strip() or None
        except (subprocess.SubprocessError, OSError):
            owner = None

    # Probe ownership before network I/O so a slow endpoint cannot consume the
    # shared Doctor deadline and make a healthy owner look unverifiable.
    effective_host = host_declared or host
    effective_port = int(port_declared) if port_declared else port
    endpoint_attributable = False
    if pid and config_valid and endpoint_configured:
        endpoint_attributable = (
            _pid_listens_on_port(
                pid,
                effective_host,
                effective_port,
                timeout_s=remaining(5),
                resolve_hostnames=probe_timeout_s is None,
            )
            is True
        )
    if probe_timeout_s is not None and effective_host.lower() != "localhost":
        try:
            ipaddress.ip_address(effective_host.strip("[]"))
        except ValueError:
            # Persisted configs accept only localhost/numeric loopback binds.
            # A legacy hostname cannot be resolved inside this synchronous
            # deadline without risking an unbounded libc DNS call.
            endpoint_attributable = False
    if endpoint_attributable:
        live, ready = (
            _endpoint_health(effective_host, effective_port)
            if probe_timeout_s is None
            else _endpoint_health(
                effective_host,
                effective_port,
                timeout_s=remaining(probe_timeout_s),
                shared_deadline=True,
            )
        )
    else:
        live, ready = None, None

    effective_user = user or (declared_user if isinstance(declared_user, str) else None)
    log_dir = log_dir_for(effective_user) if effective_user else None
    return {
        "label": label,
        "domain": DEFAULT_DOMAIN,
        "registered": registered,
        "launchctl_error": launchctl_error,
        "pid": pid,
        "owner": owner,
        "declared_user": declared_user,
        "executable": executable,
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
        "config_valid": config_valid,
        "endpoint_configured": endpoint_configured,
        "endpoint_attributable": endpoint_attributable,
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
        f"  health:                livez={'unknown' if s['livez'] is None else 'ok' if s['livez'] else 'down'} "
        f"readyz={'unknown' if s['readyz'] is None else 'ok' if s['readyz'] else 'down'} "
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
