# SPDX-License-Identifier: Apache-2.0
"""Local serve process discovery and startup progress, without engine imports."""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from functools import lru_cache


@dataclass
class Server:
    pid: int
    port: int | None
    model: str
    alias: str | None = None


@lru_cache(maxsize=1)
def _serve_actions():
    from rapid_mlx.cli_parser import build_parser

    parser = build_parser()
    return parser._subparsers._group_actions[0].choices["serve"]._option_string_actions


def scan_servers() -> list[Server]:
    """Prefer the actual listener over argv: serve may auto-select its port."""
    import psutil

    from rapid_mlx.cli_parser import _port_arg

    servers = []
    try:
        processes = list(psutil.process_iter(["pid", "cmdline"]))
    except (psutil.AccessDenied, PermissionError):
        return []
    for proc in processes:
        try:
            cmd = proc.info["cmdline"] or []
            if not cmd or os.path.basename(cmd[0]) == "caffeinate":
                continue
            if "serve" not in cmd or not any(
                "rapid-mlx" in c or "rapid_mlx" in c for c in cmd
            ):
                continue
            tokens = cmd[cmd.index("serve") + 1 :]
            port = None
            alias = None
            model = "(unknown)"
            # Reuse the parser's option arities rather than treating option
            # values (host, parsers, etc.) as a positional model.
            actions = _serve_actions()
            i = 0
            while i < len(tokens):
                token = tokens[i]
                key, sep, value = token.partition("=")
                action = actions.get(key)
                if action is not None and action.nargs != 0:
                    if not sep and i + 1 < len(tokens):
                        i += 1
                        value = tokens[i]
                    if key == "--port":
                        # ``--port 0`` (OS-assigned) or a malformed value is
                        # not a usable argv port; the listener lookup below
                        # still finds the real one, so never drop the server.
                        try:
                            port = _port_arg(value)
                        except (argparse.ArgumentTypeError, ValueError):
                            port = None
                    elif key == "--served-model-name":
                        alias = value
                elif not token.startswith("-") and model == "(unknown)":
                    model = token
                i += 1
            try:
                listeners = {
                    c.laddr.port
                    for c in proc.net_connections(kind="inet")
                    if c.status == psutil.CONN_LISTEN and c.laddr
                }
                if port in listeners:
                    pass
                elif len(listeners) == 1:
                    port = listeners.pop()
            except (psutil.AccessDenied, psutil.NoSuchProcess, PermissionError):
                pass
            servers.append(Server(proc.pid, port, model, alias))
        except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError):
            continue
    return servers


def pid_alive(pid: int) -> bool:
    import psutil

    try:
        proc = psutil.Process(pid)
        return proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False
    except (psutil.AccessDenied, PermissionError):
        return True


@dataclass
class Endpoint:
    base_url: str
    server: Server | None = None
    follow: bool = False
    model: str | None = None

    def refresh(self) -> None:
        # Explicit URLs/ports are authoritative, even if an unbound serve
        # process eventually chooses another port.
        if self.server is None and self.follow:
            resolved = resolve_endpoint(self.model, report_ignored=False)
            self.base_url, self.server = resolved.base_url, resolved.server
        if self.server is not None and self.server.port is None:
            for server in scan_servers():
                if server.pid == self.server.pid and server.port is not None:
                    self.server = server
                    if self.follow:
                        self.base_url = f"http://localhost:{server.port}/v1"
                        announce(server)
                    else:
                        from rapid_mlx.connect import _parse_base_url

                        if _parse_base_url(self.base_url)[1] != server.port:
                            # An explicit endpoint must never follow a process
                            # that has now chosen a different listener.
                            self.server = None
                    break


def announce(server: Server) -> None:
    port = f":{server.port}" if server.port else "an unbound port (8000-8009)"
    print(f"  Using the Rapid-MLX server on {port} (pid {server.pid}).")


def _same_model(left: str, right: str) -> bool:
    from rapid_mlx.model_aliases import resolve_model

    def resolved(value: str) -> str:
        try:
            return resolve_model(value)
        except Exception:
            # Alias resolution failures must not prevent matching raw names.
            return value

    def same_identity(a: str, b: str) -> bool:
        if a == b:
            return True
        a_local = a.startswith(("/", "./", "../", "~")) or os.path.exists(a)
        b_local = b.startswith(("/", "./", "../", "~")) or os.path.exists(b)
        if a_local and b_local:
            return os.path.realpath(os.path.expanduser(a)) == os.path.realpath(
                os.path.expanduser(b)
            )
        if a_local or b_local:
            return os.path.basename(a.rstrip("/")) == os.path.basename(b.rstrip("/"))
        return False

    return same_identity(left, right) or same_identity(resolved(left), resolved(right))


def _default_endpoint_ready() -> tuple[bool, bool]:
    """Return (healthy, answered) separately from process discovery."""
    try:
        with urllib.request.urlopen(
            "http://localhost:8000/health", timeout=0.25
        ) as response:
            return response.status == 200, True
    except urllib.error.HTTPError:
        # Only an owning serve process makes an unhealthy listener a loading
        # server. A foreign process must not hide a server on another port.
        return False, True
    except (OSError, ValueError):
        return False, False


def resolve_endpoint(
    model: str | None = None, *, report_ignored: bool = True
) -> Endpoint:
    default = "http://localhost:8000/v1"
    healthy, answered = _default_endpoint_ready()
    if healthy:
        return Endpoint(default)
    servers = scan_servers()
    default_server = next((s for s in servers if s.port == 8000), None)
    if answered and default_server is not None:
        return Endpoint(default, default_server, follow=True, model=model)
    candidates = (
        [
            s
            for s in servers
            if any(_same_model(model, name) for name in (s.model, s.alias) if name)
        ]
        if model
        else servers
    )
    if model and not candidates and report_ignored:
        for server in servers[:5]:
            port = str(server.port) if server.port else "unknown"
            print(
                f"  Note: rapid-mlx serve on :{port} (pid {server.pid}) serves "
                f"{server.alias or server.model}, not {model}; "
                f"pass --port {port} to use it."
            )
        if len(servers) > 5:
            print(f"  Note: and {len(servers) - 5} more unrelated servers.")
    if len(candidates) > 1:
        rows = "\n".join(
            f"  pid {s.pid}, port {s.port or 'unknown (8000-8009)'}, model {s.alias or s.model}"
            for s in candidates
        )
        raise RuntimeError(
            f"No ready server at {default}; multiple local servers:\n{rows}\n"
            "  Pass --port to select the server."
        )
    if candidates:
        server = candidates[0]
        announce(server)
        return Endpoint(
            f"http://localhost:{server.port}/v1" if server.port else default,
            server,
            follow=True,
            model=model,
        )
    return Endpoint(default, follow=True, model=model)


def server_for_url(base_url: str) -> Server | None:
    from rapid_mlx.connect import _parse_base_url

    from .server_hint import is_local_base_url

    if not is_local_base_url(base_url):
        return None
    _, port = _parse_base_url(base_url)
    return next(
        (
            s
            for s in scan_servers()
            if s.port == port or (s.port is None and 8000 <= port <= 8009)
        ),
        None,
    )


class Progress:
    def __init__(self) -> None:
        self.phase = None
        self.last = float("-inf")

    def show(self, base_url: str, now: float) -> None:
        root = base_url.rstrip("/").removesuffix("/v1")
        payload = {}
        try:
            try:
                response = urllib.request.urlopen(f"{root}/health/ready", timeout=0.25)
            except urllib.error.HTTPError as exc:
                response = exc
            with response:
                payload = json.loads(response.read())
        except (OSError, ValueError):
            pass
        if not isinstance(payload, dict):
            payload = {}
        phase = payload.get("phase", "downloading or preparing")
        if phase == self.phase and now - self.last < 10:
            return
        self.phase, self.last = phase, now
        detail = phase
        progress = payload.get("progress", {})
        if isinstance(progress, dict):
            done, total = progress.get("done"), progress.get("total")
            if (
                isinstance(done, (int, float))
                and isinstance(total, (int, float))
                and total > 0
            ):
                detail = f"{phase} {min(100, int(100 * done / total))}%"
        print(
            f"  Waiting for the Rapid-MLX server at {root} to finish "
            f"downloading/loading the model ({detail})…",
            flush=True,
        )
