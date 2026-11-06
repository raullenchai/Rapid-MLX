# SPDX-License-Identifier: Apache-2.0
"""Local serve process discovery and startup progress, without engine imports."""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from .chat_models import NonChatModelError, non_chat_kind, preferred_model


@dataclass
class Server:
    pid: int
    port: int | None
    model: str
    alias: str | None = None
    # True once ``port`` came from the process's own LISTEN socket rather
    # than its argv, i.e. it is no longer a guess.
    listening: bool = field(default=False, compare=False)
    # False when the process's sockets could not be read (AccessDenied):
    # a missing port is then unknown, not "still unbound".
    inspected: bool = field(default=True, compare=False)


def _loopback_reachable(host: str | None) -> bool:
    """True when a listener bound to *host* accepts ``localhost`` connections.

    Setup only ever configures a ``localhost`` URL, so a server bound to a
    specific LAN address is not one it can verify or wait on.
    """
    # ``localhost`` resolves to 127.0.0.1 / ::1; other 127.x addresses are
    # distinct listeners that a localhost URL never reaches.
    return host in (None, "", "localhost", "0.0.0.0", "::", "127.0.0.1", "::1")


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
            host = None
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
                    elif key == "--host":
                        host = value
                elif not token.startswith("-") and model == "(unknown)":
                    model = token
                i += 1
            if not _loopback_reachable(host):
                continue
            listening = False
            inspected = True
            try:
                bound = [
                    c.laddr
                    for c in proc.net_connections(kind="inet")
                    if c.status == psutil.CONN_LISTEN and c.laddr
                ]
                listeners = {a.port for a in bound if _loopback_reachable(a.ip)}
                if bound and not listeners:
                    # Listening, but only where localhost cannot reach it: not
                    # an unbound server that may still start on 8000-8009.
                    continue
                if port in listeners:
                    listening = True
                elif len(listeners) == 1:
                    port = listeners.pop()
                    listening = True
            except (psutil.AccessDenied, psutil.NoSuchProcess, PermissionError):
                inspected = False
            servers.append(Server(proc.pid, port, model, alias, listening, inspected))
        except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError):
            continue
    return servers


def startup_loading(base_url: str) -> bool:
    """True when ``/health`` answers 503: the serve startup probe is loading."""
    root = base_url.rstrip("/").removesuffix("/v1")
    try:
        with urllib.request.urlopen(f"{root}/health", timeout=2):
            return False
    except urllib.error.HTTPError as exc:
        return exc.code == 503
    except (OSError, ValueError):
        return False


def pid_alive(pid: int) -> bool:
    import psutil

    try:
        proc = psutil.Process(pid)
        return bool(proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE)
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
    skipped_auth: list[Server] = field(default_factory=list)
    # Hint history only: a replacement server may reuse a rejected port.
    skipped_non_chat: dict[int, NonChatModelError] = field(default_factory=dict)
    agent: str = "codex"
    # Verification can outlast discovery probes; remember its result per pid.
    verified_non_chat: dict[int, int] = field(default_factory=dict)
    _last_scan: float = field(default=float("-inf"), repr=False, compare=False)

    def refresh(self) -> None:
        # Explicit URLs/ports are authoritative, even if an unbound serve
        # process eventually chooses another port.
        if self.server is None and self.follow:
            resolved = resolve_endpoint(
                self.model,
                agent=self.agent,
                report_ignored=False,
                skipped_auth=self.skipped_auth,
                skipped_non_chat=self.skipped_non_chat,
                verified_non_chat=self.verified_non_chat,
            )
            self.base_url, self.server = resolved.base_url, resolved.server
            self.skipped_auth = resolved.skipped_auth
            self.skipped_non_chat = resolved.skipped_non_chat
        # An argv port is only a guess until the process listens: keep
        # re-reading the selected pid's real listener (throttled) so a server
        # that binds somewhere else is followed instead of waited on forever.
        now = time.monotonic()
        if (
            self.server is None
            or self.server.listening
            or (self.server.port is not None and now - self._last_scan < 2.0)
        ):
            return
        self._last_scan = now
        for server in scan_servers():
            if server.pid != self.server.pid:
                continue
            if server.port is None:
                break
            changed = server.port != self.server.port
            self.server = server
            if not changed:
                break
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


def requires_missing_key(port: int) -> bool:
    """Only an observed auth rejection excludes an automatic candidate.

    Health alone is insufficient: the server can expose it publicly while
    protecting models. Refused connections and startup errors remain eligible.
    Never read credentials from another process or Desktop's storage.
    """
    if os.environ.get("RAPID_MLX_API_KEY"):
        return False
    for path in ("/health", "/v1/models"):
        try:
            with urllib.request.urlopen(f"http://localhost:{port}{path}", timeout=0.25):
                pass
        except urllib.error.HTTPError as exc:
            exc.close()
            if exc.code in (401, 403):
                return True
        except (OSError, ValueError):
            pass
    return False


def desktop_owned(server: Server) -> bool:
    """Identify Desktop only when its record matches both process and port."""
    try:
        record = json.loads(
            (
                Path.home() / "Library/Application Support/Rapid/owned-server.json"
            ).read_text()
        )
        return (
            isinstance(record, dict)
            and record.get("pid") == server.pid
            and record.get("port") == server.port
        )
    except (OSError, ValueError):
        return False


def skipped_auth_hint(server: Server, agent: str) -> str:
    api_key_env = "RAPID_MLX_API_KEY"
    owner = (
        "belongs to Rapid-MLX Desktop"
        if desktop_owned(server)
        else "requires an API key"
    )
    return (
        f"  Server on :{server.port} {owner}; copy the API key from Rapid-MLX "
        "Desktop's Agent > Connection details > API key, and rerun with "
        f"{api_key_env}=<key> rapid-mlx agents {agent} --setup --port {server.port}; "
        "or start a CLI server with rapid-mlx serve."
    )


def non_chat_server(port: int, model: str | None = None) -> NonChatModelError | None:
    """Exclude only observed non-chat metadata; loading servers stay eligible."""
    from rapid_mlx.http_auth import rapid_mlx_auth_headers

    url = f"http://localhost:{port}/v1"
    # Keep the plain request shape for unauthenticated discovery probes.
    headers = rapid_mlx_auth_headers()
    request = (
        urllib.request.Request(f"{url}/models", headers=headers)
        if headers
        else f"{url}/models"
    )
    try:
        with urllib.request.urlopen(request, timeout=0.25) as response:
            payload = json.loads(response.read())
    except (OSError, ValueError):
        return None
    models = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(models, list):
        return None
    entries = [
        m
        for m in models
        if isinstance(m, dict) and isinstance(m.get("id"), str) and m["id"]
    ]
    matching = [m for m in entries if model and _same_model(model, m["id"])]
    # Keep aliases visible when the entire server is non-chat.
    if matching and any(not non_chat_kind(m) for m in entries):
        entries = matching
    if entries and all(non_chat_kind(m) for m in entries):
        return NonChatModelError(url, preferred_model(entries))
    return None


def resolve_endpoint(
    model: str | None = None,
    *,
    agent: str = "codex",
    report_ignored: bool = True,
    skipped_auth: list[Server] | None = None,
    skipped_non_chat: dict[int, NonChatModelError] | None = None,
    verified_non_chat: dict[int, int] | None = None,
) -> Endpoint:
    default = "http://localhost:8000/v1"
    healthy, answered = _default_endpoint_ready()
    skipped = (
        list(skipped_auth or []) if not os.environ.get("RAPID_MLX_API_KEY") else []
    )
    rejected: dict[int, bool] = {s.port: True for s in skipped if s.port is not None}

    def missing_key(port: int) -> bool:
        if port not in rejected:
            rejected[port] = requires_missing_key(port)
        return rejected[port]

    non_chat = dict(skipped_non_chat or {})
    checked: dict[int, bool] = {}
    verified = verified_non_chat or {}
    scanned = scan_servers() if verified else None
    owners = {s.port: s.pid for s in scanned or [] if s.listening}

    def incompatible(port: int) -> bool:
        if port not in checked:
            error = non_chat_server(port, model)
            checked[port] = error is not None or (
                port in verified and owners.get(port) == verified[port]
            )
            if error is not None:
                if port not in non_chat:
                    print(f"  {error.skip_hint}")
                non_chat[port] = error
        return checked[port]

    if healthy and not missing_key(8000) and not incompatible(8000):
        return Endpoint(
            default,
            follow=True,
            model=model,
            agent=agent,
            skipped_auth=skipped,
            skipped_non_chat=non_chat,
        )
    # A process whose sockets cannot be read and whose argv names no port
    # cannot be shown to own any endpoint; waiting on it could never end.
    servers = [
        s
        for s in (scanned if scanned is not None else scan_servers())
        if s.port is not None or s.inspected
    ]

    # Carry skipped ports through follow-mode refreshes so each hint prints once,
    # even when a later scan reports a different pid/model for the same port.
    def skip_auth(server: Server) -> None:
        if not any(s.port == server.port for s in skipped):
            skipped.append(server)
            print(skipped_auth_hint(server, agent))

    for server in servers:
        if server.port is not None and missing_key(server.port):
            skip_auth(server)
    if (
        (healthy or answered)
        and missing_key(8000)
        and not any(s.port == 8000 for s in skipped)
    ):
        skip_auth(Server(0, 8000, "(unknown)"))
    servers = [
        s
        for s in servers
        if s.port is None or (not missing_key(s.port) and not incompatible(s.port))
    ]
    candidates = (
        [
            s
            for s in servers
            if any(_same_model(model, name) for name in (s.model, s.alias) if name)
        ]
        if model
        else servers
    )
    default_server = next((s for s in candidates if s.port == 8000), None)
    if answered and default_server is not None:
        return Endpoint(
            default,
            default_server,
            follow=True,
            model=model,
            agent=agent,
            skipped_auth=skipped,
            skipped_non_chat=non_chat,
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
            agent=agent,
            skipped_auth=skipped,
            skipped_non_chat=non_chat,
        )
    return Endpoint(
        default,
        follow=True,
        model=model,
        agent=agent,
        skipped_auth=skipped,
        skipped_non_chat=non_chat,
    )


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
            if s.port == port
            or (s.port is None and s.inspected and 8000 <= port <= 8009)
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
