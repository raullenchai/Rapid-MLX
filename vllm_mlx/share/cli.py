# SPDX-License-Identifier: Apache-2.0
"""``rapid-mlx share <alias>`` — start a serve + open a public tunnel.

Orchestration shape:

  1. Validate alias (cheap fail-fast before booting the engine).
  2. Pick a free local port + generate a fresh 24-byte bearer key.
  3. Spawn ``rapid-mlx serve`` in a child process pointing at that port.
  4. Wait for /healthz to come back ready.
  5. Ask the rapidmlx.com control plane for a session token + subdomain.
  6. Start frpc with a single-proxy config bridging localhost → subdomain.
  7. Print the security banner + URL + key, then block until Ctrl-C.
  8. On exit, kill frpc and the serve process in that order.

State lives in ``~/.cache/rapid-mlx/share/`` — pids, logs, and the frpc
config. Key + URL are NOT persisted: each invocation issues a new key
(per user's "new key every share" preference) and a new session.
"""

from __future__ import annotations

import argparse
import os
import secrets
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import frpc_manager, session, warning


def _pick_port(preferred: int) -> int:
    """Return ``preferred`` if free, else an OS-assigned port. We bind+release
    rather than just checking — TOCTOU windows on busy systems are real.
    """
    for candidate in (preferred, 0):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(("127.0.0.1", candidate))
                return s.getsockname()[1]
        except OSError:
            continue
    raise RuntimeError("no free port available for share")


def _wait_for_healthz(
    port: int, serve_proc: subprocess.Popen[bytes] | None = None
) -> bool:
    """Poll /healthz until the child serve reports ready or exits.

    No fixed timeout: a cold first-time pull of a 70B model legitimately
    takes 10+ minutes, and silently SIGTERM-ing a healthy download is
    one of the worst UX failure modes we can ship. Instead we watch the
    child process — if it exits without ever serving /healthz we give
    up, otherwise we wait as long as it takes. Caller can Ctrl-C any
    time to abort.
    """
    url = f"http://127.0.0.1:{port}/healthz"
    while True:
        if serve_proc is not None and serve_proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionError):
            pass
        time.sleep(1)


def _spawn_serve(
    *,
    alias: str,
    port: int,
    api_key: str,
    log_path: Path,
    extra_args: list[str],
) -> subprocess.Popen[bytes]:
    # Use sys.executable + ``-m`` instead of the ``rapid-mlx`` script so
    # the share command works inside editable installs and CI environments
    # where the entrypoint script may not be on PATH.
    # ``--host 127.0.0.1`` is load-bearing here: without it serve binds
    # 0.0.0.0 and the bearer-key-gated API becomes reachable from anyone
    # on the user's LAN, not just through the frp tunnel as intended.
    cmd = [
        sys.executable,
        "-m",
        "vllm_mlx.cli",
        "serve",
        alias,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--api-key",
        api_key,
        "--log-level",
        "INFO",
        *extra_args,
    ]
    log_fp = log_path.open("ab", buffering=0)
    return subprocess.Popen(
        cmd,
        stdout=log_fp,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        # New process group so Ctrl-C in our terminal doesn't deliver
        # SIGINT to serve before we've had a chance to tear down frpc.
        start_new_session=True,
    )


def _state_dir() -> Path:
    d = Path.home() / ".cache" / "rapid-mlx" / "share"
    d.mkdir(parents=True, exist_ok=True)
    d.chmod(0o700)
    return d


def share_command(args: argparse.Namespace) -> None:
    alias: str = args.model
    extra_serve_args: list[str] = []
    if args.no_thinking:
        extra_serve_args.append("--no-thinking")
    if args.cors_origins:
        extra_serve_args.extend(["--cors-origins", args.cors_origins])

    api_key = secrets.token_hex(24)
    # Port parsing is lazy on purpose: validating RAPID_MLX_SHARE_PORT at
    # parser-build time crashes ``rapid-mlx models`` (and every other
    # unrelated subcommand) when the env var is set to garbage.
    raw_port = os.environ.get("RAPID_MLX_SHARE_PORT") if args.port is None else None
    try:
        preferred_port = int(raw_port) if raw_port is not None else (args.port or 8765)
    except ValueError:
        print(
            f"RAPID_MLX_SHARE_PORT must be an integer (got {raw_port!r})",
            file=sys.stderr,
        )
        sys.exit(2)
    if not (1 <= preferred_port <= 65535):
        print(
            f"share port {preferred_port} is outside the valid range (1-65535)",
            file=sys.stderr,
        )
        sys.exit(2)
    port = _pick_port(preferred_port)
    state_dir = _state_dir()
    serve_log = state_dir / "serve.log"
    tunnel_log = state_dir / "tunnel.log"

    print(f"Starting rapid-mlx serve ({alias} on :{port})…", file=sys.stderr)
    serve_proc = _spawn_serve(
        alias=alias,
        port=port,
        api_key=api_key,
        log_path=serve_log,
        extra_args=extra_serve_args,
    )

    try:
        if not _wait_for_healthz(port, serve_proc):
            print(
                f"serve exited before becoming ready — see {serve_log}",
                file=sys.stderr,
            )
            sys.exit(1)

        print("Requesting share session from rapidmlx.com…", file=sys.stderr)
        try:
            sess = session.request(model=alias)
        except RuntimeError as exc:
            print(f"share: {exc}", file=sys.stderr)
            sys.exit(1)

        print(
            f"Starting frpc → {sess.subdomain}.{sess.frps_host.split('.', 1)[-1]}",
            file=sys.stderr,
        )
        with tempfile.NamedTemporaryFile(
            "w", dir=state_dir, suffix=".toml", delete=False
        ) as f:
            f.write(
                frpc_manager.render_config(
                    server_addr=sess.frps_host,
                    server_port=sess.frps_port,
                    auth_token=sess.token,
                    subdomain=sess.subdomain,
                    local_port=port,
                )
            )
            config_path = Path(f.name)
        config_path.chmod(0o600)
        frpc_proc = frpc_manager.spawn(config_path, tunnel_log)

        # frpc takes ~1-3s to establish. After the settling window, if
        # the process has already exited we know the relay rejected the
        # token / the config is malformed / frps is unreachable — bail
        # instead of printing a URL pointing at a dead tunnel.
        time.sleep(3)
        if frpc_proc.poll() is not None:
            print(
                f"frpc exited before tunnel was ready — see {tunnel_log}",
                file=sys.stderr,
            )
            sys.exit(1)

        print(warning.render(sess.public_url, api_key, alias))

        # Block on the serve process; if it exits, share's done.
        serve_proc.wait()
    except KeyboardInterrupt:
        print("\nStopping share…", file=sys.stderr)
    finally:
        for proc_name, proc in (("frpc", locals().get("frpc_proc")), ("serve", serve_proc)):
            if proc is None or proc.poll() is not None:
                continue
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            except OSError:
                pass
        # Wipe the frpc config — it contains the relay token.
        cp = locals().get("config_path")
        if isinstance(cp, Path):
            cp.unlink(missing_ok=True)


def register(subparsers: argparse._SubParsersAction) -> None:
    """Wire up the ``share`` subcommand onto the top-level CLI parser."""
    p = subparsers.add_parser(
        "share",
        help="Expose a local model behind a public URL via rapidmlx.com",
        description=(
            "Start rapid-mlx serve and open a public Cloudflare-fronted "
            "URL on rapidmlx.com so you can use the model from a different "
            "device — or share it with a friend. Press Ctrl-C to stop."
        ),
    )
    p.add_argument(
        "model",
        help="Alias to serve (same names as `rapid-mlx serve`, e.g. qwen3.5-4b)",
    )
    p.add_argument(
        "--port",
        type=int,
        default=None,
        help=(
            "Local port to bind serve to (default: 8765, or "
            "$RAPID_MLX_SHARE_PORT if set)"
        ),
    )
    p.add_argument(
        "--no-thinking",
        action="store_true",
        default=True,
        help=(
            "Pass --no-thinking to serve (default: on; chat UIs see "
            "content immediately instead of waiting on a <think> prelude)"
        ),
    )
    p.add_argument(
        "--cors-origins",
        default="*",
        help=(
            "Pass --cors-origins to serve (default: '*' so browser chat "
            "UIs like Open WebUI work without extra config)"
        ),
    )
