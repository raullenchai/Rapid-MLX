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
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import frpc_manager, session, warning

# Pulled out so the routing-shape audit (tests/test_no_out_of_band_routing.py)
# sees one clean RAPID_MLX_* string literal that lives in the
# ALLOWED_RAPID_MLX_ENV_VARS allowlist — inlining the name into an
# f-string error message would yield "RAPID_MLX_SHARE_PORT must be an…"
# which is NOT on the allowlist and tripwires the audit.
_PORT_ENV_VAR = "RAPID_MLX_SHARE_PORT"


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


def _resolve_served_model_name(port: int, api_key: str) -> str | None:
    """Read the model id rapid-mlx serve is exposing via /v1/models.

    The CLI accepts a short alias (``qwen3.5-4b``) but the OpenAI
    endpoint only recognises the full HF model id
    (``mlx-community/Qwen3.5-4B-MLX-4bit``). Without this lookup the
    curl example we paste into the security banner fails on first
    try — a confusing UX for the user (and their friend).

    ``api_key`` is required because we spawn serve with ``--api-key``
    so /v1/models is bearer-gated; without the header the probe 401s
    and silently falls back to the alias.
    """
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=2) as r:
            import json

            payload = json.load(r)
        data = payload.get("data") or []
        if data and isinstance(data[0], dict):
            return data[0].get("id")
    except (urllib.error.URLError, ConnectionError, ValueError):
        return None
    return None


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


def _verify_auth_gate(port: int, api_key: str) -> bool:
    """Auth-gated proof that the process answering on ``port`` is OURS.

    /healthz is unauthenticated by design (load-balancers need it). On a
    busy host another local process can race us to the same port and
    answer /healthz while having nothing to do with rapid-mlx — and the
    tunnel would happily forward to it. We require an authenticated
    /v1/models 200 with the freshly-generated bearer before requesting a
    tunnel: only our serve has that key, so a 200 here means we're
    pointing the tunnel at our own process.
    """
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=3) as r:  # noqa: S310
            return r.status == 200
    except (urllib.error.URLError, ConnectionError, ValueError):
        return False


def _wait_for_public_url(public_url: str, timeout: float = 15.0) -> bool:
    """Confirm the tunnel is live by probing the public URL's /healthz.

    Without this, frpc can stay up (process alive, TCP connected to frps)
    while the proxy registration silently fails — the banner prints with
    a URL that 502s. Bounded so we don't hang forever if Cloudflare /
    frps are degraded.
    """
    url = f"{public_url.rstrip('/')}/healthz"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as r:  # noqa: S310
                if r.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionError):
            pass
        time.sleep(1)
    return False


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
    #
    # The bearer key is passed via ``RAPID_MLX_API_KEY`` env var, NOT
    # argv. ``ps`` exposes argv to every local user — landing the key
    # there leaks the secret that gates the public tunnel. The env var
    # is only visible to the owning process (and root). (DeepSeek
    # BLOCKING on PR #504 round 3.)
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
        "--log-level",
        "INFO",
        *extra_args,
    ]
    env = dict(os.environ)
    env["RAPID_MLX_API_KEY"] = api_key
    log_fp = log_path.open("ab", buffering=0)
    # Tighten permissions: log files default to umask-derived modes
    # (often 644 = world-readable). If serve ever logs the key as part
    # of an error or debug line, world-read leaks it. 600 forces
    # owner-only. (DeepSeek round-3 NIT #3.)
    try:
        Path(log_fp.name).chmod(0o600)
    except OSError:
        pass
    return subprocess.Popen(
        cmd,
        stdout=log_fp,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        env=env,
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
    # ``args.thinking`` comes from BooleanOptionalAction so ``--thinking``
    # turns it on and ``--no-thinking`` (or the default) turns it off. We
    # forward ``--no-thinking`` to serve only when explicitly disabled —
    # serve's own default is on, so an explicit flag is needed.
    if not args.thinking:
        extra_serve_args.append("--no-thinking")
    if args.cors_origins:
        extra_serve_args.extend(["--cors-origins", args.cors_origins])

    api_key = secrets.token_hex(24)
    # Port parsing is lazy on purpose: validating RAPID_MLX_SHARE_PORT at
    # parser-build time crashes ``rapid-mlx models`` (and every other
    # unrelated subcommand) when the env var is set to garbage.
    raw_port = os.environ.get(_PORT_ENV_VAR) if args.port is None else None
    try:
        if raw_port is not None:
            preferred_port = int(raw_port)
        elif args.port is not None:
            # ``is not None`` (not truthy): an explicit ``--port 0`` is a
            # user error that should surface as exit-2, not get silently
            # rewritten to the 8765 default.
            preferred_port = args.port
        else:
            preferred_port = 8765
    except ValueError:
        print(
            f"{_PORT_ENV_VAR} must be an integer (got {raw_port!r})",
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

    # Convert SIGTERM into a KeyboardInterrupt so the existing finally
    # block runs cleanup. Without this, a supervisor (systemd, docker,
    # ``kill <pid>``) terminates the share parent and orphans the serve
    # + frpc children, leaking a public tunnel until the user notices.
    # ``original_sigterm`` is the handler we replace; we restore it on
    # function exit so future code in the same process (e.g.
    # command-chaining) sees the prior behavior instead of inheriting
    # our KeyboardInterrupt translator. (DeepSeek round-3 NIT #2.)
    def _term_handler(signum, frame):  # noqa: ARG001
        raise KeyboardInterrupt

    original_sigterm = signal.signal(signal.SIGTERM, _term_handler)

    serve_proc: subprocess.Popen[bytes] | None = None
    frpc_proc: subprocess.Popen[bytes] | None = None
    config_path: Path | None = None
    try:
        print(f"Starting rapid-mlx serve ({alias} on :{port})…", file=sys.stderr)
        serve_proc = _spawn_serve(
            alias=alias,
            port=port,
            api_key=api_key,
            log_path=serve_log,
            extra_args=extra_serve_args,
        )
        if not _wait_for_healthz(port, serve_proc):
            print(
                f"serve exited before becoming ready — see {serve_log}",
                file=sys.stderr,
            )
            sys.exit(1)
        # Auth-gated proof: even though /healthz returned 200, confirm the
        # process answering on this port is ours (it has our key). On a
        # busy host another local serve could win the race to bind the
        # same port we asked for, and tunneling THAT process would leak
        # someone else's model + their data. Bearer-gating /v1/models
        # eliminates this class of bug — no other process has our key.
        if not _verify_auth_gate(port, api_key):
            print(
                f"serve on :{port} did not answer authenticated /v1/models — "
                f"another process may be bound to the same port. Aborting "
                f"before opening a public tunnel.",
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
        # render_config validates the relay-provided strings against
        # strict allow-lists (see frpc_manager._validate_session_fields)
        # so a hostile/buggy control plane can't inject TOML keys here.
        try:
            rendered = frpc_manager.render_config(
                server_addr=sess.frps_host,
                server_port=sess.frps_port,
                auth_token=sess.token,
                subdomain=sess.subdomain,
                local_port=port,
            )
        except RuntimeError as exc:
            print(f"share: {exc}", file=sys.stderr)
            sys.exit(1)
        with tempfile.NamedTemporaryFile(
            "w", dir=state_dir, suffix=".toml", delete=False
        ) as f:
            f.write(rendered)
            config_path = Path(f.name)
        config_path.chmod(0o600)
        # ``spawn`` chains into ``ensure()``, which can raise on a fresh
        # checkout (download failure, sha256 mismatch, unsupported
        # platform/arch). Surface those as user-readable messages
        # instead of bare tracebacks. (DeepSeek round-4 BLOCKER #2.)
        try:
            frpc_proc = frpc_manager.spawn(config_path, tunnel_log)
        except (RuntimeError, urllib.error.URLError) as exc:
            print(f"share: failed to start frpc tunnel: {exc}", file=sys.stderr)
            sys.exit(1)

        # Liveness ladder: short settle, then poll the actual public URL.
        # frpc can stay up (process alive, TCP connected to frps) while
        # the proxy registration silently fails — printing a banner with
        # a URL that 502s is worse than failing here.
        time.sleep(1)
        if frpc_proc.poll() is not None:
            print(
                f"frpc exited before tunnel was ready — see {tunnel_log}",
                file=sys.stderr,
            )
            sys.exit(1)
        if not _wait_for_public_url(sess.public_url):
            print(
                f"tunnel did not respond at {sess.public_url} within 15s — "
                f"see {tunnel_log}",
                file=sys.stderr,
            )
            sys.exit(1)

        # rapid-mlx serve registers the model under its HF id, not the
        # short alias the user typed — so the curl example needs that
        # name to actually run. Falls back to the typed alias if the
        # /v1/models probe fails (the banner still prints).
        display_model = _resolve_served_model_name(port, api_key) or alias
        # ``flush=True`` is load-bearing: when stdout is a pipe
        # (``rapid-mlx share … | tee``), Python block-buffers and the
        # banner doesn't reach the terminal until the process exits.
        print(warning.render(sess.public_url, api_key, display_model), flush=True)

        # Block on the serve process; if it exits, share's done.
        serve_proc.wait()
    except KeyboardInterrupt:
        print("\nStopping share…", file=sys.stderr)
    finally:
        # DeepSeek round-2 NIT: if a second SIGTERM arrives mid-cleanup,
        # the installed handler raises KeyboardInterrupt again and we
        # leak the second child (serve). Ignore SIGTERM for the
        # duration of cleanup — supervisor "kill -9" can still force
        # us, that's fine.
        try:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
        except (ValueError, OSError):
            pass
        for _name, proc in (("frpc", frpc_proc), ("serve", serve_proc)):
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
        if isinstance(config_path, Path):
            config_path.unlink(missing_ok=True)
        # Restore whatever SIGTERM handler we replaced. Keeps share_command
        # idempotent within the same Python process.
        try:
            signal.signal(signal.SIGTERM, original_sigterm)
        except (ValueError, OSError, TypeError):
            pass


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
    # BooleanOptionalAction is the only way to get both ``--thinking``
    # and ``--no-thinking`` from a single declaration. The previous
    # ``store_true`` + ``default=True`` was unreachable — there's no
    # ``--no-no-thinking`` and the flag silently couldn't be disabled.
    p.add_argument(
        "--thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Forward thinking-mode behavior to serve. Default off "
            "(``--no-thinking``) so chat UIs see content immediately "
            "instead of waiting on a <think> prelude. Pass ``--thinking`` "
            "to keep upstream defaults."
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
