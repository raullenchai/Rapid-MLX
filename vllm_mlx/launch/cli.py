# SPDX-License-Identifier: Apache-2.0
"""``rapid-mlx launch`` subcommand wiring.

This module exposes:

* :func:`register` — called from :mod:`vllm_mlx.cli` to wire the
  ``launch`` subparser onto the top-level argparse tree.
* :func:`launch_command` — argparse dispatch entry point.

The subcommand has three argv shapes:

* ``rapid-mlx launch list`` — print the supported clients + detection
  matrix. No state mutated.
* ``rapid-mlx launch <client>`` — patch the named client's config.
* ``rapid-mlx launch --all`` — patch every *detected* client.

All shapes accept the same set of orthogonal flags (``--model``,
``--server-url``, ``--start-server``, ``--port``, ``--dry-run``). The
positional ``<client>`` and ``--all`` are mutually exclusive — argparse
isn't aware of this because we accept either, but :func:`launch_command`
fails fast with a clear error.

``--start-server`` spawns ``rapid-mlx serve <model> --port <port>`` in
the background and writes the PID to ``~/.rapid-mlx/launch.pid`` so a
later ``kill $(cat ~/.rapid-mlx/launch.pid)`` shuts it down cleanly.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

from . import ADAPTERS

# Where we drop the PID of a ``--start-server`` subprocess. Pulled out
# so tests can monkeypatch it to a tmp_path and assert the file's
# contents without polluting the dev's real home dir.
PID_FILE = Path.home() / ".rapid-mlx" / "launch.pid"


def _print_list(*, as_json: bool = False) -> int:
    """Print the supported-clients + detection matrix.

    Output shape (one line per client):

        cline           detected
        claude-code     not detected

    Always returns 0 — listing is a read-only inspect command.
    """
    if as_json:
        from vllm_mlx.integrations import integration_targets_json

        print(json.dumps(integration_targets_json(), ensure_ascii=False))
        return 0
    width = max(len(name) for name in ADAPTERS) + 2
    print("Supported clients:")
    for name, adapter in ADAPTERS.items():
        status = "detected" if adapter.detect() else "not detected"
        print(f"  {name.ljust(width)}{status}")
    return 0


def _resolve_default_model() -> str:
    """Pick a default model alias when the user didn't pass ``--model``.

    Precedence:

    * ``RAPID_MLX_DEFAULT_MODEL`` env var (lets the operator pin one)
    * the built-in ``qwen3.5-4b-4bit`` (same default the chat REPL uses
      — a tiny, fast, well-MHI'd model that fits 24 GB Macs).

    A "last-served" file would be slightly nicer UX but adds a state
    surface we'd have to maintain across CLI versions; for the first
    cut this static default is sufficient (and matches what the README
    quickstart tells users to pull).
    """
    return os.environ.get("RAPID_MLX_DEFAULT_MODEL") or "qwen3.5-4b-4bit"


def _loopback_url_port(server_url: str) -> int | None:
    """Return a loopback URL's effective port, else ``None``."""
    try:
        parsed = urlparse(server_url)
        if parsed.scheme != "http" or parsed.hostname not in {
            "127.0.0.1",
            "localhost",
            "::1",
        }:
            return None
        return parsed.port or 80
    except ValueError:
        return None


def _start_port_available(port: int) -> bool:
    """Whether a new loopback server can bind *port* before configs change."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind(("127.0.0.1", port))
    except OSError:
        return False
    finally:
        probe.close()
    return True


def _start_server_background(model: str, port: int, api_key: str | None = None) -> int:
    """Spawn ``rapid-mlx serve <model> --port <port>`` detached.

    Writes the child PID to :data:`PID_FILE` so a later ``kill $(cat
    ~/.rapid-mlx/launch.pid)`` shuts it down. We don't wait for
    readiness — the launch command's whole point is "configure the
    client now, model load can happen in the background" — but we DO
    fail fast if the spawn itself fails (e.g. ``rapid-mlx`` not on
    PATH).

    Returns the child PID. The parent rapid-mlx process exits after
    detaching; the child becomes a session leader (``start_new_session``)
    so a closing terminal doesn't SIGHUP the serve.
    """
    PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["rapid-mlx", "serve", model, "--port", str(port)]
    # ``start_new_session=True`` is the POSIX-portable replacement for
    # setsid() — detaches the child from the parent's controlling
    # terminal so a Ctrl-C on the parent doesn't propagate.
    popen_kwargs: dict[str, object] = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "start_new_session": True,
    }
    if api_key:
        child_env = os.environ.copy()
        child_env["RAPID_MLX_API_KEY"] = api_key
        popen_kwargs["env"] = child_env
    proc = subprocess.Popen(cmd, **popen_kwargs)
    PID_FILE.write_text(str(proc.pid) + "\n", encoding="utf-8")
    return proc.pid


def launch_command(args: argparse.Namespace) -> None:
    """Argparse entry point for ``rapid-mlx launch``.

    Handles three subcommands by inspecting ``args.client`` and
    ``args.all``:

    * ``args.client == "list"`` → print detection matrix.
    * ``args.all`` → patch every detected client.
    * otherwise → patch the named client.

    All paths share the ``--dry-run`` short-circuit: when the user
    passed ``--dry-run`` we describe what we *would* do and exit 0
    without touching disk.
    """
    if args.client == "list":
        sys.exit(_print_list(as_json=args.json))

    if args.all and args.client:
        print(
            "launch: --all is mutually exclusive with a client name",
            file=sys.stderr,
        )
        sys.exit(2)

    if not args.all and not args.client:
        print(
            "launch: missing client name (or pass --all). "
            "Try `rapid-mlx launch list` to see supported clients.",
            file=sys.stderr,
        )
        sys.exit(2)

    api_key = os.environ.get("RAPID_MLX_API_KEY")

    targets: list[str]
    if args.all:
        targets = [name for name, adapter in ADAPTERS.items() if adapter.detect()]
        if not targets:
            print(
                "launch: no supported clients detected on this machine. "
                "Run `rapid-mlx launch list` to see what's checked.",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        if args.client not in ADAPTERS:
            supported = ", ".join(ADAPTERS.keys())
            print(
                f"launch: unknown client {args.client!r}. Supported: {supported}.",
                file=sys.stderr,
            )
            sys.exit(2)
        targets = [args.client]

    requested_url = getattr(args, "server_url", None)
    requested_port = getattr(args, "port", None)
    if args.start_server:
        url_port = _loopback_url_port(requested_url) if requested_url else None
        if requested_url and url_port is None:
            print(
                "launch: --start-server requires a loopback --server-url; "
                "the spawned server cannot bind a remote address.",
                file=sys.stderr,
            )
            sys.exit(2)
        if requested_port is not None and url_port is not None and requested_port != url_port:
            print(
                "launch: --port and --server-url select different ports; "
                "use one port for both the spawned server and client config.",
                file=sys.stderr,
            )
            sys.exit(2)
        start_port = requested_port or url_port
        if start_port is None:
            # OpenHands' running ingress owns :8000 by default. Its adapter
            # must reach that ingress to write settings, while Rapid needs a
            # different port for inference.
            start_port = 8001 if "openhands" in targets else 8000
        server_url = requested_url or f"http://127.0.0.1:{start_port}"
    else:
        start_port = requested_port or 8000
        server_url = requested_url or "http://127.0.0.1:8000"

    # Prefer the alias the user typed over the alias-resolved HF repo
    # id. The top-level ``main()`` in ``vllm_mlx/cli.py`` rewrites
    # ``args.model`` from e.g. ``qwen3.5-4b-4bit`` to
    # ``mlx-community/Qwen3.5-4B-MLX-4bit`` before dispatching to us —
    # but the IDE clients should request the short alias from
    # rapid-mlx (the server's ``/v1/models`` advertises the alias, and
    # request-side resolution accepts both), so we restore it here.
    # Same pattern as ``share_command`` in ``vllm_mlx/share/cli.py``.
    original_alias = getattr(args, "_original_alias", None)
    model = original_alias or args.model or _resolve_default_model()
    if args.dry_run:
        print(f"[dry-run] model={model} server-url={server_url}")
        for name in targets:
            adapter = ADAPTERS[name]
            path = adapter.current_config_path()
            installed = adapter.detect()
            print(f"[dry-run] {name}: detected={installed} would-patch={path}")
        if args.start_server:
            print(f"[dry-run] would spawn: rapid-mlx serve {model} --port {start_port}")
        return

    if args.start_server and not _start_port_available(start_port):
        print(
            f"launch: cannot start rapid-mlx on port {start_port}: the port is "
            "already in use. Choose another --port; the client URL will follow it.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Real patch path. Track per-client success so we can exit non-zero
    # if any single client failed even when others succeeded — the user
    # gets the partial-success line plus a final summary.
    failures: list[str] = []
    for name in targets:
        adapter = ADAPTERS[name]
        if not adapter.detect():
            print(
                f"  {name}: not detected on this machine — skipping. "
                "Install the client first.",
                file=sys.stderr,
            )
            failures.append(name)
            continue
        try:
            config_kwargs = {
                "server_url": server_url,
                "model": model,
            }
            if api_key:
                config_kwargs["api_key"] = api_key
            path = adapter.write_or_patch_config(**config_kwargs)
        except Exception as exc:
            print(f"  {name}: FAILED — {exc}", file=sys.stderr)
            failures.append(name)
            continue
        print(f"  Patched {name} config at {path}")

    succeeded = [n for n in targets if n not in failures]

    if args.start_server:
        if not succeeded:
            print(
                "  Skipping --start-server: no clients were patched. "
                "Install a supported client first, then re-run.",
                file=sys.stderr,
            )
        else:
            pid = _start_server_background(model, start_port, api_key=api_key)
            print(f"  Started: rapid-mlx serve {model} --port {start_port} (pid {pid})")
            print(f"  PID file: {PID_FILE}")

    if succeeded:
        print(
            "\nNow ready: open "
            + " / ".join(succeeded)
            + " and it'll route through rapid-mlx."
        )
    if failures:
        sys.exit(1)


def register(subparsers) -> None:
    """Wire up the ``launch`` subcommand onto the top-level CLI parser.

    Called from :mod:`vllm_mlx.cli` alongside the other
    ``subparsers.add_parser(...)`` blocks. Keeping the wiring here (not
    in ``cli.py``) means a future client-list change touches only this
    module.
    """
    # Deferred import: ``vllm_mlx.cli`` imports us at module load to
    # register the subcommand, so we cannot import from it at file scope
    # without forming an import cycle. Reuse ``serve``'s ``[1, 65535]``
    # port validator so `launch --port 99999` argparse-rejects up front
    # instead of failing inside the detached child after the parent has
    # already written a PID and printed "Started".
    from ..cli import _port_arg

    p = subparsers.add_parser(
        "launch",
        help="One-shot bootstrap: patch IDE/agent client config to use rapid-mlx",
        description=(
            "Detect a client (Cline, Claude Code, OpenHands) and write/patch its "
            "config to route at the rapid-mlx server. Use `rapid-mlx launch "
            "list` to see what's supported. Clients whose settings are not a "
            "writable config file — Cursor, for one — are covered by "
            "`rapid-mlx agents <name>` instead."
        ),
    )
    p.add_argument(
        "client",
        nargs="?",
        default=None,
        help=(
            'Client to configure (or "list" to print the detection matrix). '
            "Supported: " + ", ".join(ADAPTERS.keys()) + "."
        ),
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Patch every detected client. Mutually exclusive with a client name.",
    )
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Model alias the client will request from rapid-mlx "
            "(default: $RAPID_MLX_DEFAULT_MODEL or qwen3.5-4b-4bit)."
        ),
    )
    p.add_argument(
        "--server-url",
        type=str,
        default=None,
        help=(
            "rapid-mlx server URL the client will route at (default: the "
            "--start-server port, otherwise http://127.0.0.1:8000)"
        ),
    )
    p.add_argument(
        "--port",
        type=_port_arg,
        default=None,
        help=(
            "Port for --start-server (default: 8000, or 8001 when OpenHands "
            "is selected). Must be in [1, 65535]."
        ),
    )
    p.add_argument(
        "--start-server",
        action="store_true",
        help=(
            "Also spawn `rapid-mlx serve <model> --port <port>` in the "
            "background, writing the pid to ~/.rapid-mlx/launch.pid."
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would change without touching disk.",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="With `list`, emit the complete GUI/CLI integration registry as JSON.",
    )
