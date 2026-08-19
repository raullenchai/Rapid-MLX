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
import subprocess
import sys
from pathlib import Path

from . import ADAPTERS, CLIENT_ALIASES, canonical_client, cursor

# Where we drop the PID of a ``--start-server`` subprocess. Pulled out
# so tests can monkeypatch it to a tmp_path and assert the file's
# contents without polluting the dev's real home dir.
PID_FILE = Path.home() / ".rapid-mlx" / "launch.pid"


def _print_list(*, as_json: bool = False) -> int:
    """Print the supported-clients + detection matrix.

    Output shape (one line per client):

        cline           detected
        claude-code     not detected
        continue-dev    detected
        cursor          not detected

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
    print("\nNote: cursor requires an explicit public HTTPS --server-url.")
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
    # Accept alias spellings (``continue`` for ``continue-dev``) before
    # any registry lookup — see CLIENT_ALIASES and issue #2082.
    args.client = canonical_client(args.client)

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

    server_url = args.server_url
    api_key = os.environ.get("RAPID_MLX_API_KEY")
    cursor_server_url: str | None = None
    cursor_reason: str | None = None
    if args.all or args.client == "cursor":
        try:
            cursor_server_url = cursor.canonical_server_url(server_url)
        except ValueError as exc:
            cursor_reason = str(exc)

    targets: list[str]
    if args.all:
        if ADAPTERS["cursor"].detect() and (cursor_reason is not None or not api_key):
            skip_reason = cursor_reason or (
                "public endpoints require RAPID_MLX_API_KEY; never expose an "
                "unauthenticated server"
            )
            print(f"  cursor: skipped — {skip_reason}", file=sys.stderr)
        targets = [
            name
            for name, adapter in ADAPTERS.items()
            if adapter.detect()
            and (name != "cursor" or (cursor_server_url is not None and bool(api_key)))
        ]
        if not targets:
            print(
                "launch: no supported clients detected on this machine. "
                "Run `rapid-mlx launch list` to see what's checked.",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        if args.client == "cursor":
            if cursor_reason is not None:
                print(
                    f"launch: {cursor_reason}. BYOK requests are routed through "
                    "Cursor's servers; use a public HTTPS tunnel or choose "
                    "claude-code, cline, or continue-dev for a local connection.",
                    file=sys.stderr,
                )
                sys.exit(2)
            if not api_key:
                print(
                    "launch: Cursor public endpoints require RAPID_MLX_API_KEY. "
                    "Start Rapid-MLX with the same key; "
                    "never expose an unauthenticated server to the internet.",
                    file=sys.stderr,
                )
                sys.exit(2)
        if args.client not in ADAPTERS:
            supported = ", ".join(ADAPTERS.keys())
            print(
                f"launch: unknown client {args.client!r}. Supported: {supported}.",
                file=sys.stderr,
            )
            sys.exit(2)
        targets = [args.client]

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
    effective_cursor_url = cursor_server_url or server_url
    if args.dry_run:
        print(f"[dry-run] model={model} server-url={server_url}")
        for name in targets:
            adapter = ADAPTERS[name]
            path = adapter.current_config_path()
            installed = adapter.detect()
            print(f"[dry-run] {name}: detected={installed} would-patch={path}")
        if args.start_server:
            print(f"[dry-run] would spawn: rapid-mlx serve {model} --port {args.port}")
        return

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
                "server_url": effective_cursor_url if name == "cursor" else server_url,
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
            pid = _start_server_background(model, args.port, api_key=api_key)
            print(f"  Started: rapid-mlx serve {model} --port {args.port} (pid {pid})")
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
            "Detect an IDE client (Cline, Claude Code, Continue, Cursor) "
            "and write/patch its config to route at the "
            "rapid-mlx server. Use `rapid-mlx launch list` to see what's "
            "supported. Cursor requires a public HTTPS --server-url."
        ),
    )
    p.add_argument(
        "client",
        nargs="?",
        default=None,
        help=(
            'Client to configure (or "list" to print the detection matrix). '
            "Supported: "
            + ", ".join(ADAPTERS.keys())
            + ". Aliases: "
            + ", ".join(
                f"{alias} (for {target})" for alias, target in CLIENT_ALIASES.items()
            )
            + "."
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
        default="http://127.0.0.1:8000",
        help="rapid-mlx server URL the client will route at (default: http://127.0.0.1:8000)",
    )
    p.add_argument(
        "--port",
        type=_port_arg,
        default=8000,
        help=(
            "Port for --start-server (default: 8000). Must be in "
            "[1, 65535]. Ignored when --start-server is not set."
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
