# SPDX-License-Identifier: Apache-2.0
"""``rapid-mlx launch`` subcommand wiring.

This module exposes:

* :func:`register` — called from :mod:`rapid_mlx.cli` to wire the
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
import re
import shlex
import signal
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from . import ADAPTERS, CLIENT_ALIASES, canonical_client, cursor

if TYPE_CHECKING:
    from rapid_mlx.agents.base import AgentProfile

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
        from rapid_mlx.integrations import integration_targets_json

        print(json.dumps(integration_targets_json(), ensure_ascii=False))
        return 0
    width = max(len(name) for name in ADAPTERS) + 2
    print("Supported clients:")
    for name, adapter in ADAPTERS.items():
        status = "detected" if adapter.detect() else "not detected"
        print(f"  {name.ljust(width)}{status}")
    print("  claude-desktop  in-app gateway setup")
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
    current = Path(sys.argv[0])
    executable = (
        str(current.resolve())
        if current.name == "rapid-mlx" and current.is_file()
        else "rapid-mlx"
    )
    cmd = [executable, "serve", model, "--port", str(port)]
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


def _stop_failed_start(pid: int | None) -> None:
    """Stop a detached model load when launch is cancelled or fails."""
    if not isinstance(pid, int):
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        if PID_FILE.read_text(encoding="utf-8").strip() == str(pid):
            PID_FILE.unlink()
    except FileNotFoundError:
        pass


def _usable_running_model(model: str, profile: AgentProfile) -> bool:
    """Check tool support and the documented floor for a coding agent."""
    from rapid_mlx.model_auto_config import detect_model_config

    config = detect_model_config(model)
    if config is None or (
        profile.needs_function_calling and not config.tool_call_parser
    ):
        return False
    if profile.needs_function_calling and "gemma-4-12b-it" in model.lower():
        return False
    if profile.name == "codex":
        identity = (
            model
            if re.search(r"\d+(?:\.\d+)?b(?=[^a-z0-9]|$)", model.lower())
            else config.hf_path
        )
        match = re.search(r"(?<!\d)(\d+(?:\.\d+)?)b(?=[^a-z0-9]|$)", identity.lower())
        return bool(match and float(match.group(1)) >= 9)
    return True


def _same_checkpoint(first: str, second: str) -> bool:
    from rapid_mlx.model_aliases import resolve_model

    try:
        return resolve_model(first) == resolve_model(second)
    except ValueError:
        return first == second


def _print_preview(
    name: str, adapter: object, server_url: str, model: str, api_key: str | None
) -> None:
    """Print the redacted diff ``launch <client>`` would apply (dry run).

    Only adapters that can compute their change without side effects expose
    ``preview``; the rest keep the path-only dry-run line above.
    """
    preview = getattr(adapter, "preview", None)
    if preview is None:
        return
    try:
        _path, diff, notes = preview(server_url, model, api_key or "sk-noop")
    except (OSError, ValueError) as exc:
        print(f"[dry-run] {name}: cannot preview — {exc}")
        return
    for note in notes:
        print(f"[dry-run] {name}: note: {note}")
    print(diff if diff else f"[dry-run] {name}: already configured; no changes")


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

    if args.client == "claude-desktop":
        from rapid_mlx.agents import get_profile
        from rapid_mlx.agents.isolated import connection_status
        from rapid_mlx.launch import claude_desktop
        from rapid_mlx.run.cli import _select_model

        if args.dry_run:
            print(
                "[dry-run] Would save Rapid's Claude Desktop tier mapping; Claude Desktop is untouched."
            )
            return
        if getattr(args, "remove", False):
            claude_desktop.mapping_path().unlink(missing_ok=True)
            print("Claude Desktop tier mapping disabled in Rapid-MLX.")
            return
        model = getattr(args, "_original_alias", None) or _select_model(
            explicit=args.model, profile=get_profile("claude-code"), no_download=False
        )
        if model is None:
            print(
                "launch: no tool-capable model fits this Mac; choose --model",
                file=sys.stderr,
            )
            sys.exit(1)
        url, gateway_key = claude_desktop.configure(
            model,
            opus=getattr(args, "opus_model", None),
            sonnet=getattr(args, "sonnet_model", None),
            haiku=getattr(args, "haiku_model", None),
            port=args.port,
        )
        print("Claude Desktop gateway (configure inside Claude Desktop):")
        print(f"  Gateway URL: {url}\n  API key: {gateway_key}")
        gateway_port = urlsplit(url).port
        assert gateway_port is not None
        status = connection_status(gateway_port, gateway_key, model)
        if status == "key-mismatch":
            print(
                "  Restart the local server to activate this saved API key before Test."
            )
        elif status == "model-mismatch":
            print(f"  Start {model} on this port before Test.")
        elif status != "ready":
            print("  Start the local server before Test.")
        print("  1. Help → Troubleshooting → Enable Developer Mode")
        print("  2. Developer → Configure Third-Party Inference → Gateway")
        print("  3. Paste the URL and API key; Test connection")
        print("  4. Apply Changes")
        return

    from rapid_mlx.agents.isolated import SUPPORTED, saved_model

    # The short command is the day-one and day-two entry point for coding
    # agents. An explicit --isolated without --run still only saves a profile.
    resume_saved = (
        args.client in SUPPORTED
        and not getattr(args, "isolated", False)
        and not getattr(args, "remove", False)
        and not getattr(args, "all", False)
        and not getattr(args, "legacy_config", False)
        and args.server_url is None
    )
    if resume_saved:
        args.isolated = True
        args.run = True

    if (getattr(args, "run", False) or getattr(args, "check", False)) and not getattr(
        args, "isolated", False
    ):
        print("launch: --run and --check require --isolated", file=sys.stderr)
        sys.exit(2)

    if getattr(args, "isolated", False) or getattr(args, "remove", False):
        from rapid_mlx.agents import get_profile
        from rapid_mlx.agents.isolated import (
            _cli_invocation,
            configure,
            connection_status,
            remove,
            running_model,
            saved_model,
            stable_key,
            stable_port,
        )

        if not args.client or args.all:
            print("launch: name one agent for --isolated or --remove", file=sys.stderr)
            sys.exit(2)
        if args.dry_run:
            action = "remove" if getattr(args, "remove", False) else "save"
            print(f"[dry-run] Would {action} the isolated {args.client} profile.")
            return
        if getattr(args, "check", False):
            model = getattr(args, "_original_alias", None) or args.model
            if not model:
                print("launch: --check requires --model", file=sys.stderr)
                sys.exit(2)
            port = args.port if args.port is not None else stable_port()
            key = stable_key(create=False)
            if key is None:
                print(
                    "Local connection key is missing. Open Rapid Launch and connect this agent again.",
                    file=sys.stderr,
                )
                sys.exit(1)
            status = connection_status(port, key, model)
            if status == "ready":
                if args.client == "hermes":
                    try:
                        configure(
                            args.client, model, port=port, require_live_context=True
                        )
                    except (OSError, ValueError) as exc:
                        print(f"launch: {exc}", file=sys.stderr)
                        raise SystemExit(1) from exc
                return
            start_hint = (
                f"{_cli_invocation()} start {shlex.quote(args.client)} "
                f"--model {shlex.quote(model)} --no-setup --port {port}"
            )
            reason = (
                "Local server is stopped."
                if status == "stopped"
                else "Local server key or model does not match this profile."
            )
            print(f"{reason} Start it with: {start_hint}", file=sys.stderr)
            sys.exit(1)
        try:
            if getattr(args, "remove", False):
                removed = remove(args.client)
                print(
                    f"  {'Removed' if removed else 'No saved profile for'} {args.client}."
                )
                return
            profile = get_profile(args.client)
            if profile is None:
                raise ValueError(f"unknown agent: {args.client}")
            from rapid_mlx.run.cli import _select_model

            port = args.port if args.port is not None else stable_port()
            key = stable_key(create=False)
            active = (
                running_model(port, key)
                if key and getattr(args, "run", False)
                else None
            )
            explicit = getattr(args, "_original_alias", None) or args.model
            active_usable = bool(active and _usable_running_model(active, profile))
            reuse = bool(active_usable and not explicit)
            model = (
                active
                if reuse
                else explicit
                or _select_model(
                    explicit=saved_model(args.client),
                    profile=profile,
                    no_download=False,
                )
            )
            if model is None:
                raise ValueError("no model fits this Mac; choose one with --model")
            if active and not active_usable and _same_checkpoint(active, model):
                raise ValueError(
                    f"running model {active} does not meet this agent's requirements"
                )
            if active and not _same_checkpoint(active, model):
                print(
                    f"  Local server serves {active}; stop it before launching {model}.",
                    file=sys.stderr,
                )
                raise SystemExit(1)
            if args.model:
                from rapid_mlx.recommendations import physical_ram_gb
                from rapid_mlx.run.cli import _fits_host

                ram_gb = physical_ram_gb()
                if ram_gb and _fits_host(model, ram_gb) is False:
                    print(
                        f"  Warning: {model} is not recommended for this Mac's memory.",
                        file=sys.stderr,
                    )
            if profile.needs_function_calling and "gemma-4-12b-it" in model.lower():
                print(
                    "  Warning: this model is known to stall on coding-agent tool calls; "
                    "choose a recommended model for reliable edits.",
                    file=sys.stderr,
                )
            command = configure(args.client, model, port=port)
        except (OSError, ValueError) as exc:
            print(f"launch: {exc}", file=sys.stderr)
            sys.exit(1)
        day_two = f"{_cli_invocation()} launch {shlex.quote(args.client)}"
        print(f"  Saved an isolated {args.client} profile. Next time: {day_two}")
        start_hint = (
            f"{_cli_invocation()} start {shlex.quote(args.client)} --model {shlex.quote(model)} "
            f"--no-setup --port {port}"
        )
        if getattr(args, "run", False):
            if reuse:
                print(f"  Using running model {model} for {args.client}.")
            key = stable_key(create=False)
            assert key is not None  # configure() created or reused it above.
            server_pid = None
            status = connection_status(port, key, model)
            if status == "stopped":
                if not (args.start_server or resume_saved):
                    print(f"  Local server is stopped. Start it with: {start_hint}")
                    raise SystemExit(1)
                from rapid_mlx.run.cli import _confirm_download

                if not _confirm_download(model, no_download=False, yes=False):
                    raise SystemExit(1)
                print(
                    f"  Downloading or loading {model}; waiting for readiness…",
                    flush=True,
                )
                server_pid = _start_server_background(model, port, api_key=key)
                import time

                try:
                    for second in range(600):
                        status = connection_status(port, key, model)
                        if status not in {"stopped", "server-error"}:
                            break
                        if second and second % 10 == 0:
                            print(f"  Preparing local model… {second}s", flush=True)
                        time.sleep(1)
                    if status in {"stopped", "server-error"}:
                        print(
                            f"  Local server did not become ready. Retry: {start_hint}"
                        )
                        raise SystemExit(1)
                    if status != "ready":
                        print(
                            "  Local server key or model does not match this profile. "
                            f"Start the matching model with: {start_hint}",
                            file=sys.stderr,
                        )
                        raise SystemExit(1)
                except BaseException:
                    _stop_failed_start(server_pid)
                    raise
            if status != "ready":
                print(
                    "  Local server key or model does not match this profile. "
                    f"Start the matching model with: {start_hint}",
                    file=sys.stderr,
                )
                raise SystemExit(1)
            if args.client == "hermes":
                try:
                    configure(args.client, model, port=port, require_live_context=True)
                except (OSError, ValueError) as exc:
                    if server_pid is not None:
                        _stop_failed_start(server_pid)
                    print(f"launch: {exc}", file=sys.stderr)
                    raise SystemExit(1) from exc
            print(
                "  Agent ready. Its first reply may take longer while the local prompt is processed; cached replies are faster.",
                flush=True,
            )
            raise SystemExit(subprocess.call(["/bin/zsh", "-lc", command]))
        print(f"  If the local server is stopped: {start_hint}")
        return

    if args.port is None:
        args.port = 8000

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

    server_url = args.server_url or "http://127.0.0.1:8000"
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
    # id. The top-level ``main()`` in ``rapid_mlx/cli.py`` rewrites
    # ``args.model`` from e.g. ``qwen3.5-4b-4bit`` to
    # ``mlx-community/Qwen3.5-4B-MLX-4bit`` before dispatching to us —
    # but the IDE clients should request the short alias from
    # rapid-mlx (the server's ``/v1/models`` advertises the alias, and
    # request-side resolution accepts both), so we restore it here.
    # Same pattern as ``share_command`` in ``rapid_mlx/share/cli.py``.
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
            _print_preview(name, adapter, server_url, model, api_key)
        if args.start_server:
            print(f"[dry-run] would spawn: rapid-mlx serve {model} --port {args.port}")
        return

    # Real patch path. Track per-client success so we can exit non-zero
    # if any single client failed even when others succeeded — the user
    # gets the partial-success line plus a final summary.
    failures: list[str] = []
    claude_context_length = None
    if "claude-code" in targets:
        from rapid_mlx.agents.adapter import fetch_context_window
        from rapid_mlx.run.cli import _cached_context_window

        # --start-server patches the client before the detached server boots;
        # probing it now only adds a guaranteed timeout to first run.
        if not args.start_server:
            claude_context_length = fetch_context_window(
                f"{server_url.rstrip('/').removesuffix('/v1')}/v1", model
            )
        claude_context_length = claude_context_length or _cached_context_window(model)
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
            if name == "claude-code":
                config_kwargs["context_length"] = claude_context_length
            path = adapter.write_or_patch_config(**config_kwargs)
        except Exception as exc:
            print(f"  {name}: FAILED — {exc}", file=sys.stderr)
            failures.append(name)
            continue
        print(f"  Patched {name} config at {path}")
        notes = getattr(adapter, "post_setup_notes", None)
        if notes is not None:
            for line in notes(server_url, model, api_key):
                print(f"    {line}")

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

    Called from :mod:`rapid_mlx.cli` alongside the other
    ``subparsers.add_parser(...)`` blocks. Keeping the wiring here (not
    in ``cli.py``) means a future client-list change touches only this
    module.
    """
    # Deferred import: ``rapid_mlx.cli`` imports us at module load to
    # register the subcommand, so we cannot import from it at file scope
    # without forming an import cycle. Reuse ``serve``'s ``[1, 65535]``
    # port validator so `launch --port 99999` argparse-rejects up front
    # instead of failing inside the detached child after the parent has
    # already written a PID and printed "Started".
    from ..cli_parser import _port_arg

    p = subparsers.add_parser(
        "launch",
        help="Launch coding agents with private local profiles or configure editors",
        description=(
            "Launch a coding agent with a private local profile and start the "
            "server if needed, or configure an editor. Use `rapid-mlx launch "
            "list` to see what's supported. Cursor requires a public HTTPS "
            "--server-url."
        ),
    )
    p.add_argument(
        "client",
        nargs="?",
        default=None,
        help=(
            'Agent to launch or client to configure (or "list" to print the detection matrix). '
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
        "--isolated",
        action="store_true",
        help="Save a private, removable Rapid profile without editing the client's existing configuration",
    )
    p.add_argument(
        "--legacy-config",
        action="store_true",
        help="Use the original client config writer instead of isolated agent launch",
    )
    p.add_argument(
        "--run",
        action="store_true",
        help="Run the isolated agent after saving its profile",
    )
    p.add_argument(
        "--check",
        action="store_true",
        help="Check that the saved local key, port, and model are available",
    )
    p.add_argument(
        "--remove",
        action="store_true",
        help="Remove the private Rapid profile for this agent",
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
    p.add_argument("--opus-model", help="Local model assigned to the opus tier")
    p.add_argument("--sonnet-model", help="Local model assigned to the sonnet tier")
    p.add_argument("--haiku-model", help="Local model assigned to the haiku tier")
    p.add_argument(
        "--server-url",
        type=str,
        default=None,
        help="rapid-mlx server URL the client will route at (default: http://127.0.0.1:8000)",
    )
    p.add_argument(
        "--port",
        type=_port_arg,
        default=None,
        help=(
            "Port for --start-server (default: 8000), or the saved local port "
            "with --isolated (default: the port used by Desktop). Must be in [1, 65535]."
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
