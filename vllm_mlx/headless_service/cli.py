# SPDX-License-Identifier: Apache-2.0
"""``rapid-mlx service`` subcommand wiring.

This module exposes:

* :func:`register` — called from :mod:`vllm_mlx.cli` to wire the
  ``service`` subparser (and its five subverebe parsers) onto the
  top-level argparse tree.
* :func:`service_command` — the argparse dispatch entry point.

The subcommand supports these lifecycle shapes:

* ``rapid-mlx service install [--service-user U] [--model M] [serve options]
  [--dry-run]`` — validate + bootstrap the daemon.
* ``rapid-mlx service status [--json]`` — launchd / pid / endpoint health.
* ``rapid-mlx service logs [--follow] [--tail N]`` — tail daemon logs.
* ``rapid-mlx service restart [--dry-run]`` — kickstart + wait health.
* ``rapid-mlx service configure`` / ``apply`` — stage then atomically activate
  a versioned definition with health-gated rollback.
* ``rapid-mlx service credential`` — manage API auth without argv/plist leaks.
* ``rapid-mlx service upgrade`` — snapshot, upgrade, health-gate, and rollback.
* ``rapid-mlx service uninstall [--dry-run]`` — bootout + remove plist.

``--dry-run`` is accepted on install, configure, apply, upgrade, restart, and
uninstall so the operator can rehearse without touching the system;
``status``/``config``/``logs`` are read-only.
"""

from __future__ import annotations

import argparse
import sys


def register(subparsers) -> None:
    """Wire up the ``service`` subcommand onto the top-level CLI parser.

    Called from :mod:`vllm_mlx.cli` alongside the other
    ``subparsers.add_parser(...)`` blocks, mirroring how ``launch`` is
    wired. We keep the whole parser here (not in ``cli.py``) so the
    sub-command surface lives with its implementation.
    """
    from ..cli import _port_arg

    p = subparsers.add_parser(
        "service",
        help="Headless macOS service lifecycle (system LaunchDaemon)",
        description=(
            "Install, inspect, and remove Rapid-MLX as an unattended system "
            "launchd daemon — the supported path for an always-on Mac mini "
            "inference appliance that boots before any GUI login. "
            "Sub-commands: install, configure, apply, config, credential, "
            "upgrade, status, logs, restart, uninstall."
        ),
    )
    svc = p.add_subparsers(dest="service_command", help="Service sub-command")
    svc.required = True

    install = svc.add_parser(
        "install",
        help="Validate a least-privilege service account + model and install "
        "the daemon",
    )
    install.add_argument(
        "--service-user",
        type=str,
        required=True,
        help="The dedicated least-privilege, NON-administrator service "
        "account the daemon runs as (must already exist, e.g. serveuser). "
        "Refused if root or an admin-group member.",
    )
    _add_bind_args(install, _port_arg)
    install.add_argument(
        "--model",
        type=str,
        default="qwen3.5-4b-4bit",
        help="Model alias (or HF path) to serve, downloaded once by the "
        "service account (default: qwen3.5-4b-4bit).",
    )
    install.add_argument(
        "serve_args",
        nargs=argparse.REMAINDER,
        help="Additional non-secret serve flags after a `--` separator, e.g. "
        "`-- --max-num-seqs 4`. Secrets and bind overrides are refused.",
    )
    install.add_argument(
        "--dry-run",
        action="store_true",
        help="Print every install/validation step without changing anything.",
    )

    status = svc.add_parser("status", help="Report launchd/pid/endpoint health")
    _add_service_account_args(status)
    _add_bind_args(status, _port_arg)
    status.add_argument(
        "--json",
        action="store_true",
        help="Emit a machine-readable JSON status object.",
    )

    logs = svc.add_parser("logs", help="Tail the daemon stdout/stderr logs")
    _add_service_account_args(logs)
    logs.add_argument(
        "--follow",
        "-f",
        action="store_true",
        help="Stream new log lines (like tail -F), across KeepAlive restarts.",
    )
    logs.add_argument(
        "--tail",
        type=int,
        default=200,
        help="Lines per file to show without --follow (default: 200).",
    )

    restart = svc.add_parser(
        "restart", help="Kickstart the daemon and wait until it is healthy"
    )
    _add_bind_args(restart, _port_arg)
    restart.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the kickstart command without running it.",
    )

    uninstall = svc.add_parser(
        "uninstall",
        help="Remove the launchd registration and plist (never "
        "deletes models/cache/logs without separate confirmation)",
    )

    configure = svc.add_parser(
        "configure",
        help="Validate and stage a candidate service definition",
    )
    configure.add_argument("--model", default=None)
    configure.add_argument("--host", default=None)
    configure.add_argument("--port", type=_port_arg, default=None)
    configure.add_argument("--log-retention-days", type=int, default=None)
    configure.add_argument("--log-max-mb", type=int, default=None)
    configure.add_argument("--log-backup-count", type=int, default=None)
    configure.add_argument(
        "--clear-serve-args",
        action="store_true",
        help="Remove all previously configured advanced serve options.",
    )
    configure.add_argument(
        "serve_args",
        nargs=argparse.REMAINDER,
        help="Replacement non-secret serve flags after `--`.",
    )
    configure.add_argument("--dry-run", action="store_true")

    apply = svc.add_parser(
        "apply", help="Activate the staged definition with health-gated rollback"
    )
    apply.add_argument("--dry-run", action="store_true")

    config = svc.add_parser("config", help="Show active and pending definitions")

    credential = svc.add_parser(
        "credential", help="Manage the service API key through a mode-600 file"
    )
    cred = credential.add_subparsers(dest="credential_command", required=True)
    cred.add_parser("set", help="Read one API key from stdin and store it securely")
    cred.add_parser("status", help="Report whether a credential is configured")
    cred.add_parser("unset", help="Remove the configured API key")

    upgrade = svc.add_parser(
        "upgrade", help="Upgrade the service runtime with health-gated rollback"
    )
    upgrade.add_argument(
        "--version", default=None, help="Exact Rapid-MLX version (default: latest)"
    )
    upgrade.add_argument(
        "--extras",
        default=None,
        help="Comma-separated extras to assert, for example vision,embeddings",
    )
    upgrade.add_argument("--pre", action="store_true", help="Allow prereleases")
    upgrade.add_argument("--dry-run", action="store_true")

    # Internal stable entry point used by launchd. It is intentionally hidden
    # from help: operators manage the service through the public verbs above.
    run = svc.add_parser("run", help="Internal config-backed launchd runtime")
    run.add_argument("--config", required=True)
    uninstall.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the removal steps without running them.",
    )

    # Optional --label override is useful for multiple appliances; default
    # (com.rapidmlx.server) matches the documented contract.
    for sub in (
        install,
        configure,
        apply,
        config,
        credential,
        upgrade,
        status,
        logs,
        restart,
        uninstall,
        run,
    ):
        sub.add_argument(
            "--label",
            type=str,
            default=None,
            help="launchd Label (default: com.rapidmlx.server).",
        )


def _add_service_account_args(p: argparse.ArgumentParser) -> None:
    """The service account used for install/status/logs (needed to resolve
    home/log paths for diagnostics)."""
    p.add_argument(
        "--service-user",
        type=str,
        default=None,
        help="The dedicated least-privilege service account the daemon runs "
        "as (e.g. serveuser).",
    )


def _add_bind_args(p: argparse.ArgumentParser, port_arg) -> None:
    p.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Loopback bind host (default: 127.0.0.1).",
    )
    p.add_argument(
        "--port",
        type=port_arg,
        default=8000,
        help="Bind port (default: 8000). Must be in [1, 65535].",
    )


def service_command(args) -> None:
    """Argparse dispatch for ``rapid-mlx service …``. Exits non-zero on a
    user error via the individual command's return code."""
    if sys.platform != "darwin":
        print(
            "error: `rapid-mlx service` is macOS-only (it manages a system "
            "launchd daemon; launchd does not exist on this platform).",
            file=sys.stderr,
        )
        raise SystemExit(2)
    # Validate the launchd Label up front for every sub-command: it flows into
    # the plist path, `launchctl`, and `rm`, so a malicious ``--label`` must
    # never reach them (path-traversal / injection hardening).
    label = getattr(args, "label", None)
    if label is not None:
        from .common import validate_label

        try:
            validate_label(label)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            raise SystemExit(2) from None
    sub = getattr(args, "service_command", None)
    if sub == "install":
        from .install import install_command

        code = install_command(args)
    elif sub == "status":
        from .status import status_command

        code = status_command(args)
    elif sub == "logs":
        from .logs import logs_command

        code = logs_command(args)
    elif sub == "restart":
        from .restart import restart_command

        code = restart_command(args)
    elif sub == "uninstall":
        from .install import uninstall_command

        code = uninstall_command(args)
    elif sub == "configure":
        from .configure import configure_command

        code = configure_command(args)
    elif sub == "apply":
        from .configure import apply_command

        code = apply_command(args)
    elif sub == "config":
        from .configure import config_show_command

        code = config_show_command(args)
    elif sub == "credential":
        from .configure import credential_command

        code = credential_command(args)
    elif sub == "run":
        from .runtime import run_command

        code = run_command(args)
    elif sub == "upgrade":
        from .upgrade import upgrade_command

        code = upgrade_command(args)
    else:
        # argparse enforces a required service_command, so this is defensive.
        print(
            "error: expected one of install/status/logs/restart/uninstall "
            "or configure/apply/config/credential",
            file=sys.stderr,
        )
        code = 2
    if code:
        raise SystemExit(code)
