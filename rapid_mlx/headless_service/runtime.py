# SPDX-License-Identifier: Apache-2.0
"""Stable LaunchDaemon entry point backed by :class:`ServiceConfig`."""

from __future__ import annotations

import os
import pwd
import signal
import subprocess
import sys
import threading
from pathlib import Path

from .common import STDERR_LOG_NAME, STDOUT_LOG_NAME, log_dir_for
from .config import ServiceConfigError, assert_private_file, load_config
from .rotating_logs import RotatingLog


def _copy_stream(source, sink: RotatingLog, child: subprocess.Popen) -> None:
    try:
        while chunk := source.read(65536):
            try:
                sink.write(chunk)
            except OSError as exc:
                # Never leave the model process blocked forever on a full or
                # unwritable log pipe. A clean service restart is observable
                # and recoverable; a wedged inference endpoint is not.
                print(f"error: service log write failed: {exc}", file=sys.stderr)
                if child.poll() is None:
                    child.terminate()
                for _ in iter(lambda: source.read(65536), b""):
                    pass
                break
    finally:
        source.close()


def _supervise(argv: list[str], env: dict[str, str], config) -> int:
    log_dir = log_dir_for(config.service_user)
    if log_dir is None:
        raise ServiceConfigError("cannot resolve service log directory")
    log_options = {
        "max_bytes": config.log_max_mb * 1024 * 1024,
        "backup_count": config.log_backup_count,
        "retention_days": config.log_retention_days,
    }
    stdout = RotatingLog(log_dir / STDOUT_LOG_NAME, **log_options)
    stderr = RotatingLog(log_dir / STDERR_LOG_NAME, **log_options)
    child: subprocess.Popen | None = None
    requested_signal: int | None = None

    def forward(signum, _frame) -> None:
        nonlocal requested_signal
        requested_signal = signum
        if child is not None and child.poll() is None:
            child.send_signal(signum)

    previous_handlers = {}
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        previous_handlers[signum] = signal.signal(signum, forward)
    try:
        child = subprocess.Popen(
            argv,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        if requested_signal is not None:
            child.send_signal(requested_signal)
        assert child.stdout is not None and child.stderr is not None
        pumps = [
            threading.Thread(
                target=_copy_stream, args=(child.stdout, stdout, child), daemon=True
            ),
            threading.Thread(
                target=_copy_stream, args=(child.stderr, stderr, child), daemon=True
            ),
        ]
        for pump in pumps:
            pump.start()
        return_code = child.wait()
        for pump in pumps:
            pump.join(timeout=5)
        return return_code if return_code >= 0 else 128 - return_code
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        stdout.close()
        stderr.close()


def run_command(args) -> int:
    path = Path(args.config)
    try:
        config = load_config(path)
        account = pwd.getpwnam(config.service_user)
        if os.geteuid() != account.pw_uid:
            raise ServiceConfigError(
                f"service runtime uid {os.geteuid()} does not match configured "
                f"account {config.service_user} ({account.pw_uid})"
            )
        env = os.environ.copy()
        env.pop("RAPID_MLX_API_KEY", None)
        if config.credential_file:
            secret_path = Path(config.credential_file)
            if secret_path.exists():
                assert_private_file(secret_path, expected_uid=account.pw_uid)
                secret = secret_path.read_text(encoding="utf-8").strip()
                if not secret or "\n" in secret or "\r" in secret:
                    raise ServiceConfigError(
                        "credential file must contain one non-empty line"
                    )
                env["RAPID_MLX_API_KEY"] = secret
        argv = [
            config.executable,
            "serve",
            config.model,
            "--host",
            config.host,
            "--port",
            str(config.port),
            *config.serve_args,
        ]
        return _supervise(argv, env, config)
    except (KeyError, OSError, ServiceConfigError) as exc:
        print(f"error: cannot start Rapid-MLX service: {exc}", file=sys.stderr)
        return 78  # EX_CONFIG
