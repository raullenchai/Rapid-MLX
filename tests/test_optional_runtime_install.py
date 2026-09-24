# SPDX-License-Identifier: Apache-2.0
"""Opt-in installation for absent optional serving runtimes."""

from __future__ import annotations

import importlib.util
import io
import os
import select
import subprocess
import sys
from types import SimpleNamespace

import pytest

import rapid_mlx
from rapid_mlx import cli
from rapid_mlx.runtime import optional_runtime
from rapid_mlx.runtime.optional_runtime import OptionalRuntimeMissing


class _TTY(io.StringIO):
    def isatty(self) -> bool:
        return True


class _NotTTY(io.StringIO):
    def isatty(self) -> bool:
        return False


class _ExecCalled(BaseException):
    pass


def _failure(*, status: str = "absent", extra: str = "vision"):
    return OptionalRuntimeMissing(
        extra=extra,
        install_hint=f"pip install 'rapid-mlx[{extra}]'",
        detail=f"missing {extra}",
        status=status,
    )


def _isolate_handler(monkeypatch, *, stdin, stderr) -> list[object]:
    order: list[object] = []
    monkeypatch.setattr(sys, "stdin", stdin)
    monkeypatch.setattr(sys, "stderr", stderr)
    monkeypatch.setattr(
        "rapid_mlx.telemetry.server_start.failed",
        lambda stage: order.append(("failed", stage)),
    )
    monkeypatch.setattr(
        "rapid_mlx.telemetry.model_events.emit_model_serve_failed",
        lambda *_args, **_kwargs: order.append("model_serve_failed"),
    )
    monkeypatch.setattr(
        "rapid_mlx.telemetry.consent_runtime.is_desktop_sidecar", lambda: False
    )
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    return order


def _install_succeeds(monkeypatch, order: list[object]) -> None:
    def run(argv, *, check):
        order.append(("pip", argv, check))
        return subprocess.CompletedProcess(argv, 0)

    def execv(executable, argv):
        order.append(("execv", executable, argv))
        raise _ExecCalled

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(os, "execv", execv)
    monkeypatch.setattr(
        "rapid_mlx.telemetry.posthog_sender.get_sender",
        lambda: SimpleNamespace(flush=lambda timeout: order.append(("flush", timeout))),
    )


def test_tty_yes_installs_pinned_extra_and_reexecs_original_argv(monkeypatch) -> None:
    stdin = _TTY("y\n")
    stderr = _TTY()
    order = _isolate_handler(monkeypatch, stdin=stdin, stderr=stderr)
    monkeypatch.setattr(
        select,
        "select",
        lambda read, _write, _error, timeout: (
            order.append(("prompt_wait", read, timeout)) or (read, [], [])
        ),
    )
    _install_succeeds(monkeypatch, order)
    monkeypatch.setattr(sys, "executable", "/tmp/rapid/bin/python")
    original = [
        "/tmp/rapid/bin/python",
        "-m",
        "rapid_mlx.cli",
        "serve",
        "bonsai2-27b-2bit",
    ]
    monkeypatch.setattr(sys, "orig_argv", original)

    with pytest.raises(_ExecCalled):
        optional_runtime.handle_optional_runtime_missing(_failure())

    pip_argv = [
        "/tmp/rapid/bin/python",
        "-m",
        "pip",
        "install",
        f"rapid-mlx[vision]=={rapid_mlx.__version__}",
    ]
    assert order == [
        ("failed", "preflight"),
        "model_serve_failed",
        ("prompt_wait", [stdin], 30.0),
        ("pip", pip_argv, False),
        ("flush", 2.0),
        ("execv", "/tmp/rapid/bin/python", original),
    ]
    assert stderr.getvalue().endswith("Install rapid-mlx[vision] now? (~322 MB) [y/N] ")


@pytest.mark.parametrize(
    ("extra", "answer", "readable", "expected_prompt"),
    [
        ("vision", "n\n", True, "Install rapid-mlx[vision] now? (~322 MB) [y/N] "),
        ("audio", "N\n", True, "Install rapid-mlx[audio] now? (~600 MB) [y/N] "),
        ("image", "\n", True, "Install rapid-mlx[image] now? [y/N] "),
        ("video", "", False, "Install rapid-mlx[video] now? [y/N] "),
    ],
)
def test_tty_decline_or_timeout_keeps_exit_two_without_install(
    monkeypatch, extra, answer, readable, expected_prompt
) -> None:
    stdin = _TTY(answer)
    stderr = _TTY()
    _isolate_handler(monkeypatch, stdin=stdin, stderr=stderr)
    monkeypatch.setattr(
        select,
        "select",
        lambda read, _write, _error, _timeout: (read if readable else [], [], []),
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("pip must not run"),
    )

    with pytest.raises(SystemExit, match="2"):
        optional_runtime.handle_optional_runtime_missing(_failure(extra=extra))

    assert expected_prompt in stderr.getvalue()


def test_non_tty_without_yes_keeps_existing_failure_without_prompt(monkeypatch) -> None:
    stderr = _NotTTY()
    _isolate_handler(monkeypatch, stdin=_NotTTY(), stderr=stderr)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("pip must not run"),
    )

    with pytest.raises(SystemExit, match="2"):
        optional_runtime.handle_optional_runtime_missing(_failure())

    assert "Install rapid-mlx[vision] now?" not in stderr.getvalue()


def test_yes_installs_without_tty(monkeypatch) -> None:
    order = _isolate_handler(monkeypatch, stdin=_NotTTY(), stderr=_NotTTY())
    _install_succeeds(monkeypatch, order)
    monkeypatch.setattr(sys, "executable", "/tmp/rapid/bin/python")
    monkeypatch.setattr(sys, "orig_argv", ["python", "rapid-mlx", "serve", "kokoro"])

    with pytest.raises(_ExecCalled):
        optional_runtime.handle_optional_runtime_missing(
            _failure(extra="audio"), assume_yes=True
        )

    assert order[-1] == (
        "execv",
        "/tmp/rapid/bin/python",
        ["/tmp/rapid/bin/python", "rapid-mlx", "serve", "kokoro"],
    )


def test_pip_failure_prints_manual_hint_and_exits_two(monkeypatch) -> None:
    stderr = _NotTTY()
    _isolate_handler(monkeypatch, stdin=_NotTTY(), stderr=stderr)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda argv, **_kwargs: subprocess.CompletedProcess(argv, 17),
    )

    with pytest.raises(SystemExit, match="2"):
        optional_runtime.handle_optional_runtime_missing(_failure(), assume_yes=True)

    assert (
        "Install failed (rc 17); run pip install 'rapid-mlx[vision]' manually."
        in stderr.getvalue()
    )


@pytest.mark.parametrize("status", ["broken", "incompatible"])
def test_non_absent_status_never_prompts_or_installs(monkeypatch, status) -> None:
    stderr = _TTY()
    _isolate_handler(monkeypatch, stdin=_TTY("y\n"), stderr=stderr)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("pip must not run"),
    )

    with pytest.raises(SystemExit, match="2"):
        optional_runtime.handle_optional_runtime_missing(
            _failure(status=status), assume_yes=True
        )

    assert "Install rapid-mlx[vision] now?" not in stderr.getvalue()


def test_missing_pip_never_prompts(monkeypatch) -> None:
    stderr = _TTY()
    _isolate_handler(monkeypatch, stdin=_TTY("y\n"), stderr=stderr)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)

    with pytest.raises(SystemExit, match="2"):
        optional_runtime.handle_optional_runtime_missing(_failure(), assume_yes=True)

    assert "Install rapid-mlx[vision] now?" not in stderr.getvalue()


def test_desktop_bundle_interpreter_never_prompts(monkeypatch) -> None:
    stderr = _TTY()
    _isolate_handler(monkeypatch, stdin=_TTY("y\n"), stderr=stderr)
    monkeypatch.setattr(
        sys,
        "executable",
        "/Applications/Rapid-MLX Desktop.app/Contents/Resources/rapid-mlx/python/bin/python3.12",
    )

    with pytest.raises(SystemExit, match="2"):
        optional_runtime.handle_optional_runtime_missing(_failure(), assume_yes=True)

    assert "Install rapid-mlx[vision] now?" not in stderr.getvalue()


def test_desktop_sidecar_role_never_prompts(monkeypatch) -> None:
    stderr = _TTY()
    _isolate_handler(monkeypatch, stdin=_TTY("y\n"), stderr=stderr)
    monkeypatch.setattr(
        "rapid_mlx.telemetry.consent_runtime.is_desktop_sidecar", lambda: True
    )

    with pytest.raises(SystemExit, match="2"):
        optional_runtime.handle_optional_runtime_missing(_failure(), assume_yes=True)

    assert "Install rapid-mlx[vision] now?" not in stderr.getvalue()


def test_serve_yes_flag_and_help() -> None:
    parser = cli.build_parser()

    assert parser.parse_args(["serve", "model", "-y"]).yes is True
    assert parser.parse_args(["serve", "model", "--yes"]).yes is True
    help_text = parser._subparsers._group_actions[0].choices["serve"].format_help()
    assert "assume yes for prompts such as installing a missing optional extra" in (
        " ".join(help_text.split())
    )
