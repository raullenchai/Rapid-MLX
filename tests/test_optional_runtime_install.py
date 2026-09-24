# SPDX-License-Identifier: Apache-2.0
"""Opt-in installation for absent optional serving runtimes."""

from __future__ import annotations

import ast
import importlib.util
import io
import os
import select
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import rapid_mlx
from rapid_mlx import cli
from rapid_mlx.runtime import optional_runtime
from rapid_mlx.runtime.optional_runtime import OptionalRuntimeMissing

if sys.platform == "win32":
    pty = None
else:
    import pty


class _TTY(io.StringIO):
    def isatty(self) -> bool:
        return True

    def fileno(self) -> int:
        return 0


class _NotTTY(io.StringIO):
    def isatty(self) -> bool:
        return False


class _StreamWithoutIsatty:
    def __init__(self) -> None:
        self._buffer = io.StringIO()

    def write(self, value: str) -> int:
        return self._buffer.write(value)

    def flush(self) -> None:
        self._buffer.flush()

    def getvalue(self) -> str:
        return self._buffer.getvalue()


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
    monkeypatch.setattr(
        select,
        "select",
        lambda readable, _writable, _errors, _timeout: (readable, [], []),
    )
    monkeypatch.setattr(
        optional_runtime.os,
        "read",
        lambda _fd, size: stdin.read(size).encode("utf-8"),
    )
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
        ("pip", pip_argv, False),
        ("flush", 2.0),
        ("execv", "/tmp/rapid/bin/python", original),
    ]
    assert stderr.getvalue().endswith("Install rapid-mlx[vision] now? (~322 MB) [y/N] ")


@pytest.mark.parametrize(
    ("extra", "answer", "expected_prompt"),
    [
        ("vision", "n\n", "Install rapid-mlx[vision] now? (~322 MB) [y/N] "),
        ("audio", "N\n", "Install rapid-mlx[audio] now? (~600 MB) [y/N] "),
        ("image", "\n", "Install rapid-mlx[image] now? [y/N] "),
    ],
)
def test_tty_decline_keeps_exit_two_without_install(
    monkeypatch, extra, answer, expected_prompt
) -> None:
    stdin = _TTY(answer)
    stderr = _TTY()
    _isolate_handler(monkeypatch, stdin=stdin, stderr=stderr)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("pip must not run"),
    )

    with pytest.raises(SystemExit, match="2"):
        optional_runtime.handle_optional_runtime_missing(_failure(extra=extra))

    assert expected_prompt in stderr.getvalue()


def test_tty_timeout_defaults_no_without_reading_stdin(monkeypatch) -> None:
    class UnreadableTTY(_TTY):
        def readline(self, *_args, **_kwargs) -> str:
            pytest.fail("timed-out stdin must not be read")

    stdin = UnreadableTTY()
    stderr = _TTY()
    _isolate_handler(monkeypatch, stdin=stdin, stderr=stderr)
    monkeypatch.setattr(optional_runtime, "_INSTALL_PROMPT_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(select, "select", lambda *_args: ([], [], []))
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("pip must not run"),
    )

    with pytest.raises(SystemExit, match="2"):
        optional_runtime.handle_optional_runtime_missing(_failure(extra="video"))

    assert "Install rapid-mlx[video] now? [y/N] \n" in stderr.getvalue()


def test_posix_partial_response_at_deadline_defaults_no(monkeypatch) -> None:
    clock = iter([10.0, 10.0, 10.2])
    monkeypatch.setattr(optional_runtime.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(select, "select", lambda *_args: ([0], [], []))
    monkeypatch.setattr(optional_runtime.os, "read", lambda _fd, _size: b"y")

    assert optional_runtime._read_posix_prompt_response(_TTY(), 0.1) is None


def test_posix_eof_defaults_no(monkeypatch) -> None:
    monkeypatch.setattr(select, "select", lambda *_args: ([0], [], []))
    monkeypatch.setattr(optional_runtime.os, "read", lambda _fd, _size: b"")

    assert optional_runtime._read_posix_prompt_response(_TTY(), 0.1) is None


def test_closed_stdin_exception_defaults_no(monkeypatch) -> None:
    stdin = _TTY("y\n")
    stdin.close()
    stderr = _TTY()
    _isolate_handler(monkeypatch, stdin=stdin, stderr=stderr)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("pip must not run"),
    )

    with pytest.raises(SystemExit, match="2"):
        optional_runtime.handle_optional_runtime_missing(_failure())

    assert "Install rapid-mlx[vision] now?" in stderr.getvalue()


def test_windows_console_yes_uses_polled_characters(monkeypatch) -> None:
    characters = iter(["x", "\b", "y", "\r"])
    console = SimpleNamespace(
        kbhit=lambda: True,
        getwche=lambda: next(characters),
    )
    monkeypatch.setitem(sys.modules, "msvcrt", console)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(optional_runtime, "_INSTALL_PROMPT_TIMEOUT_SECONDS", 0.01)

    assert optional_runtime._prompt_to_install("vision") is True


def test_windows_console_timeout_never_reads_fake_stdin(monkeypatch) -> None:
    class UnreadableTTY(_TTY):
        def readline(self, *_args, **_kwargs) -> str:
            pytest.fail("the Windows console path must not call stdin.readline")

    monkeypatch.setattr(sys, "stdin", UnreadableTTY())
    monkeypatch.setattr(sys, "stderr", _TTY())
    monkeypatch.setitem(
        sys.modules,
        "msvcrt",
        SimpleNamespace(kbhit=lambda: False, getwche=lambda: pytest.fail("no key")),
    )
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(optional_runtime, "_INSTALL_PROMPT_TIMEOUT_SECONDS", 0.01)

    assert optional_runtime._prompt_to_install("vision") is False


def test_windows_console_exception_defaults_no(monkeypatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "msvcrt",
        SimpleNamespace(
            kbhit=lambda: (_ for _ in ()).throw(OSError("console closed")),
            getwche=lambda: "y",
        ),
    )
    monkeypatch.setattr(sys, "platform", "win32")

    assert optional_runtime._prompt_to_install("vision") is False


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


def test_isatty_exception_is_non_tty(monkeypatch) -> None:
    class BrokenTTY(_TTY):
        def isatty(self) -> bool:
            raise OSError("stream closed")

    stderr = _TTY()
    _isolate_handler(monkeypatch, stdin=BrokenTTY(), stderr=stderr)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("pip must not run"),
    )

    with pytest.raises(SystemExit, match="2"):
        optional_runtime.handle_optional_runtime_missing(_failure())

    assert "Install rapid-mlx[vision] now?" not in stderr.getvalue()


@pytest.mark.parametrize("stream_name", ["stdin", "stderr"])
@pytest.mark.parametrize("stream", [None, _StreamWithoutIsatty()])
def test_detached_or_stream_without_isatty_is_non_tty(
    monkeypatch, capsys, stream_name, stream
) -> None:
    streams = {"stdin": _TTY("y\n"), "stderr": _TTY()}
    streams[stream_name] = stream
    stderr = streams["stderr"]
    _isolate_handler(monkeypatch, stdin=streams["stdin"], stderr=stderr)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("pip must not run"),
    )

    with pytest.raises(SystemExit, match="2"):
        optional_runtime.handle_optional_runtime_missing(_failure())

    if stderr is None:
        output = capsys.readouterr().out
        assert "missing vision" in output
        assert "Install rapid-mlx[vision] now?" not in output
    else:
        assert "missing vision" in stderr.getvalue()
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


def test_pip_failure_preserves_original_multiline_hint(monkeypatch) -> None:
    stderr = _NotTTY()
    _isolate_handler(monkeypatch, stdin=_NotTTY(), stderr=stderr)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda argv, **_kwargs: subprocess.CompletedProcess(argv, 17),
    )

    hint = (
        "Install the validated vision stack into this runtime with:\n"
        "  python -m pip install 'rapid-mlx[vision]'"
    )
    failure = OptionalRuntimeMissing(
        extra="vision",
        install_hint=hint,
        detail="missing vision",
        status="absent",
    )

    optional_runtime._install_optional_extra(failure)

    assert stderr.getvalue() == f"Install failed (rc 17).\n{hint}\n"


def _read_until(fd: int, marker: bytes, timeout: float = 3.0) -> bytes:
    data = bytearray()
    deadline = time.monotonic() + timeout
    while marker not in data and time.monotonic() < deadline:
        ready, _, _ = select.select([fd], [], [], 0.1)
        if not ready:
            continue
        try:
            chunk = os.read(fd, 4096)
        except OSError:
            break
        if not chunk:
            break
        data.extend(chunk)
    return bytes(data)


def _pty_child(script: str):
    assert pty is not None
    master, slave = pty.openpty()
    proc = subprocess.Popen(
        [sys.executable, "-c", script],
        stdin=slave,
        stdout=slave,
        stderr=slave,
        close_fds=True,
        env={**os.environ, "HOME": os.environ["HOME"]},
    )
    os.close(slave)
    return proc, master


@pytest.mark.skipif(sys.platform == "win32", reason="requires POSIX pty")
def test_real_tty_timeout_does_not_consume_later_prompt_input() -> None:
    script = textwrap.dedent(
        """
        import sys
        from rapid_mlx.runtime import optional_runtime as o
        o._INSTALL_PROMPT_TIMEOUT_SECONDS = 0.05
        result = o._prompt_to_install('vision')
        print('TIMEOUT', result, flush=True)
        later = sys.stdin.readline()
        print('LATER_GOT=' + repr(later), flush=True)
        """
    )
    proc, master = _pty_child(script)
    before = _read_until(master, b"TIMEOUT False")
    os.write(master, b"FIRST\nSECOND\n")
    after = _read_until(master, b"LATER_GOT=")
    proc.wait(timeout=3)
    os.close(master)

    output = (before + after).decode(errors="replace")
    assert "LATER_GOT='FIRST\\n'" in output, output


@pytest.mark.skipif(sys.platform == "win32", reason="requires POSIX pty")
def test_real_tty_process_exits_cleanly_after_timeout() -> None:
    script = textwrap.dedent(
        """
        from rapid_mlx.runtime import optional_runtime as o
        o._INSTALL_PROMPT_TIMEOUT_SECONDS = 0.05
        print('RESULT', o._prompt_to_install('vision'), flush=True)
        """
    )
    proc, master = _pty_child(script)
    output = _read_until(master, b"RESULT False")
    proc.wait(timeout=3)
    output += _read_until(master, b"never", timeout=0.2)
    os.close(master)

    assert proc.returncode == 0
    assert b"Traceback" not in output and b"Fatal Python error" not in output


@pytest.mark.skipif(sys.platform == "win32", reason="requires POSIX pty")
def test_real_tty_stdin_closes_cleanly_after_timeout() -> None:
    script = textwrap.dedent(
        """
        import sys
        from rapid_mlx.runtime import optional_runtime as o
        o._INSTALL_PROMPT_TIMEOUT_SECONDS = 0.05
        print('RESULT', o._prompt_to_install('vision'), flush=True)
        sys.stdin.close()
        print('CLOSED', flush=True)
        """
    )
    proc, master = _pty_child(script)
    output = _read_until(master, b"RESULT False")
    output += _read_until(master, b"CLOSED", timeout=0.2)
    proc.wait(timeout=3)
    os.close(master)

    assert b"CLOSED" in output, output.decode(errors="replace")


@pytest.mark.skipif(sys.platform == "win32", reason="requires POSIX pty")
def test_raw_tty_one_byte_cannot_bypass_prompt_deadline() -> None:
    script = textwrap.dedent(
        """
        import sys
        import tty
        from rapid_mlx.runtime import optional_runtime as o
        tty.setraw(sys.stdin.fileno())
        print('RAW_READY', flush=True)
        print(
            'RESULT=' + repr(
                o._prompt_to_install('vision', timeout_seconds=0.10)
            ),
            flush=True,
        )
        """
    )
    proc, master = _pty_child(script)
    before = _read_until(master, b"Install rapid-mlx[vision] now?")
    os.write(master, b"y")
    after = _read_until(master, b"RESULT=", timeout=0.7)
    proc.wait(timeout=3)
    os.close(master)

    assert b"RAW_READY" in before
    assert b"RESULT=False" in after, (before + after).decode(errors="replace")


@pytest.mark.skipif(sys.platform == "win32", reason="requires POSIX pty")
def test_canonical_partial_line_ctrl_d_defaults_no_at_deadline() -> None:
    script = textwrap.dedent(
        """
        from rapid_mlx.runtime import optional_runtime as o
        print(
            'RESULT=' + repr(
                o._prompt_to_install('vision', timeout_seconds=0.10)
            ),
            flush=True,
        )
        """
    )
    proc, master = _pty_child(script)
    before = _read_until(master, b"Install rapid-mlx[vision] now?")
    os.write(master, b"y")
    time.sleep(0.02)
    os.write(master, b"\x04")
    after = _read_until(master, b"RESULT=", timeout=0.7)
    proc.wait(timeout=3)
    os.close(master)

    assert b"RESULT=False" in after, (before + after).decode(errors="replace")


def test_all_optional_runtime_handler_call_sites_forward_assume_yes() -> None:
    missing = []
    cli_path = Path(cli.__file__)
    server_path = Path(optional_runtime.__file__).parents[1] / "server.py"
    for path in (cli_path, server_path):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function_name = getattr(node.func, "id", None)
            if function_name not in {
                "_handle_optional_runtime_missing",
                "handle_optional_runtime_missing",
            }:
                continue
            if not any(keyword.arg == "assume_yes" for keyword in node.keywords):
                missing.append(f"{path}:{node.lineno}")

    assert missing == []


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
