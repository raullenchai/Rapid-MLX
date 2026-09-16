"""Issue #3495: the serve process must not run interpreter finalization.

On macOS, letting ``uvicorn.run`` return flows into CPython finalization
while native worker threads (the rayon pool inside ``tokenizers`` /
``llguidance``, MLX/Metal internals) are still alive; some macOS dyld
versions then crash the process with SIGSEGV during TLS teardown — a
"Python quit unexpectedly" dialog after a fully clean shutdown. The fix
is ``cli._hard_exit_after_serve()``: flush + ``os._exit(0)`` right after
the uvicorn dispatch returns on the success path.

These tests pin:
  * the helper's own contract (flush, ``os._exit(0)``, pytest guard),
  * that BOTH serve entrypoints (text ``serve_command`` and
    ``_serve_audio_mode``) actually invoke the helper after
    ``_run_uvicorn`` — a refactor that drops the call silently
    re-opens the #3495 crash on macOS 15.
"""

import sys
import types

import pytest

from rapid_mlx import cli


@pytest.fixture
def production_exit_context(monkeypatch):
    """Force the helper's production path: the guard keys on ``pytest``
    being imported in THIS process, so hide it (a real serve subprocess
    never has it imported — including pytest-spawned children, whose
    inherited ``PYTEST_CURRENT_TEST`` env must NOT disable the hard
    exit).
    """
    monkeypatch.delitem(sys.modules, "pytest", raising=False)


def test_hard_exit_flushes_streams_and_exits_zero(monkeypatch, production_exit_context):
    """Success path: flush stdout/stderr, then ``os._exit(0)``."""
    exit_calls: list[int] = []

    class _FlushRecorder:
        def __init__(self) -> None:
            self.flushed = False

        def flush(self) -> None:
            self.flushed = True

    out, err = _FlushRecorder(), _FlushRecorder()
    monkeypatch.setattr(cli.sys, "stdout", out)
    monkeypatch.setattr(cli.sys, "stderr", err)
    monkeypatch.setattr(cli.os, "_exit", lambda code: exit_calls.append(code))

    cli._hard_exit_after_serve()

    assert out.flushed, "stdout must be flushed before os._exit"
    assert err.flushed, "stderr must be flushed before os._exit"
    assert exit_calls == [0], f"expected exactly one os._exit(0), got {exit_calls!r}"


def test_hard_exit_is_skipped_when_pytest_is_imported(monkeypatch):
    """In-process suites drive serve through stubbed uvicorn runs; the
    guard must keep ``os._exit`` away from the pytest process (an
    ``os._exit(0)`` mid-suite would end the run green while skipping
    every later test).
    """
    exit_calls: list[int] = []
    monkeypatch.setattr(cli.os, "_exit", lambda code: exit_calls.append(code))
    # Simulate the in-process harness even if this suite runs in a
    # pytest-less runner: the guard keys on module presence, not env.
    monkeypatch.setitem(sys.modules, "pytest", types.ModuleType("pytest"))

    cli._hard_exit_after_serve()

    assert exit_calls == []


def test_hard_exit_swallows_flush_failures(monkeypatch, production_exit_context):
    """A dead stream (closed stderr at teardown) must not convert the
    clean exit into a traceback-driven non-zero exit.
    """

    class _Broken:
        def flush(self) -> None:
            raise ValueError("stream closed")

    exit_calls: list[int] = []
    monkeypatch.setattr(cli.sys, "stdout", _Broken())
    monkeypatch.setattr(cli.sys, "stderr", _Broken())
    monkeypatch.setattr(cli.os, "_exit", lambda code: exit_calls.append(code))

    cli._hard_exit_after_serve()

    assert exit_calls == [0]


@pytest.mark.parametrize(
    "entrypoint",
    [cli.serve_command, cli._serve_audio_mode],
    ids=["serve_command", "serve_audio_mode"],
)
def test_serve_entrypoints_invoke_hard_exit_after_uvicorn(entrypoint):
    """Both serve entrypoints MUST reference ``_hard_exit_after_serve``
    in their bytecode — the call sits immediately after the
    ``_run_uvicorn`` dispatch, so dropping it (rename, refactor,
    "cleanup") re-opens the #3495 SIGSEGV-on-shutdown window with no
    failing check elsewhere.
    """
    assert "_hard_exit_after_serve" in entrypoint.__code__.co_names, (
        f"{entrypoint.__name__} no longer calls _hard_exit_after_serve "
        "after the uvicorn dispatch — #3495 regression"
    )
