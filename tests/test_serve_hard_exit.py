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

import os
import sys
import types
from pathlib import Path

import pytest

from rapid_mlx import cli


@pytest.fixture
def production_exit_context(monkeypatch):
    """Force the helper's production path.

    The guard fires only when BOTH pytest is imported in this process
    AND ``PYTEST_CURRENT_TEST`` is set (the in-process harness shape).
    Removing the module from ``sys.modules`` alone is sufficient at the
    call phase: pytest re-sets ``PYTEST_CURRENT_TEST`` when the call
    phase starts, so any setup-phase ``delenv`` would be undone anyway
    (verified with a spy plugin — codex round-2). A real serve process
    — including a pytest-spawned child that inherits the env var —
    never has pytest imported, so the guard stays false for it.
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


def test_hard_exit_is_skipped_in_inprocess_pytest_harness(monkeypatch):
    """In-process suites drive serve through stubbed uvicorn runs; the
    guard must keep ``os._exit`` away from the pytest process (an
    ``os._exit(0)`` mid-suite would end the run green while skipping
    every later test). The guard requires BOTH the imported module and
    the pytest-owned env var — the in-process harness shape.
    """
    exit_calls: list[int] = []
    monkeypatch.setattr(cli.os, "_exit", lambda code: exit_calls.append(code))
    monkeypatch.setitem(sys.modules, "pytest", types.ModuleType("pytest"))
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "tests/test_serve_hard_exit.py::t")

    cli._hard_exit_after_serve()

    assert exit_calls == []


@pytest.mark.parametrize(
    "scenario",
    [
        # pytest-spawned serve child: env inherited, module absent
        "env_only",
        # embedded/instrumented server importing pytest, not driven by it
        "module_only",
    ],
)
def test_hard_exit_still_fires_without_full_harness_shape(
    monkeypatch, production_exit_context, scenario
):
    """A single pytest signal must NOT disable the hard exit: a real
    server process never matches both (codex round-1 NIT — an env-only
    or module-only process must take the production exit path).
    """
    exit_calls: list[int] = []

    class _NullStream:
        def flush(self) -> None:
            pass

    monkeypatch.setattr(cli.sys, "stdout", _NullStream())
    monkeypatch.setattr(cli.sys, "stderr", _NullStream())
    monkeypatch.setattr(cli.os, "_exit", lambda code: exit_calls.append(code))
    if scenario == "env_only":
        # The fixture already removed the module; pytest guarantees
        # PYTEST_CURRENT_TEST is set at the call phase, matching the
        # pytest-spawned-serve-child shape.
        pass
    else:
        # Remove the env half AT CALL TIME — setup-phase delenv is
        # undone by pytest's call-phase bookkeeping, so patch the env
        # mapping the helper actually reads instead.
        monkeypatch.setattr(
            cli.os,
            "environ",
            {k: v for k, v in os.environ.items() if k != "PYTEST_CURRENT_TEST"},
        )
        monkeypatch.setitem(sys.modules, "pytest", types.ModuleType("pytest"))

    cli._hard_exit_after_serve()

    assert exit_calls == [0], (
        f"{scenario}: hard exit must fire when only one harness signal "
        "is present — production processes never match both"
    )


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


def _entrypoint_call_sequence(entrypoint_name: str) -> list[str]:
    """Statement-level call sequence of a serve entrypoint's TOP-LEVEL body.

    Reads the AST (not bytecode co_names): a co_names check would still
    pass if the helper were merely referenced, called BEFORE
    ``_run_uvicorn``, or unreachable — codex round-1 BLOCKING #1. Only
    the function's own top-level statements count (``ast.walk`` would
    interleave nested function bodies in BFS order and scramble the
    sequence).
    """
    import ast

    source = Path(cli.__file__).read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == entrypoint_name:
            calls: list[str] = []
            for stmt in node.body:
                if (
                    isinstance(stmt, ast.Expr)
                    and isinstance(stmt.value, ast.Call)
                    and isinstance(stmt.value.func, ast.Name)
                ):
                    calls.append(stmt.value.func.id)
            return calls
    raise AssertionError(f"entrypoint {entrypoint_name} not found in cli.py")


@pytest.mark.parametrize(
    "entrypoint",
    ["serve_command", "_serve_audio_mode"],
    ids=["serve_command", "serve_audio_mode"],
)
def test_serve_entrypoints_call_hard_exit_immediately_after_uvicorn(entrypoint):
    """Both serve entrypoints MUST dispatch uvicorn and then IMMEDIATELY
    hard-exit: the final two statement-level calls in the body must be
    ``_run_uvicorn`` → ``_hard_exit_after_serve``, in that order.
    Reordering or dropping the hard exit silently re-opens the #3495
    SIGSEGV-on-shutdown window on macOS 15.
    """
    calls = _entrypoint_call_sequence(entrypoint)
    assert calls[-2:] == ["_run_uvicorn", "_hard_exit_after_serve"], (
        f"{entrypoint} statement-level call tail is {calls[-4:]!r}; expected "
        "['...', '_run_uvicorn', '_hard_exit_after_serve'] — #3495 regression"
    )
