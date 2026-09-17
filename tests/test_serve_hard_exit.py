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
    """Force the helper's production path, with the atexit pass recorded.

    Two jobs:

    * The guard fires only when BOTH pytest is imported in this process
      AND ``PYTEST_CURRENT_TEST`` is set (the in-process harness shape).
      Removing the module from ``sys.modules`` alone is sufficient at
      the call phase: pytest re-sets ``PYTEST_CURRENT_TEST`` when the
      call phase starts, so any setup-phase ``delenv`` would be undone
      anyway (verified with a spy plugin — codex round-2). A real serve
      process — including a pytest-spawned child that inherits the env
      var — never has pytest imported, so the guard stays false for it.
    * ``atexit._run_exitfuncs`` is replaced with a recorder (codex
      round-3 BLOCKING #1): letting the helper execute the REAL atexit
      pass mid-suite permanently consumes every exit handler registered
      by pytest and its plugins, breaking teardown for the rest of the
      session (and aborting the interpreter at exit — observed as a
      SIGABRT'd full-unit run). Returns the event list so tests can
      assert hook/exit ordering.
    """
    monkeypatch.delitem(sys.modules, "pytest", raising=False)
    events: list[str] = []
    monkeypatch.setattr(cli.atexit, "_run_exitfuncs", lambda: events.append("atexit"))
    return events


def test_hard_exit_flushes_streams_and_exits_zero(monkeypatch, production_exit_context):
    """Success path: run the atexit pass, flush stdout/stderr, then
    ``os._exit(0)`` — in that exact order (codex round-3 BLOCKING #1;
    round-6 BLOCKING: the flushes must be recorded INTO the same event
    stream as the exit, otherwise a refactor moving both flushes after
    the mocked ``os._exit`` — unreachable in a real process — would
    still pass on flushed booleans alone).
    """
    events = production_exit_context

    class _FlushRecorder:
        def __init__(self, event: str) -> None:
            self._event = event

        def flush(self) -> None:
            events.append(self._event)

    monkeypatch.setattr(cli.sys, "stdout", _FlushRecorder("stdout_flush"))
    monkeypatch.setattr(cli.sys, "stderr", _FlushRecorder("stderr_flush"))
    monkeypatch.setattr(cli.os, "_exit", lambda code: events.append(f"exit:{code}"))

    cli._hard_exit_after_serve()

    assert events == ["atexit", "stdout_flush", "stderr_flush", "exit:0"], (
        f"expected [atexit pass, stdout flush, stderr flush, one os._exit(0)], "
        f"got {events!r}"
    )


def test_hard_exit_is_skipped_in_inprocess_pytest_harness(monkeypatch):
    """In-process suites drive serve through stubbed uvicorn runs; the
    guard must keep ``os._exit`` — and the atexit pass — away from the
    pytest process (an ``os._exit(0)`` mid-suite would end the run green
    while skipping every later test; a consumed atexit registry breaks
    teardown for the rest of the session). The guard requires BOTH the
    imported module and the pytest-owned env var — the in-process
    harness shape.
    """
    events: list[str] = []
    monkeypatch.setattr(cli.os, "_exit", lambda code: events.append(f"exit:{code}"))
    monkeypatch.setattr(cli.atexit, "_run_exitfuncs", lambda: events.append("atexit"))
    monkeypatch.setitem(sys.modules, "pytest", types.ModuleType("pytest"))
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "tests/test_serve_hard_exit.py::t")

    cli._hard_exit_after_serve()

    assert events == []


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
    events = production_exit_context

    class _NullStream:
        def flush(self) -> None:
            pass

    monkeypatch.setattr(cli.sys, "stdout", _NullStream())
    monkeypatch.setattr(cli.sys, "stderr", _NullStream())
    monkeypatch.setattr(cli.os, "_exit", lambda code: events.append(f"exit:{code}"))
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

    assert events == ["atexit", "exit:0"], (
        f"{scenario}: hard exit (after the atexit pass) must fire when only "
        "one harness signal is present — production processes never match both"
    )


@pytest.mark.parametrize("broken_stream", ["stdout", "stderr"])
def test_hard_exit_reports_flush_failure_as_exit_120(
    monkeypatch, production_exit_context, broken_stream
):
    """A dead stream (closed stderr at teardown, broken pipe on stdout)
    must not become a traceback-driven crash, but it also must not be
    silently promoted to a successful exit: mirror CPython's
    finalization convention and report 120 (codex round-5 BLOCKING).
    """

    class _Broken:
        def flush(self) -> None:
            raise ValueError("stream closed")

    class _Null:
        def flush(self) -> None:
            pass

    events = production_exit_context
    monkeypatch.setattr(
        cli.sys, "stdout", _Broken() if broken_stream == "stdout" else _Null()
    )
    monkeypatch.setattr(
        cli.sys, "stderr", _Broken() if broken_stream == "stderr" else _Null()
    )
    monkeypatch.setattr(cli.os, "_exit", lambda code: events.append(f"exit:{code}"))

    cli._hard_exit_after_serve()

    assert events == ["atexit", "exit:120"]


def test_hard_exit_survives_a_crashing_exit_hook(monkeypatch, production_exit_context):
    """A misbehaving atexit hook (raise SystemExit mid-drain, e.g.) must
    not skip the os._exit below and re-enter the interpreter-finalization
    crash path this helper exists to prevent — the hard exit is the
    invariant, the atexit pass is best-effort (see the helper's
    BaseException rationale).
    """

    def _boom() -> None:
        raise SystemExit("telemetry drain failed")

    class _Null:
        def flush(self) -> None:
            pass

    events = production_exit_context
    monkeypatch.setattr(cli.atexit, "_run_exitfuncs", _boom)
    monkeypatch.setattr(cli.sys, "stdout", _Null())
    monkeypatch.setattr(cli.sys, "stderr", _Null())
    monkeypatch.setattr(cli.os, "_exit", lambda code: events.append(f"exit:{code}"))

    cli._hard_exit_after_serve()

    assert events == ["exit:0"], (
        f"a crashing exit hook must still end in the hard exit, got {events!r}"
    )


@pytest.mark.requires_mlx
def test_legacy_server_main_hard_exits_after_uvicorn(monkeypatch):
    """Behavioral pin for the legacy ``python -m rapid_mlx.server``
    entrypoint (codex round-4 NIT): its ``main()`` runs its own
    ``uvicorn.run`` and must actually invoke ``_hard_exit_after_serve``
    afterwards — otherwise that documented entrypoint keeps the
    unfixed #3495 finalization crash path. Mirrors the stubbing shape
    of ``test_server_main_no_mllm_skips_routing_config_fail_fast``.
    """
    from rapid_mlx import server as server_mod

    events: list[str] = []

    monkeypatch.setattr(server_mod, "_ensure_routing_config", lambda *_a, **_kw: None)
    monkeypatch.setattr(server_mod, "load_model", lambda *_a, **_kw: None)
    monkeypatch.setattr("rapid_mlx.cli._port_preflight_or_die", lambda *_a, **_kw: None)
    import uvicorn as _uvicorn

    monkeypatch.setattr(_uvicorn, "run", lambda *_a, **_kw: events.append("uvicorn"))
    monkeypatch.setattr(
        "rapid_mlx.cli._hard_exit_after_serve",
        lambda: events.append("hard_exit"),
    )
    monkeypatch.setattr(
        "rapid_mlx._version_check.prompt_upgrade_if_available", lambda: False
    )
    monkeypatch.setattr(
        "rapid_mlx._version_check.print_staleness_warning_if_any",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rapid_mlx.server",
            "--model",
            "some/uncached-hybrid-vlm-4bit",
            "--no-mllm",
        ],
    )

    server_mod.main()

    assert events == ["uvicorn", "hard_exit"], (
        f"expected exactly ['uvicorn', 'hard_exit'], got {events!r} — "
        "the legacy server entrypoint must hard-exit after the uvicorn "
        "dispatch returns (#3495)"
    )


def test_audio_serve_mode_hard_exits_immediately_after_uvicorn(
    monkeypatch, production_exit_context
):
    """Behavioral pin for ``_serve_audio_mode`` (``rapid-mlx serve kokoro``):
    it must invoke ``_hard_exit_after_serve`` immediately after
    ``_run_uvicorn`` returns — same contract as the text entrypoints.

    Hermetic on purpose: ``rapid_mlx.server`` is replaced in
    ``sys.modules`` with a stub so this test stays runnable on the
    no-MLX CI lane (the audio alias suite covers the same contract
    through the real module graph on MLX machines — and is skipped
    there, which is exactly why the changed-lines gate needs this
    Linux-runnable twin).
    """
    import rapid_mlx

    events = production_exit_context
    stub_server = types.ModuleType("rapid_mlx.server")
    stub_server.app = object()
    stub_server.configure_logging = lambda _level: "info"
    stub_server._resolve_api_key = lambda _key: None
    stub_server.configure_cors_from_env = lambda _origins: None
    stub_server.configure_trusted_hosts = lambda _hosts: None
    stub_server.register_audio_routes_if_enabled = lambda: None
    stub_server._sync_config = lambda: None
    monkeypatch.setitem(sys.modules, "rapid_mlx.server", stub_server)
    monkeypatch.setattr(rapid_mlx, "server", stub_server, raising=False)
    monkeypatch.setattr(cli, "_port_preflight_or_die", lambda *_a, **_kw: None)
    monkeypatch.setattr(
        cli, "_run_uvicorn", lambda *_a, **_kw: events.append("uvicorn")
    )
    monkeypatch.setattr(
        cli, "_hard_exit_after_serve", lambda: events.append("hard_exit")
    )

    args = types.SimpleNamespace(
        model="kokoro",
        host="127.0.0.1",
        port=8900,
        log_level="info",
        api_key=None,
        timeout=300,
        rate_limit=0,
        cors_origins=None,
        embedding_model=None,
    )
    entry = types.SimpleNamespace(
        type="tts", alias="kokoro", hf_id="dummy/kokoro-tts", default_voice=None
    )

    cli._serve_audio_mode(args, entry)

    assert events == ["uvicorn", "hard_exit"], (
        f"expected exactly ['uvicorn', 'hard_exit'], got {events!r} — "
        "the audio serve mode must hard-exit after the uvicorn dispatch "
        "returns (#3495)"
    )


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
