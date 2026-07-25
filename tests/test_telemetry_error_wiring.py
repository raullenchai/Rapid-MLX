# SPDX-License-Identifier: Apache-2.0
"""End-to-end check that a model-load failure emits an opt-in ``error``
telemetry event (Phase 2.2 error wiring).

Companion to ``test_telemetry_cli.py``: that file pins the lifecycle
(``session_start`` / ``session_end``) wiring; this one pins that the
``error`` event actually lands at a real load-failure call site, carries
the allowlisted ``category`` / ``phase``, and — critically — that the
fingerprint is the only trace of the exception (no model name, no
message text, no filesystem path).

The ``bench`` path loads synchronously via ``mlx_lm.load`` (unlike
``serve``, whose weight load is deferred into the async FastAPI lifespan),
so a missing local model directory reproduces the failure deterministically
and offline.
"""

from __future__ import annotations

import http.server
import importlib
import json
import os
import re
import subprocess
import sys
import threading

import pytest

# Every test here exercises a code path behind the mlx runtime: the CLI
# subprocess (bench), the FastAPI server lifespan, and the tool parser all
# transitively import ``mlx``. Skip the whole module cleanly where mlx is
# not installed — e.g. the Linux ``pr_validate`` / targeted-tests runner —
# so they skip rather than hard-fail with ModuleNotFoundError there. On the
# apple-silicon job (where mlx is present) they run normally.
pytest.importorskip("mlx")


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("RAPID_MLX_TELEMETRY", raising=False)
    import vllm_mlx.telemetry.state as state

    importlib.reload(state)
    return tmp_path


def _run_cli(*args, env_overrides=None, home=None):
    env = os.environ.copy()
    if home is not None:
        env["HOME"] = str(home)
        # Overriding HOME isolates the telemetry consent/client-id files
        # (they live under ``~/.rapid-mlx/``), but it also drops the
        # per-user site-packages (``~/.local/...``) from the child's
        # import path — which on some dev machines is where ``mlx_lm``
        # lives, so ``bench`` would fail at ``import mlx_lm`` BEFORE the
        # load-failure site under test. Propagate the runner's resolved
        # ``sys.path`` via PYTHONPATH so package resolution is decoupled
        # from the HOME override. (No-op in CI, where deps sit in the
        # HOME-independent environment site.)
        env["PYTHONPATH"] = os.pathsep.join(
            [p for p in sys.path if p] + [env.get("PYTHONPATH", "")]
        ).strip(os.pathsep)
    env.pop("RAPID_MLX_TELEMETRY", None)
    if env_overrides:
        env.update(env_overrides)
    return subprocess.run(
        [sys.executable, "-m", "vllm_mlx.cli", *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
        check=False,
    )


def _capture_server():
    """Local HTTP server that records every POSTed telemetry batch.

    Mirrors the harness in ``test_telemetry_cli.py`` — bind port 0 and
    read ``server_port`` to avoid a probe-then-bind race.
    """
    captured: list[dict] = []

    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802 — name dictated by stdlib
            length = int(self.headers.get("content-length", "0"))
            raw = self.rfile.read(length)
            try:
                captured.append(json.loads(raw.decode("utf-8")))
            except Exception:
                captured.append({"_raw": raw[:200].decode("utf-8", "replace")})
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"ok":true}')

        def log_message(self, *_a, **_k):  # silence
            return

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, captured


def _all_events(captured):
    return [
        ev
        for batch in captured
        if isinstance(batch.get("batch"), list)
        for ev in batch["batch"]
    ]


def test_bench_model_load_failure_emits_error_event(fake_home, tmp_path):
    """``bench`` on a directory with no ``config.json`` fails the
    synchronous load; an opt-in ``error`` event must land with the
    allowlisted ``model_load_failure`` / ``startup`` fields."""
    empty_model = tmp_path / "empty-model"
    empty_model.mkdir()

    server, captured = _capture_server()
    port = server.server_port
    try:
        _run_cli("telemetry", "enable", home=fake_home)
        r = _run_cli(
            "bench",
            str(empty_model),
            "--max-tokens",
            "4",
            home=fake_home,
            env_overrides={
                "RAPID_MLX_TELEMETRY_DEBUG": "1",
                "RAPID_MLX_TELEMETRY_ENDPOINT": f"http://127.0.0.1:{port}/v1/events",
            },
        )
        # The load failure is surfaced to the user (non-zero exit) — the
        # telemetry hook must not have swallowed it.
        assert r.returncode != 0, f"expected non-zero exit; stdout={r.stdout}"
    finally:
        server.shutdown()
        server.server_close()

    events = _all_events(captured)
    assert events, f"no telemetry POST captured (stderr={r.stderr})"

    errors = [ev for ev in events if ev.get("event") == "error"]
    assert len(errors) >= 1, (
        f"no error event; events={[e.get('event') for e in events]}"
    )
    err = errors[0]["error"]
    assert err["category"] == "model_load_failure", err
    assert err["phase"] == "startup", err
    # Fingerprint is a 16-hex digest — the ONLY trace of the exception.
    assert re.fullmatch(r"[0-9a-f]{16}", err["fingerprint"]), err

    # Privacy red-line: the offending path / message text must never ride
    # along on ANY captured payload (the error event carries only the
    # bucketed category + fingerprint + phase).
    blob = json.dumps(captured)
    assert str(empty_model) not in blob
    assert "config.json" not in blob
    assert "No such file" not in blob


def test_bench_load_failure_error_event_absent_when_opted_out(fake_home, tmp_path):
    """The same failure emits NOTHING when telemetry is left at its
    default-off state — the consent gate holds on the error path too."""
    empty_model = tmp_path / "empty-model"
    empty_model.mkdir()

    server, captured = _capture_server()
    port = server.server_port
    try:
        # No ``telemetry enable`` — consent stays default-off.
        r = _run_cli(
            "bench",
            str(empty_model),
            "--max-tokens",
            "4",
            home=fake_home,
            env_overrides={
                "RAPID_MLX_TELEMETRY_DEBUG": "1",
                "RAPID_MLX_TELEMETRY_ENDPOINT": f"http://127.0.0.1:{port}/v1/events",
            },
        )
        assert r.returncode != 0
    finally:
        server.shutdown()
        server.server_close()

    assert not _all_events(captured), "opted-out run must emit no telemetry"


async def test_serve_engine_start_failure_emits_model_load_error(monkeypatch):
    """``serve``'s real weight load runs in the async FastAPI lifespan
    (``_engine.start()``), NOT the CLI ``load_model()`` (which only does
    config read + type-detection). A failure there is THE serve model-load
    failure — it must emit ``model_load_failure`` / ``startup`` and still
    re-raise so startup aborts exactly as before.

    Driven in-process via the lifespan async-generator pattern (mirrors
    tests/test_ready_banner_timing.py). ``emit.error`` is patched to capture
    the call so this asserts the wiring, not the (separately-tested) redact
    pipeline.
    """
    import vllm_mlx._signal_observability as _sigobs
    import vllm_mlx.server as vllm_server
    from vllm_mlx.config import get_config
    from vllm_mlx.telemetry import emit as _emit

    # Keep the test hermetic: don't install real SIGTERM/SIGHUP handlers on
    # the pytest process (the failure aborts startup mid-lifespan, so the
    # full-lifespan restore in the banner test doesn't apply here).
    monkeypatch.setattr(_sigobs, "install_signal_observability", lambda: False)

    calls = []
    monkeypatch.setattr(_emit, "error", lambda **kw: calls.append(kw))

    class _BoomEngine:
        _loaded = False

        async def start(self):
            raise RuntimeError("checkpoint shards missing")

    cfg = get_config()
    saved = (vllm_server._engine, cfg.bind_host, cfg.bind_port, cfg.ready)
    vllm_server._engine = _BoomEngine()
    try:
        agen = vllm_server.lifespan(vllm_server.app)
        with pytest.raises(RuntimeError, match="checkpoint shards missing"):
            await agen.__anext__()  # startup → engine.start() raises → re-raised
    finally:
        vllm_server._engine, cfg.bind_host, cfg.bind_port, cfg.ready = saved

    assert any(
        c.get("category") == "model_load_failure" and c.get("phase") == "startup"
        for c in calls
    ), calls
    # The raw exception is handed to emit.error for fingerprinting only;
    # its message never reaches the payload (redact.fingerprint_traceback).
    assert isinstance(calls[0].get("exc"), RuntimeError)


async def test_serve_shutdown_failure_emits_shutdown_traceback(monkeypatch):
    """A crash during lifespan teardown (cache save / MCP close / engine
    stop) must emit ``shutdown_traceback`` / ``shutdown`` and re-raise so
    the shutdown path is unchanged."""
    import vllm_mlx._signal_observability as _sigobs
    import vllm_mlx.server as vllm_server
    from vllm_mlx.config import get_config
    from vllm_mlx.telemetry import emit as _emit

    monkeypatch.setattr(_sigobs, "install_signal_observability", lambda: False)

    calls = []
    monkeypatch.setattr(_emit, "error", lambda **kw: calls.append(kw))

    class _EngineBoomOnStop:
        # ``_loaded=True`` skips ``start()`` so startup reaches ``yield``;
        # no ``save_cache_to_disk`` attr so the cache-save step is a no-op;
        # warmup calls are swallowed (non-fatal) by the lifespan.
        _loaded = True

        async def stop(self):
            raise RuntimeError("engine stop boom")

    cfg = get_config()
    saved = (
        vllm_server._engine,
        cfg.bind_host,
        cfg.bind_port,
        cfg.ready,
        getattr(cfg, "draining", False),
    )
    vllm_server._engine = _EngineBoomOnStop()
    cfg.bind_host = None  # suppress the Ready banner
    cfg.bind_port = None
    try:
        agen = vllm_server.lifespan(vllm_server.app)
        await agen.__anext__()  # startup → yield
        with pytest.raises(RuntimeError, match="engine stop boom"):
            await agen.__anext__()  # shutdown → stop() raises → re-raised
    finally:
        (
            vllm_server._engine,
            cfg.bind_host,
            cfg.bind_port,
            cfg.ready,
            cfg.draining,
        ) = saved

    assert any(
        c.get("category") == "shutdown_traceback" and c.get("phase") == "shutdown"
        for c in calls
    ), calls


def test_tool_parser_crash_emits_tool_parse_error(monkeypatch):
    """When the configured tool parser raises while extracting calls, the
    request path falls back to the generic text parser AND emits a
    ``tool_parse`` / ``chat`` error. The fallback must still return normally
    (the crash is never surfaced to the user)."""
    from vllm_mlx.config import get_config
    from vllm_mlx.service import helpers
    from vllm_mlx.telemetry import emit as _emit
    from vllm_mlx.tool_parsers.abstract_tool_parser import ToolParserManager

    calls = []
    monkeypatch.setattr(_emit, "error", lambda **kw: calls.append(kw))

    class _BoomParser:
        def __init__(self, tokenizer=None):
            pass

        def reset(self):
            pass

        def extract_tool_calls(self, *a, **k):
            raise ValueError("malformed tool-call markup")

    monkeypatch.setattr(
        ToolParserManager, "get_tool_parser", staticmethod(lambda name: _BoomParser)
    )

    cfg = get_config()
    saved = (cfg.enable_auto_tool_choice, cfg.tool_call_parser)
    cfg.enable_auto_tool_choice = True
    cfg.tool_call_parser = "hermes"
    try:
        content, tool_calls = helpers._parse_tool_calls_with_parser(
            "here is some plain assistant text with no tool call", request=None
        )
    finally:
        cfg.enable_auto_tool_choice, cfg.tool_call_parser = saved

    assert any(
        c.get("category") == "tool_parse" and c.get("phase") == "chat" for c in calls
    ), calls
    # Fallback returned normally — the parser crash was swallowed, not raised.
    assert isinstance(content, str)
