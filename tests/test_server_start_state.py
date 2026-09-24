# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import builtins
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from rapid_mlx import cli, server
from rapid_mlx.runtime.primary_lifecycle import PrimaryModelLifecycle
from rapid_mlx.service import helpers
from rapid_mlx.telemetry import registry, server_start

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _reset_state():
    server_start._reset_for_tests()
    yield
    server_start._reset_for_tests()


def _capture(monkeypatch):
    events: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr("rapid_mlx.telemetry.track._upload_allowed", lambda: True)
    monkeypatch.setattr(
        "rapid_mlx.telemetry.track.track",
        lambda event, props: events.append((event, dict(props))),
    )
    return events


class _CaptureHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        length = int(self.headers["Content-Length"])
        self.server.bodies.append(self.rfile.read(length))  # type: ignore[attr-defined]
        self.send_response(200)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"{}")

    def log_message(self, format: str, *args: object) -> None:
        pass


def test_stale_marker_reports_previous_run_unterminated_once_to_loopback(tmp_path):
    home = tmp_path / "home"
    state_dir = home / ".rapid-mlx" / "state"
    state_dir.mkdir(parents=True)
    (state_dir / "serve-inflight.json").write_text(
        json.dumps(
            {
                "pid": 99_999_999,
                "utc_start": "2026-09-24T00:00:00Z",
                "app_version": "0.15.1",
            }
        ),
        encoding="utf-8",
    )
    sink = HTTPServer(("127.0.0.1", 0), _CaptureHandler)
    sink.bodies = []  # type: ignore[attr-defined]
    thread = threading.Thread(target=sink.serve_forever, daemon=True)
    thread.start()
    program = """
import rapid_mlx
from rapid_mlx.telemetry import build_gate, common_props, consent_runtime, posthog_sender, server_start, state
from rapid_mlx.telemetry.build_gate import ReleaseStamp
from rapid_mlx.telemetry.common_props import PlatformFacts

rapid_mlx.__version__ = "0.15.1"
stamp = ReleaseStamp(channel="stable", posthog_key="phc_" + "a" * 32)
build_gate.official_build = lambda: stamp
consent_runtime.upload_allowed = lambda: True
common_props.read_platform_facts = lambda: PlatformFacts(
    os="darwin", os_version="25.3", arch="arm64", chip="m1-pro",
    memory_gb=32, python_version="3.11"
)
state.get_or_create_client_id = lambda: "6f1b1d3e-4a2b-4c9d-8e7f-0a1b2c3d4e5f"
state.session_id = lambda: "0a1b2c3d-4e5f-6071-8293-a4b5c6d7e8f9"
server_start.attempted("qwen3.5-4b-4bit", load_policy="eager")
server_start.attempted("ignored", load_policy="lazy")
posthog_sender.get_sender().flush(5.0)
"""
    env = dict(os.environ, HOME=str(home))
    env["RAPID_MLX_POSTHOG_URL"] = f"http://127.0.0.1:{sink.server_port}/batch/"
    for name in (
        "CI",
        "GITHUB_ACTIONS",
        "GITLAB_CI",
        "CIRCLECI",
        "TRAVIS",
        "BUILDKITE",
        "JENKINS_URL",
        "TEAMCITY_VERSION",
        "RAPID_MLX_TELEMETRY",
        "DO_NOT_TRACK",
    ):
        env.pop(name, None)
    try:
        proc = subprocess.run(
            [sys.executable, "-c", program],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    finally:
        sink.shutdown()
        thread.join(timeout=2.0)
        sink.server_close()

    assert proc.returncode == 0, proc.stderr
    attempted = [
        item
        for body in sink.bodies  # type: ignore[attr-defined]
        for item in json.loads(body)["batch"]
        if item["event"] == "server_start_state"
    ]
    assert len(attempted) == 1
    assert attempted[0]["properties"]["previous_run_unterminated"] is True


def test_attempted_then_ready_exactly_once(monkeypatch):
    events = _capture(monkeypatch)
    server_start.attempted("qwen3.5-4b-4bit", load_policy="eager")
    server_start.attempted("ignored", load_policy="lazy")
    server_start.ready()
    server_start.ready()
    server_start.failed("bind")

    assert [props["state"] for _, props in events] == ["attempted", "ready"]
    assert all(name == "server_start_state" for name, _ in events)
    assert all("failure_stage" not in props for _, props in events)
    assert events[0][1] == {
        "state": "attempted",
        "model_type": "llm",
        "load_policy": "eager",
    }
    assert events[1][1] == {
        "state": "ready",
        "model_type": "llm",
        "load_policy": "eager",
    }
    assert all(registry.validate(name, props) == props for name, props in events)


@pytest.mark.parametrize(
    "terminal", [server_start.ready, lambda: server_start.failed("bind")]
)
def test_marker_written_at_attempted_and_removed_at_terminal_with_telemetry_disabled(
    monkeypatch, tmp_path, terminal
):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("rapid_mlx.telemetry.track._upload_allowed", lambda: False)

    server_start.attempted("qwen3.5-4b-4bit", load_policy="eager")

    marker = tmp_path / ".rapid-mlx" / "state" / "serve-inflight.json"
    assert marker.is_file()
    assert marker.stat().st_mode & 0o777 == 0o600
    assert marker.parent.stat().st_mode & 0o777 == 0o700
    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert payload["pid"] == os.getpid()
    assert payload["utc_start"].endswith("Z")
    assert isinstance(payload["app_version"], str)

    terminal()
    assert not marker.exists()


def test_live_marker_is_not_reported_overwritten_or_removed(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    marker = tmp_path / ".rapid-mlx" / "state" / "serve-inflight.json"
    marker.parent.mkdir(parents=True)
    original = {
        "pid": os.getpid(),
        "utc_start": "2026-09-24T00:00:00Z",
        "app_version": "0.15.1",
    }
    marker.write_text(json.dumps(original), encoding="utf-8")
    events = _capture(monkeypatch)

    server_start.attempted("qwen3.5-4b-4bit", load_policy="eager")
    server_start.ready()

    assert all("previous_run_unterminated" not in props for _, props in events)
    assert json.loads(marker.read_text(encoding="utf-8")) == original


@pytest.mark.parametrize(
    "stage",
    ["resolve", "download", "preflight", "prepare", "engine_start", "bind"],
)
def test_each_failure_stage_is_the_only_terminal(monkeypatch, stage):
    events = _capture(monkeypatch)
    server_start.attempted("flux-schnell", load_policy="lazy")
    server_start.failed(stage)
    server_start.ready()
    server_start.failed("bind")

    assert [props["state"] for _, props in events] == ["attempted", "failed"]
    assert events[-1][1]["failure_stage"] == stage
    assert "failure_stage" not in events[0][1]


def test_invalid_values_are_omitted_or_ignored(monkeypatch):
    events = _capture(monkeypatch)
    server_start.attempted(SimpleNamespace(), load_policy="surprise")
    server_start.failed("surprise")
    server_start.ready()

    assert events == [
        (
            "server_start_state",
            {"state": "attempted", "model_type": "other"},
        ),
        (
            "server_start_state",
            {"state": "ready", "model_type": "other"},
        ),
    ]


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("qwen3.5-4b-4bit", "eager"),
        ("flux-schnell", "lazy"),
        ("ltx-2.3-mlx-q4", "lazy"),
        ("kokoro", "lazy"),
    ],
)
def test_load_policy_matches_adapter_lanes(model, expected):
    assert server_start.load_policy(model) == expected


def test_explicit_lazy_load_overrides_adapter_policy():
    assert server_start.load_policy("qwen3.5-4b-4bit") == "eager"
    assert server_start.load_policy("qwen3.5-4b-4bit", lazy_load=True) == "lazy"


@pytest.mark.asyncio
async def test_post_ready_lazy_503_does_no_server_start_telemetry_work(monkeypatch):
    events = _capture(monkeypatch)

    class FailingLazyEngine:
        _loaded = False

        async def start(self):
            raise RuntimeError("lazy load failed")

        async def stop(self):
            self._loaded = False

    engine = FailingLazyEngine()
    lifecycle = PrimaryModelLifecycle(engine, lazy_load=True)
    monkeypatch.setattr(
        helpers,
        "get_config",
        lambda: SimpleNamespace(primary_model_lifecycle=lifecycle),
    )
    server_start.attempted("qwen3.5-4b-4bit", load_policy="lazy")
    server_start.ready()

    imports: list[str] = []
    real_import = builtins.__import__

    def reject_server_start_import(name, *args, **kwargs):
        if name == "rapid_mlx.telemetry.server_start":
            imports.append(name)
            raise AssertionError("request path imported server_start telemetry")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "rapid_mlx.telemetry.server_start")
    monkeypatch.setattr(builtins, "__import__", reject_server_start_import)

    with pytest.raises(HTTPException) as caught:
        await helpers.ensure_engine_ready(engine)

    assert caught.value.status_code == 503
    assert imports == []
    assert [props["state"] for _, props in events] == ["attempted", "ready"]


def test_emitter_base_exception_cannot_change_host_result(monkeypatch):
    def explode(*_args, **_kwargs):
        raise SystemExit(91)

    monkeypatch.setattr("rapid_mlx.telemetry.track._upload_allowed", lambda: True)
    monkeypatch.setattr("rapid_mlx.telemetry.track.track", explode)
    server_start.attempted("qwen3.5-4b-4bit", load_policy="eager")
    server_start.failed("preflight")


def test_disabled_telemetry_never_initializes_sender(monkeypatch):
    monkeypatch.setattr("rapid_mlx.telemetry.track._upload_allowed", lambda: False)

    def explode():
        raise AssertionError("disabled telemetry initialized the sender")

    monkeypatch.setattr("rapid_mlx.telemetry.posthog_sender.get_sender", explode)
    server_start.attempted("qwen3.5-4b-4bit", load_policy="eager")
    server_start.ready()


def test_attempt_setup_failure_is_inert(monkeypatch):
    monkeypatch.setattr("rapid_mlx.telemetry.track._upload_allowed", lambda: True)
    monkeypatch.setattr(
        "rapid_mlx.telemetry.posthog_sender.install_atexit",
        lambda: (_ for _ in ()).throw(RuntimeError("sender unavailable")),
    )

    server_start.attempted("qwen3.5-4b-4bit", load_policy="eager")


def test_load_policy_falls_back_to_eager_when_classification_fails(monkeypatch):
    monkeypatch.setattr(
        "rapid_mlx.telemetry.model_events.model_type",
        lambda _model: (_ for _ in ()).throw(RuntimeError("classifier unavailable")),
    )

    assert server_start.load_policy("qwen3.5-4b-4bit") == "eager"


def test_cli_main_emits_attempted_before_serve_preflight(monkeypatch, tmp_path):
    from rapid_mlx.telemetry import consent_runtime

    model = tmp_path / "local-model"
    model.mkdir()
    calls: list[tuple[str, object]] = []
    monkeypatch.setattr(consent_runtime, "startup", lambda **_kwargs: None)
    monkeypatch.setattr(cli, "_start_v2_lifecycle", lambda command: None)
    monkeypatch.setattr(
        server_start,
        "load_policy",
        lambda selected, *, lazy_load: calls.append(("policy", selected)) or "eager",
    )
    monkeypatch.setattr(
        server_start,
        "attempted",
        lambda selected, *, load_policy: calls.append(("attempted", load_policy)),
    )
    monkeypatch.setattr(
        cli,
        "serve_command",
        lambda args: calls.append(("serve", args.model)),
    )
    monkeypatch.setattr(sys, "argv", ["rapid-mlx", "serve", str(model)])

    cli.main()

    assert calls == [
        ("policy", str(model)),
        ("attempted", "eager"),
        ("serve", str(model)),
    ]


def test_failure_context_preserves_original_exception(monkeypatch):
    events = _capture(monkeypatch)
    server_start.attempted("qwen3.5-4b-4bit", load_policy="eager")
    with (
        pytest.raises(SystemExit) as caught,
        server_start.failure_stage("preflight"),
    ):
        raise SystemExit(2)

    assert caught.value.code == 2
    assert [props["state"] for _, props in events] == ["attempted", "failed"]
    assert events[-1][1]["failure_stage"] == "preflight"


def _stub_download_entry(monkeypatch):
    monkeypatch.setattr(cli, "_cache_runnability", lambda _model: False)
    monkeypatch.setattr(cli, "_offline_hub_mode_active", lambda: False)
    monkeypatch.setattr(cli, "_check_disk_space", lambda *_a, **_kw: None)
    monkeypatch.setattr(cli, "_try_mirror_prefetch", lambda *_a, **_kw: False)
    monkeypatch.setattr(
        "rapid_mlx.telemetry.model_events.emit_model_pull_failed",
        lambda *_a, **_kw: None,
    )


def test_resolve_timeout_emits_resolve_before_preserving_exit(monkeypatch):
    events = _capture(monkeypatch)
    _stub_download_entry(monkeypatch)
    monkeypatch.setattr(
        "rapid_mlx._download_gate.call_with_deadline",
        lambda *_a, **_kw: (_ for _ in ()).throw(TimeoutError()),
    )
    server_start.attempted("qwen3.5-4b-4bit", load_policy="eager")

    with pytest.raises(SystemExit) as caught:
        cli._ensure_model_downloaded("owner/model")

    assert caught.value.code == 1
    assert [(props["state"], props.get("failure_stage")) for _, props in events] == [
        ("attempted", None),
        ("failed", "resolve"),
    ]


def test_definitive_download_404_emits_download(monkeypatch):
    events = _capture(monkeypatch)
    _stub_download_entry(monkeypatch)
    monkeypatch.setattr(
        "rapid_mlx._download_gate.call_with_deadline",
        lambda _func, _timeout, *_a, **_kw: SimpleNamespace(sha="abc", siblings=[]),
    )
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda *_a, **_kw: (_ for _ in ()).throw(RuntimeError("404 missing")),
    )
    server_start.attempted("qwen3.5-4b-4bit", load_policy="eager")

    with pytest.raises(RuntimeError, match="not found on HuggingFace"):
        cli._ensure_model_downloaded("owner/model")

    assert [(props["state"], props.get("failure_stage")) for _, props in events] == [
        ("attempted", None),
        ("failed", "download"),
    ]


@pytest.mark.parametrize(
    "decorator", [cli._capture_start_failures, server._capture_start_failures]
)
def test_entrypoint_guard_never_replaces_host_exception(monkeypatch, decorator):
    monkeypatch.setattr(
        server_start,
        "fail_current",
        lambda: (_ for _ in ()).throw(SystemExit(91)),
    )

    @decorator
    def fail_host():
        raise ValueError("host failure")

    with pytest.raises(ValueError, match="host failure"):
        fail_host()
