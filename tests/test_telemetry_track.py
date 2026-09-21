# SPDX-License-Identifier: Apache-2.0
"""Mutation pins for the telemetry v2 emit and lifecycle seam."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import uuid
from collections.abc import Iterator, Mapping
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

import rapid_mlx
import rapid_mlx.cli as cli
import rapid_mlx.server as server_module
from rapid_mlx.telemetry import consent_runtime, emit, posthog_sender, state
from rapid_mlx.telemetry import track as track_module
from rapid_mlx.telemetry.build_gate import ReleaseStamp
from rapid_mlx.telemetry.common_props import PlatformFacts
from rapid_mlx.telemetry.consent_decision import (
    Decision,
    ProcessRole,
    StoredConsent,
    WriteBack,
)

REAL_UPLOAD_ALLOWED = consent_runtime.upload_allowed

REPO_ROOT = Path(__file__).resolve().parents[1]
STAMP = ReleaseStamp(channel="stable", posthog_key="phc_" + "a" * 32)
INSTALL_ID = "6f1b1d3e-4a2b-4c9d-8e7f-0a1b2c3d4e5f"
SESSION_ID = "0a1b2c3d-4e5f-6071-8293-a4b5c6d7e8f9"
FACTS = PlatformFacts(
    os="darwin",
    os_version="25.3",
    arch="arm64",
    chip="m3-ultra",
    memory_gb=64,
    python_version="3.11",
)


class RecordingSender:
    def __init__(self) -> None:
        self.calls = 0
        self.items: list[dict[str, object]] = []

    def capture(self, item: Mapping[str, object]) -> bool:
        self.calls += 1
        self.items.append(dict(item))
        return True


class ExplodingMapping(Mapping[str, object]):
    def __getitem__(self, key: str) -> object:
        raise RuntimeError("exploded getitem")

    def __iter__(self) -> Iterator[str]:
        raise RuntimeError("exploded iter")

    def __len__(self) -> int:
        return 1


@pytest.fixture(autouse=True)
def isolated_emit(monkeypatch, tmp_path):
    for name in (state.ENV_VAR, state.DO_NOT_TRACK_ENV, *state.CI_ENV_VARS):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(rapid_mlx, "__version__", "0.15.1")
    monkeypatch.setattr(track_module.common_props, "read_platform_facts", lambda: FACTS)
    monkeypatch.setattr(state, "get_or_create_client_id", lambda: INSTALL_ID)
    monkeypatch.setattr(emit, "session_id", lambda: SESSION_ID)
    monkeypatch.setattr(track_module.build_gate, "official_build", lambda: STAMP)
    monkeypatch.setattr(consent_runtime, "upload_allowed", lambda: True)
    monkeypatch.setattr(
        track_module.store, "days_since_first_run_bucket", lambda: "7-29"
    )
    track_module._reset_for_tests()
    consent_runtime._reset_runtime_state_for_tests()
    posthog_sender._reset_for_tests()
    state.set_cli_kill_switch(False)
    yield
    track_module._reset_for_tests()
    consent_runtime._reset_runtime_state_for_tests()
    posthog_sender._reset_for_tests()
    state.set_cli_kill_switch(False)


def inject_sender(monkeypatch) -> RecordingSender:
    sender = RecordingSender()
    monkeypatch.setattr(posthog_sender, "get_sender", lambda: sender)
    return sender


def test_track_swallows_builder_failure_and_hostile_mapping(monkeypatch):
    sender = inject_sender(monkeypatch)
    original_builder = track_module.common_props.build_common_props
    monkeypatch.setattr(
        track_module.common_props,
        "build_common_props",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("builder exploded")),
    )
    track_module.track("app_opened", {})
    monkeypatch.setattr(
        track_module.common_props, "build_common_props", original_builder
    )
    track_module.track("app_opened", ExplodingMapping())
    assert sender.items == []


def test_keyboard_interrupt_propagates(monkeypatch):
    inject_sender(monkeypatch)
    monkeypatch.setattr(
        track_module.common_props,
        "build_common_props",
        lambda **kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    with pytest.raises(KeyboardInterrupt):
        track_module.track("app_opened", {})


def test_unofficial_build_never_reaches_sender(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(track_module.build_gate, "official_build", lambda: None)
    monkeypatch.setattr(
        posthog_sender,
        "get_sender",
        lambda: calls.append("constructed") or RecordingSender(),
    )
    track_module.track("app_opened", {})
    assert calls == []


def test_unofficial_build_gate_precedes_every_context_side_effect(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(
        track_module.build_gate,
        "official_build",
        lambda: calls.append("gate") or None,
    )
    monkeypatch.setattr(
        track_module.common_props,
        "read_platform_facts",
        lambda: pytest.fail("platform read before build gate"),
    )
    monkeypatch.setattr(
        state,
        "get_or_create_client_id",
        lambda: pytest.fail("client id touched before build gate"),
    )
    monkeypatch.setattr(
        emit,
        "session_id",
        lambda: pytest.fail("session id touched before build gate"),
    )
    track_module.track("app_opened", {})
    assert calls == ["gate"]


def test_process_context_memoizes_unofficial_result(monkeypatch):
    calls = 0

    def gate():
        nonlocal calls
        calls += 1
        return None

    monkeypatch.setattr(track_module.build_gate, "official_build", gate)
    assert track_module._process_context() is None
    assert track_module._process_context() is None
    assert calls == 1


def test_track_drops_if_build_gate_changes_before_context(monkeypatch):
    stamps = iter((STAMP, None))
    monkeypatch.setattr(track_module.build_gate, "official_build", lambda: next(stamps))
    track_module.track("app_opened", {})


def test_denied_permission_precedes_context_and_store(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(
        track_module.build_gate,
        "official_build",
        lambda: calls.append("gate") or STAMP,
    )
    monkeypatch.setattr(
        consent_runtime,
        "upload_allowed",
        lambda: calls.append("permission") or False,
    )
    monkeypatch.setattr(
        track_module.common_props,
        "read_platform_facts",
        lambda: pytest.fail("platform read for denied user"),
    )
    monkeypatch.setattr(
        state,
        "get_or_create_client_id",
        lambda: pytest.fail("client id created for denied user"),
    )
    monkeypatch.setattr(
        track_module.store,
        "days_since_first_run_bucket",
        lambda: pytest.fail("store touched for denied user"),
    )
    track_module.track("app_opened", {})
    assert calls == ["gate", "permission"]


def test_registry_is_the_only_event_gate(monkeypatch):
    sender = inject_sender(monkeypatch)
    track_module.track("not_an_event", {})
    track_module.track("app_opened", {"x": 1})
    assert sender.calls == 0
    assert sender.items == []


def test_zero_nth_model_served_is_omitted(monkeypatch):
    sender = inject_sender(monkeypatch)
    track_module.track("app_opened", {}, nth_model_served=0)
    [item] = sender.items
    properties = item["properties"]
    assert isinstance(properties, dict)
    assert "nth_model_served" not in properties


def test_app_opened_attempted_once_and_surface_selected(monkeypatch):
    calls: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(
        track_module, "track", lambda event, props: calls.append((event, props))
    )
    track_module._emit_app_opened("server")
    track_module._emit_app_opened("server")
    assert calls == [("app_opened", {})]
    assert track_module._surface == "server"


def test_app_opened_denial_and_gate_failure_are_inert(monkeypatch):
    monkeypatch.setattr(track_module, "_upload_allowed", lambda: False)
    track_module._emit_app_opened("cli")
    monkeypatch.setattr(
        track_module,
        "_upload_allowed",
        lambda: (_ for _ in ()).throw(RuntimeError("gate failed")),
    )
    track_module._emit_app_opened("server")
    assert track_module._app_opened_attempted is False


@pytest.mark.parametrize(
    ("command", "surface"), [("models", "cli"), ("serve", "server")]
)
def test_lifecycle_surface_reaches_captured_item(monkeypatch, command, surface):
    sender = inject_sender(monkeypatch)
    monkeypatch.setattr(consent_runtime, "upload_allowed", lambda: True)
    monkeypatch.setattr(
        consent_runtime, "detect_role", lambda: ProcessRole.HEADLESS_CLI
    )
    cli._start_v2_lifecycle(command)
    [item] = sender.items
    assert item["properties"]["surface"] == surface


@pytest.mark.parametrize("command", [None, "telemetry", "feedback"])
def test_cli_lifecycle_exclusions(monkeypatch, command):
    calls: list[str] = []
    monkeypatch.setattr(
        posthog_sender, "install_atexit", lambda: calls.append("atexit")
    )
    monkeypatch.setattr(
        track_module, "_emit_app_opened", lambda surface: calls.append(surface)
    )
    monkeypatch.setattr(
        consent_runtime, "detect_role", lambda: ProcessRole.HEADLESS_CLI
    )
    cli._start_v2_lifecycle(command)
    assert calls == []


def test_cli_lifecycle_skips_sidecar(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(
        posthog_sender, "install_atexit", lambda: calls.append("atexit")
    )
    monkeypatch.setattr(
        track_module, "_emit_app_opened", lambda surface: calls.append(surface)
    )
    monkeypatch.setattr(consent_runtime, "detect_role", lambda: ProcessRole.SIDECAR)
    cli._start_v2_lifecycle("serve")
    assert calls == []


@pytest.mark.parametrize(
    ("command", "surface"), [("serve", "server"), ("models", "cli")]
)
def test_cli_lifecycle_installs_exit_hook_and_emits(monkeypatch, command, surface):
    calls: list[str] = []
    monkeypatch.setattr(
        consent_runtime, "detect_role", lambda: ProcessRole.HEADLESS_CLI
    )
    monkeypatch.setattr(
        posthog_sender, "install_atexit", lambda: calls.append("atexit")
    )
    monkeypatch.setattr(
        track_module, "_emit_app_opened", lambda value: calls.append(value)
    )
    cli._start_v2_lifecycle(command)
    assert calls == ["atexit", surface]


def test_cli_lifecycle_failure_cannot_escape(monkeypatch):
    monkeypatch.setattr(
        consent_runtime, "detect_role", lambda: ProcessRole.HEADLESS_CLI
    )
    monkeypatch.setattr(
        posthog_sender,
        "install_atexit",
        lambda: (_ for _ in ()).throw(RuntimeError("hook failed")),
    )
    cli._start_v2_lifecycle("models")


def test_shared_lifecycle_rejects_invalid_surface_and_denial(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(
        posthog_sender, "install_atexit", lambda: calls.append("atexit")
    )
    track_module.start_lifecycle("desktop")
    monkeypatch.setattr(track_module, "_upload_allowed", lambda: False)
    track_module.start_lifecycle("cli")
    assert calls == []


@pytest.mark.parametrize(("claimed", "expected"), [(True, 1), (False, 0)])
def test_active_day_requires_successful_claim(monkeypatch, claimed, expected):
    calls: list[str] = []
    monkeypatch.setattr(track_module, "track", lambda event, props: calls.append(event))
    fake_store = SimpleNamespace(claim_active_day=lambda: claimed)
    track_module.emit_active_day(_store=fake_store)
    assert calls == ["active_day"] * expected


def test_active_day_denial_precedes_store_claim(monkeypatch):
    monkeypatch.setattr(consent_runtime, "upload_allowed", lambda: False)
    fake_store = SimpleNamespace(
        claim_active_day=lambda: pytest.fail("denied active-day store write")
    )
    track_module.emit_active_day(_store=fake_store)


def test_track_import_registers_no_exit_or_fork_hook():
    code = (
        "import atexit, os\n"
        "calls = []\n"
        "atexit.register = lambda fn: calls.append(('exit', fn))\n"
        "os.register_at_fork = lambda **kw: calls.append(('fork', kw))\n"
        "import rapid_mlx.telemetry.track\n"
        "exits = [c for c in calls if c[0] == 'exit']\n"
        "assert exits == [], calls\n"
        "posthog = [c for c in calls if "
        "'posthog_sender' in repr(c)]\n"
        "assert posthog == [], calls\n"
    )
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT), HOME=os.environ["HOME"])
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr


_CLI_SUBPROCESS = """
import sys
import rapid_mlx
from rapid_mlx import cli
from rapid_mlx.telemetry import build_gate
from rapid_mlx.telemetry.build_gate import ReleaseStamp

case = sys.argv[1]
mode = sys.argv[2]
if case != "unofficial":
    build_gate.official_build = lambda: ReleaseStamp("stable", "phc_" + "a" * 32)
if case == "pre_cutoff":
    rapid_mlx.__version__ = "0.14.3"
if mode == "baseline":
    from rapid_mlx.telemetry import track
    track.start_lifecycle = lambda surface: None
sys.argv = ["rapid-mlx"]
if case == "no_telemetry":
    sys.argv.append("--no-telemetry")
sys.argv.extend(["models", "--json"])
cli.main()
"""


def _home_files(path: Path) -> set[str]:
    return {
        str(item.relative_to(path))
        for item in path.rglob("*")
        if item.is_file() or item.is_symlink()
    }


def _run_models_subprocess(
    home: Path, case: str, extra_env=None, *, baseline: bool = False
) -> subprocess.CompletedProcess[str]:
    home.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, HOME=str(home), PYTHONPATH=str(REPO_ROOT))
    env.pop(state.ENV_VAR, None)
    env.pop(state.DO_NOT_TRACK_ENV, None)
    env.pop("RAPID_MLX_PROCESS_ROLE", None)
    env.pop("RAPID_MLX_WATCHDOG_PPID", None)
    for name in state.CI_ENV_VARS:
        env.pop(name, None)
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [
            sys.executable,
            "-c",
            _CLI_SUBPROCESS,
            case,
            "baseline" if baseline else "candidate",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def test_nonofficial_cli_adds_no_v2_home_files(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    baseline_proc = _run_models_subprocess(baseline, "unofficial", baseline=True)
    candidate_proc = _run_models_subprocess(candidate, "unofficial")
    assert baseline_proc.returncode == candidate_proc.returncode == 0
    assert _home_files(candidate) == _home_files(baseline)
    assert not any("telemetry-client-id" in path for path in _home_files(candidate))


@pytest.mark.parametrize(
    ("case", "extra_env", "consent_text", "consent_is_dir"),
    [
        ("official", {"DO_NOT_TRACK": "1"}, None, False),
        ("official", {"RAPID_MLX_TELEMETRY": "0"}, None, False),
        ("no_telemetry", {}, None, False),
        (
            "official",
            {},
            "consent: false\nprompted_version: 0.15.1\nnotice_revision_seen: 1\n",
            False,
        ),
        ("official", {}, None, True),
        (
            "pre_cutoff",
            {},
            "consent: true\nprompted_version: 0.14.3\nnotice_revision_seen: 1\n",
            False,
        ),
        ("official", {"RAPID_MLX_PROCESS_ROLE": "desktop-sidecar"}, None, False),
    ],
    ids=[
        "do-not-track",
        "telemetry-env-off",
        "cli-no-telemetry",
        "stored-refusal",
        "read-error",
        "pre-cutoff",
        "sidecar",
    ],
)
def test_denied_official_cli_home_tree_matches_without_v2_lifecycle(
    tmp_path, case, extra_env, consent_text, consent_is_dir
):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    for home in (baseline, candidate):
        telemetry_dir = home / ".rapid-mlx"
        telemetry_dir.mkdir(parents=True)
        consent_path = telemetry_dir / "telemetry-consent.yaml"
        if consent_is_dir:
            consent_path.mkdir()
        elif consent_text is not None:
            consent_path.write_text(consent_text, encoding="utf-8")
    baseline_proc = _run_models_subprocess(baseline, case, extra_env, baseline=True)
    candidate_proc = _run_models_subprocess(candidate, case, extra_env)
    assert baseline_proc.returncode == candidate_proc.returncode == 0
    assert _home_files(candidate) == _home_files(baseline)
    assert not any("telemetry.db" in path for path in _home_files(candidate))
    assert not any("telemetry-client-id" in path for path in _home_files(candidate))


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


def test_allowed_official_cli_posts_one_app_opened_to_loopback(tmp_path):
    home = tmp_path / "home"
    telemetry_dir = home / ".rapid-mlx"
    telemetry_dir.mkdir(parents=True)
    (telemetry_dir / "telemetry-consent.yaml").write_text(
        "consent: true\nprompted_version: 0.15.1\nnotice_revision_seen: 1\n",
        encoding="utf-8",
    )
    server = HTTPServer(("127.0.0.1", 0), _CaptureHandler)
    server.bodies = []  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        proc = _run_models_subprocess(
            home,
            "official",
            {
                posthog_sender.POSTHOG_URL_ENV: (
                    f"http://127.0.0.1:{server.server_port}/batch/"
                )
            },
        )
    finally:
        server.shutdown()
        thread.join(timeout=2.0)
        server.server_close()
    assert proc.returncode == 0, proc.stderr
    items = [item for body in server.bodies for item in json.loads(body)["batch"]]  # type: ignore[attr-defined]
    assert [item["event"] for item in items] == ["app_opened"]
    assert items[0]["properties"]["surface"] == "cli"


def test_platform_cached_but_cohort_stamp_read_per_event(monkeypatch):
    sender = inject_sender(monkeypatch)
    facts_calls = 0
    days = iter(("0", "1"))

    def facts() -> PlatformFacts:
        nonlocal facts_calls
        facts_calls += 1
        return FACTS

    monkeypatch.setattr(track_module.common_props, "read_platform_facts", facts)
    monkeypatch.setattr(
        track_module.store, "days_since_first_run_bucket", lambda: next(days)
    )
    track_module.track("app_opened", {})
    track_module.track("active_day", {})
    assert facts_calls == 1
    assert [
        item["properties"]["days_since_first_run_bucket"] for item in sender.items
    ] == [
        "0",
        "1",
    ]


def test_surface_rejects_invalid_and_cannot_change_after_context(monkeypatch):
    inject_sender(monkeypatch)
    track_module._set_surface("desktop")
    assert track_module._surface is None
    track_module.track("app_opened", {})
    track_module._set_surface("server")
    assert track_module._surface is None


def test_none_common_props_drops_event(monkeypatch):
    sender = inject_sender(monkeypatch)
    monkeypatch.setattr(
        track_module.common_props, "build_common_props", lambda **kwargs: None
    )
    track_module.track("app_opened", {})
    assert sender.items == []


def test_active_day_store_failure_is_swallowed(monkeypatch):
    monkeypatch.setattr(
        track_module,
        "track",
        lambda event, props: pytest.fail("claim failure must not emit"),
    )
    fake_store = SimpleNamespace(
        claim_active_day=lambda: (_ for _ in ()).throw(RuntimeError("store failed"))
    )
    track_module.emit_active_day(_store=fake_store)


def test_server_shutdown_flush_is_best_effort(monkeypatch):
    calls: list[str] = []
    sender = SimpleNamespace(flush=lambda timeout: calls.append(f"flushed:{timeout}"))
    monkeypatch.setattr(posthog_sender, "get_sender", lambda: sender)
    server_module._flush_v2_telemetry()
    assert calls == ["flushed:2.0"]

    monkeypatch.setattr(
        sender,
        "flush",
        lambda timeout: (_ for _ in ()).throw(RuntimeError("flush failed")),
    )
    server_module._flush_v2_telemetry()


@pytest.mark.asyncio
async def test_lifespan_shutdown_drains_v2_once(monkeypatch):
    calls: list[str] = []
    sender = posthog_sender.PostHogSender(
        post=lambda url, body, timeout: calls.append("post") or 200,
        gate=lambda: STAMP,
        allowed=lambda: True,
        clock=lambda: 1000.0,
    )
    original_flush = sender.flush

    def recording_flush(timeout: float) -> None:
        calls.append("drain")
        original_flush(timeout)

    monkeypatch.setattr(sender, "flush", recording_flush)
    monkeypatch.setattr(posthog_sender, "get_sender", lambda: sender)
    lifespan = server_module.lifespan(server_module.app)
    await lifespan.__anext__()
    with pytest.raises(StopAsyncIteration):
        await lifespan.__anext__()
    assert calls == ["drain"]
    assert sender.capture(
        {
            "uuid": str(uuid.uuid4()),
            "event": "app_opened",
            "distinct_id": INSTALL_ID,
            "timestamp": "2026-01-01T00:00:00Z",
            "properties": {},
        }
    )
    sender.close(0.5)


def test_server_module_entrypoint_starts_shared_v2_lifecycle(monkeypatch):
    class StopAfterLifecycleError(Exception):
        pass

    calls: list[str] = []
    monkeypatch.setattr(sys, "argv", ["rapid_mlx.server"])
    monkeypatch.setattr(consent_runtime, "startup", lambda **kwargs: None)
    monkeypatch.setattr(
        track_module, "start_lifecycle", lambda surface: calls.append(surface)
    )
    monkeypatch.setattr(
        cli,
        "_port_preflight_or_die",
        lambda *args, **kwargs: (_ for _ in ()).throw(StopAfterLifecycleError()),
    )
    with pytest.raises(StopAfterLifecycleError):
        server_module.main()
    assert calls == ["server"]


def test_ci_kill_switch_is_checked_live_by_default_sender(monkeypatch):
    posts: list[bytes] = []
    decision = Decision(True, False, WriteBack(False, False, False), "existing_opt_in")
    monkeypatch.setattr(consent_runtime, "resolve", lambda: decision)
    monkeypatch.setattr(
        consent_runtime,
        "_live_stored_consent",
        lambda: StoredConsent(True, "0.15.1", 1),
    )
    monkeypatch.setattr(consent_runtime, "upload_allowed", REAL_UPLOAD_ALLOWED)
    sender = posthog_sender.PostHogSender(
        post=lambda url, body, timeout: posts.append(body) or 200,
        clock=lambda: 1000.0,
        gate=lambda: STAMP,
    )
    monkeypatch.setattr(posthog_sender, "get_sender", lambda: sender)
    track_module.track("app_opened", {})
    assert sender._accepted == 1
    monkeypatch.setenv("CI", "true")
    track_module.track("active_day", {})
    assert sender._accepted == 1
    sender.flush(1.0)
    assert posts == []
