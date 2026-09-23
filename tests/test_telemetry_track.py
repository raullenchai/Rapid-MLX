# SPDX-License-Identifier: Apache-2.0
"""Mutation pins for the telemetry v2 emit and lifecycle seam."""

from __future__ import annotations

import ast
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import uuid
from collections.abc import Iterator, Mapping
from datetime import date, datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

import rapid_mlx
import rapid_mlx.cli as cli
from rapid_mlx.telemetry import (
    common_props,
    consent_runtime,
    posthog_sender,
    redact,
    state,
)
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
REAL_READ_PLATFORM_FACTS = common_props.read_platform_facts

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


class RefusingSender(RecordingSender):
    def capture(self, item: Mapping[str, object]) -> bool:
        self.calls += 1
        self.items.append(dict(item))
        return False


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
    monkeypatch.delenv("RAPID_MLX_PROCESS_ROLE", raising=False)
    monkeypatch.delenv("RAPID_MLX_WATCHDOG_PPID", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(rapid_mlx, "__version__", "0.15.1")
    monkeypatch.setattr(track_module.common_props, "read_platform_facts", lambda: FACTS)
    monkeypatch.setattr(state, "get_or_create_client_id", lambda: INSTALL_ID)
    monkeypatch.setattr(state, "session_id", lambda: SESSION_ID)
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
        state,
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
    result = track_module.track("app_opened", {}, nth_model_served=0)
    assert result is None
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


def test_cli_sidecar_sets_desktop_surface_without_app_opened(monkeypatch):
    sender = inject_sender(monkeypatch)
    monkeypatch.setenv("RAPID_MLX_PROCESS_ROLE", "desktop-sidecar")
    cli._start_v2_lifecycle("serve")
    track_module.track("active_day", {})
    track_module.track(
        "model_served",
        {
            "model": "whisper-small",
            "model_type": "audio",
            "auto_selected": False,
            "quant": "unknown",
        },
    )
    assert [item["event"] for item in sender.items] == ["active_day", "model_served"]
    assert {item["properties"]["surface"] for item in sender.items} == {"desktop"}


def test_process_context_falls_back_to_desktop_for_sidecar(monkeypatch):
    sender = inject_sender(monkeypatch)
    monkeypatch.setenv("RAPID_MLX_PROCESS_ROLE", "desktop-sidecar")
    track_module.track("active_day", {})
    [item] = sender.items
    assert item["properties"]["surface"] == "desktop"


def test_process_context_defaults_to_cli_when_role_detection_fails(monkeypatch):
    sender = inject_sender(monkeypatch)
    monkeypatch.setattr(
        consent_runtime,
        "detect_role",
        lambda: (_ for _ in ()).throw(RuntimeError("role unavailable")),
    )
    track_module.track("active_day", {})
    [item] = sender.items
    assert item["properties"]["surface"] == "cli"


def test_watchdog_sidecar_keeps_cli_surface_without_app_opened(monkeypatch):
    sender = inject_sender(monkeypatch)
    monkeypatch.setenv("RAPID_MLX_WATCHDOG_PPID", "4242")
    cli._start_v2_lifecycle("serve")
    track_module.track("active_day", {})
    [item] = sender.items
    assert item["properties"]["surface"] == "cli"


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
        track_module,
        "start_lifecycle",
        lambda surface: (_ for _ in ()).throw(RuntimeError("lifecycle failed")),
    )
    cli._start_v2_lifecycle("models")


def test_cli_main_starts_lifecycle_after_consent(monkeypatch, capsys):
    calls: list[str] = []
    decision = Decision(True, False, WriteBack(False, False, False), "existing_opt_in")
    monkeypatch.setattr(
        consent_runtime,
        "startup",
        lambda **kwargs: calls.append("consent") or decision,
    )
    monkeypatch.setattr(
        consent_runtime, "detect_role", lambda: ProcessRole.HEADLESS_CLI
    )
    monkeypatch.setattr(
        track_module, "start_lifecycle", lambda surface: calls.append(surface)
    )
    monkeypatch.setattr(sys, "argv", ["rapid-mlx", "models", "--json"])
    cli.main()
    capsys.readouterr()
    assert calls == ["consent", "cli"]


def test_shared_lifecycle_accepts_desktop_surface_without_app_opened(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(
        posthog_sender, "install_atexit", lambda: calls.append("atexit")
    )
    monkeypatch.setattr(consent_runtime, "detect_role", lambda: ProcessRole.DESKTOP)
    track_module.start_lifecycle("server")
    assert track_module._surface == "desktop"
    assert calls == []


def test_desktop_lifecycle_stays_suppressed_after_context_resolution(monkeypatch):
    sender = inject_sender(monkeypatch)
    monkeypatch.setenv("RAPID_MLX_PROCESS_ROLE", "desktop-sidecar")
    track_module.track("active_day", {})
    calls: list[str] = []
    monkeypatch.setattr(
        posthog_sender, "install_atexit", lambda: calls.append("atexit")
    )
    monkeypatch.setattr(
        track_module, "_emit_app_opened", lambda surface: calls.append(surface)
    )
    track_module.start_lifecycle("server")
    assert [item["event"] for item in sender.items] == ["active_day"]
    assert calls == []


def test_shared_lifecycle_rejects_invalid_surface_and_denial(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(
        posthog_sender, "install_atexit", lambda: calls.append("atexit")
    )
    track_module.start_lifecycle("mobile")
    monkeypatch.setattr(track_module, "_upload_allowed", lambda: False)
    track_module.start_lifecycle("cli")
    assert calls == []


def test_shared_lifecycle_failure_cannot_escape(monkeypatch):
    monkeypatch.setattr(
        posthog_sender,
        "install_atexit",
        lambda: (_ for _ in ()).throw(RuntimeError("hook failed")),
    )
    track_module.start_lifecycle("cli")


def test_active_day_claims_before_sender_acceptance(monkeypatch):
    sender = inject_sender(monkeypatch)
    calls: list[str] = []
    original_capture = sender.capture

    def capture(item: Mapping[str, object]) -> bool:
        calls.append("capture")
        return original_capture(item)

    monkeypatch.setattr(sender, "capture", capture)
    fake_store = SimpleNamespace(claim_active_day=lambda: calls.append("claim") or True)
    track_module.emit_active_day(_store=fake_store)
    assert [item["event"] for item in sender.items] == ["active_day"]
    assert calls == ["claim", "capture"]


def test_active_day_claim_is_memoized_for_utc_day(monkeypatch):
    inject_sender(monkeypatch)
    claims: list[date] = []
    today = date(2026, 9, 21)
    monkeypatch.setattr(track_module, "_utc_day", lambda: today)
    fake_store = SimpleNamespace(claim_active_day=lambda: claims.append(today) or True)

    for _ in range(100):
        track_module.emit_active_day(_store=fake_store)

    assert claims == [today]


def test_active_day_false_claim_is_retried_and_utc_rollover_resets(monkeypatch):
    sender = inject_sender(monkeypatch)
    day = [date(2026, 9, 21)]
    claims = iter([False, True, True])
    claim_calls: list[None] = []
    monkeypatch.setattr(track_module, "_utc_day", lambda: day[0])
    fake_store = SimpleNamespace(
        claim_active_day=lambda: claim_calls.append(None) or next(claims)
    )

    track_module.emit_active_day(_store=fake_store)
    track_module.emit_active_day(_store=fake_store)
    day[0] = date(2026, 9, 22)
    track_module.emit_active_day(_store=fake_store)

    assert len(claim_calls) == 3
    assert [item["event"] for item in sender.items] == ["active_day", "active_day"]


def test_active_day_emits_once_when_fresh_store_sees_day_claimed(monkeypatch):
    sender = inject_sender(monkeypatch)
    first_process_store = SimpleNamespace(claim_active_day=lambda: True)
    second_process_store = SimpleNamespace(claim_active_day=lambda: False)

    track_module.emit_active_day(_store=first_process_store)
    track_module.emit_active_day(_store=second_process_store)

    assert [item["event"] for item in sender.items] == ["active_day"]


def test_active_day_rejects_truthy_non_true_claim(monkeypatch):
    sender = inject_sender(monkeypatch)
    fake_store = SimpleNamespace(claim_active_day=lambda: 1)
    track_module.emit_active_day(_store=fake_store)
    assert sender.items == []


def test_active_day_sender_refusal_after_claim_loses_day(monkeypatch):
    sender = RefusingSender()
    monkeypatch.setattr(posthog_sender, "get_sender", lambda: sender)
    claims: list[str] = []
    fake_store = SimpleNamespace(
        claim_active_day=lambda: claims.append("claimed") or True
    )
    track_module.emit_active_day(_store=fake_store)
    assert sender.calls == 1
    assert claims == ["claimed"]


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
from rapid_mlx.telemetry import build_gate, track
from rapid_mlx.telemetry.build_gate import ReleaseStamp
from rapid_mlx.telemetry.common_props import PlatformFacts

case = sys.argv[1]
mode = sys.argv[2]
if case != "unofficial":
    build_gate.official_build = lambda: ReleaseStamp("stable", "phc_" + "a" * 32)
if case == "pre_cutoff":
    rapid_mlx.__version__ = "0.14.3"
else:
    rapid_mlx.__version__ = "0.15.1"
track.common_props.read_platform_facts = lambda: PlatformFacts(
    os="darwin",
    os_version="25.3",
    arch="arm64",
    chip="m3-ultra",
    memory_gb=64,
    python_version="3.11",
)
if mode == "baseline":
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


@pytest.fixture(scope="module")
def official_entrypoint_layout(tmp_path_factory):
    root = tmp_path_factory.mktemp("telemetry-official-entrypoints")
    site_dir = root / "site-packages"
    package_dir = site_dir / "rapid_mlx"
    shutil.copytree(
        REPO_ROOT / "rapid_mlx",
        package_dir,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    (package_dir / "telemetry" / "_release_stamp.json").write_text(
        json.dumps({"channel": "rc", "posthog_key": "phc_" + "a" * 32}),
        encoding="utf-8",
    )
    metadata_dir = site_dir / "rapid_mlx-0.15.1.dist-info"
    metadata_dir.mkdir()
    (metadata_dir / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: rapid-mlx\nVersion: 0.15.1\n",
        encoding="utf-8",
    )

    hooks_dir = root / "hooks"
    hooks_dir.mkdir()
    (hooks_dir / "sitecustomize.py").write_text(
        """
from rapid_mlx import cli

def _capture_later_event_and_stop(*_args, **_kwargs):
    from rapid_mlx.telemetry import posthog_sender, track

    track.track("active_day", {})
    posthog_sender.get_sender().flush(5.0)
    raise SystemExit(0)

cli._port_preflight_or_die = _capture_later_event_and_stop
cli._validate_primary_lifecycle_args = _capture_later_event_and_stop
cli.models_command = _capture_later_event_and_stop
""".lstrip(),
        encoding="utf-8",
    )

    bin_dir = root / "bin"
    bin_dir.mkdir()
    console = bin_dir / "rapid-mlx"
    console.write_text(
        f"#!{sys.executable}\n"
        "from rapid_mlx.cli import cli_entrypoint\n"
        "cli_entrypoint()\n",
        encoding="utf-8",
    )
    console.chmod(0o755)
    return root, hooks_dir, site_dir, console


@pytest.mark.parametrize(
    ("entrypoint", "role_env", "expected_events", "expected_surface"),
    [
        ("module-server", "watchdog", ["active_day"], "cli"),
        ("cli-serve", "watchdog", ["active_day"], "cli"),
        ("module-server", "desktop", ["active_day"], "desktop"),
        ("cli-serve", "desktop", ["active_day"], "desktop"),
        ("module-server", "standalone", ["app_opened", "active_day"], "server"),
        ("cli-serve", "standalone", ["app_opened", "active_day"], "server"),
        ("other-cli", "standalone", ["app_opened", "active_day"], "cli"),
    ],
)
def test_entrypoint_role_surface_matrix(
    tmp_path,
    official_entrypoint_layout,
    entrypoint,
    role_env,
    expected_events,
    expected_surface,
):
    root, hooks_dir, site_dir, console = official_entrypoint_layout
    home = tmp_path / "home"
    telemetry_dir = home / ".rapid-mlx"
    telemetry_dir.mkdir(parents=True)
    (telemetry_dir / "telemetry-consent.yaml").write_text(
        "consent: true\nprompted_version: 0.15.1\nnotice_revision_seen: 1\n",
        encoding="utf-8",
    )
    fake_model = home / "fake-model"
    fake_model.mkdir()

    sink = HTTPServer(("127.0.0.1", 0), _CaptureHandler)
    sink.bodies = []  # type: ignore[attr-defined]
    thread = threading.Thread(target=sink.serve_forever, daemon=True)
    thread.start()
    env = dict(
        os.environ,
        HOME=str(home),
        USER="rc",
        PATH=f"{root / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}",
        PYTHONPATH=os.pathsep.join((str(hooks_dir), str(site_dir))),
        RAPID_MLX_POSTHOG_URL=f"http://127.0.0.1:{sink.server_port}/batch/",
        RAPID_MLX_DISABLE_VERSION_CHECK="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    )
    for name in (*state.CI_ENV_VARS, state.ENV_VAR, state.DO_NOT_TRACK_ENV):
        env.pop(name, None)
    env.pop("RAPID_MLX_PROCESS_ROLE", None)
    env.pop("RAPID_MLX_WATCHDOG_PPID", None)
    if role_env == "watchdog":
        env["RAPID_MLX_WATCHDOG_PPID"] = str(os.getpid())
    elif role_env == "desktop":
        env["RAPID_MLX_PROCESS_ROLE"] = "desktop-sidecar"

    if entrypoint == "module-server":
        command = [sys.executable, "-m", "rapid_mlx.server", "--port", "0"]
    elif entrypoint == "cli-serve":
        command = [str(console), "serve", str(fake_model), "--port", "0"]
    else:
        command = [str(console), "models", "--json"]
    try:
        proc = subprocess.run(
            command,
            cwd=home,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    finally:
        sink.shutdown()
        thread.join(timeout=2.0)
        sink.server_close()

    assert proc.returncode == 0
    assert "Traceback" not in proc.stderr
    items = [
        item
        for body in sink.bodies  # type: ignore[attr-defined]
        for item in json.loads(body)["batch"]
    ]
    assert [item["event"] for item in items] == expected_events
    assert sum(item["event"] == "app_opened" for item in items) == (
        expected_events.count("app_opened")
    )
    assert {item["properties"]["surface"] for item in items} == {expected_surface}


def test_platform_and_cohort_stamp_cached_within_utc_day(monkeypatch):
    sender = inject_sender(monkeypatch)
    facts_calls = 0
    bucket_calls = 0

    def facts() -> PlatformFacts:
        nonlocal facts_calls
        facts_calls += 1
        return FACTS

    monkeypatch.setattr(track_module.common_props, "read_platform_facts", facts)

    def bucket() -> str:
        nonlocal bucket_calls
        bucket_calls += 1
        return "7-29"

    monkeypatch.setattr(track_module.store, "days_since_first_run_bucket", bucket)
    track_module.track("app_opened", {})
    track_module.track("active_day", {})
    assert facts_calls == 1
    assert bucket_calls == 1
    assert [
        item["properties"]["days_since_first_run_bucket"] for item in sender.items
    ] == ["7-29", "7-29"]


def test_cohort_stamp_retries_after_transient_store_failure(monkeypatch):
    sender = inject_sender(monkeypatch)
    buckets = iter((None, "7-29"))
    monkeypatch.setattr(
        track_module.store, "days_since_first_run_bucket", lambda: next(buckets)
    )
    track_module.track("app_opened", {})
    track_module.track("active_day", {})
    assert "days_since_first_run_bucket" not in sender.items[0]["properties"]
    assert sender.items[1]["properties"]["days_since_first_run_bucket"] == "7-29"


@pytest.mark.parametrize("bucket", ["0", "1", "2-6", "7-29", "30+", "not-a-bucket"])
def test_only_declared_cohort_bucket_values_reach_wire(monkeypatch, bucket):
    sender = inject_sender(monkeypatch)
    monkeypatch.setattr(
        track_module.store, "days_since_first_run_bucket", lambda: bucket
    )

    track_module.track("app_opened", {})

    if bucket in track_module.store.DAY_BUCKETS:
        [item] = sender.items
        assert item["properties"]["days_since_first_run_bucket"] == bucket
    else:
        assert sender.items == []


def test_utc_day_uses_utc_clock_near_local_midnight(monkeypatch):
    class FakeDatetime:
        @classmethod
        def now(cls, tz):
            assert tz is timezone.utc
            return datetime(2026, 9, 22, 6, 30, tzinfo=tz)

    monkeypatch.setattr(track_module, "datetime", FakeDatetime)
    assert track_module._utc_day() == date(2026, 9, 22)


def test_cohort_stamp_reread_after_utc_day_rollover(monkeypatch):
    sender = inject_sender(monkeypatch)
    days = iter((date(2026, 9, 21), date(2026, 9, 21), date(2026, 9, 22)))
    buckets = iter(("7-29", "30+"))
    monkeypatch.setattr(track_module, "_utc_day", lambda: next(days))
    monkeypatch.setattr(
        track_module.store, "days_since_first_run_bucket", lambda: next(buckets)
    )
    track_module.track("app_opened", {})
    track_module.track("active_day", {})
    track_module.track("active_day", {})
    assert [
        item["properties"]["days_since_first_run_bucket"] for item in sender.items
    ] == ["7-29", "7-29", "30+"]


def test_surface_rejects_invalid_and_cannot_change_after_context(monkeypatch):
    inject_sender(monkeypatch)
    track_module._set_surface("desktop")
    assert track_module._surface == "desktop"
    track_module._set_surface("mobile")
    assert track_module._surface == "desktop"
    track_module.track("app_opened", {})
    track_module._set_surface("server")
    assert track_module._surface == "desktop"


def test_none_common_props_drops_event(monkeypatch):
    sender = inject_sender(monkeypatch)
    monkeypatch.setattr(
        track_module.common_props, "build_common_props", lambda **kwargs: None
    )
    track_module.track("app_opened", {})
    assert sender.items == []


def test_incomplete_non_apple_platform_drops_event_without_raising(monkeypatch):
    sender = inject_sender(monkeypatch)
    monkeypatch.setattr(
        redact,
        "platform_info",
        lambda: {
            "os": "linux",
            "os_version": None,
            "arch": "x86_64",
            "chip": None,
            "memory_gb": None,
            "python_version": "3.11",
        },
    )
    monkeypatch.setattr(
        track_module.common_props, "read_platform_facts", REAL_READ_PLATFORM_FACTS
    )
    track_module.track("app_opened", {})
    assert sender.items == []


def test_active_day_store_failure_is_swallowed(monkeypatch):
    sender = inject_sender(monkeypatch)
    fake_store = SimpleNamespace(
        claim_active_day=lambda: (_ for _ in ()).throw(RuntimeError("store failed"))
    )
    track_module.emit_active_day(_store=fake_store)
    assert sender.items == []


@pytest.mark.asyncio
async def test_lifespan_shutdown_drains_v2_once(monkeypatch):
    import rapid_mlx.server as server_module

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


def test_server_shutdown_flush_failure_is_swallowed(monkeypatch):
    import rapid_mlx.server as server_module

    monkeypatch.setattr(
        posthog_sender,
        "get_sender",
        lambda: (_ for _ in ()).throw(RuntimeError("sender unavailable")),
    )
    server_module._flush_v2_telemetry()


def test_server_entrypoint_lifecycle_source_contract():
    tree = ast.parse((REPO_ROOT / "rapid_mlx" / "server.py").read_text())
    main = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    startup_index = next(
        index
        for index, node in enumerate(main.body)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and ast.unparse(node.value.func) == "consent_runtime.startup"
    )
    role_assignment = next(
        node
        for node in main.body
        if isinstance(node, ast.Assign)
        and ast.unparse(node.value) == "consent_runtime.detect_role()"
    )
    guard = next(
        node
        for node in main.body
        if isinstance(node, ast.If)
        and ast.unparse(node.test)
        == "not (telemetry_v2.set_surface_for_role(role) or role is ProcessRole.SIDECAR)"
    )
    assert len(guard.body) == 1
    lifecycle_call = guard.body[0]
    assert isinstance(lifecycle_call, ast.Expr)
    assert isinstance(lifecycle_call.value, ast.Call)
    assert ast.unparse(lifecycle_call.value.func) == "telemetry_v2.start_lifecycle"
    assert role_assignment.lineno > main.body[startup_index].lineno
    assert guard.lineno > role_assignment.lineno
    assert len(lifecycle_call.value.args) == 1
    assert isinstance(lifecycle_call.value.args[0], ast.Constant)
    assert lifecycle_call.value.args[0].value == "server"


def test_server_module_entrypoint_starts_shared_v2_lifecycle(monkeypatch):
    import rapid_mlx.cli as current_cli
    import rapid_mlx.server as server_module

    class StopAfterLifecycleError(Exception):
        pass

    calls: list[str] = []
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    monkeypatch.setattr(sys, "argv", ["rapid_mlx.server", "--port", str(port)])
    monkeypatch.setattr(consent_runtime, "startup", lambda **kwargs: None)
    monkeypatch.setattr(
        track_module, "start_lifecycle", lambda surface: calls.append(surface)
    )
    monkeypatch.setattr(
        current_cli,
        "_port_preflight_or_die",
        lambda *args, **kwargs: (_ for _ in ()).throw(StopAfterLifecycleError()),
    )
    with pytest.raises(StopAfterLifecycleError):
        server_module.main()
    assert calls == ["server"]


def test_session_transport_guard_assertion_rejects_recorded_post(
    _posthog_transport_guard_assertion,
):
    calls: list[str] = []
    calls.append("https://us.i.posthog.com/batch/")
    with pytest.raises(AssertionError, match="uninjected PostHog calls"):
        _posthog_transport_guard_assertion(calls)


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
