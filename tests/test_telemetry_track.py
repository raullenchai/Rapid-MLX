# SPDX-License-Identifier: Apache-2.0
"""Mutation pins for the telemetry v2 emit and lifecycle seam."""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Iterator, Mapping
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


@pytest.mark.parametrize(("claimed", "expected"), [(True, 1), (False, 0)])
def test_active_day_requires_successful_claim(monkeypatch, claimed, expected):
    calls: list[str] = []
    monkeypatch.setattr(track_module, "track", lambda event, props: calls.append(event))
    fake_store = SimpleNamespace(claim_active_day=lambda: claimed)
    track_module.emit_active_day(_store=fake_store)
    assert calls == ["active_day"] * expected


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
    monkeypatch.setattr(
        posthog_sender, "_flush_at_exit", lambda: calls.append("flushed")
    )
    server_module._flush_v2_telemetry()
    assert calls == ["flushed"]

    monkeypatch.setattr(
        posthog_sender,
        "_flush_at_exit",
        lambda: (_ for _ in ()).throw(RuntimeError("flush failed")),
    )
    server_module._flush_v2_telemetry()


def test_ci_kill_switch_is_checked_live_by_default_sender(monkeypatch):
    posts: list[bytes] = []
    decision = Decision(True, False, WriteBack(False, False, False), "existing_opt_in")
    monkeypatch.setattr(consent_runtime, "resolve", lambda: decision)
    monkeypatch.setattr(
        consent_runtime,
        "_live_stored_consent",
        lambda: StoredConsent(True, "0.15.1", 1),
    )
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
