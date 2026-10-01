# SPDX-License-Identifier: Apache-2.0
"""Restart-loop dedupe invariants for server_start_state and app_opened.

Five installs re-running ``rapid-mlx serve`` in a tight loop produced 50k-106k
events each. These tests pin the manager-decided invariants: one
attempted/terminal pair per failure key per install per window, one
``app_opened`` per (surface, app_version) per window, no orphans created by
suppression, and #3763's consent, rollback, and lock semantics.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

import rapid_mlx
from rapid_mlx.telemetry import (
    consent_runtime,
    model_events,
    posthog_sender,
    server_start,
    state,
)
from rapid_mlx.telemetry import track as track_module
from rapid_mlx.telemetry.build_gate import ReleaseStamp
from rapid_mlx.telemetry.common_props import PlatformFacts

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

WINDOW = model_events.SERVE_FAILED_DEDUPE_SECONDS


@pytest.fixture
def loop_env(monkeypatch, tmp_path):
    """One fresh install: isolated HOME, captured emissions, fake clock."""
    for name in (state.ENV_VAR, state.DO_NOT_TRACK_ENV, *state.CI_ENV_VARS):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(rapid_mlx, "__version__", "0.15.1")
    monkeypatch.setattr(track_module.common_props, "read_platform_facts", lambda: FACTS)
    monkeypatch.setattr(state, "get_or_create_client_id", lambda: INSTALL_ID)
    monkeypatch.setattr(state, "session_id", lambda: SESSION_ID)
    monkeypatch.setattr(track_module.build_gate, "official_build", lambda: STAMP)
    monkeypatch.setattr(posthog_sender.build_gate, "official_build", lambda: STAMP)
    monkeypatch.setattr(consent_runtime, "upload_allowed", lambda: True)
    monkeypatch.setattr(
        track_module.store, "days_since_first_run_bucket", lambda: "7-29"
    )
    events: list[tuple[str, dict[str, object]]] = []

    def enqueue(accepted) -> bool:
        events.append((accepted.event, dict(accepted.props)))
        return True

    monkeypatch.setattr(track_module, "_enqueue_accepted", enqueue)
    clock = {"now": 1000.0}

    def advance(seconds: float) -> None:
        clock["now"] += seconds

    monkeypatch.setattr(model_events, "_serve_failed_clock", lambda: clock["now"])
    track_module._reset_for_tests()
    model_events._reset_for_tests()
    server_start._reset_for_tests()
    posthog_sender._reset_for_tests()
    state.set_cli_kill_switch(False)
    yield tmp_path, events, advance
    posthog_sender._reset_for_tests()
    track_module._reset_for_tests()
    model_events._reset_for_tests()
    server_start._reset_for_tests()
    state.set_cli_kill_switch(False)


def _run(env, *, outcome: str = "resolve", port_explicit: bool | None = None) -> None:
    """Simulate one ``rapid-mlx serve`` process from launch to its terminal."""
    tmp_path, _events, _advance = env
    del tmp_path
    server_start._reset_for_tests()
    server_start.attempted("test-model", load_policy="eager")
    if outcome == "ready":
        server_start.ready()
    elif outcome != "crash":
        server_start.failed(outcome, port_explicit=port_explicit)


def _states(events) -> list[str]:
    return [props["state"] for _event, props in events]


def _start_ledger(env):
    tmp_path, _events, _advance = env
    path = tmp_path / ".rapid-mlx" / "state" / "serve-start-recent.json"
    return path, (
        json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    )


def test_attempted_is_immediate_outside_any_window(loop_env):
    """A run starting with no fresh failure claim emits attempted right away."""
    _tmp_path, events, _advance = loop_env
    server_start._reset_for_tests()
    server_start.attempted("test-model", load_policy="eager")

    assert _states(events) == ["attempted"]

    server_start.failed("resolve")

    assert _states(events) == ["attempted", "failed"]


def test_loop_of_identical_failures_emits_exactly_one_pair(loop_env):
    """Invariant 2: N identical failing runs in the window emit one pair.

    Suppression drops both halves, so it never creates an orphan attempted
    and never an orphan terminal.
    """
    _tmp_path, events, advance = loop_env

    for _index in range(5):
        _run(loop_env, outcome="resolve")
        advance(2.7)

    assert _states(events) == ["attempted", "failed"]
    _path, ledger = _start_ledger(loop_env)
    assert ledger is not None and len(ledger) == 1


def test_loop_then_success_emits_its_full_pair(loop_env):
    """Invariant 2b: a run that finally reaches ready emits attempted+ready."""
    _tmp_path, events, advance = loop_env

    for _index in range(3):
        _run(loop_env, outcome="resolve")
        advance(2.7)
    _run(loop_env, outcome="ready")

    assert _states(events) == [
        "attempted",
        "failed",
        "attempted",
        "ready",
    ]
    assert events[-1][1]["state"] == "ready"


def test_loop_then_success_clears_the_failure_window(loop_env):
    """After a success a recurrence is new information again (fail-open)."""
    _tmp_path, events, advance = loop_env

    _run(loop_env, outcome="resolve")
    advance(2.7)
    _run(loop_env, outcome="ready")
    advance(2.7)
    _run(loop_env, outcome="resolve")

    assert _states(events) == [
        "attempted",
        "failed",
        "attempted",
        "ready",
        "attempted",
        "failed",
    ]


def test_loop_then_different_failure_emits_its_full_pair(loop_env):
    """Invariant 2b: a different failure_stage is not the recorded failure."""
    _tmp_path, events, advance = loop_env

    for _index in range(3):
        _run(loop_env, outcome="resolve")
        advance(2.7)
    _run(loop_env, outcome="preflight")

    assert _states(events) == [
        "attempted",
        "failed",
        "attempted",
        "failed",
    ]
    assert events[-1][1]["failure_stage"] == "preflight"


def test_bind_port_explicit_participates_in_the_failure_key(loop_env):
    """Invariant 2: the same stage with a different port_explicit is different."""
    _tmp_path, events, _advance = loop_env

    _run(loop_env, outcome="bind", port_explicit=True)
    _run(loop_env, outcome="bind", port_explicit=False)

    assert _states(events) == ["attempted", "failed", "attempted", "failed"]
    assert [props.get("port_explicit") for _e, props in events[1::2]][0:2] == [
        True,
        False,
    ]


def test_window_expiry_emits_the_pair_again(loop_env):
    """Invariant 2d: one pair per window per key, anchored at each emission.

    Suppressed repeats must not extend the window: a loop failing every
    2.7 s produces exactly two pairs across the expiry boundary.
    """
    _tmp_path, events, advance = loop_env

    _run(loop_env, outcome="resolve")
    advanced = 0.0
    while advanced < WINDOW + 1.0:
        advance(2.7)
        advanced += 2.7
        _run(loop_env, outcome="resolve")

    assert _states(events).count("failed") == 2
    assert _states(events).count("attempted") == 2


def test_crash_inside_window_emits_nothing_and_writes_no_claim(loop_env):
    """Invariant 2c: a killed run inside a window stays silent.

    The deferred attempted is never emitted and the run writes no claim, so
    it cannot extend suppression or inflate counts; the next identical
    failure is still measured against the original emitted pair's window.
    """
    _tmp_path, events, advance = loop_env

    _run(loop_env, outcome="resolve")
    _path, ledger_before = _start_ledger(loop_env)
    advance(2.7)
    _run(loop_env, outcome="crash")
    _path, ledger_after = _start_ledger(loop_env)

    assert _states(events) == ["attempted", "failed"]
    assert ledger_after == ledger_before
    advance(2.7)
    _run(loop_env, outcome="resolve")
    assert _states(events) == ["attempted", "failed"]


def test_app_opened_burst_emits_once_per_window(loop_env):
    """Invariant 3: one app_opened per (surface, app_version) per window."""
    tmp_path, events, advance = loop_env

    for _index in range(5):
        track_module._reset_for_tests()
        track_module._emit_app_opened("cli")
        advance(5)

    assert [event for event, _props in events] == ["app_opened"]

    path = tmp_path / ".rapid-mlx" / "state" / "app-opened-recent.json"
    ledger = json.loads(path.read_text(encoding="utf-8"))
    assert list(ledger) == [
        json.dumps(("app_opened", "cli", "0.15.1"), separators=(",", ":"))
    ]
    assert path.stat().st_mode & 0o777 == 0o600

    advance(WINDOW + 1)
    track_module._reset_for_tests()
    track_module._emit_app_opened("cli")

    assert [event for event, _props in events] == ["app_opened", "app_opened"]


def test_app_opened_surface_or_version_change_emits(loop_env):
    """Invariant 3: the key is (surface, app_version), per install."""
    _tmp_path, events, advance = loop_env

    track_module._emit_app_opened("cli")
    advance(5)
    track_module._reset_for_tests()
    track_module._emit_app_opened("server")

    assert [event for event, _props in events] == ["app_opened", "app_opened"]


def test_concurrent_claims_elect_exactly_one_writer(loop_env):
    """Invariant 5: simultaneous processes cannot both claim; no deadlock."""
    _tmp_path, _events, _advance = loop_env
    barrier = threading.Barrier(8)
    enqueued: list[int] = []
    results: list[int] = []

    def claim() -> None:
        barrier.wait()
        accepted = model_events._claim_ledger_key(
            server_start._serve_start_recent_path,
            ("resolve", None),
            window_seconds=WINDOW,
            on_claim=lambda: enqueued.append(1),
        )
        results.append(1 if accepted else 0)

    threads = [threading.Thread(target=claim) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5.0)

    assert not any(thread.is_alive() for thread in threads)
    assert enqueued == [1]
    assert results.count(1) == 1
    assert results.count(0) == 7


@pytest.mark.parametrize("mode", ["rejected", "raised"])
def test_enqueue_rejection_keeps_the_durable_start_claim(loop_env, mode, monkeypatch):
    """Invariant 4: rollback semantics identical to #3763 (no rollback)."""
    _tmp_path, events, _advance = loop_env

    real_enqueue = track_module._enqueue_accepted

    def fail_failed_enqueue(accepted) -> bool:
        if dict(accepted.props)["state"] == "failed":
            if mode == "raised":
                raise RuntimeError("defensive enqueue failure")
            return False
        return real_enqueue(accepted)

    monkeypatch.setattr(track_module, "_enqueue_accepted", fail_failed_enqueue)
    _run(loop_env, outcome="resolve")
    _run(loop_env, outcome="resolve")

    _path, ledger = _start_ledger(loop_env)
    assert len(ledger) == 1
    # The attempted half went out before the rejected failed enqueue; the
    # durable claim survives the rejection, so the repeat stays silent.
    assert _states(events) == ["attempted"]


def test_consent_off_emits_nothing_and_writes_no_ledger(loop_env, monkeypatch):
    """Invariant 4: consent off means no events and no ledger writes."""
    tmp_path, events, _advance = loop_env
    monkeypatch.setattr(consent_runtime, "upload_allowed", lambda: False)

    _run(loop_env, outcome="resolve")
    _run(loop_env, outcome="crash")
    track_module._reset_for_tests()
    track_module._emit_app_opened("cli")

    assert events == []
    assert not (tmp_path / ".rapid-mlx" / "state" / "serve-start-recent.json").exists()
    assert not (tmp_path / ".rapid-mlx" / "state" / "app-opened-recent.json").exists()


def test_deferred_decision_uses_a_start_time_snapshot(loop_env, monkeypatch):
    """A mid-run claim by another process never orphans an emitted attempt."""
    _tmp_path, events, _advance = loop_env

    server_start._reset_for_tests()
    server_start.attempted("test-model", load_policy="eager")
    # Another "process" claims the same failure after our attempted went out.
    assert model_events._claim_ledger_key(
        server_start._serve_start_recent_path,
        ("resolve", None),
        window_seconds=WINDOW,
    )
    server_start.failed("resolve")

    assert _states(events) == ["attempted", "failed"]


def test_mid_run_claim_silences_a_deferred_run(loop_env):
    """A deferred run losing the claim race emits nothing and orphans nothing."""
    _tmp_path, events, _advance = loop_env

    _run(loop_env, outcome="preflight")
    server_start._reset_for_tests()
    server_start.attempted("test-model", load_policy="eager")  # deferred
    # Another process claims our failure key before our terminal.
    assert model_events._claim_ledger_key(
        server_start._serve_start_recent_path,
        ("resolve", None),
        window_seconds=WINDOW,
    )
    server_start.failed("resolve")

    assert _states(events) == ["attempted", "failed"]


def test_nonfinite_clock_fails_open_to_immediate_attempted(loop_env, monkeypatch):
    _tmp_path, events, _advance = loop_env
    monkeypatch.setattr(model_events, "_serve_failed_clock", lambda: float("nan"))

    _run(loop_env, outcome="resolve")

    assert _states(events) == ["attempted", "failed"]


def test_snapshot_read_failure_fails_open(loop_env, monkeypatch):
    _tmp_path, events, _advance = loop_env
    monkeypatch.setattr(
        model_events,
        "_read_serve_failed_recent",
        lambda _path: (_ for _ in ()).throw(OSError("ledger unavailable")),
    )

    _run(loop_env, outcome="resolve")

    assert _states(events) == ["attempted", "failed"]


def test_terminal_consent_rejection_writes_no_claim(loop_env, monkeypatch):
    """A terminal rejected after an immediate attempted claims nothing."""
    _tmp_path, events, _advance = loop_env
    allowed = {"value": True}
    real_upload_allowed = track_module._upload_allowed

    def decide() -> bool:
        return real_upload_allowed() and allowed["value"]

    monkeypatch.setattr(track_module, "_upload_allowed", decide)
    server_start._reset_for_tests()
    server_start.attempted("test-model", load_policy="eager")
    allowed["value"] = False
    server_start.failed("resolve")

    assert _states(events) == ["attempted"]
    _path, ledger = _start_ledger(loop_env)
    assert ledger is None


def test_clear_ledger_is_silent_on_every_failure_shape(loop_env, monkeypatch):
    tmp_path, _events, _advance = loop_env
    path = tmp_path / ".rapid-mlx" / "state" / "serve-start-recent.json"

    model_events._clear_ledger(path)  # absent file

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"k": 1.0}), encoding="utf-8")
    monkeypatch.setattr(
        "rapid_mlx.telemetry.server_start._prepare_state_dir",
        lambda _dir: (_ for _ in ()).throw(OSError("boom")),
    )
    model_events._clear_ledger(path)  # guarded body raises
    monkeypatch.setattr(
        "rapid_mlx.telemetry.server_start._prepare_state_dir", lambda _dir: False
    )
    model_events._clear_ledger(path)  # state dir unavailable
    monkeypatch.setattr(
        "rapid_mlx.telemetry.server_start._prepare_state_dir", lambda _dir: True
    )
    monkeypatch.setattr(model_events, "_acquire_serve_failed_lock", lambda _fd: False)
    model_events._clear_ledger(path)  # bounded lock unavailable
    monkeypatch.setattr(model_events, "_acquire_serve_failed_lock", lambda _fd: True)
    monkeypatch.setattr(
        "rapid_mlx.telemetry.server_start._atomic_write_marker",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("write failed")),
    )
    model_events._clear_ledger(path)  # atomic write raises

    assert json.loads(path.read_text(encoding="utf-8")) == {"k": 1.0}


def test_ready_skips_ledger_clear_when_consent_off(loop_env, monkeypatch):
    tmp_path, _events, _advance = loop_env
    path = tmp_path / ".rapid-mlx" / "state" / "serve-start-recent.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({json.dumps(("resolve", None), separators=(",", ":")): 1000.0}),
        encoding="utf-8",
    )
    monkeypatch.setattr(track_module, "_upload_allowed", lambda: False)

    server_start.ready()

    assert json.loads(path.read_text(encoding="utf-8")) != {}


def test_clear_failure_ledger_swallows_errors(loop_env, monkeypatch):
    _tmp_path, _events, _advance = loop_env
    monkeypatch.setattr(
        model_events,
        "_clear_ledger",
        lambda _path: (_ for _ in ()).throw(OSError("clear failed")),
    )
    server_start._reset_for_tests()
    server_start.attempted("test-model", load_policy="eager")

    server_start.ready()


def test_deferred_attempted_swallows_emit_errors(loop_env, monkeypatch):
    """A raising deferred-attempted emission never blocks the ready path."""
    _tmp_path, events, _advance = loop_env

    _run(loop_env, outcome="resolve")
    server_start._reset_for_tests()
    server_start.attempted("test-model", load_policy="eager")  # deferred
    monkeypatch.setattr(
        track_module,
        "would_accept",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("gate exploded")),
    )

    server_start.ready()

    assert _states(events) == ["attempted", "failed"]


def test_deferred_attempted_is_dropped_when_rejected(loop_env, monkeypatch):
    """Consent-off at the ready terminal drops the deferred half silently."""
    _tmp_path, events, _advance = loop_env
    allowed = {"value": True}
    real_upload_allowed = track_module._upload_allowed

    def decide() -> bool:
        return real_upload_allowed() and allowed["value"]

    monkeypatch.setattr(track_module, "_upload_allowed", decide)
    _run(loop_env, outcome="resolve")
    server_start._reset_for_tests()
    server_start.attempted("test-model", load_policy="eager")  # deferred
    allowed["value"] = False
    server_start.ready()

    assert _states(events) == ["attempted", "failed"]


def test_app_opened_rejected_token_writes_nothing(loop_env, monkeypatch):
    tmp_path, _events, _advance = loop_env
    monkeypatch.setattr(
        track_module,
        "would_accept",
        lambda *_args, **_kwargs: None,
    )

    track_module._emit_app_opened("cli")

    assert not (
        tmp_path / ".rapid-mlx" / "state" / "app-opened-recent.json"
    ).exists()


def test_app_opened_claim_failure_is_contained(loop_env, monkeypatch):
    _tmp_path, _events, _advance = loop_env
    monkeypatch.setattr(
        model_events,
        "_claim_ledger_key",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("claim failed")),
    )

    track_module._emit_app_opened("cli")

    assert track_module._app_opened_attempted is True


def test_registry_has_no_new_server_start_or_app_opened_properties():
    """Invariant 6: no registry/schema change."""
    registry = json.loads(
        (Path(rapid_mlx.__file__).parent / "telemetry" / "events.json").read_text()
    )
    assert set(registry["events"]["app_opened"]["props"]) == set()
    assert set(registry["events"]["server_start_state"]["props"]) == {
        "state",
        "model_type",
        "load_policy",
        "previous_run_unterminated",
        "port_explicit",
        "failure_stage",
    }
