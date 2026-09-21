# SPDX-License-Identifier: Apache-2.0
"""Proof for the telemetry v2 consent wiring (``consent_runtime``).

This module is the ONLY caller of ``consent_decision.decide``; the tests
here pin the wiring contract the decision table relies on:

- the v2 path never calls ``state.get_consent_state()`` (Hazard 1: that
  reader collapses a desktop-written schema-1 refusal into "absent",
  which would upload on installs that said no);
- rows 1/7 gate uploading on the notice having actually been delivered
  IN THIS PROCESS (``startup`` order: notice -> write-back -> allow);
- the migrating run (row 4) never uploads, whatever lands on disk;
- sidecars (rows 2/5/8) never write and never disclose;
- the kill switch is absolute (no upload, no notice, no write-back);
- the write-back MERGES under a lock and never drops
  ``desktop_consent`` / ``schema_version`` / unknown keys;
- a mid-session opt-out goes dark on the next capture (live re-check
  with a stat-keyed TTL cache and an injectable clock — no sleeps).

The venv running these tests may have rapid-mlx installed from a
DIFFERENT worktree, so first assert the module under test really is the
file in this worktree. ``rapid_mlx.__version__`` is patched to a
post-cutoff release string in the fixture: the editable install in a
dev venv often still reports a pre-cutoff version, which would make
every decision ``pre_cutoff_runtime`` and exercise nothing.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest
import yaml

import rapid_mlx.telemetry.consent_runtime as consent_runtime_module
from rapid_mlx.telemetry import state
from rapid_mlx.telemetry.consent_decision import (
    DISCLOSURE_REVISION,
    ProcessRole,
    StoredConsent,
    WriteBack,
)
from rapid_mlx.telemetry.consent_runtime import (
    NOTICE_TEXT,
    apply_write_back,
    deliver_notice_if_needed,
    detect_role,
    kill_switch_active,
    read_stored_consent,
    resolve,
    startup,
    upload_allowed,
)

_REPO_ROOT = Path(consent_runtime_module.__file__).resolve().parents[2]

_RELEASE_VERSION = "0.15.1"  # post-cutoff, parses

#: The record the desktop app writes (JSON-that-is-valid-YAML,
#: ``schema_version: 1``, ``TelemetryConsent.writeSharedConsent`` shape).
_DESKTOP_REFUSAL = (
    "{\n"
    '  "consent" : false,\n'
    '  "desktop_consent" : false,\n'
    '  "prompted_at" : "2026-09-18T21:11:17Z",\n'
    '  "prompted_version" : "0.11.0",\n'
    '  "schema_version" : 1\n'
    "}\n"
)

_MARKER_ONLY_WB = WriteBack(False, False, True)
_MIGRATE_WB = WriteBack(True, True, True)
_NO_WB = WriteBack(False, False, False)


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    """Reroute HOME into tmp and neutralize every env-side decision input.

    Deliberately deletes the CI markers too: on GitHub Actions they are
    set for every job and would engage the env kill switch, which would
    make every decision in this file the kill-switch row and test
    nothing.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    for name in (
        "RAPID_MLX_TELEMETRY",
        "DO_NOT_TRACK",
        "RAPID_MLX_PROCESS_ROLE",
        "RAPID_MLX_WATCHDOG_PPID",
        "CI",
        "GITHUB_ACTIONS",
        "GITLAB_CI",
        "CIRCLECI",
        "TRAVIS",
        "BUILDKITE",
        "JENKINS_URL",
        "TEAMCITY_VERSION",
    ):
        monkeypatch.delenv(name, raising=False)
    # A post-cutoff release version: the editable venv often reports a
    # pre-cutoff one, which would decide ``pre_cutoff_runtime`` everywhere.
    monkeypatch.setattr(rapid_mlx_module(), "__version__", _RELEASE_VERSION)
    # Neutralize cross-test leakage of the process-global latches.
    monkeypatch.setattr(state, "_cli_kill_switch_active", False)
    consent_runtime_module._reset_runtime_state_for_tests()
    yield tmp_path
    consent_runtime_module._reset_runtime_state_for_tests()


def rapid_mlx_module():
    import rapid_mlx

    return rapid_mlx


def consent_path() -> Path:
    return state.consent_path()


def write_consent(text: str) -> Path:
    path = consent_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def read_consent_data() -> dict:
    data = yaml.safe_load(consent_path().read_text())
    assert isinstance(data, dict)
    return data


def fake_clock(start: float = 1000.0):
    """A settable monotonic clock: tests advance it, never sleep."""
    current = [start]

    def clock() -> float:
        return current[0]

    def advance(seconds: float) -> None:
        current[0] += seconds

    return clock, advance


# ---------------------------------------------------------------------------
# Module-under-test identity + the Hazard-1 grep gate
# ---------------------------------------------------------------------------


def test_module_under_test_is_this_worktrees_file():
    assert Path(__file__).resolve().parent.parent == _REPO_ROOT


def test_v2_path_never_calls_get_consent_state():
    """Hazard 1: ``get_consent_state()`` collapses schema-1 refusals into
    "absent", which the decision table would read as a fresh install and
    upload on. The v2 module must not even mention it."""
    source = Path(consent_runtime_module.__file__).read_text()
    forbidden = "get_consent" + "_state"
    assert forbidden not in source


# ---------------------------------------------------------------------------
# read_stored_consent — raw, schema-tolerant read
# ---------------------------------------------------------------------------


def test_read_absent_file_is_all_none(fake_home):
    stored = read_stored_consent()
    assert stored == StoredConsent(None, None, None)


def test_read_directory_where_file_should_be_is_all_none(fake_home):
    path = consent_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir()  # a directory: read_text() raises IsADirectoryError
    assert read_stored_consent() == StoredConsent(None, None, None)


def test_read_invalid_yaml_is_all_none(fake_home):
    write_consent("consent: [unclosed")
    assert read_stored_consent() == StoredConsent(None, None, None)


def test_read_binary_junk_is_all_none(fake_home):
    path = write_consent("")
    path.write_bytes(b"\xff\xfe\x00binary")
    assert read_stored_consent() == StoredConsent(None, None, None)


def test_read_non_mapping_documents_are_all_none(fake_home):
    for text in ("", "- a\n- b\n", "just a scalar\n"):
        write_consent(text)
        assert read_stored_consent() == StoredConsent(None, None, None)


def test_read_desktop_shaped_schema1_refusal(fake_home):
    """The desktop writes schema_version: 1 — the record MUST still read."""
    write_consent(_DESKTOP_REFUSAL)
    stored = read_stored_consent()
    assert stored.consent is False, "a collapsed schema-1 refusal would have uploaded"
    assert stored.recorded_version == "0.11.0"
    assert stored.notice_revision_seen is None


def test_read_ignores_schema_version_entirely(fake_home):
    write_consent("consent: true\nprompted_version: 9.9.9\nschema_version: 99\n")
    stored = read_stored_consent()
    assert stored.consent is True
    assert stored.recorded_version == "9.9.9"


def test_read_non_bool_consent_is_none(fake_home):
    for text in ("consent: 1\n", "consent: 'true'\n", "consent: 'yes'\n"):
        write_consent(text)
        assert read_stored_consent().consent is None


def test_read_non_str_prompted_version_is_none(fake_home):
    write_consent("prompted_version: 1\n")
    assert read_stored_consent().recorded_version is None
    write_consent("prompted_version: [0, 14, 3]\n")
    assert read_stored_consent().recorded_version is None


def test_read_notice_revision_seen_type_gate(fake_home):
    write_consent("notice_revision_seen: true\n")
    assert read_stored_consent().notice_revision_seen is None
    write_consent("notice_revision_seen: '1'\n")
    assert read_stored_consent().notice_revision_seen is None
    write_consent("notice_revision_seen: 3\n")
    assert read_stored_consent().notice_revision_seen == 3
    write_consent("notice_revision_seen: -1\n")
    assert read_stored_consent().notice_revision_seen == -1


# ---------------------------------------------------------------------------
# detect_role
# ---------------------------------------------------------------------------


def test_detect_role_honours_desktop_sidecar_env(fake_home, monkeypatch):
    monkeypatch.setenv("RAPID_MLX_PROCESS_ROLE", "desktop-sidecar")
    assert detect_role() is ProcessRole.SIDECAR


def test_detect_role_ignores_unknown_role_values(fake_home, monkeypatch):
    monkeypatch.setenv("RAPID_MLX_PROCESS_ROLE", "desktop-app")
    assert detect_role() is ProcessRole.HEADLESS_CLI


def test_detect_role_watchdog_ppid_means_sidecar(fake_home, monkeypatch):
    monkeypatch.setenv("RAPID_MLX_WATCHDOG_PPID", str(12345))
    assert detect_role() is ProcessRole.SIDECAR


def test_detect_role_garbage_watchdog_ppid_falls_through(fake_home, monkeypatch):
    monkeypatch.setenv("RAPID_MLX_WATCHDOG_PPID", "banana")
    assert detect_role() is ProcessRole.HEADLESS_CLI


def test_detect_role_watchdog_ppid_of_one_or_less_is_not_sidecar(
    fake_home, monkeypatch
):
    for value in ("1", "0", "-5"):
        monkeypatch.setenv("RAPID_MLX_WATCHDOG_PPID", value)
        assert detect_role() is ProcessRole.HEADLESS_CLI


def test_detect_role_interactive_when_stdin_and_stderr_are_ttys(fake_home, monkeypatch):
    class FakeTty:
        def __init__(self, tty: bool) -> None:
            self._tty = tty

        def isatty(self) -> bool:
            return self._tty

    monkeypatch.setattr(sys, "stdin", FakeTty(True))
    monkeypatch.setattr(sys, "stderr", FakeTty(True))
    assert detect_role() is ProcessRole.INTERACTIVE_CLI


def test_detect_role_stderr_not_tty_is_headless(fake_home, monkeypatch):
    class FakeTty:
        def __init__(self, tty: bool) -> None:
            self._tty = tty

        def isatty(self) -> bool:
            return self._tty

    monkeypatch.setattr(sys, "stdin", FakeTty(True))
    monkeypatch.setattr(sys, "stderr", FakeTty(False))
    assert detect_role() is ProcessRole.HEADLESS_CLI


# ---------------------------------------------------------------------------
# kill_switch_active — env switches OR the CLI flag, read live
# ---------------------------------------------------------------------------


def test_kill_switch_inactive_by_default(fake_home):
    assert kill_switch_active() is False


def test_kill_switch_env_falsy_rapid_mlx_telemetry(fake_home, monkeypatch):
    monkeypatch.setenv("RAPID_MLX_TELEMETRY", "0")
    assert kill_switch_active() is True


def test_kill_switch_do_not_track(fake_home, monkeypatch):
    monkeypatch.setenv("DO_NOT_TRACK", "1")
    assert kill_switch_active() is True


def test_kill_switch_ci_marker(fake_home, monkeypatch):
    monkeypatch.setenv("CI", "1")
    assert kill_switch_active() is True


def test_kill_switch_cli_flag_is_or_ed_in_live(fake_home, monkeypatch):
    assert kill_switch_active() is False
    state.set_cli_kill_switch(True)
    assert kill_switch_active() is True
    state.set_cli_kill_switch(False)
    assert kill_switch_active() is False


# ---------------------------------------------------------------------------
# resolve — memoized decision + invalid_input visibility
# ---------------------------------------------------------------------------


def test_resolve_is_memoized_and_detects_role(fake_home, monkeypatch):
    calls = []
    real_decide = consent_runtime_module.decide

    def counting_decide(*args, **kwargs):
        calls.append((args, kwargs))
        return real_decide(*args, **kwargs)

    monkeypatch.setattr(consent_runtime_module, "decide", counting_decide)
    first = resolve(role=ProcessRole.HEADLESS_CLI)
    second = resolve()
    assert first is second
    assert len(calls) == 1
    assert first.reason == "fresh_install_notice"


def test_resolve_calls_decide_once_under_concurrency(fake_home, monkeypatch):
    entered_read = threading.Event()
    release_read = threading.Event()
    calls = []
    real_read = consent_runtime_module.read_stored_consent
    real_decide = consent_runtime_module.decide

    def blocking_read():
        entered_read.set()
        assert release_read.wait(timeout=5)
        return real_read()

    def counting_decide(*args, **kwargs):
        calls.append(1)
        return real_decide(*args, **kwargs)

    monkeypatch.setattr(consent_runtime_module, "read_stored_consent", blocking_read)
    monkeypatch.setattr(consent_runtime_module, "decide", counting_decide)
    results = []
    first = threading.Thread(target=lambda: results.append(resolve()))
    first.start()
    assert entered_read.wait(timeout=5)
    others = [
        threading.Thread(target=lambda: results.append(resolve())) for _ in range(7)
    ]
    for thread in others:
        thread.start()
    release_read.set()
    first.join()
    for thread in others:
        thread.join()
    assert len(calls) == 1
    assert len(results) == 8
    assert all(result is results[0] for result in results)


def test_resolve_passes_running_version_through_untouched(fake_home, monkeypatch):
    seen = {}
    real_decide = consent_runtime_module.decide

    def spy_decide(*args, **kwargs):
        seen.update(kwargs)
        return real_decide(*args, **kwargs)

    monkeypatch.setattr(consent_runtime_module, "decide", spy_decide)
    resolve()
    assert seen["running_version"] == _RELEASE_VERSION


def test_resolve_logs_invalid_input_at_warning(fake_home, monkeypatch, caplog):
    class Hostile:
        @property
        def consent(self):
            raise RuntimeError("boom")

    monkeypatch.setattr(
        consent_runtime_module, "read_stored_consent", lambda: Hostile()
    )
    with caplog.at_level(logging.WARNING, logger="rapid_mlx.telemetry.consent_runtime"):
        decision = resolve()
    assert decision.reason == "invalid_input"
    assert decision.upload_now is False
    warning = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warning, "invalid_input must be visible at WARNING"


def test_resolve_logs_other_reasons_at_debug_only(fake_home, caplog):
    with caplog.at_level(logging.DEBUG, logger="rapid_mlx.telemetry.consent_runtime"):
        decision = resolve()
    assert decision.reason == "fresh_install_notice"
    assert any(r.levelno == logging.DEBUG for r in caplog.records)
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]


# ---------------------------------------------------------------------------
# Notice delivery — stderr, once per process, never stdout
# ---------------------------------------------------------------------------


def test_notice_is_ascii_encodable():
    NOTICE_TEXT.encode("ascii")


def test_notice_contains_the_required_copy():
    assert "turns anonymous usage reporting on by\ndefault" in NOTICE_TEXT
    assert "including for installs that previously turned it off" in NOTICE_TEXT
    assert "PostHog Cloud" in NOTICE_TEXT
    assert "rapid-mlx telemetry off" in NOTICE_TEXT
    assert "RAPID_MLX_TELEMETRY=0" in NOTICE_TEXT
    assert "DO_NOT_TRACK=1" in NOTICE_TEXT


def test_notice_goes_to_stderr_never_stdout(fake_home, capsys):
    assert deliver_notice_if_needed() is True
    captured = capsys.readouterr()
    assert "NOTICE:" in captured.err
    assert captured.out == ""


def test_notice_is_idempotent_per_process(fake_home, capsys):
    assert deliver_notice_if_needed() is True
    assert deliver_notice_if_needed() is False
    captured = capsys.readouterr()
    assert captured.err.count("NOTICE:") == 1


def test_enabled_headless_process_emits_notice_with_existing_marker(fake_home, capsys):
    write_consent(
        f"consent: true\nprompted_version: 0.15.0\n"
        f"notice_revision_seen: {DISCLOSURE_REVISION}\n"
    )
    decision = startup()
    assert decision.reason == "consented"
    assert capsys.readouterr().err.count("NOTICE:") == 1


def test_notice_skipped_when_decision_does_not_ask(fake_home, monkeypatch, capsys):
    monkeypatch.setenv("RAPID_MLX_TELEMETRY", "0")
    decision = resolve()  # kill_switch row: deliver_notice False
    assert decision.deliver_notice is False
    assert deliver_notice_if_needed(decision) is False
    assert capsys.readouterr().err == ""


def test_notice_failure_keeps_upload_blocked(fake_home, monkeypatch, capsys):
    decision = resolve()  # row 1
    assert decision.deliver_notice is True

    def boom(*args, **kwargs):
        raise OSError("closed stderr")

    original_stderr = sys.stderr
    monkeypatch.setattr(sys, "stderr", _BrokenStream(boom))
    assert deliver_notice_if_needed(decision) is False
    assert upload_allowed() is False  # delivered, not attempted
    # A later healthy stderr can still deliver it.
    monkeypatch.setattr(sys, "stderr", original_stderr)
    # resolve() is memoized; deliver on a fresh call writes and unlocks.
    assert deliver_notice_if_needed() is True
    assert upload_allowed() is False  # startup has not persisted the marker
    assert capsys.readouterr().out == ""


class _BrokenStream:
    """stderr stand-in whose write raises (closed pipe / encoding)."""

    def __init__(self, boom) -> None:
        self._boom = boom

    def write(self, *args, **kwargs):
        self._boom(*args, **kwargs)

    def flush(self) -> None:
        pass


def test_notice_unicode_error_is_swallowed(fake_home, monkeypatch, capsys):
    def boom(*args, **kwargs):
        raise UnicodeEncodeError("ascii", "x", 0, 1, "not ascii")

    monkeypatch.setattr(sys, "stderr", _BrokenStream(boom))
    assert deliver_notice_if_needed() is False
    assert upload_allowed() is False


# ---------------------------------------------------------------------------
# Rows 1/7 — upload gated on actual delivery; startup ordering
# ---------------------------------------------------------------------------


def test_row1_upload_blocked_until_notice_delivered(fake_home):
    decision = resolve()
    assert decision.reason == "fresh_install_notice"
    assert decision.upload_now is True
    assert decision.deliver_notice is True
    assert upload_allowed() is False, "notice not yet delivered"
    assert deliver_notice_if_needed(decision) is True
    assert apply_write_back(decision.write_back) is True
    assert upload_allowed() is True


def test_row7_upload_blocked_until_notice_delivered(fake_home):
    write_consent("consent: true\nprompted_version: 0.15.0\nschema_version: 2\n")
    decision = resolve()
    assert decision.reason == "consented_needs_notice"
    assert upload_allowed() is False, "notice not yet delivered"
    assert deliver_notice_if_needed(decision) is True
    assert apply_write_back(decision.write_back) is True
    assert upload_allowed() is True


def test_startup_order_is_notice_then_write_back(fake_home, monkeypatch):
    order = []
    real_deliver = consent_runtime_module.deliver_notice_if_needed
    real_apply = consent_runtime_module.apply_write_back

    def spy_deliver(*args, **kwargs):
        order.append("deliver")
        return real_deliver(*args, **kwargs)

    def spy_apply(*args, **kwargs):
        order.append("apply")
        return real_apply(*args, **kwargs)

    monkeypatch.setattr(consent_runtime_module, "deliver_notice_if_needed", spy_deliver)
    monkeypatch.setattr(consent_runtime_module, "apply_write_back", spy_apply)
    startup()
    assert order == ["deliver", "apply"], "notice MUST precede the write-back"


def test_startup_applies_write_back_once_per_process(fake_home):
    decision = startup()
    assert decision.reason == "fresh_install_notice"
    assert read_consent_data()["notice_revision_seen"] == DISCLOSURE_REVISION
    # A second startup must not replay the memoized write-back.
    consent_path().write_text("consent: false\nschema_version: 1\n")
    startup()
    assert read_consent_data() == {"consent": False, "schema_version": 1}


# ---------------------------------------------------------------------------
# Row 4 — the migrating run never uploads, even after the write-back
# ---------------------------------------------------------------------------


def test_row4_migrating_run_uploads_never_but_migrates(fake_home):
    write_consent(_DESKTOP_REFUSAL)
    decision = startup()
    assert upload_allowed() is False, (
        "a schema-1 refusal misread as absent would have uploaded (Hazard 1)"
    )
    assert decision.reason == "legacy_refusal_migrated"
    assert decision.upload_now is False, "row 4 must never upload in the migrating run"
    data = read_consent_data()
    assert data["consent"] is True
    assert data["prompted_version"] == _RELEASE_VERSION
    assert data["notice_revision_seen"] == DISCLOSURE_REVISION
    assert data["desktop_consent"] is False, "desktop_consent must survive"
    assert data["schema_version"] == 1, "schema_version must stay untouched"
    assert data["prompted_at"] == "2026-09-18T21:11:17Z"
    # Next process: the migrated record reads as plain consented (row 9).
    consent_runtime_module._reset_runtime_state_for_tests()
    assert resolve().reason == "consented"
    assert upload_allowed() is True


def test_row6_current_refusal_stays_dark(fake_home):
    write_consent(
        "consent: false\nprompted_version: 0.15.0\nschema_version: 2\n"
        "desktop_consent: false\n"
    )
    decision = startup()
    assert decision.reason == "current_refusal"
    assert upload_allowed() is False
    # No writes at all: the refusal record is preserved byte-for-byte.
    assert consent_path().read_text() == (
        "consent: false\nprompted_version: 0.15.0\nschema_version: 2\n"
        "desktop_consent: false\n"
    )


# ---------------------------------------------------------------------------
# Rows 2/5/8 — a sidecar never writes and never discloses
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "seed",
    [
        None,  # row 2: fresh install
        _DESKTOP_REFUSAL,  # row 5: legacy refusal
        "consent: true\nprompted_version: 0.15.0\nschema_version: 2\n",  # row 8
    ],
)
def test_sidecar_never_writes_and_never_discloses(fake_home, monkeypatch, capsys, seed):
    monkeypatch.setenv("RAPID_MLX_PROCESS_ROLE", "desktop-sidecar")
    if seed is not None:
        write_consent(seed)
    before = consent_path().read_text() if seed is not None else None
    decision = startup()
    assert decision.reason == "sidecar_waits_for_desktop"
    assert decision.deliver_notice is False
    assert capsys.readouterr().err == ""
    assert upload_allowed() is False
    if seed is None:
        assert not consent_path().exists()
    else:
        assert consent_path().read_text() == before


def test_apply_write_back_is_a_noop_for_sidecars(fake_home, monkeypatch):
    monkeypatch.setenv("RAPID_MLX_PROCESS_ROLE", "desktop-sidecar")
    resolve()  # memoize the SIDECAR role
    write_consent(_DESKTOP_REFUSAL)
    before = consent_path().read_text()
    assert apply_write_back(_MIGRATE_WB) is False
    assert consent_path().read_text() == before
    assert not (consent_path().parent / "telemetry-consent.yaml.lock").exists()


# ---------------------------------------------------------------------------
# Kill switch is absolute
# ---------------------------------------------------------------------------


def test_kill_switch_cli_flag_blocks_everything(fake_home, capsys):
    write_consent(_DESKTOP_REFUSAL)
    state.set_cli_kill_switch(True)
    decision = startup()
    assert decision.reason == "kill_switch"
    assert decision.deliver_notice is False
    assert capsys.readouterr().err == ""
    assert upload_allowed() is False
    # No write-back, no marker: the file is byte-identical.
    assert consent_path().read_text() == _DESKTOP_REFUSAL


def test_kill_switch_env_blocks_everything(fake_home, monkeypatch, capsys):
    monkeypatch.setenv("RAPID_MLX_TELEMETRY", "0")
    decision = startup()  # fresh install, would otherwise be row 1
    assert decision.reason == "kill_switch"
    assert capsys.readouterr().err == ""
    assert not consent_path().exists()
    assert upload_allowed() is False


# ---------------------------------------------------------------------------
# apply_write_back — merge semantics under the lock
# ---------------------------------------------------------------------------


def test_empty_write_back_touches_nothing(fake_home):
    assert apply_write_back(_NO_WB) is False
    assert not consent_path().exists()


def test_write_back_uses_resolved_decision_by_default(fake_home):
    decision = resolve()
    assert decision.reason == "fresh_install_notice"
    assert apply_write_back() is True
    assert read_consent_data() == {"notice_revision_seen": DISCLOSURE_REVISION}


def test_write_back_without_marker_request(fake_home):
    assert apply_write_back(WriteBack(True, True, False)) is True
    assert read_consent_data() == {
        "consent": True,
        "prompted_version": _RELEASE_VERSION,
    }


def test_write_back_creates_record_when_absent(fake_home):
    assert apply_write_back(_MARKER_ONLY_WB) is True
    data = read_consent_data()
    assert data == {"notice_revision_seen": DISCLOSURE_REVISION}


def test_write_back_preserves_desktop_consent_and_unknown_keys(fake_home):
    path = consent_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "consent: false\n"
        "desktop_consent: false\n"
        "prompted_at: '2026-09-18T21:11:17Z'\n"
        "prompted_version: 0.11.0\n"
        "schema_version: 1\n"
        "future_unknown_key: keepme\n"
    )
    assert apply_write_back(_MIGRATE_WB) is True
    data = read_consent_data()
    assert data["consent"] is True
    assert data["desktop_consent"] is False
    assert data["schema_version"] == 1
    assert data["prompted_at"] == "2026-09-18T21:11:17Z"
    assert data["future_unknown_key"] == "keepme"
    assert data["prompted_version"] == _RELEASE_VERSION
    assert data["notice_revision_seen"] == DISCLOSURE_REVISION


def test_write_back_only_raises_the_marker(fake_home):
    write_consent("notice_revision_seen: 5\nconsent: false\n")
    assert apply_write_back(_MARKER_ONLY_WB) is True
    assert read_consent_data()["notice_revision_seen"] == 5
    write_consent("notice_revision_seen: 'garbage'\nconsent: false\n")
    assert apply_write_back(_MARKER_ONLY_WB) is True
    assert read_consent_data()["notice_revision_seen"] == DISCLOSURE_REVISION


def test_write_back_aborts_on_unreadable_file(fake_home):
    path = write_consent("consent: [unclosed")
    before = path.read_text()
    assert apply_write_back(_MIGRATE_WB) is False
    assert path.read_text() == before


def test_write_back_survives_zero_progress_write(fake_home, monkeypatch):
    path = write_consent(_DESKTOP_REFUSAL)
    before = path.read_text()
    monkeypatch.setattr(os, "write", lambda fd, data: 0)
    assert apply_write_back(_MIGRATE_WB) is False
    assert path.read_text() == before


def test_write_back_survives_replace_failure(fake_home, monkeypatch):
    path = write_consent(_DESKTOP_REFUSAL)
    before = path.read_text()
    real_replace = os.replace

    def boom_replace(src, dst):
        if str(dst) == str(path):
            raise OSError("replace failed")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", boom_replace)
    assert apply_write_back(_MIGRATE_WB) is False
    assert path.read_text() == before
    # No .tmp litter left behind.
    leftovers = [
        p.name
        for p in path.parent.iterdir()
        if p.name not in {path.name, "telemetry-consent.yaml.lock"}
    ]
    assert leftovers == [], f"tmp litter: {leftovers}"


def test_atomic_write_preserves_original_if_tmp_cleanup_fails(fake_home, monkeypatch):
    path = write_consent(_DESKTOP_REFUSAL)

    def fail_replace(_src, _dst):
        raise OSError("replace failed")

    def fail_unlink(_self):
        raise OSError("cleanup failed")

    monkeypatch.setattr(os, "replace", fail_replace)
    monkeypatch.setattr(Path, "unlink", fail_unlink)
    with pytest.raises(OSError, match="replace failed"):
        consent_runtime_module._atomic_write(path, {"consent": True})
    assert path.read_text() == _DESKTOP_REFUSAL


# ---------------------------------------------------------------------------
# upload_allowed — live re-check with the stat-keyed TTL cache
# ---------------------------------------------------------------------------


def test_upload_allowed_row9_and_mid_session_opt_out(fake_home, monkeypatch):
    write_consent(
        f"consent: true\nprompted_version: 0.15.0\nschema_version: 2\n"
        f"notice_revision_seen: {DISCLOSURE_REVISION}\n"
    )
    clock, advance = fake_clock()
    monkeypatch.setattr(consent_runtime_module, "_clock", clock)
    assert resolve().reason == "consented"
    assert upload_allowed() is True
    # Mid-session `telemetry off`: rewrite the record, advance past the TTL.
    consent_path().write_text(
        "consent: false\nprompted_version: 0.15.1\nschema_version: 2\n"
    )
    advance(6.0)  # > _LIVE_CACHE_TTL_SECONDS
    assert upload_allowed() is False


def test_upload_allowed_live_cache_hit_within_ttl(fake_home, monkeypatch):
    write_consent(
        f"consent: true\nprompted_version: 0.15.0\nschema_version: 2\n"
        f"notice_revision_seen: {DISCLOSURE_REVISION}\n"
    )
    resolve()  # warm the decision; its read_stored_consent is uncounted
    reads = []
    real_read = consent_runtime_module.read_stored_consent

    def counting_read():
        reads.append(1)
        return real_read()

    clock, advance = fake_clock()
    monkeypatch.setattr(consent_runtime_module, "_clock", clock)
    monkeypatch.setattr(consent_runtime_module, "read_stored_consent", counting_read)
    assert upload_allowed() is True
    assert len(reads) == 1
    assert upload_allowed() is True  # cache hit: same fingerprint, within TTL
    assert len(reads) == 1
    advance(_consent_runtime_ttl())
    assert upload_allowed() is True  # TTL expired: re-read
    assert len(reads) == 2


def test_upload_allowed_re_reads_when_file_disappears(fake_home, monkeypatch):
    write_consent(
        f"consent: true\nprompted_version: 0.15.0\nschema_version: 2\n"
        f"notice_revision_seen: {DISCLOSURE_REVISION}\n"
    )
    resolve()
    reads = []
    real_read = consent_runtime_module.read_stored_consent

    def counting_read():
        reads.append(1)
        return real_read()

    clock, _advance = fake_clock()
    monkeypatch.setattr(consent_runtime_module, "_clock", clock)
    monkeypatch.setattr(consent_runtime_module, "read_stored_consent", counting_read)
    assert upload_allowed() is True
    assert len(reads) == 1
    consent_path().unlink()  # withdrawn record: fingerprint changes
    assert upload_allowed() is False  # a withdrawn record goes dark immediately
    assert len(reads) == 2


def _consent_runtime_ttl() -> float:
    return consent_runtime_module._LIVE_CACHE_TTL_SECONDS + 0.5


# ---------------------------------------------------------------------------
# Concurrency — the lock keeps merges consistent
# ---------------------------------------------------------------------------


def test_concurrent_write_backs_never_corrupt_or_lose_keys(fake_home, monkeypatch):
    write_consent(
        "consent: false\n"
        "desktop_consent: false\n"
        "prompted_at: '2026-09-18T21:11:17Z'\n"
        "prompted_version: 0.11.0\n"
        "schema_version: 1\n"
        "seed_key: keepme\n"
    )
    # Distinct per-thread stamps so the writes genuinely differ.
    monkeypatch.setattr(
        consent_runtime_module,
        "_running_version",
        lambda: threading.current_thread().name,
    )
    names = [f"writer-{i}" for i in range(8)]
    barrier = threading.Barrier(len(names))
    results: list[bool] = []
    parses: list[dict] = []

    def worker() -> None:
        barrier.wait()
        results.append(apply_write_back(_MIGRATE_WB))
        data = yaml.safe_load(consent_path().read_text())
        assert isinstance(data, dict)
        parses.append(data)

    threads = [threading.Thread(target=worker, name=name) for name in names]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert results == [True] * len(names)
    final = read_consent_data()
    assert final["consent"] is True
    assert final["desktop_consent"] is False
    assert final["schema_version"] == 1
    assert final["prompted_at"] == "2026-09-18T21:11:17Z"
    assert final["seed_key"] == "keepme"
    assert final["prompted_version"] in names
    assert final["notice_revision_seen"] == DISCLOSURE_REVISION
    for data in parses:
        assert data["desktop_consent"] is False
        assert data["seed_key"] == "keepme"
    assert len(parses) == len(names)


# ---------------------------------------------------------------------------
# Entrypoint wiring — cli.main and server.main
# ---------------------------------------------------------------------------


def test_cli_main_runs_startup_and_migrates_schema1_refusal(
    fake_home, monkeypatch, capsys
):
    from rapid_mlx import cli

    write_consent(_DESKTOP_REFUSAL)
    monkeypatch.setattr(sys, "argv", ["rapid-mlx", "version"])
    cli.main()
    data = read_consent_data()
    assert data["consent"] is True
    assert data["desktop_consent"] is False
    assert data["schema_version"] == 1
    assert data["notice_revision_seen"] == DISCLOSURE_REVISION
    assert data["prompted_version"] == _RELEASE_VERSION
    captured = capsys.readouterr()
    assert "NOTICE:" in captured.err, "the notice goes to stderr"
    assert "rapid-mlx" in captured.out  # command output is unaffected
    assert "NOTICE:" not in captured.out
    # The in-process decision was row 4: this run never uploads.
    assert upload_allowed() is False


def test_cli_main_no_telemetry_blocks_notice_and_write_back(
    fake_home, monkeypatch, capsys
):
    from rapid_mlx import cli

    write_consent(_DESKTOP_REFUSAL)
    monkeypatch.setattr(sys, "argv", ["rapid-mlx", "--no-telemetry", "version"])
    cli.main()
    captured = capsys.readouterr()
    assert "NOTICE:" not in captured.err
    assert consent_path().read_text() == _DESKTOP_REFUSAL
    assert upload_allowed() is False


def test_server_main_runs_startup_before_engine_init(fake_home, capsys):
    from rapid_mlx import server

    def boom(_args):
        raise RuntimeError("stub: stop main() right after the wiring")

    monkeypatch_module = pytest.MonkeyPatch()
    try:
        monkeypatch_module.setattr("rapid_mlx.routes.video.configure_video_jobs", boom)
        monkeypatch_module.setattr(sys, "argv", ["rapid_mlx.server"])
        with pytest.raises(SystemExit):
            server.main()
    finally:
        monkeypatch_module.undo()

    data = read_consent_data()
    assert data["notice_revision_seen"] == DISCLOSURE_REVISION
    captured = capsys.readouterr()
    assert "NOTICE:" in captured.err
    assert upload_allowed() is True


# ---------------------------------------------------------------------------
# Real-file round trip via a subprocess (fresh process, fresh memo state)
# ---------------------------------------------------------------------------


def test_subprocess_round_trip_desktop_refusal_migration(tmp_path):
    """A real file round trip in a REAL process (no memoized state): a
    desktop-shaped record goes in, the migrated record comes out, and
    the migrating run itself is not allowed to upload."""
    home = tmp_path / "home"
    telemetry_dir = home / ".rapid-mlx"
    telemetry_dir.mkdir(parents=True)
    (telemetry_dir / "telemetry-consent.yaml").write_text(_DESKTOP_REFUSAL)
    driver = tmp_path / "driver.py"
    driver.write_text(
        "import os, sys\n"
        f"os.environ['HOME'] = {str(home)!r}\n"
        "import rapid_mlx\n"
        f"rapid_mlx.__version__ = {_RELEASE_VERSION!r}\n"
        "from rapid_mlx.telemetry import consent_runtime as cr\n"
        "decision = cr.startup()\n"
        "print('reason=' + decision.reason)\n"
        "print('upload_allowed=' + str(cr.upload_allowed()))\n"
        "print('notice=' + str(cr._notice_delivered))\n"
    )
    result = subprocess.run(
        [sys.executable, str(driver)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "reason=legacy_refusal_migrated" in result.stdout
    assert "upload_allowed=False" in result.stdout
    assert "notice=True" in result.stdout
    # The notice really went to stderr, never stdout.
    assert "NOTICE:" in result.stderr
    assert "NOTICE:" not in result.stdout
    migrated = (telemetry_dir / "telemetry-consent.yaml").read_text()
    data = yaml.safe_load(migrated)
    assert data["consent"] is True
    assert data["prompted_version"] == _RELEASE_VERSION
    assert data["notice_revision_seen"] == DISCLOSURE_REVISION
    assert data["desktop_consent"] is False
    assert data["schema_version"] == 1
    print("--- migrated file ---")
    print(migrated)


# ---------------------------------------------------------------------------
# Decision passthrough — upload_allowed reflects every blocked row
# ---------------------------------------------------------------------------


def test_upload_allowed_false_for_sidecar_without_marker(fake_home, monkeypatch):
    monkeypatch.setenv("RAPID_MLX_WATCHDOG_PPID", "4242")
    resolve()
    assert upload_allowed() is False


def test_upload_allowed_false_for_current_refusal(fake_home):
    write_consent("consent: false\nprompted_version: 0.15.0\nschema_version: 2\n")
    resolve()
    assert upload_allowed() is False


def test_upload_allowed_true_for_marker_authorised_sidecar(fake_home, monkeypatch):
    monkeypatch.setenv("RAPID_MLX_PROCESS_ROLE", "desktop-sidecar")
    write_consent(
        f"prompted_version: 0.15.0\nnotice_revision_seen: {DISCLOSURE_REVISION}\n"
    )
    assert resolve().reason == "marker_authorises"
    assert upload_allowed() is True


def test_upload_allowed_respects_live_kill_switch_flip(fake_home, monkeypatch):
    write_consent(
        f"consent: true\nprompted_version: 0.15.0\nschema_version: 2\n"
        f"notice_revision_seen: {DISCLOSURE_REVISION}\n"
    )
    assert upload_allowed() is True
    monkeypatch.setenv("DO_NOT_TRACK", "1")
    assert upload_allowed() is False
