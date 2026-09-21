# SPDX-License-Identifier: Apache-2.0
"""Pin the user-facing ``rapid-mlx telemetry ...`` subcommand surface.

Smoke-level: every action returns 0, prints something resembling its
purpose. The deep behaviour (precedence, redaction, prompt skips) is
covered in ``test_telemetry_state.py`` / ``test_telemetry_consent.py``
/ ``test_telemetry_redact.py``. This file only guards the CLI wiring
itself — argparse exposes all 5 actions, dispatch reaches the right
handler, the global ``--no-telemetry`` flag is plumbed through.
"""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest
import yaml

from rapid_mlx.telemetry import state

_PROCESS_ROLE_ENV_VARS = (
    "RAPID_MLX_PROCESS_ROLE",
    "RAPID_MLX_WATCHDOG_PPID",
)


@pytest.fixture(autouse=True)
def _clean_telemetry_env(monkeypatch):
    for name in (
        state.ENV_VAR,
        state.DO_NOT_TRACK_ENV,
        *state.CI_ENV_VARS,
        *_PROCESS_ROLE_ENV_VARS,
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    importlib.reload(state)
    return tmp_path


def _child_env(home, env_overrides=None):
    import os

    env = os.environ.copy()
    env["HOME"] = str(home)
    for name in (
        state.ENV_VAR,
        state.DO_NOT_TRACK_ENV,
        *state.CI_ENV_VARS,
        *_PROCESS_ROLE_ENV_VARS,
    ):
        env.pop(name, None)
    if env_overrides:
        env.update(env_overrides)
    return env


def _run_cli(*args, env_overrides=None, home):
    """Spawn the CLI as a subprocess so argparse + dispatch run end-to-end.

    In-process invocation would short-circuit ``sys.exit`` and miss
    real-world failure modes (broken imports, missing dispatch case).
    """
    return subprocess.run(
        [sys.executable, "-m", "rapid_mlx.cli", *args],
        capture_output=True,
        text=True,
        env=_child_env(home, env_overrides),
        timeout=30,
        check=False,
    )


def test_status_default_off(fake_home):
    r = _run_cli("telemetry", "status", home=fake_home)
    assert r.returncode == 0, r.stderr
    assert "disabled" in r.stdout.lower()
    assert "default" in r.stdout.lower()


def test_status_after_enable(fake_home):
    enable = _run_cli("telemetry", "enable", home=fake_home)
    assert enable.returncode == 0, enable.stderr
    status = _run_cli("telemetry", "status", home=fake_home)
    assert status.returncode == 0
    assert "enabled" in status.stdout.lower()
    assert "consent-file" in status.stdout.lower()


def test_disable_records_false(fake_home):
    _run_cli("telemetry", "enable", home=fake_home)
    _run_cli("telemetry", "disable", home=fake_home)
    status = _run_cli("telemetry", "status", home=fake_home)
    assert "disabled" in status.stdout.lower()
    # Must say it WAS prompted — disable=record consent=False, not "never".
    assert "false" in status.stdout.lower() or "consent: false" in status.stdout.lower()


def test_disable_replaces_unreadable_record(fake_home):
    consent = fake_home / ".rapid-mlx" / "telemetry-consent.yaml"
    consent.parent.mkdir(parents=True, exist_ok=True)
    consent.write_text("unknown_key: replace_me\n")
    consent.chmod(0)
    result = _run_cli("telemetry", "disable", home=fake_home)
    assert result.returncode == 0, result.stderr
    data = yaml.safe_load(consent.read_text())
    assert data["consent"] is False
    assert "unknown_key" not in data


@pytest.mark.parametrize("action", ["enable", "disable"])
def test_explicit_consent_persist_failure_is_one_line_error_without_success(
    fake_home, monkeypatch, capsys, action
):
    import rapid_mlx.telemetry as telemetry
    from rapid_mlx import cli

    def fail_record(*_args, **_kwargs):
        raise OSError("disk denied")

    monkeypatch.setattr(telemetry, "record_consent", fail_record)
    with pytest.raises(SystemExit) as raised:
        cli.telemetry_command(SimpleNamespace(telemetry_action=action))
    assert raised.value.code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.count("\n") == 1
    assert "could not save telemetry preference" in captured.err.lower()
    assert "traceback" not in captured.err.lower()
    assert "enabled" not in captured.err.lower()
    assert "no data will be sent" not in captured.err.lower()


def test_disable_then_enable_preserve_v2_migration_fields(fake_home, monkeypatch):
    import rapid_mlx
    from rapid_mlx import cli
    from rapid_mlx.telemetry import consent_runtime, state
    from rapid_mlx.telemetry.consent_decision import DISCLOSURE_REVISION, ProcessRole

    monkeypatch.setattr(rapid_mlx, "__version__", "0.15.1")
    path = state.consent_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "consent: false\n"
        "desktop_consent: false\n"
        "prompted_version: 0.11.0\n"
        "schema_version: 1\n"
    )
    consent_runtime._reset_runtime_state_for_tests()
    consent_runtime.startup(role=ProcessRole.INTERACTIVE_CLI)

    cli.telemetry_command(SimpleNamespace(telemetry_action="disable"))
    data = yaml.safe_load(path.read_text())
    assert data["notice_revision_seen"] == DISCLOSURE_REVISION
    assert data["desktop_consent"] is False
    consent_runtime._reset_runtime_state_for_tests()
    assert consent_runtime.resolve(role=ProcessRole.INTERACTIVE_CLI).reason == (
        "current_refusal"
    )

    cli.telemetry_command(SimpleNamespace(telemetry_action="enable"))
    data = yaml.safe_load(path.read_text())
    assert data["notice_revision_seen"] == DISCLOSURE_REVISION
    assert data["desktop_consent"] is False
    consent_runtime._reset_runtime_state_for_tests()
    decision = consent_runtime.startup(role=ProcessRole.INTERACTIVE_CLI)
    assert decision.reason == "consented"
    assert consent_runtime.deliver_notice_if_needed(decision) is False


def test_preview_emits_valid_json_payload(fake_home):
    """The preview must be parseable JSON with the documented schema
    fields. If the structure drifts silently, the README docs lie and
    Phase 2's transport will send a payload that doesn't match the
    Cloudflare Worker's validator."""
    r = _run_cli("telemetry", "preview", home=fake_home)
    assert r.returncode == 0, r.stderr

    # Anchor on the first line containing ``"schema_version"`` so we
    # don't get confused by stray ``{`` in framing text (e.g. a path
    # like ``/some/{template}/dir`` would break naive parsing).
    lines = r.stdout.splitlines()
    schema_idx = next(
        (i for i, line in enumerate(lines) if '"schema_version"' in line),
        None,
    )
    assert schema_idx is not None, f"no schema_version line in:\n{r.stdout}"
    open_idx = next(i for i in range(schema_idx, -1, -1) if lines[i].strip() == "{")
    # Track brace depth so the nested platform/session sub-objects'
    # ``}`` don't fool us into closing the envelope early.
    depth = 0
    close_idx = None
    for i in range(open_idx, len(lines)):
        depth += lines[i].count("{") - lines[i].count("}")
        if depth == 0:
            close_idx = i
            break
    assert close_idx is not None, "could not find matching close brace"
    payload = json.loads("\n".join(lines[open_idx : close_idx + 1]))
    assert payload["schema_version"] == 1
    assert "client_id" in payload
    assert payload["session_id"].startswith("preview-")
    assert "platform" in payload
    for key in ("os", "arch", "chip", "memory_gb", "python_version"):
        assert key in payload["platform"]
    assert payload["event"] == "session_start"


def test_preview_shows_request_event_with_degenerate_flag(fake_home):
    """Preview must also show the highest-volume ``request`` event so users can
    audit it — including the #1250 ``output_degenerate`` bool, the only field
    derived from completion text, which ships as a bare boolean. No prompt or
    response text may appear in the sample."""
    r = _run_cli("telemetry", "preview", home=fake_home)
    assert r.returncode == 0, r.stderr
    assert '"event": "request"' in r.stdout
    assert '"output_degenerate": false' in r.stdout
    assert '"completion_empty": false' in r.stdout
    assert '"completion_abnormally_short": false' in r.stdout
    # Only bucketed numbers + booleans — no raw-text field ever surfaces.
    for forbidden in ('"prompt"', '"prompt_text"', '"completion"', '"messages"'):
        assert forbidden not in r.stdout, f"{forbidden} leaked into preview"


def test_reset_removes_consent(fake_home):
    _run_cli("telemetry", "enable", home=fake_home)
    r = _run_cli("telemetry", "reset", home=fake_home)
    assert r.returncode == 0
    status = _run_cli("telemetry", "status", home=fake_home)
    assert "never prompted" in status.stdout.lower()


def test_reset_exits_nonzero_when_removal_fails(fake_home):
    """A failed reset must NOT report success: automation running
    ``rapid-mlx telemetry reset`` needs a non-zero exit when state survives on
    disk (else it assumes telemetry is off when it may still be on)."""
    import pathlib

    _run_cli("telemetry", "enable", home=fake_home)
    # Make one marker path unremovable: a NON-EMPTY directory where reset expects
    # a file, so unlink() raises inside reset_state.
    stuck = pathlib.Path(fake_home) / ".rapid-mlx" / "activation_seen_first_inference"
    stuck.mkdir(parents=True, exist_ok=True)
    (stuck / "child").write_text("x")

    r = _run_cli("telemetry", "reset", home=fake_home)
    assert r.returncode != 0
    assert "reset incomplete" in (r.stdout + r.stderr).lower()


def test_status_with_no_action_defaults_to_status(fake_home):
    """``rapid-mlx telemetry`` (no action) should be a friendly status,
    not an argparse error — users will type the bare command first."""
    r = _run_cli("telemetry", home=fake_home)
    assert r.returncode == 0, r.stderr
    assert "telemetry" in r.stdout.lower()


def test_global_no_telemetry_flag(fake_home):
    """``--no-telemetry`` must be reachable on the root parser."""
    _run_cli("telemetry", "enable", home=fake_home)
    r = _run_cli("--no-telemetry", "telemetry", "status", home=fake_home)
    assert r.returncode == 0, r.stderr
    assert "cli-flag" in r.stdout.lower()
    assert "disabled" in r.stdout.lower()


def test_banner_failure_does_not_block_cli_command(fake_home, monkeypatch, capsys):
    from rapid_mlx import _banner, cli

    def boom(*_args, **_kwargs):
        raise RuntimeError("banner failed")

    monkeypatch.setattr(_banner, "should_show_banner", boom)
    monkeypatch.setattr(sys, "argv", ["rapid-mlx", "version"])
    cli.main()
    assert "rapid-mlx" in capsys.readouterr().out


def test_session_end_synchronously_drained_before_exit(fake_home):
    """Round 5 codex review caught the atexit-ordering risk; round 6
    sharpened the regression test. We now stand up a local HTTPServer
    that captures the actual POST body, then assert the batch carries
    BOTH ``session_start`` AND ``session_end`` envelopes. The previous
    retry-count assertion (round 5) would have passed even if
    ``session_end`` were deleted and shutdown flushed a 1-event
    batch."""
    import http.server
    import json as _json
    import threading as _threading

    captured: list[dict] = []

    class _CaptureHandler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802 — name dictated by stdlib
            length = int(self.headers.get("content-length", "0"))
            raw = self.rfile.read(length)
            try:
                captured.append(_json.loads(raw.decode("utf-8")))
            except Exception:
                captured.append({"_raw": raw[:200].decode("utf-8", "replace")})
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"ok":true}')

        def log_message(self, *_a, **_k):  # silence
            return

    # Round 10 codex review caught a probe-then-bind race in the
    # earlier version (separate ``socket.bind`` → close → re-bind).
    # Let the server bind port 0 itself and read ``server_port``.
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _CaptureHandler)
    port = server.server_port
    t = _threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        _run_cli("telemetry", "enable", home=fake_home)
        r = _run_cli(
            "models",
            home=fake_home,
            env_overrides={
                "RAPID_MLX_TELEMETRY_DEBUG": "1",
                "RAPID_MLX_TELEMETRY_ENDPOINT": f"http://127.0.0.1:{port}/v1/events",
            },
        )
        assert r.returncode == 0, r.stderr
    finally:
        server.shutdown()
        server.server_close()

    # Both lifecycle events must land — they MAY come in the same
    # batch (short ``models`` run finishes well under ``FLUSH_INTERVAL_S``)
    # or in two batches if a long-running command crosses the idle
    # flush boundary. Round 11 codex caught the prior "exactly one
    # batch" assertion as a real flake risk for long-running serve
    # processes. Collect events across all batches instead.
    assert captured, f"no POST captured (return={r.returncode}, stderr={r.stderr})"
    all_events = [
        ev
        for batch in captured
        if isinstance(batch.get("batch"), list)
        for ev in batch["batch"]
    ]
    # Round 14 codex review: comparing a ``set`` of event names hid
    # duplicate ``session_start`` / ``session_end`` emissions — a
    # double-fire from a stray atexit re-registration or a reentered
    # CLI main would still pass. Assert exact counts instead so that
    # regression is visible.
    event_names = [ev.get("event") for ev in all_events]
    starts = event_names.count("session_start")
    ends = event_names.count("session_end")
    assert starts == 1 and ends == 1, (
        f"expected exactly one session_start + one session_end across "
        f"{len(captured)} batch(es); got starts={starts}, ends={ends}, "
        f"all_events={event_names}, batches={captured}"
    )
    # ``session_id`` must be identical across both — race-condition
    # regression catch (round 6 fix #3).
    session_ids = {ev["session_id"] for ev in all_events}
    assert len(session_ids) == 1, (
        f"session_start and session_end carry different session_ids "
        f"(race in lazy init?): {session_ids}"
    )


def test_telemetry_subcommand_does_not_emit_lifecycle_events(fake_home):
    """Codex round 1 caught a "phone home before silencing the phone"
    issue: ``rapid-mlx telemetry disable`` (and ``reset``) would queue
    a ``session_start`` event before the disable action ran, because
    cli.py registered the lifecycle emit for every non-None
    subcommand. The fix excludes the ``telemetry`` subcommand entirely.

    Verified end-to-end: spawn the CLI with debug logging on, run
    ``telemetry disable`` from an opted-in state, and check the stderr
    trace contains no ``[telemetry] attempt`` lines."""
    _run_cli("telemetry", "enable", home=fake_home)
    r = _run_cli(
        "telemetry",
        "disable",
        home=fake_home,
        env_overrides={
            "RAPID_MLX_TELEMETRY_DEBUG": "1",
            # Force a non-routable endpoint so this test cannot
            # accidentally exercise the production collector even if
            # the kill-switch wiring regresses.
            "RAPID_MLX_TELEMETRY_ENDPOINT": "https://127.0.0.1:1/never",
        },
    )
    assert r.returncode == 0, r.stderr
    assert "[telemetry] attempt" not in r.stderr, (
        "telemetry subcommand queued a lifecycle event before disabling — "
        "the cli must skip session_start/session_end for the telemetry path"
    )


def test_env_kill_switch_via_subprocess(fake_home):
    _run_cli("telemetry", "enable", home=fake_home)
    r = _run_cli(
        "telemetry",
        "status",
        home=fake_home,
        env_overrides={"RAPID_MLX_TELEMETRY": "0"},
    )
    assert r.returncode == 0
    assert "env-var" in r.stdout.lower()
    assert "disabled" in r.stdout.lower()


def test_help_lists_telemetry_subcommand(fake_home):
    """Bare ``rapid-mlx --help`` must surface the telemetry subcommand
    so users discover it. Regression target: someone refactors the
    subparsers and accidentally drops the registration."""
    r = subprocess.run(
        [sys.executable, "-m", "rapid_mlx.cli", "--help"],
        env=_child_env(fake_home),
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert r.returncode == 0, r.stderr
    assert "telemetry" in r.stdout


def test_telemetry_help_lists_all_five_actions(fake_home):
    r = subprocess.run(
        [sys.executable, "-m", "rapid_mlx.cli", "telemetry", "--help"],
        env=_child_env(fake_home),
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert r.returncode == 0, r.stderr
    for action in ("status", "enable", "disable", "preview", "reset"):
        assert action in r.stdout, f"missing action {action!r} in --help"
