# SPDX-License-Identifier: Apache-2.0
"""User-facing telemetry v2 CLI contracts."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from rapid_mlx import cli
from rapid_mlx.telemetry import consent_runtime, state


@pytest.fixture(autouse=True)
def isolated_telemetry(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    for name in (state.ENV_VAR, state.DO_NOT_TRACK_ENV, *state.CI_ENV_VARS):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("RAPID_MLX_PROCESS_ROLE", raising=False)
    monkeypatch.delenv("RAPID_MLX_WATCHDOG_PPID", raising=False)
    consent_runtime._reset_runtime_state_for_tests()
    state.set_cli_kill_switch(False)
    monkeypatch.setattr(state, "_session_id", None)
    monkeypatch.setattr(cli, "_consent_mutation_event_count", 0)


def _args(action: str | None, *, no_telemetry: bool = False) -> SimpleNamespace:
    return SimpleNamespace(telemetry_action=action, no_telemetry=no_telemetry)


def _home_snapshot(home: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(home)): path.read_bytes()
        for path in home.rglob("*")
        if path.is_file()
    }


@pytest.mark.parametrize(
    "action",
    ["status", "on", "off", "reset-id", "enable", "disable", "preview", "reset"],
)
def test_parser_accepts_every_telemetry_verb(action):
    args = cli.build_parser().parse_args(["telemetry", action])
    assert args.telemetry_action == action


def test_status_prints_every_required_line(monkeypatch, capsys):
    from rapid_mlx.telemetry import build_gate
    from rapid_mlx.telemetry.build_gate import ReleaseStamp
    from rapid_mlx.telemetry.consent_decision import Decision, WriteBack

    monkeypatch.setattr(
        consent_runtime,
        "resolve",
        lambda: Decision(True, False, WriteBack(False, False, False), "consented"),
    )
    monkeypatch.setattr(consent_runtime, "upload_allowed", lambda: True)
    monkeypatch.setattr(
        build_gate,
        "official_build",
        lambda: ReleaseStamp("stable", "phc_12345678901234567890"),
    )
    cli.telemetry_command(_args("status"))
    out = capsys.readouterr().out
    for expected in (
        "Reporting:  ON",
        "Reason:     consented",
        "Upload:     allowed",
        "Build:      official (stable)",
        "Install ID:",
        "Sent to:    PostHog Cloud (US). No IP, no location, no per-person profile.",
        "Turn off:   rapid-mlx telemetry off",
        "Files:",
        "Rotate ID:  rapid-mlx telemetry reset-id",
        "Details:    https://rapidmlx.com/docs/telemetry",
    ):
        assert expected in out


def test_status_distinguishes_reason_from_blocked_upload(monkeypatch, capsys):
    from rapid_mlx.telemetry import build_gate
    from rapid_mlx.telemetry.consent_decision import Decision, WriteBack

    monkeypatch.setattr(
        consent_runtime,
        "resolve",
        lambda: Decision(True, False, WriteBack(False, False, False), "consented"),
    )
    monkeypatch.setattr(consent_runtime, "upload_allowed", lambda: True)
    monkeypatch.setattr(build_gate, "official_build", lambda: None)
    cli.telemetry_command(_args(None, no_telemetry=True))
    out = capsys.readouterr().out
    assert "Reporting:  OFF" in out
    assert "Reason:     consented" in out
    assert "Upload:     blocked" in out
    assert "Build:      unofficial build — never transmits" in out


def test_status_names_the_active_kill_switch(monkeypatch, capsys):
    from rapid_mlx.telemetry import build_gate
    from rapid_mlx.telemetry.consent_decision import Decision, WriteBack

    monkeypatch.setenv(state.ENV_VAR, "0")
    monkeypatch.setattr(
        consent_runtime,
        "resolve",
        lambda: Decision(False, False, WriteBack(False, False, False), "kill_switch"),
    )
    monkeypatch.setattr(consent_runtime, "upload_allowed", lambda: False)
    monkeypatch.setattr(build_gate, "official_build", lambda: None)
    cli.telemetry_command(_args("status"))
    assert "Reason:     kill_switch (env-var (RAPID_MLX_TELEMETRY='0'))" in (
        capsys.readouterr().out
    )


def test_status_reads_existing_id_without_rewriting_it(capsys):
    path = state.client_id_path()
    path.parent.mkdir(parents=True)
    path.write_text("existing-install-id\n")
    before = path.stat().st_mtime_ns
    cli.telemetry_command(_args("status"))
    assert "Install ID: exis…" in capsys.readouterr().out
    assert path.read_text() == "existing-install-id\n"
    assert path.stat().st_mtime_ns == before


@pytest.mark.parametrize("action", ["on", "enable"])
def test_on_and_enable_share_the_same_path(action, monkeypatch, capsys):
    calls: list[object] = []
    monkeypatch.setattr(
        "rapid_mlx.telemetry.record_consent",
        lambda value, **kwargs: calls.append((value, kwargs)),
    )
    monkeypatch.setattr(cli, "_track_telemetry_opted_in", lambda: calls.append("track"))
    monkeypatch.setattr(
        "rapid_mlx.telemetry.get_or_create_client_id", lambda: "install-id"
    )
    cli.telemetry_command(_args(action))
    assert calls[0][0] is True
    assert calls[1] == "track"
    assert "Telemetry: ENABLED" in capsys.readouterr().out


def test_on_reports_write_failure(monkeypatch, capsys):
    monkeypatch.setattr(
        "rapid_mlx.telemetry.record_consent",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("denied")),
    )
    tracked: list[bool] = []
    monkeypatch.setattr(cli, "_track_telemetry_opted_in", lambda: tracked.append(True))
    with pytest.raises(SystemExit, match="1"):
        cli.telemetry_command(_args("on"))
    assert tracked == []
    assert "could not save telemetry preference" in capsys.readouterr().err


@pytest.mark.parametrize("action", ["off", "disable"])
def test_off_flushes_opt_out_before_recording_false(action, monkeypatch, capsys):
    order: list[object] = []
    sender = SimpleNamespace(flush=lambda timeout: order.append(("flush", timeout)))
    monkeypatch.setattr(
        "rapid_mlx.telemetry.track.track",
        lambda event, props: order.append((event, props)),
    )
    monkeypatch.setattr("rapid_mlx.telemetry.posthog_sender.get_sender", lambda: sender)
    monkeypatch.setattr(
        "rapid_mlx.telemetry.record_consent",
        lambda value, **_kwargs: order.append(("consent", value)),
    )
    cli.telemetry_command(_args(action))
    assert order == [
        ("telemetry_opted_out", {"via": "cli"}),
        ("flush", 2.0),
        ("consent", False),
    ]
    assert "Telemetry: disabled" in capsys.readouterr().out


def test_off_reports_consent_write_failure(monkeypatch, capsys):
    sender = SimpleNamespace(flush=lambda _timeout: None)
    monkeypatch.setattr(
        "rapid_mlx.telemetry.track.track", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr("rapid_mlx.telemetry.posthog_sender.get_sender", lambda: sender)
    monkeypatch.setattr(
        "rapid_mlx.telemetry.record_consent",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("denied")),
    )
    with pytest.raises(SystemExit, match="1"):
        cli.telemetry_command(_args("off"))
    assert "could not save telemetry preference" in capsys.readouterr().err


def test_opt_in_helper_refreshes_delivers_tracks_and_flushes(monkeypatch):
    from rapid_mlx.telemetry.consent_decision import Decision, WriteBack

    order: list[object] = []
    decision = Decision(True, True, WriteBack(False, False, True), "notice")
    monkeypatch.setattr(
        consent_runtime, "refresh_decision", lambda: order.append("refresh") or decision
    )
    monkeypatch.setattr(
        consent_runtime,
        "deliver_notice_if_needed",
        lambda resolved: order.append(("notice", resolved)) or True,
    )
    monkeypatch.setattr(
        consent_runtime,
        "apply_write_back",
        lambda write_back: order.append(("write_back", write_back)) or True,
    )
    monkeypatch.setattr(
        "rapid_mlx.telemetry.track.track",
        lambda event, props: order.append((event, props)),
    )
    monkeypatch.setattr(
        "rapid_mlx.telemetry.posthog_sender.get_sender",
        lambda: SimpleNamespace(flush=lambda timeout: order.append(("flush", timeout))),
    )

    cli._track_telemetry_opted_in()

    assert order == [
        "refresh",
        ("notice", decision),
        ("write_back", decision.write_back),
        ("telemetry_opted_in", {"via": "cli"}),
        ("flush", 2.0),
    ]


def test_consent_event_helpers_are_bounded_and_fail_silent(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(cli, "_consent_mutation_event_count", 5)
    monkeypatch.setattr(
        "rapid_mlx.telemetry.track.track",
        lambda *_args, **_kwargs: calls.append("track"),
    )
    cli._track_telemetry_opted_out()
    cli._track_telemetry_opted_in()
    assert calls == []

    monkeypatch.setattr(cli, "_consent_mutation_event_count", 0)
    monkeypatch.setattr(
        "rapid_mlx.telemetry.track.track",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("unavailable")),
    )
    cli._track_telemetry_opted_out()

    from rapid_mlx.telemetry.consent_decision import Decision, WriteBack

    monkeypatch.setattr(cli, "_consent_mutation_event_count", 0)
    monkeypatch.setattr(
        consent_runtime,
        "refresh_decision",
        lambda: Decision(True, True, WriteBack(False, False, True), "notice"),
    )
    monkeypatch.setattr(
        consent_runtime, "deliver_notice_if_needed", lambda _decision: False
    )
    monkeypatch.setattr(consent_runtime, "notice_was_delivered", lambda: False)
    cli._track_telemetry_opted_in()

    monkeypatch.setattr(cli, "_consent_mutation_event_count", 0)
    monkeypatch.setattr(
        consent_runtime,
        "refresh_decision",
        lambda: (_ for _ in ()).throw(RuntimeError("unavailable")),
    )
    cli._track_telemetry_opted_in()


def test_preview_is_one_real_v2_batch_item(capsys):
    cli.telemetry_command(_args("preview"))
    out = capsys.readouterr().out
    assert "POST https://us.i.posthog.com/batch/" in out
    assert "nothing is sent by this command" in out
    start = out.index("{")
    item = json.loads(out[start : out.rindex("}") + 1])
    assert item["event"] == "app_opened"
    assert item["properties"]["$geoip_disable"] is True
    assert item["properties"]["$process_person_profile"] is False
    assert item["distinct_id"] == item["properties"]["install_id"]
    assert "days_since_first_run_bucket" not in item["properties"]


def test_preview_never_captures_or_flushes(monkeypatch, capsys):
    calls: list[object] = []
    sender = SimpleNamespace(
        capture=lambda item: calls.append(("capture", item)),
        flush=lambda *args: calls.append(("flush", args)),
    )
    monkeypatch.setattr("rapid_mlx.telemetry.posthog_sender.get_sender", lambda: sender)

    cli.telemetry_command(_args("preview"))

    assert calls == []
    assert "nothing is sent by this command" in capsys.readouterr().out


def test_preview_creates_identity_only_when_upload_is_allowed(monkeypatch, capsys):
    from rapid_mlx.telemetry import build_gate
    from rapid_mlx.telemetry.build_gate import ReleaseStamp

    monkeypatch.setattr(
        build_gate,
        "official_build",
        lambda: ReleaseStamp("stable", "phc_12345678901234567890"),
    )
    monkeypatch.setattr(consent_runtime, "upload_allowed", lambda: True)
    cli.telemetry_command(_args("preview"))
    assert state.client_id_path().exists()
    assert "nothing is sent by this command" in capsys.readouterr().out


@pytest.mark.parametrize("action", ["status", "preview"])
@pytest.mark.parametrize(
    ("gate", "no_telemetry"),
    [
        ("do_not_track", False),
        ("rapid_mlx_telemetry", False),
        ("cli_flag", True),
        ("stored_refusal", False),
    ],
)
def test_blocked_read_only_commands_leave_home_unchanged(
    action, gate, no_telemetry, tmp_path, monkeypatch, capsys
):
    from rapid_mlx.telemetry import build_gate
    from rapid_mlx.telemetry.build_gate import ReleaseStamp

    monkeypatch.setattr(
        build_gate,
        "official_build",
        lambda: ReleaseStamp("stable", "phc_12345678901234567890"),
    )
    if gate == "do_not_track":
        monkeypatch.setenv(state.DO_NOT_TRACK_ENV, "1")
    elif gate == "rapid_mlx_telemetry":
        monkeypatch.setenv(state.ENV_VAR, "0")
    elif gate == "stored_refusal":
        state.record_consent(False, rapid_mlx_version="0.15.0")

    before = _home_snapshot(tmp_path)
    cli.telemetry_command(_args(action, no_telemetry=no_telemetry))
    output = capsys.readouterr().out

    assert _home_snapshot(tmp_path) == before
    if action == "status":
        assert "Install ID: (not created)" in output
    else:
        item = json.loads(output[output.index("{") : output.rindex("}") + 1])
        assert "days_since_first_run_bucket" not in item["properties"]


def test_preview_prints_null_if_common_snapshot_is_invalid(monkeypatch, capsys):
    monkeypatch.setattr(
        "rapid_mlx.telemetry.common_props.build_common_props", lambda **_kwargs: None
    )
    cli.telemetry_command(_args("preview"))
    assert "\nnull\n" in capsys.readouterr().out


def test_reset_id_rotates_only_identity(monkeypatch, capsys):
    monkeypatch.setattr(state, "rotate_client_id", lambda: "new-id")
    cli.telemetry_command(_args("reset-id"))
    assert "Consent is unchanged" in capsys.readouterr().out


def test_reset_id_reports_failure(monkeypatch, capsys):
    monkeypatch.setattr(
        state,
        "rotate_client_id",
        lambda: (_ for _ in ()).throw(OSError("denied")),
    )
    with pytest.raises(SystemExit, match="1"):
        cli.telemetry_command(_args("reset-id"))
    assert "could not rotate telemetry identity" in capsys.readouterr().err


def test_reset_deletes_preference_rotates_id_and_emits_no_event(monkeypatch, capsys):
    calls: list[object] = []
    success = state.ResetStateResult(
        consent_file=state.ResetItemResult(True, True),
        consent_lock=state.ResetItemResult(True, True),
        client_id=state.ResetItemResult(True, True),
    )
    monkeypatch.setattr(state, "reset_state", lambda: calls.append("reset") or success)
    monkeypatch.setattr(
        "rapid_mlx.telemetry.track.track",
        lambda *args, **kwargs: calls.append(("track", args, kwargs)),
    )
    cli.telemetry_command(_args("reset"))
    assert calls == ["reset"]
    output = capsys.readouterr().out
    assert "deletes your stored preference" in output
    assert "client ID rotated" in output
    assert "desktop clears its answer" in output
    assert "next run is treated as a new install" in output
    assert "emits no telemetry event" in output


def test_reset_is_best_effort(monkeypatch, capsys):
    success = state.ResetStateResult(
        consent_file=state.ResetItemResult(False, True),
        consent_lock=state.ResetItemResult(False, True),
        client_id=state.ResetItemResult(False, True),
    )
    monkeypatch.setattr(state, "reset_state", lambda: success)
    cli.telemetry_command(_args("reset"))
    assert capsys.readouterr().err == ""


def test_reset_reports_marker_failure_separately(monkeypatch, capsys):
    result = state.ResetStateResult(
        consent_file=state.ResetItemResult(False, True),
        consent_lock=state.ResetItemResult(False, True),
        client_id=state.ResetItemResult(True, True),
        activation_markers=(state.ResetItemResult(True, False, ("PermissionError",)),),
    )
    monkeypatch.setattr(state, "reset_state", lambda: result)

    with pytest.raises(SystemExit, match="1"):
        cli.telemetry_command(_args("reset"))

    assert "activation marker(s) (PermissionError) remained" in capsys.readouterr().out


def test_reset_reports_marker_scan_and_id_rotation_failures(monkeypatch, capsys):
    result = state.ResetStateResult(
        consent_file=state.ResetItemResult(False, True),
        consent_lock=state.ResetItemResult(False, True),
        client_id=state.ResetItemResult(True, True),
        activation_marker_scan=state.ResetItemResult(True, False, ("OSError",)),
        client_id_rotation_errors=("PermissionError",),
    )
    monkeypatch.setattr(state, "reset_state", lambda: result)

    with pytest.raises(SystemExit, match="1"):
        cli.telemetry_command(_args("reset"))

    output = capsys.readouterr().out
    assert "Activation marker scan (OSError) failed" in output
    assert "Client ID rotation (PermissionError) failed" in output


def test_reset_reports_unwritable_state_and_keeps_files(capsys):
    state.record_consent(True, rapid_mlx_version="0.15.0")
    state.get_or_create_client_id()
    telemetry_dir = state.consent_path().parent
    os.chmod(telemetry_dir, 0o500)
    try:
        with pytest.raises(SystemExit, match="1"):
            cli.telemetry_command(_args("reset"))
    finally:
        os.chmod(telemetry_dir, 0o700)

    output = capsys.readouterr().out
    assert "Reset incomplete:" in output
    assert "PermissionError" in output
    assert state.consent_path().exists()
    assert state.client_id_path().exists()


def test_real_cli_reset_reports_unremovable_marker_not_removed_client_id(tmp_path):
    telemetry_dir = tmp_path / ".rapid-mlx"
    marker = telemetry_dir / "activation_seen_server"
    marker.mkdir(parents=True)
    client_id = telemetry_dir / "telemetry-client-id"
    client_id.write_text("old-client-id\n")

    result = subprocess.run(
        [sys.executable, "-m", "rapid_mlx.cli", "telemetry", "reset"],
        capture_output=True,
        text=True,
        check=False,
        env=os.environ.copy(),
    )

    assert result.returncode == 1
    error_type = "PermissionError" if sys.platform == "darwin" else "IsADirectoryError"
    assert (
        f"Reset incomplete: activation marker(s) ({error_type}) remained."
        in result.stdout
    )
    assert "client ID" not in result.stdout
    assert marker.is_dir()
    assert not client_id.exists()


def test_reset_empty_home_succeeds_without_creating_state(capsys, tmp_path):
    cli.telemetry_command(_args("reset"))

    assert list(tmp_path.rglob("*")) == []
    assert "no stored preference or client ID found" in capsys.readouterr().out


def test_unknown_action_is_defensively_rejected(capsys):
    with pytest.raises(SystemExit, match="1"):
        cli.telemetry_command(_args("wat"))
    assert "Unknown telemetry action" in capsys.readouterr().out
