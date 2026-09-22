# SPDX-License-Identifier: Apache-2.0
"""User-facing telemetry v2 CLI contracts."""

from __future__ import annotations

import json
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
    monkeypatch.setattr(state, "_session_id", None)


def _args(action: str | None, *, no_telemetry: bool = False) -> SimpleNamespace:
    return SimpleNamespace(telemetry_action=action, no_telemetry=no_telemetry)


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


@pytest.mark.parametrize("action", ["on", "enable"])
def test_on_and_enable_share_the_same_path(action, monkeypatch, capsys):
    calls: list[object] = []
    monkeypatch.setattr(
        "rapid_mlx.telemetry.record_consent",
        lambda value, **kwargs: calls.append((value, kwargs)),
    )
    monkeypatch.setattr(
        consent_runtime,
        "apply_write_back",
        lambda write_back: calls.append(write_back) or True,
    )
    monkeypatch.setattr(
        "rapid_mlx.telemetry.get_or_create_client_id", lambda: "install-id"
    )
    cli.telemetry_command(_args(action))
    assert calls[0][0] is True
    assert calls[1].mark_notice_seen is True
    assert "Telemetry: ENABLED" in capsys.readouterr().out


def test_on_reports_write_failure(monkeypatch, capsys):
    monkeypatch.setattr(
        "rapid_mlx.telemetry.record_consent", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(consent_runtime, "apply_write_back", lambda _write_back: False)
    with pytest.raises(SystemExit, match="1"):
        cli.telemetry_command(_args("on"))
    assert "could not save telemetry preference" in capsys.readouterr().err


@pytest.mark.parametrize("action", ["off", "disable"])
def test_off_flushes_opt_out_before_recording_false(action, monkeypatch, capsys):
    order: list[object] = []
    sender = SimpleNamespace(flush=lambda: order.append("flush"))
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
        "flush",
        ("consent", False),
    ]
    assert "Telemetry: disabled" in capsys.readouterr().out


def test_off_reports_consent_write_failure(monkeypatch, capsys):
    sender = SimpleNamespace(flush=lambda: None)
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


def test_reset_records_false_and_rotates(monkeypatch, capsys):
    calls: list[object] = []
    monkeypatch.setattr(
        "rapid_mlx.telemetry.record_consent",
        lambda value, **_kwargs: calls.append(value),
    )
    monkeypatch.setattr(state, "rotate_client_id", lambda: calls.append("rotate"))
    cli.telemetry_command(_args("reset"))
    assert calls == [False, "rotate"]
    assert "Telemetry disabled and client ID removed." in capsys.readouterr().out


def test_reset_reports_failure(monkeypatch, capsys):
    monkeypatch.setattr(
        "rapid_mlx.telemetry.record_consent",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("denied")),
    )
    with pytest.raises(SystemExit, match="1"):
        cli.telemetry_command(_args("reset"))
    assert "could not reset telemetry" in capsys.readouterr().err


def test_unknown_action_is_defensively_rejected(capsys):
    with pytest.raises(SystemExit, match="1"):
        cli.telemetry_command(_args("wat"))
    assert "Unknown telemetry action" in capsys.readouterr().out
