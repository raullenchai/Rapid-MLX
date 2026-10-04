# SPDX-License-Identifier: Apache-2.0
"""CLI wiring for the closed ``command`` prop on ``app_opened``.

Bare ``rapid-mlx`` and the ``--help`` / ``--version`` flags must never print
the telemetry disclosure; they report the launch only once the notice was
delivered on an earlier run.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

import rapid_mlx.cli as cli
from rapid_mlx.telemetry import consent_runtime


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["--help"], "help"),
        (["-h"], "help"),
        (["serve", "--help"], "help"),
        (["--version"], "version"),
        (["-V"], "version"),
        (["chat"], None),
        (["share", "m", "--", "--help"], None),
    ],
)
def test_help_or_version_flag(argv, expected):
    assert cli._help_or_version_flag(argv) == expected


def _record_quiet(monkeypatch):
    calls: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        cli,
        "_start_quiet_lifecycle",
        lambda command, *, no_telemetry: calls.append((command, no_telemetry)),
    )
    return calls


@pytest.mark.parametrize(
    ("flag", "expected"), [("--help", "help"), ("--version", "version")]
)
def test_top_level_flags_report_quietly_and_keep_stdout(
    monkeypatch, capsys, flag, expected
):
    calls = _record_quiet(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["rapid-mlx", "--no-telemetry", flag])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 0
    assert calls == [(expected, True)]
    out = capsys.readouterr().out
    assert out.startswith("usage:" if expected == "help" else "rapid-mlx ")


def test_parse_errors_report_nothing(monkeypatch, capsys):
    calls = _record_quiet(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["rapid-mlx", "not-a-command", "--help"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
    assert calls == []
    capsys.readouterr()


def test_bare_command_reports_quietly(monkeypatch, capsys):
    calls = _record_quiet(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["rapid-mlx"])
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False, raising=False)
    with pytest.raises(SystemExit):
        cli.main()
    assert calls == [("bare", False)]
    capsys.readouterr()


def test_quiet_lifecycle_sends_nothing_while_the_notice_is_pending(monkeypatch):
    events: list[str] = []
    monkeypatch.setattr(
        consent_runtime, "resolve", lambda: SimpleNamespace(deliver_notice=True)
    )
    monkeypatch.setattr(
        consent_runtime, "startup", lambda **_k: events.append("startup")
    )
    monkeypatch.setattr(cli, "_start_v2_lifecycle", events.append)
    cli._start_quiet_lifecycle("bare", no_telemetry=False)
    assert events == []


def test_quiet_lifecycle_reports_once_disclosed(monkeypatch):
    events: list[str] = []
    switch: list[bool] = []
    monkeypatch.setattr("rapid_mlx.telemetry.state.set_cli_kill_switch", switch.append)
    monkeypatch.setattr(
        consent_runtime, "resolve", lambda: SimpleNamespace(deliver_notice=False)
    )
    monkeypatch.setattr(
        consent_runtime, "startup", lambda **_k: events.append("startup")
    )
    monkeypatch.setattr(cli, "_start_v2_lifecycle", events.append)
    cli._start_quiet_lifecycle("help", no_telemetry=True)
    assert switch == [True]
    assert events == ["startup", "help"]


def test_quiet_lifecycle_never_raises(monkeypatch):
    monkeypatch.setattr(
        consent_runtime,
        "resolve",
        lambda: (_ for _ in ()).throw(RuntimeError("consent store broken")),
    )
    cli._start_quiet_lifecycle("bare", no_telemetry=False)


def test_quiet_lifecycle_prints_nothing_on_a_first_run(monkeypatch, capsys):
    # Real consent runtime against the isolated test state dir: a pending
    # first-run notice must neither print nor be marked delivered.
    monkeypatch.setattr(
        consent_runtime, "resolve", lambda: SimpleNamespace(deliver_notice=True)
    )
    cli._start_quiet_lifecycle("bare", no_telemetry=False)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
