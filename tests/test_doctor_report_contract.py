# SPDX-License-Identifier: Apache-2.0
"""Machine-readable and selective execution contracts for Doctor v2."""

from __future__ import annotations

import io
import json
from argparse import Namespace

import pytest

from vllm_mlx.doctor import env_health as eh
from vllm_mlx.doctor.cli import doctor_command, render_json, render_summary


def _report() -> eh.Report:
    section = eh.Section("Network", id="network", duration_ms=12)
    section.add(
        "API token=secret-value under /Users/tester/project",
        eh.CheckStatus.WARN,
        detail="password=hunter2",
        check_id="network.credentials",
    )
    return eh.Report(sections=[section], duration_ms=14)


def test_json_report_has_versioned_stable_shape(monkeypatch):
    monkeypatch.setenv("HOME", "/Users/tester")
    output = io.StringIO()

    render_json(_report(), stream=output)
    document = json.loads(output.getvalue())

    assert document["schemaVersion"] == 1
    assert document["status"] == "warn"
    assert document["exitCode"] == 0
    assert document["durationMs"] == 14
    assert document["sections"][0]["id"] == "network"
    check = document["sections"][0]["checks"][0]
    assert check["id"] == "network.credentials"
    assert "secret-value" not in check["summary"]
    assert "/Users/tester" not in check["summary"]
    assert "hunter2" not in check["detail"]


def test_summary_is_one_line():
    output = io.StringIO()
    render_summary(_report(), stream=output)
    assert output.getvalue().count("\n") == 1
    assert "warn" in output.getvalue()


def test_overall_status_covers_ok_and_fail():
    ok_section = eh.Section("OK")
    ok_section.add("healthy", eh.CheckStatus.OK)
    assert eh.Report(sections=[ok_section]).overall_status == "ok"

    failed_section = eh.Section("Failed")
    failed_section.add("broken", eh.CheckStatus.FAIL)
    assert eh.Report(sections=[failed_section]).overall_status == "fail"


def test_run_all_only_and_skip_use_stable_section_ids(monkeypatch):
    def alpha() -> eh.Section:
        section = eh.Section("Alpha")
        section.add("a", eh.CheckStatus.OK)
        return section

    def beta() -> eh.Section:
        section = eh.Section("Beta")
        section.add("b", eh.CheckStatus.OK)
        return section

    monkeypatch.setattr(eh, "_SECTION_BUILDERS", (alpha, beta))
    monkeypatch.setattr(eh, "_selected_runtime", lambda: None)

    report = eh.run_all(only={"alpha", "beta"}, skip={"beta"})

    assert [section.id for section in report.sections] == ["alpha"]
    assert report.sections[0].checks[0].id == "alpha.001"


def test_budget_exhaustion_is_skipped_not_warning(monkeypatch):
    monkeypatch.setattr(eh.time, "monotonic", lambda: 100.0)
    report = eh._run_all_serialized(99.0)
    assert report.n_warn == 0
    assert report.n_skipped == len(report.sections)
    assert all(
        section.checks[0].status is eh.CheckStatus.SKIPPED
        for section in report.sections
    )


def test_doctor_json_does_not_mix_human_output(monkeypatch, capsys):
    monkeypatch.setattr("vllm_mlx.doctor.cli.run_all", lambda **_: _report())
    args = Namespace(
        tier=None,
        verbose=False,
        json=True,
        summary=False,
        only=["network"],
        skip=None,
    )
    with pytest.raises(SystemExit) as exc:
        doctor_command(args)
    assert exc.value.code == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["sections"][0]["id"] == "network"


@pytest.mark.parametrize(
    "summary,expected",
    [
        (True, "Rapid-MLX Doctor: warn"),
        (False, "◆ Network"),
    ],
)
def test_doctor_human_output_modes(monkeypatch, capsys, summary, expected):
    monkeypatch.setattr("vllm_mlx.doctor.cli.run_all", lambda **_: _report())
    args = Namespace(
        tier=None,
        verbose=False,
        json=False,
        summary=summary,
        only=None,
        skip=None,
    )
    with pytest.raises(SystemExit) as exc:
        doctor_command(args)
    assert exc.value.code == 0
    assert expected in capsys.readouterr().out
