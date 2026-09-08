# SPDX-License-Identifier: Apache-2.0
"""Opt-in deep probes and detect-plan-repair-verify behavior."""

from __future__ import annotations

import io
import json
import subprocess
from argparse import Namespace
from pathlib import Path

import pytest

from vllm_mlx.doctor import env_health as eh
from vllm_mlx.doctor.cli import doctor_command, render, report_document
from vllm_mlx.doctor.repairs import (
    RepairAction,
    RepairResult,
    apply_repairs,
    plan_repairs,
    service_is_live,
)


def test_deep_runtime_reports_dependency_dns_and_route(monkeypatch):
    monkeypatch.setattr(
        eh, "_selected_runtime", lambda: (Path("/usr/bin/python3"), False)
    )
    monkeypatch.setattr(eh.sys, "platform", "darwin")

    def run(argv, **_kwargs):
        if argv[-3:] == ["-m", "pip", "check"]:
            return subprocess.CompletedProcess(
                argv, 0, "No broken requirements found.\n", ""
            )
        if argv[0] == "/sbin/route":
            return subprocess.CompletedProcess(argv, 0, " interface: en0\n", "")
        return subprocess.CompletedProcess(argv, 0, "ok\n", "")

    section = eh.section_deep_runtime(run=run)
    by_id = {check.id: check for check in section.checks}
    assert by_id["deep.python.pip-check"].status is eh.CheckStatus.OK
    assert by_id["deep.network.dns"].status is eh.CheckStatus.OK
    assert "en0" in by_id["deep.network.default-route"].label


def test_deep_timeout_is_skipped_not_failure(monkeypatch):
    monkeypatch.setattr(
        eh, "_selected_runtime", lambda: (Path("/usr/bin/python3"), False)
    )
    monkeypatch.setattr(eh.sys, "platform", "linux")

    def run(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(["probe"], 1)

    section = eh.section_deep_runtime(run=run)
    assert all(check.status is eh.CheckStatus.SKIPPED for check in section.checks)


def test_deep_operating_system_errors_are_warnings(monkeypatch):
    monkeypatch.setattr(
        eh, "_selected_runtime", lambda: (Path("/usr/bin/python3"), False)
    )
    monkeypatch.setattr(eh.sys, "platform", "linux")

    def unavailable(*_args, **_kwargs):
        raise OSError("operation unavailable")

    section = eh.section_deep_runtime(run=unavailable)
    by_id = {check.id: check for check in section.checks}
    assert by_id["deep.python.pip-check"].status is eh.CheckStatus.WARN
    assert by_id["deep.network.dns"].status is eh.CheckStatus.WARN


def test_deep_route_operating_system_error_is_warning(monkeypatch):
    monkeypatch.setattr(
        eh, "_selected_runtime", lambda: (Path("/usr/bin/python3"), False)
    )
    monkeypatch.setattr(eh.sys, "platform", "darwin")

    def run(argv, **_kwargs):
        if argv[0] == "/sbin/route":
            raise OSError("route unavailable")
        return subprocess.CompletedProcess(argv, 0, "ok", "")

    section = eh.section_deep_runtime(run=run)
    route = next(
        check for check in section.checks if check.id == "deep.network.default-route"
    )
    assert route.status is eh.CheckStatus.WARN


def test_only_deep_enables_deep_probe(monkeypatch):
    calls = []

    def run_all(**kwargs):
        calls.append(kwargs)
        return eh.Report()

    monkeypatch.setattr("vllm_mlx.doctor.cli.run_all", run_all)
    args = Namespace(
        tier=None,
        verbose=False,
        json=False,
        summary=True,
        only=["deep"],
        skip=None,
        fix=False,
        dry_run=False,
        yes=False,
        deep=False,
    )

    with pytest.raises(SystemExit) as exc:
        doctor_command(args)

    assert exc.value.code == 0
    assert calls == [{"only": {"deep"}, "skip": None, "deep": True}]


def test_pipless_bundled_runtime_is_skipped_not_dependency_failure(monkeypatch):
    monkeypatch.setattr(eh, "_selected_runtime", lambda: (Path("/app/python"), True))
    monkeypatch.setattr(eh, "_bundled_sidecar_root", lambda _runtime: Path("/app"))
    monkeypatch.setattr(eh, "_runtime_environment", lambda _runtime: "unknown")
    monkeypatch.setattr(eh.sys, "platform", "linux")

    def run(argv, **_kwargs):
        if "pip" in argv:
            return subprocess.CompletedProcess(argv, 1, "", "No module named pip")
        return subprocess.CompletedProcess(argv, 0, "ok", "")

    section = eh.section_deep_runtime(run=run)
    pip_check = next(c for c in section.checks if c.id == "deep.python.pip-check")
    assert pip_check.status is eh.CheckStatus.SKIPPED


def test_missing_pip_in_normal_runtime_is_warning(monkeypatch):
    monkeypatch.setattr(
        eh, "_selected_runtime", lambda: (Path("/usr/local/bin/python3"), False)
    )
    monkeypatch.setattr(eh, "_runtime_environment", lambda _runtime: "system")
    monkeypatch.setattr(eh.sys, "platform", "linux")

    def run(argv, **_kwargs):
        if "pip" in argv:
            return subprocess.CompletedProcess(
                argv, 1, "starting pip", "No module named pip"
            )
        return subprocess.CompletedProcess(argv, 0, "ok", "")

    pip_check = eh.section_deep_runtime(run=run).checks[0]
    assert pip_check.status is eh.CheckStatus.WARN
    assert "starting pip" in pip_check.detail
    assert "No module named pip" in pip_check.detail


def test_pip_check_separates_conflicts_from_operational_failures(monkeypatch):
    monkeypatch.setattr(
        eh, "_selected_runtime", lambda: (Path("/usr/bin/python3"), False)
    )
    monkeypatch.setattr(eh.sys, "platform", "linux")

    def run_conflict(argv, **_kwargs):
        if "pip" in argv:
            return subprocess.CompletedProcess(
                argv, 1, "demo 1.0 requires missing, which is not installed.\n", ""
            )
        return subprocess.CompletedProcess(argv, 0, "ok", "")

    conflict = eh.section_deep_runtime(run=run_conflict).checks[0]
    assert conflict.status is eh.CheckStatus.FAIL

    def run_unsupported(argv, **_kwargs):
        if "pip" in argv:
            return subprocess.CompletedProcess(
                argv, 1, "demo 1.0 is not supported on this platform\n", ""
            )
        return subprocess.CompletedProcess(argv, 0, "ok", "")

    unsupported = eh.section_deep_runtime(run=run_unsupported).checks[0]
    assert unsupported.status is eh.CheckStatus.FAIL

    def run_bad_metadata(argv, **_kwargs):
        if "pip" in argv:
            return subprocess.CompletedProcess(
                argv, 1, "", "WARNING: Error parsing dependencies of demo: bad metadata"
            )
        return subprocess.CompletedProcess(argv, 0, "ok", "")

    bad_metadata = eh.section_deep_runtime(run=run_bad_metadata).checks[0]
    assert bad_metadata.status is eh.CheckStatus.FAIL

    def run_operational_failure(argv, **_kwargs):
        if "pip" in argv:
            return subprocess.CompletedProcess(argv, 1, "", "permission denied")
        return subprocess.CompletedProcess(argv, 0, "ok", "")

    operational = eh.section_deep_runtime(run=run_operational_failure).checks[0]
    assert operational.status is eh.CheckStatus.WARN


def _service_report(*, process=eh.CheckStatus.FAIL, live=eh.CheckStatus.FAIL):
    section = eh.Section("Always-on Service", id="service")
    section.add("registered", eh.CheckStatus.OK, check_id="service.registration")
    section.add("process", process, check_id="service.process")
    section.add("live", live, check_id="service.endpoint.liveness")
    return eh.Report(sections=[section])


def test_repair_plan_only_targets_registered_unhealthy_service():
    actions = plan_repairs(_service_report())
    assert [action.id for action in actions] == ["repair.service.kickstart"]
    assert actions[0].command[0] == "/bin/launchctl"

    healthy = _service_report(process=eh.CheckStatus.OK, live=eh.CheckStatus.OK)
    assert plan_repairs(healthy) == []


def test_repair_dry_run_never_executes():
    action = plan_repairs(_service_report())

    def must_not_run(*_args, **_kwargs):
        raise AssertionError("dry-run executed a command")

    result = apply_repairs(action, dry_run=True, run=must_not_run)
    assert result[0].status == "planned"


def test_declined_interactive_repair_is_recorded(monkeypatch, capsys):
    class InteractiveInput(io.StringIO):
        def isatty(self):
            return True

    monkeypatch.setattr(
        "vllm_mlx.doctor.cli.run_all", lambda **_kwargs: _service_report()
    )
    monkeypatch.setattr("vllm_mlx.doctor.cli.sys.stdin", InteractiveInput("no\n"))
    monkeypatch.setattr(
        "vllm_mlx.doctor.repairs.plan_repairs",
        lambda _report: [
            RepairAction(
                id="repair.test",
                summary="token=secret-value",
                command=("/bin/launchctl", "password=hunter2"),
            )
        ],
    )

    def must_not_apply(*_args, **_kwargs):
        raise AssertionError("declined repair was applied")

    monkeypatch.setattr("vllm_mlx.doctor.repairs.apply_repairs", must_not_apply)
    args = Namespace(
        tier=None,
        verbose=True,
        json=False,
        summary=False,
        only=["service"],
        skip=None,
        fix=True,
        dry_run=False,
        yes=False,
        deep=False,
    )
    with pytest.raises(SystemExit) as exc:
        doctor_command(args)

    assert exc.value.code == 1
    rendered = capsys.readouterr()
    assert "not_applied" in rendered.out
    assert "operator declined" in rendered.out
    assert "secret-value" not in rendered.out + rendered.err
    assert "hunter2" not in rendered.out + rendered.err


def test_fix_json_uses_schema_v2_even_when_no_action_is_needed(monkeypatch, capsys):
    healthy = _service_report(process=eh.CheckStatus.OK, live=eh.CheckStatus.OK)
    monkeypatch.setattr(
        "vllm_mlx.doctor.cli._collect_report_isolated", lambda **_kwargs: healthy
    )
    args = Namespace(
        tier=None,
        verbose=False,
        json=True,
        summary=False,
        only=["service"],
        skip=None,
        fix=True,
        dry_run=False,
        yes=True,
        deep=False,
    )
    with pytest.raises(SystemExit) as exc:
        doctor_command(args)

    assert exc.value.code == 0
    document = json.loads(capsys.readouterr().out)
    assert document["schemaVersion"] == 2
    assert document["repairs"] == []


def test_noninteractive_fix_requires_yes_before_diagnosis(monkeypatch):
    monkeypatch.setattr("vllm_mlx.doctor.cli.sys.stdin", io.StringIO())
    monkeypatch.setattr(
        "vllm_mlx.doctor.cli._collect_report_isolated",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("diagnosis ran before authorization")
        ),
    )
    args = Namespace(
        tier=None,
        verbose=False,
        json=True,
        summary=False,
        only=["service"],
        skip=None,
        fix=True,
        dry_run=False,
        yes=False,
        deep=False,
    )
    with pytest.raises(SystemExit) as exc:
        doctor_command(args)
    assert exc.value.code == 2


@pytest.mark.parametrize("flag", ["dry_run", "yes"])
def test_repair_only_flags_require_fix(flag, capsys):
    args = Namespace(
        tier=None,
        verbose=False,
        json=False,
        summary=False,
        only=None,
        skip=None,
        fix=False,
        dry_run=flag == "dry_run",
        yes=flag == "yes",
        deep=False,
    )

    with pytest.raises(SystemExit) as exc:
        doctor_command(args)

    assert exc.value.code == 2
    assert "requires --fix" in capsys.readouterr().err


def test_verified_repair_triggers_fresh_deep_report(monkeypatch, capsys):
    reports = iter([_service_report(), eh.Report()])
    collections = []

    def collect(**kwargs):
        collections.append(kwargs)
        return next(reports)

    monkeypatch.setattr("vllm_mlx.doctor.cli._collect_report_isolated", collect)
    monkeypatch.setattr(
        "vllm_mlx.doctor.repairs.plan_repairs",
        lambda _report: [
            RepairAction(
                id="repair.test",
                summary="restart service",
                command=("/bin/launchctl", "kickstart", "service"),
            )
        ],
    )
    monkeypatch.setattr(
        "vllm_mlx.doctor.repairs.apply_repairs",
        lambda *_args, **_kwargs: [
            RepairResult(
                id="repair.test",
                status="verified",
                summary="restart service",
                command=["/bin/launchctl", "kickstart", "service"],
                detail="verification passed",
            )
        ],
    )
    args = Namespace(
        tier=None,
        verbose=False,
        json=True,
        summary=False,
        only=["service"],
        skip=None,
        fix=True,
        dry_run=False,
        yes=True,
        deep=False,
    )

    with pytest.raises(SystemExit) as exc:
        doctor_command(args)

    assert exc.value.code == 0
    assert [call["deep"] for call in collections] == [True, True]
    document = json.loads(capsys.readouterr().out)
    assert document["schemaVersion"] == 2
    assert document["repairs"][0]["status"] == "verified"


def test_repair_never_invokes_sudo_implicitly():
    action = plan_repairs(_service_report())

    def must_not_run(*_args, **_kwargs):
        raise AssertionError("non-root repair executed a command")

    result = apply_repairs(action, run=must_not_run, geteuid=lambda: 501)
    assert result[0].status == "not_applied"
    assert result[0].command[0] == "sudo"


def test_repair_is_success_only_after_verification():
    action = plan_repairs(_service_report())
    probes = iter((False, True))

    def run(argv, **_kwargs):
        return subprocess.CompletedProcess(argv, 0, "", "")

    result = apply_repairs(
        action,
        run=run,
        geteuid=lambda: 0,
        verify_service=lambda: next(probes),
        sleep=lambda _seconds: None,
    )
    assert result[0].status == "verified"
    assert "passed" in result[0].detail


def test_repair_command_success_without_health_is_unverified(monkeypatch):
    action = plan_repairs(_service_report())
    clock = iter((0.0, 0.0, 1.0, 1.0))
    monkeypatch.setattr("vllm_mlx.doctor.repairs.time.monotonic", lambda: next(clock))

    def run(argv, **_kwargs):
        return subprocess.CompletedProcess(argv, 0, "", "")

    result = apply_repairs(
        action,
        run=run,
        geteuid=lambda: 0,
        verify_service=lambda: False,
        sleep=lambda _seconds: None,
        verify_timeout_s=1.0,
    )
    assert result[0].status == "unverified"


def test_repair_without_verifier_or_with_crashing_verifier_is_unverified(monkeypatch):
    action = plan_repairs(_service_report())
    run = lambda argv, **_kwargs: subprocess.CompletedProcess(argv, 0, "", "")

    missing = apply_repairs(action, run=run, geteuid=lambda: 0)
    assert missing[0].status == "unverified"

    clock = iter((0.0, 0.0, 1.0, 1.0))
    monkeypatch.setattr("vllm_mlx.doctor.repairs.time.monotonic", lambda: next(clock))
    crashing = apply_repairs(
        action,
        run=run,
        geteuid=lambda: 0,
        verify_service=lambda: (_ for _ in ()).throw(RuntimeError("probe failed")),
        sleep=lambda _seconds: None,
        verify_timeout_s=1.0,
    )
    assert crashing[0].status == "unverified"
    assert "RuntimeError: probe failed" in crashing[0].detail


def test_repair_nonzero_without_output_has_actionable_detail():
    action = plan_repairs(_service_report())
    result = apply_repairs(
        action,
        run=lambda argv, **_kwargs: subprocess.CompletedProcess(argv, 7, "", ""),
        geteuid=lambda: 0,
    )
    assert result[0].status == "failed"
    assert "status 7" in result[0].detail


def test_repair_execution_error_is_failed():
    action = plan_repairs(_service_report())

    def unavailable(*_args, **_kwargs):
        raise OSError("launchctl unavailable")

    result = apply_repairs(action, run=unavailable, geteuid=lambda: 0)
    assert result[0].status == "failed"
    assert "launchctl unavailable" in result[0].detail


def test_repair_retries_after_transient_verifier_exception():
    action = plan_repairs(_service_report())
    probes = iter((RuntimeError("not ready"), True))

    def verify():
        result = next(probes)
        if isinstance(result, Exception):
            raise result
        return result

    result = apply_repairs(
        action,
        run=lambda argv, **_kwargs: subprocess.CompletedProcess(argv, 0, "", ""),
        geteuid=lambda: 0,
        verify_service=verify,
        sleep=lambda _seconds: None,
    )
    assert result[0].status == "verified"


def test_service_verifier_requires_registration_pid_and_liveness(monkeypatch):
    healthy = {"registered": True, "pid": 42, "livez": True}
    status = dict(healthy)
    monkeypatch.setattr(
        "vllm_mlx.headless_service.status.collect_status", lambda **_kwargs: status
    )
    assert service_is_live() is True

    for missing in ("registered", "pid", "livez"):
        value = dict(healthy)
        value[missing] = None
        status.clear()
        status.update(value)
        assert service_is_live() is False

    for invalid_pid in (True, 0, -1, "42"):
        status.clear()
        status.update(healthy, pid=invalid_pid)
        assert service_is_live() is False

    status.clear()
    status.update(healthy, livez={"error": "timeout"})
    assert service_is_live() is False


def test_repair_json_redacts_result_strings():
    report = eh.Report(
        repairs=[
            {
                "id": "repair.test",
                "status": "failed",
                "summary": "token=secret-value",
                "command": ["tool", "password=hunter2"],
                "detail": "/Users/tester token=another-secret",
            }
        ]
    )

    document = report_document(report)
    assert document["schemaVersion"] == 2
    rendered = str(document["repairs"])
    assert "secret-value" not in rendered
    assert "hunter2" not in rendered
    assert "another-secret" not in rendered


def test_repair_human_output_redacts_result_strings():
    report = eh.Report(
        repairs=[
            {
                "id": "repair.test",
                "status": "failed",
                "summary": "token=secret-value",
                "command": ["tool", "password=hunter2"],
                "detail": "token=another-secret",
            }
        ]
    )
    output = io.StringIO()
    render(report, verbose=True, stream=output)

    rendered = output.getvalue()
    assert "secret-value" not in rendered
    assert "hunter2" not in rendered
    assert "another-secret" not in rendered
