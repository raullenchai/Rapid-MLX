# SPDX-License-Identifier: Apache-2.0
"""Machine-readable and selective execution contracts for Doctor v2."""

from __future__ import annotations

import copy
import io
import json
import os
import signal
import subprocess
import sys
import time
from argparse import Namespace
from pathlib import Path

import pytest

from vllm_mlx.doctor import cli as doctor_cli
from vllm_mlx.doctor import env_health as eh
from vllm_mlx.doctor import json_worker
from vllm_mlx.doctor.cli import (
    _collect_report_isolated,
    _open_collector_pipes,
    _report_from_document,
    doctor_command,
    render,
    render_json,
    render_summary,
    report_document,
)


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


def test_json_redacts_complete_authorization_header():
    section = eh.Section("Auth", id="auth")
    section.add(
        "Authorization: Bearer abc123",
        eh.CheckStatus.WARN,
        detail="Authorization=Basic dXNlcjpwYXNz\nnext=safe",
    )
    output = io.StringIO()
    render_json(eh.Report(sections=[section]), stream=output)
    rendered = output.getvalue()
    assert "abc123" not in rendered
    assert "dXNlcjpwYXNz" not in rendered
    assert "next=safe" in rendered


@pytest.mark.parametrize(
    "detail,secret",
    [
        ("Bearer sk-secret", "sk-secret"),
        ("bearer eyJhbGciOiJIUzI1NiJ9.payload", "eyJhbGciOiJIUzI1NiJ9"),
        ("Basic dXNlcjpwYXNz", "dXNlcjpwYXNz"),
    ],
)
def test_json_redacts_standalone_authentication_schemes(detail, secret):
    section = eh.Section("Auth", id="auth")
    section.add("probe failed", eh.CheckStatus.FAIL, detail=detail)

    rendered = json.dumps(report_document(eh.Report(sections=[section])))

    assert secret not in rendered
    assert "[REDACTED]" in rendered


def test_json_redacts_quoted_secrets_with_spaces_and_escapes():
    section = eh.Section("Secrets", id="secrets")
    section.add(
        'password="correct horse \\"battery\\" staple"',
        eh.CheckStatus.WARN,
        detail="api_key='single quoted value' next=safe",
    )
    output = io.StringIO()
    render_json(eh.Report(sections=[section]), stream=output)
    rendered = output.getvalue()
    assert "correct horse" not in rendered
    assert "single quoted value" not in rendered
    # Once a sensitive assignment starts, redact the rest of that line: an
    # unquoted secret may itself contain spaces, commas, or semicolons.
    assert "next=safe" not in rendered


@pytest.mark.parametrize(
    "key",
    [
        "access_token",
        "refresh-token",
        "client_secret",
        "OPENAI_API_KEY",
        "AWS_SECRET_ACCESS_KEY",
        "signing_private_key",
        "service_credential",
        "passwd",
        "pwd",
        "ssh_passphrase",
        "cookie",
        "session_id",
        "auth",
        "connection_string",
        "access_token_value",
        "client_secret_value",
        "password_hash",
        "accessTokenValue",
        "AWSSecretAccessKey",
        "apikey",
        "privatekey",
        "connectionstring",
    ],
)
def test_json_redacts_compound_credential_keys(key):
    section = eh.Section("Secrets", id="secrets")
    section.add(f"{key}=top-secret-value", eh.CheckStatus.WARN)
    output = io.StringIO()
    render_json(eh.Report(sections=[section]), stream=output)
    assert "top-secret-value" not in output.getvalue()


@pytest.mark.parametrize(
    "url",
    [
        "https://user:password@example.com/path",
        "http://user:p%40ss@127.0.0.1:8000/v1",
        "https://ghp_secret@github.com/repo",
        "postgresql://user:password@database.internal/app",
        "ssh://deployment-token@server.internal",
        "redis+sentinel://user:password@cache.internal/0",
        "https://user:p@ss@example.com/private",
    ],
)
def test_json_redacts_url_userinfo(url):
    section = eh.Section("Network", id="network")
    section.add(f"endpoint={url}", eh.CheckStatus.WARN)
    output = io.StringIO()
    render_json(eh.Report(sections=[section]), stream=output)
    rendered = output.getvalue()
    assert "user:" not in rendered
    assert "[REDACTED]@" in rendered


@pytest.mark.parametrize("secret", ["abc,def", "abc;def", "abc def"])
def test_json_redacts_complete_unquoted_secret_line(secret):
    section = eh.Section("Secrets", id="secrets")
    section.add(f"password={secret}", eh.CheckStatus.WARN)
    output = io.StringIO()
    render_json(eh.Report(sections=[section]), stream=output)
    assert secret not in output.getvalue()


def test_json_finds_sensitive_assignment_after_safe_assignment():
    section = eh.Section("Secrets", id="secrets")
    section.add("context=diagnostic password=hunter2", eh.CheckStatus.WARN)
    output = io.StringIO()
    render_json(eh.Report(sections=[section]), stream=output)
    rendered = output.getvalue()
    assert "context=diagnostic" in rendered
    assert "hunter2" not in rendered


@pytest.mark.parametrize(
    "detail,secret",
    [
        ('{"password": "hunter2"}', "hunter2"),
        ("{'api_key': 'top-secret'}", "top-secret"),
    ],
)
def test_json_redacts_quoted_mapping_keys(detail, secret):
    section = eh.Section("Secrets", id="secrets")
    section.add("mapping", eh.CheckStatus.WARN, detail=detail)

    rendered = json.dumps(report_document(eh.Report(sections=[section])))

    assert secret not in rendered
    assert "[REDACTED]" in rendered


def test_json_redacts_multiline_secret_continuations():
    section = eh.Section("Auth")
    section.add(
        "credentials inspected",
        eh.CheckStatus.WARN,
        detail='mode=local\npassword="first line\nsecret-second"\npublic=visible',
    )

    rendered = json.dumps(report_document(eh.Report(sections=[section])))

    assert "first line" not in rendered
    assert "secret-second" not in rendered
    assert "public=visible" in rendered
    assert "mode=local" in rendered
    assert "[REDACTED]" in rendered


def test_json_redacts_authorization_through_end_of_line():
    section = eh.Section("Auth", id="auth")
    section.add("Authorization: Bearer abc,def;ghi", eh.CheckStatus.WARN)
    output = io.StringIO()
    render_json(eh.Report(sections=[section]), stream=output)
    rendered = output.getvalue()
    assert "abc" not in rendered
    assert "def" not in rendered
    assert "ghi" not in rendered


def test_summary_is_one_line():
    output = io.StringIO()
    render_summary(_report(), stream=output)
    assert output.getvalue().count("\n") == 1
    assert "warn" in output.getvalue()


def test_legacy_human_summary_omits_zero_skipped():
    from vllm_mlx.doctor.cli import render

    output = io.StringIO()
    render(_report(), stream=output)
    assert "skipped" not in output.getvalue()


def test_overall_status_covers_ok_and_fail():
    ok_section = eh.Section("OK")
    ok_section.add("healthy", eh.CheckStatus.OK)
    assert eh.Report(sections=[ok_section]).overall_status == "ok"

    failed_section = eh.Section("Failed")
    failed_section.add("broken", eh.CheckStatus.FAIL)
    assert eh.Report(sections=[failed_section]).overall_status == "fail"

    skipped_section = eh.Section("Skipped")
    skipped_section.add("not run", eh.CheckStatus.SKIPPED)
    assert eh.Report(sections=[skipped_section]).overall_status == "skipped"


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
    assert report.sections[0].checks[0].id is None


def test_empty_selection_does_not_discover_runtime(monkeypatch):
    def must_not_run():
        raise AssertionError("runtime discovery must not run")

    monkeypatch.setattr(eh, "_selected_runtime", must_not_run)
    assert eh.run_all(only={"unknown.section"}).sections == []


def test_explicit_empty_only_set_runs_no_sections(monkeypatch):
    def must_not_run():
        raise AssertionError("an explicit empty selection must run nothing")

    monkeypatch.setattr(eh, "_SECTION_BUILDERS", (must_not_run,))
    assert eh.run_all(only=set()).sections == []


def test_cli_preserves_explicit_empty_only_selection(monkeypatch):
    calls = []

    def capture_run_all(**kwargs):
        calls.append(kwargs)
        return eh.Report()

    monkeypatch.setattr("vllm_mlx.doctor.cli.run_all", capture_run_all)
    args = Namespace(
        tier=None,
        verbose=False,
        json=False,
        summary=True,
        only=[],
        skip=None,
    )

    with pytest.raises(SystemExit):
        doctor_command(args)

    assert calls == [{"only": set(), "skip": None}]


def test_redaction_respects_home_boundaries_and_benign_tokenizer_keys(monkeypatch):
    monkeypatch.setattr(
        "vllm_mlx.doctor.cli.os.path.expanduser", lambda _: "/Users/tester"
    )
    section = eh.Section("Paths", id="paths")
    section.add(
        "paths",
        eh.CheckStatus.OK,
        detail=(
            "home=/Users/tester/cache bracket=[/Users/tester] "
            "object={/Users/tester} other=/Users/tester2 tokenizer=Qwen2"
        ),
    )

    document = report_document(eh.Report(sections=[section]))
    detail = document["sections"][0]["checks"][0]["detail"]

    assert "home=~/cache" in detail
    assert "bracket=[~]" in detail
    assert "object={~}" in detail
    assert "/Users/tester2" in detail
    assert "tokenizer=Qwen2" in detail


@pytest.mark.parametrize(
    "detail",
    [
        "Authorization Bearer abc123",
        "Authorization Basic dXNlcjpwYXNz",
        "token abc",
        "token abc123",
        "token abcdefghijklmnop",
        "api_key abc123",
        "api_key abcdefgh",
        "client_secret abc123",
        "AWS_SECRET_ACCESS_KEY abc123",
        "ssh_passphrase abc123",
        "cookie abc123",
        "session_id abc123",
        "auth abc123",
        "connection_string abc123",
    ],
)
def test_json_redacts_whitespace_delimited_credentials(detail):
    section = eh.Section("Auth", id="auth")
    section.add("credential", eh.CheckStatus.WARN, detail=detail)

    rendered = json.dumps(report_document(eh.Report(sections=[section])))

    assert "abc123" not in rendered
    assert "dXNlcjpwYXNz" not in rendered
    assert "[REDACTED]" in rendered


def test_system_only_does_not_discover_runtime(monkeypatch):
    def must_not_run():
        raise AssertionError("unselected runtime checks must not run")

    monkeypatch.setattr(eh, "_selected_runtime", must_not_run)
    report = eh.run_all(only={"system"})
    assert [section.id for section in report.sections] == ["system"]


def test_budget_exhaustion_is_skipped_not_warning(monkeypatch):
    monkeypatch.setattr(eh.time, "monotonic", lambda: 100.0)
    report = eh._run_all_serialized(99.0)
    assert report.n_warn == 0
    assert report.n_skipped == len(report.sections)
    assert all(
        section.checks[0].status is eh.CheckStatus.SKIPPED
        for section in report.sections
    )


def test_expired_budget_preserves_only_and_skip_selection(monkeypatch):
    monkeypatch.setattr(eh.time, "monotonic", lambda: 100.0)
    report = eh._run_all_serialized(99.0, only={"network", "python"}, skip={"python"})
    assert [section.id for section in report.sections] == ["network"]


def test_lock_timeout_preserves_only_selection(monkeypatch):
    class BusyLock:
        def acquire(self, **_kwargs):
            return False

    monkeypatch.setattr(eh, "_DOCTOR_RUN_LOCK", BusyLock())
    report = eh.run_all(only={"network"})
    assert [section.id for section in report.sections] == ["network"]


def test_doctor_json_does_not_mix_human_output(monkeypatch, capsys):
    monkeypatch.setattr(
        "vllm_mlx.doctor.cli._collect_report_isolated", lambda **_: _report()
    )
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
    stdout = capsys.readouterr().out
    parsed = json.loads(stdout)
    assert parsed["sections"][0]["id"] == "network"


def test_doctor_json_collector_failure_is_schema_valid(monkeypatch, capsys):
    def fail_collection(*_args, **_kwargs):
        raise OSError("collector unavailable")

    monkeypatch.setattr("vllm_mlx.doctor.cli._collect_report_isolated", fail_collection)
    args = Namespace(
        tier=None,
        verbose=False,
        json=True,
        summary=False,
        only=None,
        skip=None,
    )

    with pytest.raises(SystemExit) as exc:
        doctor_command(args)

    assert exc.value.code == 1
    document = json.loads(capsys.readouterr().out)
    assert document["status"] == "fail"
    assert document["sections"][0]["checks"][0]["id"] == "doctor.json.collection"


@pytest.mark.parametrize(
    "document",
    [None, {}, {"sections": "invalid"}, {"sections": [{}]}],
)
def test_isolated_report_payload_is_strictly_validated(document):
    with pytest.raises(RuntimeError, match="invalid"):
        _report_from_document(document)


def test_isolated_report_rejects_schema_and_field_contract_violations():
    valid = report_document(_report())
    invalid_documents = []

    wrong_schema = copy.deepcopy(valid)
    wrong_schema["schemaVersion"] = 2
    invalid_documents.append(wrong_schema)

    float_schema = copy.deepcopy(valid)
    float_schema["schemaVersion"] = 1.0
    invalid_documents.append(float_schema)

    float_exit_code = copy.deepcopy(valid)
    float_exit_code["exitCode"] = 0.0
    invalid_documents.append(float_exit_code)

    extra_root = copy.deepcopy(valid)
    extra_root["unexpected"] = True
    invalid_documents.append(extra_root)

    negative_duration = copy.deepcopy(valid)
    negative_duration["durationMs"] = -1
    invalid_documents.append(negative_duration)

    invalid_version = copy.deepcopy(valid)
    invalid_version["rapidMlxVersion"] = 1
    invalid_documents.append(invalid_version)

    invalid_status = copy.deepcopy(valid)
    invalid_status["status"] = "unknown"
    invalid_documents.append(invalid_status)

    invalid_exit_code = copy.deepcopy(valid)
    invalid_exit_code["exitCode"] = True
    invalid_documents.append(invalid_exit_code)

    invalid_summary = copy.deepcopy(valid)
    invalid_summary["summary"] = []
    invalid_documents.append(invalid_summary)

    invalid_summary_count = copy.deepcopy(valid)
    invalid_summary_count["summary"]["warnings"] = True
    invalid_documents.append(invalid_summary_count)

    invalid_sections = copy.deepcopy(valid)
    invalid_sections["sections"] = {}
    invalid_documents.append(invalid_sections)

    invalid_section = copy.deepcopy(valid)
    invalid_section["sections"][0]["extra"] = True
    invalid_documents.append(invalid_section)

    invalid_section_id = copy.deepcopy(valid)
    invalid_section_id["sections"][0]["id"] = ""
    invalid_documents.append(invalid_section_id)

    invalid_section_title = copy.deepcopy(valid)
    invalid_section_title["sections"][0]["title"] = 1
    invalid_documents.append(invalid_section_title)

    invalid_section_checks = copy.deepcopy(valid)
    invalid_section_checks["sections"][0]["checks"] = {}
    invalid_documents.append(invalid_section_checks)

    invalid_check = copy.deepcopy(valid)
    invalid_check["sections"][0]["checks"][0]["extra"] = True
    invalid_documents.append(invalid_check)

    invalid_check_id = copy.deepcopy(valid)
    invalid_check_id["sections"][0]["checks"][0]["id"] = 1
    invalid_documents.append(invalid_check_id)

    invalid_check_text = copy.deepcopy(valid)
    invalid_check_text["sections"][0]["checks"][0]["detail"] = None
    invalid_documents.append(invalid_check_text)

    mismatched_summary = copy.deepcopy(valid)
    mismatched_summary["summary"]["warnings"] = 0
    invalid_documents.append(mismatched_summary)

    mismatched_status = copy.deepcopy(valid)
    mismatched_status["status"] = "ok"
    invalid_documents.append(mismatched_status)

    mismatched_exit_code = copy.deepcopy(valid)
    mismatched_exit_code["exitCode"] = 1
    invalid_documents.append(mismatched_exit_code)

    missing_title = copy.deepcopy(valid)
    del missing_title["sections"][0]["title"]
    invalid_documents.append(missing_title)

    for document in invalid_documents:
        with pytest.raises(RuntimeError, match="invalid|schemaVersion"):
            _report_from_document(document)


def test_mixed_ok_and_skipped_report_is_partial_warning():
    section = eh.Section("Mixed", id="mixed")
    section.add("healthy", eh.CheckStatus.OK)
    section.add("not run", eh.CheckStatus.SKIPPED)
    report = eh.Report(sections=[section])

    assert report.overall_status == "warn"
    assert report.exit_code == 0


def test_empty_report_is_skipped():
    report = eh.run_all(only={"network"}, skip={"network"})

    assert report.sections == []
    assert report.overall_status == "skipped"
    assert report.exit_code == 0


def test_ambiguous_token_text_prefers_redaction_safety():
    section = eh.Section("Usage", id="usage")
    section.add("metrics", eh.CheckStatus.OK, detail="token count 42")

    detail = report_document(eh.Report(sections=[section]))["sections"][0]["checks"][0][
        "detail"
    ]

    assert detail == "token [REDACTED]"


def test_json_process_isolation_collects_in_fresh_executable(monkeypatch):
    child_pids = []
    popen_options = []
    real_popen = subprocess.Popen

    def tracked_popen(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        child_pids.append(process.pid)
        popen_options.append(kwargs)
        return process

    monkeypatch.setattr(doctor_cli.subprocess, "Popen", tracked_popen)
    report = _collect_report_isolated(only={"system"}, skip=None)

    assert child_pids and child_pids[0] != os.getpid()
    assert len(popen_options) == 1
    assert popen_options[0]["stdin"] == subprocess.DEVNULL
    assert popen_options[0]["stdout"] == subprocess.DEVNULL
    assert popen_options[0]["stderr"] == subprocess.DEVNULL
    assert popen_options[0]["pass_fds"]
    assert popen_options[0]["start_new_session"] is True
    assert [section.id for section in report.sections] == ["system"]


def _run_json_worker(tmp_path, monkeypatch, request, *, result_parent=True):
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps({"deep": False, **request}))
    result_root = tmp_path if result_parent else tmp_path / "missing"
    result_path = result_root / "report.json"
    keepalive_read, keepalive_write = os.pipe()
    status_read, status_write = os.pipe()
    os.close(keepalive_write)
    monkeypatch.setattr(json_worker, "run_all", lambda **_kwargs: _report())
    try:
        exit_code = json_worker.main(
            [
                str(request_path),
                str(result_path),
                str(keepalive_read),
                str(status_write),
            ]
        )
    finally:
        os.close(status_read)
    return exit_code, result_path


def test_json_worker_publishes_atomic_report(tmp_path, monkeypatch):
    exit_code, result_path = _run_json_worker(
        tmp_path,
        monkeypatch,
        {"only": ["network"], "skip": None},
    )

    assert exit_code == 0
    message = json.loads(result_path.read_text())
    assert message["ok"] is True
    assert message["report"]["sections"][0]["id"] == "network"
    assert not result_path.with_suffix(".json.tmp").exists()


def test_json_worker_serializes_invalid_request_as_error(tmp_path, monkeypatch):
    exit_code, result_path = _run_json_worker(
        tmp_path,
        monkeypatch,
        {"only": "network", "skip": None},
    )

    assert exit_code == 0
    message = json.loads(result_path.read_text())
    assert message["ok"] is False
    assert "TypeError" in message["error"]


def test_json_worker_serializes_invalid_request_fields_as_error(tmp_path, monkeypatch):
    exit_code, result_path = _run_json_worker(
        tmp_path,
        monkeypatch,
        {"only": None, "skip": None, "unexpected": True},
    )

    assert exit_code == 0
    message = json.loads(result_path.read_text())
    assert message["ok"] is False
    assert "request fields are invalid" in message["error"]


def test_json_worker_rejects_non_boolean_deep_flag(tmp_path, monkeypatch):
    exit_code, result_path = _run_json_worker(
        tmp_path,
        monkeypatch,
        {"only": None, "skip": None, "deep": "yes"},
    )

    assert exit_code == 0
    message = json.loads(result_path.read_text())
    assert message["ok"] is False
    assert "deep flag is invalid" in message["error"]


def test_json_worker_argument_and_write_failures(tmp_path, monkeypatch):
    assert json_worker.main([]) == 2
    assert json_worker.main(["request", "result", "not-an-fd", "also-bad"]) == 2

    exit_code, result_path = _run_json_worker(
        tmp_path,
        monkeypatch,
        {"only": None, "skip": []},
        result_parent=False,
    )
    assert exit_code == 1
    assert not result_path.exists()

    # Invalid lifecycle descriptors also exercise the best-effort close path.
    request_path = tmp_path / "request-invalid-fds.json"
    request_path.write_text(json.dumps({"only": None, "skip": None}))
    assert (
        json_worker.main([str(request_path), str(result_path), "999999", "999998"]) == 1
    )


def test_json_worker_tolerates_keepalive_read_error(tmp_path, monkeypatch):
    monkeypatch.setattr(
        json_worker.os, "read", lambda *_args: (_ for _ in ()).throw(OSError("closed"))
    )
    exit_code, result_path = _run_json_worker(
        tmp_path,
        monkeypatch,
        {"only": None, "skip": None},
    )

    assert exit_code == 0
    assert result_path.exists()


class _CollectorProcess:
    pid = 424242

    def __init__(self, *, waits=None, kill_error=None):
        self.waits = list(waits or [None])
        self.kill_error = kill_error
        self.killed = False

    def wait(self, *, timeout):
        outcome = self.waits.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def kill(self):
        self.killed = True
        if self.kill_error is not None:
            raise self.kill_error


def _install_collector_result(monkeypatch, message, *, process=None):
    process = process or _CollectorProcess()

    def fake_popen(args, **_kwargs):
        Path(args[-3]).write_text(message)
        return process

    monkeypatch.setattr(doctor_cli.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(doctor_cli, "_terminate_collector_group", lambda _pid: None)
    return process


@pytest.mark.parametrize(
    "message,match",
    [
        ("not-json", "result is unreadable"),
        (json.dumps([]), "invalid message"),
        (json.dumps({"ok": "yes"}), "invalid message"),
        (json.dumps({"ok": False, "error": "boom"}), "collection failed: boom"),
    ],
)
def test_json_collector_rejects_invalid_child_messages(monkeypatch, message, match):
    _install_collector_result(monkeypatch, message)

    with pytest.raises(RuntimeError, match=match):
        _collect_report_isolated(only=None, skip=None)


def test_json_collector_closes_pipes_when_spawn_fails(monkeypatch):
    monkeypatch.setattr(
        doctor_cli.subprocess,
        "Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("spawn failed")),
    )

    with pytest.raises(OSError, match="spawn failed"):
        _collect_report_isolated(only=None, skip=None)


def test_json_collector_reports_process_group_cleanup_failure(monkeypatch):
    _install_collector_result(monkeypatch, json.dumps({"ok": False}))
    monkeypatch.setattr(
        doctor_cli,
        "_terminate_collector_group",
        lambda _pid: (_ for _ in ()).throw(OSError("killpg denied")),
    )

    with pytest.raises(RuntimeError, match="group cleanup failed"):
        _collect_report_isolated(only=None, skip=None)


def test_json_collector_falls_back_to_direct_kill(monkeypatch):
    process = _CollectorProcess(
        waits=[subprocess.TimeoutExpired("collector", 0.5), None]
    )
    _install_collector_result(monkeypatch, json.dumps({"ok": False}), process=process)

    with pytest.raises(RuntimeError, match="collection failed"):
        _collect_report_isolated(only=None, skip=None)

    assert process.killed is True


def test_json_collector_tolerates_already_exited_direct_kill(monkeypatch):
    process = _CollectorProcess(
        waits=[subprocess.TimeoutExpired("collector", 0.5), None],
        kill_error=ProcessLookupError(),
    )
    _install_collector_result(monkeypatch, json.dumps({"ok": False}), process=process)

    with pytest.raises(RuntimeError, match="collection failed"):
        _collect_report_isolated(only=None, skip=None)


def test_json_collector_reports_direct_kill_failure(monkeypatch):
    process = _CollectorProcess(
        waits=[subprocess.TimeoutExpired("collector", 0.5), None],
        kill_error=OSError("kill denied"),
    )
    _install_collector_result(monkeypatch, json.dumps({"ok": False}), process=process)

    with pytest.raises(RuntimeError, match="could not be killed"):
        _collect_report_isolated(only=None, skip=None)


def test_json_collector_reports_unreaped_process(monkeypatch):
    process = _CollectorProcess(
        waits=[
            subprocess.TimeoutExpired("collector", 0.5),
            subprocess.TimeoutExpired("collector", 0.5),
        ]
    )
    _install_collector_result(monkeypatch, json.dumps({"ok": False}), process=process)

    with pytest.raises(RuntimeError, match="could not be reaped"):
        _collect_report_isolated(only=None, skip=None)


def test_terminate_collector_group_ignores_missing_process(monkeypatch):
    monkeypatch.setattr(
        doctor_cli.os,
        "killpg",
        lambda *_args: (_ for _ in ()).throw(ProcessLookupError()),
    )

    doctor_cli._terminate_collector_group(123)


def test_json_timeout_terminates_collector_leader_and_descendant(monkeypatch, tmp_path):
    descendant_path = tmp_path / "descendant.pid"
    processes = []
    real_popen = subprocess.Popen
    script = (
        "import pathlib,subprocess,sys,time; "
        "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']); "
        "pathlib.Path(sys.argv[1]).write_text(str(child.pid)); time.sleep(60)"
    )

    def blocking_popen(_args, **kwargs):
        process = real_popen(
            [sys.executable, "-c", script, str(descendant_path)], **kwargs
        )
        processes.append(process)
        ready_deadline = time.monotonic() + 1.0
        while not descendant_path.exists() and time.monotonic() < ready_deadline:
            time.sleep(0.01)
        assert descendant_path.exists()
        return process

    monkeypatch.setattr(doctor_cli.subprocess, "Popen", blocking_popen)

    with pytest.raises(RuntimeError, match="timed out"):
        _collect_report_isolated(only={"system"}, skip=None, timeout_s=0.3)

    assert processes[0].returncode == -signal.SIGKILL
    descendant_pid = int(descendant_path.read_text())
    ps = real_popen(
        ["/bin/ps", "-o", "stat=", "-p", str(descendant_pid)],
        stdout=subprocess.PIPE,
        text=True,
    )
    descendant_state = ps.communicate(timeout=1.0)[0].strip()
    assert ps.returncode in {0, 1}
    if descendant_state and not descendant_state.startswith("Z"):
        os.kill(descendant_pid, signal.SIGKILL)
    assert not descendant_state or descendant_state.startswith("Z")


def test_json_process_isolation_detects_child_exit_without_waiting_for_timeout(
    monkeypatch,
):
    monkeypatch.setattr(doctor_cli.sys, "executable", "/usr/bin/false")
    started_at = time.monotonic()

    with pytest.raises(RuntimeError, match="exited without a report"):
        _collect_report_isolated(only={"system"}, skip=None, timeout_s=2.0)

    assert time.monotonic() - started_at < 1.0


def test_json_collector_pipe_setup_closes_partial_allocation(monkeypatch):
    real_pipe = os.pipe
    opened_fds = []
    calls = 0

    def fail_second_pipe():
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("descriptor limit")
        opened_fds.extend(real_pipe())
        return tuple(opened_fds)

    monkeypatch.setattr(doctor_cli.os, "pipe", fail_second_pipe)

    with pytest.raises(OSError, match="descriptor limit"):
        _open_collector_pipes()

    for fd in opened_fds:
        with pytest.raises(OSError):
            os.fstat(fd)


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


def test_human_summary_includes_skipped_count():
    section = eh.Section("Optional", id="optional")
    section.add("not selected", eh.CheckStatus.SKIPPED)

    output = io.StringIO()
    render(eh.Report(sections=[section]), stream=output)

    assert "1 skipped" in output.getvalue()


def test_output_modes_are_mutually_exclusive():
    from vllm_mlx.cli import build_parser

    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["doctor", "--json", "--verbose"])
    assert exc.value.code == 2
