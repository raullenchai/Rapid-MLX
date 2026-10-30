# SPDX-License-Identifier: Apache-2.0
import threading
from types import SimpleNamespace

import pytest

from scripts.pr_validate.steps import hosted_ci


def check(
    name="tests", status="completed", conclusion="success", app="github-actions", id=1
):
    return {
        "name": name,
        "status": status,
        "conclusion": conclusion,
        "app": {"slug": app},
        "id": id,
        "check_suite": {"id": 1},
        "html_url": "https://example.test/check",
    }


def run(
    workflow_id=1, suite=1, number=1, attempt=1, id=1, path=".github/workflows/ci.yml"
):
    return {
        "workflow_id": workflow_id,
        "check_suite_id": suite,
        "run_number": number,
        "run_attempt": attempt,
        "id": id,
        "path": path,
        "status": "completed",
    }


@pytest.fixture(autouse=True)
def fake_runs(monkeypatch):
    monkeypatch.setattr(
        hosted_ci, "_fetch_runs", lambda ctx: [run(), run(2, 2, path="other")]
    )


@pytest.fixture
def ctx():
    return SimpleNamespace(
        head_sha="a" * 40,
        repo="owner/repo",
        run_log=lambda msg: None,
        cancel_event=None,
    )


def test_green_and_filtering(monkeypatch, ctx):
    monkeypatch.setattr(
        hosted_ci,
        "_fetch_checks",
        lambda ctx: [
            check(),
            check("other", conclusion="skipped"),
            check("neutral", conclusion="neutral"),
            check("external", conclusion="failure", app="other"),
            check("tests", conclusion="failure", id=0),
        ],
    )
    result = hosted_ci.HostedCIStep().run(ctx)
    assert result.status == "pass"
    assert result.summary == f"1 hosted checks green on {'a' * 12}"


def test_failure(monkeypatch, ctx):
    monkeypatch.setattr(
        hosted_ci, "_fetch_checks", lambda ctx: [check(conclusion="failure")]
    )
    result = hosted_ci.HostedCIStep().run(ctx)
    assert result.status == "fail"
    assert "tests" in result.details and "https://example.test/check" in result.details


def test_poll_then_pass(monkeypatch, ctx):
    checks = iter([[check(status="in_progress", conclusion=None)], [check()]])
    monkeypatch.setattr(hosted_ci, "_fetch_checks", lambda ctx: next(checks))
    monkeypatch.setattr(hosted_ci, "_sleep", lambda seconds: None)
    assert hosted_ci.HostedCIStep().run(ctx).status == "pass"


@pytest.mark.parametrize(
    "checks", [[check(status="queued", conclusion=None)], [check("other")], []]
)
def test_timeout(monkeypatch, ctx, checks):
    monkeypatch.setenv("PR_VALIDATE_HOSTED_CI_TIMEOUT_S", "0")
    monkeypatch.setattr(hosted_ci, "_fetch_checks", lambda ctx: checks)
    result = hosted_ci.HostedCIStep().run(ctx)
    assert result.status == "fail"
    assert "tests" in result.details


def test_skip_in_ci_and_opt_out(monkeypatch, ctx):
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    assert not hosted_ci.HostedCIStep().should_run(ctx)
    monkeypatch.delenv("GITHUB_ACTIONS")
    monkeypatch.setenv("PR_VALIDATE_NO_HOSTED_CI", "1")
    assert not hosted_ci.HostedCIStep().should_run(ctx)


def test_bad_sha(ctx):
    ctx.head_sha = "bad"
    assert hosted_ci.HostedCIStep().run(ctx).status == "error"


def test_cancelled_before_fetch(monkeypatch, ctx):
    ctx.cancel_event = threading.Event()
    ctx.cancel_event.set()

    def fetch(ctx):
        pytest.fail("cancelled poll fetched checks")

    monkeypatch.setattr(hosted_ci, "_fetch_checks", fetch)
    result = hosted_ci.HostedCIStep().run(ctx)
    assert result.status == "skip"
    assert result.summary == "cancelled while waiting for hosted CI"


def test_cancelled_during_wait(monkeypatch, ctx):
    class CancelDuringWait:
        def is_set(self):
            return False

        def wait(self, seconds):
            return True

    ctx.cancel_event = CancelDuringWait()
    monkeypatch.setattr(
        hosted_ci, "_fetch_checks", lambda ctx: [check(status="queued")]
    )
    result = hosted_ci.HostedCIStep().run(ctx)
    assert result.status == "skip"
    assert result.summary == "cancelled while waiting for hosted CI"


def test_same_name_in_distinct_suites_failure(monkeypatch, ctx):
    failed = check("changes", conclusion="failure", id=1)
    passed = check("changes", id=2)
    failed["check_suite"] = {"id": 1}
    passed["check_suite"] = {"id": 2}
    monkeypatch.setattr(
        hosted_ci, "_fetch_checks", lambda ctx: [check(), failed, passed]
    )
    assert hosted_ci.HostedCIStep().run(ctx).status == "fail"


def test_superseded_failure_is_ignored(monkeypatch, ctx):
    monkeypatch.setattr(
        hosted_ci,
        "_fetch_runs",
        lambda ctx: [
            run(),
            run(
                suite=2, number=2, id=2, path=".github/workflows/ci.yml@refs/heads/main"
            ),
        ],
    )
    failed = check(conclusion="failure")
    passed = check(id=2)
    passed["check_suite"] = {"id": 2}
    monkeypatch.setattr(hosted_ci, "_fetch_checks", lambda ctx: [failed, passed])
    assert hosted_ci.HostedCIStep().run(ctx).status == "pass"


def test_tests_from_another_workflow_do_not_satisfy_gate(monkeypatch, ctx):
    monkeypatch.setenv("PR_VALIDATE_HOSTED_CI_TIMEOUT_S", "0")
    other = check()
    other["check_suite"] = {"id": 2}
    monkeypatch.setattr(hosted_ci, "_fetch_checks", lambda ctx: [other])
    result = hosted_ci.HostedCIStep().run(ctx)
    assert result.status == "fail"
    assert "tests" in result.details


def test_green_checks_wait_for_workflow(monkeypatch, ctx):
    monkeypatch.setenv("PR_VALIDATE_HOSTED_CI_TIMEOUT_S", "0")
    pending = run()
    pending.update(status="in_progress", name="CI")
    monkeypatch.setattr(hosted_ci, "_fetch_runs", lambda ctx: [pending])
    monkeypatch.setattr(hosted_ci, "_fetch_checks", lambda ctx: [check()])
    result = hosted_ci.HostedCIStep().run(ctx)
    assert result.status == "fail"
    assert "workflow CI" in result.details


def test_pull_request_failure_not_superseded_by_newer_push(monkeypatch, ctx):
    pr = run()
    pr.update(event="pull_request")
    push = run(suite=2, number=2, id=2)
    push.update(event="push")
    failed = check(conclusion="failure")
    passed = check(id=2)
    passed["check_suite"] = {"id": 2}
    monkeypatch.setattr(hosted_ci, "_fetch_runs", lambda ctx: [pr, push])
    monkeypatch.setattr(hosted_ci, "_fetch_checks", lambda ctx: [failed, passed])
    assert hosted_ci.HostedCIStep().run(ctx).status == "fail"


def test_push_tests_cannot_satisfy_pull_request_gate(monkeypatch, ctx):
    monkeypatch.setenv("PR_VALIDATE_HOSTED_CI_TIMEOUT_S", "0")
    pr = run()
    pr.update(event="pull_request")
    push = run(suite=2, number=2, id=2)
    push.update(event="push")
    passed = check(id=2)
    passed["check_suite"] = {"id": 2}
    monkeypatch.setattr(hosted_ci, "_fetch_runs", lambda ctx: [pr, push])
    monkeypatch.setattr(hosted_ci, "_fetch_checks", lambda ctx: [passed])
    assert "tests" in hosted_ci.HostedCIStep().run(ctx).details


def test_failed_workflow_without_checks_fails_immediately(monkeypatch, ctx):
    failed = run()
    failed.update(
        name="CI", conclusion="startup_failure", html_url="https://example.test/run"
    )
    monkeypatch.setattr(hosted_ci, "_fetch_runs", lambda ctx: [failed])
    monkeypatch.setattr(
        hosted_ci, "_fetch_checks", lambda ctx: pytest.fail("fetched checks")
    )
    result = hosted_ci.HostedCIStep().run(ctx)
    assert result.status == "fail"
    assert "CI" in result.details and "https://example.test/run" in result.details
