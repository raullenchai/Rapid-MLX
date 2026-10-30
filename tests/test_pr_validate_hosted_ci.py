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
        "html_url": "https://example.test/check",
    }


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
