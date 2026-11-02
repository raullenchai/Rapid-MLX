# SPDX-License-Identifier: Apache-2.0
"""Concurrency, cancellation, and ordering of validation lanes."""

import subprocess
import sys
import threading
import time

import pytest

from scripts.pr_validate import runner
from scripts.pr_validate.base import Step, StepResult
from scripts.pr_validate.context import Context

_REAL_TERMINATE = runner._terminate_child_processes


@pytest.fixture(autouse=True)
def prevent_child_process_termination(monkeypatch):
    monkeypatch.setattr(runner, "_terminate_child_processes", lambda: None)


@pytest.fixture
def repo_root_cwd(monkeypatch, tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'fake'\n")
    monkeypatch.chdir(tmp_path)
    ctx = Context(pr_number=1, work_dir=tmp_path / "artifacts")
    monkeypatch.setattr(runner, "Context", lambda **kwargs: ctx)
    monkeypatch.delenv("PR_VALIDATE_CONTINUE_AFTER_CODEX_FAIL", raising=False)
    return ctx


class FakeStep(Step):
    def __init__(self, name, action):
        self.name = name
        self.description = "fake"
        self.action = action

    def run(self, ctx):
        return self.action(self.name)


def result(name, status="pass", findings=()):
    return StepResult(name, status, status, findings=list(findings))


def test_lanes_overlap(repo_root_cwd):
    barrier = threading.Barrier(2, timeout=5)

    def action(name):
        barrier.wait()
        return result(name)

    assert (
        runner.run_pipeline(
            1, steps=[FakeStep("codex_review", action), FakeStep("lint", action)]
        )
        == 0
    )


def test_failing_lane_records_remaining_steps_as_cancelled(repo_root_cwd):
    ran = []

    def action(name):
        ran.append(name)
        return result(name, "fail" if name == "lint" else "pass")

    names = ("lint", "targeted_tests", "mac_only_tests")
    assert runner.run_pipeline(1, steps=[FakeStep(n, action) for n in names]) == 1
    assert ran == ["lint"]
    rows = repo_root_cwd.results
    assert [row.name for row in rows] == list(names)
    assert all(row.status == "skip" and "cancelled" in row.summary for row in rows[1:])


@pytest.mark.parametrize("failing_lane", ["checks", "review"])
def test_cancellation_stops_subprocess_and_remaining_steps(
    repo_root_cwd, monkeypatch, failing_lane
):
    started = threading.Event()
    process = []
    gpu = []

    def sleeper(name):
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        process.append(proc)
        started.set()
        proc.wait(timeout=15)
        return result(name)

    def fail(name):
        assert started.wait(5)
        return result(
            name, "fail", ["[BLOCKING] problem"] if name == "codex_review" else []
        )

    def terminate():
        for proc in process:
            proc.terminate()
            proc.wait(timeout=5)

    monkeypatch.setattr(runner, "_terminate_child_processes", terminate)
    review = FakeStep("codex_review", sleeper if failing_lane == "checks" else fail)
    check = FakeStep(
        "lint" if failing_lane == "checks" else "targeted_tests",
        fail if failing_lane == "checks" else sleeper,
    )
    later = FakeStep("full_unit", lambda name: result(name))
    steps = [
        review,
        check,
        later,
        FakeStep("stress_e2e_bench", lambda name: gpu.append(name) or result(name)),
    ]
    before = time.monotonic()
    assert runner.run_pipeline(1, steps=steps, fail_fast=failing_lane == "checks") == 1
    assert time.monotonic() - before < 15
    results = {r.name: r for r in repo_root_cwd.results}
    cancelled = results[
        "codex_review" if failing_lane == "checks" else "targeted_tests"
    ]
    assert cancelled.status == "skip" and cancelled.summary.startswith("cancelled:")
    if failing_lane == "review":
        assert results["full_unit"].status == "skip"
        assert results["full_unit"].summary.startswith("cancelled:")
    assert not gpu


@pytest.mark.parametrize("fail_fast", [True, False])
def test_review_infrastructure_failure_allows_checks(repo_root_cwd, fail_fast):
    ran = []

    def action(name):
        ran.append(name)
        return result(name, "fail" if name == "codex_review" else "pass")

    steps = [
        FakeStep(name, action)
        for name in ("codex_review", "lint", "targeted_tests", "stress_e2e_bench")
    ]
    assert runner.run_pipeline(1, steps=steps, fail_fast=fail_fast) == 1
    assert "lint" in ran and "targeted_tests" in ran
    assert ("stress_e2e_bench" in ran) is not fail_fast


def test_full_picture_and_order(repo_root_cwd):
    checks_done = threading.Event()

    def action(name):
        if name == "codex_review":
            assert checks_done.wait(5)
        if name == "lint":
            checks_done.set()
        return result(name, "fail" if name == "lint" else "pass")

    names = ["codex_review", "lint", "targeted_tests", "stress_e2e_bench"]
    assert (
        runner.run_pipeline(
            1, steps=[FakeStep(name, action) for name in names], fail_fast=False
        )
        == 1
    )
    assert [r.name for r in repo_root_cwd.results] == names
    assert [r.status for r in repo_root_cwd.results] == ["pass", "fail", "pass", "pass"]


def test_terminate_children_terminates_then_kills_survivors(monkeypatch):
    from unittest.mock import Mock

    child, survivor = Mock(), Mock()
    process = Mock()
    process.children.return_value = [child, survivor]
    monkeypatch.setattr(runner.psutil, "Process", lambda: process)
    wait = Mock(return_value=([child], [survivor]))
    monkeypatch.setattr(runner.psutil, "wait_procs", wait)
    _REAL_TERMINATE(grace_seconds=0.25)
    process.children.assert_called_once_with(recursive=True)
    child.terminate.assert_called_once()
    survivor.terminate.assert_called_once()
    wait.assert_called_once_with([child, survivor], timeout=0.25)
    survivor.kill.assert_called_once()
    child.kill.assert_not_called()


def test_terminate_children_warns_when_enumeration_is_denied(monkeypatch, capsys):
    from unittest.mock import Mock

    process = Mock()
    process.children.side_effect = runner.psutil.AccessDenied(pid=123)
    monkeypatch.setattr(runner.psutil, "Process", lambda: process)
    wait = Mock()
    monkeypatch.setattr(runner.psutil, "wait_procs", wait)

    _REAL_TERMINATE()

    assert "cannot enumerate child processes" in capsys.readouterr().err
    wait.assert_not_called()


def test_terminate_children_ignores_processes_that_disappear(monkeypatch):
    from unittest.mock import Mock

    child, survivor = Mock(), Mock()
    child.terminate.side_effect = runner.psutil.NoSuchProcess(pid=123)
    survivor.kill.side_effect = runner.psutil.NoSuchProcess(pid=456)
    process = Mock()
    process.children.return_value = [child, survivor]
    monkeypatch.setattr(runner.psutil, "Process", lambda: process)
    wait = Mock(return_value=([child], [survivor]))
    monkeypatch.setattr(runner.psutil, "wait_procs", wait)

    _REAL_TERMINATE()

    child.terminate.assert_called_once()
    wait.assert_called_once_with([child, survivor], timeout=5.0)
    survivor.kill.assert_called_once()


def test_blocking_review_cancels_waiting_checks(repo_root_cwd):
    started = threading.Event()

    class WaitingCheck(Step):
        name = "hosted_ci"
        description = "wait for cancellation"

        def run(self, ctx):
            started.set()
            assert ctx.cancel_event.wait(10)
            return result(self.name, "skip")

    def fail(name):
        assert started.wait(5)
        return result(name, "fail", ["[BLOCKING] problem"])

    before = time.monotonic()
    assert (
        runner.run_pipeline(1, steps=[FakeStep("codex_review", fail), WaitingCheck()])
        == 1
    )
    assert time.monotonic() - before < 5


@pytest.mark.parametrize(
    "exception", [RuntimeError("boom"), SystemExit("boom"), KeyboardInterrupt("boom")]
)
def test_lane_execute_crash_blocks_merge(repo_root_cwd, monkeypatch, exception):
    step = FakeStep("lint", lambda name: result(name))

    def crash(ctx):
        raise exception

    monkeypatch.setattr(step, "execute", crash)
    assert runner.run_pipeline(1, steps=[step]) == 1
    row = repo_root_cwd.results[0]
    assert (row.status, row.summary, row.stage) == (
        "error",
        f"lane crashed: {type(exception).__name__}: boom",
        "checks",
    )


@pytest.mark.parametrize("status", ["fail", "error"])
def test_failure_after_cancellation_is_preserved(repo_root_cwd, status):
    started = threading.Event()

    def delayed_failure(name):
        started.set()
        assert repo_root_cwd.cancel_event.wait(5)
        return result(name, status)

    def fail(name):
        assert started.wait(5)
        return result(name, "fail", ["[BLOCKING] problem"])

    assert (
        runner.run_pipeline(
            1, steps=[FakeStep("codex_review", fail), FakeStep("lint", delayed_failure)]
        )
        == 1
    )
    row = next(r for r in repo_root_cwd.results if r.name == "lint")
    assert row.status == status
    assert row.summary == f"after cancellation: {status}"


@pytest.mark.parametrize("exception", [KeyboardInterrupt, SystemExit])
def test_main_interrupt_cancels_lanes(repo_root_cwd, monkeypatch, exception):
    started = threading.Event()
    release = threading.Event()
    ran = []
    terminated = []
    real_join = threading.Thread.join
    interrupted = False
    workers = []

    def blocking(name):
        started.set()
        assert release.wait(5)
        return result(name)

    def join(thread, timeout=None):
        nonlocal interrupted
        if threading.current_thread() is threading.main_thread() and not interrupted:
            interrupted = True
            workers.append(thread)
            assert started.wait(5)
            raise exception()
        return real_join(thread, timeout)

    def terminate():
        terminated.append(True)
        release.set()

    monkeypatch.setattr(threading.Thread, "join", join)
    monkeypatch.setattr(runner, "_terminate_child_processes", terminate)
    try:
        with pytest.raises(exception):
            runner.run_pipeline(
                1,
                steps=[
                    FakeStep("lint", blocking),
                    FakeStep(
                        "targeted_tests", lambda name: ran.append(name) or result(name)
                    ),
                ],
            )
        assert terminated
        assert repo_root_cwd.cancel_event.is_set()
        assert not ran
        assert any("interrupted" in row.summary for row in repo_root_cwd.results)
    finally:
        release.set()
        for worker in workers:
            real_join(worker, 5)


def test_cli_ctrl_c_returns_130(monkeypatch, capsys):
    from scripts.pr_validate import pr_validate

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(pr_validate, "run_pipeline", interrupt)
    try:
        exit_code = pr_validate.main(["1"])
    except KeyboardInterrupt:
        pytest.fail("CLI leaked KeyboardInterrupt instead of returning 130")
    assert exit_code == 130
    assert capsys.readouterr().err.splitlines() == ["Validation interrupted (Ctrl-C)."]
