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
    runner._terminate_child_processes(grace_seconds=0.25)
    process.children.assert_called_once_with(recursive=True)
    child.terminate.assert_called_once()
    survivor.terminate.assert_called_once()
    wait.assert_called_once_with([child, survivor], timeout=0.25)
    survivor.kill.assert_called_once()
    child.kill.assert_not_called()
