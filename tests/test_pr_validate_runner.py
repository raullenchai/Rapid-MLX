# SPDX-License-Identifier: Apache-2.0
"""Tests for ``scripts.pr_validate.runner.run_pipeline``.

We want to lock in two contracts:

1. Default (``fail_fast=False``) runs every step even after one fails,
   so the scorecard surfaces the FULL picture for a maintainer review.
2. Opt-in ``fail_fast=True`` stops at the first ``fail`` / ``error``
   AFTER the always-on fetch fail-fast — so CI doesn't waste compute on
   stress/bench when an earlier cheap check already blocked the PR.

Both contracts are exercised against fake in-memory Steps via the
``steps=`` injection seam — the production STEPS list pulls real PR
data over the network and is not unit-testable here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.pr_validate.base import Step, StepResult
from scripts.pr_validate.runner import run_pipeline
from scripts.pr_validate.steps.fetch import FetchStep


class _FakeFetch(Step):
    """Stand-in for ``FetchStep`` — populates the bare minimum of
    Context fields the rest of the pipeline reads, then returns pass."""

    name = "fetch"
    description = "fake fetch"

    def run(self, ctx):  # type: ignore[no-untyped-def]
        # Other steps may read these; populate them harmlessly.
        ctx.pr_title = "test"
        ctx.pr_author = "tester"
        ctx.head_sha = "deadbeef"
        ctx.diff_path = ""
        ctx.files_changed = []
        return StepResult(name=self.name, status="pass", summary="ok")


class _FakeStep(Step):
    """A configurable step that returns a preset status."""

    def __init__(self, name: str, status: str = "pass"):
        self.name = name
        self.description = f"fake {name}"
        self._status = status

    def run(self, ctx):  # type: ignore[no-untyped-def]
        return StepResult(
            name=self.name, status=self._status, summary=f"{self._status}"
        )


@pytest.fixture
def repo_root_cwd(monkeypatch, tmp_path):
    """Context's ``__post_init__`` insists on running from a dir with a
    pyproject.toml. Build a fake one so the test doesn't have to live in
    the real repo root."""
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'fake'\n")
    monkeypatch.chdir(tmp_path)
    # Each test gets its own work_dir under tmp_path so artifacts don't
    # collide across runs.
    monkeypatch.setattr(
        "scripts.pr_validate.context.Path",
        Path,
        raising=False,
    )
    return tmp_path


def _fake_pipeline(after_fetch: list[tuple[str, str]]) -> list[Step]:
    """Build a [fake-fetch, ...named steps with a status...] pipeline."""
    return [_FakeFetch(), *(_FakeStep(name, status) for name, status in after_fetch)]


class TestFailFast:
    def test_default_runs_all_steps_after_a_fail(self, repo_root_cwd, capsys):
        """Without fail_fast, every step after fetch runs even when one
        fails — the scorecard is supposed to show ALL the issues at once."""
        steps = _fake_pipeline(
            [
                ("step_a", "pass"),
                ("step_b", "fail"),  # blocking, but fail_fast=False
                ("step_c", "pass"),
            ]
        )
        rc = run_pipeline(pr_number=999, fail_fast=False, steps=steps)
        captured = capsys.readouterr()
        # Exit code is non-zero because step_b failed.
        assert rc == 1
        # All four step headers must appear in stderr — none was skipped.
        for name in ("fetch", "step_a", "step_b", "step_c"):
            assert f"## [{name}]" in captured.err, f"missing step {name!r}"
        # Scorecard goes to stdout.
        assert "step_c" in captured.out

    def test_fail_fast_stops_at_first_fail_after_fetch(self, repo_root_cwd, capsys):
        """With fail_fast=True, step_c never runs once step_b fails."""
        steps = _fake_pipeline(
            [
                ("step_a", "pass"),
                ("step_b", "fail"),  # should stop here
                ("step_c", "pass"),  # never reached
            ]
        )
        rc = run_pipeline(pr_number=999, fail_fast=True, steps=steps)
        captured = capsys.readouterr()
        assert rc == 1
        # fetch + step_a + step_b ran; step_c did not.
        assert "## [fetch]" in captured.err
        assert "## [step_a]" in captured.err
        assert "## [step_b]" in captured.err
        assert "## [step_c]" not in captured.err
        # The fail-fast stop message must include the step name and the
        # 'subsequent steps not run' phrasing so the operator isn't
        # surprised by a short scorecard.
        assert "fail-fast: [step_b]" in captured.err
        assert "subsequent steps not run" in captured.err

    def test_fail_fast_stops_on_error_too(self, repo_root_cwd, capsys):
        """error status (step crash) should also short-circuit fail_fast."""
        steps = _fake_pipeline(
            [
                ("step_a", "pass"),
                ("step_b", "error"),  # crash counts as blocking
                ("step_c", "pass"),
            ]
        )
        rc = run_pipeline(pr_number=999, fail_fast=True, steps=steps)
        captured = capsys.readouterr()
        assert rc == 1
        assert "## [step_c]" not in captured.err
        assert "fail-fast: [step_b]" in captured.err

    def test_fail_fast_does_not_stop_on_skip(self, repo_root_cwd, capsys):
        """skip is neutral — fail-fast must NOT trigger on it, otherwise
        a high-blast gate skipping on a low-blast PR would look like a
        failure."""
        steps = _fake_pipeline(
            [
                ("step_a", "skip"),  # neutral, must continue
                ("step_b", "pass"),
                ("step_c", "pass"),
            ]
        )
        rc = run_pipeline(pr_number=999, fail_fast=True, steps=steps)
        captured = capsys.readouterr()
        assert rc == 0
        for name in ("step_a", "step_b", "step_c"):
            assert f"## [{name}]" in captured.err

    def test_fetch_failure_always_stops_regardless_of_flag(self, repo_root_cwd, capsys):
        """The pre-existing FAIL_FAST_STEPS={'fetch'} contract still
        holds even with fail_fast=False — without a successful fetch
        nothing else has anything to validate against."""

        class _BadFetch(Step):
            name = "fetch"
            description = "fake bad fetch"

            def run(self, ctx):  # type: ignore[no-untyped-def]
                return StepResult(name=self.name, status="fail", summary="bad")

        steps = [_BadFetch(), _FakeStep("step_a", "pass")]
        rc = run_pipeline(pr_number=999, fail_fast=False, steps=steps)
        captured = capsys.readouterr()
        assert rc == 1
        assert "## [fetch]" in captured.err
        assert "## [step_a]" not in captured.err
        # The "is critical" message is the hard-coded fetch fail-fast,
        # not the user-toggled one — make sure the right path fired.
        assert "is critical" in captured.err

    def test_real_fetch_step_in_global_pipeline(self):
        """Sanity: the production pipeline still has FetchStep first.
        Catches accidental reorderings in runner.py's STEPS list."""
        from scripts.pr_validate.runner import STEPS

        assert isinstance(STEPS[0], FetchStep)
