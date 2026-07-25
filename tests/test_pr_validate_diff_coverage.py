# SPDX-License-Identifier: Apache-2.0
"""Tests for the ADVISORY ``diff_coverage`` pr_validate step.

Two contracts matter most here and are the reason this file exists:

1. **It can never block a merge.** ``diff_coverage`` is measure-only.
   Every failure mode — missing tooling, no coverage.xml, an internal
   crash, diff-cover finding nothing — must resolve to ``pass`` or
   ``skip``, never ``fail`` / ``error``. If this contract ever breaks,
   an advisory measurer would start gating merges, which is exactly
   what the dev-flow proposal set out to avoid.

2. **It only runs when there is production code to measure.** Docs-only,
   tests-only, and deps-only PRs have no ``vllm_mlx`` lines for
   diff-cover to score, so the step must gate itself out cleanly rather
   than spend ~40 s instrumenting the suite for a guaranteed "no lines".
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from scripts.pr_validate.context import Context
from scripts.pr_validate.steps.diff_coverage import (
    DiffCoverageStep,
    _parse_diff_cover,
    _path_exists,
    _run_group_bounded,
)

# --------------------------------------------------------------------------
# diff-cover output samples (verbatim shape of the tool's stdout footer)
# --------------------------------------------------------------------------

_DC_WITH_LINES = """\
-------------
Diff Coverage
Diff: origin/main...HEAD, staged and unstaged changes
-------------
vllm_mlx/quantized_batch_cache.py (80.0%): Missing lines 12,45
-------------
Total:   10 lines
Missing: 2 lines
Coverage: 80%
-------------
"""

_DC_NO_LINES = "No lines with coverage information in this diff.\n"

_DC_FULL = """\
-------------
Diff Coverage
Diff: origin/main...HEAD
-------------
vllm_mlx/foo.py (100%)
-------------
Total:   7 lines
Missing: 0 lines
Coverage: 100%
-------------
"""


class TestParseDiffCover:
    def test_parses_percent_and_line_counts(self):
        parsed = _parse_diff_cover(_DC_WITH_LINES)
        assert parsed is not None
        pct, covered, total = parsed
        assert (covered, total) == (8, 10)
        assert pct == pytest.approx(80.0)

    def test_percent_keeps_resolution_from_total_and_missing(self):
        # 2/3 covered → 66.67%, computed from Total/Missing (NOT the
        # rounded "Coverage: 67%" line), so the baseline keeps decimals.
        text = "Total:   3 lines\nMissing: 1 lines\nCoverage: 67%\n"
        parsed = _parse_diff_cover(text)
        assert parsed is not None
        pct, covered, total = parsed
        assert (covered, total) == (2, 3)
        assert pct == pytest.approx(66.666, abs=0.01)

    def test_full_coverage(self):
        parsed = _parse_diff_cover(_DC_FULL)
        assert parsed == (100.0, 7, 7)

    def test_no_lines_returns_none(self):
        assert _parse_diff_cover(_DC_NO_LINES) is None

    def test_malformed_returns_none(self):
        assert _parse_diff_cover("garbage without a footer") is None
        assert _parse_diff_cover("") is None

    def test_zero_total_returns_none(self):
        # Guard against a divide-by-zero if diff-cover ever emits a
        # degenerate "Total: 0 lines".
        assert _parse_diff_cover("Total:   0 lines\nMissing: 0 lines\n") is None


class TestPathExists:
    def test_none_is_false(self):
        assert _path_exists(None) is False

    def test_real_paths(self, tmp_path):
        assert _path_exists(tmp_path) is True
        assert _path_exists(tmp_path / "nope") is False

    def test_never_raises_on_oserror(self, tmp_path, monkeypatch):
        # pathlib re-raises EACCES/EIO from .exists(); the guard must swallow
        # it and report 'not present' so the advisory contract can't break.
        def exists_boom(self, *a, **k):
            raise OSError("Permission denied")

        monkeypatch.setattr(Path, "exists", exists_boom)
        assert _path_exists(tmp_path / "x") is False


# --------------------------------------------------------------------------
# Context helper — Context.__post_init__ insists on a repo-root cwd.
# --------------------------------------------------------------------------


@pytest.fixture
def ctx_factory(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'fake'\n")
    monkeypatch.chdir(tmp_path)

    def _make(files_changed: list[str]) -> Context:
        ctx = Context(pr_number=1)
        ctx.files_changed = files_changed
        ctx.work_dir = tmp_path / "work"
        return ctx

    return _make


class TestShouldRun:
    def test_runs_on_production_change_medium_blast(self, ctx_factory):
        ctx = ctx_factory(["vllm_mlx/quantized_batch_cache.py"])
        assert ctx.blast_radius == "medium"
        assert DiffCoverageStep().should_run(ctx) is True

    def test_runs_on_production_change_high_blast(self, ctx_factory):
        ctx = ctx_factory(["vllm_mlx/scheduler.py"])
        assert ctx.blast_radius == "high"
        assert DiffCoverageStep().should_run(ctx) is True

    def test_skips_docs_only_low_blast(self, ctx_factory):
        ctx = ctx_factory(["docs/guide.md", "README.md"])
        assert ctx.blast_radius == "low"
        assert DiffCoverageStep().should_run(ctx) is False

    def test_skips_tests_only(self, ctx_factory):
        # medium blast (tests/ isn't low), but no production lines to score.
        ctx = ctx_factory(["tests/test_foo.py"])
        assert DiffCoverageStep().should_run(ctx) is False

    def test_skips_deps_only(self, ctx_factory):
        # pyproject.toml is high blast, yet still nothing under vllm_mlx/.
        ctx = ctx_factory(["pyproject.toml"])
        assert ctx.blast_radius == "high"
        assert DiffCoverageStep().should_run(ctx) is False

    def test_skips_non_python_production_file(self, ctx_factory):
        ctx = ctx_factory(["vllm_mlx/py.typed"])
        assert DiffCoverageStep().should_run(ctx) is False


# --------------------------------------------------------------------------
# The advisory contract: never fail/error, whatever the subprocesses do.
# --------------------------------------------------------------------------


def _both_tools_present(monkeypatch):
    """Make the tooling-availability probe report both tools installed."""
    import importlib.util as _u

    real = _u.find_spec

    def fake_find_spec(name, *a, **k):
        if name in ("pytest_cov", "diff_cover"):
            # Return a truthy sentinel — the step only checks ``is None``.
            return object()
        return real(name, *a, **k)

    monkeypatch.setattr(
        "scripts.pr_validate.steps.diff_coverage.importlib.util.find_spec",
        fake_find_spec,
    )


def _xml_target(cmd: list[str]) -> str | None:
    for part in cmd:
        if part.startswith("--cov-report=xml:"):
            return part.split("xml:", 1)[1]
    return None


class TestAdvisoryContract:
    def test_pass_with_finding_on_good_run(self, ctx_factory, monkeypatch):
        _both_tools_present(monkeypatch)
        ctx = ctx_factory(["vllm_mlx/quantized_batch_cache.py"])

        def fake_run(cmd, *a, **k):
            if "pytest" in cmd:
                # A CLEAN suite (exit 0) that writes coverage.xml.
                target = _xml_target(cmd)
                assert target is not None
                Path(target).write_text("<coverage/>")
                return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")
            # diff-cover invocation.
            return subprocess.CompletedProcess(cmd, 0, stdout=_DC_WITH_LINES, stderr="")

        monkeypatch.setattr(
            "scripts.pr_validate.steps.diff_coverage._run_group_bounded", fake_run
        )
        res = DiffCoverageStep().run(ctx)
        assert res.status == "pass"
        assert "80" in res.summary
        assert res.findings and "ADVISORY" in res.findings[0]

    @pytest.mark.parametrize(
        "base_sha, base_branch, expected_ref",
        [
            ("abc123def", "main", "abc123def"),  # known base → the SHA itself
            ("", "main", "origin/main"),  # no metadata → remote-qualified branch
            ("", "release-0.11", "origin/release-0.11"),  # release-branch PR
        ],
    )
    def test_diff_cover_command_is_well_formed(
        self, ctx_factory, monkeypatch, base_sha, base_branch, expected_ref
    ):
        # nit (codex #1220 r6): the happy-path mock accepted ANY diff-cover
        # invocation, so a broken module name / args / compare-ref stayed
        # green. Pin the fully-constructed command — including base compare-ref
        # resolution: the SHA when known, else ``origin/<branch>`` so it
        # resolves in a detached CI checkout (a bare local branch may not).
        _both_tools_present(monkeypatch)
        ctx = ctx_factory(["vllm_mlx/quantized_batch_cache.py"])
        ctx.base_sha = base_sha
        ctx.base_branch = base_branch
        captured: dict[str, list[str]] = {}

        def fake_run(cmd, *a, **k):
            if "pytest" in cmd:
                Path(_xml_target(cmd)).write_text("<coverage/>")
                return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")
            captured["dc_cmd"] = cmd
            return subprocess.CompletedProcess(cmd, 0, stdout=_DC_WITH_LINES, stderr="")

        monkeypatch.setattr(
            "scripts.pr_validate.steps.diff_coverage._run_group_bounded", fake_run
        )
        res = DiffCoverageStep().run(ctx)
        assert res.status == "pass"
        dc = captured["dc_cmd"]
        assert dc[:3] == [sys.executable, "-m", "diff_cover.diff_cover_tool"]
        assert dc[3].endswith("coverage.xml")
        assert dc[4] == "--compare-branch"
        assert dc[5] == expected_ref

    def test_measures_with_caveat_when_some_tests_failed(
        self, ctx_factory, monkeypatch
    ):
        # codex #1220 r4: exit 1 means the suite RAN but some tests failed —
        # coverage.xml is still a complete record of executed lines, and an
        # advisory baseline collector must survive an unrelated flaky test
        # rather than discard the whole measurement. So exit 1 is MEASURED
        # (pass), with a caveat noting the % may under-count. full_unit still
        # owns gating on the red suite.
        _both_tools_present(monkeypatch)
        ctx = ctx_factory(["vllm_mlx/quantized_batch_cache.py"])
        dc_called = {"n": 0}

        def fake_run(cmd, *a, **k):
            if "pytest" in cmd:
                Path(_xml_target(cmd)).write_text("<coverage/>")  # xml IS written
                return subprocess.CompletedProcess(cmd, 1, stdout="1 failed", stderr="")
            dc_called["n"] += 1
            return subprocess.CompletedProcess(cmd, 0, stdout=_DC_WITH_LINES, stderr="")

        monkeypatch.setattr(
            "scripts.pr_validate.steps.diff_coverage._run_group_bounded", fake_run
        )
        res = DiffCoverageStep().run(ctx)
        assert res.status == "pass"
        assert dc_called["n"] == 1  # DID proceed to diff-cover
        assert "80" in res.summary
        # The finding must flag that some tests failed (honest advisory signal).
        assert res.findings and "some unit tests failed" in res.findings[0]

    @pytest.mark.parametrize("exit_code", [2, 3, 4, 5])
    def test_skip_when_suite_interrupted_or_errored(
        self, ctx_factory, monkeypatch, exit_code
    ):
        # Exit 2-5 = interrupted / internal error / usage error / no tests
        # collected — no dependable coverage even if a stale-looking xml is
        # present. Must skip BEFORE diff-cover, unlike the exit-1 case.
        _both_tools_present(monkeypatch)
        ctx = ctx_factory(["vllm_mlx/quantized_batch_cache.py"])
        dc_called = {"n": 0}

        def fake_run(cmd, *a, **k):
            if "pytest" in cmd:
                Path(_xml_target(cmd)).write_text("<coverage/>")
                return subprocess.CompletedProcess(
                    cmd, exit_code, stdout="boom", stderr=""
                )
            dc_called["n"] += 1
            return subprocess.CompletedProcess(cmd, 0, stdout=_DC_WITH_LINES, stderr="")

        monkeypatch.setattr(
            "scripts.pr_validate.steps.diff_coverage._run_group_bounded", fake_run
        )
        res = DiffCoverageStep().run(ctx)
        assert res.status == "skip"
        assert f"exit {exit_code}" in res.summary
        assert dc_called["n"] == 0  # never reached diff-cover

    def test_stale_xml_is_removed_before_run(self, ctx_factory, monkeypatch):
        # A leftover coverage.xml from a previous run must not be able to
        # masquerade as this run's result when pytest fails to write one.
        _both_tools_present(monkeypatch)
        ctx = ctx_factory(["vllm_mlx/quantized_batch_cache.py"])
        stale = ctx.artifact_path("coverage.xml")
        stale.write_text("<coverage>STALE</coverage>")

        def fake_run(cmd, *a, **k):
            # Clean exit but DO NOT write a fresh xml. Without the
            # pre-run unlink, the stale file would survive and get scored.
            return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

        monkeypatch.setattr(
            "scripts.pr_validate.steps.diff_coverage._run_group_bounded", fake_run
        )
        res = DiffCoverageStep().run(ctx)
        assert res.status == "skip"
        assert "no coverage.xml" in res.summary

    def test_skip_when_no_coverage_xml(self, ctx_factory, monkeypatch):
        _both_tools_present(monkeypatch)
        ctx = ctx_factory(["vllm_mlx/quantized_batch_cache.py"])

        def fake_run(cmd, *a, **k):
            # Clean exit but no xml written (e.g. cov plugin misconfigured).
            return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

        monkeypatch.setattr(
            "scripts.pr_validate.steps.diff_coverage._run_group_bounded", fake_run
        )
        res = DiffCoverageStep().run(ctx)
        assert res.status == "skip"

    def test_skip_on_pytest_timeout(self, ctx_factory, monkeypatch):
        _both_tools_present(monkeypatch)
        ctx = ctx_factory(["vllm_mlx/quantized_batch_cache.py"])

        def fake_run(cmd, *a, **k):
            raise subprocess.TimeoutExpired(cmd, k.get("timeout", 1))

        monkeypatch.setattr(
            "scripts.pr_validate.steps.diff_coverage._run_group_bounded", fake_run
        )
        res = DiffCoverageStep().run(ctx)
        assert res.status == "skip"
        assert "exceeded" in res.summary

    def test_skip_on_diff_cover_timeout(self, ctx_factory, monkeypatch):
        _both_tools_present(monkeypatch)
        ctx = ctx_factory(["vllm_mlx/quantized_batch_cache.py"])

        def fake_run(cmd, *a, **k):
            if "pytest" in cmd:
                Path(_xml_target(cmd)).write_text("<coverage/>")
                return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")
            raise subprocess.TimeoutExpired(cmd, k.get("timeout", 1))

        monkeypatch.setattr(
            "scripts.pr_validate.steps.diff_coverage._run_group_bounded", fake_run
        )
        res = DiffCoverageStep().run(ctx)
        assert res.status == "skip"
        assert "diff-cover" in res.summary and "exceeded" in res.summary

    def test_skip_on_diff_cover_nonzero_exit_even_with_footer(
        self, ctx_factory, monkeypatch
    ):
        # codex #1220 r2: a failed/interrupted diff-cover that still
        # printed a parseable footer must NOT be published as success.
        _both_tools_present(monkeypatch)
        ctx = ctx_factory(["vllm_mlx/quantized_batch_cache.py"])

        def fake_run(cmd, *a, **k):
            if "pytest" in cmd:
                Path(_xml_target(cmd)).write_text("<coverage/>")
                return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")
            # diff-cover errored (exit 1) but emitted a valid-looking footer.
            return subprocess.CompletedProcess(
                cmd, 1, stdout=_DC_WITH_LINES, stderr="err"
            )

        monkeypatch.setattr(
            "scripts.pr_validate.steps.diff_coverage._run_group_bounded", fake_run
        )
        res = DiffCoverageStep().run(ctx)
        assert res.status == "skip"
        assert "diff-cover exit 1" in res.summary

    def test_skip_when_artifact_path_raises(self, ctx_factory, monkeypatch):
        # codex #1220 r2: artifact_path() does a mkdir that can raise on
        # disk-full / permission errors. That must be caught and skipped,
        # not escape through execute() as a blocking error.
        ctx = ctx_factory(["vllm_mlx/quantized_batch_cache.py"])

        def raising_artifact_path(name):
            raise OSError("Read-only file system")

        monkeypatch.setattr(ctx, "artifact_path", raising_artifact_path)
        res = DiffCoverageStep().run(ctx)
        assert res.status == "skip"
        assert res.status not in ("fail", "error")
        assert res.artifacts == []  # no log path could be resolved

    def test_skip_when_diff_cover_finds_no_lines(self, ctx_factory, monkeypatch):
        _both_tools_present(monkeypatch)
        ctx = ctx_factory(["vllm_mlx/quantized_batch_cache.py"])

        def fake_run(cmd, *a, **k):
            if "pytest" in cmd:
                Path(_xml_target(cmd)).write_text("<coverage/>")
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
            return subprocess.CompletedProcess(cmd, 0, stdout=_DC_NO_LINES, stderr="")

        monkeypatch.setattr(
            "scripts.pr_validate.steps.diff_coverage._run_group_bounded", fake_run
        )
        res = DiffCoverageStep().run(ctx)
        assert res.status == "skip"

    def test_skip_when_tooling_missing(self, ctx_factory, monkeypatch):
        # find_spec returns None for pytest_cov → clean skip, no subprocess.
        import importlib.util as _u

        real = _u.find_spec
        monkeypatch.setattr(
            "scripts.pr_validate.steps.diff_coverage.importlib.util.find_spec",
            lambda name, *a, **k: None if name == "pytest_cov" else real(name),
        )
        ctx = ctx_factory(["vllm_mlx/quantized_batch_cache.py"])
        res = DiffCoverageStep().run(ctx)
        assert res.status == "skip"
        assert "pytest-cov" in res.summary

    def test_internal_crash_downgrades_to_skip_not_error(
        self, ctx_factory, monkeypatch
    ):
        _both_tools_present(monkeypatch)
        ctx = ctx_factory(["vllm_mlx/quantized_batch_cache.py"])

        def boom(cmd, *a, **k):
            raise RuntimeError("simulated subprocess explosion")

        monkeypatch.setattr(
            "scripts.pr_validate.steps.diff_coverage._run_group_bounded", boom
        )
        res = DiffCoverageStep().run(ctx)
        # The whole point: a crash in the measurer must NOT surface as
        # error (which the scorecard treats as blocking).
        assert res.status == "skip"
        assert res.status not in ("fail", "error")

    def test_crash_with_unwritable_log_still_skips(self, ctx_factory, monkeypatch):
        # codex #1220 fix #2: if the diagnostic-log write ALSO fails
        # (disk full / permissions) inside the exception handler, the
        # step must STILL return skip, not let the write error escape as
        # a blocking error.
        _both_tools_present(monkeypatch)
        ctx = ctx_factory(["vllm_mlx/quantized_batch_cache.py"])

        def boom(cmd, *a, **k):
            raise RuntimeError("subprocess explosion")

        def unwritable(self, *a, **k):
            raise OSError("No space left on device")

        monkeypatch.setattr(
            "scripts.pr_validate.steps.diff_coverage._run_group_bounded", boom
        )
        monkeypatch.setattr(Path, "write_text", unwritable)
        res = DiffCoverageStep().run(ctx)
        assert res.status == "skip"

    def test_advisory_survives_exists_oserror(self, ctx_factory, monkeypatch):
        # codex #1220 r7: pathlib's Path.exists() re-raises OSErrors other
        # than ENOENT (e.g. EACCES / EIO). A bare .exists() in _skip could
        # then escape run()'s handler and become a blocking error — the one
        # thing this advisory step must never do. With Path.exists() raising
        # from anywhere in the flow, run() must STILL resolve to skip.
        _both_tools_present(monkeypatch)
        ctx = ctx_factory(["vllm_mlx/quantized_batch_cache.py"])

        def boom(cmd, *a, **k):
            raise RuntimeError("subprocess explosion")

        def exists_boom(self, *a, **k):
            raise OSError("Permission denied")

        monkeypatch.setattr(
            "scripts.pr_validate.steps.diff_coverage._run_group_bounded", boom
        )
        monkeypatch.setattr(Path, "exists", exists_boom)
        res = DiffCoverageStep().run(ctx)
        assert res.status == "skip"
        assert res.status not in ("fail", "error")

    def test_execute_wrapper_skips_when_gated_out(self, ctx_factory, monkeypatch):
        # Through the base execute() wrapper, a docs-only PR gates out.
        ctx = ctx_factory(["docs/x.md"])
        res = DiffCoverageStep().execute(ctx)
        assert res.status == "skip"


# --------------------------------------------------------------------------
# Registration — advisory step is wired in and positioned last.
# --------------------------------------------------------------------------


class TestRegistration:
    def test_registered_and_last(self):
        from scripts.pr_validate.runner import STEPS

        names = [s.name for s in STEPS]
        assert "diff_coverage" in names
        # Last on purpose: skip its ~40 s cost in fail-fast mode once any
        # real gate above has already blocked the PR.
        assert names[-1] == "diff_coverage"

    def test_never_declared_as_gating(self):
        # A defensive pin on the advisory intent: the step opts into
        # continue_on_error so a stray error can't halt the pipeline.
        assert DiffCoverageStep().continue_on_error is True


# --------------------------------------------------------------------------
# _run_group_bounded — the process-group-bounded subprocess helper. Real
# subprocesses (no mocking) so we actually exercise the timeout + kill.
# --------------------------------------------------------------------------


class TestRunGroupBounded:
    def test_returns_completed_process_on_success(self):
        proc = _run_group_bounded(
            [sys.executable, "-c", "print('hi')"], cwd=".", timeout=30
        )
        assert proc.returncode == 0
        assert "hi" in proc.stdout

    def test_propagates_nonzero_returncode(self):
        proc = _run_group_bounded(
            [sys.executable, "-c", "import sys; sys.exit(3)"], cwd=".", timeout=30
        )
        assert proc.returncode == 3

    def test_raises_timeout_on_a_lone_sleeper(self):
        with pytest.raises(subprocess.TimeoutExpired):
            _run_group_bounded(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                cwd=".",
                timeout=1,
            )

    def test_timeout_kills_the_whole_group_including_descendants(self, tmp_path):
        # The contract that actually matters: on timeout the ENTIRE process
        # group dies, not just the leader. A leader-only kill would orphan
        # any grandchild a timed-out pytest spawned (xdist workers, a serve
        # subprocess) and let it contaminate later validation steps — and,
        # worse, a grandchild holding the inherited stdout/stderr pipes open
        # would wedge the reap. This test distinguishes the two: it fails if
        # group-kill ever regresses to a leader-only kill.
        #
        # Liveness is detected via a HEARTBEAT the grandchild grows, NOT an
        # ``os.kill(pid, 0)`` probe: signal 0 succeeds for a not-yet-reaped
        # zombie (and, after PID reuse, for an unrelated process), so a
        # PID-based probe is racy (codex #1220 r5). Instead the grandchild
        # appends a byte to a file every 50 ms; after the group kill the file
        # STOPS growing. A monotonically-growing size can't false-negative and
        # is immune to zombie/PID-reuse — if group-kill regressed to
        # leader-only, the orphaned grandchild (reparented to init) would keep
        # appending and the size would still advance.
        #
        # Leader = ``sh``; it backgrounds the grandchild in the SAME process
        # group (non-interactive sh has no job control, so ``&`` jobs stay in
        # the shell's group), then blocks so the leader is alive when the
        # timeout fires. The grandchild inherits the subprocess pipes, so a
        # leader-only kill would also reproduce the reap-wedge from codex #1220.
        import time as _time

        heartbeat = tmp_path / "heartbeat"
        cmd = [
            "sh",
            "-c",
            f"(while true; do printf . >> '{heartbeat}'; sleep 0.05; done) & sleep 120",
        ]

        with pytest.raises(subprocess.TimeoutExpired):
            _run_group_bounded(cmd, cwd=str(tmp_path), timeout=1)

        # Grandchild must have started (heartbeat exists) before we can judge
        # whether it STOPPED. It ticks every 50 ms and the leader ran for the
        # full 1 s timeout, so the file exists well before we get here.
        deadline = _time.time() + 5.0
        while not heartbeat.exists() and _time.time() < deadline:
            _time.sleep(0.02)
        assert heartbeat.exists(), "grandchild never started — test is inconclusive"

        # Sample size, wait several heartbeat intervals, sample again. A dead
        # grandchild leaves the size fixed; a survivor grows it by ~20 bytes/s.
        size1 = heartbeat.stat().st_size
        _time.sleep(1.0)
        size2 = heartbeat.stat().st_size
        assert size1 == size2, (
            f"heartbeat kept growing after the timeout ({size1} → {size2} "
            "bytes) — the grandchild survived, so group-kill regressed to a "
            "leader-only kill"
        )
