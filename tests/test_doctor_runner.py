# SPDX-License-Identifier: Apache-2.0
"""Tests for doctor runner internals.

These pin down the exit-code contract (which is the entire point of
the doctor for CI integration), the markdown-cell escaping helper,
and the run-directory atomic reservation.
"""

from pathlib import Path

import pytest

from vllm_mlx.doctor.runner import (
    CheckResult,
    DoctorRunner,
    Status,
    md_cell,
)

# ----------------------------------------------------------------------
# Exit-code contract
# ----------------------------------------------------------------------

class TestExitCodeContract:
    """Documented contract: 0 = pass, 1 = regression, 2 = functional fail."""

    def _runner(self, tmp_path: Path) -> DoctorRunner:
        return DoctorRunner(tier="test", run_dir=tmp_path / "run")

    def _result(self, name: str, status: Status) -> CheckResult:
        return CheckResult(name=name, status=status, duration_s=0.1)

    def test_all_pass_is_zero(self, tmp_path):
        r = self._runner(tmp_path)
        r.checks = [self._result("a", Status.PASS), self._result("b", Status.PASS)]
        assert r._compute_exit_code() == 0

    def test_skip_only_is_zero(self, tmp_path):
        r = self._runner(tmp_path)
        r.checks = [self._result("a", Status.SKIP)]
        assert r._compute_exit_code() == 0

    def test_pass_with_regression_is_one(self, tmp_path):
        r = self._runner(tmp_path)
        r.checks = [
            self._result("a", Status.PASS),
            self._result("b", Status.REGRESSION),
        ]
        assert r._compute_exit_code() == 1

    def test_pass_with_fail_is_two(self, tmp_path):
        r = self._runner(tmp_path)
        r.checks = [
            self._result("a", Status.PASS),
            self._result("b", Status.FAIL),
        ]
        assert r._compute_exit_code() == 2

    def test_fail_dominates_regression(self, tmp_path):
        """Worse-signal-wins: a run with both regression and fail
        reports fail (rc=2), so CI doesn't downgrade to "just perf"."""
        r = self._runner(tmp_path)
        r.checks = [
            self._result("a", Status.REGRESSION),
            self._result("b", Status.FAIL),
        ]
        assert r._compute_exit_code() == 2

    def test_empty_runs_are_zero(self, tmp_path):
        r = self._runner(tmp_path)
        assert r._compute_exit_code() == 0


# ----------------------------------------------------------------------
# md_cell — markdown table escaping
# ----------------------------------------------------------------------

class TestMdCell:
    def test_passes_through_safe_text(self):
        assert md_cell("hello world") == "hello world"

    def test_escapes_pipes(self):
        """Stack-trace-like detail must not break the table layout."""
        assert md_cell('File "x.py" | line 42') == 'File "x.py" \\| line 42'

    def test_collapses_newlines(self):
        assert md_cell("line 1\nline 2\nline 3") == "line 1 line 2 line 3"

    def test_truncates_to_max_len(self):
        long = "a" * 200
        out = md_cell(long, max_len=50)
        assert len(out) == 50
        assert out.endswith("...")

    def test_max_len_zero_means_uncapped(self):
        long = "a" * 200
        assert md_cell(long, max_len=0) == long

    def test_handles_empty(self):
        assert md_cell("") == ""

    def test_handles_none_via_falsy_fallback(self):
        # Defensive: callers occasionally pass None for missing details.
        assert md_cell(None) == ""  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# Atomic run-dir reservation
# ----------------------------------------------------------------------

class TestRunDirReservation:
    def test_back_to_back_runs_get_distinct_dirs(self, tmp_path, monkeypatch):
        """Concurrent invocations must never share a run dir, otherwise
        report.md / result.json get clobbered."""
        from vllm_mlx.doctor import runner as runner_mod
        monkeypatch.setattr(runner_mod, "RUNS_DIR", tmp_path)
        r1 = DoctorRunner(tier="x")
        r2 = DoctorRunner(tier="x")
        r3 = DoctorRunner(tier="x")
        assert r1.run_dir != r2.run_dir != r3.run_dir
        assert r1.run_dir.exists()
        assert r2.run_dir.exists()
        assert r3.run_dir.exists()


# ----------------------------------------------------------------------
# Report rendering smoke test (catches structural regressions)
# ----------------------------------------------------------------------

class TestReportRendering:
    def test_basic_report_renders_table(self, tmp_path, monkeypatch):
        from vllm_mlx.doctor import runner as runner_mod
        monkeypatch.setattr(runner_mod, "RUNS_DIR", tmp_path)
        r = DoctorRunner(tier="test")

        def fake_check():
            return CheckResult(
                name="example",
                status=Status.PASS,
                duration_s=1.5,
                detail="all good",
            )
        r.run_check("example", fake_check)
        result = r.finalize()

        report = (Path(result.run_dir) / "report.md").read_text()
        assert "# Doctor Report — `test`" in report
        assert "| example | pass | 1.5s | all good |" in report
        assert result.exit_code == 0

    def test_report_includes_diff_sections_when_stashed(self, tmp_path, monkeypatch):
        from vllm_mlx.doctor import runner as runner_mod
        monkeypatch.setattr(runner_mod, "RUNS_DIR", tmp_path)
        r = DoctorRunner(tier="test")
        r._pending_diff_sections = [  # type: ignore[attr-defined]
            ("model-a", "| metric | base | curr | dp | s |\n| --- | --- | --- | --- | --- |\n"),
        ]

        # Need at least one check otherwise finalize() works fine but the
        # "Baseline diffs" section is the test point.
        r.run_check("noop", lambda: CheckResult("noop", Status.PASS, 0.1))
        result = r.finalize()

        report = (Path(result.run_dir) / "report.md").read_text()
        assert "## Baseline diffs" in report
        assert "### model-a" in report
        assert "| metric | base | curr | dp | s |" in report

    def test_pipe_in_detail_does_not_break_table(self, tmp_path, monkeypatch):
        from vllm_mlx.doctor import runner as runner_mod
        monkeypatch.setattr(runner_mod, "RUNS_DIR", tmp_path)
        r = DoctorRunner(tier="test")

        def crashing():
            return CheckResult(
                name="weird",
                status=Status.FAIL,
                duration_s=0.1,
                detail='Traceback: File "x.py" | line 42 | a',
            )
        r.run_check("weird", crashing)
        result = r.finalize()
        report = (Path(result.run_dir) / "report.md").read_text()

        # Find the row and confirm exactly 4 unescaped pipes (table cell
        # boundaries: leading | + 3 separators + trailing |) means the row
        # has exactly 5 cells like the header.
        for line in report.splitlines():
            if "weird" in line and line.strip().startswith("|"):
                # Count only unescaped pipes (those NOT preceded by '\').
                unescaped = 0
                for i, ch in enumerate(line):
                    if ch == "|" and (i == 0 or line[i - 1] != "\\"):
                        unescaped += 1
                assert unescaped == 5, (
                    f"row should have exactly 5 unescaped pipe boundaries, "
                    f"got {unescaped} in: {line!r}"
                )
                break
        else:
            pytest.fail("could not find the weird-check row in report")


# ----------------------------------------------------------------------
# Boot-timeout heuristic
# ----------------------------------------------------------------------

class TestSuggestedBootTimeout:
    """Picks 600s for ≥20B-shaped aliases, 180s for smaller / unknown.

    Codex round 3 P2: scoping the timeout purely by tier (180s for
    'check', 600s for 'full') broke the supported 'doctor check
    --model qwen3.5-35b' workflow.  Per-model heuristic restores it.
    """

    def test_small_models_get_short_timeout(self):
        from vllm_mlx.doctor.cli import _suggested_boot_timeout
        for alias in ("qwen3.5-4b", "qwen3-0.6b", "llama3-3b", "gemma3-1b",
                      "phi4-14b", "ministral-3b", "qwen3-vl-8b"):
            assert _suggested_boot_timeout(alias) == 180, alias

    def test_large_models_get_long_timeout(self):
        from vllm_mlx.doctor.cli import _suggested_boot_timeout
        for alias in ("qwen3.5-27b", "qwen3.5-35b", "qwopus-27b",
                      "deepseek-r1-32b", "gemma-4-26b", "hermes4-70b",
                      "qwen3.5-122b", "kimi-48b", "gpt-oss-20b"):
            assert _suggested_boot_timeout(alias) == 600, alias

    def test_unknown_alias_defaults_to_short(self):
        """Fast-fail when the heuristic doesn't know — better to time
        out a legitimately huge model and ask the user to retry with
        an explicit override than to make 'doctor check' feel sluggish
        for everyone else."""
        from vllm_mlx.doctor.cli import _suggested_boot_timeout
        assert _suggested_boot_timeout("some-random-model") == 180
        assert _suggested_boot_timeout("model-1.2") == 180  # 1.2B → not matched
