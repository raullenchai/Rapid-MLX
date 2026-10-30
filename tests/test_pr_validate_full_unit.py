# SPDX-License-Identifier: Apache-2.0
"""Pure Python contracts for optional full-suite parallelism and diagnostics."""

import subprocess
import sys
from pathlib import Path

import pytest

from scripts.pr_validate import _xdist
from scripts.pr_validate.context import Context
from scripts.pr_validate.steps.full_unit import (
    FullUnitStep,
    _extract_failed_lines,
    _last_summary_line,
)


@pytest.fixture(autouse=True)
def clean_workers(monkeypatch):
    monkeypatch.delenv("PR_VALIDATE_UNIT_WORKERS", raising=False)
    monkeypatch.delenv("PR_VALIDATE_FULL_UNIT_TIMEOUT_S", raising=False)
    monkeypatch.setattr(_xdist.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(_xdist.os, "getloadavg", lambda: (0.0, 0.0, 0.0), raising=False)


@pytest.mark.parametrize("value", ["-1", "abc", "", "2.5"])
def test_invalid_workers(monkeypatch, value):
    monkeypatch.setenv("PR_VALIDATE_UNIT_WORKERS", value)
    with pytest.raises(ValueError, match="PR_VALIDATE_UNIT_WORKERS must be"):
        _xdist.pytest_parallel_args()


@pytest.mark.parametrize("workers", [0, 1, 4, 20, 32])
def test_override(monkeypatch, workers):
    monkeypatch.setenv("PR_VALIDATE_UNIT_WORKERS", str(workers))

    def unexpected_call():
        pytest.fail("override must bypass load and core detection")

    monkeypatch.setattr(_xdist.os, "getloadavg", unexpected_call)
    monkeypatch.setattr(_xdist, "_performance_cores", unexpected_call)
    args, note = _xdist.pytest_parallel_args()
    assert args == ([] if workers <= 1 else ["-n", str(workers), "--dist", "loadgroup"])
    assert ("serially" if workers <= 1 else f"{workers} workers") in note
    assert f"PR_VALIDATE_UNIT_WORKERS={workers}" in note


def test_missing_xdist(monkeypatch):
    monkeypatch.setenv("PR_VALIDATE_UNIT_WORKERS", "4")
    monkeypatch.setattr(_xdist.importlib.util, "find_spec", lambda name: None)
    assert _xdist.pytest_parallel_args() == (
        [],
        "pytest-xdist not installed — running serially "
        "(requested 4 workers; PR_VALIDATE_UNIT_WORKERS=4)",
    )


@pytest.mark.parametrize(
    ("cores", "load1", "workers"),
    [
        (20, 0.0, 20),
        (20, 0.2, 19),
        (20, 8.3, 11),
        (20, 90.0, 4),
        (2, 90.0, 2),
        (1, 8.3, 1),
        (128, 90.0, 20),
    ],
)
def test_load_aware_default(monkeypatch, cores, load1, workers):
    monkeypatch.setattr(_xdist, "_performance_cores", lambda: cores)
    monkeypatch.setattr(_xdist.os, "getloadavg", lambda: (load1, 99.0, 99.0))
    args, note = _xdist.pytest_parallel_args()
    assert args == ([] if workers <= 1 else ["-n", str(workers), "--dist", "loadgroup"])
    assert f"load1={load1:.2f}, perf cores={cores}" in note
    assert (
        "1 pytest worker — running serially" if workers == 1 else f"{workers} workers"
    ) in note


@pytest.mark.parametrize("error", [OSError, AttributeError, None])
@pytest.mark.parametrize("cores", [6, 128])
def test_load_unavailable_falls_back(monkeypatch, error, cores):
    monkeypatch.setattr(_xdist, "_performance_cores", lambda: cores)

    def fail():
        raise error("load unavailable")

    if error is None:
        monkeypatch.delattr(_xdist.os, "getloadavg")
    else:
        monkeypatch.setattr(_xdist.os, "getloadavg", fail)
    args, note = _xdist.pytest_parallel_args()
    assert args == ["-n", str(min(20, cores)), "--dist", "loadgroup"]
    assert "load1=0.00" in note


def test_missing_xdist_includes_load(monkeypatch):
    monkeypatch.setattr(_xdist, "_performance_cores", lambda: 20)
    monkeypatch.setattr(_xdist.os, "getloadavg", lambda: (8.37, 0.0, 0.0))
    monkeypatch.setattr(_xdist.importlib.util, "find_spec", lambda name: None)
    args, note = _xdist.pytest_parallel_args()
    assert args == []
    assert "running serially" in note
    assert "requested 11 workers; load1=8.37, perf cores=20" in note


def test_default_perf_cores(monkeypatch):
    monkeypatch.setattr(_xdist.sys, "platform", "darwin")

    def sysctl(cmd, **kwargs):
        assert cmd == ["sysctl", "-n", "hw.perflevel0.physicalcpu"]
        assert kwargs == dict(capture_output=True, text=True, check=True)
        return subprocess.CompletedProcess(cmd, 0, stdout="6\n")

    monkeypatch.setattr(_xdist.subprocess, "run", sysctl)
    assert _xdist.pytest_parallel_args()[0] == ["-n", "6", "--dist", "loadgroup"]
    monkeypatch.setattr(_xdist, "_performance_cores", lambda: 128)
    assert _xdist.pytest_parallel_args()[0] == ["-n", "20", "--dist", "loadgroup"]


@pytest.mark.parametrize("output", ["0", "-2", "unknown"])
def test_sysctl_invalid_falls_back(monkeypatch, output):
    monkeypatch.setattr(_xdist.sys, "platform", "darwin")
    monkeypatch.setattr(
        _xdist.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout=output),
    )
    monkeypatch.setattr(_xdist.os, "cpu_count", lambda: 3)
    assert _xdist._performance_cores() == 3


@pytest.mark.parametrize(
    "error", [OSError(), subprocess.CalledProcessError(1, "sysctl")]
)
def test_sysctl_failure_falls_back(monkeypatch, error):
    monkeypatch.setattr(_xdist.sys, "platform", "darwin")

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(_xdist.subprocess, "run", fail)
    monkeypatch.setattr(_xdist.os, "cpu_count", lambda: None)
    assert _xdist._performance_cores() == 1


def test_non_darwin_default(monkeypatch):
    monkeypatch.setattr(_xdist.sys, "platform", "linux")
    monkeypatch.setattr(_xdist.os, "cpu_count", lambda: 4)
    assert _xdist.pytest_parallel_args()[0] == ["-n", "4", "--dist", "loadgroup"]


@pytest.mark.parametrize(
    "summary",
    [
        "2 passed in 1.23s",
        "1 failed, 2 passed in 1.23s",
        "3 skipped, 19 errors in 79s",
        "3 skipped in 1.0s",
        "2 xfailed in 1.0s",
        "1 xpassed in 1.0s",
    ],
)
@pytest.mark.parametrize("decorated", [False, True])
def test_summary(summary, decorated):
    line = f"==== {summary} ====" if decorated else summary
    assert (
        _last_summary_line(f"[gw0] [ 12%] PASSED tests/test_a.py::test_a\n{line}\n")
        == summary
    )


def test_no_summary():
    assert _last_summary_line("") == ""
    assert _last_summary_line("[gw0] [ 12%] PASSED tests/a.py::test_a") == ""
    assert _last_summary_line("3 passed\nnoise") == ""


def test_failure_lines():
    assert _extract_failed_lines("FAILED outside summary\n") == []
    assert _extract_failed_lines("""
[gw0] [ 12%] FAILED tests/test_a.py::test_a
==== short test summary info ====
FAILED tests/test_a.py::test_a - AssertionError
ERROR gw3 - worker crashed
ERROR tests/test_b.py - collection mismatch
SKIPPED tests/test_c.py
==== 1 failed, 2 errors in 1s ====
FAILED after summary
""") == [
        "FAILED tests/test_a.py::test_a - AssertionError",
        "ERROR gw3 - worker crashed",
        "ERROR tests/test_b.py - collection mismatch",
    ]


@pytest.mark.parametrize("present", [False, True])
@pytest.mark.parametrize("returncode", [0, 1])
def test_full_unit_result(tmp_path, monkeypatch, present, returncode):
    monkeypatch.setenv("PR_VALIDATE_UNIT_WORKERS", "4")
    monkeypatch.setattr(
        _xdist.importlib.util, "find_spec", lambda name: object() if present else None
    )
    ctx = Context(pr_number=1, work_dir=tmp_path, files_changed=["tests/test_a.py"])
    step = FullUnitStep()
    assert step.should_run(ctx)
    assert not step.should_run(Context(pr_number=1, files_changed=["README.md"]))
    stdout = (
        "==== 2 passed in 1s ====\n"
        if returncode == 0
        else "==== short test summary info ====\nERROR gw3 - crashed\n==== 19 errors in 79s ====\n"
    )

    def run(cmd, **kwargs):
        assert cmd == [
            sys.executable,
            "-m",
            "pytest",
            "tests/",
            "--ignore=tests/integrations",
            "--ignore=tests/test_event_loop.py",
            "-q",
            "--no-header",
            "-rfE",
            "-o",
            "faulthandler_timeout=600",
            *(["-n", "4", "--dist", "loadgroup"] if present else []),
        ]
        assert kwargs == dict(
            cwd=str(ctx.repo_root), timeout=3600 if present else 14400
        )
        return subprocess.CompletedProcess(
            cmd, returncode, stdout=stdout, stderr="stderr"
        )

    monkeypatch.setattr("scripts.pr_validate.steps.full_unit._run_group_bounded", run)
    result = step.run(ctx)
    assert result.status == ("pass" if returncode == 0 else "fail")
    assert ("2 passed" if returncode == 0 else "19 errors") in result.summary
    note = "4 workers" if present else "pytest-xdist not installed — running serially"
    assert note in result.summary
    assert note in result.details
    assert note in Path(result.artifacts[0]).read_text()
    assert "stderr" in Path(result.artifacts[0]).read_text()
    if returncode:
        assert "ERROR gw3 - crashed" in result.details


@pytest.mark.parametrize("returncode", [0, 1])
def test_full_unit_without_summary(tmp_path, monkeypatch, returncode):
    monkeypatch.setenv("PR_VALIDATE_UNIT_WORKERS", "1")
    monkeypatch.setattr(
        "scripts.pr_validate.steps.full_unit._run_group_bounded",
        lambda *a, **k: subprocess.CompletedProcess(
            a, returncode, stdout=None, stderr=None
        ),
    )
    result = FullUnitStep().run(Context(pr_number=1, work_dir=tmp_path))
    assert (
        "all tests passed" if returncode == 0 else "pytest exited 1"
    ) in result.summary


def test_failure_truncation(tmp_path, monkeypatch):
    monkeypatch.setenv("PR_VALIDATE_UNIT_WORKERS", "1")
    stdout = "==== short test summary info ====\n" + "\n".join(
        f"FAILED test_{i}" for i in range(35)
    )
    monkeypatch.setattr(
        "scripts.pr_validate.steps.full_unit._run_group_bounded",
        lambda *a, **k: subprocess.CompletedProcess(a, 1, stdout=stdout, stderr=""),
    )
    result = FullUnitStep().run(Context(pr_number=1, work_dir=tmp_path))
    assert "FAILED test_29" in result.details
    assert "FAILED test_30" not in result.details
    assert "5 more" in result.details


@pytest.mark.parametrize("value", ["0", "-5", "abc"])
def test_invalid_full_unit_timeout(tmp_path, monkeypatch, value):
    monkeypatch.setenv("PR_VALIDATE_FULL_UNIT_TIMEOUT_S", value)
    with pytest.raises(ValueError, match="PR_VALIDATE_FULL_UNIT_TIMEOUT_S must be"):
        FullUnitStep().run(Context(pr_number=1, work_dir=tmp_path))


def test_full_unit_timeout_override(tmp_path, monkeypatch):
    monkeypatch.setenv("PR_VALIDATE_FULL_UNIT_TIMEOUT_S", "120")

    def run(cmd, *, cwd, timeout):
        assert timeout == 120
        return subprocess.CompletedProcess(cmd, 0, stdout="1 passed in 1s", stderr="")

    monkeypatch.setattr("scripts.pr_validate.steps.full_unit._run_group_bounded", run)
    assert FullUnitStep().run(Context(pr_number=1, work_dir=tmp_path)).status == "pass"


def test_full_unit_timeout_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("PR_VALIDATE_FULL_UNIT_TIMEOUT_S", "120")

    def run(cmd, *, cwd, timeout):
        raise subprocess.TimeoutExpired(cmd, timeout, output="partial", stderr="dump")

    monkeypatch.setattr("scripts.pr_validate.steps.full_unit._run_group_bounded", run)
    result = FullUnitStep().run(Context(pr_number=1, work_dir=tmp_path))
    assert result.status == "fail"
    assert "120s" in result.summary
    log = Path(result.artifacts[0]).read_text()
    assert "TIMED OUT" in log
    assert "partial" in log and "dump" in log
