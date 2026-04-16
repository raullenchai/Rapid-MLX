# SPDX-License-Identifier: Apache-2.0
"""Smoke-tier checks: static analysis + import sanity, no model required."""

from __future__ import annotations

import time
from pathlib import Path

from ..runner import REPO_ROOT, CheckResult, Status, python_executable, run_subprocess


def check_pytest_unit() -> CheckResult:
    """Run the unit-test slice (excludes integration + server-required tests)."""
    t0 = time.perf_counter()
    py = python_executable()
    cmd = [
        py, "-m", "pytest",
        "tests/",
        "-q",
        "--ignore=tests/integrations",
        "--deselect", "tests/test_event_loop.py",
    ]
    rc, stdout, stderr = run_subprocess(cmd, timeout=300)
    elapsed = time.perf_counter() - t0

    # Pull the summary line ("=== N passed, M skipped in Ts ===") for the report.
    summary_line = ""
    for line in stdout.strip().splitlines()[::-1]:
        if "passed" in line or "failed" in line or "error" in line:
            summary_line = line.strip()
            break

    if rc == 0:
        return CheckResult(
            name="pytest",
            status=Status.PASS,
            duration_s=elapsed,
            detail=summary_line,
        )
    return CheckResult(
        name="pytest",
        status=Status.FAIL,
        duration_s=elapsed,
        detail=summary_line or f"pytest exited {rc}\n{stderr[-1000:]}",
    )


def check_ruff() -> CheckResult:
    """Run ruff lint over the package + tests.

    Tries ``python -m ruff`` first (works when ruff is in the same venv),
    falls back to a standalone ``ruff`` binary on PATH.  Skips gracefully
    if neither is available.
    """
    import shutil

    t0 = time.perf_counter()
    py = python_executable()

    # Try python -m ruff first
    rc, stdout, stderr = run_subprocess(
        [py, "-m", "ruff", "check", "vllm_mlx/", "tests/"],
        timeout=60,
    )
    if rc != 0 and ("No module named" in stderr or "ModuleNotFoundError" in stderr):
        # Fall back to standalone binary
        ruff_bin = shutil.which("ruff")
        if ruff_bin:
            rc, stdout, stderr = run_subprocess(
                [ruff_bin, "check", "vllm_mlx/", "tests/"],
                timeout=60,
            )
        else:
            elapsed = time.perf_counter() - t0
            return CheckResult(
                name="ruff",
                status=Status.SKIP,
                duration_s=elapsed,
                detail="ruff not installed (neither module nor binary on PATH)",
            )

    elapsed = time.perf_counter() - t0
    if rc == 0:
        return CheckResult(name="ruff", status=Status.PASS, duration_s=elapsed)
    return CheckResult(
        name="ruff",
        status=Status.FAIL,
        duration_s=elapsed,
        detail=(stdout or stderr)[-2000:],
    )


def check_imports() -> CheckResult:
    """Verify critical modules import cleanly (catches syntax errors fast)."""
    t0 = time.perf_counter()
    py = python_executable()
    code = (
        "import vllm_mlx; "
        "from vllm_mlx import cli, server, scheduler; "
        "from vllm_mlx.engine import simple, batched, hybrid; "
        "from vllm_mlx.agents import list_profiles; "
        "print(f'modules: {len(list_profiles())} agent profiles')"
    )
    rc, stdout, stderr = run_subprocess([py, "-c", code], timeout=60)
    elapsed = time.perf_counter() - t0
    if rc == 0:
        return CheckResult(
            name="imports",
            status=Status.PASS,
            duration_s=elapsed,
            detail=stdout.strip(),
        )
    return CheckResult(
        name="imports",
        status=Status.FAIL,
        duration_s=elapsed,
        detail=stderr[-1000:] or stdout[-1000:],
    )


def check_cli_sanity() -> CheckResult:
    """Smoke the CLI by invoking the no-side-effect subcommands."""
    t0 = time.perf_counter()
    py = python_executable()
    sub_cmds = [
        [py, "-m", "vllm_mlx.cli", "--help"],
        [py, "-m", "vllm_mlx.cli", "models"],
        [py, "-m", "vllm_mlx.cli", "agents"],
    ]
    failures: list[str] = []
    for sub in sub_cmds:
        rc, stdout, stderr = run_subprocess(sub, timeout=30)
        if rc != 0:
            failures.append(f"{' '.join(sub[-2:])} → rc={rc}: {stderr[-200:]}")
    elapsed = time.perf_counter() - t0
    if not failures:
        return CheckResult(
            name="cli_sanity",
            status=Status.PASS,
            duration_s=elapsed,
            detail=f"{len(sub_cmds)} subcommands OK",
        )
    return CheckResult(
        name="cli_sanity",
        status=Status.FAIL,
        duration_s=elapsed,
        detail="\n".join(failures),
    )


def check_repo_layout() -> CheckResult:
    """Verify expected files exist (catches accidental deletions)."""
    t0 = time.perf_counter()
    required: list[Path] = [
        REPO_ROOT / "pyproject.toml",
        REPO_ROOT / "vllm_mlx" / "aliases.json",
        REPO_ROOT / "vllm_mlx" / "agents" / "profiles",
    ]
    missing = [str(p.relative_to(REPO_ROOT)) for p in required if not p.exists()]
    elapsed = time.perf_counter() - t0
    if not missing:
        return CheckResult(name="repo_layout", status=Status.PASS, duration_s=elapsed)
    return CheckResult(
        name="repo_layout",
        status=Status.FAIL,
        duration_s=elapsed,
        detail=f"missing: {', '.join(missing)}",
    )
