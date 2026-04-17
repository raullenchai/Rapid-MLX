# SPDX-License-Identifier: Apache-2.0
"""
Doctor runner — orchestrates check tiers and writes per-run reports.

Each check is a small callable returning ``CheckResult``.  Tiers are just
ordered lists of checks plus optional model/server requirements.  Keeping
the framework dumb makes it easy to add new checks: drop a function into
``checks/`` and append it to a tier.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

# Repo root resolved relative to this file: vllm_mlx/doctor/runner.py → repo
REPO_ROOT = Path(__file__).resolve().parents[2]
HARNESS_DIR = REPO_ROOT / "harness"
RUNS_DIR = HARNESS_DIR / "runs"


class Status(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"
    REGRESSION = "regression"


@dataclass
class CheckResult:
    """Result of one named check inside a tier."""

    name: str
    status: Status
    duration_s: float
    detail: str = ""
    metrics: dict = field(default_factory=dict)


@dataclass
class TierResult:
    """Aggregate result of a single tier run."""

    tier: str
    started_at: str
    duration_s: float
    checks: list[CheckResult]
    run_dir: str
    exit_code: int

    @property
    def passed(self) -> bool:
        return self.exit_code == 0


class DoctorRunner:
    """Run a tier of checks and persist the report."""

    def __init__(self, tier: str, run_dir: Path | None = None):
        self.tier = tier
        self.started = _dt.datetime.now()
        if run_dir is None:
            self.run_dir = self._reserve_run_dir()
        else:
            self.run_dir = run_dir
            self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checks: list[CheckResult] = []

    def _reserve_run_dir(self) -> Path:
        """Atomically create and return a unique run directory.

        Uses ``mkdir(exist_ok=False)`` so concurrent same-second invocations
        race on the filesystem — the loser gets ``FileExistsError`` and tries
        the next suffix.  This is the only way to be safe against multiple
        tiers launched in parallel (CI matrix, user kicking off two tiers
        in two terminals, etc.).
        """
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        ts = self.started.strftime("%Y-%m-%d-%H%M%S")
        for i in range(0, 1000):
            suffix = "" if i == 0 else f"-{i}"
            candidate = RUNS_DIR / f"{ts}-{self.tier}{suffix}"
            try:
                candidate.mkdir(parents=False, exist_ok=False)
                return candidate
            except FileExistsError:
                continue
        # Practically unreachable — 1000 collisions in one second.
        raise RuntimeError(
            f"Could not reserve a unique run directory under {RUNS_DIR} "
            f"after 1000 attempts at {ts}"
        )

    # ------------------------------------------------------------------
    # Check execution
    # ------------------------------------------------------------------
    def run_check(self, name: str, fn: Callable[[], CheckResult]) -> CheckResult:
        """Execute a single check, capture timing, append to results.

        ``fn`` must construct and return its own CheckResult.  Catching
        broad ``Exception`` here is intentional: a check that crashes
        should not abort the entire tier — we record the failure and
        continue so the user sees the full picture in the report.

        The caller-supplied ``name`` always wins over ``result.name``.
        That matters for tiers that run the same check fn multiple
        times (e.g. full tier across 3 models), where the report would
        otherwise collapse entries.
        """
        print(f"  [{name}]", end=" ", flush=True)
        t0 = time.perf_counter()
        try:
            result = fn()
        except Exception as e:  # noqa: BLE001 — see docstring above
            elapsed = time.perf_counter() - t0
            result = CheckResult(
                name=name,
                status=Status.FAIL,
                duration_s=elapsed,
                detail=f"check raised: {type(e).__name__}: {e}",
            )
            logger.exception("Doctor check %s crashed", name)

        # Caller's name wins — see docstring.
        result.name = name

        # Sanity: even a passing check should report a positive duration
        if result.duration_s == 0.0:
            result.duration_s = time.perf_counter() - t0

        self.checks.append(result)
        symbol = {
            Status.PASS: "OK",
            Status.FAIL: "FAIL",
            Status.SKIP: "SKIP",
            Status.REGRESSION: "REGRESSION",
        }[result.status]
        print(f"{symbol} ({result.duration_s:.1f}s)")
        if result.detail and result.status != Status.PASS:
            for line in result.detail.splitlines():
                print(f"      {line}")
        return result

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def finalize(self) -> TierResult:
        """Write the report and return the tier-level aggregate."""
        elapsed = (_dt.datetime.now() - self.started).total_seconds()
        exit_code = self._compute_exit_code()

        result = TierResult(
            tier=self.tier,
            started_at=self.started.isoformat(timespec="seconds"),
            duration_s=elapsed,
            checks=self.checks,
            run_dir=str(self.run_dir),
            exit_code=exit_code,
        )

        # Persist machine-readable + human-readable artefacts.
        (self.run_dir / "result.json").write_text(
            json.dumps(asdict(result), indent=2, default=str)
        )
        (self.run_dir / "report.md").write_text(self._render_markdown(result))

        self._print_summary(result)
        return result

    def _compute_exit_code(self) -> int:
        """0 = all pass, 1 = regression(s), 2 = functional failure(s)."""
        if any(c.status == Status.FAIL for c in self.checks):
            return 2
        if any(c.status == Status.REGRESSION for c in self.checks):
            return 1
        return 0

    def _render_markdown(self, result: TierResult) -> str:
        lines = [
            f"# Doctor Report — `{result.tier}`",
            "",
            f"- Started: {result.started_at}",
            f"- Duration: {result.duration_s:.1f}s",
            f"- Exit code: {result.exit_code}",
            "",
            "## Checks",
            "",
            "| Check | Status | Duration | Detail |",
            "| --- | --- | --- | --- |",
        ]
        for c in result.checks:
            lines.append(
                f"| {md_cell(c.name)} | {c.status.value} | {c.duration_s:.1f}s | "
                f"{md_cell(c.detail, max_len=120)} |"
            )

        # Per-model baseline-diff tables (if any) — these are stashed by
        # _apply_baseline because they're too long to fit in a check's
        # detail cell.  Render as a separate section so report.md is
        # self-contained without needing to read diff.md alongside it.
        diff_sections = getattr(self, "_pending_diff_sections", [])
        if diff_sections:
            lines += ["", "## Baseline diffs", ""]
            for model, deltas_md in diff_sections:
                lines.append(f"### {model}")
                lines.append("")
                lines.append(deltas_md.rstrip())
                lines.append("")
        return "\n".join(lines) + "\n"

    def _print_summary(self, result: TierResult) -> None:
        n_pass = sum(1 for c in result.checks if c.status == Status.PASS)
        n_fail = sum(1 for c in result.checks if c.status == Status.FAIL)
        n_regress = sum(1 for c in result.checks if c.status == Status.REGRESSION)
        n_skip = sum(1 for c in result.checks if c.status == Status.SKIP)

        print()
        print("─" * 60)
        verdict = {0: "PASS", 1: "REGRESSION", 2: "FAIL"}[result.exit_code]
        print(
            f"Result: {verdict}  "
            f"({n_pass} pass, {n_regress} regression, {n_fail} fail, {n_skip} skip)"
        )
        print(f"Report: {self.run_dir / 'report.md'}")


def md_cell(s: str, max_len: int = 0) -> str:
    """Sanitise a string for safe inclusion in a markdown table cell.

    Replaces newlines with spaces and escapes ``|`` so the table layout
    is preserved.  Optionally truncates to ``max_len`` characters with
    a "..." suffix; ``max_len=0`` (default) leaves the cell uncapped.

    Used by both ``runner.py`` (per-check rows) and ``scorecard.py``
    (per-model rows) so escaping rules stay consistent.
    """
    s = (s or "").replace("\n", " ").replace("|", "\\|")
    if max_len and len(s) > max_len:
        s = s[: max_len - 3] + "..."
    return s


# ----------------------------------------------------------------------
# Helpers shared by tier implementations
# ----------------------------------------------------------------------


def run_subprocess(
    cmd: list[str],
    cwd: Path | None = None,
    timeout: int = 600,
    env: dict | None = None,
) -> tuple[int, str, str]:
    """Thin wrapper around subprocess.run that always captures output.

    Returns ``(returncode, stdout, stderr)``.  On TimeoutExpired we return a
    sentinel returncode of 124 so callers can treat it uniformly.
    """
    try:
        proc = subprocess.run(  # noqa: S603 — args are constructed by us
            cmd,
            cwd=cwd or REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired as e:
        stdout = e.stdout or ""
        stderr = e.stderr or ""
        # TimeoutExpired may return bytes even with text=True
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        return 124, stdout, f"TIMEOUT after {timeout}s\n{stderr}"
    return proc.returncode, proc.stdout, proc.stderr


def python_executable() -> str:
    """Prefer the same interpreter we are running under, fall back to system python3.12."""
    if sys.executable and Path(sys.executable).exists():
        return sys.executable
    py = shutil.which("python3.12") or shutil.which("python3") or "python"
    return py
