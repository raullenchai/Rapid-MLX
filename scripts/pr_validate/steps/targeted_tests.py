# SPDX-License-Identifier: Apache-2.0
"""Step 3 — diff-aware targeted test selection with negative control.

Goal: run the tests most likely to catch a regression from this PR
without running the entire ~25s suite. Plus the negative-control move
we used in PR #187: any failure on the PR branch is re-checked on main
to filter pre-existing flakes (so we don't block merge on something
that was already broken).

Selection heuristic — deliberately simple, grep-able:

1. For each Python file in the diff, derive the candidate test file
   name(s):
   - ``vllm_mlx/foo.py`` → ``tests/test_foo.py``
   - ``vllm_mlx/bar/baz.py`` → ``tests/test_baz.py``
2. For each non-test Python file, also include any test file whose
   name contains the module's stem.
3. If the diff hits a test file directly, include it.
4. If the heuristic matches nothing (e.g. PR only touches docs), skip
   the step.

We don't import-graph trace because pytest's collection cost dominates
and the heuristic catches >90% of the cases that matter. The full unit
suite (step 4) covers the rest for medium/high blast PRs.

Negative control: when targeted tests fail on the PR branch, re-run
the same set on a fresh ``git worktree`` of ``main``. Tests that fail
on both → pre-existing → don't block. Tests that pass on main but fail
on PR → real regression → BLOCK.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from ..base import Step, StepResult
from ..context import Context


class TargetedTestsStep(Step):
    name = "targeted_tests"
    description = "diff-aware tests + negative control"

    def should_run(self, ctx: Context) -> bool:
        # Skip if no python in the diff — nothing to target.
        return any(p.endswith(".py") for p in ctx.files_changed)

    def run(self, ctx: Context) -> StepResult:
        targets = _select_test_files(ctx)
        if not targets:
            return StepResult(
                name=self.name,
                status="skip",
                summary="diff has no python files mapping to tests",
            )

        # Cap the targeted set — if the diff is huge, prefer falling
        # through to step 4 (full unit) rather than re-implementing it
        # here. Otherwise we'd spend 30s collecting tests that step 4
        # will then re-run from scratch.
        if len(targets) > 25:
            return StepResult(
                name=self.name,
                status="skip",
                summary=(
                    f"too many test targets ({len(targets)}) — "
                    f"deferring to full_unit step"
                ),
            )

        ctx.run_log(f"running {len(targets)} targeted test file(s) on PR branch")

        # Run on the PR branch (current working tree should be the PR's
        # head — we don't enforce that here, but the caller setup should
        # ensure it).  TODO if we ever auto-checkout PRs: switch to the
        # PR head here too.
        pr_log = ctx.artifact_path("targeted-pr.log")
        pr_summary, pr_failed = _run_pytest(targets, pr_log, ctx.repo_root)

        if not pr_failed:
            return StepResult(
                name=self.name,
                status="pass",
                summary=f"{pr_summary} (in {len(targets)} target file(s))",
                artifacts=[str(pr_log)],
            )

        # Failures on PR branch — run negative control on main.
        ctx.run_log(
            f"{len(pr_failed)} fail on PR branch — running same tests "
            f"on main to filter pre-existing flakes"
        )
        main_log = ctx.artifact_path("targeted-main.log")
        try:
            # Use the PR's exact base SHA, not the branch tip — main may
            # have advanced since the PR was opened, and a moving
            # negative-control would misclassify regressions caused by
            # newer main commits as PR-introduced.
            main_failed = _run_on_main(
                targets,
                main_log,
                ctx.repo_root,
                ctx.base_sha or ctx.base_branch,
            )
        except Exception as e:  # noqa: BLE001
            # Worktree setup failed — surface it but don't lose the PR
            # failures we already have. Treat as fail-safe (block).
            details = (
                f"**negative control unavailable** ({type(e).__name__}: {e}). "
                "Cannot distinguish regressions from pre-existing flakes — "
                "treating all PR failures as regressions.\n\n"
                f"```\n{_failed_block(pr_failed)}\n```"
            )
            return StepResult(
                name=self.name,
                status="fail",
                summary=f"{len(pr_failed)} fail (neg control unavailable)",
                details=details,
                artifacts=[str(pr_log)],
            )

        # Classify each PR failure: regression vs pre-existing.
        regressions = sorted(set(pr_failed) - set(main_failed))
        pre_existing = sorted(set(pr_failed) & set(main_failed))
        only_on_main = sorted(
            set(main_failed) - set(pr_failed)
        )  # interesting but unused

        if not regressions:
            return StepResult(
                name=self.name,
                status="pass",
                summary=(
                    f"{len(pr_failed)} fail on PR — all also fail on main "
                    f"(pre-existing, not regressions)"
                ),
                details=(
                    "**Pre-existing failures (also fail on main, ignored):**\n```\n"
                    + _failed_block(pre_existing)
                    + "\n```"
                ),
                artifacts=[str(pr_log), str(main_log)],
            )

        details = ["**Regressions (fail on PR, pass on main):**", "```"]
        details.extend(regressions)
        details.append("```")
        if pre_existing:
            details.append("\n**Pre-existing (also fail on main, not blocking):**\n```")
            details.extend(pre_existing)
            details.append("```")
        return StepResult(
            name=self.name,
            status="fail",
            summary=f"{len(regressions)} regression(s), "
            f"{len(pre_existing)} pre-existing",
            details="\n".join(details),
            artifacts=[str(pr_log), str(main_log)],
        )


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def _select_test_files(ctx: Context) -> list[str]:
    """Return tests/ paths to run, deduped, sorted."""
    candidates: set[str] = set()
    tests_dir = ctx.repo_root / "tests"
    if not tests_dir.exists():
        return []

    for path in ctx.files_changed:
        if not path.endswith(".py"):
            continue
        # Direct hit on a test file.
        if path.startswith("tests/"):
            if (ctx.repo_root / path).exists():
                candidates.add(path)
            continue

        stem = Path(path).stem  # e.g. "scheduler"

        # Try the obvious test_<stem>.py first.
        direct = tests_dir / f"test_{stem}.py"
        if direct.exists():
            candidates.add(f"tests/test_{stem}.py")

        # Plus any test files whose name contains the stem (covers
        # e.g. test_prefix_cache_persistence.py for prefix_cache.py).
        for tf in tests_dir.glob("test_*.py"):
            if stem in tf.stem:
                candidates.add(f"tests/{tf.name}")

    return sorted(candidates)


# ---------------------------------------------------------------------------
# Pytest runner
# ---------------------------------------------------------------------------


# Use ``sys.executable`` so pytest runs in the *same* interpreter as
# pr_validate itself. The test_env_check step verifies pytest-asyncio
# (etc.) are importable in this Python; if we hardcoded ``python3.12``
# we could end up handing pytest to a sibling Python where the
# self-check never ran. Closes #185 — the prior ``python3.12`` literal
# meant a venv-installed pr_validate could collect 124 async-test
# failures because the system python3.12 didn't have the plugin.
_PYTEST_CMD = [
    sys.executable,
    "-m",
    "pytest",
    "-q",
    "--no-header",
    "--tb=no",  # we don't render tracebacks here; the artifact has them
]


def _run_pytest(targets: list[str], log_path: Path, cwd: Path) -> tuple[str, list[str]]:
    """Run pytest against ``targets``. Returns (one-line summary,
    list of FAILED node IDs). Empty failed list => clean run."""
    proc = subprocess.run(  # noqa: S603
        [*_PYTEST_CMD, *targets],
        capture_output=True,
        text=True,
        cwd=str(cwd),
    )
    log_path.write_text((proc.stdout or "") + (proc.stderr or ""))
    summary = _last_summary_line(proc.stdout) or f"exit {proc.returncode}"
    failed = _extract_failed_node_ids(proc.stdout)
    return summary, failed


def _run_on_main(
    targets: list[str], log_path: Path, repo_root: Path, base_ref: str
) -> list[str]:
    """Run the same targets on a fresh worktree of ``base_ref``.

    ``base_ref`` should be the PR's base SHA (preferred) — using a
    branch name lets ``main`` move under us. Falls back to a branch
    name if the SHA isn't available.

    Uses a temp directory; the caller's repo root + working tree stay
    untouched. We re-resolve targets relative to the worktree (some
    files may not exist on main, e.g. tests added by this PR — those
    are dropped from the negative control).
    """
    tmp = Path(tempfile.mkdtemp(prefix="pr_validate_main_"))
    try:
        # Create a worktree pointing at base_ref (sha-pinned when given).
        subprocess.run(  # noqa: S603
            ["git", "worktree", "add", "--detach", str(tmp), base_ref],
            check=True,
            capture_output=True,
            text=True,
            cwd=str(repo_root),
        )
        try:
            existing_targets = [t for t in targets if (tmp / t).exists()]
            if not existing_targets:
                # Every target test file is new in this PR. By
                # construction they don't exist on main → no failures
                # to filter; treat as no pre-existing fails.
                log_path.write_text(
                    "(no targeted test files exist on main — all are new in PR)\n"
                )
                return []

            proc = subprocess.run(  # noqa: S603
                [*_PYTEST_CMD, *existing_targets],
                capture_output=True,
                text=True,
                cwd=str(tmp),
            )
            log_path.write_text((proc.stdout or "") + (proc.stderr or ""))
            return _extract_failed_node_ids(proc.stdout)
        finally:
            # Remove the worktree even if pytest crashed.
            subprocess.run(  # noqa: S603
                ["git", "worktree", "remove", "--force", str(tmp)],
                capture_output=True,
                text=True,
                cwd=str(repo_root),
            )
    finally:
        # In case `git worktree remove` failed, nuke the dir.
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)


def _last_summary_line(stdout: str) -> str:
    for line in reversed((stdout or "").splitlines()):
        line = line.strip()
        if line.startswith("=") and ("passed" in line or "failed" in line):
            return line.strip("= ").strip()
    return ""


_FAIL_RE = re.compile(r"^FAILED\s+(\S+)")


def _extract_failed_node_ids(stdout: str) -> list[str]:
    """Pull the FAILED <node_id> lines from pytest's short summary."""
    out = []
    in_summary = False
    for line in (stdout or "").splitlines():
        if "short test summary" in line:
            in_summary = True
            continue
        if in_summary:
            if line.startswith("="):
                break
            m = _FAIL_RE.match(line)
            if m:
                out.append(m.group(1))
    return out


def _failed_block(items: list[str]) -> str:
    return "\n".join(items) if items else "(none)"
