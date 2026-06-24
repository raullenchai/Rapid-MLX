# SPDX-License-Identifier: Apache-2.0
"""Pipeline runner — owns step ordering, fail-fast policy, scorecard.

Step order is intentionally hardcoded here (not auto-discovered) so a
reviewer can grep one file to see the entire validation policy. To add
a step: write the module under ``steps/``, import it here, append to
``STEPS``.
"""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Sequence
from datetime import datetime, timezone

from .base import Step
from .context import Context
from .scorecard import render_scorecard, verdict
from .steps.cl_description_quality import CLDescriptionQualityStep
from .steps.codex_review import CodexReviewStep
from .steps.fetch import FetchStep
from .steps.full_unit import FullUnitStep
from .steps.lint import LintStep
from .steps.stress_e2e_bench import StressE2EBenchStep
from .steps.supply_chain import SupplyChainStep
from .steps.targeted_tests import TargetedTestsStep
from .steps.test_env_check import TestEnvCheckStep
from .steps.test_plan_check import TestPlanCheckStep

# Step order — see scripts/pr_validate/README.md for the rationale.
# Codex review goes early so cheap critical thinking happens before
# we spend 10 minutes on tests.
#
# ORDERING INVARIANT (#275 / codex review on #885): SupplyChainStep
# must run BEFORE any step that auto-installs from the PR's working
# tree (currently only TestEnvCheckStep). Otherwise an external PR
# that adds a malicious build hook or a fake package source to
# ``pyproject.toml`` would get its code executed inside the validator
# venv before the supply-chain scan ever flagged the change. The
# ``test_supply_chain_runs_before_auto_installing_steps`` invariant
# test in ``tests/test_pr_validate_runner.py`` pins this order so a
# future refactor can't silently regress it.
STEPS: list[Step] = [
    FetchStep(),  # 0 — fetch PR + diff + classify blast radius
    TestPlanCheckStep(),  # 0.5 — unchecked test-plan items block merge (#427 lesson)
    CLDescriptionQualityStep(),  # 0.7 — title + body rationale (Google eng-practices)
    # 0.75 — supply-chain scan must run BEFORE any step that might
    # ``pip install`` from the PR's working tree. Otherwise a
    # malicious build hook / fake package source in pyproject.toml
    # would execute inside the validator venv before the gate that
    # was supposed to flag it. See #275.
    SupplyChainStep(),
    # 0.8 — verify pytest plugins importable in the same Python pytest
    # will be invoked with. Closes #185 (was failing the targeted_tests
    # + full_unit gates with cryptic "async def functions are not
    # natively supported" pytest errors when pytest-asyncio fell out of
    # the host env). MUST run AFTER ``supply_chain`` because this step
    # may ``pip install '.[test]'`` to auto-recover a missing plugin,
    # and that install would execute attacker-controlled code from a
    # tampered pyproject.toml. The auto-install path is also disabled
    # when the diff touches dep-declaration files — see ``_test_env.py``
    # for the gate.
    TestEnvCheckStep(),
    CodexReviewStep(),  # 6 — adversarial review (codex exec, gpt-5.5)
    LintStep(),  # 2 — ruff check + format
    TargetedTestsStep(),  # 3 — diff-aware test selection + neg control
    FullUnitStep(),  # 4 — full pytest, gated on blast radius
    StressE2EBenchStep(),  # 5 — stress + e2e + bench (multi-model × agents)
]

# Steps that, if they fail, stop the pipeline immediately regardless of
# the user's preference (subsequent steps would either crash or waste
# CPU). Fetch failures mean we have nothing to validate. Most other
# failures still let later steps run by default so the scorecard
# surfaces the FULL picture rather than only the first bug — the user
# opts in to the "stop at first fail" behaviour with ``--fail-fast`` /
# ``PR_VALIDATE_FAIL_FAST=1`` (typical for CI / incoming-PR gating).
FAIL_FAST_STEPS = {"fetch"}


def run_pipeline(
    pr_number: int,
    *,
    verbose: bool = False,
    fail_fast: bool = False,
    skip_steps: Sequence[str] = (),
    steps: Sequence[Step] | None = None,
) -> int:
    """Execute the pipeline. Returns process exit code (0 = merge-safe).

    Strict scoring: ANY single ``fail`` or ``error`` blocks merge.
    ``skip`` is neutral (a step decided it didn't apply).

    With ``fail_fast=True`` the pipeline stops at the first ``fail`` /
    ``error`` after fetch — useful for CI gating where running the
    expensive stress/bench step on a PR that already failed lint or
    the codex review is just wasted compute.

    ``skip_steps`` is the list of step names to drop from the pipeline
    entirely (e.g. ``("stress_e2e_bench",)`` in CI where no MLX
    runtime is available to boot a server). Unknown names are silently
    ignored so a typo doesn't surprise-fail the run; the rendered
    scorecard lists which steps actually ran.

    ``steps`` is an injection seam for tests; production callers leave
    it ``None`` and the module-level ``STEPS`` list is used.
    """
    pipeline = STEPS if steps is None else steps
    if skip_steps:
        dropped = set(skip_steps)
        pipeline = [s for s in pipeline if s.name not in dropped]

    ctx = Context(pr_number=pr_number, verbose=verbose)
    run_id = (
        f"run-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-"
        f"{os.getpid()}-{time.time_ns()}"
    )
    # Use a unique run directory instead of reusing / deleting pr-<n>.
    # Reuse leaves stale failure logs; deletion can corrupt a concurrent run.
    ctx.work_dir = ctx.work_dir / f"pr-{pr_number}" / run_id
    ctx.work_dir.mkdir(parents=True, exist_ok=True)

    print(f"# PR #{pr_number} validation", file=sys.stderr)
    print(f"  artifacts → {ctx.work_dir}", file=sys.stderr)
    if fail_fast:
        print("  fail-fast: ON", file=sys.stderr)
    print("", file=sys.stderr)

    for step in pipeline:
        print(f"## [{step.name}] {step.description}", file=sys.stderr)
        result = step.execute(ctx)
        ctx.results.append(result)

        marker = {
            "pass": "OK",
            "fail": "FAIL",
            "skip": "skip",
            "error": "ERROR",
        }[result.status]
        print(
            f"  → {marker:6s} {result.summary} ({result.duration_seconds:.1f}s)",
            file=sys.stderr,
        )
        print("", file=sys.stderr)

        is_blocking = result.status in ("fail", "error")
        if is_blocking and step.name in FAIL_FAST_STEPS:
            print(
                f"  fail-fast: [{step.name}] is critical, stopping pipeline",
                file=sys.stderr,
            )
            break
        if is_blocking and fail_fast:
            print(
                f"  fail-fast: [{step.name}] failed and --fail-fast is on, "
                "stopping pipeline (subsequent steps not run)",
                file=sys.stderr,
            )
            break

    # Render the scorecard to stdout (so callers can pipe into PR comments).
    print(render_scorecard(ctx))

    final_verdict = verdict(ctx.results)
    print(f"\nVerdict: {final_verdict}", file=sys.stderr)

    # Exit code: 0 only if every step is pass-or-skip (strict).
    return 0 if final_verdict == "MERGE-SAFE" else 1
