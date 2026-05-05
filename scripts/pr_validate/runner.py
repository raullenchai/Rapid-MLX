# SPDX-License-Identifier: Apache-2.0
"""Pipeline runner — owns step ordering, fail-fast policy, scorecard.

Step order is intentionally hardcoded here (not auto-discovered) so a
reviewer can grep one file to see the entire validation policy. To add
a step: write the module under ``steps/``, import it here, append to
``STEPS``.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

from .base import Step
from .context import Context
from .scorecard import render_scorecard, verdict
from .steps.deepseek_review import DeepSeekReviewStep
from .steps.fetch import FetchStep
from .steps.full_unit import FullUnitStep
from .steps.lint import LintStep
from .steps.stress_e2e_bench import StressE2EBenchStep
from .steps.supply_chain import SupplyChainStep
from .steps.targeted_tests import TargetedTestsStep

# Step order — see scripts/pr_validate/README.md for the rationale.
# DeepSeek review goes early so cheap critical thinking happens before
# we spend 10 minutes on tests.
STEPS: list[Step] = [
    FetchStep(),  # 0 — fetch PR + diff + classify blast radius
    DeepSeekReviewStep(),  # 6 — adversarial review (moved to front)
    SupplyChainStep(),  # 1 — pip-audit, license, install hooks
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
    steps: Sequence[Step] | None = None,
) -> int:
    """Execute the pipeline. Returns process exit code (0 = merge-safe).

    Strict scoring: ANY single ``fail`` or ``error`` blocks merge.
    ``skip`` is neutral (a step decided it didn't apply).

    With ``fail_fast=True`` the pipeline stops at the first ``fail`` /
    ``error`` after fetch — useful for CI gating where running the
    expensive stress/bench step on a PR that already failed lint or
    DeepSeek review is just wasted compute.

    ``steps`` is an injection seam for tests; production callers leave
    it ``None`` and the module-level ``STEPS`` list is used.
    """
    pipeline = STEPS if steps is None else steps

    ctx = Context(pr_number=pr_number, verbose=verbose)
    ctx.work_dir = ctx.work_dir / f"pr-{pr_number}"
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
