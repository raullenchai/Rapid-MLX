# SPDX-License-Identifier: Apache-2.0
"""Pipeline runner — owns step ordering, fail-fast policy, scorecard.

Step order is intentionally hardcoded here (not auto-discovered) so a
reviewer can grep one file to see the entire validation policy. To add
a step: write the module under ``steps/``, import it here, append to
``STEPS``.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from collections.abc import Sequence
from datetime import datetime, timezone

from .base import Step
from .context import STEP_OPT_OUT_ENV_VARS, Context, env_truthy
from .scorecard import render_scorecard, verdict
from .steps.cl_description_quality import CLDescriptionQualityStep
from .steps.codex_review import CodexReviewStep
from .steps.diff_coverage import DiffCoverageStep
from .steps.fetch import FetchStep
from .steps.full_unit import FullUnitStep
from .steps.lint import LintStep
from .steps.review_vocabulary import ReviewVocabularyStep
from .steps.sdk_env_check import SDKEnvCheckStep
from .steps.stress_e2e_bench import StressE2EBenchStep
from .steps.supply_chain import SupplyChainStep
from .steps.targeted_tests import TargetedTestsStep
from .steps.test_env_check import TestEnvCheckStep
from .steps.test_plan_check import TestPlanCheckStep

# Step order — see scripts/pr_validate/README.md for the rationale.
STAGE_NAMES = ("admit", "review", "checks", "gpu")
STAGE_STEPS: dict[str, list[Step]] = {
    "admit": [
        FetchStep(),
        TestPlanCheckStep(),
        CLDescriptionQualityStep(),
        ReviewVocabularyStep(),  # reject reviewer-steering diff text before scanning
        # ORDERING INVARIANT (#275 / codex review on #885): SupplyChainStep
        # must run BEFORE any step that auto-installs from the PR's working
        # tree (currently only TestEnvCheckStep). Otherwise an external PR
        # that adds a malicious build hook or a fake package source to
        # ``pyproject.toml`` would get its code executed inside the validator
        # venv before the supply-chain scan ever flagged the change. The
        # ``test_supply_chain_runs_before_auto_installing_steps`` invariant
        # test in ``tests/test_pr_validate_test_env.py`` pins this order so a
        # future refactor can't silently regress it.
        SupplyChainStep(),
        TestEnvCheckStep(),
        SDKEnvCheckStep(),
    ],
    "review": [CodexReviewStep()],
    "checks": [
        LintStep(),
        TargetedTestsStep(),
        FullUnitStep(),
        # Advisory patch coverage ends checks before gpu; it never gates.
        DiffCoverageStep(),
    ],
    "gpu": [StressE2EBenchStep()],
}
STEPS: list[Step] = [step for name in STAGE_NAMES for step in STAGE_STEPS[name]]
STEP_STAGE: dict[str, str] = {
    step.name: name for name, stage_steps in STAGE_STEPS.items() for step in stage_steps
}


def stage_of(name: str) -> str:
    return STEP_STAGE.get(name, "other")


# Steps that, if they fail, stop the pipeline immediately regardless of
# the user's preference (subsequent steps would either crash or waste
# CPU). Fetch failures mean we have nothing to validate. Most other
# failures still let later steps run by default so the scorecard
# surfaces the FULL picture rather than only the first bug — the user
# opts in to the "stop at first fail" behaviour with ``--fail-fast`` /
# ``PR_VALIDATE_FAIL_FAST=1`` (typical for CI / incoming-PR gating).
# Missing SDK prerequisites invalidate the matrix and stop before costly tests,
# even when the caller wants the full picture for ordinary product failures.
# Codex review runs early (~1 min): blocking findings mean the head must be
# re-pushed, so spending 5–30 min on tests would be wasted. Only "fail" carrying
# [BLOCKING] findings stops by default. Infrastructure failures (timeout,
# auth/backend error, empty or malformed reply), "error", and "skip" let tests
# provide signal. PR_VALIDATE_CONTINUE_AFTER_CODEX_FAIL=1 restores the full
# picture, while --fail-fast still stops on any fail/error.
FAIL_FAST_STEPS = {"fetch", "sdk_env_check"}
# Only failures carrying [BLOCKING] findings stop these steps by default.
STOP_ON_BLOCKING_FINDINGS_STEPS = {"codex_review"}


def run_pipeline(
    pr_number: int,
    *,
    verbose: bool = False,
    fail_fast: bool = False,
    skip_steps: Sequence[str] = (),
    steps: Sequence[Step] | None = None,
    base: str = "",
    body_only: bool = False,
    queue_on_success: bool = False,
    stage: str = "",
) -> int:
    """Execute the pipeline. Returns process exit code (0 = merge-safe).

    Strict scoring: ANY single ``fail`` or ``error`` blocks merge.
    ``skip`` is neutral (a step decided it didn't apply).

    Codex review ``fail`` carrying ``[BLOCKING]`` findings stops before tests; set
    ``PR_VALIDATE_CONTINUE_AFTER_CODEX_FAIL=1`` to continue instead.
    Infrastructure failures (timeout, auth/backend error, empty or malformed
    reply), ``error``, and ``skip`` do not trigger this default stop.

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

    ``base`` is an explicit merge-base override (``--base <sha>``). When
    set it wins over merge-base derivation in the fetch step and is used
    as the review/diff-coverage base so the reviewer sees only the PR's
    own diff.

    ``body_only`` short-circuits the pipeline to JUST the description
    quality gate (fetch + ``cl_description_quality``). Used by
    ``pr_validate --body-only <pr#>`` to judge a PR's body hygiene
    locally without burning compute on tests/codex — the description
    gate needs only the prod-fetched title/body, never the diff.
    When set, ``steps`` and ``skip_steps`` are ignored.
    """
    if stage and stage not in STAGE_NAMES:
        print(
            f"error: unknown stage {stage!r}; choose from {', '.join(STAGE_NAMES)}",
            file=sys.stderr,
        )
        return 1
    if stage and (body_only or queue_on_success):
        print(
            "error: --stage runs a partial pipeline and cannot be combined with --body-only or --queue-on-success",
            file=sys.stderr,
        )
        return 1
    if queue_on_success and (
        stage
        or body_only
        or base
        or skip_steps
        or steps is not None
        or any(env_truthy(name) for name in STEP_OPT_OUT_ENV_VARS)
    ):
        print(
            "error: --queue-on-success requires the complete default pipeline",
            file=sys.stderr,
        )
        return 1
    if body_only:
        pipeline = [FetchStep(), CLDescriptionQualityStep()]
    else:
        pipeline = STEPS if steps is None else steps
        if stage:
            selected = {"admit", stage}
            pipeline = [
                s
                for s in pipeline
                if stage_of(s.name) in selected
                and (stage == "gpu" or s.name != "sdk_env_check")
            ]
        if skip_steps:
            dropped = set(skip_steps)
            # The SDK prerequisite check belongs to the model matrix. CI's
            # documented CPU-only partial pipeline omits that matrix and does
            # not need its SDK environment. Such partial runs still cannot
            # use --queue-on-success (checked above).
            if "stress_e2e_bench" in dropped:
                dropped.add("sdk_env_check")
            pipeline = [s for s in pipeline if s.name not in dropped]

    ctx = Context(pr_number=pr_number, verbose=verbose, base_override=base)
    if base:
        # Validate an explicit ``--base <sha>`` fails fast and clearly: a bad
        # SHA silently degrades review/diff-coverage to whatever the fallback
        # resolves, which is exactly the noise the override exists to avoid.
        probe = subprocess.run(
            ["git", "cat-file", "-e", base],
            capture_output=True,
            text=True,
            cwd=str(ctx.repo_root),
        )
        if probe.returncode != 0:
            print(
                "error: --base is not an object Git can resolve: "
                f"{base!r}\n  -> confirm the SHA exists in this clone "
                "(git cat-file -e <sha>), or drop --base and let the "
                "fetch step derive the exact merge-base automatically.",
                file=sys.stderr,
            )
            return 1
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
    if stage:
        print(f"  stage: {stage}", file=sys.stderr)
    if fail_fast:
        print("  fail-fast: ON", file=sys.stderr)
    print("", file=sys.stderr)

    for step in pipeline:
        print(
            f"## [{step.name}] {step.description} (stage: {stage_of(step.name)})",
            file=sys.stderr,
        )
        result = step.execute(ctx)
        result.stage = stage_of(step.name)
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
        if (
            result.status == "fail"
            and step.name in STOP_ON_BLOCKING_FINDINGS_STEPS
            and any(f.startswith("[BLOCKING]") for f in result.findings)
            and not env_truthy("PR_VALIDATE_CONTINUE_AFTER_CODEX_FAIL")
        ):
            print(
                f"  fail-fast: [{step.name}] found blocking issues; stopping before tests "
                "(fix and re-push; set PR_VALIDATE_CONTINUE_AFTER_CODEX_FAIL=1 "
                "to run the remaining steps anyway)",
                file=sys.stderr,
            )
            break

    # ``--body-only`` exists to reproduce the description-quality verdict
    # locally. If that single gate got SKIPPED (via
    # ``PR_VALIDATE_SKIP_DESC=1``) the run validated nothing — a silent
    # MERGE-SAFE here would mislead the author into thinking their body is
    # fine. Surface it as a loud WARNING (non-blocking, the user asked for
    # the override) rather than a clean pass.
    if body_only and env_truthy("PR_VALIDATE_SKIP_DESC"):
        desc_result = next(
            (r for r in ctx.results if r.name == "cl_description_quality"), None
        )
        if desc_result is not None and desc_result.status == "skip":
            print(
                "  WARNING: --body-only ran but the description gate was "
                "SKIPPED via PR_VALIDATE_SKIP_DESC=1 — this run validated "
                "NOTHING about the body. Unset that env var and re-run to "
                "get a real description verdict.",
                file=sys.stderr,
            )

    # Render the scorecard to stdout (so callers can pipe into PR comments).
    print(render_scorecard(ctx, partial_stage=stage))

    final_verdict = verdict(ctx.results)
    display_verdict = final_verdict
    if stage and final_verdict == "MERGE-SAFE":
        display_verdict = (
            f"STAGE {stage} PASSED (partial run — not a merge verdict; "
            "run without --stage for MERGE-SAFE)"
        )
    print(f"\nVerdict: {display_verdict}", file=sys.stderr)

    if queue_on_success and final_verdict == "MERGE-SAFE":
        from .queue_ready import queue_validated_head

        try:
            queue_validated_head(ctx, [step.name for step in STEPS])
        except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as error:
            print(f"Queue handoff stopped: {error}", file=sys.stderr)
            return 1

    # Exit code: 0 only if every step is pass-or-skip (strict).
    return 0 if final_verdict == "MERGE-SAFE" else 1
