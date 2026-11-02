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
import threading
import time
from collections.abc import Sequence
from datetime import datetime, timezone

import psutil

from . import _children
from .base import Step, StepResult
from .context import STEP_OPT_OUT_ENV_VARS, Context, env_truthy
from .scorecard import render_scorecard, verdict
from .steps.cl_description_quality import CLDescriptionQualityStep
from .steps.codex_review import CodexReviewStep
from .steps.diff_coverage import DiffCoverageStep
from .steps.fetch import FetchStep
from .steps.full_unit import FullUnitStep
from .steps.hosted_ci import HostedCIStep
from .steps.lint import LintStep
from .steps.mac_only_tests import MacOnlyTestsStep
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
        MacOnlyTestsStep(),
        HostedCIStep(),
    ],
    "gpu": [StressE2EBenchStep()],
}
NIGHTLY_CHECK_STEPS: list[Step] = [FullUnitStep(), DiffCoverageStep()]
STEPS: list[Step] = [step for name in STAGE_NAMES for step in STAGE_STEPS[name]]
STEP_STAGE: dict[str, str] = {
    step.name: name for name, stage_steps in STAGE_STEPS.items() for step in stage_steps
}
STEP_STAGE.update({step.name: "checks" for step in NIGHTLY_CHECK_STEPS})


def stage_of(name: str) -> str:
    return STEP_STAGE.get(name, "other")


# Prerequisite failures always stop. Blocking review findings also stop unless
# explicitly overridden; review infrastructure failures still allow checks to run.
# Other failures stop when fail-fast is enabled (the default).
FAIL_FAST_STEPS = {"fetch", "sdk_env_check"}
STOP_ON_BLOCKING_FINDINGS_STEPS = {"codex_review"}


def _stop_reason(step: Step, result: StepResult, fail_fast: bool) -> str | None:
    if result.status not in ("fail", "error"):
        return None
    if step.name in FAIL_FAST_STEPS:
        return f"[{step.name}] is critical, stopping pipeline"
    if step.name in STOP_ON_BLOCKING_FINDINGS_STEPS:
        if (
            result.status == "fail"
            and any(f.startswith("[BLOCKING]") for f in result.findings)
            and not env_truthy("PR_VALIDATE_CONTINUE_AFTER_CODEX_FAIL")
        ):
            return (
                f"[{step.name}] found blocking issues; stopping before tests "
                "(fix and re-push; set PR_VALIDATE_CONTINUE_AFTER_CODEX_FAIL=1 "
                "to run the remaining steps anyway)"
            )
        return None
    if fail_fast:
        return (
            f"[{step.name}] failed and --fail-fast is on, "
            "stopping pipeline (subsequent steps not run)"
        )
    return None


def _terminate_child_processes(grace_seconds: float = 5.0) -> None:
    bounded_leaders = _children.kill_groups()
    try:
        children = [
            child
            for child in psutil.Process().children(recursive=True)
            if child.pid not in bounded_leaders
        ]
    except (psutil.NoSuchProcess, psutil.AccessDenied, PermissionError) as error:
        print(f"  warning: cannot enumerate child processes: {error}", file=sys.stderr)
        return
    for child in children:
        try:
            child.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    _, survivors = psutil.wait_procs(children, timeout=grace_seconds)
    for child in survivors:
        try:
            child.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass


def run_pipeline(
    pr_number: int,
    *,
    verbose: bool = False,
    fail_fast: bool = True,
    skip_steps: Sequence[str] = (),
    steps: Sequence[Step] | None = None,
    base: str = "",
    body_only: bool = False,
    queue_on_success: bool = False,
    stage: str = "",
    local_full_suite: bool = False,
) -> int:
    """Execute the pipeline. Returns process exit code (0 = merge-safe).

    Strict scoring: ANY single ``fail`` or ``error`` blocks merge.
    ``skip`` is neutral (a step decided it didn't apply).

    Codex review ``fail`` carrying ``[BLOCKING]`` findings cancels checks; set
    ``PR_VALIDATE_CONTINUE_AFTER_CODEX_FAIL=1`` to continue instead.
    Infrastructure failures (timeout, auth/backend error, empty or malformed
    reply), ``error``, and ``skip`` do not trigger this default stop.

    With ``fail_fast=True`` ordinary failures stop their lane and cancel the
    other lane. Review infrastructure failures still allow checks to finish.
    Any failure prevents the subsequent GPU phase.

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
        pipeline = list(STEPS if steps is None else steps)
        if steps is None and local_full_suite:
            gpu_index = next(
                (i for i, s in enumerate(pipeline) if stage_of(s.name) == "gpu"),
                len(pipeline),
            )
            pipeline[gpu_index:gpu_index] = NIGHTLY_CHECK_STEPS
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
    print(
        "  fail-fast: ON" if fail_fast else "  fail-fast: OFF (--no-fail-fast)",
        file=sys.stderr,
    )
    print("", file=sys.stderr)

    serial = [s for s in pipeline if stage_of(s.name) in ("admit", "other")]
    lanes = {
        stage_name: [s for s in pipeline if stage_of(s.name) == stage_name]
        for stage_name in ("review", "checks")
    }
    gpu = [s for s in pipeline if stage_of(s.name) == "gpu"]
    lock = threading.Lock()
    cancel = threading.Event()
    ctx.cancel_event = cancel
    cancel_reason = ""

    def print_result(step: Step, result: StepResult) -> None:
        print(
            f"## [{step.name}] {step.description} (stage: {stage_of(step.name)})",
            file=sys.stderr,
        )
        marker = {"pass": "OK", "fail": "FAIL", "skip": "skip", "error": "ERROR"}[
            result.status
        ]
        print(
            f"  → {marker:6s} {result.summary} ({result.duration_seconds:.1f}s)",
            file=sys.stderr,
        )
        print("", file=sys.stderr)

    def run_serial(steps_to_run: Sequence[Step]) -> bool:
        for step in steps_to_run:
            if cancel.is_set():
                return False
            print(
                f"## [{step.name}] {step.description} (stage: {stage_of(step.name)})",
                file=sys.stderr,
            )
            result = step.execute(ctx)
            result.stage = stage_of(step.name)
            ctx.results.append(result)
            marker = {"pass": "OK", "fail": "FAIL", "skip": "skip", "error": "ERROR"}[
                result.status
            ]
            print(
                f"  → {marker:6s} {result.summary} ({result.duration_seconds:.1f}s)\n",
                file=sys.stderr,
            )
            reason = _stop_reason(step, result, fail_fast)
            if reason:
                print(f"  fail-fast: {reason}", file=sys.stderr)
                return False
        return True

    def run_lane(stage_name: str, lane: Sequence[Step]) -> None:
        nonlocal cancel_reason
        for step in lane:
            with lock:
                if cancel.is_set():
                    result = StepResult(
                        step.name,
                        "skip",
                        f"cancelled: {cancel_reason}",
                        stage=stage_name,
                    )
                    ctx.results.append(result)
                    print_result(step, result)
                    continue
                print(
                    f"## [{step.name}] started (stage: {stage_name})", file=sys.stderr
                )
            try:
                result = step.execute(ctx)
            except BaseException as e:
                result = StepResult(
                    step.name,
                    "error",
                    f"lane crashed: {type(e).__name__}: {e}",
                    stage=stage_name,
                )
            result.stage = stage_name
            terminate = False
            with lock:
                if cancel.is_set():
                    if result.status in {"fail", "error"}:
                        result.summary = f"after cancellation: {result.summary}"
                    else:
                        result.status = "skip"
                        result.summary = (
                            f"cancelled: {cancel_reason} (partial: {result.summary})"
                        )
                else:
                    reason = _stop_reason(step, result, fail_fast)
                    if reason:
                        cancel_reason = f"[{step.name}] failed in the {stage_name} lane"
                        cancel.set()
                        terminate = True
                        print(f"  fail-fast: {reason}", file=sys.stderr)
                ctx.results.append(result)
                print_result(step, result)
            if terminate:
                _terminate_child_processes()

    threads: list[threading.Thread] = []
    try:
        if run_serial(serial):
            threads = [
                threading.Thread(target=run_lane, args=(name, lane), daemon=True)
                for name, lane in lanes.items()
                if lane
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            if cancel.is_set() or (
                fail_fast and any(r.status in ("fail", "error") for r in ctx.results)
            ):
                print("  fail-fast: stopping before gpu", file=sys.stderr)
            else:
                run_serial(gpu)
    except BaseException:
        with lock:
            cancel_reason = "interrupted"
            cancel.set()
        _terminate_child_processes()
        deadline = time.monotonic() + 5.0
        for thread in threads:
            if thread.ident is not None:
                thread.join(max(0.0, deadline - time.monotonic()))
        raise

    order = {step.name: i for i, step in enumerate(pipeline)}
    ctx.results.sort(key=lambda result: order.get(result.name, len(pipeline)))

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
            queue_validated_head(ctx, [step.name for step in pipeline])
        except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as error:
            print(f"Queue handoff stopped: {error}", file=sys.stderr)
            return 1

    # Exit code: 0 only if every step is pass-or-skip (strict).
    return 0 if final_verdict == "MERGE-SAFE" else 1
