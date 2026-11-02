# SPDX-License-Identifier: Apache-2.0
"""``python3.12 -m scripts.pr_validate <PR#>`` entry point."""

from __future__ import annotations

import argparse
import os
import signal
import sys

from .context import env_truthy
from .runner import STAGE_NAMES, _terminate_child_processes, run_pipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pr_validate",
        description="Run the merge-readiness pipeline against a PR.",
    )
    parser.add_argument(
        "pr_number",
        type=int,
        help="GitHub PR number (e.g. 200)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print step output as it runs",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Compatibility no-op; fail-fast is enabled by default.",
    )
    parser.add_argument(
        "--no-fail-fast",
        action="store_true",
        help="Run every lane and step to completion for the full picture; gpu still runs after failures.",
    )
    parser.add_argument(
        "--skip-steps",
        default="",
        help=(
            "Comma-separated list of step names to drop from the pipeline. "
            "Used by CI to skip steps that need a live ``rapid-mlx serve`` "
            "(e.g. ``--skip-steps stress_e2e_bench``) since GitHub-hosted "
            "runners can't host real model inference. Also accepts the env "
            "var ``PR_VALIDATE_SKIP_STEPS`` for the same purpose. Unknown "
            "names are silently ignored."
        ),
    )
    parser.add_argument(
        "--base",
        default="",
        metavar="SHA",
        help=(
            "Explicit merge-base SHA to review/diff-coverage against. "
            "Overrides automatic merge-base derivation, so the reviewer "
            "sees ONLY this PR's diff even when the base branch tip has "
            "moved past the point where the PR branched. When absent the "
            "fetch step derives the exact git merge-base automatically."
        ),
    )
    parser.add_argument(
        "--body-only",
        action="store_true",
        help=(
            "Run ONLY the description-quality gate (fetch + "
            "cl_description_quality) and reproduce its verdict locally — no "
            "tests, no codex, no diff review. Use to check a PR's title/body "
            "hygiene cheaply (e.g. a fresh PR from the template should pass "
            "with no edits)."
        ),
    )
    parser.add_argument(
        "--stage",
        choices=STAGE_NAMES,
        default="",
        help=(
            "Run only that stage; review, checks, and gpu also re-run the cheap "
            "admit stage first for context and the supply-chain scan."
        ),
    )
    parser.add_argument(
        "--queue-on-success",
        action="store_true",
        help=(
            "After the complete local pipeline passes, wait for exact-head "
            "hosted gates, authorize readiness and request normal queue once. "
            "Never retries a previously queued PR; incompatible with partial runs."
        ),
    )
    parser.add_argument(
        "--local-full-suite",
        action="store_true",
        help="also run the full local unit suite and the advisory diff coverage; for nightly runs — the default relies on hosted CI for these",
    )
    args = parser.parse_args(argv)
    fail_fast = not args.no_fail_fast and os.environ.get(
        "PR_VALIDATE_FAIL_FAST", ""
    ).strip().lower() not in {"0", "false", "no", "off"}
    skip_raw = args.skip_steps or os.environ.get("PR_VALIDATE_SKIP_STEPS", "")
    skip_steps = tuple(s.strip() for s in skip_raw.split(",") if s.strip())

    def terminate(signum, frame):
        _terminate_child_processes(grace_seconds=0)
        raise SystemExit(128 + signum)

    previous = {}
    try:
        for signum in (signal.SIGTERM, signal.SIGHUP):
            previous[signum] = signal.signal(signum, terminate)
        return run_pipeline(
            args.pr_number,
            verbose=args.verbose,
            fail_fast=fail_fast,
            skip_steps=skip_steps,
            base=args.base,
            body_only=args.body_only,
            queue_on_success=args.queue_on_success,
            stage=args.stage,
            local_full_suite=args.local_full_suite
            or env_truthy("PR_VALIDATE_LOCAL_FULL_SUITE"),
        )
    except KeyboardInterrupt:
        print("Validation interrupted (Ctrl-C).", file=sys.stderr)
        return 130
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    sys.exit(main())
