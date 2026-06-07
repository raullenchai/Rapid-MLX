# SPDX-License-Identifier: Apache-2.0
"""``python3.12 -m scripts.pr_validate <PR#>`` entry point."""

from __future__ import annotations

import argparse
import os
import sys

from .context import env_truthy
from .runner import run_pipeline


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
        help=(
            "Stop at the first failing step instead of running the whole "
            "pipeline. Saves compute on PRs that fail an early check; "
            "loses the 'show me everything wrong at once' view. Also "
            "enabled by PR_VALIDATE_FAIL_FAST=1 in the environment."
        ),
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
    args = parser.parse_args(argv)
    fail_fast = args.fail_fast or env_truthy("PR_VALIDATE_FAIL_FAST")
    skip_raw = args.skip_steps or os.environ.get("PR_VALIDATE_SKIP_STEPS", "")
    skip_steps = tuple(s.strip() for s in skip_raw.split(",") if s.strip())
    return run_pipeline(
        args.pr_number,
        verbose=args.verbose,
        fail_fast=fail_fast,
        skip_steps=skip_steps,
    )


if __name__ == "__main__":
    sys.exit(main())
