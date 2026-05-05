# SPDX-License-Identifier: Apache-2.0
"""``python3.12 -m scripts.pr_validate <PR#>`` entry point."""

from __future__ import annotations

import argparse
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
    args = parser.parse_args(argv)
    fail_fast = args.fail_fast or env_truthy("PR_VALIDATE_FAIL_FAST")
    return run_pipeline(args.pr_number, verbose=args.verbose, fail_fast=fail_fast)


if __name__ == "__main__":
    sys.exit(main())
