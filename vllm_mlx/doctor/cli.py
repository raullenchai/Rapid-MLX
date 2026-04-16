# SPDX-License-Identifier: Apache-2.0
"""CLI entry points for ``rapid-mlx doctor``."""

from __future__ import annotations

import sys

from .runner import DoctorRunner


def doctor_command(args) -> None:
    """Dispatch to the requested tier."""
    tier = getattr(args, "tier", None) or "smoke"

    if tier == "smoke":
        result = run_smoke_tier()
    elif tier == "check":
        print("[doctor] check tier not yet implemented", file=sys.stderr)
        sys.exit(2)
    elif tier == "full":
        print("[doctor] full tier not yet implemented", file=sys.stderr)
        sys.exit(2)
    elif tier == "benchmark":
        print("[doctor] benchmark tier not yet implemented", file=sys.stderr)
        sys.exit(2)
    else:
        print(f"[doctor] unknown tier: {tier}", file=sys.stderr)
        sys.exit(2)

    sys.exit(result.exit_code)


def run_smoke_tier():
    """Static + import + CLI sanity. No model required."""
    from .checks import smoke

    print("Rapid-MLX Doctor — smoke tier")
    print("=" * 60)

    runner = DoctorRunner(tier="smoke")
    runner.run_check("repo_layout", smoke.check_repo_layout)
    runner.run_check("imports", smoke.check_imports)
    runner.run_check("ruff", smoke.check_ruff)
    runner.run_check("cli_sanity", smoke.check_cli_sanity)
    runner.run_check("pytest", smoke.check_pytest_unit)
    return runner.finalize()
