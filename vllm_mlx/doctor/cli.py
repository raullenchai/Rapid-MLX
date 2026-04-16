# SPDX-License-Identifier: Apache-2.0
"""CLI entry points for ``rapid-mlx doctor``."""

from __future__ import annotations

import sys

from .baseline import (
    DeltaStatus,
    compare,
    has_regression,
    load_baseline,
    load_thresholds,
    render_deltas_md,
    save_baseline,
)
from .runner import REPO_ROOT, CheckResult, DoctorRunner, Status

# Default model used by the check tier.  Tier 3 (full) loops a wider list.
DEFAULT_CHECK_MODEL = "qwen3.5-4b"


def _require_source_checkout() -> None:
    """Doctor depends on tests/ + harness/ + pyproject.toml."""
    sentinels = [
        REPO_ROOT / "pyproject.toml",
        REPO_ROOT / "tests",
        REPO_ROOT / "harness",
    ]
    missing = [str(p.relative_to(REPO_ROOT)) for p in sentinels if not p.exists()]
    if missing:
        print(
            "[doctor] this command requires a source checkout of Rapid-MLX.\n"
            f"        missing: {', '.join(missing)}\n"
            "        Clone https://github.com/raullenchai/Rapid-MLX and run "
            "from the repo root.",
            file=sys.stderr,
        )
        sys.exit(2)


def doctor_command(args) -> None:
    """Dispatch to the requested tier."""
    _require_source_checkout()

    tier = getattr(args, "tier", None) or "smoke"
    update_baselines = getattr(args, "update_baselines", False)

    if tier == "smoke":
        result = run_smoke_tier()
    elif tier == "check":
        result = run_check_tier(
            model=getattr(args, "model", None) or DEFAULT_CHECK_MODEL,
            update_baselines=update_baselines,
        )
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


# ---------------------------------------------------------------------
# Smoke tier
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# Check tier
# ---------------------------------------------------------------------

def run_check_tier(model: str, update_baselines: bool = False):
    """Boot a server with ``model`` and run API + perf + agent checks."""
    from .checks import api, perf, smoke
    from .server import ServerStartFailed, serve

    print(f"Rapid-MLX Doctor — check tier (model={model})")
    print("=" * 60)

    runner = DoctorRunner(tier="check")

    # Cheap static checks first — fail fast on broken syntax / missing files.
    runner.run_check("repo_layout", smoke.check_repo_layout)
    runner.run_check("imports", smoke.check_imports)

    server_log = runner.run_dir / "server.log"
    try:
        with serve(model=model, log_path=server_log) as info:
            port = info["port"]
            print(f"  [server] up on port {port}, log → {server_log.name}")
            runner.run_check(
                "regression_suite",
                lambda: api.check_regression_suite(port),
            )
            runner.run_check(
                "smoke_matrix",
                lambda: api.check_smoke_matrix(port),
            )
            perf_result = runner.run_check(
                "autoresearch",
                lambda: perf.check_autoresearch(port, runs=1),
            )
    except ServerStartFailed as exc:
        # Capture the exception locally so the closure below sees it
        # even if Python clears `exc` at the end of the except block.
        err_msg = f"{exc}\nlog: {server_log}"
        runner.run_check(
            "server_boot",
            lambda: CheckResult(
                name="server_boot",
                status=Status.FAIL,
                duration_s=0.0,
                detail=err_msg,
            ),
        )
        return runner.finalize()

    # Compare perf metrics against baseline (if one exists).
    if perf_result.metrics:
        _apply_baseline(runner, "check", model, perf_result.metrics, update_baselines)

    return runner.finalize()


def _apply_baseline(
    runner: DoctorRunner,
    tier: str,
    model: str,
    metrics: dict,
    update_baselines: bool,
) -> None:
    """Diff against baseline; flag regressions; optionally update baseline."""
    baseline = load_baseline(tier)
    thresholds = load_thresholds()

    if update_baselines:
        path = save_baseline(tier, model, metrics)
        runner.run_check(
            "baseline_update",
            lambda: CheckResult(
                name="baseline_update",
                status=Status.PASS,
                duration_s=0.0,
                detail=f"wrote {path.relative_to(REPO_ROOT)}",
            ),
        )
        return

    if baseline is None:
        # First run with no baseline — record what we saw, don't fail.
        deltas_md = "_no baseline yet — run with --update-baselines to record one_\n"
        (runner.run_dir / "diff.md").write_text(deltas_md)
        runner.run_check(
            "baseline_diff",
            lambda: CheckResult(
                name="baseline_diff",
                status=Status.SKIP,
                duration_s=0.0,
                detail="no baseline found; run --update-baselines to create",
            ),
        )
        return

    deltas = compare(metrics, baseline, thresholds)
    deltas_md = render_deltas_md(deltas)
    (runner.run_dir / "diff.md").write_text(deltas_md)

    n_regress = sum(1 for d in deltas if d.status == DeltaStatus.REGRESSION)
    n_improve = sum(1 for d in deltas if d.status == DeltaStatus.IMPROVEMENT)
    detail = f"{len(deltas)} metrics: {n_regress} regression(s), {n_improve} improvement(s)"

    status = Status.REGRESSION if has_regression(deltas) else Status.PASS
    runner.run_check(
        "baseline_diff",
        lambda: CheckResult(
            name="baseline_diff",
            status=status,
            duration_s=0.0,
            detail=detail,
        ),
    )

    # Append delta table to the report so the summary is self-contained.
    runner.checks[-1].detail += f"\n\n{deltas_md}"
