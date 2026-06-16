# SPDX-License-Identifier: Apache-2.0
"""Check tier — agent profile API tests against a live server."""

from __future__ import annotations

import time

from ...doctor.runner import CheckResult, Status


def check_agent_profile(
    profile_name: str,
    port: int,
    model_id: str | None = None,
) -> CheckResult:
    """Run one agent profile's auto-generated test plan.

    Uses ``vllm_mlx.agents.testing.AgentTestRunner`` which derives the
    test plan from the profile's capability declarations.  E2E tests
    that require the agent binary on disk are skipped automatically
    when the binary isn't installed, so this works headless.
    """
    t0 = time.perf_counter()
    try:
        # Three dots: vllm_mlx/bench/tiers/ → vllm_mlx/agents/
        from ...agents import get_profile
        from ...agents.testing import AgentTestRunner, TestStatus
    except ImportError as e:
        return CheckResult(
            name=f"agent_{profile_name}",
            status=Status.SKIP,
            duration_s=time.perf_counter() - t0,
            detail=f"agent framework unavailable: {e}",
        )

    profile = get_profile(profile_name)
    if profile is None:
        return CheckResult(
            name=f"agent_{profile_name}",
            status=Status.SKIP,
            duration_s=time.perf_counter() - t0,
            detail=f"no profile named {profile_name!r}",
        )

    base_url = f"http://127.0.0.1:{port}/v1"
    runner = AgentTestRunner(profile, base_url=base_url, model_id=model_id)
    report = runner.run()
    elapsed = time.perf_counter() - t0

    n_pass = report.passed
    n_fail = report.failed
    n_err = report.errored
    n_skip = report.skipped
    detail = (
        f"{n_pass} pass, {n_fail} fail, {n_err} error, {n_skip} skip "
        f"({len(report.results)} total)"
    )

    # Treat ERROR same as FAIL — both indicate something the doctor
    # would want to surface.  SKIP is fine (capability not declared).
    if n_fail > 0 or n_err > 0:
        # Include the failing test names so the report is actionable
        # without needing to chase down the per-profile log.
        bad = [
            f"{r.name}: {r.message[:120]}"
            for r in report.results
            if r.status in (TestStatus.FAIL, TestStatus.ERROR)
        ]
        detail += "\n  failures:\n    " + "\n    ".join(bad[:5])
        return CheckResult(
            name=f"agent_{profile_name}",
            status=Status.FAIL,
            duration_s=elapsed,
            detail=detail,
        )

    return CheckResult(
        name=f"agent_{profile_name}",
        status=Status.PASS,
        duration_s=elapsed,
        detail=detail,
    )
