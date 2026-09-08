# SPDX-License-Identifier: Apache-2.0
"""Conservative Doctor repair planning and post-action verification."""

from __future__ import annotations

import os
import subprocess
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass

from .env_health import CheckStatus, Report


@dataclass(frozen=True)
class RepairAction:
    id: str
    summary: str
    command: tuple[str, ...]
    requires_root: bool = False


@dataclass(frozen=True)
class RepairResult:
    id: str
    status: str
    summary: str
    command: list[str]
    detail: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def plan_repairs(report: Report) -> list[RepairAction]:
    """Return only repairs with a bounded action and objective verifier."""
    checks = {check.id: check for check in report.all_checks()}
    registration = checks.get("service.registration")
    process = checks.get("service.process")
    liveness = checks.get("service.endpoint.liveness")
    unhealthy = any(
        check is not None and check.status is CheckStatus.FAIL
        for check in (process, liveness)
    )
    if unhealthy and registration is not None and registration.status is CheckStatus.OK:
        return [
            RepairAction(
                id="repair.service.kickstart",
                summary="restart the registered Always-on service",
                command=(
                    "/bin/launchctl",
                    "kickstart",
                    "-k",
                    "system/com.rapidmlx.server",
                ),
                requires_root=True,
            )
        ]
    return []


def apply_repairs(
    actions: list[RepairAction],
    *,
    dry_run: bool = False,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    geteuid: Callable[[], int] = os.geteuid,
    verify_service: Callable[[], bool] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    verify_timeout_s: float = 20.0,
) -> list[RepairResult]:
    """Apply each action, then report success only after re-detection."""
    results: list[RepairResult] = []
    for action in actions:
        command = list(action.command)
        if dry_run:
            results.append(
                RepairResult(
                    id=action.id,
                    status="planned",
                    summary=action.summary,
                    command=command,
                    detail="dry run; no changes made",
                )
            )
            continue
        if action.requires_root and geteuid() != 0:
            results.append(
                RepairResult(
                    id=action.id,
                    status="not_applied",
                    summary=action.summary,
                    command=["sudo", *command],
                    detail="root is required; Doctor never invokes sudo automatically",
                )
            )
            continue
        try:
            completed = run(
                command,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            results.append(
                RepairResult(
                    id=action.id,
                    status="failed",
                    summary=action.summary,
                    command=command,
                    detail=str(exc),
                )
            )
            continue
        if completed.returncode != 0:
            failure_detail = (completed.stderr or completed.stdout).strip()
            results.append(
                RepairResult(
                    id=action.id,
                    status="failed",
                    summary=action.summary,
                    command=command,
                    detail=failure_detail
                    or f"repair command exited with status {completed.returncode}",
                )
            )
            continue

        verified = False
        verification_error: Exception | None = None
        if verify_service is not None:
            deadline = time.monotonic() + verify_timeout_s
            while time.monotonic() < deadline:
                try:
                    if verify_service():
                        verified = True
                        break
                except Exception as exc:  # noqa: BLE001 — verifier boundary
                    verification_error = exc
                sleep(min(1.0, max(0.0, deadline - time.monotonic())))
        results.append(
            RepairResult(
                id=action.id,
                status="verified" if verified else "unverified",
                summary=action.summary,
                command=command,
                detail=(
                    "post-repair liveness check passed"
                    if verified
                    else (
                        f"command succeeded but verification failed: "
                        f"{type(verification_error).__name__}: {verification_error}"
                        if verification_error is not None
                        else "command succeeded but liveness did not recover before timeout"
                    )
                ),
            )
        )
    return results


def service_is_live() -> bool:
    """Bounded verifier for the only currently automated repair."""
    from vllm_mlx.headless_service.status import collect_status

    status = collect_status(probe_timeout_s=0.5)
    pid = status.get("pid")
    return (
        status.get("registered") is True
        and isinstance(pid, int)
        and not isinstance(pid, bool)
        and pid > 0
        and status.get("livez") is True
    )
