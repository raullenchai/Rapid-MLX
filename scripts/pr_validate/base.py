# SPDX-License-Identifier: Apache-2.0
"""Base classes for pipeline steps.

Each step is a small class with three contract points: a name, a
``should_run(ctx)`` predicate (lets us gate expensive steps on blast
radius), and a ``run(ctx) -> StepResult``. Keep step modules
self-contained — the runner doesn't know what each does, only that it
returns a uniform result object the scorecard can render.

We intentionally avoid a plugin registry / entrypoints mechanism. The
runner explicitly imports + orders steps so the pipeline is grep-able
and review-time obvious. Adding a step = one import + one list entry.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .context import Context


# Status meanings:
#   pass    — step ran, nothing wrong → does not block merge
#   fail    — step ran, found a real problem → BLOCKS merge (strict mode)
#   skip    — step decided not to run (gating) → does not block, neutral
#   error   — step crashed before it could decide → BLOCKS merge (treat
#             unknown like failure; never let a broken validator silently
#             approve a PR)
StepStatus = Literal["pass", "fail", "skip", "error"]


@dataclass
class StepResult:
    """Per-step output the scorecard renders.

    ``summary`` is a one-liner shown in the verdict table; ``details``
    is the multiline markdown shown when the step failed (or when the
    user asked for ``--verbose``). ``artifacts`` are paths to log files
    in the working dir — preserved so the user can inspect after.
    """

    name: str
    status: StepStatus
    summary: str
    details: str = ""
    duration_seconds: float = 0.0
    artifacts: list[str] = field(default_factory=list)
    # Free-form findings list; populated by the adversarial-review step
    # so the scorecard can render them inline. Each is one short string.
    findings: list[str] = field(default_factory=list)
    stage: str = ""


class Step:
    """Pipeline step. Subclasses override ``run``; everything else is
    handled by the runner."""

    name: str = "unnamed"
    description: str = ""
    # If True, an ``error`` from this step still lets later steps run
    # (best-effort gating). Default is False — most steps are blocking.
    continue_on_error: bool = False

    def should_run(self, ctx: Context) -> bool:
        """Return False to skip — e.g. blast radius too low. Default
        runs every time."""
        return True

    def run(self, ctx: Context) -> StepResult:
        raise NotImplementedError

    def execute(self, ctx: Context) -> StepResult:
        """Wrapper called by the runner. Times the step, catches
        exceptions, normalizes errors so a step bug never silently
        passes a bad PR."""
        if not self.should_run(ctx):
            return StepResult(
                name=self.name,
                status="skip",
                summary="skipped (gating predicate returned False)",
                duration_seconds=0.0,
            )
        t0 = time.monotonic()
        try:
            result = self.run(ctx)
        except Exception as e:  # noqa: BLE001 — we want to catch ANY crash
            import traceback

            return StepResult(
                name=self.name,
                status="error",
                summary=f"step crashed: {type(e).__name__}: {e}",
                details=f"```\n{traceback.format_exc()}\n```",
                duration_seconds=time.monotonic() - t0,
            )
        result.duration_seconds = time.monotonic() - t0
        if result.name == "unnamed":
            result.name = self.name
        return result
