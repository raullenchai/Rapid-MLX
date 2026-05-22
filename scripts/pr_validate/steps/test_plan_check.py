# SPDX-License-Identifier: Apache-2.0
"""Step — unchecked test-plan checkbox gate.

Catches the class of bug that landed via PR #435 → re-opened #427: the
PR body's test plan listed ``- [ ] E2E verification against fishloa's
repro`` and we merged anyway. pr_validate's stress_e2e_bench passed
because it tests correctness, not the original symptom — so the SOP §6
gate ("Boot real server with the fix path — End-to-end repro the
user's failing curl request to confirm the original symptom is gone")
was the only thing standing between us and shipping a non-fix. SOP §6
is run by a human; humans skip it.

Mechanical fix: if the PR body has any unchecked Markdown checkbox
(``- [ ]`` or ``* [ ]``), block merge until the author either
verifies the item or removes it from the body. This is the same
discipline as "no failing unit tests" — the PR carries its own
verification contract; pr_validate enforces it.

Why a STEP, not a comment-only warning: warnings are ignored. A real
``fail`` status in the scorecard surfaces in the Verdict line and the
merge-readiness summary, and it stops ``--fail-fast`` runs. That's the
behavior the SOP needs to be unmissable.

Test plan markers we recognize:
* ``- [ ]`` (GitHub Markdown task list — primary form)
* ``* [ ]`` (alternative bullet)
* ``- [X]`` / ``- [x]`` / ``* [X]`` / ``* [x]`` — checked, not flagged

We DO NOT try to auto-detect which "section" of the PR body the
checkboxes appear in. Any unchecked task in any section blocks merge —
if it's in the PR body, the author put it there as a contract.
"""

from __future__ import annotations

import re

from ..base import Step, StepResult
from ..context import Context

# Match Markdown task list items in either bullet form. The body of the
# task (after ``[ ]``) is captured so the failure message can name the
# specific items left unchecked — saves the reviewer a tab-to-GitHub.
_TASK_PATTERN = re.compile(
    r"^[ \t]*[-*][ \t]+\[(?P<mark>[ xX])\][ \t]*(?P<body>.+?)[ \t]*$",
    re.MULTILINE,
)


class TestPlanCheckStep(Step):
    name = "test_plan_check"
    description = "PR body must not have unchecked checkboxes"

    def run(self, ctx: Context) -> StepResult:
        body = (ctx.pr_body or "").strip()
        if not body:
            return StepResult(
                name=self.name,
                status="pass",
                summary="no PR body — nothing to check",
            )

        unchecked: list[str] = []
        total = 0
        for m in _TASK_PATTERN.finditer(body):
            total += 1
            if m.group("mark") == " ":
                # Trim Markdown noise (links, backticks) so the summary
                # stays readable; full body still shown in details.
                snippet = m.group("body").strip()
                if len(snippet) > 100:
                    snippet = snippet[:97] + "..."
                unchecked.append(snippet)

        if total == 0:
            return StepResult(
                name=self.name,
                status="pass",
                summary="no checkbox-style test plan found",
            )

        if not unchecked:
            return StepResult(
                name=self.name,
                status="pass",
                summary=f"all {total} test-plan item(s) checked",
            )

        # Strict fail: any unchecked box blocks merge. The author either
        # ran the check (and forgot to tick) or didn't run it (and the
        # gate is doing its job). Either way the fix is fast.
        details_lines = [
            f"**{len(unchecked)} unchecked test-plan item(s)** out of {total} total:",
            "",
        ]
        for i, item in enumerate(unchecked, start=1):
            details_lines.append(f"  {i}. `[ ]` {item}")
        details_lines.append("")
        details_lines.append(
            "Either verify the item and check the box in the PR body, or "
            "remove the item if it's no longer applicable. PR #435 / #427 "
            "landed an incomplete fix because its `- [ ] E2E verification` "
            "item was left unchecked at merge — this gate exists so that "
            "doesn't repeat."
        )

        return StepResult(
            name=self.name,
            status="fail",
            summary=(
                f"{len(unchecked)}/{total} test-plan item(s) still unchecked — "
                f"first: {unchecked[0][:60]}..."
                if len(unchecked[0]) > 60
                else f"{len(unchecked)}/{total} test-plan item(s) still unchecked"
            ),
            details="\n".join(details_lines),
        )
