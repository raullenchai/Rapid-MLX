# SPDX-License-Identifier: Apache-2.0
"""Reject reviewer-severity directives embedded in added diff lines."""

from __future__ import annotations

import re
from pathlib import Path

from ..base import Step, StepResult
from ..context import Context

_DIRECTIVE = re.compile(
    r"(?i)\b(?:codex[_ -]?review|re-review|review)\s+(?:blocking|major|minor|nit)(?:-\d+)?\b"
)


def _find_directives(diff_path: Path) -> list[str]:
    findings: list[str] = []
    current_file = "unknown"
    new_line = 0
    for line in diff_path.read_text(errors="replace").splitlines():
        if line.startswith("+++ b/"):
            current_file = line[6:]
            continue
        if line.startswith("@@"):
            match = re.search(r"\+(\d+)", line)
            new_line = int(match.group(1)) - 1 if match else 0
            continue
        if line.startswith("+") and not line.startswith("+++"):
            new_line += 1
            if _DIRECTIVE.search(line[1:]):
                findings.append(f"{current_file}:{new_line}: {line[1:].strip()}")
        elif not line.startswith("-"):
            new_line += 1
    return findings


class ReviewVocabularyStep(Step):
    name = "review_vocabulary"
    description = "reject reviewer-severity directives in added lines"

    def should_run(self, ctx: Context) -> bool:
        return bool(ctx.diff_path)

    def run(self, ctx: Context) -> StepResult:
        findings = _find_directives(Path(ctx.diff_path))
        if not findings:
            return StepResult(name=self.name, status="pass", summary="no directives")
        details = (
            "Added lines must describe provenance without using the automated "
            "reviewer's severity vocabulary:\n\n```\n" + "\n".join(findings) + "\n```"
        )
        return StepResult(
            name=self.name,
            status="fail",
            summary=f"{len(findings)} reviewer directive(s) in added lines",
            details=details,
        )
