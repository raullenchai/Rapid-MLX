# SPDX-License-Identifier: Apache-2.0
"""Render the per-step results as a markdown scorecard.

Strict mode: ANY single ``fail`` or ``error`` → "DO NOT MERGE".
``skip`` is neutral. Output is markdown so the same string can be
posted as a PR comment via ``gh pr comment``.
"""

from __future__ import annotations

from .base import StepResult
from .context import Context


def verdict(results: list[StepResult]) -> str:
    """Strict — any non-pass-or-skip blocks merge."""
    blocking = [r for r in results if r.status in ("fail", "error")]
    if blocking:
        return "DO NOT MERGE"
    if not results:
        return "INCOMPLETE"
    return "MERGE-SAFE"


_STATUS_BADGE = {
    "pass": "PASS",
    "fail": "FAIL",
    "skip": "skip",
    "error": "ERROR",
}


def render_scorecard(ctx: Context, partial_stage: str = "") -> str:
    """Build the markdown report. One table row per step, then per-fail
    detail blocks below. Designed to be paste-able into a GitHub PR
    comment without further editing."""
    final = verdict(ctx.results)
    if partial_stage and final == "MERGE-SAFE":
        final = f"STAGE {partial_stage} PASSED (partial, not a merge verdict)"

    lines = []
    lines.append(f"# PR #{ctx.pr_number} validation scorecard")
    lines.append("")
    if ctx.pr_title:
        lines.append(f"**Title**: {ctx.pr_title}")
    if ctx.pr_author:
        author_label = ctx.pr_author
        if ctx.is_external_author:
            author_label += " (external)"
        lines.append(f"**Author**: {author_label}")
    if ctx.files_changed:
        lines.append(
            f"**Diff**: {len(ctx.files_changed)} file(s), "
            f"+{ctx.additions}/-{ctx.deletions} LOC, "
            f"blast radius: **{ctx.blast_radius}**"
        )
    lines.append("")
    lines.append(f"## Verdict: **{final}**")
    lines.append("")

    # Step results table.
    lines.append("| stage | step | status | summary | time |")
    lines.append("|---|---|---|---|---:|")
    for r in ctx.results:
        badge = _STATUS_BADGE[r.status]
        # Markdown-escape any pipes in summary.
        summary = r.summary.replace("|", "\\|")
        lines.append(
            f"| {r.stage or '-'} | `{r.name}` | {badge} | {summary} | {r.duration_seconds:.1f}s |"
        )
    lines.append("")

    # Detail blocks for any failure / error / important findings.
    detail_blocks = []
    for r in ctx.results:
        if r.status in ("fail", "error") or r.findings:
            label = f"{r.stage}/{r.name}" if r.stage else r.name
            block = [f"### `{label}` — {_STATUS_BADGE[r.status]}", ""]
            if r.findings:
                block.append("**Findings:**")
                for f in r.findings:
                    block.append(f"- {f}")
                block.append("")
            if r.details:
                block.append(r.details)
                block.append("")
            if r.artifacts:
                block.append("**Artifacts:**")
                for a in r.artifacts:
                    block.append(f"- `{a}`")
                block.append("")
            detail_blocks.extend(block)
    if detail_blocks:
        lines.append("## Details")
        lines.append("")
        lines.extend(detail_blocks)

    return "\n".join(lines)
