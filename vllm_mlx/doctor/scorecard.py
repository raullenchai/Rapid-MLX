# SPDX-License-Identifier: Apache-2.0
"""Scorecard rendering for the benchmark tier.

Aggregates per-cell ``CheckResult.metrics`` dicts into a single markdown
table.  Output goes to ``harness/scorecard/scorecard-{ts}.md`` and a
"latest" symlink so callers can always find the most recent one.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

from .runner import HARNESS_DIR, CheckResult, Status, md_cell

SCORECARD_DIR = HARNESS_DIR / "scorecard"

# Columns rendered in the scorecard.  Keep the list short — wide tables
# are unreadable in markdown.  Ordered by importance.
SCORECARD_COLUMNS: list[tuple[str, str, str]] = [
    # (metric_key, header, format)
    ("decode_tps", "Decode TPS", "{:.1f}"),
    ("cold_ttft_ms", "Cold TTFT", "{:.0f}ms"),
    ("cached_ttft_ms", "Cached TTFT", "{:.0f}ms"),
    ("tc_success_rate", "Tool %", "{:.0%}"),
    ("composite_score", "Score", "{:.1f}"),
]


def render_scorecard(
    cells: list[tuple[str, CheckResult]],
    skipped: list[tuple[str, str]] | None = None,
    title: str = "Rapid-MLX Benchmark Scorecard",
) -> str:
    """Render a markdown scorecard.

    Args:
        cells: list of (model, CheckResult) for each benchmarked combo.
        skipped: list of (model, reason) for combos that didn't run.
        title: H1 heading.
    """
    lines: list[str] = [
        f"# {title}",
        "",
        f"_Generated: {_dt.datetime.now().isoformat(timespec='seconds')}_",
        "",
        "| Model | " + " | ".join(h for _, h, _ in SCORECARD_COLUMNS) + " | Status |",
        "| --- | " + " | ".join("---:" for _ in SCORECARD_COLUMNS) + " | --- |",
    ]

    for model, result in cells:
        if result.status == Status.PASS and result.metrics:
            cells_md = []
            for key, _h, fmt in SCORECARD_COLUMNS:
                val = result.metrics.get(key)
                cells_md.append(fmt.format(val) if val is not None else "—")
            status_md = "OK"
        else:
            cells_md = ["—"] * len(SCORECARD_COLUMNS)
            # Fold the failure reason into the status cell so the row
            # is still useful at a glance.
            reason = (result.detail or "fail").splitlines()[0][:80]
            status_md = f"FAIL — {md_cell(reason)}"
        lines.append(
            f"| {md_cell(model)} | " + " | ".join(cells_md) + f" | {status_md} |"
        )

    if skipped:
        lines += ["", "## Skipped", ""]
        for model, reason in skipped:
            lines.append(f"- **{model}** — {reason}")

    return "\n".join(lines) + "\n"


def write_scorecard(content: str) -> Path:
    """Write the scorecard markdown to disk + update 'latest.md' alias.

    Returns the timestamped path.  ``latest.md`` is a regular file
    rather than a symlink so it stays valid on filesystems that
    disallow symlinks (Windows by default, some shared mounts).
    """
    SCORECARD_DIR.mkdir(parents=True, exist_ok=True)
    ts = _dt.datetime.now().strftime("%Y-%m-%d-%H%M%S")
    path = SCORECARD_DIR / f"scorecard-{ts}.md"
    path.write_text(content)
    (SCORECARD_DIR / "latest.md").write_text(content)
    return path
