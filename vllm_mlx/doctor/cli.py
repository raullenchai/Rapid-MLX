# SPDX-License-Identifier: Apache-2.0
"""CLI entry point for ``rapid-mlx doctor`` — pure env-health probe.

Doctor is now strictly about answering "is my install / environment broken?".
Model-validation tiers (smoke / check / full / benchmark) moved to
``rapid-mlx bench --tier ...`` in PRs #1-#3 of the doctor refactor series;
this PR rips out the dispatch and leaves only the env-health surface.

Output is modelled on ``hermes doctor``: sections of one-line ✓/⚠/✗ probes,
a summary, an exit code (0 unless any ✗). Runtime budget: ≤ 5 s end-to-end.
"""

from __future__ import annotations

import sys
from typing import Any

from .env_health import CheckStatus, Report, Section, run_all

# Removed tiers — referenced from doctor_command's error message so a user
# typing the old subcommand sees the canonical replacement, not a bare
# ``unknown argument`` traceback.
_REMOVED_TIERS = ("smoke", "check", "full", "benchmark")


# Status glyphs. ASCII fallbacks aren't supported — we already require a
# UTF-8 terminal for the section headers (◆), so a stray cp1252 user gets
# mojibake everywhere, not just on the status column.
_GLYPHS = {
    CheckStatus.OK: "✓",  # ✓
    CheckStatus.WARN: "⚠",  # ⚠
    CheckStatus.FAIL: "✗",  # ✗
}


def doctor_command(args: Any) -> None:
    """Render the env-health report and ``sys.exit`` with 0 or 1.

    ``args`` is the argparse namespace from ``vllm_mlx.cli``. We only read
    ``args.verbose`` (and reject the removed positional ``tier`` argument
    with a clear pointer to ``rapid-mlx bench --tier ...``).
    """
    # Hard removal: PRs #1-#3 deprecated the tier subcommands; this PR closes
    # the door. If a user still types ``rapid-mlx doctor smoke`` they hit
    # this branch and get pointed at the replacement.
    legacy_tier = getattr(args, "tier", None)
    if legacy_tier in _REMOVED_TIERS:
        print(
            f"rapid-mlx doctor {legacy_tier!s} was removed in 0.7.22.\n"
            f"Use:  rapid-mlx bench <model> --tier {legacy_tier}\n"
            "Doctor is now a pure environment-health check; "
            "model-validation tiers live in `rapid-mlx bench`.",
            file=sys.stderr,
        )
        sys.exit(2)

    verbose = bool(getattr(args, "verbose", False))
    report = run_all()
    render(report, verbose=verbose)
    sys.exit(report.exit_code)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render(report: Report, *, verbose: bool = False, stream=None) -> None:
    """Write the report to ``stream`` (defaults to stdout)."""
    stream = stream or sys.stdout
    write = stream.write

    write("\n")
    write("┌" + "─" * 57 + "┐\n")
    write("│" + "\U0001fa7a Rapid-MLX Doctor".center(57) + "│\n")
    write("└" + "─" * 57 + "┘\n")
    write("\n")

    for section in report.sections:
        _render_section(section, write=write, verbose=verbose)
        write("\n")

    _render_summary(report, write=write, verbose=verbose)


def _render_section(section: Section, *, write, verbose: bool) -> None:
    write(f"◆ {section.title}\n")
    for check in section.checks:
        glyph = _GLYPHS[check.status]
        write(f"  {glyph} {check.label}\n")
        if verbose and check.detail:
            write(f"      ↳ {check.detail}\n")


def _render_summary(report: Report, *, write, verbose: bool) -> None:
    write("─" * 40 + "\n")
    write(
        f"Summary: {report.n_ok} ok, "
        f"{report.n_warn} warnings, "
        f"{report.n_fail} issue"
        f"{'s' if report.n_fail != 1 else ''}\n"
    )
    if not verbose and (report.n_warn or report.n_fail):
        write("Run with `--verbose` for details on each check.\n")
