# SPDX-License-Identifier: Apache-2.0
"""Advisory step — patch (diff) coverage measurement. MEASURE-FIRST, NO GATE.

This step reports what fraction of the *production lines this PR adds or
changes* are exercised by the unit suite. It NEVER blocks a merge.

Why advisory-only (dev-flow gate proposal, 2026-07): a hard threshold
picked before we have any distribution is a number out of thin air. The
plan is to publish patch-coverage across ~20 PRs, look at where the
numbers actually land, and only *then* decide whether a gate is
justified. Until then this step exists purely to make the signal
visible in the scorecard (which is posted as a PR comment).

Mechanism:
  1. Run the same unit set ``full_unit`` runs, but under ``coverage``
     instrumentation (``pytest --cov=vllm_mlx --cov-report=xml``). This
     is a SEPARATE, self-contained run — we deliberately do NOT
     piggyback on ``full_unit``. ``full_unit`` is a merge-*gating*
     step; an advisory measurement feature must not be able to break
     the gate. The asymmetry is the whole argument: a false block on
     the gate costs a maintainer a blocked-PR investigation, while
     sharing the run would save only ~40 s. Isolation wins.
  2. Feed the coverage XML to ``diff-cover`` scoped to the diff vs the
     base branch → patch-coverage %.

Because it's advisory, EVERY failure path here returns ``pass`` or
``skip`` — never ``fail`` / ``error``. A crash in an advisory measurer
must not be the reason a good PR can't merge. The base ``execute``
wrapper would turn an *uncaught* exception into ``error`` (which the
scorecard treats as blocking), so ``run`` catches everything and
downgrades to ``skip``. The subprocess calls carry bounded timeouts so
a hung test or diff-cover can't wedge the (auto-deploy) pipeline, and
the diagnostic-log writes are best-effort (a disk-full / permission
error while logging must not escape as a blocking ``error`` either).
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
import traceback
from pathlib import Path

from ..base import Step, StepResult
from ..context import Context

# The package we measure. Coverage is scoped to production code only —
# test files aren't the subject of "is this PR's new code tested".
_COV_PACKAGE = "vllm_mlx"

# diff-cover's compare point. ``origin/main`` is the merge target and is
# always present locally after ``git fetch origin`` (which the operator
# runs before validating, per G13). diff-cover computes the merge-base
# internally, so this yields exactly the lines this PR adds on top of
# main. We prefer the branch name over ``ctx.base_sha`` because the raw
# SHA may not resolve if the local clone is shallow.
_COMPARE_BRANCH = "origin/main"

# Bounded so a hung test or a wedged diff-cover can't block the pipeline
# — the whole point of an advisory step is that it never blocks. On
# expiry ``subprocess.run`` SIGKILLs the child; we catch and skip. The
# suite timeout is deliberately generous (the instrumented full suite is
# ~3 min today) — a false timeout would just silently drop a
# measurement, so err long.
_PYTEST_TIMEOUT_S = 1800
_DIFF_COVER_TIMEOUT_S = 180


class DiffCoverageStep(Step):
    name = "diff_coverage"
    description = "advisory patch (diff) coverage — measure-only, never gates"

    # An advisory measurer must never stop the pipeline. (Belt-and-braces:
    # ``run`` already catches every exception, so ``execute`` can't reach
    # its error path — but if it somehow did, don't halt later steps.)
    continue_on_error = True

    def should_run(self, ctx: Context) -> bool:
        # Nothing to measure on docs/example-only PRs.
        if ctx.blast_radius == "low":
            return False
        # Only meaningful when the PR touches production Python under the
        # measured package — a tests-only or config-only PR has no
        # ``vllm_mlx`` lines for diff-cover to score.
        return any(
            f.startswith(f"{_COV_PACKAGE}/") and f.endswith(".py")
            for f in ctx.files_changed
        )

    def run(self, ctx: Context) -> StepResult:
        # Advisory contract: swallow ALL failures. Never fail/error.
        # Resolve the artifact path INSIDE the protected block —
        # ``artifact_path()`` does a ``mkdir`` that can itself raise on
        # disk-full / permission errors, and even that must not escape
        # through ``execute()`` as a blocking ``error`` (codex #1220).
        # The fallback tolerates ``log_path`` never getting assigned.
        log_path: Path | None = None
        try:
            log_path = ctx.artifact_path("diff-coverage.log")
            return self._measure(ctx, log_path)
        except Exception as e:  # noqa: BLE001 — advisory must not block merge
            # ``_safe_write`` never raises; guard the path being unset too.
            if log_path is not None:
                _safe_write(log_path, traceback.format_exc())
            return self._skip(
                f"advisory coverage skipped (internal error: {type(e).__name__}: {e})",
                log_path,
            )

    # ------------------------------------------------------------------

    def _measure(self, ctx: Context, log_path: Path) -> StepResult:
        # 1. Tooling present? Both are declared in the ``[test]``/``[dev]``
        #    extras, but the operator may run validate from a leaner env.
        #    Missing tooling is a clean skip, not a failure.
        if importlib.util.find_spec("pytest_cov") is None:
            return self._skip(
                "pytest-cov not installed — advisory coverage unavailable"
            )
        if importlib.util.find_spec("diff_cover") is None:
            return self._skip(
                "diff-cover not installed — advisory coverage unavailable"
            )

        xml_path = ctx.artifact_path("coverage.xml")
        # Fresh-file guarantee: never let a stale coverage.xml from an
        # earlier run masquerade as this run's result. If pytest fails to
        # regenerate it below, the ``not xml_path.exists()`` path fires
        # instead of publishing yesterday's numbers.
        xml_path.unlink(missing_ok=True)

        # 2. Instrumented suite — mirrors ``full_unit``'s selection so the
        #    coverage picture matches what we actually gate on.
        pytest_cmd = [
            sys.executable,
            "-m",
            "pytest",
            "tests/",
            "--ignore=tests/integrations",
            "--ignore=tests/test_event_loop.py",
            f"--cov={_COV_PACKAGE}",
            # Suppress the terminal cov table (noise); we only need XML.
            "--cov-report=",
            f"--cov-report=xml:{xml_path}",
            "-q",
            "--no-header",
            "-p",
            "no:cacheprovider",
        ]
        ctx.run_log("diff_coverage: running instrumented suite (advisory)…")
        try:
            pytest_proc = subprocess.run(  # noqa: S603 — fixed argv, no shell
                pytest_cmd,
                capture_output=True,
                text=True,
                cwd=str(ctx.repo_root),
                timeout=_PYTEST_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            return self._skip(
                f"advisory coverage skipped (instrumented suite exceeded "
                f"{_PYTEST_TIMEOUT_S}s — killed)"
            )

        # 3. Only trust coverage from a CLEAN, COMPLETE run. A nonzero
        #    exit means failures / interruption / collection error — a
        #    partial or failing run would contaminate the very baseline
        #    this feature exists to collect. ``full_unit`` owns surfacing
        #    a red suite; here we simply skip. (Green-suite PRs — the only
        #    ones that merge and form the baseline — exit 0, so nothing of
        #    value is lost.)
        if pytest_proc.returncode != 0:
            _safe_write(log_path, _pytest_dump(pytest_cmd, pytest_proc))
            return self._skip(
                f"advisory coverage skipped (instrumented suite exit "
                f"{pytest_proc.returncode} — not clean; see full_unit)",
                log_path,
            )

        if not xml_path.exists():
            _safe_write(log_path, _pytest_dump(pytest_cmd, pytest_proc))
            return self._skip(
                "advisory coverage skipped (no coverage.xml produced)", log_path
            )

        # 4. diff-cover: patch coverage of the changed lines vs main.
        #    Invoke via ``-m`` so it runs in the SAME interpreter the
        #    coverage was produced with (matches targeted_tests' policy).
        dc_cmd = [
            sys.executable,
            "-m",
            "diff_cover.diff_cover_tool",
            str(xml_path),
            "--compare-branch",
            _COMPARE_BRANCH,
        ]
        try:
            dc_proc = subprocess.run(  # noqa: S603 — fixed argv, no shell
                dc_cmd,
                capture_output=True,
                text=True,
                cwd=str(ctx.repo_root),
                timeout=_DIFF_COVER_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            return self._skip(
                f"advisory coverage skipped (diff-cover exceeded "
                f"{_DIFF_COVER_TIMEOUT_S}s — killed)"
            )

        _safe_write(
            log_path,
            "# diff_coverage advisory run\n\n"
            f"## pytest cmd\n`{' '.join(pytest_cmd)}`\n\n"
            f"## pytest exit: {pytest_proc.returncode}\n\n"
            f"## diff-cover cmd\n`{' '.join(dc_cmd)}`\n\n"
            f"## diff-cover exit: {dc_proc.returncode}\n\n"
            "## diff-cover stdout\n"
            + (dc_proc.stdout or "")
            + "\n## diff-cover stderr\n"
            + (dc_proc.stderr or ""),
        )

        # A nonzero diff-cover exit means it errored (bad XML, git
        # failure, interrupted) — even if it happened to print a footer
        # first, that number is not trustworthy. Skip BEFORE parsing so a
        # failed run can't publish a misleading success (codex #1220).
        # Verified exit codes: scored-lines=0, no-lines=0, bad-xml=1,
        # bad-branch=1 — so this never false-skips the happy path.
        if dc_proc.returncode != 0:
            return self._skip(
                f"advisory coverage skipped (diff-cover exit "
                f"{dc_proc.returncode} — not scored)",
                log_path,
            )

        parsed = _parse_diff_cover(dc_proc.stdout)
        if parsed is None:
            # diff-cover ran but found no changed lines it could score
            # (e.g. every changed line is a comment / blank / not in the
            # coverage source map). Nothing to report — clean skip.
            return self._skip(
                "advisory coverage skipped (no measurable production lines in diff)",
                log_path,
            )

        pct, covered, total = parsed
        summary = (
            f"patch coverage {pct:.0f}% ({covered}/{total} changed lines) · "
            "advisory — not gating"
        )
        finding = (
            f"[ADVISORY] patch (diff) coverage: {pct:.1f}% — "
            f"{covered}/{total} newly changed {_COV_PACKAGE} lines exercised "
            "by the unit suite. Measure-only; no threshold enforced yet "
            "(collecting baseline across ~20 PRs before deciding a gate)."
        )
        return StepResult(
            name=self.name,
            status="pass",  # ALWAYS pass — advisory
            summary=summary,
            findings=[finding],
            artifacts=[str(log_path), str(xml_path)],
        )

    # ------------------------------------------------------------------

    def _skip(self, summary: str, log_path: Path | None = None) -> StepResult:
        """Uniform advisory skip. Attaches the log artifact only when it
        actually made it to disk (``_safe_write`` may have swallowed a
        write error)."""
        artifacts = (
            [str(log_path)] if log_path is not None and log_path.exists() else []
        )
        return StepResult(
            name=self.name, status="skip", summary=summary, artifacts=artifacts
        )


def _safe_write(path: Path, text: str) -> None:
    """Best-effort diagnostic write. NEVER raises. An advisory step must
    still return skip/pass even if it can't write its own log (disk full,
    permissions) — otherwise a failing write inside an exception handler
    would escape as a blocking ``error`` (codex review on #1220)."""
    try:
        path.write_text(text)
    except OSError:
        pass


def _pytest_dump(cmd: list[str], proc: subprocess.CompletedProcess[str]) -> str:
    return (
        f"# instrumented pytest (exit {proc.returncode})\n\n"
        f"## cmd\n`{' '.join(cmd)}`\n\n"
        "## stdout\n" + (proc.stdout or "") + "\n## stderr\n" + (proc.stderr or "")
    )


# ----------------------------------------------------------------------
# diff-cover output parsing — we parse the stable human-readable text
# rather than a JSON report because the JSON flag name has churned
# across diff-cover versions (``--json-report`` vs ``--format json:``)
# while the text footer (``Total:`` / ``Missing:`` / ``Coverage:``) has
# been stable for years.
# ----------------------------------------------------------------------

_TOTAL_RE = re.compile(r"^Total:\s+(\d+)\s+lines?", re.MULTILINE)
_MISSING_RE = re.compile(r"^Missing:\s+(\d+)\s+lines?", re.MULTILINE)
_NO_LINES_RE = re.compile(r"No lines with coverage information", re.IGNORECASE)


def _parse_diff_cover(stdout: str) -> tuple[float, int, int] | None:
    """Return ``(percent, covered_lines, total_lines)`` or ``None`` when
    diff-cover reports nothing to score.

    Percent is computed from the exact ``Total`` / ``Missing`` counts,
    NOT from diff-cover's own ``Coverage:`` line. diff-cover *floors*
    that displayed integer (e.g. 58/1934 = 2.9989 % prints as
    ``Coverage: 2%``), which both loses resolution and can read a point
    below the true value — bad for a baseline we intend to threshold on
    later. So our headline % may read ~1 pt above the number in the
    saved diff-cover log; the ``covered/total`` counts we surface
    alongside it are the unambiguous ground truth.
    """
    text = stdout or ""
    if _NO_LINES_RE.search(text):
        return None
    tm = _TOTAL_RE.search(text)
    mm = _MISSING_RE.search(text)
    if not tm or not mm:
        return None
    total = int(tm.group(1))
    missing = int(mm.group(1))
    if total <= 0:
        return None
    covered = total - missing
    pct = 100.0 * covered / total
    return pct, covered, total
