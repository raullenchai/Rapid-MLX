# SPDX-License-Identifier: Apache-2.0
"""Step 2 — lint (ruff check + ruff format --check).

Runs against the PR's changed files (plus their containing module if
the diff hit a single file deep in a package). We don't check the
whole repo every time — that runs for ~5s on this codebase but adds
noise when an unrelated old file already has lint errors.
"""

from __future__ import annotations

import subprocess

from ..base import Step, StepResult
from ..context import Context

# We only lint Python files. Other extensions (.md, .yaml, .yml) get
# skipped — ruff doesn't apply, and dedicated linters for those would
# be a separate step.
_PY_SUFFIXES = (".py",)


class LintStep(Step):
    name = "lint"
    description = "ruff check + ruff format on changed files"

    def should_run(self, ctx: Context) -> bool:
        # Skip if the PR has no python changes at all (docs-only PRs).
        return any(p.endswith(_PY_SUFFIXES) for p in ctx.files_changed)

    def run(self, ctx: Context) -> StepResult:
        # Filter to existing .py files. A deletion-only PR would list
        # paths that no longer exist in the working tree; ruff would
        # error on those — skip them, the diff itself isn't lintable.
        py_files = [
            p
            for p in ctx.files_changed
            if p.endswith(_PY_SUFFIXES) and (ctx.repo_root / p).exists()
        ]
        if not py_files:
            return StepResult(
                name=self.name,
                status="skip",
                summary="no python files to lint (deletions only?)",
            )

        # Run check + format-check separately so we can attribute the
        # failure precisely. Both must pass.
        check_log = ctx.artifact_path("lint-check.log")
        format_log = ctx.artifact_path("lint-format.log")

        # `--force-exclude` is REQUIRED here: when ruff is handed explicit
        # file paths (as opposed to discovering them itself), it does NOT
        # apply the `[tool.ruff].exclude` / `[tool.ruff.format].exclude`
        # config unless this flag is set. Without it, vendored files that
        # pyproject deliberately excludes (deepseek_v4.py, gemma4_vendored/*,
        # hy_v3.py, ...) get lint/format-checked anyway, falsely failing the
        # step on any PR that merely touches them.
        check_rc = _run_ruff(["check", "--force-exclude", *py_files], check_log)
        format_rc = _run_ruff(
            ["format", "--check", "--force-exclude", *py_files], format_log
        )

        if check_rc == 0 and format_rc == 0:
            return StepResult(
                name=self.name,
                status="pass",
                summary=f"clean ({len(py_files)} file(s))",
                artifacts=[str(check_log), str(format_log)],
            )

        # At least one failed — surface both logs in the details so the
        # contributor sees exactly which files / lines need attention.
        details = []
        if check_rc != 0:
            details.append("**`ruff check` failures:**\n```")
            details.append(check_log.read_text().strip())
            details.append("```")
        if format_rc != 0:
            details.append("**`ruff format --check` would reformat:**\n```")
            details.append(format_log.read_text().strip())
            details.append("```")
            details.append(
                "\nFix locally with: `ruff format " + " ".join(py_files) + "`"
            )

        return StepResult(
            name=self.name,
            status="fail",
            summary=(
                f"check_rc={check_rc}, format_rc={format_rc} ({len(py_files)} file(s))"
            ),
            details="\n".join(details),
            artifacts=[str(check_log), str(format_log)],
        )


def _run_ruff(args: list[str], log_path) -> int:
    """Run ruff and tee output to log_path. Returns ruff's exit code."""
    proc = subprocess.run(  # noqa: S603
        ["ruff", *args],
        capture_output=True,
        text=True,
    )
    log_path.write_text((proc.stdout or "") + (proc.stderr or ""))
    return proc.returncode
