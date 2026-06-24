# SPDX-License-Identifier: Apache-2.0
"""Step 0.8 — test-env self-check.

Runs just after ``supply_chain`` and just before the codex review,
i.e. as the first compute-bound gate AFTER the supply-chain scan has
had a chance to flag a malicious PR (#275). Verifies that the same
Python interpreter ``targeted_tests`` and ``full_unit`` will hand to
pytest can actually load the plugins the suite needs (chiefly
``pytest_asyncio`` — pytest.ini sets ``asyncio_mode = auto``).

Without this gate, a broken env produced a 124-failure ``full_unit``
log that looked like a regression but was really tooling debt (#185 /
PR #731). The fix is to fail loudly at the front of the pipeline, with
the exact missing-package list and the canonical recovery command,
rather than letting downstream pytest crash with a cryptic plugin-
missing message.

Recovery path: by default the step *also* attempts to install the
required plugins so the rest of the pipeline can run:

* **Trusted-pins path (default + safe).** When the PR has not
  modified any dep-declaration file (``pyproject.toml``,
  ``requirements*.txt``, ``setup.py``, ``setup.cfg``), the step
  installs ``TRUSTED_TEST_PINS`` directly from PyPI (hardcoded
  version-pinned set) — bypasses the PR's working tree entirely.
* **Project-extras path.** Falls back to ``pip install '.[test]'``
  ONLY if the trusted-pins install didn't cover what's missing AND
  the PR has not touched any dep file. This path executes
  ``pyproject.toml`` build hooks, so it's gated on the PR being
  "trustworthy by file scope" — see #275 for the threat model.
* **Skip auto-install.** When the PR touched a dep file (so the
  project-extras path is unsafe) the step refuses to auto-install
  and emits a MERGE-UNSAFE-flavored warning naming the offending
  file(s) and the canonical manual recovery command.

Disable with ``PR_VALIDATE_NO_AUTO_INSTALL=1`` for hardened CI
sandboxes that must not mutate the host Python.
"""

from __future__ import annotations

from .._test_env import (
    TEST_EXTRAS_NAME,
    auto_install_disabled,
    check_test_env,
    install_test_extras,
    install_trusted_pins,
    pr_touches_dep_files,
)
from ..base import Step, StepResult
from ..context import Context


class TestEnvCheckStep(Step):
    name = "test_env_check"
    description = "verify pytest plugins importable in target Python"

    def run(self, ctx: Context) -> StepResult:
        log_path = ctx.artifact_path("test-env-check.log")

        status = check_test_env()
        if status.ok:
            log_path.write_text(
                f"interpreter: {status.interpreter}\n"
                f"status: ok\n"
                f"detail: {status.message}\n"
            )
            return StepResult(
                name=self.name,
                status="pass",
                summary=status.message,
                artifacts=[str(log_path)],
            )

        # Something's missing. Try to fix it from the canonical source
        # unless the operator opted out.
        if auto_install_disabled():
            details = (
                f"**pr_validate venv is misconfigured** — "
                f"{status.message}.\n\n"
                "Auto-install is disabled (`PR_VALIDATE_NO_AUTO_INSTALL=1`). "
                "Recover with:\n\n"
                f"```\n{status.install_hint}\n```\n"
            )
            log_path.write_text(
                f"interpreter: {status.interpreter}\n"
                f"status: fail (auto-install disabled)\n"
                f"missing: {', '.join(status.missing)}\n"
                f"hint: {status.install_hint}\n"
            )
            return StepResult(
                name=self.name,
                status="fail",
                summary=f"{status.message} — auto-install disabled",
                details=details,
                artifacts=[str(log_path)],
            )

        # Supply-chain integrity gate (#275): if the PR has modified
        # any dep-declaration file we MUST NOT run
        # ``pip install '.[test]'`` from the PR's working tree —
        # build hooks in a tampered pyproject.toml would execute
        # inside the validator venv. Try the trusted-pins path first
        # (PyPI-only, hardcoded versions); if that doesn't resolve
        # everything we still skip the project-extras path and let the
        # operator decide.
        touched_dep_files = pr_touches_dep_files(ctx.files_changed)

        ctx.run_log(
            f"missing: {', '.join(status.missing)} — installing trusted pins from PyPI"
        )
        pins_ok, pins_log = install_trusted_pins()
        post_pins = check_test_env()
        if post_pins.ok:
            log_path.write_text(
                f"interpreter: {status.interpreter}\n"
                f"status (initial): fail\n"
                f"missing (initial): {', '.join(status.missing)}\n"
                f"trusted-pins install ok: {pins_ok}\n"
                f"trusted-pins pip log:\n{pins_log}\n"
                f"status (after trusted pins): ok\n"
            )
            return StepResult(
                name=self.name,
                status="pass",
                summary=(
                    f"installed trusted-pins ({len(status.missing)} "
                    f"missing plugin(s) recovered)"
                ),
                artifacts=[str(log_path)],
            )

        # Trusted pins didn't cover it. The fallback is the PR's own
        # ``.[test]`` extras — but that's exactly the path #275 says
        # is unsafe when the PR has touched a dep-declaration file.
        if touched_dep_files:
            details = (
                f"**pr_validate venv is misconfigured** — "
                f"{status.message}.\n\n"
                "Auto-install via `pip install '.[test]'` was **refused**: "
                f"this PR modifies dep-declaration file(s) "
                f"`{', '.join(touched_dep_files)}` and installing from the "
                "PR's working tree would execute build hooks / fake "
                "package sources from the diff inside the validator "
                "venv. See #275 / the threat-model section of "
                "scripts/pr_validate/README.md.\n\n"
                "Trusted-pins install (PyPI-only, hardcoded versions) "
                f"also did not resolve everything: `{post_pins.message}`.\n\n"
                "Manual recovery — review the PR diff first, then run:\n\n"
                f"```\n{post_pins.install_hint}\n```\n\n"
                "Pip output (trusted-pins, truncated):\n\n"
                f"```\n{pins_log}\n```\n"
            )
            log_path.write_text(
                f"interpreter: {status.interpreter}\n"
                f"status: fail (auto-install refused — PR touches dep files)\n"
                f"touched dep files: {', '.join(touched_dep_files)}\n"
                f"missing (initial): {', '.join(status.missing)}\n"
                f"trusted-pins install ok: {pins_ok}\n"
                f"trusted-pins pip log:\n{pins_log}\n"
                f"status (after trusted pins): "
                f"{'ok' if post_pins.ok else 'fail'}\n"
                f"missing (after trusted pins): "
                f"{', '.join(post_pins.missing) or '(none)'}\n"
            )
            return StepResult(
                name=self.name,
                status="fail",
                summary=(
                    f"missing test packages; project-extras install "
                    f"refused (PR touches "
                    f"{', '.join(touched_dep_files)})"
                ),
                details=details,
                artifacts=[str(log_path)],
            )

        ctx.run_log(
            f"trusted pins insufficient ({post_pins.message}) — "
            f"falling back to .[{TEST_EXTRAS_NAME}]"
        )
        ok, pip_log = install_test_extras(ctx.repo_root)
        # Re-check after install — pip can report 0 and still leave a
        # plugin un-importable (rare, but happens when there's a
        # site-packages permission issue and pip silently falls back
        # to --user). The second probe is the source of truth.
        post = check_test_env()

        log_path.write_text(
            f"interpreter: {status.interpreter}\n"
            f"status (initial): fail\n"
            f"missing (initial): {', '.join(status.missing)}\n"
            f"trusted-pins install ok: {pins_ok}\n"
            f"trusted-pins pip log:\n{pins_log}\n"
            f"status (after trusted pins): "
            f"{'ok' if post_pins.ok else 'fail'}\n"
            f"project-extras auto-install ok: {ok}\n"
            f"project-extras pip log:\n{pip_log}\n"
            f"status (after install): {'ok' if post.ok else 'fail'}\n"
            f"missing (after install): {', '.join(post.missing) or '(none)'}\n"
        )

        if post.ok:
            return StepResult(
                name=self.name,
                status="pass",
                summary=(
                    f"installed .[{TEST_EXTRAS_NAME}] to recover "
                    f"{len(status.missing)} missing plugin(s)"
                ),
                artifacts=[str(log_path)],
            )

        # Auto-install ran but didn't fix it. Surface BOTH the original
        # missing list AND the pip output so the operator can diagnose.
        details = [
            f"**pr_validate venv is misconfigured** — {status.message}.\n",
            (
                f"Auto-install attempted (`pip install '.[{TEST_EXTRAS_NAME}]'`) "
                f"but the post-install probe still reports: "
                f"`{post.message}`.\n"
            ),
            "Manual recovery:\n",
            f"```\n{post.install_hint}\n```\n",
            "Pip output (truncated):\n",
            f"```\n{pip_log}\n```",
        ]
        return StepResult(
            name=self.name,
            status="fail",
            summary=(
                f"missing test packages and auto-install failed: "
                f"{', '.join(post.missing)}"
            ),
            details="\n".join(details),
            artifacts=[str(log_path)],
        )
