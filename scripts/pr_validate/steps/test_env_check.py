# SPDX-License-Identifier: Apache-2.0
"""Step 0.8 — test-env self-check.

Runs just after ``cl_description_quality`` and just before the codex
review, i.e. as the first compute-bound gate. Verifies that the same
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
canonical ``[test]`` extras from ``pyproject.toml`` and re-checks.
Disable with ``PR_VALIDATE_NO_AUTO_INSTALL=1`` for hardened CI
sandboxes that must not mutate the host Python.
"""

from __future__ import annotations

from .._test_env import (
    TEST_EXTRAS_NAME,
    auto_install_disabled,
    check_test_env,
    install_test_extras,
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

        ctx.run_log(
            f"missing: {', '.join(status.missing)} — installing .[{TEST_EXTRAS_NAME}]"
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
            f"auto-install ok: {ok}\n"
            f"pip log:\n{pip_log}\n"
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
