# SPDX-License-Identifier: Apache-2.0
"""pr_validate's own tests must not inherit operator step opt-outs (#4984).

``PR_VALIDATE_NO_STRESS=1`` / ``PR_VALIDATE_NO_CODEX=1`` are documented for
whoever runs the validator, and ``full_unit`` runs pytest in a child process
that inherits them. Runner tests that assert default-pipeline behavior used
to fail under that ambient state and flip a clean PR to DO NOT MERGE.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from scripts.pr_validate.context import STEP_OPT_OUT_ENV_VARS

REPO_ROOT = Path(__file__).resolve().parents[1]

# Tests that failed with the opt-outs exported before the fix.
_SENSITIVE_TESTS = (
    "tests/test_pr_validate_runner.py::test_queue_handoff_only_called_after_merge_safe",
    "tests/test_pr_validate_sdk_startup.py::test_sdk_probe_launch_failure_records_diagnostics_and_stops",
    "tests/test_pr_validate_sdk_startup.py::test_prerequisite_probe_reports_real_import_failure_without_sdk_or_network",
    "tests/test_pr_validate_google_review.py::TestCLDescriptionQualityTitle::test_bad_title_patterns_fail",
    "tests/test_pr_validate_google_review.py::TestCLDescriptionQualityBody::test_body_with_no_rationale_signal_fails",
)


def test_step_opt_outs_are_cleared_for_each_test():
    for name in STEP_OPT_OUT_ENV_VARS:
        assert name not in os.environ


def test_sensitive_tests_pass_with_step_opt_outs_exported():
    env = dict(os.environ)
    env.update(dict.fromkeys(STEP_OPT_OUT_ENV_VARS, "1"))
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "-o",
            "addopts=",
            *_SENSITIVE_TESTS,
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-2000:]
    assert " passed" in proc.stdout
