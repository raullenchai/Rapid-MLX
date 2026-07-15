# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the RAPID_MLX_MATRIX_NO_SKIPS decision policy.

The release-artifact matrix runs with ``RAPID_MLX_MATRIX_NO_SKIPS=1`` so a
silently-skipped cell cannot shrink required-PASS coverage. The exemption
policy is subtle enough to be worth pinning directly (codex review):

  * A strict-xfail cell that ACTUALLY xfailed (body ran, ``wasxfail`` set) is
    the ONLY approved skip exception.
  * A strict-xfail cell that merely SKIPPED on a missing prerequisite
    (``wasxfail`` unset) is NOT exempt — otherwise the matrix could pass
    "green" without ever exercising the expected failure.
  * A NON-strict / dynamic xfail (``wasxfail`` set but no strict marker) is
    NOT exempt — an un-audited xfail must not bypass the gate.
  * A plain skip on a matrix cell is NOT exempt.
  * A skip on a non-matrix test is left alone.

``conftest._skip_should_become_failure`` is the pure decision function the
``pytest_runtest_makereport`` hook delegates to; these tests drive it across
every case so the real hook logic is covered without booting a server.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from tests.integrations import conftest as matrix_conftest

_REPO_ROOT = Path(__file__).resolve().parents[2]

_MATRIX_NODEID = (
    "tests/integrations/test_agents_matrix.py::TestOpenCode::test_smoke[deepseek]"
)
_NON_MATRIX_NODEID = "tests/test_model_auto_config.py::TestHy3::test_x[hy3]"

# A sentinel standing in for pytest's ``report.wasxfail`` string (its value is
# the xfail reason; only "is not None" matters to the policy).
_WASXFAIL = "expected architectural tool-emission gap"


def test_strict_xfail_that_actually_xfailed_is_exempt():
    """Body ran and xfailed on a strict cell → approved exception → not failed."""
    assert (
        matrix_conftest._skip_should_become_failure(
            nodeid=_MATRIX_NODEID,
            wasxfail=_WASXFAIL,
            has_strict_marker=True,
        )
        is False
    )


def test_strict_xfail_that_only_skipped_on_prereq_is_not_exempt():
    """Missing prerequisite (wasxfail unset) on a strict cell → must fail.

    This is the codex-flagged hole: without the ``wasxfail`` requirement, a
    strict-xfail cell that skips because the server/client/host was absent
    would be silently exempt and the matrix would pass with missing coverage.
    """
    assert (
        matrix_conftest._skip_should_become_failure(
            nodeid=_MATRIX_NODEID,
            wasxfail=None,
            has_strict_marker=True,
        )
        is True
    )


def test_non_strict_xfail_is_not_exempt():
    """wasxfail set but no strict marker (non-strict/dynamic xfail) → must fail."""
    assert (
        matrix_conftest._skip_should_become_failure(
            nodeid=_MATRIX_NODEID,
            wasxfail=_WASXFAIL,
            has_strict_marker=False,
        )
        is True
    )


def test_plain_skip_on_matrix_cell_is_not_exempt():
    """A plain skip (no xfail at all) on a matrix cell → must fail."""
    assert (
        matrix_conftest._skip_should_become_failure(
            nodeid=_MATRIX_NODEID,
            wasxfail=None,
            has_strict_marker=False,
        )
        is True
    )


def test_skip_on_non_matrix_test_is_left_alone():
    """A skip outside the matrix modules is never upgraded, whatever its shape."""
    for wasxfail in (None, _WASXFAIL):
        for strict in (True, False):
            assert (
                matrix_conftest._skip_should_become_failure(
                    nodeid=_NON_MATRIX_NODEID,
                    wasxfail=wasxfail,
                    has_strict_marker=strict,
                )
                is False
            )


# --------------------------------------------------------------------------- #
# End-to-end: the real hook must fail a skipped cell under NO_SKIPS.
# --------------------------------------------------------------------------- #
#
# The unit tests above exercise the pure decision helper; this subprocess test
# proves the ``pytest_runtest_makereport`` hook is actually WIRED to it, so
# deleting or breaking the hook cannot leave the no-skip suite green (codex
# review). With no server reachable, every matrix cell skips at the
# family-guard fixture; under ``RAPID_MLX_MATRIX_NO_SKIPS=1`` the hook must
# convert those skips to failures and the pytest process must exit non-zero.


# pytest exit codes (from ``pytest.ExitCode``): 0=all passed, 1=tests failed,
# 2=interrupted, 3=internal error, 4=usage error, 5=no tests collected. We
# require EXACTLY 1 (a genuine test failure) so a collection error / conftest
# crash / usage error cannot make the assertion pass on a non-zero fluke.
_EXIT_TESTS_FAILED = 1
_EXIT_ALL_PASSED = 0
# The exact message the no-skip hook injects when it converts a skip.
_HOOK_MESSAGE = "release artifact matrix forbids skipped cells"

_PLAIN_CELL = (
    "tests/integrations/test_agents_matrix.py::TestCodexCLI::test_smoke[deepseek]"
)


def test_no_skips_hook_fails_a_skipped_cell_end_to_end():
    result = _run_pytest_cell(_PLAIN_CELL, {"RAPID_MLX_MATRIX_NO_SKIPS": "1"})
    combined = result.stdout + result.stderr
    assert result.returncode == _EXIT_TESTS_FAILED, (
        "NO_SKIPS run of a skipped matrix cell must exit with pytest code 1 "
        f"(tests failed), got {result.returncode} — a non-1 non-zero code "
        "means a collection/internal/usage error, not the hook doing its job.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert _HOOK_MESSAGE in combined, (
        "the failure must carry the no-skip hook's specific message "
        f"({_HOOK_MESSAGE!r}) — otherwise the cell failed for an unrelated "
        f"reason and this test is not proving the hook is wired.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def test_no_skips_off_leaves_a_skipped_cell_green_end_to_end():
    # Sanity: without the env var, the same skipped cell stays a green skip
    # (the gate only bites when the release runner asks for it).
    result = _run_pytest_cell(_PLAIN_CELL, env={})
    assert result.returncode == _EXIT_ALL_PASSED, (
        "Without NO_SKIPS, a skipped matrix cell should stay green (exit 0), "
        f"but pytest exited {result.returncode}.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert _HOOK_MESSAGE not in (result.stdout + result.stderr), (
        "the no-skip hook must NOT fire when RAPID_MLX_MATRIX_NO_SKIPS is unset."
    )


def _run_pytest_cell(nodeid: str, env: dict[str, str]) -> subprocess.CompletedProcess:
    import os

    run_env = os.environ.copy()
    # Ensure no operator server is picked up: point at an unroutable port so
    # the family guard skips every cell deterministically.
    run_env["RAPID_MLX_BASE_URL"] = "http://127.0.0.1:1/v1"
    run_env.update(env)
    return subprocess.run(
        [sys.executable, "-m", "pytest", nodeid, "-p", "no:cacheprovider", "-q"],
        cwd=str(_REPO_ROOT),
        env=run_env,
        capture_output=True,
        text=True,
        timeout=120,
    )
