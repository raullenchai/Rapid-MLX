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

from tests.integrations import conftest as matrix_conftest

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
