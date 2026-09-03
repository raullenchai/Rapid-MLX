# SPDX-License-Identifier: Apache-2.0
"""Tests for the pr_validate ``supply_chain`` step, focusing on the
test-roster-enrollment exception (issue #2522).

Issue #2522: external PRs that ONLY enroll a test in the explicit CI roster
(``tests/<name>.py \\`` added to a ``.github/workflows/`` list, nothing else
changed in any workflow file) must be surfaced as a WARNING — with the added
lines — not ``[BLOCKING]``. Any other workflow edit by an external author stays
BLOCKING.
"""

from __future__ import annotations

import pytest

from scripts.pr_validate.context import Context
from scripts.pr_validate.steps.supply_chain import SupplyChainStep


@pytest.fixture
def ctx_factory(tmp_path):
    """A Context pre-wired with a diff file + changed-files plus external or
    internal author, without requiring the real GitHub fetch."""
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'fake'\n")

    def _make(
        files_changed: list[str],
        diff_text: str,
        *,
        external: bool = True,
    ) -> Context:
        diff_path = tmp_path / "pr.diff"
        diff_path.write_text(diff_text)
        ctx = Context(pr_number=1)
        ctx.files_changed = list(files_changed)
        ctx.diff_path = str(diff_path)
        ctx.pr_is_external = external
        return ctx

    return _make


def _run(ctx) -> object:
    return SupplyChainStep().run(ctx)


ROSTER_ONLY_DIFF = """\
diff --git a/.github/workflows/ci.yml b/.github/workflows/ci.yml
--- a/.github/workflows/ci.yml
+++ b/.github/workflows/ci.yml
@@ -470 +470,2 @@
             tests/test_suffix_decoding.py \\
+            tests/test_new_lane_contract.py \\
"""

MIXED_DIFF = """\
diff --git a/.github/workflows/ci.yml b/.github/workflows/ci.yml
--- a/.github/workflows/ci.yml
+++ b/.github/workflows/ci.yml
@@ -470 +470,2 @@
             tests/test_suffix_decoding.py \\
+            tests/test_new_lane_contract.py \\
+            - uses: actions/checkout@not-a-pinned-sha \\
"""


class TestRosterOnlyEnrollment:
    def test_external_roster_only_is_warning_not_blocking(self, ctx_factory):
        """#2522: an external author adding only ``tests/<name>.py \\`` to the
        roster must NOT be [BLOCKING] — it's the expected shape of a
        'I added a test' contribution."""
        ctx = ctx_factory(
            [".github/workflows/ci.yml", "tests/test_new_lane_contract.py"],
            ROSTER_ONLY_DIFF,
            external=True,
        )
        result = _run(ctx)
        assert result.status == "pass", result.summary
        # Surface it as a warning listing the enrollment line.
        body = "\n".join(result.findings)
        assert "[BLOCKING]" not in body
        assert "modifies install/CI hook(s)" in body
        assert "tests/test_new_lane_contract.py" in body

    def test_external_mixed_roster_plus_edit_stays_blocking(self, ctx_factory):
        """#2522: a workflow edit that ALSO changes something non-roster
        (e.g. an unpinned action) must remain [BLOCKING] for an external
        author."""
        ctx = ctx_factory(
            [".github/workflows/ci.yml", "tests/test_new_lane_contract.py"],
            MIXED_DIFF,
            external=True,
        )
        result = _run(ctx)
        assert result.status == "fail", result.summary
        assert any("[BLOCKING]" in f for f in result.findings)

    def test_external_unpinned_action_change_stays_blocking(self, ctx_factory):
        """#2522: a non-roster workflow change (unpinned action line) with no
        roster enrollment at all remains BLOCKING."""
        diff = """\
diff --git a/.github/workflows/ci.yml b/.github/workflows/ci.yml
--- a/.github/workflows/ci.yml
+++ b/.github/workflows/ci.yml
@@ -49 +49 @@
-            - uses: actions/checkout@v4
+            - uses: actions/checkout@main
"""
        ctx = ctx_factory(
            [".github/workflows/ci.yml", "tests/test_new_lane_contract.py"],
            diff,
            external=True,
        )
        result = _run(ctx)
        assert result.status == "fail", result.summary
        assert any("[BLOCKING]" in f for f in result.findings)

    @pytest.mark.parametrize(
        "unsafe_path",
        (
            "tests/../payload.py",
            "tests/test_ok.py;curl.py",
            "tests/$(payload).py",
            "tests/'payload'.py",
        ),
    )
    def test_shell_or_traversal_path_stays_blocking(self, ctx_factory, unsafe_path):
        """A roster-looking line must not become a shell-injection bypass."""
        diff = ROSTER_ONLY_DIFF.replace("tests/test_new_lane_contract.py", unsafe_path)
        ctx = ctx_factory(
            [".github/workflows/ci.yml", unsafe_path],
            diff,
            external=True,
        )
        result = _run(ctx)
        assert result.status == "fail", result.summary
        assert any("[BLOCKING]" in f for f in result.findings)

    def test_roster_path_not_added_by_pr_stays_blocking(self, ctx_factory):
        """The exception applies to a contributed test, not an arbitrary line."""
        ctx = ctx_factory(
            [".github/workflows/ci.yml"],
            ROSTER_ONLY_DIFF,
            external=True,
        )
        result = _run(ctx)
        assert result.status == "fail", result.summary
        assert any("[BLOCKING]" in f for f in result.findings)

    def test_other_workflow_roster_lookalike_stays_blocking(self, ctx_factory):
        """Only the explicit ci.yml roster is eligible for the exception."""
        other = ROSTER_ONLY_DIFF.replace(
            ".github/workflows/ci.yml", ".github/workflows/release.yml"
        )
        ctx = ctx_factory(
            [".github/workflows/release.yml", "tests/test_new_lane_contract.py"],
            other,
            external=True,
        )
        result = _run(ctx)
        assert result.status == "fail", result.summary
        assert any("[BLOCKING]" in f for f in result.findings)

    def test_internal_author_roster_only_is_warning(self, ctx_factory):
        """#2522 regression: even for an INTERNAL author the roster-only
        change is surfaced as a warning (not silently swallowed), but the
        status stays pass."""
        ctx = ctx_factory(
            [".github/workflows/ci.yml"],
            ROSTER_ONLY_DIFF,
            external=False,
        )
        result = _run(ctx)
        assert result.status == "pass", result.summary
        assert "modifies install/CI hook(s)" in "\n".join(result.findings)

    def test_non_workflow_hook_change_stays_blocking_for_external(self, ctx_factory):
        """#2522: the roster-only exception applies ONLY to workflow roster
        enrollment; a conftest.py hook change by an external author stays
        BLOCKING."""
        ctx = ctx_factory(
            ["tests/conftest.py"],
            "diff --git a/tests/conftest.py b/tests/conftest.py\n"
            "--- a/tests/conftest.py\n+++ b/tests/conftest.py\n"
            "@@ -1 +1 @@\n-x\n+y\n",
            external=True,
        )
        result = _run(ctx)
        assert result.status == "fail", result.summary
        assert any("[BLOCKING]" in f for f in result.findings)
