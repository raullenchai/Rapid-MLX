# SPDX-License-Identifier: Apache-2.0
"""Tests for the supply-chain step's roster-only workflow exception (#2522).

Issue #2522: an external contributor who adds a NEW test and enrolls it in
the explicit CI test roster (appending ``tests/<name>.py \\`` to the list in
``.github/workflows/ci.yml``) was getting ``[BLOCKING]`` — the gate was
blocking the exact contribution it exists to encourage. The fix downgrades a
PURELY roster-only workflow edit (only ``tests/<name>.py \\`` lines added, no
removed lines, no other workflow edit) from ``[BLOCKING]`` to ``[warning]``
for external authors, while keeping every other external workflow edit at
``[BLOCKING]``.

These tests are the contract that drove the change. The roster-only path and
the mixed / arbitrary paths must stay distinct.
"""

from __future__ import annotations

import pytest

from scripts.pr_validate.context import Context
from scripts.pr_validate.steps.supply_chain import (
    _WORKFLOW_PREFIX,
    SupplyChainStep,
    _roster_only_workflows,
)

# The test file an external "I added a test" PR enrolls.
_ENROLLED = "tests/test_serving_lane_reason_contract.py"

# A unified diff adding ONE line to the Linux test roster in ci.yml — the
# exact "encourage me" case from issue #2522 / PR #2514.
_ROSTER_ONLY_DIFF = f"""\
diff --git a/.github/workflows/ci.yml b/.github/workflows/ci.yml
index 1111111..2222222 100644
--- a/.github/workflows/ci.yml
+++ b/.github/workflows/ci.yml
@@ -441 +441,2 @@
             tests/test_mllm_hybrid_probe.py \\
+            {_ENROLLED} \\
"""

# Same roster addition PLUS a real structural edit to the same workflow
# (swapping ``runs-on``) — a mixed case that must still BLOCK.
_MIXED_DIFF = f"""\
diff --git a/.github/workflows/ci.yml b/.github/workflows/ci.yml
index 1111111..2222222 100644
--- a/.github/workflows/ci.yml
+++ b/.github/workflows/ci.yml
@@ -80 +80 @@
-            runs-on: ubuntu-latest
+            runs-on: macos-14
@@ -441 +441,2 @@
             tests/test_mllm_hybrid_probe.py \\
+            {_ENROLLED} \\
"""

# An arbitrary workflow edit alone (no roster) — must still BLOCK.
_ARBITRARY_DIFF = """\
diff --git a/.github/workflows/ci.yml b/.github/workflows/ci.yml
index 1111111..2222222 100644
--- a/.github/workflows/ci.yml
+++ b/.github/workflows/ci.yml
@@ -80 +80 @@
-            runs-on: ubuntu-latest
+            runs-on: macos-14
"""

_WORKFLOW = ".github/workflows/ci.yml"


def _ctx(
    diff: str,
    files_changed: list[str],
    *,
    external: bool,
    tmp_path: pytest.TempPathFactory,
) -> Context:
    """A context pointing at a tmpfile diff with the given file set. Pytest
    runs from the repo root (which has pyproject.toml), so Context's
    ``__post_init__`` repo-root check is satisfied without chdir."""
    ctx = Context(pr_number=999, repo="x/y")
    ctx.pr_is_external = external
    ctx.files_changed = files_changed
    diff_path = tmp_path / "pr.diff"
    diff_path.write_text(diff)
    ctx.diff_path = str(diff_path)
    return ctx


# ---------------------------------------------------------------------------
# _roster_only_workflows — the pure parser
# ---------------------------------------------------------------------------


def test_roster_only_classifies_single_addition(tmp_path):
    roster_only, additions = _roster_only_workflows(
        _ROSTER_ONLY_DIFF, {_WORKFLOW, _ENROLLED}
    )
    assert roster_only == {_WORKFLOW}
    assert additions[_WORKFLOW] == [_ENROLLED]


def test_roster_only_addition_not_in_files_changed_stays_blocking(tmp_path):
    # The roster line names a test the PR did NOT add — not an enrollment.
    roster_only, _ = _roster_only_workflows(_ROSTER_ONLY_DIFF, {_WORKFLOW})
    assert roster_only == set()


def test_mixed_edit_not_roster_only(tmp_path):
    roster_only, _ = _roster_only_workflows(_MIXED_DIFF, {_WORKFLOW, _ENROLLED})
    assert roster_only == set()


def test_arbitrary_workflow_edit_not_roster_only(tmp_path):
    roster_only, _ = _roster_only_workflows(_ARBITRARY_DIFF, {_WORKFLOW})
    assert roster_only == set()


# ---------------------------------------------------------------------------
# SupplyChainStep.run — the gate behavior
# ---------------------------------------------------------------------------


def test_external_roster_only_is_warning_not_blocking(tmp_path):
    """Acceptance criterion 1: roster-only external change → warning with the
    enrolled hunk, NOT [BLOCKING]."""
    ctx = _ctx(
        _ROSTER_ONLY_DIFF,
        [_WORKFLOW, _ENROLLED],
        external=True,
        tmp_path=tmp_path,
    )
    result = SupplyChainStep().run(ctx)
    # Warnings only → the step passes (does not gate).
    assert result.status == "pass"
    assert "[BLOCKING]" not in " ".join(result.findings)
    hook = next(f for f in result.findings if "install/CI hook" in f)
    assert "[warning]" in hook
    # The diff hunk (enrolled path) must be surfaced for human eyeballs.
    assert _ENROLLED in hook


def test_internal_roster_only_is_warning(tmp_path):
    """An internal author's roster-only change is already a warning (the
    default); the fix must not regress it."""
    ctx = _ctx(
        _ROSTER_ONLY_DIFF,
        [_WORKFLOW, _ENROLLED],
        external=False,
        tmp_path=tmp_path,
    )
    result = SupplyChainStep().run(ctx)
    assert result.status == "pass"
    assert "[BLOCKING]" not in " ".join(result.findings)


def test_external_mixed_roster_plus_edit_still_blocking(tmp_path):
    """Acceptance criterion 2 / mixed case: roster addition + a real workflow
    edit keeps [BLOCKING]."""
    ctx = _ctx(
        _MIXED_DIFF,
        [_WORKFLOW, _ENROLLED],
        external=True,
        tmp_path=tmp_path,
    )
    result = SupplyChainStep().run(ctx)
    assert result.status == "fail"
    assert "[BLOCKING]" in " ".join(result.findings)


def test_external_arbitrary_workflow_edit_still_blocking(tmp_path):
    """Acceptance criterion 2: any other workflow edit alone stays BLOCKING."""
    ctx = _ctx(
        _ARBITRARY_DIFF,
        [_WORKFLOW],
        external=True,
        tmp_path=tmp_path,
    )
    result = SupplyChainStep().run(ctx)
    assert result.status == "fail"
    assert "[BLOCKING]" in " ".join(result.findings)


def test_external_roster_plus_nonworkflow_hook_still_blocking(tmp_path):
    """The exception is scoped to workflow files: a roster-only ci.yml edit
    PLUS a change to a non-workflow hook (conftest.py) must still BLOCK."""
    conftest_diff = (
        _ROSTER_ONLY_DIFF
        + """\
diff --git a/conftest.py b/conftest.py
index 3333333..4444444 100644
--- a/conftest.py
+++ b/conftest.py
@@ -1 +1,2 @@
 import os
+import subprocess
"""
    )
    ctx = _ctx(
        conftest_diff,
        [_WORKFLOW, _ENROLLED, "conftest.py"],
        external=True,
        tmp_path=tmp_path,
    )
    result = SupplyChainStep().run(ctx)
    assert result.status == "fail"
    assert "[BLOCKING]" in " ".join(result.findings)


def test_roster_only_prefix_constant_is_workflow_scope():
    """The exception deliberately targets the workflow tree, matching the
    issue's "explicit roster list in .github/workflows/ci.yml" framing."""
    assert _WORKFLOW_PREFIX == ".github/workflows/"
