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

from pathlib import Path

import pytest

from scripts.pr_validate.context import Context
from scripts.pr_validate.steps.supply_chain import (
    _WORKFLOW_PREFIX,
    SupplyChainStep,
    _roster_only_workflows,
)

# HEAD content of the real workflow roster, read from the working tree. The
# fixtures below are line-tuned to it (roster spans 432-487), so the pure
# parser tests validate against the same ground truth the step reads at run
# time. If the roster ever moves, these tests fail loudly and the fixtures
# must be re-synced.
_REPO = Path(__file__).resolve().parents[1]
_WORKFLOW_HEAD = {
    ".github/workflows/ci.yml": (_REPO / ".github/workflows/ci.yml").read_text(),
}

# The test file an external "I added a test" PR enrolls.
_ENROLLED = "tests/test_serving_lane_reason_contract.py"

# The diff of the NEWLY-created test file being enrolled (``new file mode``
# makes it count as a fresh contribution — required for the downgrade).
_NEW_TEST_DIFF = f"""\
diff --git a/{_ENROLLED} b/{_ENROLLED}
new file mode 100644
index 0000000..1111111
--- /dev/null
+++ b/{_ENROLLED}
@@ -0,0 +1,3 @@
+"\"test contract run on the new lane\"\"
+from test_stuff import run

"""

# A unified diff adding ONE line to the Linux test roster in ci.yml — the
# exact "encourage me" case from issue #2522 / PR #2514. Includes the new
# test file so the enrollment points at a genuinely new contribution.
_ROSTER_ONLY_DIFF = f"""\
{_NEW_TEST_DIFF}diff --git a/.github/workflows/ci.yml b/.github/workflows/ci.yml
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
{_NEW_TEST_DIFF}diff --git a/.github/workflows/ci.yml b/.github/workflows/ci.yml
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

# A "roster-only" diff that ALSO removes a line whose content begins with
# ``-- ``, so the diff line starts ``--- ``, colliding with the file-header
# marker. The parser must NOT drop that removed line once inside the hunk —
# otherwise a structural change gets misclassified as roster-only and
# downgraded (regression for codex r1 finding #1).
_COLLIDING_HEADER_DIFF = f"""\
{_NEW_TEST_DIFF}diff --git a/.github/workflows/ci.yml b/.github/workflows/ci.yml
index 1111111..2222222 100644
--- a/.github/workflows/ci.yml
+++ b/.github/workflows/ci.yml
@@ -8 +8,0 @@
--- --quiet-node
@@ -441 +442,2 @@
             tests/test_mllm_hybrid_probe.py \\
+            {_ENROLLED} \\
"""

# An enrollment of an EXISTING (modified, not new) test file — must NOT get
# the downgrade, because the exemption is for "I added a new test".
_MODIFIED_TEST_DIFF = """\
diff --git a/tests/test_existing.py b/tests/test_existing.py
index 1111111..2222222 100644
--- a/tests/test_existing.py
+++ b/tests/test_existing.py
@@ -1 +1 @@
-import os
+import os  # touched, not new
diff --git a/.github/workflows/ci.yml b/.github/workflows/ci.yml
index 1111111..2222222 100644
--- a/.github/workflows/ci.yml
+++ b/.github/workflows/ci.yml
@@ -441 +441,2 @@
             tests/test_mllm_hybrid_probe.py \\
+            tests/test_existing.py \\
"""

_WORKFLOW = ".github/workflows/ci.yml"

# A NEW test is genuinely added, but its ``tests/foo.py \`` line is inserted
# at the top of the workflow file (NOT inside the explicit ``pytest \`` test
# roster). It must NOT qualify as roster-only — the token alone, outside the
# roster list, is not an enrollment (codex r1 #2).
_NON_ROSTER_CONTEXT_DIFF = f"""\
{_NEW_TEST_DIFF}diff --git a/.github/workflows/ci.yml b/.github/workflows/ci.yml
index 1111111..2222222 100644
--- a/.github/workflows/ci.yml
+++ b/.github/workflows/ci.yml
@@ -1,3 +1,4 @@
 name: CI
 on: [push]
+tests/test_serving_lane_reason_contract.py \\
 jobs:
"""

# A second new test file, enrolled alongside the primary one — used by the
# hunk-boundary fixture below so BOTH hunks carry a fully-legitimate
# new-file enrollment under the (buggy) flat-list logic.
_OTHER_NEW = "tests/test_first_new.py"
_OTHER_NEW_DIFF = f"""\
diff --git a/{_OTHER_NEW} b/{_OTHER_NEW}
new file mode 100644
index 0000000..2222222
--- /dev/null
+++ b/{_OTHER_NEW}
@@ -0,0 +1 @@
+pass

"""

# Regression for codex r1 #2 (hunk boundaries): the FIRST added line of a
# later hunk is anchored ONLY if we ignore hunk boundaries. Hunk 1 is a real
# roster append (context `tests/...py \` + added entry); hunk 2's first line
# is a roster-shaped token placed OUTSIDE the roster — under the buggy flat
# list it would inherit hunk 1's roster anchor and be misclassified. The new
# code rejects any added line that opens its own hunk.
_HUNK_BOUNDARY_DIFF = f"""\
{_NEW_TEST_DIFF}{_OTHER_NEW_DIFF}diff --git a/.github/workflows/ci.yml b/.github/workflows/ci.yml
index 1111111..2222222 100644
--- a/.github/workflows/ci.yml
+++ b/.github/workflows/ci.yml
@@ -441 +441,2 @@
             tests/test_mllm_hybrid_probe.py \\
+            {_OTHER_NEW} \\
@@ -1 +2,2 @@
+            {_ENROLLED} \\
"""

# Regression for codex r1 #2 (opening pytest command must continue): a line
# following a plain ``pytest -q`` (NO trailing backslash) runs as its own
# shell command and does NOT open the multi-line roster, so an added
# ``tests/foo.py \`` after it is not an enrollment.
_NON_CONTINUING_PYTEST_DIFF = f"""\
{_NEW_TEST_DIFF}diff --git a/.github/workflows/ci.yml b/.github/workflows/ci.yml
index 1111111..2222222 100644
--- a/.github/workflows/ci.yml
+++ b/.github/workflows/ci.yml
@@ -354,1 +354,2 @@
          pytest -q
+            {_ENROLLED} \\
"""

# Regression for codex r1 round-3: extended diff metadata on the workflow
# (here a mode change) is structural and disqualifies a file even when it also
# carries a valid roster addition.
_MODE_CHANGE_DIFF = f"""\
{_NEW_TEST_DIFF}diff --git a/.github/workflows/ci.yml b/.github/workflows/ci.yml
old mode 100644
index 1111111..2222222 100644
--- a/.github/workflows/ci.yml
+++ b/.github/workflows/ci.yml
@@ -441 +441,2 @@
             tests/test_mllm_hybrid_probe.py \\
+            {_ENROLLED} \\
"""

# Regression for codex r1 round-2: a rename of the workflow file, even with a
# roster addition, is not a roster-only enrollment.
_RENAME_DIFF = f"""\
{_NEW_TEST_DIFF}diff --git a/.github/workflows/ci.yml b/.github/workflows/renamed.yml
similarity index 99%
rename from .github/workflows/ci.yml
rename to .github/workflows/renamed.yml
index 1111111..2222222 100644
--- a/.github/workflows/ci.yml
+++ b/.github/workflows/renamed.yml
@@ -441 +441,2 @@
             tests/test_mllm_hybrid_probe.py \\
+            {_ENROLLED} \\
"""

# Regression for codex r1 round-2 (non-pytest command anchor): the added
# ``tests/foo.py \\`` sits inside a DIFFERENT multiline command (an uploader
# receiving test-file arguments), not the pytest roster. Its continuation
# chain traces back to ``upload-artifact \\`` — not a ``pytest \\`` opener —
# so it must NOT be downgraded.
_NON_PYTEST_COMMAND_DIFF = f"""\
{_NEW_TEST_DIFF}diff --git a/.github/workflows/ci.yml b/.github/workflows/ci.yml
index 1111111..2222222 100644
--- a/.github/workflows/ci.yml
+++ b/.github/workflows/ci.yml
@@ -198,4 +198,5 @@
             upload-artifact \\
               tests/test_asset_1.py \\
               tests/test_asset_2.py \\
+              {_ENROLLED} \\
"""


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
        _ROSTER_ONLY_DIFF, {_WORKFLOW, _ENROLLED}, _WORKFLOW_HEAD
    )
    assert roster_only == {_WORKFLOW}
    assert additions[_WORKFLOW] == [_ENROLLED]


def test_roster_only_addition_not_in_files_changed_stays_blocking(tmp_path):
    # The roster line names a test the PR did NOT add — not an enrollment.
    roster_only, _ = _roster_only_workflows(
        _ROSTER_ONLY_DIFF, {_WORKFLOW}, _WORKFLOW_HEAD
    )
    assert roster_only == set()


def test_mixed_edit_not_roster_only(tmp_path):
    roster_only, _ = _roster_only_workflows(
        _MIXED_DIFF, {_WORKFLOW, _ENROLLED}, _WORKFLOW_HEAD
    )
    assert roster_only == set()


def test_arbitrary_workflow_edit_not_roster_only(tmp_path):
    roster_only, _ = _roster_only_workflows(
        _ARBITRARY_DIFF, {_WORKFLOW}, _WORKFLOW_HEAD
    )
    assert roster_only == set()


def test_removed_line_colliding_with_header_marker_not_roster_only(tmp_path):
    """Regression (codex r1 #1): a removed line whose content begins with
    ``-- `` produces a diff line starting ``--- ``, which must be treated as
    a real removed change once inside the hunk — NOT skipped as a file header.
    Otherwise a workflow with an additional structural change could be
    misclassified as roster-only."""
    roster_only, _ = _roster_only_workflows(
        _COLLIDING_HEADER_DIFF, {_WORKFLOW, _ENROLLED}, _WORKFLOW_HEAD
    )
    assert roster_only == set()


def test_enrolling_modified_existing_test_not_roster_only(tmp_path):
    """Regression (codex r1 #2): the downgrade is for a NEWLY added test, so
    enrolling a test this PR merely edits must not qualify."""
    roster_only, _ = _roster_only_workflows(
        _MODIFIED_TEST_DIFF, {_WORKFLOW, "tests/test_existing.py"}, _WORKFLOW_HEAD
    )
    assert roster_only == set()


def test_roster_token_outside_roster_list_not_roster_only(tmp_path):
    """Regression (codex r1 #2): a ``tests/foo.py \\`` line added at the top of
    the workflow (outside the `pytest \\` roster) is not an enrollment, even
    though the token and new-file checks pass — it must NOT be downgraded."""
    roster_only, _ = _roster_only_workflows(
        _NON_ROSTER_CONTEXT_DIFF, {_WORKFLOW, _ENROLLED}, _WORKFLOW_HEAD
    )
    assert roster_only == set()


def test_hunk_boundary_not_roster_only(tmp_path):
    """Regression (codex r1 #2 / hunk boundaries): a roster token that is the
    FIRST added line of its own hunk must not inherit an anchor from the
    previous hunk's final roster line. Without this, a token placed outside
    the roster could be misclassified and downgraded."""
    roster_only, _ = _roster_only_workflows(
        _HUNK_BOUNDARY_DIFF, {_WORKFLOW, _ENROLLED, _OTHER_NEW}, _WORKFLOW_HEAD
    )
    assert roster_only == set()


def test_non_continuing_pytest_command_not_roster_only(tmp_path):
    """Regression (codex r1 #2): an added ``tests/foo.py \\`` line directly
    after a `pytest -q` command with NO continuation backslash is not in the
    multi-line roster and must not be treated as an enrollment."""
    roster_only, _ = _roster_only_workflows(
        _NON_CONTINUING_PYTEST_DIFF, {_WORKFLOW, _ENROLLED}, _WORKFLOW_HEAD
    )
    assert roster_only == set()


def test_mode_change_with_roster_addition_not_roster_only(tmp_path):
    """Regression (codex r1 round-2): a mode change on the workflow file is
    structural, so even with a valid-looking roster addition the file must
    not be downgraded."""
    roster_only, _ = _roster_only_workflows(
        _MODE_CHANGE_DIFF, {_WORKFLOW, _ENROLLED}, _WORKFLOW_HEAD
    )
    assert roster_only == set()


def test_rename_with_roster_addition_not_roster_only(tmp_path):
    """Regression (codex r1 round-2): renaming the workflow file is not a
    roster-only enrollment even combined with a roster addition."""
    roster_only, _ = _roster_only_workflows(
        _RENAME_DIFF,
        {_WORKFLOW, "tests/test_serving_lane_reason_contract.py"},
        _WORKFLOW_HEAD,
    )
    assert roster_only == set()


def test_non_roster_line_not_roster_only(tmp_path):
    """Regression (codex r1 round-3): an added ``tests/foo.py \\`` line whose
    target line number is NOT an actual pytest-roster entry in the merged file
    (e.g. inside an unrelated multiline command an uploader) is not an
    enrollment — verified against ground truth, not hunk context."""
    roster_only, _ = _roster_only_workflows(
        _NON_PYTEST_COMMAND_DIFF, {_WORKFLOW, _ENROLLED}, _WORKFLOW_HEAD
    )
    assert roster_only == set()


def test_new_files_detects_created_file():
    from scripts.pr_validate.steps.supply_chain import _new_files

    assert _NEW_TEST_DIFF.split("diff --git a/")[1]  # sanity: top block is the new file
    new = _new_files(_ROSTER_ONLY_DIFF)
    assert _ENROLLED in new


def test_mid_list_no_backslash_entry_is_not_roster_only():
    """Regression (codex r1 round-4): a no-backslash ``tests/x.py`` line
    sandwiched BEFORE an existing continuing roster entry would cut the
    pytest command short and run later paths as separate commands — it is
    not a terminal enrollment and must NOT be downgraded."""
    from scripts.pr_validate.steps.supply_chain import _pytest_roster_lines

    head = (
        "name: CI\n"
        "jobs:\n"
        "  t:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - name: Run MLX-dependent tests\n"
        "        run: |\n"
        "          pytest \\\n"
        "            tests/test_alpha.py \\\n"
        "            tests/test_mid.py\n"  # no backslash
        "            tests/test_beta.py \\\n"
    )
    lines = _pytest_roster_lines(head)
    # line 10 (the sandwiched no-backslash entry) must NOT count as a genuine
    # roster terminal (it would cut the pytest command and run beta as a
    # separate command).
    assert 10 not in lines

    # End-to-end: adding that no-backslash line at position 9 is not a valid
    # enrollment and stays BLOCKING (the path is new, but the position is not
    # a genuine roster terminal). The added test must be "new" to isolate the
    # position check.
    newfile = (
        "diff --git a/tests/test_mid.py b/tests/test_mid.py\n"
        "new file mode 100644\n"
        "index 0000000..9999999\n"
        "--- /dev/null\n"
        "+++ b/tests/test_mid.py\n"
        "@@ -0,0 +1 @@\n"
        "+pass\n"
    )
    inserted = (
        "diff --git a/.github/workflows/ci.yml b/.github/workflows/ci.yml\n"
        "--- a/.github/workflows/ci.yml\n"
        "+++ b/.github/workflows/ci.yml\n"
        "@@ -9 +10,2 @@\n"
        "+            tests/test_mid.py\n"
        "             tests/test_beta.py \\\n"
    )
    roster_only, _ = _roster_only_workflows(
        newfile + inserted,
        {".github/workflows/ci.yml", "tests/test_mid.py"},
        {".github/workflows/ci.yml": head},
    )
    assert roster_only == set()


def test_terminal_roster_entry_without_backslash_is_enrollment():
    """A genuine final argument at the end of the literal block is safe."""
    from scripts.pr_validate.steps.supply_chain import _pytest_roster_lines

    head = (
        "name: CI\n"
        "jobs:\n"
        "  t:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - name: Run MLX-dependent tests\n"
        "        run: |\n"
        "          pytest \\\n"
        "            tests/test_alpha.py \\\n"
        "            tests/test_terminal.py\n"
    )
    # A no-backslash final entry added at the same position.
    diff = (
        "diff --git a/.github/workflows/ci.yml b/.github/workflows/ci.yml\n"
        "--- a/.github/workflows/ci.yml\n"
        "+++ b/.github/workflows/ci.yml\n"
        "@@ -9 +9,2 @@\n"
        "            tests/test_alpha.py \\\n"
        "+            tests/test_appended.py\n"
    )
    lines = _pytest_roster_lines(head)
    # alpha (line 9) is continuing; terminal (line 10) has no backslash but is
    # the genuine end of the literal block.
    assert 10 in lines

    # End-to-end: the genuinely terminal no-backslash edit is an enrollment.
    newfile = (
        "diff --git a/tests/test_appended.py b/tests/test_appended.py\n"
        "new file mode 100644\n"
        "index 0000000..1111111\n"
        "--- /dev/null\n"
        "+++ b/tests/test_appended.py\n"
        "@@ -0,0 +1 @@\n"
        "+pass\n"
    )
    roster_only, additions = _roster_only_workflows(
        newfile + diff,
        {".github/workflows/ci.yml", "tests/test_appended.py"},
        {".github/workflows/ci.yml": head},
    )
    assert roster_only == {".github/workflows/ci.yml"}
    assert additions[".github/workflows/ci.yml"] == ["tests/test_appended.py"]


def test_no_backslash_before_pytest_options_stays_blocking():
    """Regression: a missing continuation before existing pytest options
    terminates pytest and turns those options into shell commands.  It must
    never receive the roster-only downgrade."""
    from scripts.pr_validate.steps.supply_chain import _pytest_roster_lines

    path = "tests/test_new_external.py"
    head = (
        "- name: Run MLX-dependent tests\n"
        "  run: |\n"
        "    pytest \\\n"
        "      tests/test_existing.py \\\n"
        f"      {path}\n"
        "      -v --tb=short \\\n"
        "      --cov=rapid_mlx\n"
    )
    line = 5
    assert line not in _pytest_roster_lines(head)

    newfile = (
        f"diff --git a/{path} b/{path}\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        f"+++ b/{path}\n"
        "@@ -0,0 +1 @@\n"
        "+pass\n"
    )
    workflow = (
        "diff --git a/.github/workflows/ci.yml b/.github/workflows/ci.yml\n"
        "--- a/.github/workflows/ci.yml\n"
        "+++ b/.github/workflows/ci.yml\n"
        "@@ -4 +5,2 @@\n"
        f"+  {path}\n"
        "   -v --tb=short \\\n"
    )
    roster_only, additions = _roster_only_workflows(
        newfile + workflow,
        {".github/workflows/ci.yml", path},
        {".github/workflows/ci.yml": head},
    )
    assert roster_only == set()
    assert additions == {}


def test_unrelated_pytest_step_in_ci_stays_blocking():
    """Matching command syntax elsewhere in ci.yml is not privileged."""
    from scripts.pr_validate.steps.supply_chain import _pytest_roster_lines

    head = (
        "- name: Some other tests\n"
        "  run: |\n"
        "    pytest \\\n"
        "      tests/test_unrelated.py \\\n"
        "      -q\n"
    )
    assert _pytest_roster_lines(head) == set()


def test_other_workflow_pytest_roster_stays_blocking():
    """The narrow exception belongs only to ci.yml's reviewed Apple/MLX
    roster, not arbitrary pytest commands in other executable workflows."""
    workflow_path = ".github/workflows/rapid-mac-ci.yml"
    test_path = "tests/test_new_external.py"
    head = f"pytest \\\n  {test_path} \\\n  -q\n"
    diff = (
        f"diff --git a/{test_path} b/{test_path}\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        f"+++ b/{test_path}\n"
        "@@ -0,0 +1 @@\n"
        "+pass\n"
        f"diff --git a/{workflow_path} b/{workflow_path}\n"
        f"--- a/{workflow_path}\n"
        f"+++ b/{workflow_path}\n"
        "@@ -1 +2,2 @@\n"
        f"+  {test_path} \\\n"
        "   -q\n"
    )
    roster_only, additions = _roster_only_workflows(
        diff,
        {workflow_path, test_path},
        {workflow_path: head},
    )
    assert roster_only == set()
    assert additions == {}


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


def test_external_roster_only_unreadable_workflow_stays_blocking(tmp_path, monkeypatch):
    """Regression (codex r1 round-3): if the workflow file cannot be read
    (missing / decode / permission), validation must NOT crash and must
    conservatively treat the change as NOT roster-only → [BLOCKING]."""
    real_read_text = Path.read_text

    def _flaky(self, *a, **k):
        if ".github/workflows/" in str(self).replace("\\", "/"):
            raise OSError("denied")
        return real_read_text(self, *a, **k)

    monkeypatch.setattr("scripts.pr_validate.steps.supply_chain.Path.read_text", _flaky)
    ctx = _ctx(
        _ROSTER_ONLY_DIFF,
        [_WORKFLOW, _ENROLLED],
        external=True,
        tmp_path=tmp_path,
    )
    result = SupplyChainStep().run(ctx)  # must not raise
    assert result.status == "fail"
    assert "[BLOCKING]" in " ".join(result.findings)


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
