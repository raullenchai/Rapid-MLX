# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the external-PR CI roster exception (#2522)."""

from __future__ import annotations

from pathlib import Path

from scripts.pr_validate.context import Context
from scripts.pr_validate.steps.supply_chain import SupplyChainStep


def _run_supply_chain(tmp_path: Path, diff: str, files: list[str]):
    diff_path = tmp_path / "pr.diff"
    diff_path.write_text(diff)
    ctx = Context(
        pr_number=2522,
        pr_is_external=True,
        diff_path=str(diff_path),
        files_changed=files,
        work_dir=tmp_path / "artifacts",
    )
    return SupplyChainStep().run(ctx)


def test_external_roster_only_workflow_change_is_warning(tmp_path):
    diff = """diff --git a/.github/workflows/ci.yml b/.github/workflows/ci.yml
index 1111111..2222222 100644
--- a/.github/workflows/ci.yml
+++ b/.github/workflows/ci.yml
@@ -431,6 +431,7 @@ jobs:
           pytest \\
             tests/test_server.py \\
+            tests/test_new_contract.py \\
             tests/test_paged_cache.py \\
             -v --tb=short
"""

    result = _run_supply_chain(tmp_path, diff, [".github/workflows/ci.yml"])

    assert result.status == "pass"
    assert result.summary == "1 warning(s) — human review wanted"
    assert len(result.findings) == 1
    finding = result.findings[0]
    assert "[BLOCKING]" not in finding
    assert "only adds inert test path(s)" in finding
    assert "@@ -431,6 +431,7 @@ jobs:" in finding
    assert "+            tests/test_new_contract.py \\" in finding


def test_external_mixed_workflow_change_remains_blocking(tmp_path):
    diff = """diff --git a/.github/workflows/ci.yml b/.github/workflows/ci.yml
index 1111111..2222222 100644
--- a/.github/workflows/ci.yml
+++ b/.github/workflows/ci.yml
@@ -431,6 +431,8 @@ jobs:
           pytest \\
             tests/test_server.py \\
+            tests/test_new_contract.py \\
+            --disable-warnings \\
             tests/test_paged_cache.py \\
             -v --tb=short
"""

    result = _run_supply_chain(tmp_path, diff, [".github/workflows/ci.yml"])

    assert result.status == "fail"
    assert any("[BLOCKING]" in finding for finding in result.findings)
    assert not any(
        "only adds inert test path(s)" in finding for finding in result.findings
    )


def test_external_roster_removal_remains_blocking(tmp_path):
    diff = """diff --git a/.github/workflows/ci.yml b/.github/workflows/ci.yml
index 1111111..2222222 100644
--- a/.github/workflows/ci.yml
+++ b/.github/workflows/ci.yml
@@ -431,6 +431,6 @@ jobs:
           pytest \\
-            tests/test_old_contract.py \\
+            tests/test_new_contract.py \\
             tests/test_paged_cache.py \\
             -v --tb=short
"""

    result = _run_supply_chain(tmp_path, diff, [".github/workflows/ci.yml"])

    assert result.status == "fail"
    assert any("[BLOCKING]" in finding for finding in result.findings)


def test_tests_like_command_outside_roster_remains_blocking(tmp_path):
    diff = """diff --git a/.github/workflows/ci.yml b/.github/workflows/ci.yml
index 1111111..2222222 100644
--- a/.github/workflows/ci.yml
+++ b/.github/workflows/ci.yml
@@ -20,3 +20,4 @@ jobs:
         run: |
           echo preparing \\
+          tests/payload.py \\
           echo done
"""

    result = _run_supply_chain(tmp_path, diff, [".github/workflows/ci.yml"])

    assert result.status == "fail"
    assert any("[BLOCKING]" in finding for finding in result.findings)


def test_roster_path_traversal_remains_blocking(tmp_path):
    diff = """diff --git a/.github/workflows/ci.yml b/.github/workflows/ci.yml
index 1111111..2222222 100644
--- a/.github/workflows/ci.yml
+++ b/.github/workflows/ci.yml
@@ -431,3 +431,4 @@ jobs:
           pytest \\
+            tests/../../scripts/payload.py \\
             tests/test_server.py \\
"""

    result = _run_supply_chain(tmp_path, diff, [".github/workflows/ci.yml"])

    assert result.status == "fail"
    assert any("[BLOCKING]" in finding for finding in result.findings)


def test_unmatched_quote_and_missing_continuation_remain_blocking(tmp_path):
    unsafe_entries = ("'tests/test_bad.py \\", "tests/test_bad.py")
    for entry in unsafe_entries:
        diff = f"""diff --git a/.github/workflows/ci.yml b/.github/workflows/ci.yml
index 1111111..2222222 100644
--- a/.github/workflows/ci.yml
+++ b/.github/workflows/ci.yml
@@ -431,3 +431,4 @@ jobs:
           pytest \\
+            {entry}
             tests/test_server.py \\
"""

        result = _run_supply_chain(tmp_path, diff, [".github/workflows/ci.yml"])

        assert result.status == "fail"
        assert any("[BLOCKING]" in finding for finding in result.findings)


def test_workflow_rename_with_roster_addition_remains_blocking(tmp_path):
    diff = """diff --git a/.github/workflows/old.yml b/.github/workflows/ci.yml
similarity index 95%
rename from .github/workflows/old.yml
rename to .github/workflows/ci.yml
index 1111111..2222222 100644
--- a/.github/workflows/old.yml
+++ b/.github/workflows/ci.yml
@@ -431,3 +431,4 @@ jobs:
           pytest \\
+            tests/test_new_contract.py \\
             tests/test_server.py \\
"""

    result = _run_supply_chain(tmp_path, diff, [".github/workflows/ci.yml"])

    assert result.status == "fail"
    assert any("[BLOCKING]" in finding for finding in result.findings)


def test_hunk_content_cannot_spoof_diff_file_header(tmp_path):
    diff = """diff --git a/.github/workflows/ci.yml b/.github/workflows/ci.yml
index 1111111..2222222 100644
--- a/.github/workflows/ci.yml
+++ b/.github/workflows/ci.yml
@@ -431,3 +431,6 @@ jobs:
           pytest \\
+            tests/test_new_contract.py \\
+++ b/.github/workflows/ci.yml
+permissions: write-all
             tests/test_server.py \\
"""

    result = _run_supply_chain(tmp_path, diff, [".github/workflows/ci.yml"])

    assert result.status == "fail"
    assert any("[BLOCKING]" in finding for finding in result.findings)
