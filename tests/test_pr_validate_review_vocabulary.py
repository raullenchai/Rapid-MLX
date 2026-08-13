# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from scripts.pr_validate.steps.review_vocabulary import _find_directives


def test_finds_directives_only_on_added_lines(tmp_path: Path):
    old_directive = "review " + "BLOCKING"
    new_directive = "codex_review " + "NIT-2"
    diff = tmp_path / "pr.diff"
    diff.write_text(
        "diff --git a/a.py b/a.py\n"
        "--- a/a.py\n"
        "+++ b/a.py\n"
        "@@ -10,2 +10,3 @@\n"
        f"-# {old_directive} old text\n"
        "+# raised during review\n"
        f"+# {new_directive} new text\n"
        " context\n"
    )
    assert _find_directives(diff) == [f"a.py:11: # {new_directive} new text"]


def test_allows_ordinary_review_provenance(tmp_path: Path):
    diff = tmp_path / "pr.diff"
    diff.write_text(
        "diff --git a/a.swift b/a.swift\n"
        "--- a/a.swift\n"
        "+++ b/a.swift\n"
        "@@ -0,0 +1,2 @@\n"
        "+// Found during PR #123 review round 2.\n"
        "+let value = 1\n"
    )
    assert _find_directives(diff) == []


def test_step_is_registered():
    from scripts.pr_validate.runner import STEPS

    assert "review_vocabulary" in [step.name for step in STEPS]
    names = [step.name for step in STEPS]
    assert names.index("review_vocabulary") < names.index("codex_review")
