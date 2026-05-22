# SPDX-License-Identifier: Apache-2.0
"""Tests for ``TestPlanCheckStep`` — the unchecked-checkbox gate.

This step exists because PR #435 merged with ``- [ ] E2E verification
against fishloa's repro`` still unchecked in its body; that incomplete
verification meant the fix only landed for streaming clients and #427
had to be re-opened. The gate enforces the basic discipline that the
PR body's contract must be honored before merge.
"""

from __future__ import annotations

from scripts.pr_validate.context import Context
from scripts.pr_validate.steps.test_plan_check import TestPlanCheckStep


def _ctx(body: str) -> Context:
    """Build a context shell with just the PR body populated — the step
    only reads ``ctx.pr_body`` so everything else can be default.
    """
    ctx = Context(pr_number=999, repo="x/y")
    ctx.pr_body = body
    return ctx


def test_no_body_passes():
    result = TestPlanCheckStep().run(_ctx(""))
    assert result.status == "pass"


def test_body_without_checkboxes_passes():
    body = """## Summary
Add a new endpoint to clear the cache.

## Notes
- It is fast.
- It is documented.
"""
    result = TestPlanCheckStep().run(_ctx(body))
    assert result.status == "pass"
    assert "no checkbox" in result.summary


def test_all_checked_passes():
    body = """## Test plan
- [x] Unit tests pass
- [X] CI green
- [x] Manual smoke test
"""
    result = TestPlanCheckStep().run(_ctx(body))
    assert result.status == "pass"
    assert "all 3" in result.summary


def test_unchecked_item_fails():
    """The exact pattern that landed PR #435 / #427: a test plan with
    an unchecked end-to-end item that was never verified before merge.
    """
    body = """## Test plan
- [x] Unit tests pass
- [ ] E2E verification against fishloa's repro: rapid-mlx serve ...
- [x] Lint clean
"""
    result = TestPlanCheckStep().run(_ctx(body))
    assert result.status == "fail"
    assert "1/3" in result.summary
    assert "E2E verification" in result.details


def test_multiple_unchecked_lists_all():
    body = """- [ ] First item
- [x] Second item
* [ ] Third item (alternative bullet)
- [ ] Fourth item
"""
    result = TestPlanCheckStep().run(_ctx(body))
    assert result.status == "fail"
    assert "3/4" in result.summary
    # All unchecked items should appear in details
    assert "First item" in result.details
    assert "Third item" in result.details
    assert "Fourth item" in result.details


def test_alternative_bullet_form():
    """``* [ ]`` and ``- [ ]`` are both valid GitHub task-list syntax;
    both must be detected so authors can't bypass the gate by switching
    bullet style.
    """
    body = """## Test plan
* [ ] Manual repro
"""
    result = TestPlanCheckStep().run(_ctx(body))
    assert result.status == "fail"
    assert "Manual repro" in result.details


def test_long_item_body_is_truncated_in_details():
    """A test plan can have long items (commands, URLs). Details should
    keep them readable — truncate per-item at 100 chars.
    """
    long_command = "rapid-mlx serve TheCluster/Qwen3.6-35B-A3B-MLX-mixed-9bit " * 4
    body = f"- [ ] {long_command}\n"
    result = TestPlanCheckStep().run(_ctx(body))
    assert result.status == "fail"
    # The full command shouldn't appear verbatim — should be truncated.
    assert "..." in result.details


def test_leading_whitespace_still_matches():
    """Nested task lists (indented under a header) must still trigger
    the gate. PRs commonly put checkboxes under ``## Test plan``.
    """
    body = "## Test plan\n  - [ ] Indented item under heading\n"
    result = TestPlanCheckStep().run(_ctx(body))
    assert result.status == "fail"
    assert "Indented item" in result.details
