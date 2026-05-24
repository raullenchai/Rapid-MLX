# SPDX-License-Identifier: Apache-2.0
"""Tests for the Google eng-practices integration in ``pr_validate``.

Covers two new pieces of functionality:

1. ``_split_findings_by_tier`` — partitions DeepSeek findings into
   ``[BLOCKING]`` (fails the gate) vs ``[NIT]`` (surfaces but passes),
   with untagged findings defaulting to BLOCKING so a forgotten prefix
   can't silently downgrade a real bug.

2. ``CLDescriptionQualityStep`` — title + body hygiene gate from
   Google's CL-descriptions guidance. Rejects bad titles, empty
   bodies, and bodies with no rationale signal.

Motivating example: PR #467 (chat-route empty content fix) spent 5
rounds in a DeepSeek spiral because every reply produced fresh style
preferences and the previous prompt couldn't downgrade them. With
tiering, those would have been single-round ``[NIT]``s and the PR
would have merged after one round of substantive fixes.
"""

from __future__ import annotations

import pytest

from scripts.pr_validate.context import Context
from scripts.pr_validate.steps.cl_description_quality import (
    CLDescriptionQualityStep,
)
from scripts.pr_validate.steps.deepseek_review import _split_findings_by_tier

# ---------------------------------------------------------------------------
# _split_findings_by_tier
# ---------------------------------------------------------------------------


class TestSplitFindingsByTier:
    """The tier-split helper is what implements Google's "approve when
    improvement is clear" — only ``[BLOCKING]`` findings fail the gate.
    """

    def test_empty_input_returns_empty_lists(self):
        blocking, nits = _split_findings_by_tier([])
        assert blocking == []
        assert nits == []

    def test_all_blocking(self):
        findings = [
            "[BLOCKING] routes/x.py:10 — race condition on shared dict.",
            "[BLOCKING] routes/y.py:42 — missing await on async call.",
        ]
        blocking, nits = _split_findings_by_tier(findings)
        assert len(blocking) == 2
        assert nits == []
        # Prefix is stripped — keeps scorecard tidy.
        assert all(not b.startswith("[BLOCKING]") for b in blocking)
        assert "race condition" in blocking[0]

    def test_all_nit(self):
        findings = [
            "[NIT] tests/x.py:5 — could rename `val` to `value`.",
            "[NIT] tests/y.py:9 — comment could be clearer.",
        ]
        blocking, nits = _split_findings_by_tier(findings)
        assert blocking == []
        assert len(nits) == 2
        assert all(not n.startswith("[NIT]") for n in nits)

    def test_mixed_tiers(self):
        findings = [
            "[BLOCKING] a.py:1 — real bug.",
            "[NIT] b.py:2 — style.",
            "[BLOCKING] c.py:3 — another bug.",
        ]
        blocking, nits = _split_findings_by_tier(findings)
        assert len(blocking) == 2
        assert len(nits) == 1
        assert "real bug" in blocking[0]
        assert "another bug" in blocking[1]
        assert "style" in nits[0]

    def test_untagged_defaults_to_blocking(self):
        """A finding without ``[BLOCKING]``/``[NIT]`` must default to
        BLOCKING. A forgotten prefix should fail-safe (block merge) so
        the model can't accidentally downgrade a real bug by skipping
        the tag — and the reviewer notices the model forgot."""
        findings = [
            "routes/x.py:10 — missing input validation.",
            "[NIT] tests/y.py:5 — minor style.",
        ]
        blocking, nits = _split_findings_by_tier(findings)
        assert len(blocking) == 1
        assert len(nits) == 1
        # The untagged finding keeps its original text verbatim so the
        # reviewer sees the model forgot the prefix.
        assert "missing input validation" in blocking[0]

    def test_case_insensitive_prefix(self):
        findings = [
            "[blocking] a.py:1 — bug.",
            "[Nit] b.py:2 — style.",
        ]
        blocking, nits = _split_findings_by_tier(findings)
        assert len(blocking) == 1
        assert len(nits) == 1

    def test_prefix_with_extra_whitespace(self):
        """The regex tolerates leading/trailing whitespace inside the
        tag area because some models indent or pad."""
        findings = [
            "  [BLOCKING]   a.py:1 — bug.",
            "[NIT]    b.py:2 — style.",
        ]
        blocking, nits = _split_findings_by_tier(findings)
        assert len(blocking) == 1
        assert len(nits) == 1
        assert blocking[0].startswith("a.py")
        assert nits[0].startswith("b.py")


# ---------------------------------------------------------------------------
# CLDescriptionQualityStep
# ---------------------------------------------------------------------------


def _ctx(title: str = "", body: str = "") -> Context:
    """Build a context shell with just title + body — the step only
    reads those fields."""
    ctx = Context(pr_number=999, repo="x/y")
    ctx.pr_title = title
    ctx.pr_body = body
    return ctx


class TestCLDescriptionQualityTitle:
    """Title checks: empty, too-short, and bad-pattern blacklist."""

    def test_empty_title_fails(self):
        result = CLDescriptionQualityStep().run(_ctx(title="", body="Why: x"))
        assert result.status == "fail"
        assert "empty" in result.summary.lower()

    def test_one_word_title_fails(self):
        result = CLDescriptionQualityStep().run(_ctx(title="patch", body="Why: x"))
        assert result.status == "fail"
        assert "short" in result.summary.lower() or "weak" in result.summary.lower()

    def test_two_word_title_fails(self):
        """3-word minimum after CC-prefix strip. ``fix bug`` is 2 words."""
        result = CLDescriptionQualityStep().run(_ctx(title="fix bug", body="Why: x"))
        assert result.status == "fail"

    @pytest.mark.parametrize(
        "title",
        [
            "fix bug",
            "fix build",
            "fix tests",
            "add patch",
            "wip",
            "various changes",
            "small change",
            "patch",
            "update",
            "tweaks",
            "cleanup",
            "misc",
            "minor fix",
        ],
    )
    def test_bad_title_patterns_fail(self, title: str):
        """The blacklist Google explicitly calls out as bad CL
        descriptions. Each must fail the gate."""
        result = CLDescriptionQualityStep().run(_ctx(title=title, body="Why: x"))
        assert result.status == "fail", f"expected fail for title {title!r}"

    def test_cc_prefix_stripped_before_bad_pattern_check(self):
        """A bare ``fix: bug`` should fail. A proper
        ``fix(routes): default empty content to ...`` (>3 substantive
        words) must pass — the conventional-commit prefix is stripped."""
        bad = CLDescriptionQualityStep().run(_ctx(title="fix: bug", body="Why: x"))
        assert bad.status == "fail"

        good = CLDescriptionQualityStep().run(
            _ctx(
                title="fix(routes): default empty content to satisfy openai shape",
                body="Why: x",
            )
        )
        assert good.status == "pass"

    def test_cc_prefix_does_not_consume_real_word(self):
        """A title like ``feat: add new endpoint`` is 3 words after
        stripping ``feat:`` and should pass title check."""
        result = CLDescriptionQualityStep().run(
            _ctx(title="feat: add new endpoint", body="Why: x")
        )
        assert result.status == "pass"

    def test_good_title_with_long_descriptive_form(self):
        result = CLDescriptionQualityStep().run(
            _ctx(
                title="fix(api): honor max_completion_tokens on chat completions",
                body="Why: openai sdk >=1.45 stopped sending max_tokens",
            )
        )
        assert result.status == "pass"


class TestCLDescriptionQualityBody:
    """Body checks: existence + rationale-signal detection."""

    def test_empty_body_fails(self):
        result = CLDescriptionQualityStep().run(
            _ctx(title="feat: add a new endpoint", body="")
        )
        assert result.status == "fail"
        assert "body" in result.summary.lower() and "empty" in result.summary.lower()

    def test_body_with_no_rationale_signal_fails(self):
        """A body that's pure description with no ``why``, no
        ``Closes #``, no ``because`` — Google's bar fails."""
        body = "This patch changes the foo helper. It updates the bar list."
        result = CLDescriptionQualityStep().run(
            _ctx(title="feat: add foo helper update", body=body)
        )
        assert result.status == "fail"
        assert "rationale" in result.summary.lower()

    @pytest.mark.parametrize(
        "body",
        [
            "## Why\nWe need this because the api changed.",
            "## Summary\nFoo\n\n## Rationale\nBar.",
            "## Motivation\nThe legacy path was broken.",
            "## Background\nUsers reported timeouts.",
            "Why: the api shape changed in v2",
            "**Why:** the api shape changed in v2",
            "Closes #123",
            "fixes #456 — broken since last release",
            "Resolves #789",
            "Refs #100",
            "Adds the helper because the legacy path was deprecated.",
        ],
    )
    def test_body_with_rationale_signal_passes(self, body: str):
        result = CLDescriptionQualityStep().run(
            _ctx(title="feat: add the new helper", body=body)
        )
        assert result.status == "pass", f"expected pass for body {body!r}"

    def test_summary_heading_alone_is_sufficient(self):
        """A ``## Summary`` heading counts as a rationale signal —
        contributors often use the template's Summary section as their
        primary explanation."""
        body = "## Summary\n- new endpoint\n- updated docs"
        result = CLDescriptionQualityStep().run(
            _ctx(title="feat: add new endpoint", body=body)
        )
        assert result.status == "pass"


class TestCLDescriptionQualityOverride:
    """The ``PR_VALIDATE_SKIP_DESC=1`` escape hatch — for two-line
    dep-bumps where rationale is genuinely overkill."""

    def test_override_env_skips_step(self, monkeypatch):
        monkeypatch.setenv("PR_VALIDATE_SKIP_DESC", "1")
        result = CLDescriptionQualityStep().run(_ctx(title="wip", body=""))
        assert result.status == "skip"
        assert "PR_VALIDATE_SKIP_DESC" in result.summary

    def test_no_override_runs_normally(self, monkeypatch):
        monkeypatch.delenv("PR_VALIDATE_SKIP_DESC", raising=False)
        result = CLDescriptionQualityStep().run(_ctx(title="wip", body=""))
        assert result.status == "fail"
