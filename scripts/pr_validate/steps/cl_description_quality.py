# SPDX-License-Identifier: Apache-2.0
"""Step — PR description quality gate.

Enforces the basic CL-description hygiene from Google eng-practices
(https://github.com/google/eng-practices/blob/master/review/developer/cl-descriptions.md):

> "A bad CL description... 'Fix bug' is not adequate. What bug? What
> did you do to fix it? Other similarly bad descriptions include:
> 'Fix build', 'Add patch', 'Moving code from A to B', 'Phase 1'."

> "The first line should be: A short summary of what is being done.
> Complete sentence, written as though it was an order. Followed by
> a longer description... what problem is being solved, and why this
> is the best approach. Any shortcomings of the approach."

Concretely:

1. **Title is informative** — at least 3 words and is NOT one of the
   well-known bad patterns ("fix bug", "fix build", "wip", "various
   changes", "small change", "patch", "update", "tweaks").
2. **Body exists** — empty PR bodies fail (no rationale = no review
   context = future-grep loses the why).
3. **Body has rationale** — at least one of: a "## Why" / "## Summary"
   / "## Rationale" / "## Motivation" section, OR a "Closes #" /
   "Fixes #" / "Refs #" issue link, OR a `Why:` line. We're lenient
   on form but strict on the principle: the PR must explain WHY
   something is changing, not just WHAT.

Why a STEP, not a comment-only warning: every other gate is hard, so
description quality should be too. A poorly-described PR is hard to
review, hard to bisect after the fact, and signals the author hasn't
thought through the change. Failing the gate forces a 30-second
rewrite that pays back forever in code archaeology.

Override: an author who insists their two-line "Bump dep X to Y" PR
needs no rationale can use ``PR_VALIDATE_SKIP_DESC=1``. Don't make
that the norm.
"""

from __future__ import annotations

import re

from ..base import Step, StepResult
from ..context import Context, env_truthy

# Title patterns that fail. Matched against the LOWERCASED title with
# leading conventional-commit prefix (e.g. ``fix:`` / ``feat(routes):``)
# stripped. Each entry is a full-title regex anchored to ^…$ so we don't
# accidentally flag "Fix bug in X-Y-Z scheduler" — only the bare phrase.
_BAD_TITLE_PATTERNS = (
    re.compile(r"^fix\s+bug\.?$"),
    re.compile(r"^fix\s+build\.?$"),
    re.compile(r"^fix\s+tests?\.?$"),
    re.compile(r"^add\s+patch\.?$"),
    re.compile(r"^small\s+change\.?$"),
    re.compile(r"^various\s+changes\.?$"),
    re.compile(r"^various\s+fixes\.?$"),
    re.compile(r"^tweaks?\.?$"),
    re.compile(r"^update\.?$"),
    re.compile(r"^patch\.?$"),
    re.compile(r"^wip\.?$"),
    re.compile(r"^cleanup\.?$"),
    re.compile(r"^changes\.?$"),
    re.compile(r"^updates?\.?$"),
    re.compile(r"^misc\.?$"),
    re.compile(r"^minor\s+(?:fix|change|update)\.?$"),
)

# Conventional-commit prefix (e.g. ``fix(routes):``, ``feat:``,
# ``docs(benchmarks):``) is stripped before the bad-title check so the
# substantive title is what we evaluate.
_CC_PREFIX = re.compile(r"^[a-z]+(?:\([^)]+\))?:\s*", re.IGNORECASE)

# Rationale signals — any of these in the body satisfies the
# "explain WHY" rule. Order = how cheap they are to look for.
# Note: the leading ``[\s>*+\-]*`` tolerates whitespace and common
# markdown list/quote prefixes (``- Why:``, ``* **Why:**``, ``> ##
# Why``) so an indented or nested rationale line still counts.
# ``re.MULTILINE`` makes ``^`` match each line's start.
_LINE_PREFIX = r"[\s>*+\-]*"
_RATIONALE_SIGNALS = (
    re.compile(
        rf"^{_LINE_PREFIX}#+\s*(?:why|summary|rationale|motivation|background|context)\b",
        re.IGNORECASE | re.MULTILINE,
    ),
    re.compile(rf"^{_LINE_PREFIX}\*\*Why:\*\*", re.IGNORECASE | re.MULTILINE),
    re.compile(rf"^{_LINE_PREFIX}Why:\s", re.IGNORECASE | re.MULTILINE),
    re.compile(r"\b(?:closes|fixes|resolves|refs)\s+#\d+", re.IGNORECASE),
    re.compile(r"\bbecause\b", re.IGNORECASE),
)

_OVERRIDE_ENV = "PR_VALIDATE_SKIP_DESC"


class CLDescriptionQualityStep(Step):
    name = "cl_description_quality"
    description = "PR title + body have rationale (Google eng-practices)"

    def run(self, ctx: Context) -> StepResult:
        # Use env_truthy so ``PR_VALIDATE_SKIP_DESC=0`` correctly leaves
        # the gate enabled — bare ``os.environ.get`` would return the
        # string "0" which is truthy in Python and would silently skip.
        if env_truthy(_OVERRIDE_ENV):
            return StepResult(
                name=self.name,
                status="skip",
                summary=f"skipped via {_OVERRIDE_ENV}=1",
            )

        title = (ctx.pr_title or "").strip()
        body = (ctx.pr_body or "").strip()

        # 1) Title check — strip a conventional-commit prefix if present
        # so "fix: bug" still trips the bad-pattern net, but
        # "fix(routes): default empty content to ..." doesn't.
        bare_title = _CC_PREFIX.sub("", title).strip().lower()
        if not bare_title:
            return StepResult(
                name=self.name,
                status="fail",
                summary="PR title is empty",
                details=(
                    "Title must be a short, informative summary. See "
                    "Google's CL-descriptions guidance: "
                    "https://github.com/google/eng-practices/blob/master/review/developer/cl-descriptions.md"
                ),
            )
        if len(bare_title.split()) < 3:
            return StepResult(
                name=self.name,
                status="fail",
                summary=f"PR title too short ({len(bare_title.split())} words after prefix strip)",
                details=(
                    f"Title (after stripping any conventional-commit prefix): "
                    f"`{bare_title}`\n\n"
                    "Google eng-practices: 'Should be informative enough that "
                    "future code searchers don't have to read your CL.' At "
                    "least 3 meaningful words required. Examples: "
                    "`fix(routes): reject audio_url on text-only models` vs "
                    "`fix: bug`."
                ),
            )
        for bad in _BAD_TITLE_PATTERNS:
            if bad.match(bare_title):
                return StepResult(
                    name=self.name,
                    status="fail",
                    summary=f"PR title matches known weak pattern: '{bare_title}'",
                    details=(
                        f"Title (post-prefix-strip): `{bare_title}`\n\n"
                        "Google eng-practices calls these out as bad CL "
                        "descriptions: 'Fix bug', 'Fix build', 'Add patch', "
                        "'WIP', etc. Rewrite to say WHAT and WHY in <70 "
                        "chars. Examples:\n"
                        "- `fix(api): honor max_completion_tokens on chat completions`\n"
                        "- `docs(benchmarks): add DFlash bench for Qwen3.6-35B`\n"
                        "- `refactor(scheduler): extract admission control to dedicated module`"
                    ),
                )

        # 2) Body must exist
        if not body:
            return StepResult(
                name=self.name,
                status="fail",
                summary="PR body is empty",
                details=(
                    "Every PR needs a body explaining the WHY: what problem "
                    "is being solved and why this approach. "
                    "Google eng-practices: "
                    "https://github.com/google/eng-practices/blob/master/review/developer/cl-descriptions.md\n\n"
                    "Minimum useful template:\n"
                    "```\n"
                    "## Summary\n"
                    "- <one-line of what changed>\n\n"
                    "## Why\n"
                    "<the problem this solves>\n\n"
                    "## Test plan\n"
                    "- [x] <verification>\n"
                    "```"
                ),
            )

        # 3) Body must contain a rationale signal
        for pattern in _RATIONALE_SIGNALS:
            if pattern.search(body):
                return StepResult(
                    name=self.name,
                    status="pass",
                    summary=f"title OK + body has rationale ({len(body)} chars)",
                )

        return StepResult(
            name=self.name,
            status="fail",
            summary="PR body has no rationale signal (no 'why', no 'closes #', no 'because')",
            details=(
                f"Body is {len(body)} chars but contains no recognizable "
                "rationale signal. Add one of:\n"
                "- A heading like `## Why`, `## Rationale`, `## Motivation`, "
                "or `## Background`.\n"
                "- An issue link: `Closes #NNN` / `Fixes #NNN` / `Refs #NNN`.\n"
                "- An inline `Why:` line.\n"
                "- A `because`-clause explaining the change.\n\n"
                "Google eng-practices: "
                "'Explain... what problem is being solved, and why this "
                "is the best approach.'"
            ),
        )
