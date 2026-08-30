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
    re.compile(r"^updates?\.?$"),  # covers "update" and "updates"
    re.compile(r"^patch\.?$"),
    re.compile(r"^wip\.?$"),
    re.compile(r"^cleanup\.?$"),
    re.compile(r"^changes\.?$"),
    re.compile(r"^misc\.?$"),
    re.compile(r"^minor\s+(?:fix|change|update)\.?$"),
)

# Conventional-commit prefix (e.g. ``fix(routes):``, ``feat:``,
# ``docs(benchmarks):``, ``feat!:`` for breaking changes) is stripped
# before the bad-title check so the substantive title is what we
# evaluate. The ``!?`` permits the breaking-change marker spec uses.
_CC_PREFIX = re.compile(r"^[a-z]+(?:\([^)]+\))?!?:\s*", re.IGNORECASE)

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

# Strip HTML comments (``<!-- ... -->``) BEFORE any scoring so template
# boilerplate / rationale hidden inside ``<!-- -->`` can't satisfy the
# "body has rationale" rule. Issue #2510: an UNEDITED
# ``.github/PULL_REQUEST_TEMPLATE.md`` (with ``[x]`` boxes ticked) was a
# FALSE green because the template's ``## Why`` / ``## Scope`` headings
# matched the rationale signals even though all the real prose sat inside
# HTML comments.
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)

# Markdown headings ``#``..``######``, for slicing the body into sections.
_HEADING = re.compile(r"^(#{1,6})\s+(\S.*)$", re.MULTILINE)

# The two contract headings whose sections must carry real prose. A PR
# that uses the template but leaves ``## Why`` / ``## Scope`` empty (their
# only content being an HTML comment) must FAIL — the whole point of issue
# #2510. Only these two matter; optional/empty sections elsewhere (e.g. a
# newer ``## Author`` field) are deliberately NOT gated so a legitimate PR
# with real Why/Scope prose but an empty Author field still passes.
_CONTRACT_HEADINGS = ("why", "scope")

# Markdown furniture stripped from the front of a line before deciding
# whether it counts as "substantive prose" — blank lines, bullet/quote
# prefixes, and task-list checkboxes (``- [ ]`` / ``- [x]``).
_FURNITURE = re.compile(r"^[\s>*+\-\[\]xX]*")


def _strip_comments(body: str) -> str:
    """Return ``body`` with every ``<!-- ... -->`` HTML comment removed.

    Comment content is inert markdown that GitHub renders hidden; it must
    not count toward "does this PR explain WHY?" — an author can leave the
    entire template in place (all prose inside comments) and the body
    would look well-formed while carrying zero real rationale.
    """
    return _HTML_COMMENT.sub("", body)


def _sections(body: str) -> dict[str, str]:
    """Split a body into per-heading sections keyed by lowercased heading
    text. A section runs from its heading to the next heading at the SAME
    or HIGHER level (a ``###`` subheading stays within its parent). Each
    value is the section's prose (the heading line itself excluded)."""
    matches = list(_HEADING.finditer(body))
    sections: dict[str, str] = {}
    for i, m in enumerate(matches):
        level = len(m.group(1))
        start = m.end()
        end = len(body)
        for j in range(i + 1, len(matches)):
            if len(matches[j].group(1)) <= level:
                end = matches[j].start()
                break
        name = m.group(2).strip().lower()
        sections[name] = body[start:end]
    return sections


def _is_substantive_prose(text: str) -> bool:
    """True if ``text`` contains at least one real word after stripping
    markdown furniture (bullets, quotes, checkboxes, blank lines) from each
    line. Rejects a section whose "content" is only markdown decoration or
    that is empty once HTML comments (the template's hiding place) are
    removed."""
    for line in text.splitlines():
        stripped = _FURNITURE.sub("", line)
        # ``[^\W\d_]`` matches any unicode letter (word char, not digit/_). A
        # real rationale line always has at least one letter after furniture.
        if re.search(r"[^\W\d_]", stripped):
            return True
    return False


_OVERRIDE_ENV = "PR_VALIDATE_SKIP_DESC"


class CLDescriptionQualityStep(Step):
    name = "cl_description_quality"
    description = "PR title + body have rationale (Google eng-practices)"

    def run(self, ctx: Context) -> StepResult:
        if ctx.is_mergify_merge_candidate:
            return StepResult(
                name=self.name,
                status="skip",
                summary="trusted Mergify candidate aggregates reviewed PR descriptions",
            )

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
        # Everything below scores the COMMENT-STRIPPED body so template
        # boilerplate hidden in ``<!-- -->`` can never satisfy "body exists"
        # or "body has rationale" (issue #2510).
        stripped = _strip_comments(body).strip()

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

        # 2) Body must exist. Scored on the comment-stripped body: a body
        # whose only content is ``<!-- … -->`` is effectively empty (issue
        # #2510) — the template's skeleton must not read as a real body.
        if not stripped:
            return StepResult(
                name=self.name,
                status="fail",
                summary="PR body is empty (or only HTML comments)",
                details=(
                    "Every PR needs a body explaining the WHY: what problem "
                    "is being solved and why this approach. "
                    "Google eng-practices: "
                    "https://github.com/google/eng-practices/blob/master/review/developer/cl-descriptions.md\n\n"
                    "Minimum useful template — the PR contract headings:\n"
                    "```\n"
                    "## Why\n"
                    "<the problem this solves>\n\n"
                    "## Scope\n"
                    "- <what changed>\n\n"
                    "## Non-goals\n"
                    "- <explicitly not done, or 'none'>\n\n"
                    "## Acceptance\n"
                    "- <the observable contract the change satisfies>\n\n"
                    "## Verification\n"
                    "- [x] <what you ran and it showed>\n\n"
                    "## Behaviour delta\n"
                    "- <before -> after, when a default/lane/policy changes>\n"
                    "```"
                ),
            )

        # 3) Body must explain WHY.
        #
        # 3a) If the author used the ``## Why`` / ``## Scope`` contract
        # headings, those sections must carry real prose — NOT just the
        # heading. An UNEDITED template has all its prose inside HTML
        # comments, so after comment-strip ``## Why`` is immediately
        # followed by ``## Scope`` with nothing in between → empty → FAIL.
        # Only these two headings are gated; optional/empty sections
        # elsewhere (e.g. a newer ``## Author`` field) never false-fail a
        # PR that has real Why/Scope prose.
        sections = _sections(stripped)
        for heading in _CONTRACT_HEADINGS:
            if heading in sections and not _is_substantive_prose(sections[heading]):
                return StepResult(
                    name=self.name,
                    status="fail",
                    summary=f"`## {heading.title()}` section is empty (unfilled template)",
                    details=(
                        f"The PR has a `## {heading.title()}` heading but that "
                        "section carries no prose — its only content is an "
                        "HTML comment or markdown furniture. The template's "
                        "comment blocks are guide rails, not a filled-in "
                        "answer.\n\n"
                        f"Rewrite the `## {heading.title()}` section with the "
                        "actual rationale (issue #2510): what problem is "
                        "being solved and why this approach.\n\n"
                        f"Current section (after removing `<!-- -->` "
                        f"comments):\n```\n{sections[heading].strip() or '(empty)'}\n```"
                    ),
                )

        # 3b) Body must carry a rationale signal. Evaluated on the
        # comment-stripped body so comment-only prose (e.g. a lone
        # ``<!-- because … -->``) never satisfies the check. This keeps the
        # lenient paths intact for PRs that DON'T use the template headings
        # — an inline ``Why:`` line, a ``Closes #N`` link, or a because-
        # clause, and the ``## Why`` heading itself for template users.
        for pattern in _RATIONALE_SIGNALS:
            if pattern.search(stripped):
                return StepResult(
                    name=self.name,
                    status="pass",
                    summary=f"title OK + body has rationale ({len(stripped)} chars)",
                )

        return StepResult(
            name=self.name,
            status="fail",
            summary="PR body has no rationale signal (no 'why', no 'closes #', no 'because')",
            details=(
                f"Body is {len(stripped)} chars but contains no recognizable "
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
