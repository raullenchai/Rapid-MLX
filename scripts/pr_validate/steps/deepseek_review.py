# SPDX-License-Identifier: Apache-2.0
"""Step 6 — DeepSeek V4 Pro adversarial review of the diff.

Productionizes the one-off /tmp/deepseek_review_*.py scripts we've been
building per-PR. Reads the prompt template from
``scripts/pr_validate/prompts/deepseek_review.md``, sends the diff,
parses the response. Findings are surfaced in the scorecard verbatim
so a maintainer can decide whether to act on each.

Failure policy is deliberately conservative: we mark the step ``fail``
ONLY if DeepSeek's reply contains markers that look like blocking
findings (lines starting with "1.", "2.", … and mentioning specific
files). A reply of "No blocking issues found." → ``pass``. Network
failures or a missing API key → ``skip`` with a clear summary, NOT
``fail`` — we don't want a temporarily-down API to block every PR.
The strictness is at the human-review layer, not the API layer.

The API key MUST be supplied via ``DEEPSEEK_API_KEY`` — there is no
in-source default (the repo is public). Without the env var the step
skips gracefully so contributors without keys can still run the rest
of the pipeline. The user's personal key is documented in
``memory/knowledge/deepseek_api_key.md`` for local development.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

from ..base import Step, StepResult
from ..context import Context, env_truthy

# NB: do NOT default the API key here. The repo is public; even a
# personal review-budget key should never live in version control.
# Users provide it via ``DEEPSEEK_API_KEY`` (see
# ``memory/knowledge/deepseek_api_key.md`` for where the key is stored
# locally). Without the env var the step skips gracefully.
ENDPOINT = "https://api.deepseek.com/v1/chat/completions"
MODEL = "deepseek-v4-pro"

# Hard cap on diff size we send. DeepSeek can take more, but past
# ~120KB of diff the signal-to-noise of the review drops sharply (the
# model starts skimming). For very large PRs we send the diff
# truncated at a file boundary and note which files were omitted.
MAX_DIFF_BYTES = 120_000

# Token budget. Reasoning-model behavior is highly variable: a simple
# diff burns ~6K reasoning tokens; a complex multi-commit one (like
# PR #203's 22KB / 5-commit diff) burns ~16K reasoning tokens before
# emitting any visible content. We observed the response getting cut
# mid-finding at MAX_TOKENS=16384 because reasoning consumed the full
# budget. 32K leaves room for ~16K reasoning + ~16K visible reply.
MAX_TOKENS = 32_768

# How long we wait for the API. DeepSeek V4 Pro reasoning takes
# 30-90s typical, up to 5min for big diffs.
TIMEOUT_SECONDS = 600

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "deepseek_review.md"


class DeepSeekReviewStep(Step):
    name = "deepseek_review"
    description = "DeepSeek V4 Pro adversarial review of diff"

    def should_run(self, ctx: Context) -> bool:
        # Allow opt-out (CI without API access, offline dev, etc.).
        if env_truthy("PR_VALIDATE_NO_DEEPSEEK"):
            return False
        # Skip if there's no diff to review (shouldn't happen post-fetch
        # but defend against it — empty review wastes API budget).
        return bool(ctx.diff_path) and Path(ctx.diff_path).stat().st_size > 0

    def run(self, ctx: Context) -> StepResult:
        # httpx is in our deps already; importing inside the method
        # keeps the framework import-light for steps that don't need it.
        try:
            import httpx
        except ImportError:
            return StepResult(
                name=self.name,
                status="skip",
                summary="httpx not installed — `pip install httpx`",
            )

        api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not api_key:
            return StepResult(
                name=self.name,
                status="skip",
                summary=(
                    "no DEEPSEEK_API_KEY set (export it to enable adversarial "
                    "review; see memory/knowledge/deepseek_api_key.md)"
                ),
            )

        diff_full = Path(ctx.diff_path).read_text()
        diff, omitted_files, truncated = _truncate_diff_at_file_boundary(
            diff_full, MAX_DIFF_BYTES
        )

        if not PROMPT_PATH.exists():
            return StepResult(
                name=self.name,
                status="error",
                summary=f"prompt template missing at {PROMPT_PATH}",
            )
        system_prompt = PROMPT_PATH.read_text()

        user_prompt = _build_user_prompt(ctx, diff, omitted_files, truncated)

        # Save what we sent — useful for debugging "why did the review
        # say X" without re-running.
        sent_path = ctx.artifact_path("deepseek-request.txt")
        sent_path.write_text(
            f"=== SYSTEM ===\n{system_prompt}\n\n=== USER ===\n{user_prompt}"
        )

        ctx.run_log(f"calling {MODEL} ({len(diff.encode())} bytes of diff)…")

        try:
            with httpx.Client(timeout=TIMEOUT_SECONDS) as client:
                resp = client.post(
                    ENDPOINT,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": MODEL,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        "temperature": 0.0,
                        "max_tokens": MAX_TOKENS,
                    },
                )
        except httpx.RequestError as e:
            # Catch the whole request-failure tree: TimeoutException,
            # NetworkError (ConnectError, ReadError), and ProtocolError
            # (RemoteProtocolError when the server cuts the connection
            # mid-response — observed under DeepSeek API load).
            return StepResult(
                name=self.name,
                status="skip",
                summary=f"API unreachable: {type(e).__name__}",
            )

        if resp.status_code != 200:
            return StepResult(
                name=self.name,
                status="skip",
                summary=f"API returned {resp.status_code}",
                details=f"```\n{resp.text[:1000]}\n```",
            )

        body = resp.json()
        choices = body.get("choices") or []
        if not choices:
            # DeepSeek can return an empty choices array on rate-limit
            # or content-policy refusal. Don't crash — surface as skip
            # with the body for debugging.
            return StepResult(
                name=self.name,
                status="skip",
                summary="API returned no choices (rate limit? policy?)",
                details=f"```json\n{json.dumps(body, indent=2)[:1000]}\n```",
            )
        content = choices[0].get("message", {}).get("content", "") or ""
        usage = body.get("usage", {})

        review_path = ctx.artifact_path("deepseek-review.md")
        review_path.write_text(content)
        usage_path = ctx.artifact_path("deepseek-usage.json")
        usage_path.write_text(json.dumps(usage, indent=2))

        # Parse the response. The prompt asks for a numbered list; we
        # detect "no findings" by an explicit phrase, otherwise we
        # extract numbered items as findings.
        findings = _extract_findings(content)
        no_issues = _is_clean_review(content)

        if no_issues and not findings:
            return StepResult(
                name=self.name,
                status="pass",
                summary="DeepSeek found no blocking issues",
                artifacts=[str(review_path), str(usage_path)],
            )

        # Findings present → fail the step in strict mode. Maintainer
        # decides per-finding whether to act; the scorecard surfaces
        # them all in the details. (Some reviewers may want a
        # human-decides flow here; that's what looking at the artifact
        # is for. The step's role is to surface, not to triage.)
        summary = f"{len(findings)} finding(s)"
        if omitted_files:
            summary += f" (diff truncated — {len(omitted_files)} file(s) not reviewed)"
        elif truncated:
            summary += " (diff truncated — single large file, partial review)"
        return StepResult(
            name=self.name,
            status="fail",
            summary=summary,
            findings=findings,
            details=(
                "**Full review:**\n\n"
                f"{content}\n\n"
                f"_(Saved to `{review_path}`. "
                f"Token usage: {usage.get('total_tokens', '?')})_"
            ),
            artifacts=[str(review_path), str(usage_path)],
        )


def _build_user_prompt(
    ctx: Context, diff: str, omitted_files: list[str], truncated: bool = False
) -> str:
    """Compose the user message: PR context + directory listings + diff.

    The directory-context section is what stops the canonical false
    positive class "you added X but didn't update Y" / "X is missing"
    when X actually exists outside the diff. Without it, DeepSeek
    flagged PR #179 for having no ``feature_request.yml`` even though
    the file already lived in ``.github/ISSUE_TEMPLATE/`` (just outside
    the diff). With it, the listing makes sibling files visible.
    """
    lines = [
        f"# PR #{ctx.pr_number}: {ctx.pr_title}",
        "",
        f"**Author**: {ctx.pr_author}{' (external/fork)' if ctx.pr_is_external else ''}",
        f"**Files**: {len(ctx.files_changed)} ({ctx.additions}+/{ctx.deletions}-)",
        f"**Blast radius**: {ctx.blast_radius}",
        "",
        "## Description",
        "",
        ctx.pr_body or "_(no description)_",
        "",
    ]
    dir_context = _gather_directory_context(ctx)
    if dir_context:
        lines.append(dir_context)
        lines.append("")
    lines.append("## Diff")
    lines.append("")
    if omitted_files:
        omitted_str = ", ".join(f"`{f}`" for f in omitted_files)
        lines.append(
            f"_Note: diff capped at {MAX_DIFF_BYTES} bytes — truncated at a file "
            f"boundary. **The following files were NOT included in this review and "
            f"MUST NOT be assumed clean**: {omitted_str}. "
            "Full diff is on disk; review only what's shown below._"
        )
        lines.append("")
    elif truncated:
        # Single large file that had to be raw-sliced (no earlier boundary).
        lines.append(
            f"_Note: diff truncated to {MAX_DIFF_BYTES} bytes (single large file). "
            "The shown diff may be incomplete; review cautiously._"
        )
        lines.append("")
    lines.append("```diff")
    lines.append(diff)
    lines.append("```")
    return "\n".join(lines)


# Header line is one of:
#   diff --git a/<path> b/<path>            (no spaces in path)
#   diff --git "a/<escaped>" "b/<escaped>"  (path with spaces / specials)
# A single byte regex handles both. group(1) wins for the quoted form,
# group(2) for the unquoted. Operating on bytes avoids the O(N·L)
# re-encode-the-prefix dance that string-mode would force.
_FILE_HEADER_RE = re.compile(
    rb'^diff --git (?:"a/((?:[^"\\]|\\.)*)"|a/(\S+)) ',
    re.MULTILINE,
)


def _truncate_diff_at_file_boundary(
    diff: str, max_bytes: int
) -> tuple[str, list[str], bool]:
    """Truncate *diff* to *max_bytes* at the nearest preceding file boundary.

    Returns ``(kept_diff, omitted_file_paths, was_truncated)``.  If the diff
    fits, returns the original string, an empty list, and ``False``.  If
    even the first file exceeds the limit we fall back to a raw byte slice
    (better than nothing) and list all remaining files as omitted.

    Sizes are measured in UTF-8 bytes (not Python character counts) to match
    what the HTTP layer actually sends.
    """
    diff_bytes = diff.encode()
    if len(diff_bytes) <= max_bytes:
        return diff, [], False

    # Find every per-file header — byte offset and a-side path.
    positions: list[tuple[int, str]] = []
    for m in _FILE_HEADER_RE.finditer(diff_bytes):
        path_bytes = m.group(1) if m.group(1) is not None else m.group(2)
        # Quoted form uses C-style escapes (\\, \", \t, \n, octal); we don't
        # try to fully unescape — the path is only used for human-readable
        # "files NOT reviewed" listings, slight visual artifact is OK.
        path = path_bytes.decode("utf-8", errors="replace")
        positions.append((m.start(), path))

    # Walk forward and find the last file whose header fits within max_bytes.
    # If we never break, kept_end naturally lands on the last header position
    # — which is exactly what we want: drop the last file (its content is
    # what overflows) so we only ship complete per-file diffs.
    kept_end = 0
    for pos, _ in positions:
        if pos > max_bytes:
            break
        kept_end = pos

    if kept_end == 0:
        # Either no headers found, or even the first file alone overflows.
        # Raw-slice to the byte limit and list the remaining files (if any)
        # as omitted; the first file is shown partially.  ``errors="ignore"``
        # drops a trailing incomplete UTF-8 sequence rather than replacing
        # it with U+FFFD — the latter would re-encode to 3 bytes and could
        # push us *over* ``max_bytes``, defeating the cap.
        raw = diff_bytes[:max_bytes].decode("utf-8", errors="ignore")
        omitted = [path for _, path in positions[1:]]
        return raw, omitted, True

    # Slice at the byte boundary, then decode — safe because the boundary
    # is always at the start of a header line (pure ASCII).
    kept_diff = diff_bytes[:kept_end].decode()
    omitted = [path for pos, path in positions if pos >= kept_end]
    return kept_diff, omitted, True


# Cap on how many directories we'll list — a 50-file refactor PR
# shouldn't bloat the prompt with 20 directory listings. Most PRs touch
# 1-5 dirs.
_MAX_DIRS_LISTED = 15
# Per-directory cap. Anything more is noise; we tag the overflow with
# a count so the model knows there's more.
_MAX_FILES_PER_DIR = 30


def _is_safe_listing_path(d: str) -> bool:
    """Return True iff *d* is safe to feed into ``gh api repos/.../contents/<d>``.

    We reject:
    * ``.`` — current dir; GitHub's contents API 404s on it.
    * ``..`` and ``../*`` — parent-traversal in the path-component sense.
      A plain ``startswith("..")`` would also reject legitimate names like
      ``..hidden`` or ``..env``; we only want the ``..`` *component* form.
    * absolute paths — never come from ``gh pr diff`` and could probe
      outside the repo's tree on a misbehaving server.

    *d* is the directory part of a changed-file path (``os.path.dirname``).
    Empty input returns False — caller should already have skipped it.
    """
    if not d:
        return False
    normalized = os.path.normpath(d)
    if normalized in (".", ".."):
        return False
    if normalized.startswith("../"):
        return False
    if os.path.isabs(normalized):
        return False
    return True


def _gather_directory_context(ctx: Context) -> str:
    """Return a markdown section listing files in each directory the PR
    touches, fetched at HEAD via ``gh api``.

    Empty string if we can't query (no head_sha, gh missing, all errors)
    — in which case the review degrades to the old diff-only behavior.
    Never raises; this is a context enhancement, not a gate.
    """
    if not ctx.head_sha or not ctx.files_changed:
        return ""

    # Collect unique directory paths from changed files. Root files
    # (no dirname) get represented by "" which we map to the repo root
    # listing — usually less interesting, so we skip it. Reject any
    # path that escapes via ``..`` or is absolute — git+GitHub usually
    # block these on commit but we'd rather not feed an unsanitized
    # path to ``gh api`` (the URL-resolved request could probe outside
    # the PR's tree on a misbehaving server).
    dirs: set[str] = set()
    for path in ctx.files_changed:
        d = os.path.dirname(path)
        if not _is_safe_listing_path(d):
            continue
        dirs.add(os.path.normpath(d))

    if not dirs:
        return ""

    sorted_dirs = sorted(dirs)
    capped = sorted_dirs[:_MAX_DIRS_LISTED]

    sections: list[str] = []
    for d in capped:
        files = _list_repo_dir(ctx.repo, ctx.head_sha, d)
        if not files:
            # Either dir is gone at HEAD, gh failed, or empty — skip
            # silently rather than misleading the model with "(empty)"
            # which could itself trigger a false positive.
            continue
        listing_lines = [f"  - `{f}`" for f in files[:_MAX_FILES_PER_DIR]]
        if len(files) > _MAX_FILES_PER_DIR:
            listing_lines.append(f"  - … ({len(files) - _MAX_FILES_PER_DIR} more)")
        sections.append(
            f"### `{d}/` (post-PR state — fetched from HEAD)\n"
            + "\n".join(listing_lines)
        )

    if not sections:
        return ""

    overflow_note = ""
    if len(sorted_dirs) > _MAX_DIRS_LISTED:
        overflow_note = (
            f"\n_(Listing first {_MAX_DIRS_LISTED} of {len(sorted_dirs)} "
            "touched directories; rest omitted to keep the prompt small.)_"
        )

    return (
        "## Directory context\n\n"
        "Files that exist in directories the diff touches, at the PR's "
        "HEAD commit. Use this to avoid 'X is missing' false positives "
        "— a sibling file you don't see in the diff might still be "
        "present. Don't claim a file is missing without checking here "
        "first.\n" + overflow_note + "\n\n" + "\n\n".join(sections)
    )


def _list_repo_dir(repo: str, ref: str, path: str) -> list[str]:
    """List entry names in ``repo``/``path`` at ``ref`` via ``gh api``.

    Returns just file/dir basenames sorted. Empty list on any failure
    (network, 404, malformed JSON, missing gh) — caller treats absence
    of context as "no enhancement", never a hard error.
    """
    try:
        proc = subprocess.run(  # noqa: S603
            [
                "gh",
                "api",
                f"repos/{repo}/contents/{path}?ref={ref}",
                "--jq",
                ".[] | .name",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except Exception:  # noqa: BLE001 — directory context is best-effort, never a gate
        return []
    if proc.returncode != 0:
        return []
    names = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    return sorted(names)


# Phrases that indicate a clean review. Match case-insensitively. If
# any of these appear in the response and we extract zero numbered
# findings, treat it as a clean pass.
_CLEAN_PATTERNS = (
    re.compile(r"no\s+blocking\s+issues?\s+found", re.IGNORECASE),
    re.compile(r"no\s+issues?\s+found", re.IGNORECASE),
    re.compile(r"^\s*looks?\s+good", re.IGNORECASE | re.MULTILINE),
)


def _is_clean_review(text: str) -> bool:
    return any(p.search(text) for p in _CLEAN_PATTERNS)


# Extract numbered findings — match lines starting with "1.", "2.",
# etc. (with optional leading whitespace and bold). One short line
# per finding for the scorecard table.
_FINDING_RE = re.compile(
    r"^\s*(?:\*\*)?(\d+)\.?\)?\s*(?:\*\*)?\s+(.+?)(?:\*\*)?\s*$",
    re.MULTILINE,
)


def _extract_findings(text: str) -> list[str]:
    """Pull numbered list items as findings. Truncates each to a
    reasonable length for the scorecard table; full text lives in the
    artifact file."""
    findings = []
    for match in _FINDING_RE.finditer(text):
        body = match.group(2).strip().rstrip("*").strip()
        # Cap per-finding length; long ones go in the artifact.
        if len(body) > 240:
            body = body[:237] + "…"
        findings.append(body)
    # Sometimes the model returns markdown headings like "### 1. Title"
    # which the regex above catches via the leading number; that's
    # fine. Deduplicate just in case (very long replies sometimes
    # repeat the summary at the bottom).
    seen = set()
    out = []
    for f in findings:
        if f not in seen:
            out.append(f)
            seen.add(f)
    return out
