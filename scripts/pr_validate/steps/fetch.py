# SPDX-License-Identifier: Apache-2.0
"""Step 0 — fetch the PR.

Wraps ``gh pr view`` + ``gh pr diff`` so every later step has a stable
view of what's being validated. Records the diff and metadata into the
context. Fail-fast: if we can't fetch the PR, nothing else can run.

We deliberately do NOT check out the PR branch into the working tree.
Steps that need to run code from the PR (lint, tests) check out into a
``git worktree`` so the user's local working tree stays untouched. This
matters: a half-validated PR shouldn't leave your editor pointed at
random foreign code.
"""

from __future__ import annotations

import json
import shlex
import shutil
import subprocess

from ..base import Step, StepResult
from ..context import Context


class FetchStep(Step):
    name = "fetch"
    description = "fetch PR + diff + classify blast radius"

    def run(self, ctx: Context) -> StepResult:
        # Preflight: gh is the entire interface to GitHub. Without it
        # we can't even start; better to fail with a clear message than
        # let `subprocess.run` raise FileNotFoundError ("[Errno 2] No
        # such file or directory: 'gh'") which leaks no fix instruction.
        if not shutil.which("gh"):
            return StepResult(
                name=self.name,
                status="error",
                summary=(
                    "`gh` CLI not installed — `brew install gh` "
                    "(or see https://cli.github.com)"
                ),
            )

        # Pull PR metadata as JSON so we get author, head ref, etc.
        # without scraping HTML.
        try:
            meta_raw = _gh(
                f"pr view {ctx.pr_number} --repo {ctx.repo} "
                "--json number,title,body,author,isCrossRepository,"
                "headRefOid,headRefName,baseRefOid,additions,deletions,"
                "files,state,mergeStateStatus"
            )
        except subprocess.CalledProcessError as e:
            return StepResult(
                name=self.name,
                status="error",
                summary="`gh pr view` failed (auth? network? typo in PR#?)",
                details=f"```\n{e.stderr or e.stdout}\n```",
            )

        meta = json.loads(meta_raw)

        ctx.pr_title = meta.get("title", "")
        ctx.pr_body = meta.get("body", "") or ""
        ctx.pr_author = (meta.get("author") or {}).get("login", "")
        # gh CLI doesn't expose authorAssociation via --json; use
        # isCrossRepository as the external-author proxy. Fork-based PRs
        # are external by definition. Misses the rare case where a
        # collaborator opens a PR from a fork, but that's a false-positive
        # in the safe direction (more scrutiny, not less).
        ctx.pr_is_external = bool(meta.get("isCrossRepository", False))
        ctx.head_sha = meta.get("headRefOid", "")
        ctx.head_branch = meta.get("headRefName", "")
        ctx.base_sha = meta.get("baseRefOid", "")
        ctx.additions = meta.get("additions", 0)
        ctx.deletions = meta.get("deletions", 0)
        ctx.files_changed = sorted(
            f["path"] for f in (meta.get("files") or []) if "path" in f
        )

        # Pull the full diff. We save to disk so the codex review and
        # supply chain steps can stream-read it without re-running gh.
        diff_path = ctx.artifact_path("pr.diff")
        try:
            diff = _gh(f"pr diff {ctx.pr_number} --repo {ctx.repo}")
        except subprocess.CalledProcessError as e:
            return StepResult(
                name=self.name,
                status="error",
                summary="`gh pr diff` failed",
                details=f"```\n{e.stderr or e.stdout}\n```",
            )
        diff_path.write_text(diff)
        ctx.diff_path = str(diff_path)

        # Refuse closed/merged PRs as a default — the user can still
        # run validation on them by editing this gate, but the common
        # case of "is this merge-safe" wants an OPEN PR.
        state = meta.get("state", "")
        if state != "OPEN":
            return StepResult(
                name=self.name,
                status="fail",
                summary=f"PR is {state}, not OPEN — refusing to validate "
                "(re-open if you want a re-grade)",
            )

        # Sanity-check the merge state too — DIRTY means the branch
        # has merge conflicts; we'd be validating an unmergeable
        # branch and the result would be misleading.
        merge_state = meta.get("mergeStateStatus", "")
        if merge_state == "DIRTY":
            return StepResult(
                name=self.name,
                status="fail",
                summary="PR has merge conflicts (mergeStateStatus=DIRTY) — "
                "rebase before validating",
            )

        ctx.run_log(
            f"fetched: '{ctx.pr_title[:60]}' by {ctx.pr_author} "
            f"({ctx.additions}+/{ctx.deletions}- LOC, "
            f"{len(ctx.files_changed)} files, blast={ctx.blast_radius})"
        )

        return StepResult(
            name=self.name,
            status="pass",
            summary=(
                f"{len(ctx.files_changed)} files, "
                f"+{ctx.additions}/-{ctx.deletions} LOC, "
                f"blast={ctx.blast_radius}"
            ),
            artifacts=[ctx.diff_path],
        )


def _gh(cmd: str) -> str:
    """Run a `gh` subcommand, return stdout. Raises on non-zero exit."""
    result = subprocess.run(  # noqa: S603 — args are shell-split, not a string
        ["gh", *shlex.split(cmd)],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout
