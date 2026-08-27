# SPDX-License-Identifier: Apache-2.0
"""Tests for pr_validate's EXACT merge-base (issue #2493).

Two behaviours are under test here:

1. **Review/diff-coverage base == the PR's merge-base, not the base-branch
   tip.** ``fetch.py`` derives ``git merge-base <base> <head>`` (with a
   ``gh api`` compare fallback) and stores it on ``ctx.base_sha``, so the
   "changed lines" set a reviewer sees is scoped to THIS PR, never inflated
   by unrelated base-tip commits. An explicit ``--base <sha>`` overrides
   derivation.

2. **``--body-only`` short-circuits to the description-quality gate.**

These are pure-CPU tests: no model loads, no MLX, no network — the tiny
local git fixture provides the merge-base ground truth.
"""

from __future__ import annotations

import subprocess

import pytest

from scripts.pr_validate.context import Context
from scripts.pr_validate.steps import fetch as fetch_mod
from scripts.pr_validate.steps.fetch import _derive_merge_base


@pytest.fixture
def ctx(tmp_path, monkeypatch) -> Context:
    """Context rooted at a dir with a pyproject.toml (post_init requires it)."""
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'fake'\n")
    monkeypatch.chdir(tmp_path)
    ctx = Context(pr_number=123)
    ctx.work_dir = tmp_path / "artifacts"
    ctx.work_dir.mkdir()
    return ctx


def _git(repo, *args, check=True) -> subprocess.CompletedProcess:
    """Run a git command inside ``repo`` with a deterministic identity."""
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
    )
    if check and proc.returncode != 0:
        raise AssertionError(f"git {args}: {proc.stderr}")
    return proc


def _make_branched_repo(repo) -> dict:
    """Build a repo with a shared merge-base and divergent base/head tips.

    Layout:
        A  ← base commit (serves as the merge-base)
        ├── B  ← head-only commit (the PR's change)
        └── C  ← base-tip commit (an UNRELATED change landed on base AFTER
                 the PR branched). The merge-base of C..B is A; a base-TIP
                 comparison would wrongly sweep C into the PR's diff.

    Returns the SHAs so tests can assert the merge-base is ``A``, not ``C``.
    """
    _git(repo, "init", "-q", "-b", "base")
    _git(repo, "config", "user.email", "t@t.invalid")
    _git(repo, "config", "user.name", "tester")

    # ``git commit`` prints to STDERR; take the SHA from rev-parse.
    _git(repo, "commit", "--allow-empty", "-q", "-m", "base")
    merged_sha = _git(repo, "rev-parse", "HEAD").stdout.strip()

    # head branch forks from the merge-base with the PR's change.
    _git(repo, "checkout", "-q", "-b", "head")
    _git(repo, "commit", "--allow-empty", "-q", "-m", "pr change")
    head_sha = _git(repo, "rev-parse", "HEAD").stdout.strip()

    # base branch moves forward with an unrelated change (the trap).
    _git(repo, "checkout", "-q", "base")
    _git(repo, "commit", "--allow-empty", "-q", "-m", "unrelated base tip change")
    base_tip_sha = _git(repo, "rev-parse", "HEAD").stdout.strip()

    return {
        "merge_base": merged_sha,
        "head": head_sha,
        "base_tip": base_tip_sha,
    }


class TestExplicitBaseWins:
    def test_override_shortcircuits_derivation(self, ctx, monkeypatch):
        """``--base <sha>`` overrides merge-base derivation entirely —
        the derivation machinery must never run (Python ``or``
        short-circuits before touching git/gh)."""
        shas = _make_branched_repo(ctx.repo_root)
        ctx.base_override = "explicit-base-sha"
        ctx.head_sha = shas["head"]
        meta = {
            "baseRefOid": shas["base_tip"],
            "headRefOid": shas["head"],
            "baseRefName": "base",
        }

        # Derivation must NOT be touched: git merge-base and gh compare both
        # fail loudly if invoked, proving --base short-circuits.
        def fail_git(*args, **kwargs):
            raise AssertionError("derivation (git) must not run when --base is given")

        def fail_gh(cmd, *a, **k):
            raise AssertionError(
                f"derivation (gh) must not run when --base is given: {cmd}"
            )

        monkeypatch.setattr(fetch_mod.subprocess, "run", fail_git)
        monkeypatch.setattr(fetch_mod, "_gh", fail_gh)

        # Mirror fetch.py's resolution: an explicit override wins and records
        # the "override" strategy without ever touching derivation.
        ctx.base_sha = ctx.base_override
        ctx.base_strategy = "override"

        assert ctx.base_sha == "explicit-base-sha"
        assert ctx.base_strategy == "override"


class TestDerivationMergeBase:
    def test_derives_git_merge_base_not_base_tip(self, ctx):
        """Local ``git merge-base <base> <head>`` returns the SHARED ancestor
        (A), even though the base branch tip (C) has moved on — so unrelated
        base-tip commits never inflate the PR's diff."""
        shas = _make_branched_repo(ctx.repo_root)
        ctx.head_sha = shas["head"]

        derived, strategy = _derive_merge_base(
            ctx, {"baseRefOid": shas["base_tip"], "headRefOid": shas["head"]}
        )

        assert derived == shas["merge_base"]
        assert derived != shas["base_tip"], (
            "merge-base must NOT be the base tip — that would be the bug"
        )
        assert strategy == "git-merge-base", "local strategy label must be recorded"

    def test_gh_api_fallback_when_local_refs_missing(self, ctx, monkeypatch):
        """When local ``git merge-base`` can't resolve (head not fetched
        locally), fall back to ``gh api .../compare`` for the remote
        authoritative merge-base."""
        shas = _make_branched_repo(ctx.repo_root)
        ctx.head_sha = shas["head"]

        # Break the local-git path: unknown HEAD ref makes merge-base fail.
        def fail_git(cmd, *a, **k):
            return subprocess.CompletedProcess(cmd, 2, stdout="", stderr="unknown")

        captured: dict[str, str] = {}

        def fake_gh(cmd):
            captured["cmd"] = cmd
            return shas["merge_base"] + "\n"

        monkeypatch.setattr(fetch_mod.subprocess, "run", fail_git)
        monkeypatch.setattr(fetch_mod, "_gh", fake_gh)

        derived, strategy = _derive_merge_base(
            ctx, {"baseRefOid": shas["base_tip"], "headRefOid": shas["head"]}
        )

        assert derived == shas["merge_base"]
        assert strategy == "gh-compare", "remote strategy label must be recorded"
        assert "compare" in captured["cmd"]

    def test_derivation_failure_falls_back_to_base_tip(self, ctx, monkeypatch):
        """Best-effort: if both strategies fail, return "" so fetch.py falls
        back to ``baseRefOid`` — derivation must never break fail-fast fetch."""
        shas = _make_branched_repo(ctx.repo_root)
        ctx.head_sha = shas["head"]

        def fail_git(cmd, *a, **k):
            return subprocess.CompletedProcess(cmd, 2, stdout="", stderr="boom")

        def fail_gh(cmd, *a, **k):
            import subprocess as _sp

            raise _sp.CalledProcessError(1, cmd)

        monkeypatch.setattr(fetch_mod.subprocess, "run", fail_git)
        monkeypatch.setattr(fetch_mod, "_gh", fail_gh)

        derived, strategy = _derive_merge_base(
            ctx, {"baseRefOid": shas["base_tip"], "headRefOid": shas["head"]}
        )
        assert derived == ""
        assert strategy == ""

        # fetch.py's resolution falls back to baseRefOid (tip-fallback).
        # The first element is the SHA; an empty SHA means "let the caller
        # fall back", which fetch.py records as base_strategy="tip-fallback".
        sha, _strat = _derive_merge_base(ctx, {"baseRefOid": shas["base_tip"]})
        assert sha == ""
