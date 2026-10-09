#!/usr/bin/env python3
"""Fail-closed release source identity gate for main or the 0.16 frozen ref."""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Sequence

FROZEN_VERSION = "0.16.0"
FROZEN_BRANCH = "release/0.16.0"
FROZEN_REF = f"refs/heads/{FROZEN_BRANCH}"
FROZEN_PRODUCT_SHA = "837d6d6234936f6886caf1edff9f5d01eaee1330"
FROZEN_ALLOWED_PATHS = {
    ".github/workflows/auto-release.yml",
    ".github/workflows/release-preflight.yml",
    "scripts/check_main_head.py",
    "scripts/check_release_environment.py",
    "scripts/check_release_source_ref.py",
    "tests/test_check_main_head.py",
    "tests/test_check_release_environment.py",
    "tests/test_check_release_source_ref.py",
    "tests/release/test_desktop_release_promotion.sh",
    "docs/development/releasing.md",
    "pyproject.toml",
    "apps/rapid-mac/Resources/Info.plist",
    "apps/rapid-mac/CHANGELOG.md",
    "docs/release-notes/v0.16.0.md",
    "docs/release-notes/unreleased.md",
}
FROZEN_METADATA_PATHS = {
    "pyproject.toml",
    "apps/rapid-mac/Resources/Info.plist",
    "apps/rapid-mac/CHANGELOG.md",
    "docs/release-notes/v0.16.0.md",
    "docs/release-notes/unreleased.md",
}


class ReleaseSourceError(Exception):
    pass


def _sha(value: str, label: str) -> None:
    if len(value) != 40 or any(c not in "0123456789abcdef" for c in value):
        raise ReleaseSourceError(f"{label} must be a lowercase full SHA, got {value!r}")


def check_source(
    *,
    source_ref: str,
    live_sha: str,
    accepted_sha: str,
    release_sha: str,
    version: str,
    repo: str = ".",
    bump_base_sha: str | None = None,
) -> list[str]:
    for label, value in (
        ("live", live_sha),
        ("accepted", accepted_sha),
        ("release", release_sha),
    ):
        _sha(value, label)
    if live_sha != accepted_sha or accepted_sha != release_sha:
        raise ReleaseSourceError(
            f"live source ref is no longer the validated candidate: live={live_sha} "
            f"accepted={accepted_sha} release={release_sha}"
        )
    if source_ref == "refs/heads/main":
        return [f"source ref refs/heads/main == accepted == release: {release_sha}"]
    if source_ref != FROZEN_REF or version != FROZEN_VERSION:
        raise ReleaseSourceError(
            f"production source must be main or exact {FROZEN_REF} for {FROZEN_VERSION}"
        )
    ancestor = subprocess.run(
        [
            "git",
            "-C",
            repo,
            "merge-base",
            "--is-ancestor",
            FROZEN_PRODUCT_SHA,
            release_sha,
        ]
    )
    if ancestor.returncode != 0:
        raise ReleaseSourceError("frozen product SHA is not an ancestor of release SHA")
    merges = subprocess.check_output(
        [
            "git",
            "-C",
            repo,
            "rev-list",
            "--min-parents=2",
            f"{FROZEN_PRODUCT_SHA}..{release_sha}",
        ],
        text=True,
    ).splitlines()
    if merges:
        raise ReleaseSourceError("frozen release history must remain linear")
    changed = subprocess.check_output(
        [
            "git",
            "-C",
            repo,
            "diff",
            "--name-only",
            f"{FROZEN_PRODUCT_SHA}..{release_sha}",
        ],
        text=True,
    ).splitlines()
    disallowed = sorted(set(changed) - FROZEN_ALLOWED_PATHS)
    if disallowed:
        raise ReleaseSourceError(
            "frozen release contains non-policy/non-metadata changes: "
            + ", ".join(disallowed)
        )
    if bump_base_sha is not None:
        _sha(bump_base_sha, "bump base")
        bump_paths = subprocess.check_output(
            [
                "git",
                "-C",
                repo,
                "diff",
                "--name-only",
                f"{bump_base_sha}..{release_sha}",
            ],
            text=True,
        ).splitlines()
        disallowed_bump_paths = sorted(set(bump_paths) - FROZEN_METADATA_PATHS)
        if not bump_paths or disallowed_bump_paths:
            got = (
                ", ".join(disallowed_bump_paths)
                if disallowed_bump_paths
                else "no changes"
            )
            raise ReleaseSourceError(
                f"frozen bump must change only version/release-note metadata; got: {got}"
            )
    commits = subprocess.check_output(
        ["git", "-C", repo, "rev-list", f"{FROZEN_PRODUCT_SHA}..{release_sha}"],
        text=True,
    ).splitlines()
    for commit in commits:
        commit_paths = subprocess.check_output(
            [
                "git",
                "-C",
                repo,
                "diff-tree",
                "--no-commit-id",
                "--name-only",
                "-r",
                commit,
            ],
            text=True,
        ).splitlines()
        disallowed_commit_paths = sorted(set(commit_paths) - FROZEN_ALLOWED_PATHS)
        if disallowed_commit_paths:
            raise ReleaseSourceError(
                f"frozen release commit {commit} touched disallowed paths: "
                + ", ".join(disallowed_commit_paths)
            )
    return [
        f"source ref {FROZEN_REF} == accepted == release: {release_sha}",
        f"frozen product ancestor: {FROZEN_PRODUCT_SHA}",
        f"frozen delta restricted to {len(changed)} approved policy/metadata paths",
    ]


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--source-ref", required=True)
    p.add_argument("--live-sha", required=True)
    p.add_argument("--accepted-sha", required=True)
    p.add_argument("--release-sha", required=True)
    p.add_argument("--version", required=True)
    p.add_argument("--repo", default=".")
    p.add_argument("--bump-base-sha")
    a = p.parse_args(argv)
    try:
        print(
            "\n".join(
                check_source(
                    source_ref=a.source_ref,
                    live_sha=a.live_sha,
                    accepted_sha=a.accepted_sha,
                    release_sha=a.release_sha,
                    version=a.version,
                    repo=a.repo,
                    bump_base_sha=a.bump_base_sha,
                )
            )
        )
    except (ReleaseSourceError, subprocess.SubprocessError) as exc:
        print(f"release source: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
