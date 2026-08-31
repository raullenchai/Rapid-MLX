# SPDX-License-Identifier: Apache-2.0
"""Verify exact-source mlx-bound proof for a trusted Mergify queue candidate.

The Mergify CLI supplies immutable queue metadata from the candidate's Git note.
This verifier then walks the candidate's complete first-parent lineage from the
real PR base, including integrations queued ahead of the current batch. Every
commit in that range must be a two-parent ``Merge of #N`` commit. If an
integration changes ``pyproject.toml``, its exact second parent must have a
coherence attestation on the same-repository source PR whose current head still
equals that immutable second parent.

Candidate identity (same repository, Mergify App author, standard branch prefix)
is enforced by the calling workflow before this script runs.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass

_SHA = re.compile(r"^[0-9a-f]{40}$")
_MERGE_SUBJECT = re.compile(r"^Merge of #(\d+)$")
_REPOSITORY = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


class AttestationError(RuntimeError):
    """Queue metadata, Git lineage, or exact-head proof is invalid."""


@dataclass(frozen=True)
class QueueMetadata:
    checking_base_sha: str
    pull_numbers: tuple[int, ...]


@dataclass(frozen=True)
class Integration:
    number: int
    merge_sha: str
    previous_candidate_sha: str
    exact_source_sha: str
    changes_pyproject: bool


def parse_queue_metadata(raw: str) -> QueueMetadata:
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as exc:
        raise AttestationError(f"invalid Mergify queue metadata: {exc}") from exc

    base = payload.get("checking_base_sha")
    sources = payload.get("pull_requests")
    if not isinstance(base, str) or not _SHA.fullmatch(base):
        raise AttestationError("queue metadata has no valid checking_base_sha")
    if not isinstance(sources, list) or not sources:
        raise AttestationError("queue metadata has no source pull requests")

    numbers: list[int] = []
    for source in sources:
        number = source.get("number") if isinstance(source, dict) else None
        if not isinstance(number, int) or isinstance(number, bool) or number <= 0:
            raise AttestationError("queue metadata has an invalid source PR number")
        numbers.append(number)
    if len(set(numbers)) != len(numbers):
        raise AttestationError("queue metadata contains duplicate source PRs")
    return QueueMetadata(base, tuple(numbers))


def _git(
    *args: str, cwd: str | None = None, check: bool = True
) -> subprocess.CompletedProcess:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )
    if check and result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "unknown Git error"
        raise AttestationError(f"git {' '.join(args)} failed: {detail}")
    return result


def _require_first_parent_ancestor(
    base: str, candidate: str, *, cwd: str | None
) -> None:
    ancestry = _git(
        "merge-base", "--is-ancestor", base, candidate, cwd=cwd, check=False
    )
    if ancestry.returncode != 0:
        raise AttestationError(f"{base} is not an ancestor of candidate {candidate}")
    first_parent = _git(
        "rev-list", "--first-parent", candidate, cwd=cwd
    ).stdout.splitlines()
    if base not in first_parent:
        raise AttestationError(f"{base} is not on candidate's first-parent lineage")


def _integrations_between(
    base: str, candidate: str, *, cwd: str | None = None
) -> list[Integration]:
    _require_first_parent_ancestor(base, candidate, cwd=cwd)
    output = _git(
        "log",
        "--first-parent",
        "--format=%H%x09%s",
        f"{base}..{candidate}",
        cwd=cwd,
    ).stdout
    integrations: list[Integration] = []
    for line in output.splitlines():
        sha, separator, subject = line.partition("\t")
        match = _MERGE_SUBJECT.fullmatch(subject) if separator else None
        if not match:
            raise AttestationError(
                f"unexpected first-parent commit {sha or '<unknown>'}: {subject!r}"
            )
        parents = _git("show", "-s", "--format=%P", sha, cwd=cwd).stdout.split()
        if len(parents) != 2:
            raise AttestationError(f"integration {sha} is not a two-parent merge")
        pyproject = _git(
            "diff",
            "--quiet",
            parents[0],
            sha,
            "--",
            "pyproject.toml",
            cwd=cwd,
            check=False,
        )
        if pyproject.returncode not in (0, 1):
            raise AttestationError(
                f"cannot inspect pyproject.toml at integration {sha}"
            )
        integrations.append(
            Integration(
                number=int(match.group(1)),
                merge_sha=sha,
                previous_candidate_sha=parents[0],
                exact_source_sha=parents[1],
                changes_pyproject=pyproject.returncode == 1,
            )
        )
    return integrations


def collect_integrations(
    metadata: QueueMetadata,
    *,
    real_base_sha: str,
    candidate_sha: str,
    cwd: str | None = None,
) -> list[Integration]:
    if not _SHA.fullmatch(real_base_sha) or not _SHA.fullmatch(candidate_sha):
        raise AttestationError("real base or candidate SHA is invalid")

    all_integrations = _integrations_between(real_base_sha, candidate_sha, cwd=cwd)
    current_batch = _integrations_between(
        metadata.checking_base_sha, candidate_sha, cwd=cwd
    )
    # Git log is newest-first; queue metadata records integration order.
    current_numbers = tuple(reversed([item.number for item in current_batch]))
    if current_numbers != metadata.pull_numbers:
        raise AttestationError(
            "queue metadata does not match candidate integration commits "
            f"({metadata.pull_numbers!r} != {current_numbers!r})"
        )

    numbers = [item.number for item in all_integrations]
    if len(numbers) != len(set(numbers)):
        raise AttestationError("candidate contains duplicate PR integrations")
    return all_integrations


def _github_json(url: str, token: str) -> dict:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "rapid-mlx-merge-queue-guard",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.load(response)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise AttestationError(f"cannot read GitHub provenance: {exc}") from exc
    if not isinstance(payload, dict):
        raise AttestationError("GitHub provenance response is not an object")
    return payload


def source_pr_attested(repository: str, number: int, sha: str, token: str) -> bool:
    if (
        not _REPOSITORY.fullmatch(repository)
        or not isinstance(number, int)
        or number <= 0
        or not _SHA.fullmatch(sha)
    ):
        raise AttestationError("repository or exact source SHA is invalid")
    if not token:
        raise AttestationError("GITHUB_TOKEN is unavailable")

    source = _github_json(
        f"https://api.github.com/repos/{repository}/pulls/{number}",
        token,
    )
    if source.get("head", {}).get("repo", {}).get("full_name") != repository:
        return False
    if source.get("head", {}).get("sha") != sha:
        return False

    labels = {
        label.get("name", "").strip().lower()
        for label in source.get("labels", [])
        if isinstance(label, dict)
    }
    if "mlx-coherence-swept" in labels:
        return True
    for line in (source.get("body") or "").splitlines():
        if line.strip().lower().startswith("coherence-sweep:"):
            return bool(line.split(":", 1)[1].strip())
    return False


def main() -> int:
    try:
        metadata = parse_queue_metadata(os.environ.get("QUEUE_METADATA", ""))
        integrations = collect_integrations(
            metadata,
            real_base_sha=os.environ.get("REAL_BASE_SHA", ""),
            candidate_sha=os.environ.get("CANDIDATE_SHA", ""),
        )
        guarded = [item for item in integrations if item.changes_pyproject]
        for item in guarded:
            if not source_pr_attested(
                os.environ.get("GITHUB_REPOSITORY", ""),
                item.number,
                item.exact_source_sha,
                os.environ.get("GITHUB_TOKEN", ""),
            ):
                raise AttestationError(
                    f"PR #{item.number} source {item.exact_source_sha} lacks an "
                    "exact-head coherence attestation"
                )

        output = os.environ.get("GITHUB_OUTPUT")
        if not output:
            raise AttestationError("GITHUB_OUTPUT is unavailable")
        with open(output, "a", encoding="utf-8") as stream:
            stream.write("attested=true\n")
        proved = ", ".join(
            f"#{item.number}@{item.exact_source_sha[:9]}" for item in guarded
        )
        print(
            f"[mergify-mlx-attestation] verified: {proved or 'no pyproject integrations'}"
        )
        return 0
    except AttestationError as exc:
        print(f"[mergify-mlx-attestation] BLOCKED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
