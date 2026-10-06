"""Publish freshly revalidated candidate admission and revoke inflight mapped gates."""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import subprocess
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from scripts import ci_candidate_consumer as consumer
from scripts import queue_tree_evidence as evidence

CONTEXT = "candidate-admission/ci"
ARTIFACT_RE = re.compile(r"candidate-qualification-ci-([0-9a-f]{40})")


def verify_admission(
    client: evidence.GitHubClient, producer_run_id: int
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "verified": False,
        "authorizes_merge": False,
        "authorizes_reduced_ci": False,
    }
    try:
        if type(producer_run_id) is not int or producer_run_id < 1:
            raise evidence.EvidenceError("invalid producer run")
        run = consumer._producer_run(client, producer_run_id)
        page = client.json(
            f"repos/{client.repo}/actions/runs/{producer_run_id}/artifacts",
            "per_page=100",
        )
        count = page.get("total_count")
        if (
            type(count) is not int
            or not 0 < count < 100
            or len(page["artifacts"]) != count
        ):
            raise evidence.EvidenceError("incomplete admission artifact listing")
        matches = [
            (artifact, ARTIFACT_RE.fullmatch(str(artifact.get("name", ""))))
            for artifact in page["artifacts"]
        ]
        matches = [(artifact, match) for artifact, match in matches if match]
        if len(matches) != 1:
            raise evidence.EvidenceError("expected one exact qualification artifact")
        artifact, match = matches[0]
        if (
            artifact.get("workflow_run", {}).get("id") != producer_run_id
            or artifact.get("expired") is not False
        ):
            raise evidence.EvidenceError("admission artifact is stale or unrelated")
        sha = match[1]
        status = consumer._status(client, sha)
        target = evidence.RUN_URL_RE.fullmatch(str(status.get("target_url", "")))
        if (
            not target
            or target.group("repo") != client.repo
            or int(target.group("run_id")) != producer_run_id
        ):
            raise evidence.EvidenceError("producer notification was superseded")
        consumed = consumer.consume_qualified(client, sha)
        if consumed.get("verified") is not True or consumed.get("kind") not in {
            "full",
            "mapped",
            "engine-not-required",
        }:
            raise evidence.EvidenceError(
                "full admission rejected: " + consumed.get("reason", "")
            )
        if (
            consumer._producer_run(client, producer_run_id) != run
            or consumer._status(client, sha) != status
        ):
            raise evidence.EvidenceError("producer/index changed before admission")
        # Repeat the actual artifact/source/base/tree verification, not just index.
        current = consumer.consume_qualified(client, sha)
        if current != consumed or current.get("verified") is not True:
            raise evidence.EvidenceError("candidate changed before admission")
        # Full-consumer output binds source proof, not notification provenance.
        # A new producer for identical proof must not validate this old notice.
        if (
            consumer._producer_run(client, producer_run_id) != run
            or consumer._status(client, sha) != status
        ):
            raise evidence.EvidenceError(
                "producer/index changed after final verification"
            )
        if consumed["kind"] == "mapped":
            result["rollout_generation"] = consumed["qualification"][
                "rollout_generation"
            ]
        result.update(
            verified=True,
            candidate_sha=sha,
            producer_run_id=producer_run_id,
            producer_attempt=run["run_attempt"],
            source_run_id=consumed["qualification"]["source_run_id"],
            source_attempt=consumed["qualification"]["source_attempt"],
            kind=consumed["kind"],
        )
    except (
        evidence.EvidenceError,
        KeyError,
        TypeError,
        ValueError,
        AttributeError,
        OSError,
        subprocess.SubprocessError,
    ) as exc:
        result["reason"] = str(exc)[:500]
    return result


def verify_source_admission(
    client: evidence.GitHubClient, source_run_id: int, source_attempt: int
) -> dict[str, Any]:
    """Start alongside qualification; only its completed real proof can admit."""
    rejected = {
        "verified": False,
        "authorizes_merge": False,
        "authorizes_reduced_ci": False,
    }
    try:
        if any(type(v) is not int or v < 1 for v in (source_run_id, source_attempt)):
            raise evidence.EvidenceError("invalid triggering CI identity/attempt")
        source = copy.deepcopy(
            client.json(f"repos/{client.repo}/actions/runs/{source_run_id}")
        )
        sha = source.get("head_sha")
        evidence._require_sha(sha)
        if (
            type(source.get("id")) is not int
            or type(source.get("run_attempt")) is not int
            or source.get("id") != source_run_id
            or source.get("run_attempt") != source_attempt
            or source.get("path") != evidence.CI_WORKFLOW_PATH
            or source.get("event") != "pull_request"
            or source.get("repository", {}).get("full_name") != client.repo
            or source.get("head_repository", {}).get("full_name") != client.repo
            or not evidence.CANDIDATE_RE.fullmatch(str(source.get("head_branch", "")))
            or source.get("status") != "completed"
            or source.get("conclusion") != "success"
        ):
            raise evidence.EvidenceError("trigger is not the current successful own CI")
        deadline = time.monotonic() + 45
        while True:
            try:
                status = copy.deepcopy(consumer._status(client, sha))
            except evidence.EvidenceError as exc:
                if str(exc) != "qualification status is absent":
                    raise
                status = None
            if status:
                target = evidence.RUN_URL_RE.fullmatch(
                    str(status.get("target_url", ""))
                )
                if not target or target.group("repo") != client.repo:
                    raise evidence.EvidenceError("unknown source producer target")
                producer_id = int(target.group("run_id"))
                producer = client.json(
                    f"repos/{client.repo}/actions/runs/{producer_id}"
                )
                if producer.get("status") == "completed":
                    result = verify_admission(client, producer_id)
                else:
                    result = {"verified": False, "reason": "producer not ready"}
                if result.get("verified") is True:
                    if (
                        result.get("candidate_sha") != sha
                        or result.get("source_run_id") != source_run_id
                        or result.get("source_attempt") != source_attempt
                        or client.json(
                            f"repos/{client.repo}/actions/runs/{source_run_id}"
                        )
                        != source
                    ):
                        raise evidence.EvidenceError("admission changed triggering CI")
                    if (
                        consumer._producer_run(client, result["producer_run_id"])[
                            "run_attempt"
                        ]
                        != result["producer_attempt"]
                        or consumer._status(client, sha) != status
                    ):
                        raise evidence.EvidenceError(
                            "producer/index changed after source verification"
                        )
                    return result
                if producer.get("status") == "completed":
                    return result
                rejected["reason"] = result.get("reason", "producer not ready")
            if time.monotonic() >= deadline:
                rejected["reason"] = "producer wait expired: " + rejected.get(
                    "reason", "qualification status is absent"
                )
                return rejected
            time.sleep(1)
    except (
        evidence.EvidenceError,
        KeyError,
        TypeError,
        ValueError,
        AttributeError,
        OSError,
        subprocess.SubprocessError,
    ) as exc:
        rejected["reason"] = str(exc)[:500]
        return rejected


def _write_status(
    client: evidence.GitHubClient, sha: str, state: str, target: str
) -> None:
    evidence._require_sha(sha)
    url = evidence.RUN_URL_RE.fullmatch(target)
    if not url or url.group("repo") != client.repo:
        raise evidence.EvidenceError("invalid admission publication URL")
    subprocess.run(
        [
            client.gh,
            "api",
            "--method",
            "POST",
            f"repos/{client.repo}/statuses/{sha}",
            "-f",
            f"state={state}",
            "-f",
            f"context={CONTEXT}",
            "-f",
            "description=Current trusted candidate validation",
            "-f",
            f"target_url={target}",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )


def _notification_sha(client: evidence.GitHubClient, run_id: int) -> str | None:
    # Failed/rerun producers can revoke a previously indexed success. Metadata
    # is authenticated, but this selector grants no positive qualification.
    if type(run_id) is not int or run_id < 1:
        raise evidence.EvidenceError("invalid notification identity")
    run = client.json(f"repos/{client.repo}/actions/runs/{run_id}")
    workflow = client.json(
        f"repos/{client.repo}/actions/workflows/candidate-qualification.yml"
    )
    if (
        run.get("id") != run_id
        or run.get("workflow_id") != workflow.get("id")
        or run.get("path") != consumer.WORKFLOW
        or workflow.get("path") != consumer.WORKFLOW
        or run.get("event") != "workflow_run"
        or run.get("repository", {}).get("full_name") != client.repo
    ):
        raise evidence.EvidenceError("untrusted notification identity")
    page = client.json(
        f"repos/{client.repo}/actions/runs/{run_id}/artifacts", "per_page=100"
    )
    count = page.get("total_count")
    if (
        type(count) is not int
        or not 0 <= count < 100
        or len(page["artifacts"]) != count
    ):
        raise evidence.EvidenceError("incomplete notification artifacts")
    matches = [
        (a, ARTIFACT_RE.fullmatch(str(a.get("name", "")))) for a in page["artifacts"]
    ]
    matches = [(a, m) for a, m in matches if m]
    if not matches:
        return None
    if len(matches) != 1 or matches[0][0].get("workflow_run", {}).get("id") != run_id:
        raise evidence.EvidenceError("ambiguous notification artifact")
    return matches[0][1][1]


def publish_admission(
    client: evidence.GitHubClient,
    producer_run_id: int,
    expected: dict,
    target: str,
    *,
    evidence_uploaded: bool = False,
) -> dict:
    # Only mutate the SHA selected by an authenticated original notification.
    # An older notification must never overwrite a newer producer's gate.
    sha = _notification_sha(client, producer_run_id)
    if sha is None:
        return {"published": False, "reason": "notification has no qualification"}
    index = consumer._status(client, sha)
    url = evidence.RUN_URL_RE.fullmatch(str(index.get("target_url", "")))
    if (
        not url
        or url.group("repo") != client.repo
        or int(url.group("run_id")) != producer_run_id
    ):
        return {"published": False, "reason": "notification superseded"}
    current = verify_admission(client, producer_run_id)
    valid = (
        evidence_uploaded is True
        and current.get("verified") is True
        and current == expected
        and expected.get("candidate_sha") == sha
    )
    # Re-read the index after the last actual consumer, including revocation.
    if consumer._status(client, sha) != index:
        return {"published": False, "reason": "index changed before publication"}
    _write_status(client, sha, "success" if valid else "failure", target)
    # Rollback may arrive during the status POST. Repair this publication if
    # its live proof changed; never claim an atomic transaction across APIs.
    if valid and verify_admission(client, producer_run_id) != expected:
        latest = consumer._status(client, sha)
        if latest == index or not consumer.consume_qualified(client, sha).get(
            "verified"
        ):
            _write_status(client, sha, "failure", target)
        valid = False
    return {"published": True, "verified": valid, "candidate_sha": sha}


def rollback(client: evidence.GitHubClient, target: str) -> dict:
    from scripts.ci_candidate_rollout import enabled

    # The latest rollback dispatch (including pending) disables the live
    # generation before this job starts; no unsupported Variables API writes.
    main = client.json(f"repos/{client.repo}/git/ref/heads/main")["object"]["sha"]
    if enabled(client, main):
        raise evidence.EvidenceError("a newer activation superseded rollback")
    pages = client.json(
        f"repos/{client.repo}/pulls", "state=open", "per_page=100", paginate=True
    )
    results = []
    for page in pages:
        if not isinstance(page, list):
            raise evidence.EvidenceError("incomplete rollback candidate listing")
        for pull in page:
            if (
                pull.get("user", {}).get("login") != "mergify[bot]"
                or pull.get("head", {}).get("repo", {}).get("full_name") != client.repo
                or pull.get("base", {}).get("repo", {}).get("full_name") != client.repo
                or pull.get("base", {}).get("ref") != "main"
                or not evidence.CANDIDATE_RE.fullmatch(
                    str(pull.get("head", {}).get("ref", ""))
                )
            ):
                continue
            sha = pull["head"]["sha"]
            evidence._require_sha(sha)
            consumed = consumer.consume_qualified(client, sha)
            # Full repairs and legitimate policy exemptions remain usable.
            # Mapped proofs cannot survive the now-disabled live switch.
            preserve = consumed.get("verified") is True and consumed.get("kind") in {
                "full",
                "engine-not-required",
            }
            if not preserve:
                _write_status(client, sha, "failure", target)
            results.append({"candidate_sha": sha, "preserved": preserve})
    return {"disabled": True, "candidates": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--reuse-api-connection", action="store_true")
    identity = parser.add_mutually_exclusive_group(required=False)
    identity.add_argument("--producer-run-id", type=int)
    identity.add_argument("--source-run-id", type=int)
    parser.add_argument("--source-attempt", type=int)
    parser.add_argument("--publish-target-url")
    parser.add_argument("--expected", type=Path)
    parser.add_argument(
        "--evidence-uploaded", choices=("true", "false"), default="false"
    )
    parser.add_argument("--rollback", action="store_true")
    parser.add_argument("--github-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    client = evidence.GitHubClient(args.repo)
    if args.rollback:
        result = rollback(client, args.publish_target_url)
    elif args.expected:
        result = publish_admission(
            client,
            args.producer_run_id,
            json.loads(args.expected.read_text()),
            args.publish_target_url,
            evidence_uploaded=args.evidence_uploaded == "true",
        )
    elif args.source_run_id is not None:
        result = verify_source_admission(client, args.source_run_id, args.source_attempt)
    else:
        result = verify_admission(client, args.producer_run_id)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    with args.github_output.open("a") as output:
        output.write("verified=" + str(result.get("verified", False)).lower() + "\n")
        if result.get("verified"):
            output.write("candidate_sha=" + result["candidate_sha"] + "\n")
            output.write("producer_run_id=" + str(result["producer_run_id"]) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
