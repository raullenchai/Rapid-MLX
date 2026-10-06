"""Deploy a trusted full-candidate consumer before queue enrollment."""

from __future__ import annotations

import argparse
import copy
import json
import re
import subprocess
import time
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
        consumed = consumer.consume_full(client, sha)
        if consumed.get("verified") is not True or consumed.get("kind") != "full":
            raise evidence.EvidenceError(
                "full admission rejected: " + consumed.get("reason", "")
            )
        if (
            consumer._producer_run(client, producer_run_id) != run
            or consumer._status(client, sha) != status
        ):
            raise evidence.EvidenceError("producer/index changed before admission")
        # Repeat the actual artifact/source/base/tree verification, not just index.
        current = consumer.consume_full(client, sha)
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
        result.update(
            verified=True,
            candidate_sha=sha,
            producer_run_id=producer_run_id,
            producer_attempt=run["run_attempt"],
            source_run_id=consumed["qualification"]["source_run_id"],
            source_attempt=consumed["qualification"]["source_attempt"],
            kind="full",
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
                result = verify_admission(client, int(target.group("run_id")))
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    identity = parser.add_mutually_exclusive_group(required=True)
    identity.add_argument("--producer-run-id", type=int)
    identity.add_argument("--source-run-id", type=int)
    parser.add_argument("--source-attempt", type=int)
    parser.add_argument("--github-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    client = evidence.GitHubClient(args.repo)
    result = (
        verify_source_admission(client, args.source_run_id, args.source_attempt)
        if args.source_run_id is not None
        else verify_admission(client, args.producer_run_id)
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    with args.github_output.open("a") as output:
        output.write("verified=" + str(result["verified"]).lower() + "\n")
        if result["verified"]:
            output.write("candidate_sha=" + result["candidate_sha"] + "\n")
            output.write("producer_run_id=" + str(result["producer_run_id"]) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
