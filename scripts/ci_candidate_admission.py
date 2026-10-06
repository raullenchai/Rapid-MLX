"""Deploy a trusted full-candidate consumer before queue enrollment."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--producer-run-id", type=int, required=True)
    parser.add_argument("--github-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify_admission(evidence.GitHubClient(args.repo), args.producer_run_id)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    with args.github_output.open("a") as output:
        output.write("verified=" + str(result["verified"]).lower() + "\n")
        if result["verified"]:
            output.write("candidate_sha=" + result["candidate_sha"] + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
