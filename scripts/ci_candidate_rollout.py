"""Default-off candidate routing and live trusted qualification policy."""

from __future__ import annotations

import argparse
import base64
import copy
import json
import subprocess
import zipfile
from pathlib import Path

from scripts import ci_candidate_execution as execution
from scripts import ci_candidate_qualification as producer
from scripts import queue_tree_evidence as evidence
from scripts.classify_ci_changes import classify as classify_lanes

VARIABLE = "RAPID_MLX_CANDIDATE_CANARY"
ADMISSION_CONDITION = "check-success = @github-actions/candidate-admission/ci"
CONTROLS = (
    *producer.CONTROL_PATHS,
    ".mergify.yml",
    "scripts/ci_candidate_rollout.py",
    "scripts/ci_candidate_execution.py",
    "scripts/ci_candidate_consumer.py",
    "scripts/ci_candidate_admission.py",
    "scripts/ci_candidate_mapped_transport.py",
    ".github/workflows/candidate-admission.yml",
)


class ImmutableContentsClient:
    """Reuse commit-pinned file metadata within one CLI; live evidence stays fresh."""

    def __init__(self, client: evidence.GitHubClient) -> None:
        self._client = client
        self._contents: dict[tuple[str, str], dict] = {}

    def __getattr__(self, name):
        return getattr(self._client, name)

    def json(self, endpoint: str, *fields: str, paginate: bool = False):
        immutable = (
            endpoint.startswith(f"repos/{self.repo}/contents/")
            and len(fields) == 1
            and fields[0].startswith("ref=")
            and evidence.SHA_RE.fullmatch(fields[0][4:]) is not None
            and not paginate
        )
        key = (endpoint, fields[0]) if immutable else None
        if key in self._contents:
            return copy.deepcopy(self._contents[key])
        value = self._client.json(endpoint, *fields, paginate=paginate)
        if (
            immutable
            and isinstance(value, dict)
            and value.get("type") == "file"
            and isinstance(value.get("sha"), str)
            and evidence.SHA_RE.fullmatch(value["sha"]) is not None
        ):
            self._contents[key] = copy.deepcopy(value)
        return value


def enabled(client: evidence.GitHubClient, trusted: str) -> dict:
    # Repository variables are a startup hint, not a live authorization API:
    # GITHUB_TOKEN cannot request Variables permissions. Use this controller's
    # authenticated latest dispatch, readable with existing actions:read.
    path = ".github/workflows/candidate-admission.yml"
    workflow = client.json(
        f"repos/{client.repo}/actions/workflows/candidate-admission.yml"
    )
    if workflow.get("path") != path:
        raise evidence.EvidenceError("unknown rollout controller")
    pages = client.json(
        f"repos/{client.repo}/actions/workflows/{workflow['id']}/runs",
        "event=workflow_dispatch",
        "per_page=100",
        paginate=True,
    )
    runs = []
    for page in pages:
        records = page.get("workflow_runs")
        if not isinstance(records, list):
            raise evidence.EvidenceError("incomplete rollout generation listing")
        runs.extend(records)
    if not runs:
        return {}
    run = max(
        runs,
        key=lambda r: (
            evidence._field(r, "run_number", int),
            evidence._field(r, "run_attempt", int),
        ),
    )
    if (
        run.get("path") != path
        or run.get("workflow_id") != workflow.get("id")
        or run.get("event") != "workflow_dispatch"
        or run.get("head_branch") != "main"
        or run.get("repository", {}).get("full_name") != client.repo
        or run.get("actor", {}).get("type") != "User"
        or run.get("triggering_actor", {}).get("type") != "User"
        or run.get("status") != "completed"
        or run.get("conclusion") != "success"
    ):
        return {}
    pinned(client, run["head_sha"], trusted)
    jobs = evidence._successful_jobs(client, run)
    evidence._require_unique_success(jobs, "activate")
    rollback = [j for j in jobs if j.get("name") == "rollback"]
    if len(rollback) != 1 or rollback[0].get("conclusion") != "skipped":
        raise evidence.EvidenceError("latest dispatch did not activate rollout")
    return {"run_id": run["id"], "attempt": run["run_attempt"], "head": run["head_sha"]}


def enrolled(client: evidence.GitHubClient, base: str) -> bool:
    import yaml

    response = client.json(f"repos/{client.repo}/contents/.mergify.yml", f"ref={base}")
    if response.get("type") != "file" or response.get("encoding") != "base64":
        raise evidence.EvidenceError("missing queue policy")
    try:
        policy = yaml.safe_load(base64.b64decode(response["content"], validate=False))
    except yaml.YAMLError as exc:
        raise evidence.EvidenceError("invalid queue policy") from exc
    queues = policy["queue_rules"]
    if len(queues) != 2 or {q["name"] for q in queues} != {"mac-batch", "no-mac-batch"}:
        raise evidence.EvidenceError("unknown queue enrollment")
    return all(ADMISSION_CONDITION in q.get("merge_conditions", []) for q in queues)


def ready(client: evidence.GitHubClient, base: str) -> dict:
    generation = enabled(client, base)
    return generation if generation and enrolled(client, base) else {}


def pinned(client: evidence.GitHubClient, sha: str, trusted: str) -> None:
    for path in CONTROLS:
        if evidence._path_blob(client, sha, path) != evidence._path_blob(
            client, trusted, path
        ):
            raise evidence.EvidenceError("candidate changes rollout controller")


def select_route(
    client: evidence.GitHubClient, run_id: int, head: str, base: str
) -> dict:
    result = {"reduced": False, "tests": []}
    try:
        generation = ready(client, base)
        if not generation:
            return result
        selected = execution.select_shadow(client, run_id, head, base, True)
        if not selected.get("selected"):
            raise evidence.EvidenceError(
                selected.get("reason", "candidate shadow rejected")
            )
        pinned(client, head, base)
        if ready(client, base) != generation:
            raise evidence.EvidenceError("rollout changed during selection")
        result.update(reduced=True, tests=selected["tests"], head=head, base=base)
    except (
        evidence.EvidenceError,
        KeyError,
        TypeError,
        ValueError,
        AttributeError,
        ImportError,
    ) as exc:
        result["reason"] = str(exc)[:500]
    return result


def _not_required(client: evidence.GitHubClient, source_run: int, trusted: str) -> dict:
    run = client.json(f"repos/{client.repo}/actions/runs/{source_run}")
    pull, base, tree = producer._candidate(client, run)
    latest = producer._latest(client, run["head_sha"], run["head_branch"])
    if (latest["id"], latest["run_attempt"]) != (run["id"], run["run_attempt"]):
        raise evidence.EvidenceError("policy exemption source is stale")
    policy = classify_lanes(producer._paths(client, base, run["head_sha"]))
    if policy.engine:
        raise evidence.EvidenceError("combined policy requires engine")
    pinned(client, run["head_sha"], trusted)
    jobs = evidence._successful_jobs(client, run)
    for name in ("changes", "lint", "tests"):
        evidence._require_unique_success(jobs, name)
    lane = [
        j
        for j in jobs
        if j.get("name") in ("merge-lane-mac", "merge-lane-no-mac")
        and j.get("conclusion") == "success"
    ]
    if len(lane) != 1 or lane[0]["name"] != (
        "merge-lane-mac" if policy.desktop else "merge-lane-no-mac"
    ):
        raise evidence.EvidenceError("missing policy merge lane")
    for name in (
        "engine-contracts",
        "type-check",
        "mlx-bound-guard",
        "test-matrix",
        "test-apple-silicon",
        "linux-coverage",
        "changed-lines-coverage",
        "l1-smoke",
        "source-canary-unit",
        "candidate-canary-unit",
    ):
        matches = [
            j
            for j in jobs
            if j.get("name") == name or str(j.get("name", "")).startswith(name + " (")
        ]
        if (
            len(matches) != 1
            or matches[0].get("status") != "completed"
            or matches[0].get("conclusion") != "skipped"
        ):
            raise evidence.EvidenceError("policy exemption has unexpected engine job")
    if producer._latest(
        client, run["head_sha"], run["head_branch"]
    ) != latest or producer._candidate(client, run)[1:] != (base, tree):
        raise evidence.EvidenceError("policy exemption changed")
    if client.json(f"repos/{client.repo}/git/ref/heads/main")["object"]["sha"] != base:
        raise evidence.EvidenceError("policy exemption main changed")
    return {
        "schema": producer.SCHEMA,
        "repository": client.repo,
        "qualified": True,
        "authorizes_reduced_ci": False,
        "kind": "engine-not-required",
        "candidate_sha": run["head_sha"],
        "candidate_tree": tree,
        "candidate_pr": pull["number"],
        "base_sha": base,
        "source_run_id": source_run,
        "source_attempt": run["run_attempt"],
    }


def qualify_source(
    client: evidence.GitHubClient, source_run: int, trusted: str
) -> dict:
    full = producer.qualify_candidate(client, source_run, trusted)
    if full.get("qualified"):
        return full
    try:
        run = client.json(f"repos/{client.repo}/actions/runs/{source_run}")
        _, base, _ = producer._candidate(client, run)
        if not classify_lanes(producer._paths(client, base, run["head_sha"])).engine:
            return _not_required(client, source_run, trusted)
        generation = ready(client, base)
        if not generation:
            raise evidence.EvidenceError("mapped qualification rollout disabled")
        pinned(client, run["head_sha"], trusted)
        from scripts.ci_candidate_mapped_transport import _input

        proof = _input(client, run)
        mapped = producer.qualify_candidate(
            client, source_run, trusted, mapped_enabled=True, mapped_input=proof
        )
        if (
            not mapped.get("qualified")
            or mapped.get("kind") != "mapped"
            or ready(client, base) != generation
        ):
            raise evidence.EvidenceError("mapped qualification rejected or revoked")
        return dict(mapped, rollout_generation=generation)
    except (
        evidence.EvidenceError,
        KeyError,
        TypeError,
        ValueError,
        AttributeError,
        OSError,
        ImportError,
        subprocess.SubprocessError,
        zipfile.BadZipFile,
        UnicodeError,
    ) as exc:
        return dict(
            full, qualified=False, authorizes_reduced_ci=False, reason=str(exc)[:500]
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--run-id", required=True, type=int)
    parser.add_argument("--head", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--github-output", required=True, type=Path)
    args = parser.parse_args()
    result = select_route(
        ImmutableContentsClient(evidence.GitHubClient(args.repo)),
        args.run_id,
        args.head,
        args.base,
    )
    with args.github_output.open("a") as output:
        output.write("candidate_reduced=" + str(result["reduced"]).lower() + "\n")
        if result["reduced"]:
            output.write(
                "candidate_shadow=true\ncandidate_shadow_tests="
                + " ".join(result["tests"])
                + "\n"
            )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
