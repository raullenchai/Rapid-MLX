"""Opt-in mapped shadow execution inputs; complete candidate CI stays required."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from scripts import ci_candidate_qualification as producer
from scripts import queue_tree_evidence as evidence
from scripts.ci_main_qualification import qualify_main
from scripts.ci_mapped_execution import validate_manifests
from scripts.classify_ci_changes import source_canary_tests

CONTROL_PATH = "scripts/ci_candidate_execution.py"


def _current_run(client: evidence.GitHubClient, run: dict) -> None:
    runs = evidence._workflow_runs(client, evidence.CI_WORKFLOW, run["head_sha"])
    candidates = [
        r
        for r in runs
        if r.get("head_sha") == run["head_sha"]
        and r.get("head_branch") == run["head_branch"]
        and r.get("event") == "pull_request"
    ]
    if not candidates:
        raise evidence.EvidenceError("missing current shadow run")
    latest = max(
        candidates,
        key=lambda r: (
            evidence._field(r, "id", int),
            evidence._field(r, "run_attempt", int),
        ),
    )
    if (latest["id"], latest["run_attempt"]) != (run["id"], run["run_attempt"]):
        raise evidence.EvidenceError("shadow run was superseded")


def select_shadow(
    client: evidence.GitHubClient, run_id: int, head: str, base: str, enabled: bool
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "selected": False,
        "tests": [],
        "authorizes_reduced_ci": False,
    }
    if enabled is not True:
        return result
    try:
        evidence._require_sha(head)
        evidence._require_sha(base)
        if type(run_id) is not int or run_id < 1:
            raise evidence.EvidenceError("invalid shadow run")
        run = client.json(f"repos/{client.repo}/actions/runs/{run_id}")
        _, actual_base, _ = producer._candidate(client, run)
        _current_run(client, run)
        if (
            run.get("id") != run_id
            or run.get("head_sha") != head
            or actual_base != base
            or type(run.get("run_attempt")) is not int
            or run["run_attempt"] < 1
        ):
            raise evidence.EvidenceError("shadow candidate identity mismatch")
        if run.get("conclusion") in ("failure", "cancelled"):
            raise evidence.EvidenceError("shadow candidate was abandoned")
        tests = source_canary_tests(producer._paths(client, base, head))
        if not tests:
            raise evidence.EvidenceError("combined diff is not allowlisted")
        for path in (*producer.CONTROL_PATHS, CONTROL_PATH):
            if evidence._path_blob(client, head, path) != evidence._path_blob(
                client, base, path
            ):
                raise evidence.EvidenceError("shadow controller changed")
        if not qualify_main(client, base)["qualified"]:
            raise evidence.EvidenceError("shadow lacks current full main")
        current = client.json(f"repos/{client.repo}/actions/runs/{run_id}")
        _current_run(client, current)
        if (
            current.get("head_sha") != head
            or current.get("run_attempt") != run["run_attempt"]
            or current.get("conclusion") in ("failure", "cancelled")
        ):
            raise evidence.EvidenceError("shadow source changed")
        if (
            producer._candidate(client, current)[1] != base
            or not qualify_main(client, base)["qualified"]
        ):
            raise evidence.EvidenceError("shadow main/candidate changed")
        result.update(
            selected=True,
            tests=list(tests),
            head=head,
            base=base,
            source_run_id=run_id,
            source_attempt=run["run_attempt"],
        )
    except (
        evidence.EvidenceError,
        KeyError,
        TypeError,
        ValueError,
        AttributeError,
    ) as exc:
        result["reason"] = str(exc)[:500]
    return result


def pack_input(
    base: str,
    head: str,
    run_id: int,
    attempt: int,
    tests: list[str],
    baseline: dict,
    executed: dict,
    junit: bytes,
    coverage: bytes,
    tested_sha: str,
) -> dict:
    for sha in (base, head, tested_sha):
        evidence._require_sha(sha)
    if (
        type(run_id) is not int
        or run_id < 1
        or type(attempt) is not int
        or attempt < 1
        or tested_sha != head
    ):
        raise ValueError("pack identity is invalid")
    validate_manifests(baseline, executed, tests)
    cases = ET.fromstring(junit).findall(".//testcase")
    if (
        len(cases) != len(executed["nodes"])
        or not cases
        or any(
            c.find(tag) is not None
            for c in cases
            for tag in ("skipped", "failure", "error")
        )
    ):
        raise ValueError("JUnit is incomplete or failed")
    cov = ET.fromstring(coverage)
    if (
        cov.tag != "coverage"
        or int(cov.get("lines-valid", "0")) <= 0
        or not cov.findall(".//line")
    ):
        raise ValueError("coverage must measure production lines")
    return {
        "schema": producer.INPUT_SCHEMA,
        "scope": "candidate-mapped-only",
        "head": head,
        "base": base,
        "source_run_id": run_id,
        "source_attempt": attempt,
        "tested_sha": tested_sha,
        "tests": tests,
        "baseline": baseline,
        "executed": executed,
        "diagnostics": {
            "junit_sha256": hashlib.sha256(junit).hexdigest(),
            "coverage_sha256": hashlib.sha256(coverage).hexdigest(),
            "production_changed_line_pilot_proven": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    select = sub.add_parser("select")
    select.add_argument("--repo", required=True)
    select.add_argument("--enabled", choices=("true", "false"), default="false")
    select.add_argument("--github-output", required=True, type=Path)
    select.add_argument("--run-id", required=True, type=int)
    select.add_argument("--head", required=True)
    select.add_argument("--base", required=True)
    pack = sub.add_parser("pack")
    for name in ("baseline", "executed", "junit", "coverage", "output"):
        pack.add_argument("--" + name, required=True, type=Path)
    for name in ("head", "base"):
        pack.add_argument("--" + name, required=True)
    pack.add_argument("--run-id", required=True, type=int)
    pack.add_argument("--attempt", required=True, type=int)
    pack.add_argument("--tests", required=True, nargs="+")
    args = parser.parse_args()
    if args.command == "select":
        result = select_shadow(
            evidence.GitHubClient(args.repo),
            args.run_id,
            args.head,
            args.base,
            args.enabled == "true",
        )
        with args.github_output.open("a") as output:
            output.write("candidate_shadow=" + str(result["selected"]).lower() + "\n")
            output.write("candidate_shadow_tests=" + " ".join(result["tests"]) + "\n")
        print(json.dumps(result))
    else:
        tested = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
        subprocess.run(["git", "diff", "--quiet", "HEAD", "--"], check=True)
        record = pack_input(
            args.base,
            args.head,
            args.run_id,
            args.attempt,
            args.tests,
            json.loads(args.baseline.read_text()),
            json.loads(args.executed.read_text()),
            args.junit.read_bytes(),
            args.coverage.read_bytes(),
            tested,
        )
        args.output.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
