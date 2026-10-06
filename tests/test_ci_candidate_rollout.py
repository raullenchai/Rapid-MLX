"""Real mapped source proof stays separate from full and live rollback policy."""

import base64

import pytest
import yaml

from scripts import ci_candidate_qualification as producer
from scripts import ci_candidate_rollout as rollout
from scripts import queue_tree_evidence as evidence
from tests.test_ci_candidate_mapped_transport import setup
from tests.test_queue_tree_evidence import CANDIDATE, MAIN, REPO, TRUSTED


def configured(monkeypatch):
    client, proof, job, artifact = setup(monkeypatch)
    for path in rollout.CONTROLS:
        for sha in (CANDIDATE, MAIN, TRUSTED):
            client.blobs[sha, path] = "f" * 40
    client.responses[f"repos/{REPO}/git/ref/heads/main"] = {"object": {"sha": MAIN}}
    original = client.json
    policy = {
        "queue_rules": [
            {"name": n, "merge_conditions": [rollout.ADMISSION_CONDITION]}
            for n in ("mac-batch", "no-mac-batch")
        ]
    }
    generation = {
        "id": 300,
        "run_attempt": 1,
        "workflow_id": 3,
        "run_number": 1,
        "actor": {"type": "User"},
        "triggering_actor": {"type": "User"},
        "path": ".github/workflows/candidate-admission.yml",
        "event": "workflow_dispatch",
        "head_branch": "main",
        "head_sha": TRUSTED,
        "repository": {"full_name": REPO},
        "status": "completed",
        "conclusion": "success",
    }
    client.job_records[300] = [
        {
            "id": 3001,
            "run_attempt": 1,
            "name": "activate",
            "status": "completed",
            "conclusion": "success",
        },
        {
            "id": 3002,
            "run_attempt": 1,
            "name": "rollback",
            "status": "completed",
            "conclusion": "skipped",
        },
    ]

    def read(endpoint, *fields, **kw):
        if endpoint.endswith("/actions/workflows/candidate-admission.yml"):
            return {"id": 3, "path": ".github/workflows/candidate-admission.yml"}
        if endpoint.endswith("/actions/workflows/3/runs"):
            return [{"workflow_runs": [generation] if generation else []}]
        if endpoint.endswith("/contents/.mergify.yml"):
            return {
                "type": "file",
                "encoding": "base64",
                "sha": "f" * 40,
                "content": base64.b64encode(yaml.safe_dump(policy).encode()).decode(),
            }
        return original(endpoint, *fields, **kw)

    monkeypatch.setattr(client, "json", read)
    return client, proof, job, artifact, generation, policy


def test_actual_zip_mapped_qualification_requires_live_enrollment(monkeypatch):
    client, *_ = configured(monkeypatch)
    result = rollout.qualify_source(client, 20, TRUSTED)
    assert result["qualified"] and result["kind"] == "mapped"
    assert result["authorizes_reduced_ci"]
    assert result["schema"] not in evidence.SCHEMAS.values()


@pytest.mark.parametrize(
    "bad",
    [
        "off",
        "absent",
        "no-gate",
        "one-queue",
        "bare-context",
        "controller",
        "red-main",
        "coverage",
        "missing-node",
        "mixed",
        "critical",
        "source-cancelled",
        "api",
    ],
)
def test_real_mapped_path_fails_closed(monkeypatch, bad):
    client, proof, job, _, generation, policy = configured(monkeypatch)
    if bad == "off":
        generation["conclusion"] = "cancelled"
    elif bad == "absent":
        generation.clear()
    elif bad in {"no-gate", "bare-context"}:
        policy["queue_rules"][0]["merge_conditions"] = (
            [] if bad == "no-gate" else ["check-success = candidate-admission/ci"]
        )
    elif bad == "one-queue":
        policy["queue_rules"].pop()
    elif bad == "controller":
        client.blobs[CANDIDATE, "scripts/ci_candidate_rollout.py"] = "e" * 40
    elif bad == "red-main":
        monkeypatch.setattr(producer, "qualify_main", lambda *a: {"qualified": False})
    elif bad == "coverage":
        job["steps"][1]["conclusion"] = "failure"
    elif bad == "missing-node":
        proof["executed"]["reports"].pop()
    elif bad in {"mixed", "critical"}:
        files = client.responses[f"repos/{REPO}/compare/{MAIN}...{CANDIDATE}"]["files"]
        files.append(
            {"filename": "rapid_mlx/server.py" if bad == "critical" else "unmapped.py"}
        )
    elif bad == "source-cancelled":
        client.responses[f"repos/{REPO}/actions/runs/20"]["conclusion"] = "cancelled"
    else:
        monkeypatch.setattr(
            rollout,
            "ready",
            lambda *a: (_ for _ in ()).throw(evidence.EvidenceError("API failed")),
        )
    result = rollout.qualify_source(client, 20, TRUSTED)
    assert not result["qualified"] and not result["authorizes_reduced_ci"]


def test_switch_revoked_after_actual_zip_validation(monkeypatch):
    client, *_ = configured(monkeypatch)
    calls = 0
    original = rollout.ready

    def ready(*args):
        nonlocal calls
        calls += 1
        return original(*args) if calls == 1 else False

    monkeypatch.setattr(rollout, "ready", ready)
    result = rollout.qualify_source(client, 20, TRUSTED)
    assert calls == 2 and not result["qualified"]


def test_full_repair_never_depends_on_reduced_switch(monkeypatch):
    from tests.test_ci_candidate_qualification import fixture

    client, _ = fixture(monkeypatch)
    monkeypatch.setattr(
        rollout, "ready", lambda *a: pytest.fail("full repair used rollout")
    )
    assert rollout.qualify_source(client, 20, TRUSTED)["kind"] == "full"


@pytest.mark.parametrize(
    "bad",
    [
        "pending",
        "failure",
        "wrong-branch",
        "wrong-path",
        "wrong-repo",
        "wrong-workflow",
        "no-activate",
        "rollback",
        "pin",
    ],
)
def test_live_generation_is_authenticated_and_fails_closed(monkeypatch, bad):
    client, _, _, _, generation, _ = configured(monkeypatch)
    if bad == "pending":
        generation["status"] = "in_progress"
    elif bad == "failure":
        generation["conclusion"] = "failure"
    elif bad == "wrong-branch":
        generation["head_branch"] = "untrusted"
    elif bad == "wrong-path":
        generation["path"] = ".github/workflows/other.yml"
    elif bad == "wrong-repo":
        generation["repository"]["full_name"] = "fork/repo"
    elif bad == "wrong-workflow":
        generation["workflow_id"] = 4
    elif bad == "no-activate":
        client.job_records[300][0]["conclusion"] = "skipped"
    elif bad == "rollback":
        client.job_records[300][1]["conclusion"] = "success"
    else:
        client.blobs[TRUSTED, "scripts/ci_candidate_rollout.py"] = "e" * 40
    assert not rollout.qualify_source(client, 20, TRUSTED)["qualified"]


def test_generation_changes_even_between_two_successful_activations(monkeypatch):
    client, *_ = configured(monkeypatch)
    calls = 0
    original = rollout.ready

    def read(*args):
        nonlocal calls
        calls += 1
        generation = original(*args)
        return generation if calls == 1 else dict(generation, run_id=301)

    monkeypatch.setattr(rollout, "ready", read)
    assert not rollout.qualify_source(client, 20, TRUSTED)["qualified"]


def test_actual_route_uses_whole_combined_mapping_and_rechecks_generation(monkeypatch):
    client, proof, *_ = configured(monkeypatch)
    monkeypatch.setattr(
        rollout.execution, "qualify_main", lambda *a: {"qualified": True}
    )
    result = rollout.select_route(client, 20, CANDIDATE, MAIN)
    assert result["reduced"] and result["tests"] == proof["tests"]
    client.responses[f"repos/{REPO}/compare/{MAIN}...{CANDIDATE}"]["files"].append(
        {"filename": "README.md"}
    )
    assert not rollout.select_route(client, 20, CANDIDATE, MAIN)["reduced"]


@pytest.mark.parametrize("path", ["README.md", "apps/rapid-mac/Sources/RapidApp.swift"])
def test_trusted_engine_policy_exemption_does_not_require_activation(monkeypatch, path):
    client, _, _, _, generation, _ = configured(monkeypatch)
    generation.clear()
    client.responses[f"repos/{REPO}/compare/{MAIN}...{CANDIDATE}"]["files"] = [
        {"filename": path}
    ]
    if path == "README.md":
        next(j for j in client.job_records[20] if j["name"] == "merge-lane-mac")[
            "name"
        ] = "merge-lane-no-mac"
    for job in client.job_records[20]:
        if job["name"] not in (
            "changes",
            "lint",
            "tests",
            "merge-lane-mac",
            "merge-lane-no-mac",
        ):
            job["conclusion"] = "skipped"
    result = rollout.qualify_source(client, 20, TRUSTED)
    assert result["qualified"] and result["kind"] == "engine-not-required"
    assert not result["authorizes_reduced_ci"]
    client.responses[f"repos/{REPO}/compare/{MAIN}...{CANDIDATE}"]["files"].append(
        {"filename": "rapid_mlx/server.py"}
    )
    assert not rollout.qualify_source(client, 20, TRUSTED)["qualified"]


@pytest.mark.parametrize("cached", [False, True])
def test_mapped_real_archive_consumption_and_live_revocation(monkeypatch, cached):
    import io
    import json
    import zipfile
    from types import SimpleNamespace

    from scripts import ci_candidate_admission as admission
    from scripts import ci_candidate_consumer as consumer
    from tests.test_ci_candidate_consumer import setup as setup_consumer

    client, record, _, _, _ = setup_consumer(monkeypatch)
    download_full = consumer.subprocess.run
    mapped, proof, _, artifact, generation, _ = configured(monkeypatch)
    # Preserve actual producer/status/archive metadata; use mapped source jobs,
    # strict activation policy, controller blobs and current combined diff.
    client.job_records[20] = mapped.job_records[20]
    client.job_records[300] = mapped.job_records[300]
    client.blobs.update(mapped.blobs)
    client.responses.update(
        {k: v for k, v in mapped.responses.items() if "/runs/20/artifacts" not in k}
    )
    client.responses[f"repos/{REPO}/actions/runs/20/artifacts"] = {
        "total_count": 1,
        "artifacts": [dict(artifact, id=102)],
    }
    original = client.json
    mapped_json = mapped.json

    def read(endpoint, *args, **kwargs):
        if (
            "/actions/workflows/3/runs" in endpoint
            or endpoint.endswith("/actions/workflows/candidate-admission.yml")
            or endpoint.endswith("/contents/.mergify.yml")
        ):
            return mapped_json(endpoint, *args, **kwargs)
        return original(endpoint, *args, **kwargs)

    monkeypatch.setattr(client, "json", read)

    def download(command, **kwargs):
        if command[-1].endswith("/artifacts/102/zip"):
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as archive:
                archive.writestr("candidate-mapped-input.json", json.dumps(proof))
            return SimpleNamespace(stdout=buf.getvalue())
        return download_full(command, **kwargs)

    monkeypatch.setattr(consumer.subprocess, "run", download)
    qualified = rollout.qualify_source(client, 20, MAIN)
    assert qualified["qualified"], qualified
    record.clear()
    record.update(qualified)
    assert not consumer.consume_full(client, CANDIDATE)["verified"]
    assert consumer.consume_qualified(client, CANDIDATE)["kind"] == "mapped"
    reads = []
    raw_json = client.json

    def counted(endpoint, *fields, **kwargs):
        reads.append((endpoint, fields))
        return raw_json(endpoint, *fields, **kwargs)

    monkeypatch.setattr(client, "json", counted)
    if cached:
        client = rollout.ImmutableContentsClient(client)
    accepted = admission.verify_admission(client, 100)
    assert admission.verify_admission(client, 100) == accepted
    contents = [r for r in reads if "/contents/" in r[0]]
    if cached:
        assert len(contents) == len(set(contents))
        assert len(contents) < 100
    else:
        assert len(contents) > 800
    assert accepted["verified"] and accepted["rollout_generation"]["run_id"] == 300
    generation["status"] = "in_progress"  # Rollback is visible before its job executes.
    assert not consumer.consume_qualified(client, CANDIDATE)["verified"]
    assert not admission.verify_admission(client, 100)["verified"]


@pytest.mark.parametrize(
    "bad",
    [
        None,
        "coverage",
        "cancelled",
        "static",
        "unexpected-full",
        "source-route",
        "source-preflight",
        "main",
        "reuse",
        "not-selected",
    ],
)
def test_real_stable_aggregate_requires_exact_mapped_route(bad):
    import re
    import subprocess
    from pathlib import Path

    jobs = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / ".github/workflows/ci.yml").read_text()
    )["jobs"]
    values = {f"needs.{job}.result": "success" for job in jobs["tests"]["needs"]}
    for job in (
        "source-canary-unit",
        "test-matrix",
        "test-apple-silicon",
        "linux-coverage",
        "changed-lines-coverage",
        "l1-smoke",
    ):
        values[f"needs.{job}.result"] = "skipped"
    values.update(
        {
            "needs.changes.outputs.reuse_ci": "false",
            "needs.changes.outputs.engine": "true",
            "needs.changes.outputs.full_gate": "true",
            "needs.changes.outputs.source_canary": "false",
            "needs.changes.outputs.source_preflight": "false",
            "needs.changes.outputs.candidate_shadow": "true",
            "needs.changes.outputs.candidate_reduced": "true",
            "github.event_name": "pull_request",
        }
    )
    if bad in {"coverage", "cancelled"}:
        values["needs.candidate-canary-unit.result"] = (
            "failure" if bad == "coverage" else "cancelled"
        )
    elif bad == "static":
        values["needs.type-check.result"] = "failure"
    elif bad == "unexpected-full":
        values["needs.test-matrix.result"] = "success"
    elif bad == "source-route":
        values["needs.changes.outputs.source_canary"] = "true"
    elif bad == "source-preflight":
        values["needs.changes.outputs.source_preflight"] = "true"
    elif bad == "main":
        values["github.event_name"] = "push"
    elif bad == "reuse":
        values["needs.changes.outputs.reuse_ci"] = "true"
    elif bad == "not-selected":
        values["needs.changes.outputs.candidate_shadow"] = "false"
    script = re.sub(
        r"\$\{\{\s*(.*?)\s*\}\}",
        lambda m: values.get(m[1], ""),
        jobs["tests"]["steps"][0]["run"],
    )
    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert (result.returncode == 0) is (bad is None), result.stdout + result.stderr


def test_full_main_backstop_and_mandatory_queue_policy_are_real_workflow_conditions():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    jobs = yaml.safe_load((root / ".github/workflows/ci.yml").read_text())["jobs"]
    assert (
        "vars.RAPID_MLX_CANDIDATE_CANARY != 'true'" in jobs["queue-tree-evidence"]["if"]
    )
    for job in (
        "test-matrix",
        "test-apple-silicon",
        "linux-coverage",
        "changed-lines-coverage",
        "l1-smoke",
    ):
        assert "needs.changes.outputs.candidate_reduced != 'true'" in jobs[job]["if"]
    queues = yaml.safe_load((root / ".mergify.yml").read_text())["queue_rules"]
    assert all(rollout.ADMISSION_CONDITION in q["merge_conditions"] for q in queues)
    assert all(rollout.ADMISSION_CONDITION not in q["queue_conditions"] for q in queues)


@pytest.mark.parametrize(
    "bad", ["workflow", "listing", "missing-policy", "invalid-yaml"]
)
def test_unsupported_authorization_apis_force_ordinary_full(monkeypatch, bad):
    client, *_ = configured(monkeypatch)
    original = client.json

    def read(endpoint, *args, **kwargs):
        if bad == "workflow" and endpoint.endswith("/candidate-admission.yml"):
            return {"id": 3, "path": "wrong"}
        if bad == "listing" and endpoint.endswith("/workflows/3/runs"):
            return [{"workflow_runs": None}]
        if endpoint.endswith("/contents/.mergify.yml"):
            if bad == "missing-policy":
                return {"type": "dir"}
            if bad == "invalid-yaml":
                return {
                    "type": "file",
                    "encoding": "base64",
                    "content": base64.b64encode(b"[").decode(),
                }
        return original(endpoint, *args, **kwargs)

    monkeypatch.setattr(client, "json", read)
    assert not rollout.select_route(client, 20, CANDIDATE, MAIN)["reduced"]


def test_selection_rejects_live_aba_and_default_off(monkeypatch):
    client, _, _, _, generation, _ = configured(monkeypatch)
    monkeypatch.setattr(
        rollout.execution, "qualify_main", lambda *a: {"qualified": True}
    )
    original = rollout.ready
    calls = 0

    def read(*args):
        nonlocal calls
        calls += 1
        result = original(*args)
        return result if calls == 1 else dict(result, attempt=2)

    monkeypatch.setattr(rollout, "ready", read)
    assert not rollout.select_route(client, 20, CANDIDATE, MAIN)["reduced"]
    monkeypatch.setattr(rollout, "ready", original)
    generation.clear()
    assert rollout.select_route(client, 20, CANDIDATE, MAIN) == {
        "reduced": False,
        "tests": [],
    }


@pytest.mark.parametrize(
    "bad", ["engine", "stale", "lane", "engine-job", "changed", "main"]
)
def test_policy_exemption_rejects_wrong_scope_and_live_mutations(monkeypatch, bad):
    client, *_ = configured(monkeypatch)
    client.responses[f"repos/{REPO}/compare/{MAIN}...{CANDIDATE}"]["files"] = [
        {"filename": "README.md"}
    ]
    for job in client.job_records[20]:
        if job["name"] == "merge-lane-mac":
            job["name"] = "merge-lane-no-mac"
        elif job["name"] not in ("changes", "lint", "tests"):
            job["conclusion"] = "skipped"
    if bad == "engine":
        client.responses[f"repos/{REPO}/compare/{MAIN}...{CANDIDATE}"]["files"].append(
            {"filename": "rapid_mlx/server.py"}
        )
    elif bad == "lane":
        next(j for j in client.job_records[20] if j["name"] == "merge-lane-no-mac")[
            "conclusion"
        ] = "failure"
    elif bad == "engine-job":
        next(j for j in client.job_records[20] if j["name"] == "engine-contracts")[
            "conclusion"
        ] = "success"
    elif bad == "main":
        client.responses[f"repos/{REPO}/git/ref/heads/main"]["object"]["sha"] = TRUSTED
    else:
        original = producer._latest
        calls = 0

        def latest(*args):
            nonlocal calls
            calls += 1
            result = original(*args)
            return dict(result, id=21) if bad == "stale" or calls > 1 else result

        monkeypatch.setattr(producer, "_latest", latest)
    with pytest.raises(evidence.EvidenceError):
        rollout._not_required(client, 20, TRUSTED)


@pytest.mark.parametrize("selected", [True, False])
def test_actual_route_cli_emits_only_valid_mapped_test_selection(
    monkeypatch, tmp_path, selected
):
    import runpy
    import sys

    client, _, _, _, generation, _ = configured(monkeypatch)
    monkeypatch.setattr(
        rollout.execution, "qualify_main", lambda *a: {"qualified": True}
    )
    if not selected:
        generation.clear()
    monkeypatch.setattr(evidence, "GitHubClient", lambda repo: client)
    output = tmp_path / "route"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "route",
            "--repo",
            REPO,
            "--run-id",
            "20",
            "--head",
            CANDIDATE,
            "--base",
            MAIN,
            "--github-output",
            str(output),
        ],
    )
    with pytest.warns(RuntimeWarning, match="found in sys.modules"):
        runpy.run_module("scripts.ci_candidate_rollout", run_name="__main__")
    values = dict(line.split("=", 1) for line in output.read_text().splitlines())
    assert values["candidate_reduced"] == str(selected).lower()
    assert ("candidate_shadow_tests" in values) is selected


@pytest.mark.parametrize("case", ["equal", "different", "rejected"])
def test_producer_cli_rechecks_exact_uploaded_record_before_index(
    monkeypatch, tmp_path, case
):
    import json
    import runpy
    import sys

    from tests.test_ci_candidate_qualification import fixture

    client, _ = fixture(monkeypatch)
    result = rollout.qualify_source(client, 20, TRUSTED)
    if case == "different":
        result = dict(result, source_attempt=2)
    elif case == "rejected":
        client.responses[f"repos/{REPO}/actions/runs/20"]["conclusion"] = "cancelled"
        result = rollout.qualify_source(client, 20, TRUSTED)
    expected = tmp_path / "expected"
    expected.write_text(json.dumps(result))
    monkeypatch.setattr(evidence, "GitHubClient", lambda repo: client)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "producer",
            "--repo",
            REPO,
            "--source-run-id",
            "20",
            "--trusted-ref",
            TRUSTED,
            "--expected",
            str(expected),
            "--output",
            str(tmp_path / "out"),
        ],
    )
    with pytest.warns(RuntimeWarning, match="found in sys.modules"):
        if case == "equal":
            runpy.run_module("scripts.ci_candidate_qualification", run_name="__main__")
            assert json.loads((tmp_path / "out").read_text()) == result
        else:
            with pytest.raises(evidence.EvidenceError):
                runpy.run_module(
                    "scripts.ci_candidate_qualification", run_name="__main__"
                )
            assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "response",
    [
        {"type": "dir"},
        {
            "type": "file",
            "encoding": "base64",
            "content": base64.b64encode(b"[").decode(),
        },
    ],
)
def test_queue_enrollment_parser_rejects_missing_or_malformed_policy(
    monkeypatch, response
):
    client, *_ = configured(monkeypatch)
    monkeypatch.setattr(client, "json", lambda *a, **kw: response)
    with pytest.raises(evidence.EvidenceError):
        rollout.enrolled(client, MAIN)


@pytest.mark.parametrize(
    "endpoint,fields,paginate",
    [
        (f"repos/{REPO}/contents/AGENTS.md", ("ref=main",), False),
        (f"repos/{REPO}/contents/AGENTS.md", (f"ref={MAIN}", "x=y"), False),
        (f"repos/{REPO}/contents/AGENTS.md", (), False),
        (f"repos/{REPO}/contents/AGENTS.md", (f"ref={MAIN}",), True),
        ("repos/other/repo/contents/AGENTS.md", (f"ref={MAIN}",), False),
        (f"repos/{REPO}/git/ref/heads/main", (f"ref={MAIN}",), False),
        (f"repos/{REPO}/actions/runs/20", (f"ref={MAIN}",), False),
        (f"repos/{REPO}/commits/{MAIN}/status", (f"ref={MAIN}",), False),
    ],
)
def test_immutable_cache_never_caches_mutable_or_ambiguous_queries(
    endpoint, fields, paginate
):
    from types import SimpleNamespace

    calls = []

    def read(*args, **kwargs):
        calls.append((args, kwargs))
        return {"type": "file", "sha": MAIN, "sequence": len(calls)}

    client = rollout.ImmutableContentsClient(SimpleNamespace(repo=REPO, json=read))
    assert client.json(endpoint, *fields, paginate=paginate)["sequence"] == 1
    assert client.json(endpoint, *fields, paginate=paginate)["sequence"] == 2


@pytest.mark.parametrize(
    "bad",
    [
        None,
        [],
        {},
        {"type": "dir", "sha": MAIN},
        {"type": "file", "sha": 1},
        {"type": "file", "sha": "wrong"},
        "error",
    ],
)
def test_immutable_cache_does_not_retain_errors_or_invalid_metadata(bad):
    from types import SimpleNamespace

    calls = []

    def read(*args, **kwargs):
        calls.append(args)
        if bad == "error":
            raise evidence.EvidenceError("temporary API failure")
        return bad

    client = rollout.ImmutableContentsClient(SimpleNamespace(repo=REPO, json=read))
    for _ in range(2):
        if bad == "error":
            with pytest.raises(evidence.EvidenceError):
                client.json(f"repos/{REPO}/contents/file", f"ref={MAIN}")
        else:
            assert client.json(f"repos/{REPO}/contents/file", f"ref={MAIN}") == bad
    assert len(calls) == 2


def test_immutable_cache_is_process_local_deep_copied_and_commit_bound():
    from types import SimpleNamespace

    calls = []
    original = {"type": "file", "sha": MAIN, "nested": {"value": "original"}}

    def read(*args, **kwargs):
        calls.append(args)
        return original

    raw = SimpleNamespace(repo=REPO, json=read, jobs=lambda run: [run])
    client = rollout.ImmutableContentsClient(raw)
    path = f"repos/{REPO}/contents/file"
    client.json(path, f"ref={MAIN}")["nested"]["value"] = "modified"
    assert client.json(path, f"ref={MAIN}")["nested"]["value"] == "original"
    client.json(path, f"ref={MAIN}")["nested"]["value"] = "again"
    assert client.json(path, f"ref={MAIN}")["nested"]["value"] == "original"
    client.json(path, f"ref={CANDIDATE}")
    rollout.ImmutableContentsClient(raw).json(path, f"ref={MAIN}")
    assert len(calls) == 3
    assert client.jobs(20) == [20]
