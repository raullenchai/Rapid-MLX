"""A producer notification cannot bypass current authenticated full evidence."""

import copy
import json
import runpy
import sys
from pathlib import Path

import pytest
import yaml

from scripts import ci_candidate_admission as admission
from tests.test_ci_candidate_consumer import setup
from tests.test_queue_tree_evidence import CANDIDATE, REPO


def test_real_archive_and_full_validation_admission(monkeypatch):
    client, _, _, _, _ = setup(monkeypatch)
    result = admission.verify_admission(client, 100)
    assert result["verified"] and result["kind"] == "full"
    assert result["candidate_sha"] == CANDIDATE
    assert result["source_run_id"] == 20 and result["producer_attempt"] == 1
    assert not result["authorizes_merge"] and not result["authorizes_reduced_ci"]


@pytest.mark.parametrize(
    "change",
    [
        "invalid-run",
        "failed-producer",
        "truncated",
        "duplicate",
        "missing",
        "expired",
        "wrong-run",
        "superseded",
        "mapped",
        "cancelled-source",
        "closed-candidate",
        "main-moved",
    ],
)
def test_bad_notification_or_evidence_never_publishes_success(monkeypatch, change):
    client, record, status, run, artifact = setup(monkeypatch)
    prefix = f"repos/{REPO}"
    page = client.responses[f"{prefix}/actions/runs/100/artifacts"]
    run_id = 100
    if change == "invalid-run":
        run_id = True
    elif change == "failed-producer":
        run["conclusion"] = "failure"
    elif change == "truncated":
        page["total_count"] = 100
    elif change == "duplicate":
        page["artifacts"].append(dict(artifact, id=102))
        page["total_count"] = 2
    elif change == "missing":
        artifact["name"] = "unrelated"
    elif change == "expired":
        artifact["expired"] = True
    elif change == "wrong-run":
        artifact["workflow_run"]["id"] = 101
    elif change == "superseded":
        status["target_url"] = f"https://github.com/{REPO}/actions/runs/101"
    elif change == "mapped":
        record["kind"] = "mapped"
    elif change == "cancelled-source":
        client.responses[f"{prefix}/actions/runs/20"]["conclusion"] = "cancelled"
    elif change == "closed-candidate":
        client.responses[f"{prefix}/pulls"] = []
    else:
        client.responses[f"{prefix}/git/ref/heads/main"]["object"]["sha"] = "e" * 40
    result = admission.verify_admission(client, run_id)
    assert not result["verified"] and "candidate_sha" not in result
    assert not result["authorizes_merge"] and not result["authorizes_reduced_ci"]


@pytest.mark.parametrize("change", ["producer", "index", "candidate"])
def test_last_rereads_reject_mutations(monkeypatch, change):
    client, _, _, _, _ = setup(monkeypatch)
    if change == "candidate":
        original = admission.consumer.consume_full
        calls = 0

        def consume(*args):
            nonlocal calls
            calls += 1
            return original(*args) if calls == 1 else {"verified": False}

        monkeypatch.setattr(admission.consumer, "consume_full", consume)
    else:
        name = "_producer_run" if change == "producer" else "_status"
        original = getattr(admission.consumer, name)
        calls = 0

        # Initial selector, consume_full initial/final, then final selector.
        def read(*args):
            nonlocal calls
            calls += 1
            result = copy.deepcopy(original(*args))
            if calls == 4:
                result["mutated"] = True
            return result

        monkeypatch.setattr(admission.consumer, name, read)
    assert not admission.verify_admission(client, 100)["verified"]


@pytest.mark.parametrize("good", [True, False])
def test_cli_only_exposes_validated_sha(monkeypatch, tmp_path, good):
    client, _, _, run, _ = setup(monkeypatch)
    if not good:
        run["conclusion"] = "failure"
    monkeypatch.setattr(admission.evidence, "GitHubClient", lambda repo: client)
    out, record = tmp_path / "out", tmp_path / "record"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "admission",
            "--repo",
            REPO,
            "--producer-run-id",
            "100",
            "--github-output",
            str(out),
            "--output",
            str(record),
        ],
    )
    admission.main()
    values = dict(line.split("=", 1) for line in out.read_text().splitlines())
    assert values["verified"] == str(good).lower()
    assert ("candidate_sha" in values) is good
    assert json.loads(record.read_text())["verified"] is good


def test_module_entrypoint_rejects_invalid_run_without_api(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "admission",
            "--repo",
            REPO,
            "--producer-run-id",
            "0",
            "--github-output",
            str(tmp_path / "out"),
            "--output",
            str(tmp_path / "record"),
        ],
    )
    with pytest.warns(RuntimeWarning, match="found in sys.modules"):
        runpy.run_module("scripts.ci_candidate_admission", run_name="__main__")
    assert (tmp_path / "out").read_text() == "verified=false\n"


def test_workflow_uses_trusted_checkout_and_indexes_after_upload():
    root = Path(__file__).resolve().parents[1]
    workflow = yaml.safe_load(
        (root / ".github/workflows/candidate-admission.yml").read_text()
    )
    assert workflow["permissions"] == {
        "actions": "read",
        "contents": "read",
        "statuses": "write",
    }
    job = workflow["jobs"]["admit"]
    assert "github.event_name == 'workflow_run'" in job["if"]
    assert "github.event.workflow_run.conclusion == 'success'" in job["if"]
    for guard in (
        "event == 'pull_request'",
        "head_repository.full_name == github.repository",
        "'mergify/merge-queue/'",
    ):
        assert guard in job["if"]
    assert workflow["on" if "on" in workflow else True]["workflow_run"][
        "workflows"
    ] == ["CI", "Candidate qualification"]
    steps = job["steps"]
    assert steps[0]["with"] == {
        "ref": "${{ github.sha }}",
        "persist-credentials": False,
    }
    assert all(
        len(step["uses"].split("@")[1]) == 40 for step in steps if "uses" in step
    )
    upload, index = steps[-2:]
    assert upload["if"] == "steps.result.outputs.verified == 'true'"
    assert index["if"] == "always() && steps.result.outcome == 'success'"
    assert "--expected" in index["run"]
    assert "--publish-target-url" in index["run"]
    validation = next(s for s in steps if s.get("id") == "result")
    assert '--producer-run-id "$TRIGGER_RUN"' in validation["run"]
    assert '--source-run-id "$TRIGGER_RUN" --source-attempt "$TRIGGER_ATTEMPT"' in validation["run"]
    assert (root / ".mergify.yml").read_text().count(
        "check-success = @github-actions/candidate-admission/ci"
    ) == 2


@pytest.mark.parametrize("change", ["new-producer", "new-attempt"])
def test_second_consumer_cannot_switch_notification_provenance(monkeypatch, change):
    client, _, status, run, artifact = setup(monkeypatch)
    prefix = f"repos/{REPO}"
    original_json = client.json
    monkeypatch.setattr(
        client, "json", lambda *a, **kw: copy.deepcopy(original_json(*a, **kw))
    )
    original_consume = admission.consumer.consume_full
    original_download = admission.consumer.subprocess.run
    calls = 0
    verified_consumptions = []

    def download(command, **kwargs):
        if command[-1].endswith("/artifacts/201/zip"):
            command = [*command[:-1], f"{prefix}/actions/artifacts/101/zip"]
        return original_download(command, **kwargs)

    monkeypatch.setattr(admission.consumer.subprocess, "run", download)

    def consume(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            if change == "new-producer":
                client.responses[f"{prefix}/actions/runs/200"] = dict(run, id=200)
                client.job_records[200] = copy.deepcopy(client.job_records[100])
                client.responses[f"{prefix}/actions/runs/200/artifacts"] = {
                    "total_count": 1,
                    "artifacts": [dict(artifact, id=201, workflow_run={"id": 200})],
                }
                client.responses[f"{prefix}/commits/{CANDIDATE}/statuses"].append(
                    dict(
                        status,
                        id=502,
                        target_url=f"https://github.com/{REPO}/actions/runs/200",
                    )
                )
            else:
                run["run_attempt"] = 2
                for job in client.job_records[100]:
                    job["run_attempt"] = 2
        result = original_consume(*args)
        verified_consumptions.append(result)
        return result

    monkeypatch.setattr(admission.consumer, "consume_full", consume)
    result = admission.verify_admission(client, 100)
    # Both real ZIP/live full verifications succeed with identical source proof.
    assert calls == 2 and verified_consumptions[0] == verified_consumptions[1]
    assert verified_consumptions[1]["verified"]
    assert not result["verified"] and "candidate_sha" not in result
    assert "producer/index" in result["reason"]


@pytest.mark.parametrize("pending", ["index", "producer"])
def test_parallel_start_waits_then_consumes_real_completed_archive(
    monkeypatch, pending
):
    client, _, _, run, _ = setup(monkeypatch)
    original = admission.consumer._status
    calls = []
    if pending == "producer":
        run["status"] = "in_progress"

    def status(*args):
        calls.append(1)
        if pending == "index" and len(calls) == 1:
            raise admission.evidence.EvidenceError("qualification status is absent")
        return original(*args)

    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        run["status"] = "completed"

    monkeypatch.setattr(admission.consumer, "_status", status)
    monkeypatch.setattr(admission.time, "sleep", sleep)
    result = admission.verify_source_admission(client, 20, 1)
    assert sleeps == [1]
    assert result["verified"] and result["source_run_id"] == 20
    assert result["source_attempt"] == 1 and result["producer_run_id"] == 100
    assert not result["authorizes_merge"] and not result["authorizes_reduced_ci"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("id", 21),
        ("id", True),
        ("run_attempt", 2),
        ("run_attempt", True),
        ("path", "other.yml"),
        ("event", "push"),
        ("head_branch", "main"),
        ("repository", {"full_name": "other/repo"}),
        ("head_repository", {"full_name": "other/repo"}),
        ("status", "in_progress"),
        ("conclusion", "failure"),
        ("head_sha", "bad"),
    ],
)
def test_parallel_source_identity_rejects_wrong_or_stale_trigger(
    monkeypatch, field, value
):
    client, _, _, _, _ = setup(monkeypatch)
    client.responses[f"repos/{REPO}/actions/runs/20"][field] = value
    assert not admission.verify_source_admission(client, 20, 1)["verified"]


@pytest.mark.parametrize("run,attempt", [(0, 1), (True, 1), (20, None), (20, False)])
def test_parallel_source_invalid_inputs_never_call_api(run, attempt):
    from types import SimpleNamespace

    client = SimpleNamespace(json=lambda *a: pytest.fail("invalid input queried API"))
    assert not admission.verify_source_admission(client, run, attempt)["verified"]


@pytest.mark.parametrize(
    "case", ["absent", "revoked", "foreign", "closed", "main", "api"]
)
def test_parallel_wait_is_bounded_and_never_bypasses_full_guard(monkeypatch, case):
    client, _, status, _, _ = setup(monkeypatch)
    clock = iter([0, 46])
    monkeypatch.setattr(admission.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(
        admission.time, "sleep", lambda *a: pytest.fail("wait exceeded deadline")
    )
    if case == "absent":
        monkeypatch.setattr(
            admission.consumer,
            "_status",
            lambda *a: (_ for _ in ()).throw(
                admission.evidence.EvidenceError("qualification status is absent")
            ),
        )
    elif case == "revoked":
        status["state"] = "failure"
    elif case == "foreign":
        status["target_url"] = "https://github.com/other/repo/actions/runs/100"
    elif case == "closed":
        client.responses[f"repos/{REPO}/pulls"] = []
    elif case == "main":
        client.responses[f"repos/{REPO}/git/ref/heads/main"]["object"]["sha"] = "e" * 40
    else:
        monkeypatch.setattr(
            admission.consumer,
            "_status",
            lambda *a: (_ for _ in ()).throw(
                admission.evidence.EvidenceError("API unavailable")
            ),
        )
    result = admission.verify_source_admission(client, 20, 1)
    assert not result["verified"] and "candidate_sha" not in result and result["reason"]


def test_parallel_trigger_cannot_accept_another_successful_source(monkeypatch):
    client, _, _, _, _ = setup(monkeypatch)
    client.responses[f"repos/{REPO}/actions/runs/21"] = dict(
        client.responses[f"repos/{REPO}/actions/runs/20"], id=21
    )
    result = admission.verify_source_admission(client, 21, 1)
    assert (
        not result["verified"] and result["reason"] == "admission changed triggering CI"
    )


def test_parallel_trigger_final_read_rejects_rerun_boundary(monkeypatch):
    client, _, _, _, _ = setup(monkeypatch)
    original = admission.verify_admission

    def verify(*args):
        result = original(*args)
        assert result["verified"]
        client.responses[f"repos/{REPO}/actions/runs/20"]["run_attempt"] = 2
        return result

    monkeypatch.setattr(admission, "verify_admission", verify)
    assert not admission.verify_source_admission(client, 20, 1)["verified"]


def test_parallel_cli_exposes_only_real_producer_identity(monkeypatch, tmp_path):
    client, _, _, _, _ = setup(monkeypatch)
    monkeypatch.setattr(admission.evidence, "GitHubClient", lambda *a: client)
    out, record = tmp_path / "out", tmp_path / "record"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "admission",
            "--repo",
            REPO,
            "--source-run-id",
            "20",
            "--source-attempt",
            "1",
            "--github-output",
            str(out),
            "--output",
            str(record),
        ],
    )
    admission.main()
    values = dict(line.split("=", 1) for line in out.read_text().splitlines())
    assert values == {
        "verified": "true",
        "candidate_sha": CANDIDATE,
        "producer_run_id": "100",
    }
    assert json.loads(record.read_text())["source_run_id"] == 20


@pytest.mark.parametrize(
    "trigger,run", [("CI", "20"), ("Candidate qualification", "100")]
)
def test_actual_rendered_observer_shell_uses_real_verified_cli(
    monkeypatch, tmp_path, trigger, run
):
    import os
    import subprocess
    from contextlib import nullcontext

    from scripts import ci_github_transport

    shell_run = subprocess.run
    client, _, _, _, _ = setup(monkeypatch)
    monkeypatch.setattr(
        ci_github_transport, "PersistentGitHubClient", lambda *a: nullcontext(client)
    )
    workflow = yaml.safe_load(
        (
            Path(__file__).resolve().parents[1]
            / ".github/workflows/candidate-admission.yml"
        ).read_text()
    )
    shell = workflow["jobs"]["admit"]["steps"][1]["run"]
    captured, out = tmp_path / "args", tmp_path / "out"
    env = dict(
        os.environ,
        TRIGGER_NAME=trigger,
        TRIGGER_RUN=run,
        TRIGGER_ATTEMPT="1",
        GITHUB_REPOSITORY=REPO,
        GITHUB_OUTPUT=str(out),
        RUNNER_TEMP=str(tmp_path),
        CAPTURE=str(captured),
    )
    shell_run(
        ["bash", "-c", 'python() { printf "%s\\0" "$@" > "$CAPTURE"; }\n' + shell],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    args = captured.read_bytes().decode().rstrip("\0").split("\0")
    assert args[:2] == ["-m", "scripts.ci_candidate_admission"]
    assert "--reuse-api-connection" in args
    monkeypatch.setattr(admission.evidence, "GitHubClient", lambda *a: client)
    monkeypatch.setattr(sys, "argv", ["admission", *args[2:]])
    admission.main()
    result = json.loads((tmp_path / "candidate-admission.json").read_text())
    assert result["verified"] and result["source_run_id"] == 20
    assert result["producer_run_id"] == 100 and result["source_attempt"] == 1


@pytest.mark.parametrize("change", ["index", "attempt"])
def test_parallel_final_source_read_cannot_switch_producer_provenance(
    monkeypatch, change
):
    client, _, status, run, _ = setup(monkeypatch)
    original = admission.verify_admission

    def verify(*args):
        result = original(*args)
        assert result["verified"]
        if change == "index":
            status["id"] += 1
        else:
            run["run_attempt"] = 2
            for job in client.job_records[100]:
                job["run_attempt"] = 2
        return result

    monkeypatch.setattr(admission, "verify_admission", verify)
    result = admission.verify_source_admission(client, 20, 1)
    assert not result["verified"]
    assert result["reason"] == "producer/index changed after source verification"


@pytest.mark.parametrize("boundary", ["before", "during-post", "superseded"])
def test_fresh_publisher_revokes_stale_proof_and_never_overwrites_newer_notice(
    monkeypatch, boundary
):
    client, _, status, run, _ = setup(monkeypatch)
    expected = admission.verify_admission(client, 100)
    writes = []

    def write(client, sha, state, target):
        writes.append((sha, state))
        if boundary == "during-post":
            run["run_attempt"] = 2

    monkeypatch.setattr(admission, "_write_status", write)
    if boundary == "before":
        run["run_attempt"] = 2
    elif boundary == "superseded":
        status["target_url"] = f"https://github.com/{REPO}/actions/runs/200"
    result = admission.publish_admission(
        client, 100, expected, f"https://github.com/{REPO}/actions/runs/400"
    )
    if boundary == "superseded":
        assert not result["published"] and not writes
    else:
        assert result["published"] and not result["verified"]
        assert writes == (
            [(CANDIDATE, "failure")]
            if boundary == "before"
            else [(CANDIDATE, "success"), (CANDIDATE, "failure")]
        )


def test_live_rollback_revokes_queue_gate_using_existing_trusted_status_identity(
    monkeypatch,
):
    from scripts import ci_candidate_rollout as rollout

    client, _, _, _, _ = setup(monkeypatch)
    client.gh = "gh"
    original = client.json
    pulls = client.responses[f"repos/{REPO}/pulls"]
    monkeypatch.setattr(
        client,
        "json",
        lambda endpoint, *a, **kw: (
            [pulls]
            if endpoint.endswith("/pulls") and kw.get("paginate")
            else original(endpoint, *a, **kw)
        ),
    )
    monkeypatch.setattr(rollout, "enabled", lambda *a: {})
    # Actual full evidence is preserved on the first sweep.
    writes = []
    monkeypatch.setattr(
        admission,
        "_write_status",
        lambda client, sha, state, target: writes.append((sha, state)),
    )
    target = f"https://github.com/{REPO}/actions/runs/400"
    result = admission.rollback(client, target)
    assert result["disabled"] and result["candidates"] == [
        {"candidate_sha": CANDIDATE, "preserved": True}
    ]
    assert writes == []
    client.responses[f"repos/{REPO}/actions/runs/20"]["conclusion"] = "cancelled"
    result = admission.rollback(client, target)
    assert not result["candidates"][0]["preserved"]
    assert writes == [(CANDIDATE, "failure")]
    monkeypatch.setattr(rollout, "enabled", lambda *a: {"run_id": 401})
    with pytest.raises(admission.evidence.EvidenceError, match="superseded"):
        admission.rollback(client, target)


def test_status_writer_uses_structured_commit_bound_github_post(monkeypatch):
    from types import SimpleNamespace

    client = admission.evidence.GitHubClient(REPO)
    commands = []
    monkeypatch.setattr(
        admission.subprocess,
        "run",
        lambda command, **kw: (
            commands.append((command, kw)) or SimpleNamespace(returncode=0)
        ),
    )
    admission._write_status(
        client, CANDIDATE, "failure", f"https://github.com/{REPO}/actions/runs/400"
    )
    command, options = commands[0]
    assert command[:5] == [
        "gh",
        "api",
        "--method",
        "POST",
        f"repos/{REPO}/statuses/{CANDIDATE}",
    ]
    assert "context=candidate-admission/ci" in command and "state=failure" in command
    assert options["check"] is True
    with pytest.raises(admission.evidence.EvidenceError):
        admission._write_status(
            client,
            CANDIDATE,
            "success",
            "https://github.com/fork/repo/actions/runs/400",
        )
    assert len(commands) == 1


def test_failed_producer_retry_revokes_previously_green_admission(monkeypatch):
    client, _, _, run, _ = setup(monkeypatch)
    run["conclusion"] = "cancelled"
    writes = []
    monkeypatch.setattr(
        admission,
        "_write_status",
        lambda client, sha, state, target: writes.append((sha, state)),
    )
    rejected = admission.verify_admission(client, 100)
    assert not rejected["verified"] and "candidate_sha" not in rejected
    result = admission.publish_admission(
        client, 100, rejected, f"https://github.com/{REPO}/actions/runs/400"
    )
    assert result["published"] and not result["verified"]
    assert writes == [(CANDIDATE, "failure")]


def test_notification_without_artifact_never_invents_status_target(monkeypatch):
    client, *_ = setup(monkeypatch)
    client.responses[f"repos/{REPO}/actions/runs/100/artifacts"] = {
        "total_count": 0,
        "artifacts": [],
    }
    monkeypatch.setattr(
        admission, "_write_status", lambda *a: pytest.fail("invented SHA")
    )
    assert not admission.publish_admission(
        client, 100, {"verified": False}, f"https://github.com/{REPO}/actions/runs/400"
    )["published"]


@pytest.mark.parametrize(
    "bad", ["boolean", "foreign-workflow", "truncated", "duplicate"]
)
def test_revocation_selector_authenticates_notification_without_trusting_artifact_contents(
    monkeypatch, bad
):
    client, _, _, run, artifact = setup(monkeypatch)
    page = client.responses[f"repos/{REPO}/actions/runs/100/artifacts"]
    if bad == "boolean":
        run_id = True
    else:
        run_id = 100
        if bad == "foreign-workflow":
            run["workflow_id"] = 999
        elif bad == "truncated":
            page["total_count"] = 100
        else:
            page["artifacts"].append(dict(artifact, id=102))
            page["total_count"] = 2
    with pytest.raises(admission.evidence.EvidenceError):
        admission._notification_sha(client, run_id)


def test_publisher_last_index_reread_prevents_old_status_mutation(monkeypatch):
    client, _, status, *_ = setup(monkeypatch)
    expected = admission.verify_admission(client, 100)
    monkeypatch.setattr(admission, "verify_admission", lambda *a: expected)
    original = admission.consumer._status
    calls = 0

    def read(*args):
        nonlocal calls
        calls += 1
        result = copy.deepcopy(original(*args))
        return result if calls == 1 else dict(result, id=502)

    monkeypatch.setattr(admission.consumer, "_status", read)
    monkeypatch.setattr(
        admission, "_write_status", lambda *a: pytest.fail("stale index published")
    )
    assert not admission.publish_admission(
        client, 100, expected, f"https://github.com/{REPO}/actions/runs/400"
    )["published"]


@pytest.mark.parametrize("mode", ["publish", "rollback"])
def test_real_cli_publication_and_rollback_paths(monkeypatch, tmp_path, mode):
    from scripts import ci_candidate_rollout as rollout

    client, *_ = setup(monkeypatch)
    original = client.json
    pulls = client.responses[f"repos/{REPO}/pulls"]
    monkeypatch.setattr(
        client,
        "json",
        lambda endpoint, *a, **kw: (
            [pulls]
            if endpoint.endswith("/pulls") and kw.get("paginate")
            else original(endpoint, *a, **kw)
        ),
    )
    monkeypatch.setattr(rollout, "enabled", lambda *a: {})
    monkeypatch.setattr(admission.evidence, "GitHubClient", lambda repo: client)
    writes = []
    monkeypatch.setattr(admission, "_write_status", lambda *a: writes.append(a))
    args = [
        "admission",
        "--repo",
        REPO,
        "--producer-run-id",
        "100",
        "--github-output",
        str(tmp_path / "out"),
        "--output",
        str(tmp_path / "result"),
        "--publish-target-url",
        f"https://github.com/{REPO}/actions/runs/400",
    ]
    if mode == "publish":
        expected = tmp_path / "expected"
        expected.write_text(json.dumps(admission.verify_admission(client, 100)))
        args += ["--expected", str(expected)]
    else:
        args += ["--rollback"]
    monkeypatch.setattr(sys, "argv", args)
    admission.main()
    result = json.loads((tmp_path / "result").read_text())
    assert (
        result.get("published") is True
        if mode == "publish"
        else result["disabled"] is True
    )


def test_rollback_does_not_touch_unrelated_prs_or_claim_success_on_malformed_listing(
    monkeypatch,
):
    from scripts import ci_candidate_rollout as rollout

    client, *_ = setup(monkeypatch)
    original = client.json
    pull = copy.deepcopy(client.responses[f"repos/{REPO}/pulls"][0])
    pull["user"]["login"] = "user"
    listing = [[pull]]
    monkeypatch.setattr(
        client,
        "json",
        lambda endpoint, *a, **kw: (
            listing
            if endpoint.endswith("/pulls") and kw.get("paginate")
            else original(endpoint, *a, **kw)
        ),
    )
    monkeypatch.setattr(rollout, "enabled", lambda *a: {})
    monkeypatch.setattr(
        admission, "_write_status", lambda *a: pytest.fail("unrelated PR mutated")
    )
    target = f"https://github.com/{REPO}/actions/runs/400"
    assert admission.rollback(client, target)["candidates"] == []
    listing[:] = [{"not": "a page"}]
    with pytest.raises(admission.evidence.EvidenceError):
        admission.rollback(client, target)
