# SPDX-License-Identifier: Apache-2.0
"""Exercise the real handoff with inert GitHub responses; never mutate GitHub."""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

from scripts.pr_validate import queue_ready as q
from scripts.pr_validate.base import StepResult
from scripts.pr_validate.context import Context

HEAD = "a" * 40
EXPECTED = ["fetch", "codex_review", "full_unit"]


@pytest.fixture
def harness(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").touch()
    ctx = Context(
        pr_number=42,
        head_sha=HEAD,
        base_strategy="git-merge-base",
        pr_title="fix: behavior",
    )
    ctx.results = [
        StepResult(name=name, status="pass", summary="ok") for name in EXPECTED
    ]
    pull = {
        "state": "open",
        "draft": False,
        "title": "fix: behavior",
        "head": {"sha": HEAD, "ref": "fix/test", "repo": {"full_name": ctx.repo}},
        "base": {"ref": "main"},
        "labels": [],
    }
    checks = [
        {
            "id": index,
            "name": name,
            "app": {"slug": "github-actions"},
            "status": "completed",
            "conclusion": "success",
        }
        for index, name in enumerate(
            sorted(q.REQUIRED | {"changed-lines-coverage", "merge-lane-mac"}), 1
        )
    ]
    state = SimpleNamespace(
        ctx=ctx,
        pull=pull,
        checks=checks,
        comments=[],
        mutations=[],
        statuses=[],
        fail_comment=False,
        on_authorize=None,
        wait=False,
        permission="write",
        calls=[],
    )

    def gh(*args):
        state.calls.append(args)
        endpoint = args[1]
        if endpoint == "user":
            return {"login": "maintainer"}
        if endpoint.endswith("/permission"):
            return {"permission": state.permission}
        if endpoint.endswith("/pulls/42"):
            return copy.deepcopy(state.pull)
        if "/check-runs?" in endpoint:
            return [{"check_runs": copy.deepcopy(state.checks)}]
        if "/statuses?" in endpoint:
            if state.on_authorize:
                state.on_authorize()
            return [copy.deepcopy(state.statuses)]
        if "/comments?" in endpoint:
            # Page one intentionally empty: dedup must see later pages.
            return [[], copy.deepcopy(state.comments)]
        if endpoint.endswith("/labels"):
            state.mutations.append(args)
            label = args[-1].split("=", 1)[1]
            state.pull["labels"] = [{"name": label}]
            state.statuses = [
                {"id": 1, "context": "merge-ready-head", "state": "success"}
            ]
            return []
        if endpoint.endswith("/comments"):
            state.mutations.append(args)
            if state.fail_comment:
                raise RuntimeError("ambiguous transport")
            state.comments.append({"body": args[-1].split("=", 1)[1]})
            return {"id": 100, "html_url": "https://example.invalid/comment/100"}
        raise AssertionError(args)

    monkeypatch.setattr(q, "gh", gh)
    monkeypatch.setattr(
        q.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=str(tmp_path))
    )
    monkeypatch.setattr(q.time, "sleep", lambda _: None)
    return state


def run(h):
    q.queue_validated_head(h.ctx, EXPECTED)


def test_success_and_second_invocation_never_repeat(harness):
    run(harness)
    run(harness)
    assert len(harness.mutations) == 2
    assert harness.comments == [{"body": "@mergifyio queue mac-batch"}]
    receipt = next((harness.ctx.repo_root / "rapid-queue-receipts").glob("*.json"))
    assert json.loads(receipt.read_text())["state"] == "issued"


def test_no_mac_route_and_deferred_source_coverage(harness):
    for check in harness.checks:
        if check["name"] == "merge-lane-mac":
            check["name"] = "merge-lane-no-mac"
        if check["name"] == "changed-lines-coverage":
            check["conclusion"] = "skipped"
    run(harness)
    assert harness.comments == [{"body": "@mergifyio queue no-mac-batch"}]


@pytest.mark.parametrize(
    "change",
    [
        lambda h: h.ctx.results.pop(),
        lambda h: setattr(h.ctx.results[1], "status", "skip"),
        lambda h: setattr(h.ctx.results[-1], "status", "fail"),
        lambda h: h.ctx.results.append(h.ctx.results[0]),
        lambda h: setattr(h.ctx, "head_sha", "short"),
        lambda h: setattr(h.ctx, "base_strategy", "tip-fallback"),
        lambda h: h.pull.update(state="closed"),
        lambda h: h.pull.update(draft=True),
        lambda h: h.pull.update(body="unchecked test plan"),
        lambda h: h.pull.update(title="fix: different rationale"),
        lambda h: h.pull["head"].update(sha="b" * 40),
        lambda h: h.pull["head"]["repo"].update(full_name="fork/repo"),
        lambda h: h.pull["head"].update(ref="mergify/merge-queue/test"),
        lambda h: h.pull["base"].update(ref="release/0.16.0"),
        lambda h: h.pull.update(labels=[{"name": "dequeued"}]),
        lambda h: h.pull.update(labels=[{"name": "version-bump"}]),
        lambda h: h.pull.update(labels=[{"name": "skip-version-bump"}]),
        lambda h: h.pull.update(title="chore: bump version to 0.17.0"),
        lambda h: setattr(h, "permission", "read"),
    ],
)
def test_rejections_never_mutate(harness, change):
    change(harness)
    with pytest.raises(ValueError):
        run(harness)
    assert harness.mutations == []


@pytest.mark.parametrize("name", sorted(q.REQUIRED | {"changed-lines-coverage"}))
@pytest.mark.parametrize("conclusion", ["failure", "cancelled", "neutral"])
def test_hosted_failure_never_authorizes(harness, name, conclusion):
    next(c for c in harness.checks if c["name"] == name)["conclusion"] = conclusion
    with pytest.raises(ValueError):
        run(harness)
    assert harness.mutations == []


def test_latest_wrong_app_cannot_override_failed_actions(harness):
    check = next(c for c in harness.checks if c["name"] == "tests")
    check["conclusion"] = "failure"
    harness.checks.append(
        {**check, "id": 100, "conclusion": "success", "app": {"slug": "other"}}
    )
    with pytest.raises(ValueError):
        run(harness)


def test_latest_pending_does_not_reuse_old_success(harness):
    check = next(c for c in harness.checks if c["name"] == "tests")
    harness.checks.append(
        {**check, "id": 100, "status": "in_progress", "conclusion": None}
    )
    assert q.hosted_lane(harness.ctx) is None


def test_two_lanes_reject(harness):
    harness.checks.append(
        {**harness.checks[-1], "id": 100, "name": "merge-lane-no-mac"}
    )
    with pytest.raises(ValueError, match="ambiguous"):
        run(harness)


def test_paginated_prior_owner_command_stops(harness):
    harness.comments = [{"body": "@mergifyio queue mac-batch"}]
    run(harness)
    assert harness.mutations == []


@pytest.mark.parametrize("state", ["queued", "dequeued", "merged"])
def test_provider_history_prevents_recovery_without_an_owner_command(harness, state):
    payload = json.dumps({"state": state, "queued_at": "2026-10-10T10:00:00Z"})
    harness.comments = [
        {
            "user": {"login": "mergify[bot]"},
            "body": f"-*- Mergify Payload -*-\n{payload}\n-*- Mergify Payload End -*-",
        }
    ]
    run(harness)
    assert harness.mutations == []


def test_provider_unadmitted_notice_does_not_block_first_request(harness):
    harness.comments = [
        {"user": {"login": "mergify[bot]"}, "body": "No queue request yet"},
        {
            "user": {"login": "mergify[bot]"},
            "body": '-*- Mergify Payload -*-\n{"queued_at":null}\n-*- Mergify Payload End -*-',
        },
    ]
    run(harness)
    assert harness.comments[-1]["body"] == "@mergifyio queue mac-batch"


def test_provider_auto_admission_never_gets_a_second_command(harness):
    harness.pull["labels"] = [{"name": "queued"}]
    run(harness)
    assert harness.mutations == []


def test_provider_admits_while_authorizing(harness):
    def admit():
        harness.pull["labels"].append({"name": "queued"})

    harness.on_authorize = admit
    run(harness)
    assert not harness.comments


def test_conflicting_ready_labels_never_reset(harness):
    harness.pull["labels"] = [{"name": "merge-ready"}]
    with pytest.raises(ValueError, match="conflicting"):
        run(harness)
    assert harness.mutations == []


def test_existing_ready_authorization_does_not_relabel(harness):
    harness.pull["labels"] = [{"name": "merge-ready-mac"}]
    harness.statuses = [{"id": 1, "context": "merge-ready-head", "state": "success"}]
    run(harness)
    assert len(harness.mutations) == 1


def test_ambiguous_transport_receipt_blocks_retry(harness):
    harness.fail_comment = True
    with pytest.raises(RuntimeError, match="ambiguous"):
        run(harness)
    with pytest.raises(FileExistsError):
        run(harness)
    assert len(harness.mutations) == 2
    receipt = next((harness.ctx.repo_root / "rapid-queue-receipts").glob("*.json"))
    assert json.loads(receipt.read_text())["state"] == "issuing"


@pytest.mark.parametrize("mutation", ["head", "label", "gate", "binder"])
def test_change_during_authorization_never_queues(harness, mutation):
    def change():
        if mutation == "head":
            harness.pull["head"]["sha"] = "b" * 40
        elif mutation == "label":
            harness.pull["labels"] = []
        elif mutation == "gate":
            next(c for c in harness.checks if c["name"] == "tests")["conclusion"] = (
                "failure"
            )
        else:
            harness.statuses.append(
                {"id": 2, "context": "merge-ready-head", "state": "failure"}
            )

    harness.on_authorize = change
    with pytest.raises(ValueError):
        run(harness)
    assert not harness.comments


def test_wait_expired_never_mutates(harness, monkeypatch):
    monkeypatch.setattr(q.time, "monotonic", lambda: 1)
    with pytest.raises(RuntimeError, match="expired"):
        q.queue_validated_head(harness.ctx, EXPECTED, wait_seconds=0)
    assert harness.mutations == []


def test_missing_or_incomplete_lane_cannot_admit(harness):
    lane = next(c for c in harness.checks if c["name"] == "merge-lane-mac")
    lane["conclusion"] = None
    assert q.hosted_lane(harness.ctx) is None
    lane.update(conclusion="success", status="in_progress")
    assert q.hosted_lane(harness.ctx) is None


def test_waits_for_both_hosted_checks_and_authorization(harness, monkeypatch):
    check = next(c for c in harness.checks if c["name"] == "tests")
    check.update(status="in_progress", conclusion=None)
    original = q.gh
    status_reads = []
    sleeps = []

    def gh(*args):
        if "/statuses?" in args[1]:
            status_reads.append(1)
            if len(status_reads) == 1:
                return [[]]
        return original(*args)

    def sleep(seconds):
        sleeps.append(seconds)
        check.update(status="completed", conclusion="success")

    monkeypatch.setattr(q, "gh", gh)
    monkeypatch.setattr(q.time, "sleep", sleep)
    run(harness)
    assert sleeps == [15, 15]
    assert len(harness.comments) == 1


def test_concurrent_request_while_waiting_is_not_duplicated(harness, monkeypatch):
    original = q.existing_request
    calls = []

    def request(ctx):
        calls.append(1)
        return len(calls) > 1 or original(ctx)

    monkeypatch.setattr(q, "existing_request", request)
    run(harness)
    assert harness.mutations == []


def test_readiness_removed_before_wait_stops(harness, monkeypatch):
    original = q.check_pull

    def pull(ctx):
        if harness.mutations:
            harness.pull["labels"] = []
        return original(ctx)

    monkeypatch.setattr(q, "check_pull", pull)
    with pytest.raises(ValueError, match="label changed"):
        run(harness)
    assert not harness.comments


def test_authorization_deadline_never_queues(harness, monkeypatch):
    ticks = [0]
    monkeypatch.setattr(q.time, "monotonic", lambda: ticks[0])
    monkeypatch.setattr(q.time, "sleep", lambda _: ticks.__setitem__(0, ticks[0] + 15))
    monkeypatch.setattr(q, "authorized", lambda _: False)
    with pytest.raises(RuntimeError, match="readiness wait expired"):
        q.queue_validated_head(harness.ctx, EXPECTED, wait_seconds=15)
    assert not harness.comments


def test_gate_restarts_while_authorizing_stops(harness):
    def restart():
        next(c for c in harness.checks if c["name"] == "tests").update(
            status="in_progress", conclusion=None
        )

    harness.on_authorize = restart
    with pytest.raises(ValueError, match="gates changed"):
        run(harness)
    assert not harness.comments


def test_gh_never_retries_mutation(monkeypatch):
    calls = []

    def execute(*args, **kwargs):
        calls.append(args)
        return SimpleNamespace(stdout='{"id":1}')

    monkeypatch.setattr(q.subprocess, "run", execute)
    assert q.gh("api", "endpoint") == {"id": 1}
    assert len(calls) == 1
