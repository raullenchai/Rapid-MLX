# SPDX-License-Identifier: Apache-2.0
"""Main failures are rerun once and reverted only with unambiguous evidence."""

import json
import runpy
import subprocess
from unittest.mock import Mock, call

import pytest

from scripts import main_guard as g
from scripts.merge_on_green import APIError, GitHub
from tests.test_merge_on_green import Fake

SHA = "a" * 40
PARENT = "b" * 40
BRANCH = f"revert/{SHA}"


def run(**overrides):
    return (
        dict(
            id=10,
            workflow_id=1,
            name="CI",
            run_attempt=2,
            head_branch="main",
            head_sha=SHA,
            status="completed",
            conclusion="failure",
            event="push",
            html_url="https://example.test/run/10",
        )
        | overrides
    )


def setup_api():
    api = Fake()
    api.data["actions/runs/10/attempts/2/jobs"] = [
        dict(conclusion="failure", steps=[dict(name="Run tests", conclusion="failure")])
    ]
    api.data["commits/" + SHA] = dict(
        parents=[dict(sha=PARENT)], commit=dict(message="Fix behavior\n\nBody")
    )
    api.data["actions/workflows/1/runs?branch=main&event=push"] = [
        run(id=9, head_sha=PARENT, conclusion="success")
    ]
    api.data[f"commits/{SHA}/pulls"] = [
        dict(
            number=3,
            merged_at="today",
            base=dict(ref="main"),
            merge_commit_sha=SHA,
            user=dict(login="author"),
        )
    ]
    api.data[f"pulls?state=all&head=owner:{BRANCH}"] = []
    api.data[f"git/ref/heads/{BRANCH}"] = APIError("missing", 404)
    api.data["labels/post-merge-revert"] = {}
    api.data[("POST", "pulls")] = dict(number=4, html_url="https://example.test/pull/4")
    api.data[("POST", "issues/4/labels")] = {}
    api.data["issues/3/comments"] = []
    api.data[("POST", "issues/3/comments")] = {}
    for label in ("main-red", "nightly-failure"):
        api.data[f"labels/{label}"] = {}
        api.data[f"issues?state=open&labels={label}"] = []
    api.data[("POST", "issues")] = {}
    return api


@pytest.mark.parametrize(
    "overrides,expected",
    [
        (dict(head_branch="feature"), "Non-main"),
        (dict(event="pull_request"), "ignored"),
        *[
            (dict(conclusion=c), "ignored")
            for c in (
                "success",
                "cancelled",
                "skipped",
                "neutral",
            )
        ],
    ],
)
def test_non_failures_ignored(overrides, expected):
    api = setup_api()
    assert expected in g.guard(api, run(**overrides))
    assert not api.calls


@pytest.mark.parametrize(
    "steps",
    [
        [],
        [dict(name="Set up job", conclusion="failure")],
        [
            dict(name="Set up runner", conclusion="failure"),
            dict(name="Run tests", conclusion="skipped"),
        ],
    ],
)
def test_infra_skipped(steps):
    api = setup_api()
    api.data["actions/runs/10/attempts/2/jobs"] = [
        dict(conclusion="failure", steps=steps)
    ]
    assert "Infrastructure" in g.guard(api, run())


def test_first_attempt_rerun():
    api = setup_api()
    api.data["actions/runs/10/attempts/1/jobs"] = api.data[
        "actions/runs/10/attempts/2/jobs"
    ]
    api.data[("POST", "actions/runs/10/rerun-failed-jobs")] = None
    assert "rerun once" in g.guard(api, run(run_attempt=1))
    assert api.calls[-1] == ("actions/runs/10/rerun-failed-jobs", "POST", None, False)


@pytest.mark.parametrize(
    "overrides,superseded",
    [
        (dict(id=11), True),
        (dict(id=9), False),
        (dict(id=10), False),
        (dict(id=11, conclusion="failure"), False),
        (dict(id=11, conclusion="cancelled"), False),
        (dict(id=11, status="in_progress"), False),
        (dict(id=11, event="pull_request"), False),
        (dict(id=11, head_branch="feature"), False),
    ],
)
def test_second_attempt_superseded_only_by_newer_completed_green_push(
    monkeypatch, overrides, superseded
):
    api = setup_api()
    api.data["actions/workflows/1/runs?branch=main&event=push"].append(
        run(head_sha="c" * 40, conclusion="success") | overrides
    )
    prepare = Mock()
    monkeypatch.setattr(g, "revert_commit", prepare)
    assert g.guard(api, run()) == (
        "Failure superseded by a newer green main run"
        if superseded
        else "Revert PR proposed"
    )
    if superseded:
        prepare.assert_not_called()
        assert not any(c[1] != "GET" for c in api.calls)
        assert not any(c[0].startswith("commits/") for c in api.calls)
    else:
        prepare.assert_called_once_with(api, SHA, BRANCH)
        assert sum(c[:2] == ("pulls", "POST") for c in api.calls) == 1


def test_attributable_direct_child_and_nearest_non_cancelled_run():
    api = setup_api()
    history = api.data["actions/workflows/1/runs?branch=main&event=push"]
    history.extend(
        [
            run(id=12, head_sha=PARENT, conclusion="cancelled"),
            run(id=13, head_sha=PARENT, status="in_progress", conclusion=None),
            run(id=14, head_sha=PARENT, event="pull_request"),
            run(id=15, head_sha=PARENT, head_branch="feature"),
        ]
    )
    assert g.attributable(api, run())[0]


@pytest.mark.parametrize("mode", ["previous-red", "range", "root", "bounded"])
def test_ambiguous_attribution_tracks_range(mode):
    api = setup_api()
    if mode == "previous-red":
        api.data["actions/workflows/1/runs?branch=main&event=push"][0]["conclusion"] = (
            "failure"
        )
    elif mode == "range":
        api.data[f"commits/{SHA}"]["parents"][0]["sha"] = "c" * 40
        api.data["commits/" + "c" * 40] = dict(parents=[dict(sha=PARENT)])
    elif mode == "root":
        api.data[f"commits/{SHA}"]["parents"] = []
    else:
        api.data["actions/workflows/1/runs?branch=main&event=push"] = []
        api.data[f"commits/{PARENT}"] = dict(parents=[dict(sha=PARENT)])
    assert "recorded" in g.guard(api, run())
    body = next(c[2]["body"] for c in api.calls if c[:2] == ("issues", "POST"))
    assert SHA in body and "Range:" in body


@pytest.mark.parametrize(
    "title,has_pr",
    [("chore: bump version to 1.0", True), ('Revert "Fix"', True), ("Fix", False)],
)
def test_protected_commits_no_revert(title, has_pr):
    api = setup_api()
    api.data[f"commits/{SHA}"]["commit"]["message"] = title
    if not has_pr:
        api.data[f"commits/{SHA}/pulls"][0]["merged_at"] = None
    assert "protected" in g.guard(api, run())


def test_revert_pr_and_author_comment_idempotence(monkeypatch):
    api = setup_api()
    revert = Mock()
    monkeypatch.setattr(g, "revert_commit", revert)
    assert g.guard(api, run()) == "Revert PR proposed"
    revert.assert_called_once_with(api, SHA, BRANCH)
    pr = next(c for c in api.calls if c[:2] == ("pulls", "POST"))
    assert pr[3] and pr[2]["title"] == 'Revert "Fix behavior"'
    assert "#3" in pr[2]["body"] and "NOT auto-merged" in pr[2]["body"]
    assert all("merge-ready" not in str(c) for c in api.calls)
    comment = next(c[2] for c in api.calls if c[:2] == ("issues/3/comments", "POST"))
    assert (
        "@author" in comment["body"]
        and "https://example.test/pull/4" in comment["body"]
    )
    api.data[f"pulls?state=all&head=owner:{BRANCH}"] = [
        dict(number=4, html_url="url", state="open")
    ]
    api.data["issues/3/comments"] = [comment]
    api.calls.clear()
    assert "already exists" in g.guard(api, run())
    assert not any(c[1] == "POST" for c in api.calls)
    revert.assert_called_once()
    api.data["issues/3/comments"] = []
    assert "already exists" in g.guard(api, run())
    assert any(c[:2] == ("issues/3/comments", "POST") for c in api.calls)


def test_existing_branch_api_error():
    api = setup_api()
    api.data[f"git/ref/heads/{BRANCH}"] = APIError("forbidden", 403)
    with pytest.raises(APIError):
        g.guard(api, run())


def test_missing_pat_and_conflict_fallback(monkeypatch):
    api = setup_api()
    api.pat = ""
    with pytest.raises(APIError, match="RELEASE_PAT"):
        g.guard(api, run())
    api.pat = "write"
    monkeypatch.setattr(g, "revert_commit", Mock(side_effect=APIError("conflict")))
    assert "conflict" in g.guard(api, run())
    assert any("conflict" in str(c[2]) for c in api.calls if c[1] == "POST")


def test_label_and_issue_refresh():
    api = Fake()
    api.data["labels/main-red"] = APIError("missing", 404)
    api.data[("POST", "labels")] = {}
    api.data["issues?state=open&labels=main-red"] = [
        dict(number=8),
        dict(number=9, pull_request={}),
    ]
    api.data[("PATCH", "issues/8")] = {}
    g.issue(api, "main-red", "Main CI is failing", "failure")
    assert api.calls[-1][0:2] == ("issues/8", "PATCH")
    api.data[("POST", "labels")] = APIError("already exists", 422)
    g.ensure_label(api, "main-red")
    api.data[("POST", "labels")] = APIError("forbidden", 403)
    with pytest.raises(APIError):
        g.ensure_label(api, "main-red")
    api.data["labels/main-red"] = APIError("forbidden", 403)
    with pytest.raises(APIError):
        g.ensure_label(api, "main-red")


def test_nightly_open_refresh_and_close():
    api = setup_api()
    nightly = run(event="schedule")
    key = "actions/workflows/1/runs?branch=main&event=schedule"
    api.data[key] = [run(id=9, event="schedule", head_sha=PARENT, conclusion="success")]
    assert "failure recorded" in g.guard(api, nightly)
    body = next(c[2]["body"] for c in api.calls if c[:2] == ("issues", "POST"))
    assert f"{PARENT}..{SHA}" in body
    api.data["issues?state=open&labels=nightly-failure"] = [dict(number=8)]
    api.data[("PATCH", "issues/8")] = {}
    api.data[key] = []
    g.guard(api, nightly)
    assert "no green nightly on record" in api.calls[-1][2]["body"]
    for workflow in g.NIGHTLIES:
        api.data[f"actions/workflows/{workflow}/runs?branch=main&event=schedule"] = []
    assert "pending" in g.guard(api, nightly | dict(conclusion="success"))
    for workflow in g.NIGHTLIES:
        api.data[f"actions/workflows/{workflow}/runs?branch=main&event=schedule"] = [
            run(event="schedule", conclusion="success")
        ]
    api.data[("POST", "issues/8/comments")] = {}
    assert "recovery recorded" in g.guard(api, nightly | dict(conclusion="success"))
    assert api.calls[-1][2] == dict(state="closed")
    assert "ignored" in g.guard(api, nightly | dict(conclusion="cancelled"))


@pytest.mark.parametrize("page_size", [100, 1])
def test_run_history_pagination_is_bounded(monkeypatch, page_size):
    api = GitHub("owner/repo", "read")
    request = Mock(
        side_effect=[
            dict(workflow_runs=[run(id=400 - page * 100 - i) for i in range(page_size)])
            for page in range(3)
        ]
    )
    monkeypatch.setattr(api, "request", request)
    history = g.runs(api, "ci.yml", "push")
    pages = 3 if page_size == 100 else 1
    assert len(history) == pages * page_size
    assert [r["id"] for r in history] == sorted(
        [r["id"] for r in history], reverse=True
    )
    assert request.call_args_list == [
        call(
            "actions/workflows/ci.yml/runs?branch=main&event=push"
            f"&per_page=100&page={page}"
        )
        for page in range(1, pages + 1)
    ]


@pytest.mark.parametrize("runner_temp", ["set", "empty", "unset"])
def test_git_revert_uses_fresh_main_and_pat(monkeypatch, tmp_path, runner_temp):
    api = Fake()
    if runner_temp == "unset":
        monkeypatch.delenv("RUNNER_TEMP", raising=False)
    else:
        monkeypatch.setenv("RUNNER_TEMP", str(tmp_path) if runner_temp == "set" else "")
    temporary_directory = Mock(wraps=g.tempfile.TemporaryDirectory)
    monkeypatch.setattr(g.tempfile, "TemporaryDirectory", temporary_directory)
    command = Mock(return_value=Mock(returncode=0))
    monkeypatch.setattr(g.subprocess, "run", command)
    g.revert_commit(api, SHA, BRANCH)
    temporary_directory.assert_called_once_with(
        prefix="main-guard-revert-", dir=str(tmp_path) if runner_temp == "set" else None
    )
    commands = [c.args[0] for c in command.call_args_list]
    assert commands[0][-3:] == ["main", "https://github.com/owner/repo.git", "."]
    assert commands[-1][-3:] == ["push", "origin", f"HEAD:refs/heads/{BRANCH}"]
    assert all("core.hooksPath=/dev/null" in c for c in commands)
    assert (
        "AUTHORIZATION: basic" in command.call_args.kwargs["env"]["GIT_CONFIG_VALUE_0"]
    )
    command.return_value.returncode = 1
    with pytest.raises(APIError, match="clone"):
        g.revert_commit(api, SHA, BRANCH)
    command.side_effect = subprocess.TimeoutExpired("git", 180)
    with pytest.raises(subprocess.TimeoutExpired):
        g.revert_commit(api, SHA, BRANCH)
    api.pat = ""
    with pytest.raises(APIError, match="RELEASE_PAT"):
        g.revert_commit(api, SHA, BRANCH)


def test_cli(monkeypatch, tmp_path):
    api = setup_api()
    event = tmp_path / "event.json"
    event.write_text(json.dumps(dict(workflow_run=dict(id=10))))
    api.data["actions/runs/10"] = run(conclusion="success")
    monkeypatch.setattr(g, "GitHub", lambda *args: api)
    arguments = ["--repo", "owner/repo", "--event-path", str(event)]
    assert g.main(arguments) == 0
    api.data["actions/runs/10"]["status"] = "in_progress"
    assert g.main(arguments) == 0
    api.data["actions/runs/10"] = APIError("offline")
    assert g.main(arguments) == 1
    monkeypatch.delenv("GITHUB_EVENT_PATH", raising=False)
    with pytest.raises(SystemExit):
        g.main(["--repo", "owner/repo"])
    monkeypatch.setattr(g.sys, "argv", ["main_guard", *arguments])
    monkeypatch.setenv("GH_TOKEN", "read")
    monkeypatch.setattr(
        "scripts.merge_on_green.urllib.request.urlopen",
        Mock(side_effect=ValueError("offline")),
    )
    with pytest.raises(SystemExit) as result:
        runpy.run_module("scripts.main_guard", run_name="__main__")
    assert result.value.code == 1


@pytest.mark.parametrize("facade", ["tests", "desktop-tests"])
@pytest.mark.parametrize("steps", [[], [dict(name="Set up job", conclusion="failure")]])
def test_setup_failure_with_failed_facade_is_infrastructure(facade, steps):
    api = setup_api()
    api.data["actions/runs/10/attempts/2/jobs"] = [
        dict(name="workload", conclusion="failure", steps=steps),
        dict(
            name=facade,
            conclusion="failure",
            steps=[dict(name="Check results", conclusion="failure")],
        ),
    ]
    assert g.guard(api, run()) == "Infrastructure-only failure ignored"
    assert not any(c[1] == "POST" for c in api.calls)


@pytest.mark.parametrize("facade", ["tests", "desktop-tests"])
def test_only_failed_facade_is_real_failure(facade):
    api = setup_api()
    api.data["actions/runs/10/attempts/1/jobs"] = [
        dict(name=facade, conclusion="failure", steps=[])
    ]
    api.data[("POST", "actions/runs/10/rerun-failed-jobs")] = None
    assert g.guard(api, run(run_attempt=1)) == "Failed jobs rerun once"


@pytest.mark.parametrize("creation_fails", [False, True])
def test_orphan_revert_branch_resumes_pr_creation(monkeypatch, creation_fails):
    api = setup_api()
    api.data[f"git/ref/heads/{BRANCH}"] = {}
    prepare = Mock()
    monkeypatch.setattr(g, "revert_commit", prepare)
    if creation_fails:
        api.data[("POST", "pulls")] = APIError("PR creation refused", 422)
        assert g.guard(api, run()) == "Revert PR creation failure recorded"
        assert any(
            "PR creation refused" in str(c[2])
            for c in api.calls
            if c[:2] == ("issues", "POST")
        )
    else:
        assert g.guard(api, run()) == "Revert PR proposed"
        pr = next(c for c in api.calls if c[:2] == ("pulls", "POST"))
        assert pr[3] and pr[2]["head"] == BRANCH
        assert pr[2]["title"] == 'Revert "Fix behavior"'
        assert "#3" in pr[2]["body"] and "NOT auto-merged" in pr[2]["body"]
        assert any(c[:2] == ("issues/4/labels", "POST") for c in api.calls)
        assert any(c[:2] == ("issues/3/comments", "POST") for c in api.calls)
    prepare.assert_not_called()


@pytest.mark.parametrize("refresh", [False, True])
@pytest.mark.parametrize("label", ["main-red", "nightly-failure"])
def test_tracking_issue_dedupes_concurrent_creation_and_refresh(refresh, label):
    api = setup_api()
    listing = f"issues?state=open&labels={label}"
    reads = 0

    def handler(path, method, data, pat):
        nonlocal reads
        if path == listing:
            reads += 1
            if not refresh and reads == 1:
                return []
            return [
                dict(number=12),
                dict(number=2, pull_request={}),
                dict(number=8),
                dict(number=10),
            ]
        if method in {"POST", "PATCH"}:
            return {}
        return NotImplemented

    api.handler = handler
    g.issue(api, label, "Failure", "Current failure")
    assert (
        "issues/8",
        "PATCH",
        {"body": f"<!-- {label} -->\nCurrent failure"},
        False,
    ) in api.calls
    for number in (10, 12):
        comment = (
            f"issues/{number}/comments",
            "POST",
            {"body": "Duplicate of #8"},
            False,
        )
        close = (f"issues/{number}", "PATCH", {"state": "closed"}, False)
        assert comment in api.calls and close in api.calls
        assert api.calls.index(comment) < api.calls.index(close)
    assert not any(c[0].startswith("issues/2") for c in api.calls)
    assert reads >= (1 if refresh else 2)


@pytest.mark.parametrize("conclusion", ["timed_out", "startup_failure"])
def test_push_infrastructure_conclusions_rerun_once_then_track(conclusion):
    api = setup_api()
    api.data[("POST", "actions/runs/10/rerun-failed-jobs")] = None
    assert "rerun once" in g.guard(api, run(conclusion=conclusion, run_attempt=1))
    assert api.calls == [("actions/runs/10/rerun-failed-jobs", "POST", None, False)]
    api.calls.clear()
    assert "recorded" in g.guard(api, run(conclusion=conclusion))
    body = next(c[2]["body"] for c in api.calls if c[:2] == ("issues", "POST"))
    assert SHA in body and "https://example.test/run/10" in body and conclusion in body
    assert all(c[0].startswith(("labels/", "issues")) for c in api.calls)


@pytest.mark.parametrize("attempt", [1, 2])
def test_push_action_required_only_tracks(attempt):
    api = setup_api()
    assert "recorded" in g.guard(
        api, run(conclusion="action_required", run_attempt=attempt)
    )
    assert any(
        c[:2] == ("issues", "POST") and c[2]["labels"] == ["main-red"]
        for c in api.calls
    )
    assert all(c[0].startswith(("labels/", "issues")) for c in api.calls)


@pytest.mark.parametrize(
    "conclusion",
    ["failure", "timed_out", "startup_failure", "action_required", "unknown", None],
)
@pytest.mark.parametrize("existing", [False, True])
def test_nightly_unhealthy_conclusions_open_or_refresh(conclusion, existing):
    api = setup_api()
    api.data["actions/workflows/1/runs?branch=main&event=schedule"] = []
    if existing:
        api.data["issues?state=open&labels=nightly-failure"] = [dict(number=8)]
        api.data[("PATCH", "issues/8")] = {}
    assert "failure recorded" in g.guard(
        api, run(event="schedule", conclusion=conclusion)
    )
    update = next(
        c
        for c in api.calls
        if c[:2] == (("issues/8", "PATCH") if existing else ("issues", "POST"))
    )
    assert (
        SHA in update[2]["body"] and "https://example.test/run/10" in update[2]["body"]
    )


@pytest.mark.parametrize("conclusion", ["cancelled", "skipped", "neutral"])
def test_nightly_ignored_conclusions(conclusion):
    api = setup_api()
    assert "ignored" in g.guard(api, run(event="schedule", conclusion=conclusion))
    assert not api.calls


@pytest.mark.parametrize("has_pr", [False, True])
@pytest.mark.parametrize("notified", [False, True])
@pytest.mark.parametrize(
    "failure",
    [
        APIError("push rejected"),
        OSError("git failed"),
        subprocess.TimeoutExpired("git", 180),
    ],
)
def test_concurrent_revert_branch_creation(monkeypatch, has_pr, notified, failure):
    api = setup_api()
    listing = f"pulls?state=all&head=owner:{BRANCH}"
    revert = dict(number=4, html_url="https://example.test/pull/4")

    def prepare(*args):
        api.data[f"git/ref/heads/{BRANCH}"] = {}
        raise failure

    def handler(path, method, data, pat):
        if (path, method) == ("pulls", "POST") and has_pr:
            api.data[listing] = [revert]
            raise APIError("PR already exists for head", 422)
        return NotImplemented

    api.handler = handler
    monkeypatch.setattr(g, "revert_commit", prepare)
    if notified:
        api.data["issues/3/comments"] = [dict(body=f"<!-- main-guard:{SHA} -->")]
    assert g.guard(api, run()) == (
        "Revert PR already exists" if has_pr else "Revert PR proposed"
    )
    assert sum(c[0] == f"git/ref/heads/{BRANCH}" for c in api.calls) == 2
    assert sum(c[0] == listing for c in api.calls) == (2 if has_pr else 1)
    writes = [c for c in api.calls if c[1] != "GET"]
    expected_writes = ["pulls"]
    if not has_pr:
        expected_writes.append("issues/4/labels")
    if not notified:
        expected_writes.append("issues/3/comments")
    assert [c[0] for c in writes] == expected_writes
    pr = writes[0]
    assert pr[3] and pr[2]["head"] == BRANCH and pr[2]["base"] == "main"
    if not notified:
        assert "@author" in writes[-1][2]["body"]
        assert revert["html_url"] in writes[-1][2]["body"]


@pytest.mark.parametrize("status", [403, 422, 500])
def test_revert_push_failure_with_branch_records_pr_creation_failure(
    monkeypatch, status
):
    api = setup_api()

    def prepare(*args):
        api.data[f"git/ref/heads/{BRANCH}"] = {}
        raise APIError("push response lost")

    monkeypatch.setattr(g, "revert_commit", prepare)
    api.data[("POST", "pulls")] = APIError("PR creation refused", status)
    assert g.guard(api, run()) == "Revert PR creation failure recorded"
    recorded = next(c for c in api.calls if c[:2] == ("issues", "POST"))
    assert recorded[2]["labels"] == ["main-red"]
    assert "PR creation refused" in recorded[2]["body"]
    assert not any(c[:2] == ("issues/3/comments", "POST") for c in api.calls)


@pytest.mark.parametrize("notified", [False, True])
def test_concurrent_revert_pr_creation(monkeypatch, notified):
    api = setup_api()
    monkeypatch.setattr(g, "revert_commit", Mock())
    listing = f"pulls?state=all&head=owner:{BRANCH}"

    def handler(path, method, data, pat):
        if (path, method) == ("pulls", "POST"):
            api.data[listing] = [dict(number=4, html_url="https://example.test/pull/4")]
            raise APIError("PR already exists for head", 422)
        return NotImplemented

    api.handler = handler
    if notified:
        api.data["issues/3/comments"] = [dict(body=f"<!-- main-guard:{SHA} -->")]
    assert g.guard(api, run()) == "Revert PR already exists"
    assert sum(c[0] == listing for c in api.calls) == 2
    assert [c[0] for c in api.calls if c[1] != "GET"] == (
        ["pulls"] if notified else ["pulls", "issues/3/comments"]
    )


@pytest.mark.parametrize("race", [False, True])
@pytest.mark.parametrize(
    "state,merged_at", [("closed", None), ("open", None), ("closed", "today")]
)
def test_existing_revert_pr_state_reporting(monkeypatch, race, state, merged_at):
    api = setup_api()
    listing = f"pulls?state=all&head=owner:{BRANCH}"
    revert = dict(
        number=4,
        html_url="https://example.test/pull/4",
        state=state,
        merged_at=merged_at,
    )
    prepare = Mock()
    monkeypatch.setattr(g, "revert_commit", prepare)
    if race:
        api.data[f"git/ref/heads/{BRANCH}"] = {}

        def handler(path, method, data, pat):
            if (path, method) == ("pulls", "POST"):
                api.data[listing] = [revert]
                raise APIError("PR already exists for head", 422)
            return NotImplemented

        api.handler = handler
    else:
        api.data[listing] = [revert]
    closed_unmerged = state == "closed" and not merged_at
    assert g.guard(api, run()) == (
        "Closed revert PR recorded" if closed_unmerged else "Revert PR already exists"
    )
    prepare.assert_not_called()
    assert sum(c[:2] == ("pulls", "POST") for c in api.calls) == int(race)
    assert not any(c[:2] == ("issues/4/labels", "POST") for c in api.calls)
    if closed_unmerged:
        recorded = next(c[2] for c in api.calls if c[:2] == ("issues", "POST"))
        assert recorded["title"] == "Main CI is failing"
        assert recorded["labels"] == ["main-red"]
        assert revert["html_url"] in recorded["body"]
        assert f"{PARENT}..{SHA}" in recorded["body"]
        assert "CI: https://example.test/run/10" in recorded["body"]
    else:
        assert not any(c[:2] == ("issues", "POST") for c in api.calls)
        comment = next(
            c[2]["body"] for c in api.calls if c[:2] == ("issues/3/comments", "POST")
        )
        assert "@author" in comment and revert["html_url"] in comment


@pytest.mark.parametrize("status", [403, 500])
def test_revert_branch_recheck_errors_fail_closed(monkeypatch, status):
    api = setup_api()

    def prepare(*args):
        api.data[f"git/ref/heads/{BRANCH}"] = APIError("lookup failed", status)
        raise APIError("push failed")

    monkeypatch.setattr(g, "revert_commit", prepare)
    with pytest.raises(APIError, match="lookup failed"):
        g.guard(api, run())
    assert not any(c[1] != "GET" for c in api.calls)


@pytest.mark.parametrize(
    "sha,created_at,expected",
    [
        (SHA, "2026-10-15T09:17:00Z", "recorded"),
        (PARENT, "2026-10-17T09:17:00Z", "pending"),
        (PARENT, "2026-10-17T15:17:00Z", "pending"),
        (PARENT, "2026-10-17T03:17:00Z", "pending"),
        (PARENT, "2026-10-17T15:17:01Z", "pending"),
        (PARENT, "2026-10-17T03:16:59Z", "pending"),
        (PARENT, "2026-10-16T09:17:00Z", "pending"),
    ],
)
def test_nightly_recovery_requires_same_cycle(sha, created_at, expected):
    api = setup_api()
    green = run(
        event="schedule", conclusion="success", created_at="2026-10-17T09:17:00Z"
    )
    for workflow, latest in zip(
        g.NIGHTLIES, [green, green | dict(head_sha=sha, created_at=created_at)]
    ):
        api.data[f"actions/workflows/{workflow}/runs?branch=main&event=schedule"] = [
            latest,
            green | dict(id=9),
        ]
    api.data["issues?state=open&labels=nightly-failure"] = [
        dict(number=8),
        dict(number=9, pull_request={}),
    ]
    api.data[("POST", "issues/8/comments")] = {}
    api.data[("PATCH", "issues/8")] = {}
    assert g.guard(api, green) == f"Nightly recovery {expected}"
    writes = [c for c in api.calls if c[1] != "GET"]
    assert [c[0] for c in writes] == (
        ["issues/8/comments", "issues/8"] if expected == "recorded" else []
    )


@pytest.mark.parametrize(
    "status,conclusion", [("in_progress", "success"), ("completed", "failure")]
)
def test_nightly_recovery_requires_latest_success_for_every_workflow(
    status, conclusion
):
    api = setup_api()
    green = run(event="schedule", conclusion="success")
    for workflow in g.NIGHTLIES:
        api.data[f"actions/workflows/{workflow}/runs?branch=main&event=schedule"] = [
            green
        ]
    api.data[f"actions/workflows/{g.NIGHTLIES[1]}/runs?branch=main&event=schedule"] = [
        green | dict(id=11, status=status, conclusion=conclusion),
        green,
    ]
    assert g.guard(api, green) == "Nightly recovery pending"
    assert not any(c[1] != "GET" for c in api.calls)
