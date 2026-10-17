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
                "startup_failure",
                "timed_out",
                "action_required",
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
        dict(number=4, html_url="url", state="closed")
    ]
    api.data["issues/3/comments"] = [comment]
    api.calls.clear()
    assert "already exists" in g.guard(api, run())
    assert not any(c[1] == "POST" for c in api.calls)
    revert.assert_called_once()
    api.data["issues/3/comments"] = []
    assert "already exists" in g.guard(api, run())
    assert any(c[:2] == ("issues/3/comments", "POST") for c in api.calls)


def test_existing_branch_and_api_error():
    api = setup_api()
    api.data[f"git/ref/heads/{BRANCH}"] = {}
    assert "branch already" in g.guard(api, run())
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
