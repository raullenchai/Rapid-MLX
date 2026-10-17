# SPDX-License-Identifier: Apache-2.0
"""Exact-head merge authorization and fail-closed release freeze contracts."""

import copy
import runpy
import urllib.error
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml

from scripts import merge_on_green as m

SHA = "a" * 40


class Fake(m.GitHub):
    def __init__(self):
        super().__init__("owner/repo", "read", "write")
        self.data = {}
        self.calls = []
        self.handler = None

    def request(self, path, method="GET", data=None, *, pat=False):
        self.calls.append((path, method, data, pat))
        if pat and not self.pat:
            raise m.APIError("RELEASE_PAT is required")
        if self.handler:
            value = self.handler(path, method, data, pat)
            if value is not NotImplemented:
                return copy.deepcopy(value)
        value = self.data.get((method, path), self.data.get(path, NotImplemented))
        if isinstance(value, Exception):
            raise value
        if value is NotImplemented:
            raise AssertionError(f"Unexpected request: {method} {path}")
        return copy.deepcopy(value)

    def items(self, path, key=None, *, max_pages=None):
        return self.request(path)


def pull():
    return dict(
        number=1,
        state="open",
        draft=False,
        title="Fix behavior",
        base=dict(ref="main", repo=dict(full_name="owner/repo")),
        head=dict(sha=SHA, repo=dict(full_name="owner/repo")),
        labels=[dict(name="merge-ready")],
        mergeable=True,
        mergeable_state="clean",
    )


def check(name, identifier=1, conclusion="success", app="github-actions"):
    return dict(
        name=name,
        id=identifier,
        conclusion=conclusion,
        status="completed",
        app=dict(slug=app),
    )


def ready():
    api = Fake()
    api.data["pulls/1"] = pull()
    api.data["pulls?state=open&base=main"] = [dict(number=1)]
    api.data[f"commits/{SHA}/check-runs?filter=all"] = [
        check(n)
        for n in ("tests", "desktop-tests", "version-bump-guard", "merge-lane-no-mac")
    ]
    api.data[f"commits/{SHA}/statuses"] = [
        dict(context="merge-ready-head", id=1, state="success")
    ]
    api.data["actions/workflows/auto-release.yml/runs"] = []
    for status in ("queued", "in_progress", "waiting", "pending", "requested"):
        api.data[f"actions/workflows/auto-release.yml/runs?status={status}"] = []
    api.data["commits/main"] = dict(sha=SHA, commit=dict(message="Fix behavior"))
    api.data[("PUT", "pulls/1/merge")] = dict(merged=True, message="merged")
    return api


@pytest.mark.parametrize(
    "path,value",
    [
        (("state",), "closed"),
        (("base", "ref"), "other"),
        (("draft",), True),
        (("head", "repo"), None),
        (("head", "repo", "full_name"), "fork/repo"),
        (("base", "repo", "full_name"), "elsewhere/repo"),
        (("labels",), []),
        (("labels",), [dict(name="merge-ready"), dict(name="merge-ready-mac")]),
        (("labels",), [dict(name="merge-ready"), dict(name="version-bump")]),
        (("labels",), [dict(name="merge-ready"), dict(name="skip-version-bump")]),
        (("title",), "chore: bump version to 1.0"),
        (("mergeable",), False),
        (("mergeable",), None),
        (("mergeable_state",), "unknown"),
        (("mergeable_state",), "dirty"),
        (("mergeable_state",), "blocked"),
        (("mergeable_state",), "behind"),
    ],
)
def test_each_pr_condition_rejects(path, value):
    api = ready()
    pr = pull()
    target = pr
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value
    assert m.eligibility(api, pr)


@pytest.mark.parametrize("state", ["clean", "unstable", "has_hooks"])
def test_allowed_mergeability_without_up_to_date_requirement(state):
    api = ready()
    pr = pull()
    pr["mergeable_state"] = state
    assert not m.eligibility(api, pr)


@pytest.mark.parametrize(
    "name", ["tests", "desktop-tests", "version-bump-guard", "merge-lane-no-mac"]
)
@pytest.mark.parametrize(
    "conclusion", [None, "failure", "cancelled", "skipped", "missing", "pending"]
)
def test_required_checks_fail_closed(name, conclusion):
    api = ready()
    checks = api.data[f"commits/{SHA}/check-runs?filter=all"]
    checks[:] = [c for c in checks if c["name"] != name]
    if conclusion != "missing":
        item = check(name, 2, conclusion)
        if conclusion == "pending":
            item["status"] = "in_progress"
        checks.append(item)
    assert name in m.eligibility(api, pull())


def test_latest_checks_and_foreign_app():
    api = ready()
    checks = api.data[f"commits/{SHA}/check-runs?filter=all"]
    checks.extend(
        [check("tests", 200, "failure", "foreign"), check("tests", 0, "failure")]
    )
    assert not m.eligibility(api, pull())
    checks.append(check("tests", 2, "failure"))
    assert "tests" in m.eligibility(api, pull())
    checks.append(check("tests", 3))
    assert not m.eligibility(api, pull())
    checks[:] = [
        c
        for c in checks
        if c["name"] != "tests" or c["app"]["slug"] != "github-actions"
    ]
    assert "tests" in m.eligibility(api, pull())


@pytest.mark.parametrize(
    "conclusion,expected", [("success", ""), ("skipped", ""), ("failure", "type-check")]
)
def test_optional_type_check(conclusion, expected):
    api = ready()
    api.data[f"commits/{SHA}/check-runs?filter=all"].append(
        check("type-check", 1, conclusion)
    )
    assert expected in m.eligibility(api, pull())


def test_mac_lane_requires_correct_check():
    api = ready()
    pr = pull()
    pr["labels"] = [dict(name="merge-ready-mac")]
    assert "merge-lane-mac" in m.eligibility(api, pr)
    api.data[f"commits/{SHA}/check-runs?filter=all"].append(check("merge-lane-mac"))
    assert not m.eligibility(api, pr)


@pytest.mark.parametrize("state", ["pending", "failure", "error", "missing"])
def test_latest_status_binding(state):
    api = ready()
    statuses = api.data[f"commits/{SHA}/statuses"]
    if state == "missing":
        statuses.clear()
    else:
        statuses.append(dict(context="merge-ready-head", id=2, state=state))
    statuses.append(dict(context="other", id=100, state="success"))
    assert "merge-ready-head" in m.eligibility(api, pull())


def test_freeze_variable_and_api_error():
    api = ready()
    assert "RELEASE_FREEZE" in m.freeze(api, "true")
    assert not api.calls
    api.data["commits/main"] = m.APIError("offline")
    assert "failed" in m.freeze(api)
    api.data["commits/main"] = {}
    assert "failed" in m.freeze(api)


@pytest.mark.parametrize(
    "jobs,expected",
    [
        ([], "undetermined"),
        ([dict(name="detect", status="in_progress")], "undetermined"),
        ([dict(name="detect", status="completed")], "undetermined"),
        (
            [
                dict(name="detect", status="completed"),
                dict(name="release", status="queued"),
            ],
            "releasing",
        ),
        (
            [
                dict(name="detect", status="completed"),
                dict(name="release-final-ci", status="completed", conclusion="success"),
            ],
            "releasing",
        ),
        (
            [
                dict(name="detect", status="completed"),
                dict(name="release", status="completed", conclusion="skipped"),
            ],
            "",
        ),
    ],
)
def test_release_detection(jobs, expected):
    api = ready()
    api.data["actions/workflows/auto-release.yml/runs"] = [dict(id=7, status="waiting")]
    api.data["actions/runs/7"] = dict(id=7, status="waiting")
    api.data["actions/runs/7/jobs"] = jobs
    assert expected in m.freeze(api)
    if not expected:
        assert m.freeze(api) == ""


def test_completed_run_does_not_freeze_and_bump_head_race():
    api = ready()
    api.data["actions/workflows/auto-release.yml/runs"] = [
        dict(id=7, status="completed")
    ]
    assert m.freeze(api) == ""
    api.data["commits/main"]["commit"]["message"] = "chore: bump version to 1.0\nbody"
    key = f"actions/workflows/auto-release.yml/runs?event=push&head_sha={SHA}"
    api.data[key] = [dict(head_sha=SHA, event="workflow_dispatch", status="completed")]
    assert "awaits" in m.freeze(api)
    api.data[key].append(dict(head_sha=SHA, event="push", status="completed"))
    assert not m.freeze(api)


def test_live_mergeability_bounded_poll(monkeypatch):
    api = ready()
    api.data["pulls/1"]["mergeable"] = None
    sleep = Mock()
    monkeypatch.setattr(m.time, "sleep", sleep)
    assert m.live_pr(api, 1)["mergeable"] is None
    assert sleep.call_count == 2


def test_merge_freeze_first_dry_run_and_success():
    api = ready()
    assert m.merge(api, manual_freeze="true")[0][1] == "frozen"
    assert not api.calls
    assert m.merge(api, dry_run=True)[0][1] == "eligible"
    assert not any(c[1] == "PUT" for c in api.calls)
    assert m.merge(api)[0][1] == "merged"
    write = next(c for c in api.calls if c[1] == "PUT")
    assert write[2] == dict(
        merge_method="squash", sha=SHA, commit_title="Fix behavior (#1)"
    )
    assert write[3]


@pytest.mark.parametrize("status", [405, 409])
def test_merge_head_races_continue(status):
    api = ready()
    api.data[("PUT", "pulls/1/merge")] = m.APIError("head changed", status)
    assert m.merge(api)[0][1] == "skip"


def test_skip_reasons_no_label_and_refused_merge():
    api = ready()
    api.data["pulls/1"]["labels"] = []
    assert m.merge(api) == []
    assert m.merge(api, number=1)[0][1] == "skip"
    api = ready()
    api.data[("PUT", "pulls/1/merge")] = dict(merged=False)
    assert m.merge(api)[0][1] == "skip"


@pytest.mark.parametrize(
    "mode", ["freeze-before", "freeze-after", "moved", "label-changed"]
)
def test_rechecks_and_one_at_a_time(monkeypatch, mode):
    api = ready()
    api.data["pulls?state=open&base=main"] += [dict(number=2)]
    api.data["pulls/2"] = pull() | dict(number=2)
    freezes = iter(
        ["", "frozen"]
        if mode == "freeze-before"
        else ["", "", "frozen"]
        if mode == "freeze-after"
        else ["", "", ""]
    )
    monkeypatch.setattr(m, "freeze", lambda *a: next(freezes))
    reads = 0

    def handler(path, method, data, pat):
        nonlocal reads
        if path == "pulls/1":
            reads += 1
            if reads == 2 and mode in {"moved", "label-changed"}:
                api.data["pulls/1"]["head"]["sha"] = (
                    "b" * 40 if mode == "moved" else SHA
                )
                api.data["pulls/1"]["labels"] = []
        if path == "pulls/2":
            return pull() | dict(labels=[])
        return NotImplemented

    api.handler = handler
    rows = m.merge(api)
    assert any(r[1] in {"frozen", "skip"} for r in rows)
    writes = [c for c in api.calls if c[1] == "PUT"]
    assert len(writes) == (1 if mode == "freeze-after" else 0)


def test_missing_pat_fails_closed():
    api = ready()
    api.pat = ""
    with pytest.raises(m.APIError, match="RELEASE_PAT"):
        m.merge(api)
    api.data[("PUT", "pulls/1/merge")] = m.APIError("forbidden", 403)
    api.pat = "token"
    with pytest.raises(m.APIError):
        m.merge(api)


def test_transport(monkeypatch):
    response = Mock()
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    response.read.return_value = b'{"merged":true}'
    urlopen = Mock(return_value=response)
    monkeypatch.setattr(m.urllib.request, "urlopen", urlopen)
    api = m.GitHub("owner/repo", "read", "pat")
    assert api.request("pulls/1/merge", "PUT", {}, pat=True)["merged"]
    assert urlopen.call_args[0][0].headers["Authorization"] == "Bearer pat"
    response.read.return_value = b""
    assert api.request("test", "POST") is None
    for exc in (
        urllib.error.HTTPError("url", 409, "conflict", {}, None),
        ValueError("bad json"),
    ):
        urlopen.side_effect = exc
        with pytest.raises(m.APIError):
            api.request("test")
    for pat in (False, True):
        with pytest.raises(m.APIError):
            m.GitHub("repo", "").request("test", pat=pat)


def test_paginated_transport(monkeypatch):
    api = m.GitHub("repo", "read")
    request = Mock(side_effect=[dict(jobs=[{}] * 100), dict(jobs=[{}])])
    monkeypatch.setattr(api, "request", request)
    assert len(api.items("jobs?filter=all", "jobs")) == 101
    request.side_effect = None
    request.return_value = []
    assert api.items("list") == []
    request.return_value = "malformed"
    with pytest.raises(m.APIError, match="Malformed"):
        api.items("list")
    request.return_value = [{}] * 100
    with pytest.raises(m.APIError, match="Pagination"):
        api.items("list")


def test_summary_and_cli(monkeypatch, tmp_path):
    output = tmp_path / "summary"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(output))
    m.summary([("1", "skip", "pipe|new\nline\r")])
    assert "pipe\\|new line " in output.read_text()
    monkeypatch.delenv("GITHUB_STEP_SUMMARY")
    m.summary([])
    monkeypatch.setattr(m, "GitHub", lambda *a: ready())
    assert m.main(["--repo", "owner/repo", "--dry-run", "--pr", "1"]) == 0
    monkeypatch.setattr(m, "merge", Mock(side_effect=m.APIError("offline")))
    assert m.main(["--repo", "owner/repo"]) == 1
    monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
    with pytest.raises(SystemExit):
        m.main([])
    monkeypatch.setattr(
        m.sys, "argv", ["merge_on_green", "--repo", "owner/repo", "--dry-run"]
    )
    monkeypatch.setattr(
        m.urllib.request, "urlopen", Mock(side_effect=ValueError("offline"))
    )
    with pytest.raises(SystemExit) as result:
        runpy.run_module("scripts.merge_on_green", run_name="__main__")
    assert (
        result.value.code == 0
    )  # freeze errors stop merging without failing the workflow


def test_workflow_security_and_triggers():
    root = Path(__file__).resolve().parents[1]
    for name in ("merge-on-green", "main-guard"):
        data = yaml.safe_load((root / f".github/workflows/{name}.yml").read_text())
        triggers = data.get("on", data.get(True))
        assert triggers["workflow_run"]["types"] == ["completed"]
        assert data["concurrency"]["cancel-in-progress"] is False
        steps = next(iter(data["jobs"].values()))["steps"]
        checkout = next(
            s for s in steps if s.get("uses", "").startswith("actions/checkout@")
        )
        assert checkout["with"] == dict(
            ref="${{ github.event.repository.default_branch }}",
            **{"persist-credentials": False},
        )
        write = steps[-1]
        assert write["env"]["RELEASE_PAT"] == "${{ secrets.RELEASE_PAT }}"
        if name == "merge-on-green":
            assert triggers["pull_request_target"]["types"] == ["labeled"]
            assert triggers["schedule"] == [dict(cron="*/5 * * * *")]
            assert set(triggers["workflow_run"]["workflows"]) == {
                "CI",
                "rapid-mac CI",
                "Version bump guard",
                "Bind merge-ready to head",
                "Auto-release on version bump",
            }
            assert (
                triggers["workflow_dispatch"]["inputs"]["dry_run"]["type"] == "boolean"
            )
            assert write["if"] == "vars.MERGE_ON_GREEN_ENABLED == 'true'"
            assert checkout["if"] == write["if"]
            assert all(v == "read" for v in data["permissions"].values())
        else:
            assert data["concurrency"]["group"] == (
                "main-guard-${{ github.event.workflow_run.id }}"
            )
            assert triggers["workflow_run"]["branches"] == ["main"]
            assert data["permissions"] == {
                "contents": "read",
                "actions": "write",
                "issues": "write",
                "pull-requests": "write",
            }
    for name in ("ci", "rapid-mac-ci"):
        data = yaml.safe_load((root / f".github/workflows/{name}.yml").read_text())
        triggers = data.get("on", data.get(True))
        assert triggers["schedule"] == [dict(cron="17 9 * * *")]
        assert "workflow_dispatch" in triggers and "merge_group" in triggers
        assert (
            "github.event_name == 'schedule' && '-nightly' || ''"
            in data["concurrency"]["group"]
        )
        assert data["jobs"]["changes"]["if"] == "${{ !cancelled() }}"
        if name == "ci":
            matrix = str(data["jobs"]["test-matrix"]["steps"])
            assert "--cov=scripts.merge_on_green" in matrix
            assert "--cov=scripts.main_guard" in matrix
            assert (
                "needs.changes.outputs.full_gate == 'true'"
                in data["jobs"]["l1-smoke"]["if"]
            )
        assert (
            "github.event_name == 'push'" in data["jobs"]["queue-tree-evidence"]["if"]
        )


def test_multiple_merges_are_ordered_with_freeze_between(monkeypatch):
    api = ready()
    api.data["pulls?state=open&base=main"] = [dict(number=2), dict(number=1)]
    api.data["pulls/2"] = pull() | dict(number=2)
    api.data[("PUT", "pulls/2/merge")] = dict(merged=True)
    events = []
    monkeypatch.setattr(m, "freeze", lambda *args: events.append("freeze") or "")

    def handler(path, method, data, pat):
        if method == "PUT":
            events.append(path)
        return NotImplemented

    api.handler = handler
    assert [r[0] for r in m.merge(api)] == ["1", "2"]
    assert events == [
        "freeze",
        "freeze",
        "pulls/1/merge",
        "freeze",
        "freeze",
        "pulls/2/merge",
        "freeze",
    ]


def test_live_head_changed_with_green_new_evidence():
    api = ready()
    new_sha = "b" * 40
    api.data[f"commits/{new_sha}/check-runs?filter=all"] = api.data[
        f"commits/{SHA}/check-runs?filter=all"
    ]
    api.data[f"commits/{new_sha}/statuses"] = api.data[f"commits/{SHA}/statuses"]
    reads = 0

    def handler(path, method, data, pat):
        nonlocal reads
        if path == "pulls/1":
            reads += 1
            if reads == 2:
                api.data[path]["head"]["sha"] = new_sha
        return NotImplemented

    api.handler = handler
    assert m.merge(api)[0] == ("1", "skip", "Head moved")
    assert not any(c[1] == "PUT" for c in api.calls)


@pytest.mark.parametrize("status", [405, 409])
def test_head_race_does_not_block_next_pr(status):
    api = ready()
    api.data["pulls?state=open&base=main"].append(dict(number=2))
    api.data["pulls/2"] = pull() | dict(number=2)
    api.data[("PUT", "pulls/1/merge")] = m.APIError("race", status)
    api.data[("PUT", "pulls/2/merge")] = dict(merged=True)
    assert [r[1] for r in m.merge(api)] == ["skip", "merged"]


def test_transport_timeout_freezes(monkeypatch):
    monkeypatch.setattr(
        m.urllib.request, "urlopen", Mock(side_effect=TimeoutError("timeout"))
    )
    assert "failed" in m.freeze(m.GitHub("owner/repo", "read"))


def test_unknown_mergeability_can_resolve(monkeypatch):
    api = ready()
    monkeypatch.setattr(m.time, "sleep", Mock())
    reads = 0

    def handler(path, method, data, pat):
        nonlocal reads
        reads += 1
        return pull() | dict(mergeable_state="unknown" if reads == 1 else "clean")

    api.handler = handler
    assert m.live_pr(api, 1)["mergeable_state"] == "clean"
    assert reads == 2


ACTIVE_STATUSES = ("queued", "in_progress", "waiting", "pending", "requested")
RUNS = "actions/workflows/auto-release.yml/runs"


@pytest.mark.parametrize("status", [*ACTIVE_STATUSES, "new_status"])
def test_release_status_transition_retains_recent_snapshot(monkeypatch, status):
    api = m.GitHub("owner/repo", "read")

    def response(path):
        if "?status=" in path:
            # The run moves between queried statuses and is absent from both.
            return dict(workflow_runs=[])
        if path == f"{RUNS}?per_page=100&page=1":
            return dict(
                workflow_runs=[dict(id=7, status=status)]
                + [dict(id=i + 100, status="completed") for i in range(99)]
            )
        if path == "actions/runs/7":
            return dict(id=7, status="in_progress")
        if path == "actions/runs/7/jobs?per_page=100&page=1":
            return dict(jobs=[])
        raise AssertionError(path)

    request = Mock(side_effect=response)
    monkeypatch.setattr(api, "request", request)
    assert m.freeze(api) == "Release run 7 detection undetermined"
    assert [c.args[0] for c in request.call_args_list] == [
        f"{RUNS}?per_page=100&page=1",
        *[f"{RUNS}?status={s}&per_page=100&page=1" for s in ACTIVE_STATUSES],
        "actions/runs/7",
        "actions/runs/7/jobs?per_page=100&page=1",
    ]


@pytest.mark.parametrize("status", ACTIVE_STATUSES)
def test_older_active_release_found_without_scanning_unfiltered_history(
    monkeypatch, status
):
    api = m.GitHub("owner/repo", "read")

    def response(path):
        if path == f"{RUNS}?per_page=100&page=1":
            return dict(
                workflow_runs=[dict(id=i + 100, status="completed") for i in range(100)]
            )
        if path == f"{RUNS}?per_page=100&page=2":
            # The active run exists here, but scanning unfiltered history is forbidden.
            pytest.fail("Unfiltered release history must remain bounded to one page")
        if path == f"{RUNS}?status={status}&per_page=100&page=1":
            return dict(
                workflow_runs=[dict(id=i + 200, status=status) for i in range(100)]
            )
        if path == f"{RUNS}?status={status}&per_page=100&page=2":
            return dict(workflow_runs=[dict(id=7, status=status)])
        if "?status=" in path:
            return dict(workflow_runs=[])
        if path.startswith("actions/runs/") and "/jobs" not in path:
            return dict(
                id=int(path.rsplit("/", 1)[1]),
                status="in_progress" if path == "actions/runs/7" else "completed",
            )
        if path == "actions/runs/7/jobs?per_page=100&page=1":
            return dict(jobs=[])
        if path == "commits/main":
            return dict(sha=SHA, commit=dict(message="Fix behavior"))
        raise AssertionError(path)

    request = Mock(side_effect=response)
    monkeypatch.setattr(api, "request", request)
    assert m.freeze(api) == "Release run 7 detection undetermined"
    assert any(
        c.args[0] == f"{RUNS}?status={status}&per_page=100&page=2"
        for c in request.call_args_list
    )


@pytest.mark.parametrize("live_status", ["completed", "in_progress"])
def test_release_union_deduplicates_and_rechecks_live_status(live_status):
    api = ready()
    run = dict(id=7, status="waiting")
    api.data[RUNS] = [run]
    for status in ACTIVE_STATUSES:
        api.data[f"{RUNS}?status={status}"] = [run]
    api.data["actions/runs/7"] = dict(id=7, status=live_status)
    api.data["actions/runs/7/jobs"] = [
        dict(name="detect", status="completed"),
        dict(
            name="release",
            status="queued" if live_status == "completed" else "completed",
            conclusion="skipped",
        ),
    ]
    assert m.freeze(api) == ""
    paths = [c[0] for c in api.calls]
    assert paths.count("actions/runs/7") == 1
    assert paths.count("actions/runs/7/jobs") == (live_status != "completed")


@pytest.mark.parametrize(
    "path",
    [
        RUNS,
        *[f"{RUNS}?status={s}" for s in ACTIVE_STATUSES],
        "actions/runs/7",
        "actions/runs/7/jobs",
    ],
)
def test_release_union_api_errors_fail_closed(path):
    api = ready()
    api.data[RUNS] = [dict(id=7, status="waiting")]
    api.data["actions/runs/7"] = dict(id=7, status="waiting")
    api.data[path] = m.APIError("offline")
    assert m.freeze(api) == "Release freeze evaluation failed: offline"
