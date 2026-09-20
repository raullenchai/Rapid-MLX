# SPDX-License-Identifier: Apache-2.0
"""Fail-closed contracts for the managed singleton merge queue."""

import json
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / ".mergify.yml"
REQUIRED_CHECKS = {
    "check-success = @github-actions/tests",
    "check-success = @github-actions/desktop-tests",
    "check-success = @github-actions/version-bump-guard",
}
HEAD_AUTHORIZATION = "check-success = merge-ready-head"
LANE_CHECKS = {
    "no-mac-batch": "check-success = @github-actions/merge-lane-no-mac",
    "mac-batch": "check-success = @github-actions/merge-lane-mac",
}


def _config() -> dict[str, object]:
    return yaml.safe_load(CONFIG.read_text())


def _rules_by_name(kind: str) -> dict[str, dict[str, object]]:
    return {rule["name"]: rule for rule in _config()[kind]}


def test_queue_runs_single_ready_prs_without_batch_features_or_fill_waits():
    config = _config()
    queue = config["merge_queue"]
    rules = _rules_by_name("queue_rules")

    assert queue["mode"] == "serial"
    assert "max_parallel_checks" not in queue
    assert queue["skip_intermediate_results"] is False
    assert set(rules) == {"no-mac-batch", "mac-batch"}
    for rule in rules.values():
        assert "batch_size" not in rule
        assert "batch_max_wait_time" not in rule
        assert "batch_max_failure_resolution_attempts" not in rule
    assert {rule["checks_timeout"] for rule in rules.values()} == {"90 min"}


def test_queue_avoids_subscription_gated_batch_and_scope_features():
    config = _config()

    assert "scopes" not in config
    assert config["merge_queue"]["mode"] == "serial"


def test_queue_revalidates_every_required_check_on_the_candidate():
    rules = _rules_by_name("queue_rules")

    for name, rule in rules.items():
        assert set(rule["queue_conditions"]) >= REQUIRED_CHECKS
        assert HEAD_AUTHORIZATION in rule["queue_conditions"]
        assert LANE_CHECKS[name] in rule["queue_conditions"]
        assert not ({*LANE_CHECKS.values()} - {LANE_CHECKS[name]}) & set(
            rule["queue_conditions"]
        )
        assert set(rule["merge_conditions"]) == REQUIRED_CHECKS
        assert HEAD_AUTHORIZATION not in rule["merge_conditions"]
        assert not set(LANE_CHECKS.values()) & set(rule["merge_conditions"])
        assert rule["branch_protection_injection_mode"] == "queue"
        assert rule["queue_branch_prefix"] == "mergify/merge-queue/"


def test_ready_labels_autoqueue_without_unsupported_recovery_rules():
    config = _config()
    queues = _rules_by_name("queue_rules")
    auto_merge = config["merge_protections_settings"]["auto_merge_conditions"]

    assert auto_merge == [{"or": ["label = merge-ready", "label = merge-ready-mac"]}]
    assert "pull_request_rules" not in config

    expected_labels = {
        "no-mac-batch": {"label = merge-ready", "-label = merge-ready-mac"},
        "mac-batch": {"label = merge-ready-mac", "-label = merge-ready"},
    }
    for name, queue_rule in queues.items():
        assert expected_labels[name] <= set(queue_rule["queue_conditions"])
        assert "-from-fork" in queue_rule["queue_conditions"]
        assert queue_rule["max_checks_retries"] == 0


def test_ready_labels_are_mutually_exclusive_in_every_rule():
    for rule in _config()["queue_rules"]:
        conditions = set(rule["queue_conditions"])
        assert {"label = merge-ready", "-label = merge-ready-mac"} <= conditions or {
            "label = merge-ready-mac",
            "-label = merge-ready",
        } <= conditions


def test_release_bumps_cannot_enter_the_general_merge_queue():
    config = _config()
    exclusions = {
        "-label = version-bump",
        "-label = skip-version-bump",
        "-title ~= ^chore: bump version to ",
    }

    for queue_rule in config["queue_rules"]:
        assert exclusions <= set(queue_rule["queue_conditions"])
        assert queue_rule["merge_method"] == "squash"


def test_head_updates_rely_on_sha_bound_authorization_without_label_mutation():
    assert not (ROOT / ".github/workflows/revoke-merge-ready.yml").exists()


def test_ready_authorization_is_bound_to_the_exact_head_commit():
    workflow = yaml.load(
        (ROOT / ".github/workflows/authorize-merge-ready.yml").read_text(),
        Loader=yaml.BaseLoader,
    )

    assert workflow["on"] == {
        "pull_request_target": {"types": ["labeled", "synchronize", "unlabeled"]}
    }
    assert "concurrency" not in workflow
    assert workflow["permissions"] == {}

    job = workflow["jobs"]["authorize-ready-head"]
    assert "head.repo.full_name == github.repository" in job["if"]
    assert "merge-ready" in job["if"]
    assert "merge-ready-mac" in job["if"]
    assert "github.event.action == 'labeled'" in job["if"]
    assert "concurrency" not in job
    assert job["permissions"] == {
        "issues": "write",
        "pull-requests": "read",
        "statuses": "write",
    }

    (step,) = job["steps"]
    assert step["uses"].startswith("actions/github-script@")
    script = step["with"]["script"]
    assert "github.rest.repos.createCommitStatus" in script
    assert "headSha = context.payload.pull_request.head.sha" in script
    assert "sha: headSha" in script
    assert 'context: "merge-ready-head"' in script
    assert "present.length === 1" in script
    assert "GITHUB_RUN_ATTEMPT" in script
    assert "github.rest.pulls.get" in script
    assert "livePull.head.sha === context.payload.pull_request.head.sha" in script
    assert "github.paginate" in script
    assert "github.rest.issues.deleteComment" in script
    assert "merge-requeue" not in script
    assert "checkout" not in script.lower()


def _run_authorization_script(
    *,
    labels: list[str],
    run_attempt: int = 1,
    event_label: str = "merge-ready-mac",
    live_head: str = "head-sha",
    fail_status_call: int | None = None,
    fail_get: bool = False,
    fail_comments: bool = False,
    comments: list[dict[str, object]] | None = None,
    cleanup_live_head: str = "head-sha",
) -> dict[str, object]:
    """Execute the exact github-script body against deterministic API mocks."""

    workflow = yaml.load(
        (ROOT / ".github/workflows/authorize-merge-ready.yml").read_text(),
        Loader=yaml.BaseLoader,
    )
    script = workflow["jobs"]["authorize-ready-head"]["steps"][0]["with"]["script"]
    scenario = json.dumps(
        {
            "labels": labels,
            "runAttempt": run_attempt,
            "eventLabel": event_label,
            "liveHead": live_head,
            "failStatusCall": fail_status_call,
            "failGet": fail_get,
            "failComments": fail_comments,
            "comments": comments or [],
            "cleanupLiveHead": cleanup_live_head,
        }
    )
    harness = f"""
const scenario = {scenario};
const calls = [];
const statusArgs = [];
let statusCalls = 0;
let pullCalls = 0;
process.env.GITHUB_RUN_ATTEMPT = String(scenario.runAttempt);
const context = {{
  repo: {{ owner: "owner", repo: "repo" }},
  issue: {{ number: 42 }},
  serverUrl: "https://github.example",
  payload: {{
    label: {{ name: scenario.eventLabel }},
    pull_request: {{ head: {{ sha: "head-sha" }} }},
  }},
}};
const github = {{
  paginate: async (method, args) => method(args).then((response) => response.data),
  rest: {{
    pulls: {{ get: async () => {{
      pullCalls += 1;
      calls.push(["get"]);
      if (scenario.failGet) throw new Error("get failure");
      return {{ data: {{
        head: {{ sha: pullCalls === 1 ? scenario.liveHead : scenario.cleanupLiveHead }},
        labels: scenario.labels.map((name) => ({{ name }})),
      }} }};
    }} }},
    repos: {{ createCommitStatus: async (args) => {{
      statusCalls += 1;
      statusArgs.push(args);
      calls.push(["status", args.state]);
      if (scenario.failStatusCall === statusCalls) throw new Error("status failure");
    }} }},
    issues: {{
      listComments: async () => {{
        calls.push(["comments"]);
        if (scenario.failComments) throw new Error("comments failure");
        return {{ data: scenario.comments }};
      }},
      deleteComment: async (args) => calls.push(["delete", args.comment_id]),
    }},
  }},
}};
const core = {{
  setFailed: (message) => calls.push(["failed", message]),
  warning: (message) => calls.push(["warning", message]),
}};
(async () => {{
  try {{
    await (async () => {{
{script}
    }})();
  }} catch (error) {{
    calls.push(["threw", error.message]);
  }}
  process.stdout.write(JSON.stringify({{ calls, statusArgs }}));
}})();
"""
    completed = subprocess.run(
        ["node", "-e", harness],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def test_initial_authorization_publishes_success_for_the_exact_head():
    result = _run_authorization_script(labels=["merge-ready-mac"])

    assert result["calls"] == [
        ["status", "pending"],
        ["get"],
        ["status", "success"],
        ["comments"],
    ]


def test_fresh_authorization_removes_the_bot_owned_stale_notice():
    result = _run_authorization_script(
        labels=["merge-ready-mac"],
        comments=[
            {
                "id": 7,
                "body": "<!-- merge-ready-stale-head:head-sha --> old",
                "user": {"login": "github-actions[bot]"},
            },
            {
                "id": 8,
                "body": "<!-- merge-ready-stale-head --> human note",
                "user": {"login": "maintainer"},
            },
        ],
    )

    assert result["calls"] == [
        ["status", "pending"],
        ["get"],
        ["status", "success"],
        ["comments"],
        ["get"],
        ["delete", 7],
    ]


def test_comment_cleanup_failure_does_not_revoke_exact_head_authorization():
    result = _run_authorization_script(labels=["merge-ready-mac"], fail_comments=True)

    assert result["calls"] == [
        ["status", "pending"],
        ["get"],
        ["status", "success"],
        ["comments"],
        ["warning", "Could not remove stale merge-ready notice: comments failure"],
    ]


def test_old_authorization_does_not_delete_a_new_heads_notice():
    result = _run_authorization_script(
        labels=["merge-ready-mac"],
        comments=[
            {
                "id": 7,
                "body": "<!-- merge-ready-stale-head:new-head --> newer",
                "user": {"login": "github-actions[bot]"},
            }
        ],
    )

    assert result["calls"] == [
        ["status", "pending"],
        ["get"],
        ["status", "success"],
        ["comments"],
    ]

    pushed_during_cleanup = _run_authorization_script(
        labels=["merge-ready-mac"],
        cleanup_live_head="new-head",
        comments=[
            {
                "id": 7,
                "body": "<!-- merge-ready-stale-head:head-sha --> old",
                "user": {"login": "github-actions[bot]"},
            }
        ],
    )
    assert pushed_during_cleanup["calls"] == [
        ["status", "pending"],
        ["get"],
        ["status", "success"],
        ["comments"],
        ["get"],
    ]


def test_head_update_notice_is_actionable_without_mutating_authorization():
    workflow = yaml.load(
        (ROOT / ".github/workflows/authorize-merge-ready.yml").read_text(),
        Loader=yaml.BaseLoader,
    )
    job = workflow["jobs"]["notify-stale-ready-head"]

    assert "github.event.action == 'synchronize'" in job["if"]
    assert "github.event.action == 'unlabeled'" in job["if"]
    assert "head.repo.full_name == github.repository" in job["if"]
    assert job["permissions"] == {
        "issues": "write",
        "pull-requests": "read",
        "statuses": "read",
    }
    assert job["concurrency"] == {
        "group": "merge-ready-stale-notice-${{ github.event.pull_request.number }}",
        "cancel-in-progress": "false",
    }
    (step,) = job["steps"]
    script = step["with"]["script"]
    assert "livePull.head.sha !== eventHead" in script
    assert 'status.context === "merge-ready-head"' in script
    assert 'latestAuthorization?.state === "success"' in script
    assert "re-apply exactly one" in script
    assert "github.rest.issues.updateComment" in script
    assert "github.rest.issues.createComment" in script
    assert 'comment.user?.login === "github-actions[bot]"' in script
    assert "github.rest.repos.createCommitStatus" not in script
    assert "checkout" not in script.lower()


def _run_head_update_notice(
    *,
    live_head: str = "head-sha",
    refreshed_live_head: str | None = None,
    labels: list[str] | None = None,
    refreshed_labels: list[str] | None = None,
    statuses: list[dict[str, str]] | None = None,
    refreshed_statuses: list[dict[str, str]] | None = None,
    post_statuses: list[dict[str, str]] | None = None,
    comments: list[dict[str, object]] | None = None,
    action: str = "synchronize",
    fail_delete: bool = False,
) -> list[list[object]]:
    workflow = yaml.load(
        (ROOT / ".github/workflows/authorize-merge-ready.yml").read_text(),
        Loader=yaml.BaseLoader,
    )
    script = workflow["jobs"]["notify-stale-ready-head"]["steps"][0]["with"]["script"]
    scenario = json.dumps(
        {
            "liveHead": live_head,
            "refreshedLiveHead": refreshed_live_head,
            "labels": labels if labels is not None else ["merge-ready-mac"],
            "refreshedLabels": refreshed_labels,
            "statuses": statuses or [],
            "refreshedStatuses": refreshed_statuses,
            "postStatuses": post_statuses,
            "comments": comments or [],
            "action": action,
            "failDelete": fail_delete,
        }
    )
    harness = f"""
const scenario = {scenario};
const calls = [];
let statusCalls = 0;
let pullCalls = 0;
const context = {{
  repo: {{ owner: "owner", repo: "repo" }},
  issue: {{ number: 42 }},
  payload: {{
    action: scenario.action,
    pull_request: {{ head: {{ sha: "head-sha" }} }},
  }},
}};
const github = {{
  paginate: async (method, args) => method(args).then((response) => response.data),
  rest: {{
    pulls: {{ get: async () => {{
      pullCalls += 1;
      calls.push(["get"]);
      return {{ data: {{
        head: {{
          sha: pullCalls === 1 || scenario.refreshedLiveHead === null
            ? scenario.liveHead
            : scenario.refreshedLiveHead,
        }},
        labels: (
          pullCalls === 1 || scenario.refreshedLabels === null
            ? scenario.labels
            : scenario.refreshedLabels
        ).map((name) => ({{ name }})),
      }} }};
    }} }},
    repos: {{ listCommitStatusesForRef: async () => {{
      statusCalls += 1;
      calls.push(["statuses"]);
      return {{
        data: statusCalls === 1
          ? scenario.statuses
          : statusCalls === 2 || scenario.postStatuses === null
            ? (scenario.refreshedStatuses ?? scenario.statuses)
            : scenario.postStatuses,
      }};
    }} }},
    issues: {{
      listComments: async () => {{
        calls.push(["comments"]);
        return {{ data: scenario.comments }};
      }},
      createComment: async (args) => {{
        calls.push(["create", args.body]);
        return {{ data: {{ id: 99, body: args.body }} }};
      }},
      updateComment: async (args) => {{
        calls.push(["update", args.body]);
        return {{ data: {{ id: args.comment_id, body: args.body }} }};
      }},
      deleteComment: async (args) => {{
        calls.push(["delete", args.comment_id]);
        if (scenario.failDelete) throw new Error("delete failure");
      }},
    }},
  }},
}};
const core = {{ warning: (message) => calls.push(["warning", message]) }};
(async () => {{
  await (async () => {{
{script}
  }})();
  process.stdout.write(JSON.stringify(calls));
}})();
"""
    completed = subprocess.run(
        ["node", "-e", harness],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def test_head_update_notice_creates_or_updates_one_actionable_comment():
    created = _run_head_update_notice()
    assert [call[0] for call in created] == [
        "get",
        "statuses",
        "comments",
        "statuses",
        "get",
        "create",
        "statuses",
        "get",
    ]
    created_comment = next(call for call in created if call[0] == "create")
    assert "merge-ready-stale-head:head-sha" in created_comment[1]
    assert "remove and re-apply" in created_comment[1]
    assert "head-sha" in created_comment[1]

    updated = _run_head_update_notice(
        comments=[
            {
                "id": 7,
                "body": "<!-- merge-ready-stale-head:old-head --> old",
                "user": {"type": "Bot", "login": "github-actions[bot]"},
            }
        ]
    )
    assert [call[0] for call in updated] == [
        "get",
        "statuses",
        "comments",
        "statuses",
        "get",
        "update",
        "statuses",
        "get",
    ]


def test_head_update_notice_skips_newer_heads_and_fresh_authorization():
    assert _run_head_update_notice(live_head="newer-head") == [["get"]]
    assert _run_head_update_notice(
        statuses=[{"context": "merge-ready-head", "state": "success"}]
    ) == [["get"], ["statuses"], ["comments"]]
    assert _run_head_update_notice(
        refreshed_statuses=[{"context": "merge-ready-head", "state": "success"}]
    ) == [["get"], ["statuses"], ["comments"], ["statuses"]]
    assert _run_head_update_notice(refreshed_live_head="newer-head") == [
        ["get"],
        ["statuses"],
        ["comments"],
        ["statuses"],
        ["get"],
    ]
    assert _run_head_update_notice(
        refreshed_labels=[],
        comments=[
            {
                "id": 7,
                "body": "<!-- merge-ready-stale-head:old-head --> old",
                "user": {"login": "github-actions[bot]"},
            }
        ],
    ) == [
        ["get"],
        ["statuses"],
        ["comments"],
        ["statuses"],
        ["get"],
        ["delete", 7],
    ]


def test_ready_label_removal_clears_an_existing_stale_notice():
    calls = _run_head_update_notice(
        action="unlabeled",
        labels=[],
        comments=[
            {
                "id": 7,
                "body": "<!-- merge-ready-stale-head:old-head --> old",
                "user": {"login": "github-actions[bot]"},
            }
        ],
    )

    assert calls == [["get"], ["comments"], ["get"], ["delete", 7]]

    newer_head_calls = _run_head_update_notice(
        action="unlabeled",
        live_head="newer-head",
        labels=[],
        comments=[
            {
                "id": 7,
                "body": "<!-- merge-ready-stale-head:old-head --> old",
                "user": {"login": "github-actions[bot]"},
            }
        ],
    )
    assert newer_head_calls == [["get"], ["comments"], ["get"], ["delete", 7]]


def test_fresh_authorization_self_heals_a_leftover_notice():
    comments = [
        {
            "id": 7,
            "body": "<!-- merge-ready-stale-head:head-sha --> old",
            "user": {"login": "github-actions[bot]"},
        }
    ]
    statuses = [{"context": "merge-ready-head", "state": "success"}]

    assert _run_head_update_notice(statuses=statuses, comments=comments) == [
        ["get"],
        ["statuses"],
        ["comments"],
        ["delete", 7],
    ]
    assert _run_head_update_notice(
        statuses=statuses,
        comments=comments,
        fail_delete=True,
    ) == [
        ["get"],
        ["statuses"],
        ["comments"],
        ["delete", 7],
        ["warning", "Could not remove stale merge-ready notice: delete failure"],
    ]


def test_notice_post_write_recheck_closes_authorization_race():
    calls = _run_head_update_notice(
        post_statuses=[{"context": "merge-ready-head", "state": "success"}]
    )

    assert [call[0] for call in calls] == [
        "get",
        "statuses",
        "comments",
        "statuses",
        "get",
        "create",
        "statuses",
        "get",
        "delete",
    ]
    assert calls[-1] == ["delete", 99]


def test_status_or_live_pull_failure_remains_fail_closed():
    assert _run_authorization_script(labels=["merge-ready-mac"], fail_status_call=1)[
        "calls"
    ] == [
        ["status", "pending"],
        ["threw", "status failure"],
    ]
    assert _run_authorization_script(labels=["merge-ready-mac"], fail_get=True)[
        "calls"
    ] == [
        ["status", "pending"],
        ["get"],
        ["threw", "get failure"],
    ]


def test_historical_authorization_rerun_cannot_replay_the_label_event():
    result = _run_authorization_script(labels=["merge-ready-mac"], run_attempt=2)

    assert result["calls"] == [
        [
            "failed",
            "A merge-ready authorization event cannot be replayed; remove and re-apply the ready label.",
        ]
    ]


def test_stale_head_or_double_ready_labels_fail_authorization():
    for labels, live_head in (
        (["merge-ready-mac"], "new-head"),
        (["merge-ready", "merge-ready-mac"], "head-sha"),
    ):
        result = _run_authorization_script(labels=labels, live_head=live_head)
        assert result["calls"] == [
            ["status", "pending"],
            ["get"],
            ["status", "failure"],
            ["failed", "Apply exactly one merge-ready label"],
        ]


def test_operations_guide_uses_provider_supported_terminal_requeue():
    docs = (ROOT / "docs/engineering/operations/path-aware-merge-queue.md").read_text()

    assert "@mergifyio queue no-mac-batch" in docs
    assert "@mergifyio queue mac-batch" in docs
    assert "does not bypass `queue_conditions`" in docs
    assert "merge-requeue-trigger" not in docs
    assert "merge-requeue-required" not in docs
