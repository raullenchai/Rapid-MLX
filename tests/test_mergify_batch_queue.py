# SPDX-License-Identifier: Apache-2.0
"""Fail-closed contracts for the managed batch merge queue."""

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


def _config() -> dict[str, object]:
    return yaml.safe_load(CONFIG.read_text())


def _rules_by_name(kind: str) -> dict[str, dict[str, object]]:
    return {rule["name"]: rule for rule in _config()[kind]}


def test_queue_batches_four_ready_prs_after_a_bounded_wait():
    config = _config()
    queue = config["merge_queue"]
    rules = _rules_by_name("queue_rules")

    assert queue["mode"] == "serial"
    assert queue["max_parallel_checks"] == 1
    assert queue["skip_intermediate_results"] is False
    assert set(rules) == {"no-mac-batch", "mac-batch"}
    assert rules["no-mac-batch"]["batch_size"] == 4
    assert rules["no-mac-batch"]["batch_max_wait_time"] == "5 min"
    assert rules["mac-batch"]["batch_size"] == 4
    assert rules["mac-batch"]["batch_max_wait_time"] == "15 min"
    assert {rule["checks_timeout"] for rule in rules.values()} == {"90 min"}


def test_queue_revalidates_every_required_check_on_the_combined_batch():
    rules = _rules_by_name("queue_rules")

    for rule in rules.values():
        assert set(rule["queue_conditions"]) >= REQUIRED_CHECKS
        assert HEAD_AUTHORIZATION in rule["queue_conditions"]
        assert set(rule["merge_conditions"]) == REQUIRED_CHECKS
        assert HEAD_AUTHORIZATION not in rule["merge_conditions"]
        assert rule["branch_protection_injection_mode"] == "queue"
        assert rule["queue_branch_prefix"] == "mergify/merge-queue/"


def test_ready_labels_autoqueue_without_enabling_blind_retries():
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
        assert queue_rule["batch_max_failure_resolution_attempts"] == 2


def test_ready_labels_are_mutually_exclusive_in_every_rule():
    for rule in _config()["queue_rules"]:
        conditions = set(rule["queue_conditions"])
        assert {"label = merge-ready", "-label = merge-ready-mac"} <= conditions or {
            "label = merge-ready-mac",
            "-label = merge-ready",
        } <= conditions


def test_release_bumps_cannot_enter_the_general_batch_queue():
    config = _config()
    exclusions = {
        "-label = version-bump",
        "-label = skip-version-bump",
        "-title ~= ^chore: bump version to ",
    }

    for queue_rule in config["queue_rules"]:
        assert exclusions <= set(queue_rule["queue_conditions"])
        assert queue_rule["merge_method"] == "squash"


def test_head_updates_revoke_both_merge_ready_authorizations():
    workflow = yaml.load(
        (ROOT / ".github/workflows/revoke-merge-ready.yml").read_text(),
        Loader=yaml.BaseLoader,
    )

    assert workflow["on"] == {"pull_request_target": {"types": ["synchronize"]}}
    assert workflow["permissions"] == {}

    job = workflow["jobs"]["revoke-merge-ready"]
    assert "merge-ready" in job["if"]
    assert "merge-ready-mac" in job["if"]
    assert job["permissions"] == {"pull-requests": "write"}

    (step,) = job["steps"]
    assert step["uses"].startswith("actions/github-script@")
    script = step["with"]["script"]
    assert '["merge-ready", "merge-ready-mac"]' in script
    assert "github.rest.issues.removeLabel" in script
    assert "checkout" not in script.lower()


def test_ready_authorization_is_bound_to_the_exact_head_commit():
    workflow = yaml.load(
        (ROOT / ".github/workflows/authorize-merge-ready.yml").read_text(),
        Loader=yaml.BaseLoader,
    )

    assert workflow["on"] == {"pull_request_target": {"types": ["labeled"]}}
    assert workflow["permissions"] == {}

    job = workflow["jobs"]["authorize-ready-head"]
    assert "head.repo.full_name == github.repository" in job["if"]
    assert "merge-ready" in job["if"]
    assert "merge-ready-mac" in job["if"]
    assert job["permissions"] == {"statuses": "write"}

    (step,) = job["steps"]
    assert step["uses"].startswith("actions/github-script@")
    script = step["with"]["script"]
    assert "github.rest.repos.createCommitStatus" in script
    assert "sha: context.payload.pull_request.head.sha" in script
    assert 'context: "merge-ready-head"' in script
    assert "present.length === 1" in script
    assert "checkout" not in script.lower()
