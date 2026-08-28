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


def _config() -> dict[str, object]:
    return yaml.safe_load(CONFIG.read_text())


def test_queue_batches_four_ready_prs_after_a_bounded_wait():
    config = _config()
    queue = config["merge_queue"]
    (rule,) = config["queue_rules"]

    assert queue["mode"] == "serial"
    assert queue["max_parallel_checks"] == 1
    assert queue["skip_intermediate_results"] is False
    assert rule["batch_size"] == 4
    assert rule["batch_max_wait_time"] == "15 min"
    assert rule["checks_timeout"] == "90 min"


def test_queue_revalidates_every_required_check_on_the_combined_batch():
    (rule,) = _config()["queue_rules"]

    assert set(rule["queue_conditions"]) >= REQUIRED_CHECKS
    assert set(rule["merge_conditions"]) == REQUIRED_CHECKS
    assert rule["branch_protection_injection_mode"] == "queue"
    assert rule["queue_branch_prefix"] == "mergify/merge-queue/"


def test_ready_label_queues_without_enabling_blind_retries():
    config = _config()
    (queue_rule,) = config["queue_rules"]
    (enqueue_rule,) = config["pull_request_rules"]

    assert enqueue_rule["actions"] == {"queue": {"name": "main-batch"}}
    assert "label = merge-ready" in enqueue_rule["conditions"]
    assert "-from-fork" in enqueue_rule["conditions"]
    assert "-from-fork" in queue_rule["queue_conditions"]
    assert queue_rule["max_checks_retries"] == 0
    assert queue_rule["batch_max_failure_resolution_attempts"] == 2


def test_release_bumps_cannot_enter_the_general_batch_queue():
    config = _config()
    (queue_rule,) = config["queue_rules"]
    (enqueue_rule,) = config["pull_request_rules"]
    exclusions = {
        "-label = version-bump",
        "-label = skip-version-bump",
        "-title ~= ^chore: bump version to ",
    }

    assert exclusions <= set(queue_rule["queue_conditions"])
    assert exclusions <= set(enqueue_rule["conditions"])
    assert queue_rule["merge_method"] == "squash"
