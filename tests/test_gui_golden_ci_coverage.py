# SPDX-License-Identifier: Apache-2.0
"""The macOS golden gate must not silently omit named GUI journeys."""

from __future__ import annotations

import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
HARNESS = ROOT / "apps/rapid-mac/scripts/gui-golden-flows.sh"
WORKFLOW = ROOT / ".github/workflows/rapid-mac-ci.yml"
MANIFEST = ROOT / "apps/rapid-mac/Tests/GUIGoldenFlows/journeys.yaml"

# `chat-depth` requires all five turns to be simultaneously realised in AX.
# The hosted runner's 1024x681 app window virtualises the oldest messages, so
# that assertion is valid on larger local displays but false by construction in
# CI. The rc2 state-upgrade journey was added during the release window and is
# verified locally on Studio; hosted dictation-shard gating is tracked in
# #2365. Keep both exceptions exact: any additional omission is accidental.
CI_EXCLUSIONS = {"chat-depth", "dictation-rc2-upgrade"}


def harness_flows() -> set[str]:
    source = HARNESS.read_text()
    dispatcher = source.rsplit('case "$FLOW" in', 1)[1].split("esac", 1)[0]
    return set(re.findall(r"^    ([a-z][a-z0-9-]+)\)", dispatcher, re.MULTILINE)) - {
        "all"
    }


def workflow_flows() -> set[str]:
    steps = yaml.safe_load(WORKFLOW.read_text())["jobs"]["gui-golden-flows"]["steps"]
    return {
        match.group(1)
        for step in steps
        if (
            match := re.search(
                r"gui-golden-flows\.sh --flow ([a-z0-9-]+)", step.get("run", "")
            )
        )
    }


def workflow_steps() -> list[dict[str, object]]:
    return yaml.safe_load(WORKFLOW.read_text())["jobs"]["gui-golden-flows"]["steps"]


def manifest_journeys() -> list[dict[str, object]]:
    payload = yaml.safe_load(MANIFEST.read_text())
    assert payload["version"] == 1
    return payload["journeys"]


def diagnostic_flows() -> set[str]:
    workflow = WORKFLOW.read_text()
    loop = workflow.split("for flow in ", 1)[1].split("; do", 1)[0]
    return set(loop.split())


def baseline_flows() -> set[str]:
    source = HARNESS.read_text()
    owners: set[str] = set()
    for match in re.finditer(
        r"^flow_([a-z0-9_]+)\(\) \{\n(.*?)(?=^\})", source, re.MULTILINE | re.DOTALL
    ):
        if re.search(r"\bbaseline\s", match.group(2)):
            owners.add(match.group(1).replace("_", "-"))
    return owners


def test_every_named_flow_is_gated_or_explicitly_excluded():
    named = harness_flows()
    gated = workflow_flows()
    assert named - gated == CI_EXCLUSIONS
    assert not gated - named


def test_manifest_is_the_complete_unique_flow_inventory():
    journeys = manifest_journeys()
    names = [str(journey["name"]) for journey in journeys]
    assert len(names) == len(set(names))
    assert set(names) == harness_flows()


def test_manifest_fields_are_valid_and_fail_closed():
    allowed_groups = {
        "chat",
        "audio",
        "models",
        "onboarding-settings",
        "images",
        "app-lifecycle",
    }
    allowed_risks = {"low", "medium", "high"}
    allowed_drivers = {"ax", "xcuitest", "hybrid"}
    allowed_tiers = {"pr", "local"}
    allowed_fixtures = {
        "audio-models",
        "cached-model",
        "campaign",
        "crash-once",
        "delayed-transcription",
        "document",
        "native-file-drag",
        "fake-sidecar",
        "generated-images",
        "isolated-home",
        "large-window",
        "loopback-telemetry-sink",
        "low-memory",
        "mixed-capability-catalog",
        "resident-model",
        "slow-download",
        "slow-stream",
        "two-images",
        "update-busy",
        "update-state",
        "zh-Hans",
    }
    expected_keys = {
        "name",
        "group",
        "risk",
        "driver",
        "ci_tier",
        "fixtures",
        "source_paths",
        "owns_baseline",
    }

    for journey in manifest_journeys():
        assert set(journey) == expected_keys
        assert isinstance(journey["name"], str) and journey["name"]
        assert journey["group"] in allowed_groups
        assert journey["risk"] in allowed_risks
        assert journey["driver"] in allowed_drivers
        assert journey["ci_tier"] in allowed_tiers
        assert isinstance(journey["fixtures"], list) and journey["fixtures"]
        assert all(
            isinstance(fixture, str) and fixture in allowed_fixtures
            for fixture in journey["fixtures"]
        )
        assert isinstance(journey["source_paths"], list) and journey["source_paths"]
        assert all(
            isinstance(path, str) and path.startswith("apps/rapid-mac/")
            for path in journey["source_paths"]
        )
        assert all((ROOT / path).exists() for path in journey["source_paths"])
        assert isinstance(journey["owns_baseline"], bool)


def test_manifest_ci_tiers_match_the_workflow_contract():
    pr_flows = {
        str(journey["name"])
        for journey in manifest_journeys()
        if journey["ci_tier"] == "pr"
    }
    local_flows = {
        str(journey["name"])
        for journey in manifest_journeys()
        if journey["ci_tier"] == "local"
    }
    assert pr_flows == workflow_flows()
    assert local_flows == CI_EXCLUSIONS


def test_manifest_baseline_ownership_matches_harness_usage():
    declared = {
        str(journey["name"])
        for journey in manifest_journeys()
        if journey["owns_baseline"]
    }
    assert declared == baseline_flows()


def test_result_evidence_records_timing_and_artifact_location():
    source = HARNESS.read_text()
    writer = source.split("write_result() {", 1)[1].split("\n}", 1)[0]
    finish = source.split("finish() {", 1)[1].split("\n}", 1)[0]
    dispatch_tail = source.rsplit('case "$FLOW" in', 1)[1].split(
        'log "PASS — $FLOW"', 1
    )[0]

    assert "started_at: $started_at" in writer
    assert "duration_seconds: $duration_seconds" in writer
    assert "artifact_path: $artifact_path" in writer
    assert '--argjson exit_code "$exit_code"' in writer
    assert 'write_result fail "$status"' in finish
    assert "write_result pass 0" in dispatch_tail


def test_early_precondition_failure_writes_typed_result_evidence(tmp_path: Path):
    output = tmp_path / "not-created-yet"
    missing_app = tmp_path / "missing.app"
    result = subprocess.run(
        ["bash", str(HARNESS), "--flow", "fresh-install"],
        env={
            **os.environ,
            "HOME": str(tmp_path),
            "RAPID_GUI_GOLDEN_OUT": str(output),
            "RAPID_GUI_SOURCE_APP": str(missing_app),
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    payload = json.loads((output / "result.json").read_text())
    assert payload["status"] == "fail"
    assert payload["flow"] == "fresh-install"
    assert payload["app"] == str(missing_app)
    assert payload["exit_code"] == result.returncode
    assert isinstance(payload["duration_seconds"], int)
    assert payload["duration_seconds"] >= 0
    assert payload["artifact_path"] == str(output)
    datetime.strptime(payload["started_at"], "%Y-%m-%dT%H:%M:%SZ")


def test_help_does_not_require_harness_runtime_dependencies():
    result = subprocess.run(
        ["bash", str(HARNESS), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Usage: gui-golden-flows.sh" in result.stdout


def test_interrupt_and_termination_flow_through_exit_evidence_handler():
    source = HARNESS.read_text()
    assert "trap finish EXIT" in source
    assert "trap 'exit 130' INT" in source
    assert "trap 'exit 143' TERM" in source


def test_failure_diagnostic_regenerates_every_ci_baseline_and_nothing_else():
    assert diagnostic_flows() == workflow_flows() & baseline_flows()


def test_failure_diagnostic_skips_regeneration_for_semantic_failures():
    steps = workflow_steps()
    (diagnostic,) = [
        step
        for step in steps
        if step.get("name") == "Regenerate baselines on this runner (diagnostic)"
    ]
    run = str(diagnostic.get("run", ""))
    assert 'for result in "$GOLDEN_ROOT"/*/result.json' in run
    assert 'find "$(dirname "$result")" -name \'*.observed.txt\'' in run
    assert "No structural baseline mismatch" in run


def test_all_named_flows_run_before_one_blocking_verdict():
    steps = workflow_steps()
    flow_steps = [
        step for step in steps if str(step.get("name", "")).startswith("Golden flow:")
    ]
    assert len(flow_steps) == len(workflow_flows())
    assert all(step.get("continue-on-error") is True for step in flow_steps)

    verdicts = [
        step for step in steps if step.get("name") == "Require every named golden flow"
    ]
    assert len(verdicts) == 1
    verdict = verdicts[0]
    assert verdict.get("if") == "always()"
    assert 'expected = int(os.environ["EXPECTED_FLOW_COUNT"])' in str(
        verdict.get("run", "")
    )


def test_golden_job_builds_the_release_ui_surface():
    """Release baselines cannot be compared against Debug-only controls."""
    workflow = yaml.safe_load(WORKFLOW.read_text())
    build_steps = [
        step
        for step in workflow["jobs"]["gui-app-build"]["steps"]
        if step.get("name") == "Build release-shaped GUI app"
    ]
    assert len(build_steps) == 1
    assert build_steps[0].get("env", {}).get("RAPID_BUILD_CONFIG") == "release"
    assert build_steps[0].get("env", {}).get("SKIP_SIDECAR") == "1"
    assert "gui-app-build" in workflow["jobs"]["gui-golden-flows"]["needs"]
