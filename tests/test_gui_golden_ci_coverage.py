# SPDX-License-Identifier: Apache-2.0
"""The macOS golden gate must not silently omit named GUI journeys."""

from __future__ import annotations

import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
HARNESS = ROOT / "apps/rapid-mac/scripts/gui-golden-flows.sh"
WORKFLOW = ROOT / ".github/workflows/rapid-mac-ci.yml"
MANIFEST = ROOT / "apps/rapid-mac/Tests/GUIGoldenFlows/journeys.yaml"
SWIFT_TESTS = ROOT / "apps/rapid-mac/Tests/RapidTests"

# `chat-depth` requires all five turns to be simultaneously realised in AX.
# The hosted runner's 1024x681 app window virtualises the oldest messages, so
# that assertion is valid on larger local displays but false by construction in
# CI. This is the one explicit exception; keep it exact: any additional
# omission is accidental. The rc2 state-upgrade journey used to be local-only
# but was gated into the hosted audio shard by #2365.
CI_EXCLUSIONS = {"chat-depth"}


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


def swift_journeys() -> set[str]:
    return {
        str(journey["name"])
        for journey in manifest_journeys()
        if journey["driver"] == "swift"
    }


def swift_suite_titles() -> set[str]:
    """Journeys carried by in-process `swift test` suites.

    A `driver: swift` journey's coverage lives in a Swift Testing suite named
    `Golden journey: <name>`; the title is the machine-checkable link between
    the manifest inventory and the code that honours it.
    """

    # Commented-out code must not satisfy a coverage gate, and a named suite
    # only counts when its file also declares at least one live @Test —
    # source-level approximations of "this journey actually executes".
    # Anything subtler (a @Test that compiles but asserts nothing) is the
    # Swift build's and reviewer's territory, not a regex's.
    titles: set[str] = set()
    for path in SWIFT_TESTS.glob("*.swift"):
        file_titles: set[str] = set()
        has_live_test = False
        for line in path.read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith("//"):
                continue
            if stripped.startswith("@Test"):
                has_live_test = True
            match = re.match(r'@Suite\("Golden journey: ([a-z0-9-]+)"', stripped)
            if match:
                file_titles.add(match.group(1))
        if has_live_test:
            titles.update(file_titles)
    return titles


def test_manifest_is_the_complete_unique_flow_inventory():
    journeys = manifest_journeys()
    names = [str(journey["name"]) for journey in journeys]
    assert len(names) == len(set(names))
    assert set(names) - swift_journeys() == harness_flows()


def test_swift_driver_journeys_have_their_golden_suite():
    assert swift_journeys() == swift_suite_titles()


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
    allowed_drivers = {"ax", "xcuitest", "hybrid", "swift"}
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
    # `driver: swift` journeys ride the build job's `swift test`, which runs
    # on every PR; they must not also occupy a GUI shard step.
    pr_flows = {
        str(journey["name"])
        for journey in manifest_journeys()
        if journey["ci_tier"] == "pr"
    } - swift_journeys()
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


def test_persona_update_isolation_is_applied_to_launch_and_relaunch(tmp_path):
    """Run the real launcher with a tiny owned child instead of a GUI app."""
    source = HARNESS.read_text()
    function = source[
        source.index("launch_persona_app() {") : source.index("\nstart_persona() {")
    ]
    persona = tmp_path / "persona"
    persona.mkdir()
    capture = tmp_path / "update-policy"
    launch = persona / "launch.sh"
    launch.write_text(
        '#!/bin/sh\nprintf "%s" "$RAPIDMLX_NO_UPDATE_CHECK" > "$ENV_CAPTURE"\nsleep 10\n'
    )
    launch.chmod(0o755)
    for mode in ("truncate", "append"):
        for override in (False, True):
            capture.unlink(missing_ok=True)
            script = (
                "begin_launch() { :; }\n"
                + function
                + """
trap 'if [[ -n "${APP_PID:-}" ]]; then kill -- "-$APP_PID" 2>/dev/null || true; wait "$APP_PID" 2>/dev/null || true; fi' EXIT
launch_persona_app "$MODE"
for _ in {1..40}; do [[ -f "$ENV_CAPTURE" ]] && exit 0; sleep 0.05; done
exit 1
"""
            )
            prefix = (
                "PERSONA_ENV=(RAPIDMLX_NO_UPDATE_CHECK=0)\n"
                if override
                else "PERSONA_ENV=()\n"
            )
            result = subprocess.run(
                ["bash", "-euo", "pipefail", "-c", prefix + script],
                env={
                    **os.environ,
                    "PERSONA": str(persona),
                    "OUT": str(tmp_path),
                    "ROOT": str(ROOT / "apps/rapid-mac"),
                    "ENV_CAPTURE": str(capture),
                    "MODE": mode,
                },
                capture_output=True,
                text=True,
                timeout=10,
            )
            assert result.returncode == 0, result.stdout + result.stderr
            assert capture.read_text() == ("0" if override else "1")


def test_updater_journeys_opt_into_their_existing_fetch_path():
    source = HARNESS.read_text()
    for flow in ("update-state", "update-busy", "no-dead-controls"):
        part = source.split(f"flow_{flow.replace('-', '_')}() {{", 1)[1].split(
            "\n}", 1
        )[0]
        assert "RAPIDMLX_NO_UPDATE_CHECK=0" in part
        assert "RAPID_GUI_GOLDEN_MODE=1" in part
        assert (
            "RAPID_GUI_UPDATE_BUSY_FIXTURE=1"
            if flow == "update-busy"
            else "RAPID_GUI_UPDATE_CURRENT_FIXTURE=1"
        ) in part


def test_update_card_geometry_guard_rejects_overlap_and_missing_evidence(tmp_path):
    source = HARNESS.read_text()
    function = source.split("assert_update_does_not_cover_readiness() {", 1)[1].split(
        "\n}\n", 1
    )[0]
    script = (
        'die() { echo "$*" >&2; exit 1; }\n'
        + "assert_update_does_not_cover_readiness() {"
        + function
        + '\n}\nassert_update_does_not_cover_readiness "$TREE"\n'
    )
    card = {"x": 100, "y": 100, "width": 200, "height": 100}
    cases = [
        ({"x": 150, "y": 150, "width": 50, "height": 30}, False),
        ({"x": 150, "y": 70, "width": 50, "height": 30}, True),
        ({"x": 310, "y": 150, "width": 50, "height": 30}, True),
        ({"x": 150, "y": 150, "width": 0, "height": 30}, False),
        (None, False),
    ]
    for action, expected in cases:
        elements = [{"identifier": "UpdateCard", "bounds": card}]
        if action is not None:
            elements.append({"identifier": "Readiness.Action", "bounds": action})
        tree = tmp_path / "tree.json"
        tree.write_text(json.dumps({"data": {"ui_elements": elements}}))
        result = subprocess.run(
            ["bash", "-euo", "pipefail", "-c", script],
            env={**os.environ, "TREE": str(tree)},
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert (result.returncode == 0) is expected, result.stderr


def test_busy_update_journey_requires_unobscured_model_start():
    flow = HARNESS.read_text().split("flow_update_busy() {", 1)[1].split("\n}", 1)[0]
    assert flow.index("assert_update_does_not_cover_readiness") < flow.index(
        "    start_model\n"
    )
    assert 'assert_fake_server_starts "$OUT/fake-events.jsonl" 1 "$FAKE_ALIAS"' in flow
    assert 'wait_identifier UpdateCard "$OUT/update-card-after-start.json"' in flow


# Execute the real Bash evidence boundary without launching a GUI. Keep these
# in the existing registered desktop contract suite so all GUI lanes run them.
def run_shell(tmp_path, body):
    return subprocess.run(
        [
            "/bin/bash",
            "-c",
            'source "$HARNESS"; OUT_ROOT="$EVIDENCE"; mkdir -p "$OUT_ROOT"; cleanup_persona() { :; }; cleanup_operator_server() { :; }; cleanup_telemetry_sink() { :; }; trap finish EXIT; '
            + body,
        ],
        env={
            **os.environ,
            "HARNESS": str(HARNESS),
            "EVIDENCE": str(tmp_path),
            "GUI_FLOWS": "",
        },
        text=True,
        capture_output=True,
    )


def read_journey(tmp_path, name):
    return json.loads((tmp_path / "journeys" / f"{name}.json").read_text())


def test_each_journey_keeps_its_own_cost_and_artifacts(tmp_path):
    result = run_shell(
        tmp_path,
        """
        sample() { begin_launch; sleep 0.06; finish_launch; sleep 0.03; }
        run_journey first sample
        run_journey second sample
    """,
    )
    assert result.returncode == 0, result.stderr
    for name in ("first", "second"):
        record = read_journey(tmp_path, name)
        assert record["flow"] == name
        assert record["status"] == "pass"
        assert record["exit_code"] == 0
        assert record["launch_count"] == 1
        assert record["launch_duration_seconds"] >= 0.06
        assert record["execution_duration_seconds"] >= 0.03
        assert record["duration_seconds"] == pytest.approx(
            record["launch_duration_seconds"] + record["execution_duration_seconds"]
        )
        assert record["artifact_path"] == str(tmp_path)


@pytest.mark.parametrize("code", [1, 7, 130, 143])
def test_failed_or_cancelled_journey_preserves_prior_pass_and_stops(tmp_path, code):
    result = run_shell(
        tmp_path,
        f"""
        good() {{ :; }}
        broken() {{ begin_launch; sleep 0.03; exit {code}; }}
        run_journey first good
        run_journey broken broken
        run_journey never good
    """,
    )
    assert result.returncode == code, result.stderr
    assert read_journey(tmp_path, "first")["status"] == "pass"
    failed = read_journey(tmp_path, "broken")
    assert failed["status"] == ("cancelled" if code in (130, 143) else "fail")
    assert failed["exit_code"] == code
    assert failed["launch_duration_seconds"] >= 0.03
    assert not (tmp_path / "journeys" / "never.json").exists()


def test_relaunches_are_accumulated(tmp_path):
    result = run_shell(
        tmp_path,
        """
        sample() { begin_launch; sleep 0.03; finish_launch; begin_launch; sleep 0.03; finish_launch; }
        run_journey relaunch sample
    """,
    )
    assert result.returncode == 0, result.stderr
    record = read_journey(tmp_path, "relaunch")
    assert record["launch_count"] == 2
    assert record["launch_duration_seconds"] >= 0.06


@pytest.mark.parametrize(
    "helper", ["cleanup_persona", "cleanup_operator_server", "cleanup_telemetry_sink"]
)
@pytest.mark.parametrize("code", [1, 7])
def test_cleanup_failure_cannot_leave_active_journey_green(tmp_path, helper, code):
    result = run_shell(
        tmp_path,
        f"""
        {helper}() {{ return {code}; }}
        normal() {{ :; }}
        run_journey cleanup normal
        run_journey never normal
    """,
    )
    assert result.returncode == code, result.stderr
    record = read_journey(tmp_path, "cleanup")
    assert record["status"] == "fail"
    assert record["exit_code"] == code
    assert not (tmp_path / "journeys/never.json").exists()


def test_cleanup_time_is_part_of_execution_cost(tmp_path):
    result = run_shell(
        tmp_path,
        """
        cleanup_persona() { sleep 0.03; }
        normal() { :; }
        run_journey cleanup normal
    """,
    )
    assert result.returncode == 0, result.stderr
    record = read_journey(tmp_path, "cleanup")
    assert record["launch_duration_seconds"] == 0
    assert record["execution_duration_seconds"] >= 0.03


def test_failed_assertion_aborts_the_function_before_success(tmp_path):
    result = run_shell(
        tmp_path,
        """
        broken() { false; touch "$OUT_ROOT/false-green"; }
        run_journey assertion broken
    """,
    )
    assert result.returncode == 1, result.stderr
    assert not (tmp_path / "false-green").exists()
    assert read_journey(tmp_path, "assertion")["status"] == "fail"


def test_early_zero_exit_does_not_report_success(tmp_path):
    result = run_shell(
        tmp_path, "premature() { exit 0; }; run_journey premature premature"
    )
    assert result.returncode == 1, result.stderr
    assert read_journey(tmp_path, "premature")["exit_code"] == 1


@pytest.mark.parametrize("signal,code", [("INT", 130), ("TERM", 143)])
def test_signal_records_interrupted_launch(tmp_path, signal, code):
    result = run_shell(
        tmp_path,
        f"""
        trap 'exit {code}' {signal}
        interrupted() {{ begin_launch; kill -s {signal} $$; }}
        run_journey interrupted interrupted
    """,
    )
    assert result.returncode == code, result.stderr
    record = read_journey(tmp_path, "interrupted")
    assert record["status"] == "cancelled"
    assert record["launch_count"] == 1


def test_real_all_dispatcher_emits_independent_records(tmp_path):
    # Run the actual dispatch table with tiny journey bodies; assertions and
    # evidence handling remain real. This catches a forgotten wrapper in all.
    dispatcher = HARNESS.read_text().rsplit('case "$FLOW" in', 1)[1].split("esac", 1)[0]
    names = re.findall(r"^    ([a-z][a-z0-9-]+)\)", dispatcher, re.M)
    functions = set(re.findall(r"\bflow_[a-z0-9_]+", dispatcher))
    definitions = "\n".join(f"{name}() {{ :; }}" for name in functions)
    result = run_shell(
        tmp_path, definitions + '\nFLOW=all\ncase "$FLOW" in' + dispatcher + "esac"
    )
    assert result.returncode == 0, result.stderr
    all_block = dispatcher.split("    all)", 1)[1].split(";;", 1)[0]
    expected = set(re.findall(r"\bflow_([a-z0-9_]+)", all_block))
    assert {p.stem for p in (tmp_path / "journeys").glob("*.json")} == {
        name.replace("_", "-") for name in expected
    }
    for name in set(names) - {"all"}:
        assert f"{name}) run_journey {name} flow_" in dispatcher
