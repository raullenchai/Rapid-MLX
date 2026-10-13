from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = ROOT / ".github/workflows/rapid-mac-ci.yml"
PREFLIGHT = ROOT / "scripts/verify_macos_ci_runner.sh"
MAC_JOBS = ("build", "gui-app-build", "gui-golden-flows")
RUNNER_EXPRESSION = (
    "${{ vars.MACOS_RUNNER == 'macos-26' && 'macos-26' || 'manzanita-standard' }}"
)
WATCHDOG_EXPRESSION = "${{ vars.MACOS_RUNNER == 'macos-26' && '180' || '90' }}"


def test_desktop_jobs_use_the_allowlisted_runner_selection():
    jobs = yaml.safe_load(WORKFLOW.read_text())["jobs"]

    for name in MAC_JOBS:
        assert jobs[name]["runs-on"] == RUNNER_EXPRESSION


def test_desktop_jobs_fail_closed_on_the_runner_contract_before_building():
    jobs = yaml.safe_load(WORKFLOW.read_text())["jobs"]

    for name in MAC_JOBS:
        steps = jobs[name]["steps"]
        checkout_index = next(
            index
            for index, step in enumerate(steps)
            if str(step.get("uses", "")).startswith("actions/checkout@")
        )
        preflight_index = next(
            index
            for index, step in enumerate(steps)
            if step.get("name") == "Verify Apple Silicon runner contract"
        )
        assert preflight_index == checkout_index + 1
        assert (
            steps[preflight_index]["run"] == "../../scripts/verify_macos_ci_runner.sh"
        )


def test_runner_preflight_pins_the_required_architecture_and_toolchain_floors():
    source = PREFLIGHT.read_text()

    assert '[[ "$architecture" != "arm64" ]]' in source
    assert "require_major_at_least Xcode 16" in source
    assert "require_major_at_least Swift 6" in source
    assert "require_major_at_least macOS_SDK 15" in source
    assert (
        "for command_name in codesign python3 swift swiftc xcodebuild xcrun" in source
    )
    assert 'require_command "$command_name"' in source


def test_only_the_allowlisted_fallback_extends_the_in_process_watchdog():
    jobs = yaml.safe_load(WORKFLOW.read_text())["jobs"]
    swift_test = next(
        step for step in jobs["build"]["steps"] if step.get("name") == "swift test"
    )
    assert swift_test["env"]["RAPID_DESKTOP_WATCHDOG_QUIET_SECONDS"] == (
        WATCHDOG_EXPRESSION
    )

    watchdog = (
        ROOT / "apps/rapid-mac/Tests/RapidTests/Support/CIHangWatchdog.swift"
    ).read_text()
    assert 'environment["RAPID_DESKTOP_WATCHDOG_QUIET_SECONDS"] == "180"' in watchdog
    assert "? 180 : 90" in watchdog
