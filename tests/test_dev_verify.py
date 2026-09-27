# SPDX-License-Identifier: Apache-2.0
"""Behavioral contract for the local dev verification command.

`scripts/dev_verify.py` must turn a changed-path diff or a Desktop report area
into a plan built only from the production routers and the journey manifest,
execute it fail-closed, and record commit-bound evidence. A blocked check, a
missing diff, or an unimplemented journey is never allowed to read as green.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest
import yaml

from scripts import dev_verify
from scripts.dev_verify import (
    AREA_JOURNEYS,
    Check,
    build_plan,
    execute_checks,
    parse_journey_result,
    swift_filter_command,
)

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests/fixtures/dev_verify/desktop_reports.yaml"


def plan_for(**kwargs) -> dev_verify.Plan:
    defaults = dict(
        diff_base=None,
        paths_file=None,
        area=None,
        journey=None,
        plan=False,
        build=False,
        out=None,
        json=False,
    )
    return build_plan(argparse.Namespace(**{**defaults, **kwargs}))


def paths_file(tmp_path: Path, paths: list[str]) -> Path:
    target = tmp_path / "changed-paths.txt"
    target.write_text("\n".join(paths) + "\n")
    return target


def check_ids(plan: dev_verify.Plan) -> list[str]:
    return [check.id for check in plan.checks]


# --------------------------------------------------------------------------
# Diff/paths selection reuses the production routers
# --------------------------------------------------------------------------


def test_images_path_selects_only_the_image_journey(tmp_path: Path):
    plan = plan_for(
        paths_file=paths_file(
            tmp_path, ["apps/rapid-mac/Sources/Rapid/Images/ImageClient.swift"]
        )
    )
    assert plan.lanes == {"engine": False, "desktop": True, "docs_only": False}
    assert plan.flows == ["image-generation"]
    assert "engine:lint" not in check_ids(plan)
    assert "desktop:contracts" in check_ids(plan)
    journey = next(c for c in plan.checks if c.id == "journey:image-generation")
    assert journey.kind == "gui-journey"
    assert "--flow image-generation" in journey.command
    assert journey.evidence_paths and journey.evidence_paths[0].endswith(
        "journeys/image-generation/result.json"
    )


def test_engine_path_selects_lint_and_unit_and_no_desktop(tmp_path: Path):
    plan = plan_for(paths_file=paths_file(tmp_path, ["rapid_mlx/server/app.py"]))
    assert plan.lanes["engine"] is True
    assert plan.lanes["desktop"] is False
    assert check_ids(plan) == ["engine:lint", "engine:unit"]
    assert all(c.kind != "gui-journey" for c in plan.checks)


def test_docs_only_diff_runs_lint_only_and_says_why(tmp_path: Path):
    plan = plan_for(paths_file=paths_file(tmp_path, ["docs/index.md"]))
    assert plan.lanes["docs_only"] is True
    assert check_ids(plan) == ["engine:lint"]
    assert "docs-only" in plan.checks[0].why


def test_chat_source_change_implicates_the_groups_swift_suites(tmp_path: Path):
    plan = plan_for(
        paths_file=paths_file(
            tmp_path, ["apps/rapid-mac/Sources/Rapid/Chat/ChatViewModel.swift"]
        )
    )
    swift = next(c for c in plan.checks if c.id == "desktop:swift-test")
    for name in ("message-actions", "math-rendering", "slow-stream-stop"):
        assert name in swift.command
    # The bash router never emits swift journeys; they must not become
    # harness flows.
    assert not any(c.id.startswith("journey:message-actions") for c in plan.checks)


def test_unknown_desktop_path_fails_closed_to_broad_coverage(tmp_path: Path):
    plan = plan_for(
        paths_file=paths_file(
            tmp_path, ["apps/rapid-mac/Sources/Rapid/BrandNewSurface.swift"]
        )
    )
    assert set(plan.flows) == set(dev_verify.select_all_flows())
    swift = next(c for c in plan.checks if c.id == "desktop:swift-test")
    # Broad selection must run the whole suite, not a filter.
    assert "desktop-test-timeout.sh" in swift.command


def test_shared_ui_component_expands_to_every_flow(tmp_path: Path):
    plan = plan_for(
        paths_file=paths_file(
            tmp_path, ["apps/rapid-mac/Sources/Rapid/UI/Components/EmptyState.swift"]
        )
    )
    assert set(plan.flows) == set(dev_verify.select_all_flows())


def test_invalid_or_missing_diff_never_yields_a_plan(tmp_path: Path):
    empty = paths_file(tmp_path, [])
    with pytest.raises(RuntimeError, match="green no-op"):
        plan_for(paths_file=empty)


def test_unknown_journey_and_unknown_area_are_usage_errors():
    with pytest.raises(RuntimeError, match="unknown journey"):
        plan_for(journey=["not-a-journey"])
    with pytest.raises(RuntimeError, match="unknown Desktop area"):
        plan_for(area="Not a real area")


def test_diff_base_resolution_covers_worktree_and_untracked(tmp_path: Path):
    """Hermetic on any checkout: an untracked probe file must appear in the
    resolved diff, proving worktree changes are part of PR-shaped input."""
    probe = ROOT / "rapid-dev-verify-probe-untracked.txt"
    probe.write_text("probe")
    try:
        paths, base_sha = dev_verify.resolve_diff("origin/main")
        assert base_sha
        assert "rapid-dev-verify-probe-untracked.txt" in paths
    finally:
        probe.unlink(missing_ok=True)


# --------------------------------------------------------------------------
# Area mode yields candidates plus explicit gaps
# --------------------------------------------------------------------------


def test_area_plan_yields_candidates_and_gaps():
    plan = plan_for(area="Speed — answers arrive slowly, or the app feels sluggish")
    assert plan.area_gaps, "a gap must be explicit, never silent"
    assert "model-switch-active-request" in plan.flows
    assert plan.lanes["desktop"] is True
    assert plan.lanes["engine"] is False


def test_area_plan_for_settings_runs_the_persistence_journeys():
    plan = plan_for(area="Settings")
    journeys = {c.id for c in plan.checks if c.kind == "gui-journey"}
    assert "journey:settings-persistence" in journeys
    assert "journey:settings-mtp" in journeys


def test_broad_area_maps_to_every_bash_flow():
    plan = plan_for(area="Somewhere else / not sure")
    bash_flows = dev_verify._bash_flow_journeys(plan.flows)
    assert set(bash_flows) == set(dev_verify.select_all_flows())


def test_explicit_journey_pins_a_single_flow(tmp_path: Path):
    plan = plan_for(journey=["settings-persistence"])
    assert plan.flows == ["settings-persistence"]
    assert plan.flow_reasons["settings-persistence"] == (
        "explicitly requested via --journey"
    )
    assert "desktop:contracts" in check_ids(plan)


def test_explicit_swift_journey_becomes_a_suite_filter_not_a_flow():
    plan = plan_for(journey=["message-actions"])
    assert "desktop:swift-test" in check_ids(plan)
    swift = next(c for c in plan.checks if c.id == "desktop:swift-test")
    assert "message-actions" in swift.command
    assert not any(c.kind == "gui-journey" for c in plan.checks)


def test_explicit_local_tier_journey_is_allowed_with_a_note():
    plan = plan_for(journey=["chat-depth"])
    assert any("local-tier" in note for note in plan.notes)


def test_build_flag_prepends_the_existing_build_script():
    plan = plan_for(area="Settings", build=True)
    assert plan.checks[0].id == "desktop:build-app"
    assert plan.checks[0].command == "./scripts/build.sh"
    assert plan.checks[0].env["SKIP_SIDECAR"] == "1"


def test_journey_names_are_regex_literal_in_suite_filters():
    command = swift_filter_command(["message-actions", "math-rendering"])
    assert command == (
        'swift test --filter "Golden journey: (message-actions|math-rendering)"'
    )
    with pytest.raises(RuntimeError, match="filter-safe"):
        swift_filter_command(["bad(name"])


# --------------------------------------------------------------------------
# Execution is fail-closed and records evidence
# --------------------------------------------------------------------------


def _run_plan_with_commands(
    tmp_path: Path, commands: list[tuple[str, str, int]]
) -> tuple[dev_verify.Plan, Path]:
    checks = [
        Check(
            id=check_id,
            kind=kind,
            command=command,
            why="synthetic check for executor contract",
        )
        for check_id, kind, command in commands
    ]
    # Executor runs commands from ROOT/check.cwd; neutralize for synthetics.
    for check in checks:
        check.cwd = str(tmp_path)
    evidence = tmp_path / "evidence"
    execute_checks(checks, evidence)
    return checks, evidence


def test_failing_check_is_fail_with_log_evidence(tmp_path: Path):
    checks, evidence = _run_plan_with_commands(
        tmp_path,
        [
            ("synthetic:fail", "lint", "echo before-failure && exit 3"),
        ],
    )
    assert checks[0].status == "fail"
    assert checks[0].exit_code == 3
    log = Path(checks[0].log_path)
    assert log.exists() and "before-failure" in log.read_text()
    assert log.is_relative_to(evidence)


def test_passing_check_is_pass(tmp_path: Path):
    checks, _ = _run_plan_with_commands(
        tmp_path,
        [
            ("synthetic:pass", "lint", "true"),
        ],
    )
    assert checks[0].status == "pass"
    assert checks[0].exit_code == 0


def test_missing_tool_is_blocked_never_pass(tmp_path: Path, monkeypatch):
    checks = [
        Check(
            id="journey:settings-persistence",
            kind="gui-journey",
            command="true",
            why="synthetic",
            env={"RAPID_GUI_GOLDEN_OUT": str(tmp_path / "j")},
        )
    ]
    monkeypatch.setattr(dev_verify.shutil, "which", lambda name: None)
    execute_checks(checks, tmp_path / "evidence")
    assert checks[0].status == "blocked"
    assert "jq" in checks[0].blocked_reason


def test_missing_built_app_is_blocked_with_recovery_hint(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(dev_verify, "GUI_APP", str(tmp_path / "Missing.app"))
    checks = [
        Check(
            id="journey:settings-persistence",
            kind="gui-journey",
            command="true",
            why="synthetic",
            env={"RAPID_GUI_GOLDEN_OUT": str(tmp_path / "j")},
        )
    ]
    execute_checks(checks, tmp_path / "evidence")
    assert checks[0].status == "blocked"
    assert "--build" in checks[0].blocked_reason


def test_harness_result_parsing_accepts_only_executable_verdicts(tmp_path: Path):
    good = tmp_path / "pass.json"
    good.write_text(json.dumps({"status": "pass", "flow": "x"}))
    assert parse_journey_result(good) == (
        "pass",
        dev_verify.parse_journey_result(good)[1],
    )

    bad = tmp_path / "fail.json"
    bad.write_text(json.dumps({"status": "fail", "flow": "x"}))
    assert parse_journey_result(bad)[0] == "fail"

    missing = tmp_path / "missing.json"
    assert parse_journey_result(missing)[0] == "fail"

    nonsense = tmp_path / "nonsense.json"
    nonsense.write_text(json.dumps({"status": "skipped"}))
    status, detail = parse_journey_result(nonsense)
    assert status == "fail" and "skipped" in detail


def test_gui_journey_agrees_with_harness_result_json(tmp_path: Path, monkeypatch):
    out = tmp_path / "j"
    out.mkdir(parents=True)
    (out / "result.json").write_text(json.dumps({"status": "fail", "flow": "f"}))
    # The preflight/tool gate is its own contract; this test pins the verdict
    # parsing: harness exit 0 with a fail result.json stays red.
    monkeypatch.setattr(dev_verify, "_tool_missing", lambda check: None)
    checks = [
        Check(
            id="journey:fake",
            kind="gui-journey",
            command="true",  # harness exits 0 but reported fail
            why="synthetic",
            env={"RAPID_GUI_GOLDEN_OUT": str(out)},
        )
    ]
    for check in checks:
        check.cwd = str(tmp_path)
    execute_checks(checks, tmp_path / "evidence")
    assert checks[0].status == "fail", (
        "a harness fail status must never collapse into a green check"
    )


def test_blocked_build_gates_journeys_even_when_a_stale_app_exists(
    tmp_path: Path, monkeypatch
):
    """--build plus no swift toolchain plus an old binary on disk must never
    produce green journey evidence against that stale binary."""
    stale_app = tmp_path / "Stale.app"
    stale_app.mkdir()
    monkeypatch.setattr(dev_verify, "GUI_APP", str(stale_app))
    monkeypatch.setattr(
        dev_verify.shutil,
        "which",
        lambda name: None if name in ("swift", "xcodebuild") else "/usr/bin/jq",
    )
    out = tmp_path / "j"
    writes_pass = (
        'mkdir -p "$RAPID_GUI_GOLDEN_OUT" && '
        'printf \'{"status":"pass","flow":"f"}\' '
        '> "$RAPID_GUI_GOLDEN_OUT/result.json"'
    )
    checks = [
        Check(
            id="desktop:build-app",
            kind="build-app",
            command="true",
            why="synthetic",
        ),
        Check(
            id="journey:settings-persistence",
            kind="gui-journey",
            command=writes_pass,
            why="synthetic",
            env={"RAPID_GUI_GOLDEN_OUT": str(out), **dev_verify.TELEMETRY_NEUTRAL_ENV},
        ),
    ]
    execute_checks(checks, tmp_path / "evidence")
    assert checks[0].status == "blocked"
    assert checks[1].status == "blocked"
    assert "stale" in checks[1].blocked_reason


def test_malformed_manifest_is_a_clean_usage_error(tmp_path: Path, monkeypatch):
    def broken_manifest():
        raise yaml.YAMLError("scan error")

    monkeypatch.setattr(dev_verify, "journey_index", broken_manifest)
    rc = dev_verify.main(
        ["--journey", "settings-persistence", "--out", str(tmp_path / "e")]
    )
    assert rc == 2


def test_explicit_swift_journey_is_recorded_in_result_input(tmp_path: Path):
    plan = plan_for(journey=["message-actions"])
    payload = dev_verify.result_payload(plan, tmp_path, "a" * 40, "", "", "plan")
    assert payload["input"]["explicit_journeys"] == ["message-actions"]


def test_journey_names_never_reach_a_shell_unvalidated(tmp_path: Path, monkeypatch):
    """journeys.yaml is PR-controlled: a crafted journey name must fail the
    plan instead of executing inside the command string."""
    malicious = "x; touch pwned-proof"
    real_index = dev_verify.journey_index
    monkeypatch.setattr(
        dev_verify,
        "journey_index",
        lambda: {
            malicious: {
                "name": malicious,
                "group": "chat",
                "driver": "ax",
                "ci_tier": "pr",
                "source_paths": ["apps/rapid-mac/Sources/Rapid/Chat/"],
            },
            **real_index(),
        },
    )
    with pytest.raises(RuntimeError, match="safe command argument"):
        plan_for(journey=[malicious])
    assert not (ROOT / "pwned-proof").exists()


def test_relative_out_resolves_against_repo_root(tmp_path: Path):
    plan = plan_for(area="Settings", out="relative-evidence-dir")
    assert plan.evidence_dir.is_absolute()
    assert str(plan.evidence_dir).startswith(str(ROOT))


def test_result_artifacts_include_the_check_log(tmp_path: Path):
    checks, _ = _run_plan_with_commands(
        tmp_path,
        [
            ("synthetic:pass", "lint", "true"),
        ],
    )
    plan = dev_verify.Plan(
        lanes={"engine": True, "desktop": False, "docs_only": False},
        flows=[],
        flow_reasons={},
        area=None,
        area_gaps=[],
        notes=[],
        checks=checks,
        input_mode="paths-file",
        evidence_dir=tmp_path,
    )
    payload = dev_verify.result_payload(plan, tmp_path, "a" * 40, "", "", "execute")
    check = payload["checks"][0]
    assert check["log_path"] in check["artifact_paths"]


def test_a_deliberately_failing_selected_check_fails_main_and_leaves_artifacts(
    tmp_path: Path, monkeypatch
):
    """The acceptance contract: a failing check fails the command, and the
    failure artifact path is in the result — not a swallowed green."""
    monkeypatch.setattr(dev_verify, "LINT_COMMAND", "echo boom-from-test && exit 9")
    monkeypatch.setattr(dev_verify, "UNIT_COMMAND", "true")  # keep this test fast
    rc = dev_verify.main(
        [
            "--paths-file",
            str(paths_file(tmp_path, ["rapid_mlx/server/app.py"])),
            "--out",
            str(tmp_path / "evidence"),
        ]
    )
    assert rc == 1
    payload = json.loads((tmp_path / "evidence" / "verify-result.json").read_text())
    assert payload["status"] == "fail"
    failed = next(c for c in payload["checks"] if c["id"] == "engine:lint")
    assert failed["exit_code"] == 9
    assert (
        failed["log_path"] and "boom-from-test" in Path(failed["log_path"]).read_text()
    )
    assert failed["log_path"] in failed["artifact_paths"]


def test_explicit_journey_survives_a_docs_only_diff(tmp_path: Path):
    """An explicit request is a promise: a docs-only lane must not swallow it."""
    plan = plan_for(
        paths_file=paths_file(tmp_path, ["docs/index.md"]),
        journey=["settings-persistence"],
    )
    assert plan.lanes["desktop"] is True
    assert "journey:settings-persistence" in check_ids(plan)
    assert "engine:lint" in check_ids(plan)


def test_area_reason_upgrades_to_explicit_for_pinned_journeys():
    plan = plan_for(area="Settings", journey=["settings-persistence"])
    assert plan.flow_reasons["settings-persistence"] == (
        "explicitly requested via --journey"
    )
    assert plan.flows.count("settings-persistence") == 1


def test_failed_build_blocks_journeys_instead_of_running_stale_app(
    tmp_path: Path,
):
    checks = [
        Check(
            id="desktop:build-app",
            kind="build-app",
            command="echo build-boom && exit 1",
            why="synthetic",
            cwd=str(tmp_path),
        ),
        Check(
            id="journey:settings-persistence",
            kind="gui-journey",
            command="true",
            why="synthetic",
            cwd=str(tmp_path),
            env={
                "RAPID_GUI_GOLDEN_OUT": str(tmp_path / "j"),
                **dev_verify.TELEMETRY_NEUTRAL_ENV,
            },
        ),
    ]
    # The app exists on disk (stale), so only the run-order guard protects
    # the commit-bound evidence contract.
    execute_checks(checks, tmp_path / "evidence")
    assert checks[0].status == "fail"
    assert checks[1].status == "blocked"
    assert "stale" in checks[1].blocked_reason


def test_shell_routing_env_cannot_leak_into_a_journey(tmp_path: Path, monkeypatch):
    """A stale GUI_FLOWS in the invoking shell would make the harness SKIP
    and exit 0 without a result.json — that must not read as green."""
    monkeypatch.setenv("GUI_FLOWS", '["image-generation"]')
    monkeypatch.setenv("RAPID_GUI_SOURCE_APP", "/some/stale.app")
    captured: dict[str, str] = {}

    original_run = dev_verify.subprocess.run

    def spy_run(*args, **kwargs):
        captured.update(kwargs["env"])
        return type("Proc", (), {"returncode": 0})()

    monkeypatch.setattr(dev_verify.subprocess, "run", spy_run)
    checks = [
        Check(
            id="journey:settings-persistence",
            kind="gui-journey",
            command="true",
            why="synthetic",
            env={
                "RAPID_GUI_GOLDEN_OUT": str(tmp_path / "j"),
                **dev_verify.TELEMETRY_NEUTRAL_ENV,
            },
        )
    ]
    monkeypatch.setattr(dev_verify, "_tool_missing", lambda check: None)
    monkeypatch.setattr(
        dev_verify,
        "parse_journey_result",
        lambda path: ("pass", "synthetic"),
    )
    execute_checks(checks, tmp_path / "evidence")
    assert "GUI_FLOWS" not in captured
    assert "RAPID_GUI_SOURCE_APP" not in captured
    assert captured["RAPID_MLX_TELEMETRY"] == "0"
    assert checks[0].status == "pass"


def test_result_payload_is_commit_bound_and_counts_blocked_as_fail(
    tmp_path: Path,
):
    plan = plan_for(area="Settings")
    for check in plan.checks:
        check.cwd = str(tmp_path)
        check.status = "pass"
    plan.checks[-1].status = "blocked"
    payload = dev_verify.result_payload(
        plan, tmp_path, "a" * 40, "origin/main", "b" * 40, "execute"
    )
    assert payload["status"] == "fail"
    assert payload["head_sha"] == "a" * 40
    assert payload["diff_base"] == "origin/main"
    assert payload["base_sha"] == "b" * 40
    for check in payload["checks"]:
        assert set(check) >= {
            "id",
            "kind",
            "status",
            "exit_code",
            "command",
            "why",
            "prerequisites",
            "log_path",
            "artifact_paths",
        }


def test_plan_only_result_is_never_a_pass(tmp_path: Path):
    plan = plan_for(area="Settings")
    payload = dev_verify.result_payload(plan, tmp_path, "a" * 40, "", "", "plan")
    assert payload["status"] == "plan-only"
    assert all(c["status"] == "planned" for c in payload["checks"])


# --------------------------------------------------------------------------
# Historical Desktop reports route as recorded (Echo's triage examples)
# --------------------------------------------------------------------------


def test_ten_representative_reports_route_as_recorded():
    reports = yaml.safe_load(FIXTURES.read_text())
    assert len(reports) == 10
    expected = {entry["issue"]: entry for entry in reports}

    for entry in reports:
        plan = plan_for(area=entry["area"])
        candidate_ids = {
            c.id.removeprefix("journey:")
            for c in plan.checks
            if c.kind == "gui-journey"
        }
        swift = next((c for c in plan.checks if c.id == "desktop:swift-test"), None)
        swift_names = [
            name
            for name in dev_verify._implicated_swift_journeys(plan.flows)
            if swift and name in swift.command
        ]
        reachable = set(candidate_ids) | set(swift_names)
        if entry.get("uncovered"):
            assert not entry.get("expected_journeys")
            continue
        for name in entry["expected_journeys"]:
            assert name in reachable, (
                f"issue #{entry['issue']} expects journey {name}; plan for"
                f" area '{entry['area']}' offers {sorted(reachable)}"
            )

    # The acceptance cases are present: settings persistence and model crash
    # recovery must be exercised by at least one recorded report each.
    exercised = {
        name for entry in reports for name in (entry.get("expected_journeys") or [])
    }
    assert "settings-persistence" in exercised
    assert "model-crash-recovery" in exercised
    assert any(entry.get("uncovered") for entry in reports), (
        "at least one report must demonstrate the explicit uncovered path"
    )
    assert all(entry["area"] in AREA_JOURNEYS for entry in reports)
