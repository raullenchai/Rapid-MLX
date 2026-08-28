# SPDX-License-Identifier: Apache-2.0
"""Contracts for lane-scoped full-CI promotion.

These tests intentionally inspect the workflows: ordinary PRs must keep their
fast product checks while release-grade model/GUI lanes are reserved for a
``full-ci`` promotion or an integration candidate.
"""

import os
import shlex
import subprocess
from collections import Counter
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
ENGINE_WORKFLOW = ROOT / ".github/workflows/ci.yml"
DESKTOP_WORKFLOW = ROOT / ".github/workflows/rapid-mac-ci.yml"
VERSION_WORKFLOW = ROOT / ".github/workflows/version-check.yml"


def _step_run(workflow: Path, job: str, step_name: str) -> str:
    steps = yaml.safe_load(workflow.read_text())["jobs"][job]["steps"]
    (step,) = [candidate for candidate in steps if candidate.get("name") == step_name]
    return str(step["run"])


def _job(workflow: Path, job: str) -> dict[str, object]:
    return yaml.safe_load(workflow.read_text())["jobs"][job]


def _explicit_pytest_nodes(command: str) -> list[str] | None:
    """Return explicit test nodes, rejecting an unrecognised pytest launcher."""
    tokens = shlex.split(command)
    explicit_nodes = [
        token for token in tokens if token.startswith("tests/") and ".py" in token
    ]
    if not explicit_nodes:
        return None

    if tokens[0] == "pytest":
        argument_offset = 1
    elif (
        len(tokens) >= 3
        and Path(tokens[0]).name.startswith("python")
        and tokens[1:3] == ["-m", "pytest"]
    ):
        argument_offset = 3
    else:
        raise AssertionError(f"unsupported pytest command form: {command}")

    return [
        token
        for token in tokens[argument_offset:]
        if token.startswith("tests/") and ".py" in token
    ]


def _pytest_invocations(workflow: Path) -> list[tuple[str, list[str]]]:
    """Return explicit test nodes from each recognised workflow pytest command."""
    jobs = yaml.safe_load(workflow.read_text())["jobs"]
    invocations: list[tuple[str, list[str]]] = []
    for job_name, job in jobs.items():
        for step in job.get("steps", []):
            run = str(step.get("run", "")).replace("\\\n", " ")
            for line in run.splitlines():
                command = line.strip()
                explicit_nodes = _explicit_pytest_nodes(command)
                if explicit_nodes is None:
                    continue
                invocations.append(
                    (f"{workflow.name}:{job_name}:{step.get('name')}", explicit_nodes)
                )
    return invocations


def _assert_no_duplicate_explicit_test_nodes(
    invocations: list[tuple[str, list[str]]],
) -> None:
    duplicates: list[str] = []
    for location, nodes in invocations:
        repeated = sorted(node for node, count in Counter(nodes).items() if count > 1)
        duplicates.extend(f"{location}: {node}" for node in repeated)

    assert not duplicates, "duplicate explicit pytest nodes:\n" + "\n".join(duplicates)


def test_pytest_command_parser_supports_workflow_launchers():
    node = "tests/test_cli_models.py"
    assert _explicit_pytest_nodes(f"pytest {node} -q") == [node]
    assert _explicit_pytest_nodes(f"python -m pytest {node} -q") == [node]
    assert _explicit_pytest_nodes(f"python3.12 -m pytest {node} -q") == [node]


def test_pytest_command_parser_rejects_unknown_launcher_with_explicit_nodes():
    with pytest.raises(AssertionError, match="unsupported pytest command form"):
        _explicit_pytest_nodes("uv run pytest tests/test_cli_models.py -q")


def test_duplicate_explicit_test_nodes_are_rejected():
    node = "tests/test_cli_models.py"
    with pytest.raises(AssertionError, match=node):
        _assert_no_duplicate_explicit_test_nodes([("ci.yml:tests", [node, node])])


def test_workflow_pytest_invocations_do_not_repeat_explicit_test_nodes():
    invocations: list[tuple[str, list[str]]] = []
    for workflow in (ENGINE_WORKFLOW, DESKTOP_WORKFLOW):
        workflow_invocations = _pytest_invocations(workflow)
        assert workflow_invocations, (
            f"no explicit workflow pytest invocations parsed from {workflow.name}"
        )
        invocations.extend(workflow_invocations)

    _assert_no_duplicate_explicit_test_nodes(invocations)


def _workflow_strings(workflow: Path) -> dict[str, object]:
    """Keep ``on`` as a string instead of YAML 1.1's boolean True."""
    return yaml.load(workflow.read_text(), Loader=yaml.BaseLoader)


def test_product_promotion_keeps_diff_scope_and_accepts_integration_heads():
    for workflow, job_name, step_name in (
        (ENGINE_WORKFLOW, "changes", "Classify validation lanes"),
        (DESKTOP_WORKFLOW, "changes", "Classify desktop lane"),
    ):
        run = _step_run(workflow, job_name, step_name)
        assert 'git diff --no-renames --name-only "$PR_BASE_SHA" "$GITHUB_SHA"' in run
        assert '[ "$FULL_CI" = true ] ||' in run
        assert '[ "$HEAD_REPO" = "$REPO" ] &&' in run
        assert '[[ "$HEAD_REF" == train/* ]] ||' in run
        assert '[[ "$HEAD_REF" =~ ^mergify/merge-queue/[0-9a-f]{10}$ ]]' in run
        assert 'echo "full_gate=$full_gate"' in run


def _execute_promotion(
    workflow: Path, job_name: str, step_name: str, **env: str
) -> str:
    run = _step_run(workflow, job_name, step_name)
    start = run.index("full_gate=false")
    end = run.index('echo "full_gate=$full_gate"', start)
    snippet = run[start : end + len('echo "full_gate=$full_gate" >> "$GITHUB_OUTPUT"')]
    output = Path(env.pop("GITHUB_OUTPUT"))
    subprocess.run(
        ["bash", "-c", snippet],
        check=True,
        env=os.environ | env | {"GITHUB_OUTPUT": str(output)},
    )
    return output.read_text().strip()


def test_fork_cannot_claim_train_branch_promotion(tmp_path):
    for index, (workflow, job_name, step_name) in enumerate(
        (
            (ENGINE_WORKFLOW, "changes", "Classify validation lanes"),
            (DESKTOP_WORKFLOW, "changes", "Classify desktop lane"),
        )
    ):
        common = {
            "FULL_CI": "false",
            "HEAD_REF": "train/spoofed",
            "REPO": "owner/repo",
        }
        assert (
            _execute_promotion(
                workflow,
                job_name,
                step_name,
                GITHUB_OUTPUT=str(tmp_path / f"fork-{index}"),
                HEAD_REPO="attacker/fork",
                **common,
            )
            == "full_gate=false"
        )
        assert (
            _execute_promotion(
                workflow,
                job_name,
                step_name,
                GITHUB_OUTPUT=str(tmp_path / f"internal-{index}"),
                HEAD_REPO="owner/repo",
                **common,
            )
            == "full_gate=true"
        )


@pytest.mark.parametrize(
    ("head_ref", "head_repo", "expected"),
    (
        ("mergify/merge-queue/a76b1f3d25", "owner/repo", "full_gate=true"),
        ("mergify/merge-queue/A76B1F3D25", "owner/repo", "full_gate=false"),
        ("mergify/merge-queue/a76b1f3d2", "owner/repo", "full_gate=false"),
        ("mergify/merge-queue/a76b1f3d25-extra", "owner/repo", "full_gate=false"),
        ("tmp-mergify/merge-queue/a76b1f3d25", "owner/repo", "full_gate=false"),
        ("mergify/merge-queue/a76b1f3d25", "attacker/fork", "full_gate=false"),
    ),
)
def test_only_exact_internal_mergify_batch_heads_promote_full_ci(
    tmp_path, head_ref, head_repo, expected
):
    for index, (workflow, job_name, step_name) in enumerate(
        (
            (ENGINE_WORKFLOW, "changes", "Classify validation lanes"),
            (DESKTOP_WORKFLOW, "changes", "Classify desktop lane"),
        )
    ):
        assert (
            _execute_promotion(
                workflow,
                job_name,
                step_name,
                GITHUB_OUTPUT=str(tmp_path / f"mergify-{index}"),
                FULL_CI="false",
                HEAD_REF=head_ref,
                HEAD_REPO=head_repo,
                REPO="owner/repo",
            )
            == expected
        )


def test_only_promoted_heads_allocate_expensive_lanes():
    l1 = str(_job(ENGINE_WORKFLOW, "l1-smoke")["if"])
    assert "needs.changes.outputs.engine == 'true'" in l1
    assert "needs.changes.outputs.full_gate == 'true'" in l1

    for job_name in ("gui-app-build", "gui-golden-flows"):
        condition = str(_job(DESKTOP_WORKFLOW, job_name)["if"])
        assert "needs.changes.outputs.desktop == 'true'" in condition
        assert "needs.changes.outputs.full_gate == 'true'" in condition


def test_unpromoted_engine_aggregate_requires_fast_linux_and_apple_lanes():
    run = _step_run(ENGINE_WORKFLOW, "tests", "Check test results")
    no_lane = run.index('if [ "$expected" != "true" ]')
    engine_gate = run.index("needs.engine-contracts.result")
    linux_gate = run.index("needs.test-matrix.result")
    apple_gate = run.index("needs.test-apple-silicon.result")
    promotion = run.index('needs.changes.outputs.full_gate }}" = "true"')
    assert no_lane < engine_gate < linux_gate < apple_gate < promotion
    assert "status remains pending" not in run
    assert 'needs.l1-smoke.result }}" != "skipped"' in run


def test_unpromoted_desktop_aggregate_requires_fast_lane_and_skips_gui():
    run = _step_run(DESKTOP_WORKFLOW, "desktop-tests", "Check desktop results")
    no_lane = run.index('if [ "$DESKTOP_EXPECTED" != true ]')
    build_gate = run.index('for result in "$IDENTIFIERS"')
    promotion = run.index('if [ "$FULL_GATE" = true ]')
    skipped_gate = run.index('elif [ "$GUI_APP_BUILD" != skipped ]')
    assert no_lane < build_gate < promotion < skipped_gate
    assert "status remains pending" not in run


def test_required_aggregates_do_not_publish_shadow_commit_statuses():
    for workflow, job_name in (
        (ENGINE_WORKFLOW, "tests"),
        (DESKTOP_WORKFLOW, "desktop-tests"),
    ):
        job = _job(workflow, job_name)
        assert "permissions" not in job
        names = {step.get("name") for step in job["steps"]}
        assert "Start an internal PR merge status pending" not in names
        assert "Settle a successful internal full-CI status" not in names

    assert not (ROOT / ".github/workflows/full-ci-label-gate.yml").exists()


def test_main_pushes_keep_full_engine_and_desktop_validation():
    for workflow in (ENGINE_WORKFLOW, DESKTOP_WORKFLOW):
        triggers = _workflow_strings(workflow)["on"]
        assert triggers["push"]["branches"] == ["main"]

    engine = _step_run(ENGINE_WORKFLOW, "changes", "Classify validation lanes")
    desktop = _step_run(DESKTOP_WORKFLOW, "changes", "Classify desktop lane")
    assert "else" in engine and "echo 'full_gate=true'" in engine
    assert 'if [ "$EVENT_NAME" != "pull_request" ]' in desktop
    assert "echo 'full_gate=true'" in desktop


def test_all_strict_required_workflows_emit_on_merge_group():
    for workflow in (ENGINE_WORKFLOW, DESKTOP_WORKFLOW, VERSION_WORKFLOW):
        triggers = _workflow_strings(workflow)["on"]
        assert triggers["merge_group"]["types"] == ["checks_requested"]

    version = _workflow_strings(VERSION_WORKFLOW)
    guard = version["jobs"]["version-bump-guard"]
    queue_step = next(
        step
        for step in guard["steps"]
        if step.get("name") == "Pass — PR contract already validated before merge queue"
    )
    assert queue_step["if"] == "github.event_name == 'merge_group'"


def test_gui_golden_job_requires_both_desktop_lane_and_full_promotion():
    condition = str(_job(DESKTOP_WORKFLOW, "gui-golden-flows")["if"])
    assert "needs.changes.outputs.desktop == 'true'" in condition
    assert "needs.changes.outputs.full_gate == 'true'" in condition


def test_engine_only_contracts_are_not_universal_pr_guards():
    universal_steps = {
        step.get("name") for step in _job(ENGINE_WORKFLOW, "lint")["steps"]
    }
    engine_steps = {
        step.get("name") for step in _job(ENGINE_WORKFLOW, "engine-contracts")["steps"]
    }
    assert {
        "GitHub Actions SHA pinning",
        "Workflow expression sanity",
        "Model-management architecture SSOT",
        "Run ruff lint",
        "Run ruff format check",
        "Engine ↔ desktop app version sync",
    } <= universal_steps
    assert {
        "CLI ↔ Config fidelity audit",
        "Release-script offline tests",
        "Installer offline tests",
        "Parser microbench",
    } <= engine_steps
    assert not universal_steps & {
        "CLI ↔ Config fidelity audit",
        "Release-script offline tests",
        "Installer offline tests",
        "Parser microbench",
    }


def test_engine_jobs_follow_fail_closed_engine_classification():
    for job_name in ("engine-contracts", "type-check"):
        job = _job(ENGINE_WORKFLOW, job_name)
        assert job["needs"] == "changes"
        assert str(job["if"]) == "needs.changes.outputs.engine == 'true'"

    bound_guard = _job(ENGINE_WORKFLOW, "mlx-bound-guard")
    assert bound_guard["needs"] == "changes"
    condition = str(bound_guard["if"])
    assert "github.event_name == 'pull_request'" in condition
    assert "needs.changes.outputs.engine == 'true'" in condition


def test_type_check_enforces_shrink_only_error_budget():
    type_check = _job(ENGINE_WORKFLOW, "type-check")
    steps = type_check["steps"]
    ratchet = next(
        step
        for step in steps
        if step.get("name") == "Enforce shrink-only mypy error budget"
    )

    assert "continue-on-error" not in ratchet
    assert ratchet["run"] == "python scripts/check_mypy_error_budget.py"
    install = next(step for step in steps if step.get("name") == "Install dependencies")
    assert "pip install --requirement config/mypy-requirements.txt" in install["run"]
    requirements = (ROOT / "config/mypy-requirements.txt").read_text().splitlines()
    pins = [line for line in requirements if line and not line.startswith("#")]
    assert pins
    assert all("==" in pin for pin in pins)
    assert {pin.split("==", maxsplit=1)[0] for pin in pins} >= {
        "mypy",
        "pydantic",
        "pydantic_core",
        "fastapi",
        "starlette",
        "typing_extensions",
    }
    unit_roster = _step_run(
        ENGINE_WORKFLOW, "test-matrix", "Run unit tests (no MLX required)"
    )
    assert "tests/test_check_mypy_error_budget.py" in unit_roster


def test_combined_platform_job_enforces_changed_lines_coverage_without_baseline():
    test_matrix = _job(ENGINE_WORKFLOW, "test-matrix")
    checkout = next(
        step
        for step in test_matrix["steps"]
        if str(step.get("uses", "")).startswith("actions/checkout@")
    )
    assert checkout["with"]["fetch-depth"] == 0

    gate_job = _job(ENGINE_WORKFLOW, "changed-lines-coverage")
    install = next(
        step
        for step in gate_job["steps"]
        if step.get("name") == "Install coverage tools"
    )
    assert '"diff-cover==8.0.3"' in install["run"]

    gate = next(
        step
        for step in gate_job["steps"]
        if step.get("name") == "Combine coverage and enforce changed lines"
    )
    assert gate["env"] == {"PR_BASE_SHA": "${{ github.event.pull_request.base.sha }}"}
    assert "continue-on-error" not in gate
    assert "coverage combine" in gate["run"]
    assert "coverage-data/linux/coverage-linux-3.11.data" in gate["run"]
    assert "coverage-data/apple/coverage-apple.data" in gate["run"]
    assert "coverage.xml" in gate["run"]
    assert '--compare-branch "$PR_BASE_SHA"' in gate["run"]
    assert "--show-uncovered" in gate["run"]
    assert "--fail-under 100" in gate["run"]
    assert "--cov-fail-under" not in gate["run"]
