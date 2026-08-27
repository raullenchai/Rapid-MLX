from pathlib import Path

import yaml

WORKFLOW = Path(__file__).parents[1] / ".github" / "workflows" / "ci.yml"


def _workflow() -> tuple[str, dict]:
    text = WORKFLOW.read_text()
    parsed = yaml.safe_load(text)
    return text, parsed


def test_changed_lines_gate_unions_linux_and_apple_coverage() -> None:
    text, workflow = _workflow()
    jobs = workflow["jobs"]

    linux = jobs["test-matrix"]
    apple = jobs["test-apple-silicon"]
    gate = jobs["changed-lines-coverage"]

    assert "coverage-linux-${{ matrix.python-version }}.data" in text
    assert "coverage-apple.data" in text
    assert "--cov=vllm_mlx" in apple["steps"][-2]["run"]
    assert set(gate["needs"]) == {
        "changes",
        "test-matrix",
        "test-apple-silicon",
    }

    gate_run = gate["steps"][-1]["run"]
    assert "coverage combine" in gate_run
    assert "coverage-data/linux/coverage-linux-3.11.data" in gate_run
    assert "coverage-data/apple/coverage-apple.data" in gate_run
    assert "--fail-under 100" in gate_run

    aggregate_needs = set(jobs["tests"]["needs"])
    assert "changed-lines-coverage" in aggregate_needs
    aggregate_run = jobs["tests"]["steps"][0]["run"]
    assert "needs.changed-lines-coverage.result" in aggregate_run

    # Non-engine PRs return before inspecting the intentionally skipped union
    # job, preserving the path-aware required-check facade.
    early_exit = aggregate_run.index('if [ "$expected" != "true" ]')
    union_check = aggregate_run.index("needs.changed-lines-coverage.result")
    assert early_exit < union_check


def test_linux_coverage_lane_has_template_support_and_deselects_known_baseline_failures() -> (
    None
):
    _, workflow = _workflow()
    linux = workflow["jobs"]["test-matrix"]
    install = next(
        step for step in linux["steps"] if step.get("name") == "Install dependencies"
    )["run"]
    run = next(
        step
        for step in linux["steps"]
        if step.get("name") == "Run unit tests (no MLX required)"
    )["run"]

    assert "jinja2" in install.split()
    assert (
        "--deselect=tests/test_cohere_command_reasoning_parser.py::"
        "test_prompt_priming_detects_command_markers_and_mixed_templates"
    ) in run
    assert (
        "--deselect=tests/test_postprocessor.py::"
        "TestStreamingPostProcessorReasoning::"
        "test_1570_distill_parser_stays_active_when_thinking_flag_is_false"
    ) in run


def test_apple_coverage_roster_contains_only_tracked_tests() -> None:
    _, workflow = _workflow()
    apple_run = workflow["jobs"]["test-apple-silicon"]["steps"][-2]["run"]
    test_paths = [
        token.rstrip(" \\")
        for token in apple_run.splitlines()
        if token.strip().startswith("tests/")
    ]

    assert test_paths
    for relative_path in test_paths:
        assert (WORKFLOW.parents[2] / relative_path.strip()).is_file(), relative_path


def test_coverage_data_is_commit_bound_and_fail_closed() -> None:
    text, workflow = _workflow()
    jobs = workflow["jobs"]

    assert text.count("coverage-${{ github.sha }}") == 4
    for job_name in ("test-matrix", "test-apple-silicon"):
        upload = next(
            step
            for step in jobs[job_name]["steps"]
            if step.get("name", "").startswith("Upload ")
            and "coverage data" in step.get("name", "").lower()
        )
        assert upload["with"]["if-no-files-found"] == "error"
        assert upload["with"]["retention-days"] == 1


def test_coverage_paths_are_portable_across_runner_operating_systems() -> None:
    config = (WORKFLOW.parents[2] / ".coveragerc").read_text()
    assert "relative_files = True" in config
    assert "source = vllm_mlx" in config
