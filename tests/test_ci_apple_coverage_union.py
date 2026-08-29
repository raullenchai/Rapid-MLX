from pathlib import Path

import yaml

WORKFLOW = Path(__file__).parents[1] / ".github" / "workflows" / "ci.yml"
PYPROJECT = Path(__file__).parents[1] / "pyproject.toml"
REQ_CI_LINUX = Path(__file__).parents[1] / "config" / "requirements-ci-linux.txt"


def _workflow() -> tuple[str, dict]:
    text = WORKFLOW.read_text()
    parsed = yaml.safe_load(text)
    return text, parsed


def _ci_linux_extra() -> list[str]:
    """Parse the canonical ``[ci-linux]`` test-dependency list from pyproject.

    Version-agnostic: ``tomllib`` is stdlib on 3.11+ only, and the Linux
    test-matrix explicitly includes 3.10. The declared CI dependency set
    installs the API-compatible ``tomli`` backport there.
    """
    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.10
        import tomli as tomllib
    with PYPROJECT.open("rb") as fh:
        return tomllib.load(fh)["project"]["optional-dependencies"]["ci-linux"]


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
    aggregate_run = next(
        step["run"]
        for step in jobs["tests"]["steps"]
        if step.get("name") == "Check test results"
    )
    assert "needs.changed-lines-coverage.result" in aggregate_run

    # Non-engine PRs return before inspecting the intentionally skipped union
    # job, preserving the path-aware required-check facade.
    early_exit = aggregate_run.index('if [ "$expected" != "true" ]')
    union_check = aggregate_run.index("needs.changed-lines-coverage.result")
    assert early_exit < union_check


def test_linux_coverage_lane_declares_complete_ci_linux_discovery_surface() -> None:
    """#2445 / #2446 root cause was the undeclared ``jinja2`` dep on the Linux
    lane: ``_should_start_in_thinking`` renders chat templates via
    ``transformers.utils.chat_template_utils._compile_jinja_template``, which
    needs jinja2 at runtime, and neither transformers nor the old ad hoc install
    line pulled it transitively. With jinja2 now declared in the ``[ci-linux]``
    extra, both formerly --deselect-ed tests run un-deselected.

    The lane must stay no-MLX (Apple-Silicon-only), so the install is
    ``-e . --no-deps`` (the
    base package's deps include mlx) + the extra's test deps from
    ``config/requirements-ci-linux.txt`` — never ``-e ".[ci-linux]"``, which
    would pull mlx onto Linux.

    This guard fails closed if any of the coordination contracts regress:
    jinja2 (or anything else) drifts out of ``[ci-linux]``, the synced
    requirements file falls out of step with the extra, the lane stops using
    the declared set (ad hoc install / safe ``-e .[ci-linux]``), or someone
    re-adds a ``--deselect`` instead of fixing the underlying dep.
    """
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

    # The lane installs the package WITHOUT deps (no mlx) + the declared test
    # deps from the synced requirements file — never the mlx-pulling editable
    # full install.
    assert "pip install -e . --no-deps" in install
    assert "pip install --requirement config/requirements-ci-linux.txt" in install
    assert 'pip install -e ".[ci-linux]"' not in install

    # The synced requirements file == the canonical [ci-linux] extra.
    req_lines = [
        line.strip()
        for line in REQ_CI_LINUX.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert req_lines == _ci_linux_extra()

    # jinja2 is declared in the ci-linux extra in pyproject.toml.
    pyproject = PYPROJECT.read_text()
    assert "ci-linux" in pyproject
    assert "jinja2" in pyproject

    # The two reasoning-template tests run through directory discovery; no
    # hand-maintained per-file roster or deselection can silently drop them.
    assert "--deselect=" not in run
    assert "tests/test_" not in run
    assert "pytest \\\n  tests \\" in run
    assert (
        '-m "not requires_mlx and not real_hf_cache and not requires_network '
        'and not slow and not integration and not needle"' in run
    )


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
