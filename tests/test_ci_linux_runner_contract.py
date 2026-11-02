"""Keep Linux runner routing opt-in and fork-isolated across PR/main workflows."""

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github/workflows"
ROUTED = (
    "ci.yml",
    "rapid-mac-ci.yml",
    "version-check.yml",
    "pr-validate.yml",
    "merge-on-green.yml",
    "main-guard.yml",
    "authorize-merge-ready.yml",
)
MAC_JOBS = {
    "ci.yml": {"test-apple-silicon", "test-apple-silicon-mac-only", "l1-smoke"},
    "rapid-mac-ci.yml": {
        "build",
        "gui-app-build",
        "gui-golden-flows",
        "release-toolchain-build",
    },
}
# Only multi-minute jobs ride the heavy pool; seconds-long control-plane jobs
# use their own switch so they never queue behind test shards.
HEAVY_JOBS = {"ci.yml": {"test-matrix"}}


def runner_expression(variable: str) -> str:
    return (
        f"${{{{ fromJSON((vars.{variable} != '' && (!github.event.pull_request || "
        "github.event.pull_request.head.repo.full_name == github.repository) && "
        "(!github.event.workflow_run || "
        "github.event.workflow_run.head_repository.full_name == github.repository)) && "
        f"""vars.{variable} || '"ubuntu-latest"') }}}}"""
    )


HEAVY_EXPRESSION = runner_expression("CI_LINUX_RUNNER")
LIGHT_EXPRESSION = runner_expression("CI_LINUX_LIGHT_RUNNER")


@pytest.mark.parametrize("filename", ROUTED)
def test_every_linux_job_uses_exact_fork_safe_runner_expression(filename):
    source = (WORKFLOWS / filename).read_text()
    jobs = yaml.safe_load(source)["jobs"]
    mac_jobs = MAC_JOBS.get(filename, set())
    assert mac_jobs <= jobs.keys()
    for name, job in jobs.items():
        if name in mac_jobs:
            if filename == "ci.yml" or name == "release-toolchain-build":
                assert job["runs-on"] == "macos-15"
            else:
                assert job["runs-on"] == (
                    "${{ vars.MACOS_RUNNER == 'macos-26' && 'macos-26' "
                    "|| 'manzanita-standard' }}"
                )
        elif name in HEAVY_JOBS.get(filename, set()):
            assert job["runs-on"] == HEAVY_EXPRESSION, name
        else:
            assert job["runs-on"] == LIGHT_EXPRESSION, name
    assert HEAVY_JOBS.get(filename, set()) <= jobs.keys()
    assert "runs-on: ubuntu-latest" not in source
    assert "gh variable delete CI_LINUX_LIGHT_RUNNER" in source
    assert (
        "Fork heads (pull_request, pull_request_target, workflow_run) always use "
        "ubuntu-latest." in source
    )


@pytest.mark.parametrize(
    "workflow",
    sorted(
        {
            WORKFLOWS / name
            for name in (
                "auto-release.yml",
                "rapid-mac-release.yml",
                "publish.yml",
                "pages.yml",
                "aggregate-bench.yml",
                "post-dmg-engine-recovery.yml",
                "mirror-drift-check.yml",
                "validate-community-submission.yml",
                "agent-gate.yml",
            )
        }
        | set(WORKFLOWS.glob("release-*.yml"))
    ),
    ids=lambda path: path.name,
)
def test_excluded_workflows_do_not_reference_linux_runner_variable(workflow):
    assert "CI_LINUX_RUNNER" not in workflow.read_text()
