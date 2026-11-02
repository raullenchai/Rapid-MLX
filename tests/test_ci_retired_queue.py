"""The active gates have no retired integration paths and fail closed."""

import json
import re
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def jobs(name):
    return yaml.safe_load((ROOT / ".github/workflows" / name).read_text())["jobs"]


def test_retired_queue_is_absent():
    assert not (ROOT / ".mergify.yml").exists()
    for path in (ROOT / ".github/workflows").glob("*.y*ml"):
        text = path.read_text()
        assert "mergify" not in text.lower(), path
        assert "merge_group" not in yaml.load(text, Loader=yaml.BaseLoader)["on"], path
        for job in yaml.safe_load(text).get("jobs", {}).values():
            assert not str(job.get("if", "")).strip().endswith(("&&", "||")), path
        for dead in (
            "reuse_ci",
            "reuse_mac",
            "source_preflight",
            "source_canary",
            "candidate_reduced",
            "candidate_shadow",
            "queue-tree-evidence",
        ):
            assert dead not in text, (path, dead)


def verdict(workflow, name, values, env=None, cancelled=False):
    failed = False
    for step in jobs(workflow)[name]["steps"]:
        if step.get("if") == "${{ !cancelled() }}" and cancelled:
            continue
        if step.get("if") == "${{ cancelled() }}" and not cancelled:
            continue
        command = re.sub(
            r"\$\{\{\s*(.*?)\s*\}\}", lambda m: values.get(m[1], ""), step["run"]
        )
        result = subprocess.run(
            ["bash", "-c", command], env=env, capture_output=True, text=True
        )
        failed |= result.returncode != 0
    return not failed


ENGINE = (
    "lint",
    "engine-contracts",
    "type-check",
    "mlx-bound-guard",
    "test-matrix",
    "linux-coverage",
    "test-apple-silicon",
    "test-apple-silicon-mac-only",
    "changed-lines-coverage",
    "l1-smoke",
)
DESKTOP = (
    "IDENTIFIERS",
    "IDENTIFIER_TESTS",
    "HARNESS_CONTRACTS",
    "BUILD",
    "RELEASE_BUILD",
    "GUI_APP_BUILD",
    "GOLDEN_FLOWS",
)


@pytest.mark.parametrize(
    "event,full",
    [
        ("pull_request", False),
        ("pull_request", True),
        ("push", True),
        ("schedule", True),
        ("workflow_dispatch", True),
    ],
)
@pytest.mark.parametrize("bad", ["failure", "cancelled", "skipped", ""])
def test_engine_aggregate_requires_every_expected_job(event, full, bad):
    values = {
        "github.event_name": event,
        "needs.changes.result": "success",
        "needs.changes.outputs.engine": "true",
        "needs.changes.outputs.full_gate": str(full).lower(),
    }
    required = set(ENGINE)
    if event != "pull_request":
        required -= {"mlx-bound-guard", "changed-lines-coverage"}
    if not full:
        required.remove("l1-smoke")
    values.update(
        {
            f"needs.{name}.result": "success" if name in required else "skipped"
            for name in ENGINE
        }
    )
    assert verdict("ci.yml", "tests", values)
    for name in required | {"changes"}:
        assert not verdict("ci.yml", "tests", values | {f"needs.{name}.result": bad}), (
            name
        )
    assert not verdict("ci.yml", "tests", values, cancelled=True)
    if not full:
        assert not verdict(
            "ci.yml", "tests", values | {"needs.l1-smoke.result": "success"}
        )


@pytest.mark.parametrize("flow_count", [0, 1])
@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("bad", ["failure", "cancelled", "skipped", ""])
def test_desktop_aggregate_requires_every_expected_job(full, bad, flow_count):
    env = {
        "DESKTOP_EXPECTED": "true",
        "FULL_GATE": str(full).lower(),
        "GUI_FLOW_COUNT": str(flow_count),
    }
    required = set(DESKTOP if full and flow_count else DESKTOP[:5])
    env.update({name: "success" if name in required else "skipped" for name in DESKTOP})
    values = {"needs.changes.result": "success"}
    assert verdict("rapid-mac-ci.yml", "desktop-tests", values, env)
    for name in required:
        assert not verdict(
            "rapid-mac-ci.yml", "desktop-tests", values, env | {name: bad}
        ), name
    assert not verdict(
        "rapid-mac-ci.yml", "desktop-tests", {"needs.changes.result": bad}, env
    )
    assert not verdict("rapid-mac-ci.yml", "desktop-tests", values, env, cancelled=True)
    if not full or not flow_count:
        for name in DESKTOP[5:]:
            assert not verdict(
                "rapid-mac-ci.yml", "desktop-tests", values, env | {name: "success"}
            )


@pytest.mark.parametrize("event", ["push", "schedule", "workflow_dispatch"])
def test_non_pr_engine_route_always_runs_full_gate(tmp_path, event):
    output = tmp_path / "outputs"
    step = next(
        step
        for step in jobs("ci.yml")["changes"]["steps"]
        if step.get("name") == "Classify validation lanes"
    )
    subprocess.run(
        ["bash", "-c", step["run"]],
        env={"EVENT_NAME": event, "GITHUB_OUTPUT": str(output)},
        check=True,
    )
    values = dict(line.split("=", 1) for line in output.read_text().splitlines())
    assert values["engine"] == values["desktop"] == values["full_gate"] == "true"
    assert values["linux_matrix_mode"] == "full"
    assert len(json.loads(values["test_matrix"])["include"]) == 6
    assert len(json.loads(values["l1_matrix"])["include"]) == 5
