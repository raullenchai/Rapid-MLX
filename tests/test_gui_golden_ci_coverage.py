# SPDX-License-Identifier: Apache-2.0
"""The macOS golden gate must not silently omit named GUI journeys."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
HARNESS = ROOT / "apps/rapid-mac/scripts/gui-golden-flows.sh"
WORKFLOW = ROOT / ".github/workflows/rapid-mac-ci.yml"

# `chat-depth` requires all five turns to be simultaneously realised in AX.
# The hosted runner's 1024x681 app window virtualises the oldest messages, so
# that assertion is valid on larger local displays but false by construction in
# CI. Keep this exception exact: any additional omission is accidental.
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


def test_failure_diagnostic_regenerates_every_ci_baseline_and_nothing_else():
    assert diagnostic_flows() == workflow_flows() & baseline_flows()


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
    assert f"expected = {len(workflow_flows())}" in str(verdict.get("run", ""))
