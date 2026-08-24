# SPDX-License-Identifier: Apache-2.0
"""Fail-closed contracts for source-aware Desktop GUI journey routing."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from scripts.select_gui_flows import all_flows, select, shard_matrix

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "apps/rapid-mac/Tests/GUIGoldenFlows/journeys.yaml"
WORKFLOW = ROOT / ".github/workflows/rapid-mac-ci.yml"
HARNESS = ROOT / "apps/rapid-mac/scripts/gui-golden-flows.sh"


def _groups() -> dict[str, set[str]]:
    journeys = yaml.safe_load(MANIFEST.read_text())["journeys"]
    result: dict[str, set[str]] = {}
    for journey in journeys:
        if journey["ci_tier"] == "pr":
            result.setdefault(journey["group"], set()).add(journey["name"])
    return result


def test_chat_component_selects_the_complete_chat_group():
    selected = set(select(["apps/rapid-mac/Sources/Rapid/Chat/ChatViewModel.swift"]))
    assert _groups()["chat"] <= selected
    assert "image-generation" not in selected


def test_image_component_selects_only_the_image_group():
    assert set(select(["apps/rapid-mac/Sources/Rapid/Images/ImageClient.swift"])) == {
        "image-generation"
    }


def test_user_facing_image_and_audio_controls_select_their_journeys():
    assert "image-generation" in select(
        ["apps/rapid-mac/Sources/Rapid/UI/ImagesView.swift"]
    )
    assert _groups()["audio"] <= set(
        select(["apps/rapid-mac/Sources/Rapid/UI/DictationView.swift"])
    )


def test_shared_ui_expands_to_every_declared_consumer_group():
    selected = set(
        select(["apps/rapid-mac/Sources/Rapid/UI/Components/EmptyState.swift"])
    )
    assert selected == set(all_flows())


def test_unknown_desktop_path_fails_closed():
    assert select(["apps/rapid-mac/Sources/NewSurface.swift"]) == all_flows()


def test_new_file_cannot_inherit_ownership_from_mixed_ui_directory():
    assert (
        select(["apps/rapid-mac/Sources/Rapid/UI/NewImagesToolbar.swift"])
        == all_flows()
    )


def test_unmapped_top_level_presentation_file_fails_closed():
    assert select(["apps/rapid-mac/Sources/Rapid/AboutPanel.swift"]) == all_flows()


def test_file_prefix_collision_does_not_inherit_explicit_file_ownership():
    assert (
        select(["apps/rapid-mac/Sources/Rapid/UI/ChatView.swift.preview"])
        == all_flows()
    )


@pytest.mark.parametrize(
    "shared_control",
    ["ReadinessBanner.swift", "ModelPickerBar.swift", "InstructionTextEditor.swift"],
)
def test_shared_controls_without_explicit_ownership_run_every_flow(shared_control):
    assert select([f"apps/rapid-mac/Sources/Rapid/UI/{shared_control}"]) == all_flows()


def test_empty_or_invalid_diff_fails_closed():
    assert select([]) == all_flows()
    assert select(["docs/readme.md"]) == all_flows()


def test_harness_manifest_and_workflow_changes_fail_closed():
    for path in (
        "apps/rapid-mac/Tests/GUIGoldenFlows/journeys.yaml",
        "apps/rapid-mac/scripts/gui-golden-flows.sh",
        ".github/workflows/rapid-mac-ci.yml",
        "scripts/select_gui_flows.py",
    ):
        assert select([path]) == all_flows()


def test_cli_emits_compact_json_and_github_outputs(tmp_path: Path):
    output = tmp_path / "github-output"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/select_gui_flows.py"),
            "--github-output",
            str(output),
            "apps/rapid-mac/Sources/Rapid/Images/ImageClient.swift",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout == ""
    lines = output.read_text().splitlines()
    assert json.loads(lines[0].removeprefix("gui_flows=")) == ["image-generation"]
    assert lines[1] == "gui_flow_count=1"
    assert json.loads(lines[2].removeprefix("gui_shards=")) == {
        "include": [
            {
                "group": "images",
                "gui_flows": '["image-generation"]',
                "flow_count": 1,
            }
        ]
    }


def test_shards_partition_every_selected_flow_once_by_manifest_group():
    selected = all_flows()
    matrix = shard_matrix(selected)
    shards = matrix["include"]
    assert [shard["group"] for shard in shards] == sorted(_groups())

    flattened = [
        flow for shard in shards for flow in json.loads(str(shard["gui_flows"]))
    ]
    assert set(flattened) == set(selected)
    assert len(flattened) == len(set(flattened))
    assert all(
        group_flows == sorted(group_flows, key=selected.index)
        for group_flows in (json.loads(str(shard["gui_flows"])) for shard in shards)
    )
    assert all(
        shard["flow_count"] == len(json.loads(str(shard["gui_flows"])))
        for shard in shards
    )


def test_shard_matrix_rejects_empty_or_unknown_flow_sets():
    with pytest.raises(ValueError, match="known PR journeys"):
        shard_matrix([])
    with pytest.raises(ValueError, match="known PR journeys"):
        shard_matrix(["not-a-flow"])


def test_unselected_harness_step_is_a_true_noop(tmp_path: Path):
    output = tmp_path / "must-not-exist"
    result = subprocess.run(
        ["bash", str(HARNESS), "--flow", "fresh-install"],
        env={
            **os.environ,
            "GUI_FLOWS": '["image-generation"]',
            "RAPID_GUI_GOLDEN_OUT": str(output),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "SKIP" in result.stdout
    assert not output.exists()


def test_workflow_passes_real_diff_to_router_and_consumes_its_outputs():
    workflow = WORKFLOW.read_text()
    assert "python scripts/select_gui_flows.py" in workflow
    assert "--paths-file /tmp/changed-paths" in workflow
    assert "gui_flows: ${{ steps.policy.outputs.gui_flows }}" in workflow
    assert "gui_shards: ${{ steps.policy.outputs.gui_shards }}" in workflow
    assert "matrix: ${{ fromJSON(needs.changes.outputs.gui_shards) }}" in workflow
    assert "GUI_FLOWS: ${{ matrix.gui_flows }}" in workflow
    assert "EXPECTED_FLOW_COUNT: ${{ matrix.flow_count }}" in workflow
    assert "fail-fast: false" in workflow
