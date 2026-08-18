# SPDX-License-Identifier: Apache-2.0
"""Source guards for GUI journeys that prove control outcomes, not just presses."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
HARNESS = ROOT / "apps/rapid-mac/scripts/gui-golden-flows.sh"
WORKFLOW = ROOT / ".github/workflows/rapid-mac-ci.yml"


def test_audio_readiness_actions_start_both_selected_models_and_clear_the_gate():
    source = HARNESS.read_text()
    flow = source.split("flow_audio_readiness() {", 1)[1].split("\n}", 1)[0]

    assert flow.count('press "$OUT/') >= 3
    assert flow.count('.subcommand == "pull"') == 2
    assert '.alias == "fake-qwen3-tts"' in flow
    assert '.alias == "fake-whisper-small"' in flow
    assert "before its pull completed" in flow
    assert "Speech stayed behind Download & start" in flow
    assert "Transcription stayed behind Download & start" in flow


def test_audio_control_journey_is_blocking_gui_ci_and_has_failure_evidence():
    workflow = WORKFLOW.read_text()

    assert "Golden flow: audio-readiness" in workflow
    assert "--flow audio-readiness" in workflow
    diagnostic = workflow.split("Regenerate baselines on this runner (diagnostic)", 1)[
        1
    ]
    assert "image-generation audio-readiness" in diagnostic


@pytest.mark.parametrize(
    "flow",
    ["no-dead-controls", "catalog-integrity", "update-state", "launch-integrations"],
)
def test_semantic_control_audits_are_blocking_gui_ci(flow: str):
    workflow = WORKFLOW.read_text()

    assert f"Golden flow: {flow}" in workflow
    assert f"--flow {flow}" in workflow
    diagnostic = workflow.split("Regenerate baselines on this runner (diagnostic)", 1)[
        1
    ]
    assert flow in diagnostic
