# SPDX-License-Identifier: Apache-2.0
"""Source guards for GUI journeys that prove control outcomes, not just presses."""

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
HARNESS = ROOT / "apps/rapid-mac/scripts/gui-golden-flows.sh"
WORKFLOW = ROOT / ".github/workflows/rapid-mac-ci.yml"


def test_audio_readiness_actions_start_the_selected_model_and_clear_the_gate():
    source = HARNESS.read_text()
    flow = source.split("flow_audio_readiness() {", 1)[1].split("\n}", 1)[0]

    assert flow.count('press "$OUT/') >= 3
    # Speech synthesis and file transcription each preserve the explicit
    # Download → Start lifecycle. Dictation owns no readiness banner: it loads
    # only after the user invokes the global hotkey.
    assert flow.count('.subcommand == "pull"') == 2
    assert '.alias == "fake-qwen3-tts"' in flow
    assert '.alias == "fake-whisper-small"' in flow
    assert "before its pull completed" in flow
    assert "Speech loaded automatically after a download-only action" in flow
    assert "Opening Audio started a model before any user action" in flow
    assert "Opening Dictation loaded its model before the user dictated" in flow
    assert "Transcription loaded automatically after Download" in flow
    assert "Audio.Transcription.Run" in flow
    assert "Audio.Transcription.Result" in flow


def test_audio_control_journey_is_blocking_gui_ci_and_has_failure_evidence():
    workflow = WORKFLOW.read_text()

    assert "Golden flow: audio-readiness" in workflow
    assert "--flow audio-readiness" in workflow
    diagnostic = workflow.split("Regenerate baselines on this runner (diagnostic)", 1)[
        1
    ]
    assert "image-generation audio-readiness" in diagnostic


SNAPSHOTS = ROOT / "apps/rapid-mac/Tests/GUIGoldenFlows/__Snapshots__"


@pytest.mark.parametrize(
    "flow",
    ["no-dead-controls", "catalog-integrity", "update-state", "launch-integrations"],
)
def test_semantic_control_audits_are_blocking_gui_ci(flow: str):
    """Each semantic audit is gated, and its failure leaves usable evidence.

    What "evidence" means depends on whether the flow owns committed AX
    baselines. update-state and launch-integrations do, so they belong in the
    regenerate-on-failure diagnostic loop (whose exact membership is pinned by
    test_failure_diagnostic_regenerates_every_ci_baseline_and_nothing_else:
    gated flows WITH baselines, nothing else — a snapshot-less audit in that
    loop has nothing to regenerate). no-dead-controls and catalog-integrity
    carry no snapshots; their evidence is the flow's own output directory,
    which must sit inside the artifact uploaded on failure.
    """
    workflow = WORKFLOW.read_text()

    assert f"Golden flow: {flow}" in workflow
    assert f"--flow {flow}" in workflow

    if any(p.name.startswith(flow) for p in SNAPSHOTS.glob("*.txt")):
        diagnostic = workflow.split(
            "Regenerate baselines on this runner (diagnostic)", 1
        )[1]
        assert flow in diagnostic
    else:
        steps = yaml.safe_load(workflow)["jobs"]["gui-golden-flows"]["steps"]
        audit = next(s for s in steps if s.get("name") == f"Golden flow: {flow}")
        out = audit.get("env", {}).get("RAPID_GUI_GOLDEN_OUT", "")
        assert out == "${{ runner.temp }}/golden/" + flow
        upload = next(s for s in steps if s.get("name") == "Upload AX evidence")
        assert upload.get("if") == "failure()"
        paths = [
            ln.strip()
            for ln in str(upload.get("with", {}).get("path", "")).splitlines()
            if ln.strip()
        ]
        assert "${{ runner.temp }}/golden" in paths
