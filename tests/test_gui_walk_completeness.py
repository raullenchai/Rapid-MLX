# SPDX-License-Identifier: Apache-2.0
"""Static contract for completeness-gated GUI absence assertions."""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_rapid_ax_reports_descendant_walk_completeness():
    source = (ROOT / "apps/rapid-mac/scripts/rapid-ax.swift").read_text()
    assert '"walk": [' in source
    assert '"complete": elementWalkComplete' in source
    assert source.count("elementWalkComplete = false") >= 2
    assert "Set<AXUIElement>()" in source
    assert "elementWalkComplete && windowListComplete" in source


def test_catalog_absence_checks_require_complete_walks():
    source = (ROOT / "apps/rapid-mac/scripts/gui-golden-flows.sh").read_text()
    flow = source.split("flow_catalog_integrity() {", 1)[1].split("\n}", 1)[0]
    assert flow.count(".data.walk.complete == true") == 2
    assert flow.count('test("fake-video-alias")') == 2
    assert 'ModelPickerBar.ModelMenu" and .value == "fake-alias' in flow
    assert "Settings.ModelManagement.Row.fake-alias" in flow
    complete_offsets = [
        index
        for index in range(len(flow))
        if flow.startswith(".data.walk.complete == true", index)
    ]
    absence_offsets = [
        index
        for index in range(len(flow))
        if flow.startswith('test("fake-video-alias")', index)
    ]
    assert all(
        complete < absence
        for complete, absence in zip(complete_offsets, absence_offsets, strict=True)
    )


def test_empty_persona_environment_is_safe_on_macos_bash3():
    source = (ROOT / "apps/rapid-mac/scripts/gui-golden-flows.sh").read_text()
    safe = '${PERSONA_ENV[@]+"${PERSONA_ENV[@]}"}'
    assert source.count(safe) == 3

    program = r"""
set -u
PERSONA_ENV=()
empty=0
for value in "${PERSONA_ENV[@]+"${PERSONA_ENV[@]}"}"; do empty=$((empty + 1)); done
PERSONA_ENV=("ONE=two words" "THREE=four")
printf '%s\n' "$empty" "${PERSONA_ENV[@]+"${PERSONA_ENV[@]}"}"
"""
    result = subprocess.run(
        ["/bin/bash", "-c", program],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.splitlines() == ["0", "ONE=two words", "THREE=four"]


def test_semantic_gui_flows_do_not_use_screen_capture_as_window_oracle():
    source = (ROOT / "apps/rapid-mac/scripts/gui-golden-flows.sh").read_text()
    wait = source.split("wait_for_window() {", 1)[1].split("\n}", 1)[0]
    assert '"$AX_DRIVER" dump "$APP_PID"' in wait
    assert ".data.windows.complete == true" in wait
    assert "if flow_requires_screen_recording" in wait
    assert "if ! refresh_main_window_id" in wait
    assert "pb list windows" not in wait

    refresh = source.split("refresh_main_window_id() {", 1)[1].split("\n}", 1)[0]
    assert 'MAIN_WINDOW_ID=""' in refresh
    assert "pb list windows" in refresh
    assert "|| return 1" in refresh

    see = source.split("see_main() {", 1)[1].split("\n}", 1)[0]
    assert "flow_requires_screen_recording && ! refresh_main_window_id" in see

    settings = source.split("open_settings() {", 1)[1].split("\n}", 1)[0]
    assert "ax_window_present Settings" in settings
    assert "if flow_requires_screen_recording" in settings
    assert settings.index("if flow_requires_screen_recording") < settings.index(
        "pb list windows"
    )

    permissions = source.split("require_tools() {", 1)[1].split("\n}", 1)[0]
    assert 'any(.data.permissions[]?; .name == "Accessibility"' in permissions
    screen_condition = permissions.index("if flow_requires_screen_recording")
    required_query = permissions.index("select(.isRequired)")
    assert screen_condition < required_query

    screen_gate = source.split("flow_requires_screen_recording() {", 1)[1].split(
        "\n}", 1
    )[0]
    assert "all) return 0" in screen_gate
    program = f"""
flow_requires_screen_recording() {{{screen_gate}
}}
for FLOW in catalog-integrity settings-persistence fresh-install all; do
    flow_requires_screen_recording
    printf '%s=%s\\n' "$FLOW" "$?"
done
"""
    result = subprocess.run(
        ["/bin/bash", "-c", program],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.splitlines() == [
        "catalog-integrity=1",
        "settings-persistence=1",
        "fresh-install=1",
        "all=0",
    ]


def test_restored_tool_flow_waits_for_relaunched_sidecar_before_sending():
    source = (ROOT / "apps/rapid-mac/scripts/gui-golden-flows.sh").read_text()
    flow = source.split("flow_restored_tools() {", 1)[1].split("\n}", 1)[0]
    opened = flow.index('press "$OUT/restored.json"')
    ready = flow.index('wait_send_idle "$OUT/restored-ready.json"')
    followup = flow.index('send_prompt "What about technology?')
    assert opened < ready < followup
