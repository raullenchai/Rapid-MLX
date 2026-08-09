# SPDX-License-Identifier: Apache-2.0
"""Static contract for completeness-gated GUI absence assertions."""

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
