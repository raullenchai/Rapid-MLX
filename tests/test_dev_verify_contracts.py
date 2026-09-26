# SPDX-License-Identifier: Apache-2.0
"""Sync contracts binding dev_verify's static tables to their sources of truth.

The Desktop issue-form areas, the journey manifest, and the local verification
command's area mapping must agree in both directions. Drift in any of them
would silently mis-route Desktop reports, so every direction is asserted here
and the mapping is kept adjacent to a documented handling rule for areas
without relevant automated journeys.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from scripts.dev_verify import AREA_JOURNEYS, AreaMapping, journey_index

ROOT = Path(__file__).resolve().parent.parent
DESKTOP_BUG_FORM = ROOT / ".github/ISSUE_TEMPLATE/desktop_bug.yml"
MANIFEST = ROOT / "apps/rapid-mac/Tests/GUIGoldenFlows/journeys.yaml"
REPORTS = ROOT / "tests/fixtures/dev_verify/desktop_reports.yaml"


def form_areas() -> list[str]:
    """The `area` dropdown options, in form order, parsed from the YAML."""
    form = yaml.safe_load(DESKTOP_BUG_FORM.read_text())
    area_input = next(
        block
        for block in form["body"]
        if block.get("type") == "dropdown" and block["id"] == "area"
    )
    return area_input["attributes"]["options"]


def test_area_mapping_keys_equal_the_issue_form_options_exactly():
    assert set(AREA_JOURNEYS) == set(form_areas()), (
        "desktop_bug.yml areas and dev_verify AREA_JOURNEYS drifted; update"
        " both together and the ten-report fixture"
    )


def test_every_mapped_journey_exists_in_the_manifest():
    known = journey_index()
    for area, mapping in AREA_JOURNEYS.items():
        for name in mapping.journeys:
            assert name in known, f"area '{area}' references unknown journey {name}"
            journey = known[name]
            assert journey["driver"] in ("ax", "hybrid", "swift")
            assert journey["ci_tier"] in ("pr", "local"), (
                f"area '{area}' references retired/nightly journey {name}"
            )


def test_every_pr_tier_bash_journey_is_reachable_from_some_area():
    """The inventory must stay triage-reachable: a journey no report area can
    reach is coverage that issue intake can never select on purpose."""
    known = journey_index()
    reachable: set[str] = set()
    for mapping in AREA_JOURNEYS.values():
        reachable.update(mapping.journeys)
    for name, journey in known.items():
        if journey["ci_tier"] == "pr" and journey["driver"] != "swift":
            assert name in reachable, (
                f"pr-tier harness journey {name} is unreachable from every"
                " Desktop area; add it to the matching AREA_JOURNEYS entry"
            )


def test_mapping_records_are_wellformed():
    for area, mapping in AREA_JOURNEYS.items():
        assert isinstance(mapping, AreaMapping)
        assert mapping.journeys, f"area '{area}' maps to no journeys"
        assert len(mapping.journeys) == len(set(mapping.journeys))
        if area == "Somewhere else / not sure":
            # The broad area must remain genuinely broad: every pr-tier bash
            # journey is a candidate when the reporter could not localize.
            known = journey_index()
            bash_pr = {
                name
                for name, journey in known.items()
                if journey["ci_tier"] == "pr" and journey["driver"] != "swift"
            }
            assert bash_pr <= set(mapping.journeys)


def test_gap_handling_is_documented_for_uncovered_areas():
    """An area without a relevant automated journey must carry an explicit
    gap string: the documented handling is to reproduce manually and report
    the gap, never to imply journey coverage."""
    with_gaps = {area for area, m in AREA_JOURNEYS.items() if m.gaps}
    assert "Speed — answers arrive slowly, or the app feels sluggish" in with_gaps
    assert "Installing or launching for the first time" in with_gaps


def test_historical_report_fixture_stays_in_sync():
    reports = yaml.safe_load(REPORTS.read_text())
    areas = set(form_areas())
    known = journey_index()
    for entry in reports:
        assert entry["area"] in areas, f"#{entry['issue']} uses a retired area"
        for name in entry.get("expected_journeys") or []:
            assert name in known, f"#{entry['issue']} expects unknown journey"
            mapping = AREA_JOURNEYS[entry["area"]]
            assert name in mapping.journeys, (
                f"#{entry['issue']} expects {name}, which its area no longer"
                " maps; update the mapping and the fixture together"
            )
        if entry.get("uncovered"):
            assert entry.get("reason"), "uncovered entries need a reason"
