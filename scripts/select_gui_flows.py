#!/usr/bin/env python3
"""Select Desktop GUI journeys affected by changed repository paths.

The journey manifest is the source of truth. A path selects every PR journey
whose declared source prefix contains it, then expands to the complete journey
group so related controls are exercised together. Paths which can affect the
harness, packaging, shared app surface, or cannot be mapped fail closed to the
entire PR journey set.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterable
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "apps/rapid-mac/Tests/GUIGoldenFlows/journeys.yaml"
DESKTOP_ROOT = "apps/rapid-mac/"
SOURCE_ROOT = f"{DESKTOP_ROOT}Sources/Rapid/"
UNSAFE_BROAD_PREFIXES = {f"{SOURCE_ROOT}UI/"}

# These inputs influence every journey or the routing policy itself. Keep this
# list intentionally broad: an unnecessary full run is cheaper than a false
# green caused by an unsafe narrow match.
FULL_SUITE_PREFIXES = (
    ".github/workflows/rapid-mac-ci.yml",
    "apps/rapid-mac/Package",
    "apps/rapid-mac/Resources/",
    "apps/rapid-mac/Sources/Rapid/UI/Components/",
    "apps/rapid-mac/Tests/GUIGoldenFlows/",
    "apps/rapid-mac/scripts/build",
    "apps/rapid-mac/scripts/fake-rapid-mlx.sh",
    "apps/rapid-mac/scripts/gui-golden-flows.sh",
    "apps/rapid-mac/scripts/rapid-ax.swift",
    "scripts/select_gui_flows.py",
    "tests/test_gui_flow_routing.py",
    "tests/test_gui_golden_ci_coverage.py",
)


def _manifest() -> list[dict[str, object]]:
    """Read the routing fields without adding a network dependency to CI.

    The full manifest schema is validated with PyYAML in gui-harness-contracts.
    The classifier job intentionally remains stdlib-only, so this strict reader
    accepts exactly the simple scalar/inline-list shape used by the routing
    fields and fails the job if that representation changes.
    """

    source = MANIFEST.read_text()
    if not re.search(r"(?m)^version: 1\s*$", source):
        raise ValueError("unsupported GUI journey manifest version")

    journeys: list[dict[str, object]] = []
    for block in re.split(r"(?m)^  - name: ", source)[1:]:
        lines = block.splitlines()
        values: dict[str, object] = {"name": lines[0].strip()}
        for line in lines[1:]:
            match = re.match(r"^    (group|ci_tier): ([a-z-]+)\s*$", line)
            if match:
                values[match.group(1)] = match.group(2)
                continue
            match = re.match(r"^    source_paths: \[(.*)\]\s*$", line)
            if match:
                values["source_paths"] = [
                    item.strip() for item in match.group(1).split(",") if item.strip()
                ]
        required = {"name", "group", "ci_tier", "source_paths"}
        if set(values) != required or not values["source_paths"]:
            raise ValueError(f"malformed GUI journey routing fields: {values['name']}")
        journeys.append(values)
    if not journeys:
        raise ValueError("GUI journey manifest contains no journeys")
    return journeys


def _pr_journeys() -> list[dict[str, object]]:
    return [journey for journey in _manifest() if journey["ci_tier"] == "pr"]


def all_flows() -> list[str]:
    return [str(journey["name"]) for journey in _pr_journeys()]


def shard_matrix(flows: Iterable[str]) -> dict[str, list[dict[str, object]]]:
    """Partition selected flows by their manifest group for a CI matrix."""

    selected = set(flows)
    journeys = _pr_journeys()
    known = {str(journey["name"]) for journey in journeys}
    if not selected or not selected <= known:
        raise ValueError("GUI shard input must contain known PR journeys")

    groups: dict[str, list[str]] = {}
    for journey in journeys:
        name = str(journey["name"])
        if name in selected:
            groups.setdefault(str(journey["group"]), []).append(name)

    return {
        "include": [
            {
                "group": group,
                "gui_flows": json.dumps(group_flows, separators=(",", ":")),
                "flow_count": len(group_flows),
            }
            for group, group_flows in sorted(groups.items())
        ]
    }


def _matches(path: str, declared: str) -> bool:
    # UI is a mixed-responsibility directory. A new file under it cannot
    # inherit ownership from the broad inventory prefix: only an explicit file
    # or a narrower declared subdirectory may route narrowly. Cohesive domain
    # folders such as Chat/, Audio/, and Images/ intentionally allow new files
    # to inherit their journey ownership.
    if declared in UNSAFE_BROAD_PREFIXES:
        return False
    normalized = declared.rstrip("/")
    return path == normalized or (declared.endswith("/") and path.startswith(declared))


def select(paths: Iterable[str]) -> list[str]:
    normalized = sorted(
        {path.strip().removeprefix("./") for path in paths if path.strip()}
    )
    journeys = _pr_journeys()
    if not normalized:
        return all_flows()

    selected_groups: set[str] = set()
    for path in normalized:
        if path.startswith(FULL_SUITE_PREFIXES):
            return all_flows()

        # Non-Desktop files do not narrow the Desktop lane. They can coexist
        # with Desktop files in a cross-cutting PR and are handled by their own
        # CI lane.
        if not path.startswith(DESKTOP_ROOT):
            continue

        matches = [
            journey
            for journey in journeys
            if any(_matches(path, str(prefix)) for prefix in journey["source_paths"])
        ]
        if not matches:
            # New Desktop production/support paths are unsafe until classified.
            return all_flows()
        selected_groups.update(str(journey["group"]) for journey in matches)

    if not selected_groups:
        # The caller expected a Desktop lane but supplied no classifiable
        # Desktop path. Treat an invalid diff as a full-suite request.
        return all_flows()

    return [
        str(journey["name"])
        for journey in journeys
        if str(journey["group"]) in selected_groups
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", help="Changed repository-relative paths")
    parser.add_argument("--paths-file", type=argparse.FileType("r"))
    parser.add_argument("--github-output", type=argparse.FileType("a"))
    args = parser.parse_args()

    paths = list(args.paths)
    if args.paths_file:
        paths.extend(args.paths_file.read().splitlines())
    flows = select(paths)
    payload = json.dumps(flows, separators=(",", ":"))
    shards = json.dumps(shard_matrix(flows), separators=(",", ":"))

    if args.github_output:
        print(f"gui_flows={payload}", file=args.github_output)
        print(f"gui_flow_count={len(flows)}", file=args.github_output)
        print(f"gui_shards={shards}", file=args.github_output)
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
