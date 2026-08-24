#!/usr/bin/env python3
"""Validate and render the model-management architecture manifest."""

from __future__ import annotations

import argparse
import html
import sys
import textwrap
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
ARCH_DIR = ROOT / "docs/engineering/architecture/model-management"
MANIFEST = ARCH_DIR / "architecture.yaml"
README = ARCH_DIR / "README.md"
GENERATED_DIR = ARCH_DIR / "generated"
MMD = GENERATED_DIR / "architecture.mmd"
SVG = GENERATED_DIR / "architecture.svg"

STATUS_STYLE = {
    "implemented": ("#dcfce7", "#16a34a", "🟢"),
    "partial": ("#fef3c7", "#d97706", "🟡"),
    "in_progress": ("#dbeafe", "#2563eb", "🔵"),
    "migrating": ("#f3e8ff", "#9333ea", "🟣"),
    "planned": ("#f1f5f9", "#64748b", "⚪"),
    "blocked": ("#fee2e2", "#dc2626", "🔴"),
    "deprecated": ("#e5e7eb", "#374151", "⚫"),
}


def load_manifest() -> dict[str, Any]:
    data = yaml.safe_load(MANIFEST.read_text())
    if not isinstance(data, dict):
        raise ValueError("architecture.yaml must contain a mapping")
    return data


def validate(data: dict[str, Any]) -> None:
    errors: list[str] = []
    if data.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    lanes = data.get("lanes", {})
    components = data.get("components", {})
    allowed = set(data.get("status_values", []))
    if allowed != set(STATUS_STYLE):
        errors.append("status_values must match the renderer's supported statuses")
    if not isinstance(components, dict) or not components:
        errors.append("components must be a non-empty mapping")
        components = {}

    for component_id, component in components.items():
        prefix = f"components.{component_id}"
        if component.get("lane") not in lanes:
            errors.append(f"{prefix}.lane references an unknown lane")
        if component.get("status") not in allowed:
            errors.append(f"{prefix}.status is invalid")
        if component.get("owner") not in {"atlas", "pixel", "vector", "harbor", "echo"}:
            errors.append(f"{prefix}.owner is invalid")
        if component.get("phase") not in data.get("phases", {}):
            errors.append(f"{prefix}.phase references an unknown phase")
        for key in ("name", "summary", "missing"):
            if not component.get(key):
                errors.append(f"{prefix}.{key} is required")
        for path_string in component.get("files", []) + component.get("tests", []):
            if not (ROOT / path_string).exists():
                errors.append(
                    f"{prefix} references missing repository path: {path_string}"
                )
        for path_string in component.get("evidence", []) + component.get(
            "migrations", []
        ):
            if not (ARCH_DIR / path_string).exists():
                errors.append(
                    f"{prefix} references missing architecture path: {path_string}"
                )

    for index, relation in enumerate(data.get("relations", [])):
        if not isinstance(relation, list) or len(relation) != 3:
            errors.append(f"relations[{index}] must be [source, target, label]")
            continue
        source, target, _label = relation
        if source not in components:
            errors.append(f"relations[{index}] has unknown source: {source}")
        if target not in components:
            errors.append(f"relations[{index}] has unknown target: {target}")

    if errors:
        raise ValueError("\n".join(f"- {error}" for error in errors))


def mermaid(data: dict[str, Any]) -> str:
    lines = ["flowchart LR"]
    components = data["components"]
    for lane_id, lane_name in data["lanes"].items():
        lines.append(f'  subgraph lane_{lane_id}["{lane_name}"]')
        for component_id, component in components.items():
            if component["lane"] != lane_id:
                continue
            label = component["name"].replace('"', "'")
            status = component["status"]
            lines.append(f'    {component_id}["{label}<br/><small>{status}</small>"]')
        lines.append("  end")
    lines.append("")
    for source, target, label in data["relations"]:
        lines.append(f"  {source} -->|{label}| {target}")
    lines.append("")
    for status, (fill, stroke, _icon) in STATUS_STYLE.items():
        lines.append(
            f"  classDef {status} fill:{fill},stroke:{stroke},color:#172033,stroke-width:2px"
        )
    for component_id, component in components.items():
        lines.append(f"  class {component_id} {component['status']}")
    return "\n".join(lines) + "\n"


def _replace_generated(text: str, section: str, body: str) -> str:
    start = f"<!-- architecture-{section}:start -->"
    end = f"<!-- architecture-{section}:end -->"
    before, separator, rest = text.partition(start)
    if not separator or end not in rest:
        raise ValueError(f"README is missing generated markers for {section}")
    _old, separator, after = rest.partition(end)
    return f"{before}{start}\n{body.rstrip()}\n{separator}{after}"


def status_table(data: dict[str, Any]) -> str:
    rows = [
        "<!-- Generated by scripts/render_model_management_architecture.py. Do not edit. -->",
        "| Component | Status | Phase | Owner | Implementation / evidence | Remaining work |",
        "| --- | --- | ---: | --- | --- | --- |",
    ]
    for component in data["components"].values():
        status = component["status"]
        icon = STATUS_STYLE[status][2]
        refs: list[str] = []
        for key in ("files", "tests"):
            refs.extend(
                f"[`{Path(path).name}`](../../../../{path})"
                for path in component.get(key, [])
            )
        refs.extend(f"[evidence]({path})" for path in component.get("evidence", []))
        refs.extend(f"[migration]({path})" for path in component.get("migrations", []))
        rows.append(
            f"| {component['name']} | {icon} `{status}` | {component['phase']} | "
            f"{component['owner'].title()} | {' · '.join(refs) or '—'} | {component['missing']} |"
        )
    return "\n".join(rows)


def phase_table(data: dict[str, Any]) -> str:
    rows = [
        "<!-- Generated by scripts/render_model_management_architecture.py. Do not edit. -->",
        "| Phase | Name | Exit condition |",
        "| ---: | --- | --- |",
    ]
    for phase_id, phase in data["phases"].items():
        rows.append(f"| {phase_id} | {phase['name']} | {phase['exit']} |")
    return "\n".join(rows)


def readme(data: dict[str, Any], diagram: str) -> str:
    rendered = README.read_text()
    rendered = _replace_generated(
        rendered,
        "diagram",
        "<!-- Generated by scripts/render_model_management_architecture.py. Do not edit. -->\n"
        f"```mermaid\n{diagram.rstrip()}\n```",
    )
    rendered = _replace_generated(rendered, "status", status_table(data))
    return _replace_generated(rendered, "phases", phase_table(data))


def svg(data: dict[str, Any]) -> str:
    lanes = list(data["lanes"].items())
    by_lane = {
        lane_id: [
            (component_id, component)
            for component_id, component in data["components"].items()
            if component["lane"] == lane_id
        ]
        for lane_id, _ in lanes
    }
    lane_width, gutter, left, top = 310, 18, 28, 110
    node_height, node_gap = 102, 20
    max_nodes = max(len(nodes) for nodes in by_lane.values())
    width = left * 2 + len(lanes) * lane_width + (len(lanes) - 1) * gutter
    height = top + max_nodes * (node_height + node_gap) + 90
    positions: dict[str, tuple[float, float]] = {}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="#f8fbff"/>',
        '<style>text{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}.title{font-size:25px;font-weight:700;fill:#14213d}.lane{font-size:16px;font-weight:700;fill:#243b63}.node{font-size:14px;font-weight:650;fill:#172033}.status{font-size:12px;fill:#475569}.edge{stroke:#94a3b8;stroke-width:1.4;fill:none;marker-end:url(#arrow)}</style>',
        '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><path d="M0,0 L0,6 L8,3 z" fill="#64748b"/></marker></defs>',
        f'<text x="{left}" y="42" class="title">{html.escape(data["title"])}</text>',
        f'<text x="{left}" y="70" class="status">architecture v{html.escape(data["architecture_version"])} · generated from architecture.yaml</text>',
    ]
    for lane_index, (lane_id, lane_name) in enumerate(lanes):
        x = left + lane_index * (lane_width + gutter)
        parts.append(
            f'<rect x="{x}" y="88" width="{lane_width}" height="{height - 116}" rx="14" fill="#ffffff" stroke="#cbd5e1"/>'
        )
        parts.append(
            f'<text x="{x + 14}" y="118" class="lane">{html.escape(lane_name)}</text>'
        )
        for node_index, (component_id, component) in enumerate(by_lane[lane_id]):
            y = top + 28 + node_index * (node_height + node_gap)
            fill, stroke, icon = STATUS_STYLE[component["status"]]
            positions[component_id] = (x + lane_width / 2, y + node_height / 2)
            parts.append(
                f'<rect x="{x + 12}" y="{y}" width="{lane_width - 24}" height="{node_height}" rx="10" fill="{fill}" stroke="{stroke}" stroke-width="2"/>'
            )
            name_lines = textwrap.wrap(
                component["name"].replace(" → ", " →\n"),
                width=34,
                break_long_words=False,
            )[:2]
            parts.append(f'<text x="{x + 26}" y="{y + 28}" class="node">')
            for line_index, line in enumerate(name_lines):
                parts.append(
                    f'<tspan x="{x + 26}" y="{y + 28 + line_index * 20}">{html.escape(line)}</tspan>'
                )
            parts.append("</text>")
            parts.append(
                f'<text x="{x + 26}" y="{y + 78}" class="status">{icon} {component["status"]} · phase {component["phase"]} · {component["owner"]}</text>'
            )
    edge_parts: list[str] = []
    for source, target, _label in data["relations"]:
        sx, sy = positions[source]
        tx, ty = positions[target]
        edge_parts.append(
            f'<path d="M{sx},{sy} L{tx},{ty}" class="edge" opacity=".42"/>'
        )
    parts[6:6] = edge_parts
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check", action="store_true", help="fail if generated files drift"
    )
    args = parser.parse_args()
    try:
        data = load_manifest()
        validate(data)
        diagram = mermaid(data)
        outputs = {
            MMD: diagram,
            SVG: svg(data),
            README: readme(data, diagram),
        }
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"architecture validation failed:\n{exc}", file=sys.stderr)
        return 1

    drift = [
        path
        for path, content in outputs.items()
        if not path.exists() or path.read_text() != content
    ]
    if args.check:
        if drift:
            print("generated architecture files are stale:", file=sys.stderr)
            for path in drift:
                print(f"- {path.relative_to(ROOT)}", file=sys.stderr)
            print(
                "run: python3 scripts/render_model_management_architecture.py",
                file=sys.stderr,
            )
            return 1
        print("model-management architecture is valid and up to date")
        return 0

    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    for path, content in outputs.items():
        path.write_text(content)
        print(f"wrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
