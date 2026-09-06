# SPDX-License-Identifier: Apache-2.0
"""Keep the README image matrix aligned with the shipping alias catalog."""

from __future__ import annotations

from pathlib import Path

from vllm_mlx.cli import _available_models_json_payload

ROOT = Path(__file__).resolve().parents[1]
START = "<!-- image-model-matrix:start -->"
END = "<!-- image-model-matrix:end -->"
RELEASE_START = "<!-- image-release-matrix:start -->"
RELEASE_END = "<!-- image-release-matrix:end -->"


def _matrix_rows() -> dict[str, list[str]]:
    readme = (ROOT / "README.md").read_text()
    assert readme.count(START) == 1
    assert readme.count(END) == 1
    table = readme.split(START, 1)[1].split(END, 1)[0]
    rows: dict[str, list[str]] = {}
    for line in table.splitlines():
        if not line.startswith("| `"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        alias = cells[0].strip("`")
        assert alias not in rows, f"duplicate README image alias: {alias}"
        rows[alias] = cells
    return rows


def _gib(size_bytes: int) -> str:
    return f"{size_bytes / 1024**3:.1f} GiB"


def _release_matrix_aliases() -> set[str]:
    runbook = (
        ROOT / "docs/engineering/operations/image-release-dogfood-matrix.md"
    ).read_text()
    assert runbook.count(RELEASE_START) == 1
    assert runbook.count(RELEASE_END) == 1
    table = runbook.split(RELEASE_START, 1)[1].split(RELEASE_END, 1)[0]
    return {
        line.split("|", 2)[1].strip().strip("`")
        for line in table.splitlines()
        if line.startswith("| `")
    }


def test_readme_image_matrix_matches_shipping_catalog() -> None:
    catalog = {
        entry["alias"]: entry for entry in _available_models_json_payload()["image"]
    }
    atomic = {
        entry["alias"]: entry["capabilities"]["operation_modes"]
        for entry in _available_models_json_payload()["atomic"]["snapshot"]["aliases"]
        if entry["alias"] in catalog
    }
    rows = _matrix_rows()

    assert set(rows) == set(catalog)
    assert _release_matrix_aliases() == set(catalog)
    assert next(iter(rows)) == "flux2-klein-4b"
    for alias, entry in catalog.items():
        cells = rows[alias]
        assert len(cells) == 6
        operations = set(atomic[alias])
        if {"text_to_image", "image_to_image"}.issubset(operations):
            expected_modes = "Generate + edit"
        elif "image_to_image" in operations:
            expected_modes = "Edit"
        else:
            expected_modes = "Generate"
        assert cells[2] == expected_modes
        assert cells[3] == _gib(entry["size_bytes"])
        assert cells[4] == f"{entry['min_memory_gb']:g} GB"
        assert cells[5] == str(entry["default_steps"])


def test_readme_alias_totals_match_shipping_catalog() -> None:
    payload = _available_models_json_payload()
    counts = {kind: len(payload[kind]) for kind in ("text", "image", "video", "audio")}
    total = sum(counts.values())
    expected = (
        f"{counts['text']} text + {counts['image']} image + "
        f"{counts['video']} video + {counts['audio']} audio aliases, {total} total"
    )

    assert expected in (ROOT / "README.md").read_text()
