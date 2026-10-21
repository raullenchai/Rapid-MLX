# SPDX-License-Identifier: Apache-2.0
"""Keep image release coverage and the README catalog path current."""

from __future__ import annotations

from pathlib import Path

from rapid_mlx.cli import _available_models_json_payload

ROOT = Path(__file__).resolve().parents[1]
RELEASE_START = "<!-- image-release-matrix:start -->"
RELEASE_END = "<!-- image-release-matrix:end -->"


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


def test_image_release_matrix_matches_shipping_catalog() -> None:
    catalog = {
        entry["alias"]: entry for entry in _available_models_json_payload()["image"]
    }
    assert _release_matrix_aliases() == set(catalog)
    assert catalog


def test_readme_links_to_model_catalog() -> None:
    assert "https://rapidmlx.com/docs/aliases.html" in (ROOT / "README.md").read_text()
