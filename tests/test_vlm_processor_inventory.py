# SPDX-License-Identifier: Apache-2.0
"""Offline checks for the generated catalog qualification."""

import importlib.util
import json
import re
from pathlib import Path

import pytest

from rapid_mlx.utils.model_file_guard import validate_vlm_processor_code

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/vlm_processor_inventory.json"
RERUN = "Re-run python -m scripts.vlm_processor_inventory"


@pytest.fixture
def inventory():
    assert FIXTURE.exists(), RERUN
    return json.loads(FIXTURE.read_text())


def test_catalog_coverage(inventory):
    catalog = json.loads((ROOT / "rapid_mlx/aliases.json").read_text())
    aliases = inventory["aliases"]
    assert {
        a for a, v in catalog.items() if v.get("supports_image_input") is True
    } <= aliases.keys(), RERUN
    for alias, value in aliases.items():
        assert alias in catalog, f"{alias}: removed; {RERUN}"
        for key in ("hf_path", "revision", "subfolder"):
            assert value.get(key) == catalog[alias].get(key), (
                f"{alias}: stale {key}; {RERUN}"
            )


def test_routes(inventory):
    for group in ("aliases", "auxiliary"):
        for name, value in inventory[group].items():
            assert value.get("route") != "forced-trust", name


def test_recorded_metadata_is_loadable(inventory, tmp_path):
    for alias, value in inventory["aliases"].items():
        path = tmp_path / alias
        path.mkdir(parents=True)
        metadata = {"config.json": {"model_type": value["model_type"]}}
        for name, keys in value["auto_map"].items():
            metadata.setdefault(name, {})["auto_map"] = dict.fromkeys(
                keys, "custom.Code"
            )
        for name, model_file in value["model_file"].items():
            metadata.setdefault(name, {})["model_file"] = model_file
        for name, data in metadata.items():
            (path / name).write_text(json.dumps(data))
        config = metadata["config.json"]
        assert (
            validate_vlm_processor_code(path, config, trust_remote_code=False) is None
        ), alias
        if value["auto_map"] or value["model_file"]:
            with pytest.raises(ValueError, match="explicitly opt in"):
                validate_vlm_processor_code(
                    path, {**config, "model_type": "got"}, trust_remote_code=False
                )


def test_dependency_forced_trust_routes_reviewed():
    if importlib.util.find_spec("mlx_vlm") is None:
        pytest.skip("mlx_vlm is not installed")
    spec = importlib.util.find_spec("mlx_vlm")
    models = Path(next(iter(spec.submodule_search_locations))) / "models"
    reviewed = {
        "got",
        "inkling",
        "moondream2",
        "minicpmv4_6",
        "minicpmo",
        "internvl_chat",
        "z_image",
    }
    hits = set()
    for file in models.glob("*/*.py"):
        for line in file.read_text(encoding="utf-8").splitlines():
            if re.search(r"\btrust_remote_code\s*=\s*True\b", line.split("#", 1)[0]):
                hits.add(file.parent.name)
    assert hits <= reviewed, (
        f"Re-review FORCED_TRUST_PROCESSOR_TYPES: {hits - reviewed}"
    )
    from rapid_mlx.utils.model_file_guard import (
        AUXILIARY_TRUST_PROCESSOR_TYPES,
        FORCED_TRUST_PROCESSOR_TYPES,
    )

    for directory, types in (
        ("got", {"got"}),
        ("inkling", {"inkling", "inkling_mm_model"}),
        ("moondream2", {"moondream1", "moondream2"}),
    ):
        assert types <= FORCED_TRUST_PROCESSOR_TYPES | AUXILIARY_TRUST_PROCESSOR_TYPES
        text = "\n".join(
            p.read_text(encoding="utf-8") for p in (models / directory).glob("*.py")
        )
        registrations = re.findall(
            r"install_auto_processor_patch\s*\(\s*(\[[^\]]*\]|['\"][^'\"]+['\"])",
            text,
            re.S,
        )
        for model_type in types:
            assert any(
                re.search(rf"['\"]{re.escape(model_type)}['\"]", registration)
                for registration in registrations
            ), f"Missing {model_type} registration"
