"""Dependency floor contracts for upstream model support."""

from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]


def _project_dependencies() -> dict[str, str]:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    dependencies = {}
    for raw in pyproject["project"]["dependencies"]:
        name = raw.split("[", 1)[0].split(">", 1)[0].split("=", 1)[0].strip()
        dependencies[name] = raw
    return dependencies


def _version_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split("."))


def _has_minimum(requirement: str, minimum: str) -> bool:
    needle = ">="
    assert needle in requirement
    version = requirement.split(needle, 1)[1].split(",", 1)[0].strip()
    return _version_tuple(version) >= _version_tuple(minimum)


def test_mlx_vlm_floor_includes_step37_flash_support_and_followups():
    dependencies = _project_dependencies()

    assert _has_minimum(dependencies["mlx-vlm"], "0.6.2")


def test_mlx_lm_floor_matches_current_mlx_vlm_runtime_requirement():
    dependencies = _project_dependencies()

    assert _has_minimum(dependencies["mlx-lm"], "0.31.3")
