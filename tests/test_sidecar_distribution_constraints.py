from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
from packaging.requirements import Requirement

SCRIPT = (
    Path(__file__).parents[1]
    / "apps"
    / "rapid-mac"
    / "scripts"
    / "check-sidecar-distributions.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("sidecar_constraints", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def constraints():
    return _load_module()


def test_emitted_constraints_are_stable_and_complete(constraints) -> None:
    emitted = constraints.emit_constraints().splitlines()
    assert emitted[0].startswith("# Release-tested versions")
    assert emitted[1:] == [
        "mlx==0.32.2",
        "transformers==5.12.1",
        "mlx-vlm==0.6.16",
        "mflux==0.19.0",
        "mlx-video-with-audio==0.1.36",
        "mlx-arsenal==0.12.1",
    ]


def test_emit_constraints_needs_only_the_python_standard_library() -> None:
    result = subprocess.run(
        [sys.executable, "-S", str(SCRIPT), "--emit-constraints"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "mlx-vlm==0.6.16" in result.stdout
    assert "mlx-video-with-audio==0.1.36" in result.stdout


def test_exact_reduced_vision_runtime_is_allowed(constraints) -> None:
    assert constraints.is_validated_compatibility_exception(
        owner="mlx-vlm",
        owner_version="0.6.16",
        requirement=Requirement("transformers>=5.14.0"),
        actual="5.12.1",
    )


@pytest.mark.parametrize(
    ("owner", "owner_version", "requirement", "actual"),
    [
        ("other", "0.6.16", "transformers>=5.14.0", "5.12.1"),
        ("mlx-vlm", "0.6.15", "transformers>=5.14.0", "5.12.1"),
        ("mlx-vlm", "0.6.17", "transformers>=5.14.0", "5.12.1"),
        ("mlx-vlm", "0.6.16", "transformers>=5.13.0", "5.12.1"),
        ("mlx-vlm", "0.6.16", "other>=5.14.0", "5.12.1"),
        ("mlx-vlm", "0.6.16", "transformers>=5.14.0", "5.12.0"),
        ("mlx-vlm", "0.6.16", "transformers>=5.14.0", "5.13.1"),
    ],
)
def test_every_nearby_mismatch_still_fails_closed(
    constraints, owner: str, owner_version: str, requirement: str, actual: str
) -> None:
    assert not constraints.is_validated_compatibility_exception(
        owner=owner,
        owner_version=owner_version,
        requirement=Requirement(requirement),
        actual=actual,
    )


def _write_metadata(
    root: Path, name: str, version: str, requires: tuple[str, ...] = ()
) -> None:
    dist_info = root / f"{name.replace('-', '_')}-{version}.dist-info"
    dist_info.mkdir()
    lines = ["Metadata-Version: 2.1", f"Name: {name}", f"Version: {version}"]
    lines.extend(f"Requires-Dist: {requirement}" for requirement in requires)
    (dist_info / "METADATA").write_text("\n".join(lines) + "\n")


def test_find_errors_reads_real_distribution_metadata(
    constraints, tmp_path: Path
) -> None:
    _write_metadata(tmp_path, "transformers", "5.12.1")
    _write_metadata(
        tmp_path,
        "mlx-vlm",
        "0.6.16",
        ("transformers>=5.14.0; python_version >= '3'",),
    )
    assert constraints.find_errors(tmp_path) == []


def test_find_errors_rejects_nonexception_conflict_from_metadata(
    constraints, tmp_path: Path
) -> None:
    _write_metadata(tmp_path, "transformers", "5.12.1")
    _write_metadata(tmp_path, "other-vision", "1.0", ("transformers>=5.14.0",))
    assert constraints.find_errors(tmp_path) == [
        "other-vision requires transformers>=5.14.0, but bundled transformers==5.12.1"
    ]


def test_find_errors_rejects_empty_distribution_directory(
    constraints, tmp_path: Path
) -> None:
    assert constraints.find_errors(tmp_path) == [
        f"no installed distributions found in {tmp_path}"
    ]
