# SPDX-License-Identifier: Apache-2.0
"""Release distributions must not publish direct URL dependencies."""

from __future__ import annotations

import io
import tarfile
import zipfile
from pathlib import Path

import pytest

from scripts.verify_dependency_metadata import verify_dist

REPO_ROOT = Path(__file__).resolve().parents[1]


def _metadata(requirement: str) -> bytes:
    return (
        "Metadata-Version: 2.4\n"
        "Name: rapid-mlx\n"
        "Version: 0.15.4\n"
        f"Requires-Dist: {requirement}\n\n"
    ).encode()


def _dist(
    tmp_path: Path, wheel_requirement: str, sdist_requirement: str | None = None
) -> Path:
    dist = tmp_path / "dist"
    dist.mkdir()
    wheel_metadata = _metadata(wheel_requirement)
    sdist_metadata = _metadata(sdist_requirement or wheel_requirement)
    wheel = dist / "rapid_mlx-0.15.4-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("rapid_mlx-0.15.4.dist-info/METADATA", wheel_metadata)
    sdist = dist / "rapid_mlx-0.15.4.tar.gz"
    with tarfile.open(sdist, "w:gz") as archive:
        info = tarfile.TarInfo("rapid_mlx-0.15.4/PKG-INFO")
        info.size = len(sdist_metadata)
        archive.addfile(info, io.BytesIO(sdist_metadata))
        # setuptools also writes an internal egg-info copy. The verifier must
        # inspect the canonical root metadata rather than treating this normal
        # duplicate as ambiguity.
        egg_info = tarfile.TarInfo("rapid_mlx-0.15.4/rapid_mlx.egg-info/PKG-INFO")
        egg_info.size = len(sdist_metadata)
        archive.addfile(egg_info, io.BytesIO(sdist_metadata))
    return dist


@pytest.mark.parametrize(
    "requirement",
    [
        "tensorfold @ git+https://example.invalid/TensorFold.git@deadbeef; extra == 'fast'",
        "tensorfold@https://example.invalid/tensorfold-0.5.0.whl",
        "tensorfold[metal, server] @ https://example.invalid/tensorfold.tar.gz",
    ],
)
@pytest.mark.parametrize("bad_artifact", ["wheel", "sdist"])
def test_wheel_and_sdist_reject_direct_references(
    tmp_path: Path, requirement: str, bad_artifact: str
) -> None:
    indexed = "mlx>=0.32.3; platform_system == 'Darwin'"
    dist = _dist(
        tmp_path,
        requirement if bad_artifact == "wheel" else indexed,
        requirement if bad_artifact == "sdist" else indexed,
    )
    with pytest.raises(ValueError, match="direct URL dependencies are not publishable"):
        verify_dist(dist)


def test_indexed_requirements_pass_for_wheel_and_sdist(tmp_path: Path) -> None:
    verify_dist(_dist(tmp_path, "mlx>=0.32.3; platform_system == 'Darwin'"))


@pytest.mark.parametrize(
    "workflow",
    ["publish.yml", "release-artifact-matrix.yml", "release-preflight.yml"],
)
def test_every_python_release_build_runs_dependency_metadata_gate(
    workflow: str,
) -> None:
    source = (REPO_ROOT / ".github" / "workflows" / workflow).read_text()
    assert source.count("python scripts/verify_dependency_metadata.py dist/") == 1
