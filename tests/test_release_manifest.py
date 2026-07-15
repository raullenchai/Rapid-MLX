#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the release artifact manifest helper."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _REPO_ROOT / "scripts" / "release_manifest.py"
_SHA = "a" * 40


@pytest.fixture(scope="module")
def manifest_module():
    spec = importlib.util.spec_from_file_location("release_manifest", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _dist(tmp_path: Path) -> tuple[Path, Path, Path]:
    dist = tmp_path / "dist"
    dist.mkdir()
    wheel = dist / "rapid_mlx-0.10.9-py3-none-any.whl"
    sdist = dist / "rapid_mlx-0.10.9.tar.gz"
    wheel.write_bytes(b"wheel-bytes")
    sdist.write_bytes(b"sdist-bytes")
    return dist, wheel, sdist


def test_create_then_verify_round_trips(manifest_module, tmp_path):
    dist, _, _ = _dist(tmp_path)
    manifest = manifest_module.create_manifest(
        dist_dir=dist, source_sha=_SHA, version="0.10.9"
    )
    output = tmp_path / "release-manifest.json"
    manifest_module.write_manifest(manifest, output)
    assert (
        manifest_module.verify_manifest(dist_dir=dist, manifest_path=output) == manifest
    )


def test_verify_rejects_changed_artifact(manifest_module, tmp_path):
    dist, wheel, _ = _dist(tmp_path)
    output = tmp_path / "release-manifest.json"
    manifest_module.write_manifest(
        manifest_module.create_manifest(
            dist_dir=dist, source_sha=_SHA, version="0.10.9"
        ),
        output,
    )
    wheel.write_bytes(b"replacement")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        manifest_module.verify_manifest(dist_dir=dist, manifest_path=output)


def test_create_rejects_unexpected_distribution_shape(manifest_module, tmp_path):
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "rapid_mlx-0.10.9-py3-none-any.whl").write_bytes(b"wheel")
    with pytest.raises(ValueError, match="exactly one"):
        manifest_module.create_manifest(
            dist_dir=dist, source_sha=_SHA, version="0.10.9"
        )


def test_create_rejects_extra_distribution_file(manifest_module, tmp_path):
    dist, _, _ = _dist(tmp_path)
    (dist / "rapid_mlx-0.10.9.zip").write_bytes(b"unexpected")
    with pytest.raises(ValueError, match="no extra files"):
        manifest_module.create_manifest(
            dist_dir=dist, source_sha=_SHA, version="0.10.9"
        )


def test_create_rejects_non_commit_sha(manifest_module, tmp_path):
    dist, _, _ = _dist(tmp_path)
    with pytest.raises(ValueError, match="40-character"):
        manifest_module.create_manifest(
            dist_dir=dist, source_sha="not-a-sha", version="0.10.9"
        )


def test_create_rejects_artifact_version_mismatch(manifest_module, tmp_path):
    dist, _, _ = _dist(tmp_path)
    with pytest.raises(ValueError, match="does not match release version"):
        manifest_module.create_manifest(
            dist_dir=dist, source_sha=_SHA, version="0.10.10"
        )


def test_verify_rejects_manifest_version_mismatch(manifest_module, tmp_path):
    dist, _, _ = _dist(tmp_path)
    manifest = manifest_module.create_manifest(
        dist_dir=dist, source_sha=_SHA, version="0.10.9"
    )
    manifest["version"] = "0.10.10"
    output = tmp_path / "release-manifest.json"
    manifest_module.write_manifest(manifest, output)

    with pytest.raises(ValueError, match="does not match release version"):
        manifest_module.verify_manifest(dist_dir=dist, manifest_path=output)


def test_verify_rejects_duplicate_manifest_artifact(manifest_module, tmp_path):
    dist, _, _ = _dist(tmp_path)
    manifest = manifest_module.create_manifest(
        dist_dir=dist, source_sha=_SHA, version="0.10.9"
    )
    manifest["artifacts"].append(manifest["artifacts"][0])
    output = tmp_path / "release-manifest.json"
    manifest_module.write_manifest(manifest, output)

    with pytest.raises(ValueError, match="exactly two unique artifacts"):
        manifest_module.verify_manifest(dist_dir=dist, manifest_path=output)
