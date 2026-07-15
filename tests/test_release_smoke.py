#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for release-smoke artifact selection."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _REPO_ROOT / "scripts" / "release_smoke.py"


@pytest.fixture(scope="module")
def release_smoke():
    spec = importlib.util.spec_from_file_location("release_smoke", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_release_artifacts_returns_wheel_then_sdist(release_smoke, tmp_path):
    wheel = tmp_path / "rapid_mlx-0.10.9-py3-none-any.whl"
    sdist = tmp_path / "rapid_mlx-0.10.9.tar.gz"
    wheel.write_bytes(b"wheel")
    sdist.write_bytes(b"sdist")
    assert release_smoke.release_artifacts(tmp_path) == (
        wheel.resolve(),
        sdist.resolve(),
    )


def test_release_artifacts_rejects_missing_sdist(release_smoke, tmp_path):
    (tmp_path / "rapid_mlx-0.10.9-py3-none-any.whl").write_bytes(b"wheel")
    with pytest.raises(ValueError, match="exactly one"):
        release_smoke.release_artifacts(tmp_path)


def test_release_artifacts_rejects_multiple_wheels(release_smoke, tmp_path):
    (tmp_path / "rapid_mlx-0.10.8-py3-none-any.whl").write_bytes(b"one")
    (tmp_path / "rapid_mlx-0.10.9-py3-none-any.whl").write_bytes(b"two")
    (tmp_path / "rapid_mlx-0.10.9.tar.gz").write_bytes(b"sdist")
    with pytest.raises(ValueError, match="exactly one"):
        release_smoke.release_artifacts(tmp_path)


def test_release_artifacts_rejects_extra_file(release_smoke, tmp_path):
    wheel = tmp_path / "rapid_mlx-0.10.9-py3-none-any.whl"
    sdist = tmp_path / "rapid_mlx-0.10.9.tar.gz"
    wheel.write_bytes(b"wheel")
    sdist.write_bytes(b"sdist")
    (tmp_path / "unexpected.txt").write_text("not a release artifact")

    with pytest.raises(ValueError, match="no extra files"):
        release_smoke.release_artifacts(tmp_path)


def test_release_artifacts_rejects_version_mismatch(release_smoke, tmp_path):
    """A wheel and sdist from different builds must not be accepted together."""
    (tmp_path / "rapid_mlx-0.10.9-py3-none-any.whl").write_bytes(b"wheel")
    (tmp_path / "rapid_mlx-0.10.8.tar.gz").write_bytes(b"sdist")

    with pytest.raises(ValueError, match="different versions"):
        release_smoke.release_artifacts(tmp_path)


def test_release_artifacts_rejects_missing_directory(release_smoke, tmp_path):
    """A --dist-dir that does not exist must raise a clear error, not traceback."""
    with pytest.raises(ValueError, match="not a directory"):
        release_smoke.release_artifacts(tmp_path / "does-not-exist")
