#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the Desktop release manifest helper."""

from __future__ import annotations

import importlib.util
import plistlib
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _REPO_ROOT / "scripts" / "desktop_manifest.py"
_SHA = "a" * 40


@pytest.fixture(scope="module")
def desktop_manifest():
    spec = importlib.util.spec_from_file_location("desktop_manifest", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _app(tmp_path: Path, *, version: str = "0.13.0-rc2", build: str = "164") -> Path:
    """Build a dummy ``apps/rapid-mac`` layout with a plist-bearing .app."""

    app_dir = tmp_path / "app"
    contents = app_dir / "build" / "Rapid-MLX Desktop.app" / "Contents"
    contents.mkdir(parents=True)
    plist = {
        "CFBundleIdentifier": "com.rapidmlx.rapid",
        "CFBundleShortVersionString": version,
        "CFBundleVersion": build,
    }
    (contents / "Info.plist").write_bytes(plistlib.dumps(plist))
    return app_dir


def _dmg(tmp_path: Path, app_dir: Path, name: str = "rapid-mlx-desktop.dmg") -> Path:
    dmg = app_dir / "build" / name
    dmg.parent.mkdir(parents=True, exist_ok=True)
    dmg.write_bytes(b"dmg-bytes")
    return dmg


def _create(desktop_manifest, app_dir, dmg, **overrides):
    kwargs = dict(
        app_dir=app_dir,
        dmg_path=dmg,
        source_sha=_SHA,
        version="0.13.0-rc2",
        app_tag="rapid-mac-v0.13.0-rc2",
        signed=True,
        delta_compared=True,
    )
    kwargs.update(overrides)
    return desktop_manifest.create_manifest(**kwargs)


def test_create_then_verify_round_trips(desktop_manifest, tmp_path):
    app_dir = _app(tmp_path)
    dmg = _dmg(tmp_path, app_dir)
    manifest = _create(desktop_manifest, app_dir, dmg)
    output = tmp_path / "desktop-manifest.json"
    desktop_manifest.write_manifest(manifest, output)
    assert (
        desktop_manifest.verify_manifest(
            app_dir=app_dir, dmg_path=dmg, manifest_path=output
        )
        == manifest
    )
    assert manifest["signed"] is True
    assert manifest["artifact_kind"] == "desktop-dmg"
    assert manifest["embedded_version"] == {
        "CFBundleShortVersionString": "0.13.0-rc2",
        "CFBundleVersion": "164",
    }
    assert len(manifest["artifacts"]) == 1
    (artifact,) = manifest["artifacts"]
    assert artifact["filename"] == "rapid-mlx-desktop.dmg"
    assert artifact["size"] == dmg.stat().st_size
    assert artifact["sha256"] == desktop_manifest.sha256(dmg)


def test_accepts_stable_version_and_unspecified_signed(desktop_manifest, tmp_path):
    app_dir = _app(tmp_path, version="0.13.1")
    dmg = _dmg(tmp_path, app_dir)
    manifest = _create(
        desktop_manifest,
        app_dir,
        dmg,
        version="0.13.1",
        app_tag="rapid-mac-v0.13.1",
        signed=None,
    )
    assert manifest["signed"] is False
    assert manifest["version"] == "0.13.1"


def test_verify_rejects_changed_dmg(desktop_manifest, tmp_path):
    app_dir = _app(tmp_path)
    dmg = _dmg(tmp_path, app_dir)
    manifest = _create(desktop_manifest, app_dir, dmg)
    output = tmp_path / "desktop-manifest.json"
    desktop_manifest.write_manifest(manifest, output)
    dmg.write_bytes(b"replacement-bytes")
    with pytest.raises(ValueError, match="SHA-256 mismatch|size mismatch"):
        desktop_manifest.verify_manifest(
            app_dir=app_dir, dmg_path=dmg, manifest_path=output
        )


def test_create_rejects_non_commit_sha(desktop_manifest, tmp_path):
    app_dir = _app(tmp_path)
    dmg = _dmg(tmp_path, app_dir)
    with pytest.raises(ValueError, match="40-character"):
        _create(desktop_manifest, app_dir, dmg, source_sha="not-a-sha")


def test_create_rejects_invalid_version(desktop_manifest, tmp_path):
    app_dir = _app(tmp_path)
    dmg = _dmg(tmp_path, app_dir)
    with pytest.raises(ValueError, match="invalid release version"):
        _create(
            desktop_manifest,
            app_dir,
            dmg,
            version="not-a-version",
            app_tag="rapid-mac-vnot-a-version",
        )


def test_create_rejects_app_tag_version_mismatch(desktop_manifest, tmp_path):
    app_dir = _app(tmp_path)
    dmg = _dmg(tmp_path, app_dir)
    with pytest.raises(ValueError, match="does not match version"):
        _create(desktop_manifest, app_dir, dmg, app_tag="rapid-mac-v0.13.0-rc3")


def test_create_rejects_embedded_version_mismatch(desktop_manifest, tmp_path):
    app_dir = _app(tmp_path, version="0.13.0-rc1")
    dmg = _dmg(tmp_path, app_dir)
    with pytest.raises(ValueError, match="CFBundleShortVersionString"):
        _create(desktop_manifest, app_dir, dmg)  # manifest version is rc2


def test_create_rejects_missing_built_app(desktop_manifest, tmp_path):
    app_dir = tmp_path / "app"  # no built .app
    app_dir.mkdir(parents=True)
    dmg = _dmg(tmp_path, app_dir)
    with pytest.raises(ValueError, match="Info.plist"):
        _create(desktop_manifest, app_dir, dmg)


def test_create_rejects_missing_dmg(desktop_manifest, tmp_path):
    app_dir = _app(tmp_path)
    dmg = app_dir / "build" / "rapid-mlx-desktop.dmg"  # never written
    with pytest.raises(ValueError, match="DMG not found"):
        _create(desktop_manifest, app_dir, dmg)


def test_verify_rejects_unknown_schema_or_project(desktop_manifest, tmp_path):
    app_dir = _app(tmp_path)
    dmg = _dmg(tmp_path, app_dir)
    manifest = _create(desktop_manifest, app_dir, dmg)
    manifest["project"] = "not-rapid-mlx"
    output = tmp_path / "desktop-manifest.json"
    desktop_manifest.write_manifest(manifest, output)
    with pytest.raises(ValueError, match="unknown schema or project"):
        desktop_manifest.verify_manifest(
            app_dir=app_dir, dmg_path=dmg, manifest_path=output
        )


def test_verify_rejects_extra_artifact(desktop_manifest, tmp_path):
    app_dir = _app(tmp_path)
    dmg = _dmg(tmp_path, app_dir)
    manifest = _create(desktop_manifest, app_dir, dmg)
    manifest["artifacts"].append(dict(manifest["artifacts"][0]))
    output = tmp_path / "desktop-manifest.json"
    desktop_manifest.write_manifest(manifest, output)
    with pytest.raises(ValueError, match="exactly one artifact"):
        desktop_manifest.verify_manifest(
            app_dir=app_dir, dmg_path=dmg, manifest_path=output
        )


def test_verify_rejects_embedded_version_drift(desktop_manifest, tmp_path):
    app_dir = _app(tmp_path, build="164")
    dmg = _dmg(tmp_path, app_dir)
    manifest = _create(desktop_manifest, app_dir, dmg)
    output = tmp_path / "desktop-manifest.json"
    desktop_manifest.write_manifest(manifest, output)
    # A same-SHA rebuild that changed the embedded version must not verify.
    (
        app_dir / "build" / "Rapid-MLX Desktop.app" / "Contents" / "Info.plist"
    ).write_bytes(
        plistlib.dumps(
            {
                "CFBundleIdentifier": "com.rapidmlx.rapid",
                "CFBundleShortVersionString": "0.13.0-rc2",
                "CFBundleVersion": "165",
            }
        )
    )
    with pytest.raises(ValueError, match="embedded versions do not match"):
        desktop_manifest.verify_manifest(
            app_dir=app_dir, dmg_path=dmg, manifest_path=output
        )


def test_signed_manifest_claims_signing_and_notary_gates(desktop_manifest, tmp_path):
    # A signed build records signed-build + the notarization/stapling gates.
    app_dir = _app(tmp_path)
    dmg = _dmg(tmp_path, app_dir)
    manifest = _create(
        desktop_manifest, app_dir, dmg
    )  # signed=True, delta_compared=True
    gates = manifest["validation_gate"].split("|")
    assert manifest["signed"] is True
    for claimed in (
        "signed-build",
        "app-notarize",
        "dmg-build",
        "dmg-size-delta",
        "validate-dmg",
        "dmg-notarize",
        "final-validate-dmg",
    ):
        assert claimed in gates, f"missing {claimed} in {gates}"
    assert "ad-hoc-build" not in gates


def test_unsigned_manifest_never_claims_signing_or_notary_gates(
    desktop_manifest, tmp_path
):
    # An ad-hoc/unsigned dry-run must not claim any signing or notarization gate.
    app_dir = _app(tmp_path)
    dmg = _dmg(tmp_path, app_dir)
    manifest = _create(desktop_manifest, app_dir, dmg, signed=False)
    gates = manifest["validation_gate"].split("|")
    assert manifest["signed"] is False
    assert "ad-hoc-build" in gates
    assert "signed-build" not in gates
    assert "app-notarize" not in gates
    assert "dmg-notarize" not in gates
    assert "final-validate-dmg" not in gates
    # Round-trips: verify recomputes the same (gate-truthful) string.
    output = tmp_path / "unsigned-manifest.json"
    desktop_manifest.write_manifest(manifest, output)
    assert (
        desktop_manifest.verify_manifest(
            app_dir=app_dir, dmg_path=dmg, manifest_path=output
        )
        == manifest
    )


def test_manifest_does_not_claim_dmg_size_delta_when_skipped(
    desktop_manifest, tmp_path
):
    # No previous release baseline -> the DMG size-delta gate did not run, so the
    # manifest must not claim it. Verify recomputes and still matches.
    app_dir = _app(tmp_path)
    dmg = _dmg(tmp_path, app_dir)
    manifest = _create(desktop_manifest, app_dir, dmg, delta_compared=False)
    gates = manifest["validation_gate"].split("|")
    assert "dmg-size-delta" not in gates
    assert manifest["dmg_size_delta_compared"] is False
    output = tmp_path / "no-delta-manifest.json"
    desktop_manifest.write_manifest(manifest, output)
    assert (
        desktop_manifest.verify_manifest(
            app_dir=app_dir, dmg_path=dmg, manifest_path=output
        )
        == manifest
    )


def test_verify_rejects_self_claimed_signed_gates_for_unsigned_build(
    desktop_manifest, tmp_path
):
    # An unsigned manifest lying that it ran a signing/notary gate must fail
    # closed on verify — a gate truthfulness is structural, not just recorded.
    app_dir = _app(tmp_path)
    dmg = _dmg(tmp_path, app_dir)
    manifest = _create(
        desktop_manifest, app_dir, dmg, signed=False, delta_compared=True
    )
    manifest["validation_gate"] = (
        "signed-build|bundle-size|app-notarize|dmg-build|dmg-size-delta|"
        "validate-dmg|dmg-notarize|final-validate-dmg"
    )
    output = tmp_path / "lying-manifest.json"
    desktop_manifest.write_manifest(manifest, output)
    with pytest.raises(
        ValueError, match="validation gate does not match what actually ran"
    ):
        desktop_manifest.verify_manifest(
            app_dir=app_dir, dmg_path=dmg, manifest_path=output
        )


def test_verify_rejects_claimed_delta_when_not_compared(desktop_manifest, tmp_path):
    # Manifest claims dmg-size-delta but says no comparison happened -> fail.
    app_dir = _app(tmp_path)
    dmg = _dmg(tmp_path, app_dir)
    manifest = _create(
        desktop_manifest, app_dir, dmg, signed=True, delta_compared=False
    )
    manifest["validation_gate"] = (
        "signed-build|bundle-size|app-notarize|dmg-build|dmg-size-delta|"
        "validate-dmg|dmg-notarize|final-validate-dmg"
    )
    output = tmp_path / "lying-delta.json"
    desktop_manifest.write_manifest(manifest, output)
    with pytest.raises(
        ValueError, match="validation gate does not match what actually ran"
    ):
        desktop_manifest.verify_manifest(
            app_dir=app_dir, dmg_path=dmg, manifest_path=output
        )
