# SPDX-License-Identifier: Apache-2.0
"""Contracts for the commit-bound GUI app artifact."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts/gui_app_artifact.py"
WORKFLOW = ROOT / ".github/workflows/rapid-mac-ci.yml"


def load_module():
    spec = importlib.util.spec_from_file_location("gui_app_artifact", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_manifest_round_trip_binds_archive_and_commit(tmp_path: Path):
    module = load_module()
    archive = tmp_path / "Rapid-MLX-Desktop.zip"
    archive.write_bytes(b"signed app bytes")
    manifest = tmp_path / "manifest.json"

    module.create(archive, "abc123", manifest)
    module.verify(archive, manifest, "abc123")

    payload = json.loads(manifest.read_text())
    assert payload["schema_version"] == 1
    assert payload["build_config"] == "release"
    assert payload["sidecar"] == "skipped"


@pytest.mark.parametrize("mutation", ["archive", "commit", "manifest"])
def test_verification_fails_closed_on_substitution(tmp_path: Path, mutation: str):
    module = load_module()
    archive = tmp_path / "Rapid-MLX-Desktop.zip"
    archive.write_bytes(b"original")
    manifest = tmp_path / "manifest.json"
    module.create(archive, "abc123", manifest)

    expected_sha = "abc123"
    if mutation == "archive":
        archive.write_bytes(b"substituted")
    elif mutation == "commit":
        expected_sha = "different"
    else:
        payload = json.loads(manifest.read_text())
        payload["sidecar"] = "bundled"
        manifest.write_text(json.dumps(payload))

    with pytest.raises(SystemExit, match="provenance verification failed"):
        module.verify(archive, manifest, expected_sha)


def test_workflow_builds_once_and_consumes_verified_artifact():
    jobs = yaml.safe_load(WORKFLOW.read_text())["jobs"]
    producer = jobs["gui-app-build"]
    consumer = jobs["gui-golden-flows"]
    producer_source = json.dumps(producer)
    consumer_source = json.dumps(consumer)
    verify_step = next(
        step
        for step in consumer["steps"]
        if step.get("name") == "Verify and extract GUI app"
    )

    assert "./scripts/build.sh" in producer_source
    assert "rapid-gui-app-${{ github.sha }}" in producer_source
    assert "gui-app-build" in consumer["needs"]
    assert "download-artifact" in consumer_source
    assert '--expected-source-sha "$GITHUB_SHA"' in verify_step["run"]
    assert "./scripts/build.sh" not in consumer_source
    assert "codesign --verify --deep --strict" in consumer_source


def test_required_gui_contract_job_runs_artifact_tests():
    jobs = yaml.safe_load(WORKFLOW.read_text())["jobs"]
    contract_source = json.dumps(jobs["gui-harness-contracts"])
    assert "tests/test_gui_app_artifact.py" in contract_source
