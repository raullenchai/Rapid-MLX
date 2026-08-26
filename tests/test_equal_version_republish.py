from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts/check_equal_version_republish.py"


@pytest.fixture(scope="module")
def checker():
    spec = importlib.util.spec_from_file_location(
        "check_equal_version_republish", SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _fixtures(tmp_path: Path):
    dmg = tmp_path / "rapid-mlx-desktop.dmg"
    dmg.write_bytes(b"exact rerun dmg")
    digest = hashlib.sha256(dmg.read_bytes()).hexdigest()
    latest = {
        "version": "0.13.0",
        "dmg_url": f"https://example.invalid/{digest}.dmg",
        "dmg_sha256": digest,
        "dmg_size": dmg.stat().st_size,
    }
    current = tmp_path / "current.json"
    candidate = tmp_path / "candidate.json"
    current.write_text(json.dumps(latest))
    candidate.write_text(json.dumps(latest))
    release = tmp_path / "release.json"
    release.write_text(
        json.dumps(
            {
                "assets": [
                    {
                        "name": "rapid-mlx-desktop.dmg",
                        "state": "uploaded",
                        "size": dmg.stat().st_size,
                        "digest": f"sha256:{digest}",
                    }
                ]
            }
        )
    )
    return current, candidate, dmg, release


def test_identical_equal_version_is_noop_with_existing_release(checker, tmp_path):
    current, candidate, dmg, release = _fixtures(tmp_path)
    evidence = checker.verify(
        current_path=current,
        candidate_path=candidate,
        dmg_path=dmg,
        release_path=release,
    )
    assert "mutable updater no-op" in evidence


def test_identical_equal_version_allows_missing_release_recovery(checker, tmp_path):
    current, candidate, dmg, _ = _fixtures(tmp_path)
    evidence = checker.verify(
        current_path=current,
        candidate_path=candidate,
        dmg_path=dmg,
        release_path=None,
    )
    assert "mutable updater no-op" in evidence


def test_mismatched_equal_version_fails_before_mutation(checker, tmp_path):
    current, candidate, dmg, release = _fixtures(tmp_path)
    value = json.loads(candidate.read_text())
    value["dmg_sha256"] = "a" * 64
    candidate.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="exact-run DMG"):
        checker.verify(
            current_path=current,
            candidate_path=candidate,
            dmg_path=dmg,
            release_path=release,
        )


def test_mismatched_existing_release_fails_before_mutation(checker, tmp_path):
    current, candidate, dmg, release = _fixtures(tmp_path)
    value = json.loads(release.read_text())
    value["assets"][0]["digest"] = "sha256:" + "b" * 64
    release.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="Release DMG identity differs"):
        checker.verify(
            current_path=current,
            candidate_path=candidate,
            dmg_path=dmg,
            release_path=release,
        )


@pytest.mark.parametrize("pointer_state", ["missing", "older", "malformed"])
def test_release_mismatch_fails_independent_of_pointer_state(
    checker, tmp_path, pointer_state
):
    current, candidate, dmg, release = _fixtures(tmp_path)
    if pointer_state == "missing":
        current.unlink()
    elif pointer_state == "older":
        value = json.loads(current.read_text())
        value["version"] = "0.12.17"
        current.write_text(json.dumps(value))
    else:
        current.write_text("not json")
    value = json.loads(release.read_text())
    value["assets"][0]["digest"] = "sha256:" + "c" * 64
    release.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="Release DMG identity differs"):
        checker.verify_exact_artifact(
            candidate_path=candidate,
            dmg_path=dmg,
            release_path=release,
        )


@pytest.mark.parametrize("pointer_state", ["missing", "older"])
def test_matching_release_permits_pointer_recovery(checker, tmp_path, pointer_state):
    current, candidate, dmg, release = _fixtures(tmp_path)
    if pointer_state == "missing":
        current.unlink()
    elif pointer_state == "older":
        value = json.loads(current.read_text())
        value["version"] = "0.12.17"
        current.write_text(json.dumps(value))
    version, _, _ = checker.verify_exact_artifact(
        candidate_path=candidate,
        dmg_path=dmg,
        release_path=release,
    )
    assert version == "0.13.0"


def test_matching_release_does_not_authorize_malformed_pointer(checker, tmp_path):
    current, candidate, dmg, release = _fixtures(tmp_path)
    current.write_text("not json")
    with pytest.raises(ValueError, match="cannot read current latest.json"):
        checker.verify(
            current_path=current,
            candidate_path=candidate,
            dmg_path=dmg,
            release_path=release,
        )
