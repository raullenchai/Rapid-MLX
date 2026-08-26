#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the LIVE release-blocker query (mock gh only)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _REPO_ROOT / "scripts" / "check_release_blockers.py"
_SHA = "b" * 40


@pytest.fixture(scope="module")
def blockers():
    spec = importlib.util.spec_from_file_location("check_release_blockers", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def gh(tmp_path, monkeypatch):
    """A mock `gh` whose output is driven by $MOCK_GH_OUT / $MOCK_GH_RC."""

    (tmp_path / "gh").write_text(
        "#!/usr/bin/env bash\n"
        'if [[ -n "${MOCK_GH_RC:-}" ]]; then exit "$MOCK_GH_RC"; fi\n'
        'printf "%s" "${MOCK_GH_OUT:-[]}"\n'
    )
    (tmp_path / "gh").chmod(0o755)
    return str(tmp_path / "gh")


def _waiver(tmp_path, ids, *, name="waivers-0.13.0-rc2.json") -> Path:
    d = tmp_path / "waivers"
    d.mkdir()
    p = d / name
    p.write_text(
        json.dumps(
            {
                "version": "0.13.0-rc2",
                "waivers": [
                    {"issue": i, "reason": f"waived for rc2 ({i})", "by": "ds0732"}
                    for i in ids
                ],
            }
        )
    )
    return d


def _issue(
    num, title="blocker title", url="https://github.com/x/y/issues/{num}", **extra
):
    rec = {"number": num, "title": title, "url": url}
    rec.update(extra)
    return rec


def _run(blockers, gh, waivers_dir, *, out="[]", rc="0", expected=None):
    import os

    os.environ["MOCK_GH_OUT"] = out
    if rc and rc != "0":  # only signal a failure rc so the success path prints
        os.environ["MOCK_GH_RC"] = rc
    try:
        return blockers.check_live_blockers(
            version="0.13.0-rc2",
            source_sha=_SHA,
            gh=gh,
            repo="raullenchai/Rapid-MLX",
            waivers_dir=waivers_dir,
            expected_open_ids=expected,
        )
    finally:
        os.environ.pop("MOCK_GH_OUT", None)
        os.environ.pop("MOCK_GH_RC", None)


def test_malformed_source_sha_fails(blockers, gh, tmp_path):
    # Evidence must never be bound to a malformed source-sha.
    import os

    os.environ["MOCK_GH_OUT"] = "[]"
    try:
        with pytest.raises(blockers.BlockerCheckError, match="40-character"):
            blockers.check_live_blockers(
                version="0.13.0-rc2",
                source_sha="not-a-full-sha",
                gh=gh,
                repo="raullenchai/Rapid-MLX",
                waivers_dir=tmp_path / "waivers",
                expected_open_ids=None,
            )
    finally:
        os.environ.pop("MOCK_GH_OUT", None)


def test_no_open_blockers_passes(blockers, gh, tmp_path):
    d = tmp_path / "waivers"
    d.mkdir()
    evidence, ids = _run(blockers, gh, d)
    assert ids == []
    assert any("<none>" in e for e in evidence)


def test_open_blockers_all_waived_passes(blockers, gh, tmp_path):
    out = json.dumps([_issue(2301), _issue(2298)])
    d = _waiver(tmp_path, [2301, 2298])
    evidence, ids = _run(blockers, gh, d, out=out)
    assert ids == [2298, 2301]
    assert any("WAIVED by @ds0732" in e for e in evidence)


def test_open_blocker_without_waiver_fails(blockers, gh, tmp_path):
    out = json.dumps([_issue(9999), _issue(2301)])
    d = _waiver(tmp_path, [2301])
    with pytest.raises(blockers.BlockerCheckError, match="without a waiver"):
        _run(blockers, gh, d, out=out)


def test_missing_waiver_file_with_open_blocker_fails(blockers, gh, tmp_path):
    d = tmp_path / "waivers"
    d.mkdir()
    out = json.dumps([_issue(2301)])
    with pytest.raises(blockers.BlockerCheckError, match="without a waiver"):
        _run(blockers, gh, d, out=out)


def test_toctou_change_fails_closed(blockers, gh, tmp_path):
    d = tmp_path / "waivers"
    d.mkdir()
    # no open blockers now, but candidate-time had #2301 -> set changed
    with pytest.raises(blockers.BlockerCheckError, match="release-blocker set changed"):
        _run(blockers, gh, d, out="[]", expected=[2301])


def test_toctou_same_set_passes(blockers, gh, tmp_path):
    out = json.dumps([_issue(2301)])
    d = _waiver(tmp_path, [2301])
    evidence, ids = _run(blockers, gh, d, out=out, expected=[2301])
    assert ids == [2301]


def test_gh_nonzero_fails_closed(blockers, gh, tmp_path):
    d = tmp_path / "waivers"
    d.mkdir()
    with pytest.raises(blockers.BlockerCheckError, match="gh issue list failed"):
        _run(blockers, gh, d, rc="1")


def test_gh_invalid_json_fails_closed(blockers, gh, tmp_path):
    d = tmp_path / "waivers"
    d.mkdir()
    with pytest.raises(blockers.BlockerCheckError, match="invalid JSON"):
        _run(blockers, gh, d, out="{ not json")


def test_pull_request_records_excluded(blockers, gh, tmp_path):
    # REST /issues returns PRs too; a PR record must never count as a blocker.
    pr = {"number": 4444, "title": "some pr", "url": "u", "pull_request": {"url": "p"}}
    d = _waiver(tmp_path, [2301])
    out = json.dumps([_issue(2301), pr])
    evidence, ids = _run(blockers, gh, d, out=out)
    assert ids == [2301]


def test_waiver_file_wrong_version_fails(blockers, gh, tmp_path):
    # A waiver file named for a different version is never loaded, so an open
    # blocker for rc2 stays uncovered and fails closed.
    d = _waiver(tmp_path, [2301], name="waivers-0.13.0-rc3.json")
    out = json.dumps([_issue(2301)])
    with pytest.raises(blockers.BlockerCheckError, match="without a waiver"):
        _run(blockers, gh, d, out=out)


def test_expected_empty_set_toctou_rejects_changed_set(blockers, gh, tmp_path):
    # Candidate-time open set was EMPTY; pre-tag now has #2301 open. Even though
    # the expected set is empty, the change must fail closed (zero != none).
    d = _waiver(tmp_path, [2301])
    out = json.dumps([_issue(2301)])
    with pytest.raises(blockers.BlockerCheckError, match="set changed"):
        # expected=[] means "expect an empty set", NOT "skip the check".
        _run(blockers, gh, d, out=out, expected=[])


def test_expected_empty_set_matches_empty_passes(blockers, gh, tmp_path):
    d = tmp_path / "waivers"
    d.mkdir()
    evidence, ids = _run(blockers, gh, d, out="[]", expected=[])
    assert ids == []
    assert any("<none>" in e for e in evidence)


def test_duplicate_open_issue_record_fails(blockers, gh, tmp_path):
    d = _waiver(tmp_path, [2301])
    out = json.dumps([_issue(2301), _issue(2301)])
    with pytest.raises(blockers.BlockerCheckError, match="duplicate open issue"):
        _run(blockers, gh, d, out=out)


def test_malformed_issue_record_no_number_fails(blockers, gh, tmp_path):
    d = _waiver(tmp_path, [2301])
    out = json.dumps([_issue(2301), {"title": "oops", "url": "u"}])
    with pytest.raises(blockers.BlockerCheckError, match="malformed issue number"):
        _run(blockers, gh, d, out=out)


def test_malformed_non_object_record_fails(blockers, gh, tmp_path):
    d = _waiver(tmp_path, [2301])
    out = json.dumps([_issue(2301), "not-an-object"])
    with pytest.raises(blockers.BlockerCheckError, match="malformed record"):
        _run(blockers, gh, d, out=out)


def test_duplicate_waiver_fails(blockers, gh, tmp_path):
    d = tmp_path / "waivers"
    d.mkdir()
    p = d / "waivers-0.13.0-rc2.json"
    p.write_text(
        json.dumps(
            {
                "version": "0.13.0-rc2",
                "waivers": [
                    {"issue": 2301, "reason": "a", "by": "ds0732"},
                    {"issue": 2301, "reason": "b", "by": "ds0732"},
                ],
            }
        )
    )
    out = json.dumps([_issue(2301)])
    with pytest.raises(blockers.BlockerCheckError, match="duplicates waiver"):
        _run(blockers, gh, d, out=out)


def test_stale_extra_waiver_fails(blockers, gh, tmp_path):
    # The waiver file waives both #2301 and #2298, but only #2301 is actually
    # open. Waiving #2298 — which is not part of the live open set — is stale.
    d = _waiver(tmp_path, [2301, 2298])
    out = json.dumps([_issue(2301)])
    with pytest.raises(blockers.BlockerCheckError, match="stale waiver"):
        _run(blockers, gh, d, out=out)
