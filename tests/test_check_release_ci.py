# SPDX-License-Identifier: Apache-2.0
"""Offline contracts for the exact-SHA production release CI gate."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

import scripts.check_release_ci as release_ci
from scripts.check_release_ci import (
    ReleaseCIGateError,
    RequiredWorkflow,
    verify,
)

SHA = "a" * 40
REPO = "raullenchai/Rapid-MLX"
REQUIREMENTS = (
    RequiredWorkflow("ci.yml", "tests"),
    RequiredWorkflow("rapid-mac-ci.yml", "desktop-tests"),
)


def _record(
    run_id: int,
    *,
    status: str = "completed",
    conclusion: str | None = "success",
    sha: str = SHA,
    event: str = "push",
) -> dict:
    return {
        "id": run_id,
        "head_sha": sha,
        "event": event,
        "status": status,
        "conclusion": conclusion,
        "html_url": f"https://example.test/runs/{run_id}",
    }


def _mock_gh(
    tmp_path: Path,
    *,
    ci_runs: list[dict],
    mac_runs: list[dict],
    jobs: dict[int, list],
) -> Path:
    state = tmp_path / "state"
    state.mkdir()
    (state / "ci.json").write_text(json.dumps(ci_runs))
    (state / "mac.json").write_text(json.dumps(mac_runs))
    (state / "jobs.json").write_text(json.dumps({str(k): v for k, v in jobs.items()}))
    gh = tmp_path / "gh"
    gh.write_text(
        textwrap.dedent(
            f"""\
            #!/usr/bin/env python3
            import json, pathlib, sys
            state = pathlib.Path({str(state)!r})
            url = sys.argv[2]
            if '/actions/workflows/ci.yml/runs' in url:
                payload = json.loads((state / 'ci.json').read_text())
                pages = payload if payload and isinstance(payload[0], list) else [payload]
                print(json.dumps([{{'workflow_runs': runs}} for runs in pages]))
            elif '/actions/workflows/rapid-mac-ci.yml/runs' in url:
                payload = json.loads((state / 'mac.json').read_text())
                pages = payload if payload and isinstance(payload[0], list) else [payload]
                print(json.dumps([{{'workflow_runs': runs}} for runs in pages]))
            elif '/actions/runs/' in url and url.endswith('/jobs'):
                run_id = url.split('/actions/runs/', 1)[1].split('/', 1)[0]
                payload = json.loads((state / 'jobs.json').read_text())[run_id]
                pages = payload if payload and isinstance(payload[0], list) else [payload]
                print(json.dumps([{{'jobs': jobs}} for jobs in pages]))
            else:
                print('unexpected URL: ' + url, file=sys.stderr)
                raise SystemExit(2)
            """
        )
    )
    gh.chmod(0o755)
    return gh


def _jobs(name: str, conclusion: str = "success") -> list[dict]:
    return [{"name": name, "conclusion": conclusion}]


def test_accepts_both_exact_sha_aggregate_facades(tmp_path: Path) -> None:
    gh = _mock_gh(
        tmp_path,
        ci_runs=[_record(10)],
        mac_runs=[_record(20)],
        jobs={10: _jobs("tests"), 20: _jobs("desktop-tests")},
    )
    messages = verify(
        source_sha=SHA,
        repo=REPO,
        requirements=REQUIREMENTS,
        gh=str(gh),
        deadline_sec=0,
        sleep_sec=0,
    )
    assert "run 10" in messages[0]
    assert "run 20" in messages[1]


def test_red_required_facade_blocks_even_when_other_workflow_passes(
    tmp_path: Path,
) -> None:
    gh = _mock_gh(
        tmp_path,
        ci_runs=[_record(10)],
        mac_runs=[_record(20, conclusion="failure")],
        jobs={10: _jobs("tests"), 20: _jobs("desktop-tests", "failure")},
    )
    with pytest.raises(ReleaseCIGateError, match="desktop-tests.*failure"):
        verify(
            source_sha=SHA,
            repo=REPO,
            requirements=REQUIREMENTS,
            gh=str(gh),
            deadline_sec=0,
            sleep_sec=0,
        )


def test_finds_required_facade_after_first_jobs_page(tmp_path: Path) -> None:
    gh = _mock_gh(
        tmp_path,
        ci_runs=[_record(10)],
        mac_runs=[_record(20)],
        jobs={
            10: [[{"name": "matrix", "conclusion": "success"}], _jobs("tests")],
            20: _jobs("desktop-tests"),
        },
    )
    messages = verify(
        source_sha=SHA,
        repo=REPO,
        requirements=REQUIREMENTS,
        gh=str(gh),
        deadline_sec=0,
        sleep_sec=0,
    )
    assert "run 10" in messages[0]


def test_cancelled_duplicate_does_not_hide_earlier_success(tmp_path: Path) -> None:
    gh = _mock_gh(
        tmp_path,
        ci_runs=[_record(11, conclusion="cancelled"), _record(10)],
        mac_runs=[_record(20)],
        jobs={10: _jobs("tests"), 20: _jobs("desktop-tests")},
    )
    messages = verify(
        source_sha=SHA,
        repo=REPO,
        requirements=REQUIREMENTS,
        gh=str(gh),
        deadline_sec=0,
        sleep_sec=0,
    )
    assert "run 10" in messages[0]


def test_cancelled_first_page_does_not_hide_older_success(tmp_path: Path) -> None:
    gh = _mock_gh(
        tmp_path,
        ci_runs=[[_record(11, conclusion="cancelled")], [_record(10)]],
        mac_runs=[_record(20)],
        jobs={10: _jobs("tests"), 20: _jobs("desktop-tests")},
    )
    messages = verify(
        source_sha=SHA,
        repo=REPO,
        requirements=REQUIREMENTS,
        gh=str(gh),
        deadline_sec=0,
        sleep_sec=0,
    )
    assert "run 10" in messages[0]


def test_only_cancelled_evidence_waits_for_replacement_then_times_out(
    tmp_path: Path,
) -> None:
    gh = _mock_gh(
        tmp_path,
        ci_runs=[_record(10, conclusion="cancelled")],
        mac_runs=[_record(20)],
        jobs={20: _jobs("desktop-tests")},
    )
    with pytest.raises(ReleaseCIGateError, match="timed out waiting") as exc:
        verify(
            source_sha=SHA,
            repo=REPO,
            requirements=REQUIREMENTS,
            gh=str(gh),
            deadline_sec=0,
            sleep_sec=0,
        )
    assert "cancellation-only evidence" in str(exc.value)


def test_completed_run_without_conclusion_fails_closed(tmp_path: Path) -> None:
    gh = _mock_gh(
        tmp_path,
        ci_runs=[_record(10, conclusion=None)],
        mac_runs=[_record(20)],
        jobs={20: _jobs("desktop-tests")},
    )
    with pytest.raises(ReleaseCIGateError, match="completed run without a conclusion"):
        verify(
            source_sha=SHA,
            repo=REPO,
            requirements=REQUIREMENTS,
            gh=str(gh),
            deadline_sec=0,
            sleep_sec=0,
        )


def test_active_run_waits_instead_of_accepting_stale_success(tmp_path: Path) -> None:
    gh = _mock_gh(
        tmp_path,
        ci_runs=[_record(11, status="in_progress", conclusion=None), _record(10)],
        mac_runs=[_record(20)],
        jobs={10: _jobs("tests"), 20: _jobs("desktop-tests")},
    )
    with pytest.raises(ReleaseCIGateError, match="timed out waiting"):
        verify(
            source_sha=SHA,
            repo=REPO,
            requirements=REQUIREMENTS,
            gh=str(gh),
            deadline_sec=0,
            sleep_sec=0,
        )


def test_newer_success_is_not_blocked_by_older_active_duplicate(tmp_path: Path) -> None:
    gh = _mock_gh(
        tmp_path,
        ci_runs=[_record(12), _record(11, status="in_progress", conclusion=None)],
        mac_runs=[_record(20)],
        jobs={12: _jobs("tests"), 20: _jobs("desktop-tests")},
    )
    messages = verify(
        source_sha=SHA,
        repo=REPO,
        requirements=REQUIREMENTS,
        gh=str(gh),
        deadline_sec=0,
        sleep_sec=0,
    )
    assert "run 12" in messages[0]


def test_success_snapshot_is_reconfirmed_before_release(monkeypatch) -> None:
    requirement = (RequiredWorkflow("ci.yml", "tests"),)
    responses = iter(
        [
            ("success", "run 10 passed", 10),
            ("wait", "run 11 is in_progress", 11),
            ("success", "run 11 passed", 11),
            ("success", "run 11 passed", 11),
        ]
    )
    calls = 0

    def fake_evaluate(*_args):
        nonlocal calls
        calls += 1
        return next(responses)

    monkeypatch.setattr(release_ci, "_evaluate", fake_evaluate)
    messages = verify(
        source_sha=SHA,
        repo=REPO,
        requirements=requirement,
        deadline_sec=1,
        sleep_sec=0,
    )

    assert messages == ["run 11 passed"]
    assert calls == 4


@pytest.mark.parametrize("bad_sha", ["abc", "A" * 40, "g" * 40])
def test_invalid_source_sha_is_rejected_before_api(
    tmp_path: Path, bad_sha: str
) -> None:
    with pytest.raises(ReleaseCIGateError, match="source SHA"):
        verify(
            source_sha=bad_sha,
            repo=REPO,
            requirements=REQUIREMENTS,
            gh=str(tmp_path / "missing-gh"),
            deadline_sec=0,
            sleep_sec=0,
        )
