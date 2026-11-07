"""Completion evidence belongs to one wrapper invocation, not mixed CI output."""

from __future__ import annotations

import importlib.util
import io
import json
import os
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "apps/rapid-mac/scripts"
CHECKER = SCRIPTS / "verify-test-completion.py"


def event(kind: str, **payload) -> dict:
    return {"version": 0, "kind": "event", "payload": {"kind": kind, **payload}}


def complete() -> list[dict]:
    return [
        event("runStarted"),
        event("testStarted", testID="Suite/test()"),
        event("testCaseStarted", testID="Suite/test()"),
        event("testCaseEnded", testID="Suite/test()"),
        event("testEnded", testID="Suite/test()"),
        event("runEnded"),
    ]


@pytest.fixture
def checker():
    spec = importlib.util.spec_from_file_location("desktop_completion", CHECKER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check(checker, records):
    return checker.check_completion(
        io.StringIO("".join(json.dumps(r) + "\n" for r in records))
    )


def test_complete_parameterized_and_multiple_runs(checker):
    records = complete()
    records[3:3] = [
        event("testCaseStarted", testID="Suite/test()"),
        event("testCaseEnded", testID="Suite/test()"),
    ]
    records.insert(1, {"version": 0, "kind": "test", "payload": {"kind": "function"}})
    records.insert(1, {"version": 0, "kind": "futureRecord", "payload": {}})
    records.insert(1, event("futureEvent"))
    records.insert(-1, event("testSkipped", testID="Suite/skipped()"))
    records.insert(-1, event("issueRecorded", issue={"isKnown": True}))
    assert check(checker, records + complete()) == (2, 2)


@pytest.mark.parametrize(
    "records",
    [
        [],
        complete()[:-1],
        complete() + [event("runStarted")],
        complete() + [{"version": 0, "kind": "test", "payload": {"kind": "function"}}],
        [event("runEnded")],
        [event("runStarted"), event("runStarted")],
        [event("testStarted", testID="x")],
        [event("testSkipped", testID="x")],
        [event("runStarted"), event("testStarted", testID="x"), event("runEnded")],
        [event("runStarted"), event("testEnded", testID="x")],
        [event("runStarted"), event("testStarted")],
        complete()[:-1] + [event("testCancelled"), event("runEnded")],
        complete()[:-1]
        + [event("issueRecorded", issue={"isKnown": False}), event("runEnded")],
        [event("runStarted"), event("runEnded")],
        [{"version": 1, "kind": "event", "payload": {"kind": "runEnded"}}],
        [{"version": False, "kind": "event", "payload": {"kind": "runEnded"}}],
        [{"version": 0, "kind": "event", "payload": None}],
        [None],
    ],
)
def test_incomplete_or_invalid_stream_is_rejected(checker, records):
    with pytest.raises(ValueError):
        check(checker, records)


def test_malformed_json_and_message_text_cannot_complete(checker):
    with pytest.raises(ValueError):
        checker.check_completion(io.StringIO('{"kind":'))
    with pytest.raises(ValueError):
        check(
            checker,
            [event("runStarted", messages=[{"text": "runEnded; tests passed"}])],
        )


def test_skipped_selection_and_nonfailing_issue_keep_framework_semantics(checker):
    records = [
        event("runStarted"),
        event("testSkipped", testID="Suite/disabled()"),
        event("issueRecorded", issue={"isFailure": False, "isKnown": False}),
        event("runEnded"),
    ]
    assert check(checker, records) == (1, 0)


def test_checker_reports_missing_invalid_and_complete_files(checker, tmp_path, capsys):
    assert checker.main([]) == 2
    assert checker.main([str(tmp_path / "missing")]) == 1
    stream = tmp_path / "events.jsonl"
    stream.write_text("not JSON\n")
    assert checker.main([str(stream)]) == 1
    stream.write_text("\n" + "".join(json.dumps(r) + "\n" for r in complete()))
    assert checker.main([str(stream)]) == 0
    captured = capsys.readouterr()
    assert "incomplete Desktop test run" in captured.err
    assert "completion verified: 1 run(s), 1 test lifecycle(s)" in captured.out


def test_checker_script_entrypoint(tmp_path, monkeypatch, capsys):
    stream = tmp_path / "events.jsonl"
    stream.write_text("".join(json.dumps(r) + "\n" for r in complete()))
    monkeypatch.setattr(sys, "argv", [str(CHECKER), str(stream)])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(str(CHECKER), run_name="__main__")
    assert exit_info.value.code == 0
    assert "completion verified" in capsys.readouterr().out


@pytest.fixture
def wrapper(tmp_path):
    # Copy the actual wrapper/checker into a package-shaped fixture. No Swift build.
    scripts = tmp_path / "package/scripts"
    scripts.mkdir(parents=True)
    for name in ("desktop-test-timeout.sh", "verify-test-completion.py"):
        if (SCRIPTS / name).exists():
            shutil.copy2(SCRIPTS / name, scripts / name)
    bindir = tmp_path / "bin"
    bindir.mkdir()
    swift = bindir / "swift"
    swift.write_text(
        "#!/usr/bin/env python3\n"
        "import json,os,pathlib,sys,time\n"
        "args=sys.argv[1:]\n"
        "assert args[0]=='test', args\n"
        "pathlib.Path(os.environ['ARGS']).write_text(json.dumps(args))\n"
        "print('ordinary stdout',flush=True)\n"
        "print('ordinary stderr',file=sys.stderr,flush=True)\n"
        "mode=os.environ['MODE']\n"
        "if mode=='timeout': time.sleep(60)\n"
        "if '--event-stream-output-path' in args and mode!='missing':\n"
        " p=pathlib.Path(args[args.index('--event-stream-output-path')+1])\n"
        " records=json.loads(os.environ['EVENTS'])\n"
        " p.write_text(''.join(json.dumps(r)+'\\n' for r in records))\n"
        "else: print('Test run with 1 test passed after 0.001 seconds.')\n"
        "sys.exit(7 if mode=='nonzero' else 0)\n"
    )
    swift.chmod(0o755)
    watchdog = bindir / "watchdog"
    watchdog.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "$MODE" == timeout ]]; then\n'
        ' for ((i=0;i<100;i++)); do [[ -f "$ARGS" ]] && break; sleep 0.01; done\n'
        ' echo diagnostic > "$3/sample.txt"; exit 1; fi\n'
        'while kill -0 "$1" 2>/dev/null; do sleep 0.01; done\n'
    )
    watchdog.chmod(0o755)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    prior = artifacts / "prior-command.jsonl"
    prior.write_text("".join(json.dumps(r) + "\n" for r in complete()))
    env = os.environ | {
        "PATH": str(bindir) + os.pathsep + os.environ["PATH"],
        "RAPID_WATCHDOG_BIN": str(watchdog),
        "RAPID_DESKTOP_HANG_ARTIFACT_DIR": str(artifacts),
        "RAPID_DESKTOP_DEADLINE_MINUTES": "1",
        "RAPID_DESKTOP_TEST_SKIP": "OnlyOnePriorCommandTest",
        "ARGS": str(tmp_path / "args.json"),
        "EVENTS": json.dumps(complete()),
    }

    def run(mode="complete", records=None, watchdog_available=True):
        local = env | {"MODE": mode}
        if not watchdog_available:
            local["RAPID_WATCHDOG_BIN"] = str(bindir / "missing-watchdog")
        if records is not None:
            local["EVENTS"] = json.dumps(records)
        result = subprocess.run(
            ["bash", str(scripts / "desktop-test-timeout.sh")],
            env=local,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result, artifacts, json.loads(Path(env["ARGS"]).read_text())

    return run


def test_wrapper_requires_current_run_and_preserves_skip_output(wrapper):
    result, artifacts, args = wrapper()
    assert result.returncode == 0, result.stderr
    assert "ordinary stdout" in result.stdout
    assert "ordinary stderr" in result.stderr
    assert args[:3] == ["test", "--no-parallel", "--skip"]
    assert args[3] == "OnlyOnePriorCommandTest"
    assert args[args.index("--event-stream-version") + 1] == "0"
    assert list(artifacts.iterdir()) == [artifacts / "prior-command.jsonl"]


def test_wrapper_zero_exit_truncation_fails_despite_prior_success(wrapper):
    result, artifacts, _ = wrapper(records=complete()[:-1])
    assert result.returncode != 0
    assert "incomplete" in result.stderr.lower()
    assert list(artifacts.glob("*/swift-testing-events.jsonl"))
    assert (artifacts / "prior-command.jsonl").exists()


def test_wrapper_preserves_nonzero_status_and_evidence(wrapper):
    result, artifacts, _ = wrapper(mode="nonzero", records=[])
    assert result.returncode == 7
    assert list(artifacts.glob("*/swift-testing-events.jsonl"))


def test_wrapper_missing_current_stream_cannot_use_prior_summary(wrapper):
    result, artifacts, _ = wrapper(mode="missing")
    assert result.returncode != 0
    assert "incomplete Desktop test run" in result.stderr
    assert (artifacts / "prior-command.jsonl").exists()


def test_wrapper_timeout_keeps_124_and_diagnostics(wrapper):
    result, artifacts, _ = wrapper(mode="timeout")
    assert result.returncode == 124
    assert list(artifacts.glob("*/sample.txt"))


def test_wrapper_without_watchdog_still_requires_completion(wrapper):
    result, _, _ = wrapper(watchdog_available=False)
    assert result.returncode == 0
    assert "running without hang backstop" in result.stderr
    result, _, _ = wrapper(records=complete()[:-1], watchdog_available=False)
    assert result.returncode != 0
    assert "incomplete Desktop test run" in result.stderr
