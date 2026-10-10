# SPDX-License-Identifier: Apache-2.0
"""Execute the real Bash evidence boundary without launching a GUI."""

import json
import os
import re
import subprocess
from pathlib import Path

import pytest

HARNESS = (
    Path(__file__).resolve().parents[1] / "apps/rapid-mac/scripts/gui-golden-flows.sh"
)


def run_shell(tmp_path, body):
    return subprocess.run(
        [
            "/bin/bash",
            "-c",
            'source "$HARNESS"; OUT_ROOT="$EVIDENCE"; mkdir -p "$OUT_ROOT"; cleanup_persona() { :; }; cleanup_operator_server() { :; }; cleanup_telemetry_sink() { :; }; trap finish EXIT; '
            + body,
        ],
        env={
            **os.environ,
            "HARNESS": str(HARNESS),
            "EVIDENCE": str(tmp_path),
            "GUI_FLOWS": "",
        },
        text=True,
        capture_output=True,
    )


def read_journey(tmp_path, name):
    return json.loads((tmp_path / "journeys" / f"{name}.json").read_text())


def test_each_journey_keeps_its_own_cost_and_artifacts(tmp_path):
    result = run_shell(
        tmp_path,
        """
        sample() { begin_launch; sleep 0.06; finish_launch; sleep 0.03; }
        run_journey first sample
        run_journey second sample
    """,
    )
    assert result.returncode == 0, result.stderr
    for name in ("first", "second"):
        record = read_journey(tmp_path, name)
        assert record["flow"] == name
        assert record["status"] == "pass"
        assert record["exit_code"] == 0
        assert record["launch_count"] == 1
        assert record["launch_duration_seconds"] >= 0.06
        assert record["execution_duration_seconds"] >= 0.03
        assert record["duration_seconds"] == pytest.approx(
            record["launch_duration_seconds"] + record["execution_duration_seconds"]
        )
        assert record["artifact_path"] == str(tmp_path)


@pytest.mark.parametrize("code", [1, 7, 130, 143])
def test_failed_or_cancelled_journey_preserves_prior_pass_and_stops(tmp_path, code):
    result = run_shell(
        tmp_path,
        f"""
        good() {{ :; }}
        broken() {{ begin_launch; sleep 0.03; exit {code}; }}
        run_journey first good
        run_journey broken broken
        run_journey never good
    """,
    )
    assert result.returncode == code, result.stderr
    assert read_journey(tmp_path, "first")["status"] == "pass"
    failed = read_journey(tmp_path, "broken")
    assert failed["status"] == ("cancelled" if code in (130, 143) else "fail")
    assert failed["exit_code"] == code
    assert failed["launch_duration_seconds"] >= 0.03
    assert not (tmp_path / "journeys" / "never.json").exists()


def test_relaunches_are_accumulated(tmp_path):
    result = run_shell(
        tmp_path,
        """
        sample() { begin_launch; sleep 0.03; finish_launch; begin_launch; sleep 0.03; finish_launch; }
        run_journey relaunch sample
    """,
    )
    assert result.returncode == 0, result.stderr
    record = read_journey(tmp_path, "relaunch")
    assert record["launch_count"] == 2
    assert record["launch_duration_seconds"] >= 0.06


def test_cleanup_failure_cannot_leave_active_journey_green(tmp_path):
    result = run_shell(
        tmp_path,
        """
        cleanup_persona() { return 1; }
        broken() { exit 0; }
        run_journey cleanup broken
    """,
    )
    assert result.returncode == 1, result.stderr
    assert read_journey(tmp_path, "cleanup")["status"] == "fail"


def test_failed_assertion_aborts_the_function_before_success(tmp_path):
    result = run_shell(
        tmp_path,
        """
        broken() { false; touch "$OUT_ROOT/false-green"; }
        run_journey assertion broken
    """,
    )
    assert result.returncode == 1, result.stderr
    assert not (tmp_path / "false-green").exists()
    assert read_journey(tmp_path, "assertion")["status"] == "fail"


def test_early_zero_exit_does_not_report_success(tmp_path):
    result = run_shell(
        tmp_path, "premature() { exit 0; }; run_journey premature premature"
    )
    assert result.returncode == 1, result.stderr
    assert read_journey(tmp_path, "premature")["exit_code"] == 1


@pytest.mark.parametrize("signal,code", [("INT", 130), ("TERM", 143)])
def test_signal_records_interrupted_launch(tmp_path, signal, code):
    result = run_shell(
        tmp_path,
        f"""
        trap 'exit {code}' {signal}
        interrupted() {{ begin_launch; kill -s {signal} $$; }}
        run_journey interrupted interrupted
    """,
    )
    assert result.returncode == code, result.stderr
    record = read_journey(tmp_path, "interrupted")
    assert record["status"] == "cancelled"
    assert record["launch_count"] == 1


def test_real_all_dispatcher_emits_independent_records(tmp_path):
    # Run the actual dispatch table with tiny journey bodies; assertions and
    # evidence handling remain real. This catches a forgotten wrapper in all.
    dispatcher = HARNESS.read_text().rsplit('case "$FLOW" in', 1)[1].split("esac", 1)[0]
    names = re.findall(r"^    ([a-z][a-z0-9-]+)\)", dispatcher, re.M)
    functions = set(re.findall(r"\bflow_[a-z0-9_]+", dispatcher))
    definitions = "\n".join(f"{name}() {{ :; }}" for name in functions)
    result = run_shell(
        tmp_path, definitions + '\nFLOW=all\ncase "$FLOW" in' + dispatcher + "esac"
    )
    assert result.returncode == 0, result.stderr
    all_block = dispatcher.split("    all)", 1)[1].split(";;", 1)[0]
    expected = set(re.findall(r"\bflow_([a-z0-9_]+)", all_block))
    assert {p.stem for p in (tmp_path / "journeys").glob("*.json")} == {
        name.replace("_", "-") for name in expected
    }
    for name in set(names) - {"all"}:
        assert f"{name}) run_journey {name} flow_" in dispatcher
