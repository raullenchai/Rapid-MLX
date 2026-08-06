# SPDX-License-Identifier: Apache-2.0
"""The perf gate must not emit a verdict a busy machine cannot support.

#1527. `stress_e2e_bench` compares a request median against a committed
baseline at a 5% threshold, with nothing isolating the measurement from GPU
work the machine does on its own behalf. On a developer's Mac that is not
hypothetical — the animated desktop decodes video on the GPU.

Observed on PR #1526, a change that provably cannot affect the bench path:

    [BLOCKING] cold +42.5% vs baseline (threshold 5%)

while an interleaved capture on a quiet machine put the same commit at
cold -0.2%. Contamination of tens of percent against a 5% threshold does
not degrade the signal, it replaces it — and a gate that fails for reasons
unrelated to the diff teaches everyone to re-roll it, at which point it has
stopped being a gate.

Two properties, tested here against the real comparison logic with the HTTP
call stubbed (no model, no server — the defect is in how captures are
combined and judged):

* **minimum, not mean.** Contention only ever ADDS time, so the minimum
  across repeats is the best estimate of the uncontaminated value.
* **decline, don't guess.** When repeats disagree by more than the model's
  own threshold, the environment is noisier than the effect being measured
  and no honest verdict exists. Report that.
"""

from __future__ import annotations

import json
import pathlib
import sys
import types

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.pr_validate.steps import stress_e2e_bench as seb  # noqa: E402

BASELINE = {
    "schema": 1,
    "captured_at": "2026-08-06T00:00:00Z",
    "family": "test",
    "sample_runs": 3,
    "regression_threshold_pct": {
        "cold_request_ms_median": 5,
        "warm_request_ms_median": 5,
    },
    "model": {
        "id": "test/model",
        "revision": "deadbeef",
        "quantization": "8bit",
        "engine": "batched",
    },
    "environment": {
        "hardware": {"chip": "Apple M3 Ultra", "memory_gb": 256},
        "software": {
            "rapid_mlx_version": "0.12.4",
            "rapid_mlx_commit": "0" * 40,
            "python": "3.12.13",
            "macos": "26.5.2",
            "mlx": "0.31.2",
            "mlx_lm": "0.31.3",
            "mlx_vlm": "0.6.4",
            "mlx_audio": "0.4.6",
        },
    },
    "metrics": {
        "cold_request_ms_median": 250.0,
        "warm_request_ms_median": 400.0,
        "speedup_x": 0.625,
    },
}


def _run(monkeypatch, cold_per_capture, warm_per_capture, tmp_path, base=None):
    """Drive ``_run_bench`` with scripted latencies."""
    baseline_dir = tmp_path / seb.BASELINE_DIR
    if not baseline_dir.exists():
        baseline_dir.mkdir(parents=True)
        (baseline_dir / "bench-test--model.json").write_text(json.dumps(BASELINE))
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir(exist_ok=True)

    ctx = types.SimpleNamespace(
        repo_root=tmp_path,
        artifact_path=lambda name: artifacts / name,
        base_sha="0" * 40,
        base_branch="main",
    )
    choice = types.SimpleNamespace(model_id="test/model", quality_tier="full")

    monkeypatch.setattr(seb, "_cached_model_revision", lambda _m: "deadbeef")
    monkeypatch.setattr(seb, "_current_chip", lambda: "Apple M3 Ultra")

    state = {"n": 0}

    class FakeResponse:
        def __init__(self, ms):
            self._ms = ms

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({"usage": {"completion_tokens": 10}}).encode()

    times = {"now": 0.0}

    def fake_time():
        return times["now"]

    def fake_urlopen(req, timeout=None):
        idx = state["n"] % 12
        capture = state["n"] // 12
        state["n"] += 1
        if idx < 5:
            ms = cold_per_capture[capture]
        elif idx < 7:
            ms = 1.0
        else:
            ms = warm_per_capture[capture]
        times["now"] += ms / 1000.0
        return FakeResponse(ms)

    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(seb.time, "time", fake_time)

    # ``base`` is what a base-ref replay would measure: None means the
    # replay is unavailable, a tuple means it returned those numbers.
    monkeypatch.setattr(seb, "_bench_on_base", lambda _c, _ch: base)

    return seb._run_bench(ctx, choice)


def test_minimum_capture_wins_so_one_busy_repeat_cannot_fail_the_gate(
    monkeypatch, tmp_path
):
    """A contaminated first capture must not decide the verdict.

    250ms then 251ms against a 250ms baseline: the machine hiccuped for one
    capture in a way the spread still tolerates, and the clean one is the
    honest number.
    """
    result = _run(monkeypatch, [258.0, 250.0], [400.0, 400.0], tmp_path)
    assert result["status"] == "pass", result["summary"]

    artifact = json.loads(pathlib.Path(result["artifact"]).read_text())
    assert artifact["cold_request_ms_median"] == pytest.approx(250.0), artifact
    assert len(artifact["captures"]) == 2


def test_a_machine_noisier_than_the_threshold_gets_no_verdict(monkeypatch, tmp_path):
    """The #1526 shape: repeats disagree by far more than 5%."""
    result = _run(monkeypatch, [347.0, 252.0], [420.0, 376.0], tmp_path)

    assert result["status"] == "skip", result
    assert "not quiet enough" in result["summary"], result["summary"]
    # The numbers are still recorded — declining to gate is not declining
    # to measure.
    artifact = json.loads(pathlib.Path(result["artifact"]).read_text())
    assert artifact["cold_request_ms_median"] == pytest.approx(252.0)
    assert artifact["capture_spread_pct"]["cold_request_ms_median"] > 5


def test_a_real_regression_on_a_quiet_machine_still_fails(monkeypatch, tmp_path):
    """The gate must not become unfailable.

    Captures agree (quiet machine), the PR is 40% over baseline, and the
    base ref measured in the same session is NOT. That is a regression the
    diff caused and has to be reported as one.
    """
    result = _run(
        monkeypatch,
        [350.0, 351.0],
        [400.0, 400.0],
        tmp_path,
        base=(250.0, 400.0),
    )

    assert result["status"] == "fail", result
    assert "perf regression" in result["summary"], result["summary"]
    assert "BASE REF" in result["summary"], result["summary"]


def test_steady_contention_is_caught_by_the_base_ref_arm(monkeypatch, tmp_path):
    """The hole the spread check cannot see.

    A steady GPU consumer inflates every capture equally, so repeats agree
    and the spread filter passes them through — and the committed baseline,
    captured on some other day, then reads the inflation as this PR's doing.
    Measuring the base ref in the SAME session under the SAME conditions is
    what cancels it.
    """
    result = _run(
        monkeypatch,
        [350.0, 351.0],
        [560.0, 561.0],
        tmp_path,
        base=(349.0, 559.0),
    )

    assert result["status"] == "skip", result
    assert "base ref measured HERE, NOW" in result["summary"], result["summary"]

    artifact = json.loads(pathlib.Path(result["artifact"]).read_text())
    assert "base_ref_comparison" in artifact, artifact
    assert artifact["base_ref_comparison"]["cold_delta_pct"] < 5


def test_an_unavailable_base_replay_falls_back_and_says_so(monkeypatch, tmp_path):
    """No silent pass when the A/B cannot be run."""
    result = _run(monkeypatch, [350.0, 351.0], [400.0, 400.0], tmp_path, base=None)

    assert result["status"] == "fail", result
    assert "could not be A/B" in result["summary"], result["summary"]


def test_cold_prompts_differ_between_captures(monkeypatch, tmp_path):
    """The second capture must not be able to hit the prefix cache.

    Reusing the same five cold prompts against the same server would make
    the second capture's "cold" median a WARM measurement, and the minimum
    across captures would then systematically understate cold — hiding the
    regressions this exists to catch.
    """
    seen: list[str] = []

    real_dumps = json.dumps

    def spy(obj, *a, **k):
        if isinstance(obj, dict) and "messages" in obj:
            seen.append(obj["messages"][-1]["content"])
        return real_dumps(obj, *a, **k)

    monkeypatch.setattr(seb.json, "dumps", spy)
    _run(monkeypatch, [250.0, 251.0], [400.0, 401.0], tmp_path, base=None)

    cold = [p for p in seen if p.startswith("Cold prompt")]
    assert len(cold) == 10, cold
    assert len(set(cold)) == 10, f"cold prompts repeated across captures: {cold}"


def test_a_quiet_machine_within_threshold_passes(monkeypatch, tmp_path):
    result = _run(monkeypatch, [250.0, 251.0], [400.0, 401.0], tmp_path)
    assert result["status"] == "pass", result["summary"]
    assert "within" in result["summary"]


def test_spread_is_recorded_even_when_the_verdict_is_clean(monkeypatch, tmp_path):
    """A surprising verdict must be auditable without re-running."""
    result = _run(monkeypatch, [250.0, 251.0], [400.0, 401.0], tmp_path)
    artifact = json.loads(pathlib.Path(result["artifact"]).read_text())
    assert "capture_spread_pct" in artifact
    assert artifact["capture_spread_pct"]["cold_request_ms_median"] == pytest.approx(
        0.4, abs=0.05
    )
