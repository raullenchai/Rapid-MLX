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


def _run(monkeypatch, cold_per_capture, warm_per_capture, tmp_path):
    """Drive ``_run_bench`` with scripted latencies."""
    baseline_dir = tmp_path / seb.BASELINE_DIR
    if not baseline_dir.exists():
        baseline_dir.mkdir(parents=True)
        (baseline_dir / "bench-test--model.json").write_text(json.dumps(BASELINE))
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir(exist_ok=True)

    ctx = types.SimpleNamespace(
        repo_root=tmp_path, artifact_path=lambda name: artifacts / name
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

    Both captures agree — the machine is quiet — and both are 40% over
    baseline. That is a regression and has to be reported as one.
    """
    result = _run(monkeypatch, [350.0, 351.0], [400.0, 400.0], tmp_path)

    assert result["status"] == "fail", result
    assert "perf regression" in result["summary"], result["summary"]


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
