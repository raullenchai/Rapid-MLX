# SPDX-License-Identifier: Apache-2.0
"""A flagged bench is settled by an A/B, not by the committed baseline alone.

#1527. `stress_e2e_bench` compared a request median against a baseline
captured on some other day, on a machine in some other state, with nothing
isolating the measurement from GPU work the machine does on its own behalf.
On a developer's Mac that is not hypothetical — the animated desktop decodes
video on the GPU.

PR #1526 changes a batch-merge helper, and the bench sends requests
sequentially so no merge occurs. The gate reported `cold +42.5%` anyway; an
interleaved hand capture put the same commit at `cold -0.2%`.

The shape that survived review:

* the committed-baseline comparison keeps ONE capture — the same statistic
  the baselines were captured with, so nothing is biased by a change of
  sampling protocol;
* a flag is not a verdict. It defers to `_bench_ab_against_base`, which
  measures both arms **here, now**, in ten complete ABBA blocks. Each block
  contributes a paired log PR/base effect. A PR is cleared only when the
  one-sided 95% Student-t upper bound is within the existing regression
  limit; definite regressions fail and overlapping uncertainty declines.

These tests drive the real orchestration with the server context and the
HTTP call stubbed. Stubbing `_bench_ab_against_base` itself would leave them
green no matter how the A/B behaved, which is the failure mode they exist to
prevent.
"""

from __future__ import annotations

import contextlib
import json
import math
import pathlib
import sys
import types

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.pr_validate.steps import stress_e2e_bench as seb  # noqa: E402

THRESHOLDS = {"cold_request_ms_median": 5.0, "warm_request_ms_median": 5.0}

BASELINE = {
    "schema": 1,
    "captured_at": "2026-08-06T00:00:00Z",
    "family": "test",
    "sample_runs": 3,
    "regression_threshold_pct": dict(THRESHOLDS),
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

CHOICE = types.SimpleNamespace(
    model_id="test/model", quality_tier="full", family="test"
)
BASE_ROOT = "/base-worktree"


def _ctx(tmp_path):
    baseline_dir = tmp_path / seb.BASELINE_DIR
    baseline_dir.mkdir(parents=True, exist_ok=True)
    (baseline_dir / "bench-test--model.json").write_text(json.dumps(BASELINE))
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir(exist_ok=True)
    return types.SimpleNamespace(
        repo_root=tmp_path,
        artifact_path=lambda name: artifacts / name,
        base_sha="0" * 40,
        base_branch="main",
    )


def _harness(monkeypatch, tmp_path, base_ms, pr_ms, git_run=None):
    """Drive the real A/B with servers and HTTP stubbed.

    ``base_ms`` / ``pr_ms`` give ``(cold_ms, warm_ms)`` per round for each
    arm. Returns ``(result, arm_order, prompts_seen)``.
    """
    ctx = _ctx(tmp_path)
    monkeypatch.setattr(seb.tempfile, "mkdtemp", lambda prefix="": BASE_ROOT)
    monkeypatch.setattr(pathlib.Path, "rmdir", lambda self: None)
    monkeypatch.setattr(
        seb.subprocess,
        "run",
        git_run or (lambda *a, **k: types.SimpleNamespace(returncode=0)),
    )

    live = {"arm": None}
    order: list[str] = []
    seen: list[str] = []
    counts = {"base": 0, "pr": 0}

    @contextlib.contextmanager
    def fake_server(choice, ctx_, *, repo_root, artifact_prefix="", **kw):
        live["arm"] = "base" if str(repo_root) == BASE_ROOT else "pr"
        order.append(live["arm"])
        try:
            yield "server.log"
        finally:
            live["arm"] = None

    monkeypatch.setattr(seb, "_server_in_repo", fake_server)

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({"usage": {"completion_tokens": 10}}).encode()

    clock = {"now": 0.0}

    def fake_urlopen(req, timeout=None):
        prompt = json.loads(req.data)["messages"][-1]["content"]
        seen.append(prompt)
        arm = live["arm"]
        n = counts[arm]
        counts[arm] += 1
        round_no = n // 12
        table = base_ms if arm == "base" else pr_ms
        if prompt == "warmup":
            ms = 1.0
        elif prompt.startswith("Cold prompt"):
            ms = table[round_no][0]
        else:
            ms = table[round_no][1]
        clock["now"] += ms / 1000.0
        return FakeResponse()

    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(seb.time, "time", lambda: clock["now"])

    result = seb._bench_ab_against_base(ctx, CHOICE, THRESHOLDS)
    return result, order, seen


def test_the_ab_shares_one_round_loop():
    """Both arms must be measured inside the same round, not in two passes.

    Two separate loops would mean all of one arm, then all of the other —
    any drift over the run lands entirely on whichever went second.
    """
    import inspect

    src = inspect.getsource(seb._bench_ab_against_base)
    assert src.count("for round_no") == 1, "both arms must share one round loop"
    assert "for arm in order" in src


def test_both_arms_are_measured_every_round(monkeypatch, tmp_path):
    _, order, _ = _harness(
        monkeypatch,
        tmp_path,
        [(250, 400)] * seb.AB_ROUNDS,
        [(251, 401)] * seb.AB_ROUNDS,
    )
    assert len(order) == 2 * seb.AB_ROUNDS, order
    assert all(
        sorted(order[offset : offset + 2]) == ["base", "pr"]
        for offset in range(0, len(order), 2)
    ), order


def test_a_regression_survives_the_ab(monkeypatch, tmp_path):
    """Both arms quiet, PR consistently slower — that is the PR's doing."""
    result, _, _ = _harness(
        monkeypatch,
        tmp_path,
        [(250, 400)] * seb.AB_ROUNDS,
        [(350, 400)] * seb.AB_ROUNDS,
    )
    assert result["status"] == "fail", result
    assert "regression confirmed" in result["summary"], result["summary"]


def test_steady_contention_cancels(monkeypatch, tmp_path):
    """A steady consumer inflates both arms; the A/B sees through it.

    No spread check can catch this — repeats agree, because the contention
    never changes. Only measuring the base ref under the same conditions
    does.
    """
    result, _, _ = _harness(
        monkeypatch,
        tmp_path,
        [(349, 559)] * seb.AB_ROUNDS,
        [(350, 560)] * seb.AB_ROUNDS,
    )
    assert result["status"] == "pass", result
    assert "not this PR" in result["summary"], result["summary"]


def test_one_transient_base_burst_does_not_make_drift_permanently_inconclusive(
    monkeypatch, tmp_path
):
    """One early base burst is retained but does not dominate ten paired blocks."""
    result, _, _ = _harness(
        monkeypatch,
        tmp_path,
        [(250, 560), *([(250, 400)] * (seb.AB_ROUNDS - 1))],
        [(250, 400)] * seb.AB_ROUNDS,
    )
    assert result["status"] == "pass", result


def test_one_transient_pr_burst_is_retained_without_becoming_a_false_regression(
    monkeypatch, tmp_path
):
    result, _, _ = _harness(
        monkeypatch,
        tmp_path,
        [(250, 400)] * seb.AB_ROUNDS,
        [(250, 560), *([(250, 400)] * (seb.AB_ROUNDS - 1))],
    )
    assert result["status"] == "pass", result


def test_cold_uncertainty_is_not_silently_dropped(monkeypatch, tmp_path):
    cold_uncertain = [
        ((250, 400) if block % 2 == 0 else (600, 400))
        for block in range(seb.AB_BLOCKS)
        for _ in range(2)
    ]
    result, _, _ = _harness(
        monkeypatch,
        tmp_path,
        [(400, 400)] * seb.AB_ROUNDS,
        cold_uncertain,
    )
    assert result["status"] == "skip", result
    assert "uncertainty" in result["summary"], result["summary"]


def test_a_warm_regression_survives_cold_noise(monkeypatch, tmp_path):
    """Uncertain cold evidence cannot waive a definite warm regression."""
    cold_uncertain = [
        ((250, 500) if block % 2 == 0 else (600, 500))
        for block in range(seb.AB_BLOCKS)
        for _ in range(2)
    ]
    result, _, _ = _harness(
        monkeypatch,
        tmp_path,
        [(400, 400)] * seb.AB_ROUNDS,
        cold_uncertain,
    )
    assert result["status"] == "fail", result
    assert "regression confirmed" in result["summary"], result["summary"]


def test_both_arms_see_matched_prompt_sets(monkeypatch, tmp_path):
    """Symmetry: round N uses the same prompts on both sides.

    A different prompt set per arm would compare two different workloads and
    call the difference a regression.
    """
    _, _, seen = _harness(
        monkeypatch,
        tmp_path,
        [(250, 400)] * seb.AB_ROUNDS,
        [(250, 400)] * seb.AB_ROUNDS,
    )
    cold = [p for p in seen if p.startswith("Cold prompt")]
    assert len(cold) == 5 * 2 * seb.AB_ROUNDS, len(cold)
    assert len(set(cold)) == 5 * seb.AB_ROUNDS, sorted(set(cold))
    assert all(cold.count(p) == 2 for p in set(cold))


def test_rounds_use_different_cold_prompts(monkeypatch, tmp_path):
    """Round 2 must not be able to hit round 1's prefix cache."""
    _, _, seen = _harness(
        monkeypatch,
        tmp_path,
        [(250, 400)] * seb.AB_ROUNDS,
        [(250, 400)] * seb.AB_ROUNDS,
    )
    cold = {p for p in seen if p.startswith("Cold prompt")}
    assert any(p.startswith("Cold prompt #0.") for p in cold)
    assert any(p.startswith("Cold prompt #1.") for p in cold)
    assert any(p.startswith("Cold prompt #0.19 ") for p in cold)


def test_cleanup_failure_does_not_replace_the_verdict(monkeypatch, tmp_path):
    """A leaked worktree is a nuisance; a crashed gate step is not."""

    def flaky(*a, **k):
        if a and "remove" in a[0]:
            raise seb.subprocess.TimeoutExpired(cmd=a[0], timeout=120)
        return types.SimpleNamespace(returncode=0)

    result, _, _ = _harness(
        monkeypatch,
        tmp_path,
        [(250, 400)] * seb.AB_ROUNDS,
        [(250, 400)] * seb.AB_ROUNDS,
        git_run=flaky,
    )
    assert result["status"] == "pass", result
    assert "could not run" not in result["summary"], result["summary"]


def test_the_fast_path_keeps_the_baselines_own_statistic():
    """One capture against the committed baseline — no protocol change.

    Repeating and taking a minimum there would bias the PR measurement
    downward against a baseline aggregated differently, so a regression near
    the threshold could pass purely because the sampling changed. Repetition
    belongs in the A/B, where both arms get it.
    """
    import inspect

    src = inspect.getsource(seb._measure_bench)
    assert "cold = statistics.median(cold_times)" in src
    assert "warm = statistics.median(warm_times)" in src
    assert "captures" not in src


def test_arm_order_is_counterbalanced_across_rounds(monkeypatch, tmp_path):
    """Interleaving is not enough — the ORDER has to alternate too.

    base/PR/base/PR still runs base first every round, so a repeatable order
    effect (thermal buildup, a cache warming across the round, the OS
    settling after the first model load) lands on the PR arm every time. It
    is consistent, so the spread check sees nothing, and it reads as a
    regression.
    """
    _, order, _ = _harness(
        monkeypatch,
        tmp_path,
        [(250, 400)] * seb.AB_ROUNDS,
        [(250, 400)] * seb.AB_ROUNDS,
    )
    assert order == ["base", "pr", "pr", "base"] * seb.AB_BLOCKS, order
    assert order.count("base") == order.count("pr") == seb.AB_ROUNDS


def test_the_ab_writes_the_artifact_the_docs_promise(monkeypatch, tmp_path):
    """`harness/README.md` tells people to read the spreads — so emit them."""
    result, _, _ = _harness(
        monkeypatch,
        tmp_path,
        [(250, 400)] * seb.AB_ROUNDS,
        [(350, 400)] * seb.AB_ROUNDS,
    )
    path = pathlib.Path(result["artifact"])
    assert path.exists(), result
    data = json.loads(path.read_text())
    assert data["rounds"] == 20
    assert data["blocks"] == 10
    assert len(data["base_captures"]) == len(data["pr_captures"]) == 20
    assert set(data["paired_decisions"]) == {"cold", "warm"}
    assert len(data["paired_decisions"]["cold"]["block_log_ratios"]) == 10


def test_retained_real_aa_and_ab_captures_clear_without_threshold_tuning():
    fixture = json.loads(
        (REPO_ROOT / "tests/fixtures/stress_ab_retained_metrics.json").read_text()
    )
    assert fixture["harness_head"] == "167bd27c4b4e36213c5412b33a9adfd1563784f9"
    assert (
        fixture["harness_blob_stress_e2e_bench"]
        == "68c3681c684276aaf6fe30bcb80fdd80738d4d41"
    )
    expected_upper = {
        "aa_base167_n20": {"cold": 3.8884, "warm": 4.4623},
        "ab_4338_f38fbd716_n20": {"cold": 3.9024, "warm": 4.1780},
    }
    for name, run in fixture["runs"].items():
        assert len(run["captures"]) == 40
        assert [(row["round"], row["arm"]) for row in run["captures"]] == [
            pair
            for block in range(10)
            for pair in (
                (block * 2, "A"),
                (block * 2, "B"),
                (block * 2 + 1, "B"),
                (block * 2 + 1, "A"),
            )
        ]
        for metric in ("cold", "warm"):
            base = [
                row[f"{metric}_ms_median"]
                for row in run["captures"]
                if row["arm"] == "A"
            ]
            pr = [
                row[f"{metric}_ms_median"]
                for row in run["captures"]
                if row["arm"] == "B"
            ]
            decision = seb._paired_metric_decision(base, pr, 5.0)
            assert decision["status"] == "pass", (name, metric, decision)
            assert decision["upper_pct"] == pytest.approx(
                expected_upper[name][metric], abs=0.0001
            )


def test_fixed_abba_blocks_cancel_linear_log_drift_and_first_boot_order_effect():
    base = []
    pr = []
    for block in range(seb.AB_BLOCKS):
        # ABBA positions 0 and 3 belong to base; 1 and 2 belong to PR.
        # Linear log-time drift has the same within-block mean for both arms.
        values = [
            400 * math.exp(0.01 * (4 * block + position)) for position in range(4)
        ]
        base.extend((values[0], values[3]))
        pr.extend((values[1], values[2]))
    decision = seb._paired_metric_decision(base, pr, 5.0)
    assert decision["status"] == "pass"
    assert decision["mean_pct"] == pytest.approx(0.0, abs=1e-10)


@pytest.mark.parametrize("regression_pct", [5.01, 8.0])
def test_constant_regression_above_the_existing_limit_definitely_fails(
    regression_pct,
):
    base = [400.0] * seb.AB_ROUNDS
    pr = [400.0 * (1 + regression_pct / 100)] * seb.AB_ROUNDS
    decision = seb._paired_metric_decision(base, pr, 5.0)
    assert decision["status"] == "fail"
    assert decision["lower_pct"] > 5.0


def test_constant_equality_at_the_limit_is_accepted_by_the_reviewed_boundary():
    base = [400.0] * seb.AB_ROUNDS
    pr = [420.0] * seb.AB_ROUNDS
    decision = seb._paired_metric_decision(base, pr, 5.0)
    assert decision["status"] == "pass"
    assert decision["upper_pct"] == pytest.approx(5.0)


def test_overlap_is_uncertainty_not_a_positive_result():
    base = [400.0] * seb.AB_ROUNDS
    pr = [420.0 if block != 0 else 440.0 for block in range(seb.AB_ROUNDS)]
    decision = seb._paired_metric_decision(base, pr, 5.0)
    assert decision["status"] == "skip"
    assert decision["lower_pct"] <= 5.0 < decision["upper_pct"]


@pytest.mark.parametrize(
    ("base", "pr", "message"),
    [
        ([400.0] * 19, [400.0] * 20, "exactly 20"),
        ([400.0] * 20, [400.0] * 19, "exactly 20"),
        ([400.0] * 19 + [0.0], [400.0] * 20, "finite and positive"),
        ([400.0] * 20, [400.0] * 19 + [float("nan")], "finite and positive"),
        ([400.0] * 20, [400.0] * 19 + [float("inf")], "finite and positive"),
    ],
)
def test_invalid_or_incomplete_measurements_fail_closed(base, pr, message):
    with pytest.raises(ValueError, match=message):
        seb._paired_metric_decision(base, pr, 5.0)


@pytest.mark.parametrize("threshold", [-0.01, float("nan"), float("inf")])
def test_invalid_thresholds_fail_closed(threshold):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        seb._paired_metric_decision(
            [400.0] * seb.AB_ROUNDS, [400.0] * seb.AB_ROUNDS, threshold
        )


def test_collector_retains_all_captures_when_a_measurement_is_invalid(
    monkeypatch, tmp_path
):
    pr = [(250, 400)] * seb.AB_ROUNDS
    pr[0] = (250, float("nan"))
    result, _, _ = _harness(monkeypatch, tmp_path, [(250, 400)] * seb.AB_ROUNDS, pr)
    assert result["status"] == "fail"
    artifact = json.loads(pathlib.Path(result["artifact"]).read_text())
    assert len(artifact["base_captures"]) == len(artifact["pr_captures"]) == 20
    assert artifact["paired_decisions"] == {}
    assert "finite and positive" in artifact["measurement_error"]


# --- review round 4: assert the STEP's outcome, not its source ----------
#
# The previous versions of these two grepped `inspect.getsource`, so they
# stayed green whether or not the branch was reachable, whether the right
# value was passed, and — the one that matters — whether an inconclusive A/B
# actually changed the step's verdict. Drive `run()` instead.


def _run_step(monkeypatch, tmp_path, ab_status, ab_summary="stubbed"):
    """Run StressE2EBenchStep with the matrix stubbed to one flagged bench."""
    ctx = _ctx(tmp_path)
    ctx.files_changed = ["rapid_mlx/scheduler.py"]
    ctx.blast_radius = "high"
    ctx.run_log = lambda *a, **k: None

    monkeypatch.setattr(
        seb,
        "_select_models",
        lambda *a, **k: [CHOICE],
        raising=False,
    )
    monkeypatch.setattr(
        seb, "_load_registry", lambda *a, **k: {"agents": []}, raising=False
    )

    @contextlib.contextmanager
    def fake_server(choice, ctx_, **kw):
        yield "server.log"

    monkeypatch.setattr(seb, "_server_in_repo", fake_server)
    monkeypatch.setattr(seb, "_server", fake_server, raising=False)
    monkeypatch.setattr(
        seb,
        "_run_bench",
        lambda ctx_, choice, **kw: {
            "status": "fail",
            "summary": "perf regression: cold +42.5% vs baseline",
            "artifact": str(tmp_path / "artifacts" / "bench.json"),
            "thresholds": dict(THRESHOLDS),
            "executed": True,
        },
    )
    monkeypatch.setattr(
        seb,
        "_run_stress",
        lambda *a, **k: {"status": "pass", "summary": "8/8", "executed": True},
    )
    monkeypatch.setattr(
        seb,
        "_bench_ab_against_base",
        lambda *a, **k: {"status": ab_status, "summary": ab_summary},
    )
    return seb.StressE2EBenchStep().run(ctx)


def test_an_inconclusive_ab_fails_the_step(monkeypatch, tmp_path):
    """An unanswered perf question must not clear the PR.

    Letting the step succeed here would ship a real regression whenever
    unrelated noise made the A/B inconclusive — a worse trade than the false
    BLOCKING this change removes.
    """
    result = _run_step(monkeypatch, tmp_path, "skip", "machine not quiet enough")
    assert result.status == "fail", result
    joined = " ".join(result.findings)
    assert "[INCONCLUSIVE]" in joined, joined
    assert "[NOT-THIS-PR]" not in joined, joined


def test_an_answered_ab_that_clears_the_pr_passes_the_step(monkeypatch, tmp_path):
    result = _run_step(monkeypatch, tmp_path, "pass", "not this PR: cold +0.2%")
    assert result.status == "pass", result
    joined = " ".join(result.findings)
    assert "[NOT-THIS-PR]" in joined, joined
    assert "[BLOCKING]" not in joined, joined


def test_a_confirmed_regression_fails_the_step(monkeypatch, tmp_path):
    result = _run_step(
        monkeypatch, tmp_path, "fail", "perf regression confirmed against base"
    )
    assert result.status == "fail", result
    assert any("[BLOCKING]" in f for f in result.findings), result.findings


def test_the_manifest_records_one_settled_verdict_per_bench(monkeypatch, tmp_path):
    """Not a raw failure AND an A/B verdict that contradict each other."""
    _run_step(monkeypatch, tmp_path, "pass", "not this PR")
    manifest = json.loads(
        (tmp_path / "artifacts" / "stress-e2e-manifest.json").read_text()
    )
    entries = [e for e in manifest if e.get("kind") == "bench"]
    statuses = [e["status"] for e in entries]
    assert "fail" not in statuses, entries
    assert "preliminary" in statuses, entries
