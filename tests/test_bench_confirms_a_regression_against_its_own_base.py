# SPDX-License-Identifier: Apache-2.0
"""A frozen baseline cannot tell a slow diff from a slow afternoon.

`stress_e2e_bench` compared each PR's measurement against a number
recorded on some earlier day. That answers "is this build slower than the
day we recorded it", which stops being the question the moment the
machine stops behaving the way it did that day — thermals, background
load, a dependency bump.

Issue #1547 is what that looks like in practice. Benching the baseline's
OWN commit, on the machine that captured it, reproduced the "regression"
it was supposed to disprove::

    Qwen3.5-35B-A3B-8bit, warm request median
      recorded baseline (f2097169, 05:12Z) : 375.8 ms
      f2097169 re-measured that evening    : 427.5 ms   (+13.8%)
      main                                 : 418.6 ms
      the PR under review                  : 418.0 ms

Every PR inherited the drift and the gate failed all of them
identically, which is worse than having no gate: a check that is always
red trains reviewers to click past it.

So a baseline miss is no longer a verdict. It triggers the on-base re-run
this step already trusts for stress and agent failures — the merge base
is measured back to back with the PR, on this machine, in this session —
and the PR is judged against THAT. Drift moves both numbers and cancels;
a real regression does not.
"""

import json

import pytest

from scripts.pr_validate.steps import stress_e2e_bench


@pytest.fixture
def ctx(tmp_path):
    """Minimal stand-in for the pieces of ``Context`` the bench touches."""

    class _Ctx:
        repo_root = tmp_path
        work_dir = tmp_path / "artifacts"

        def artifact_path(self, name):
            p = self.work_dir / name
            p.parent.mkdir(parents=True, exist_ok=True)
            return p

    return _Ctx()


@pytest.fixture
def choice():
    return stress_e2e_bench.ModelChoice(
        family="qwen3.5",
        model_id="mlx-community/Qwen3.5-35B-A3B-8bit",
        ram_gb_required=40.0,
        quality_tier="tier1",
        extra_args=[],
    )


@pytest.fixture
def baseline(ctx, choice):
    """A schema-v1 baseline recording 250/375 ms, 5% either side."""
    safe = stress_e2e_bench._safe_name(choice.model_id)
    path = ctx.repo_root / stress_e2e_bench.BASELINE_DIR / f"bench-{safe}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema": 1,
                "model": {"id": choice.model_id, "revision": "deadbeef"},
                "metrics": {
                    "cold_request_ms_median": 250.0,
                    "warm_request_ms_median": 375.0,
                },
                "regression_threshold_pct": {
                    "cold_request_ms_median": 5,
                    "warm_request_ms_median": 5,
                },
                "environment": {"hardware": {"chip": "Apple M3 Ultra"}},
            }
        )
    )
    return path


@pytest.fixture(autouse=True)
def _pin_environment(monkeypatch):
    """Keep the revision / chip preconditions out of the way — they have
    their own guards, and neither is what these tests are about."""
    monkeypatch.setattr(
        stress_e2e_bench, "_cached_model_revision", lambda *a, **k: "deadbeef"
    )
    monkeypatch.setattr(stress_e2e_bench, "_current_chip", lambda: "Apple M3 Ultra")


def _fake_measure(cold, warm):
    def _inner(ctx, choice, *, artifact_prefix=""):
        metrics = {
            "model": choice.model_id,
            "cold_request_ms_median": cold,
            "warm_request_ms_median": warm,
            "speedup_x": cold / warm if warm else 0,
        }
        path = ctx.artifact_path(
            f"{artifact_prefix}bench-{stress_e2e_bench._safe_name(choice.model_id)}.json"
        )
        path.write_text(json.dumps(metrics))
        return {"metrics": metrics, "artifact": str(path)}

    return _inner


def _base_measuring(cold, warm):
    """``_run_base_check`` stand-in that reports the base at cold/warm."""

    def _inner(ctx, choice, kind, agent):
        assert kind == "bench", f"bench failure asked for a {kind!r} base check"
        return {
            "status": "pass",
            "summary": f"base cold={cold:.0f}ms warm={warm:.0f}ms",
            "artifact": "base.json",
            "metrics": {
                "cold_request_ms_median": cold,
                "warm_request_ms_median": warm,
            },
            "executed": True,
        }

    return _inner


def _pr_failure(cold, warm, ctx, choice, monkeypatch):
    """Run the real ``_run_bench`` and return its (failing) result."""
    monkeypatch.setattr(stress_e2e_bench, "_measure_bench", _fake_measure(cold, warm))
    result = stress_e2e_bench._run_bench(ctx, choice)
    assert result["status"] == "fail", result["summary"]
    return result


def _base(cold, warm):
    return {
        "status": "pass",
        "summary": f"base cold={cold:.0f}ms warm={warm:.0f}ms",
        "artifact": "base.json",
        "metrics": {
            "cold_request_ms_median": cold,
            "warm_request_ms_median": warm,
        },
        "executed": True,
    }


def test_the_bench_defers_its_base_check_until_the_port_is_free(
    ctx, choice, baseline, monkeypatch
):
    """``_run_bench`` runs inside the PR's server context, which owns
    BENCH_PORT. A base server started from there can never bind, so the
    confirming measurement must NOT happen inside ``_run_bench`` — it
    carries its numbers out and the caller resolves them afterwards."""

    def _explode(*a, **k):
        raise AssertionError(
            "_run_bench started a base check while the PR server still held BENCH_PORT"
        )

    monkeypatch.setattr(stress_e2e_bench, "_run_base_check", _explode)
    result = _pr_failure(345.0, 418.0, ctx, choice, monkeypatch)
    assert result["bench_comparison"], (
        "the numbers the resolver needs did not travel out"
    )


def test_a_baseline_miss_the_base_also_reproduces_is_not_this_diffs_fault(
    ctx, choice, baseline, monkeypatch
):
    """The #1547 shape: PR and base are both ~12% over the frozen number,
    and identical to each other. Nothing regressed between them."""
    pr = _pr_failure(345.0, 418.0, ctx, choice, monkeypatch)

    verdict = stress_e2e_bench._resolve_bench_against_base(
        pr, _base(344.0, 419.0), choice.model_id
    )

    assert verdict["preexisting"], verdict["finding"]
    assert "[PRE-EXISTING]" in verdict["finding"]
    # The report must not bury the drift — the baseline still needs a refresh.
    assert "refresh" in verdict["finding"].lower()


def test_a_regression_the_base_does_not_share_still_fails(
    ctx, choice, baseline, monkeypatch
):
    """The control is only allowed to excuse drift it actually shares."""
    pr = _pr_failure(300.0, 450.0, ctx, choice, monkeypatch)

    verdict = stress_e2e_bench._resolve_bench_against_base(
        pr, _base(250.0, 375.0), choice.model_id
    )

    assert not verdict["preexisting"], verdict["finding"]
    assert "[BLOCKING]" in verdict["finding"]
    assert "vs the base measured in this run" in verdict["finding"]


def test_a_drifted_machine_does_not_hide_a_regression_stacked_on_top(
    ctx, choice, baseline, monkeypatch
):
    """Drift and a real regression at once: the diff is still slower than
    the base, so the base cannot excuse it."""
    pr = _pr_failure(345.0, 481.0, ctx, choice, monkeypatch)

    verdict = stress_e2e_bench._resolve_bench_against_base(
        pr, _base(344.0, 418.0), choice.model_id
    )

    assert not verdict["preexisting"], verdict["finding"]


def test_a_measurement_inside_the_baseline_is_never_deferred(
    ctx, choice, baseline, monkeypatch
):
    """A green bench must resolve on the spot — no base measurement, and
    nothing handed to the resolver."""
    monkeypatch.setattr(stress_e2e_bench, "_measure_bench", _fake_measure(252.0, 380.0))

    result = stress_e2e_bench._run_bench(ctx, choice)

    assert result["status"] == "pass", result["summary"]
    assert "bench_comparison" not in result


def test_an_unmeasurable_base_reports_the_miss_without_claiming_it_is_confirmed(
    ctx, choice, baseline, monkeypatch
):
    """No control (worktree failed, base server would not boot) means the
    baseline verdict stands — stated as unconfirmed, not as proven."""
    pr = _pr_failure(345.0, 418.0, ctx, choice, monkeypatch)

    verdict = stress_e2e_bench._resolve_bench_against_base(
        pr,
        {
            "status": "error",
            "summary": "base check failed: git worktree add exited 128",
            "executed": False,
        },
        choice.model_id,
    )

    assert not verdict["preexisting"], verdict["finding"]
    assert "could not measure the base" in verdict["finding"]
    assert "git worktree add exited 128" in verdict["finding"]


def test_thermal_scale_drift_cannot_buy_an_excuse(ctx, choice, baseline, monkeypatch):
    """The base is measured after stress + the agent matrix, so the machine
    is warmer than it was for the PR's bench on a fresh server. That biases
    the base slow, and a slow base is what excuses a PR — the wrong
    direction. A base that is only a couple of percent off must therefore
    excuse nothing: it never breaches the gate on its own."""
    # Numbers chosen to land in the gap this clause exists for — if the
    # PR/base comparison alone already tripped, the clause would never be
    # reached and this test would pass without testing it:
    #   baseline warm 375, threshold 5% (trips above 393.75)
    #   PR   warm 400  → +6.7% vs baseline, so the bench fails and we land here
    #   base warm 385  → PR is only +3.9% vs base: WITHIN threshold
    #   base drift     → +2.7% vs baseline: does NOT breach on its own
    # Without the clause that combination reads as "drift, excused". With
    # it, a base that never fails the gate cannot vouch for one that does.
    pr = _pr_failure(255.0, 400.0, ctx, choice, monkeypatch)

    verdict = stress_e2e_bench._resolve_bench_against_base(
        pr, _base(252.0, 385.0), choice.model_id
    )

    assert not verdict["preexisting"], verdict["finding"]


def test_a_dependency_change_is_never_excused_by_the_base(
    ctx, choice, baseline, monkeypatch
):
    """The control checks out base SOURCE but runs it against the CURRENTLY
    installed environment. A slowdown that arrived with a dependency bump
    therefore lands in the base measurement too, and the two agreeing proves
    nothing. When the diff touches packaging, the frozen baseline keeps the
    last word."""
    pr = _pr_failure(345.0, 418.0, ctx, choice, monkeypatch)

    verdict = stress_e2e_bench._resolve_bench_against_base(
        pr,
        _base(344.0, 419.0),
        choice.model_id,
        files_changed=["pyproject.toml", "vllm_mlx/engine.py"],
    )

    assert not verdict["preexisting"], verdict["finding"]
    assert "dependency/packaging" in verdict["finding"]

    # Same numbers, ordinary source change → the base is allowed to speak.
    ordinary = stress_e2e_bench._resolve_bench_against_base(
        pr, _base(344.0, 419.0), choice.model_id, files_changed=["vllm_mlx/engine.py"]
    )
    assert ordinary["preexisting"], ordinary["finding"]
