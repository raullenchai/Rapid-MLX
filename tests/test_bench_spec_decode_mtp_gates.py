from dataclasses import replace
from types import SimpleNamespace

import pytest

from bench.bench_spec_decode_mtp import (
    RunResult,
    _assert_repo_module,
    _balanced_condition_order,
    _evaluate_landing_gates,
    _summarize,
    _tokenizer_stop_tokens,
    _with_paired_speedup,
)


def _run(condition: str, run: int, prompt: int, digest: str) -> RunResult:
    return RunResult(
        condition=condition,
        run_idx=run,
        prompt_idx=prompt,
        decode_tok_per_sec=10.0,
        n_tokens=10,
        accept_attempts=0,
        accept_count=0,
        elapsed_seconds=1.0,
        decode_elapsed_seconds=1.0,
        prompt_eval_seconds=0.0,
        end_to_end_tok_per_sec=10.0,
        token_sha256=digest,
        verify_kernel_calls=0,
        verify_kernel_fallbacks=0,
        verify_sync_seconds=0.0,
        draft_seconds=0.0,
        residual_sync_seconds=0.0,
        verify_calls=0,
        prompt_lookup_proposals=0,
        prompt_lookup_drafted_tokens=0,
        prompt_lookup_accepted_tokens=0,
        prompt_lookup_rejections=0,
        prompt_lookup_mtp_sync_seconds=0.0,
    )


def test_landing_gates_pass_complete_lossless_speedup():
    results = {
        "none": [_run("none", 0, 0, "same")],
        "mtp": [_run("mtp", 0, 0, "same")],
    }
    gates = _evaluate_landing_gates(
        results,
        expected_pairs=1,
        require_lossless=True,
        min_speedup=1.1,
        speedup=1.25,
    )
    assert gates["passed"] is True


def test_landing_gates_report_hash_mismatch():
    results = {
        "none": [_run("none", 0, 0, "ar")],
        "mtp": [_run("mtp", 0, 0, "mtp")],
    }
    gates = _evaluate_landing_gates(
        results,
        expected_pairs=1,
        require_lossless=True,
        min_speedup=None,
        speedup=2.0,
    )
    assert gates["passed"] is False
    assert gates["lossless"]["per_baseline"]["none"]["mismatches"] == [
        {"run_idx": 0, "prompt_idx": 0}
    ]


def test_optional_lossless_gate_still_reports_observed_mismatch():
    results = {
        "none": [_run("none", 0, 0, "ar")],
        "mtp": [_run("mtp", 0, 0, "mtp")],
    }
    gates = _evaluate_landing_gates(
        results,
        expected_pairs=1,
        require_lossless=False,
        min_speedup=None,
        speedup=2.0,
    )
    assert gates["passed"] is True
    assert gates["lossless"]["required"] is False
    assert gates["lossless"]["passed"] is False
    assert gates["lossless"]["per_baseline"]["none"]["passed"] is False


def test_landing_gates_fail_closed_on_missing_pair():
    results = {"none": [_run("none", 0, 0, "same")], "mtp": []}
    gates = _evaluate_landing_gates(
        results,
        expected_pairs=1,
        require_lossless=True,
        min_speedup=1.1,
        speedup=None,
    )
    assert gates["passed"] is False
    assert gates["complete_pairs"] == 0
    assert gates["performance"]["passed"] is False


def test_summary_pools_decode_time_not_prompt_eval_time():
    run = replace(
        _run("mtp", 0, 0, "same"),
        elapsed_seconds=2.0,
        decode_elapsed_seconds=1.0,
        prompt_eval_seconds=1.0,
    )

    summary = _summarize("mtp", [run], None)

    assert summary.pooled_tok_per_sec == 10.0


def test_tokenizer_stop_tokens_prefers_plural_set():
    tokenizer = SimpleNamespace(eos_token_ids={1, 2}, eos_token_id=1)
    assert _tokenizer_stop_tokens(tokenizer) == {1, 2}


def test_tokenizer_stop_tokens_falls_back_to_singular():
    tokenizer = SimpleNamespace(eos_token_ids=None, eos_token_id=7)
    assert _tokenizer_stop_tokens(tokenizer) == {7}


def test_balanced_condition_order_rotates_each_arm_through_each_slot():
    conditions = ("none", "ar", "mtp")
    orders = [_balanced_condition_order(conditions, cell) for cell in range(3)]

    assert orders == [
        ("none", "ar", "mtp"),
        ("ar", "mtp", "none"),
        ("mtp", "none", "ar"),
    ]
    assert [order[0] for order in orders] == ["none", "ar", "mtp"]


def test_same_generator_comparison_keeps_its_own_paired_reference():
    ar = [replace(_run("ar", 0, 0, "ar"), decode_tok_per_sec=10.0)]
    mtp = [
        replace(
            _run("mtp", 0, 0, "mtp"),
            decode_tok_per_sec=12.0,
            n_tokens=12,
        )
    ]
    summary = _with_paired_speedup(
        _summarize("mtp_vs_ar", mtp, baseline_tok_per_sec=10.0),
        mtp,
        ar,
    )

    assert summary.speedup_vs_baseline == 1.2
    assert summary.paired_speedup_median == 1.2


def test_landing_gate_prefers_product_baseline_when_both_baselines_run():
    results = {
        "none": [_run("none", 0, 0, "same")],
        "ar": [_run("ar", 0, 0, "same")],
        "mtp": [_run("mtp", 0, 0, "same")],
    }

    gates = _evaluate_landing_gates(
        results,
        expected_pairs=1,
        require_lossless=False,
        min_speedup=1.0,
        speedup=1.1,
    )

    assert gates["performance"]["reference_arm"] == "none"


def test_benchmark_rejects_installed_package_outside_checkout():
    fake_module = SimpleNamespace(__file__="/tmp/site-packages/vllm_mlx/__init__.py")
    with pytest.raises(RuntimeError, match="outside this checkout"):
        _assert_repo_module(fake_module)
