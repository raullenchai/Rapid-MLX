from dataclasses import replace
from types import SimpleNamespace

from bench.bench_spec_decode_mtp import (
    RunResult,
    _evaluate_landing_gates,
    _summarize,
    _tokenizer_stop_tokens,
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
    assert gates["lossless"]["mismatches"] == [{"run_idx": 0, "prompt_idx": 0}]


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
