# SPDX-License-Identifier: Apache-2.0
"""Hermetic tests for the KV-quant differential quality gate (absorb #4).

All metric tests use SYNTHETIC baseline/candidate token streams + logprob
vectors + byte counts — no model inference. A single ``@pytest.mark.slow`` smoke
runs the real harness on ONE small already-cached model and SKIPS cleanly when
the model isn't in the local HF cache (never downloads).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from vllm_mlx.kv_quant_gate import (
    FAIL,
    NA,
    PASS,
    AgreementResult,
    LogitDivergence,
    RetentionResult,
    build_report,
    default_thresholds,
    greedy_agreement_rate,
    is_valid_json,
    logit_divergence,
    memory_delta,
    structured_output_retention,
)


def _logsoftmax(logits: list[float]) -> np.ndarray:
    arr = np.asarray(logits, dtype=np.float64)
    return arr - float(np.log(np.sum(np.exp(arr))))


# ---------------------------------------------------------------------------
# Greedy agreement
# ---------------------------------------------------------------------------
def test_greedy_agreement_identical():
    r = greedy_agreement_rate([1, 2, 3, 4], [1, 2, 3, 4])
    assert r.rate == 1.0
    assert r.matched == 4
    assert r.total == 4
    assert r.first_divergence_index is None


def test_greedy_agreement_one_token_diff():
    # Diverges at index 2 -> 2 leading matches out of 4 compared.
    r = greedy_agreement_rate([1, 2, 3, 4], [1, 2, 9, 4])
    assert r.first_divergence_index == 2
    assert r.matched == 2
    assert r.total == 4
    assert r.rate == 0.5


def test_greedy_agreement_diverge_immediately():
    r = greedy_agreement_rate([5, 6], [9, 6])
    assert r.first_divergence_index == 0
    assert r.matched == 0
    assert r.rate == 0.0


def test_greedy_agreement_empty_streams():
    r = greedy_agreement_rate([], [])
    assert r.total == 0
    assert r.rate == 1.0
    assert r.first_divergence_index is None


def test_greedy_agreement_premature_eos_scores_below_one():
    """RED-GREEN: a candidate that ends early after an all-matching prefix must
    NOT score 1.0. With the old ``min`` denominator this returned 1.0, hiding the
    premature-EOS degradation the gate exists to catch; the ``max`` denominator
    counts the missing suffix positions as disagreement.
    """
    # baseline 5 tokens, candidate terminated at 3 (premature EOS), prefix matches
    r = greedy_agreement_rate([1, 2, 3, 4, 5], [1, 2, 3])
    assert r.total == 5  # denominator is the LONGER stream, not the shorter
    assert r.matched == 3
    assert r.rate == 0.6
    assert r.first_divergence_index == 3  # boundary where the short stream ended


def test_greedy_agreement_candidate_longer_also_penalized():
    r = greedy_agreement_rate([1, 2], [1, 2, 3, 4])
    assert r.total == 4
    assert r.matched == 2
    assert r.rate == 0.5
    assert r.first_divergence_index == 2


def test_greedy_agreement_empty_vs_nonempty_scores_zero():
    r = greedy_agreement_rate([], [1, 2, 3])
    assert r.total == 3
    assert r.matched == 0
    assert r.rate == 0.0


# ---------------------------------------------------------------------------
# Logit divergence
# ---------------------------------------------------------------------------
def test_logit_divergence_identical_is_zero():
    base = [_logsoftmax([2.0, 1.0, 0.0]), _logsoftmax([0.0, 3.0, 1.0])]
    r = logit_divergence(base, base)
    assert r.compared_steps == 2
    assert r.mean_kl == pytest.approx(0.0, abs=1e-9)
    assert r.max_kl == pytest.approx(0.0, abs=1e-9)
    assert r.top1_agreement_rate == 1.0


def test_logit_divergence_degraded_positive_kl_and_top1_drop():
    base = [_logsoftmax([5.0, 0.0, 0.0])]  # argmax 0
    cand = [_logsoftmax([0.0, 5.0, 0.0])]  # argmax 1
    r = logit_divergence(base, cand)
    assert r.compared_steps == 1
    assert r.mean_kl > 0.1
    assert r.top1_agreement_rate == 0.0


def test_logit_divergence_compare_len_bounds_steps():
    base = [_logsoftmax([3.0, 0.0]), _logsoftmax([0.0, 3.0]), _logsoftmax([1.0, 0.0])]
    cand = list(base)
    r = logit_divergence(base, cand, compare_len=1)
    assert r.compared_steps == 1


def test_logit_divergence_catastrophic_neg_inf_is_finite():
    # Baseline has mass at index 0; candidate assigns it -inf logprob.
    base = [_logsoftmax([10.0, 0.0])]
    cand = [np.array([-np.inf, 0.0])]
    r = logit_divergence(base, cand)
    assert math.isfinite(r.mean_kl)
    assert r.mean_kl > 1.0  # clearly flags the divergence


def test_logit_divergence_nan_in_logits_is_finite():
    # A NaN anywhere in either vector must NOT poison the aggregate KL (which
    # would make the whole gate report NaN and silently pass every threshold).
    base = [_logsoftmax([4.0, 0.0, 1.0]), _logsoftmax([1.0, 2.0, 0.0])]
    cand = [np.array([np.nan, 0.0, 1.0]), _logsoftmax([1.0, 2.0, 0.0])]
    r = logit_divergence(base, cand)
    assert math.isfinite(r.mean_kl)
    assert math.isfinite(r.max_kl)
    assert r.mean_kl > 0.0  # the NaN step is charged the ceiling, not dropped
    assert r.compared_steps == 2


def test_logit_divergence_all_nan_stays_finite_and_high():
    base = [_logsoftmax([3.0, 0.0])]
    cand = [np.array([np.nan, np.nan])]
    r = logit_divergence(base, cand)
    assert math.isfinite(r.mean_kl)
    assert r.mean_kl > 1.0


def test_logit_divergence_unnormalized_shift_invariant():
    # RED-GREEN (codex round 4 blocking #2): the SAME distribution expressed as
    # raw logits offset by DIFFERENT additive constants is identical as a
    # probability distribution -> forward KL must be ~0. The half-normalized code
    # (weights renormalized, but the log-ratio taken on raw values) reported a
    # spurious non-zero here; renormalizing BOTH vectors via log-sum-exp fixes it.
    logits = np.array([2.0, 1.0, 0.5, -1.0, 3.0])
    base = [logits + 7.0]  # shifted up
    cand = [logits - 4.0]  # shifted down — same softmax
    r = logit_divergence(base, cand)
    assert r.compared_steps == 1
    assert r.mean_kl == pytest.approx(0.0, abs=1e-9)
    assert r.max_kl == pytest.approx(0.0, abs=1e-9)
    assert r.top1_agreement_rate == 1.0


def test_logit_divergence_raw_logits_match_logsoftmax_inputs():
    # Feeding raw logits must give the same KL as feeding their log_softmax — the
    # metric renormalizes internally, so callers need not pre-normalize.
    base_logits = np.array([4.0, 1.0, 0.0, 2.0])
    cand_logits = np.array([3.0, 2.0, 0.0, 1.0])
    r_raw = logit_divergence([base_logits], [cand_logits])
    r_norm = logit_divergence([_logsoftmax(base_logits)], [_logsoftmax(cand_logits)])
    assert r_raw.mean_kl == pytest.approx(r_norm.mean_kl, abs=1e-9)
    assert r_raw.mean_kl > 0.0


def test_logit_divergence_empty():
    r = logit_divergence([], [])
    assert r.compared_steps == 0
    assert r.mean_kl == 0.0
    assert r.top1_agreement_rate == 1.0


def test_logit_divergence_shape_mismatch_step_skipped():
    base = [_logsoftmax([1.0, 2.0, 3.0])]
    cand = [_logsoftmax([1.0, 2.0])]  # wrong vocab size -> skipped
    r = logit_divergence(base, cand)
    assert r.compared_steps == 0


# ---------------------------------------------------------------------------
# Structured-output retention + JSON helpers
# ---------------------------------------------------------------------------
def test_is_valid_json_strict_whole_output():
    # Whole output IS one JSON value (optionally a single fence) -> valid.
    assert is_valid_json('{"a": 1}')
    assert is_valid_json('```json\n{"a": 1}\n```')
    assert is_valid_json("[1, 2, 3]")
    assert is_valid_json("  42  ")  # a scalar is a valid JSON document
    # Empty / non-JSON -> invalid.
    assert not is_valid_json("not json at all")
    assert not is_valid_json("")


def test_is_valid_json_strict_rejects_prose_wrapped_json():
    """RED-GREEN: under the 'ONLY JSON' contract, JSON buried in prose is NOT
    valid. A lenient raw_decode/first-brace scan would (wrongly) accept these;
    strict whole-output parsing rejects them so a candidate that stops honoring
    the format contract is caught.
    """
    assert not is_valid_json('Sure! Here you go: {"a": [1, 2, 3]} — done.')
    assert not is_valid_json("Sorry, [1]")
    assert not is_valid_json('see [x] then {"a": 1}')
    assert not is_valid_json('result: [1, 2] and {"ok": true}')


def test_retention_both_valid():
    r = structured_output_retention([('{"a": 1}', '{"a": 2}')])
    assert r.attributable == 1
    assert r.retained == 1
    assert r.rate == 1.0
    assert r.baseline_invalid == 0


def test_retention_candidate_breaks():
    r = structured_output_retention([('{"a": 1}', "sorry, no json")])
    assert r.attributable == 1
    assert r.retained == 0
    assert r.rate == 0.0


def test_retention_baseline_invalid_is_excluded():
    r = structured_output_retention([("garbage", "garbage")])
    assert r.attributable == 0
    assert r.baseline_invalid == 1
    assert r.rate is None  # N/A — nothing attributable


def test_retention_mixed():
    pairs = [
        ('{"a": 1}', '{"a": 1}'),  # attributable, retained
        ('{"b": 2}', "broke"),  # attributable, lost
        ("not json", "not json"),  # excluded
    ]
    r = structured_output_retention(pairs)
    assert r.attributable == 2
    assert r.retained == 1
    assert r.rate == 0.5
    assert r.baseline_invalid == 1


def test_retention_shape_mismatch_not_retained():
    """RED-GREEN: an object prompt must NOT 'retain' as an array/scalar. Generic
    JSON validity would count ``[1, 2]`` as retained; shape-aware retention
    (same top-level type) correctly rejects it.
    """
    r = structured_output_retention([('{"a": 1}', "[1, 2]")])
    assert r.attributable == 1
    assert r.retained == 0
    assert r.rate == 0.0


def test_retention_dropped_key_not_retained():
    """RED-GREEN: a candidate object that drops a baseline key is NOT retained —
    the requested field vanished under quantization.
    """
    r = structured_output_retention([('{"name": "x", "age": 3}', '{"name": "x"}')])
    assert r.attributable == 1
    assert r.retained == 0
    assert r.rate == 0.0
    # Superset of keys is fine (candidate added a field but kept the contract).
    r2 = structured_output_retention([('{"name": "x"}', '{"name": "y", "extra": 1}')])
    assert r2.retained == 1


# ---------------------------------------------------------------------------
# Memory delta
# ---------------------------------------------------------------------------
def test_memory_delta_typical_int8():
    d = memory_delta(1000, 500)
    assert d.saved_bytes == 500
    assert d.reduction_ratio == 2.0
    assert d.saved_pct == 50.0


def test_memory_delta_candidate_zero_is_degenerate():
    d = memory_delta(1000, 0)
    assert d.reduction_ratio == 0.0  # treated as no saving, not div-by-zero


def test_memory_delta_no_saving():
    d = memory_delta(1000, 1000)
    assert d.reduction_ratio == 1.0
    assert d.saved_bytes == 0


# ---------------------------------------------------------------------------
# Baseline KV-cache dtype detection (harness helper, hermetic — no mlx load)
# ---------------------------------------------------------------------------
class _FakeDtype:
    def __init__(self, name: str) -> None:
        self._name = name

    def __str__(self) -> str:  # mimics mlx: str(mx.bfloat16) == "mlx.core.bfloat16"
        return self._name


class _FakeArr:
    def __init__(self, dtype: _FakeDtype) -> None:
        self.dtype = dtype


class _FakeKVLayer:
    """A plain (non-quantized) KVCache exposes ``keys`` as a single array."""

    def __init__(self, dtype_name: str) -> None:
        self.keys = _FakeArr(_FakeDtype(dtype_name))


class _FakeQuantLayer:
    """A QuantizedKVCache exposes ``keys`` as a (packed, scales, biases) tuple."""

    def __init__(self) -> None:
        self.keys = (_FakeArr(_FakeDtype("uint32")), object(), object())


def test_baseline_kv_dtype_reads_real_dtype_not_hardcoded():
    """RED-GREEN: report must reflect the loaded model's actual KV dtype.

    Hardcoding ``"bf16"`` mislabels the many fp16 mlx checkpoints; the helper
    strips the ``mlx.core.`` prefix and returns the true element dtype.
    """
    from scripts.kv_quant_quality_gate import _baseline_kv_dtype

    assert _baseline_kv_dtype([_FakeKVLayer("mlx.core.bfloat16")]) == "bfloat16"
    assert _baseline_kv_dtype([_FakeKVLayer("mlx.core.float16")]) == "float16"


def test_baseline_kv_dtype_ignores_quantized_and_returns_unknown():
    from scripts.kv_quant_quality_gate import _baseline_kv_dtype

    # A quantized layer's ``keys`` is a tuple -> not a plain baseline dtype.
    assert _baseline_kv_dtype([_FakeQuantLayer()]) == "unknown"
    assert _baseline_kv_dtype([]) == "unknown"


# ---------------------------------------------------------------------------
# Decode strips special tokens (harness helper) — codex round 4 blocking #1
# ---------------------------------------------------------------------------
class _FakeTokenizerWithEOS:
    """Greedy generation stops ON the EOS token (id 9), which is appended to the
    stream; a naive decode leaves ``<|im_end|>`` trailing and breaks strict JSON.
    """

    _VOCAB = {1: "{", 2: '"ok":', 3: "true", 4: "}", 9: "<|im_end|>"}

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        text = "".join(self._VOCAB[i] for i in ids)
        if skip_special_tokens:
            text = text.replace("<|im_end|>", "")
        return text


class _OldTokenizerNoKwarg:
    def decode(self, ids) -> str:  # no skip_special_tokens kwarg at all
        return "x" * len(ids)


def test_decode_text_strips_trailing_eos_so_strict_json_holds():
    """RED-GREEN: retention decoded WITHOUT skip_special_tokens keeps a trailing
    EOS control token, which fails the strict whole-output JSON parse and falsely
    scores structured retention N/A/FAIL. ``_decode_text`` must strip it.
    """
    from scripts.kv_quant_quality_gate import _decode_text
    from vllm_mlx.kv_quant_gate import is_valid_json

    tok = _FakeTokenizerWithEOS()
    with_eos = [1, 2, 3, 4, 9]  # JSON then the appended EOS
    assert tok.decode(with_eos) == '{"ok":true}<|im_end|>'
    assert not is_valid_json(tok.decode(with_eos))  # the bug: prose-wrapped
    text = _decode_text(tok, with_eos)
    assert text == '{"ok":true}'
    assert is_valid_json(text)  # fixed: strict JSON survives


def test_decode_text_falls_back_when_kwarg_unsupported():
    from scripts.kv_quant_quality_gate import _decode_text

    # A tokenizer whose decode rejects the kwarg must not crash the gate.
    assert _decode_text(_OldTokenizerNoKwarg(), [1, 2, 3]) == "xxx"


# ---------------------------------------------------------------------------
# Offline enforcement is reliable even if huggingface_hub was imported first
# (codex round 5 blocking) — hermetic, no model load.
# ---------------------------------------------------------------------------
def test_enable_hf_offline_overrides_preimported_constant():
    """RED-GREEN: ``huggingface_hub.constants`` snapshots ``HF_HUB_OFFLINE`` from
    the env ONCE at import; ``is_offline_mode()`` returns that frozen global. If
    the hub was imported while ONLINE, setting the env var alone leaves
    ``is_offline_mode()`` False and a network fetch can still happen.
    ``_enable_hf_offline`` must also patch the in-memory constant so the "no
    fetch" contract holds regardless of import order.
    """
    import os

    hf = pytest.importorskip("huggingface_hub.constants")
    from scripts.kv_quant_quality_gate import _enable_hf_offline

    saved_const = hf.HF_HUB_OFFLINE
    saved_env = {
        k: os.environ.get(k) for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    }
    try:
        # Simulate a process that imported the hub while online.
        hf.HF_HUB_OFFLINE = False
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
        assert hf.is_offline_mode() is False  # precondition: online

        _enable_hf_offline()

        # Env-var-only would leave this False; the constant patch flips it.
        assert hf.is_offline_mode() is True
        assert os.environ["HF_HUB_OFFLINE"] == "1"
        assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
    finally:
        hf.HF_HUB_OFFLINE = saved_const
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ---------------------------------------------------------------------------
# Thresholds + report schema + PASS/FAIL logic
# ---------------------------------------------------------------------------
def test_default_thresholds_int4_more_lenient_than_int8():
    t8 = default_thresholds("int8")
    t4 = default_thresholds("int4")
    assert t4.min_greedy_agreement < t8.min_greedy_agreement
    assert t4.max_mean_kl > t8.max_mean_kl
    assert t4.min_memory_reduction_ratio > t8.min_memory_reduction_ratio


def test_default_thresholds_rejects_unsupported_dtype():
    """RED-GREEN: an unknown dtype must raise, not silently borrow int8's
    thresholds (which would give a misleading verdict for a typo/future dtype).
    """
    for bad in ("int2", "bf16", "fp8", ""):
        with pytest.raises(ValueError):
            default_thresholds(bad)


def _passing_inputs():
    """Metric objects that sit exactly at / above the int8 thresholds -> all PASS."""
    t = default_thresholds("int8")
    return dict(
        thresholds=t,
        agreement=AgreementResult(
            total=10, matched=6, rate=t.min_greedy_agreement, first_divergence_index=6
        ),
        logits=LogitDivergence(
            compared_steps=8,
            mean_kl=t.max_mean_kl,  # exactly at bound -> PASS (<=)
            max_kl=t.max_mean_kl,
            top1_agreement_rate=0.9,
        ),
        retention=RetentionResult(
            attributable=2, retained=2, rate=1.0, baseline_invalid=0
        ),
        memory=memory_delta(2000, 1000),  # 2.0x >= 1.5
    )


def _base_report_kwargs():
    return dict(
        model="synthetic",
        hf_path="synthetic/path",
        baseline_dtype="bf16",
        candidate_dtype="int8",
        num_prompts=5,
        advisory=True,
        chip={
            "raw": "Apple M3 Ultra",
            "is_apple_silicon": True,
            "generation": 3,
            "variant": "Ultra",
            "is_m3_or_newer": True,
        },
    )


def test_report_schema_has_expected_fields():
    report = build_report(**_base_report_kwargs(), **_passing_inputs())
    d = report.to_dict()
    for key in (
        "schema",
        "kind",
        "advisory",
        "model",
        "hf_path",
        "baseline_dtype",
        "candidate_dtype",
        "num_prompts",
        "chip",
        "thresholds",
        "agreement",
        "logits",
        "retention",
        "memory",
        "niah",
        "metrics",
        "overall",
    ):
        assert key in d, key
    assert set(d["metrics"]) == {
        "greedy_agreement",
        "logit_mean_kl",
        "structured_retention",
        "memory_reduction",
    }
    # to_dict must be JSON-serializable end to end.
    import json

    json.loads(json.dumps(d))


def test_report_all_pass_at_thresholds():
    report = build_report(**_base_report_kwargs(), **_passing_inputs())
    assert report.overall == PASS
    for m in report.metrics.values():
        assert m["outcome"] == PASS


def test_report_agreement_just_below_threshold_fails():
    inputs = _passing_inputs()
    t = inputs["thresholds"]
    inputs["agreement"] = AgreementResult(
        total=100,
        matched=int((t.min_greedy_agreement - 0.01) * 100),
        rate=t.min_greedy_agreement - 0.01,
        first_divergence_index=1,
    )
    report = build_report(**_base_report_kwargs(), **inputs)
    assert report.metrics["greedy_agreement"]["outcome"] == FAIL
    assert report.overall == FAIL


def test_report_kl_above_threshold_fails():
    inputs = _passing_inputs()
    t = inputs["thresholds"]
    inputs["logits"] = LogitDivergence(
        compared_steps=4,
        mean_kl=t.max_mean_kl + 0.5,
        max_kl=t.max_mean_kl + 0.5,
        top1_agreement_rate=0.5,
    )
    report = build_report(**_base_report_kwargs(), **inputs)
    assert report.metrics["logit_mean_kl"]["outcome"] == FAIL
    assert report.overall == FAIL


def test_report_memory_below_reduction_fails():
    inputs = _passing_inputs()
    inputs["memory"] = memory_delta(1000, 900)  # 1.11x < 1.5x
    report = build_report(**_base_report_kwargs(), **inputs)
    assert report.metrics["memory_reduction"]["outcome"] == FAIL
    assert report.overall == FAIL


def test_report_na_metrics_do_not_fail_overall():
    inputs = _passing_inputs()
    # No comparable logit steps -> logit_mean_kl NA.
    inputs["logits"] = LogitDivergence(
        compared_steps=0, mean_kl=0.0, max_kl=0.0, top1_agreement_rate=1.0
    )
    # No attributable JSON prompt -> structured_retention NA.
    inputs["retention"] = RetentionResult(
        attributable=0, retained=0, rate=None, baseline_invalid=3
    )
    report = build_report(**_base_report_kwargs(), **inputs)
    assert report.metrics["logit_mean_kl"]["outcome"] == NA
    assert report.metrics["structured_retention"]["outcome"] == NA
    # Remaining graded metrics still pass -> overall PASS.
    assert report.overall == PASS


def test_report_niah_participates_in_overall():
    passing = _passing_inputs()
    skipped = build_report(
        **_base_report_kwargs(), **passing, niah={"status": "skipped"}
    )
    assert skipped.overall == PASS  # skipped NIAH is NA

    failed = build_report(**_base_report_kwargs(), **passing, niah={"status": "fail"})
    assert failed.overall == FAIL  # a failing NIAH drags overall down

    passed = build_report(**_base_report_kwargs(), **passing, niah={"status": "pass"})
    assert passed.overall == PASS


def test_human_summary_renders():
    report = build_report(**_base_report_kwargs(), **_passing_inputs())
    text = report.human_summary()
    assert "KV-quant differential quality gate" in text
    assert "ADVISORY" in text
    assert "OVERALL:" in text


# ---------------------------------------------------------------------------
# Harness guards (hermetic — no model load; the guards run before any import).
# ---------------------------------------------------------------------------
def _run_gate_kwargs(**overrides):
    base = dict(
        model_arg="dummy",
        candidate_dtypes=["int8"],
        prompts=[{"kind": "text", "prompt": "hi"}],
        max_tokens=8,
        mem_tokens=16,
        kv_group_size=64,
        advisory=True,
        run_niah=False,
        json_out=None,
    )
    base.update(overrides)
    return base


@pytest.mark.parametrize(
    "override",
    [
        {"max_tokens": 0},
        {"max_tokens": -1},
        {"mem_tokens": 0},
        {"kv_group_size": 0},
        {"prompts": []},
        {"candidate_dtypes": []},  # empty -> enforced no-op "pass" without it
    ],
)
def test_run_gate_rejects_bad_config_before_load(override):
    """RED-GREEN: non-positive token budgets / empty prompts raise up front.

    A zero/negative budget yields empty generations that would score a vacuous
    1.0 agreement — an enforced gate must never 'pass' without measuring. The
    guard runs before any mlx_lm import, so this is hermetic (no model load).
    """
    from scripts.kv_quant_quality_gate import run_gate

    with pytest.raises(ValueError):
        run_gate(**_run_gate_kwargs(**override))


def test_positive_int_argparse_type_rejects_nonpositive():
    import argparse

    from scripts.kv_quant_quality_gate import _positive_int

    assert _positive_int("48") == 48
    for bad in ("0", "-3"):
        with pytest.raises(argparse.ArgumentTypeError):
            _positive_int(bad)


def test_niah_fail_closed_on_unknown_ram():
    """RED-GREEN: when RAM is undetected (None) NIAH must SKIP, not run.

    The chip qualifies (M3 Ultra) so the RAM guard is the deciding factor;
    unknown capacity must fail closed rather than assume headroom.
    """
    from scripts.kv_quant_quality_gate import _maybe_run_niah
    from vllm_mlx.chip_tier import classify_chip_tier

    chip = classify_chip_tier("Apple M3 Ultra")
    assert chip.is_m3_or_newer  # precondition — RAM guard is what decides
    result = _maybe_run_niah(
        None,
        None,
        chip,
        enabled=True,
        max_tokens=8,
        kv_bits=8,
        kv_group_size=64,
        eos_ids=set(),
        total_ram_gb=None,
    )
    assert result["status"] == "skipped"
    assert "undetected" in result["reason"] or "fail-closed" in result["reason"]


@pytest.mark.real_hf_cache
def test_niah_skips_when_not_requested_and_sub_m3():
    from scripts.kv_quant_quality_gate import _maybe_run_niah
    from vllm_mlx.chip_tier import classify_chip_tier

    m3 = classify_chip_tier("Apple M3 Ultra")
    m1 = classify_chip_tier("Apple M1")
    # Not requested -> skipped regardless of chip.
    r1 = _maybe_run_niah(
        None,
        None,
        m3,
        enabled=False,
        max_tokens=8,
        kv_bits=8,
        kv_group_size=64,
        eos_ids=set(),
        total_ram_gb=256.0,
    )
    assert r1["status"] == "skipped" and "not requested" in r1["reason"]
    # Sub-M3 chip -> skipped even when requested with ample RAM.
    r2 = _maybe_run_niah(
        None,
        None,
        m1,
        enabled=True,
        max_tokens=8,
        kv_bits=8,
        kv_group_size=64,
        eos_ids=set(),
        total_ram_gb=256.0,
    )
    assert r2["status"] == "skipped" and "below M3" in r2["reason"]


# ---------------------------------------------------------------------------
# Gated end-to-end smoke — real harness, cached small model only.
# ---------------------------------------------------------------------------
# Small, quantizable (dense, non-sliding-window) text models to try, smallest
# first. The smoke SKIPS unless one is already in the local HF cache.
_SMOKE_MODEL_CANDIDATES = [
    "mlx-community/Qwen3-0.6B-4bit",
    "mlx-community/Qwen3-0.6B-8bit",
    "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "mlx-community/Phi-3-mini-4k-instruct-4bit",
]


# A COMPLETE snapshot needs config + tokenizer config + a tokenizer data file +
# ALL weight shards. Probing only ``config.json`` (or an index without its
# shards) would let ``mlx_lm.load`` fetch a missing file over the network —
# violating the cache-only (no-download) contract.
_REQUIRED_CACHE_FILES = ("config.json", "tokenizer_config.json")
# Tokenizer payload — a repo ships exactly one of these; require at least one.
_TOKENIZER_DATA_FILES = ("tokenizer.json", "tokenizer.model", "vocab.json")


def _model_fully_cached(repo: str) -> bool:
    """True iff a COMPLETE local snapshot of ``repo`` is in the HF cache.

    Deterministic cache probe (``try_to_load_from_cache`` — a real ``str`` path
    means present). Verifies config + tokenizer config + a tokenizer data file +
    a full weight set: either a single ``model.safetensors`` OR a
    ``model.safetensors.index.json`` whose ``weight_map`` shards are ALL cached.
    No network, no exception-message classification (the flaky pattern the gemma4
    tests warned against).
    """
    try:
        import json as _json
        from pathlib import Path as _Path

        from huggingface_hub import try_to_load_from_cache
    except Exception:
        return False

    def cached_path(filename: str) -> str | None:
        try:
            path = try_to_load_from_cache(repo, filename)
        except Exception:
            return None
        return path if isinstance(path, str) else None

    if not all(cached_path(f) for f in _REQUIRED_CACHE_FILES):
        return False
    if not any(cached_path(f) for f in _TOKENIZER_DATA_FILES):
        return False

    # Single-file weights.
    if cached_path("model.safetensors"):
        return True
    # Sharded weights: the index must be cached AND every shard it references.
    index_path = cached_path("model.safetensors.index.json")
    if index_path is None:
        return False
    try:
        weight_map = _json.loads(_Path(index_path).read_text())["weight_map"]
        shards = set(weight_map.values())
    except Exception:
        return False
    return bool(shards) and all(cached_path(shard) for shard in shards)


def _first_cached_model() -> str | None:
    """Return the first candidate with a COMPLETE local snapshot, else None."""
    for repo in _SMOKE_MODEL_CANDIDATES:
        if _model_fully_cached(repo):
            return repo
    return None


@pytest.mark.slow
def test_smoke_real_harness_on_cached_model(tmp_path, monkeypatch):
    """End-to-end: load a real cached small model, run the gate, assert the report.

    Cache-only (never downloads): skips unless a COMPLETE snapshot is cached, and
    forces HF offline mode so the load can NEVER reach the network. Run with
    ``pytest --run-slow -m slow tests/test_kv_quant_gate.py``.
    """
    pytest.importorskip("mlx_lm")
    repo = _first_cached_model()
    if repo is None:
        pytest.skip(
            "no COMPLETE small-model snapshot in the local HF cache — smoke is "
            "cache-only (no download). Pre-cache one of: "
            + ", ".join(_SMOKE_MODEL_CANDIDATES)
        )

    # Belt-and-suspenders: even though the snapshot is complete, force offline so
    # a stray fetch is impossible (raises instead of downloading).
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    from scripts.kv_quant_quality_gate import run_gate

    out = tmp_path / "report.json"
    rc = run_gate(
        model_arg=repo,
        candidate_dtypes=["int8"],
        prompts=[
            {"kind": "text", "prompt": "Say hello in one word."},
            {
                "kind": "json",
                "prompt": 'Return ONLY {"ok": true} as JSON. No prose.',
            },
        ],
        max_tokens=16,
        mem_tokens=32,
        kv_group_size=64,
        advisory=True,  # advisory -> exit 0 regardless
        run_niah=False,
        json_out=out,
    )
    assert rc == 0  # advisory always 0

    import json

    reports = json.loads(out.read_text())
    assert len(reports) == 1
    rep = reports[0]
    assert rep["schema"] == 1
    assert rep["candidate_dtype"] == "int8"
    # Baseline dtype is READ from the real cache, not hardcoded — dense mlx
    # checkpoints are bfloat16 or float16, never a quantized "int*" label.
    assert rep["baseline_dtype"] in {"bfloat16", "float16"}
    assert 0.0 <= rep["agreement"]["rate"] <= 1.0
    # int8 KV must actually save memory vs bf16 on a dense model.
    assert rep["memory"]["reduction_ratio"] > 1.0
    assert rep["memory"]["candidate_bytes"] < rep["memory"]["baseline_bytes"]
    assert rep["overall"] in {PASS, FAIL, NA}
    # chip tier must be recorded and internally consistent.
    chip = rep["chip"]
    assert "is_apple_silicon" in chip
    if chip["is_apple_silicon"]:
        assert chip["is_m3_or_newer"] == (chip["generation"] >= 3)
