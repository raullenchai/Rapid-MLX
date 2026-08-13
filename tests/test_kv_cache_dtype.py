# SPDX-License-Identifier: Apache-2.0
"""Unit tests for :mod:`vllm_mlx.kv_cache_dtype` (R15 task #300).

The resolver is pure (no I/O, no model loading) — these tests pin the
matrix of {dtype × safelist hit × reasoning override} that the CLI
banner and Prometheus gauge depend on.
"""

from __future__ import annotations

import pytest

from vllm_mlx.kv_cache_dtype import (
    DEFAULT_KV_CACHE_DTYPE,
    KV_CACHE_DTYPES,
    REASONING_KV_CACHE_DTYPE,
    dtype_to_quantization_bits,
    resolve_kv_cache_dtype,
)

# ---------------------------------------------------------------------------
# Basic invariants
# ---------------------------------------------------------------------------


def test_default_is_bf16():
    """Locks the #1853 contract — bf16 is the default, quant is opt-in.

    The R15 #300 int4 default assumed a bandwidth win, but the live
    serve path (QuantizedBatchKVCache, dequant-on-read) materializes
    full-precision K/V every decode step: measured 16k decode was
    int4 98.2 tok/s vs bf16 134.6 on qwen3.5-4b. Anyone flipping the
    default back to a quantized dtype must first land a fused
    quantized-attention read path and re-run the long-context bench.
    """
    assert DEFAULT_KV_CACHE_DTYPE == "bf16"
    assert "bf16" in KV_CACHE_DTYPES


def test_reasoning_default_is_int8():
    """Reasoning profile pins to int8 (sub-4-bit drops -20pt on AIME)."""
    assert REASONING_KV_CACHE_DTYPE == "int8"


def test_dtype_to_quantization_bits_matrix():
    """Each dtype maps onto a single, well-defined SchedulerConfig pair."""
    assert dtype_to_quantization_bits("bf16") == (False, 8)
    assert dtype_to_quantization_bits("int8") == (True, 8)
    assert dtype_to_quantization_bits("int4") == (True, 4)


def test_dtype_to_quantization_bits_rejects_unknown():
    with pytest.raises(ValueError):
        dtype_to_quantization_bits("fp4")


def test_resolve_rejects_unknown_dtype():
    with pytest.raises(ValueError):
        resolve_kv_cache_dtype("fp4")


# ---------------------------------------------------------------------------
# Happy path — plain Qwen-style transformer, no safelist hit
# ---------------------------------------------------------------------------


def test_default_qwen_keeps_int4():
    """A vanilla Qwen3 config should NOT be downgraded."""
    decision = resolve_kv_cache_dtype(
        "int4",
        model_name="qwen3.5-9b-4bit",
        hf_path="mlx-community/Qwen3.5-9B-MLX-4bit",
        hf_config={"model_type": "qwen3", "hidden_size": 4096},
    )
    assert decision.dtype == "int4"
    assert decision.downgraded is False
    assert "qwen3.5-9b-4bit" in decision.reason


def test_bf16_request_is_passed_through_even_on_safelist():
    """An explicit bf16 request must never be silently upgraded."""
    decision = resolve_kv_cache_dtype(
        "bf16",
        model_name="gemma-3-27b-4bit",
        hf_config={"model_type": "gemma3", "sliding_window": 4096},
    )
    assert decision.dtype == "bf16"
    assert decision.downgraded is False


def test_int8_request_keeps_int8_on_safe_model():
    decision = resolve_kv_cache_dtype("int8", model_name="qwen3-1.7b-4bit")
    assert decision.dtype == "int8"
    assert decision.downgraded is False


# ---------------------------------------------------------------------------
# Sliding-window safelist (Gemma 3, GPT-OSS)
# ---------------------------------------------------------------------------


def test_gemma3_config_field_triggers_downgrade():
    """sliding_window in hf_config flips int4 → bf16."""
    decision = resolve_kv_cache_dtype(
        "int4",
        model_name="gemma-3-27b-4bit",
        hf_config={"model_type": "gemma3", "sliding_window": 4096},
    )
    assert decision.dtype == "bf16"
    assert decision.downgraded is True
    assert "sliding-window" in decision.reason.lower()


def test_gemma3_substring_triggers_downgrade():
    """When config is unavailable, alias substring still catches Gemma 3."""
    decision = resolve_kv_cache_dtype(
        "int4",
        model_name="gemma3-12b-4bit",
        hf_path="mlx-community/gemma-3-12b-it-4bit",
    )
    assert decision.dtype == "bf16"
    assert decision.downgraded is True


def test_gpt_oss_substring_triggers_downgrade():
    decision = resolve_kv_cache_dtype(
        "int4",
        model_name="gpt-oss-20b-mxfp4-q8",
        hf_path="mlx-community/gpt-oss-20b-MXFP4-Q8",
    )
    assert decision.dtype == "bf16"
    assert decision.downgraded is True


def test_gpt_oss_model_type_triggers_downgrade():
    """model_type=gpt_oss in hf_config triggers safelist."""
    decision = resolve_kv_cache_dtype(
        "int4",
        model_name="some-arbitrary-name",
        hf_config={"model_type": "gpt_oss"},
    )
    assert decision.dtype == "bf16"
    assert decision.downgraded is True


def test_alias_metadata_sliding_window_wins_over_config():
    """An alias-level ``sliding_window: true`` triggers downgrade even
    if hf_config doesn't say so (carve-out for upcoming releases)."""
    decision = resolve_kv_cache_dtype(
        "int4",
        model_name="some-future-alias",
        hf_config={"model_type": "mystery", "hidden_size": 4096},
        alias_metadata={"sliding_window": True},
    )
    assert decision.dtype == "bf16"
    assert decision.downgraded is True


# ---------------------------------------------------------------------------
# MLA safelist (DeepSeek V3+, Kimi K2.5)
# ---------------------------------------------------------------------------


def test_deepseek_v3_config_triggers_downgrade():
    """q_lora_rank + kv_lora_rank pair in hf_config triggers MLA safelist."""
    decision = resolve_kv_cache_dtype(
        "int4",
        model_name="deepseek-v3-mock",
        hf_config={
            "model_type": "deepseek_v3",
            "q_lora_rank": 1536,
            "kv_lora_rank": 512,
        },
    )
    assert decision.dtype == "bf16"
    assert decision.downgraded is True
    assert "mla" in decision.reason.lower()


def test_bailing_hybrid_model_type_triggers_downgrade():
    """Ling 3.0 (bailing_hybrid) MLA layers share DeepSeek's compressed-K
    layout; model_type alone must trigger the safelist — the checkpoint
    name ("ling...") matches no _MLA_PATTERNS entry."""
    decision = resolve_kv_cache_dtype(
        "int4",
        model_name="/models/ling_v3_tiny_fp8",
        hf_config={
            "model_type": "bailing_hybrid",
            "q_lora_rank": 256,
            "kv_lora_rank": 512,
        },
    )
    assert decision.dtype == "bf16"
    assert decision.downgraded is True
    assert "mla" in decision.reason.lower()


def test_deepseek_v4_substring_triggers_downgrade():
    decision = resolve_kv_cache_dtype(
        "int4",
        model_name="deepseek-v4-flash-4bit",
        hf_path="mlx-community/DeepSeek-V4-Flash-4bit",
    )
    assert decision.dtype == "bf16"
    assert decision.downgraded is True


def test_kimi_k2_substring_triggers_downgrade():
    decision = resolve_kv_cache_dtype(
        "int4",
        model_name="kimi-k2-instruct",
        hf_path="mlx-community/Kimi-K2-Instruct",
    )
    assert decision.dtype == "bf16"
    assert decision.downgraded is True


def test_mla_only_q_lora_rank_does_not_trigger():
    """A LoRA-adapted Qwen with q_lora_rank but no kv_lora_rank is NOT
    MLA — the safelist must not over-fire here or every LoRA model gets
    silently quality-tested."""
    decision = resolve_kv_cache_dtype(
        "int4",
        model_name="qwen3-lora-finetune",
        hf_config={"model_type": "qwen3", "q_lora_rank": 64},
    )
    assert decision.dtype == "int4"
    assert decision.downgraded is False


def test_alias_metadata_is_mla_wins():
    decision = resolve_kv_cache_dtype(
        "int4",
        model_name="some-future-alias",
        alias_metadata={"is_mla": True},
    )
    assert decision.dtype == "bf16"
    assert decision.downgraded is True


def test_mla_rank_pair_without_family_signal_does_not_trigger():
    """codex r1 BLOCKING #3: a non-DeepSeek/Kimi model that happens to
    ship both ``q_lora_rank`` and ``kv_lora_rank`` (e.g. a LoRA-quant
    toolkit reusing the field names, or a future architecture) must
    NOT be force-downgraded to bf16 — that would be a silent perf
    regression on an architecture nobody benched as MLA.

    The fix requires a family signal (alias metadata, canonical
    ``model_type`` in ``_MLA_MODEL_TYPES``, or a name-pattern hit)
    in ADDITION to the rank pair.
    """
    decision = resolve_kv_cache_dtype(
        "int4",
        model_name="some-novel-arch-9b",
        hf_path="custom/some-novel-arch-9b",
        hf_config={
            "model_type": "novel_arch",
            "q_lora_rank": 1536,
            "kv_lora_rank": 512,
        },
    )
    assert decision.dtype == "int4"
    assert decision.downgraded is False


def test_mla_unknown_deepseek_v5_name_fails_closed():
    """codex r2 NIT (was BLOCKING #3 follow-up): when a future
    DeepSeek release ships a name/``model_type`` we don't recognise
    yet (e.g. ``deepseek_v5``), the safelist must fail CLOSED —
    leave ``int4`` in place rather than guess at MLA based on rank
    pairs alone. The previous test name suggested the rank pair
    *triggered* the downgrade, which is the opposite of what the
    body asserts; renamed to make a future failure self-explain.
    """
    decision = resolve_kv_cache_dtype(
        "int4",
        model_name="deepseek-v5-test",
        hf_path="deepseek-ai/DeepSeek-V5",
        hf_config={
            "model_type": "deepseek_v5",  # not in our pinned set yet
            "q_lora_rank": 1536,
            "kv_lora_rank": 512,
        },
    )
    # ``deepseek-v5-test`` is not a documented pattern but the existing
    # ``deepseek-v3`` / ``deepseek-v4`` substrings won't match. So this
    # test really verifies that BOTH the substring search and the
    # model_type check fail closed — i.e., we should NOT downgrade.
    assert decision.dtype == "int4"
    assert decision.downgraded is False


# ---------------------------------------------------------------------------
# Reasoning profile
# ---------------------------------------------------------------------------


def test_reasoning_pins_to_int8_from_int4():
    decision = resolve_kv_cache_dtype(
        "int4",
        reasoning=True,
        model_name="qwen3.5-thinking-9b",
    )
    assert decision.dtype == "int8"
    assert decision.downgraded is True
    assert "reasoning" in decision.reason.lower()


def test_reasoning_pins_to_int8_from_bf16():
    """Even bf16 → int8 under --reasoning (the profile is intentional)."""
    decision = resolve_kv_cache_dtype(
        "bf16",
        reasoning=True,
        model_name="qwen3.5-thinking-9b",
    )
    assert decision.dtype == "int8"
    assert decision.downgraded is True


def test_reasoning_with_int8_is_not_downgraded():
    decision = resolve_kv_cache_dtype(
        "int8",
        reasoning=True,
        model_name="qwen3.5-thinking-9b",
    )
    assert decision.dtype == "int8"
    assert decision.downgraded is False


def test_reasoning_wins_over_safelist():
    """An MLA model with --reasoning still gets int8, not bf16.

    Rationale: --reasoning is an explicit operator opt-in for a known
    workload class; the safelist's job is to protect default-flag users
    from silent quality regressions, not to second-guess the operator
    who's already chosen a profile.
    """
    decision = resolve_kv_cache_dtype(
        "int4",
        reasoning=True,
        model_name="deepseek-v3",
        hf_config={
            "model_type": "deepseek_v3",
            "q_lora_rank": 1536,
            "kv_lora_rank": 512,
        },
    )
    assert decision.dtype == "int8"
    # Downgrade flag is True because requested != effective.
    assert decision.downgraded is True


# ---------------------------------------------------------------------------
# Reason strings are stable enough for the operator banner
# ---------------------------------------------------------------------------


def test_bf16_reason_is_neutral_and_not_a_downgrade():
    decision = resolve_kv_cache_dtype("bf16", model_name="qwen3-4b-4bit")
    # argparse cannot distinguish an explicit bf16 from the parser
    # default, so the reason must not claim either; what matters is
    # that bf16 resolution is never reported as a downgrade.
    assert "bf16" in decision.reason
    assert "default" not in decision.reason
    assert decision.downgraded is False


def test_int4_override_reason_cites_decode_cost():
    decision = resolve_kv_cache_dtype("int4", model_name="qwen3-4b-4bit")
    # Opting into quantized KV must leave a breadcrumb in the log that
    # the operator accepted the O(context) dequant-on-read decode cost.
    assert "#1853" in decision.reason
    assert decision.downgraded is False


def test_safelist_reason_mentions_sliding_window_for_gemma3():
    decision = resolve_kv_cache_dtype(
        "int4",
        model_name="gemma3-27b-4bit",
        hf_config={"model_type": "gemma3", "sliding_window": 4096},
    )
    assert "sliding-window" in decision.reason.lower()
    assert "bf16" in decision.reason
