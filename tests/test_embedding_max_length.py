# SPDX-License-Identifier: Apache-2.0
"""Unit tests for configurable embedding max input length (issue #1381).

These exercise the pure resolution / overflow logic with a mocked model and
tokenizer, so they need no ``mlx_embeddings`` model download and never touch
the GPU. The acceptance matrix from the issue is covered: Qwen3-Embedding at
32K, BERT at 512, explicit lower overrides, over-model values, both overflow
policies, and mixed-length batches.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mlx")
pytestmark = pytest.mark.requires_mlx


import logging

from vllm_mlx.embedding import (
    _FALLBACK_MAX_LENGTH,
    _MODEL_MAX_SENTINEL_THRESHOLD,
    EmbeddingEngine,
    EmbeddingInputTooLongError,
    normalize_max_length_setting,
)

# HuggingFace's "unset" sentinel for tokenizer.model_max_length.
_HF_SENTINEL = 1_000_000_000_000_000_019_884_624_838_656


class _FakeConfig:
    def __init__(self, **attrs):
        for key, value in attrs.items():
            setattr(self, key, value)


class _FakeModel:
    def __init__(self, config=None):
        if config is not None:
            self.config = config


class _FakeTokenizer:
    """Minimal tokenizer stand-in: one token per whitespace-delimited word."""

    def __init__(self, model_max_length=None):
        if model_max_length is not None:
            self.model_max_length = model_max_length

    def encode(self, text):
        return text.split()


def make_engine(
    *,
    model_max=None,
    tok_max=None,
    max_length="auto",
    overflow_policy="truncate",
    resolve=True,
):
    eng = EmbeddingEngine(
        "fake-model", max_length=max_length, overflow_policy=overflow_policy
    )
    config = _FakeConfig(
        **({"max_position_embeddings": model_max} if model_max is not None else {})
    )
    eng._model = _FakeModel(config=config)
    eng._tokenizer = _FakeTokenizer(model_max_length=tok_max)
    if resolve:
        eng._resolve_effective_max_length()
    return eng


# --------------------------------------------------------------------------
# normalize_max_length_setting
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        ("auto", "auto"),
        ("AUTO", "auto"),
        ("  auto ", "auto"),
        ("1024", 1024),
        (4096, 4096),
        (1, 1),
    ],
)
def test_normalize_valid(value, expected):
    assert normalize_max_length_setting(value) == expected


@pytest.mark.parametrize("value", ["0", "-5", "foo", "", 0, -1, True, False, 3.5])
def test_normalize_invalid(value):
    with pytest.raises(ValueError):
        normalize_max_length_setting(value)


def test_bad_overflow_policy_rejected():
    with pytest.raises(ValueError):
        EmbeddingEngine("m", overflow_policy="nope")


def test_bad_max_length_rejected_at_construction():
    with pytest.raises(ValueError):
        EmbeddingEngine("m", max_length="not-a-number")


# --------------------------------------------------------------------------
# _discover_model_max_length + effective length resolution
# --------------------------------------------------------------------------


def test_discover_from_model_config():
    eng = make_engine(model_max=32768, resolve=False)
    assert eng._discover_model_max_length() == 32768


def test_discover_from_tokenizer_when_no_config():
    eng = make_engine(model_max=None, tok_max=512, resolve=False)
    assert eng._discover_model_max_length() == 512


def test_discover_ignores_hf_sentinel_on_tokenizer():
    eng = make_engine(model_max=None, tok_max=_HF_SENTINEL, resolve=False)
    assert eng._discover_model_max_length() is None
    assert _HF_SENTINEL >= _MODEL_MAX_SENTINEL_THRESHOLD


def test_discover_ignores_sentinel_on_config_falls_back_to_tokenizer():
    eng = make_engine(model_max=_HF_SENTINEL, tok_max=512, resolve=False)
    assert eng._discover_model_max_length() == 512


def test_discover_none_when_nothing_declared():
    eng = make_engine(model_max=None, tok_max=None, resolve=False)
    assert eng._discover_model_max_length() is None


def test_auto_uses_model_max():
    # Qwen3-Embedding-4B: 32K context.
    eng = make_engine(model_max=32768, max_length="auto")
    assert eng.effective_max_length == 32768


def test_auto_falls_back_when_unknown():
    eng = make_engine(model_max=None, tok_max=None, max_length="auto")
    assert eng.effective_max_length == _FALLBACK_MAX_LENGTH


def test_bert_at_512():
    eng = make_engine(model_max=512, max_length="auto")
    assert eng.effective_max_length == 512


def test_explicit_override_below_model_max():
    eng = make_engine(model_max=32768, max_length=1024)
    assert eng.effective_max_length == 1024


def test_explicit_override_above_model_max_is_clamped(caplog):
    with caplog.at_level(logging.WARNING):
        eng = make_engine(model_max=32768, max_length=99999)
    assert eng.effective_max_length == 32768
    assert "clamping" in caplog.text.lower()


def test_explicit_override_when_model_max_unknown():
    eng = make_engine(model_max=None, tok_max=None, max_length=4096)
    assert eng.effective_max_length == 4096


# --------------------------------------------------------------------------
# overflow policy
# --------------------------------------------------------------------------


def test_truncate_policy_warns_and_counts(caplog):
    eng = make_engine(model_max=512, overflow_policy="truncate")
    with caplog.at_level(logging.WARNING):
        eng._enforce_overflow([600, 100])  # one over the 512 limit
    assert eng.num_truncations == 1
    assert "truncat" in caplog.text.lower()


def test_truncate_policy_counts_each_over_input():
    eng = make_engine(model_max=512, overflow_policy="truncate")
    eng._enforce_overflow([600, 700, 100])  # two over
    assert eng.num_truncations == 2


def test_truncate_policy_no_overflow_is_silent(caplog):
    eng = make_engine(model_max=512, overflow_policy="truncate")
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        eng._enforce_overflow([100, 200, 511, 512])
    assert eng.num_truncations == 0
    assert caplog.text == ""


def test_error_policy_raises_with_counts():
    eng = make_engine(model_max=512, overflow_policy="error")
    with pytest.raises(EmbeddingInputTooLongError) as exc_info:
        eng._enforce_overflow([100, 600, 900])
    err = exc_info.value
    assert err.observed_tokens == 600  # first over-limit input
    assert err.allowed_tokens == 512
    assert err.index == 1
    assert eng.num_truncations == 0  # error policy never truncates


def test_error_policy_passes_when_within_limit():
    eng = make_engine(model_max=512, overflow_policy="error")
    eng._enforce_overflow([100, 512])  # 512 == limit, not over
    assert eng.num_truncations == 0


# --------------------------------------------------------------------------
# count_tokens is capped at the effective limit (usage no longer over-reports)
# --------------------------------------------------------------------------


def test_count_tokens_capped_at_effective_limit():
    eng = make_engine(model_max=512, max_length="auto")
    long_text = " ".join(["tok"] * 600)  # 600 "tokens"
    # Capped at 512, not the pre-truncation 600.
    assert eng.count_tokens([long_text]) == 512


def test_count_tokens_sums_capped_per_input():
    eng = make_engine(model_max=512, max_length="auto")
    texts = [" ".join(["a"] * 600), "short one here"]
    assert eng.count_tokens(texts) == 512 + 3


def test_count_tokens_uncapped_within_limit():
    eng = make_engine(model_max=32768, max_length="auto")
    assert eng.count_tokens(["one two three four"]) == 4


# --------------------------------------------------------------------------
# acceptance scenarios straight from the issue
# --------------------------------------------------------------------------


def test_qwen3_embedding_32k_long_input_not_truncated():
    eng = make_engine(model_max=32768, max_length="auto", overflow_policy="truncate")
    eng._enforce_overflow([1000, 4096, 8192])  # all < 32768
    assert eng.num_truncations == 0


def test_mixed_length_batch_only_over_limit_counted():
    eng = make_engine(model_max=512, max_length="auto", overflow_policy="truncate")
    eng._enforce_overflow([100, 600])  # one within, one over
    assert eng.num_truncations == 1
