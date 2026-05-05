# SPDX-License-Identifier: Apache-2.0
"""Sampling primitives for native MTP speculative decoding.

The acceptance rule follows Leviathan/Chen speculative decoding: a draft token
sampled from q is accepted with probability min(1, p / q), where p is the
target distribution for the same position. On rejection, the replacement token
is sampled from the positive residual distribution (p - q)+.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx


@dataclass(frozen=True)
class MTPDistributionParams:
    """Sampling distribution parameters shared by target and draft paths."""

    temperature: float = 0.7
    top_p: float = 0.9
    min_p: float = 0.0
    top_k: int = 0


def _apply_filters(logprobs: mx.array, params: MTPDistributionParams) -> mx.array:
    """Apply mlx-lm sampling filters without drawing a token."""
    try:
        from mlx_lm.sample_utils import apply_min_p, apply_top_k, apply_top_p
    except ImportError:
        return logprobs

    if 0.0 < params.top_p < 1.0:
        logprobs = apply_top_p(logprobs, params.top_p)
    if params.min_p != 0.0:
        logprobs = apply_min_p(logprobs, params.min_p)
    if params.top_k > 0:
        logprobs = apply_top_k(logprobs, params.top_k)
    return logprobs


def distribution_logprobs(
    logits: mx.array,
    params: MTPDistributionParams,
) -> mx.array:
    """Return normalized log probabilities for the configured sampler."""
    base = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    filtered = _apply_filters(base, params)

    if params.temperature == 0:
        argmax = mx.argmax(filtered, axis=-1, keepdims=True)
        one_hot = mx.zeros_like(filtered)
        one_hot = mx.put_along_axis(one_hot, argmax, mx.array(1.0), axis=-1)
        return mx.log(mx.where(one_hot > 0, one_hot, 0.0))

    scaled = filtered * (1.0 / params.temperature)
    return scaled - mx.logsumexp(scaled, axis=-1, keepdims=True)


def sample_from_logprobs(logprobs: mx.array) -> mx.array:
    """Sample token IDs from already-normalized log probabilities."""
    return mx.random.categorical(logprobs)


def _token_logprobs(logprobs: mx.array, tokens: mx.array) -> mx.array:
    return mx.take_along_axis(logprobs, tokens[:, None], axis=-1)[:, 0]


def acceptance_mask(
    target_logprobs: mx.array,
    draft_logprobs: mx.array,
    draft_tokens: mx.array,
) -> mx.array:
    """Return boolean accept decisions for each batch row."""
    target_lp = _token_logprobs(target_logprobs, draft_tokens)
    draft_lp = _token_logprobs(draft_logprobs, draft_tokens)
    accept_prob = mx.minimum(mx.exp(target_lp - draft_lp), mx.array(1.0))
    rolls = mx.random.uniform(shape=accept_prob.shape)
    return rolls <= accept_prob


def residual_logprobs(
    target_logprobs: mx.array,
    draft_logprobs: mx.array,
) -> mx.array:
    """Return normalized log probabilities for the residual (p - q)+."""
    residual = mx.maximum(mx.exp(target_logprobs) - mx.exp(draft_logprobs), 0.0)
    total = mx.sum(residual, axis=-1, keepdims=True)
    fallback = mx.exp(target_logprobs)
    residual = mx.where(total > 0, residual / total, fallback)
    return mx.log(mx.maximum(residual, 1e-45))
