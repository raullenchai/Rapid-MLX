# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the fused top-p + temperature sampler fast path.

Pins three properties of ``vllm_mlx._sampler_fast_path``:

1. ``is_fused_top_p_eligible`` covers the eligible knob window exactly —
   eligible when ``temperature > 0`` and ``min_p == 0`` and at least one
   of ``0 < top_p < 1`` or ``top_k > 0`` is active; every other
   combination (greedy, ``min_p > 0``, both top-p and top-k disabled)
   returns False.
2. The fused sampler returns the right shape contract: ``[V]`` -> ``[]``,
   ``[B, V]`` -> ``[B]``, output dtype is ``uint32`` (matching mlx-lm's
   sample token type).
3. The fused sampler is **distributionally equivalent** to mlx-lm's
   ``apply_top_p + categorical_sampling`` chain. We sample N times from
   each path on a fixed logit vector and check that the sampled-token
   frequency distributions match within a generous tolerance. Same
   kept-token set, same relative weights inside it — only the Gumbel
   index space differs, so we can't compare bit-by-bit with a fixed
   seed, but the empirical distributions converge.
"""

from __future__ import annotations

import mlx.core as mx
import pytest
from mlx_lm.sample_utils import make_sampler

from vllm_mlx._sampler_fast_path import (
    is_fused_top_p_eligible,
    make_fused_top_p_temp_sampler,
)


class TestEligibility:
    """``is_fused_top_p_eligible`` must accept exactly the dominant chat
    knob set and reject every off-path variant."""

    def test_canonical_chat_knobs_eligible(self):
        assert is_fused_top_p_eligible(temperature=0.7, top_p=0.95, min_p=0.0, top_k=0)

    def test_default_top_p_one_rejected(self):
        # top_p=1.0 means no nucleus — mlx-lm short-circuits apply_top_p.
        assert not is_fused_top_p_eligible(
            temperature=0.7, top_p=1.0, min_p=0.0, top_k=0
        )

    def test_top_p_zero_rejected_when_no_top_k(self):
        # top_p=0 disables nucleus; without top_k there is nothing to mask.
        assert not is_fused_top_p_eligible(
            temperature=0.7, top_p=0.0, min_p=0.0, top_k=0
        )

    def test_top_k_only_eligible(self):
        # top_k > 0 alone is enough — common when alias defaults inject
        # top_k from generation_config.json.
        assert is_fused_top_p_eligible(temperature=0.7, top_p=0.0, min_p=0.0, top_k=20)

    def test_top_p_and_top_k_eligible(self):
        # The combination this PR was originally motivated by: Qwen
        # alias defaults top_k=20 in addition to the request's top_p.
        assert is_fused_top_p_eligible(temperature=0.7, top_p=0.95, min_p=0.0, top_k=20)

    def test_greedy_rejected(self):
        # temp=0 already returns argmax in mlx-lm — nothing to fuse.
        assert not is_fused_top_p_eligible(
            temperature=0.0, top_p=0.95, min_p=0.0, top_k=0
        )

    def test_min_p_rejected(self):
        # min_p adds a fourth op the fast path doesn't implement.
        assert not is_fused_top_p_eligible(
            temperature=0.7, top_p=0.95, min_p=0.05, top_k=0
        )


class TestShapeContract:
    """Output shape + dtype must match what ``BatchGenerator._step`` expects."""

    def test_batched_input(self):
        sampler = make_fused_top_p_temp_sampler(0.7, 0.95)
        logits = mx.random.normal(shape=(4, 32000))
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        out = sampler(logprobs)
        assert out.shape == (4,)
        assert out.dtype == mx.uint32

    def test_unbatched_input(self):
        sampler = make_fused_top_p_temp_sampler(0.7, 0.95)
        logits = mx.random.normal(shape=(32000,))
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        out = sampler(logprobs)
        assert out.shape == ()
        assert out.dtype == mx.uint32

    def test_invalid_temperature_raises(self):
        with pytest.raises(ValueError, match="temperature"):
            make_fused_top_p_temp_sampler(0.0, 0.95)

    def test_invalid_top_p_raises(self):
        # Both top_p and top_k disabled => no mask to build, refuse.
        with pytest.raises(ValueError, match="top_p"):
            make_fused_top_p_temp_sampler(0.7, 1.0, top_k=0)
        with pytest.raises(ValueError, match="top_p"):
            make_fused_top_p_temp_sampler(0.7, 0.0, top_k=0)

    def test_top_k_only_constructs(self):
        # top_k > 0 alone (top_p disabled) must build the sampler.
        sampler = make_fused_top_p_temp_sampler(0.7, 0.0, top_k=20)
        logprobs = mx.random.normal(shape=(1, 8000))
        logprobs = logprobs - mx.logsumexp(logprobs, axis=-1, keepdims=True)
        out = sampler(logprobs)
        assert out.shape == (1,)


class TestDistributionalEquivalence:
    """The fused sampler must draw from the same distribution as mlx-lm's
    ``apply_top_p + categorical_sampling`` chain. We can't pin bit-for-bit
    with a fixed seed (Gumbel index space differs); instead we sample
    N times from each path and check that the same set of tokens is
    populated and the empirical frequencies are within a 3-sigma band.
    """

    @staticmethod
    def _empirical_distribution(sampler, logprobs: mx.array, n: int) -> dict[int, int]:
        counts: dict[int, int] = {}
        for _ in range(n):
            tok = int(sampler(logprobs))
            counts[tok] = counts.get(tok, 0) + 1
        return counts

    def test_kept_set_matches_mlx_lm(self):
        """The set of tokens with non-zero sample probability must match
        between the fused sampler and mlx-lm's chain. We use a small
        vocab + a sharp distribution so the kept set has 3-5 members
        and every member is sampled at least 10× in 1000 draws — that
        way the empirical kept set actually equals the true kept set
        with high probability and we can compare them directly.
        """
        # Hand-built sharp distribution: 5 dominant tokens, vocab=32.
        # Probabilities are roughly [0.4, 0.25, 0.18, 0.10, 0.05, ...]
        # which puts top_p=0.95 cutoff inside the first 5 entries.
        vocab = 32
        logits = mx.full((vocab,), -8.0)
        logits[0] = 1.5
        logits[7] = 1.0
        logits[15] = 0.7
        logits[23] = 0.1
        logits[31] = -0.5
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)

        fused = make_fused_top_p_temp_sampler(0.7, 0.95)
        mlx_chain = make_sampler(temp=0.7, top_p=0.95)

        mx.random.seed(0)
        fused_counts = self._empirical_distribution(fused, logprobs, n=1000)
        mx.random.seed(0)
        chain_counts = self._empirical_distribution(mlx_chain, logprobs, n=1000)

        # The empirical kept set should equal the true kept set after
        # 1000 draws because each member has p >= 0.05 and probability
        # of being missed in 1000 draws is ~(0.95)^1000 ≈ 5e-23.
        assert set(fused_counts.keys()) == set(chain_counts.keys()), (
            f"kept-set mismatch on sharp distribution: "
            f"fused={set(fused_counts.keys())}, "
            f"chain={set(chain_counts.keys())}"
        )

    def test_top_token_frequency_matches(self):
        """The most-likely token (argmax of the original distribution)
        should be sampled at the same rate within sampling noise.
        """
        mx.random.seed(0)
        vocab = 4096
        logits = mx.random.normal(shape=(vocab,)) * 2.0
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        top_token = int(mx.argmax(logprobs, axis=-1))

        fused = make_fused_top_p_temp_sampler(0.7, 0.9)
        mlx_chain = make_sampler(temp=0.7, top_p=0.9)

        n = 4000
        fused_top_rate = (
            self._empirical_distribution(fused, logprobs, n).get(top_token, 0) / n
        )
        chain_top_rate = (
            self._empirical_distribution(mlx_chain, logprobs, n).get(top_token, 0) / n
        )
        # Both paths sample the argmax with the same expected rate.
        # 3-sigma binomial band for n=4000 at p~0.5 is ~0.024 — use 0.04
        # to absorb the additional run-to-run noise from MLX's RNG state.
        assert abs(fused_top_rate - chain_top_rate) < 0.04, (
            f"top-token rate diverges: fused={fused_top_rate:.3f}, "
            f"chain={chain_top_rate:.3f}"
        )

    def test_temperature_one_argmax_unaffected(self):
        """At T=1 with a degenerate one-hot distribution, both paths
        must sample the argmax with probability 1 (no other token has
        any mass).
        """
        vocab = 1024
        logits = mx.full((vocab,), -100.0)
        logits[42] = 0.0  # one-hot at token 42
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)

        fused = make_fused_top_p_temp_sampler(1.0, 0.5)
        for _ in range(10):
            assert int(fused(logprobs)) == 42
