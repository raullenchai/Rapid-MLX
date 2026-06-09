# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the fused top-p + temperature sampler fast path.

Pins three properties of ``vllm_mlx._sampler_fast_path``:

1. ``is_fused_top_p_eligible`` covers the eligible knob window exactly —
   eligible iff ``temperature > 0`` AND ``min_p == 0`` AND
   ``0 < top_p < 1``. ``top_k`` is optional (additional mask layered on
   top of the active nucleus cut). Every other combination — greedy,
   ``min_p > 0``, top-k-only (``top_p`` disabled) — returns False.
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

    def test_top_k_only_rejected(self):
        # Codex round-2 BLOCKER #2 fix: top-k-only configurations fall
        # through to mlx-lm's chain because ``apply_top_k`` uses a cheaper
        # ``mx.partition`` primitive than our full-vocab ``argsort``. The
        # fast path's win comes from collapsing apply_top_p + categorical;
        # without nucleus we have nothing to collapse.
        assert not is_fused_top_p_eligible(
            temperature=0.7, top_p=0.0, min_p=0.0, top_k=20
        )
        assert not is_fused_top_p_eligible(
            temperature=0.7, top_p=1.0, min_p=0.0, top_k=20
        )

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
        # ``top_p`` must be strictly in (0, 1). Outside that window the
        # fused path can't beat mlx-lm (top_p == 1 short-circuits;
        # top_p == 0 with top_k routes through partition).
        with pytest.raises(ValueError, match="top_p"):
            make_fused_top_p_temp_sampler(0.7, 1.0, top_k=0)
        with pytest.raises(ValueError, match="top_p"):
            make_fused_top_p_temp_sampler(0.7, 0.0, top_k=0)

    def test_top_k_only_rejected_at_construction(self):
        # top_k > 0 with top_p disabled must refuse — eligibility predicate
        # already excludes this path so the constructor is the second
        # safety net (defense in depth against caller bugs).
        with pytest.raises(ValueError, match="top_p"):
            make_fused_top_p_temp_sampler(0.7, 0.0, top_k=20)
        with pytest.raises(ValueError, match="top_p"):
            make_fused_top_p_temp_sampler(0.7, 1.0, top_k=20)

    def test_tiny_top_p_does_not_produce_all_inf(self):
        # Codex round-2 BLOCKER #1 — sub-fp32-epsilon top_p makes
        # ``1.0 - top_p`` round to 1.0 in the float32 comparison and
        # ``cumulative > 1 - top_p`` evaluates all-false. Without the
        # top-1 OR guarantee the sampler would feed all -inf logits to
        # ``mx.random.categorical`` and either crash or produce garbage.
        # We assert it returns a real token id matching the argmax.
        vocab = 1024
        logits = mx.random.normal(shape=(vocab,)) * 3.0
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        expected_argmax = int(mx.argmax(logprobs, axis=-1))

        sampler = make_fused_top_p_temp_sampler(0.7, 1e-9)
        for _ in range(5):
            tok = int(sampler(logprobs))
            assert tok == expected_argmax, (
                f"tiny top_p must keep at least the argmax; got token "
                f"{tok}, expected {expected_argmax}"
            )


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

    def test_per_token_frequency_matches_mlx_lm(self):
        """Codex round-3 NIT #4: check the per-token frequency for every
        kept token, not just the argmax marginal. A sampler that gives
        the wrong relative probabilities to the rest of the kept set
        would pass the prior argmax-only assertion but fail this one.

        Strategy: use a sharp 32-vocab distribution where the kept set is
        4-5 tokens with probabilities differing by 3-8x. After 4000 draws
        each kept token's empirical frequency has a 3-sigma binomial
        band of ~0.024 at p~0.3, so a 0.05 tolerance comfortably absorbs
        MLX RNG noise but still flags any kept token whose relative
        weight is off by more than ~50%.
        """
        vocab = 32
        logits = mx.full((vocab,), -8.0)
        logits[0] = 2.0  # ~0.40
        logits[5] = 1.5  # ~0.24
        logits[12] = 1.0  # ~0.15
        logits[20] = 0.5  # ~0.09
        logits[27] = 0.0  # ~0.05
        logits[31] = -0.5  # ~0.03
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)

        fused = make_fused_top_p_temp_sampler(0.7, 0.95)
        mlx_chain = make_sampler(temp=0.7, top_p=0.95)

        n = 4000
        mx.random.seed(0)
        fused_counts = self._empirical_distribution(fused, logprobs, n)
        mx.random.seed(0)
        chain_counts = self._empirical_distribution(mlx_chain, logprobs, n)

        kept = set(fused_counts.keys()) | set(chain_counts.keys())
        assert kept, "no tokens sampled — check the test setup"
        # Per-token freq must match within 5% (3-sigma slack for n=4000).
        for tok in kept:
            fused_freq = fused_counts.get(tok, 0) / n
            chain_freq = chain_counts.get(tok, 0) / n
            assert abs(fused_freq - chain_freq) < 0.05, (
                f"per-token freq diverges for token {tok}: "
                f"fused={fused_freq:.3f}, chain={chain_freq:.3f}"
            )

    def test_top_token_frequency_matches(self):
        """The most-likely token (argmax of the original distribution)
        should be sampled at the same rate within sampling noise.
        Complements ``test_per_token_frequency_matches_mlx_lm`` by
        running on the larger noisy 4096-vocab distribution so we catch
        any Gumbel-noise weighting bug that only shows up at scale.
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

    def test_top_k_tie_at_boundary_keeps_k_tokens(self):
        """Tied logits at the ``top_k`` cutoff: mlx-lm's ``apply_top_k``
        uses ``mx.argpartition`` (also position-based, unstable on ties),
        so both paths keep exactly ``top_k`` tokens. They may arbitrarily
        disagree on WHICH tied token gets picked, but the kept-set SIZE
        is invariant and the distribution shape is unaffected.

        Codex round 4 raised a BLOCKER claiming mlx-lm was partition+
        threshold ("keep all ties"); inspecting ``apply_top_k`` in
        mlx-lm 0.21 falsified that — it is position-based via
        ``argpartition``. This test pins the verified contract so future
        codex rounds don't re-raise the same false BLOCKER.
        """
        vocab = 16
        # Three strictly larger, then a tied pair at the boundary,
        # then strict losers. top_k=4 cutoff lands on the tie.
        logits = mx.full((vocab,), -8.0)
        logits[0] = 3.0  # strictly largest
        logits[3] = 2.5  # strictly second
        logits[7] = 2.0  # strictly third
        logits[10] = 1.5  # tied with token 13 — kth value
        logits[13] = 1.5  # tied with token 10
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)

        fused = make_fused_top_p_temp_sampler(0.7, 0.99, top_k=4)
        mlx_chain = make_sampler(temp=0.7, top_p=0.99, top_k=4)

        mx.random.seed(0)
        fused_counts = self._empirical_distribution(fused, logprobs, n=2000)
        mx.random.seed(0)
        chain_counts = self._empirical_distribution(mlx_chain, logprobs, n=2000)

        # Both paths must keep exactly 4 tokens. WHICH tied token wins
        # may differ across paths (unstable tie-break is allowed) but
        # the size is invariant.
        assert len(fused_counts) == 4, (
            f"fused kept {len(fused_counts)} tokens, expected 4; "
            f"got {set(fused_counts.keys())}"
        )
        assert len(chain_counts) == 4, (
            f"mlx-lm chain kept {len(chain_counts)} tokens, expected 4; "
            f"got {set(chain_counts.keys())}"
        )
        # The 3 strictly larger tokens are always kept by both paths.
        assert {0, 3, 7}.issubset(set(fused_counts.keys())), (
            f"strict winners 0, 3, 7 must be in fused kept set, "
            f"got {set(fused_counts.keys())}"
        )
        assert {0, 3, 7}.issubset(set(chain_counts.keys())), (
            f"strict winners 0, 3, 7 must be in mlx-lm kept set, "
            f"got {set(chain_counts.keys())}"
        )
        # Exactly ONE of the tied tokens (10 or 13) is in each kept set.
        tied_in_fused = {10, 13} & set(fused_counts.keys())
        tied_in_chain = {10, 13} & set(chain_counts.keys())
        assert len(tied_in_fused) == 1, (
            f"exactly one tied token expected in fused kept set, got {tied_in_fused}"
        )
        assert len(tied_in_chain) == 1, (
            f"exactly one tied token expected in mlx-lm kept set, got {tied_in_chain}"
        )

    def test_top_p_plus_top_k_kept_set_matches_mlx_lm(self):
        """When both top_p and top_k are active the fused sampler must
        produce the SAME kept set as mlx-lm's ``apply_top_p`` ->
        ``apply_top_k`` -> categorical chain. Codex round-2 NIT #4: the
        prior tests only pinned shape for combined configs; this asserts
        the intersection (top_p ∩ top_k) actually matches.
        """
        vocab = 32
        logits = mx.full((vocab,), -8.0)
        # Sharp distribution: dominant tokens get most of the mass.
        logits[0] = 2.0
        logits[5] = 1.5
        logits[12] = 1.0
        logits[20] = 0.5
        logits[27] = 0.0
        logits[31] = -0.5
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)

        # top_k=3 forces only the top-3 (tokens 0, 5, 12) to be kept,
        # even though top_p=0.95 would otherwise admit 4-5 tokens.
        fused = make_fused_top_p_temp_sampler(0.7, 0.95, top_k=3)
        mlx_chain = make_sampler(temp=0.7, top_p=0.95, top_k=3)

        mx.random.seed(0)
        fused_counts = self._empirical_distribution(fused, logprobs, n=1000)
        mx.random.seed(0)
        chain_counts = self._empirical_distribution(mlx_chain, logprobs, n=1000)

        assert set(fused_counts.keys()) == set(chain_counts.keys()), (
            f"top_p+top_k kept-set mismatch: fused={set(fused_counts.keys())}, "
            f"chain={set(chain_counts.keys())}"
        )
        # Sanity-check the intersection actually constrained the set:
        # top_k=3 means at most 3 tokens are sampleable.
        assert len(fused_counts) <= 3, (
            f"top_k=3 should cap kept set at 3, got {len(fused_counts)}"
        )
