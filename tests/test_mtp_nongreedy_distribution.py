# SPDX-License-Identifier: Apache-2.0
"""Distribution-level tests for non-greedy MTP speculative decoding.

Why this file exists (and why wiring tests are not enough)
---------------------------------------------------------

``tests/test_mtp_cli_wiring.py`` proves that ``temp`` / ``top_p`` reach
``mtp_generate_step``. That says nothing about whether the tokens that
come out are drawn from the right distribution. Speculative decoding is
only "free" if the emitted sequence is distributed EXACTLY as plain
autoregressive sampling from the target — a sampler that is merely
"close" is a silent quality regression that no wiring assertion can see.

The specific failure this file pins: the verify path needs an
INDEPENDENT ``u ~ U(0,1)`` per proposed position, because the accept
test is a Bernoulli(min(1, p/q)) trial at each position. Sharing one
scalar ``u`` across all K positions keeps the accept probability at the
FIRST proposed position correct while making accept events across
positions perfectly rank-correlated. Conditioning on position 1 being
accepted implies ``u < p(d1)/q(d1)``, which biases ``u`` small and so
RAISES the acceptance probability at position 2; the emitted token
there skews toward the draft distribution ``q`` instead of being
resampled from the residual. The defect exists only for K >= 2, since
at K = 1 there is nothing for the single draw to correlate with.

Measured, not assumed: reverting ``generator.py``'s draw to a scalar
makes the K=2 stream miss plain autoregressive sampling by TV 0.029 on
the marginal alone (0.066 at K=3, 0.024 under top-p). So the bias
reaches the per-token marginal too, not only the joint — the correct
statement is that a shared draw skews BOTH, because the stream contains
those biased position-2+ tokens. Both statistics are asserted below;
either one alone would catch this particular defect, and the joint
additionally catches a class of errors that preserve marginals.

So every test here compares a K >= 2 run against a K = 0 run of the
SAME generator over the SAME mocked model. ``max_k=0`` with
``disable_auto_k=True`` parks the depth controller at zero every round,
which walks ``generator.py``'s "Round K=0 ... Plain backbone forward
emits ONE committed token" branch — i.e. plain autoregressive sampling
through the identical filter chain. Using the generator itself as the
reference (rather than a hand-rolled softmax) means the test cannot
pass by re-implementing top-p truncation the same wrong way twice.

Test statistic
--------------

The mocked model returns the SAME logits at every position, so under
correct sampling the emitted stream is i.i.d. Categorical(p_filtered).
We therefore compare two things between the K = 0 and K >= 2 streams:

* the per-token marginal distribution, and
* the lag-1 pair distribution (t_i, t_{i+1}).

Correct sampling makes consecutive tokens independent, so the lag-1
joint must factor into the marginals; the shared draw induces exactly
the within-round dependence described above, and also shifts the
marginal. Asserting both keeps the file sensitive to sampler errors
that move only one of them.
"""

from __future__ import annotations

import math

import pytest

mx = pytest.importorskip("mlx.core")


@pytest.fixture(autouse=True)
def _reset_mtp_module_state():
    """Reset the MTP module-level singletons + ``generation_stream``.

    Same three pieces of cross-test state as the autouse fixture in
    ``test_mtp_spec_decode.py`` / ``test_mtp_lossless.py``; see the
    fixture there for the full rationale on each.
    """
    import sys

    import mlx.core as mx
    import mlx_lm.generate  # noqa: F401 — ensure module exists in sys.modules

    from vllm_mlx.spec_decode.mtp.accept_counter import (
        reset_global_counter_for_tests,
    )
    from vllm_mlx.spec_decode.mtp.cache_patch import _unpatch_for_tests
    from vllm_mlx.spec_decode.mtp.draft_k_controller_v2 import reset_controllers

    def _reset():
        _unpatch_for_tests()
        reset_global_counter_for_tests()
        reset_controllers()
        sys.modules["mlx_lm.generate"].generation_stream = mx.default_stream(
            mx.default_device()
        )

    _reset()
    yield
    _reset()


class _FixedDistributionModel:
    """Mocked Qwen3.5-shaped model with POSITION-INDEPENDENT distributions.

    Satisfies the same contract surface as
    ``tests.test_mtp_spec_decode._MockedQwen35Model`` (``__call__`` with
    ``return_hidden`` / ``n_confirmed``, ``mtp_forward``,
    ``make_mtp_cache``, ``layers``), but scripts DISTRIBUTIONS rather
    than argmax tokens — which is what a sampling test needs and what
    the argmax-scripting mock cannot express.

    The backbone returns ``log(target_probs)`` and the MTP head returns
    ``log(draft_probs)`` at every position, ignoring the input tokens
    entirely. Combined with ``temp=1.0`` (the generator computes
    ``logprobs - logsumexp`` then divides by ``temp``), the target
    sampling distribution is exactly ``target_probs`` and the draft's is
    exactly ``draft_probs``.

    Position-independence is the point: it makes the ground-truth
    autoregressive stream i.i.d., so any dependence between consecutive
    emitted tokens is attributable to the speculative machinery rather
    than to the model.
    """

    def __init__(
        self,
        target_probs: list[float],
        draft_probs: list[float],
        hidden_size: int = 8,
    ):
        assert len(target_probs) == len(draft_probs)
        assert math.isclose(sum(target_probs), 1.0, rel_tol=1e-6)
        assert math.isclose(sum(draft_probs), 1.0, rel_tol=1e-6)
        self.vocab = len(target_probs)
        self.hidden_size = hidden_size
        # log(p) as logits: the generator renormalises with logsumexp,
        # which is a no-op on an already-normalised log-prob vector.
        self._target_logits = mx.log(mx.array(target_probs))
        self._draft_logits = mx.log(mx.array(draft_probs))
        self.layers = []

    @staticmethod
    def _tile(row: mx.array, batch: int, steps: int) -> mx.array:
        return mx.broadcast_to(row[None, None, :], (batch, steps, row.shape[0]))

    def __call__(
        self,
        inputs,
        cache=None,
        input_embeddings=None,
        return_hidden: bool = False,
        n_confirmed: int = 0,
    ):
        batch, steps = inputs.shape
        logits = self._tile(self._target_logits, batch, steps)
        if return_hidden:
            return logits, mx.zeros((batch, steps, self.hidden_size))
        return logits

    def mtp_forward(self, hidden, next_token_ids, mtp_cache):
        batch, steps = next_token_ids.shape
        return self._tile(self._draft_logits, batch, steps)

    def make_mtp_cache(self):
        return []


def _run_stream(
    target_probs: list[float],
    draft_probs: list[float],
    *,
    max_k: int,
    max_tokens: int,
    seed: int,
    temp: float = 1.0,
    top_p: float = 0.0,
) -> list[int]:
    """Emit ``max_tokens`` tokens from ``mtp_generate_step`` at depth ``max_k``.

    ``disable_auto_k=True`` pins the depth: ``max_k=0`` parks every
    round (plain autoregressive reference), ``max_k=K`` runs a full
    chain-of-K verify every round. Both paths share the same filter
    chain and the same mocked distributions, so a difference in the
    emitted stream's statistics is a difference in the SAMPLER.
    """
    from vllm_mlx.spec_decode.mtp.generator import mtp_generate_step

    mx.random.seed(seed)
    model = _FixedDistributionModel(target_probs, draft_probs)
    return [
        tok
        for tok, _lp, _fd in mtp_generate_step(
            mx.array([0], mx.uint32),
            model,
            max_tokens=max_tokens,
            temp=temp,
            top_p=top_p,
            disable_auto_k=True,
            max_k=max_k,
        )
    ]


def _marginal(stream: list[int], vocab: int) -> list[float]:
    counts = [0] * vocab
    for tok in stream:
        counts[tok] += 1
    total = float(len(stream))
    return [c / total for c in counts]


def _lag1_joint(stream: list[int], vocab: int) -> list[float]:
    """Empirical distribution of consecutive emitted pairs, flattened."""
    counts = [0] * (vocab * vocab)
    for first, second in zip(stream, stream[1:]):
        counts[first * vocab + second] += 1
    total = float(len(stream) - 1)
    return [c / total for c in counts]


def _tv(a: list[float], b: list[float]) -> float:
    """Total variation distance between two discrete distributions."""
    return 0.5 * sum(abs(x - y) for x, y in zip(a, b))


# A target/draft pair chosen so the accept ratio p/q spans a wide range
# (4.0, 1.5, 0.667, 0.25 across the four tokens). The wider that spread,
# the more a shared ``u`` couples the per-position accept events — and
# so the larger the joint-distribution error a shared draw produces.
_TARGET = [0.4, 0.3, 0.2, 0.1]
_DRAFT = [0.1, 0.2, 0.3, 0.4]

# Sample budget. The lag-1 table has ``vocab**2`` cells; at 24k tokens
# the Monte-Carlo noise floor on its TV distance is well under the
# thresholds asserted below, which were set from the observed spread
# across seeds with margin.
_N_TOKENS = 24_000
_TV_MARGINAL_MAX = 0.02
_TV_JOINT_MAX = 0.03


@pytest.mark.parametrize("max_k", [2, 3])
def test_kge2_joint_matches_autoregressive(max_k):
    """Chain-of-K sampling must match plain autoregressive JOINTLY, not just marginally.

    This is the regression test for the shared-``u`` defect. Verified
    against the defect: restoring the scalar draw fails this test at
    TV 0.029 (K=2) and 0.066 (K=3) on the marginal, both far outside the
    thresholds, because accepting position 1 biases ``u`` low, inflating
    position 2's accept rate and pulling the following token toward the
    draft distribution.
    """
    vocab = len(_TARGET)
    baseline = _run_stream(_TARGET, _DRAFT, max_k=0, max_tokens=_N_TOKENS, seed=1234)
    spec = _run_stream(_TARGET, _DRAFT, max_k=max_k, max_tokens=_N_TOKENS, seed=1234)

    assert _tv(_marginal(spec, vocab), _marginal(baseline, vocab)) < _TV_MARGINAL_MAX
    assert _tv(_lag1_joint(spec, vocab), _lag1_joint(baseline, vocab)) < _TV_JOINT_MAX


def test_k2_joint_matches_autoregressive_under_top_p():
    """The same joint contract must hold with a top-p filter engaged.

    top_p truncates and renormalises BOTH the target and the draft
    distribution before the accept ratio and the residual are computed,
    so the accept test operates on a different (and much peakier) pair
    of distributions than the unfiltered case. Pinning it separately
    guards the filtered path's accept/residual arithmetic.
    """
    # Both distributions put their tail on the SAME token (t4, 2%), so
    # top_p=0.95 truncates t4 out of BOTH and leaves a shared support
    # {0,1,2,3}. That matters: if truncation left the drafter's top
    # token outside the target's kept set, that token would reject
    # deterministically (p=0) regardless of ``u``, closing the very
    # correlation channel this test exists to probe. Over the shared
    # support the accept ratio still spans 1.94 -> 0.52, so a shared
    # draw has room to couple the positions.
    target = [0.35, 0.25, 0.20, 0.18, 0.02]
    draft = [0.18, 0.20, 0.25, 0.35, 0.02]
    vocab = len(target)
    baseline = _run_stream(
        target, draft, max_k=0, max_tokens=_N_TOKENS, seed=99, top_p=0.95
    )
    spec = _run_stream(
        target, draft, max_k=2, max_tokens=_N_TOKENS, seed=99, top_p=0.95
    )

    assert _tv(_marginal(spec, vocab), _marginal(baseline, vocab)) < _TV_MARGINAL_MAX
    assert _tv(_lag1_joint(spec, vocab), _lag1_joint(baseline, vocab)) < _TV_JOINT_MAX


def test_k2_joint_matches_autoregressive_residual_dominant():
    """Rejection/residual path: a drafter that proposes mostly-wrong tokens.

    The draft mass sits almost entirely where the target has almost
    none, so most positions REJECT and the emitted token comes from the
    normalised residual ``max(p - q, 0)`` rather than from the draft.
    That makes this the case where residual arithmetic — not accept
    arithmetic — determines the output distribution, and it is the one
    a test built only around high-accept-rate drafters would never
    exercise.
    """
    target = [0.55, 0.25, 0.15, 0.05]
    draft = [0.02, 0.03, 0.15, 0.80]
    vocab = len(target)

    baseline = _run_stream(target, draft, max_k=0, max_tokens=_N_TOKENS, seed=7)
    spec = _run_stream(target, draft, max_k=2, max_tokens=_N_TOKENS, seed=7)

    assert _tv(_marginal(spec, vocab), _marginal(baseline, vocab)) < _TV_MARGINAL_MAX
    assert _tv(_lag1_joint(spec, vocab), _lag1_joint(baseline, vocab)) < _TV_JOINT_MAX


def test_baseline_stream_is_iid():
    """Sanity-check the reference: the K=0 stream must itself be i.i.d.

    Every other test in this file reads a lag-1 dependence as evidence
    of a speculative-sampling defect. That inference is only valid if
    the reference stream has no lag-1 dependence to begin with — which
    holds because the mocked model's distribution is position-
    independent. If this test ever fails, the others' diagnosis is
    unsound and should not be trusted.
    """
    vocab = len(_TARGET)
    baseline = _run_stream(_TARGET, _DRAFT, max_k=0, max_tokens=_N_TOKENS, seed=2024)
    marginal = _marginal(baseline, vocab)
    product = [marginal[i] * marginal[j] for i in range(vocab) for j in range(vocab)]
    assert _tv(_lag1_joint(baseline, vocab), product) < _TV_JOINT_MAX
