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
through the identical filter chain. This isolates rejection
sampling from filtering. The serving-parity tests below separately compare
the actual serving sampler to detect errors shared by both generator depths.

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
pytestmark = pytest.mark.requires_mlx


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

    from rapid_mlx.spec_decode.mtp.accept_counter import (
        reset_global_counter_for_tests,
    )
    from rapid_mlx.spec_decode.mtp.cache_patch import _unpatch_for_tests
    from rapid_mlx.spec_decode.mtp.draft_k_controller_v2 import reset_controllers

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
        self._backbone_rows = {}
        self._hidden_rows = {}
        self._draft_rows = {}

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
        shape = (batch, steps)
        # These outputs depend only on shape. Reuse their fixed tensors,
        # but return fresh array wrappers so caller writes cannot change the
        # distributions or hidden states returned by a later forward.
        if shape not in self._backbone_rows:
            self._backbone_rows[shape] = self._tile(self._target_logits, batch, steps)
        logits = mx.array(self._backbone_rows[shape])
        if return_hidden:
            if shape not in self._hidden_rows:
                self._hidden_rows[shape] = mx.zeros((batch, steps, self.hidden_size))
            return logits, mx.array(self._hidden_rows[shape])
        return logits

    def mtp_forward(self, hidden, next_token_ids, mtp_cache):
        batch, steps = next_token_ids.shape
        shape = (batch, steps)
        if shape not in self._draft_rows:
            self._draft_rows[shape] = self._tile(self._draft_logits, batch, steps)
        return mx.array(self._draft_rows[shape])

    def make_mtp_cache(self):
        return []


@pytest.mark.parametrize("shape", [(1, 1), (1, 4), (2, 3)])
def test_fixed_model_outputs_ignore_input_values_and_caller_writes(shape):
    model = _FixedDistributionModel(_TARGET, _DRAFT)
    inputs = mx.zeros(shape, dtype=mx.uint32)
    changed_inputs = mx.ones(shape, dtype=mx.uint32)
    target = model._tile(mx.log(mx.array(_TARGET)), *shape)
    draft = model._tile(mx.log(mx.array(_DRAFT)), *shape)

    logits, hidden = model(inputs, return_hidden=True)
    draft_logits = model.mtp_forward(hidden, inputs, [])
    logits[0, 0, 0] = 42.0
    hidden[0, 0, 0] = 42.0
    draft_logits[0, 0, 0] = 42.0

    next_logits, next_hidden = model(changed_inputs, return_hidden=True)
    assert mx.array_equal(next_logits, target)
    assert mx.array_equal(next_hidden, mx.zeros((*shape, model.hidden_size)))
    assert mx.array_equal(model.mtp_forward(next_hidden, changed_inputs, []), draft)

    plain_logits = model(inputs)
    plain_logits[0, 0, 0] = 42.0
    assert mx.array_equal(model(changed_inputs), target)


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
    from rapid_mlx.spec_decode.mtp.generator import mtp_generate_step

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


@pytest.mark.parametrize("dtype_name", ["float32", "bfloat16"])
@pytest.mark.parametrize("temp", [0.3, 1.0, 1.7])
@pytest.mark.parametrize("top_k", [0, 3])
def test_mtp_filtered_probabilities_match_serving(monkeypatch, dtype_name, temp, top_k):
    """Compare the actual categorical inputs, including token IDs at ties."""
    from tests.test_mtp_sampling_parity import sampling_harness

    env, captured, categorical = sampling_harness(mx, temp=temp, top_k=top_k)
    logits = mx.array([[-3, -2, 0, 0, 0, 2]], dtype=getattr(mx, dtype_name))
    _, logprobs, acceptance = env["_process_and_sample"](None, logits)
    mtp_scaled = captured.pop()
    monkeypatch.setattr(mx.random, "categorical", categorical)
    env["make_fused_top_p_temp_sampler"](temp, 0.95, top_k)(logprobs)
    sorted_scaled = captured.pop()
    indices = mx.argsort(mx.exp(logprobs.astype(mx.float32)), axis=-1)
    serving_scaled = mx.put_along_axis(
        mx.full_like(sorted_scaled, -float("inf")), indices, sorted_scaled, axis=-1
    )
    assert mx.array_equal(mx.isfinite(mtp_scaled), mx.isfinite(serving_scaled))
    assert mx.allclose(mx.exp(acceptance), mx.softmax(serving_scaled), atol=1e-7)
    assert acceptance.dtype == mx.float32
    # Stable ascending ties put higher token IDs in the retained top-k tail.
    if top_k == 3:
        assert not mx.isfinite(mtp_scaled[0, 2]).item()
        assert mx.isfinite(mtp_scaled[0, 3]).item()
        assert mx.isfinite(mtp_scaled[0, 4]).item()


@pytest.mark.parametrize("dtype_name", ["float32", "bfloat16"])
def test_nucleus_threshold_uses_fp32_mass(monkeypatch, dtype_name):
    """A cutoff between rounded and FP32 tail mass must follow serving.

    On the old MTP path, BF16 exp rounds the smallest probability before
    comparing cumulative mass, retaining a different nucleus (or cumsum
    rejects BF16 entirely on runtimes without that operation).
    """
    from tests.test_mtp_sampling_parity import sampling_harness

    logits = mx.array([-3, -1, 0, 1], dtype=getattr(mx, dtype_name))
    logprobs = logits - mx.logsumexp(logits, keepdims=True)
    mass = mx.exp(logprobs.astype(mx.float32))[0].item()
    rounded = mx.exp(logprobs.astype(mx.bfloat16))[0].item()
    assert mass != rounded
    threshold = (mass + rounded) / 2
    top_p = 1 - threshold
    env, captured, categorical = sampling_harness(mx, temp=1.0, top_p=top_p)
    _, _, acceptance = env["_process_and_sample"](None, logits)
    mtp_scaled = captured.pop()
    monkeypatch.setattr(mx.random, "categorical", categorical)
    env["make_fused_top_p_temp_sampler"](1.0, top_p)(logprobs)
    sorted_scaled = captured.pop()
    assert mx.array_equal(mx.isfinite(mtp_scaled), mx.isfinite(sorted_scaled))
    assert mx.allclose(mx.exp(acceptance), mx.softmax(sorted_scaled), atol=1e-7)
    assert mx.isfinite(mtp_scaled[0]).item() == (mass > threshold)


@pytest.mark.parametrize("draft", [[0.55, 0.25, 0.15, 0.05], [0.02, 0.03, 0.15, 0.80]])
def test_k2_matches_actual_serving_distribution(monkeypatch, draft):
    """High acceptance and rejection-heavy runs use the serving distribution."""
    from rapid_mlx._sampler_fast_path import make_fused_top_p_temp_sampler

    target = [0.55, 0.25, 0.15, 0.05]
    logits = mx.log(mx.array(target)).astype(mx.bfloat16)
    logprobs = logits - mx.logsumexp(logits, keepdims=True)
    captured = []

    def categorical(scaled):
        captured.append(mx.softmax(scaled))
        return mx.argmax(scaled, axis=-1)

    # Measure the actual serving sampler's weights without Monte Carlo noise.
    with monkeypatch.context() as patch:
        patch.setattr(mx.random, "categorical", categorical)
        make_fused_top_p_temp_sampler(0.7, 0.95, 3)(logprobs)
    indices = mx.argsort(mx.exp(logprobs.astype(mx.float32)))
    expected = mx.put_along_axis(
        mx.zeros_like(logprobs.astype(mx.float32)), indices, captured[0], axis=-1
    )
    model = _FixedDistributionModel(target, draft)
    model._target_logits = logits
    model._draft_logits = model._draft_logits.astype(mx.bfloat16)
    from rapid_mlx.spec_decode.mtp.generator import mtp_generate_step

    mx.random.seed(731)
    stream = [
        token
        for token, _, _ in mtp_generate_step(
            mx.array([0], mx.uint32),
            model,
            max_tokens=12_000,
            temp=0.7,
            top_p=0.95,
            top_k=3,
            disable_auto_k=True,
            max_k=2,
        )
    ]
    probabilities = expected.astype(mx.float32).tolist()
    joint = [a * b for a in probabilities for b in probabilities]
    assert _tv(_marginal(stream, 4), probabilities) < 0.025
    assert _tv(_lag1_joint(stream, 4), joint) < 0.04
    assert set(stream) <= {i for i, p in enumerate(probabilities) if p > 0}


@pytest.mark.parametrize("dtype_name", ["float32", "bfloat16"])
@pytest.mark.parametrize(
    "top_p,min_p,top_k", [(0, 0, 0), (1, 0, 2), (0.95, 0.1, 3), (0.95, 0, 3)]
)
def test_fallback_distribution_matches_serving(
    monkeypatch, dtype_name, top_p, min_p, top_k
):
    """Fallback settings, including a disabled fast path, keep serving masks."""
    from mlx_lm import sample_utils

    from rapid_mlx.spec_decode.mtp.generator import _make_sampler_chain
    from tests.test_mtp_sampling_parity import sampling_harness

    chain, _ = _make_sampler_chain(top_p=top_p, min_p=min_p, top_k=top_k)
    env, captured, _ = sampling_harness(mx, temp=0.7, fused=False, chain=chain)
    logits = mx.array([-3, 0, 0, 0, 2], dtype=getattr(mx, dtype_name))
    _, logprobs, acceptance = env["_process_and_sample"](None, logits)
    mtp_scaled = captured.pop()

    def categorical(masked, temp):
        captured.append(masked * (1 / temp))
        return mx.argmax(masked, axis=-1)

    monkeypatch.setattr(sample_utils, "categorical_sampling", categorical)
    sample_utils.make_sampler(temp=0.7, top_p=top_p, min_p=min_p, top_k=top_k)(logprobs)
    serving_scaled = captured.pop()
    assert mx.array_equal(mtp_scaled, serving_scaled)
    assert mx.allclose(
        mx.exp(acceptance), mx.softmax(serving_scaled.astype(mx.float32)), atol=1e-7
    )
    assert acceptance.dtype == mx.float32


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16", "float32"])
@pytest.mark.parametrize(
    "top_p,min_p,top_k", [(0.95, 0, 3), (0, 0.1, 3), (1e-9, 1, 100)]
)
@pytest.mark.parametrize(
    "options",
    [
        {},
        {"min_tokens_to_keep": 3},
        {"xtc_probability": 0.5, "xtc_threshold": 0.1, "xtc_special_tokens": [5]},
        {"min_tokens_to_keep": 3, "xtc_probability": 0.5, "xtc_threshold": 0.2},
    ],
)
def test_seeded_mtp_filtered_probabilities_match_serving(
    monkeypatch, dtype_name, top_p, min_p, top_k, options
):
    from rapid_mlx._seeded_sampler import make_seeded_sampler
    from tests.test_mtp_sampling_parity import sampling_harness

    env, captured, _ = sampling_harness(
        mx, temp=0.7, top_p=top_p, min_p=min_p, top_k=top_k, seeded=True, **options
    )
    draw = mx.array(0.25)
    monkeypatch.setattr(mx.random, "uniform", lambda *, key: draw)
    logits = mx.array([-3, -2, 0, 0, 0, 2], dtype=getattr(mx, dtype_name))
    _, logprobs, acceptance = env["_process_and_sample"](None, logits, draw)
    mtp_scaled = captured.pop()

    def capture(logits, *, key):
        captured.append(logits)
        return mx.argmax(logits, axis=-1)

    monkeypatch.setattr(mx.random, "categorical", capture)
    make_seeded_sampler(
        seed=42, temperature=0.7, top_p=top_p, min_p=min_p, top_k=top_k, **options
    )(logprobs)
    serving_scaled = captured.pop()
    assert mx.array_equal(mtp_scaled, serving_scaled)
    assert mx.allclose(mx.exp(acceptance), mx.softmax(serving_scaled), atol=1e-7)
    if not options and (top_p, min_p, top_k) == (0.95, 0, 3):
        assert mx.isfinite(mtp_scaled).tolist() == [
            False,
            False,
            True,
            True,
            False,
            True,
        ]


def test_seeded_xtc_interleaving_preserves_rng_isolation():
    from rapid_mlx._seeded_sampler import make_seeded_sampler

    def sampler(seed):
        return make_seeded_sampler(
            seed=seed,
            temperature=0.7,
            min_p=0.9,
            min_tokens_to_keep=3,
            xtc_probability=0.5,
            xtc_threshold=0.2,
        )

    row = mx.log(mx.array([0.5, 0.3, 0.15, 0.05]))
    baseline = sampler(42)
    expected = [baseline(row).item() for _ in range(30)]
    first, second, unrelated = sampler(42), sampler(42), sampler(99)
    actual_first, actual_second = [], []
    for _ in range(30):
        actual_first.append(first(row).item())
        unrelated(row).item()
        actual_second.append(second(row).item())
    assert expected == actual_first == actual_second
    assert first.lane_rng.draws == second.lane_rng.draws == 60
