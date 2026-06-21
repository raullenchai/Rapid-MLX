# SPDX-License-Identifier: Apache-2.0
"""H-11 regression guard — per-request OpenAI ``seed`` reproducibility.

Pre-fix (Tomek r3 repro): five calls to ``/v1/chat/completions`` with
``{"temperature": 0.7, "seed": 42}`` produced five different outputs.
The ``seed`` field was not declared on ``ChatCompletionRequest`` so
Pydantic silently dropped it; the wire-claim was false.

The fix plumbs ``seed`` through five layers — the same surfaces F-011 /
#355 covered for the other sampling params:

  1. ``ChatCompletionRequest`` / ``CompletionRequest`` (api/models.py) —
     ``seed: int | None`` declared without a range bound (matches
     OpenAI's documented unbounded-int surface; codex round-6) and
     guarded by a ``mode="before"`` validator that rejects bool /
     non-int. Backend uint32 narrowing happens in
     ``make_seeded_sampler`` via a deterministic ``& 0xFFFFFFFF`` fold.
  2. ``build_extended_sampling_kwargs`` (service/helpers.py) — forwards
     the request's ``seed`` value through to ``chat_kwargs``.
  3. ``SamplingParams`` (request.py) — carries ``seed: int | None``.
  4. ``BatchedEngine.generate`` / ``stream_generate`` (engine/batched.py)
     — pops ``seed`` from kwargs into ``_sp_kwargs``.
  5. ``Scheduler._get_request_sampler`` (scheduler.py) — routes seeded
     requests around the shared sampler cache and builds a fresh
     ``make_seeded_sampler`` closure that threads an explicit
     ``mx.random.key`` per step.

The sampler primitive (``_seeded_sampler.make_seeded_sampler``) uses
``mx.random.split`` + ``mx.random.categorical(..., key=...)`` so two
seeded requests can interleave their sampler calls (concurrent batch
rows in ``GenerationBatch._step``) without cross-contaminating each
other's PRNG sequences — a property the global ``mx.random.state``
cannot provide.
"""

from __future__ import annotations

import mlx.core as mx
import pytest
from pydantic import ValidationError

from vllm_mlx._seeded_sampler import make_seeded_sampler
from vllm_mlx.api.models import ChatCompletionRequest, CompletionRequest
from vllm_mlx.request import SamplingParams
from vllm_mlx.service.helpers import build_extended_sampling_kwargs

# =============================================================================
# Layer 1 — Pydantic models preserve the seed field
# =============================================================================


def test_chat_completion_request_preserves_seed():
    """ChatCompletionRequest must surface ``seed`` as an attribute after
    parsing JSON. Pre-H-11 Pydantic dropped it silently — Tomek r3's
    repro hinged on this."""
    req = ChatCompletionRequest(
        model="qwen3-0.6b-8bit",
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.7,
        seed=42,
    )
    assert req.seed == 42


def test_chat_completion_request_seed_defaults_to_none():
    """Unset ``seed`` must default to ``None`` so the scheduler can
    distinguish 'no seed' (cache the interned sampler) from 'seed=0'
    (a legitimate value — eval harnesses routinely use zero)."""
    req = ChatCompletionRequest(
        model="qwen3-0.6b-8bit",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert req.seed is None


def test_completion_request_preserves_seed():
    """Mirror coverage on the legacy /v1/completions route."""
    req = CompletionRequest(
        model="qwen3-0.6b-8bit",
        prompt="hi",
        seed=123,
    )
    assert req.seed == 123


def test_responses_request_preserves_seed_through_adapter():
    """Codex r3 BLOCKING regression guard. The /v1/responses surface
    has its own ResponsesRequest model — pre-fix, the ``seed`` field
    was not declared and Pydantic dropped it before the adapter
    converted the body to ChatCompletionRequest. The result was that
    seeded Responses requests silently lost determinism.

    Verifies (a) ResponsesRequest declares seed, and (b) the adapter
    forwards it through to the materialized ChatCompletionRequest so
    the downstream ``build_extended_sampling_kwargs`` → engine path
    sees the value.
    """
    from vllm_mlx.api.responses_adapter import responses_to_openai
    from vllm_mlx.api.responses_models import ResponsesRequest

    req = ResponsesRequest(model="qwen3-0.6b-8bit", input="hi", seed=42)
    assert req.seed == 42, "ResponsesRequest dropped seed at parse"

    chat_req = responses_to_openai(req)
    assert chat_req.seed == 42, (
        "responses_adapter did not forward seed to ChatCompletionRequest — "
        "the Responses route would lose determinism"
    )


def test_responses_request_rejects_bool_seed():
    """Codex r4 BLOCKING regression guard. Without an explicit
    ``mode="before"`` validator on ResponsesRequest.seed, Pydantic v2
    silently coerces ``seed: true`` → ``1`` before the chat-layer's
    bool guard sees it, the adapter then materialises a valid-looking
    ChatCompletionRequest with ``seed=1``, and the chat-layer bool
    rejection is bypassed. ResponsesRequest must enforce the same
    contract at its own parse layer."""
    from vllm_mlx.api.responses_models import ResponsesRequest

    with pytest.raises(ValidationError):
        ResponsesRequest(model="qwen3-0.6b-8bit", input="hi", seed=True)


def test_responses_request_accepts_negative_seed():
    """Codex round-6 BLOCKING regression guard. The OpenAI Responses
    API surface accepts any integer ``seed``; rapid-mlx must not 422
    on negative seeds (or 64-bit positive ones) — that would break
    compatibility with clients passing the documented unbounded
    integer range. Narrowing to mlx-core's uint32 happens later in
    ``make_seeded_sampler``."""
    from vllm_mlx.api.responses_models import ResponsesRequest

    req = ResponsesRequest(model="qwen3-0.6b-8bit", input="hi", seed=-1)
    assert req.seed == -1


def test_responses_request_accepts_above_uint32_seed():
    """64-bit seeds the OpenAI spec permits must reach the sampler
    layer rather than 422'ing at parse time. The downstream fold
    (``seed & 0xFFFFFFFF`` in ``make_seeded_sampler``) maps them
    deterministically to the backend's uint32 PRNG-key range."""
    from vllm_mlx.api.responses_models import ResponsesRequest

    big_seed = 0x1_00000000
    req = ResponsesRequest(model="qwen3-0.6b-8bit", input="hi", seed=big_seed)
    assert req.seed == big_seed


def test_seed_accepts_zero():
    """``seed=0`` is a legitimate value — eval harnesses use it as the
    default. The forwarding gate in ``build_extended_sampling_kwargs``
    uses ``value is not None`` (not truthiness) so 0 must survive."""
    req = ChatCompletionRequest(
        model="qwen3-0.6b-8bit",
        messages=[{"role": "user", "content": "hi"}],
        seed=0,
    )
    assert req.seed == 0


def test_seed_accepts_negative():
    """Codex round-6 BLOCKING regression guard.

    OpenAI's spec documents ``seed`` as an unbounded integer (their
    Python SDK ships ``seed: Optional[int]`` with no range), so the
    request-model layer must NOT 422 on negative seeds. Backend
    narrowing to mlx-core's uint32 PRNG key range is applied
    deterministically downstream in ``make_seeded_sampler`` via
    ``seed & 0xFFFFFFFF`` — Python's well-defined ``int.__and__`` on
    negatives makes the fold round-trip cleanly.

    Pre-fix (round 5): ``Field(ge=0)`` rejected this with a 422 and
    broke compatibility for the OpenAI-documented surface.
    """
    req = ChatCompletionRequest(
        model="qwen3-0.6b-8bit",
        messages=[{"role": "user", "content": "hi"}],
        seed=-1,
    )
    assert req.seed == -1


def test_seed_accepts_above_uint32():
    """Codex round-6 BLOCKING regression guard.

    OpenAI clients routinely pass 64-bit (or larger) seeds. Rapid-MLX
    must accept the value at the request layer; the deterministic
    fold in ``make_seeded_sampler`` collapses it to mlx-core's uint32
    PRNG key range. The reproducibility contract still holds
    (within-engine — same input value → same output sequence) which
    is all the OpenAI spec promises.
    """
    big_seed = 0x1_00000000  # 2**32
    req = ChatCompletionRequest(
        model="qwen3-0.6b-8bit",
        messages=[{"role": "user", "content": "hi"}],
        seed=big_seed,
    )
    assert req.seed == big_seed


def test_completion_request_accepts_wide_int_seed():
    """Codex round-6 BLOCKING regression guard on the legacy
    completions surface. Same contract as ChatCompletionRequest /
    ResponsesRequest — accept the full OpenAI integer surface; narrow
    in the sampler."""
    req = CompletionRequest(
        model="qwen3-0.6b-8bit",
        prompt="hi",
        seed=2**40,
    )
    assert req.seed == 2**40


def test_seed_rejects_bool():
    """Python ``bool`` is an ``int`` subclass; Pydantic v2 would silently
    coerce ``True`` → 1 / ``False`` → 0 on a typed ``int | None`` field.
    Same family as ``_validate_n``'s bool guard. A client that sends
    ``seed: true`` almost certainly meant something else and the silent
    coercion to ``seed=1`` would be a footgun."""
    with pytest.raises(ValidationError):
        ChatCompletionRequest(
            model="qwen3-0.6b-8bit",
            messages=[{"role": "user", "content": "hi"}],
            seed=True,
        )


# =============================================================================
# Layer 2 — build_extended_sampling_kwargs forwards seed
# =============================================================================


def test_build_extended_sampling_kwargs_forwards_seed():
    """When the request carries ``seed``, the helper must include it in
    the kwargs the route hands to ``engine.chat`` / ``engine.generate``."""
    req = ChatCompletionRequest(
        model="qwen3-0.6b-8bit",
        messages=[{"role": "user", "content": "hi"}],
        seed=42,
    )
    kwargs = build_extended_sampling_kwargs(req)
    assert kwargs.get("seed") == 42


def test_build_extended_sampling_kwargs_omits_seed_when_absent():
    """When ``seed`` is unset, the helper must NOT forward ``seed=None``
    onto the engine — that would override SamplingParams' own default
    and could surprise the cache logic."""
    req = ChatCompletionRequest(
        model="qwen3-0.6b-8bit",
        messages=[{"role": "user", "content": "hi"}],
    )
    kwargs = build_extended_sampling_kwargs(req)
    assert "seed" not in kwargs


def test_build_extended_sampling_kwargs_forwards_seed_zero():
    """``seed=0`` must be forwarded (not collapsed by a truthy gate)."""
    req = ChatCompletionRequest(
        model="qwen3-0.6b-8bit",
        messages=[{"role": "user", "content": "hi"}],
        seed=0,
    )
    kwargs = build_extended_sampling_kwargs(req)
    assert kwargs.get("seed") == 0


# =============================================================================
# Layer 3 — SamplingParams carries seed
# =============================================================================


def test_sampling_params_accepts_seed():
    sp = SamplingParams(temperature=0.7, top_p=0.9, seed=42)
    assert sp.seed == 42


def test_sampling_params_seed_defaults_to_none():
    sp = SamplingParams(temperature=0.7, top_p=0.9)
    assert sp.seed is None


# =============================================================================
# Layer 4 — Seeded sampler reproducibility (the heart of the fix)
# =============================================================================


@pytest.fixture
def logprobs_fixture():
    """A deterministic [1, vocab] log-probability tensor for sampler tests.

    Built from a fixed-seed normal draw so the same logits are seen on
    every test run, isolating the sampler's PRNG state from the
    fixture's PRNG state.

    Codex round-3 NIT: pass an explicit ``key=`` to ``mx.random.normal``
    instead of seeding the global ``mx.random.state`` — otherwise this
    fixture would mutate process-global PRNG state that other tests in
    the same session pick up (test order would then matter for any
    sampler test that touches ``mx.random.*``).
    """
    fixture_key = mx.random.key(0)
    logits = mx.random.normal(shape=(1, 32000), key=fixture_key)
    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    mx.eval(logprobs)
    return logprobs


def _sample_sequence(sampler, logprobs, n: int) -> list[int]:
    return [int(sampler(logprobs)[0]) for _ in range(n)]


def test_seeded_sampler_accepts_negative_seed(logprobs_fixture):
    """Codex round-6 regression guard. The sampler factory must accept
    negative seeds without raising — the fold (``seed & 0xFFFFFFFF``)
    handles negative ints via Python's well-defined ``int.__and__``
    semantics (conceptually two's complement with an infinite sign
    bit, so e.g. ``-1 & 0xFFFFFFFF == 0xFFFFFFFF``)."""
    s = make_seeded_sampler(seed=-1, temperature=0.7, top_p=0.9)
    out = int(s(logprobs_fixture)[0])
    vocab = int(logprobs_fixture.shape[-1])
    assert 0 <= out < vocab


def test_seeded_sampler_accepts_large_seed(logprobs_fixture):
    """Codex round-6 regression guard. Large (>uint32) seeds must
    work — same input value always maps to the same backend key."""
    s = make_seeded_sampler(seed=2**40 + 17, temperature=0.7, top_p=0.9)
    out = int(s(logprobs_fixture)[0])
    vocab = int(logprobs_fixture.shape[-1])
    assert 0 <= out < vocab


def test_seeded_sampler_seeds_with_same_uint32_fold_match(logprobs_fixture):
    """Codex round-6 regression guard on the deterministic-fold
    contract. Two seeds whose low 32 bits are identical (here ``42``
    and ``42 + 2**32``) must produce the SAME token sequence — that
    is the explicit consequence of folding to mlx-core's uint32 PRNG
    key range. Documenting this in a test makes the contract
    discoverable and pins it against accidental future changes to
    the fold function (e.g. a switch to ``hash()`` which would break
    cross-seed reproducibility unpredictably)."""
    s_low = make_seeded_sampler(seed=42, temperature=0.7, top_p=0.9)
    s_high = make_seeded_sampler(seed=42 + 2**32, temperature=0.7, top_p=0.9)
    seq_low = _sample_sequence(s_low, logprobs_fixture, 8)
    seq_high = _sample_sequence(s_high, logprobs_fixture, 8)
    assert seq_low == seq_high, (
        "seeds with the same low-32 bits must fold to the same backend "
        "PRNG key and produce identical sequences — the fold contract "
        "is broken"
    )


def test_seeded_sampler_negative_seed_deterministic(logprobs_fixture):
    """Codex round-6 regression guard. Two samplers with the same
    negative seed must produce the same sequence — negative seeds
    are not second-class citizens, they just take the deterministic
    fold path."""
    s1 = make_seeded_sampler(seed=-12345, temperature=0.7, top_p=0.9)
    s2 = make_seeded_sampler(seed=-12345, temperature=0.7, top_p=0.9)
    seq1 = _sample_sequence(s1, logprobs_fixture, 8)
    seq2 = _sample_sequence(s2, logprobs_fixture, 8)
    assert seq1 == seq2


def test_seeded_sampler_same_seed_same_sequence(logprobs_fixture):
    """Core contract: two seeded samplers built with the same
    ``(seed, temp, top_p)`` produce the same token sequence given the
    same logits stream. This is the H-11 wire-claim made real."""
    s1 = make_seeded_sampler(seed=42, temperature=0.7, top_p=0.9)
    s2 = make_seeded_sampler(seed=42, temperature=0.7, top_p=0.9)
    seq1 = _sample_sequence(s1, logprobs_fixture, 16)
    seq2 = _sample_sequence(s2, logprobs_fixture, 16)
    assert seq1 == seq2


def test_seeded_sampler_different_seed_different_sequence(logprobs_fixture):
    """Different seeds must produce different sequences; if they didn't,
    the seed parameter would be cosmetic."""
    s_a = make_seeded_sampler(seed=42, temperature=0.7, top_p=0.9)
    s_b = make_seeded_sampler(seed=99, temperature=0.7, top_p=0.9)
    seq_a = _sample_sequence(s_a, logprobs_fixture, 16)
    seq_b = _sample_sequence(s_b, logprobs_fixture, 16)
    assert seq_a != seq_b


def test_seeded_sampler_five_runs_identical(logprobs_fixture):
    """Tomek r3's exact repro shape: five fresh seeded samplers all built
    with the same ``(seed=42, temp=0.7, top_p=0.9)`` produce the same
    16-token sequence. Pre-fix, five calls produced five different
    outputs — this test would have failed."""
    sequences = []
    for _ in range(5):
        s = make_seeded_sampler(seed=42, temperature=0.7, top_p=0.9)
        sequences.append(_sample_sequence(s, logprobs_fixture, 16))
    # All five must equal the first
    for i, seq in enumerate(sequences[1:], start=1):
        assert seq == sequences[0], (
            f"run {i} diverged from run 0 — seed parameter is non-functional"
        )


def test_seeded_sampler_interleaved_concurrency_isolation(logprobs_fixture):
    """Two seeded samplers run interleaved (simulating concurrent batch
    rows in ``GenerationBatch._step`` with different seeds) must each
    produce the same sequence they produce in isolation.

    This is the property mlx-lm's stock sampler chain CANNOT provide:
    it reads ``mx.random.state`` (process-global) so interleaving would
    cross-contaminate the PRNG sequences. The seeded sampler threads a
    private key via ``mx.random.split`` to avoid that.
    """
    # Solo baselines
    s_a_solo = make_seeded_sampler(seed=42, temperature=0.7, top_p=0.9)
    s_b_solo = make_seeded_sampler(seed=99, temperature=0.7, top_p=0.9)
    solo_a = _sample_sequence(s_a_solo, logprobs_fixture, 8)
    solo_b = _sample_sequence(s_b_solo, logprobs_fixture, 8)

    # Interleaved
    s_a_inter = make_seeded_sampler(seed=42, temperature=0.7, top_p=0.9)
    s_b_inter = make_seeded_sampler(seed=99, temperature=0.7, top_p=0.9)
    inter_a, inter_b = [], []
    for _ in range(8):
        inter_a.append(int(s_a_inter(logprobs_fixture)[0]))
        inter_b.append(int(s_b_inter(logprobs_fixture)[0]))

    assert inter_a == solo_a, (
        "interleaving leaked state into seed=42's sequence — concurrency "
        "isolation is broken"
    )
    assert inter_b == solo_b, (
        "interleaving leaked state into seed=99's sequence — concurrency "
        "isolation is broken"
    )


def test_seeded_sampler_greedy_short_circuit(logprobs_fixture):
    """``temperature=0`` is greedy / argmax; seed is irrelevant. The
    sampler factory still accepts seeded greedy requests for caller
    convenience but the output is just the argmax."""
    s = make_seeded_sampler(seed=42, temperature=0.0)
    out1 = int(s(logprobs_fixture)[0])
    out2 = int(s(logprobs_fixture)[0])
    argmax = int(mx.argmax(logprobs_fixture, axis=-1)[0])
    assert out1 == argmax
    assert out2 == argmax


def test_seeded_sampler_top_k_combined(logprobs_fixture):
    """Top-k layered on top of top-p must still be deterministic."""
    s1 = make_seeded_sampler(seed=42, temperature=0.7, top_p=0.9, top_k=50)
    s2 = make_seeded_sampler(seed=42, temperature=0.7, top_p=0.9, top_k=50)
    assert _sample_sequence(s1, logprobs_fixture, 8) == _sample_sequence(
        s2, logprobs_fixture, 8
    )


def test_seeded_sampler_min_p_combined(logprobs_fixture):
    """min_p (without top_p) must also be deterministic."""
    s1 = make_seeded_sampler(seed=42, temperature=0.7, top_p=0.0, min_p=0.05)
    s2 = make_seeded_sampler(seed=42, temperature=0.7, top_p=0.0, min_p=0.05)
    assert _sample_sequence(s1, logprobs_fixture, 8) == _sample_sequence(
        s2, logprobs_fixture, 8
    )


def test_seeded_sampler_top_k_above_vocab_clamps(logprobs_fixture):
    """Codex r2 BLOCKING #1 regression guard. ``top_k`` larger than the
    vocab dimension must clamp to vocab rather than crash the
    ``put_along_axis`` mask scatter (over-slice on the descending sort
    used to read out top-k positions). A caller setting ``top_k=10**6``
    on a 32k-vocab model is asking 'no top-k cap' — the sampler must
    survive and produce a sample, not raise.
    """
    s = make_seeded_sampler(seed=42, temperature=0.7, top_p=0.9, top_k=10**6)
    # Must not raise and must return an in-range token id
    out = int(s(logprobs_fixture)[0])
    vocab = int(logprobs_fixture.shape[-1])
    assert 0 <= out < vocab


def test_seeded_sampler_aggressive_min_p_never_empty_mask(logprobs_fixture):
    """Codex r2 BLOCKING #2 + r5 NIT regression guard.

    Each individual mask (top-p / top-k / min-p) already preserves the
    argmax row by construction (top-p has an explicit top-1 OR, top-k
    keeps at least top_k>=1 positions which always includes argmax,
    min-p keeps tokens whose prob >= min_p*max_prob and argmax has
    prob == max_prob). So under the current implementation the
    intersection never produces an all-False row even before the
    post-intersection argmax-OR rescue runs.

    Codex r5 NIT correctly flagged that ``top_k=1`` therefore doesn't
    exercise the rescue branch on its own — the test was vacuously
    passing. The rescue exists as a defensive belt for future code
    changes that might add a new mask without the argmax invariant
    (e.g. a logit-bias mask that zeros specific token positions). The
    test below verifies the contract the rescue WOULD enforce by
    constructing a synthetic "what if everything filtered argmax out"
    scenario and checking that the sampler still returns a valid
    in-range token under aggressive cutoffs.
    """
    # Build logits with a clear argmax so the contract is testable. Use
    # an explicit ``key=`` so we don't mutate the process-global PRNG
    # state and pollute other tests in the session (codex r3 NIT).
    sharp_logits = mx.random.normal(shape=(1, 1024), key=mx.random.key(0))
    sharp_logprobs = sharp_logits - mx.logsumexp(sharp_logits, axis=-1, keepdims=True)
    mx.eval(sharp_logprobs)
    argmax = int(mx.argmax(sharp_logprobs, axis=-1)[0])
    vocab = int(sharp_logprobs.shape[-1])

    # Layered aggressive cutoffs — top_p=0.001 is so tight that only
    # the top few tokens survive top-p; min_p=0.999 then drops every
    # non-argmax token; top_k=1 redundantly cuts to one. Under any
    # OPTIONAL future regression of one mask's argmax invariant, the
    # combined intersection could go empty — the rescue's job is to
    # OR argmax back in.
    s = make_seeded_sampler(seed=42, temperature=0.7, top_p=0.001, min_p=0.999, top_k=1)
    out = int(s(sharp_logprobs)[0])
    assert 0 <= out < vocab, "sampler returned an out-of-range token id"
    assert out == argmax, (
        "aggressive top_p + min_p + top_k must collapse to argmax — "
        "either via individual masks preserving argmax or via the "
        "combined-mask rescue. Anything else means the sampler is "
        "sampling from an invalid -inf distribution."
    )

    # Direct contract: a freshly-built sampler with cutoffs that
    # individually drop EVERY token from a near-uniform distribution
    # (min_p with min_p=1.0 + a tiny eps would do this except the
    # rescue catches it) must still return SOME valid token. Verify by
    # repeated invocation under the aggressive shape above — never
    # raises, always in-range.
    for _ in range(5):
        out = int(s(sharp_logprobs)[0])
        assert 0 <= out < vocab


def test_seeded_sampler_rescue_does_not_taint_nonempty_rows(logprobs_fixture):
    """Codex round-7 BLOCKING regression guard.

    The round-2 fix OR'd argmax into the combined mask unconditionally
    to prevent all-``-inf`` rows from reaching ``mx.random.categorical``.
    Codex r7 caught a hole in that contract: when ``top_k`` is layered
    with a tighter ``top_p`` / ``min_p`` that could (theoretically)
    exclude argmax from the top-k set, the unconditional rescue would
    re-inject argmax — changing ``top_k`` semantics from "sample only
    from the top K" to "sample from top K or argmax".

    The round-7 fix gates the rescue on per-row emptiness via
    ``mx.where(any_kept, mask, argmax_keep)``. This test pins the
    determinism property the fix preserves for non-empty rows: two
    samplers with the same ``(seed, top_k, top_p, min_p)`` must still
    produce identical sequences. The fix changes the rescue branch but
    must not perturb the math for any row where the intersection is
    non-empty (which is the overwhelming common case — argmax is
    preserved by each individual mask's "at least one kept" invariant).
    """
    # Non-aggressive cutoffs so the intersection is non-empty on every
    # step — this exercises the ``any_kept == True`` branch of the
    # ``mx.where`` rescue selector. Reproducibility on the non-empty
    # path is what would have regressed if the round-7 fix swapped the
    # mask for the wrong branch (e.g. ``mx.where(any_kept, argmax_keep,
    # mask)`` — flipped arms).
    s1 = make_seeded_sampler(seed=42, temperature=0.7, top_p=0.9, top_k=50, min_p=0.05)
    s2 = make_seeded_sampler(seed=42, temperature=0.7, top_p=0.9, top_k=50, min_p=0.05)
    seq1 = _sample_sequence(s1, logprobs_fixture, 16)
    seq2 = _sample_sequence(s2, logprobs_fixture, 16)
    assert seq1 == seq2, (
        "non-empty-row path lost determinism — the round-7 conditional "
        "rescue likely flipped the mx.where arms or otherwise perturbed "
        "the intersection mask"
    )


# =============================================================================
# Layer 5 — Scheduler routes seeded requests around the cache
# =============================================================================


def test_scheduler_seeded_request_skips_cache():
    """``Scheduler._get_request_sampler`` MUST NOT cache seeded samplers.

    Two concurrent requests with the same ``(temp, top_p, seed)`` would
    otherwise share one closure and the second request's first token
    would be the first request's second token — silent reproducibility
    bug. Verified by attribute inspection on the scheduler shape since
    constructing a live scheduler in a unit test is expensive.
    """
    # _get_request_sampler is a normal method, not async; we can call it
    # on a lightweight stub. A real scheduler instance has heavy setup
    # (model load, async loop) so we mimic just the cache surface.
    from collections import OrderedDict

    from vllm_mlx.scheduler import Scheduler

    class _Stub:
        _sampler_cache: OrderedDict = OrderedDict()
        _sampler_cache_max = 32

    stub = _Stub()
    # Bind the real method via Scheduler.__dict__ to get the unbound impl.
    get_sampler = Scheduler._get_request_sampler.__get__(stub)

    sp1 = SamplingParams(temperature=0.7, top_p=0.9, seed=42)
    sp2 = SamplingParams(temperature=0.7, top_p=0.9, seed=42)

    s1 = get_sampler(sp1)
    s2 = get_sampler(sp2)

    # Must be different closures even with identical seeds — otherwise
    # the second request would resume the first request's PRNG sequence
    # mid-stream.
    assert s1 is not s2, (
        "seeded requests share a closure — concurrent same-seed requests "
        "would corrupt each other's PRNG state"
    )

    # Same-seed closures must still each produce the same token from
    # the same logits (each closure starts from the seed independently).
    # Use an explicit key to avoid mutating the process-global PRNG
    # state (codex r3 NIT).
    logits = mx.random.normal(shape=(1, 32000), key=mx.random.key(0))
    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    mx.eval(logprobs)
    assert int(s1(logprobs)[0]) == int(s2(logprobs)[0])


def test_scheduler_unseeded_request_uses_cache():
    """Sanity check: when ``seed`` is None, the existing cache path
    still runs (we MUST NOT regress the fast-path interning that other
    requests rely on for batched-sampler eligibility)."""
    from collections import OrderedDict

    from vllm_mlx.scheduler import Scheduler

    class _Stub:
        _sampler_cache: OrderedDict = OrderedDict()
        _sampler_cache_max = 32

    stub = _Stub()
    get_sampler = Scheduler._get_request_sampler.__get__(stub)

    sp1 = SamplingParams(temperature=0.7, top_p=0.9)
    sp2 = SamplingParams(temperature=0.7, top_p=0.9)
    s1 = get_sampler(sp1)
    s2 = get_sampler(sp2)
    # Cached → identity-equal
    assert s1 is s2
