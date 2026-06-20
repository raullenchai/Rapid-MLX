# SPDX-License-Identifier: Apache-2.0
"""H-11 regression guard — per-request OpenAI ``seed`` reproducibility.

Pre-fix (Tomek r3 repro): five calls to ``/v1/chat/completions`` with
``{"temperature": 0.7, "seed": 42}`` produced five different outputs.
The ``seed`` field was not declared on ``ChatCompletionRequest`` so
Pydantic silently dropped it; the wire-claim was false.

The fix plumbs ``seed`` through five layers — the same surfaces F-011 /
#355 covered for the other sampling params:

  1. ``ChatCompletionRequest`` / ``CompletionRequest`` (api/models.py) —
     ``seed: int | None`` declared with a ``[0, 2**32-1]`` range bound
     and a ``mode="before"`` validator that rejects bool / non-int.
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


def test_seed_rejects_negative():
    """Negative seeds are not in the ``mx.random.key`` accepted range
    (uint32). Pre-fix Pydantic would have coerced any int unchecked."""
    with pytest.raises(ValidationError):
        ChatCompletionRequest(
            model="qwen3-0.6b-8bit",
            messages=[{"role": "user", "content": "hi"}],
            seed=-1,
        )


def test_seed_rejects_above_uint32():
    """``mx.random.key`` accepts only ``[0, 2**32-1]``; reject anything
    larger with a clean 422 rather than letting it overflow downstream."""
    with pytest.raises(ValidationError):
        ChatCompletionRequest(
            model="qwen3-0.6b-8bit",
            messages=[{"role": "user", "content": "hi"}],
            seed=0x1_00000000,  # 2**32
        )


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
    """
    mx.random.seed(0)
    logits = mx.random.normal(shape=(1, 32000))
    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    mx.eval(logprobs)
    return logprobs


def _sample_sequence(sampler, logprobs, n: int) -> list[int]:
    return [int(sampler(logprobs)[0]) for _ in range(n)]


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
    mx.random.seed(0)
    logits = mx.random.normal(shape=(1, 32000))
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
