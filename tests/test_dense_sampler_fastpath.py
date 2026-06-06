# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the dense-LLM batched-sampler fast path.

Mirrors ``test_mllm_batch_generator``'s shape on the dense path. The MLLM
fast path lives inside our ``MLLMBatchGenerator._step``; the dense path
lives inside mlx-lm's ``GenerationBatch._step`` which we cannot edit, so
we monkey-patch the bound ``_step`` to swap ``samplers`` to all-None +
``fallback_sampler`` to the shared sampler when the running batch is
homogeneous.

The tests stub a minimal ``GenerationBatch``-shaped object so we don't
have to load a model. They lock in the four behaviors that matter:

1. Homogeneous batch (all samplers identity-equal) → swaps to fast path
   (``self.samplers`` is observed as all-None inside the wrapped call;
   ``self.fallback_sampler`` is the shared sampler).
2. Heterogeneous batch → leaves ``self.samplers`` untouched.
3. B=1 → no swap (degenerate case; identity-equality with empty rest is
   true but the patch must NOT engage since the fast-path savings only
   exist for B ≥ 2).
4. Swap is reversed after the call returns — even on exception — so the
   per-request samplers are restored for the NEXT step.
"""

from __future__ import annotations

import types

import pytest

from vllm_mlx.scheduler import _install_dense_sampler_fastpath


class _FakeGenBatch:
    """Stub matching mlx-lm ``GenerationBatch`` attributes the fast path touches."""

    def __init__(self, samplers, fallback):
        self.samplers = samplers
        self.fallback_sampler = fallback
        self.observed_samplers = None
        self.observed_fallback = None
        self.step_calls = 0
        self.raise_in_step: Exception | None = None

    def _step(self):
        self.step_calls += 1
        self.observed_samplers = list(self.samplers)
        self.observed_fallback = self.fallback_sampler
        if self.raise_in_step is not None:
            raise self.raise_in_step
        return ([0] * len(self.samplers), [None] * len(self.samplers))


class _FakeBatchGen:
    def __init__(self, gen_batch):
        self._generation_batch = gen_batch


def _install(gen_batch):
    _install_dense_sampler_fastpath(_FakeBatchGen(gen_batch))


def test_homogeneous_batch_swaps_to_fast_path():
    """All samplers are the same callable → ``self.samplers`` is all-None
    inside the wrapped step and ``self.fallback_sampler`` is the shared one."""
    shared = lambda x: x  # noqa: E731 — stand-in for make_sampler closure
    original_fallback = lambda x: x  # noqa: E731
    gb = _FakeGenBatch(
        samplers=[shared, shared, shared, shared], fallback=original_fallback
    )
    _install(gb)

    gb._step()

    assert gb.observed_samplers == [None, None, None, None]
    assert gb.observed_fallback is shared
    # Restoration: outside the call, the per-request samplers are back.
    assert gb.samplers == [shared, shared, shared, shared]
    assert gb.fallback_sampler is original_fallback


def test_heterogeneous_batch_keeps_per_request_samplers():
    """Mixed sampler identities → patch is a no-op; mlx-lm's per-row loop runs."""
    s1 = lambda x: x  # noqa: E731
    s2 = lambda x: x  # noqa: E731
    original_fallback = lambda x: x  # noqa: E731
    gb = _FakeGenBatch(samplers=[s1, s2, s1, s2], fallback=original_fallback)
    _install(gb)

    gb._step()

    # Slow path observed: samplers and fallback unchanged inside the call.
    assert gb.observed_samplers == [s1, s2, s1, s2]
    assert gb.observed_fallback is original_fallback


def test_b1_does_not_engage_fast_path():
    """B=1 is degenerate — patch must NOT swap (no perf upside, and
    swapping just adds attribute writes per step)."""
    sampler = lambda x: x  # noqa: E731
    original_fallback = lambda x: x  # noqa: E731
    gb = _FakeGenBatch(samplers=[sampler], fallback=original_fallback)
    _install(gb)

    gb._step()

    assert gb.observed_samplers == [sampler]
    assert gb.observed_fallback is original_fallback


def test_homogeneous_with_first_none_does_not_engage():
    """If samplers[0] is None we cannot share — even if the rest match,
    ``None`` means mlx-lm will reach for ``fallback_sampler`` per row.
    The patch's identity-equality check is gated on ``first is not None``."""
    original_fallback = lambda x: x  # noqa: E731
    gb = _FakeGenBatch(samplers=[None, None], fallback=original_fallback)
    _install(gb)

    gb._step()

    # mlx-lm's slow-path branch is `any(self.samplers)` — all-None already
    # takes the fast path naturally; we just must not synthesize a swap.
    assert gb.observed_samplers == [None, None]
    assert gb.observed_fallback is original_fallback


def test_swap_is_reversed_on_exception():
    """If ``_step`` raises, the per-request samplers MUST still be
    restored. Otherwise the next step would silently see the wrong
    sampling distribution for those requests.

    Uses ``pytest.raises`` so a regression that stops raising is caught —
    a bare ``try/except`` would let the test silently pass."""
    shared = lambda x: x  # noqa: E731
    original_fallback = lambda x: x  # noqa: E731
    gb = _FakeGenBatch(samplers=[shared, shared], fallback=original_fallback)
    boom = RuntimeError("metal blew up")
    gb.raise_in_step = boom
    _install(gb)

    with pytest.raises(RuntimeError) as excinfo:
        gb._step()
    assert excinfo.value is boom

    # After the exception, swap must be reverted.
    assert gb.samplers == [shared, shared]
    assert gb.fallback_sampler is original_fallback


def test_install_is_safe_when_step_already_a_plain_closure():
    """SuffixDecoding writes ``gb._step = _suffix_step`` (a plain closure,
    not a bound method). The fast-path installer must wrap it without
    requiring ``__func__``."""
    shared = lambda x: x  # noqa: E731
    captured = {"called": False}

    def suffix_like_step():  # zero-arg closure, mimics _install_suffix_decoding
        captured["called"] = True
        return ([0, 0], [None, None])

    gb = _FakeGenBatch(samplers=[shared, shared], fallback=lambda x: x)
    gb._step = suffix_like_step  # type: ignore[method-assign]
    _install(gb)

    gb._step()

    assert captured["called"]
    # Outside the wrapped call samplers are restored to the per-request list.
    assert gb.samplers == [shared, shared]


def test_install_no_op_when_generation_batch_missing():
    """Defensive: older mlx-lm shapes without ``_generation_batch`` must
    not crash the installer — they just skip the fast path."""

    class _BareBatchGen:
        _generation_batch = None

    # Should not raise.
    _install_dense_sampler_fastpath(_BareBatchGen())

    class _NoStepBatch:
        pass

    class _BareBatchGen2:
        _generation_batch = _NoStepBatch()

    _install_dense_sampler_fastpath(_BareBatchGen2())


def test_sampler_cache_interns_by_param_tuple():
    """``Scheduler._get_request_sampler`` must return the SAME callable
    for identical params — that's what lets the fast-path detector
    short-circuit via ``is`` comparison."""
    from collections import OrderedDict

    from vllm_mlx.scheduler import Scheduler

    # We need a Scheduler instance to exercise _get_request_sampler, but
    # we don't want the heavy __init__ (loads model). Construct via
    # __new__ + manual init of just the attributes the method reads.
    sched = Scheduler.__new__(Scheduler)
    sched._sampler_cache = OrderedDict()
    sched._sampler_cache_max = 32

    class _SP:
        temperature = 0.7
        top_p = 0.95
        min_p = 0.0
        top_k = 20

    class _SP_same:
        temperature = 0.7
        top_p = 0.95
        min_p = 0.0
        top_k = 20

    class _SP_diff:
        temperature = 1.0
        top_p = 0.95
        min_p = 0.0
        top_k = 20

    a = sched._get_request_sampler(_SP())
    b = sched._get_request_sampler(_SP_same())
    c = sched._get_request_sampler(_SP_diff())

    assert a is b, "identical params must reuse cached sampler — required for fast path"
    assert a is not c, "different temperature must produce a distinct sampler"
    assert len(sched._sampler_cache) == 2


def test_sampler_cache_is_bounded_lru():
    """The cache key is request-controlled (clients pick temp/top_p), so
    the cache MUST be bounded. Without a cap, an adversarial client
    streaming unique floats grows ``_sampler_cache`` without bound for
    the process lifetime."""
    from collections import OrderedDict

    from vllm_mlx.scheduler import Scheduler

    sched = Scheduler.__new__(Scheduler)
    sched._sampler_cache = OrderedDict()
    sched._sampler_cache_max = 4  # small cap for test legibility

    class _SP:
        def __init__(self, temp):
            self.temperature = temp
            self.top_p = 0.95
            self.min_p = 0.0
            self.top_k = 20

    # Fill past the cap.
    samplers = [sched._get_request_sampler(_SP(0.1 * i)) for i in range(6)]
    assert len(sched._sampler_cache) == 4, "cap must hold even under churn"

    # Oldest entries (temp=0.0, 0.1) are evicted.
    keys = list(sched._sampler_cache.keys())
    assert (0.0, 0.95, 0.0, 20) not in keys
    assert (0.1, 0.95, 0.0, 20) not in keys
    # Newest entry is at the LRU tail.
    assert keys[-1] == (0.5, 0.95, 0.0, 20)

    # Hot-key LRU bookkeeping: hitting an existing key bumps it to MRU.
    sched._get_request_sampler(_SP(0.2))  # was middle of the LRU
    assert list(sched._sampler_cache.keys())[-1] == (0.2, 0.95, 0.0, 20)


def test_method_type_wrapper_sees_correct_self():
    """The installed patch is bound via ``types.MethodType`` — ``self``
    inside the wrapper must be the actual generation_batch instance
    (not whatever was passed at install time)."""
    shared = lambda x: x  # noqa: E731
    gb = _FakeGenBatch(samplers=[shared, shared], fallback=lambda x: x)
    _install(gb)

    assert isinstance(gb._step, types.MethodType)
    assert gb._step.__self__ is gb
