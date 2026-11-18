"""Per-request seeded sampler for the OpenAI ``seed`` parameter (H-11).

mlx-lm's stock sampler chain (``mlx_lm.sample_utils.make_sampler``) reads
PRNG state from the process-global ``mx.random.state``: every step
``apply_top_p`` / ``apply_top_k`` / ``apply_min_p`` /
``categorical_sampling`` thread state in/out via ``@partial(mx.compile,
inputs=mx.random.state, outputs=mx.random.state)``. The fused fast path
in ``rapid_mlx/_sampler_fast_path.py`` likewise calls
``mx.random.categorical`` with no ``key=`` argument — same global-state
dependency. Neither carries a per-request seed, so the OpenAI ``seed``
parameter on ``/v1/chat/completions`` was parsed, validated, and silently
dropped — Tomek r3's H-11 repro (five calls with ``{temperature: 0.7,
seed: 42}`` → five different outputs).

This module ships a sampler factory that **threads a per-request PRNG
key explicitly** via ``mx.random.categorical(..., key=...)``. State lives
inside the closure as a single-element list holding the current
``mx.random.key``; each call ``mx.random.split`` consumes one half for
the active step and stows the other for the next. Two consequences flow
from this design:

* **Concurrency safety.** Every seeded request carries its own key
  state; there is no read/write of ``mx.random.state``. Interleaved
  calls from different requests in the same multi-row batch (scheduler
  per-row dispatch in ``scheduler.py::_mtp_step``) can't corrupt each
  other's sequences. Verified by an isolated micro-test before the
  scheduler integration landed — two seeded samplers run interleaved
  produce the same sequences they produce in isolation.

* **No cache eligibility.** The scheduler's sampler cache
  (``Scheduler._get_request_sampler``) interns samplers by
  ``(temp, top_p, min_p, top_k)`` so identity-equality on
  ``GenerationBatch.samplers`` can detect homogeneous batches and engage
  the dense-sampler fast path (``_install_dense_sampler_fastpath``).
  Seeded samplers MUST NOT be interned — they carry mutable closure
  state and two requests sharing the same closure would observe each
  other's keys. The caller side (``Scheduler._get_request_sampler``)
  routes ``sampling_params.seed`` requests around the cache.

Math: matches mlx-lm's ``make_sampler`` semantically (top-p first on
unscaled probs, then top-k, then min-p mask, then temperature scaling,
then ``mx.random.categorical``). Bit-level outputs do not match mlx-lm's
chain for the same ``mx.random.seed`` because we sample with an
explicit key instead of through ``categorical_sampling``'s
``@mx.compile`` boundary — but reproducibility is **within-engine** as
the OpenAI spec promises ("we cannot guarantee determinism across model
versions or backends"). Two calls to rapid-mlx with the same
``(seed, temperature, top_p, top_k, min_p, prompt)`` produce the same
token stream; that's the contract H-11 makes good on.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

import mlx.core as mx


class RequestSeededRNG:
    """Generation-thread-owned carried key for one seeded request."""

    __slots__ = ("_key", "_draws")

    def __init__(self, seed: int):
        self._key = mx.random.key(int(seed))
        self._draws = 0

    @property
    def key(self) -> mx.array:
        return self._key

    @property
    def draws(self) -> int:
        return self._draws

    def next_key(self) -> mx.array:
        # Preserve the sampler's historical split contract exactly: consume
        # the first child and carry the second child into the next draw.
        subkey, self._key = mx.random.split(self._key)
        self._draws += 1
        return subkey


def _apply_argmax_rescue(mask: mx.array, argmax_idx: mx.array) -> mx.array:
    """Apply the round-7 conditional argmax rescue to a combined mask.

    Factored out as a module-level helper so the rescue's intersection-
    preserving contract can be unit-tested directly without driving
    the whole sampler closure. Codex round-8 BLOCKING regression guard:
    the prior round-7 sampler test only proved same-seed determinism
    and would pass under the OLD unconditional ``mask | argmax_keep``
    behaviour, so the test was vacuous. Exposing the rescue as a pure
    helper lets the test inject a hand-built non-empty mask that
    excludes ``argmax_idx`` and assert the rescue does NOT add argmax
    back — the property the round-7 fix exists to enforce.

    Args:
        mask: Bool tensor shaped ``[..., vocab]`` — the combined
            kept-token mask after applying top-p / top-k / min-p.
        argmax_idx: Int tensor shaped ``[..., 1]`` — the per-row
            argmax positions (``mx.argmax(work, axis=-1, keepdims=True)``).

    Returns:
        Bool tensor shaped ``[..., vocab]`` with the round-2 invariant:
        each row has at least one ``True`` entry, and rows that already
        had at least one ``True`` entry are returned UNCHANGED (no
        argmax injected). Rows that were all ``False`` fall back to a
        single-True at ``argmax_idx``.

    The conditional gate ``mx.where(any_kept, mask, argmax_keep)`` is
    what makes the contract observable: under the old unconditional
    form (``mask | argmax_keep``), a non-empty row that excluded
    argmax would still have argmax OR'd in — violating ``top_k``
    intersection semantics.
    """
    # Build the single-True argmax mask in vocab order.
    argmax_keep = mx.zeros(mask.shape, dtype=mx.bool_)
    argmax_keep = mx.put_along_axis(
        argmax_keep,
        argmax_idx,
        mx.ones(argmax_idx.shape, dtype=mx.bool_),
        axis=-1,
    )
    # Per-row "any token kept after intersection?" detector. ``keepdims=True``
    # gives shape ``[..., 1]`` which broadcasts against ``[..., vocab]``
    # under ``mx.where``.
    any_kept = mx.any(mask, axis=-1, keepdims=True)
    return mx.where(any_kept, mask, argmax_keep)


def _make_seeded_logits_filter(
    *,
    temperature: float,
    top_p: float = 0.0,
    min_p: float = 0.0,
    top_k: int = 0,
    min_tokens_to_keep: int = 1,
    xtc_probability: float = 0.0,
    xtc_threshold: float = 0.0,
    xtc_special_tokens: list[int] | None = None,
) -> Callable[..., mx.array]:
    """Share seeded serving's masks and temperature with speculative sampling.

    XTC runs on the retained distribution before temperature, preserves the
    least probable candidate above its threshold, and exempts special tokens.
    Its caller supplies a request-owned draw so draft and verify can reuse it.
    Minimum token retention applies only to min-p, preserving other masks.
    """
    if min_p > 0 and (
        not isinstance(min_tokens_to_keep, int) or min_tokens_to_keep < 1
    ):
        raise ValueError("min_tokens_to_keep must be a positive integer")
    if xtc_probability > 0 and not (xtc_probability <= 1 and 0 <= xtc_threshold <= 0.5):
        raise ValueError("XTC requires probability in [0, 1] and threshold in [0, 0.5]")
    temp_inv = 1.0 / float(temperature)
    use_top_p = 0.0 < top_p < 1.0
    use_top_k = top_k > 0
    use_min_p = min_p > 0.0
    top_p_threshold = 1.0 - float(top_p)
    min_p_val = float(min_p)
    top_k_val = int(top_k)

    def filter_logits(
        logprobs: mx.array, *, xtc_draw: mx.array | None = None
    ) -> mx.array:
        work = logprobs.astype(mx.float32) if logprobs.dtype != mx.float32 else logprobs
        vocab = work.shape[-1]

        mask = mx.ones(work.shape, dtype=mx.bool_)

        if use_top_p:
            probs = mx.exp(work)
            sorted_indices = mx.argsort(probs, axis=-1)
            sorted_probs = mx.take_along_axis(probs, sorted_indices, axis=-1)
            cumulative = mx.cumsum(sorted_probs, axis=-1)
            sorted_mask = cumulative > top_p_threshold
            top_one = mx.arange(vocab) == (vocab - 1)
            sorted_mask = sorted_mask | top_one
            vocab_mask = mx.zeros(work.shape, dtype=mx.bool_)
            vocab_mask = mx.put_along_axis(
                vocab_mask, sorted_indices, sorted_mask, axis=-1
            )
            mask = mask & vocab_mask

        if use_top_k:
            effective_top_k = min(top_k_val, vocab)
            sorted_desc = mx.argsort(-work, axis=-1)
            top_k_positions = sorted_desc[..., :effective_top_k]
            vocab_mask_k = mx.zeros(work.shape, dtype=mx.bool_)
            ones = mx.ones(top_k_positions.shape, dtype=mx.bool_)
            vocab_mask_k = mx.put_along_axis(
                vocab_mask_k, top_k_positions, ones, axis=-1
            )
            mask = mask & vocab_mask_k

        if use_min_p:
            probs_for_min = mx.exp(work)
            max_prob = mx.max(probs_for_min, axis=-1, keepdims=True)
            min_p_mask = probs_for_min >= (min_p_val * max_prob)
            if min_tokens_to_keep > 1:
                positions = mx.argsort(-work, axis=-1)[..., :min_tokens_to_keep]
                min_p_mask = mx.put_along_axis(
                    min_p_mask,
                    positions,
                    mx.ones(positions.shape, dtype=mx.bool_),
                    axis=-1,
                )
            mask = mask & min_p_mask

        argmax_idx = mx.argmax(work, axis=-1, keepdims=True)
        mask = _apply_argmax_rescue(mask, argmax_idx)

        if xtc_probability > 0:
            if xtc_draw is None:
                raise ValueError("seeded XTC requires a request-owned draw")
            filtered = mx.where(mask, work, -mx.inf)
            probs = mx.softmax(filtered, axis=-1)
            floor = mx.min(
                mx.where(probs > xtc_threshold, probs, mx.inf), axis=-1, keepdims=True
            )
            remove = probs > floor
            if xtc_special_tokens:
                remove[..., xtc_special_tokens] = False
            mask = mask & ((xtc_draw > xtc_probability) | ~remove)

        masked = mx.where(mask, work * temp_inv, -mx.inf)

        return masked

    return filter_logits


def make_seeded_sampler(
    *,
    seed: int,
    temperature: float,
    top_p: float = 0.0,
    min_p: float = 0.0,
    top_k: int = 0,
    min_tokens_to_keep: int = 1,
    xtc_probability: float = 0.0,
    xtc_threshold: float = 0.0,
    xtc_special_tokens: list[int] | None = None,
) -> Callable[[mx.array], mx.array]:
    """Build a per-request sampler that threads an explicit PRNG key.

    Args:
        seed: PRNG seed. Used once at construction to derive the initial
            ``mx.random.key``; subsequent calls consume / refresh the key
            via ``mx.random.split``.
        temperature: Sampling temperature. ``temperature == 0.0`` returns
            a greedy sampler (argmax) — seed is irrelevant for argmax,
            so callers SHOULD route greedy requests around this factory
            and use ``mx.argmax`` directly. We still accept ``0.0`` here
            so the scheduler doesn't need a separate branch.
        top_p: Nucleus cutoff in ``[0, 1]``. ``0.0`` (default) disables.
        min_p: Min-p cutoff in ``[0, 1]``. ``0.0`` (default) disables.
        top_k: Top-k cutoff. ``0`` (default) disables.
        min_tokens_to_keep: Minimum number retained by the min-p mask.
        xtc_probability: Probability of excluding high-probability candidates.
        xtc_threshold: Probability floor for XTC candidates before temperature.
        xtc_special_tokens: Token IDs exempt from XTC exclusion.

    Returns:
        ``sampler(logprobs) -> token_ids`` matching the shape contract
        the scheduler's per-row dispatch expects: ``logprobs`` is
        ``[..., vocab]`` (already log-softmaxed), output drops the
        vocab axis.

    Reproducibility contract: two distinct callable instances built with
    the same ``(seed, temperature, top_p, min_p, top_k)`` and fed the
    same ``logprobs`` sequence produce the same token sequence. State
    is local to each closure; concurrent use of two callables (different
    seeds or same seed) does not cross-contaminate.
    """
    if temperature == 0.0:
        # Greedy short-circuit — argmax is deterministic, no PRNG needed.
        # Seeded requests with ``temperature=0`` still get a valid (just
        # trivially deterministic) sampler so callers don't have to branch.
        def greedy(logprobs: mx.array) -> mx.array:
            return mx.argmax(logprobs, axis=-1)

        greedy_with_state: Any = greedy
        greedy_with_state.lane_rng = None
        greedy_with_state.request_seeded = True
        return greedy

    # ``temperature == 0`` is already handled by the greedy short-circuit
    # above; reaching this branch with ``<= 0`` means the caller passed a
    # negative value (API layer rejects this via Field bounds, but a direct
    # engine caller bypassing the schema must still get a coherent error).
    if temperature < 0.0:
        raise ValueError(
            f"seeded sampler requires temperature >= 0 (got {temperature})"
        )

    filter_logits = _make_seeded_logits_filter(
        temperature=temperature,
        top_p=top_p,
        min_p=min_p,
        top_k=top_k,
        min_tokens_to_keep=min_tokens_to_keep,
        xtc_probability=xtc_probability,
        xtc_threshold=xtc_threshold,
        xtc_special_tokens=xtc_special_tokens,
    )

    # Per-request key state lives in a one-element list so the closure
    # can rebind it (Python doesn't expose ``nonlocal`` on assignment
    # without an enclosing function-local binding; the list sidesteps
    # that without adding ``nonlocal`` boilerplate).
    #
    # Codex round-6 BLOCKING fix: the API layer now accepts the full
    # OpenAI-documented integer range (no ``Field(ge=, le=)`` bound on
    # ``seed``) to match clients that pass 64-bit or negative integer
    # seeds. mlx-core's ``mx.random.key`` requires a non-negative
    # ``uint32``, so we fold the public seed value to the backend's
    # PRNG key range HERE — once, at sampler construction. The fold
    # is purely deterministic (``seed & 0xFFFFFFFF``), which means:
    #
    #   * Same input seed always maps to the same backend key, so
    #     reproducibility is preserved within rapid-mlx. (OpenAI's spec
    #     only promises within-engine determinism — they explicitly
    #     warn "we cannot guarantee determinism across model versions
    #     or backends".)
    #   * Negative seeds work via Python's well-defined ``int.__and__``
    #     semantics on negatives (conceptually two's complement with
    #     an infinite sign bit), so e.g. ``-1 & 0xFFFFFFFF == 0xFFFFFFFF``
    #     rather than raising or silently truncating to garbage.
    #   * Two callers passing seeds that happen to fold to the same
    #     uint32 (e.g. ``42`` and ``42 + 2**32``) get the same
    #     sequence — this is an honest consequence of the backend's
    #     32-bit key space and matches the silent narrowing JAX /
    #     mlx-lm would do internally anyway.
    seed_uint32 = int(seed) & 0xFFFFFFFF
    lane_rng = RequestSeededRNG(seed_uint32)

    # Codex round-5 BLOCKING #2 defensive belt: serialize key-state
    # advancement so two concurrent callers of THE SAME closure can't
    # race on the ``state[0] = next_key`` write.
    #
    # In the current scheduler architecture this can't actually happen
    # — each request gets its own closure via
    # ``Scheduler._get_request_sampler`` (seeded path bypasses the
    # interning cache; see scheduler.py for the rationale block) and
    # the per-row dispatch loop in ``GenerationBatch._step`` /
    # ``_mtp_step`` invokes samplers sequentially on the mlx-step
    # thread. The only way two concurrent calls reach the same closure
    # is if a future change (a) shares one seeded closure across rows
    # or (b) introduces threaded sampler dispatch. Defending here costs
    # one uncontended ``Lock.acquire`` per step (~50 ns on M-series),
    # so the price of removing the corner case as a footgun for future
    # refactors is negligible compared to the multi-millisecond per-step
    # cost we already pay on the rest of the chain. Codex r5 flagged
    # this as BLOCKING; the lock makes the contract explicit and
    # closes the hypothetical race.
    state_lock = threading.Lock()

    def sampler(logprobs: mx.array) -> mx.array:
        # Pull a fresh subkey for this step. Held under the per-closure
        # lock so the read-split-write is atomic — two concurrent
        # callers can't both read the same carried key and then both
        # advance it (which would reuse one subkey and skip the other).
        # See the ``state_lock`` block above for why this is defensive
        # rather than load-bearing under the current scheduler.
        with state_lock:
            xtc_draw = (
                mx.random.uniform(key=lane_rng.next_key())
                if xtc_probability > 0
                else None
            )
            key = lane_rng.next_key()
        masked = filter_logits(logprobs, xtc_draw=xtc_draw)
        return mx.random.categorical(masked, key=key)

    # The priming sample and the speculative verifier execute on the same
    # scheduler-owned generation thread. Expose the carried key object so a
    # B=1 self-MTP request continues after the priming draw rather than
    # re-deriving the public seed and replaying draw zero.
    sampler_with_state: Any = sampler
    sampler_with_state.lane_rng = lane_rng
    sampler_with_state.request_seeded = True

    return sampler
