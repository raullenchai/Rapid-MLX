"""Per-request seeded sampler for the OpenAI ``seed`` parameter (H-11).

mlx-lm's stock sampler chain (``mlx_lm.sample_utils.make_sampler``) reads
PRNG state from the process-global ``mx.random.state``: every step
``apply_top_p`` / ``apply_top_k`` / ``apply_min_p`` /
``categorical_sampling`` thread state in/out via ``@partial(mx.compile,
inputs=mx.random.state, outputs=mx.random.state)``. The fused fast path
in ``vllm_mlx/_sampler_fast_path.py`` likewise calls
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

import mlx.core as mx


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


def make_seeded_sampler(
    *,
    seed: int,
    temperature: float,
    top_p: float = 0.0,
    min_p: float = 0.0,
    top_k: int = 0,
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

        return greedy

    # ``temperature == 0`` is already handled by the greedy short-circuit
    # above; reaching this branch with ``<= 0`` means the caller passed a
    # negative value (API layer rejects this via Field bounds, but a direct
    # engine caller bypassing the schema must still get a coherent error).
    if temperature < 0.0:
        raise ValueError(
            f"seeded sampler requires temperature >= 0 (got {temperature})"
        )

    temp_inv = 1.0 / float(temperature)
    use_top_p = 0.0 < top_p < 1.0
    use_top_k = top_k > 0
    use_min_p = min_p > 0.0
    top_p_threshold = 1.0 - float(top_p)
    min_p_val = float(min_p)
    top_k_val = int(top_k)

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
    state = [mx.random.key(seed_uint32)]

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
        # Promote bfloat16/float16 to float32 — same rationale as the
        # fused fast path: half precision rounds the top-p cutoff away
        # on production logits and ``mx.cumsum`` over bfloat16 is
        # unsupported in MLX 0.21.
        work = logprobs.astype(mx.float32) if logprobs.dtype != mx.float32 else logprobs
        vocab = work.shape[-1]

        # Build the kept-token mask in VOCAB order so we can sample
        # in vocab space and return the token id directly. This is
        # different from the fused fast path (which samples in sorted
        # space to fuse the scatter); the seeded path optimises for
        # clarity + math parity with mlx-lm, not for tok/s — seeded
        # requests are an eval / snapshot use case where 1-3 ms / token
        # is acceptable.
        mask = mx.ones(work.shape, dtype=mx.bool_)

        if use_top_p:
            # Top-p: sort probs ascending, find cumulative-from-low
            # cutoff, OR-back the top-1 (argmax) to guarantee at least
            # one sampleable token (mirrors mlx-lm's invariant and
            # the fast path's ``top_one_mask`` belt). Re-scatter the
            # sorted mask back into vocab order.
            #
            # Boundary semantics: ``cumulative > 1 - top_p`` is a
            # STRICT comparison and matches both
            # ``mlx_lm.sample_utils.apply_top_p`` (which uses
            # ``cumulative_probs > 1 - top_p`` — see mlx-lm
            # 0.31.3 sample_utils.py:222) and the fused fast path in
            # ``_sampler_fast_path.py``. Codex round-5 raised a BLOCKING
            # claim that this should be ``>=`` so a token whose
            # cumulative prob is exactly the cutoff is kept; that would
            # actually DIVERGE from mlx-lm's contract (and from the
            # HuggingFace reference apply_top_p the upstream points to).
            # The boundary token is recovered via the OR-with-argmax
            # rescue at the bottom of this function on the rare cases
            # where the strict cut alone would drop the head of the
            # nucleus — matching mlx-lm's documented "at least one
            # token kept" invariant. Bit-level parity with the stock
            # sampler is the priority for the seeded path because the
            # reproducibility contract advertises within-engine
            # determinism.
            probs = mx.exp(work)
            sorted_indices = mx.argsort(probs, axis=-1)
            sorted_probs = mx.take_along_axis(probs, sorted_indices, axis=-1)
            cumulative = mx.cumsum(sorted_probs, axis=-1)
            sorted_mask = cumulative > top_p_threshold
            # Guarantee top-1 (last position under ascending sort)
            top_one = mx.arange(vocab) == (vocab - 1)
            sorted_mask = sorted_mask | top_one
            # Scatter back to vocab order
            vocab_mask = mx.zeros(work.shape, dtype=mx.bool_)
            vocab_mask = mx.put_along_axis(
                vocab_mask, sorted_indices, sorted_mask, axis=-1
            )
            mask = mask & vocab_mask

        if use_top_k:
            # Top-k: keep the ``top_k`` largest logits. Use argsort
            # descending then mark positions <= top_k. Mirrors the
            # fast path's intersection semantics — kept-set size is
            # exactly ``min(top_k, vocab)``.
            #
            # Codex round-2 BLOCKING #1 fix: clamp ``top_k_val`` to
            # the vocab dimension. ``sorted_desc[..., :top_k_val]``
            # with ``top_k_val > vocab`` silently returns the whole
            # tensor (NumPy/MLX over-slice semantics), but the
            # subsequent ``put_along_axis`` would then write
            # ``vocab + something`` positions into a shape that only
            # has ``vocab`` slots — undefined behaviour in MLX. The
            # clamp normalises the contract so ``top_k=10**9`` on a
            # 32k-vocab model collapses to "all tokens eligible"
            # rather than crashing the sampler.
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
            # Min-p: keep tokens whose prob >= min_p * max_prob.
            # Mirrors mlx-lm's ``apply_min_p`` formula.
            probs_for_min = mx.exp(work)
            max_prob = mx.max(probs_for_min, axis=-1, keepdims=True)
            min_p_mask = probs_for_min >= (min_p_val * max_prob)
            mask = mask & min_p_mask

        # Codex round-2 BLOCKING #2 + round-7 BLOCKING fix: guarantee
        # at least one sampleable token WITHOUT changing the intersection
        # semantics for non-empty rows.
        #
        # The original round-2 fix unconditionally OR'd argmax into the
        # combined mask, which fixed the empty-row crash but silently
        # broke ``top_k`` when combined with a tight ``top_p`` / ``min_p``
        # that excluded argmax from the top-k set — the rescue would
        # re-introduce a token the caller's intersection contract had
        # already filtered out. Codex r7 caught this: ``top_k`` no
        # longer means "sample only from the top K" when layered with
        # a stricter cutoff.
        #
        # The conditional form below preserves the round-2 contract
        # (no all-``-inf`` rows ever reach ``mx.random.categorical``)
        # while only invoking the argmax rescue when the row would
        # otherwise be empty. Non-empty rows keep their exact mask,
        # so ``top_k`` / ``top_p`` / ``min_p`` intersection semantics
        # match the documented behaviour and the fused fast path.
        #
        # Empty-row case (combined cutoffs filtered every token): fall
        # back to argmax so the sampler returns a sensible token
        # rather than uint32 garbage from a degenerate ``-inf``
        # distribution. This matches mlx-lm's "at least one token
        # kept" invariant on the individual ``apply_top_p`` /
        # ``apply_top_k`` / ``apply_min_p`` masks — each preserves a
        # token by construction, so empty rows can only arise from
        # the combined intersection, which is exactly the case the
        # rescue targets.
        argmax_idx = mx.argmax(work, axis=-1, keepdims=True)
        mask = _apply_argmax_rescue(mask, argmax_idx)

        # Apply mask + temperature scaling. Use ``-inf`` for masked
        # positions so categorical never picks them.
        masked = mx.where(mask, work * temp_inv, -mx.inf)

        # Pull a fresh subkey for this step. Held under the per-closure
        # lock so the read-split-write is atomic — two concurrent
        # callers can't both read the same ``state[0]`` and then both
        # advance it (which would reuse one subkey and skip the other).
        # See the ``state_lock`` block above for why this is defensive
        # rather than load-bearing under the current scheduler.
        with state_lock:
            cur_key, next_key = mx.random.split(state[0])
            state[0] = next_key
        return mx.random.categorical(masked, key=cur_key)

    return sampler
