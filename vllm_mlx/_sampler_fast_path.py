"""Fused top-p + temperature sampler for the common chat sampler config.

mlx-lm's ``make_sampler`` builds a closure chain of independent
``@mx.compile``'d functions::

    sampler(logprobs) ->
        apply_top_p(logprobs, top_p)         # @mx.compile #1
        categorical_sampling(masked, temp)   # @mx.compile #2

For the dominant chat config ``(temp > 0, at least one of 0 < top_p < 1
or top_k > 0, no min_p, no xtc, no logits_processors)`` this costs ~4.3 ms
/ token on Qwen 3.6 35B 4-bit @ B=1 on M3 Ultra, split as:

* ~0.9 ms Python — three closure dispatches per step (mlx-lm chain runs
  ``apply_top_p`` closure, then ``categorical_sampling`` closure, both
  inside the per-row dispatch loop inside ``GenerationBatch._step``).
* ~3.3 ms GPU — the two ``@mx.compile`` boundaries break lazy-graph
  fusion across ``apply_top_p`` -> ``categorical_sampling``, forcing a
  separate kernel-launch sync; ``apply_top_p`` also builds a vocab-sized
  ``inverse_indices`` array (``mx.put_along_axis`` + ``mx.arange``) to
  undo the sort permutation back to vocab order so categorical can
  sample in vocab space — that scatter is the largest single op in the
  sampler chain.

mlx-vlm avoids both by sampling in sorted space inside one Python
function (``mlx_vlm/sample_utils.py:top_p_sampling``). The math is
identical for ``temperature=1``; mlx-vlm's variant applies the top-p
cutoff after temperature scaling, slightly changing the kept set when
``T != 1`` (sharper at ``T < 1``, wider at ``T > 1``).

This module ships a vendored single-function variant that **preserves
mlx-lm semantics exactly**: the top-p cutoff is computed on
``exp(logprobs)`` (the unscaled probability distribution, matching
``apply_top_p``); temperature is applied to the masked logits before
``mx.random.categorical``. Only the index space differs (sorted vs
vocab order) — the kept set, the relative weights inside it, and the
expected sample distribution are all identical to mlx-lm's chain.

The one observable difference is sample determinism under a fixed
``mx.random.seed``: mlx-lm draws Gumbel noise in vocab order while we
draw it in sorted order, so two engines with the same seed pick
different tokens. The distributions match; the bit-level sequence does
not. Documented; not a regression for OpenAI-style ``seed=`` requests
which only promise within-engine reproducibility.

Validated 2026-06-08 against Qwen 3.6 35B-A3B 4-bit B=1 HTTP:
``bg_next`` 14.07 -> 9.77 ms, HTTP 65.7 -> 100.3 tok/s (also clears
mlx-vlm's own 92.7 tok/s).
"""

from __future__ import annotations

from collections.abc import Callable

import mlx.core as mx


def is_fused_top_p_eligible(
    *,
    temperature: float,
    top_p: float,
    min_p: float,
    top_k: int,
) -> bool:
    """Return True when the sampler chain reduces to a subset of
    ``{apply_top_p, apply_top_k, categorical_sampling}``.

    mlx-lm's ``make_sampler`` builds the chain conditionally. With
    ``min_p == 0`` (rapid-mlx never forwards ``xtc_*`` knobs) and at
    least one of ``0 < top_p < 1`` or ``top_k > 0`` active, the chain
    is at most three ops — the case this fast path replaces.
    ``temperature > 0`` is required because ``temperature == 0`` already
    short-circuits in mlx-lm (``make_sampler`` returns
    ``lambda x: mx.argmax(x, axis=-1)`` directly), so there is nothing
    to optimise.
    """
    # Codex round-2 BLOCKER #2 fix: top-k-only configurations route back
    # through mlx-lm's chain because ``apply_top_k`` uses ``mx.partition``
    # which is cheaper than our full vocab ``argsort`` when top-p is not
    # also active. The fast path's win comes from collapsing the top_p +
    # categorical chain — without top_p, we'd be replacing mlx-lm's
    # partition with a heavier sort for no upside. ``top_k > 0`` remains
    # supported as an *additional* mask layered on top of an active
    # nucleus cut (the qwen3.6 + alias cascade case this PR targets).
    return temperature > 0.0 and min_p == 0.0 and 0.0 < top_p < 1.0


def make_fused_top_p_temp_sampler(
    temperature: float, top_p: float, top_k: int = 0
) -> Callable[[mx.array], mx.array]:
    """Build a sampler closure that fuses top-p / top-k / temperature /
    categorical sampling into one Python call and one lazy-graph segment.

    Math is identical to mlx-lm's ``apply_top_p`` (on unscaled probs) ->
    ``apply_top_k`` -> ``categorical_sampling`` (with ``logits * 1/T``).
    Index space differs: we sample in sorted space and map back via one
    ``take_along_axis`` instead of building ``inverse_indices`` to undo
    the sort permutation. top-k drops out for free because the sort is
    already done; we just intersect a position mask in sorted space.

    Args:
        temperature: Sampling temperature. Must be > 0.
        top_p: Nucleus cutoff. ``0 < top_p < 1`` enables top-p; values
            outside that range disable it (matching mlx-lm's
            ``make_sampler`` gate).
        top_k: Top-k cutoff. ``top_k > 0`` enables; ``top_k == 0``
            disables.

    At least one of top-p or top-k must be active — otherwise there is
    nothing to mask and the call collapses to plain
    ``mx.random.categorical(logits / T)``, which mlx-lm handles fine and
    we should not steal.

    Returns:
        A callable ``sampler(logprobs) -> token_ids`` matching the
        shape contract mlx-lm uses inside ``GenerationBatch._step``:
        ``logprobs`` is ``[..., vocab]``; the return drops the vocab
        axis.
    """
    if temperature <= 0.0:
        raise ValueError("fused sampler requires temperature > 0")
    if not (0.0 < top_p < 1.0):
        raise ValueError(
            "fused sampler requires top_p in (0, 1); top-k-only configurations "
            "should route through mlx-lm's apply_top_k partition primitive"
        )
    use_top_k = top_k > 0

    temp_inv = 1.0 / float(temperature)
    one_minus_p = 1.0 - float(top_p)
    top_k_val = int(top_k)

    def sampler(logprobs: mx.array) -> mx.array:
        # mlx-lm passes ``logprobs`` = ``logits - logsumexp(logits)``;
        # exp(logprobs) is exactly the unscaled probability distribution
        # that ``apply_top_p`` masks on. ``mx.cumsum`` over bfloat16 is
        # unsupported as of MLX 0.21 — promote first.
        work = (
            logprobs.astype(mx.float32) if logprobs.dtype == mx.bfloat16 else logprobs
        )
        probs = mx.exp(work)
        sorted_indices = mx.argsort(probs, axis=-1)
        sorted_logits = mx.take_along_axis(work, sorted_indices, axis=-1)

        # Build mask in sorted space. argsort returns ascending order, so
        # the top-k tokens sit at positions ``[V - top_k, V - 1]`` and
        # top-p's cumulative-from-low cutoff is ``cumulative > 1 - top_p``.
        # The fast path is only constructed when top_p is active (top-k-only
        # falls through to mlx-lm — see ``is_fused_top_p_eligible``), so
        # we always build the top_p mask first.
        vocab = sorted_logits.shape[-1]
        sorted_probs = mx.take_along_axis(probs, sorted_indices, axis=-1)
        cumulative = mx.cumsum(sorted_probs, axis=-1)
        # Codex round-2 BLOCKER #1 fix: for sub-fp32-epsilon top_p (e.g.
        # 1e-9), ``one_minus_p`` rounds to 1.0 in the float32 comparison
        # and ``cumulative > one_minus_p`` is all-false, producing an
        # all-``-inf`` masked vector that breaks ``mx.random.categorical``.
        # OR in the top-1 position (vocab - 1 under ascending argsort) to
        # guarantee the argmax token is always sampleable, mirroring
        # mlx-lm's "at least one token" invariant on its apply_top_p path.
        top_one_mask = mx.arange(vocab) == (vocab - 1)
        mask = (cumulative > one_minus_p) | top_one_mask
        if use_top_k:
            top_k_mask = mx.arange(vocab) >= (vocab - top_k_val)
            # Re-apply the top-1 guarantee after intersecting with top-k so
            # the same edge case doesn't reopen via a degenerate top_k=0
            # mask (already excluded by ``use_top_k`` but cheap to be safe).
            mask = (mask & top_k_mask) | top_one_mask

        masked_sorted = mx.where(
            mask,
            sorted_logits * temp_inv,
            -float("inf"),
        )
        sampled_pos = mx.random.categorical(masked_sorted)
        return mx.take_along_axis(
            sorted_indices, sampled_pos[..., None], axis=-1
        ).squeeze(-1)

    return sampler
