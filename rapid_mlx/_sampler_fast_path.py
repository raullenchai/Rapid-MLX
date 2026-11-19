"""Fused top-p + temperature sampler for the common chat sampler config.

mlx-lm's ``make_sampler`` builds a closure chain of independent
``@mx.compile``'d functions::

    sampler(logprobs) ->
        apply_top_p(logprobs, top_p)         # @mx.compile #1
        categorical_sampling(masked, temp)   # @mx.compile #2

For the dominant chat config ``(temp > 0, 0 < top_p < 1, no min_p, no
xtc, no logits_processors)`` (with optional ``top_k > 0`` layered on top
of the active nucleus cut) this costs ~4.3 ms / token on Qwen 3.6 35B
4-bit @ B=1 on M3 Ultra, split as:

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
the unscaled top-p rule**: the top-p cutoff is computed on
``exp(logprobs)`` (the unscaled probability distribution, matching
``apply_top_p``); temperature is applied to the masked logits before
``mx.random.categorical``. Only the index space differs (sorted vs
vocab order). Untied inputs preserve the kept set and relative weights;
cutoff ties follow the sorted-position rule documented below.

The observable differences from the fallback chain are:

1. Sample determinism under a fixed ``mx.random.seed``: mlx-lm draws
   Gumbel noise in vocab order while we draw it in sorted order, so
   two engines with the same seed pick different tokens. The
   distributions match; the bit-level sequence does not. Not a
   regression for OpenAI-style ``seed=`` requests which only promise
   within-engine reproducibility.

2. Top-k ties use the stable ascending sort order: higher token IDs
   occupy the retained tail at a tied cutoff. The partition-based
   fallback can retain different IDs, which changes token support.
   MTP shares this fused mask whenever plain serving selects it.

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
    """Return True when the sampler chain reduces to ``apply_top_p`` plus
    an optional ``apply_top_k`` plus ``categorical_sampling``, and the
    fused replacement is expected to win against mlx-lm.

    Eligible iff ``temperature > 0`` AND ``min_p == 0`` AND
    ``0 < top_p < 1`` (with ``top_k`` optional — if set, layered as an
    additional mask on top of the active nucleus cut).

    Top-k-only configurations are explicitly NOT eligible: mlx-lm's
    ``apply_top_k`` uses ``mx.partition`` which is cheaper than our
    full vocab ``argsort`` when top-p isn't also active, so swapping in
    the fused path would be a regression there. ``temperature == 0`` is
    also not eligible because mlx-lm already short-circuits to
    ``argmax`` directly — nothing to optimise.
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
        top_p: Nucleus cutoff. Must be in ``(0, 1)`` — the fast path
            requires an active nucleus cut. ``top_p == 1`` short-circuits
            in mlx-lm (no mask to apply); ``top_p == 0`` is degenerate.
        top_k: Top-k cutoff. ``top_k > 0`` enables the additional mask
            layered on top of top-p; ``top_k == 0`` (default) leaves
            only top-p active.

    Active top-p is required — top-k-only configurations should route
    through mlx-lm's ``apply_top_k`` (``mx.partition``-based) primitive
    because that's cheaper than our full vocab ``argsort`` when nucleus
    isn't also active. See ``is_fused_top_p_eligible``.

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
    temp_inv = 1.0 / float(temperature)
    one_minus_p = 1.0 - float(top_p)
    top_k_val = int(top_k)

    def sampler(logprobs: mx.array) -> mx.array:
        masked_sorted, sorted_indices = _fused_top_p_logits(
            logprobs, temp_inv, one_minus_p, top_k_val
        )
        sampled_pos = mx.random.categorical(masked_sorted)
        return mx.take_along_axis(
            sorted_indices, sampled_pos[..., None], axis=-1
        ).squeeze(-1)

    return sampler


def _fused_top_p_logits(
    logprobs: mx.array, temp_inv: float, one_minus_p: float, top_k: int
) -> tuple[mx.array, mx.array]:
    """Return temperature-scaled, filtered logits and token IDs in sorted space.

    Both serving and MTP use this mask. Promote the already-normalized input
    before exp/cumsum, retain at least the final sorted position, and intersect
    top-k by position. Stable ascending ties retain the higher token IDs.
    """
    work = logprobs.astype(mx.float32) if logprobs.dtype != mx.float32 else logprobs
    probs = mx.exp(work)
    sorted_indices = mx.argsort(probs, axis=-1)
    sorted_logits = mx.take_along_axis(work, sorted_indices, axis=-1)
    vocab = sorted_logits.shape[-1]
    sorted_probs = mx.take_along_axis(probs, sorted_indices, axis=-1)
    cumulative = mx.cumsum(sorted_probs, axis=-1)
    top_one_mask = mx.arange(vocab) == (vocab - 1)
    mask = (cumulative > one_minus_p) | top_one_mask
    if top_k > 0:
        top_k_mask = mx.arange(vocab) >= (vocab - top_k)
        mask = (mask & top_k_mask) | top_one_mask
    return mx.where(mask, sorted_logits * temp_inv, -float("inf")), sorted_indices
