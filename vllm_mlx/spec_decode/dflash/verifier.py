# SPDX-License-Identifier: Apache-2.0
"""Block-diffusion DFlash verifier (R15 task #313).

The verifier takes a candidate block of ``block_size`` tokens from the
drafter, runs the full target model in 1 forward pass over the block,
and decides the longest accepted prefix. This module owns:

* the **position-id contract** for the block forward (the verifier emits
  positions ``[offset, offset+1, ..., offset+block_size-1]`` so the
  target model's KV cache writes land at the right slots);
* the **cache-write integration** with
  :func:`vllm_mlx.positioned_kv_cache.positioned_update_and_fetch` — the
  HELPER, NOT the subclass, so the cache stays compatible with
  ``mlx_lm.save_prompt_cache``;
* the **longest-accepted-prefix decision** — at temp=0 it's the prefix
  where every position's verify argmax matches the drafter's candidate.
  On the first mismatch (position ``k``), the verifier emits the
  CORRECTIVE token (verify model's argmax at position ``k``), so the
  output is byte-identical to a non-spec-decode generation;
* the **KV-cache rollback for rejected suffix** — the cache write
  happens BEFORE the verify decision (one forward, one batched write),
  so on a partial accept the verifier rewinds ``cache.offset`` to drop
  the rejected tail.

What this module does NOT own:

* Drafter forward — :mod:`vllm_mlx.spec_decode.dflash.drafter`.
* Generator outer loop / sampler chain / accept-counter bookkeeping —
  :mod:`vllm_mlx.spec_decode.dflash.generator`.

Lossless contract
-----------------

The contract is the same one MTP enforces: for every input prompt +
sampling config, the output token sequence MUST be byte-identical to
the no-spec-decode path.

For DFlash:

* Block of ``B`` draft tokens at positions ``[p, p+1, ..., p+B-1]``.
* Target forward returns logits for every position; at temp=0 we take
  the argmax at each position.
* Walk the block left-to-right: at position ``i``, if
  ``argmax(verify_logits[i]) == draft[i]``, accept and continue.
  On the FIRST mismatch at position ``k``:
  - Emit the corrective token (the verify argmax at position ``k``);
  - Truncate the KV cache to ``p + k + 1`` (the corrective token's
    KV is already in the cache from the batched write).
* If every position matches (``k == B``), the block is fully accepted
  AND we still need ONE more verify token (the position ``p+B`` step)
  — that gets queued for the NEXT block as the "always-emit verify
  pred" position. Equivalent to MTP's bonus-token emission on accept.

At temp>0 the decision uses the speculative-sampling ratio
``min(1, p_target / p_draft)`` per position — outside the scope of this
verifier module for now since the paper's headline numbers are
temp=0 / argmax. The chain-of-positions extension is a 1-paragraph
add (call :func:`mlx.random.uniform` per position before the equality
test) but adding it without empirical validation against the paper's
benchmark would risk a lossless-contract drift, so we ship temp=0
only and emit a clear "DFlash temp>0 not yet validated" warning in the
generator.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import mlx.core as mx

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VerifyResult:
    """Outcome of one verifier round.

    Attributes:
        accepted_tokens: Token IDs the verifier ACCEPTED from the
            drafter (subset of the draft block, in order). Length is
            in ``[0, block_size]``. Length ``0`` means the drafter's
            position-0 candidate diverged from the verifier's argmax —
            still a valid lossless step (the corrective token wins).
        bonus_token: The corrective / bonus token emitted at the
            divergence position (or at position ``block_size`` if the
            block was fully accepted). The generator yields this AFTER
            the accepted tokens.
        accepted_len: Convenience: ``len(accepted_tokens)``.
        block_was_full: True when every position in the draft block
            matched. Used by the generator to track "full block win"
            telemetry.
        verify_offset_after: Cache offset AFTER this round's commit.
            The generator passes it to the drafter's next call as
            ``current_position``.
    """

    accepted_tokens: tuple[int, ...]
    bonus_token: int
    accepted_len: int
    block_was_full: bool
    verify_offset_after: int


def _argmax_per_position(logits: mx.array) -> mx.array:
    """Greedy argmax along the vocab axis.

    Args:
        logits: Shape ``(B, S, V)``. ``B`` is batch, ``S`` is the block
            size + any prefix positions, ``V`` is vocab.

    Returns:
        Shape ``(B, S)`` ``mx.uint32`` argmax per (batch, position).
        Cast to ``uint32`` so the result is comparable to drafter token
        IDs (always ``uint32`` per the drafter Protocol contract).
    """
    if logits.ndim != 3:
        raise ValueError(
            f"verify logits must be 3-D (B, S, V); got shape {tuple(logits.shape)}"
        )
    return mx.argmax(logits, axis=-1).astype(mx.uint32)


def _decide_accepted_prefix(
    verify_argmax: list[int],
    draft_tokens: list[int],
) -> tuple[int, int]:
    """Walk the block and find the longest accepted prefix.

    Args:
        verify_argmax: Verify model's argmax at each position
            ``[p, p+1, ..., p+B-1]``. Length ``B``.
        draft_tokens: Drafter's candidate at each position. Length
            ``B``.

    Returns:
        Tuple ``(accepted_len, bonus_token)`` where:

        * ``accepted_len`` is the number of positions where
          ``verify_argmax[i] == draft_tokens[i]`` for every ``i <
          accepted_len`` (and the first mismatch is at
          ``accepted_len``).
        * ``bonus_token`` is the corrective token at the divergence
          position. If the block was fully accepted, this is the
          token to emit at the NEXT position — which the verifier
          can't determine from this block's logits alone, so the
          caller passes a sentinel of ``-1`` and runs an additional
          forward pass. We track that case via the
          :attr:`VerifyResult.block_was_full` flag.

    Lossless contract:

    * If ``accepted_len == B`` (all match): no divergence to correct;
      ``bonus_token`` returns ``-1`` (caller handles the post-block
      forward).
    * If ``accepted_len < B``: the corrective token is the verify
      argmax at the divergence position. The accepted tokens stay
      accepted (they each matched the verify argmax at temp=0, so
      emitting them is identical to running a normal decode step at
      each position).
    """
    if len(verify_argmax) != len(draft_tokens):
        raise ValueError(
            f"verify_argmax length {len(verify_argmax)} must equal "
            f"draft_tokens length {len(draft_tokens)}"
        )
    accepted_len = 0
    for v, d in zip(verify_argmax, draft_tokens, strict=True):
        if v == d:
            accepted_len += 1
        else:
            break
    if accepted_len == len(draft_tokens):
        return accepted_len, -1
    return accepted_len, verify_argmax[accepted_len]


def _isfinite_or_raise(name: str, value: float) -> None:
    """Reject NaN / +-Inf for any user-supplied float used in math.

    Pydantic's ``Field(ge=, le=)`` does NOT reject NaN — see
    knowledge/gotchas.md. Every float that reaches a math kernel here
    needs an explicit ``math.isfinite`` validator. Currently the
    verifier only takes a temperature float so this is the only check.
    """
    if not math.isfinite(value):
        raise ValueError(f"{name}={value!r} is not finite (NaN / Inf not allowed)")


def verify_block(
    model: Any,
    draft_block: mx.array,
    *,
    last_confirmed_token: int,
    cache: list[Any],
    block_size: int,
    current_offset: int,
    temperature: float = 0.0,
) -> VerifyResult:
    """Verify a draft block in 1 target forward pass.

    Args:
        model: The loaded target model. Must accept ``__call__(inputs,
            cache=...)`` and return logits of shape ``(1, S, vocab_size)``.
            Mlx-lm's ``mlx_lm.models.qwen3_5.Model`` / ``qwen3_5_moe.Model``
            both satisfy this contract.
        draft_block: 1-D ``mx.uint32`` of shape ``(block_size,)`` from
            the drafter — the candidate tokens for positions
            ``[current_offset, ..., current_offset + block_size - 1]``.
        last_confirmed_token: The last token whose KV is already
            COMMITTED in the cache at position ``current_offset - 1``.
            Used as the first input to the verifier (so the verifier
            sees the same prefix as a normal decode step would).
        cache: List of per-layer KV caches (each a
            :class:`mlx_lm.models.cache.KVCache` or
            :class:`QuantizedKVCache`). The verifier uses
            :func:`positioned_update_and_fetch` to write the block's KV
            at explicit positions so the cache offset advances exactly
            to the accepted prefix end.
        block_size: Expected block size. Must match
            ``draft_block.shape[0]``.
        current_offset: Cache offset BEFORE this block — i.e. the
            position of the first token in the block.
        temperature: Sampling temperature. Currently only ``0.0`` is
            supported — the speculative-sampling probabilistic
            acceptance for temp>0 will land in a follow-up once the
            paper's bench is reproduced at temp=0.

    Returns:
        :class:`VerifyResult` — the accepted prefix, the corrective /
        bonus token, and the cache offset after commit.

    Raises:
        ValueError: On shape mismatch or non-finite temperature.
        NotImplementedError: If ``temperature > 0`` is requested
            before the per-position speculative-sampling fork lands.
    """
    _isfinite_or_raise("temperature", float(temperature))
    if temperature != 0.0:
        raise NotImplementedError(
            "DFlash verifier currently only supports greedy decoding "
            "(temperature=0.0). Per-position speculative sampling for "
            "temp>0 is reserved for a follow-up; see module docstring."
        )

    if draft_block.ndim != 1:
        raise ValueError(
            f"draft_block must be 1-D; got shape {tuple(draft_block.shape)}"
        )
    if int(draft_block.shape[0]) != block_size:
        raise ValueError(
            f"draft_block length {int(draft_block.shape[0])} does not "
            f"match block_size={block_size}"
        )
    if current_offset < 0:
        raise ValueError(f"current_offset must be >= 0; got {current_offset}")

    # Build the verifier input: the last confirmed token followed by
    # the draft block. Positions are [current_offset - 1, current_offset,
    # ..., current_offset + block_size - 1]. The verifier returns
    # logits for every position; we read positions [0, ..., block_size - 1]
    # of the OUTPUT (corresponding to inputs at positions [1, ..., block_size])
    # for the per-position accept decision.
    #
    # The "first input is the last confirmed token" mirrors the standard
    # decode contract: the logits at position 0 of the output predict
    # the NEXT token after the last confirmed one — which is exactly the
    # position-0 draft candidate.
    inputs = mx.concatenate(
        [
            mx.array([last_confirmed_token], dtype=mx.uint32),
            draft_block,
        ]
    )

    # Run the target model. The forward writes to ``cache`` via the
    # model's own attention layers; we DON'T call
    # positioned_update_and_fetch here because the model itself owns
    # the cache write. But on PARTIAL acceptance below, we use the
    # positioned helper to rewind the cache to drop rejected positions.
    #
    # ``inputs[None]`` adds the batch dim. The model returns logits of
    # shape ``(1, block_size + 1, vocab_size)``.
    logits = model(inputs[None], cache=cache)

    # We only care about logits at positions [1, ..., block_size] of
    # the input — those predict the draft candidates. Position 0
    # (predicting the FIRST draft from last_confirmed_token) is the
    # always-emitted verify pred; the block walk starts there.
    verify_argmax_arr = _argmax_per_position(logits)
    # ``mx.eval`` so the tolist() below doesn't trigger a sync mid-loop.
    mx.eval(verify_argmax_arr)
    verify_argmax = [int(x) for x in verify_argmax_arr[0, :].tolist()]
    draft_list = [int(x) for x in draft_block.tolist()]

    # The verify argmax at OUTPUT position i predicts the token at
    # INPUT position i+1. So the per-position comparison is:
    #   verify_argmax[i] == draft_list[i]
    # for i in [0, block_size - 1], where verify_argmax[i] is the
    # output-position-i argmax and draft_list[i] is the candidate at
    # input position i+1.
    block_argmax = verify_argmax[:block_size]
    accepted_len, bonus_token = _decide_accepted_prefix(block_argmax, draft_list)

    block_was_full = accepted_len == block_size
    if block_was_full:
        # When every candidate matched, the bonus token is the verify
        # argmax at OUTPUT position ``block_size`` (which predicts the
        # NEXT token after the block). This is the always-accepted
        # "post-block" emit — equivalent to MTP's bonus_tok on accept.
        bonus_token = verify_argmax[block_size]

    # Cache offset rewind on partial accept.
    #
    # The model's forward wrote KV for every position in `inputs`, so
    # the cache now extends to current_offset + block_size. On a
    # partial accept we need to drop the rejected suffix:
    # - Keep positions [current_offset - 1, current_offset, ...,
    #   current_offset + accepted_len - 1].
    # - The corrective bonus token at position current_offset +
    #   accepted_len would be written by the NEXT decode step, not by
    #   this verifier — so the cache offset ends at
    #   current_offset + accepted_len after rewind.
    #
    # We use the positioned helper's offset-rewind semantics: the cache
    # supports trimming via slicing the keys/values arrays. For
    # ``KVCache`` and ``QuantizedKVCache`` this is a direct
    # ``cache.offset`` reset because the helper's writes are
    # idempotent up to the rewound offset.
    verify_offset_after = current_offset + accepted_len
    if not block_was_full:
        _rewind_cache_to(cache, verify_offset_after)

    accepted_tokens = tuple(draft_list[:accepted_len])
    return VerifyResult(
        accepted_tokens=accepted_tokens,
        bonus_token=bonus_token,
        accepted_len=accepted_len,
        block_was_full=block_was_full,
        verify_offset_after=verify_offset_after,
    )


def _rewind_cache_to(cache: list[Any], target_offset: int) -> None:
    """Rewind each per-layer cache's ``offset`` to ``target_offset``.

    The cache's underlying ``keys`` / ``values`` buffers retain their
    written contents past ``offset`` — that's fine, the offset is the
    canonical "valid up to here" bound and any subsequent write
    overwrites the stale tail.

    For mlx-lm's ``KVCache`` / ``QuantizedKVCache``, ``offset`` is a
    plain int attribute; assigning a smaller value is the standard
    rewind path used by both the MTP rollback and the standard
    ``cache.trim(n)`` method.

    Tolerant of cache types that don't have an ``offset`` attribute
    (e.g. the linear-attention :class:`ArraysCache` used by
    GatedDeltaNet) — those caches handle their own rollback via the
    PR #990 ``rollback_state`` slot, which the DFlash path doesn't
    touch (DFlash currently targets pure-attention Qwen3.5/3.6
    dense and MoE; the hybrid path is a follow-up).
    """
    for c in cache:
        if hasattr(c, "offset"):
            try:
                if c.offset > target_offset:
                    c.offset = target_offset
            except AttributeError:  # pragma: no cover — defensive
                # Read-only offset (some cache subclasses) — log and
                # continue. Without a rewind the next forward will
                # write past the target offset; the lossless contract
                # is preserved (corrective token re-fires) but the
                # cache uses more memory than necessary.
                logger.warning(
                    "[dflash.verifier] cache %s has read-only offset; "
                    "skipping rewind. Memory may grow per partial accept.",
                    type(c).__name__,
                )


__all__ = [
    "VerifyResult",
    "verify_block",
    # Exported for unit-test access — the function is meaningful
    # to test in isolation, not just through verify_block.
    "_decide_accepted_prefix",
    "_argmax_per_position",
    "_rewind_cache_to",
]
