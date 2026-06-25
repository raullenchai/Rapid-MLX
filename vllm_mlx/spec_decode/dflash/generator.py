# SPDX-License-Identifier: Apache-2.0
"""DFlash block-diffusion generator loop (R15 task #313).

Pairs a :class:`BlockDiffusionDrafter` with the target model's
:func:`verify_block` to yield tokens at temp=0 with the lossless
contract preserved.

Public surface
--------------

:func:`dflash_generate_step` — generator function matching the
:func:`vllm_mlx.spec_decode.mtp.generator.mtp_generate_step` signature
in spirit (yields ``(token_int, logprobs, from_draft)`` triples) so the
upstream scheduler dispatch can branch on
:attr:`SchedulerConfig.spec_decode` and call the right generator
without conditional kwargs.

Differences from MTP's generator
--------------------------------

* DFlash takes a SEPARATE drafter (not a built-in head). The drafter
  is constructed once at request start and reset between requests via
  :meth:`BlockDiffusionDrafter.reset`.
* Each outer-loop iteration emits up to ``block_size + 1`` tokens (the
  accepted prefix plus the always-emitted verify bonus). MTP emits up
  to 2 tokens (verify + accepted draft).
* The verifier's KV-cache write goes through
  :func:`vllm_mlx.positioned_kv_cache.positioned_update_and_fetch`
  via the model's own forward — so the cache contract is the same as
  monotonic decode, plus the offset rewind on partial accept.

Lossless contract
-----------------

For every prompt + sampling config (temp=0), the yielded token sequence
MUST be byte-identical to a no-spec-decode generation. The contract is
enforced by:

* :func:`vllm_mlx.spec_decode.dflash.verifier.verify_block` always
  emitting the verify model's argmax at the divergence position;
* the generator NEVER yielding the rejected draft suffix;
* the next outer-loop iteration starting from the bonus token (the
  verifier's pred) as ``last_confirmed_token``, identical to what
  the no-spec-decode path would feed.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Generator
from typing import Any

import mlx.core as mx

from .accept_counter import get_global_counter
from .drafter import BlockDiffusionDrafter
from .verifier import verify_block

logger = logging.getLogger(__name__)


def _validate_block_size(block_size: int) -> None:
    """Reject non-positive or non-integer block sizes early."""
    if not isinstance(block_size, int):
        raise TypeError(f"block_size must be int; got {type(block_size).__name__}")
    if block_size < 1:
        raise ValueError(f"block_size must be >= 1; got {block_size}")


def _validate_temperature(temperature: float) -> None:
    """Reject NaN / Inf temperature.

    DFlash currently only supports temp=0.0 (greedy). A small epsilon
    around zero is accepted to forgive float-rounded ``temp=0.0`` from
    JSON parsing.
    """
    if not math.isfinite(float(temperature)):
        raise ValueError(
            f"temperature={temperature!r} is not finite (NaN / Inf not allowed)"
        )
    if temperature != 0.0:
        raise NotImplementedError(
            "DFlash generator currently only supports temperature=0.0. "
            "See vllm_mlx/spec_decode/dflash/verifier.py docstring for "
            "the speculative-sampling extension plan."
        )


def dflash_generate_step(
    prompt: mx.array,
    model: Any,
    drafter: BlockDiffusionDrafter,
    *,
    block_size: int = 16,
    max_tokens: int = 256,
    temperature: float = 0.0,
    accept_counter: Any = None,
) -> Generator[tuple[int, mx.array | None, bool], None, None]:
    """Generate tokens via DFlash spec decode, yielding one token at a time.

    Args:
        prompt: 1-D ``mx.uint32`` prompt token IDs. The verifier
            consumes the prompt via a prefill forward pass before the
            first block; the first yielded token comes from that
            prefill (always ``from_draft=False``).
        model: Target model — see
            :func:`vllm_mlx.spec_decode.dflash.verifier.verify_block`
            for the interface.
        drafter: The block-diffusion drafter. The generator calls
            :meth:`BlockDiffusionDrafter.reset` once at start.
        block_size: Draft block size. Must match
            ``drafter.block_size`` — raises if not.
        max_tokens: Maximum tokens to emit (including the prefill
            primary). The generator stops after yielding this many
            even if the verifier is mid-block.
        temperature: Sampling temperature. Currently only ``0.0`` is
            supported.
        accept_counter: Optional override for the process-global
            :class:`DFlashAcceptCounter`. Tests pass a fresh counter
            for isolation; production callers pass ``None`` and the
            module-global counter is used.

    Yields:
        Tuples ``(token_int, logprobs, from_draft_bool)``. ``logprobs``
        is ``None`` for the simple greedy path (no logprob stream needed
        at temp=0); future temp>0 support would populate it.

    Raises:
        ValueError: On block-size mismatch / bad prompt shape.
        NotImplementedError: On ``temperature > 0``.
    """
    _validate_block_size(block_size)
    _validate_temperature(temperature)
    if drafter.block_size != block_size:
        raise ValueError(
            f"drafter.block_size={drafter.block_size} does not match "
            f"requested block_size={block_size}"
        )
    if prompt.ndim != 1:
        raise ValueError(f"prompt must be 1-D; got shape {tuple(prompt.shape)}")
    if prompt.shape[0] == 0:
        raise ValueError("prompt must contain at least one token")
    if max_tokens < 1:
        raise ValueError(f"max_tokens must be >= 1; got {max_tokens}")

    if accept_counter is None:
        accept_counter = get_global_counter()

    drafter.reset()

    # Lazy import inside function body to keep CLI startup fast (avoids
    # pulling mlx-lm cache builders at module-import time).
    from mlx_lm.models import cache as _cache_module

    cache: list[Any] = _cache_module.make_prompt_cache(model)

    # ---- Prefill: run the prompt through the target model in one pass
    # ----------------------------------------------------------------
    # We don't use the verifier here because no draft block exists
    # yet; the prefill is a plain decode-style forward whose argmax at
    # the LAST position becomes the first emitted token. Equivalent to
    # mtp_generate_step's cold-start backbone call.
    prefill_logits = model(prompt[None], cache=cache)
    last_logit = prefill_logits[:, -1, :]
    primary_token = int(mx.argmax(last_logit, axis=-1).item())

    emitted = 0
    yield primary_token, None, False
    emitted += 1
    if emitted >= max_tokens:
        return

    last_confirmed = primary_token
    # The cache offset after prefill is len(prompt); the FIRST block
    # is drafted at position len(prompt) (the slot right after the
    # primary token, since the primary is what predicted that slot).
    current_offset = int(prompt.shape[0])

    while emitted < max_tokens:
        # 1. Draft a block.
        # ``current_offset`` is the position of the FIRST token in the
        # block. The prefix passed is the prompt + primary + everything
        # accepted so far (drafters typically only need the tail; we
        # pass the full thing for safety — the StubBlockDiffusionDrafter
        # ignores it entirely, the mlx-vlm adapter forwards through).
        prefix_so_far = mx.array([last_confirmed], dtype=mx.uint32)
        try:
            draft_block = drafter.draft_block(prefix_so_far, current_offset)
        except IndexError:
            # Stub drafter exhausted the script — used in unit tests
            # to terminate the loop cleanly without hitting max_tokens.
            return
        # Bump attempts BEFORE the verify decision so the counter is
        # consistent under a midway exception.
        accept_counter.record_attempt()

        # 2. Verify the block.
        result = verify_block(
            model,
            draft_block,
            last_confirmed_token=last_confirmed,
            cache=cache,
            block_size=block_size,
            current_offset=current_offset,
            temperature=temperature,
        )

        # 3. Bookkeep the accept event.
        # ``accepted_len == 0`` → no draft accepted, just the corrective
        # bonus token. Counts as a reject for the accept-rate gauge but
        # the bonus is still emitted (lossless step). Treating the
        # ``accepted_len > 0`` case as the "accept event" matches MTP's
        # contract: accepts < attempts iff at least one position
        # in the block matched.
        if result.accepted_len > 0:
            # Bonus tokens saved: accepted_len - 1 (the "first verify
            # pred" position is always emitted regardless of spec decode,
            # so only the SUBSEQUENT accepted positions count as
            # speedup). Lower bound at 0.
            bonus_saved = max(0, result.accepted_len - 1)
            accept_counter.record_accept(tokens_saved=bonus_saved)
        else:
            accept_counter.record_reject()

        # 4. Emit accepted tokens, then the bonus token.
        for tok in result.accepted_tokens:
            yield int(tok), None, True
            emitted += 1
            if emitted >= max_tokens:
                return

        # Bonus token: either the corrective verify pred at the
        # divergence (partial accept) OR the post-block prediction
        # (full block accept). Both are always-emitted lossless
        # tokens; from_draft=False because they come from the
        # verifier, not the drafter.
        if result.bonus_token >= 0:
            yield int(result.bonus_token), None, False
            emitted += 1
            if emitted >= max_tokens:
                return

        # 5. Advance state for the next iteration.
        # last_confirmed is the most recent emitted token (the bonus).
        # current_offset is the position immediately AFTER the bonus
        # token — that's verify_offset_after + 1 because the bonus
        # itself sits at position verify_offset_after.
        last_confirmed = int(result.bonus_token)
        current_offset = result.verify_offset_after + 1
