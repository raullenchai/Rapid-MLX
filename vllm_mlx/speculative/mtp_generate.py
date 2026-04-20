# SPDX-License-Identifier: Apache-2.0
"""
Shared MTP (Multi-Token Prediction) decode loop.

This module provides the core MTP always-advance decode step used by
BatchedEngine.

Architecture:
  model(input, cache, return_hidden=True) -> (logits, hidden)
  model.mtp_forward(hidden, next_token, mtp_cache) -> draft_logits
  Verify: model([primary, draft], cache) -> accept or reject

The caller is responsible for:
  - Creating the cache (make_prompt_cache + make_mtp_cache)
  - Running prefill
  - Iterating the generator and collecting tokens
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import mlx.core as mx

logger = logging.getLogger(__name__)


@dataclass
class MTPStats:
    """Running statistics for MTP decode."""

    accepted: int = 0
    rejected: int = 0
    errors: int = 0

    @property
    def total(self) -> int:
        return self.accepted + self.rejected

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.total if self.total > 0 else 0.0


@dataclass
class MTPOutput:
    """Output from a single MTP decode step."""

    token: int
    logprobs: mx.array
    is_draft: bool = False  # True if this token was from MTP draft (accepted)


def _snapshot_rnn_state(cache: list) -> dict[int, Any]:
    """Deep-copy non-trimmable (RNN/DeltaNet) cache layers.

    Returns dict mapping layer index -> copied state.
    Only snapshots layers that have .state but are NOT trimmable.
    """
    snapshots = {}
    for ci, c in enumerate(cache):
        if not (hasattr(c, "is_trimmable") and c.is_trimmable()):
            if hasattr(c, "state"):
                orig_state = c.state
                copied = [mx.array(s) if s is not None else None for s in orig_state]
                if isinstance(orig_state, tuple):
                    copied = tuple(copied)
                snapshots[ci] = copied
    return snapshots


def _restore_rnn_state(cache: list, snapshots: dict[int, Any]) -> None:
    """Restore non-trimmable cache layers from snapshots."""
    for ci, snap in snapshots.items():
        cache[ci].state = snap


def _trim_cache(cache: list, n: int) -> None:
    """Trim all trimmable cache layers by n positions."""
    for c in cache:
        if hasattr(c, "is_trimmable") and c.is_trimmable() and hasattr(c, "trim"):
            c.trim(n)


def mtp_generate_step(
    model: Any,
    prompt_tokens: mx.array,
    cache: list,
    sampler: Any,
    max_tokens: int = 256,
    optimistic: bool = False,
) -> Iterator[MTPOutput]:
    """Core MTP decode generator used by BatchedEngine.

    Implements the always-advance strategy:
    1. Forward pass (or use skip_state from previous step)
    2. Sample primary token
    3. MTP head drafts one token
    4. Verify [primary, draft] in single model forward
    5. Accept → emit both tokens, store skip_state
       Reject → emit primary only, rollback cache

    Args:
        model: Model with return_hidden, mtp_forward, make_mtp_cache support
        prompt_tokens: Tokenized prompt as mx.array [1, seq_len]
        cache: Pre-created cache (model cache + MTP cache)
        sampler: Sampling function (logits -> token)
        max_tokens: Maximum tokens to generate
        optimistic: If True, always accept drafts without verification

    Yields:
        MTPOutput for each generated token (primary and accepted drafts)
    """
    from mlx_lm.sample_utils import make_sampler

    draft_sampler = make_sampler(temp=0.0)  # Greedy for drafts
    stats = MTPStats()

    # Detect hybrid cache (mix of trimmable KV + non-trimmable RNN)
    is_hybrid = any(
        hasattr(c, "is_trimmable") and c.is_trimmable() for c in cache
    ) and any(hasattr(c, "is_trimmable") and not c.is_trimmable() for c in cache)

    # Prefill with return_hidden
    model_output = model(prompt_tokens, cache=cache, return_hidden=True)
    if isinstance(model_output, tuple):
        logits, hidden_states = model_output
    else:
        # Model doesn't support return_hidden — fall back to standard decode
        logger.warning("[MTP] model doesn't support return_hidden, falling back")
        logits = model_output
        # Standard decode fallback
        logits = logits[:, -1, :]
        for _ in range(max_tokens):
            token = sampler(logits)
            mx.eval(token)
            lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            yield MTPOutput(token=token.item(), logprobs=lp[0])
            logits = model(token[:, None], cache=cache)
            logits = logits[:, -1, :]
        return

    logits = logits[:, -1, :]
    # hidden_states: keep last position for MTP
    mx.eval(logits)

    skip_state = None
    token_count = 0

    while token_count < max_tokens:
        # --- Step 1: Get logits from skip_state or current ---
        if skip_state is not None:
            logits = skip_state["logits"]
            hidden_states = skip_state["hidden"]
            skip_state = None

        # --- Step 2: Sample primary token ---
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        primary = sampler(logprobs)
        mx.eval(primary)
        yield MTPOutput(token=primary.item(), logprobs=logprobs[0], is_draft=False)
        token_count += 1

        if token_count >= max_tokens:
            break

        # --- Step 3: MTP draft ---
        try:
            h_for_mtp = (
                hidden_states[:, -1:, :] if hidden_states.ndim == 3 else hidden_states
            )
            draft_logits = model.mtp_forward(
                h_for_mtp,
                primary[:, None],
                mtp_cache=None,
            )
            draft_logits = draft_logits[:, -1, :]
            draft = draft_sampler(draft_logits)

            # --- Step 4: Snapshot RNN state for hybrid models ---
            rnn_snapshots = _snapshot_rnn_state(cache) if is_hybrid else {}

            # --- Step 5: Verify [primary, draft] in one forward pass ---
            verify_input = mx.concatenate([primary[:, None], draft[:, None]], axis=1)
            verify_output = model(verify_input, cache=cache, return_hidden=True)
            if isinstance(verify_output, tuple):
                verify_logits, verify_hidden = verify_output
            else:
                verify_logits = verify_output
                verify_hidden = None

            # --- Step 6: Accept or Reject ---
            if optimistic:
                # Always accept, zero sync
                draft_lp = verify_logits[:, 0, :] - mx.logsumexp(
                    verify_logits[:, 0, :], axis=-1, keepdims=True
                )
                if verify_hidden is not None:
                    skip_state = {
                        "logits": verify_logits[:, 1, :],
                        "hidden": verify_hidden[:, -1:, :],
                    }
                    mx.async_eval(
                        skip_state["logits"], skip_state["hidden"], draft, draft_lp
                    )
                yield MTPOutput(token=draft.item(), logprobs=draft_lp[0], is_draft=True)
                token_count += 1
                stats.accepted += 1
            else:
                # Verified mode
                verify_pred = mx.argmax(verify_logits[:, 0, :], axis=-1)
                mx.eval(verify_pred, draft)

                if verify_pred.item() == draft.item():
                    # --- ACCEPT ---
                    draft_lp = verify_logits[:, 0, :] - mx.logsumexp(
                        verify_logits[:, 0, :], axis=-1, keepdims=True
                    )
                    if verify_hidden is not None:
                        skip_state = {
                            "logits": verify_logits[:, 1, :],
                            "hidden": verify_hidden[:, -1:, :],
                        }
                        mx.async_eval(skip_state["logits"], skip_state["hidden"])
                    yield MTPOutput(
                        token=draft.item(), logprobs=draft_lp[0], is_draft=True
                    )
                    token_count += 1
                    stats.accepted += 1
                else:
                    # --- REJECT ---
                    if rnn_snapshots:
                        # Hybrid: undo both P and D, restore RNN, re-advance with P
                        _trim_cache(cache, 2)
                        _restore_rnn_state(cache, rnn_snapshots)
                        rerun_out = model(
                            primary[:, None], cache=cache, return_hidden=True
                        )
                        if isinstance(rerun_out, tuple):
                            rerun_logits, rerun_hidden = rerun_out
                            skip_state = {
                                "logits": rerun_logits[:, -1, :],
                                "hidden": rerun_hidden[:, -1:, :],
                            }
                            mx.async_eval(skip_state["logits"], skip_state["hidden"])
                    else:
                        # Pure attention: simple trim(1)
                        _trim_cache(cache, 1)
                        if verify_hidden is not None:
                            skip_state = {
                                "logits": verify_logits[:, 0, :],
                                "hidden": verify_hidden[:, 0:1, :],
                            }
                            mx.async_eval(skip_state["logits"], skip_state["hidden"])
                    stats.rejected += 1

        except Exception as e:
            logger.debug(f"[MTP] draft/verify failed: {e}")
            skip_state = None
            stats.errors += 1
            # Fall back to normal forward for next token
            out = model(primary[:, None], cache=cache, return_hidden=True)
            if isinstance(out, tuple):
                logits, hidden_states = out
            else:
                logits = out
                hidden_states = None
            logits = logits[:, -1, :]
            mx.eval(logits)
            continue

        # If skip_state is None (verify_hidden was None), do fresh forward
        if skip_state is None:
            last_token = mx.array([[primary.item()]])
            out = model(last_token, cache=cache, return_hidden=True)
            if isinstance(out, tuple):
                logits, hidden_states = out
            else:
                logits = out
                hidden_states = None
            logits = logits[:, -1, :]
            mx.eval(logits)

    logger.info(
        "[MTP] decode stats: %d accepted, %d rejected, %d errors "
        "(%.1f%% acceptance rate)",
        stats.accepted,
        stats.rejected,
        stats.errors,
        stats.acceptance_rate * 100,
    )
