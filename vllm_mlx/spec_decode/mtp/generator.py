# SPDX-License-Identifier: Apache-2.0
"""Vendored ``mtp_generate_step`` from mlx-lm PR #990 (commit ``50c164fb``).

The function is a near-verbatim port of upstream
``mlx_lm/generate.py::mtp_generate_step``. Three things had to change
to make it importable against our installed mlx-lm 0.31.3:

1. ``make_sampler_chain`` does not exist upstream — PR #990 adds it.
   We define a local fallback :func:`_make_sampler_chain` with the
   exact same signature. When upstream merges, callers can switch to
   ``mlx_lm.sample_utils.make_sampler_chain`` without any other
   change.
2. ``apply_xtc`` does not yet accept a ``p_draw`` argument upstream —
   PR #990 adds it for shared-draw determinism. We wrap upstream's
   ``apply_xtc`` and override the draw with the cell when the
   ``p_draw`` slot is set, falling back to a fresh draw otherwise.
3. The accept-rate counter (:class:`MTPAcceptCounter`) is rapid-mlx's
   addition. PR #990 just prints an accept ratio at the end; we
   instead bump
   :func:`vllm_mlx.spec_decode.mtp.accept_counter.get_global_counter`
   on every attempt / accept, which surfaces through the Prometheus
   ``rapid_mlx_spec_decode_*`` series.

Everything else — the verify / accept logic, the rollback path, the
probabilistic-acceptance ``min(1, p_target/p_draft)`` test, the
residual-distribution sample on rejection — is the upstream code
unchanged. That is intentional. The lossless contract (byte-identical
to non-spec-decode for the same prompt + seed at temp=0) lives in the
verify / accept arithmetic, and rewriting it would risk a divergence
that a unit test against a single mocked model can't catch.

Public signature mirrors upstream so callers can swap
``mtp_generate_step(prompt, model, ...)`` for
``mlx_lm.generate.generate_step(prompt, model, ...)`` with only kwarg
adjustments.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Generator
from functools import partial
from typing import Any

import mlx.core as mx

# Force the ArraysCache rollback_state patch on first import — the
# generator references ``cache.rollback_state`` directly inside
# ``_rollback_draft``, and the patch lifts that attribute from a
# missing-class-attr to a class-default-None.
from .accept_counter import get_global_counter
from .cache_patch import patch_arrays_cache_rollback_state
from .draft_k_controller import (
    DraftKController,
)
from .draft_k_controller import (
    get_global_controller as _get_global_draft_k_controller,
)

patch_arrays_cache_rollback_state()

logger = logging.getLogger(__name__)

# Match upstream PR #990 cache-clear cadence verbatim (``_CACHE_CLEAR_INTERVAL = 256``).
_CACHE_CLEAR_INTERVAL = 256


# ---------------------------------------------------------------------------
# Local sampler-chain helpers — vendor of the new ``make_sampler_chain``
# from PR #990 ``mlx_lm/sample_utils.py``. When upstream merges, this
# block becomes a one-line ``from mlx_lm.sample_utils import
# make_sampler_chain``.
# ---------------------------------------------------------------------------


def _apply_xtc_with_shared_draw(
    logits: mx.array,
    xtc_probability: float,
    xtc_threshold: float,
    xtc_special_tokens: list[int],
    p_draw: mx.array | None,
) -> mx.array:
    """XTC sampler with optional shared draw (PR #990 surface).

    Vendored from PR #990's ``apply_xtc(p_draw=...)`` addition. When
    ``p_draw`` is ``None`` we delegate to upstream's bare ``apply_xtc``
    (which makes a fresh internal draw); when ``p_draw`` is supplied
    we replicate the upstream gate inline using the provided draw so
    the draft and verify steps share the same apply/skip decision.

    Sharing the draw is what makes XTC sampling deterministic across
    the draft + verify pair — without it, the verify step could
    independently roll a different XTC decision from the draft step
    and the acceptance ratio drops sharply (and worse, the lossless
    contract at temp=0 quietly breaks because the verify step would
    mask different special tokens than the draft).
    """
    from mlx_lm.sample_utils import apply_xtc

    if p_draw is None:
        return apply_xtc(logits, xtc_probability, xtc_threshold, xtc_special_tokens)

    # Inline replication of PR #990's apply_xtc body with the supplied
    # draw — this fork only executes when XTC + shared draw are BOTH
    # active, which is rare (operators rarely combine XTC with MTP).
    # Matches PR #990 mlx_lm/sample_utils.py:300-306 verbatim.
    if not (0 <= xtc_threshold <= 0.5):
        raise ValueError(f"xtc_threshold must be in [0, 0.5]; got {xtc_threshold}")
    probs = mx.softmax(logits, axis=-1)
    mask = probs > xtc_threshold
    n_above = mask.sum(axis=-1, keepdims=True)
    mask = mx.where(n_above > 1, mask, mx.zeros_like(mask))
    if xtc_special_tokens:
        mask[..., xtc_special_tokens] = False
    return mx.where(
        p_draw > xtc_probability,
        logits,
        mx.where(mask, -mx.inf, logits),
    )


def _make_sampler_chain(
    top_p: float = 0.0,
    top_k: int = 0,
    min_p: float = 0.0,
    min_tokens_to_keep: int = 1,
    xtc_probability: float = 0.0,
    xtc_threshold: float = 0.0,
    xtc_special_tokens: list[int] | None = None,
) -> tuple[list[Callable[[mx.array], mx.array]], list | None]:
    """Vendored ``make_sampler_chain`` (PR #990, sample_utils.py:1028).

    Returns ``(chain, xtc_cell)`` where ``xtc_cell`` is a single-slot
    mutable list used to share the XTC draw across the draft and
    verify steps. ``xtc_cell`` is ``None`` when XTC is disabled.
    """
    from mlx_lm.sample_utils import apply_min_p, apply_top_k, apply_top_p

    xtc_special_tokens = xtc_special_tokens or []
    xtc_cell: list | None = [None] if xtc_probability > 0.0 else None
    chain: list[Callable[[mx.array], mx.array]] = []
    if 0 < top_p < 1.0:
        chain.append(lambda x: apply_top_p(x, top_p))
    if min_p != 0.0:
        chain.append(lambda x: apply_min_p(x, min_p, min_tokens_to_keep))
    if xtc_probability > 0.0:
        # Capture xtc_cell by reference — closure reads the current
        # cell[0] each invocation, so writes from the outer loop are
        # visible inside the lambda.
        def _xtc(x, _cell=xtc_cell):
            return _apply_xtc_with_shared_draw(
                x,
                xtc_probability,
                xtc_threshold,
                xtc_special_tokens,
                _cell[0],
            )

        chain.append(_xtc)
    if top_k > 0:
        chain.append(lambda x: apply_top_k(x, top_k))
    return chain, xtc_cell


# ---------------------------------------------------------------------------
# The vendored generator. Body mirrors PR #990 mlx_lm/generate.py:662-997
# line-by-line; comments and rapid-mlx accept-counter hooks added.
# ---------------------------------------------------------------------------


def mtp_generate_step(
    prompt: mx.array,
    model: Any,
    *,
    max_tokens: int = 256,
    logits_processors: list[Callable[[mx.array, mx.array], mx.array]] | None = None,
    prompt_cache: Any | None = None,
    prefill_step_size: int = 2048,
    kv_bits: int | None = None,
    kv_group_size: int = 64,
    quantized_kv_start: int = 0,
    input_embeddings: mx.array | None = None,
    temp: float = 0.0,
    top_p: float = 0.0,
    top_k: int = 0,
    min_p: float = 0.0,
    min_tokens_to_keep: int = 1,
    xtc_probability: float = 0.0,
    xtc_threshold: float = 0.0,
    xtc_special_tokens: list[int] | None = None,
    accept_counter=None,
    draft_k_controller: DraftKController | None = None,
) -> Generator[tuple[int, mx.array, bool], None, None]:
    """Generator that uses the model's native MTP head for spec decode.

    Vendored verbatim from mlx-lm PR #990
    ``mlx_lm/generate.py::mtp_generate_step``. Each iteration runs one
    backbone forward pass (over the current token plus its pending
    draft) and one MTP forward pass (to propose the next draft). Up
    to two tokens are emitted per backbone step: one always-accepted
    backbone token and one conditionally-accepted draft token.

    Requirements on ``model``:

    * Implements ``mtp_forward(hidden, next_token_ids, mtp_cache)``
      returning logits of shape ``(B, N, vocab_size)``.
    * Implements ``make_mtp_cache()`` returning a list of caches the
      MTP transformer layers can write into.
    * Accepts ``return_hidden=True`` in ``__call__`` and returns
      ``(logits, hidden)`` where ``hidden`` is the pre-norm backbone
      hidden state at every position.
    * Accepts ``n_confirmed=int`` in ``__call__`` (used by the
      GatedDeltaNet layer to snapshot its SSM/conv state at the
      confirmed boundary so the generator can roll back on draft
      rejection).

    The :func:`vllm_mlx.spec_decode.mtp.qwen3_5_inject.inject_mtp_support`
    helper installs all four on a freshly-loaded
    ``mlx_lm.models.qwen3_5.TextModel`` instance.

    Yields:
        Tuples of ``(token_int, logprobs_array, from_draft_bool)``.
        ``from_draft`` is ``True`` when the token came from an
        accepted MTP draft, ``False`` when it came from the backbone.

    Args:
        accept_counter: Optional override for the process-global
            :class:`MTPAcceptCounter`. Tests pass a fresh counter to
            isolate measurements; production callers pass ``None``
            and the module-global counter is used.
        draft_k_controller: Optional runtime auto-tune controller for
            the number of draft tokens per verify pass (PR-1 of the
            0.9.11 Gemma-4 MTP roadmap). When ``None`` (the default),
            the module-global controller is consulted via
            :func:`~vllm_mlx.spec_decode.mtp.draft_k_controller.get_global_controller`;
            when that is also ``None`` the generator is in
            static-``k`` mode and behaves byte-identically to the
            pre-PR-1 code path. When set, every verify-pass outcome
            (accept or reject) is recorded on the controller AFTER
            the accept/reject decision has been made. The controller
            does NOT change the arithmetic of this generator — the
            lossless verify/accept contract from PR #990 is
            preserved verbatim. Reading ``current_k()`` and looping
            over ``k`` draft tokens per verify pass is a follow-up
            PR (Gemma-4 sidecar); PR-1 only lands the feedback
            input.
    """
    from mlx_lm.generate import generation_stream, maybe_quantize_kv_cache
    from mlx_lm.models import cache as _cache_module
    from mlx_lm.sample_utils import categorical_sampling

    xtc_special_tokens = xtc_special_tokens or []
    if accept_counter is None:
        accept_counter = get_global_counter()
    # Fall back to the module-global controller when the caller
    # doesn't pass one. ``get_global_controller()`` returns ``None``
    # when auto-tune has not been installed at boot — that keeps the
    # hot loop cost to one ``is None`` check per verify pass, so the
    # static-``k`` fast path is byte-identical to the pre-PR-1 code.
    if draft_k_controller is None:
        draft_k_controller = _get_global_draft_k_controller()

    y = prompt.astype(mx.uint32)
    prev_tokens: mx.array | None = None

    if prompt_cache is None:
        model_cache = _cache_module.make_prompt_cache(model)
        mtp_cache = model.make_mtp_cache()
    else:
        # Split a pre-built cache at backbone length. If MTP entries
        # are absent (e.g. cache made by make_prompt_cache), construct
        # them.
        n_main = len(model.layers)
        model_cache = prompt_cache[:n_main]
        mtp_cache = prompt_cache[n_main:] or model.make_mtp_cache()

    _is_greedy = temp == 0

    _filter_chain, _xtc_cell = (
        _make_sampler_chain(
            top_p,
            top_k,
            min_p,
            min_tokens_to_keep,
            xtc_probability,
            xtc_threshold,
            xtc_special_tokens,
        )
        if not _is_greedy
        else ([], None)
    )

    quantize_cache_fn = partial(
        maybe_quantize_kv_cache,
        quantized_kv_start=quantized_kv_start,
        kv_group_size=kv_group_size,
        kv_bits=kv_bits,
    )

    def _process_and_sample(tokens, logits, xtc_draw=None):
        if logits_processors:
            logits = logits[None]
            for processor in logits_processors:
                logits = processor(tokens, logits)
            logits = logits.squeeze(0)
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        if _filter_chain:
            if _xtc_cell is not None:
                _xtc_cell[0] = xtc_draw  # None = fresh draw; mx.array = shared
            masked = logprobs
            for f in _filter_chain:
                masked = f(masked)
            token = categorical_sampling(masked, temp)
            scaled = masked / temp
            lp_accept = scaled - mx.logsumexp(scaled, axis=-1, keepdims=True)
        elif _is_greedy:
            token = mx.argmax(logprobs, axis=-1)
            lp_accept = logprobs
        else:
            token = categorical_sampling(logprobs, temp)
            scaled = logprobs / temp
            lp_accept = scaled - mx.logsumexp(scaled, axis=-1, keepdims=True)
        return token, logprobs, lp_accept

    def _clear_rollback():
        for c in model_cache:
            if hasattr(c, "rollback_state"):
                c.rollback_state = None

    def _rollback_draft():
        """Restore caches to the state after the confirmed token.

        SSM layers (ArraysCache): restore the conv/ssm snapshot saved
        by GatedDeltaNet after the confirmed token.
        Attention layers (KVCache): trim the draft-token entry.
        """
        for c in model_cache:
            if hasattr(c, "rollback_state") and c.rollback_state is not None:
                conv_snap, ssm_snap = c.rollback_state
                c[0] = conv_snap
                c[1] = ssm_snap
                c.rollback_state = None
            elif c.is_trimmable():
                c.trim(1)

    def _step_backbone(yy, prev, n_predict=1, n_confirmed=0, xtc_draw=None):
        """Run backbone on ``yy`` and return (tokens, logprobs, accept_lps, hidden, prev)."""
        with mx.stream(generation_stream):
            logits, hidden = model(
                yy[None],
                cache=model_cache,
                return_hidden=True,
                n_confirmed=n_confirmed,
            )
            logits = logits[:, -n_predict:, :]
            quantize_cache_fn(model_cache)
            toks: list = []
            lps: list = []
            accept_lps: list = []
            for i in range(n_predict):
                if logits_processors:
                    prev = (
                        mx.concatenate([prev, yy[i : i + 1]])
                        if prev is not None
                        else yy[i : i + 1]
                    )
                # Shared XTC draw only for position 0 (verify position).
                draw = xtc_draw if i == 0 else None
                tok, lp, alp = _process_and_sample(
                    prev, logits[:, i, :].squeeze(0), draw
                )
                toks.append(tok)
                lps.append(lp)
                accept_lps.append(alp)
            return (
                mx.stack(toks),
                mx.stack(lps),
                mx.stack(accept_lps),
                hidden,
                prev,
            )

    def _step_mtp(hidden_last, main_tok, prev, *, cache_commit=None):
        """Run MTP head and return (draft_tok, draft_lp, draft_accept_lp, xtc_draw)."""
        if cache_commit is not None:
            align_h, align_tok = cache_commit
            hidden_last = mx.concatenate([align_h, hidden_last], axis=1)
            next_ids = mx.concatenate(
                [align_tok.reshape(1, 1), main_tok.reshape(1, 1)], axis=1
            )
        else:
            next_ids = main_tok.reshape(1, 1)
        with mx.stream(generation_stream):
            mtp_logits = model.mtp_forward(hidden_last, next_ids, mtp_cache)
            quantize_cache_fn(mtp_cache)
            mtp_logits = mtp_logits[:, -1, :].squeeze(0)
            if logits_processors:
                tokens_for_proc = (
                    mx.concatenate([prev, main_tok.reshape(-1)])
                    if prev is not None
                    else main_tok.reshape(-1)
                )
            else:
                tokens_for_proc = prev
            xtc_draw = mx.random.uniform() if _xtc_cell is not None else None
            draft_tok, draft_lp, draft_accept_lp = _process_and_sample(
                tokens_for_proc, mtp_logits, xtc_draw
            )
        return draft_tok, draft_lp, draft_accept_lp, xtc_draw

    def _prefill(yy, embeddings):
        # Leave exactly 1 token for _step_backbone so the decode loop
        # starts clean.
        total = len(embeddings) if embeddings is not None else yy.size
        while total > 1:
            n = min(prefill_step_size, total - 1)
            if embeddings is not None:
                _, hidden = model(
                    yy[:n][None],
                    cache=model_cache,
                    return_hidden=True,
                    input_embeddings=embeddings[:n][None],
                )
                embeddings = embeddings[n:]
            else:
                _, hidden = model(yy[:n][None], cache=model_cache, return_hidden=True)
            model.mtp_forward(hidden, yy[1 : n + 1][None], mtp_cache)
            quantize_cache_fn(mtp_cache)
            quantize_cache_fn(model_cache)
            mx.eval([c.state for c in model_cache + mtp_cache if hasattr(c, "state")])
            yy = yy[n:]
            total -= n
            mx.clear_cache()
        return yy

    with mx.stream(generation_stream):
        y = _prefill(y, input_embeddings)

    ntoks = 0
    last_cache_block = 0
    draft_tok = draft_lp = draft_accept_lp = draft_xtc_draw = None

    while ntoks < max_tokens:
        if draft_tok is None:
            # No pending draft: run backbone only, generate first draft.
            toks, lps, accept_lps, hidden, prev_tokens = _step_backbone(
                y, prev_tokens, n_predict=1
            )
            mx.eval(toks)
            main_tok, main_lp = toks[0], lps[0]
            ntoks += 1
            yield main_tok.item(), main_lp, False
            if ntoks >= max_tokens:
                return
            hidden_at_main = hidden[:, -1:, :]
            draft_tok, draft_lp, draft_accept_lp, draft_xtc_draw = _step_mtp(
                hidden_at_main, main_tok, prev_tokens
            )
            mx.eval(draft_tok)
            y = mx.array([main_tok.item()], mx.uint32)
        else:
            # Verify draft: run backbone over [y, draft_tok].
            # n_confirmed=1 causes GatedDeltaNet to snapshot its SSM/conv
            # state at the confirmed boundary so rejection rolls back.
            y_with_draft = mx.concatenate([y, mx.array([draft_tok.item()], mx.uint32)])
            toks, lps, accept_lps, hidden, prev_tokens = _step_backbone(
                y_with_draft,
                prev_tokens,
                n_predict=2,
                n_confirmed=1,
                xtc_draw=draft_xtc_draw,
            )
            u = mx.random.uniform()
            mx.eval(toks, draft_tok, u)

            verify_pred, bonus_tok = toks[0], toks[1]
            verify_lp, bonus_lp = lps[0], lps[1]
            verify_accept_lp = accept_lps[0]
            draft_tok_id = draft_tok.item()

            # Bump attempts BEFORE deciding accept/reject so the
            # counter is consistent under a midway exception.
            accept_counter.record_attempt()

            if _is_greedy:
                accept = verify_pred.item() == draft_tok_id
            else:
                # Probabilistic acceptance: min(1, p_target/p_draft).
                log_accept = (
                    verify_accept_lp[draft_tok_id] - draft_accept_lp[draft_tok_id]
                ).item()
                accept = log_accept >= 0 or u.item() < math.exp(log_accept)

            hidden_at_confirmed = hidden[:, 0:1, :]
            hidden_at_draft = hidden[:, 1:2, :]

            if accept:
                _clear_rollback()
                accept_counter.record_accept(tokens_saved=1)
                # PR-1 auto-tune feedback: record the accepted verify
                # outcome. When the controller is not installed this
                # branch is skipped entirely, preserving the pre-PR-1
                # code path byte-for-byte.
                if draft_k_controller is not None:
                    draft_k_controller.record_attempt(accepted=True)
                ntoks += 1
                yield draft_tok_id, draft_lp, True
                if ntoks >= max_tokens:
                    return
                ntoks += 1
                yield bonus_tok.item(), bonus_lp, False
                if ntoks >= max_tokens:
                    return
                # Next draft: cache-commit aligns the cache for the
                # accepted draft token and generates the next draft in
                # the same batched forward.
                draft_tok, draft_lp, draft_accept_lp, draft_xtc_draw = _step_mtp(
                    hidden_at_draft,
                    bonus_tok,
                    prev_tokens,
                    cache_commit=(hidden_at_confirmed, draft_tok),
                )
                mx.eval(draft_tok)
                y = mx.array([bonus_tok.item()], mx.uint32)
            else:
                _rollback_draft()
                accept_counter.record_reject()
                # PR-1 auto-tune feedback: record the rejected verify
                # outcome. Same reasoning as the accept branch — the
                # ``is None`` check keeps the static-``k`` path
                # unchanged when the controller is not installed.
                if draft_k_controller is not None:
                    draft_k_controller.record_attempt(accepted=False)
                if logits_processors and prev_tokens is not None:
                    prev_tokens = prev_tokens[:-1]  # discard rejected
                verify_tok_id = verify_pred.item()
                if not _is_greedy:
                    # Residual-distribution sample on reject so the
                    # output marginal still equals the target distro.
                    p_target = mx.exp(verify_accept_lp)
                    p_draft = mx.exp(draft_accept_lp)
                    residual = mx.maximum(p_target - p_draft, 0.0)
                    z = residual.sum(keepdims=True)
                    dist = mx.where(z > 0, residual, p_target)
                    verify_tok_id = mx.random.categorical(
                        mx.log(dist).reshape(1, -1)
                    ).item()
                ntoks += 1
                yield verify_tok_id, verify_lp, False
                if ntoks >= max_tokens:
                    return
                # Next draft from MTP at y's hidden state.
                draft_tok, draft_lp, draft_accept_lp, draft_xtc_draw = _step_mtp(
                    hidden_at_confirmed,
                    mx.array([verify_tok_id], mx.uint32),
                    prev_tokens,
                )
                mx.eval(draft_tok)
                y = mx.array([verify_tok_id], mx.uint32)
        block = ntoks // _CACHE_CLEAR_INTERVAL
        if block > last_cache_block:
            mx.clear_cache()
            last_cache_block = block
