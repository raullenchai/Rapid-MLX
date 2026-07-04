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
import time
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
from .draft_k_controller_v2 import DepthController, get_or_create_controller

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
    # 0.9.13 PR-B (Ollama-style EV depth controller). ``model_id`` keys
    # the process-global controller registry so cost + acceptance state
    # persists across requests for the same model+drafter combination.
    # ``max_k`` bounds the depth the controller may pick; the current
    # generator body only implements K∈{0,1} (park + chain-of-1), so
    # picks above 1 are clamped and logged. ``disable_auto_k=True``
    # keeps the pre-PR-B chain-of-1 behavior unchanged — useful for
    # A/B benching the controller against a fixed K=1 baseline.
    model_id: str | None = None,
    max_k: int = 1,
    disable_auto_k: bool = False,
    # 0.9.13 PR-C EOS holdout. When an accepted draft is a stop token
    # (Gemma 4 ``<end_of_turn>``, tokenizer ``eos_token_id``, or any
    # id in the request's assembled stop set), positions past that
    # index would never have been reached in real decode. Ollama's
    # ``speculate.go:456-472`` caps ``observed`` at the EOS position
    # so the acceptance-rate EWMA does not learn a spurious "position
    # N+1 rate = 0" from a hypothetical rejection that would never
    # have been performed. Emitted token stream is unchanged (the
    # caller still stops at EOS on its side); only the controller's
    # training window shrinks. ``None`` disables the holdout.
    stop_tokens: set[int] | None = None,
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
    """
    import inspect as _inspect

    from mlx_lm.generate import generation_stream, maybe_quantize_kv_cache
    from mlx_lm.models import cache as _cache_module
    from mlx_lm.sample_utils import categorical_sampling

    xtc_special_tokens = xtc_special_tokens or []
    if accept_counter is None:
        accept_counter = get_global_counter()

    # ------------------------------------------------------------------
    # Chain-of-K drafter-hidden-cascade capability probe.
    #
    # The Google Gemma 4 assistant inject (0.9.13 Fix 3) exposes an
    # optional ``return_hidden=True`` kwarg on ``mtp_forward`` that
    # returns ``post_projection(h)`` alongside the drafter's logits.
    # Chaining THIS hidden into the next iteration's ``hidden_last``
    # slot (per Google's ``draft_block``) is the ONLY way to produce
    # a non-degenerate K>=2 chain on a shared-K/V drafter — target
    # cache doesn't advance across chain calls, so a target-hidden-
    # frozen cascade produces d_1 == d_2 == d_3 at temp=0.
    #
    # Qwen 3.5's MTP head has its own advancing KVCache, so the
    # target-hidden-frozen cascade still produces distinct drafts and
    # the capability flag stays off for backwards compat.
    # ------------------------------------------------------------------
    try:
        _mtp_supports_hidden = (
            "return_hidden" in _inspect.signature(model.mtp_forward).parameters
        )
    except (TypeError, ValueError):  # pragma: no cover — non-introspectable
        _mtp_supports_hidden = False

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

    def _rollback_draft(n_to_drop: int = 1):
        """Restore caches by dropping the last ``n_to_drop`` draft tokens.

        SSM layers (ArraysCache): restore the conv/ssm snapshot saved
        by GatedDeltaNet at the confirmed boundary. The snapshot is
        taken at a SINGLE offset (``n_confirmed`` positions from end),
        so ``n_to_drop`` MUST match that offset — chain-of-K with
        partial accept is not representable in the current one-snapshot
        model. The generator prevents this by clamping ``max_k`` to 1
        when any SSM cache is present (see ``_has_ssm_cache`` at
        decode-loop start); callers that reach this path with
        ``n_to_drop > 1`` on an SSM cache trip an assertion because a
        silent partial-rollback would corrupt the SSM state and break
        the lossless contract.

        Attention layers (KVCache): trim the last ``n_to_drop`` draft
        entries.
        """
        for c in model_cache:
            if hasattr(c, "rollback_state") and c.rollback_state is not None:
                # SSM path: single-snapshot rollback, only n_to_drop==1
                # is representable in the current on-disk snapshot slot.
                # The controller-side clamp keeps chain-of-K away from
                # this branch; assert here as a defense in depth in case
                # a caller wires K>=2 without adjusting the SSM cache.
                if n_to_drop != 1:
                    raise AssertionError(
                        f"_rollback_draft(n_to_drop={n_to_drop}) on SSM "
                        "cache: only single-token rollback is supported. "
                        "Chain-of-K on SSM-hybrid targets is not wired "
                        "yet — the generator should have clamped max_k=1."
                    )
                conv_snap, ssm_snap = c.rollback_state
                c[0] = conv_snap
                c[1] = ssm_snap
                c.rollback_state = None
            elif c.is_trimmable():
                c.trim(n_to_drop)

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

    def _step_mtp(hidden_last, main_tok, prev, *, cache_commit=None, want_hidden=False):
        """Run MTP head and return draft state.

        Returns ``(draft_tok, draft_lp, draft_accept_lp, xtc_draw, drafter_hidden_or_None)``.
        ``drafter_hidden_or_None`` is populated only when ``want_hidden=True``
        AND the injected ``mtp_forward`` accepts ``return_hidden`` — the
        caller is responsible for guarding on ``model_supports_hidden``
        before setting the flag. For the Gemma 4 Google-assistant path
        this is the drafter's ``post_projection(h)`` (``(B, N,
        backbone_hidden)``) at the last predicted position, ready to
        chain into the next iteration's ``hidden_last`` slot.
        """
        if cache_commit is not None:
            align_h, align_tok = cache_commit
            hidden_last = mx.concatenate([align_h, hidden_last], axis=1)
            next_ids = mx.concatenate(
                [align_tok.reshape(1, 1), main_tok.reshape(1, 1)], axis=1
            )
        else:
            next_ids = main_tok.reshape(1, 1)
        drafter_hidden_last = None
        with mx.stream(generation_stream):
            if want_hidden:
                mtp_logits, mtp_hidden = model.mtp_forward(
                    hidden_last, next_ids, mtp_cache, return_hidden=True
                )
                # Keep only the LAST predicted-position hidden — that's
                # what feeds the next chain iteration. Shape
                # ``(B, 1, backbone_hidden)``.
                drafter_hidden_last = mtp_hidden[:, -1:, :]
            else:
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
        return draft_tok, draft_lp, draft_accept_lp, xtc_draw, drafter_hidden_last

    def _step_mtp_chain(hidden_last, main_tok, prev, K, *, cache_commit=None):
        """Generate ``K`` sequential drafts by cascading MTP calls.

        Two cascade shapes are supported, selected by the injected
        ``mtp_forward`` surface:

        1. **Drafter-hidden cascade (preferred)**. When ``mtp_forward``
           accepts ``return_hidden=True`` (Gemma 4 Google-assistant
           inject, ``0.9.13`` Fix 3), the drafter's own
           ``post_projection(h)`` is fed into the next iteration's
           ``hidden_last`` slot. This mirrors Google's
           ``Gemma4AssistantDraftModel.draft_block`` in
           ``mlx_vlm/speculative/drafters/gemma4_assistant/gemma4_assistant.py``:
           each successive iteration sees a fresh drafter-hidden
           context instead of reusing target's frozen hidden, and each
           successive iteration's token IS shaped by the prior
           iteration's own reasoning. This is what makes chain-of-K
           produce a NON-DEGENERATE chain on Google's assistant (where
           K/V is shared with target and does NOT advance across
           chain calls, so a target-hidden-frozen cascade would
           produce d_1 == d_2 == d_3 at temp=0).
        2. **Target-hidden cascade (fallback, K=1 baseline for Qwen 3.5
           and any pre-Fix-3 inject)**. The Qwen 3.5 MTP head has its
           own KVCache that advances across chain calls, so cascading
           on target's hidden with the MTP cache doing the position
           refinement still produces distinct drafts; kept as-is for
           Qwen 3.5 which does not implement ``return_hidden`` on the
           MTP path yet.

        Args:
            hidden_last: [B, 1, H] backbone hidden state at the last
                confirmed position (main_tok's position).
            main_tok: the last confirmed backbone token (drives d_1).
            prev: rolling ``prev_tokens`` tensor for logits_processors,
                or None.
            K: chain length. MUST be >= 1; caller is responsible for
                skipping when the controller parked at K=0.
            cache_commit: optional first-call cache-commit tuple
                (see ``_step_mtp``). Only applied on the FIRST call in
                the chain — subsequent chain calls carry ``None``.

        Returns:
            Tuple of four Python lists, each of length ``K``:
            ``(draft_toks, draft_lps, draft_accept_lps, xtc_draws)``.
        """
        draft_toks: list = []
        draft_lps: list = []
        draft_accept_lps: list = []
        xtc_draws: list = []
        prev_tok = main_tok
        cur_hidden = hidden_last
        cur_commit = cache_commit
        for _k in range(K):
            d_tok, d_lp, d_alp, d_xtc, d_hidden = _step_mtp(
                cur_hidden,
                prev_tok,
                prev,
                cache_commit=cur_commit,
                want_hidden=_mtp_supports_hidden and K >= 2,
            )
            # Materialize before chaining — the next iteration needs
            # ``prev_tok.item()`` inside ``_step_mtp`` (via reshape,
            # not .item(), but the MLX graph needs the value pinned).
            mx.eval(d_tok)
            draft_toks.append(d_tok)
            draft_lps.append(d_lp)
            draft_accept_lps.append(d_alp)
            xtc_draws.append(d_xtc)
            prev_tok = d_tok
            # Drafter-hidden cascade: swap in the drafter's own
            # ``post_projection(h)`` for the next iteration's hidden
            # slot when available. Falls back to holding
            # ``hidden_last`` constant on injects that don't expose
            # ``return_hidden`` (Qwen 3.5 today).
            if d_hidden is not None:
                mx.eval(d_hidden)
                cur_hidden = d_hidden
            cur_commit = None
        return draft_toks, draft_lps, draft_accept_lps, xtc_draws

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
    # 0.9.13 PR-B Fix 3: state generalized from a scalar ``draft_tok``
    # to a list of pending drafts. ``pending_drafts`` is None when the
    # controller parked (K=0) or on the bootstrap primary-only step;
    # otherwise it's a list of ``(tok, lp, accept_lp, xtc_draw)``
    # tuples of length K, one per chained MTP draft.
    pending_drafts: list | None = None

    # ------------------------------------------------------------------
    # 0.9.13 PR-B: Ollama-style EV draft-K controller.
    #
    # When ``disable_auto_k=False`` (default), a per-model controller
    # picks K ∈ {0..max_k_effective} each round. K=0 "parks" — plain
    # decode, no drafter — which fixes the prose-slowdown regression
    # from PR-A K=1 (the drafter cost dominates on low-accept content).
    # When ``disable_auto_k=True``, the pre-PR-B chain-of-1 behavior
    # is preserved for A/B benching.
    #
    # 0.9.13 PR-B Fix 3: K≥2 chain-of-K lifted. The verify path now
    # accepts K sequential drafts with a single ``(K+1)``-position
    # backbone forward, per Ollama's ``speculate.go::accept`` batching.
    # The SSM-hybrid (ArraysCache) rollback is not compatible with the
    # per-position snapshot Ollama uses, so any model whose cache list
    # contains an SSM slot is clamped to K=1 at loop-start below —
    # chain-of-K on SSM targets needs the ``PrepareSnapshots([offsets])``
    # per-position machinery, which is a separate work item.
    # ------------------------------------------------------------------
    # SSM detection: patched ArraysCache carries a ``rollback_state``
    # class attribute (see ``cache_patch.py``); KVCache does not. This
    # is the cheapest, most stable class-level signal for the SSM path
    # available without importing the two cache classes here.
    _has_ssm_cache = any(hasattr(c, "rollback_state") for c in model_cache)
    if not disable_auto_k:
        # Chain-of-K on SSM targets not implemented; clamp to K=1 with
        # a startup log (once per generator instance is cheap enough
        # given ``mtp_generate_step`` is called per-request).
        _max_k_hw = 1 if _has_ssm_cache else max(0, max_k)
        if _has_ssm_cache and max_k > 1:
            logger.info(
                "[MTP-chain-of-K] SSM cache detected in model_cache — "
                "clamping max_k from %d to 1 (chain-of-K on SSM-hybrid "
                "targets needs per-position snapshots not yet wired). "
                "Set --mtp-max-k=1 to silence this log.",
                max_k,
            )
        max_k_effective = _max_k_hw
        _controller: DepthController | None = get_or_create_controller(
            model_id or "__default__", max_k=max_k_effective
        )
    else:
        # ``disable_auto_k`` keeps the pre-0.9.13 fixed-K=1 A/B-bench
        # behavior — no controller, no chain-of-K, verbatim chain-of-1.
        max_k_effective = 1
        _controller = None

    # next_k: the K the controller wants for the UPCOMING round. Determines
    # whether we generate a draft at end of the current round. Bootstrap
    # value is the controller's initial pick_k (0 if fresh, else the
    # scheduled depth from the previous request).
    next_k = _controller.pick_k() if _controller is not None else 1

    def _record_round(k_used: int, round_wall_ms: float, accepts: list[bool]) -> None:
        """Fold a round outcome into the controller (if enabled)."""
        if _controller is None:
            return
        _controller.record(k_used, round_wall_ms, accepts)

    while ntoks < max_tokens:
        round_start_perf = time.perf_counter()
        if pending_drafts is None:
            # -------------------------------------------------------
            # Round K=0 (either bootstrap or a park). Plain backbone
            # forward emits ONE committed token.
            # -------------------------------------------------------
            toks, lps, accept_lps, hidden, prev_tokens = _step_backbone(
                y, prev_tokens, n_predict=1
            )
            mx.eval(toks)
            main_tok, main_lp = toks[0], lps[0]
            round_wall_ms = (time.perf_counter() - round_start_perf) * 1000.0
            _record_round(0, round_wall_ms, [])

            ntoks += 1
            yield main_tok.item(), main_lp, False
            if ntoks >= max_tokens:
                return

            # Decide K for the NEXT round.
            next_k = _controller.pick_k() if _controller is not None else 1

            hidden_at_main = hidden[:, -1:, :]
            if next_k >= 1:
                # Chain-of-K: generate ``next_k`` drafts cascaded via
                # MTP. next_k==1 is the plain single-draft path.
                d_toks, d_lps, d_alps, d_xtcs = _step_mtp_chain(
                    hidden_at_main, main_tok, prev_tokens, next_k
                )
                pending_drafts = list(zip(d_toks, d_lps, d_alps, d_xtcs))
            else:
                # Parking again: no draft. Next round enters this
                # branch with ``pending_drafts is None`` and pays no
                # drafter cost — the whole point of park.
                pending_drafts = None
            y = mx.array([main_tok.item()], mx.uint32)
        else:
            # -------------------------------------------------------
            # Verify path with K = len(pending_drafts) drafts.
            #
            # 0.9.13 PR-C: single-sync verify. Ollama's
            # ``speculate.go:428-439`` pre-samples the residual at
            # EVERY draft position and the bonus at position K, then
            # batches them into ONE ``mx.eval`` alongside the accept
            # mask. The prior implementation issued two host syncs
            # (verify tokens first, then residual on the reject path);
            # merging them cuts one host round-trip per verify round
            # and — for K≥2 — replaces a per-position Python accept
            # loop with a single vectorized comparison.
            #
            # K=1: single verify + bonus (byte-equal to pre-PR-C at
            # greedy temp=0 because residual == verify-argmax there).
            # K>=2: batched (K+1)-position backbone forward, sequential
            # accept-reject per Ollama's ``speculate.go::accept``. The
            # SSM path is clamped at loop-start so K>=2 only reaches
            # this branch on pure-attention targets (KVCache.trim() is
            # the only rollback needed).
            # -------------------------------------------------------
            k_len = len(pending_drafts)
            draft_toks_arr = [rec[0] for rec in pending_drafts]
            draft_lps_arr = [rec[1] for rec in pending_drafts]
            draft_alps_arr = [rec[2] for rec in pending_drafts]
            first_xtc_draw = pending_drafts[0][3]

            # Assemble [y, d_1, ..., d_K] for the batched target forward.
            # Stacking on device (rather than materializing each draft
            # via ``.item()``) keeps the whole graph lazy up to the
            # single ``mx.eval`` below.
            drafts_arr = (
                mx.stack([d.reshape(-1) for d in draft_toks_arr])
                .reshape(-1)
                .astype(mx.uint32)
            )
            y_with_drafts = mx.concatenate([y, drafts_arr])

            toks, lps, accept_lps, hidden, prev_tokens = _step_backbone(
                y_with_drafts,
                prev_tokens,
                n_predict=k_len + 1,
                # n_confirmed = k_len only matters on SSM targets (which
                # are clamped to k_len=1 above). Passing k_len keeps the
                # semantics uniform: "the last k_len positions are
                # drafts, snapshot before them".
                n_confirmed=k_len,
                xtc_draw=first_xtc_draw,
            )

            # One shared uniform for all positions' probabilistic
            # accept tests. Ollama uses a per-position Bernoulli draw;
            # at greedy temp=0 the draw is ignored (accept iff argmax
            # match), so this only matters for temp>0 where the same
            # ``u`` biases all positions the same way — closer to
            # Ollama's per-position draw than reusing the sampler
            # chain's XTC cell would be.
            u = mx.random.uniform()
            drafts_i32 = drafts_arr.astype(mx.int32)

            # --------------------------------------------------------
            # Pre-compute accept-mask, residual-at-every-position, and
            # bonus-at-position-K on device. Single ``mx.eval`` below
            # materializes all four at once. Ollama's speculate.go:428-439.
            # --------------------------------------------------------
            if _is_greedy:
                # ``toks[i]`` IS target's argmax at position i. Accept
                # iff it matches the draft; residual == verify == toks[i]
                # (residual distribution at greedy is a point mass on
                # target's argmax, which coincides with target argmax).
                accept_mask_arr = toks[:k_len].astype(mx.int32) == drafts_i32
                residual_toks_arr = toks[:k_len]
                bonus_tok_arr = toks[k_len]
            else:
                # Vectorized per-position log-accept over the K draft
                # positions with a shared draw ``u``.
                v_alps = accept_lps[:k_len]  # (K, V)
                d_alps_stack = mx.stack(draft_alps_arr)  # (K, V)
                idx = drafts_i32.reshape(-1, 1)  # (K, 1)
                v_at = mx.take_along_axis(v_alps, idx, axis=1).squeeze(-1)
                d_at = mx.take_along_axis(d_alps_stack, idx, axis=1).squeeze(-1)
                log_accept = v_at - d_at  # (K,)
                accept_mask_arr = (log_accept >= 0) | (u < mx.exp(log_accept))

                # Residual distribution at every position — sampled
                # up-front so a reject at position i doesn't cost a
                # second eval. Formula matches the prior per-reject code:
                # ``max(p_target - p_draft, 0)``, falling back to
                # ``p_target`` if the max clamp zeroed everything.
                p_target = mx.exp(v_alps)  # (K, V)
                p_draft = mx.exp(d_alps_stack)  # (K, V)
                residual = mx.maximum(p_target - p_draft, 0.0)  # (K, V)
                z = residual.sum(axis=-1, keepdims=True)  # (K, 1)
                dist = mx.where(z > 0, residual, p_target)  # (K, V)
                residual_toks_arr = mx.random.categorical(mx.log(dist))
                # Bonus already sampled per-position inside _step_backbone
                # for temp>0 (categorical over target distro at position K).
                bonus_tok_arr = toks[k_len]

            # ------- SINGLE SYNC -------
            mx.eval(toks, accept_mask_arr, residual_toks_arr, bonus_tok_arr, u)

            # ------- Host-side read (all values already resident) -------
            accept_flags = accept_mask_arr.tolist()
            residual_ids = residual_toks_arr.tolist()
            bonus_id = int(bonus_tok_arr.item())
            draft_ids = drafts_arr.tolist()

            # Bump attempts by K (one per draft position considered).
            for _ in range(k_len):
                accept_counter.record_attempt()

            # Sequential accept-reject walk (host-only; no MLX ops).
            accepts: list[bool] = []
            accepted_count = 0
            for i in range(k_len):
                ok = bool(accept_flags[i])
                accepts.append(ok)
                if ok:
                    accepted_count += 1
                else:
                    break

            # -------- 0.9.13 PR-C: EOS holdout --------
            # When an accepted draft is a stop token, positions past it
            # would never be reached in real decode. Ollama's
            # ``speculate.go:456-472`` sets ``observed = i + 1`` and
            # ``keep = i`` at the EOS, so the acceptance-rate EWMA
            # never sees the (nonexistent) positions past the natural
            # terminator. We use the same idea: truncate ``accepts``
            # to the EOS index+1 for the record call; cap
            # ``accepted_count`` so we stop emitting at (and including)
            # EOS instead of continuing to a bonus or residual.
            eos_cut = False
            accepts_for_record = accepts
            if stop_tokens:
                for j in range(accepted_count):
                    if int(draft_ids[j]) in stop_tokens:
                        eos_cut = True
                        accepts_for_record = accepts[: j + 1]
                        accepted_count = j + 1
                        break

            round_wall_ms = (time.perf_counter() - round_start_perf) * 1000.0
            _record_round(k_len, round_wall_ms, accepts_for_record)

            # Emit the accepted drafts (capped at EOS position when set).
            for i in range(accepted_count):
                accept_counter.record_accept(tokens_saved=1)
                ntoks += 1
                yield int(draft_ids[i]), draft_lps_arr[i], True
                if ntoks >= max_tokens:
                    return

            if eos_cut:
                # Emitted EOS via an accepted draft. Caller will detect
                # the stop token and terminate; skip bonus / residual /
                # drafter-chain setup entirely. The cache is left with
                # the un-emitted drafts past EOS still committed to it,
                # but the request terminates here so the cache is
                # discarded by the scheduler at request boundary.
                return

            if accepted_count == k_len:
                # All K drafts accepted → emit the bonus token
                # (target's prediction one past the last draft).
                _clear_rollback()
                ntoks += 1
                yield bonus_id, lps[k_len], False
                if ntoks >= max_tokens:
                    return
                last_committed_tok_id = bonus_id
                last_committed_hidden = hidden[:, k_len : k_len + 1, :]
                y = mx.array([bonus_id], mx.uint32)
            else:
                # Reject at position ``accepted_count``. Emit target's
                # pre-sampled residual there (byte-equal to the prior
                # ``verify_pred.item()`` on greedy since residual ==
                # target argmax at temp=0), and drop the remaining
                # (k_len - accepted_count) unaccepted drafts from the
                # caches.
                n_to_drop = k_len - accepted_count
                _rollback_draft(n_to_drop)
                accept_counter.record_reject()
                if logits_processors and prev_tokens is not None:
                    # Discard the ``n_to_drop`` rejected positions
                    # from prev_tokens (they were appended by
                    # _step_backbone during the batched verify).
                    prev_tokens = prev_tokens[:-n_to_drop]

                # Also trim mtp_cache by the same n_to_drop — those
                # positions were appended by _step_mtp_chain and
                # correspond to the rejected drafts. The MTP KV
                # cache is per-layer KVCache (see qwen3_5_inject
                # make_mtp_cache / gemma4_inject) — always trimmable.
                for mc in mtp_cache:
                    if mc.is_trimmable():
                        mc.trim(n_to_drop)

                verify_tok_id = int(residual_ids[accepted_count])

                ntoks += 1
                yield verify_tok_id, lps[accepted_count], False
                if ntoks >= max_tokens:
                    return
                last_committed_tok_id = verify_tok_id
                # hidden at position ``accepted_count`` is the state
                # AFTER the last accepted draft (or after y when
                # accepted_count=0) — this is what MTP conditions on
                # for the next draft chain.
                last_committed_hidden = hidden[
                    :, accepted_count : accepted_count + 1, :
                ]
                y = mx.array([verify_tok_id], mx.uint32)

            # Decide K for the next round BEFORE generating the
            # next chain (a park decision skips drafter cost).
            next_k = _controller.pick_k() if _controller is not None else 1
            if next_k >= 1:
                # Chain-carry: on all-accept the mtp_cache must
                # advance by one extra position for the just-accepted
                # LAST draft so the head's attention sees it before
                # predicting the next round's first draft. The old
                # K=1 code did this via ``cache_commit`` on the
                # single _step_mtp call, which batched
                # ``(align_h=hidden_at_last_accepted_pre, align_tok=
                # accepted_draft, next_id=bonus_tok)`` into one
                # mtp_forward with 2 positions. We replicate here
                # only when this round was all-accept — on partial
                # accept the reject path already trims mtp_cache to
                # ``accepted_count`` positions and the residual
                # doesn't need a carry (its own hidden is what the
                # first chain call conditions on).
                if accepted_count == k_len:
                    # Position of last accepted draft is at index
                    # ``accepted_count - 1`` in the k+1-length hidden.
                    # For k_len=1 all-accept, this is hidden[:, 0:1].
                    align_h = hidden[:, accepted_count - 1 : accepted_count, :]
                    align_tok = draft_toks_arr[accepted_count - 1]
                    cache_commit = (align_h, align_tok)
                else:
                    cache_commit = None
                last_committed_tok = mx.array([last_committed_tok_id], mx.uint32)
                d_toks, d_lps, d_alps, d_xtcs = _step_mtp_chain(
                    last_committed_hidden,
                    last_committed_tok,
                    prev_tokens,
                    next_k,
                    cache_commit=cache_commit,
                )
                pending_drafts = list(zip(d_toks, d_lps, d_alps, d_xtcs))
            else:
                pending_drafts = None

        block = ntoks // _CACHE_CLEAR_INTERVAL
        if block > last_cache_block:
            mx.clear_cache()
            last_cache_block = block
