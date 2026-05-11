# SPDX-License-Identifier: Apache-2.0
"""
Install path for combining dFlash block-diffusion drafting with the
existing MTP / continuous batching machinery.

`_install_dflash` monkey-patches a `BatchGenerator` so that each
generation step runs:

  1. Use skip_state from previous verify, OR run target forward with
     `capture_layer_ids = cfg.target_layer_ids` to obtain final logits,
     final hidden (post-norm), and the intermediate hidden states at
     the requested layers.
  2. Sample the primary token P from final logits.
  3. Run dFlash with the captured hidden -> greedy-decoded block of
     K_max candidate tokens.
  4. Verify [P, d_0, ..., d_{K-1}] in a single target forward (with
     capture). Greedy-accept the longest prefix where
     argmax(verify_logits[i]) == d_i.
  5. Skip_state for next step: stores verify (or rerun) logits +
     hidden + captured at the last cached position.

Notes
-----
* Only batch_size == 1. Multi-row requires padding + per-row accept.
* ngram drafting and the trailing MTP head are bypassed for the dFlash
  install path. The standard `_install_mtp` continues to handle the
  non-dFlash case.
* Acceptance is greedy. dFlash papers also use greedy verify, and it
  avoids the extra residual-sampling work.
"""

from __future__ import annotations

import logging
from typing import Any

import mlx.core as mx

from .dflash_drafter import DFlashConfig, DFlashDraftModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Target model accessors (target may be wrapped Outer or inner Qwen3-Next)
# ---------------------------------------------------------------------------


def _embed_tokens(model: Any, token_ids: mx.array) -> mx.array:
    if hasattr(model, "language_model"):
        return model.language_model.model.embed_tokens(token_ids)
    return model.model.embed_tokens(token_ids)


def _project_to_logits(model: Any, hidden: mx.array) -> mx.array:
    args = (
        model.language_model.args
        if hasattr(model, "language_model")
        else model.args
    )
    if getattr(args, "tie_word_embeddings", False):
        if hasattr(model, "language_model"):
            return model.language_model.model.embed_tokens.as_linear(hidden)
        return model.model.embed_tokens.as_linear(hidden)
    if hasattr(model, "language_model"):
        return model.language_model.lm_head(hidden)
    return model.lm_head(hidden)


def _cache_offset(prompt_cache) -> int:
    if not prompt_cache:
        return 0
    for c in prompt_cache:
        off = getattr(c, "offset", None)
        if off is not None:
            return int(off)
    return 0


def _trim_kv_layers(prompt_cache, n: int) -> None:
    if n <= 0 or not prompt_cache:
        return
    for c in prompt_cache:
        if hasattr(c, "is_trimmable") and c.is_trimmable() and hasattr(c, "trim"):
            c.trim(n)


def _snapshot_rnn(prompt_cache) -> dict[int, Any]:
    snaps: dict[int, Any] = {}
    if not prompt_cache:
        return snaps
    for ci, c in enumerate(prompt_cache):
        if hasattr(c, "is_trimmable") and c.is_trimmable():
            continue
        if hasattr(c, "state"):
            state = c.state
            copied = [mx.array(s) if s is not None else None for s in state]
            if isinstance(state, tuple):
                copied = tuple(copied)
            snaps[ci] = copied
    return snaps


def _restore_rnn(prompt_cache, snaps: dict[int, Any]) -> None:
    for ci, snap in snaps.items():
        prompt_cache[ci].state = snap


# ---------------------------------------------------------------------------
# dFlash drafting
# ---------------------------------------------------------------------------


def _dflash_draft_block(
    drafter: DFlashDraftModel,
    cfg: DFlashConfig,
    target_model: Any,
    captured: list[mx.array],
    captured_start_pos: int,
    skip_first_n: int,
    keep: int,
) -> mx.array:
    """Run dFlash and return greedy-decoded token IDs (B, keep).

    Slot semantics:
        - dflash_out[0] predicts position L+1 (overlaps the primary
          token sampled from the target).
        - dflash_out[1] predicts L+2. Empirically weak (least
          bidirectional lookahead) on this MLX port — the caller
          typically replaces this slot with an MTP-head draft.
        - dflash_out[i>=2] predict L+i+1 and tend to align with
          target argmax once context is non-trivial.

    `skip_first_n` controls how many leading dflash slots to drop.
    `keep` controls how many remaining slots to return.
    """
    block = cfg.block_size
    ctx_len = captured[0].shape[1]
    batch_size = captured[0].shape[0]

    mask_ids = mx.full(
        (batch_size, block), cfg.mask_token_id, dtype=mx.int32
    )
    noise_embed = _embed_tokens(target_model, mask_ids)

    target_hidden_concat = mx.concatenate(captured, axis=-1)

    last_ctx_pos = captured_start_pos + ctx_len - 1
    position_ids = mx.arange(
        captured_start_pos,
        last_ctx_pos + 1 + block,
        dtype=mx.int32,
    )
    position_ids = mx.broadcast_to(
        position_ids[None, :], (batch_size, ctx_len + block)
    )

    h = drafter(noise_embed, target_hidden_concat, position_ids)
    logits = _project_to_logits(target_model, h)
    drafts = mx.argmax(logits, axis=-1).astype(mx.int32)
    end = min(skip_first_n + keep, block)
    return drafts[:, skip_first_n:end]


# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------


def _install_dflash(
    batch_gen: Any,
    model: Any,
    drafter_path: str,
    num_drafts: int,
    target_temperature: float = 0.7,
    target_top_p: float = 0.9,
    target_min_p: float = 0.0,
    target_top_k: int = 0,
    optimistic: bool = False,
    stats: dict[str, int] | None = None,
) -> None:
    """Wire dFlash drafter into a BatchGenerator.

    Drafter weights are loaded LAZILY inside the worker thread that
    runs `_step`, so the resulting MLX arrays live on the worker's
    default stream and avoid the cross-thread "no Stream(gpu, N)"
    error MLX raises when arrays cross thread boundaries.
    """
    from .dflash_drafter import load_dflash_drafter
    from .native_mtp import (
        MTPDistributionParams,
        distribution_logprobs,
    )

    # Lazy-load: keep the path, materialise the drafter on first use
    # in the worker thread. Cache the loaded objects in a holder.
    _holder: dict[str, Any] = {"drafter": None, "cfg": None}

    def _ensure_drafter() -> tuple[Any, DFlashConfig]:
        if _holder["drafter"] is None:
            logger.info("[dFlash] lazy-loading drafter from %s", drafter_path)
            d, c = load_dflash_drafter(drafter_path)
            _holder["drafter"] = d
            _holder["cfg"] = c
            logger.info("[dFlash] drafter loaded")
        return _holder["drafter"], _holder["cfg"]

    # Probe the config eagerly so we can size things up-front. Loading
    # config.json is cheap and creates no MLX arrays.
    import json
    from pathlib import Path

    cfg_raw = json.loads((Path(drafter_path) / "config.json").read_text())
    cfg = DFlashConfig.from_dict(cfg_raw)

    target_params = MTPDistributionParams(
        temperature=target_temperature,
        top_p=target_top_p,
        min_p=target_min_p,
        top_k=target_top_k,
    )

    _stats = stats if stats is not None else {
        "accepted": 0,
        "rejected": 0,
        "errors": 0,
        "dflash_drafted": 0,
        "dflash_accepted": 0,
    }

    # Skip state: lets next step skip its target forward.
    # Format: {logits: (B, V), hidden: (B, 1, H),
    #          captured: list[(B, ctx_len, H)],
    #          captured_start_pos: int}  where captured_start_pos is
    #          the absolute target position of captured[:, 0, :].
    _skip_state: list[dict | None] = [None]
    _deferred_drafts: dict[int, list[dict]] = {}

    # Per-uid rolling captured-hidden window for multi-position drafter
    # context. dict[uid] -> (captured_list, start_pos).
    _ctx_window: dict[int, tuple[list[Any], int]] = {}

    # Cap at block_size - 1 because slot 0 of dFlash overlaps the
    # primary's position and is dropped (see `_dflash_draft_block`).
    K_max = max(1, min(int(num_drafts), cfg.block_size - 1))
    capture_ids = list(cfg.target_layer_ids)
    # Multi-position context window for the drafter (in addition to
    # the latest hidden). Drafter was trained with multi-position
    # bidirectional attention; ctx_len=1 is degenerate. Keep the most
    # recent CTX_WINDOW positions of captured hidden across steps.
    CTX_WINDOW = 64

    if hasattr(batch_gen, "_step"):
        raise NotImplementedError(
            "dFlash install requires mlx-lm 0.31+ GenerationBatch API."
        )

    generation_cls = batch_gen._generation_batch.__class__
    if not hasattr(generation_cls, "_rapid_dflash_orig_step"):
        generation_cls._rapid_dflash_orig_step = generation_cls._step
    _orig_step = generation_cls._rapid_dflash_orig_step

    def _capture_last(captured: list[mx.array]) -> list[mx.array]:
        return [c[:, -1:, :] for c in captured]

    def _append_ctx(uid: int, new_captured: list[mx.array], new_start_pos: int) -> None:
        """Append new positions to the rolling per-uid context window."""
        existing = _ctx_window.get(uid)
        if existing is None:
            _ctx_window[uid] = (
                [c[:, -CTX_WINDOW:, :] for c in new_captured],
                max(0, new_start_pos - max(0, new_captured[0].shape[1] - CTX_WINDOW)),
            )
            return
        old_captured, old_start = existing
        merged = []
        for o, n in zip(old_captured, new_captured):
            cat = mx.concatenate([o, n], axis=1)
            if cat.shape[1] > CTX_WINDOW:
                cat = cat[:, -CTX_WINDOW:, :]
            merged.append(cat)
        new_total_len = merged[0].shape[1]
        # The end position is unchanged (still at new last-position).
        end_pos = new_start_pos + new_captured[0].shape[1] - 1
        new_start = end_pos - new_total_len + 1
        _ctx_window[uid] = (merged, new_start)

    def _trim_ctx(uid: int, n: int) -> None:
        """Drop the last n positions from the per-uid context window."""
        existing = _ctx_window.get(uid)
        if existing is None or n <= 0:
            return
        old_captured, start = existing
        new_len = max(0, old_captured[0].shape[1] - n)
        if new_len == 0:
            _ctx_window.pop(uid, None)
            return
        trimmed = [c[:, :new_len, :] for c in old_captured]
        _ctx_window[uid] = (trimmed, start)

    def _ctx_for(uid: int) -> tuple[list[mx.array], int] | None:
        return _ctx_window.get(uid)

    def _dflash_generation_step(gen_self):
        if not gen_self.uids:
            return [], []

        gen_self._current_tokens = gen_self._next_tokens
        gen_self._current_logprobs = gen_self._next_logprobs
        input_tokens = gen_self._current_tokens
        batch_size = input_tokens.shape[0]

        # dFlash drafting only supports batch=1.
        if batch_size != 1:
            return _orig_step(gen_self)

        skip = _skip_state[0]
        if skip is not None and skip["logits"].shape[0] != batch_size:
            skip = None
            _skip_state[0] = None

        uid = gen_self.uids[0]

        if skip is None:
            try:
                model_out = model(
                    input_tokens[:, None],
                    cache=gen_self.prompt_cache,
                    capture_layer_ids=capture_ids,
                )
            except TypeError:
                # Model wasn't patched for capture_layer_ids — abort.
                return _orig_step(gen_self)
            if not isinstance(model_out, tuple) or len(model_out) != 3:
                return _orig_step(gen_self)
            logits_full, hidden_states, captured = model_out
            logits = logits_full[:, -1, :]
            cache_off = _cache_offset(gen_self.prompt_cache)
            target_pos = cache_off - 1
            new_input_len = captured[0].shape[1]
            new_start_pos = target_pos - new_input_len + 1
            _append_ctx(uid, captured, new_start_pos)
        else:
            logits = skip["logits"]
            hidden_states = skip["hidden"]
            target_pos = skip["target_pos"]
            _skip_state[0] = None

        # --- Logits processors on primary slot ---
        token_context = []
        if any(gen_self.logits_processors):
            token_context = [
                tc.update_and_fetch(input_tokens[i : i + 1])
                for i, tc in enumerate(gen_self._token_context)
            ]
            processed = []
            for e in range(batch_size):
                sl = logits[e : e + 1]
                for proc in gen_self.logits_processors[e]:
                    sl = proc(token_context[e], sl)
                processed.append(sl)
            logits = mx.concatenate(processed, axis=0)

        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        if any(gen_self.samplers):
            samples = []
            for e in range(batch_size):
                s = gen_self.samplers[e] or gen_self.fallback_sampler
                samples.append(s(logprobs[e : e + 1]))
            primary_tokens = mx.concatenate(samples, axis=0)
        else:
            primary_tokens = gen_self.fallback_sampler(logprobs)
        primary_logprobs = list(logprobs)

        try:
            # --- dFlash drafts (greedy block) + MTP-head slot 0 ---
            # Slot 0 of the verify draft block uses the MTP head's
            # prediction for position L+2 (target's own auxiliary
            # head). Slots 1..K_max-1 use dFlash predictions for
            # positions L+3..L+K_max+1. This complements MTP+dFlash:
            # MTP nails the immediate next-next token, dFlash extends
            # the block in parallel.
            drafter_obj, cfg_local = _ensure_drafter()
            ctx = _ctx_for(uid)
            if ctx is None:
                raise RuntimeError("no context window for uid")
            ctx_captured, ctx_start = ctx

            # MTP head prediction for L+2 (token after primary).
            mtp_token_arr: mx.array | None = None
            if hasattr(model, "mtp_forward"):
                try:
                    mtp_out = model.mtp_forward(
                        hidden_states[:, -1:, :],
                        primary_tokens[:, None],
                        mtp_cache=None,
                    )
                    if isinstance(mtp_out, tuple):
                        mtp_logits = mtp_out[0]
                    else:
                        mtp_logits = mtp_out
                    mtp_logits = mtp_logits[:, -1, :]
                    mtp_token_arr = mx.argmax(mtp_logits, axis=-1).astype(mx.int32)
                except Exception:
                    mtp_token_arr = None

            if mtp_token_arr is not None:
                # Reserve slot 0 for MTP. dFlash fills remaining slots
                # 1..K_max-1, drawn from dflash_out[2..K_max].
                dflash_keep = K_max - 1
                dflash_part = _dflash_draft_block(
                    drafter_obj, cfg_local, model,
                    ctx_captured, ctx_start,
                    skip_first_n=2,
                    keep=dflash_keep,
                )
                drafts = mx.concatenate(
                    [mtp_token_arr[:, None], dflash_part], axis=1
                )
            else:
                # No MTP head — fall back to pure dFlash slots 1..K_max.
                drafts = _dflash_draft_block(
                    drafter_obj, cfg_local, model,
                    ctx_captured, ctx_start,
                    skip_first_n=1,
                    keep=K_max,
                )
            mx.eval(drafts)

            # --- Snapshot RNN/SSM cache before verify in case we need
            # to roll back on partial accept ---
            rnn_snaps = _snapshot_rnn(gen_self.prompt_cache)

            # --- Verify [P, drafts...] ---
            verify_input = mx.concatenate(
                [primary_tokens[:, None], drafts], axis=1
            )
            verify_out = model(
                verify_input,
                cache=gen_self.prompt_cache,
                capture_layer_ids=capture_ids,
            )
            if not isinstance(verify_out, tuple) or len(verify_out) != 3:
                raise RuntimeError("verify did not return capture tuple")
            verify_logits, verify_hidden, verify_captured = verify_out

            target_argmax = mx.argmax(verify_logits, axis=-1).astype(mx.int32)
            mx.eval(target_argmax)
            target_arr = target_argmax[0]  # (K_max+1,)
            draft_arr = drafts[0]          # (K_max,)
            accept_mask = target_arr[:K_max] == draft_arr
            mx.eval(accept_mask)
            accept_list = accept_mask.tolist()
            accepted_count = 0
            for v in accept_list:
                if bool(v):
                    accepted_count += 1
                else:
                    break

            target_distribution = distribution_logprobs(
                verify_logits[0], target_params
            )

            deferred: list[dict] = []
            for d in range(accepted_count):
                deferred.append({
                    "token_array": drafts[0, d : d + 1],
                    "logprobs": target_distribution[d],
                })

            if accepted_count == K_max:
                # Full accept: cache already advanced through P + K_max
                # drafts. Skip_state from verify slot K_max (predicts
                # the position AFTER the last draft). Append the full
                # verify_captured to the rolling context window.
                consumed = 1 + K_max
                _append_ctx(uid, verify_captured, target_pos + 1)
                _skip_state[0] = {
                    "logits": verify_logits[:, K_max, :],
                    "hidden": verify_hidden[:, -1:, :],
                    "target_pos": target_pos + consumed,
                }
                mx.async_eval(
                    _skip_state[0]["logits"],
                    _skip_state[0]["hidden"],
                )
            else:
                # Partial / no accept. Roll back the verify's K_max+1
                # cache positions, restore RNN, then re-run a smaller
                # forward over [P, accepted_drafts..., correction] so
                # the cache + RNN state end up consistent with the
                # actually-emitted token stream.
                _trim_kv_layers(gen_self.prompt_cache, 1 + K_max)
                _restore_rnn(gen_self.prompt_cache, rnn_snaps)

                correction = target_arr[accepted_count : accepted_count + 1]
                deferred.append({
                    "token_array": correction,
                    "logprobs": target_distribution[accepted_count],
                })

                rerun_input = mx.concatenate(
                    [primary_tokens[:, None]]
                    + [drafts[:, d : d + 1] for d in range(accepted_count)]
                    + [correction[None, :]],
                    axis=1,
                )
                rerun_out = model(
                    rerun_input,
                    cache=gen_self.prompt_cache,
                    capture_layer_ids=capture_ids,
                )
                if not isinstance(rerun_out, tuple) or len(rerun_out) != 3:
                    raise RuntimeError("rerun did not return capture tuple")
                rerun_logits, rerun_hidden, rerun_captured = rerun_out
                _append_ctx(uid, rerun_captured, target_pos + 1)
                _skip_state[0] = {
                    "logits": rerun_logits[:, -1, :],
                    "hidden": rerun_hidden[:, -1:, :],
                    "target_pos": target_pos + rerun_input.shape[1],
                }
                mx.async_eval(
                    _skip_state[0]["logits"],
                    _skip_state[0]["hidden"],
                )

            _stats["accepted"] += accepted_count
            _stats["rejected"] += (K_max - accepted_count)
            _stats["dflash_drafted"] += K_max
            _stats["dflash_accepted"] += accepted_count

            uid = gen_self.uids[0]
            _deferred_drafts[uid] = deferred
        except Exception as e:
            if _stats["errors"] < 3:
                logger.warning("[dFlash] draft/verify failed: %s", e)
            else:
                logger.debug("[dFlash] draft/verify failed: %s", e)
            _stats["errors"] += 1
            _skip_state[0] = None
            for u in gen_self.uids:
                _deferred_drafts.pop(u, None)

        gen_self._next_tokens = primary_tokens
        gen_self._next_logprobs = primary_logprobs
        mx.async_eval(
            gen_self._next_tokens, gen_self._next_logprobs, token_context
        )
        mx.eval(input_tokens, gen_self._current_logprobs)

        tokens_out = input_tokens.tolist()
        for tl, t in zip(gen_self.tokens, tokens_out):
            tl.append(t)

        total = _stats["accepted"] + _stats["rejected"]
        if total > 0 and total % 100 == 0:
            print(
                f"[dFlash] stats accepted={_stats['accepted']} "
                f"rejected={_stats['rejected']} errors={_stats['errors']} "
                f"block_accept_rate="
                f"{100.0 * _stats['dflash_accepted'] / max(1, _stats['dflash_drafted']):.1f}% "
                f"avg_accept_per_block="
                f"{_stats['dflash_accepted'] / max(1, _stats['dflash_drafted'] / K_max):.2f}",
                flush=True,
            )
        return tokens_out, gen_self._current_logprobs

    batch_gen._inner_next = batch_gen._next
    generation_cls._step = _dflash_generation_step

    def _dflash_next_new(self=batch_gen):
        prev_deferred = {}
        gen_batch = self._generation_batch
        if len(gen_batch) == 0:
            _skip_state[0] = None
            _deferred_drafts.clear()
            _ctx_window.clear()
        if len(gen_batch) > 0:
            for uid in gen_batch.uids:
                if uid in _deferred_drafts:
                    prev_deferred[uid] = _deferred_drafts.pop(uid)

        prompt_responses, gen_responses = self._inner_next()
        if not prev_deferred or not gen_responses:
            return prompt_responses, gen_responses

        augmented = []
        draft_end_uids = set()
        gen_batch = self._generation_batch
        for response in gen_responses:
            augmented.append(response)
            uid = response.uid
            if response.finish_reason is not None or uid not in prev_deferred:
                continue
            items = prev_deferred.pop(uid)
            for item in items:
                draft_t = int(item["token_array"].item())
                draft_lp = item["logprobs"]
                finish_reason = None
                current_state = None
                match_sequence = None
                prompt_cache = None
                all_tokens = None
                if uid in gen_batch.uids:
                    e = gen_batch.uids.index(uid)
                    gen_batch._num_tokens[e] += 1
                    gen_batch.tokens[e].append(draft_t)
                    (
                        gen_batch._matcher_states[e],
                        match_sequence,
                        current_state,
                    ) = gen_batch.state_machines[e].match(
                        gen_batch._matcher_states[e], draft_t
                    )
                    if gen_batch._num_tokens[e] >= gen_batch.max_tokens[e]:
                        finish_reason = "length"
                    if match_sequence is not None and current_state is None:
                        finish_reason = "stop"
                    if finish_reason is not None:
                        prompt_cache = gen_batch.extract_cache(e)
                        all_tokens = gen_batch.tokens[e]
                        draft_end_uids.add(uid)
                augmented.append(
                    gen_batch.Response(
                        uid=uid,
                        token=draft_t,
                        logprobs=draft_lp,
                        finish_reason=finish_reason,
                        current_state=current_state,
                        match_sequence=match_sequence,
                        prompt_cache=prompt_cache,
                        all_tokens=all_tokens,
                    )
                )
                if finish_reason is not None:
                    break

        if draft_end_uids and len(gen_batch) > 0:
            for uid in draft_end_uids:
                _deferred_drafts.pop(uid, None)
                _ctx_window.pop(uid, None)
            keep = [
                i for i, uid in enumerate(gen_batch.uids)
                if uid not in draft_end_uids
            ]
            gen_batch.filter(keep)

        return prompt_responses, augmented

    batch_gen._next = _dflash_next_new
    logger.info(
        "[dFlash] installed: block_size=%d num_drafts=%d capture_layers=%s",
        cfg.block_size,
        num_drafts,
        cfg.target_layer_ids,
    )
