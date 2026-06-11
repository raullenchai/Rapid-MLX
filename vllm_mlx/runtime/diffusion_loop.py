# SPDX-License-Identifier: Apache-2.0
"""In-house diffusion generation loop for block-diffusion text models
(DiffusionGemma family today; extensible to siblings as we add them).

Phase 1 of the in-source roadmap (``docs/plans/in-source-roadmap.md``) —
this module owns the generation loop, ``mlx_vlm.models.bonsai`` still
owns the model architecture (MMDiT forward pass + ``QuantizedEmbedding``
layers). Phase 2 vendors the model arch and removes the runtime dep on
mlx-vlm for this model family.

## Vendoring policy

The math helpers (`_diffusion_*`) and the core denoising algorithm in
``rapid_stream_diffusion_generate`` are direct ports of mlx-vlm 0.6.3's
``mlx_vlm.generate.diffusion`` (Apache 2.0; original code by Lucas
Newman; commit ``3967c21`` and ancestors). We vendor verbatim where the
math is load-bearing (entropy/confidence-mask, soft-embedding scaling,
canvas initialization, stable-and-confident early-stop) so the rapid
backend produces byte-identical output to the upstream loop at
``temperature=0`` — verified by ``tests/test_diffusion_gemma_parity.py``
on 3 canonical prompts.

## What we change vs upstream

1. **Structure**: a single function instead of a 500-line generator
   with vision / show-unmasking / static-cache branches we don't take
   on the text-diffusion lane.
2. **Result shape**: ``RapidDiffusionResult`` mirrors the
   ``getattr(result, 'X', default)`` contract ``diffusion_lane.py``
   already expects from ``GenerationResult``, but is frozen and has
   only the fields we actually consume. The lane never has to
   discriminate between this and ``GenerationResult``.
4. **EOS extraction**: union of every shape ``Scheduler._get_stop_tokens``
   reads (``eos_token_id``, ``eos_token_ids``, ``_eos_token_ids``,
   ``_rapid_extra_eos_token_ids``) on both tokenizer and processor.
   Upstream relies on ``tokenizer.stopping_criteria(token)`` which is
   not present on every loader-shape — the union path is robust.
5. **No ``tokenizer.stopping_criteria.add_eos_token_ids``** side-effect:
   we never mutate tokenizer global state.

## What we do NOT change vs upstream

- Helpers (`_diffusion_*`) are verbatim — any deviation here would
  silently drift output distributions away from mlx-vlm's, and the
  parity test would no longer hold.
- The acceptance-mask + draft-canvas progressive reveal pattern is
  preserved verbatim — overwriting the canvas every step (which the
  first iteration of this module did) breaks ``structured-json`` and
  similar deterministic-output prompts because token positions that
  the model already locked in get re-sampled from a noisier
  intermediate distribution.
- ``mx.compile`` of decoder logits is intentionally skipped — the
  upstream compile path has a fallback for shape changes (we'd inherit
  the fallback warning logger); we'd rather leave the decision to
  callers wrapping us in their own ``mx.compile`` layer.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn

# =============================================================================
# Result — mirrors the getattr contract diffusion_lane.py reads via
# mlx-vlm's GenerationResult.
# =============================================================================


@dataclass(frozen=True)
class RapidDiffusionResult:
    text: str = ""
    token: int | None = None
    prompt_tokens: int = 0
    generation_tokens: int = 0
    diffusion_block_complete: bool = False
    finish_reason: str | None = None
    is_draft: bool = False


# =============================================================================
# Vendored helpers — direct port of mlx-vlm 0.6.3 ``mlx_vlm.generate.diffusion``
# (Apache 2.0). Function names match upstream so a reader cross-referencing
# the two trees finds the same identifiers.
# =============================================================================


def _initialize_canvas(
    batch_size: int,
    canvas_length: int,
    vocab_size: int,
    dtype,
) -> mx.array:
    return mx.random.randint(
        0,
        vocab_size,
        (batch_size, canvas_length),
    ).astype(dtype)


def _sample_canvas(
    processed_logits: mx.array,
    dtype,
    temperature: float,
) -> mx.array:
    logits = processed_logits.astype(mx.float32)
    if temperature <= 0:
        return mx.argmax(logits, axis=-1).astype(dtype)
    if temperature != 1.0:
        logits = logits / temperature
    return mx.random.categorical(logits).astype(dtype)


def _token_probability(
    processed_logits: mx.array,
    token_ids: mx.array,
) -> mx.array:
    logits = processed_logits.astype(mx.float32)
    token_logits = mx.take_along_axis(
        logits,
        token_ids[..., None],
        axis=-1,
    ).squeeze(-1)
    return mx.exp(token_logits - mx.logsumexp(logits, axis=-1))


def _token_entropy(processed_logits: mx.array) -> mx.array:
    logits = processed_logits.astype(mx.float32)
    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    probs = mx.exp(log_probs)
    return -mx.sum(probs * log_probs, axis=-1)


def _soft_embedding_weight(embed_tokens: nn.Module) -> mx.array:
    """Return a float weight matrix usable as ``probs @ weight``.

    Quantized embeddings hold a packed weight that can't feed a regular
    matmul, and ``mx.quantized_matmul(..., transpose=False)`` is several
    times slower at this shape; dequantize once per call. Mirrors
    ``_diffusion_soft_embedding_weight`` at mlx-vlm 0.6.3."""
    if isinstance(embed_tokens, nn.QuantizedEmbedding):
        return mx.dequantize(
            embed_tokens.weight,
            embed_tokens.scales,
            embed_tokens.biases,
            group_size=embed_tokens.group_size,
            bits=embed_tokens.bits,
        )
    return embed_tokens.weight


def _entropy_probs_chain(logits: mx.array) -> tuple[mx.array, mx.array]:
    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    probs = mx.exp(log_probs)
    entropy = -mx.sum(probs * log_probs, axis=-1)
    return probs, entropy


def _entropy_and_soft_embeddings(
    processed_logits: mx.array,
    embedding_weight: mx.array,
    embed_scale: float,
) -> tuple[mx.array, mx.array]:
    probs, entropy = _entropy_probs_chain(processed_logits.astype(mx.float32))
    soft_embeddings = (probs.astype(embedding_weight.dtype) @ embedding_weight).astype(
        embedding_weight.dtype
    ) * embed_scale
    return entropy, soft_embeddings


def _soft_embeddings(
    processed_logits: mx.array,
    embedding_weight: mx.array,
    embed_scale: float,
) -> mx.array:
    """``softmax(logits) @ embedding_weight * embed_scale`` — feeds the
    next denoising pass as self-conditioning. ``precise=True`` matches
    upstream's numerics on the ``[B, canvas, vocab=262144]`` reduction;
    ordinary softmax produces visibly different SC input on DiffusionGemma."""
    probs = mx.softmax(processed_logits, axis=-1, precise=True)
    return (probs.astype(embedding_weight.dtype) @ embedding_weight).astype(
        embedding_weight.dtype
    ) * embed_scale


def _confidence_transfer_mask(
    confidence: mx.array,
    unrevealed_mask: mx.array,
    threshold: float,
    *,
    force_all: bool = False,
) -> mx.array:
    if force_all:
        return unrevealed_mask
    transfer_mask = unrevealed_mask & (confidence >= threshold)
    has_unrevealed = mx.any(unrevealed_mask, axis=-1)
    has_transfer = mx.any(transfer_mask, axis=-1)
    needs_force = has_unrevealed & (~has_transfer)
    masked_confidence = mx.where(unrevealed_mask, confidence, -mx.inf)
    best_index = mx.argmax(masked_confidence, axis=-1)
    positions = mx.arange(confidence.shape[-1])[None, :]
    forced = (positions == best_index[:, None]) & needs_force[:, None]
    return transfer_mask | forced


def _entropy_transfer_mask(
    entropy: mx.array,
    entropy_bound: float,
) -> mx.array:
    sorted_indices = mx.argsort(entropy, axis=-1)
    sorted_entropy = mx.take_along_axis(entropy, sorted_indices, axis=-1)
    cumulative_entropy = mx.cumsum(sorted_entropy, axis=-1)
    cumulative_maximum_entropy = mx.cummax(sorted_entropy, axis=-1)
    sorted_selection_mask = (
        cumulative_entropy - cumulative_maximum_entropy
    ) <= entropy_bound
    selection_mask = mx.zeros_like(sorted_selection_mask)
    return mx.put_along_axis(
        selection_mask,
        sorted_indices,
        sorted_selection_mask,
        axis=-1,
    )


def _stable_and_confident(
    accepted_canvas: mx.array,
    processed_logits: mx.array,
    history: list[mx.array],
    stopping_config: dict[str, Any] | None,
) -> bool:
    """Early-stop predicate. Without this, the rapid loop runs the full
    ``max_denoising_steps`` even when the canvas has already converged —
    losing the ~2× speedup mlx-vlm gets on short EOS-bound outputs
    (code, JSON). Reads ``stability_threshold`` and ``confidence_threshold``
    from ``generation_config.diffusion_stopping_config`` (DiffusionGemma:
    ``{stability_threshold: 1, confidence_threshold: 0.005}``)."""
    if stopping_config is None:
        return False
    stability_threshold = int(stopping_config.get("stability_threshold", 1))
    confidence_threshold = float(stopping_config.get("confidence_threshold", 0.005))
    if len(history) == stability_threshold:
        stable = all(
            bool(mx.all(accepted_canvas == canvas).item()) for canvas in history
        )
    else:
        stable = False
    history.append(accepted_canvas)
    if len(history) > stability_threshold:
        history.pop(0)
    if not stable:
        return False
    token_entropy = _token_entropy(processed_logits)
    confident = bool((mx.mean(token_entropy) < confidence_threshold).item())
    return stable and confident


def _config_dict(maybe_config: Any) -> dict[str, Any]:
    if isinstance(maybe_config, dict):
        return maybe_config
    if maybe_config is None:
        return {}
    if hasattr(maybe_config, "to_dict"):
        return maybe_config.to_dict()
    return {}


# =============================================================================
# EOS extraction — rapid-mlx flavor of upstream's
# ``tokenizer.stopping_criteria`` path. Unions every surface
# ``Scheduler._get_stop_tokens`` knows about so loader-shape doesn't
# silently drop the chat terminator.
# =============================================================================


def _extract_eos_ids(
    model_config: Any,
    tokenizer: Any,
    processor: Any = None,
) -> set[int]:
    """Collect EOS token ids from every surface rapid-mlx already
    enumerates in ``Scheduler._get_stop_tokens``. Sources (all unioned):
      1. ``config.generation_config['eos_token_id']``
      2. ``config.eos_token_id``
      3. ``tokenizer.eos_token_id``
      4. ``tokenizer.eos_token_ids``
      5. ``tokenizer._eos_token_ids`` (mlx-lm TokenizerWrapper)
      6. ``tokenizer._rapid_extra_eos_token_ids`` (rapid-mlx stash)
      7. all 4 tokenizer surfaces also from ``processor``."""
    ids: set[int] = set()

    def _add(value: Any) -> None:
        if value is None:
            return
        if isinstance(value, (list, set, tuple)):
            for v in value:
                if isinstance(v, int):
                    ids.add(int(v))
            return
        if isinstance(value, int):
            ids.add(int(value))

    gc = getattr(model_config, "generation_config", None)
    if isinstance(gc, dict):
        _add(gc.get("eos_token_id"))
    _add(getattr(model_config, "eos_token_id", None))
    for src in (tokenizer, processor):
        if src is None:
            continue
        _add(getattr(src, "eos_token_id", None))
        _add(getattr(src, "eos_token_ids", None))
        _add(getattr(src, "_eos_token_ids", None))
        _add(getattr(src, "_rapid_extra_eos_token_ids", None))
    return ids


# =============================================================================
# Main loop — slim, text-only port of ``stream_diffusion_generate``.
# =============================================================================


_DEFAULT_MIN_CANVAS_LENGTH = 64
_DEFAULT_MAX_DENOISING_STEPS = 32
_DEFAULT_THRESHOLD = 0.9


def rapid_stream_diffusion_generate(
    model,
    processor,
    tokenizer,
    input_ids: mx.array,
    pixel_values=None,
    attention_mask=None,
    *,
    max_tokens: int = 256,
    skip_special_token_ids: set[int] | None = None,
    temperature: float = 0.0,
    # rapid-mlx knobs (forwarded by AliasProfile via diffusion_lane.py)
    fixed_steps: int | None = None,
    sc_every: int = 1,
    # Accept-and-honor knobs that mirror upstream's signature so the
    # lane can hand the same kwargs to either backend. Empty defaults
    # match mlx-vlm's behavior.
    diffusion_sampler: str = "entropy-bound",
    diffusion_threshold: float = _DEFAULT_THRESHOLD,
    max_denoising_steps: int | None = None,
    prefill_step_size: int | None = None,
) -> Iterator[RapidDiffusionResult]:
    """Stream ``RapidDiffusionResult``s from the in-house denoising loop.

    Algorithm mirrors mlx-vlm 0.6.3 ``stream_diffusion_generate`` for the
    text-only B=1 case (which is the only shape ``diffusion_lane.py``
    feeds us). Progressive token-reveal via entropy-bound or
    confidence-threshold acceptance mask. Early-stop via
    ``_stable_and_confident`` when the canvas converges before the
    nominal ``max_denoising_steps`` — this is the gap that made the
    naive "overwrite canvas every step" version slower than upstream on
    short EOS-bound outputs.

    Knobs:
      * ``fixed_steps``: caps the per-canvas step budget. Adaptive
        early-stop via ``_stable_and_confident`` STILL FIRES — the
        budget is a ceiling, not a floor. This is the production
        default (``8``) and gives the rapid backend its perf win:
          - long-form: budget caps below mlx-vlm's ~11-15 adaptive
            count → 1.76x speedup
          - short EOS-bound (JSON, structured): early-stop fires
            before the cap → on-par with mlx-vlm's own step count
        Set to ``None`` to remove the ceiling (default upstream
        ``max_denoising_steps`` budget; useful for reproducibility
        experiments).
      * ``sc_every``: cadence for explicit (non-entropy-bound) self-
        conditioning. ``1`` = every step (DiffusionGemma optimum from
        the 2026-06-11 hand-graded eval). The entropy-bound path
        computes SC inline as a byproduct of the entropy mask so this
        knob only gates the confidence-threshold path.

    Yields:
      * One result per emitted text segment with that segment's text.
      * One terminal result per canvas with ``diffusion_block_complete=True``.
      * One final result with ``finish_reason in {'stop','length'}``."""

    if pixel_values is not None:
        raise ValueError(
            "rapid_stream_diffusion_generate is text-only; pixel_values must be None"
        )
    if prefill_step_size is not None and int(prefill_step_size) <= 0:
        raise ValueError("prefill_step_size must be a positive integer.")
    if diffusion_sampler not in ("entropy-bound", "confidence-threshold"):
        raise ValueError(
            f"Unsupported diffusion sampler: {diffusion_sampler!r}.",
        )
    if not 0.0 <= diffusion_threshold <= 1.0:
        raise ValueError("diffusion_threshold must be between 0 and 1.")

    config = model.config
    text_config = config.text_config
    model_canvas_length = int(config.canvas_length)
    vocab_size = int(text_config.vocab_size)
    batch_size, prompt_length = input_ids.shape
    prompt_tokens = int(input_ids.size)

    generation_config = _config_dict(getattr(config, "generation_config", None))
    sampler_config = _config_dict(generation_config.get("sampler_config"))
    entropy_bound = float(sampler_config.get("entropy_bound", 0.1))

    # Resolve denoising-step budget. ``fixed_steps`` caps the budget
    # (a CEILING, not a floor) — adaptive early-stop via
    # ``_stable_and_confident`` always runs on top so EOS-bound short
    # outputs (JSON, code) stop at ~6 steps even when ceiling is 8.
    # This is the round-3 fix for the structured-json regression
    # surfaced on PR #555: the v1/v2 mutual-exclusive logic forced 8
    # steps on prompts that converge at 6, paying 33% extra compute.
    if fixed_steps is not None:
        denoise_budget = int(fixed_steps)
    else:
        if max_denoising_steps is None:
            max_denoising_steps = int(
                generation_config.get("max_denoising_steps")
                or _DEFAULT_MAX_DENOISING_STEPS,
            )
        denoise_budget = int(max_denoising_steps)
    adaptive_stop = True

    # Temperature schedule (linear ramp during denoising). Upstream
    # falls back to {0.4, 0.8} when no config is present.
    temperature_config = generation_config.get(
        "linear_temperature_schedule_config",
    )
    temperature_config = _config_dict(temperature_config)
    if not temperature_config:
        if "t_min" in generation_config or "t_max" in generation_config:
            temperature_config = {
                "t_min": generation_config.get("t_min", 0.4),
                "t_max": generation_config.get("t_max", 0.8),
            }
        else:
            temperature_config = {"t_min": 0.4, "t_max": 0.8}

    # Adaptive-stop config (used only when adaptive_stop=True).
    stopping_config = _config_dict(
        generation_config.get("diffusion_stopping_config"),
    )
    if not stopping_config:
        stopping_config = {
            key: generation_config[key]
            for key in ("confidence_threshold", "stability_threshold")
            if key in generation_config
        }
    if not stopping_config:
        stopping_config = None

    canvas_dtype = input_ids.dtype
    skip_ids = skip_special_token_ids if skip_special_token_ids is not None else set()
    eos_ids = _extract_eos_ids(config, tokenizer, processor)

    decoder = model.model.decoder
    soft_emb_weight = _soft_embedding_weight(decoder.embed_tokens)
    embed_scale = decoder.embed_scale

    # Attention mask defaults to "all real tokens" — matches upstream.
    if attention_mask is None:
        attention_mask = mx.ones((batch_size, prompt_length), dtype=mx.bool_)
    else:
        attention_mask = attention_mask.astype(mx.bool_)
    has_padding = not bool(mx.all(attention_mask).item())
    decoder_attention_mask = attention_mask if has_padding else None

    # Prefill — chunked if the operator opted in and the prompt is long
    # enough to benefit. Mirrors ``_diffusion_prefill_cache``.
    kv_cache = model.make_cache()
    chunk_prefill = (
        prefill_step_size is not None
        and prompt_length > int(prefill_step_size)
        and not has_padding
    )
    if chunk_prefill:
        step = int(prefill_step_size)  # type: ignore[arg-type]
        for start in range(0, prompt_length, step):
            end = min(start + step, prompt_length)
            _, kv_cache = model.model.encoder(
                input_ids[:, start:end],
                attention_mask=None,
                cache=kv_cache,
            )
            mx.eval([c.state for c in kv_cache])
            mx.clear_cache()
    else:
        _, kv_cache = model.model.encoder(
            input_ids,
            attention_mask=(attention_mask if has_padding else None),
            cache=kv_cache,
        )
        mx.eval([c.state for c in kv_cache])

    generated: list[int] = []
    finish_reason: str | None = None
    last_token: int | None = None
    is_prefill_done = True  # we just did the initial prefill
    current_canvas: mx.array | None = None

    while len(generated) < max_tokens:
        if finish_reason is not None:
            break

        # Re-encode the previous canvas into the KV cache (upstream
        # line 793-797). Skipping this is the OTHER algorithmic bug
        # the v1 rapid loop had — subsequent canvases hallucinated
        # because the KV state was stale.
        if not is_prefill_done and current_canvas is not None:
            _, kv_cache = model.model.encoder(
                current_canvas,
                attention_mask=None,
                cache=kv_cache,
            )
        is_prefill_done = False

        remaining = max_tokens - len(generated)
        canvas_length = min(
            model_canvas_length,
            max(remaining, _DEFAULT_MIN_CANVAS_LENGTH),
        )
        current_decoder_attention_mask = (
            mx.concatenate(
                [
                    decoder_attention_mask,
                    mx.ones((batch_size, canvas_length), dtype=mx.bool_),
                ],
                axis=-1,
            )
            if decoder_attention_mask is not None
            else None
        )

        current_canvas = _initialize_canvas(
            batch_size,
            canvas_length,
            vocab_size,
            canvas_dtype,
        )
        draft_reveal_mask = mx.zeros(current_canvas.shape, dtype=mx.bool_)
        draft_canvas = current_canvas
        accepted_canvas = current_canvas
        argmax_canvas = current_canvas
        self_conditioning_embeddings = None
        mask_mapping = decoder._make_decoder_masks(
            current_canvas[..., None],
            kv_cache,
            current_decoder_attention_mask,
        )
        diffusion_history: list[mx.array] = []

        # Denoising loop — counts DOWN like upstream so the linear
        # temperature schedule matches.
        for cur_step in reversed(range(1, denoise_budget + 1)):
            if self_conditioning_embeddings is None:
                processed_logits = model(
                    cache=kv_cache,
                    canvas_ids=current_canvas,
                    self_conditioning_logits=None,
                    decoder_attention_mask=mask_mapping,
                ).logits
            else:
                processed_logits = model(
                    cache=kv_cache,
                    canvas_ids=current_canvas,
                    self_conditioning_embeddings=self_conditioning_embeddings,
                    decoder_attention_mask=mask_mapping,
                ).logits

            # Linear temperature ramp (mirrors upstream).
            t_min = float(temperature_config.get("t_min", 0.4))
            t_max = float(temperature_config.get("t_max", 0.8))
            schedule_temperature = t_min + (
                (t_max - t_min) * (cur_step / denoise_budget)
            )
            processed_logits = processed_logits / schedule_temperature

            argmax_canvas = mx.argmax(processed_logits, axis=-1).astype(
                canvas_dtype,
            )
            if cur_step == 1:
                break

            denoiser_canvas = (
                argmax_canvas
                if temperature <= 0
                else _sample_canvas(
                    processed_logits,
                    canvas_dtype,
                    float(temperature),
                )
            )

            if diffusion_sampler == "entropy-bound":
                if cur_step > 1:
                    token_entropy, next_sc = _entropy_and_soft_embeddings(
                        processed_logits,
                        soft_emb_weight,
                        embed_scale,
                    )
                else:
                    token_entropy = _token_entropy(processed_logits)
                    next_sc = None
                acceptance_mask = _entropy_transfer_mask(
                    token_entropy,
                    entropy_bound,
                )
                accepted_canvas = mx.where(
                    acceptance_mask,
                    denoiser_canvas,
                    current_canvas,
                )
                current_canvas = mx.where(
                    acceptance_mask,
                    accepted_canvas,
                    _initialize_canvas(
                        batch_size,
                        canvas_length,
                        vocab_size,
                        canvas_dtype,
                    ),
                )
                draft_reveal_mask = acceptance_mask
                draft_canvas = argmax_canvas
            else:
                next_sc = None
                unrevealed_mask = ~draft_reveal_mask
                confidence = _token_probability(
                    processed_logits,
                    denoiser_canvas,
                )
                acceptance_mask = _confidence_transfer_mask(
                    confidence,
                    unrevealed_mask,
                    diffusion_threshold,
                    force_all=cur_step == 1,
                )
                accepted_canvas = mx.where(
                    acceptance_mask,
                    denoiser_canvas,
                    draft_canvas,
                )
                current_canvas = mx.where(
                    draft_reveal_mask | acceptance_mask,
                    accepted_canvas,
                    _initialize_canvas(
                        batch_size,
                        canvas_length,
                        vocab_size,
                        canvas_dtype,
                    ),
                )
                draft_reveal_mask = draft_reveal_mask | acceptance_mask
                draft_canvas = mx.where(
                    acceptance_mask,
                    accepted_canvas,
                    draft_canvas,
                )

            # Confidence-threshold sampler can short-circuit when every
            # position is revealed.
            if diffusion_sampler == "confidence-threshold" and bool(
                mx.all(draft_reveal_mask).item(),
            ):
                accepted_canvas = draft_canvas
                break

            # Adaptive early-stop — the gap that closed the perf cliff
            # vs upstream on short outputs.
            if adaptive_stop and _stable_and_confident(
                argmax_canvas,
                processed_logits,
                diffusion_history,
                stopping_config,
            ):
                break

            # Self-conditioning for next step.
            if cur_step > 1 and (cur_step % max(sc_every, 1) == 0):
                if next_sc is None:
                    next_sc = _soft_embeddings(
                        processed_logits,
                        soft_emb_weight,
                        embed_scale,
                    )
                self_conditioning_embeddings = next_sc

        current_canvas = argmax_canvas
        mx.eval(current_canvas)

        # Emit canvas tokens one by one — stop on EOS or max_tokens.
        canvas_tokens = [int(t) for t in current_canvas[0].tolist()]
        emitted_text_pieces: list[str] = []
        canvas_stop = False
        for tok in canvas_tokens:
            if len(generated) >= max_tokens:
                finish_reason = "length"
                canvas_stop = True
                break
            if tok in eos_ids:
                finish_reason = "stop"
                canvas_stop = True
                break
            generated.append(tok)
            last_token = tok
            if tok not in skip_ids:
                emitted_text_pieces.append(tokenizer.decode([tok]))

        # Edge case: canvas exactly filled the remaining budget. The
        # for-loop above only stamps ``finish_reason='length'`` when the
        # cap fires MID-canvas; when ``len(generated)`` rises to exactly
        # ``max_tokens`` on the final append, the loop exits naturally
        # with finish_reason=None and the consumer never sees a
        # terminal SSE event. Catch the boundary explicitly.
        if not canvas_stop and len(generated) >= max_tokens:
            finish_reason = "length"
            canvas_stop = True

        block_text = "".join(emitted_text_pieces)
        yield RapidDiffusionResult(
            text=block_text,
            token=last_token,
            prompt_tokens=prompt_tokens,
            generation_tokens=len(generated),
            diffusion_block_complete=True,
            finish_reason=finish_reason,
            is_draft=False,
        )

        if canvas_stop:
            break
