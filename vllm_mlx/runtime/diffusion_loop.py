# SPDX-License-Identifier: Apache-2.0
"""In-house diffusion generation loop for DiffusionGemma 26B-A4B.

Drop-in replacement for ``mlx_vlm.generate.diffusion.stream_diffusion_generate``.
Produces an iterator yielding ``RapidDiffusionResult`` instances whose
attribute shape matches what ``runtime/diffusion_lane.py`` expects from
mlx-vlm's own ``DiffusionGenerationResult`` (``getattr(result, 'X', …)``
sites at ``diffusion_lane.py:1229-1252``).

Why we wrote this:
  - mlx-vlm's loop uses entropy-bound adaptive stopping (median ~23
    denoising steps per canvas) which is conservative by ~3× on
    M3 Ultra and non-deterministic at temperature=0 (filed as
    mlx-vlm#1351).
  - With a fixed-step + per-step self-conditioning policy we hit
    127 tok/s on the canonical DiffusionGemma benchmark prompt (vs
    upstream's 30-46 tok/s baseline), at quality on par with
    upstream's ``max_denoising_steps=16`` setting — see quality eval
    ``research/diffusion-gemma/quality/eval-20260611-1001.md``.

Scope:
  - **Generation loop only.** We re-use mlx-vlm's loaded model — same
    weights, same transformer arch (``model.model.encoder`` for prefill,
    ``model.model.decoder._make_decoder_masks`` for attention masks,
    ``model(canvas_ids=…, self_conditioning_embeddings=…)`` for the
    denoising forward pass), same KV cache class. Only the loop body
    (canvas init, fixed-step denoising, self-conditioning cadence,
    EOS-aware multi-canvas advancement) is ours.

The handful of small generation utilities that live in mlx-vlm's
``generate/diffusion.py`` as ``_diffusion_initialize_canvas`` /
``_diffusion_soft_embedding_weight`` / ``_diffusion_soft_embeddings``
are inlined below — they're a few lines of mx math each, and importing
them under their ``_`` private name would couple us to mlx-vlm
internals (their loop calls them; ours is parallel). When mlx-vlm
refactors their loop we don't care, because we don't share the loop.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RapidDiffusionResult:
    """Per-canvas yield record. Attribute names + semantics match what
    ``diffusion_lane.py``'s ``stream_diffusion_generate`` consumer reads
    via ``getattr(result, …, default)``. New consumers should use this
    dataclass directly so the typing is explicit."""

    text: str
    token: int | None
    prompt_tokens: int
    generation_tokens: int
    diffusion_block_complete: bool
    finish_reason: str | None
    # ``is_draft`` is False for every yield we produce — we don't emit
    # mid-canvas previews. Kept for ``getattr(result, "is_draft", False)``
    # call-site symmetry with mlx-vlm's draft results.
    is_draft: bool = False


# =============================================================================
# Math helpers — vendored equivalents of mlx-vlm's diffusion utilities
# =============================================================================


def _initialize_canvas(batch_size: int, canvas_length: int, vocab_size: int,
                       dtype: Any) -> mx.array:
    """Start a denoising canvas as uniform random token ids in
    ``[0, vocab_size)`` — same convention as
    ``mlx_vlm.generate.diffusion._diffusion_initialize_canvas`` at
    mlx-vlm 0.6.3. DiffusionGemma's first denoising pass overwrites the
    whole canvas via self-attention on the prompt KV cache, so the
    initial values are noise, not zeros."""
    return mx.random.randint(0, vocab_size, (batch_size, canvas_length)).astype(dtype)


def _soft_embedding_weight(embed_layer: nn.Module) -> mx.array:
    """Return a float weight matrix usable as ``probs @ weight``.

    For 4-bit / quantized DiffusionGemma checkpoints the packed
    ``embed_layer.weight`` can't feed a regular matmul — we have to
    dequantize once. mlx-vlm does the same at
    ``_diffusion_soft_embedding_weight`` (mlx-vlm 0.6.3); their note:
    ``mx.quantized_matmul(..., transpose=False)`` is several times
    slower at this shape, so dequant-once is the cheaper path."""
    if isinstance(embed_layer, nn.QuantizedEmbedding):
        return mx.dequantize(
            embed_layer.weight,
            embed_layer.scales,
            embed_layer.biases,
            group_size=embed_layer.group_size,
            bits=embed_layer.bits,
        )
    return embed_layer.weight


def _sample_canvas(logits: mx.array, dtype: Any, temperature: float) -> mx.array:
    """Per-position argmax (greedy) or categorical sample (sampled).

    Vendored from ``mlx_vlm.generate.diffusion._diffusion_sample_canvas``
    at mlx-vlm 0.6.3. The semantics are:
      - ``temperature <= 0``: greedy argmax (deterministic at given seed).
      - ``temperature == 1.0``: categorical sample without rescaling.
      - ``temperature in (0, 1) or (1, inf)``: rescale logits then sample.

    Shape: ``[B, L, V]`` → ``[B, L]`` token ids in ``dtype``.

    Keeping this matched to upstream's behavior is important — chat
    clients (BCG, Big-AGI, etc.) default to ``temperature=0.7`` if the
    user hasn't overridden it. Diverging here would silently change
    sampled outputs vs. the mlx-vlm baseline."""
    logits_f32 = logits.astype(mx.float32)
    if temperature <= 0:
        return mx.argmax(logits_f32, axis=-1).astype(dtype)
    if temperature != 1.0:
        logits_f32 = logits_f32 / temperature
    return mx.random.categorical(logits_f32).astype(dtype)


def _soft_embeddings(logits: mx.array, embedding_weight: mx.array,
                     embed_scale: float) -> mx.array:
    """``softmax(logits) @ embedding_weight * embed_scale`` — feeds the
    next denoising pass as self-conditioning context.

    Mirrors ``_diffusion_soft_embeddings`` at mlx-vlm 0.6.3 (uses
    ``precise=True`` softmax to match upstream's numerics on the
    ``[B, canvas, vocab=262144]`` reduction; ordinary softmax produces
    visibly different self-conditioning input for the next step on
    DiffusionGemma).

    Shape: ``[B, canvas, vocab]`` @ ``[vocab, hidden]`` →
    ``[B, canvas, hidden]``, scaled."""
    probs = mx.softmax(logits, axis=-1, precise=True)
    return (
        probs.astype(embedding_weight.dtype) @ embedding_weight
    ).astype(embedding_weight.dtype) * embed_scale


# =============================================================================
# Main loop
# =============================================================================


def _extract_eos_ids(model_config: Any, tokenizer: Any) -> set[int]:
    """Collect EOS token ids from the model's ``generation_config`` and
    the tokenizer's ``eos_token_id`` field. We union both because
    DiffusionGemma's checkpoint splits the two between
    ``config.json::eos_token_id`` (single id) and
    ``generation_config.json::eos_token_id`` (list incl. ``<end_of_turn>``)."""
    ids: set[int] = set()

    gc = getattr(model_config, "generation_config", None)
    if isinstance(gc, dict):
        raw = gc.get("eos_token_id")
        if isinstance(raw, list):
            ids.update(int(x) for x in raw)
        elif isinstance(raw, int):
            ids.add(int(raw))

    cfg_eos = getattr(model_config, "eos_token_id", None)
    if isinstance(cfg_eos, list):
        ids.update(int(x) for x in cfg_eos)
    elif isinstance(cfg_eos, int):
        ids.add(int(cfg_eos))

    tok_eos = getattr(tokenizer, "eos_token_id", None)
    if isinstance(tok_eos, int):
        ids.add(int(tok_eos))

    return ids


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
    fixed_steps: int = 8,
    sc_every: int = 1,
    # Accept-and-ignore parameters that ``stream_diffusion_generate``
    # also takes — keeps the call site at ``diffusion_lane.py:1211`` from
    # having to discriminate between backends. Anything we don't honor:
    #   - ``temperature``: the rapid loop is greedy-only by design.
    #     ``diffusion_lane.py`` is responsible for routing non-zero
    #     temperature to the mlx-vlm backend before we get here.
    #   - ``diffusion_sampler``: argmax only; same routing rule.
    #   - ``max_denoising_steps``: ``fixed_steps`` is the source of truth.
    #   - ``prefill_step_size``: full prefill always (DiffusionGemma's
    #     prompts are short relative to the 4-bit prefill cost).
    temperature: float = 0.0,
    diffusion_sampler=None,
    max_denoising_steps: int | None = None,
    prefill_step_size: int | None = None,
) -> Iterator[RapidDiffusionResult]:
    """Yield one ``RapidDiffusionResult`` per completed canvas plus a
    final result with ``finish_reason='stop'`` or ``'length'``.

    Caller contract is the same as mlx-vlm's
    ``stream_diffusion_generate``: results with
    ``diffusion_block_complete=True`` carry the canvas's joined text;
    the terminal result carries ``finish_reason``."""

    if pixel_values is not None:
        # The text-diffusion lane is not supposed to land any
        # vision input. Fail loud rather than silently ignore — caller
        # has a bug.
        raise ValueError(
            "rapid_stream_diffusion_generate is text-only; "
            "pixel_values must be None"
        )
    # Note: ``temperature`` is honored per-step via ``_sample_canvas``.
    # 0.0 → greedy argmax (deterministic), >0 → categorical sample.
    # Matches mlx-vlm's ``_diffusion_sample_canvas`` semantics so chat
    # clients defaulting to temperature=0.7 (BCG, Big-AGI) get the same
    # output distribution as the upstream loop, at rapid-loop speed.

    config = model.config
    canvas_length = int(config.canvas_length)
    vocab_size = int(config.text_config.vocab_size)
    batch_size, prompt_length = input_ids.shape

    skip_ids = skip_special_token_ids if skip_special_token_ids is not None else set()
    eos_ids = _extract_eos_ids(config, tokenizer)

    decoder = model.model.decoder
    soft_emb_weight = _soft_embedding_weight(decoder.embed_tokens)
    embed_scale = decoder.embed_scale

    # Prefill — sets up KV cache as if the prompt had been seen.
    kv_cache = model.make_cache()
    _, kv_cache = model.model.encoder(input_ids, attention_mask=None, cache=kv_cache)
    mx.eval([c.state for c in kv_cache])

    generated: list[int] = []
    last_emitted_idx = 0  # length of ``generated`` already streamed out
    finish_reason: str | None = None
    last_token_id: int = 0

    while len(generated) < max_tokens:
        if finish_reason is not None:
            break

        remaining = max_tokens - len(generated)
        # Sized to the canvas (mask tokens that lap past the request
        # cap are clipped at emit time). Floor at 64 so a tiny
        # ``max_tokens`` doesn't shrink the canvas below DiffusionGemma's
        # minimum useful denoising width — short canvases under-denoise
        # (see quality eval: shortform 'scat shorter' token corruption).
        cur_canvas_len = min(canvas_length, max(remaining, 64))

        canvas = _initialize_canvas(batch_size, cur_canvas_len, vocab_size,
                                    input_ids.dtype)
        mask_mapping = decoder._make_decoder_masks(canvas[..., None], kv_cache, None)

        # Denoising loop — fixed_steps passes over the canvas. Each pass:
        #   1. Forward(canvas, sc_emb) -> logits [B, L, V]
        #   2. canvas = argmax(logits)
        #   3. Maybe recompute soft-embedding for next pass
        sc_emb: mx.array | None = None
        last_logits: mx.array | None = None
        for step in range(fixed_steps):
            if sc_emb is None:
                out = model(
                    cache=kv_cache,
                    canvas_ids=canvas,
                    self_conditioning_logits=None,
                    decoder_attention_mask=mask_mapping,
                )
            else:
                out = model(
                    cache=kv_cache,
                    canvas_ids=canvas,
                    self_conditioning_embeddings=sc_emb,
                    decoder_attention_mask=mask_mapping,
                )
            last_logits = out.logits
            canvas = _sample_canvas(last_logits, input_ids.dtype, float(temperature))
            mx.eval(canvas)

            # Compute self-conditioning for the NEXT step on the
            # configured cadence. The empirical optimum on
            # DiffusionGemma is sc_every=1 (every step); values >=2
            # collapse output quality (eval 2026-06-11).
            if step < fixed_steps - 1 and (step + 1) % sc_every == 0:
                sc_emb = _soft_embeddings(last_logits, soft_emb_weight, embed_scale)
                mx.eval(sc_emb)

        # Walk the finished canvas tokens, append to running buffer,
        # stop at EOS / max_tokens.
        canvas_tokens = [int(t) for t in canvas[0].tolist()]
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
            last_token_id = tok

        # Emit one result per completed canvas (or terminal piece).
        new_token_ids = [
            t for t in generated[last_emitted_idx:]
            if t not in skip_ids
        ]
        last_emitted_idx = len(generated)
        block_text = tokenizer.decode(new_token_ids) if new_token_ids else ""

        yield RapidDiffusionResult(
            text=block_text,
            token=last_token_id if generated else None,
            prompt_tokens=int(prompt_length),
            generation_tokens=len(generated),
            diffusion_block_complete=True,
            finish_reason=finish_reason,
            is_draft=False,
        )

        if canvas_stop:
            break

        # Extend KV cache with the accepted canvas tokens for the next
        # canvas — same call shape as mlx-vlm's loop does between
        # canvases. We pass the ORIGINAL canvas (not the skip-id
        # filtered view) so the cache state matches what the next
        # canvas's decoder mask expects.
        _, kv_cache = model.model.encoder(canvas, attention_mask=None, cache=kv_cache)
        mx.eval([c.state for c in kv_cache])

    # If we ran out of the while-loop without setting finish_reason
    # (e.g. exact max_tokens hit between canvases), emit a terminal
    # result so the consumer's "no finish_reason → synthetic stop"
    # branch at ``diffusion_lane.py:1290`` doesn't fire on top of our
    # already-terminal yield.
    if finish_reason is None:
        yield RapidDiffusionResult(
            text="",
            token=last_token_id if generated else None,
            prompt_tokens=int(prompt_length),
            generation_tokens=len(generated),
            diffusion_block_complete=True,
            finish_reason="length",
            is_draft=False,
        )


__all__ = [
    "RapidDiffusionResult",
    "rapid_stream_diffusion_generate",
]
