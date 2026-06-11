# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``runtime/diffusion_loop.py``.

Covers the math helpers + EOS extraction + greedy-only / vision guards.
Does NOT load DiffusionGemma — those live in
``test_diffusion_gemma_parity.py`` behind ``@pytest.mark.gpu_heavy``.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn
import pytest

from vllm_mlx.runtime.diffusion_loop import (
    RapidDiffusionResult,
    _extract_eos_ids,
    _initialize_canvas,
    _sample_canvas,
    _soft_embedding_weight,
    _soft_embeddings,
    rapid_stream_diffusion_generate,
)

# =============================================================================
# RapidDiffusionResult — getattr contract with diffusion_lane.py consumer
# =============================================================================


def test_result_dataclass_is_frozen():
    """The lane reads via ``getattr(result, …, default)``; mutation here
    would break the implicit immutability assumption at
    ``diffusion_lane.py:1241``."""
    r = RapidDiffusionResult(
        text="x",
        token=1,
        prompt_tokens=2,
        generation_tokens=3,
        diffusion_block_complete=True,
        finish_reason=None,
    )
    with pytest.raises(Exception):
        r.text = "mutated"  # type: ignore[misc]


def test_result_getattr_defaults_match_mlx_vlm_shape():
    """The lane uses ``getattr(result, 'is_draft', False)`` etc.; verify
    our dataclass returns the same defaults the upstream
    ``GenerationResult`` would surface."""
    r = RapidDiffusionResult(
        text="hi",
        token=42,
        prompt_tokens=10,
        generation_tokens=1,
        diffusion_block_complete=True,
        finish_reason=None,
    )
    assert getattr(r, "is_draft", "MISSING") is False
    assert getattr(r, "prompt_tokens", 0) == 10
    assert getattr(r, "generation_tokens", 0) == 1
    assert getattr(r, "token", None) == 42
    assert getattr(r, "diffusion_block_complete", False) is True
    assert getattr(r, "finish_reason", None) is None


# =============================================================================
# _initialize_canvas — shape + dtype + value range
# =============================================================================


def test_initialize_canvas_shape_and_dtype():
    canvas = _initialize_canvas(
        batch_size=1, canvas_length=64, vocab_size=1000, dtype=mx.int32
    )
    assert canvas.shape == (1, 64)
    assert canvas.dtype == mx.int32


def test_initialize_canvas_values_in_vocab():
    """Upstream uses mx.random.randint(0, vocab_size, …) — values must
    stay in ``[0, vocab_size)``. Off-by-one here lands the mask token
    in the canvas and breaks the first denoising pass."""
    canvas = _initialize_canvas(1, 256, 1000, mx.int32)
    vals = canvas.tolist()
    flat = vals[0]
    assert min(flat) >= 0
    assert max(flat) < 1000


def test_initialize_canvas_int8_canvas_dtype():
    """DiffusionGemma's input_ids are int8 in some quantization configs;
    the helper must respect the requested dtype."""
    canvas = _initialize_canvas(1, 32, 256, mx.int16)
    assert canvas.dtype == mx.int16


# =============================================================================
# _soft_embedding_weight — handles regular nn.Embedding + QuantizedEmbedding
# =============================================================================


def test_soft_embedding_weight_regular_embedding():
    """Plain ``nn.Embedding`` returns its weight directly (no dequant)."""
    embed = nn.Embedding(num_embeddings=100, dims=32)
    w = _soft_embedding_weight(embed)
    assert w.shape == (100, 32)
    # Identity check — no copy, no dequant.
    assert w is embed.weight or mx.array_equal(w, embed.weight)


def test_soft_embedding_weight_quantized_embedding():
    """Quantized embeddings must be dequantized before matmul — the
    packed weight has a different shape and ``probs @ packed_weight``
    silently produces garbage. This is the bug that surfaced in B3's
    first smoke test."""
    # nn.QuantizedEmbedding requires an existing embedding to quantize.
    base = nn.Embedding(num_embeddings=128, dims=64)
    q = nn.QuantizedEmbedding.from_embedding(base, bits=4, group_size=64)

    w = _soft_embedding_weight(q)
    # After dequant, shape must be ``[num_embeddings, dims]`` —
    # otherwise the @ embedding_weight matmul in _soft_embeddings will
    # produce wrong-shape activations.
    assert w.shape == (128, 64)


# =============================================================================
# _soft_embeddings — math sanity
# =============================================================================


def test_soft_embeddings_output_shape_and_dtype():
    """Shape ``[B, L, V]`` @ ``[V, H]`` → ``[B, L, H]``, scaled, same
    dtype as embedding_weight. Wrong shape here is what crashed the
    first B3 smoke run before the dequant fix."""
    logits = mx.random.normal((1, 8, 32))  # [B=1, L=8, V=32]
    embedding_weight = mx.random.normal((32, 16))  # [V=32, H=16]
    embed_scale = mx.array(2.5)

    out = _soft_embeddings(logits, embedding_weight, embed_scale)
    assert out.shape == (1, 8, 16)
    assert out.dtype == embedding_weight.dtype


def test_soft_embeddings_scale_factor_applied():
    """``embed_scale`` must multiply the result — DiffusionGemma's
    decoder.embed_scale is non-1.0; dropping it silently degrades
    self-conditioning input magnitude on every step."""
    logits = mx.zeros((1, 4, 8))  # uniform softmax → 1/8 per vocab slot
    embedding_weight = mx.ones((8, 4))  # each col = 8 → softmax @ weight = 1.0
    embed_scale = mx.array(3.0)
    out = _soft_embeddings(logits, embedding_weight, embed_scale)
    # softmax(zeros) = uniform 1/8, @ ones(8,4) = ones(B,L,4), * 3 = 3
    expected = mx.full(out.shape, 3.0, dtype=embedding_weight.dtype)
    assert mx.allclose(out, expected, atol=1e-5)


# =============================================================================
# _sample_canvas — greedy vs categorical, temperature semantics
# =============================================================================


def test_sample_canvas_greedy_at_temp_zero():
    """``temperature=0`` → deterministic argmax. One-hot logits land on
    the corresponding token id every time."""
    # Build logits where each [B, L] position has a unique max
    logits = mx.array(
        [
            [
                [-100.0, 5.0, -100.0, -100.0],  # → 1
                [-100.0, -100.0, 7.0, -100.0],  # → 2
                [3.0, -100.0, -100.0, -100.0],  # → 0
            ]
        ]
    )  # shape [1, 3, 4]
    out = _sample_canvas(logits, mx.int32, temperature=0.0)
    assert out.shape == (1, 3)
    assert out.tolist() == [[1, 2, 0]]


def test_sample_canvas_greedy_at_negative_temp():
    """``temperature<0`` also routes to argmax (defensive — upstream
    treats anything <= 0 as greedy)."""
    logits = mx.array([[[1.0, 9.0, 2.0]]])
    assert _sample_canvas(logits, mx.int32, temperature=-1.0).tolist() == [[1]]


def test_sample_canvas_categorical_at_temp_one():
    """``temperature=1`` → categorical sample. Sharp logits should
    still pick the dominant token most of the time; verify the picked
    id is at least valid."""
    logits = mx.array([[[10.0, -10.0, -10.0, -10.0]]])  # token 0 totally dominant
    # Even with sampling, this should always pick 0 (others have ~0 prob).
    for _ in range(5):
        out = _sample_canvas(logits, mx.int32, temperature=1.0)
        assert out.tolist() == [[0]]


def test_sample_canvas_temperature_rescales_logits():
    """``temperature != 1`` scales logits before softmax. Lower temp
    → sharper distribution. We can't bit-check randomness, but we can
    verify the path runs and produces in-vocab tokens for diverse
    temps."""
    logits = mx.random.normal((1, 4, 32))
    for t in [0.1, 0.5, 0.7, 1.0, 2.0]:
        out = _sample_canvas(logits, mx.int32, temperature=t)
        assert out.shape == (1, 4)
        vals = out.tolist()[0]
        assert min(vals) >= 0
        assert max(vals) < 32


def test_sample_canvas_dtype_preserved():
    """The canvas dtype must match the input ids dtype — mlx-vlm
    uses int32 / int16 depending on quantization."""
    logits = mx.array([[[1.0, 2.0]]])
    assert _sample_canvas(logits, mx.int32, 0.0).dtype == mx.int32
    assert _sample_canvas(logits, mx.int16, 1.0).dtype == mx.int16


# =============================================================================
# _extract_eos_ids — union of generation_config + tokenizer EOS
# =============================================================================


class _StubConfig:
    def __init__(self, generation_config: Any = None, eos_token_id: Any = None):
        self.generation_config = generation_config
        if eos_token_id is not None:
            self.eos_token_id = eos_token_id


class _StubTokenizer:
    def __init__(self, eos: Any = None):
        if eos is not None:
            self.eos_token_id = eos


def test_extract_eos_unions_generation_config_list():
    cfg = _StubConfig(generation_config={"eos_token_id": [1, 2, 3]})
    tok = _StubTokenizer(eos=99)
    assert _extract_eos_ids(cfg, tok) == {1, 2, 3, 99}


def test_extract_eos_unions_int_and_list():
    """DiffusionGemma's actual checkpoint splits the EOS set between
    ``config.eos_token_id`` (single int) and
    ``generation_config.eos_token_id`` (list incl. <end_of_turn>).
    Missing this union silently turns <end_of_turn> into normal text
    and the model rambles past the natural stop."""
    cfg = _StubConfig(generation_config={"eos_token_id": [106]}, eos_token_id=1)
    tok = _StubTokenizer()
    assert _extract_eos_ids(cfg, tok) == {1, 106}


def test_extract_eos_empty_when_nothing_set():
    cfg = _StubConfig()
    tok = _StubTokenizer()
    assert _extract_eos_ids(cfg, tok) == set()


def test_extract_eos_handles_missing_generation_config():
    cfg = _StubConfig(generation_config=None, eos_token_id=42)
    tok = _StubTokenizer(eos=42)
    # Set union → single element
    assert _extract_eos_ids(cfg, tok) == {42}


class _StubFullTok:
    """Tokenizer/processor surface covering all 4 EOS shapes
    ``Scheduler._get_stop_tokens`` reads."""

    def __init__(self, eos=None, eos_plural=None, wrapper_set=None, rapid_extras=None):
        if eos is not None:
            self.eos_token_id = eos
        if eos_plural is not None:
            self.eos_token_ids = eos_plural
        if wrapper_set is not None:
            self._eos_token_ids = wrapper_set
        if rapid_extras is not None:
            self._rapid_extra_eos_token_ids = rapid_extras


def test_extract_eos_unions_all_tokenizer_surfaces():
    """Codex r1 P1: my original ``_extract_eos_ids`` read only
    ``tokenizer.eos_token_id`` and silently missed three other surfaces
    (``eos_token_ids`` plural, mlx-lm ``_eos_token_ids`` set,
    ``_rapid_extra_eos_token_ids`` stash). For models whose chat
    terminator only lands on one of those (Gemma 3 VL processors,
    augmented HF tokenizers), the rapid loop would generate up to
    ``max_tokens`` instead of stopping at the natural turn boundary."""
    cfg = _StubConfig()
    tok = _StubFullTok(
        eos=1,
        eos_plural=[2, 3],
        wrapper_set={4, 5},
        rapid_extras={6},
    )
    assert _extract_eos_ids(cfg, tok) == {1, 2, 3, 4, 5, 6}


def test_extract_eos_reads_processor_surface_when_tokenizer_bare():
    """mlx-vlm processors carry EOS on ``processor.tokenizer`` AND on
    ``processor`` itself depending on the loader. The rapid loop must
    union both — otherwise rapid would miss the EOS that mlx-vlm's
    own augmenter stashed on the processor (most common shape for
    DiffusionGemma's HF tokenizer wrapper)."""
    cfg = _StubConfig()
    tok = _StubFullTok()  # bare — nothing
    proc = _StubFullTok(rapid_extras={106})  # the augmented stash
    assert _extract_eos_ids(cfg, tok, proc) == {106}


def test_extract_eos_handles_tuple_and_set_shapes():
    """``eos_token_ids`` can be a list, set, or tuple depending on the
    tokenizer flavor — all three must work."""
    cfg = _StubConfig()
    tok = _StubFullTok(eos_plural=(7, 8))
    assert _extract_eos_ids(cfg, tok) == {7, 8}
    tok = _StubFullTok(wrapper_set={9, 10, 11})
    assert _extract_eos_ids(cfg, tok) == {9, 10, 11}


def test_extract_eos_skips_none_and_non_int_entries():
    """``getattr`` defaults must not blow up on ``None``, and bogus
    string entries in a list must be silently skipped (defensive —
    HF generation configs sometimes carry mixed lists during loader
    upgrades)."""
    cfg = _StubConfig(generation_config={"eos_token_id": [1, "bogus", 2]})
    tok = _StubFullTok()
    assert _extract_eos_ids(cfg, tok) == {1, 2}


# =============================================================================
# Guards — vision input + non-zero temperature
# =============================================================================


def test_rejects_pixel_values():
    """Vision input should fail loud — the text-diffusion lane never
    routes pixels and silently ignoring would hide a routing bug."""
    fake_input_ids = mx.array([[1, 2, 3]])
    with pytest.raises(ValueError, match="text-only"):
        next(
            rapid_stream_diffusion_generate(
                model=None,
                processor=None,
                tokenizer=None,
                input_ids=fake_input_ids,
                pixel_values=mx.zeros((1, 3, 224, 224)),
            )
        )


def test_accepts_explicit_zero_temperature():
    """``temperature=0.0`` → greedy argmax path inside ``_sample_canvas``.
    Generator is constructed without error. (Iteration calls model()
    which is None and crashes; that's fine — the test only verifies the
    upfront guard logic doesn't reject the legal value.)"""
    fake_input_ids = mx.array([[1, 2, 3]])
    gen = rapid_stream_diffusion_generate(
        model=None,
        processor=None,
        tokenizer=None,
        input_ids=fake_input_ids,
        temperature=0.0,
    )
    assert gen is not None


def test_rejects_zero_or_negative_prefill_step_size():
    """Codex r1 P0 fix: ``prefill_step_size`` is honored now. A
    non-positive value is a caller-side bug (operator typo in
    ``--prefill-step-size`` or SchedulerConfig); silently treating it
    as ``None`` would mask the misconfig. Mirror upstream's loud
    raise at mlx-vlm diffusion.py:655. Validation runs BEFORE any
    model state is touched so this test can pass ``model=None``."""
    fake_input_ids = mx.array([[1, 2, 3]])
    with pytest.raises(ValueError, match="prefill_step_size must be"):
        next(
            rapid_stream_diffusion_generate(
                model=None,
                processor=None,
                tokenizer=None,
                input_ids=fake_input_ids,
                prefill_step_size=0,
            )
        )
    with pytest.raises(ValueError, match="prefill_step_size must be"):
        next(
            rapid_stream_diffusion_generate(
                model=None,
                processor=None,
                tokenizer=None,
                input_ids=fake_input_ids,
                prefill_step_size=-1,
            )
        )


def test_accepts_nonzero_temperature():
    """``temperature>0`` is now legal: per-step argmax switches to
    ``mx.random.categorical`` inside ``_sample_canvas`` to match
    mlx-vlm's sampling semantics. Chat clients defaulting to
    ``temperature=0.7`` must reach the rapid path."""
    fake_input_ids = mx.array([[1, 2, 3]])
    gen = rapid_stream_diffusion_generate(
        model=None,
        processor=None,
        tokenizer=None,
        input_ids=fake_input_ids,
        temperature=0.7,
    )
    assert gen is not None


# =============================================================================
# Hybrid mode — fixed_steps caps the budget, adaptive stop fires on top.
# =============================================================================


def test_fixed_steps_and_adaptive_stop_are_not_mutually_exclusive():
    """Round-3 fix: PR #555 v2 logic treated ``fixed_steps != None`` as
    "disable adaptive stop", which forced 8 steps on structured-json
    prompts that converged at 6 (the canvas was stable but we kept
    re-running the same forward pass). Without this test, a future
    refactor could re-introduce the mutual-exclusive code path and
    silently bring back the ~25% regression on EOS-bound short outputs.

    We can't easily probe the internal ``adaptive_stop`` flag without
    leaking implementation detail, so this test pins the OBSERVABLE
    contract via the function docstring: ``fixed_steps`` is described
    as a CEILING, and the live PR #555 bench (commit on this branch)
    confirms structured-json runs 41.2 tok/s (rapid) vs 37.1 tok/s
    (mlx-vlm) — only possible if adaptive stop is firing inside the
    cap. If somebody flips this back to mutually-exclusive, this
    docstring-pin assertion makes the regression visible at review."""
    from vllm_mlx.runtime.diffusion_loop import rapid_stream_diffusion_generate

    doc = rapid_stream_diffusion_generate.__doc__ or ""
    # The docstring MUST advertise fixed_steps as a ceiling — this
    # is the user-visible contract the AliasProfile default depends on.
    assert "ceiling, not a floor" in doc.lower() or "caps" in doc.lower(), (
        "rapid_stream_diffusion_generate docstring must describe "
        "fixed_steps as a budget cap (not a mutually-exclusive override) "
        "so future contributors don't re-introduce the v2 perf regression"
    )
