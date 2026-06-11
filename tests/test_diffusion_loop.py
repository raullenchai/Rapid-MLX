# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``runtime/diffusion_loop.py``.

Covers the math helpers + EOS extraction + greedy-only / vision guards.
Does NOT load DiffusionGemma — those live in
``test_diffusion_gemma_parity.py`` behind ``@pytest.mark.slow`` (run
with ``pytest --run-slow``).
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
    Codex round 2 [P1]: generator BODIES don't execute until iteration,
    so the previous version was a false-green — a future regression that
    added an upfront ``raise ValueError("temperature must be > 0")`` would
    silently pass. Drive the generator with ``next()`` so the body runs.
    ``model=None`` crashes downstream with some other error class once we
    reach the forward pass — that's fine; what matters is that the upfront
    validation does NOT reject the legal ``temperature=0.0`` value."""
    fake_input_ids = mx.array([[1, 2, 3]])
    gen = rapid_stream_diffusion_generate(
        model=None,
        processor=None,
        tokenizer=None,
        input_ids=fake_input_ids,
        temperature=0.0,
    )
    with pytest.raises(Exception) as exc_info:
        next(gen)
    # Body executed and crashed downstream (model=None); the upfront
    # validation did NOT raise ``ValueError`` against the legal value.
    assert not (
        isinstance(exc_info.value, ValueError)
        and "temperature" in str(exc_info.value).lower()
    ), f"Upfront validation rejected legal temperature=0.0: {exc_info.value}"


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


def test_rejects_zero_or_negative_fixed_steps():
    """Codex round 4 [P1]: programmatic callers (not just JSON loaders)
    can pass ``fixed_steps=0``. Without entry validation,
    ``denoise_budget`` resolves to ``0``, the for-loop body never runs,
    and ``_initialize_canvas`` random ids are emitted as model output.
    AliasProfile's loader already enforces ``>= 1`` on the JSON surface
    (``test_fixed_steps_must_be_positive_int``); this test pins the
    same contract on the runtime entry."""
    fake_input_ids = mx.array([[1, 2, 3]])
    with pytest.raises(ValueError, match="fixed_steps must be"):
        next(
            rapid_stream_diffusion_generate(
                model=None,
                processor=None,
                tokenizer=None,
                input_ids=fake_input_ids,
                fixed_steps=0,
            )
        )
    with pytest.raises(ValueError, match="fixed_steps must be"):
        next(
            rapid_stream_diffusion_generate(
                model=None,
                processor=None,
                tokenizer=None,
                input_ids=fake_input_ids,
                fixed_steps=-3,
            )
        )


def test_strict_int_validation_rejects_float_and_bool():
    """Codex round 5 [NIT]: ``int(x)`` silently truncates ``1.5`` and
    accepts ``True`` (bool is an int subclass). Strict validation mirrors
    ``model_aliases._coerce`` so the runtime entry refuses what the JSON
    loader refuses."""
    fake_input_ids = mx.array([[1, 2, 3]])
    for knob_name, knob_kwargs in [
        ("fixed_steps", {"fixed_steps": 1.5}),
        ("fixed_steps", {"fixed_steps": True}),
        ("max_denoising_steps", {"max_denoising_steps": 1.5}),
        ("max_denoising_steps", {"max_denoising_steps": True}),
        ("prefill_step_size", {"prefill_step_size": 1.5}),
        ("prefill_step_size", {"prefill_step_size": False}),
    ]:
        with pytest.raises(ValueError, match=f"{knob_name} must be"):
            next(
                rapid_stream_diffusion_generate(
                    model=None,
                    processor=None,
                    tokenizer=None,
                    input_ids=fake_input_ids,
                    **knob_kwargs,  # type: ignore[arg-type]
                )
            )


def test_stop_string_truncates_block_text():
    """Codex round 7 [P1]: rapid backend must honor request-level
    string stops. Use the fake-model framework from
    ``test_fixed_steps_and_adaptive_stop_are_not_mutually_exclusive``
    — stable peaky logits make the canvas predictable; we override
    the tokenizer to decode token ``5`` as the string ``"STOPHERE"``
    so the entire canvas decodes to ``"STOPHEREE...STOPHEREE"``.
    With ``stop=["STOPHERE"]`` the very first canvas should yield
    ``""`` (cut at position 0) and ``finish_reason="stop"``."""

    class _StopTok:
        all_special_ids: list[int] = []
        eos_token_id = None

        def decode(self, ids):  # type: ignore[no-untyped-def]
            # Each token 5 → "STOPHERE"; any other id → "?"
            return "".join("STOPHERE" if int(i) == 5 else "?" for i in ids)

    model = _StableLogitsFakeModel()
    tok = _StopTok()
    input_ids = mx.array([[1, 2, 3]])

    last = None
    for r in rapid_stream_diffusion_generate(
        model=model,
        processor=None,
        tokenizer=tok,
        input_ids=input_ids,
        fixed_steps=8,
        max_tokens=8,
        temperature=0.0,
        stop=["STOPHERE"],
    ):
        last = r
    assert last is not None
    assert last.finish_reason == "stop", (
        f"stop string should have terminated; got finish_reason={last.finish_reason!r}"
    )
    # The first canvas's block_text starts with "STOPHERE..." so the
    # cut is at index 0; emitted text should be empty.
    assert last.text == "", (
        f"text should be truncated at the stop position (got {last.text!r})"
    )


def test_stop_string_validation_rejects_non_string_list():
    """codex round 7 [P1]: invalid ``stop`` shape must fail loud, not
    silently no-op. Mirror the lane's ``_normalize_stops`` permissive
    contract (str / list[str] / None) but raise on other types."""
    fake_input_ids = mx.array([[1, 2, 3]])
    with pytest.raises(ValueError, match="stop must be"):
        next(
            rapid_stream_diffusion_generate(
                model=None,
                processor=None,
                tokenizer=None,
                input_ids=fake_input_ids,
                stop=42,  # type: ignore[arg-type]
            )
        )


def test_stop_string_validation_rejects_non_string_list_element():
    """codex round 8 [P1]: ``stop=[123]`` MUST raise — previously
    ``[s for s in stop if isinstance(s, str) and s]`` silently dropped
    the non-string element and the resulting empty list made the
    caller's stop request a no-op. Raise loud rather than coerce."""
    fake_input_ids = mx.array([[1, 2, 3]])
    with pytest.raises(ValueError, match="stop list elements must be strings"):
        next(
            rapid_stream_diffusion_generate(
                model=None,
                processor=None,
                tokenizer=None,
                input_ids=fake_input_ids,
                stop=[123],  # type: ignore[list-item]
            )
        )
    # Mixed list: even one bad element → raise (don't silently filter).
    with pytest.raises(ValueError, match="stop list elements must be strings"):
        next(
            rapid_stream_diffusion_generate(
                model=None,
                processor=None,
                tokenizer=None,
                input_ids=fake_input_ids,
                stop=["</answer>", 42, "</done>"],  # type: ignore[list-item]
            )
        )


def test_stop_string_wins_over_length_when_earlier():
    """codex round 8 [P1]: when a canvas decodes to text that BOTH
    contains a stop string AND hits the max_tokens cap on the same
    canvas, the earlier cut must win. Previously the ``if not
    canvas_stop`` guard let length-from-max_tokens win unconditionally,
    leaving the stop string un-trimmed in the emitted text."""

    class _StopTok:
        all_special_ids: list[int] = []
        eos_token_id = None

        def decode(self, ids):  # type: ignore[no-untyped-def]
            # Each token 5 → "STOPHERE"; canvas of [5,5,5,5] decodes to
            # "STOPHERESTOPHERESTOPHERESTOPHERE" — stop at index 0.
            return "".join("STOPHERE" if int(i) == 5 else "?" for i in ids)

    model = _StableLogitsFakeModel()
    tok = _StopTok()
    input_ids = mx.array([[1, 2, 3]])

    # max_tokens=4 matches canvas_length exactly, so the length cap fires
    # on the same canvas as the stop match. Without the r8 fix, length
    # wins and "STOPHERESTOPHERESTOPHERESTOPHERE" leaks out unfiltered.
    last = None
    full_text = ""
    for r in rapid_stream_diffusion_generate(
        model=model,
        processor=None,
        tokenizer=tok,
        input_ids=input_ids,
        fixed_steps=8,
        max_tokens=4,
        temperature=0.0,
        stop=["STOPHERE"],
    ):
        full_text += r.text
        last = r
    assert last is not None
    assert last.finish_reason == "stop", (
        f"stop should win over length when earlier; got {last.finish_reason!r}"
    )
    assert "STOPHERE" not in full_text, (
        f"stop string leaked into emitted text (got {full_text!r})"
    )


def test_stop_string_cross_canvas_holdback():
    """codex round 8 [P1]: a stop string that straddles a canvas
    boundary must still be enforced — we hold back ``max_stop_len - 1``
    chars from each non-terminal canvas so the next canvas can complete
    the match. Without the holdback, the first canvas's tail leaks
    verbatim and the consumer sees the stop string un-trimmed."""

    class _AlternatingTok:
        """Decode canvas N: first call returns ``"AAAA"``, every
        subsequent call returns ``"BBBB"``. The stop string ``"AB"``
        straddles the canvas boundary at position 3-4 of the full
        emitted prefix."""

        all_special_ids: list[int] = []
        eos_token_id = None

        def __init__(self) -> None:
            self.calls = 0

        def decode(self, ids):  # type: ignore[no-untyped-def]
            self.calls += 1
            return "AAAA" if self.calls == 1 else "BBBB"

    model = _StableLogitsFakeModel()
    tok = _AlternatingTok()
    input_ids = mx.array([[1, 2, 3]])

    full_text = ""
    last = None
    for r in rapid_stream_diffusion_generate(
        model=model,
        processor=None,
        tokenizer=tok,
        input_ids=input_ids,
        fixed_steps=8,
        max_tokens=8,
        temperature=0.0,
        stop=["AB"],
    ):
        full_text += r.text
        last = r

    assert last is not None
    assert last.finish_reason == "stop", (
        f"cross-canvas stop should terminate with finish_reason='stop'; "
        f"got {last.finish_reason!r}"
    )
    # Full emit should cut at position 3 ("AAA"), NOT leak the 'A' that
    # would have shipped with canvas 1 absent the holdback.
    assert full_text == "AAA", (
        f"cross-canvas stop should cut at position 3 (got {full_text!r}); "
        "an 'AAAA' or longer prefix means the holdback didn't fire."
    )
    assert "AB" not in full_text, (
        f"stop string 'AB' leaked into emitted text (got {full_text!r})"
    )


def test_stop_string_none_and_empty_are_noop():
    """``stop=None``, ``stop=""``, ``stop=[]``, ``stop=["", ""]`` must
    all behave as "no stop strings" — exercise the entry validation
    drops empty strings silently like the lane does."""
    fake_input_ids = mx.array([[1, 2, 3]])
    for empty_stop in (None, "", [], ["", ""]):
        # Should validate cleanly; downstream model=None crash is
        # expected and verifies entry validation passed.
        gen = rapid_stream_diffusion_generate(
            model=None,
            processor=None,
            tokenizer=None,
            input_ids=fake_input_ids,
            stop=empty_stop,  # type: ignore[arg-type]
        )
        with pytest.raises(Exception) as exc_info:
            next(gen)
        # The downstream crash must NOT be a stop-related ValueError.
        assert not (
            isinstance(exc_info.value, ValueError) and "stop" in str(exc_info.value)
        ), f"empty-stop variant {empty_stop!r} rejected at entry: {exc_info.value}"


def test_rejects_padded_inputs():
    """Codex round 6 [P1]: the per-canvas re-prefill grows the KV cache
    by ``current_canvas`` each iteration, but ``decoder_attention_mask``
    was only built from ``prompt + current_canvas`` lengths. On padded
    inputs the mask would diverge from cache length on canvas 2+, so
    attention silently looks at the wrong slots. The lane never pads
    today (B=1, no padding), so reject at entry until a multi-canvas
    + padding implementation lands."""
    fake_input_ids = mx.array([[1, 2, 3, 4]])
    # Mask with at least one False slot → has_padding = True → reject.
    padded_mask = mx.array([[True, True, False, False]])
    with pytest.raises(ValueError, match="padded inputs"):
        next(
            rapid_stream_diffusion_generate(
                model=None,
                processor=None,
                tokenizer=None,
                input_ids=fake_input_ids,
                attention_mask=padded_mask,
            )
        )


def test_rejects_invalid_sc_every():
    """Codex round 6 [P1]: ``sc_every`` gates
    ``cur_step % max(sc_every, 1)`` mid-loop. A float raises
    ``TypeError`` 32 steps into generation; ``0``/negative silently
    coerce to ``1`` via the ``max(...)`` guard. Mirror the same strict
    validation as the budget knobs."""
    fake_input_ids = mx.array([[1, 2, 3]])
    for bad_val in (None, 1.5, True, 0, -1):
        with pytest.raises(ValueError, match="sc_every must be"):
            next(
                rapid_stream_diffusion_generate(
                    model=None,
                    processor=None,
                    tokenizer=None,
                    input_ids=fake_input_ids,
                    sc_every=bad_val,  # type: ignore[arg-type]
                )
            )


def test_rejects_batch_size_greater_than_one():
    """Codex round 5 [NIT]: the canvas decode step reads
    ``current_canvas[0]`` only — B>1 inputs would silently drop all but
    the first row's output. Reject at entry until per-row streaming
    is implemented."""
    fake_input_ids_b2 = mx.array([[1, 2, 3], [4, 5, 6]])
    with pytest.raises(ValueError, match="B=1 only"):
        next(
            rapid_stream_diffusion_generate(
                model=None,
                processor=None,
                tokenizer=None,
                input_ids=fake_input_ids_b2,
            )
        )


def test_rejects_zero_or_negative_max_denoising_steps():
    """Codex round 4 [P1]: see ``test_rejects_zero_or_negative_fixed_steps``
    — same bug class on the ``max_denoising_steps`` knob, which is the
    accept-and-honor mirror of mlx-vlm's upstream signature. Per-request
    overrides also need entry validation."""
    fake_input_ids = mx.array([[1, 2, 3]])
    with pytest.raises(ValueError, match="max_denoising_steps must be"):
        next(
            rapid_stream_diffusion_generate(
                model=None,
                processor=None,
                tokenizer=None,
                input_ids=fake_input_ids,
                max_denoising_steps=0,
            )
        )
    with pytest.raises(ValueError, match="max_denoising_steps must be"):
        next(
            rapid_stream_diffusion_generate(
                model=None,
                processor=None,
                tokenizer=None,
                input_ids=fake_input_ids,
                max_denoising_steps=-2,
            )
        )


def test_accepts_nonzero_temperature():
    """``temperature>0`` is now legal: per-step argmax switches to
    ``mx.random.categorical`` inside ``_sample_canvas`` to match
    mlx-vlm's sampling semantics. Chat clients defaulting to
    ``temperature=0.7`` must reach the rapid path. Codex round 2 [P1]:
    same false-green class as ``test_accepts_explicit_zero_temperature``
    — drive ``next()`` so the generator body runs and we can assert the
    upfront guard does NOT reject ``temperature=0.7``."""
    fake_input_ids = mx.array([[1, 2, 3]])
    gen = rapid_stream_diffusion_generate(
        model=None,
        processor=None,
        tokenizer=None,
        input_ids=fake_input_ids,
        temperature=0.7,
    )
    with pytest.raises(Exception) as exc_info:
        next(gen)
    assert not (
        isinstance(exc_info.value, ValueError)
        and "temperature" in str(exc_info.value).lower()
    ), f"Upfront validation rejected legal temperature=0.7: {exc_info.value}"


# =============================================================================
# Hybrid mode — fixed_steps caps the budget, adaptive stop fires on top.
# =============================================================================


class _StableLogitsFakeModel:
    """Minimal fake model: every per-step forward returns peaky logits
    where token ``5`` dominates. argmax_canvas is the same on every
    step from step 1, so ``_stable_and_confident`` MUST fire on step 2
    (history has 1 entry from step 1, current matches). Counts forward
    calls so the test can assert the loop exited well before consuming
    the ``fixed_steps`` ceiling.

    Codex round 3 [P1]: this replaces the prior docstring-pin assertion
    which would have silently passed if production code re-introduced
    the v1/v2 mutually-exclusive ``adaptive_stop = (fixed_steps is None)``
    logic. With the fake model, that regression makes ``forward_calls``
    jump from ~2 to the full ceiling and the test fails loud.
    """

    class _TC:
        vocab_size = 16

    class _C:
        text_config: Any
        canvas_length = 4
        generation_config: dict[str, Any] = {
            # diffusion_stopping_config with low thresholds so
            # _stable_and_confident fires aggressively on peaky logits.
            "diffusion_stopping_config": {
                "stability_threshold": 1,
                "confidence_threshold": 0.5,
            },
            "sampler_config": {"entropy_bound": 1.0},
        }
        eos_token_id = None

    class _ET:
        # Regular (non-quantized) weight — _soft_embedding_weight returns
        # the .weight attribute directly. (vocab=16, dim=4)
        weight: mx.array

    class _Decoder:
        embed_scale = 1.0
        embed_tokens: Any

        @staticmethod
        def _make_decoder_masks(canvas_ids, kv_cache, decoder_attention_mask):
            # The loop only forwards this opaquely into model() as
            # ``decoder_attention_mask`` — our fake ignores it.
            return None

    class _Inner:
        decoder: Any

        @staticmethod
        def encoder(input_ids, attention_mask=None, cache=None):
            return None, cache or []

    def __init__(self) -> None:
        self.forward_calls = 0
        # Peaky logits — one-hot at token 5 with a huge spike so entropy
        # is ~0. Broadcast to (batch=1, canvas_length=4, vocab=16).
        spike = mx.where(
            mx.arange(16) == 5,
            mx.array(1000.0),
            mx.array(0.0),
        )
        self._logits = mx.broadcast_to(spike[None, None, :], (1, 4, 16))
        # Build the nested config/embed_tokens/decoder shape the loop
        # reads. Done in __init__ so the fake has stable per-instance
        # state (rather than class-level mutable defaults).
        tc = self._TC()
        cfg = self._C()
        cfg.text_config = tc
        et = self._ET()
        et.weight = mx.zeros((16, 4))
        dec = self._Decoder()
        dec.embed_tokens = et
        inner = self._Inner()
        inner.decoder = dec
        self.config = cfg
        self.model = inner

    def make_cache(self) -> list:
        class _Cache:
            state = mx.zeros((1,))

        return [_Cache()]

    def __call__(self, cache=None, canvas_ids=None, **kw):
        self.forward_calls += 1

        class _R:
            pass

        r = _R()
        r.logits = self._logits
        return r


class _FakeTokenizerForLoopTest:
    all_special_ids: list[int] = []
    eos_token_id = None

    def decode(self, ids):  # type: ignore[no-untyped-def]
        return "".join(chr(0x41 + (int(i) % 26)) for i in ids)


def test_fixed_steps_and_adaptive_stop_are_not_mutually_exclusive():
    """Codex round 3 [P1]: pin that adaptive early-stop fires INSIDE a
    large ``fixed_steps`` ceiling. With a fake model whose canvas
    stabilizes on step 1 (peaky logits → argmax is deterministic and
    identical every step), ``_stable_and_confident`` MUST fire on step 2
    of the denoising loop, regardless of how large ``fixed_steps`` is.

    If a future refactor re-introduces the v1/v2 mutually-exclusive
    behavior (``adaptive_stop = fixed_steps is None``), ``forward_calls``
    will balloon from ~2 to the full ceiling and this test fails loud.
    This is the structured-json regression the round-3 hybrid mode
    closed."""
    from vllm_mlx.runtime.diffusion_loop import rapid_stream_diffusion_generate

    model = _StableLogitsFakeModel()
    tok = _FakeTokenizerForLoopTest()
    input_ids = mx.array([[1, 2, 3]])

    # ``fixed_steps=64`` is the ceiling; if adaptive stop is disabled
    # the loop will burn all 64 forward passes per canvas. With it on,
    # the stable-logits fake exits at step 2.
    gen = rapid_stream_diffusion_generate(
        model=model,
        processor=None,
        tokenizer=tok,
        input_ids=input_ids,
        fixed_steps=64,
        max_tokens=4,  # one canvas-worth — single denoising pass
        temperature=0.0,
    )
    # Drain the generator so the loop runs to completion.
    list(gen)

    assert model.forward_calls < 10, (
        f"adaptive_stop should have fired ~step 2 of the ``fixed_steps=64`` "
        f"ceiling on stable peaky logits, but the loop made "
        f"{model.forward_calls} forward calls — the v1/v2 mutually-exclusive "
        f"regression (``adaptive_stop = fixed_steps is None``) is back."
    )
