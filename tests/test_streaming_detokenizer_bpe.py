# SPDX-License-Identifier: Apache-2.0
"""Regression tests for D-DETOK-BPE: GPT-2 byte-level BPE mojibake leak.

Bug: ``mlx-community/DeepSeek-R1-*`` distills ship a
``tokenizer.json`` whose ``decoder`` chain is the Llama SentencePiece
chain (``Replace("▁", " ")`` -> ``ByteFallback`` -> ``Fuse`` ->
``Strip``) while the *vocab* is GPT-2 byte-level. Every
``tokenizer.decode([byte_level_id])`` then leaks the raw pretty
form (``Ġ``, ``Ċ``, ``âĢľ``, ``Â°`` …) instead of inverting it back to
the original byte sequence.

This test suite locks the contract on three surfaces:

1. ``repair_byte_level_decoder`` swaps the broken decoder in place.
2. The ``IncrementalDecoder`` (used by ``Scheduler`` for streaming
   ``delta.reasoning_content`` / ``delta.content``) emits clean UTF-8.
3. Cumulative ``tokenizer.decode(ids)`` (used by ``/v1/completions[0]
   .text`` and non-stream ``content`` / ``reasoning_content``) is
   clean.

The fixture builds a synthetic byte-level BPE tokenizer that mirrors
the malformation — no model download required, runs in <100 ms in CI.
"""

from __future__ import annotations

import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from vllm_mlx.utils.decode import IncrementalDecoder
from vllm_mlx.utils.tokenizer import (
    _BYTE_LEVEL_MOJIBAKE_MARKERS,
    repair_byte_level_decoder,
)

# Synthetic byte-level vocab that reproduces the bug deterministically.
# Each id maps to its byte-level pretty token; the "ground truth" decoded
# text after a correct ByteLevel decoder swap is shown in the inverse
# comment.
_VOCAB = {
    "<pad>": 0,
    "<s>": 1,
    "</s>": 2,
    "Hello": 3,  # 'Hello'
    "ĠHello": 4,  # ' Hello'
    "Ġworld": 5,  # ' world'
    "Ċ": 6,  # '\n'
    "ĠLet": 7,  # ' Let'
    "Ġme": 8,  # ' me'
    "Ġcompute": 9,  # ' compute'
    "ĊĊ": 10,  # '\n\n'
    "ĠThe": 11,  # ' The'
    "Ġanswer": 12,  # ' answer'
    "Ġis": 13,  # ' is'
    "4": 14,  # '4'
    ".": 15,  # '.'
    "<think>": 16,  # '<think>'
    "</think>": 17,  # '</think>'
}

# ids -> expected decoded text once the decoder is repaired.
_REASONING_SEQUENCE_IDS = [16, 6, 7, 8, 9, 10, 17, 6, 11, 12, 13, 14, 15]
_REASONING_SEQUENCE_TEXT = "<think>\n Let me compute\n\n</think>\n The answer is4."

# Sentinels for content streaming (no <think>).
_CONTENT_SEQUENCE_IDS = [4, 5, 6, 11, 12, 13, 14, 15]
_CONTENT_SEQUENCE_TEXT = " Hello world\n The answer is4."


def _build_broken_tokenizer() -> PreTrainedTokenizerFast:
    """Construct a tokenizer with the D-DETOK-BPE malformation.

    Encoder = GPT-2 ByteLevel (correct), decoder = SentencePiece chain
    (broken — leaks pretty tokens verbatim).
    """
    model = models.BPE(vocab=_VOCAB, merges=[])
    tok = Tokenizer(model)
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.Sequence(
        [
            decoders.Replace("▁", " "),
            decoders.ByteFallback(),
            decoders.Fuse(),
            decoders.Strip(" ", 1, 0),
        ]
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=tok,
        eos_token="</s>",
        bos_token="<s>",
        pad_token="<pad>",
    )


def _build_healthy_tokenizer() -> PreTrainedTokenizerFast:
    """Same vocab but with the correct ByteLevel decoder from the start."""
    model = models.BPE(vocab=_VOCAB, merges=[])
    tok = Tokenizer(model)
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    return PreTrainedTokenizerFast(
        tokenizer_object=tok,
        eos_token="</s>",
        bos_token="<s>",
        pad_token="<pad>",
    )


def _assert_no_mojibake(text: str, context: str = "") -> None:
    """Assert that ``text`` contains no GPT-2 byte-level pretty markers."""
    bad = [m for m in _BYTE_LEVEL_MOJIBAKE_MARKERS if m in text]
    assert not bad, f"mojibake leak in {context}: text={text!r} markers={bad!r}"


# ---------------------------------------------------------------------------
# 1. The repair function itself
# ---------------------------------------------------------------------------


class TestRepairByteLevelDecoder:
    """repair_byte_level_decoder swaps the decoder; idempotent; targeted."""

    def test_broken_tokenizer_leaks_mojibake_before_repair(self) -> None:
        tok = _build_broken_tokenizer()
        decoded = tok.decode(_REASONING_SEQUENCE_IDS, skip_special_tokens=False)
        # Sanity: the bug is reproducible — Ġ and Ċ both appear.
        assert "Ġ" in decoded
        assert "Ċ" in decoded

    def test_repair_returns_true_on_broken_tokenizer(self) -> None:
        tok = _build_broken_tokenizer()
        assert repair_byte_level_decoder(tok) is True

    def test_repair_clears_mojibake(self) -> None:
        tok = _build_broken_tokenizer()
        repair_byte_level_decoder(tok)
        decoded = tok.decode(_REASONING_SEQUENCE_IDS, skip_special_tokens=False)
        _assert_no_mojibake(decoded, context="full decode after repair")
        assert decoded == _REASONING_SEQUENCE_TEXT

    def test_repair_is_idempotent(self) -> None:
        tok = _build_broken_tokenizer()
        assert repair_byte_level_decoder(tok) is True
        # Second call: nothing to repair, returns False, output unchanged.
        assert repair_byte_level_decoder(tok) is False
        decoded = tok.decode(_REASONING_SEQUENCE_IDS, skip_special_tokens=False)
        assert decoded == _REASONING_SEQUENCE_TEXT

    def test_repair_is_noop_on_healthy_tokenizer(self) -> None:
        tok = _build_healthy_tokenizer()
        # Already healthy — no repair, decode already clean.
        before = tok.decode(_REASONING_SEQUENCE_IDS, skip_special_tokens=False)
        assert repair_byte_level_decoder(tok) is False
        after = tok.decode(_REASONING_SEQUENCE_IDS, skip_special_tokens=False)
        assert before == after
        _assert_no_mojibake(after, context="healthy tokenizer decode")

    def test_repair_handles_none_gracefully(self) -> None:
        assert repair_byte_level_decoder(None) is False

    def test_repair_skips_tokenizers_without_byte_level_vocab(self) -> None:
        """A tokenizer without ``Ġ``/``Ċ`` in its vocab is left alone."""
        # Build a vocab with no byte-level markers.
        from tokenizers.models import WordLevel

        plain_vocab = {"<pad>": 0, "hello": 1, "world": 2}
        rust = Tokenizer(WordLevel(plain_vocab))
        rust.decoder = decoders.WordPiece()
        plain = PreTrainedTokenizerFast(tokenizer_object=rust, pad_token="<pad>")
        # No mojibake markers in vocab — probe finds nothing — no repair.
        assert repair_byte_level_decoder(plain) is False

    def test_repair_restores_original_decoder_when_swap_fails_verification(
        self,
    ) -> None:
        """If ByteLevel swap doesn't clear the mojibake either (unknown
        vocab shape), the original decoder must be restored — *not* left
        as ByteLevel. Locks codex r1 BLOCKING contract."""
        # Build a vocab where every "byte-level-looking" token actually
        # decodes through a custom decoder that ByteLevel can't undo.
        # We simulate this by giving the vocab a Ġ-prefixed token whose
        # byte-level inverse still contains Ġ — patho­logical, but the
        # contract is: we revert on failure.
        vocab = {"<pad>": 0, "Ġevil": 1}
        model = models.BPE(vocab=vocab, merges=[])
        rust = Tokenizer(model)
        rust.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        # Custom decoder that simply returns the pretty token verbatim —
        # this is the BROKEN-on-byte-level shape we want to repair.
        original = decoders.Sequence(
            [
                decoders.Replace("▁", " "),
                decoders.ByteFallback(),
                decoders.Fuse(),
                decoders.Strip(" ", 1, 0),
            ]
        )
        rust.decoder = original
        tok = PreTrainedTokenizerFast(tokenizer_object=rust, pad_token="<pad>")

        # Sanity: the broken decode contains Ġ.
        assert "Ġ" in tok.decode([1], skip_special_tokens=False)

        # Monkey-patch ``tokenizers.decoders.ByteLevel`` to be a no-op
        # decoder (returns the input unchanged — still leaks Ġ) so the
        # verification path will fail and trigger the revert branch.
        import tokenizers.decoders as _decmod

        from vllm_mlx.utils import tokenizer as _toktools

        class _NoopDecoder(decoders.Decoder):
            def decode(self, tokens: list[str]) -> str:
                return "".join(tokens)

        # Patch only inside the call so we don't leak global state.
        real_byte_level = _decmod.ByteLevel
        try:
            _decmod.ByteLevel = _NoopDecoder
            assert _toktools.repair_byte_level_decoder(tok) is False
        finally:
            _decmod.ByteLevel = real_byte_level

        # The contract: original decoder is back in place.
        assert tok.backend_tokenizer.decoder.__class__.__name__ == "Sequence", (
            f"expected Sequence (original), got "
            f"{tok.backend_tokenizer.decoder.__class__.__name__}"
        )


# ---------------------------------------------------------------------------
# 2. Streaming surface: IncrementalDecoder (delta.* path)
# ---------------------------------------------------------------------------


class TestIncrementalDecoderNoLeak:
    """The streaming ``delta`` path used by the chat-completions SSE
    handler and the OpenAI completions adapter."""

    @pytest.mark.parametrize(
        "ids,expected",
        [
            (_REASONING_SEQUENCE_IDS, _REASONING_SEQUENCE_TEXT),
            (_CONTENT_SEQUENCE_IDS, _CONTENT_SEQUENCE_TEXT),
        ],
        ids=["reasoning_block", "content_block"],
    )
    def test_streaming_deltas_are_clean_after_repair(
        self, ids: list[int], expected: str
    ) -> None:
        tok = _build_broken_tokenizer()
        # The fix: repair happens at tokenizer load time in production.
        repair_byte_level_decoder(tok)

        decoder = IncrementalDecoder(tok)
        emitted = ""
        for tid in ids:
            delta = decoder.add_token(tid)
            _assert_no_mojibake(delta, context=f"delta for id={tid}")
            emitted += delta

        assert emitted == expected, (
            f"streaming sum mismatch:\n  got:      {emitted!r}\n"
            f"  expected: {expected!r}"
        )

    def test_streaming_without_repair_leaks(self) -> None:
        """Negative control: without the repair, mojibake leaks. This
        is the canonical D-DETOK-BPE reproducer — if this test fails
        the bug is no longer reproducible by this fixture (which would
        invalidate the positive assertions)."""
        tok = _build_broken_tokenizer()
        # Deliberately skip repair.
        decoder = IncrementalDecoder(tok)
        emitted = ""
        for tid in _REASONING_SEQUENCE_IDS:
            emitted += decoder.add_token(tid)
        assert any(m in emitted for m in _BYTE_LEVEL_MOJIBAKE_MARKERS), (
            f"expected mojibake in unrepaired streaming output, got {emitted!r}"
        )


# ---------------------------------------------------------------------------
# 3. Non-stream + raw /v1/completions surface: cumulative
#    tokenizer.decode(ids) is what builds ``message.content``,
#    ``message.reasoning_content``, and ``completions[0].text``.
# ---------------------------------------------------------------------------


class TestCumulativeDecodeNoLeak:
    @pytest.mark.parametrize(
        "ids,expected",
        [
            (_REASONING_SEQUENCE_IDS, _REASONING_SEQUENCE_TEXT),
            (_CONTENT_SEQUENCE_IDS, _CONTENT_SEQUENCE_TEXT),
        ],
        ids=["reasoning_block", "content_block"],
    )
    def test_full_decode_is_clean(self, ids: list[int], expected: str) -> None:
        tok = _build_broken_tokenizer()
        repair_byte_level_decoder(tok)
        decoded = tok.decode(ids, skip_special_tokens=False)
        _assert_no_mojibake(decoded, context="full decode")
        assert decoded == expected

    def test_per_token_decode_is_clean(self) -> None:
        """The legacy ``tokenizer.decode([single_id])`` path used by
        ``output_router.py`` line 188 (current text), 310 (response
        token), 430 (control token), and ``service/helpers.py``
        line 1370 (probe text) is *also* clean post-repair."""
        tok = _build_broken_tokenizer()
        repair_byte_level_decoder(tok)
        for tid in _REASONING_SEQUENCE_IDS:
            single = tok.decode([tid], skip_special_tokens=False)
            _assert_no_mojibake(single, context=f"per-token decode id={tid}")


# ---------------------------------------------------------------------------
# 4. UTF-8-safety preserved (the IncrementalDecoder's U+FFFD hold-back
#    contract must keep working after the decoder swap).
# ---------------------------------------------------------------------------


class TestUtf8SafetyPreserved:
    def test_repair_does_not_break_emoji_hold_back(self) -> None:
        """The ByteLevel decoder swap must not disturb the
        IncrementalDecoder's UTF-8 incomplete-sequence hold-back
        (deferred-emit) contract — multibyte characters still arrive
        intact instead of being split across deltas."""
        # Build a vocab with multibyte UTF-8 split across two ids.
        # In GPT-2 byte-level, U+00B0 (°) = bytes 0xC2 0xB0 = pretty
        # token "Â°"; a real emoji like 😀 (U+1F600) = F0 9F 98 80 =
        # "ðŁĺĢ".
        vocab = {
            "<pad>": 0,
            "</s>": 1,
            "Ġ25": 2,  # ' 25'
            "Â°": 3,  # '°' (2-byte UTF-8)
            "C": 4,
            # Split the emoji across two tokens.
            "ðŁ": 5,  # first half of 😀 (bytes F0 9F)
            "ĺĢ": 6,  # second half (bytes 98 80)
        }
        model = models.BPE(vocab=vocab, merges=[])
        rust = Tokenizer(model)
        rust.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        # Start broken: SentencePiece-style decoder.
        rust.decoder = decoders.Sequence(
            [
                decoders.Replace("▁", " "),
                decoders.ByteFallback(),
                decoders.Fuse(),
                decoders.Strip(" ", 1, 0),
            ]
        )
        tok = PreTrainedTokenizerFast(
            tokenizer_object=rust, pad_token="<pad>", eos_token="</s>"
        )
        assert repair_byte_level_decoder(tok) is True

        # Degree-sign sequence: " 25°C"
        decoder = IncrementalDecoder(tok)
        deltas = [decoder.add_token(t) for t in [2, 3, 4]]
        assert "".join(deltas) == " 25°C"
        for d in deltas:
            _assert_no_mojibake(d, context="degree-sign streaming")

        # Emoji split-across-tokens: 😀 must not leave a U+FFFD in the
        # first delta — the IncrementalDecoder holds back until the
        # second token completes it.
        decoder2 = IncrementalDecoder(tok)
        first_delta = decoder2.add_token(5)
        assert "�" not in first_delta
        # The full character only materialises after the second id.
        second_delta = decoder2.add_token(6)
        assert (first_delta + second_delta) == "😀"
        _assert_no_mojibake(first_delta + second_delta, context="emoji")
