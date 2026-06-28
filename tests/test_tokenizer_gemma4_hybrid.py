# SPDX-License-Identifier: Apache-2.0
"""Regression tests for issue #950 — Gemma 4 hybrid tokenizer space corruption.

PR #793 (``repair_byte_level_decoder``) misfires on hybrid SentencePiece-
metaspace + GPT-2-byte-level tokenizers like Gemma 4
(``mlx-community/gemma-4-26b-a4b-it-4bit``). Mechanism:

  1. Gemma 4's vocab contains legitimate GPT-2 byte tokens (id 240630 =
     ``ĉ`` for tab) alongside the dominant SentencePiece ``▁``-prefixed
     space tokens.
  2. PR #793's probe finds the legit ``ĉ`` byte token, sees the SP
     decoder fails to invert it (``ByteFallback`` doesn't know about
     GPT-2 pretty-byte encoding), and declares the tokenizer "broken".
  3. PR #793 swaps the whole decoder for a bare ``ByteLevel()``,
     dropping the load-bearing ``Replace("▁", " ")`` step.
  4. Post-swap verify only re-decodes the probe byte token (now clean)
     — never notices spaces are now ``▁``-corrupted.

  Result: every space in ``content``, ``reasoning_content``, streaming
  deltas, and ``/v1/completions[0].text`` becomes ``▁`` (U+2581).

The fix (two gates inside ``repair_byte_level_decoder``):

  * **Gate 2** — bail before mutating if the existing decoder already
    contains a ``Replace("▁", " ")`` step AND a ``encode("a b c")``
    sample produces tokens containing ``▁`` (the load-bearing
    metaspace indicator). The latter check is the disambiguator: a
    pure GPT-2 byte-level vocab with a *mis-applied* SP decoder (the
    PR #793 target, DeepSeek-R1 distills) ALSO has the ``Replace``
    step in its decoder, but its vocab uses ``Ġ`` (not ``▁``) for
    spaces — so the encode-sample check correctly distinguishes the
    two cases.

  * **Gate 3** — even if a future hybrid we haven't seen slips past
    gate 2, after the swap decode ``encode("a b c")`` and assert no
    ``▁`` leaks. If it does, revert to the original decoder.

This test file pins both gates with a synthetic reproducer (no
download required, runs in CI) AND a real-Gemma-4 reproducer
(``AutoTokenizer.from_pretrained`` only — no model load).
"""

from __future__ import annotations

import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from vllm_mlx.utils.tokenizer import (
    _METASPACE_MARKER,
    _decoder_has_metaspace_replace,
    repair_byte_level_decoder,
)

# Real Gemma 4 alias the bug reporter used. The model weights are NOT
# downloaded — ``AutoTokenizer.from_pretrained`` only fetches the
# tokenizer files (tokenizer.json + tokenizer_config.json + sidecars),
# which total a few MB and are cached on first run.
_GEMMA4_ALIAS = "mlx-community/gemma-4-26b-a4b-it-4bit"


def _build_synthetic_gemma4_hybrid() -> PreTrainedTokenizerFast:
    """Synthetic reproducer of Gemma 4's hybrid tokenizer shape.

    Vocab combines:
      * SentencePiece ``▁``-prefixed space tokens (dominant)
      * A legit GPT-2-pretty byte token (``ĉ`` = tab) — the probe trap
        that fooled PR #793 into mis-classifying Gemma 4 as broken.

    Decoder is the real Gemma 4 chain:
      ``Sequence([Replace("▁", " "), ByteFallback(), Fuse()])``

    On THIS tokenizer:
      * ``decode([▁quick id])`` correctly returns ``" quick"`` (via the
        ``Replace`` step).
      * ``decode([ĉ id])`` returns ``"ĉ"`` verbatim because
        ``ByteFallback`` doesn't know GPT-2 pretty-byte encoding.

    The pre-fix ``repair_byte_level_decoder`` "fixes" the latter at the
    cost of the former. The fix gate 2 must bail before mutating so the
    spaces survive.
    """
    vocab = {
        "<pad>": 0,
        "<eos>": 1,
        "<unk>": 2,
        "▁the": 3,
        "the": 4,
        "▁quick": 5,
        "▁brown": 6,
        "▁fox": 7,
        # The trap: legit GPT-2-pretty byte token that PR #793's probe
        # would have flagged as "broken decoder".
        "ĉ": 8,
        # Single-char SP tokens so ``encode("a b c")`` produces the
        # ``▁`` metaspace marker tokens that gate 2's disambiguator
        # depends on.
        "▁a": 9,
        "▁b": 10,
        "▁c": 11,
    }
    wl = models.WordLevel(vocab=vocab, unk_token="<unk>")
    rust = Tokenizer(wl)
    # Match Gemma 4's tokenizer pipeline (Metaspace converts ASCII
    # spaces into the ``▁`` metaspace marker at the pre-tokenizer
    # stage). Without this, ``encode("a b c")`` cannot produce
    # ``▁``-containing tokens and gate 2's disambiguator can't tell
    # the synthetic hybrid apart from a pure-byte-level vocab.
    rust.pre_tokenizer = pre_tokenizers.Metaspace()
    rust.decoder = decoders.Sequence(
        [
            decoders.Replace("▁", " "),
            decoders.ByteFallback(),
            decoders.Fuse(),
        ]
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=rust,
        eos_token="<eos>",
        pad_token="<pad>",
        unk_token="<unk>",
    )


# ---------------------------------------------------------------------------
# Gate 2 helper: _decoder_has_metaspace_replace
# ---------------------------------------------------------------------------


class TestDecoderHasMetaspaceReplace:
    """Pin the helper that drives gate 2 — must detect ``Replace('▁',' ')``
    whether top-level or nested inside a ``Sequence``."""

    def test_detects_top_level_replace(self) -> None:
        d = decoders.Replace("▁", " ")
        assert _decoder_has_metaspace_replace(d) is True

    def test_detects_replace_nested_in_sequence(self) -> None:
        d = decoders.Sequence(
            [
                decoders.Replace("▁", " "),
                decoders.ByteFallback(),
                decoders.Fuse(),
            ]
        )
        assert _decoder_has_metaspace_replace(d) is True

    def test_returns_false_for_bare_byte_level(self) -> None:
        d = decoders.ByteLevel()
        assert _decoder_has_metaspace_replace(d) is False

    def test_returns_false_for_replace_with_different_pattern(self) -> None:
        # Different character — not the metaspace marker.
        d = decoders.Replace("X", " ")
        assert _decoder_has_metaspace_replace(d) is False

    def test_returns_false_for_replace_with_different_content(self) -> None:
        # Correct pattern but different replacement — not a space.
        d = decoders.Replace("▁", "_")
        assert _decoder_has_metaspace_replace(d) is False


# ---------------------------------------------------------------------------
# Gate 2: synthetic reproducer — hybrid tokenizer must NOT have its
# decoder swapped, and spaces must round-trip cleanly.
# ---------------------------------------------------------------------------


class TestGate2SyntheticHybrid:
    """The pre-fix bug: PR #793 would swap the decoder out, dropping
    ``Replace("▁", " ")``, and the next ``decode`` round-trip would
    leak ``▁`` into model output. Gate 2 must bail BEFORE the swap."""

    def test_synthetic_hybrid_decode_clean_before_repair(self) -> None:
        """Sanity: the synthetic hybrid tokenizer round-trips spaces
        correctly with its native decoder. This is the contract we
        must not break."""
        tok = _build_synthetic_gemma4_hybrid()
        ids = [4, 5, 6, 7]  # "the ▁quick ▁brown ▁fox"
        assert tok.decode(ids) == "the quick brown fox"

    def test_repair_returns_false_on_synthetic_hybrid(self) -> None:
        """Gate 2: with a metaspace ``Replace`` step in the decoder AND
        ``▁`` in the spaced-encoding sample, repair must short-circuit
        without mutation."""
        tok = _build_synthetic_gemma4_hybrid()
        assert repair_byte_level_decoder(tok) is False

    def test_repair_does_not_corrupt_spaces_on_synthetic_hybrid(self) -> None:
        """The actual user-visible bug: every space becomes ``▁`` after
        repair. This test FAILS pre-fix and PASSES post-fix."""
        tok = _build_synthetic_gemma4_hybrid()
        ids = [4, 5, 6, 7]
        repair_byte_level_decoder(tok)
        # No metaspace marker leakage post-repair.
        decoded = tok.decode(ids)
        assert _METASPACE_MARKER not in decoded, (
            f"space corruption regression: decoded={decoded!r}"
        )
        assert decoded == "the quick brown fox"

    def test_repair_leaves_decoder_unchanged_on_synthetic_hybrid(self) -> None:
        """Gate 2 is a no-op contract: not just decode parity, but the
        live decoder object must remain identical. Catches a future
        regression where someone refactors gate 2 to "swap then revert"
        instead of "bail outright"."""
        tok = _build_synthetic_gemma4_hybrid()
        decoder_before = tok.backend_tokenizer.decoder
        before_state = decoder_before.__getstate__()
        repair_byte_level_decoder(tok)
        after_state = tok.backend_tokenizer.decoder.__getstate__()
        assert before_state == after_state, (
            f"decoder mutated by gate-2 path:\n"
            f"  before: {before_state!r}\n  after:  {after_state!r}"
        )


# ---------------------------------------------------------------------------
# Gate 2: real Gemma 4 reproducer (tokenizer-only, no model weights).
# This is the exact reproducer from the bug report.
# ---------------------------------------------------------------------------


class TestGate2RealGemma4:
    """Pin the issue-#950 reproducer end-to-end on the real Gemma 4
    tokenizer. Tokenizer files only — no model weights, no MLX load."""

    @pytest.fixture(scope="class")
    def gemma4_tokenizer(self):
        return AutoTokenizer.from_pretrained(_GEMMA4_ALIAS)

    def test_issue_950_reproducer_fixed(self, gemma4_tokenizer) -> None:
        """The exact reproducer from issue #950: ``decode(encode("the
        quick brown fox"))`` must round-trip cleanly even AFTER calling
        ``repair_byte_level_decoder``. Pre-fix the spaces became ``▁``;
        post-fix the round-trip is identity."""
        tok = gemma4_tokenizer
        text = "the quick brown fox"
        ids = tok.encode(text, add_special_tokens=False)
        # Sanity: native round-trip works.
        assert tok.decode(ids) == text, "Gemma 4 native decode is broken"
        repair_byte_level_decoder(tok)
        # The fix: round-trip still works after calling repair.
        assert tok.decode(ids) == text, (
            f"issue #950 regression: decode after repair = {tok.decode(ids)!r}, "
            f"expected {text!r}"
        )

    def test_gemma4_repair_returns_false(self, gemma4_tokenizer) -> None:
        """Gate 2 short-circuits: repair returns False (no mutation) on
        Gemma 4 — not True (swap applied)."""
        tok = gemma4_tokenizer
        assert repair_byte_level_decoder(tok) is False

    def test_gemma4_multiple_spaced_phrases(self, gemma4_tokenizer) -> None:
        """Broad coverage: a variety of spaced inputs all round-trip
        cleanly after repair, locking that the fix covers more than
        just the single phrase the bug report used."""
        tok = gemma4_tokenizer
        repair_byte_level_decoder(tok)
        for text in [
            "hello world",
            "a b c d e f",
            "The answer is 42.",
            "Lorem ipsum dolor sit amet",
            "  leading and trailing spaces  ",
        ]:
            ids = tok.encode(text, add_special_tokens=False)
            decoded = tok.decode(ids)
            assert _METASPACE_MARKER not in decoded, (
                f"metaspace leaked for {text!r}: decoded={decoded!r}"
            )

    def test_gemma4_decoder_not_mutated(self, gemma4_tokenizer) -> None:
        """Gate 2: the real Gemma 4 decoder object must remain
        identical (same JSON state) after a repair call."""
        tok = gemma4_tokenizer
        before = tok.backend_tokenizer.decoder.__getstate__()
        repair_byte_level_decoder(tok)
        after = tok.backend_tokenizer.decoder.__getstate__()
        assert before == after, "Gemma 4 decoder unexpectedly mutated"


# ---------------------------------------------------------------------------
# PR #793 still works: DeepSeek/Qwen-style mis-paired SP decoder over
# pure GPT-2-byte-level vocab. Gate 2's disambiguator (encode("a b c")
# must produce ``▁``-containing tokens) MUST distinguish this case from
# the Gemma 4 hybrid and let the swap proceed.
# ---------------------------------------------------------------------------


class TestPR793PathStillRepairs:
    """The PR #793 win — DeepSeek-R1 distills with a mis-paired Llama SP
    decoder over a pure GPT-2-byte-level vocab — must STILL get its
    decoder swapped. Locks that gate 2's disambiguator works."""

    def test_simulated_deepseek_r1_distill_still_repairs(self) -> None:
        """Take a real GPT-2-byte-level tokenizer (Qwen3 — same shape as
        DeepSeek-R1 distills) and force-replace its decoder with the
        broken Llama SP chain. This reproduces the exact malformation
        PR #793 was meant to fix. Gate 2 must NOT fire here (vocab
        encodes spaces as ``Ġ``, not ``▁``), so the swap proceeds and
        ``decode`` is repaired."""
        tok = AutoTokenizer.from_pretrained("mlx-community/Qwen3-0.6B-4bit")
        # Force the bug: swap in the Llama SP decoder chain. The vocab
        # itself is unchanged (still pure GPT-2 byte-level).
        tok.backend_tokenizer.decoder = decoders.Sequence(
            [
                decoders.Replace("▁", " "),
                decoders.ByteFallback(),
                decoders.Fuse(),
                decoders.Strip(" ", 1, 0),
            ]
        )
        ids = tok.encode("hello world", add_special_tokens=False)
        # Pre-repair: the ``Ġ`` mojibake leaks because the SP decoder
        # doesn't know how to invert GPT-2 pretty-byte tokens.
        broken = tok.decode(ids)
        assert "Ġ" in broken, (
            f"reproducer didn't break Qwen3 decoder as expected: {broken!r}"
        )
        # Post-repair: swap fires, mojibake is gone.
        assert repair_byte_level_decoder(tok) is True
        repaired = tok.decode(ids)
        assert repaired == "hello world", f"PR #793 repair path broken: {repaired!r}"

    def test_pure_byte_level_vocab_with_sp_decoder_falls_to_gate_2_safely(
        self,
    ) -> None:
        """A synthetic vocab with NO ``▁`` tokens but a SP-style decoder
        — exactly the shape PR #793's existing
        ``test_streaming_detokenizer_bpe.py::_build_broken_tokenizer``
        fixture targets. Gate 2's encode-sample check finds no ``▁`` in
        the vocab so the swap must proceed."""
        vocab = {
            "<pad>": 0,
            "<s>": 1,
            "</s>": 2,
            "Hello": 3,
            "ĠHello": 4,
            "Ġworld": 5,
            "Ċ": 6,
        }
        bpe = models.BPE(vocab=vocab, merges=[])
        rust = Tokenizer(bpe)
        rust.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        rust.decoder = decoders.Sequence(
            [
                decoders.Replace("▁", " "),
                decoders.ByteFallback(),
                decoders.Fuse(),
                decoders.Strip(" ", 1, 0),
            ]
        )
        tok = PreTrainedTokenizerFast(
            tokenizer_object=rust,
            eos_token="</s>",
            bos_token="<s>",
            pad_token="<pad>",
        )
        # Pre-repair: ``Ġ`` leaks.
        assert "Ġ" in tok.decode([4, 5], skip_special_tokens=False)
        # Gate 2 sees the SP ``Replace`` in the decoder, but
        # ``encode("a b c")`` on this synthetic vocab returns no
        # tokens containing ``▁`` (the BPE has no ``▁`` entries) — so
        # gate 2's disambiguator clears it and the swap proceeds.
        assert repair_byte_level_decoder(tok) is True
        assert "Ġ" not in tok.decode([4, 5], skip_special_tokens=False)


# ---------------------------------------------------------------------------
# Gate 3: belt-and-braces — even a hybrid that slips past gate 2 (e.g.
# a future model with a metaspace ``Replace`` we don't recognise via
# ``__getstate__``) must have its swap reverted when the post-swap
# spaced-sample decode leaks ``▁``.
# ---------------------------------------------------------------------------


class TestGate3SpacedSampleVerification:
    """Lock the gate-3 fail-closed contract."""

    def test_gate_3_reverts_when_metaspace_leaks_post_swap(self, monkeypatch) -> None:
        """Simulate a "future hybrid we haven't seen": force gate 2 to
        clear (by neutering ``_decoder_has_metaspace_replace``), so the
        swap runs. The swap is now a regression because the vocab uses
        ``▁``. Gate 3's spaced-sample decode must catch it and revert.
        """
        tok = _build_synthetic_gemma4_hybrid()
        # Neuter gate 2 — pretend the decoder has no metaspace step,
        # so the swap path will run.
        from vllm_mlx.utils import tokenizer as _toktools

        monkeypatch.setattr(
            _toktools, "_decoder_has_metaspace_replace", lambda d: False
        )

        # Capture the original decoder state for revert verification.
        original_state = tok.backend_tokenizer.decoder.__getstate__()

        # Repair must run, hit the swap, and gate 3 must catch the
        # metaspace leak in the post-swap spaced sample.
        result = repair_byte_level_decoder(tok)
        assert result is False, "gate 3 should have caught metaspace leak"

        # Decoder reverted to original state — not left as ByteLevel.
        assert tok.backend_tokenizer.decoder.__getstate__() == original_state, (
            "gate 3 must revert decoder when spaced-sample verify fails"
        )

        # And of course the spaces still round-trip cleanly.
        assert tok.decode([4, 5, 6, 7]) == "the quick brown fox"
