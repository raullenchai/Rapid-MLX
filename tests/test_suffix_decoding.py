# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``SuffixDecodingDrafter``.

These exercise the drafter in isolation — no model, no MLX. The headline
behaviour we verify:

  - Empty history → no draft.
  - Repeated literal phrase → drafter recovers the continuation.
  - Ambiguous continuation → drafter truncates at the confidence floor.
  - Max-draft cap is honored.
  - History trimming preserves index correctness past the cap.

Acceptance reporting and stats accounting are also covered, since the
PoC reads its headline metric (mean accepted per step) directly off
``DraftStats``.
"""

from __future__ import annotations

import pytest

from vllm_mlx.speculative.suffix_decoding import (
    DraftStats,
    SuffixDecodingDrafter,
)


class TestDrafterBasics:
    def test_empty_history_no_draft(self):
        drafter = SuffixDecodingDrafter()
        assert drafter.get_draft() == []
        assert drafter.stats.n_step_calls == 1
        assert drafter.stats.n_drafts_returned == 0

    def test_repeated_literal_recovers_continuation(self):
        # Phrase "1 2 3 4 5" appears twice, then suffix matches the start
        # of the third occurrence — drafter should propose the remainder
        # up to ``max_draft_tokens``. Both prior occurrences agree on
        # tokens 3, 4, 5, 1 (the "1" comes from the next repeat starting),
        # so 4-length draft is correct.
        drafter = SuffixDecodingDrafter(max_draft_tokens=4, max_suffix_len=4)
        drafter.add_prompt_tokens([1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2])
        draft = drafter.get_draft()
        assert draft == [3, 4, 5, 1]

    def test_ambiguous_continuation_truncates(self):
        # Phrase (1, 2) is followed by 3 once and by 9 once (50/50).
        # With min_confidence=0.6, drafter must refuse the first token.
        drafter = SuffixDecodingDrafter(
            max_draft_tokens=4, max_suffix_len=2, min_confidence=0.6
        )
        drafter.add_prompt_tokens([1, 2, 3, 0, 1, 2, 9, 0, 1, 2])
        draft = drafter.get_draft()
        assert draft == [], "50/50 split must not pass 0.6 floor"

    def test_truncation_at_first_low_confidence(self):
        # (1, 2) → 3 always, then (1, 2, 3) → 4 once and → 5 once.
        # First draft token (3) should land; second (after voting on
        # ambiguous (1, 2, 3) continuation) should be cut.
        drafter = SuffixDecodingDrafter(
            max_draft_tokens=4, max_suffix_len=2, min_confidence=0.6
        )
        drafter.add_prompt_tokens([1, 2, 3, 4, 0, 1, 2, 3, 5, 0, 1, 2])
        draft = drafter.get_draft()
        # Both positions agree on token 3 at offset 0 — accepted.
        # At offset 1, after filtering to "matched 3", positions split
        # 50/50 on 4 vs 5 — drafter stops.
        assert draft == [3]

    def test_max_draft_cap_honored(self):
        # Long deterministic continuation but cap at 3.
        drafter = SuffixDecodingDrafter(max_draft_tokens=3, max_suffix_len=2)
        drafter.add_prompt_tokens([1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2])
        draft = drafter.get_draft()
        assert len(draft) == 3
        assert draft == [3, 4, 5]

    def test_skips_self_match(self):
        # Only one occurrence of the suffix — there's nothing to learn
        # from, so no draft should be returned.
        drafter = SuffixDecodingDrafter(max_draft_tokens=4, max_suffix_len=2)
        drafter.add_prompt_tokens([1, 2, 3])
        # Suffix is (2, 3) and only ends at position 2. Predicting
        # ourselves is forbidden, so no draft.
        assert drafter.get_draft() == []


class TestHistoryTrimming:
    def test_trim_preserves_recent_lookups(self):
        drafter = SuffixDecodingDrafter(
            max_draft_tokens=4, max_suffix_len=2, max_history=10
        )
        # 12 tokens → first 2 dropped from local _tokens but absolute
        # positions are still consistent.
        drafter.add_prompt_tokens([1, 2, 9, 9, 9, 9, 9, 9, 9, 9, 1, 2])
        # _shift = 2, _tokens length 10. Suffix (1, 2) appears at abs end
        # position 11 (current). The other (1, 2) at abs 0/1 was trimmed
        # → drafter should NOT find a continuation (stale positions for
        # the trimmed range resolve to local index < 0 and are skipped).
        draft = drafter.get_draft()
        # Confirm no draft is returned (no surviving (1, 2) continuation).
        # If we kept the prefix, draft would be [9].
        assert draft == []

    def test_index_robust_after_many_adds(self):
        drafter = SuffixDecodingDrafter(
            max_draft_tokens=2, max_suffix_len=2, max_history=20
        )
        # Stream 50 tokens of a periodic pattern; lookups still work for
        # the most recent (1, 2) appearance.
        for i in range(50):
            drafter.add_generated_token(i % 5)
        # Last few tokens are 5, 6, 7, 8, 9 mod 5 → 0, 1, 2, 3, 4.
        # Suffix (3, 4) is followed by 0 in the recent window.
        draft = drafter.get_draft()
        assert draft == [0, 1] or draft == [0]


class TestStats:
    def test_stats_track_proposals_and_accepts(self):
        # max_draft=3 so the headline arithmetic is clean:
        # one prior (1, 2) → continuation [3, 4, 5] capped at 3.
        drafter = SuffixDecodingDrafter(max_draft_tokens=3, max_suffix_len=2)
        drafter.add_prompt_tokens([1, 2, 3, 4, 5, 1, 2])
        draft = drafter.get_draft()
        assert draft == [3, 4, 5]
        # Verifier accepted the first 2 of the 3 drafts.
        drafter.record_acceptance(num_accepted=2)
        assert drafter.stats.total_drafts_proposed == 1
        assert drafter.stats.total_draft_tokens_proposed == 3
        assert drafter.stats.total_draft_tokens_accepted == 2
        assert drafter.stats.acceptance_rate == pytest.approx(2 / 3)
        assert drafter.stats.n_step_calls == 1
        assert drafter.stats.mean_accepted_per_step == 2.0

    def test_stats_dict_roundtrip(self):
        s = DraftStats(
            total_drafts_proposed=10,
            total_draft_tokens_proposed=40,
            total_draft_tokens_accepted=22,
            n_step_calls=15,
            n_drafts_returned=10,
        )
        d = s.as_dict()
        assert d["acceptance_rate"] == pytest.approx(0.55)
        # 22 accepts over 15 step calls = 1.4666 mean accepts per step.
        # That's the upper-bound "speedup hint" our PoC reports.
        assert d["mean_accepted_per_step"] == pytest.approx(22 / 15, abs=1e-4)


class TestValidation:
    def test_rejects_invalid_max_draft(self):
        with pytest.raises(ValueError):
            SuffixDecodingDrafter(max_draft_tokens=0)

    def test_rejects_invalid_suffix_len(self):
        with pytest.raises(ValueError):
            SuffixDecodingDrafter(max_suffix_len=0)

    def test_rejects_invalid_confidence(self):
        with pytest.raises(ValueError):
            SuffixDecodingDrafter(min_confidence=1.5)
        with pytest.raises(ValueError):
            SuffixDecodingDrafter(min_confidence=-0.1)


class TestRealisticAgentWorkload:
    """Integration-flavored — feed the drafter a token stream that
    resembles an agent loop (prompt echo + repeated tool-name patterns)
    and confirm the drafts are non-trivial.
    """

    def test_tool_name_echo(self):
        # Imagine tokens for "user wants to call tool: get_weather"
        # generated, then context for the next call wraps the same
        # pattern. We expect (call_tool, get_weather) to be drafted.
        # Encode as integer ids for simplicity.
        prompt = (
            [10, 11, 12, 13, 14]  # system header
            + [20, 21, 22]  # "call tool:"
            + [30, 31]  # "get weather"
            + [40, 41, 42]  # response
            + [50, 51, 52]  # follow-up
            + [20, 21, 22]  # another "call tool:" — drafter sees pattern
        )
        drafter = SuffixDecodingDrafter(max_draft_tokens=3, max_suffix_len=3)
        drafter.add_prompt_tokens(prompt)
        # Last suffix (20, 21, 22) matched once before — continuation was
        # 30, 31, 40 in absolute positions. We should draft those.
        draft = drafter.get_draft()
        assert draft == [30, 31, 40]


class TestInstallSuffixDecoding:
    """Tests for the GenerationBatch monkey-patch installer.

    These cover the *wiring* — allowlist gate, install side-effects,
    fallback behaviour — without requiring a real model. End-to-end
    output-correctness is covered by ``scripts/bench_suffix_decoding.py``
    and the PoC sweep in ``evals/results/SUFFIX_POC_REPORT.md``.

    The hook patches ``BatchGenerator._generation_batch._step`` and
    ``.next`` (mlx-lm 0.31+ moved the step from BatchGenerator down to
    GenerationBatch). Tests use a minimal fake with the same surface.
    """

    def _make_fake_bg(self):
        """Build a minimal fake BatchGenerator + GenerationBatch.

        Captures the attributes the install function reads. Intentionally
        avoids MagicMock because hasattr() on a MagicMock returns True
        for any name, which masks the "no-op install" check.
        """

        class _GB:
            pass

        gb = _GB()
        gb._step = lambda: ([], [])  # original step (returns empty)
        gb.next = lambda: []  # original next

        class _BG:
            pass

        bg = _BG()
        bg._generation_batch = gb
        return bg, gb

    def test_hybrid_profile_skips_install(self):
        """``supports_spec_decode == False`` → install is a no-op.

        The whole point of the per-model profile is that hybrid models
        (Qwen3.5/3.6 GatedDeltaNet, Granite 4 Mamba2) never get
        SuffixDecoding installed — chunked-batched verify corrupts their
        output (see SUFFIX_POC_REPORT.md table 4).
        """
        from unittest.mock import MagicMock

        from vllm_mlx.model_auto_config import ModelConfig
        from vllm_mlx.scheduler import _install_suffix_decoding

        bg, gb = self._make_fake_bg()
        orig_step = gb._step
        orig_next = gb.next

        profile = ModelConfig(
            is_hybrid=True,
            supports_spec_decode=False,
        )

        _install_suffix_decoding(
            bg,
            model=MagicMock(),
            profile=profile,
            max_draft=8,
            max_suffix_len=4,
            min_confidence=0.3,
            requests={},
            uid_to_request_id={},
        )

        # Refusing to install means leaving _step / next pristine.
        assert gb._step is orig_step
        assert gb.next is orig_next
        # No telemetry attribute either (on bg or gb).
        assert not hasattr(bg, "_suffix_stats")
        assert not hasattr(gb, "_suffix_stats")

    def test_pure_attention_profile_installs(self):
        """Default ``supports_spec_decode=True`` → step/next replaced + stats attached."""
        from unittest.mock import MagicMock

        from vllm_mlx.model_auto_config import ModelConfig
        from vllm_mlx.scheduler import _install_suffix_decoding

        bg, gb = self._make_fake_bg()
        orig_step = gb._step
        orig_next = gb.next

        profile = ModelConfig()  # defaults: supports_spec_decode=True
        assert profile.supports_spec_decode

        _install_suffix_decoding(
            bg,
            model=MagicMock(),
            profile=profile,
            max_draft=8,
            max_suffix_len=4,
            min_confidence=0.3,
            requests={},
            uid_to_request_id={},
        )

        # Both ._step and .next on the GenerationBatch should be replaced.
        assert gb._step is not orig_step
        assert gb.next is not orig_next
        # Telemetry attribute attached for /metrics surfacing.
        assert hasattr(bg, "_suffix_stats")
        assert hasattr(gb, "_suffix_stats")
        stats = bg._suffix_stats
        assert stats["verify_steps"] == 0
        assert stats["fallthrough_steps"] == 0
        assert stats["draft_tokens_proposed"] == 0
        assert stats["tokens_accepted"] == 0
        # bg and gb share the same stats dict
        assert bg._suffix_stats is gb._suffix_stats

    def test_no_profile_treats_as_supported(self):
        """``profile=None`` → install proceeds (default-on for unknown families)."""
        from unittest.mock import MagicMock

        from vllm_mlx.scheduler import _install_suffix_decoding

        bg, gb = self._make_fake_bg()
        orig_step = gb._step

        _install_suffix_decoding(
            bg,
            model=MagicMock(),
            profile=None,
            max_draft=8,
            max_suffix_len=4,
            min_confidence=0.3,
            requests={},
            uid_to_request_id={},
        )

        assert gb._step is not orig_step
        assert hasattr(bg, "_suffix_stats")

    def test_step_falls_through_when_no_generation_batch(self):
        """If BatchGenerator has no ``_generation_batch`` (older mlx-lm),
        install logs a warning and returns without patching anything."""
        from unittest.mock import MagicMock

        from vllm_mlx.scheduler import _install_suffix_decoding

        # bg WITHOUT _generation_batch — simulates pre-0.31 mlx-lm
        class _BG:
            pass

        bg = _BG()

        _install_suffix_decoding(
            bg,
            model=MagicMock(),
            profile=None,
            max_draft=8,
            max_suffix_len=4,
            min_confidence=0.3,
            requests={},
            uid_to_request_id={},
        )

        assert not hasattr(bg, "_suffix_stats")

    def test_stats_include_non_trimmable_cache_counter(self):
        """Defense-in-depth: pre-flight check counter is wired into stats.

        Even though ``profile.supports_spec_decode`` already gates install on
        hybrid arches, the verify path also pre-checks every cache layer
        is trimmable. If a ``_BaseCache`` (or vendored DeltaNet/Mamba
        cache) ever slipped through, we'd fall through and bump
        ``ft_non_trimmable_cache`` instead of corrupting state silently.
        """
        from unittest.mock import MagicMock

        from vllm_mlx.scheduler import _install_suffix_decoding

        bg, _gb = self._make_fake_bg()

        _install_suffix_decoding(
            bg,
            model=MagicMock(),
            profile=None,
            max_draft=8,
            max_suffix_len=4,
            min_confidence=0.3,
            requests={},
            uid_to_request_id={},
        )

        assert "ft_non_trimmable_cache" in bg._suffix_stats
        assert bg._suffix_stats["ft_non_trimmable_cache"] == 0

    def test_drafter_pruned_when_primary_finishes(self):
        """Per-uid drafters must be dropped when the primary finishes.

        Without the cleanup, ``_drafters`` accumulates one entry per
        completed request for the lifetime of the BatchGenerator. Each
        drafter can hold up to ``max_history`` indexed tokens (default
        32K) — a real memory leak on long-running servers.
        """
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        from vllm_mlx.scheduler import _install_suffix_decoding
        from vllm_mlx.speculative.suffix_decoding import SuffixDecodingDrafter

        bg, gb = self._make_fake_bg()
        # The wrapped next() will call _orig_next() which returns a single
        # finished response — exercising the primary-finish cleanup branch.
        finished_response = SimpleNamespace(uid=42, finish_reason="stop")
        gb.next = lambda: [finished_response]

        _install_suffix_decoding(
            bg,
            model=MagicMock(),
            profile=None,
            max_draft=8,
            max_suffix_len=4,
            min_confidence=0.3,
            requests={},
            uid_to_request_id={},
        )

        drafters = gb._suffix_drafters
        # Plant a drafter for uid 42 to mimic a verify step having run.
        drafters[42] = SuffixDecodingDrafter()
        # Also stash a pending emit so we exercise the same branch the
        # production code does (drop-pending + drop-drafter).
        # Closure-scoped _pending_emits is reachable indirectly: the
        # wrapped next() pops both before falling through.
        gb.next()  # invokes the wrapped _suffix_next via the install
        assert 42 not in drafters, (
            "Drafter for finished uid was retained — _drafters leak"
        )
