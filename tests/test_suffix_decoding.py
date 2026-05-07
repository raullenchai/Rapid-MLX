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
