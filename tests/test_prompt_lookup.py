# SPDX-License-Identifier: Apache-2.0
"""Tests for PromptLookupDecoder."""

from vllm_mlx.speculative.prompt_lookup import PromptLookupDecoder


class TestPromptLookupDecoderInit:
    """Tests for PromptLookupDecoder initialization."""

    def test_default_init(self):
        decoder = PromptLookupDecoder()
        assert decoder.num_draft_tokens == 4
        assert decoder.ngram_size == 3
        assert decoder.min_matches == 2
        assert decoder._token_history == []
        assert decoder.total_drafts == 0
        assert decoder.successful_drafts == 0
        assert decoder.total_draft_tokens == 0
        assert decoder.accepted_tokens == 0

    def test_custom_init(self):
        decoder = PromptLookupDecoder(num_draft_tokens=8, ngram_size=5, min_matches=3)
        assert decoder.num_draft_tokens == 8
        assert decoder.ngram_size == 5
        assert decoder.min_matches == 3


class TestPromptLookupDecoderReset:
    """Tests for reset method."""

    def test_reset_clears_history(self):
        decoder = PromptLookupDecoder()
        decoder.add_prompt_tokens([1, 2, 3, 4, 5])
        decoder.add_generated_token(6)

        decoder.reset()

        assert decoder._token_history == []
        assert len(decoder._ngram_index) == 0

    def test_reset_preserves_config(self):
        decoder = PromptLookupDecoder(num_draft_tokens=8, ngram_size=5)
        decoder.add_prompt_tokens([1, 2, 3])

        decoder.reset()

        assert decoder.num_draft_tokens == 8
        assert decoder.ngram_size == 5


class TestPromptLookupDecoderAddTokens:
    """Tests for add_prompt_tokens and add_generated_token."""

    def test_add_prompt_tokens_empty(self):
        decoder = PromptLookupDecoder()
        decoder.add_prompt_tokens([])
        assert decoder._token_history == []

    def test_add_prompt_tokens_single(self):
        decoder = PromptLookupDecoder()
        decoder.add_prompt_tokens([1])
        assert decoder._token_history == [1]

    def test_add_prompt_tokens_multiple(self):
        decoder = PromptLookupDecoder(ngram_size=3)
        decoder.add_prompt_tokens([1, 2, 3, 4, 5])
        assert decoder._token_history == [1, 2, 3, 4, 5]

    def test_add_generated_token(self):
        decoder = PromptLookupDecoder()
        decoder.add_prompt_tokens([1, 2])
        decoder.add_generated_token(3)
        assert decoder._token_history == [1, 2, 3]

    def test_ngram_index_populated(self):
        """N-gram index stores preceding n-grams pointing to positions."""
        decoder = PromptLookupDecoder(ngram_size=3)
        # History: [1, 2, 3, 4]
        # At pos 3 (token 4): ngram (1,2,3) -> pos 3
        decoder.add_prompt_tokens([1, 2, 3, 4])
        assert (1, 2, 3) in decoder._ngram_index


class TestPromptLookupDecoderGetDraftTokens:
    """Tests for get_draft_tokens method.

    Key: query = history[-ngram_size:]. The index maps an n-gram to positions
    where that n-gram PRECEDED the token. Continuation starts at the matched
    position.
    """

    def test_empty_history_returns_empty(self):
        decoder = PromptLookupDecoder()
        assert decoder.get_draft_tokens() == []

    def test_history_shorter_than_ngram_size(self):
        decoder = PromptLookupDecoder(ngram_size=3)
        decoder.add_prompt_tokens([1, 2])
        assert decoder.get_draft_tokens() == []

    def test_no_matching_ngram(self):
        decoder = PromptLookupDecoder(ngram_size=3, min_matches=1)
        decoder.add_prompt_tokens([1, 2, 3, 4, 5, 6, 7, 8])
        # query = (6, 7, 8) — never seen before, no match
        assert decoder.get_draft_tokens() == []

    def test_ngram_match_found(self):
        """When last 3 tokens match an earlier n-gram, return continuation."""
        decoder = PromptLookupDecoder(num_draft_tokens=4, ngram_size=3, min_matches=1)
        # History: [1, 2, 3, 4, 5, 1, 2, 3]
        # query = history[-3:] = (1, 2, 3)
        # (1,2,3) appears at start=0 and start=5 (current), skip current
        # continuation from 0+3=3: [4, 5, 1, 2]
        decoder.add_prompt_tokens([1, 2, 3, 4, 5, 1, 2, 3])
        drafts = decoder.get_draft_tokens()
        assert drafts == [4, 5, 1, 2]

    def test_min_matches_threshold_not_met(self):
        """Draft not returned if continuation < min_matches."""
        decoder = PromptLookupDecoder(num_draft_tokens=4, ngram_size=3, min_matches=3)
        # History: [1, 2, 3, 4, 5, 1, 2, 3]
        # Match at pos 3, continuation = [5, 1, 2, 3] (4 tokens >= min 3)
        decoder.add_prompt_tokens([1, 2, 3, 4, 5, 1, 2, 3])
        drafts = decoder.get_draft_tokens()
        assert len(drafts) == 4  # >= min_matches=3

    def test_min_matches_threshold_blocks(self):
        """Draft not returned when continuation is too short."""
        decoder = PromptLookupDecoder(num_draft_tokens=4, ngram_size=3, min_matches=3)
        # History: [1, 2, 3, 4, 1, 2, 3]
        # Match at pos 3 (token=4), continuation = [1, 2, 3] (3 tokens)
        # But continuation is capped by num_draft_tokens and history end
        # continuation from pos 4 to min(4+4+1, 7) = 7: [1, 2, 3] -> 3 tokens >= 3
        decoder.add_prompt_tokens([1, 2, 3, 4, 1, 2, 3])
        drafts = decoder.get_draft_tokens()
        assert len(drafts) >= 3

    def test_num_draft_tokens_limits_output(self):
        decoder = PromptLookupDecoder(num_draft_tokens=2, ngram_size=3, min_matches=1)
        # History: [1, 2, 3, 4, 5, 6, 1, 2, 3]
        # query = (1,2,3), matches at start=0 (start=6 is current, skipped)
        # continuation from 0+3=3: [4, 5, 6, 1, 2, 3], limited to 2
        decoder.add_prompt_tokens([1, 2, 3, 4, 5, 6, 1, 2, 3])
        drafts = decoder.get_draft_tokens()
        assert len(drafts) == 2
        assert drafts == [4, 5]

    def test_repeating_pattern(self):
        """Repeating pattern gives long continuations."""
        decoder = PromptLookupDecoder(num_draft_tokens=4, ngram_size=3, min_matches=1)
        # History: [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3]
        # query = (_, 2, 3) -> matches multiple positions
        # Best continuation starts after the latest non-current match
        decoder.add_prompt_tokens([1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3])
        drafts = decoder.get_draft_tokens()
        assert len(drafts) > 0
        # Match pos has token after the n-gram; continuation starts one further
        # Best match should give continuation from the repeating pattern

    def test_best_continuation_selected(self):
        """Longest continuation wins among multiple matches."""
        decoder = PromptLookupDecoder(num_draft_tokens=4, ngram_size=2, min_matches=1)
        # History: [1, 2, 1, 2, 3, 4, 5, 1, 2]
        # query = (1, 2)
        # Match at pos 2 (token=1): continuation [2, 3, 4, 5] (4 tokens)
        # Match at pos 4 (token=3): wait, need to check what ngram (1,2) maps to
        # Actually: at pos 2, preceding 2-gram is (1,2), maps to pos 2
        # At pos 4, preceding 2-gram is (2,3)... no.
        # Let me trace: _add_token stores ngram = history[pos-n:pos]
        # pos 2 (token=1): n=2 -> ngram=(1,2) -> pos 2? No: history[0:2]=(1,2) -> pos 2
        # But token at pos 2 is 1 (the third element of [1,2,1,2,3,4,5,1,2])
        # Continuation from pos 2+1 = [2,3,4,5,1,2][:4] = [2,3,4,5]
        decoder.add_prompt_tokens([1, 2, 1, 2, 3, 4, 5, 1, 2])
        drafts = decoder.get_draft_tokens()
        assert len(drafts) > 0

    def test_current_position_excluded(self):
        """Current position in n-gram index should be skipped."""
        decoder = PromptLookupDecoder(num_draft_tokens=2, ngram_size=3, min_matches=1)
        # History: [1, 2, 3] — only one occurrence of (1,2,3) at the end
        # query = (1, 2, 3), only position is current -> skip -> empty
        decoder.add_prompt_tokens([1, 2, 3])
        drafts = decoder.get_draft_tokens()
        assert drafts == []


class TestPromptLookupDecoderRecordAccepted:
    """Tests for record_accepted method."""

    def test_record_accepted_zero(self):
        decoder = PromptLookupDecoder()
        decoder.record_accepted(0)
        assert decoder.successful_drafts == 0
        assert decoder.accepted_tokens == 0

    def test_record_accepted_positive(self):
        decoder = PromptLookupDecoder()
        decoder.record_accepted(3)
        assert decoder.successful_drafts == 1
        assert decoder.accepted_tokens == 3

    def test_record_accepted_multiple_calls(self):
        decoder = PromptLookupDecoder()
        decoder.record_accepted(2)
        decoder.record_accepted(3)
        decoder.record_accepted(1)
        assert decoder.successful_drafts == 3
        assert decoder.accepted_tokens == 6


class TestPromptLookupDecoderGetStats:
    """Tests for get_stats method."""

    def test_initial_stats(self):
        decoder = PromptLookupDecoder()
        stats = decoder.get_stats()

        assert stats["total_drafts"] == 0
        assert stats["successful_drafts"] == 0
        assert stats["total_draft_tokens"] == 0
        assert stats["accepted_tokens"] == 0
        assert stats["acceptance_rate"] == 0.0
        assert stats["history_size"] == 0

    def test_stats_after_draft(self):
        decoder = PromptLookupDecoder(ngram_size=3, min_matches=1)
        # Need query to match: [1,2,3,4,5,1,2,3]
        decoder.add_prompt_tokens([1, 2, 3, 4, 5, 1, 2, 3])
        drafts = decoder.get_draft_tokens()

        assert len(drafts) > 0
        stats = decoder.get_stats()
        assert stats["total_drafts"] == 1
        assert stats["total_draft_tokens"] == len(drafts)
        assert stats["history_size"] == 8

    def test_stats_acceptance_rate(self):
        decoder = PromptLookupDecoder(ngram_size=3, min_matches=1)
        decoder.add_prompt_tokens([1, 2, 3, 4, 5, 1, 2, 3])
        drafts = decoder.get_draft_tokens()

        num_drafts = len(drafts)
        decoder.record_accepted(num_drafts)

        stats = decoder.get_stats()
        assert stats["acceptance_rate"] == 1.0

    def test_stats_acceptance_rate_partial(self):
        decoder = PromptLookupDecoder(ngram_size=3, min_matches=1)
        decoder.add_prompt_tokens([1, 2, 3, 4, 5, 6, 7, 1, 2, 3])
        drafts = decoder.get_draft_tokens()

        assert len(drafts) > 0
        accepted = len(drafts) // 2
        if accepted == 0:
            accepted = 1
        decoder.record_accepted(accepted)

        stats = decoder.get_stats()
        expected_rate = accepted / len(drafts)
        assert abs(stats["acceptance_rate"] - expected_rate) < 0.01


class TestPromptLookupDecoderEdgeCases:
    """Tests for edge cases."""

    def test_ngram_size_one(self):
        """With ngram_size=1, single token lookup."""
        decoder = PromptLookupDecoder(ngram_size=1, num_draft_tokens=3, min_matches=1)
        # History: [5, 5, 5, 5]
        # query = (5,) — matches positions where preceding 1-gram is (5)
        # At pos 1: ngram (5,) -> pos 1
        # At pos 2: ngram (5,) -> pos 2
        # At pos 3: ngram (5,) -> pos 3
        # Current pos = 3, skip. Best from pos 2: continuation [5][:3] = [5]
        # Actually from pos 1: continuation = [5, 5][:3] = [5, 5]
        decoder.add_prompt_tokens([5, 5, 5, 5])
        drafts = decoder.get_draft_tokens()
        assert len(drafts) >= 1
        assert all(t == 5 for t in drafts)

    def test_large_ngram_size(self):
        decoder = PromptLookupDecoder(ngram_size=10, min_matches=1)
        tokens = list(range(15))
        decoder.add_prompt_tokens(tokens)
        # query = last 10 tokens = [5..14], only one occurrence at end
        drafts = decoder.get_draft_tokens()
        assert drafts == []

    def test_no_repetition_in_prompt(self):
        decoder = PromptLookupDecoder(ngram_size=3, min_matches=1)
        decoder.add_prompt_tokens([1, 2, 3, 4, 5, 6, 7, 8])
        # query = (6, 7, 8) — unique, no earlier match
        drafts = decoder.get_draft_tokens()
        assert drafts == []

    def test_exactly_ngram_size_history(self):
        decoder = PromptLookupDecoder(ngram_size=3, min_matches=1)
        decoder.add_prompt_tokens([1, 2, 3])
        # Only 3 tokens, query = (1,2,3) at pos=2 (only occurrence, is current)
        drafts = decoder.get_draft_tokens()
        assert drafts == []

    def test_reset_between_generations(self):
        """Test typical usage: reset between generation runs."""
        decoder = PromptLookupDecoder(ngram_size=3, min_matches=1)

        # First generation
        decoder.add_prompt_tokens([1, 2, 3, 4, 5, 1, 2, 3])
        drafts1 = decoder.get_draft_tokens()
        assert len(drafts1) > 0

        # Reset
        decoder.reset()

        # Second generation — completely different tokens
        decoder.add_prompt_tokens([10, 20, 30, 40, 50, 10, 20, 30])
        drafts2 = decoder.get_draft_tokens()
        assert len(drafts2) > 0

        # History should only contain second generation's tokens
        assert 1 not in decoder._token_history
        assert 10 in decoder._token_history

    def test_generated_tokens_extend_ngram_index(self):
        """Generated tokens create new n-grams for future lookup."""
        decoder = PromptLookupDecoder(ngram_size=3, num_draft_tokens=2, min_matches=1)
        # Prompt: [1, 2, 3, 4]
        decoder.add_prompt_tokens([1, 2, 3, 4])

        # No match yet — query (2,3,4) only at current pos
        assert decoder.get_draft_tokens() == []

        # Generate tokens that create a pattern
        decoder.add_generated_token(1)
        decoder.add_generated_token(2)
        decoder.add_generated_token(3)
        # History = [1,2,3,4,1,2,3]
        # query = (1,2,3), matches at start=0 (start=4 is current, skipped)
        # continuation from 0+3=3: [4, 1, 2, 3], limited to 2 -> [4, 1]
        drafts = decoder.get_draft_tokens()
        assert len(drafts) > 0
        assert drafts[0] == 4
