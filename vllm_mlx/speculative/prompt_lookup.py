# SPDX-License-Identifier: Apache-2.0
"""
Prompt Lookup Decoding for speculative token generation.

This module implements Prompt Lookup Decoding, a draft-model-free approach
to speculative decoding that uses n-gram matching from the prompt and
generated text to predict future tokens.

Reference: https://github.com/apoorvumang/prompt-lookup-decoding
"""

import logging
from collections import defaultdict
import mlx.core as mx

logger = logging.getLogger(__name__)


class PromptLookupDecoder:
    """
    Prompt Lookup Decoder for speculative token generation.

    Uses n-gram matching to find repeating patterns in the prompt
    and generated text to speculate future tokens without a draft model.

    This is particularly effective for:
    - Code generation (repetitive patterns)
    - Structured text (JSON, XML, markdown)
    - Text with common phrases
    - Translation tasks (similar patterns)

    Args:
        num_draft_tokens: Number of tokens to draft (default: 4)
        ngram_size: Size of n-gram to match (default: 3)
        min_matches: Minimum number of matching tokens required (default: 2)
    """

    def __init__(
        self,
        num_draft_tokens: int = 4,
        ngram_size: int = 3,
        min_matches: int = 2,
    ):
        self.num_draft_tokens = num_draft_tokens
        self.ngram_size = ngram_size
        self.min_matches = min_matches

        # Token history for n-gram lookup
        self._token_history: list[int] = []

        # N-gram index: maps (token_1, ..., token_n) -> [positions]
        self._ngram_index: dict[tuple, list[int]] = defaultdict(list)

        # Statistics
        self.total_drafts = 0
        self.successful_drafts = 0
        self.total_draft_tokens = 0
        self.accepted_tokens = 0

    def reset(self):
        """Reset the decoder state for a new generation."""
        self._token_history = []
        self._ngram_index = defaultdict(list)

    def add_prompt_tokens(self, tokens: list[int]):
        """
        Add prompt tokens to the history for lookup.

        Args:
            tokens: List of prompt token IDs
        """
        for token in tokens:
            self._add_token(token)

    def _add_token(self, token: int):
        """Add a single token to history and update n-gram index."""
        self._token_history.append(token)

        # Update n-gram index for all n-gram sizes up to ngram_size
        pos = len(self._token_history) - 1
        for n in range(1, min(self.ngram_size + 1, pos + 1)):
            if pos >= n:
                ngram = tuple(self._token_history[pos - n : pos])
                self._ngram_index[ngram].append(pos)

    def add_generated_token(self, token: int):
        """
        Add a generated token to history.

        Args:
            token: Generated token ID
        """
        self._add_token(token)

    def get_draft_tokens(self) -> list[int]:
        """
        Get draft tokens based on n-gram lookup.

        Returns:
            List of draft token IDs (may be empty if no match found)
        """
        if len(self._token_history) < self.ngram_size:
            return []

        # Get the last ngram_size tokens as the query
        query = tuple(self._token_history[-self.ngram_size :])

        # Look up positions where this n-gram occurred
        positions = self._ngram_index.get(query, [])

        if not positions:
            return []

        # Find the best match (most recent, or one with longest continuation)
        draft_tokens = []
        best_continuation_length = 0

        for pos in positions:
            # Skip if this is the current position
            if pos == len(self._token_history) - 1:
                continue

            # Check if there are tokens after this position
            if pos < len(self._token_history) - 1:
                # Get continuation tokens
                continuation_end = min(
                    pos + self.num_draft_tokens + 1, len(self._token_history)
                )
                continuation = self._token_history[pos + 1 : continuation_end]

                if len(continuation) > best_continuation_length:
                    best_continuation_length = len(continuation)
                    draft_tokens = continuation[: self.num_draft_tokens]

        if len(draft_tokens) >= self.min_matches:
            self.total_drafts += 1
            self.total_draft_tokens += len(draft_tokens)
            return draft_tokens

        return []

    def record_accepted(self, num_accepted: int):
        """Record statistics about accepted draft tokens."""
        if num_accepted > 0:
            self.successful_drafts += 1
            self.accepted_tokens += num_accepted

    def get_stats(self) -> dict:
        """Get decoder statistics."""
        acceptance_rate = 0.0
        if self.total_draft_tokens > 0:
            acceptance_rate = self.accepted_tokens / self.total_draft_tokens

        return {
            "total_drafts": self.total_drafts,
            "successful_drafts": self.successful_drafts,
            "total_draft_tokens": self.total_draft_tokens,
            "accepted_tokens": self.accepted_tokens,
            "acceptance_rate": acceptance_rate,
            "history_size": len(self._token_history),
        }


def prompt_lookup_generate_step(
    prompt: mx.array,
    model,
    *,
    num_draft_tokens: int = 4,
    ngram_size: int = 3,
    max_tokens: int = 256,
    sampler=None,
    logits_processors=None,
    prompt_cache=None,
    prefill_step_size: int = 512,
):
    """
    Generator for token generation with prompt lookup speculation.

    This is a drop-in replacement for generate_step that uses n-gram
    matching for speculation instead of a draft model.

    Args:
        prompt: Input token array
        model: The main model
        num_draft_tokens: Number of tokens to draft
        ngram_size: N-gram size for matching
        max_tokens: Maximum tokens to generate
        sampler: Token sampler (default: argmax)
        logits_processors: Optional logits processors
        prompt_cache: Optional KV cache
        prefill_step_size: Prefill chunk size

    Yields:
        Tuple of (token, logprobs, from_draft)
    """
    from mlx_lm.models import cache

    y = prompt.astype(mx.uint32)

    # Create KV cache if not provided
    if prompt_cache is None:
        prompt_cache = cache.make_prompt_cache(model)

    sampler = sampler or (lambda x: mx.argmax(x, axis=-1))

    # Initialize prompt lookup decoder
    decoder = PromptLookupDecoder(
        num_draft_tokens=num_draft_tokens,
        ngram_size=ngram_size,
    )

    # Add prompt tokens to decoder
    decoder.add_prompt_tokens(prompt.tolist())

    def _step(tokens: mx.array, n_predict: int = 1):
        """Run model on tokens and get predictions."""
        logits = model(tokens[None], cache=prompt_cache)
        logits = logits[:, -n_predict:, :]

        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        sampled = sampler(logprobs)

        return sampled.squeeze(0), logprobs.squeeze(0)

    def _prefill(tokens: mx.array):
        """Process prompt tokens in chunks."""
        while tokens.size > prefill_step_size:
            model(tokens[:prefill_step_size][None], cache=prompt_cache)
            mx.eval([c.state for c in prompt_cache])
            tokens = tokens[prefill_step_size:]
            mx.clear_cache()
        return tokens

    # Prefill prompt
    y = _prefill(y)

    # Generate first token
    current_token, logprobs = _step(y)
    mx.eval(current_token, logprobs)

    ntoks = 0

    while ntoks < max_tokens:
        # Yield current token
        yield current_token.item(), logprobs, False
        ntoks += 1

        if ntoks >= max_tokens:
            break

        # Add token to decoder history
        decoder.add_generated_token(current_token.item())

        # Try to get draft tokens via n-gram lookup
        draft_tokens = decoder.get_draft_tokens()

        if draft_tokens and len(draft_tokens) > 0:
            # Speculative path: verify draft tokens
            verify_input = mx.array([current_token.item()] + draft_tokens, mx.uint32)
            verified_tokens, verified_logprobs = _step(
                verify_input, n_predict=len(draft_tokens) + 1
            )
            mx.eval(verified_tokens, verified_logprobs)

            verified_tokens = verified_tokens.tolist()

            # Check how many draft tokens were accepted
            n_accepted = 0
            for i, (draft_t, verify_t) in enumerate(
                zip(draft_tokens, verified_tokens[:-1])
            ):
                if draft_t == verify_t:
                    n_accepted += 1
                    ntoks += 1
                    decoder.add_generated_token(draft_t)
                    yield draft_t, verified_logprobs[i], True  # from_draft=True
                    if ntoks >= max_tokens:
                        break
                else:
                    break

            decoder.record_accepted(n_accepted)

            if ntoks >= max_tokens:
                break

            # Trim cache for rejected tokens
            if n_accepted < len(draft_tokens):
                cache.trim_prompt_cache(prompt_cache, len(draft_tokens) - n_accepted)

            # Next token is the first non-accepted verification result
            current_token = mx.array(verified_tokens[n_accepted], mx.uint32)
            logprobs = verified_logprobs[n_accepted]
        else:
            # Standard path: single token generation
            next_token, next_logprobs = _step(
                mx.array([current_token.item()], mx.uint32)
            )
            mx.eval(next_token, next_logprobs)
            current_token = next_token
            logprobs = next_logprobs

        if ntoks % 256 == 0:
            mx.clear_cache()

    # Log final stats
    stats = decoder.get_stats()
    if stats["total_drafts"] > 0:
        logger.info(
            f"Prompt Lookup stats: {stats['accepted_tokens']}/{stats['total_draft_tokens']} "
            f"tokens accepted ({stats['acceptance_rate']:.1%})"
        )
