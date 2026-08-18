# SPDX-License-Identifier: Apache-2.0
# Adapted from MTPLX context_copy.py.
# Copyright 2026 Youssof Altoukhi and MTPLX contributors.
"""Prompt-lookup proposals for high-overlap speculative decoding."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptLookupMatch:
    """A continuation found in the immutable prompt."""

    start: int
    matched_suffix: int
    tokens: tuple[int, ...]


class PromptLookupIndex:
    """Index prompt n-grams and propose their following token block.

    Generated text is used only as the lookup query.  It is never added to
    the index, which prevents self-repetition from turning into a fast loop.
    """

    def __init__(
        self,
        prompt: list[int] | tuple[int, ...],
        *,
        min_ngram: int = 6,
        max_ngram: int = 10,
        max_candidates: int = 32,
    ) -> None:
        if min_ngram < 2:
            raise ValueError("min_ngram must be at least 2")
        if max_ngram < min_ngram:
            raise ValueError("max_ngram must be >= min_ngram")
        if max_candidates < 1:
            raise ValueError("max_candidates must be positive")

        self.prompt = tuple(int(token) for token in prompt)
        self.min_ngram = int(min_ngram)
        self.max_ngram = int(max_ngram)
        self.max_candidates = int(max_candidates)
        self._positions: dict[tuple[int, ...], list[int]] = {}

        # Store the position immediately following each n-gram.  A position at
        # len(prompt) has no continuation and is intentionally excluded.
        for end in range(self.min_ngram, len(self.prompt)):
            gram = self.prompt[end - self.min_ngram : end]
            self._positions.setdefault(gram, []).append(end)

    def propose(
        self,
        generated: list[int] | tuple[int, ...],
        *,
        max_tokens: int = 24,
    ) -> PromptLookupMatch | None:
        """Return the best prompt continuation for ``generated``'s suffix."""

        if max_tokens < 1 or len(generated) < self.min_ngram:
            return None
        suffix = tuple(int(token) for token in generated[-self.min_ngram :])
        candidates = self._positions.get(suffix)
        if not candidates:
            return None

        best_start: int | None = None
        best_suffix = self.min_ngram - 1
        max_extension = self.max_ngram - self.min_ngram
        history_len = len(generated)
        for start in reversed(candidates[-self.max_candidates :]):
            extension = 0
            while (
                extension < max_extension
                and start - self.min_ngram - 1 - extension >= 0
                and history_len - self.min_ngram - 1 - extension >= 0
                and self.prompt[start - self.min_ngram - 1 - extension]
                == generated[history_len - self.min_ngram - 1 - extension]
            ):
                extension += 1
            matched = self.min_ngram + extension
            if matched > best_suffix:
                best_start = start
                best_suffix = matched
                if extension == max_extension:
                    break

        if best_start is None:
            return None
        proposal = self.prompt[best_start : best_start + max_tokens]
        if not proposal:
            return None
        return PromptLookupMatch(best_start, best_suffix, proposal)


__all__ = ["PromptLookupIndex", "PromptLookupMatch"]
