# SPDX-License-Identifier: Apache-2.0
"""
Incremental token decoder with multi-byte character safety.

This module provides ``IncrementalDecoder``, used by the Scheduler to safely decode
streaming tokens without emitting broken multi-byte characters (emoji, CJK).
"""


class IncrementalDecoder:
    """Decode tokens incrementally with multi-byte character safety.

    Accumulates token IDs and re-decodes from scratch each step,
    computing deltas from the last good position. If a delta contains
    U+FFFD (incomplete multi-byte sequence), holds back until more
    tokens complete the character.

    Example::

        decoder = IncrementalDecoder(tokenizer, skip_special_tokens=False)
        for tok_id in stream:
            delta = decoder.add_token(tok_id)
            if delta:
                send_to_client(delta)
    """

    __slots__ = ("_tokenizer", "_skip_special_tokens", "_token_ids", "_prev_text")

    def __init__(self, tokenizer, skip_special_tokens: bool = False):
        self._tokenizer = tokenizer
        self._skip_special_tokens = skip_special_tokens
        self._token_ids: list[int] = []
        self._prev_text: str = ""

    def add_token(self, token_id: int) -> str:
        """Add a token and return the safe new text delta.

        Returns an empty string if the new token creates an incomplete
        multi-byte character (U+FFFD), deferring output until the
        character is complete.
        """
        self._token_ids.append(token_id)
        full_text = self._tokenizer.decode(
            self._token_ids, skip_special_tokens=self._skip_special_tokens
        )
        delta = full_text[len(self._prev_text) :]
        if "\ufffd" in delta:
            return ""  # hold back until character completes
        self._prev_text = full_text
        return delta

    def get_full_text(self) -> str:
        """Get the full decoded text so far (re-decodes all tokens)."""
        if not self._token_ids:
            return ""
        return self._tokenizer.decode(
            self._token_ids, skip_special_tokens=self._skip_special_tokens
        )

    @property
    def token_ids(self) -> list[int]:
        """Read-only access to accumulated token IDs."""
        return self._token_ids

    def reset(self):
        """Reset state for a new sequence."""
        self._token_ids.clear()
        self._prev_text = ""
