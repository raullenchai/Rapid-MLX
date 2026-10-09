# SPDX-License-Identifier: Apache-2.0
"""Tokenize prompt variants by re-encoding only the text that differs.

The prefix-boundary probe renders two or three variants of the request's chat
prompt (without the generation marker, with a placeholder next turn) and
compares their tokens with the real prompt's.  In an agent session those
variants share tens of thousands of tokens of system prompt, tools and history
with the real prompt, so encoding each one in full repeats the most expensive
host-side step of a warm turn for no new information.

A tokenizer splits its input on added tokens (``<|im_end|>`` and friends)
before any merge runs, so the text before an added token is tokenized
independently of the text after it.  When two rendered prompts share a head
that ends at such a token, the variant's tokens are the real prompt's tokens
for that head followed by the encoding of the variant's own tail.  That claim
is checked on the real prompt (whose full tokens are known) before it is used;
any mismatch, missing tokenizer surface or exception returns ``None`` and the
caller encodes the variant in full, exactly as before.

The result only feeds the prefix-boundary heuristic: the scheduler always
snapshots the real prompt's own token prefix, so a variant that tokenizes
differently from what this module predicts (for example an added token
matched after normalization rather than literally) can cost the next turn
some cache reuse but cannot change any output.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

# How far back from the first differing character to look for an added token.
# Chat templates close every message with one, so a cut is normally found
# within a single message; a miss simply falls back to a full encode.
_SEARCH_WINDOW_CHARS = 16384


def common_prefix_length(left: str, right: str) -> int:
    """Return the length of the longest common prefix of two strings."""

    high = min(len(left), len(right))
    if left[:high] == right[:high]:
        return high
    low = 0
    # Invariant: the first ``low`` characters match, the first ``high`` do not.
    while high - low > 1:
        middle = (low + high) // 2
        if left[:middle] == right[:middle]:
            low = middle
        else:
            high = middle
    return low


def added_token_markers(tokenizer: Any) -> tuple[str, ...]:
    """Return the tokenizer's added-token strings, or ``()`` when unavailable."""

    get_added_vocab = getattr(tokenizer, "get_added_vocab", None)
    if not callable(get_added_vocab):
        return ()
    try:
        vocab = get_added_vocab()
    except Exception:
        return ()
    if not isinstance(vocab, dict):
        return ()
    return tuple(marker for marker in vocab if isinstance(marker, str) and marker)


def _last_marker_end(text: str, limit: int, markers: Sequence[str]) -> int:
    """End offset of the last marker that lies entirely within ``text[:limit]``."""

    start = max(0, limit - _SEARCH_WINDOW_CHARS)
    best = 0
    for marker in markers:
        index = text.rfind(marker, start, limit)
        if index >= 0:
            best = max(best, index + len(marker))
    return best


def _marker_spans(text: str, cut: int, markers: Sequence[str]) -> bool:
    """Whether some marker occurrence starts before ``cut`` and ends after it.

    Added tokens are matched longest-first, so a longer added token that
    starts inside the head could swallow characters past the cut on one
    text and not on the other.  Such a cut does not split both texts the
    same way and must not be used.
    """

    for marker in markers:
        width = len(marker)
        # Any occurrence wholly inside this window starts at or after
        # ``cut - width + 1`` and ends at or before ``cut + width - 1``, so it
        # starts before the cut and ends after it.
        if text.find(marker, max(0, cut - width + 1), cut + width - 1) >= 0:
            return True
    return False


def encode_sharing_head(
    real_text: str,
    real_tokens: Sequence[int],
    variant_text: str,
    *,
    encode_tail: Callable[[str], Sequence[int]],
    markers: Sequence[str],
) -> list[int] | None:
    """Tokenize ``variant_text`` reusing ``real_tokens`` for the shared head.

    ``encode_tail`` must encode text without adding special tokens (no BOS),
    because the tail is spliced after tokens that already carry them.  Returns
    ``None`` whenever the shortcut cannot be proven on ``real_text``.
    """

    if not markers or not real_tokens:
        return None
    shared = common_prefix_length(real_text, variant_text)
    if shared < 2:
        return None
    # Cut strictly inside the shared text so both tails begin with the same
    # character; a marker that merely ends at the divergence point could be
    # the prefix of a longer added token on one side only.
    cut = _last_marker_end(real_text, shared - 1, markers)
    if cut <= 0:
        return None
    if _marker_spans(real_text, cut, markers) or _marker_spans(
        variant_text, cut, markers
    ):
        return None
    try:
        real_tail = [int(token) for token in encode_tail(real_text[cut:])]
        head_length = len(real_tokens) - len(real_tail)
        if head_length <= 0:
            return None
        if [int(token) for token in real_tokens[head_length:]] != real_tail:
            return None
        variant_tail = [int(token) for token in encode_tail(variant_text[cut:])]
    except Exception:
        return None
    return [int(token) for token in real_tokens[:head_length]] + variant_tail


__all__ = [
    "added_token_markers",
    "common_prefix_length",
    "encode_sharing_head",
]
