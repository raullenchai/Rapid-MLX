# SPDX-License-Identifier: Apache-2.0
"""Token-level delta splitting for streaming-parser tests.

Direct port of vLLM ``tests/tool_parsers/utils.py:114-126``
(``split_string_into_token_deltas``) plus a ``stream_interval`` batching
variant used by vLLM ``tests/tool_parsers/test_hermes_tool_parser.py``
to vary delta granularity in parametrized tests.

Why we vary stream_interval: bugs surface differently at different
delta boundaries. vLLM issue #19056 (Hermes boolean-arg leak) only
manifested at ``stream_interval > 1``. Cluster issues #444 / #447 /
#448 / #455 / #468 / #480 each leak channel markers at specific delta
splits — parametrizing the interval is the cheapest way to fuzz the
exact byte boundary that triggers each leak.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def split_string_into_token_deltas(tokenizer: Any, text: str) -> list[str]:
    """Split ``text`` into per-token decoded fragments.

    Port of vLLM ``tools_parsers/utils.py:114-126``. Each returned
    string is the decoded suffix added by emitting one more token from
    the tokenizer. Joining them reconstructs the original text
    byte-for-byte. (Confirmed via ``test_streaming_invariants_round_trip``.)

    Args:
        tokenizer: Anything with ``.encode(text, add_special_tokens=False)
            -> list[int]`` and ``.decode(token_ids) -> str`` — covers
            ``PreTrainedTokenizerBase`` and our test fakes.
        text: The string to split.

    Returns:
        Per-token decoded fragments. ``"".join(deltas) == text``.
    """
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    previously_decoded_text = ""
    deltas: list[str] = []
    for i in range(1, len(token_ids) + 1):
        current_tokens = token_ids[:i]
        current_text = tokenizer.decode(current_tokens)
        new_text = current_text[len(previously_decoded_text) :]
        previously_decoded_text = current_text
        deltas.append(new_text)
    return deltas


def batch_deltas_with_stream_interval(
    deltas: Iterable[str], stream_interval: int
) -> list[str]:
    """Concatenate every ``stream_interval`` deltas into a single emission.

    Models with ``stream_interval=N`` emit every N decoded tokens, not
    every token. Same pattern as vLLM's hermes parser test parametrize
    over ``[1, 2, 3, 5, 8]`` — exercises the parser's ability to handle
    multi-token deltas, where channel markers may straddle a boundary
    (e.g. one delta ends with ``"<|chan"`` and the next starts with
    ``"nel|>"``).

    Args:
        deltas: Per-token fragments from ``split_string_into_token_deltas``.
        stream_interval: Number of per-token fragments to merge into one
            emitted delta. Must be >= 1.

    Returns:
        List of merged deltas. ``"".join(result) == "".join(deltas)``.
    """
    assert stream_interval >= 1, f"stream_interval must be >= 1; got {stream_interval}"

    deltas_list = list(deltas)
    if stream_interval == 1:
        return deltas_list

    batched: list[str] = []
    buffer = ""
    for i, piece in enumerate(deltas_list, start=1):
        buffer += piece
        if i % stream_interval == 0:
            batched.append(buffer)
            buffer = ""
    if buffer:
        # Flush any trailing fragments — keeps the round-trip property
        # ``"".join(result) == "".join(deltas)`` regardless of total
        # delta count vs stream_interval.
        batched.append(buffer)
    return batched
