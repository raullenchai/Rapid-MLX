# SPDX-License-Identifier: Apache-2.0
"""
Channel-scoped stop-string matching for Harmony-format models (gpt-oss).

Harmony emits two channels the client cares about:

* ``analysis`` — chain-of-thought / reasoning (surfaces as
  ``reasoning_content`` in the OpenAI-shape response).
* ``final`` — the user-facing answer (surfaces as ``content``).

User-supplied ``stop=[...]`` sequences on ``/v1/chat/completions`` are
part of the **client-visible** wire contract — the client asks the
server to truncate ``content`` at that marker. They MUST NOT apply to
the analysis channel: agents like OpenHands CodeActAgent set
``stop=['</execute_ipython>', ...]`` and the model's CoT routinely
mentions those markers while reasoning about which action to take.
Applying the stop to the raw stream terminates the request mid-CoT and
``content`` never emits (issue #1049).

The helpers in this module let the scheduler (both ``Scheduler`` and
``MLLMScheduler``) scope user-supplied stops to just the final-channel
region of the decoded surface. The analysis-channel region and any
harmony control markers are left stop-agnostic — only harmony's own
channel delimiters (``<|end|>`` / ``<|return|>`` / ``<|channel|>``)
terminate the analysis channel, as the protocol defines.

Non-harmony models bypass this entirely (see
``harmony_family_from_tokenizer``); their stop-matching is unchanged.
"""

from __future__ import annotations

from typing import Any

# The sentinel that opens the final channel body. Emitted verbatim by
# the tokenizer as a single-token multi-byte sequence for gpt-oss
# family tokenizers; the raw decoded surface always contains the full
# literal once the model has switched to the final channel.
HARMONY_FINAL_MARKER = "<|channel|>final<|message|>"

# Harmony protocol control markers that terminate the final-channel
# body. Per the openai-harmony spec, ``<|end|>`` closes a message,
# ``<|return|>`` marks end-of-conversation-turn, and ``<|call|>``
# closes a commentary channel body carrying a tool call. Any of these
# appearing AFTER ``HARMONY_FINAL_MARKER`` in the decoded surface
# ends the final-channel body at the earliest occurrence.
#
# ``<|channel|>`` is intentionally NOT in this set even though it
# opens the next channel block: in the (extremely rare) analysis→final
# →analysis transition the caller's ``rfind`` re-anchors to the LATEST
# ``<|channel|>final<|message|>`` marker anyway, so treating ``<|channel|>``
# as a terminator would only add a false-positive risk against final-
# channel content that legitimately contains the literal string
# ``<|channel|>`` (e.g. answering a user question about harmony
# format — codex round-2 NIT).
HARMONY_FINAL_TERMINATORS = ("<|end|>", "<|return|>", "<|call|>")

# Cheap heuristic gate — any harmony-format output contains at least
# one of these sentinels early in the raw decoded surface. Used to
# short-circuit the final-channel search when the raw text obviously
# has no harmony wire envelope (defensive: the primary gate is the
# model-family flag on the scheduler; this belt-and-braces check
# avoids applying channel-scoped logic to text that doesn't look
# harmony-shaped even if the flag is somehow set on the wrong model).
_HARMONY_ENVELOPE_SENTINEL = "<|channel|>"


def find_harmony_final_span(decoded_so_far: str) -> tuple[int, int] | None:
    """Return the (start, end) byte offsets of the harmony final-channel
    body inside ``decoded_so_far``, or ``None`` if the model has not yet
    switched to the final channel.

    Semantics:

    * ``start`` is the byte offset immediately after the last
      ``<|channel|>final<|message|>`` marker in ``decoded_so_far``.
      Using ``rfind`` matches how the (rare) analysis-after-final path
      is handled: only the most recent final-channel opening is
      searchable.
    * ``end`` is the byte offset of the earliest control marker
      (``<|end|>``, ``<|return|>``, ``<|call|>``, ``<|channel|>``)
      after ``start``, or ``len(decoded_so_far)`` if none. That gives
      streaming callers a live growing window as the model emits final
      tokens, and a bounded window once a terminator appears.
    """
    if _HARMONY_ENVELOPE_SENTINEL not in decoded_so_far:
        return None
    marker_idx = decoded_so_far.rfind(HARMONY_FINAL_MARKER)
    if marker_idx < 0:
        return None
    body_start = marker_idx + len(HARMONY_FINAL_MARKER)
    end = len(decoded_so_far)
    for term in HARMONY_FINAL_TERMINATORS:
        pos = decoded_so_far.find(term, body_start)
        if pos != -1 and pos < end:
            end = pos
    return (body_start, end)


def find_stop_in_final_channel(
    decoded_so_far: str, stop_params: list[str]
) -> tuple[str, int] | None:
    """Search ``stop_params`` inside the harmony final-channel body only.

    Returns ``(stop_str, global_offset)`` for the earliest match, or
    ``None`` if:

    * the model has not yet emitted ``<|channel|>final<|message|>``, or
    * no user stop appears inside the final-channel body.

    ``global_offset`` is the offset inside ``decoded_so_far`` (not
    inside the final-channel substring) so the caller's trim math is a
    drop-in replacement for the pre-fix
    ``decoded_so_far.index(stop_str)`` path.
    """
    span = find_harmony_final_span(decoded_so_far)
    if span is None:
        return None
    body_start, body_end = span
    if body_start >= body_end:
        return None
    body = decoded_so_far[body_start:body_end]
    best: tuple[str, int] | None = None
    for stop_str in stop_params:
        if not stop_str:
            continue
        local_idx = body.find(stop_str)
        if local_idx == -1:
            continue
        global_idx = body_start + local_idx
        if best is None or global_idx < best[1]:
            best = (stop_str, global_idx)
    return best


def is_harmony_family_tokenizer(tokenizer: Any) -> bool:
    """Return True iff the tokenizer belongs to the gpt-oss/harmony
    family. Used as the model-level gate for scoping user-supplied
    stops to the final channel.

    Detection strategy:

    1. Prefer the vocab-based check (``<|channel|>`` marker present) —
       this matches how ``OutputRouter.from_tokenizer`` detects harmony
       shape and is definitive for any tokenizer with a ``get_vocab``
       API (HF fast tokenizers, mlx-lm wrappers).
    2. Fall back to the name-based allowlist
       (``_is_known_harmony_identity``) for tokenizers whose vocab is
       expensive to enumerate (mock tokenizers in unit tests, custom
       wrappers).

    Both checks are cheap and neither imports the openai-harmony
    optional dep; this helper is safe to call from the scheduler init
    path even when the harmony encoder is not installed.
    """
    if tokenizer is None:
        return False
    # (1) Vocab-based check — direct marker presence.
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if callable(get_vocab):
        try:
            vocab = get_vocab()
        except Exception:  # noqa: BLE001
            vocab = None
        if (
            isinstance(vocab, dict)
            and "<|channel|>" in vocab
            and "<|message|>" in vocab
        ):
            return True
    # (2) Name-based fallback. Imported lazily so a broken output_router
    # module (e.g. optional dep missing) does not cascade into the
    # scheduler import.
    name_or_path = getattr(tokenizer, "name_or_path", "") or ""
    if not name_or_path:
        return False
    try:
        from ..output_router_harmony import _is_known_harmony_identity

        return _is_known_harmony_identity(str(name_or_path))
    except Exception:  # noqa: BLE001
        return False
