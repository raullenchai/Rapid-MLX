# SPDX-License-Identifier: Apache-2.0
"""
Per-request n-gram speculative draft state, gated by `<think>` block,
tool-call XML region, per-request self-tuning, and adaptive K based on
n-gram match confidence.

Wires `PromptLookupDecoder` into the MTP scheduler path with several
extras:

- Token-level `<think>` state machine (Qwen/DeepSeek style).
- Text-level `<tool_call>` state machine (Qwen3 multi-token BPE marker).
- Confidence-aware lookup: compute K based on how many prior occurrences
  of the n-gram had the *same* continuation tail.
- Self-tuning: after a warmup window, disable drafting for a request
  whose running acceptance rate is below threshold.
- Tool-call gate: skip drafting inside `<tool_call>...</tool_call>`.
"""

from __future__ import annotations

import logging
from collections import Counter

from .prompt_lookup import PromptLookupDecoder

logger = logging.getLogger(__name__)


class ThinkStateTracker:
    """Track `<think>...</think>` nesting via raw token IDs.

    Qwen3.6 / DeepSeek emit `<think>` as a single special token, so a
    direct token-equality check is enough.
    """

    def __init__(
        self, think_start_id: int | None, think_end_id: int | None
    ) -> None:
        self.think_start_id = think_start_id
        self.think_end_id = think_end_id
        self.thinking = False

    def feed(self, token_id: int) -> None:
        if self.think_start_id is not None and token_id == self.think_start_id:
            self.thinking = True
        elif self.think_end_id is not None and token_id == self.think_end_id:
            self.thinking = False

    def feed_many(self, token_ids) -> None:
        for t in token_ids:
            self.feed(int(t))


class TextToolCallTracker:
    """Track `<tool_call>...</tool_call>` regions via incremental decode.

    Qwen3 tool call markers are tokenized as multi-token BPE sequences,
    so a single-token equality check (like ThinkStateTracker) does not
    work. Instead we maintain a rolling decoded-text buffer and look for
    the literal markers as substrings.

    Buffer is bounded; older text is trimmed to keep memory flat across
    long generations.
    """

    START_MARKER = "<tool_call>"
    END_MARKER = "</tool_call>"
    BUFFER_TRIM_THRESHOLD = 1024
    BUFFER_KEEP_TAIL = 512

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
        self.buffer = ""
        self.tool_call_depth = 0
        self._enabled = tokenizer is not None and hasattr(tokenizer, "decode")

    @property
    def in_tool_call(self) -> bool:
        return self.tool_call_depth > 0

    @property
    def enabled(self) -> bool:
        return self._enabled

    def feed(self, token_id: int) -> None:
        if not self._enabled:
            return
        try:
            piece = self.tokenizer.decode([int(token_id)])
        except Exception:
            return
        if not piece:
            return
        self.buffer += piece
        if len(self.buffer) > self.BUFFER_TRIM_THRESHOLD:
            # Keep enough tail to span across an in-flight marker that
            # straddles the trim boundary.
            self.buffer = self.buffer[-self.BUFFER_KEEP_TAIL :]
        self._scan_buffer()

    def feed_many(self, token_ids) -> None:
        if not self._enabled:
            return
        for t in token_ids:
            self.feed(int(t))

    def _scan_buffer(self) -> None:
        """Consume markers from the buffer head, updating depth."""
        # Loop because a single decode chunk could contain start + end
        # in sequence (e.g. an empty tool call or markers reaching the
        # buffer in the same chunk).
        while True:
            if self.tool_call_depth > 0:
                end_pos = self.buffer.find(self.END_MARKER)
                if end_pos == -1:
                    return
                self.tool_call_depth -= 1
                self.buffer = self.buffer[end_pos + len(self.END_MARKER) :]
            else:
                start_pos = self.buffer.find(self.START_MARKER)
                if start_pos == -1:
                    # Trim the head so substring search stays cheap.
                    if len(self.buffer) > len(self.START_MARKER):
                        self.buffer = self.buffer[
                            -(len(self.START_MARKER) - 1) :
                        ] if len(self.buffer) >= len(self.START_MARKER) else self.buffer
                    return
                self.tool_call_depth += 1
                self.buffer = self.buffer[start_pos + len(self.START_MARKER) :]


class NgramRequestState:
    """Per-request n-gram drafter with gating, adaptive K, and self-tune.

    The decoder is seeded with the prompt tokens so first-step lookups
    can hit prefix patterns. The think tracker is also fed the prompt
    so the initial state matches the chat template (Qwen reasoning
    models inject `<think>` at the assistant prefix → tracker starts
    in thinking=True, ready to draft on the very first generated
    token). The tool-call tracker is NOT seeded with the prompt — it
    only tracks the assistant's own emissions because tool-call XML
    only appears in the assistant turn.

    Per-request running stats (`tokens_drafted`, `tokens_accepted`)
    drive the self-tune disable; once a request's running acceptance
    drops below `self_tune_disable_threshold` after warmup, drafting
    is suppressed for the rest of that request.
    """

    SELF_TUNE_WARMUP_TOKENS = 32
    SELF_TUNE_BUMP_RATE = 0.75

    def __init__(
        self,
        prompt_tokens,
        think_start_id: int | None,
        think_end_id: int | None,
        num_draft_tokens: int = 4,
        ngram_size: int = 3,
        min_matches: int = 2,
        only_in_think: bool = True,
        # New: confidence + adaptive + self-tune + tool_call.
        min_occurrences: int = 1,
        adaptive_k: bool = True,
        skip_tool_calls: bool = True,
        self_tune: bool = True,
        self_tune_disable_threshold: float = 0.30,
        tokenizer=None,
    ) -> None:
        self.decoder = PromptLookupDecoder(
            num_draft_tokens=num_draft_tokens,
            ngram_size=ngram_size,
            min_matches=min_matches,
        )
        self.tracker = ThinkStateTracker(think_start_id, think_end_id)
        self.tool_call_tracker = (
            TextToolCallTracker(tokenizer) if skip_tool_calls else None
        )
        self.only_in_think = only_in_think
        self.skip_tool_calls = skip_tool_calls
        self.min_occurrences = max(1, int(min_occurrences))
        self.adaptive_k = bool(adaptive_k)
        self.self_tune = bool(self_tune)
        self.self_tune_disable_threshold = float(self_tune_disable_threshold)

        prompt_list = [int(t) for t in prompt_tokens]
        self.decoder.add_prompt_tokens(prompt_list)
        self.tracker.feed_many(prompt_list)
        # tool_call tracker is intentionally NOT fed the prompt: tool
        # calls only appear in assistant generations, and seeding with
        # prompt text would also include user-typed `<tool_call>` text
        # (rare but possible) which would falsely flip state.

        # Stats — exposed via get_stats() for aggregate logging.
        self.drafts_attempted = 0
        self.tokens_drafted = 0
        self.tokens_accepted = 0
        self._self_tune_disabled = False

    @property
    def thinking(self) -> bool:
        return self.tracker.thinking

    @property
    def in_tool_call(self) -> bool:
        return (
            self.tool_call_tracker is not None
            and self.tool_call_tracker.in_tool_call
        )

    @property
    def self_tune_disabled(self) -> bool:
        return self._self_tune_disabled

    def feed_token(self, token_id: int) -> None:
        """Add an emitted token to history; updates decoder + trackers."""
        tid = int(token_id)
        self.decoder.add_generated_token(tid)
        self.tracker.feed(tid)
        if self.tool_call_tracker is not None:
            self.tool_call_tracker.feed(tid)

    def feed_many(self, token_ids) -> None:
        for t in token_ids:
            self.feed_token(t)

    def should_draft(self) -> bool:
        """Return True iff this step is eligible for n-gram drafting.

        Combines three gates:
        1. Think-block gate (`only_in_think`).
        2. Tool-call gate (`skip_tool_calls`).
        3. Self-tune gate (disabled once running acceptance drops below
           threshold after warmup).
        """
        if self._self_tune_disabled:
            return False
        if self.only_in_think and not self.tracker.thinking:
            return False
        if self.skip_tool_calls and self.in_tool_call:
            return False
        return True

    def get_drafts(self) -> list[int]:
        """Backwards-compatible: drafts based on current decoder history."""
        return self.decoder.get_draft_tokens()

    def lookup_drafts_with_pending(
        self, pending_token: int, max_k: int | None = None
    ) -> list[int]:
        """Return drafts treating ``pending_token`` as a virtual tail token.

        Applies confidence-aware filtering:
        - Group prior occurrences of the n-gram by their *continuation*
          tail (next K tokens after the n-gram).
        - The most common continuation gets selected as the candidate.
        - If its frequency is below `min_occurrences`, return [].
        - If `adaptive_k` is True, cap K by frequency:
          K_cap = min(max_k, frequency + 1).

        Does not mutate decoder/tracker state.
        """
        decoder = self.decoder
        history = decoder._token_history
        n = decoder.ngram_size
        K_request = max_k if max_k is not None else decoder.num_draft_tokens
        if K_request <= 0:
            return []
        if len(history) + 1 < n:
            return []
        if n == 1:
            query = (int(pending_token),)
        else:
            query = tuple(history[-(n - 1) :]) + (int(pending_token),)

        positions = decoder._ngram_index.get(query, [])
        if not positions:
            return []

        # Build continuations grouped by tail. Use up to K_request tokens
        # of tail per occurrence so frequency reflects matches at the
        # requested depth (a longer continuation length only counts if
        # there are at least that many history tokens after the match).
        current_start_virtual = len(history) + 1 - n
        continuations: list[tuple[int, ...]] = []
        for start in positions:
            if start == current_start_virtual:
                continue
            cont_begin = start + n
            cont_end = min(cont_begin + K_request, len(history))
            cont = tuple(history[cont_begin:cont_end])
            if cont:
                continuations.append(cont)
        if not continuations:
            return []

        # Score by frequency of the EXACT prefix (so partial agreement
        # at depth d counts toward depth d, not full continuation).
        # Build a per-depth frequency table of the most-common token at
        # each depth, conditional on prior tokens matching.
        # Simpler version: pick the most common full continuation; trim
        # length by its occurrence count.
        counter = Counter(continuations)
        most_common, freq = counter.most_common(1)[0]

        if freq < self.min_occurrences:
            return []

        K_max = K_request
        if self.adaptive_k:
            # freq=1 → K=2, freq=2 → K=3, freq=3 → K=4 (capped at K_request).
            K_max = max(1, min(K_request, freq + 1))

        drafts = list(most_common[:K_max])
        if len(drafts) < decoder.min_matches:
            return []

        # Cycle guard: if a prefix of length p>=3 of [pending, *drafts]
        # equals the same-length suffix of recent history, the model is
        # about to repeat a token cycle of period <= p. Suppress the
        # draft so spec-decode does not amplify the loop. Cheap: O(L^2)
        # worst case where L = len(drafts)+1 (small, ~5–7), only runs
        # after a draft is otherwise accepted.
        predicted = [int(pending_token)] + drafts
        max_p = min(len(predicted), len(history))
        for p in range(max_p, 2, -1):
            if list(history[-p:]) == predicted[:p]:
                return []

        decoder.total_drafts += 1
        decoder.total_draft_tokens += len(drafts)
        return drafts

    def record_outcome(self, drafted: int, accepted: int) -> None:
        if drafted <= 0:
            return
        self.drafts_attempted += 1
        self.tokens_drafted += drafted
        self.tokens_accepted += accepted

        # Self-tune: disable for the rest of the request if running
        # acceptance is bad after warmup.
        if (
            self.self_tune
            and not self._self_tune_disabled
            and self.tokens_drafted >= self.SELF_TUNE_WARMUP_TOKENS
        ):
            rate = self.tokens_accepted / max(1, self.tokens_drafted)
            if rate < self.self_tune_disable_threshold:
                self._self_tune_disabled = True
                logger.debug(
                    "[ngram] self-tune disabled drafter "
                    "(running rate=%.2f, threshold=%.2f)",
                    rate,
                    self.self_tune_disable_threshold,
                )

    def running_acceptance_rate(self) -> float | None:
        if self.tokens_drafted <= 0:
            return None
        return self.tokens_accepted / self.tokens_drafted

    def get_stats(self) -> dict:
        rate = (
            self.tokens_accepted / self.tokens_drafted
            if self.tokens_drafted > 0
            else 0.0
        )
        return {
            "drafts_attempted": self.drafts_attempted,
            "tokens_drafted": self.tokens_drafted,
            "tokens_accepted": self.tokens_accepted,
            "acceptance_rate": rate,
            "self_tune_disabled": self._self_tune_disabled,
            "in_tool_call": self.in_tool_call,
            "thinking": self.thinking,
        }


def lookup_think_token_ids(tokenizer) -> tuple[int | None, int | None]:
    """Resolve `<think>` and `</think>` token IDs from a tokenizer vocab.

    Returns (None, None) if the model doesn't have these tokens. In that
    case n-gram-only-in-think gating cannot work and the caller should
    skip ngram setup (or set only_in_think=False).
    """
    actual = tokenizer
    if hasattr(actual, "tokenizer") and not hasattr(actual, "get_vocab"):
        actual = actual.tokenizer
    try:
        vocab = actual.get_vocab()
    except Exception:
        return (None, None)
    return vocab.get("<think>"), vocab.get("</think>")
