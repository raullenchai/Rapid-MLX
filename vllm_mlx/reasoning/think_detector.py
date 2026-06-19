# SPDX-License-Identifier: Apache-2.0
"""Autonomous ``<think>`` opener detection — runtime parser selection.

The historical dispatch lives in ``vllm_mlx.model_auto_config``: a
name-based regex picks ``reasoning_parser="deepseek_r1"`` (or a
variant) for each known model family that opens with ``<think>``. PR
#715 added an entry for VibeThinker; the 2026-06-19 systematic-fix PR
called the per-model regex pattern out as the antipattern to avoid and
this module is the first step on the deprecation path.

Strategy (post-deprecation):

  1. Aliases declare a boolean ``can_emit_think`` capability flag
     (instead of selecting a specific reasoning_parser implementation).
     The flag is observable from the upstream HF model card / README
     content during alias creation — no source-code regex needed.

  2. ``ThinkDetector`` runs against the first ~64 chars of the model's
     output. If those chars open with ``<think>`` (with or without
     leading whitespace), the detector routes the remainder through
     the ``BaseThinkingReasoningParser`` family (deepseek_r1 /
     vibethinker / qwen3 — all three share the same wire shape after
     the opener).

  3. Aliases that explicitly disable thinking
     (``reasoning_parser=null``, e.g. ``qwen3-vl-2b-4bit``) skip the
     detector and route output to ``content`` unconditionally.

For the present PR the dispatch in ``model_auto_config`` is unchanged
(removal exceeds the 300 LOC scope budget). The ``ThinkDetector``
class is a forward-compatibility hook — subclasses of
``BaseThinkingReasoningParser`` already share the autonomous-think
behaviour (Case 3 in ``BaseThinkingReasoningParser.extract_reasoning``
+ the ``NO_TAG_CONTENT_THRESHOLD`` knob), so once aliases gain
``can_emit_think`` the existing parser families absorb the work
without any new regex.
"""

from __future__ import annotations

import re

_AUTONOMOUS_THINK_OPENER = re.compile(r"^\s{0,32}<think>")


def looks_like_autonomous_think(prefix: str) -> bool:
    """Return True when ``prefix`` (typically the first N decoded
    characters of the model's output) opens with ``<think>``.

    Tolerates up to 32 chars of leading whitespace because models
    distilled on Qwen / DeepSeek-R1 sometimes emit a couple of
    space / newline tokens BEFORE the opener.

    This function takes NO model name and NO alias config — it makes
    a structural decision purely from the model's emitted bytes. That
    is the property that makes it a non-regex (in the antipattern
    sense) dispatch: the regex it does use describes the WIRE FORMAT,
    not any particular model name.
    """
    if not prefix:
        return False
    return _AUTONOMOUS_THINK_OPENER.match(prefix) is not None


class ThinkDetector:
    """Per-stream detector that picks a thinking parser at runtime.

    Currently UNWIRED — the legacy regex dispatch in
    ``vllm_mlx.model_auto_config`` still owns parser selection. This
    class is the migration target: once aliases gain a
    ``can_emit_think`` boolean, the dispatch becomes:

      * alias.reasoning_parser explicit  → use that
      * alias.can_emit_think is True     → ThinkDetector picks at
                                            runtime
      * neither                          → no reasoning extraction

    The detector itself is stateless across streams; instantiate once
    per request.
    """

    __slots__ = ("_decided", "_routes_to_thinking")

    # Number of opening bytes to inspect before committing. Matches
    # ``DeepSeekR1ReasoningParser.NO_TAG_CONTENT_THRESHOLD`` (64) by
    # default so the detector and the streaming parser agree on the
    # window.
    INSPECT_BYTES = 64

    def __init__(self) -> None:
        self._decided = False
        self._routes_to_thinking = False

    def feed(self, accumulated_text: str) -> bool:
        """Inspect ``accumulated_text`` and return True iff this stream
        should be routed through a thinking parser.

        Once a decision is made it sticks — repeated feeds with longer
        text don't flip the routing. The decision becomes final at
        ``len(accumulated_text) >= INSPECT_BYTES`` OR when the
        autonomous opener is detected in the prefix.
        """
        if self._decided:
            return self._routes_to_thinking
        if looks_like_autonomous_think(accumulated_text):
            self._routes_to_thinking = True
            self._decided = True
        elif len(accumulated_text) >= self.INSPECT_BYTES:
            self._routes_to_thinking = False
            self._decided = True
        return self._routes_to_thinking

    @property
    def is_decided(self) -> bool:
        return self._decided

    def reset(self) -> None:
        self._decided = False
        self._routes_to_thinking = False
