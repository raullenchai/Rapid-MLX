# SPDX-License-Identifier: Apache-2.0
"""
Regression tests for the Hy3 (Tencent Hunyuan 3) reasoning parser.

Hy3 emits ``<think:opensource>…</think:opensource>`` reasoning spans. The
parser normalizes the ``:opensource`` suffix to the plain ``<think>`` /
``</think>`` shape and delegates to the qwen3 parser, so every qwen3
semantic (Case 1/2/3/4, streaming multi-block, SSE-boundary withhold,
tool-call promotion, D-STOP-THINK finalize suppression) applies verbatim.
"""

from __future__ import annotations

import pytest

from rapid_mlx.reasoning import get_parser
from rapid_mlx.reasoning.hy3_parser import Hy3ReasoningParser, _normalize_hy3_tags


def test_parser_is_registered():
    """The parser must appear in the reasoning registry under both
    aliases so ``reasoning_parser="hy_v3"`` and ``reasoning_parser="hy3"``
    (CLI convenience) both resolve."""
    assert get_parser("hy_v3") is Hy3ReasoningParser
    assert get_parser("hy3") is Hy3ReasoningParser


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("<think:opensource>a</think:opensource>", "<think>a</think>"),
        ("<think>a</think>", "<think>a</think>"),  # plain — unchanged
        # Mixed open+close suffixes still normalize.
        ("<think:v1>a</think:opensource>", "<think>a</think>"),
        # Non-tag text is unchanged.
        ("normal text with no tags", "normal text with no tags"),
        # Empty-string safe.
        ("", ""),
    ],
)
def test_normalize_hy3_tags(raw, expected):
    assert _normalize_hy3_tags(raw) == expected


def test_extract_reasoning_suffixed_tags():
    """The canonical Hy3 emission — ``<think:opensource>…</think:opensource>``
    — must split cleanly into reasoning and content."""
    parser = Hy3ReasoningParser()
    r, c = parser.extract_reasoning(
        "<think:opensource>Let me think about it.</think:opensource>The answer is 42."
    )
    assert r == "Let me think about it."
    assert c == "The answer is 42."


def test_extract_reasoning_plain_tags_still_work():
    """The parser MUST accept the plain ``<think>`` shape too so a future
    Hy3 revision that drops the suffix (or a mixed-checkpoint dogfood
    session) doesn't regress."""
    parser = Hy3ReasoningParser()
    r, c = parser.extract_reasoning("<think>reasoning</think>content")
    assert r == "reasoning"
    assert c == "content"


def test_extract_reasoning_implicit_close_only():
    """Case-2 (chat template injects ``<think>`` into the prompt) — only
    the close tag appears in the output. Suffixed variant must route the
    same way."""
    parser = Hy3ReasoningParser()
    r, c = parser.extract_reasoning("reasoning here</think:opensource>final answer")
    assert r == "reasoning here"
    assert c == "final answer"


def test_extract_reasoning_no_tags():
    """Case-4 with ``enable_thinking`` unset — no tags → pure content."""
    parser = Hy3ReasoningParser()
    r, c = parser.extract_reasoning("just a response")
    assert r is None
    assert c == "just a response"


def test_streaming_tag_atomic_deltas():
    """The common streaming case — every tag arrives whole in a single
    SSE delta. Reasoning bytes route to ``reasoning``, post-close bytes
    to ``content``."""
    parser = Hy3ReasoningParser()
    parser.reset_state()

    def step(prev: str, delta: str):
        cur = prev + delta
        return parser.extract_reasoning_streaming(prev, cur, delta)

    prev = ""
    # Opener token — nothing emitted (structural).
    m1 = step(prev, "<think:opensource>")
    prev = "<think:opensource>"
    # After normalization the base parser sees ``<think>`` as opener.
    assert m1 is None
    # Reasoning bytes flow to ``reasoning``.
    m2 = step(prev, "Let me ")
    prev += "Let me "
    assert m2 is not None
    assert m2.reasoning == "Let me "
    assert m2.content is None
    m3 = step(prev, "think.")
    prev += "think."
    assert m3.reasoning == "think."
    # Close token — structural.
    m4 = step(prev, "</think:opensource>")
    prev += "</think:opensource>"
    assert m4 is None or m4.content in (None, "")
    # Post-close content bytes.
    m5 = step(prev, "Paris")
    assert m5 is not None
    assert m5.content == "Paris"


def test_streaming_suffix_tag_split_across_boundary_preserves_invariant():
    """Codex round-1 BLOCKING #2 regression test. When
    ``<think:opensource>`` straddles the SSE chunk boundary (e.g. delta
    1 ends with ``<think:opens``, delta 2 opens with ``ource>``), the
    partial-tag suffix MUST be withheld so the base qwen3 state machine
    sees consistent ``previous_norm``/``current_norm``/``delta_norm``.
    Without this fix, ``current_norm`` collapses the completed tag to
    ``<think>`` on delta 2 while ``previous_norm`` still ends with
    ``<think:opens``, breaking the base parser's invariant and leaking
    tag fragments into the wrong channel."""
    parser = Hy3ReasoningParser()
    parser.reset_state()

    prev = ""

    def step(delta: str):
        nonlocal prev
        cur = prev + delta
        msg = parser.extract_reasoning_streaming(prev, cur, delta)
        prev = cur
        return msg

    # Half the opener arrives.
    m1 = step("<think:opens")
    # Must not emit tag fragments — either None or reasoning-only up to
    # the withhold point (empty here since the whole delta is inside
    # the partial-tag suffix).
    assert m1 is None or (m1.content is None)
    # Second half completes the tag.
    m2 = step("ource>Hello")
    # After completion the reasoning body ``Hello`` should reach
    # ``reasoning`` (not ``content``).
    assert m2 is not None
    assert m2.reasoning == "Hello"
    assert m2.content is None
    # Close tag, also split — first half.
    m3 = step("</think:opens")
    assert m3 is None or (m3.content is None and m3.reasoning is None)
    # Second half of close + trailing content.
    m4 = step("ource>Paris")
    assert m4 is not None
    assert m4.content == "Paris"


def test_streaming_pre_colon_prefix_still_withholds():
    """Codex round-5 BLOCKING #2 regression test. Boundary split
    ``"<think"`` then ``":opensource>Hello"`` — the FIRST delta ends
    with just ``<think`` (no colon). The round-1 straddle regex
    required the colon to be present, so the ``<think`` prefix fell
    through to the qwen3 base withhold. Qwen3's base hold only reserves
    prefixes of the plain ``<think>`` shape and releases when the
    trailing char is not ``>`` — the next tick's ``:`` character caused
    qwen3 to release the hold and leak ``:opensource>`` as plain
    content.

    Widening the straddle regex to make the ``:LABEL`` suffix optional
    fixes this — ``<think`` and ``</think`` are now withheld until the
    tag either completes as bare ``<think>`` or gains a suffix and
    completes as ``<think:LABEL>``."""
    parser = Hy3ReasoningParser()
    parser.reset_state()

    prev = ""

    def step(delta: str):
        nonlocal prev
        cur = prev + delta
        msg = parser.extract_reasoning_streaming(prev, cur, delta)
        prev = cur
        return msg

    # Delta 1: ``<think`` alone — no colon yet. Must NOT leak the ``<think``
    # bytes to any channel; must NOT enter reasoning mode either.
    m1 = step("<think")
    assert m1 is None or (m1.content is None and m1.reasoning is None), (
        f"Colon-less <think prefix leaked to a channel: {m1!r}"
    )
    # Delta 2: ``:opensource>`` + reasoning body. The completed tag
    # should now enter reasoning mode; ``:opensource>`` MUST NOT surface
    # as content.
    m2 = step(":opensource>Hello")
    assert m2 is not None
    assert m2.content is None, (
        f":opensource> leaked as content — regressing round-5 BLOCKING #2: {m2!r}"
    )
    assert m2.reasoning == "Hello"


def test_streaming_pre_colon_close_prefix_still_withholds():
    """Codex round-5 BLOCKING #2 companion — same shape for the close
    tag. ``</think`` alone on delta N, ``:opensource>tail`` on delta
    N+1. Must not leak ``:opensource>`` as reasoning or content."""
    parser = Hy3ReasoningParser()
    parser.reset_state()

    prev = ""

    def step(delta: str):
        nonlocal prev
        cur = prev + delta
        msg = parser.extract_reasoning_streaming(prev, cur, delta)
        prev = cur
        return msg

    # Establish reasoning mode.
    step("<think:opensource>")
    step("body")
    # Now split the close tag: bare ``</think`` first.
    m1 = step("</think")
    assert m1 is None or (m1.content is None and m1.reasoning is None), (
        f"Colon-less </think close prefix leaked to a channel: {m1!r}"
    )
    # Complete with suffix + tail content.
    m2 = step(":opensource>tail")
    assert m2 is not None
    assert m2.reasoning is None
    # The tail content MUST reach ``content`` unadulterated — no
    # ``:opensource>`` leak.
    assert m2.content == "tail", f"Close suffix leaked with the tail content: {m2!r}"


def test_finalize_streaming_delegates_to_qwen3():
    """``finalize_streaming`` MUST inherit qwen3's D-STOP-THINK
    suppression semantics on truncation. Smoke-test that the delegation
    path doesn't crash on a suffixed accumulated buffer."""
    parser = Hy3ReasoningParser()
    parser.reset_state()
    # A ``<think:opensource>`` opener with no close — the base returns
    # None (no correction) by default; qwen3's override returns
    # reasoning on ``finish_reason="length"``.
    msg = parser.finalize_streaming(
        "<think:opensource>partial thought",
        finish_reason="length",
    )
    assert msg is not None
    assert msg.reasoning == "partial thought"


def test_is_open_in_think_recognises_suffixed_opener():
    """The finalize-on-truncation router calls ``is_open_in_think`` to
    decide whether to route the buffer as reasoning. The Hy3 parser
    MUST recognise a suffixed opener as such."""
    parser = Hy3ReasoningParser()
    assert parser.is_open_in_think("<think:opensource>partial") is True
    assert parser.is_open_in_think("no think here") is False
    assert (
        parser.is_open_in_think("<think:opensource>closed</think:opensource>tail")
        is False
    )


def _collect_reasoning_stream(full: str) -> tuple[str, str]:
    """Char-by-char drive a fresh parser over ``full``; return
    ``(content, reasoning)`` accumulated across deltas + finalize."""
    parser = Hy3ReasoningParser()
    parser.reset_state()
    prev = ""
    content = ""
    reasoning = ""
    for ch in full:
        cur = prev + ch
        msg = parser.extract_reasoning_streaming(prev, cur, ch)
        prev = cur
        if msg is None:
            continue
        if msg.content:
            content += msg.content
        if msg.reasoning:
            reasoning += msg.reasoning
    # Flush any withheld trailing straddle suffix at stream end.
    fin = parser.finalize_streaming(prev)
    if fin is not None:
        if fin.content:
            content += fin.content
        if fin.reasoning:
            reasoning += fin.reasoning
    return content, reasoning


def test_streaming_falsified_think_prefix_in_content_not_dropped():
    """codex R7 BLOCKING #3 regression. After a completed think block, content
    that starts with a partial-think prefix which then FALSIFIES into ordinary
    text (``<think`` → ``<thinking``) must surface intact — NOT be dropped or
    corrupted. The pre-fix held only the full ``<think`` root, so the withheld
    span grew non-monotonically (``see <thin`` held nothing, then ``see
    <think`` held 6 bytes), the visible span retreated, and the base machine
    re-emitted already-shown bytes as garbage (``see `` → ``k>see``). Widening
    the straddle matcher to every tag PREFIX keeps the hold monotonic."""
    content, reasoning = _collect_reasoning_stream(
        "<think:opensource>ok</think:opensource>see <thinking here"
    )
    assert reasoning == "ok"
    assert content == "see <thinking here"


def test_streaming_falsified_think_prefix_inside_reasoning_not_dropped():
    """The same falsified prefix INSIDE the reasoning span (``<thinker``) must
    survive as reasoning, and the post-close content must be clean."""
    content, reasoning = _collect_reasoning_stream(
        "<think:opensource>weigh <thinker note</think:opensource>answer"
    )
    assert reasoning == "weigh <thinker note"
    assert content == "answer"


def test_streaming_content_ending_in_held_lt_released_at_finalize():
    """codex R8 BLOCKING: content that ENDS in a lone ``<`` (or ``<think``) —
    a partial-tag prefix the streaming path withholds every tick — must be
    released at finalize, not dropped. Our widened straddle hold reserves even a
    lone ``<``, so ``finalize_streaming`` re-surfaces the held non-tag suffix."""
    content, reasoning = _collect_reasoning_stream(
        "<think:opensource>r</think:opensource>trailing <"
    )
    assert reasoning == "r"
    assert content == "trailing <"


def test_streaming_content_ending_in_held_think_prefix_released_at_finalize():
    """Same release for a longer held prefix (``<think``) that never completed
    a tag — the full ``done <think`` must reach content."""
    content, reasoning = _collect_reasoning_stream(
        "<think:opensource>r</think:opensource>done <think"
    )
    assert reasoning == "r"
    assert content == "done <think"


def test_finalize_does_not_double_emit_held_tail():
    """codex R9 BLOCKING: ``finalize_streaming`` must NOT double-emit the held
    tail when the base finalize already surfaced it. The released tail appears
    EXACTLY once — assert the trailing ``<think`` occurs a single time in the
    accumulated content (no ``<think<think`` duplication)."""
    content, _reasoning = _collect_reasoning_stream(
        "<think:opensource>r</think:opensource>done <think"
    )
    assert content == "done <think"
    assert content.count("<think") == 1


def test_finalize_direct_call_appends_held_tail_once():
    """Direct ``finalize_streaming`` on a buffer ending in a held ``<`` returns
    the tail once — and calling it again on the same buffer is idempotent (no
    growth), guarding the not-already-present check."""
    parser = Hy3ReasoningParser()
    parser.reset_state()
    # Prime streaming state up to the held boundary.
    full = "<think:opensource>r</think:opensource>tail <"
    prev = ""
    for ch in full:
        cur = prev + ch
        parser.extract_reasoning_streaming(prev, cur, ch)
        prev = cur
    fin = parser.finalize_streaming(full)
    assert fin is not None
    # The held "<" is released as content exactly once.
    assert (fin.content or "").endswith("<")
    assert (fin.content or "").count("<") == 1


def test_finalize_streaming_partial_close_tag_not_leaked_as_reasoning():
    """codex R17 BLOCKING: a stream truncated mid-CLOSE-tag while still inside
    an open think span (``<think:opensource>r</think`` — note ``</think`` has
    NO closing ``>``) must NOT leak the incomplete ``</think`` delimiter into
    reasoning OR content. A partial close-tag prefix is opaque markup the model
    started emitting, not user-visible text — drop it. Contrast R8: a partial
    OPEN-tag prefix in already-closed content (``done <think``) IS legitimate
    content and is still surfaced (asserted above)."""
    parser = Hy3ReasoningParser()
    parser.reset_state()
    dm = parser.finalize_streaming("<think:opensource>r</think")
    reasoning = getattr(dm, "reasoning", None) if dm else None
    content = getattr(dm, "content", None) if dm else None
    # The incomplete close delimiter must not appear in either channel.
    assert "</think" not in (reasoning or ""), (
        f"partial close tag leaked into reasoning: {reasoning!r}"
    )
    assert "</think" not in (content or ""), (
        f"partial close tag leaked into content: {content!r}"
    )
    # No stray ``<`` markup from the delimiter should survive anywhere either.
    assert "</" not in (reasoning or "") and "</" not in (content or "")


def test_finalize_streaming_wellformed_close_tag_surfaces_reasoning():
    """Control for R17: a PROPERLY closed think span
    (``<think:opensource>r</think:opensource>``) surfaces ``r`` as reasoning
    with no stray tag markup — the partial-close drop must not disturb the
    well-formed path."""
    parser = Hy3ReasoningParser()
    parser.reset_state()
    reasoning, content = parser.extract_reasoning(
        "<think:opensource>r</think:opensource>"
    )
    assert reasoning is not None and "r" in reasoning
    assert "<think" not in reasoning and "</think" not in reasoning
    # No leftover delimiter markup in content either.
    assert "</think" not in (content or "") and "<think" not in (content or "")
