# SPDX-License-Identifier: Apache-2.0
"""Pins the engine non-streaming path's use of ``OutputRouter`` for
channel-aware content/reasoning extraction — the root-cause fix for
issue #442.

Before this change, the engine produced ``GenerationOutput.text`` via
``clean_output_text`` (text-based regex parsing of decoded output) and
left reasoning extraction entirely to the route's ``ReasoningParser``.
That had three failure modes that all reduced to "regex requires
terminators the model didn't emit":

  1. Truncated analysis (no ``<|end|>``) → ``HarmonyReasoningParser``
     pattern missed the block, ``reasoning_content`` came back ``None``.
  2. Same truncation → ``_clean_gpt_oss_output`` "no final channel"
     branch only stripped marker tokens, leaving the analysis BODY
     in ``message.content`` (issue #442's titular leak).
  3. The streaming path *already* used ``OutputRouter`` correctly via
     ``_stream_with_output_router`` — non-streaming silently diverged.

The fix routes the completed token sequence through the same
``OutputRouter`` state machine the streaming path trusts. The router
tracks channel state at the token level (it doesn't care about ``<|end|>``
markers — it knows it's in analysis the moment the analysis word token
fires), so truncated output is handled identically to terminated.

The engine's ``_route_tokens_for_channels`` returns
``(reasoning_text, content_text)``; the content override only fires
when the router authoritatively says "no content, only reasoning" so
tool-call paths (where harmony commentary needs to survive intact for
the route's tool parser) keep their existing behavior. These tests pin
each branch of that override decision.
"""

from __future__ import annotations

import pytest

from vllm_mlx.engine.batched import BatchedEngine


class _FakeTokenizer:
    """Minimal stand-in matching the ``OutputRouter.from_tokenizer``
    interface — only ``get_vocab`` and ``decode`` are touched."""

    def __init__(self, vocab: dict[str, int]):
        self._vocab = vocab
        self._id_to_text = {v: k for k, v in vocab.items()}

    def get_vocab(self) -> dict[str, int]:
        return self._vocab

    def decode(self, ids):
        return "".join(self._id_to_text.get(i, f"<UNK:{i}>") for i in ids)


# Harmony token IDs from openai/gpt-oss-20b (same constants the real
# router reads). Keep in sync with tests/test_output_router.py.
_HARMONY_VOCAB = {
    "<|return|>": 200002,
    "<|constrain|>": 200003,
    "<|channel|>": 200005,
    "<|start|>": 200006,
    "<|end|>": 200007,
    "<|message|>": 200008,
    "<|call|>": 200012,
    "analysis": 35644,
    "final": 17196,
    "assistant": 173781,
    "Reason": 1,
    "ing": 2,
    "Answer": 3,
    "Plain": 4,
}


@pytest.fixture
def engine() -> BatchedEngine:
    """Engine with a harmony tokenizer wired in. We don't load a real
    model — ``_route_tokens_for_channels`` only touches the tokenizer
    and the router state machine.
    """
    eng = BatchedEngine("test-model")
    eng._loaded = True
    eng._is_mllm = False
    eng._tokenizer = _FakeTokenizer(_HARMONY_VOCAB)
    return eng


def test_truncated_analysis_drops_content_and_recovers_reasoning(engine):
    """The exact #442 production failure: model finishes mid-thinking
    (no ``<|end|>``), no final channel ever opened. Before the fix,
    ``clean_output_text``'s else-branch returned the analysis body as
    ``content`` and the parser couldn't recover ``reasoning_content``.
    Router-based extraction sees the analysis channel from the
    ``analysis`` word token regardless of whether ``<|end|>`` fires.
    """
    # <|channel|> analysis <|message|> Reason ing  (no <|end|>)
    token_ids = [200005, 35644, 200008, 1, 2]
    reasoning, content = engine._route_tokens_for_channels(
        token_ids, fallback_text="Reasoning"
    )
    assert reasoning == "Reasoning", (
        f"#442: router did not recover reasoning from truncated analysis: {reasoning!r}"
    )
    assert content == "", f"#442: analysis body leaked into content: {content!r}"


def test_terminated_analysis_only_also_drops_content(engine):
    """``<|channel|>analysis<|message|>...<|end|>`` with no final
    channel: same behavior as truncated — content must be empty,
    reasoning must be populated. Pins the symmetric case (the bug
    reproduces both with and without ``<|end|>``).
    """
    token_ids = [200005, 35644, 200008, 1, 2, 200007]
    reasoning, content = engine._route_tokens_for_channels(
        token_ids, fallback_text="anything"
    )
    assert reasoning == "Reasoning"
    assert content == ""


def test_analysis_then_final_keeps_fallback_content(engine):
    """Happy path: analysis followed by final channel. The router sees
    BOTH a CONTENT event (final-channel "Answer") and a REASONING event
    (analysis-channel "Reasoning"). Override condition (content is None
    AND reasoning exists) is FALSE → keep the fallback text from
    ``clean_output_text``. This pins the existing PR #436 / #440
    behavior so the new override doesn't regress the common case.
    """
    token_ids = [
        200005,
        35644,
        200008,  # <|channel|>analysis<|message|>
        1,
        2,  # Reasoning
        200007,  # <|end|>
        200006,
        173781,  # <|start|>assistant  (PR #441 swallows the role)
        200005,
        17196,
        200008,  # <|channel|>final<|message|>
        3,  # Answer
        200002,  # <|return|>
    ]
    reasoning, content = engine._route_tokens_for_channels(
        token_ids, fallback_text="Answer"
    )
    assert reasoning == "Reasoning"
    assert content == "Answer", "happy path must not clobber the text-cleaning result"


def test_pure_content_no_thinking_keeps_fallback(engine):
    """``<|channel|>final<|message|>Plain<|return|>`` — content-only,
    no analysis. Router emits CONTENT and no REASONING. Override
    condition is FALSE (reasoning empty) → fallback content preserved.
    Without this guard the engine would clobber pure-content responses
    with ``text=""`` on every non-reasoning model.
    """
    token_ids = [200005, 17196, 200008, 4, 200002]
    reasoning, content = engine._route_tokens_for_channels(
        token_ids, fallback_text="Plain"
    )
    assert reasoning == ""
    assert content == "Plain"


def test_no_router_returns_fallback_untouched(engine):
    """If ``_create_output_router`` returns ``None`` (tokenizer doesn't
    have channel tokens — e.g. plain Llama), the engine must NOT try
    to do anything channel-aware. Returns empty reasoning + the
    original ``fallback_text`` so routes' ReasoningParser path keeps
    handling extraction for older formats.
    """
    plain_engine = BatchedEngine("test-model")
    plain_engine._loaded = True
    plain_engine._is_mllm = False
    plain_engine._tokenizer = _FakeTokenizer({"plain": 100})
    reasoning, content = plain_engine._route_tokens_for_channels(
        [100, 100], fallback_text="plain text"
    )
    assert reasoning == ""
    assert content == "plain text"


def test_empty_token_list_returns_fallback(engine):
    """Defensive: empty token IDs (race during error paths) must not
    crash. Returns empty reasoning + the fallback text.
    """
    reasoning, content = engine._route_tokens_for_channels([], fallback_text="whatever")
    assert reasoning == ""
    assert content == "whatever"


def test_tool_call_routing_preserves_fallback_text(engine, monkeypatch):
    """PR #515 round-1: when the router emits a TOOL_CALL channel
    (commentary tool call) the override MUST NOT fire — fallback_text
    has to survive intact so the route's ``_parse_tool_calls_with_parser``
    can extract the call from the harmony wire format. Verified via a
    live diff against the pre-PR baseline; this test pins the
    invariant so a future override-condition change doesn't silently
    regress non-stream tool calls.
    """

    class _ToolCallRouter:
        def reset(self):
            pass

        def feed_sequence(self, _ids):
            return {
                "content": None,
                "reasoning": "Need to call the function",
                "tool_calls": [
                    "<|channel|>commentary to=functions.get_weather "
                    '<|constrain|>json<|message|>{"city":"NYC"}<|call|>'
                ],
            }

    monkeypatch.setattr(engine, "_create_output_router", lambda: _ToolCallRouter())
    reasoning, content = engine._route_tokens_for_channels(
        [200005, 12606, 815, 200008, 1],
        fallback_text='<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{"city":"NYC"}<|call|>',
    )
    assert reasoning == "Need to call the function"
    # The override must NOT clobber fallback_text — tool-call commentary
    # MUST survive to the route's HarmonyToolParser.
    assert "functions.get_weather" in content, (
        f"tool-call fallback_text clobbered; got {content!r}"
    )
    assert '{"city":"NYC"}' in content


def test_routed_tool_call_supplements_fallback_text(engine, monkeypatch):
    """PR #515 (codex round-3 BLOCKING origin / round-11 NIT retarget):
    when ``OutputRouter.feed_sequence`` surfaces a structured tool call
    in ``routed["tool_calls"]`` but ``_clean_gpt_oss_output`` has
    stripped the commentary wire text from ``fallback_text``, the
    engine must pipe each routed call through by appending its
    reconstructed wire text so the route's ``HarmonyToolParser`` can
    extract it. Idempotent: when ``fallback_text`` already carries
    the SAME canonical signature, don't double-append.

    (Round-6 flipped ``HarmonyStreamingRouter.finalize()`` to the
    vLLM / SGLang safer-default — finalize now always returns None;
    tool calls only surface via ``feed()`` on a fully closed
    ``<|call|>``-terminated commentary message. The fake-router stub
    below stands in for that ``feed_sequence`` path, not finalize.)
    """

    class _DrainedToolCallRouter:
        def reset(self):
            pass

        def feed_sequence(self, _ids):
            return {
                "content": None,
                "reasoning": "Calling the tool",
                "tool_calls": [
                    "<|channel|>commentary to=functions.get_weather "
                    '<|constrain|>json<|message|>{"city":"NYC"}<|call|>'
                ],
            }

    monkeypatch.setattr(
        engine, "_create_output_router", lambda: _DrainedToolCallRouter()
    )

    # Case A: fallback_text was already stripped — must be supplemented.
    reasoning, content = engine._route_tokens_for_channels(
        [200005, 12606, 815, 200008, 1],
        fallback_text="",  # stripped because clean_output_text didn't bail
    )
    assert reasoning == "Calling the tool"
    assert "functions.get_weather" in content, (
        f"finalize-drained tool call must be appended to empty fallback_text; "
        f"got {content!r}"
    )
    assert '{"city":"NYC"}' in content

    # PR #515 round-5 BLOCKING refinement: idempotency must be per
    # wire-text, not "any commentary marker present" — the multi-tool
    # case is covered in
    # ``test_multi_tool_fallback_with_existing_marker_still_appends_new``.

    # Case B: fallback_text already contains THIS exact tool call —
    # don't double-append.
    tc_wire = (
        "<|channel|>commentary to=functions.get_weather "
        '<|constrain|>json<|message|>{"city":"NYC"}<|call|>'
    )
    reasoning, content = engine._route_tokens_for_channels(
        [200005, 12606, 815, 200008, 1], fallback_text=tc_wire
    )
    assert reasoning == "Calling the tool"
    assert content == tc_wire, (
        f"fallback_text already carrying THIS tool call must not be doubled; "
        f"got {content!r}"
    )
    assert content.count("functions.get_weather") == 1


def test_multi_tool_fallback_with_existing_marker_still_appends_new(
    engine, monkeypatch
):
    """PR #515 round-5 BLOCKING: in a multi-tool turn (model emits
    tool call #1 cleanly with ``<|call|>``, then is truncated mid-tool
    call #2), ``fallback_text`` already carries call #1's commentary
    marker. The previous bulk guard ``"<|channel|>commentary" not in
    fallback_text`` then suppressed appending ALL reconstructed tool
    calls, dropping call #2. The fix checks each reconstructed wire
    text individually so call #1 isn't doubled and call #2 still
    reaches ``HarmonyToolParser``.
    """
    call1_text = (
        "<|channel|>commentary to=functions.get_weather "
        '<|constrain|>json<|message|>{"city":"NYC"}<|call|>'
    )
    call2_text = (
        "<|channel|>commentary to=functions.get_news "
        '<|constrain|>json<|message|>{"topic":"tech"}<|call|>'
    )

    class _MultiToolRouter:
        def reset(self):
            pass

        def feed_sequence(self, _ids):
            # Router surfaces BOTH tool calls.
            return {
                "content": None,
                "reasoning": None,
                "tool_calls": [call1_text, call2_text],
            }

    monkeypatch.setattr(engine, "_create_output_router", lambda: _MultiToolRouter())

    # fallback_text has call #1 already (from raw text bail-out) but
    # NOT call #2 (finalize() synthesized it because raw was truncated
    # before <|call|>).
    fallback_in = call1_text
    reasoning, content = engine._route_tokens_for_channels(
        [200005], fallback_text=fallback_in
    )

    # Call #1 must not be doubled.
    assert content.count("functions.get_weather") == 1, (
        f"call #1 must not be doubled; got {content!r}"
    )
    # Call #2 must be appended.
    assert "functions.get_news" in content, (
        f"finalize-synthesized call #2 must be appended; got {content!r}"
    )
    assert '{"topic":"tech"}' in content


def test_fallback_dedup_normalizes_whitespace_variance(engine, monkeypatch):
    """Codex round-7 NIT (PR #515): the previous verbatim substring
    check ``tc_text not in fallback_text`` doubled tool calls when the
    model's emit and the router's canonical reconstruction differ only
    by whitespace runs (e.g. one space vs two between ``to=...`` and
    ``<|constrain|>``). The dedup must compare by the structural
    ``(recipient, normalized_body)`` tuple instead so the same call
    matches across spacing variants.
    """
    # Model emitted the call with double-space between recipient and
    # constrain (a plausible variance — gpt-oss tokenizer roundtrips
    # may decode trailing whitespace before the special token). Router
    # reconstructs the canonical single-space form. Body bytes are
    # identical (both come from the same body-token decode).
    model_emit = (
        "<|channel|>commentary to=functions.get_weather  <|constrain|>json"
        '<|message|>{"city":"NYC"}<|call|>'
    )
    router_reconstructs = (
        "<|channel|>commentary to=functions.get_weather <|constrain|>json"
        '<|message|>{"city":"NYC"}<|call|>'
    )

    class _SpacingVariantRouter:
        def reset(self):
            pass

        def feed_sequence(self, _ids):
            return {
                "content": None,
                "reasoning": None,
                "tool_calls": [router_reconstructs],
            }

    monkeypatch.setattr(
        engine, "_create_output_router", lambda: _SpacingVariantRouter()
    )

    reasoning, content = engine._route_tokens_for_channels(
        [200005], fallback_text=model_emit
    )
    # Spacing-variant of the same call must NOT double-append.
    assert content.count("functions.get_weather") == 1, (
        f"spacing-variant duplicate of same call must not double-append; "
        f"got {content!r}"
    )
    # And a DIFFERENT call (different body, even if same recipient)
    # still gets appended.
    different_body_call = (
        "<|channel|>commentary to=functions.get_weather <|constrain|>json"
        '<|message|>{"city":"Paris"}<|call|>'
    )

    class _DifferentBodyRouter:
        def reset(self):
            pass

        def feed_sequence(self, _ids):
            return {
                "content": None,
                "reasoning": None,
                "tool_calls": [different_body_call],
            }

    monkeypatch.setattr(engine, "_create_output_router", lambda: _DifferentBodyRouter())
    reasoning, content = engine._route_tokens_for_channels(
        [200005], fallback_text=model_emit
    )
    assert content.count("functions.get_weather") == 2, (
        f"same recipient with different body must append; got {content!r}"
    )
    assert '{"city":"Paris"}' in content


def test_fallback_dedup_preserves_internal_body_whitespace(engine, monkeypatch):
    """Codex round-8 BLOCKING (PR #515): two legitimate calls
    ``{"q":"a b"}`` and ``{"q":"a  b"}`` carry semantically distinct
    arguments and must dedup to DIFFERENT signatures. The round-7
    body-whitespace-collapse path made them collide and could drop the
    later call. Signature now uses the exact body bytes — distinct
    internal whitespace ⇒ distinct signature ⇒ both calls survive.
    """
    call_a_one_space = (
        "<|channel|>commentary to=functions.search <|constrain|>json"
        '<|message|>{"q":"a b"}<|call|>'
    )
    call_a_two_space = (
        "<|channel|>commentary to=functions.search <|constrain|>json"
        '<|message|>{"q":"a  b"}<|call|>'
    )

    class _DistinctBodyRouter:
        def reset(self):
            pass

        def feed_sequence(self, _ids):
            return {
                "content": None,
                "reasoning": None,
                "tool_calls": [call_a_two_space],
            }

    monkeypatch.setattr(engine, "_create_output_router", lambda: _DistinctBodyRouter())

    # fallback_text already has the single-space-body variant from the
    # model's clean emit. The router separately reconstructed the
    # double-space-body call (different tool turn). Dedup must NOT
    # collapse them.
    reasoning, content = engine._route_tokens_for_channels(
        [200005], fallback_text=call_a_one_space
    )
    assert content.count("functions.search") == 2, (
        f"distinct internal-whitespace bodies must NOT dedup; got {content!r}"
    )
    assert '{"q":"a b"}' in content
    assert '{"q":"a  b"}' in content


def test_non_canonical_router_wire_text_is_dropped(engine, monkeypatch):
    """Codex round-8 BLOCKING (PR #515): when the router emits a wire
    text that doesn't match the canonical
    ``to=functions.<name>...<|message|>...<|call|>`` shape (e.g.
    truncated header, missing ``<|call|>`` terminator, or a malformed
    recipient that nonetheless survived router-side validation), the
    engine must NOT append it to fallback_text — feeding malformed wire
    text to HarmonyToolParser either fails to parse or anchors on the
    wrong markers and surfaces a corrupt tool call. Drop instead.
    """
    malformed = "<|channel|>commentary to=functions.broken<|message|>{}"
    # NOTE: no <|call|> terminator → signature regex won't match.

    class _MalformedRouter:
        def reset(self):
            pass

        def feed_sequence(self, _ids):
            return {
                "content": None,
                "reasoning": None,
                "tool_calls": [malformed],
            }

    monkeypatch.setattr(engine, "_create_output_router", lambda: _MalformedRouter())

    fallback = "raw fallback text without any tool call"
    reasoning, content = engine._route_tokens_for_channels(
        [200005], fallback_text=fallback
    )
    # Non-canonical wire text must not be appended — fallback stays.
    assert content == fallback, (
        f"non-canonical wire text must be dropped, not appended; got {content!r}"
    )


def test_identical_calls_twice_in_turn_both_survive_dedup(engine, monkeypatch):
    """Codex round-9 BLOCKING (PR #515): when the model legitimately
    emits the SAME tool call twice in one turn (user asked for the
    weather twice in one prompt), the model's raw fallback_text
    carries two wire-text instances AND the router structures both.
    The round-7/8 ``set`` dedup collapsed identical signatures to one
    and silently dropped the second; the fix uses ``Counter``
    multiplicity so each existing match consumes ONE routed call and
    the duplicates both survive.
    """
    same_call = (
        "<|channel|>commentary to=functions.get_weather <|constrain|>json"
        '<|message|>{"city":"NYC"}<|call|>'
    )
    fallback_two_calls = same_call + same_call

    class _TwoIdenticalRouter:
        def reset(self):
            pass

        def feed_sequence(self, _ids):
            return {
                "content": None,
                "reasoning": None,
                "tool_calls": [same_call, same_call],
            }

    monkeypatch.setattr(engine, "_create_output_router", lambda: _TwoIdenticalRouter())

    # Two in fallback, two in routed → two existing matches consume
    # both routed calls → nothing appended → final count stays at 2.
    reasoning, content = engine._route_tokens_for_channels(
        [200005], fallback_text=fallback_two_calls
    )
    assert content.count("functions.get_weather") == 2, (
        f"identical-twice (model+router both saw both) must dedup to 2; got {content!r}"
    )

    # And if the model emitted ONE but the router structured TWO
    # (truncation rescued one), the second must be appended.
    fallback_one_call = same_call

    class _OneInFallbackTwoRouted:
        def reset(self):
            pass

        def feed_sequence(self, _ids):
            return {
                "content": None,
                "reasoning": None,
                "tool_calls": [same_call, same_call],
            }

    monkeypatch.setattr(
        engine, "_create_output_router", lambda: _OneInFallbackTwoRouted()
    )
    reasoning, content = engine._route_tokens_for_channels(
        [200005], fallback_text=fallback_one_call
    )
    assert content.count("functions.get_weather") == 2, (
        f"one-in-fallback + two-routed must merge to 2; got {content!r}"
    )


def test_signature_regex_requires_commentary_frame(engine, monkeypatch):
    """Codex round-9 BLOCKING (PR #515): the previous signature regex
    matched a BARE ``to=functions.X<|message|>...<|call|>`` shape
    without requiring the ``<|channel|>commentary`` prefix. A
    fallback_text where ``clean_output_text`` stripped the commentary
    header but left the tail (or ordinary content happening to contain
    those substrings) then false-matched as an existing tool call,
    suppressing the router's real reconstructed wire text. The fix
    anchors the regex on the full commentary frame.
    """
    tc_wire = (
        "<|channel|>commentary to=functions.get_weather <|constrain|>json"
        '<|message|>{"city":"NYC"}<|call|>'
    )
    # Stripped fallback_text — the commentary header is gone, only the
    # tail survives. This must NOT register as an existing signature.
    stripped_fallback = (
        'to=functions.get_weather <|constrain|>json<|message|>{"city":"NYC"}<|call|>'
    )

    class _OneCallRouter:
        def reset(self):
            pass

        def feed_sequence(self, _ids):
            return {
                "content": None,
                "reasoning": None,
                "tool_calls": [tc_wire],
            }

    monkeypatch.setattr(engine, "_create_output_router", lambda: _OneCallRouter())
    reasoning, content = engine._route_tokens_for_channels(
        [200005], fallback_text=stripped_fallback
    )
    # Router's canonical wire text must be appended — the stripped
    # fallback's bare tail does NOT count as an existing signature.
    assert content.count("<|channel|>commentary") == 1, (
        f"stripped fallback must not false-suppress; got {content!r}"
    )
    assert "<|channel|>commentary to=functions.get_weather" in content


def test_router_exception_falls_back_cleanly(engine, monkeypatch):
    """If the router blows up mid-sequence (e.g. token id outside the
    vocab causes a decode failure), the engine must not propagate the
    exception — fall back to text-based cleaning. The streaming router
    handles this the same way; symmetry matters.
    """

    def _exploding_router():
        class _BoomRouter:
            def reset(self):
                pass

            def feed_sequence(self, _ids):
                raise RuntimeError("explosion in router")

        return _BoomRouter()

    monkeypatch.setattr(engine, "_create_output_router", _exploding_router)
    reasoning, content = engine._route_tokens_for_channels(
        [1, 2, 3], fallback_text="fallback"
    )
    assert reasoning == ""
    assert content == "fallback"
