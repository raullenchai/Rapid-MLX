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
``OutputRouter`` state machine the streaming path trusts.

Round-15 refactor (PR #515 codex round-12 / round-14 BLOCKING
closure): ``_route_tokens_for_channels`` returns
``(reasoning_text, content_text, structured_tool_calls)``. The third
value is the engine's structured tool-call passthrough — when the
router natively parses tool calls (currently
``HarmonyStreamingRouter`` via openai-harmony's ``StreamableParser``),
the bytes-faithful ``[{"name", "arguments"}]`` payload flows through
``GenerationOutput.tool_calls`` so the route layer bypasses text-based
regex extraction entirely. This eliminates the wire-text round-trip
that previously corrupted tool calls whose JSON arguments contained
literal harmony sentinel substrings.
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


# Harmony token IDs from openai/gpt-oss-20b-mxfp4-q8 (same constants the real
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
    reasoning, content, tool_calls = engine._route_tokens_for_channels(
        token_ids, fallback_text="Reasoning"
    )
    assert reasoning == "Reasoning", (
        f"#442: router did not recover reasoning from truncated analysis: {reasoning!r}"
    )
    assert content == "", f"#442: analysis body leaked into content: {content!r}"
    assert tool_calls is None


def test_terminated_analysis_only_also_drops_content(engine):
    """``<|channel|>analysis<|message|>...<|end|>`` with no final
    channel: same behavior as truncated — content must be empty,
    reasoning must be populated. Pins the symmetric case (the bug
    reproduces both with and without ``<|end|>``).
    """
    token_ids = [200005, 35644, 200008, 1, 2, 200007]
    reasoning, content, tool_calls = engine._route_tokens_for_channels(
        token_ids, fallback_text="anything"
    )
    assert reasoning == "Reasoning"
    assert content == ""
    assert tool_calls is None


def test_analysis_then_final_keeps_fallback_content(engine):
    """Happy path: analysis followed by final channel. The router sees
    BOTH a CONTENT event (final-channel "Answer") and a REASONING event
    (analysis-channel "Reasoning"). Override condition (content is None
    AND reasoning exists) is FALSE → keep the fallback text from
    ``clean_output_text``.
    """
    token_ids = [
        200005,
        35644,
        200008,  # <|channel|>analysis<|message|>
        1,
        2,  # Reasoning
        200007,  # <|end|>
        200006,
        173781,  # <|start|>assistant
        200005,
        17196,
        200008,  # <|channel|>final<|message|>
        3,  # Answer
        200002,  # <|return|>
    ]
    reasoning, content, tool_calls = engine._route_tokens_for_channels(
        token_ids, fallback_text="Answer"
    )
    assert reasoning == "Reasoning"
    assert content == "Answer", "happy path must not clobber the text-cleaning result"
    assert tool_calls is None


def test_pure_content_no_thinking_keeps_fallback(engine):
    """``<|channel|>final<|message|>Plain<|return|>`` — content-only,
    no analysis. Router emits CONTENT and no REASONING. Override
    condition is FALSE (reasoning empty) → fallback content preserved.
    """
    token_ids = [200005, 17196, 200008, 4, 200002]
    reasoning, content, tool_calls = engine._route_tokens_for_channels(
        token_ids, fallback_text="Plain"
    )
    assert reasoning == ""
    assert content == "Plain"
    assert tool_calls is None


def test_no_router_returns_fallback_untouched(engine):
    """If ``_create_output_router`` returns ``None`` (tokenizer doesn't
    have channel tokens — e.g. plain Llama), the engine must NOT try
    to do anything channel-aware.
    """
    plain_engine = BatchedEngine("test-model")
    plain_engine._loaded = True
    plain_engine._is_mllm = False
    plain_engine._tokenizer = _FakeTokenizer({"plain": 100})
    reasoning, content, tool_calls = plain_engine._route_tokens_for_channels(
        [100, 100], fallback_text="plain text"
    )
    assert reasoning == ""
    assert content == "plain text"
    assert tool_calls is None


def test_empty_token_list_returns_fallback(engine):
    """Defensive: empty token IDs (race during error paths) must not
    crash.
    """
    reasoning, content, tool_calls = engine._route_tokens_for_channels(
        [], fallback_text="whatever"
    )
    assert reasoning == ""
    assert content == "whatever"
    assert tool_calls is None


def test_structured_tool_call_passthrough_drops_fallback_text(engine, monkeypatch):
    """Round-15 refactor: when the router natively surfaces structured
    tool calls (HarmonyStreamingRouter), the engine MUST surface them
    via ``structured_tool_calls`` and force ``content`` to the
    router's CONTENT-channel result. The ``fallback_text`` from
    ``clean_output_text`` may still carry un-cleaned commentary
    headers that would otherwise bleed into the user-facing
    ``content`` field. The route layer then bypasses text-based
    extraction entirely.
    """

    class _StructuredToolCallRouter:
        def reset(self):
            pass

        def feed_sequence(self, _ids):
            return {
                "content": None,
                "reasoning": "Need to call the function",
                "tool_calls": [
                    {"name": "get_weather", "arguments": '{"city":"NYC"}'},
                ],
            }

    monkeypatch.setattr(
        engine, "_create_output_router", lambda: _StructuredToolCallRouter()
    )
    fallback = (
        "<|channel|>commentary to=functions.get_weather <|constrain|>json"
        '<|message|>{"city":"NYC"}<|call|>'
    )
    reasoning, content, tool_calls = engine._route_tokens_for_channels(
        [200005, 12606, 815, 200008, 1], fallback_text=fallback
    )
    assert reasoning == "Need to call the function"
    # Structured passthrough — fallback_text's commentary residue MUST
    # NOT leak into content. The route layer reads tool_calls instead.
    assert content == "", (
        f"structured passthrough must clear content of commentary residue; "
        f"got {content!r}"
    )
    assert tool_calls == [{"name": "get_weather", "arguments": '{"city":"NYC"}'}]


def test_structured_tool_call_passthrough_preserves_final_content(engine, monkeypatch):
    """When the model emits BOTH a tool call AND a final-channel
    response (compound assistant turn), structured passthrough must
    preserve the final-channel CONTENT alongside the tool_calls. The
    router's CONTENT result is the user-facing text; the tool_calls
    are surfaced separately for the route layer.
    """

    class _CompoundRouter:
        def reset(self):
            pass

        def feed_sequence(self, _ids):
            return {
                "content": "The answer is 42.",
                "reasoning": None,
                "tool_calls": [
                    {"name": "lookup", "arguments": '{"q":"meaning"}'},
                ],
            }

    monkeypatch.setattr(engine, "_create_output_router", lambda: _CompoundRouter())
    reasoning, content, tool_calls = engine._route_tokens_for_channels(
        [200005], fallback_text="any fallback"
    )
    assert reasoning == ""
    assert content == "The answer is 42.", (
        f"final-channel content must survive alongside structured tool calls; "
        f"got {content!r}"
    )
    assert tool_calls == [{"name": "lookup", "arguments": '{"q":"meaning"}'}]


def test_structured_tool_call_passthrough_preserves_sentinel_bearing_arguments(
    engine, monkeypatch
):
    """Round-15 closure for codex round-12 / round-14 BLOCKING: a tool
    call whose JSON arguments contain a literal harmony sentinel
    substring (``{"text":"<|call|>"}``) MUST flow through bytes-
    faithfully via the structured payload. The previous wire-text
    round-trip lost these calls — the downstream regex parser
    anchored on the embedded sentinel and truncated the JSON.
    """
    sentinel_bearing_args = '{"text":"use <|call|>"}'

    class _SentinelBodyRouter:
        def reset(self):
            pass

        def feed_sequence(self, _ids):
            return {
                "content": None,
                "reasoning": None,
                "tool_calls": [
                    {"name": "echo", "arguments": sentinel_bearing_args},
                ],
            }

    monkeypatch.setattr(engine, "_create_output_router", lambda: _SentinelBodyRouter())
    reasoning, content, tool_calls = engine._route_tokens_for_channels(
        [200005], fallback_text=""
    )
    assert tool_calls is not None and len(tool_calls) == 1
    # Bytes-faithful: the sentinel substring inside the JSON body is
    # preserved exactly. Pre-refactor this was lost because the wire-
    # text reconstruction abstained (rounds 7/9) or because regex
    # extraction anchored on the embedded sentinel (rounds 12/14).
    assert tool_calls[0] == {"name": "echo", "arguments": sentinel_bearing_args}


def test_multiple_structured_tool_calls_pass_through_in_order(engine, monkeypatch):
    """A multi-tool turn must surface all tool calls in emission
    order. Distinct calls (same recipient with different args, or
    different recipients) MUST NOT be deduped — multiplicity is part
    of the model's intent.
    """
    calls = [
        {"name": "get_weather", "arguments": '{"city":"NYC"}'},
        {"name": "get_news", "arguments": '{"topic":"tech"}'},
        {"name": "get_weather", "arguments": '{"city":"Paris"}'},
    ]

    class _MultiRouter:
        def reset(self):
            pass

        def feed_sequence(self, _ids):
            return {"content": None, "reasoning": None, "tool_calls": list(calls)}

    monkeypatch.setattr(engine, "_create_output_router", lambda: _MultiRouter())
    reasoning, content, tool_calls = engine._route_tokens_for_channels(
        [200005], fallback_text=""
    )
    assert tool_calls == calls, "multi-tool turn must preserve order and multiplicity"


def test_identical_structured_calls_twice_both_survive(engine, monkeypatch):
    """When the model legitimately emits the SAME tool call twice in
    one turn, BOTH must surface — multiplicity carries semantic intent
    (user asked the same question twice, model called the tool twice).
    """
    same_call = {"name": "get_weather", "arguments": '{"city":"NYC"}'}

    class _TwoIdenticalRouter:
        def reset(self):
            pass

        def feed_sequence(self, _ids):
            return {
                "content": None,
                "reasoning": None,
                "tool_calls": [dict(same_call), dict(same_call)],
            }

    monkeypatch.setattr(engine, "_create_output_router", lambda: _TwoIdenticalRouter())
    reasoning, content, tool_calls = engine._route_tokens_for_channels(
        [200005], fallback_text=""
    )
    assert tool_calls == [same_call, same_call], (
        f"identical-twice calls must both survive; got {tool_calls!r}"
    )


def test_router_exception_falls_back_cleanly(engine, monkeypatch):
    """If the router blows up mid-sequence (e.g. token id outside the
    vocab causes a decode failure), the engine must not propagate the
    exception — fall back to text-based cleaning.
    """

    def _exploding_router():
        class _BoomRouter:
            def reset(self):
                pass

            def feed_sequence(self, _ids):
                raise RuntimeError("explosion in router")

        return _BoomRouter()

    monkeypatch.setattr(engine, "_create_output_router", _exploding_router)
    reasoning, content, tool_calls = engine._route_tokens_for_channels(
        [1, 2, 3], fallback_text="fallback"
    )
    assert reasoning == ""
    assert content == "fallback"
    assert tool_calls is None


def test_legacy_router_emitting_wire_text_strings_routes_through_fallback(
    engine, monkeypatch
):
    """Backwards compatibility: the legacy ``OutputRouter`` (gemma4 /
    think-tag) emits TOOL_CALL events whose ``text`` is wire-format
    string (single tool_call sentinel block decoded into one string).
    The engine MUST treat those as non-structured — leave
    ``fallback_text`` intact for the legacy text-based parser path.
    Only HarmonyStreamingRouter currently emits dicts.
    """

    class _LegacyRouter:
        def reset(self):
            pass

        def feed_sequence(self, _ids):
            return {
                "content": "Some content",
                "reasoning": None,
                "tool_calls": ["<some_legacy_wire_text>"],
            }

    monkeypatch.setattr(engine, "_create_output_router", lambda: _LegacyRouter())
    reasoning, content, tool_calls = engine._route_tokens_for_channels(
        [200005], fallback_text="raw fallback"
    )
    assert tool_calls is None, (
        f"legacy wire-text tool calls must not surface as structured; "
        f"got {tool_calls!r}"
    )
    # And fallback_text is preserved since the structured path didn't fire.
    assert content == "raw fallback"
