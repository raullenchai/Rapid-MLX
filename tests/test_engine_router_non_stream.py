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


def test_finalize_drained_tool_call_supplements_fallback_text(engine, monkeypatch):
    """PR #515 round-3 BLOCKING: when ``HarmonyStreamingRouter.finalize()``
    synthesizes a tool-call event from a truncated commentary (no
    ``<|call|>`` in the raw text), ``_clean_gpt_oss_output``'s
    bail-out may not match — the strip regex then removes the
    commentary header from ``fallback_text``, and the route's
    ``HarmonyToolParser`` sees no tool call at all. The engine must
    pipe ``routed["tool_calls"]`` through by appending the
    reconstructed wire text so the downstream parser can extract it.
    Idempotent: when the raw text already carries ``<|channel|>commentary``,
    don't append (avoids double-parsing).
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

    # Case B: fallback_text already has commentary markers — don't append
    # (idempotent; avoids the downstream parser seeing the same call twice).
    pre_existing = (
        "<|channel|>commentary to=functions.add <|constrain|>json"
        '<|message|>{"a":1,"b":2}<|call|>'
    )
    reasoning, content = engine._route_tokens_for_channels(
        [200005, 12606, 815, 200008, 1], fallback_text=pre_existing
    )
    assert reasoning == "Calling the tool"
    assert content == pre_existing, (
        f"fallback_text with existing commentary must not be double-appended; "
        f"got {content!r}"
    )


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
