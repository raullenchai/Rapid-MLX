"""Shared parser contracts: source ownership, bounded work and lifecycle."""

import itertools
import random

import pytest

from rapid_mlx.parsing.output_demux import (
    ChannelGrammar,
    ContentDelta,
    Control,
    EventReducer,
    Finish,
    Lexeme,
    OutputDemultiplexer,
    ParseContext,
    ParseFailure,
    ReasoningDelta,
    Span,
    Terminal,
    ToolArgs,
    ToolEnd,
    ToolStart,
)
from rapid_mlx.prompt_parse_context import PromptParseContext


def context(channel="content", **kwargs):
    return ParseContext(
        prompt=PromptParseContext("", (), channel, None, "chatml"), **kwargs
    )


def run(raw, chunks, ctx=None):
    parser = OutputDemultiplexer(ctx or context())
    events = []
    for chunk in chunks:
        events.extend(parser.feed(chunk))
        assert len(parser.pending) < parser.max_marker_chars
    events.extend(parser.finish())
    assert parser.finish() == parser.cancel() == []
    cursor = 0
    for event in events:
        assert event.span.start == cursor
        cursor = event.span.end
    assert cursor == len(raw) == parser.commit_cursor
    reduced = EventReducer()
    reduced.feed(events)
    return "".join(reduced.content), "".join(reduced.reasoning), parser


@pytest.mark.parametrize(
    "raw,visible,reasoning",
    [
        ("Hello world", "Hello world", ""),
        ("<think>secret</think>answer", "answer", "secret"),
        ("<think>A<think>B</think>C</think>D", "CD", "AB"),
        ("<think>x</think></think>y", "y", "x"),
        ("a<thorn>b</th", "a<thorn>b</th", ""),
        ("a<<think>秘密</think>🦊<", "a<🦊<", "秘密"),
        ("<think>x</think>answer<think>y", "answer", "xy"),
    ],
)
def test_partition_invariance_and_source_ownership(raw, visible, reasoning):
    partitions = [[raw], list(raw)]
    partitions += [[raw[:i], raw[i:]] for i in range(len(raw) + 1)]
    rng = random.Random(73)
    for _ in range(20):
        points = sorted(
            {0, len(raw), *(rng.randrange(len(raw) + 1) for _ in range(10))}
        )
        partitions.append([raw[a:b] for a, b in itertools.pairwise(points)])
    for chunks in partitions:
        actual, private, parser = run(raw, chunks)
        assert (actual, private) == (visible, reasoning)
        assert parser.input_bytes == len(raw.encode())
        assert parser.examined_chars <= 64 * len(raw)


def test_prompt_owned_initial_state_and_unknown_contract():
    raw = "private</think>public"
    assert run(raw, list(raw), context("reasoning"))[:2] == ("public", "private")
    unknown = ParseContext(default_channel="reasoning")
    assert run(raw, [raw], unknown)[:2] == ("public", "private")
    assert ParseContext().initial_channel == "content"


@pytest.mark.parametrize("reason", ["stop", "length", "cancelled"])
@pytest.mark.parametrize("channel", ["content", "reasoning"])
def test_terminal_partial_marker_returns_to_original_channel_once(reason, channel):
    parser = OutputDemultiplexer(context(channel))
    assert parser.feed("</thi") == []
    events = parser.cancel() if reason == "cancelled" else parser.finish(reason)
    cls = ContentDelta if channel == "content" else ReasoningDelta
    assert events == [cls(Span(0, 5), "</thi"), Finish(Span(5, 5), reason)]
    assert parser.finish() == []
    with pytest.raises(ValueError, match="terminal"):
        parser.feed("later")


def test_atomic_tokens_override_spelling_and_flush_earlier_prefix():
    parser = OutputDemultiplexer(
        context(
            terminals=(
                Terminal("reasoning_start", "<think>", (17,)),
                Terminal("reasoning_end", "</think>", (18,)),
            )
        )
    )
    assert parser.feed("<thi") == []
    assert parser.feed("atomic", token_id=17) == [
        ContentDelta(Span(0, 4), "<thi"),
        Control(Span(4, 10), "reasoning_start"),
    ]
    assert parser.feed("secret", token_id=19) == [
        ReasoningDelta(Span(10, 16), "secret")
    ]
    assert parser.feed("", token_id=18) == [Control(Span(16, 16), "reasoning_end")]
    assert parser.feed("answer") == [ContentDelta(Span(16, 22), "answer")]
    assert parser.finish() == [Finish(Span(22, 22), "stop")]


@pytest.mark.parametrize(
    "terminals",
    [
        (Terminal("x", ""),),
        (Terminal("x", "a"), Terminal("y", "a")),
        (Terminal("x", "a"), Terminal("y", "ab")),
        (Terminal("x", "a", (1,)), Terminal("y", "b", (1,))),
    ],
)
def test_invalid_lexer_configuration(terminals):
    with pytest.raises(ValueError):
        OutputDemultiplexer(context(terminals=terminals))


def test_empty_grammar_and_unknown_control():
    parser = OutputDemultiplexer(context(terminals=()))
    assert parser.feed("<think>") == [ContentDelta(Span(0, 7), "<think>")]
    assert parser.max_marker_chars == 0
    grammar = ChannelGrammar(context())
    assert grammar.consume(Lexeme(Span(0, 3), "eos", "turn_end")) == [
        Control(Span(0, 3), "turn_end")
    ]


def test_native_event_bypass_preserves_literal_controls_and_tools():
    parser = OutputDemultiplexer(context())
    native = [
        ContentDelta(Span(0, 7), "<think>"),
        ToolStart(Span(7, 8), 0, "read"),
        ToolArgs(Span(8, 10), 0, "{}"),
        ToolEnd(Span(10, 11), 0),
    ]
    assert parser.accept_native(native, end_offset=11) == native
    with pytest.raises(ValueError, match="reclassified"):
        parser.feed("anything")
    reducer = EventReducer()
    reducer.feed(native + parser.finish())
    assert reducer.content == ["<think>"]
    assert reducer.calls == [{"name": "read", "arguments": "{}"}]
    assert reducer.finish_reason == "stop"
    with pytest.raises(ValueError, match="lifecycle"):
        parser.accept_native([], end_offset=11)


def test_native_invalid_lifecycle_and_source_spans():
    parser = OutputDemultiplexer(context())
    with pytest.raises(ValueError, match="owned by finish"):
        parser.accept_native([Finish(Span(0, 0), "stop")], end_offset=0)
    parser.accept_native([ContentDelta(Span(0, 1), "a")], end_offset=1)
    with pytest.raises(ValueError, match="lifecycle"):
        parser.accept_native([], end_offset=0)
    with pytest.raises(ValueError, match="nonoverlapping"):
        parser.accept_native([ContentDelta(Span(0, 1), "a")], end_offset=1)
    text_parser = OutputDemultiplexer(context())
    text_parser.feed("x")
    with pytest.raises(ValueError, match="lifecycle"):
        text_parser.accept_native([], end_offset=1)


def test_adapter_cannot_lose_or_replay_source_at_finish():
    class LostGrammar:
        def consume(self, lexeme):
            return []

        def finish(self, offset, reason):
            return []

    parser = OutputDemultiplexer(context(), LostGrammar())
    parser.feed("a")
    with pytest.raises(ValueError, match="unowned"):
        parser.finish()

    class BadGrammar(LostGrammar):
        def consume(self, lexeme):
            return [ContentDelta(Span(0, 2), "aa")]

    parser = OutputDemultiplexer(context(), BadGrammar())
    with pytest.raises(ValueError, match="nonoverlapping"):
        parser.feed("a")


def test_candidate_adapter_can_hold_then_resolve_once():
    class CandidateGrammar:
        def __init__(self):
            self.parts = []

        def consume(self, lexeme):
            self.parts.append(lexeme.text)
            return []

        def finish(self, offset, reason):
            # A rejected uncommitted candidate is literal in its original lane.
            return [ReasoningDelta(Span(0, offset), "".join(self.parts))]

    parser = OutputDemultiplexer(context(), CandidateGrammar())
    assert parser.feed("candidate") == []
    assert parser.finish() == [
        ReasoningDelta(Span(0, 9), "candidate"),
        Finish(Span(9, 9), "stop"),
    ]
    assert parser.finish() == []


def test_reducer_failure_incomplete_and_duplicate_call_contracts():
    reducer = EventReducer()
    span = Span(0, 0)
    reducer.feed(
        [
            ToolStart(span, 0, "read"),
            ToolArgs(span, 0, '{"path":'),
            ParseFailure(span, "invalid_arguments", 0),
            ToolStart(span, 1, "pending"),
            ParseFailure(span, "invalid_candidate"),
        ]
    )
    assert reducer.calls == []
    assert len(reducer.failures) == 2
    with pytest.raises(ValueError, match="terminal call"):
        reducer.feed([ToolEnd(span, 0)])
    with pytest.raises(ValueError, match="duplicate"):
        reducer.feed([ToolStart(span, 1, "pending")])
    reducer.feed(
        [ToolStart(span, 2, "okay"), ToolArgs(span, 2, "{}"), ToolEnd(span, 2)]
    )
    with pytest.raises(ValueError, match="terminal call"):
        reducer.feed([ToolArgs(span, 2, "more")])
    for index in (0, 2):
        with pytest.raises(ValueError, match="terminal call"):
            reducer.feed([ParseFailure(span, "late_failure", index)])
    reducer.feed([Finish(span, "length")])
    assert reducer.calls == [{"name": "okay", "arguments": "{}"}]
    with pytest.raises(ValueError, match="after finish"):
        reducer.feed([ContentDelta(span, "late")])


def test_marker_work_is_linear_at_release_benchmark_sizes():
    ratios = []
    for size in (1024, 4096, 16384, 65536):
        raw = ("<thiX" * (size // 5 + 1))[:size]
        visible, private, parser = run(raw, [raw[i : i + 5] for i in range(0, size, 5)])
        assert visible == raw and private == ""
        ratios.append(parser.examined_chars / size)
    assert max(ratios) / min(ratios) < 1.01


def test_plain_content_is_emitted_in_each_chunk_without_eof_wait():
    parser = OutputDemultiplexer(context())
    for chunk in ["Hello ", "world. ", "More text."]:
        assert parser.feed(chunk)
        assert not parser.pending
    parser.finish()


def test_rejected_native_batch_does_not_partially_advance_cursor():
    parser = OutputDemultiplexer(context())
    with pytest.raises(ValueError, match="nonoverlapping"):
        parser.accept_native(
            [ContentDelta(Span(0, 1), "a"), ContentDelta(Span(0, 1), "b")], end_offset=2
        )
    assert parser.offset == parser.commit_cursor == 0
    assert parser.feed("okay") == [ContentDelta(Span(0, 4), "okay")]


def test_grammar_cannot_publish_an_independent_finish():
    class PrematureGrammar:
        def consume(self, lexeme):
            return [Finish(lexeme.span, "stop")]

        def finish(self, offset, reason):
            return []

    parser = OutputDemultiplexer(context(), PrematureGrammar())
    with pytest.raises(ValueError, match="owned by finish"):
        parser.feed("x")


@pytest.mark.parametrize("events", [[], [ContentDelta(Span(0, 1), "a")]])
def test_native_batch_rejects_unowned_tail_before_finalization(events):
    parser = OutputDemultiplexer(context())
    with pytest.raises(ValueError, match="complete source span"):
        parser.accept_native(events, end_offset=2)
    assert parser.offset == parser.commit_cursor == 0
    # Rejection did not select native mode or strand finalization.
    assert parser.feed("okay") == [ContentDelta(Span(0, 4), "okay")]
    assert parser.finish() == [Finish(Span(4, 4), "stop")]


@pytest.mark.parametrize("failure", ["consume", "finish"])
def test_adapter_exception_is_terminal_and_cannot_replay(failure):
    class FailingGrammar(ChannelGrammar):
        def consume(self, lexeme):
            if failure == "consume" and lexeme.terminal:
                raise RuntimeError("adapter failed after earlier text")
            return super().consume(lexeme)

        def finish(self, offset, reason):
            if failure == "finish":
                raise RuntimeError("adapter terminal failure")
            return []

    parser = OutputDemultiplexer(context(), FailingGrammar(context()))
    with pytest.raises(RuntimeError):
        parser.feed("already owned<think>")
        parser.finish()
    for operation in (lambda: parser.feed("again"), parser.finish, parser.cancel):
        with pytest.raises(ValueError, match="parser failed"):
            operation()
    with pytest.raises(ValueError, match="lifecycle"):
        parser.accept_native([], end_offset=parser.offset)


def test_release_pending_resolves_held_prefix_once():
    ctx = context(terminals=(Terminal("reasoning_start", "<think>"),))
    parser = OutputDemultiplexer(ctx)
    assert parser.feed("Hi <thi") == [ContentDelta(Span(0, 3), "Hi ")]
    assert parser.release_pending() == [ContentDelta(Span(3, 7), "<thi")]
    assert parser.release_pending() == []
    parser.finish()
    assert parser.release_pending() == []


def test_release_pending_failure_is_terminal():
    class FailingGrammar(ChannelGrammar):
        def consume(self, lexeme):
            raise RuntimeError("adapter failed on released text")

    ctx = context(terminals=(Terminal("reasoning_start", "<think>"),))
    parser = OutputDemultiplexer(ctx, FailingGrammar(ctx))
    parser.feed("<thi")
    with pytest.raises(RuntimeError):
        parser.release_pending()
    assert parser.release_pending() == []
    with pytest.raises(ValueError, match="parser failed"):
        parser.finish()


def test_empty_feed_does_not_select_a_routing_mode():
    parser = OutputDemultiplexer(context())
    assert parser.feed("") == []
    assert parser.accept_native([ContentDelta(Span(0, 1), "a")], end_offset=1)
    assert parser.feed("") == []
    parser.finish()
    other = OutputDemultiplexer(context())
    assert other.accept_native([], end_offset=0) == []
    assert other.feed("a") == [ContentDelta(Span(0, 1), "a")]


@pytest.mark.parametrize("cls", [ContentDelta, ReasoningDelta])
def test_native_text_length_must_match_claimed_span(cls):
    parser = OutputDemultiplexer(context())
    with pytest.raises(ValueError, match="length must match"):
        parser.accept_native([cls(Span(0, 1), "long")], end_offset=1)
    assert parser.offset == parser.commit_cursor == 0
    assert parser.feed("a") == [ContentDelta(Span(0, 1), "a")]


def test_native_rejection_is_atomic_for_non_value_validation_errors():
    parser = OutputDemultiplexer(context())
    with pytest.raises(TypeError):
        parser.accept_native([ContentDelta(Span(0, 1), None)], end_offset=1)
    assert parser.offset == parser.commit_cursor == 0
    assert parser.feed("a") == [ContentDelta(Span(0, 1), "a")]
    assert parser.finish() == [Finish(Span(1, 1), "stop")]


def test_indexed_failure_before_start_cannot_become_an_executable_call():
    reducer = EventReducer()
    span = Span(0, 0)
    reducer.feed([ParseFailure(span, "invalid_header", 0)])
    with pytest.raises(ValueError, match="failed tool index"):
        reducer.feed([ToolStart(span, 0, "failed")])
    with pytest.raises(ValueError, match="terminal call"):
        reducer.feed([ParseFailure(span, "another_failure", 0)])
    assert reducer.calls == []
    reducer.feed(
        [
            ToolStart(span, 1, "valid"),
            ToolArgs(span, 1, "{}"),
            ToolEnd(span, 1),
            Finish(span, "stop"),
        ]
    )
    assert reducer.calls == [{"name": "valid", "arguments": "{}"}]
    assert len(reducer.failures) == 1
