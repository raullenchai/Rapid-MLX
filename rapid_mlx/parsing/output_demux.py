# SPDX-License-Identifier: Apache-2.0
"""Incremental output owner, initially used only for shadow comparisons.

Offsets are half-open Unicode code-point offsets in the decoded generation, not
prompt offsets or tokenizer indices. Every consumed source span has one owner,
including controls. Zero-width lifecycle events do not own source bytes.
Atomic token terminals take precedence over their decoded spelling. Callers that
cannot align token IDs to decoded spans must use the incremental text interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

from ..prompt_parse_context import PromptParseContext

Channel = Literal["content", "reasoning"]


@dataclass(frozen=True)
class Span:
    start: int
    end: int


@dataclass(frozen=True)
class ContentDelta:
    span: Span
    text: str


@dataclass(frozen=True)
class ReasoningDelta:
    span: Span
    text: str


@dataclass(frozen=True)
class Control:
    span: Span
    name: str


@dataclass(frozen=True)
class ToolStart:
    span: Span
    index: int
    name: str


@dataclass(frozen=True)
class ToolArgs:
    span: Span
    index: int
    text: str


@dataclass(frozen=True)
class ToolEnd:
    span: Span
    index: int


@dataclass(frozen=True)
class ParseFailure:
    span: Span
    code: str
    index: int | None = None


@dataclass(frozen=True)
class Finish:
    span: Span
    reason: str


Event = (
    ContentDelta
    | ReasoningDelta
    | Control
    | ToolStart
    | ToolArgs
    | ToolEnd
    | ParseFailure
    | Finish
)


@dataclass(frozen=True)
class Terminal:
    name: str
    text: str
    token_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class ParseContext:
    prompt: PromptParseContext | None = None
    # Unknown prompts require an explicit grammar default, never a text guess.
    default_channel: Channel = "content"
    terminals: tuple[Terminal, ...] = (
        Terminal("reasoning_start", "<think>"),
        Terminal("reasoning_end", "</think>"),
    )
    authorized_tools: frozenset[str] = frozenset()
    max_header_chars: int = 1024
    max_argument_chars: int = 1048576

    @property
    def initial_channel(self) -> Channel:
        return (
            self.prompt.initial_channel if self.prompt else None
        ) or self.default_channel


@dataclass(frozen=True)
class Lexeme:
    span: Span
    text: str
    terminal: str | None = None


class Grammar(Protocol):
    """Format adapters consume lexemes once; candidates may retain owned spans.

    Candidate storage is separate from the lexer's bounded marker holdback.
    Adapters must bound uncommitted headers, validate tool names against the
    original declarations, and fail committed calls without replaying them.
    """

    def consume(self, lexeme: Lexeme) -> list[Event]: ...

    def finish(self, offset: int, reason: str) -> list[Event]: ...


class ChannelGrammar:
    """Explicit reasoning controls; format-specific tool adapters follow later."""

    def __init__(self, context: ParseContext):
        self.channel = context.initial_channel

    def consume(self, lexeme: Lexeme) -> list[Event]:
        if lexeme.terminal is not None:
            if lexeme.terminal == "reasoning_start":
                self.channel = "reasoning"
            elif lexeme.terminal == "reasoning_end":
                self.channel = "content"
            return [Control(lexeme.span, lexeme.terminal)]
        cls = ReasoningDelta if self.channel == "reasoning" else ContentDelta
        return [cls(lexeme.span, lexeme.text)]

    def finish(self, offset: int, reason: str) -> list[Event]:
        return []


class OutputDemultiplexer:
    """One lexer, grammar and monotonic commit cursor per generation.

    Text lookahead is strictly shorter than the longest configured terminal.
    Each input character examines at most that fixed terminal length. No prior
    response prefix is rescanned. ``examined_chars`` counts actual prefix tests,
    allowing deterministic scaling assertions independent of wall-clock noise.
    """

    def __init__(self, context: ParseContext, grammar: Grammar | None = None):
        self.context = context
        self.grammar = grammar if grammar is not None else ChannelGrammar(context)
        self._terminals = {t.text: t.name for t in context.terminals}
        if len(self._terminals) != len(context.terminals) or any(
            not t for t in self._terminals
        ):
            raise ValueError("terminals must be nonempty and unique")
        self._prefixes = {
            text[:i] for text in self._terminals for i in range(1, len(text))
        }
        if self._prefixes.intersection(self._terminals):
            raise ValueError("terminal spellings must be prefix-free")
        self._tokens: dict[int, str] = {}
        for terminal in context.terminals:
            for token in terminal.token_ids:
                if token in self._tokens:
                    raise ValueError("terminal token IDs must be unique")
                self._tokens[token] = terminal.name
        self.max_marker_chars = max(map(len, self._terminals), default=0)
        self.pending = ""
        self.offset = 0
        self.commit_cursor = 0
        self.examined_chars = 0
        self.input_bytes = 0
        self._finished = False
        self._failed = False
        self._mode: str | None = None

    def _commit(
        self, events: list[Event], *, terminal: bool = False, complete: bool = False
    ) -> list[Event]:
        cursor = self.commit_cursor
        for event in events:
            if isinstance(event, Finish) and not terminal:
                raise ValueError("terminal events are owned by finish")
            if (
                event.span.start != cursor
                or not event.span.start <= event.span.end <= self.offset
            ):
                raise ValueError(
                    "events must own consecutive, nonoverlapping source spans"
                )
            if (
                isinstance(event, (ContentDelta, ReasoningDelta))
                and len(event.text) != event.span.end - event.span.start
            ):
                raise ValueError("text delta length must match its source span")
            cursor = event.span.end
        if complete and cursor != self.offset:
            raise ValueError("native batch must cover its complete source span")
        self.commit_cursor = cursor
        return events

    def _consume(self, lexeme: Lexeme) -> list[Event]:
        return self._commit(self.grammar.consume(lexeme))

    def feed(self, text: str, *, token_id: int | None = None) -> list[Event]:
        if self._failed:
            raise ValueError("parser failed; create a new request-local instance")
        if self._finished:
            raise ValueError("cannot feed a terminal parser")
        if not text and token_id not in self._tokens:
            return []
        if self._mode == "native":
            raise ValueError("native events cannot be reclassified as text")
        try:
            return self._feed(text, token_id=token_id)
        except Exception:
            # Adapter state may already have advanced. Never retry or replay
            # after a failed feed; fail this request-local parser permanently.
            self._failed = True
            raise

    def _feed(self, text: str, *, token_id: int | None) -> list[Event]:
        self._mode = "text"
        self.input_bytes += len(text.encode("utf-8"))
        result: list[Event] = []
        if token_id is not None and token_id in self._tokens:
            # The pending text preceded this atomic control. Its possible
            # spelling cannot absorb any portion of the new token.
            result += self._release_pending()
            start = self.offset
            self.offset += len(text)
            result += self._consume(
                Lexeme(Span(start, self.offset), text, self._tokens[token_id])
            )
            return result
        plain: list[str] = []
        plain_start = self.offset - len(self.pending)

        def flush():
            if plain:
                value = "".join(plain)
                result.extend(
                    self._consume(
                        Lexeme(Span(plain_start, plain_start + len(value)), value)
                    )
                )
                plain.clear()

        for char in text:
            self.offset += 1
            self.pending += char
            while self.pending:
                self.examined_chars += len(self.pending)
                if self.pending in self._terminals:
                    flush()
                    start = self.offset - len(self.pending)
                    result += self._consume(
                        Lexeme(
                            Span(start, self.offset),
                            self.pending,
                            self._terminals[self.pending],
                        )
                    )
                    self.pending = ""
                    plain_start = self.offset
                    break
                if self.pending in self._prefixes:
                    break
                plain.append(self.pending[0])
                self.pending = self.pending[1:]
        flush()
        return result

    def _release_pending(self) -> list[Event]:
        if not self.pending:
            return []
        lexeme = Lexeme(
            Span(self.offset - len(self.pending), self.offset), self.pending
        )
        self.pending = ""
        return self._consume(lexeme)

    def release_pending(self) -> list[Event]:
        """Resolve a held marker prefix that an out-of-band event interrupted."""
        if self._failed or self._finished:
            return []
        try:
            return self._release_pending()
        except Exception:
            self._failed = True
            raise

    def accept_native(self, events: list[Event], *, end_offset: int) -> list[Event]:
        """Accept already routed events without interpreting their text again.

        Native adapters supply offsets in their decoded generation coordinate
        space, including control spans. Text/native modes cannot be mixed.
        """
        if (
            self._failed
            or self._finished
            or self._mode == "text"
            or end_offset < self.offset
        ):
            raise ValueError("invalid native event lifecycle")
        if any(isinstance(event, Finish) for event in events):
            raise ValueError("terminal events are owned by finish")
        old_offset = self.offset
        self.offset = end_offset
        try:
            result = self._commit(events, complete=True)
        except Exception:
            self.offset = old_offset
            raise
        if events:
            self._mode = "native"
        return result

    def finish(self, reason: str = "stop") -> list[Event]:
        if self._failed:
            raise ValueError("parser failed; create a new request-local instance")
        if self._finished:
            return []
        try:
            return self._finish(reason)
        except Exception:
            self._failed = True
            raise

    def _finish(self, reason: str) -> list[Event]:
        result: list[Event] = []
        result += self._release_pending()
        if self._mode != "native":
            result += self._commit(self.grammar.finish(self.offset, reason))
        if self.commit_cursor != self.offset:
            raise ValueError("grammar left source spans unowned at finish")
        result += self._commit(
            [Finish(Span(self.offset, self.offset), reason)], terminal=True
        )
        self._finished = True
        return result

    def cancel(self) -> list[Event]:
        return self.finish("cancelled")


@dataclass
class ReducedCall:
    index: int
    name: str
    arguments: list[str] = field(default_factory=list)
    complete: bool = False
    failed: bool = False


class EventReducer:
    """Buffered and streaming consumers accumulate the same ordered events.

    Only successfully ended calls appear in ``calls``. Argument syntax and
    authorization are grammar responsibilities; the reducer never fabricates
    arguments or repairs a failed/incomplete call.
    """

    def __init__(self):
        self.content: list[str] = []
        self.reasoning: list[str] = []
        self._calls: dict[int, ReducedCall] = {}
        self._failed_indices: set[int] = set()
        self.failures: list[ParseFailure] = []
        self.finish_reason: str | None = None

    def feed(self, events: list[Event]):
        for event in events:
            if self.finish_reason is not None:
                raise ValueError("events after finish")
            if isinstance(event, ContentDelta):
                self.content.append(event.text)
            elif isinstance(event, ReasoningDelta):
                self.reasoning.append(event.text)
            elif isinstance(event, ToolStart):
                if event.index in self._calls or event.index in self._failed_indices:
                    raise ValueError("duplicate or failed tool index")
                self._calls[event.index] = ReducedCall(event.index, event.name)
            elif isinstance(event, (ToolArgs, ToolEnd)):
                call = self._calls[event.index]
                if call.complete or call.failed:
                    raise ValueError("tool event after terminal call")
                if isinstance(event, ToolArgs):
                    call.arguments.append(event.text)
                else:
                    call.complete = True
            elif isinstance(event, ParseFailure):
                if event.index in self._failed_indices:
                    raise ValueError("failure after terminal call")
                if event.index is not None and event.index in self._calls:
                    call = self._calls[event.index]
                    if call.complete or call.failed:
                        raise ValueError("failure after terminal call")
                    call.failed = True
                if event.index is not None:
                    self._failed_indices.add(event.index)
                self.failures.append(event)
            elif isinstance(event, Finish):
                self.finish_reason = event.reason

    @property
    def calls(self) -> list[dict]:
        return [
            {"name": call.name, "arguments": "".join(call.arguments)}
            for call in self._calls.values()
            if call.complete and not call.failed
        ]
