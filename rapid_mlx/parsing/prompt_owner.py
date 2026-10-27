# SPDX-License-Identifier: Apache-2.0
"""Request-local prompt channel ownership.

The existing demultiplexer owns channel transitions. For Flint and LFM2.5,
consumers pass only its content events to their existing tool parser; reasoning
never enters tool input. Selected Qwen-reasoning families with tool-call
envelopes additionally let the owner close each envelope and decode it once
with the configured parser's complete-output extractor, so buffered and
streaming consumers receive the same typed calls.
"""

import re
from dataclasses import replace

from ..engine.base import GenerationOutput
from ..prompt_parse_context import PromptParseContext
from .output_demux import (
    ChannelGrammar,
    ContentDelta,
    Control,
    Event,
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


class ContentToolGrammar(ChannelGrammar):
    """Tool envelopes own their contents after the reasoning phase."""

    def __init__(self, context):
        super().__init__(context)
        self.tool_end: str | None = None
        self.reasoning_seen = self.channel == "reasoning"

    def consume(self, lexeme: Lexeme) -> list[Event]:
        # vllm/parser/abstract_parser.py::DelegatingParser.parse_delta@5df63a7:
        # tool payload stays in the tool phase until its own closing control.
        # The first closing control ends the envelope: neither format escapes
        # it inside argument values (flint_sglang/tool_parser.py non-greedy
        # func_detail_regex; LFM's end marker is a single special token).
        if self.tool_end is not None:
            if lexeme.terminal == self.tool_end:
                self.tool_end = None
                if lexeme.terminal == "lfm_tool_end":
                    return [Control(lexeme.span, lexeme.terminal)]
            return [ContentDelta(lexeme.span, lexeme.text)]
        # mlx_lm/tool_parsers/pythonic.py::parse_tool_call@4cfc647:
        # native LFM envelope tokens delimit the pythonic payload. Consume
        # controls here; the existing tool parser receives only that payload.
        if lexeme.terminal in ("lfm_tool_start", "lfm_tool_end"):
            if self.channel == "reasoning":
                return [ReasoningDelta(lexeme.span, lexeme.text)]
            self.tool_end = (
                "lfm_tool_end" if lexeme.terminal == "lfm_tool_start" else None
            )
            return [Control(lexeme.span, lexeme.terminal)]
        if lexeme.terminal in ("tool_start", "tool_end"):
            if self.channel == "content":
                self.tool_end = "tool_end" if lexeme.terminal == "tool_start" else None
                return [ContentDelta(lexeme.span, lexeme.text)]
            return [ReasoningDelta(lexeme.span, lexeme.text)]
        # python/sglang/srt/parser/reasoning_parser.py::
        # BaseReasoningFormatDetector._parse_streaming_increment_impl@baa9bef:
        # a closer ends reasoning only while reasoning is open. A stray closer
        # is literal output; removing it would hide a model/grammar failure.
        if lexeme.terminal == "reasoning_end" and self.channel == "content":
            return [ContentDelta(lexeme.span, lexeme.text)]
        # Same detector: the start marker is honored once (stripped_think_start).
        # A later opener, e.g. inside an unwrapped call's argument, is literal.
        if lexeme.terminal == "reasoning_start":
            if self.reasoning_seen and self.channel == "content":
                return [ContentDelta(lexeme.span, lexeme.text)]
            self.reasoning_seen = True
        return super().consume(lexeme)


# Markup argument values (GLM ``<arg_value>``, Qwen XML ``<parameter=...>``)
# are opaque to the envelope grammar, like JSON strings: until the value's own
# closer, markers inside them are argument text for the configured extractor.
VALUE_CLOSERS = {"<arg_value>": "</arg_value>", "<parameter=": "</parameter>"}
VALUE_MARKER = re.compile(
    "|".join(map(re.escape, [*VALUE_CLOSERS, *VALUE_CLOSERS.values()]))
)
VALUE_TAIL = max(map(len, [*VALUE_CLOSERS, *VALUE_CLOSERS.values()])) - 1

# Families whose Qwen-derived reasoning grammar pairs with a ``<tool_call>``
# envelope. Each format keeps its existing complete-output extractor; the
# owner only decides which source span is a tool envelope.
QWEN_REASONING = frozenset({"qwen3", "hy_v3", "hy3"})
JSON_ENVELOPE_TOOLS = frozenset(
    {"hermes", "nous", "qwen3_coder", "qwen", "qwen3", "qwen3_xml"}
)
GLM_ENVELOPE_TOOLS = frozenset({"glm47", "glm4"})


class EnvelopeTools:
    """One request's envelope terminals and complete-call decoder."""

    def __init__(self, parser_name: str, request, tokenizer=None):
        from ..tool_parsers import ToolParserManager

        self.request = (
            request.model_dump() if hasattr(request, "model_dump") else request
        )
        self.parser = ToolParserManager.get_tool_parser(parser_name)(tokenizer)
        # The owner already separated reasoning: the decoder sees only the
        # envelope, so reasoning markers inside it are argument text.
        self.parser.skip_reasoning_parsing = True
        self.terminals = (
            Terminal("tool_start", "<tool_call>"),
            Terminal("tool_end", "</tool_call>"),
            # GLM argument markers are tool-grammar controls; an argument
            # list whose opener was not decoded still belongs to the call.
            *(
                (Terminal("tool_args", "<arg_key>"),)
                if parser_name in GLM_ENVELOPE_TOOLS
                else ()
            ),
        )

    def decode(self, text: str) -> list[dict]:
        # vllm/entrypoints/openai/tool_parsers/abstract_tool_parser.py::
        # ToolParser.extract_tool_calls@5df63a7: one complete-output parse per
        # owned envelope. Buffered and streaming consumers share this result.
        self.parser.reset()
        result = self.parser.extract_tool_calls(text, self.request)
        return list(result.tool_calls) if result.tools_called else []


def _field(request, name):
    return (
        request.get(name) if isinstance(request, dict) else getattr(request, name, None)
    )


def selected_envelope_tools(
    cfg, request, reasoning_max_tokens=None
) -> EnvelopeTools | None:
    """Select the owner's envelope grammar for one request, or keep legacy."""
    from ..service.helpers import tool_choice_is_none
    from ..tool_call_authorization import declared_names

    parser = getattr(cfg, "tool_call_parser", None)
    if (
        getattr(cfg, "reasoning_parser_name", None) not in QWEN_REASONING
        or parser not in JSON_ENVELOPE_TOOLS | GLM_ENVELOPE_TOOLS
        or not getattr(cfg, "enable_auto_tool_choice", False)
        or request is None
        or not declared_names(request)
        # Disabled tools keep their literal-text contract.
        or tool_choice_is_none(request)
        # Reasoning budgets remain owned by the existing reasoning parser.
        or reasoning_max_tokens is not None
        or _field(request, "reasoning_max_tokens") is not None
    ):
        return None
    engine = getattr(cfg, "engine", None)
    return EnvelopeTools(parser, request, getattr(engine, "_tokenizer", None))


def requested_thinking(cfg, request) -> bool:
    """Fallback channel evidence when the rendered prompt state is unknown."""
    from ..service.helpers import _effective_enable_thinking, _resolve_enable_thinking

    return (
        _effective_enable_thinking(
            _resolve_enable_thinking(request), served_model_identity(cfg)
        )
        is not False
    )


class EnvelopeToolGrammar(ChannelGrammar):
    """Reasoning first; each ``<tool_call>`` envelope is one tool phase.

    Text outside envelopes is ordinary content or reasoning. An envelope is a
    tool phase nested in the channel it interrupts. A reasoning block the
    output opened resumes after it and ends only at its closer; reasoning
    inherited from the prompt ends at the tool phase. Every closer outside
    an envelope is a control; only one that ends an open block changes the
    channel. A closed envelope is
    decoded once; a closer or opener inside a JSON string or markup argument
    value is argument text. An envelope that yields no call, a partial envelope interrupted by a
    new opener or by its reasoning block's closer, and an envelope left open at finish stay verbatim in the
    channel they arrived in. A closed GLM argument list without its opener is
    an unpublished parse failure.
    """

    def __init__(self, context: ParseContext, tools: EnvelopeTools):
        super().__init__(context)
        self.tools = tools
        self.reasoning_seen = self.channel == "reasoning"
        self.reasoning_open = self.channel == "reasoning"
        # True while a reasoning block opened by the output itself is open.
        self.delimited = False
        self.envelope: list[Lexeme] | None = None
        self.json_body: bool | None = False
        self.json_string = self.json_escape = False
        self.value_closer: str | None = None
        self.value_tail = ""
        self.origin = self.channel
        self.calls: dict[int, dict] = {}

    def _literal(self, lexeme: Lexeme, channel: str | None = None) -> Event:
        cls = (
            ReasoningDelta if (channel or self.channel) == "reasoning" else ContentDelta
        )
        return cls(lexeme.span, lexeme.text)

    def consume(self, lexeme: Lexeme) -> list[Event]:
        terminal = lexeme.terminal
        if self.envelope is not None:
            if terminal == "tool_start" and not self._in_argument():
                # A new opener outside an argument value starts the next envelope;
                # the unfinished one before it is ordinary text.
                flushed = self._flush()
                self._open(lexeme)
                return flushed
            if (
                terminal == "reasoning_end"
                and self.origin == "reasoning"
                and self.reasoning_open
                and not self._in_argument()
            ):
                # The reasoning block the envelope interrupted ends first; the
                # unfinished envelope stays verbatim in that reasoning.
                return self._flush() + self.consume(lexeme)
            self.envelope.append(lexeme)
            if terminal == "tool_end" and not self._in_argument():
                return self._close()
            self._scan(lexeme.text)
            return []
        # The tool parser owns the envelope; the channel it interrupts is
        # resolved when the envelope closes.
        if terminal in ("tool_start", "tool_args"):
            self._open(lexeme)
            return []
        # python/sglang/srt/parser/reasoning_parser.py::
        # BaseReasoningFormatDetector@baa9bef: the opener is honored once and a
        # closer ends only an open block. A later opener in content is literal.
        if terminal == "reasoning_start":
            if self.reasoning_seen and self.channel == "content":
                return [self._literal(lexeme)]
            self.reasoning_seen = self.reasoning_open = self.delimited = True
            self.channel = "reasoning"
            return [Control(lexeme.span, terminal)]
        if terminal == "reasoning_end":
            # Every closer is a reasoning control (the #5271 contract of the
            # existing think parser); one without an open block changes no
            # channel.
            if self.reasoning_open:
                self.reasoning_open = self.delimited = False
                self.channel = "content"
            return [Control(lexeme.span, terminal)]
        return [self._literal(lexeme)]

    def _open(self, lexeme: Lexeme) -> None:
        self.envelope = [lexeme]
        # Only a ``<tool_call>`` body may be JSON; GLM arguments are not.
        self.json_body = None if lexeme.terminal == "tool_start" else False
        self.json_string = self.json_escape = False
        self.value_closer = None
        self.value_tail = ""
        self.origin = self.channel

    def _flush(self) -> list[Event]:
        """Publish the current envelope's source verbatim in its channel."""
        lexemes, self.envelope = self.envelope or [], None
        span = Span(lexemes[0].span.start, lexemes[-1].span.end)
        self.channel = self.origin
        text = "".join(lexeme.text for lexeme in lexemes)
        return [self._literal(Lexeme(span, text))]

    def _calls(self, span: Span, calls: list[dict]) -> list[Event]:
        events: list[Event] = []
        end = Span(span.end, span.end)
        for position, call in enumerate(calls):
            index = len(self.calls)
            self.calls[index] = call
            events += [
                ToolStart(span if position == 0 else end, index, call["name"]),
                ToolArgs(end, index, call["arguments"]),
                ToolEnd(end, index),
            ]
        return events

    def _in_argument(self) -> bool:
        return self.json_string or self.value_closer is not None

    def _scan(self, text: str) -> None:
        """Track whether the envelope body is inside an argument value.

        The JSON extractors decode the body with a JSON decoder, so a closer
        spelled inside an argument string belongs to the arguments; markup
        extractors read a value up to its own closing marker. State is
        incremental: each body character is read once.
        """
        for position, char in enumerate(text):
            if self.json_body is None:
                if char.isspace():
                    continue
                self.json_body = char == "{"
            if not self.json_body:
                self._scan_values(text[position:])
                return
            if self.json_escape:
                self.json_escape = False
            elif char == "\\":
                self.json_escape = self.json_string
            elif char == '"':
                self.json_string = not self.json_string

    def _scan_values(self, text: str) -> None:
        buffer = self.value_tail + text
        end = 0
        for match in VALUE_MARKER.finditer(buffer):
            marker = match.group()
            if self.value_closer is None:
                self.value_closer = VALUE_CLOSERS.get(marker)
            elif marker == self.value_closer:
                self.value_closer = None
            end = match.end()
        self.value_tail = buffer[max(end, len(buffer) - VALUE_TAIL) :]

    def _close(self, closed: bool = True) -> list[Event]:
        assert self.envelope is not None
        lexemes = self.envelope
        calls = self.tools.decode("".join(lexeme.text for lexeme in lexemes))
        if not calls:
            if closed and lexemes[0].terminal == "tool_args":
                # A closed GLM argument list without its call opener is tool
                # syntax, not prose; it is an unpublished parse failure.
                self.envelope = None
                self.channel = self.origin if self.delimited else "content"
                span = Span(lexemes[0].span.start, lexemes[-1].span.end)
                return [ParseFailure(span, "invalid_tool_call")]
            # Complete-output extractors return undecodable text as content:
            # the whole source span stays verbatim in the channel it arrived in.
            return self._flush()
        self.envelope = None
        span = Span(lexemes[0].span.start, lexemes[-1].span.end)
        # python/sglang/srt/parser/reasoning_parser.py::
        # BaseReasoningFormatDetector@baa9bef: a delimited block ends only at
        # its closer. vllm/reasoning/kimi_k2_reasoning_parser.py@5df63a7: a
        # tool section ends reasoning the output did not delimit.
        self.channel = self.origin if self.delimited else "content"
        return self._calls(span, calls)

    def finish(self, offset: int, reason: str) -> list[Event]:
        return [] if self.envelope is None else self._close(closed=False)


def served_model_identity(cfg) -> str:
    """Both configured names: a local path may not carry the family name."""
    names = (getattr(cfg, "model_path", ""), getattr(cfg, "model_name", ""))
    return " ".join(name for name in names if isinstance(name, str) and name)


def owns_prompt_state(model: str | None, *, flint: bool) -> bool:
    """Families whose output channel follows the rendered prompt state."""
    return flint or "lfm2.5" in (model or "").lower()


def owned_initial_channel(context: PromptParseContext) -> str:
    """Channel the rendered assistant suffix leaves open for owned families."""
    suffix = context.assistant_suffix
    return (
        "reasoning" if suffix.rfind("<think>") > suffix.rfind("</think>") else "content"
    )


class PromptOutputOwner:
    def __init__(
        self,
        model: str | None = "",
        *,
        tools: EnvelopeTools | None = None,
    ):
        self.model = (model or "").lower()
        self.tools = tools
        self.parser: OutputDemultiplexer | None = None
        self.prompt_context: PromptParseContext | None = None
        self.content: list[str] = []
        self.reasoning: list[str] = []
        self.calls: list[dict] = []

    @classmethod
    def for_request(
        cls, cfg, request, *, thinking=None, reasoning_max_tokens=None
    ) -> "PromptOutputOwner":
        tools = selected_envelope_tools(cfg, request, reasoning_max_tokens)
        if tools is not None and thinking is None:
            thinking = requested_thinking(cfg, request)
        # A thinking-off template closes reasoning before generation; its
        # implicit-reasoning policy stays with the existing parser.
        return cls(
            served_model_identity(cfg), tools=tools if thinking is not False else None
        )

    @property
    def active(self) -> bool:
        return self.parser is not None

    def _initialize(self, output: GenerationOutput) -> None:
        context = getattr(output, "prompt_parse_context", None) or self.prompt_context
        self.prompt_context = context
        if (
            self.parser is None
            and context is not None
            and getattr(output, "channel", None) is None
            and not getattr(output, "tool_calls", None)
        ):
            if owns_prompt_state(
                self.model, flint=context.template_contract == "flint"
            ):
                # mlx_lm/server.py::ResponseGenerator._tokenize@4cfc647:
                # rendered prompt state, never the requested thinking switch.
                context = replace(
                    context, initial_channel=owned_initial_channel(context)
                )
                config = ParseContext(
                    prompt=context,
                    terminals=(
                        Terminal("reasoning_start", "<think>"),
                        Terminal("reasoning_end", "</think>"),
                        *(
                            (
                                Terminal("lfm_tool_start", "<|tool_call_start|>"),
                                Terminal("lfm_tool_end", "<|tool_call_end|>"),
                            )
                            if "lfm2.5" in self.model
                            else (
                                Terminal("tool_start", "<tool_call>"),
                                Terminal("tool_end", "</tool_call>"),
                            )
                        ),
                    ),
                )
                self.parser = OutputDemultiplexer(config, ContentToolGrammar(config))
        if (
            self.parser is None
            and self.tools is not None
            and getattr(output, "channel", None) is None
            and not getattr(output, "tool_calls", None)
            and not (
                context is not None
                and owns_prompt_state(
                    self.model, flint=context.template_contract == "flint"
                )
            )
        ):
            # Rendered prompt state first; selected requests asked for
            # thinking, so an unknown current-turn channel starts in reasoning.
            config = ParseContext(
                prompt=context,
                default_channel="reasoning",
                terminals=(
                    Terminal("reasoning_start", "<think>"),
                    Terminal("reasoning_end", "</think>"),
                    *self.tools.terminals,
                ),
            )
            self.parser = OutputDemultiplexer(
                config, EnvelopeToolGrammar(config, self.tools)
            )

    def release_held(self, output: GenerationOutput) -> list[Event]:
        """Resolve held marker text before a native event is routed.

        A native event cannot complete a marker spelling started by earlier
        text, so the held prefix is plain text of the current channel and must
        be emitted ahead of the native event to keep stream order.
        """
        if self.parser is None or (
            getattr(output, "channel", None) is None
            and not getattr(output, "tool_calls", None)
        ):
            return []
        return self._record(self.parser.release_pending())

    def observe(self, output: GenerationOutput) -> list[Event] | None:
        self._initialize(output)
        # Native metadata is authoritative on every event, including streams
        # whose first event used text parsing. Let the existing route handlers
        # consume structured calls without feeding their argument summaries back
        # into the text grammar. Retain native narration in the final projection.
        channel = getattr(output, "channel", None)
        if channel is not None or getattr(output, "tool_calls", None):
            selected = self.prompt_context is not None and owns_prompt_state(
                self.model, flint=self.prompt_context.template_contract == "flint"
            )
            if selected and channel in (None, "content"):
                self.content.append(output.new_text)
            return None
        if self.parser is None:
            return None
        events = self.parser.feed(output.new_text)
        if output.finished:
            events += self.parser.finish(output.finish_reason or "stop")
        return self._record(events)

    def finish(self) -> list[Event]:
        return self._record(self.parser.finish()) if self.parser else []

    def _record(self, events: list[Event]) -> list[Event]:
        # vllm/parser/abstract_parser.py::DelegatingParser.parse_delta@5df63a7:
        # reasoning first; tool parsing receives only the content phase.
        for event in events:
            if isinstance(event, ContentDelta):
                self.content.append(event.text)
            elif isinstance(event, ReasoningDelta):
                self.reasoning.append(event.text)
            elif isinstance(event, ToolEnd):
                self.calls.append(self.call(event))
        return events

    def call(self, event: ToolEnd) -> dict:
        """The decoded call a ``ToolEnd`` event completed."""
        assert self.parser is not None
        grammar = self.parser.grammar
        assert isinstance(grammar, EnvelopeToolGrammar)
        return grammar.calls[event.index]


def split_buffered(
    output: GenerationOutput, model: str | None, *, cfg=None, request=None
):
    """Apply the same owner before buffered tool extraction."""
    if getattr(output, "reasoning_text", ""):
        return output, False
    owner = (
        PromptOutputOwner.for_request(cfg, request)
        if cfg is not None
        else PromptOutputOwner(model)
    )
    owner._initialize(output)
    if not owner.active:
        return output, False
    owner.observe(
        replace(output, new_text=output.raw_text or output.text, finished=True)
    )
    content = "".join(owner.content)
    reasoning = "".join(owner.reasoning)
    if owner.tools is not None:
        # The selected families keep the existing buffered reasoning-parser
        # normalization: complete fields are trimmed, so separator whitespace
        # around a decoded envelope is not a visible answer.
        content, reasoning = content.strip(), reasoning.strip()
    return replace(
        output,
        text=content,
        raw_text=content,
        reasoning_text=reasoning,
        # Decoded envelopes use the existing structured-call lane.
        tool_calls=owner.calls or output.tool_calls,
    ), True
