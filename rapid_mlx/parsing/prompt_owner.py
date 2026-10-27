# SPDX-License-Identifier: Apache-2.0
"""Request-local prompt channel ownership.

The existing demultiplexer owns channel transitions. For Flint and LFM2.5,
consumers pass only its content events to their existing tool parser; reasoning
never enters tool input. Selected Qwen-reasoning families with tool-call
envelopes additionally let the owner close each envelope and decode it once
with the configured parser's complete-output extractor, so buffered and
streaming consumers receive the same typed calls.
"""

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

    Text outside envelopes is ordinary content or reasoning. An envelope
    opened from reasoning ends that reasoning phase, and a later closer for the
    still-open block is consumed once. A closed envelope is decoded once; a
    span that yields no call is a parse failure and is not published. An
    envelope left open at finish is decoded, or stays literal in its channel.
    """

    def __init__(self, context: ParseContext, tools: EnvelopeTools):
        super().__init__(context)
        self.tools = tools
        self.reasoning_seen = self.channel == "reasoning"
        self.reasoning_open = self.channel == "reasoning"
        self.envelope: list[Lexeme] | None = None
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
            self.envelope.append(lexeme)
            return self._close() if terminal == "tool_end" else []
        # vllm/reasoning/kimi_k2_reasoning_parser.py@5df63a7: a tool section
        # opener ends the reasoning phase; the tool parser owns the section.
        if terminal in ("tool_start", "tool_args"):
            self.envelope = [lexeme]
            self.origin = self.channel
            return []
        # python/sglang/srt/parser/reasoning_parser.py::
        # BaseReasoningFormatDetector@baa9bef: the opener is honored once and a
        # closer ends only an open block; other markers are literal output.
        if terminal == "reasoning_start":
            if self.reasoning_seen and self.channel == "content":
                return [self._literal(lexeme)]
            self.reasoning_seen = self.reasoning_open = True
            self.channel = "reasoning"
            return [Control(lexeme.span, terminal)]
        if terminal == "reasoning_end" and self.reasoning_open:
            self.reasoning_open = False
            self.channel = "content"
            return [Control(lexeme.span, terminal)]
        return [self._literal(lexeme)]

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

    def _close(self) -> list[Event]:
        lexemes, self.envelope = self.envelope or [], None
        span = Span(lexemes[0].span.start, lexemes[-1].span.end)
        self.channel = "content"
        calls = self.tools.decode("".join(lexeme.text for lexeme in lexemes))
        if not calls:
            return [ParseFailure(span, "invalid_tool_call")]
        return self._calls(span, calls)

    def finish(self, offset: int, reason: str) -> list[Event]:
        if self.envelope is None:
            return []
        lexemes, self.envelope = self.envelope, None
        span = Span(lexemes[0].span.start, lexemes[-1].span.end)
        text = "".join(lexeme.text for lexeme in lexemes)
        calls = self.tools.decode(text)
        if calls:
            return self._calls(span, calls)
        return [self._literal(Lexeme(span, text), self.origin)]


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
        return self.parser.grammar.calls[event.index]


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
