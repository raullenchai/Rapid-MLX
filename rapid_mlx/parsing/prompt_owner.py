# SPDX-License-Identifier: Apache-2.0
"""Request-local prompt channel ownership for Flint and LFM2.5.

The existing demultiplexer owns channel transitions. Consumers pass only its
content events to their existing tool parser; reasoning never enters tool input.
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
    ReasoningDelta,
    Terminal,
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
    def __init__(self, model: str | None = ""):
        self.model = (model or "").lower()
        self.parser: OutputDemultiplexer | None = None
        self.prompt_context: PromptParseContext | None = None
        self.content: list[str] = []
        self.reasoning: list[str] = []

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
        return events


def split_buffered(output: GenerationOutput, model: str | None):
    """Apply the same owner before buffered tool extraction."""
    if getattr(output, "reasoning_text", ""):
        return output, False
    owner = PromptOutputOwner(model)
    owner._initialize(output)
    if not owner.active:
        return output, False
    owner.observe(
        replace(output, new_text=output.raw_text or output.text, finished=True)
    )
    content = "".join(owner.content)
    return replace(
        output, text=content, raw_text=content, reasoning_text="".join(owner.reasoning)
    ), True
