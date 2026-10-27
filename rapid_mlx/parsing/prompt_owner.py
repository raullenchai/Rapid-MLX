# SPDX-License-Identifier: Apache-2.0
"""Request-local prompt channel ownership for Flint and LFM2.5.

The existing demultiplexer owns channel transitions. Consumers pass only its
content events to their existing tool parser; reasoning never enters tool input.
"""

from dataclasses import replace

from ..engine.base import GenerationOutput
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
        self.in_tool = False

    def consume(self, lexeme: Lexeme) -> list[Event]:
        # mlx_lm/tool_parsers/pythonic.py::parse_tool_call@4cfc647:
        # native LFM envelope tokens delimit the pythonic payload. Consume
        # controls here; the existing tool parser receives only that payload.
        if lexeme.terminal in ("lfm_tool_start", "lfm_tool_end"):
            if self.channel == "reasoning":
                return [ReasoningDelta(lexeme.span, lexeme.text)]
            self.in_tool = lexeme.terminal == "lfm_tool_start"
            return [Control(lexeme.span, lexeme.terminal)]
        # vllm/parser/abstract_parser.py::DelegatingParser.parse_delta@5df63a7:
        # once a content tool starts, its bytes belong to the tool parser.
        if self.in_tool:
            if lexeme.terminal == "tool_end":
                self.in_tool = False
            return [ContentDelta(lexeme.span, lexeme.text)]
        if lexeme.terminal in ("tool_start", "tool_end"):
            if self.channel == "content":
                self.in_tool = lexeme.terminal == "tool_start"
                return [ContentDelta(lexeme.span, lexeme.text)]
            return [ReasoningDelta(lexeme.span, lexeme.text)]
        # python/sglang/srt/parser/reasoning_parser.py::
        # BaseReasoningFormatDetector._parse_streaming_increment_impl@baa9bef:
        # a closer ends reasoning only while reasoning is open. A stray closer
        # is literal output; removing it would hide a model/grammar failure.
        if lexeme.terminal == "reasoning_end" and self.channel == "content":
            return [ContentDelta(lexeme.span, lexeme.text)]
        return super().consume(lexeme)


class PromptOutputOwner:
    def __init__(self, model: str | None = ""):
        self.model = (model or "").lower()
        self.parser: OutputDemultiplexer | None = None
        self.content: list[str] = []
        self.reasoning: list[str] = []

    @property
    def active(self) -> bool:
        return self.parser is not None

    def observe(self, output: GenerationOutput) -> list[Event] | None:
        context = getattr(output, "prompt_parse_context", None)
        if self.parser is None and context is not None and output.channel is None:
            if context.template_contract == "flint" or "lfm2.5" in self.model:
                # mlx_lm/server.py::ResponseGenerator._tokenize@4cfc647:
                # rendered prompt state, never the requested thinking switch.
                suffix = context.assistant_suffix
                initial = (
                    "reasoning"
                    if suffix.rfind("<think>") > suffix.rfind("</think>")
                    else "content"
                )
                context = replace(context, initial_channel=initial)
                config = ParseContext(
                    prompt=context,
                    terminals=(
                        Terminal("reasoning_start", "<think>"),
                        Terminal("reasoning_end", "</think>"),
                        Terminal("tool_start", "<tool_call>"),
                        Terminal("tool_end", "</tool_call>"),
                        *(
                            (
                                Terminal("lfm_tool_start", "<|tool_call_start|>"),
                                Terminal("lfm_tool_end", "<|tool_call_end|>"),
                            )
                            if "lfm2.5" in self.model
                            else ()
                        ),
                    ),
                )
                self.parser = OutputDemultiplexer(config, ContentToolGrammar(config))
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
    owner = PromptOutputOwner(model)
    events = owner.observe(
        replace(output, new_text=output.raw_text or output.text, finished=True)
    )
    if events is None:
        return output, False
    content = "".join(owner.content)
    return replace(
        output, text=content, raw_text=content, reasoning_text="".join(owner.reasoning)
    ), True
