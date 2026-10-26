# SPDX-License-Identifier: Apache-2.0
"""Selected-family parsing session shared by all API surfaces and reductions."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from fastapi import HTTPException

from ..domain.events import StreamEvent
from ..tool_call_authorization import declared_names
from .output_demux import (
    ContentDelta,
    EventReducer,
    OutputDemultiplexer,
    ParseContext,
    ParseFailure,
    ReasoningDelta,
    ToolArgs,
    ToolEnd,
    ToolStart,
)
from .selected_grammar import SelectedGrammar, ToolPolicy, terminals

QWEN_REASONING = frozenset({"qwen3", "hy_v3", "hy3"})
GLM_REASONING = frozenset({"glm4", "glm5"})
HERMES_TOOLS = frozenset({"hermes", "nous", "qwen3_coder"})
QWEN_TOOLS = frozenset({"qwen", "qwen3", "qwen3_xml"})
GLM_TOOLS = frozenset({"glm47", "glm4"})
PAIRED_TOOLS = frozenset(
    {"auto", "generic", "nemotron", "nemotron3", "muse", "seed_oss", "seed", "gpt_oss"}
)


def supported(cfg, tools_requested):
    reasoning = cfg.reasoning_parser_name
    tool = cfg.tool_call_parser
    if reasoning is None and cfg.reasoning_parser is not None:
        return False
    if reasoning in QWEN_REASONING | GLM_REASONING:
        return (
            not tools_requested
            or tool is None
            or tool in HERMES_TOOLS | QWEN_TOOLS | GLM_TOOLS
            or (reasoning in QWEN_REASONING and tool in PAIRED_TOOLS)
        )
    return (
        bool(tools_requested)
        and reasoning is None
        and tool in HERMES_TOOLS | QWEN_TOOLS | GLM_TOOLS
    )


def _request_dict(request):
    return (
        request.model_dump(exclude_none=True)
        if hasattr(request, "model_dump")
        else request or {}
    )


class OutputParseError(HTTPException):
    def __init__(self, message="Generated tool call was incomplete or invalid"):
        super().__init__(502, message)


@dataclass
class ParsedResult:
    content: str
    reasoning: str
    calls: list[dict]
    failures: list[ParseFailure]
    required: bool = False
    required_name: str | None = None


class ParsingSession:
    def __init__(
        self,
        cfg,
        request,
        *,
        tools_requested,
        thinking=None,
        authorization=None,
        promote_reasoning=True,
        reasoning_max_tokens=None,
    ):
        self.cfg = cfg
        self.request = _request_dict(request)
        self.tools_requested = tools_requested
        self.thinking = thinking
        self.authorization = frozenset(
            authorization if authorization is not None else declared_names(request)
        )
        if not tools_requested:
            self.authorization = frozenset()
        self.schemas = {
            tool["function"]["name"]: tool["function"].get("parameters", {})
            for tool in self.request.get("tools", [])
            if isinstance(tool, dict) and isinstance(tool.get("function"), dict)
        }
        self.promote_reasoning = promote_reasoning
        self.reasoning_max_tokens = reasoning_max_tokens
        self._reasoning_chars = 0
        self.parser = None
        self.reducer = EventReducer()
        self.native = False
        self.finished = False
        self.ids = {}
        self.last_output = None
        self.prompt = None
        self.completed = []
        choice = self.request.get("tool_choice")
        self.required_name = (
            choice.get("function", {}).get("name") or choice.get("name")
            if isinstance(choice, dict)
            else None
        )
        self.drop_calls = choice == "none"
        self._public_indices = {}

    def _initialize(self, output):
        if getattr(output, "channel", None) is not None or getattr(
            output, "tool_calls", None
        ):
            self.native = True
            return
        prompt = getattr(output, "prompt_parse_context", None) or self.prompt
        reasoning = self.cfg.reasoning_parser_name
        default = (
            "reasoning"
            if reasoning in QWEN_REASONING or reasoning == "glm5"
            else "content"
        )
        policy = ToolPolicy(
            self.cfg.tool_call_parser,
            self.schemas,
            bool(
                self.tools_requested
                and self.cfg.enable_auto_tool_choice
                and self.cfg.tool_call_parser
            ),
            self.promote_reasoning,
        )
        context = ParseContext(
            prompt=prompt,
            default_channel=default,
            terminals=terminals(policy),
            authorized_tools=self.authorization,
        )
        self.parser = OutputDemultiplexer(context, SelectedGrammar(context, policy))

    def feed(self, output):
        if self.finished:
            raise ValueError("generation after parser finish")
        self.last_output = output
        self.prompt = getattr(output, "prompt_parse_context", None) or self.prompt
        if (
            self.parser is None
            and not self.native
            and not output.new_text
            and getattr(output, "channel", None) is None
            and not getattr(output, "tool_calls", None)
        ):
            return []
        if self.parser is None and not self.native:
            self._initialize(output)
        if self.native:
            return None
        calls = getattr(output, "tool_calls", None)
        channel = getattr(output, "channel", None)
        if calls or channel is not None:
            events = []
            routed_text = ""
            routed_channel = None
            if channel is None:
                events += self.parser.feed(output.new_text or "")
            elif channel == "content" or (channel == "reasoning" and not calls):
                routed_text = output.new_text or ""
                routed_channel = channel
            elif channel not in {"tool_call", "reasoning"}:
                raise OutputParseError("Unknown engine output channel")
            try:
                metadata = self.parser.accept_metadata(
                    calls or [],
                    text=routed_text,
                    channel=routed_channel,
                )
            except (ValueError, TypeError, AttributeError, RecursionError) as exc:
                raise OutputParseError() from exc
            return self._convert(events + metadata)
        return self._convert(self.parser.feed(output.new_text or ""))

    def _convert(self, events):
        # Reasoning budgets may suppress private text, never promote it into
        # public content or inject synthetic grammar markers.
        self.reducer.feed(events)
        result = []
        for event in events:
            if isinstance(event, ContentDelta):
                result.append(StreamEvent("content", content=event.text))
            elif isinstance(event, ReasoningDelta):
                text = event.text
                if self.reasoning_max_tokens is not None:
                    remaining = max(
                        0, self.reasoning_max_tokens * 4 - self._reasoning_chars
                    )
                    text = text[:remaining]
                self._reasoning_chars += len(event.text)
                if text:
                    result.append(StreamEvent("reasoning", reasoning=text))
            elif isinstance(event, ToolStart):
                if (
                    self.drop_calls
                    or (
                        self.required_name is not None
                        and event.name != self.required_name
                    )
                    or (
                        (
                            self.request.get("parallel_tool_calls") is False
                            or self.required_name is not None
                        )
                        and self._public_indices
                    )
                ):
                    continue
                index = len(self._public_indices)
                self._public_indices[event.index] = index
                self.ids[event.index] = event.call_id or "call_" + uuid.uuid4().hex[:12]
                result.append(
                    StreamEvent(
                        "tool_call",
                        tool_calls=[
                            {
                                "index": index,
                                "id": self.ids[event.index],
                                "type": "function",
                                "function": {"name": event.name, "arguments": ""},
                            }
                        ],
                    )
                )
            elif isinstance(event, ToolArgs) and event.index in self._public_indices:
                result.append(
                    StreamEvent(
                        "tool_call",
                        tool_calls=[
                            {
                                "index": self._public_indices[event.index],
                                "function": {"arguments": event.text},
                            }
                        ],
                    )
                )
            elif isinstance(event, ToolEnd):
                self.completed.append(event.index)
            elif isinstance(event, ParseFailure) and event.index is not None:
                result.append(
                    StreamEvent(
                        "parse_failure",
                        metadata={"code": event.code, "tool_index": event.index},
                    )
                )
        return result

    def finish(self, reason=None):
        if self.finished or self.native:
            return []
        self.finished = True
        if self.parser is None:
            return []
        return self._convert(
            self.parser.finish(
                reason or getattr(self.last_output, "finish_reason", None) or "stop"
            )
        )

    @property
    def result(self):
        calls = [
            (index, call)
            for index, call in zip(self.completed, self.reducer.calls)
            if index in self._public_indices
        ]
        calls = [
            {**call, "id": self.ids.get(index, "call_" + uuid.uuid4().hex[:12])}
            for index, call in calls
        ]
        reasoning = "".join(self.reducer.reasoning)
        if self.reasoning_max_tokens is not None:
            reasoning = reasoning[: max(0, self.reasoning_max_tokens * 4)]
        choice = self.request.get("tool_choice")
        required_name = None
        if isinstance(choice, dict):
            required_name = choice.get("function", {}).get("name") or choice.get("name")
        return ParsedResult(
            "".join(self.reducer.content),
            reasoning,
            calls,
            self.reducer.failures,
            choice == "required" or required_name is not None,
            required_name,
        )


def make_session(
    cfg,
    request=None,
    *,
    tools_requested=None,
    thinking=None,
    authorization=None,
    promote_reasoning=True,
    reasoning_max_tokens=None,
):
    if tools_requested is None:
        tools_requested = bool(_request_dict(request).get("tools"))
    if not supported(cfg, tools_requested):
        return None
    return ParsingSession(
        cfg,
        request,
        tools_requested=tools_requested,
        thinking=thinking,
        authorization=authorization,
        promote_reasoning=promote_reasoning,
        reasoning_max_tokens=reasoning_max_tokens,
    )


def parse_buffered(
    cfg,
    output,
    request,
    *,
    thinking=None,
    promote_reasoning=True,
    reasoning_max_tokens=None,
):
    if (
        getattr(output, "channel", None) is not None
        or getattr(output, "tool_calls", None)
        or getattr(output, "reasoning_text", None)
    ):
        return None
    session = make_session(
        cfg,
        request,
        thinking=thinking,
        promote_reasoning=promote_reasoning,
        reasoning_max_tokens=reasoning_max_tokens,
    )
    if session is None:
        return None
    from types import SimpleNamespace

    session.feed(
        SimpleNamespace(
            new_text=getattr(output, "raw_text", None) or output.text,
            prompt_parse_context=getattr(output, "prompt_parse_context", None),
            finish_reason=getattr(output, "finish_reason", None),
        )
    )
    session.finish()
    return session.result


def protocol_calls(parsed):
    if any(failure.index is not None for failure in parsed.failures):
        raise OutputParseError()
    if parsed.required and (
        not parsed.calls
        or (
            parsed.required_name is not None
            and any(call["name"] != parsed.required_name for call in parsed.calls)
        )
    ):
        raise OutputParseError("Generation did not fulfill the requested tool choice")
    from ..api.models import FunctionCall, ToolCall

    return [
        ToolCall(
            id=call["id"],
            type="function",
            function=FunctionCall(name=call["name"], arguments=call["arguments"]),
        )
        for call in parsed.calls
    ] or None
