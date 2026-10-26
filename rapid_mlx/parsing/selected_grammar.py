# SPDX-License-Identifier: Apache-2.0
"""Selected thinking/tool grammars over the shared source-ownership core."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, cast

from .output_demux import (
    ChannelGrammar,
    ContentDelta,
    Control,
    Event,
    Lexeme,
    ParseContext,
    ParseFailure,
    ReasoningDelta,
    Span,
    Terminal,
    ToolArgs,
    ToolEnd,
    ToolStart,
)
from .tool_values import _json, decode_body, validate_arguments


@dataclass(frozen=True)
class ToolPolicy:
    format_name: str | None = None
    schemas: dict = field(default_factory=dict)
    enabled: bool = False
    promote_reasoning: bool = True


def terminals(policy: ToolPolicy) -> tuple[Terminal, ...]:
    result = [
        Terminal("reasoning_start", "<think>"),
        Terminal("reasoning_end", "</think>"),
        Terminal("backtick", "`"),
        Terminal("tilde", "~"),
    ]
    if policy.enabled:
        result += [
            Terminal("tool_start", "<tool_call>"),
            Terminal("tool_end", "</tool_call>"),
            Terminal("function_start", "<function="),
            Terminal("named_function_start", '<function name="'),
            Terminal("function_end", "</function>"),
            Terminal("parameter_start", "<parameter="),
            Terminal("named_parameter_start", '<parameter name="'),
            Terminal("parameter_end", "</parameter>"),
            Terminal("raw_start", "{"),
            Terminal("raw_end", "}"),
        ]
        if policy.format_name in {"qwen", "qwen3", "qwen3_xml", "auto", "generic"}:
            result += [
                Terminal("bracket_start", "[Calling tool:"),
                Terminal("bracket_end", ")]"),
            ]
        if policy.format_name == "muse":
            result += [
                Terminal("muse_start", "<atem:function_calls>"),
                Terminal("muse_end", "</atem:function_calls>"),
                Terminal("muse_invoke", '<atem:invoke name="'),
                Terminal("muse_invoke_end", "</atem:invoke>"),
                Terminal("muse_parameter", '<atem:parameter name="'),
                Terminal("muse_parameter_end", "</atem:parameter>"),
            ]
        if policy.format_name in {"seed_oss", "seed", "gpt_oss"}:
            result += [
                Terminal("seed_start", "<seed:tool_call>"),
                Terminal("seed_end", "</seed:tool_call>"),
            ]
        if policy.format_name in {"glm47", "glm4"}:
            result += [
                Terminal("orphan_start", "<arg_key>"),
                Terminal("arg_value_start", "<arg_value>"),
                Terminal("arg_key_end", "</arg_key>"),
                Terminal("arg_value_end", "</arg_value>"),
            ]
    return tuple(result)


def _coalesce_args(events: list[Event]) -> list[Event]:
    """One argument delta per contiguous source run in this input chunk."""
    result: list[Event] = []
    pending: list[str] = []
    first: ToolArgs | None = None
    end = 0
    for event in events:
        if first is not None and not (
            isinstance(event, ToolArgs)
            and event.index == first.index
            and event.span.start == end
        ):
            result.append(
                ToolArgs(Span(first.span.start, end), first.index, "".join(pending))
            )
            first, pending = None, []
        if isinstance(event, ToolArgs):
            if first is None:
                first = event
            pending.append(event.text)
            end = event.span.end
        else:
            result.append(event)
    if first is not None:
        result.append(
            ToolArgs(Span(first.span.start, end), first.index, "".join(pending))
        )
    return result


class JsonEnvelope:
    """Single-pass envelope scanner; full JSON validation occurs once at close.

    The common name-first object-argument shape exposes argument source spans
    incrementally. Other valid key orders remain a bounded logical candidate,
    converted once at completion; no earlier response text is rescanned.
    """

    def __init__(self):
        self.depth = 0
        self.in_string = False
        self.escape = False
        self.capture: list[str] = []
        self.role = None
        self.key = None
        self.expect = "key"
        self.name = None
        self.argument_seen = False
        self.argument_start = None
        self.argument_end = None
        self.invalid = False
        self.started = False
        self.track_values = True

    def feed(self, char: str, offset: int):
        if self.in_string:
            if self.role is not None:
                self.capture.append(char)
            if self.escape:
                self.escape = False
            elif char == "\\":
                self.escape = True
            elif char == '"':
                self.in_string = False
                if self.role is not None:
                    try:
                        value = json.loads("".join(self.capture))
                    except ValueError:
                        self.invalid = True
                    else:
                        if self.role == "key":
                            self.key = value
                            self.expect = "colon"
                        elif self.role == "name":
                            self.name = value
                            self.expect = "comma"
                    self.capture = []
                    self.role = None
            return
        if not self.started:
            if char.isspace():
                return
            self.started = True
            if char != "{":
                self.invalid = True
                return
            self.depth = 1
            return
        if self.depth == 1 and self.track_values:
            if self.expect == "key" and char == '"':
                self.role = "key"
            elif self.expect == "colon" and char == ":":
                self.expect = "value"
            elif self.expect == "value" and not char.isspace():
                if self.key == "name" and char == '"':
                    self.role = "name"
                if self.key in {"arguments", "parameters"}:
                    self.argument_seen = True
                    if char == "{" and self.name is not None:
                        self.argument_start = offset
                self.expect = "comma"
            elif char == ",":
                self.expect = "key"
        if char == '"':
            self.in_string = True
            if self.role is not None:
                self.capture = ['"']
        elif char in "{[":
            self.depth += 1
            if self.depth > 128:
                self.invalid = True
        elif char in "}]":
            self.depth -= 1
            if (
                self.argument_start is not None
                and self.argument_end is None
                and self.depth == 1
            ):
                self.argument_end = offset + 1
            if self.depth < 0:
                self.invalid = True


class SelectedGrammar(ChannelGrammar):
    """One lane owns each span, including tool promotion and orphan controls.

    Literal policy: fenced examples are text. Invalid/unclosed uncommitted
    candidates retain their original lane, including an incomplete GLM header
    after prose (R6). A committed call fails explicitly and is never replayed.
    """

    def __init__(self, context: ParseContext, policy: ToolPolicy):
        super().__init__(context)
        self.context = context
        self.policy = policy
        from ..tool_parsers.qwen3coder_tool_parser import _MarkdownCodeTracker

        # Reuse only its incremental fence-state scanner, never its prefix
        # queries or tool parsing. This request's grammar remains the owner.
        self.code = _MarkdownCodeTracker()
        self.inline_ticks = None
        self.tick_run = 0
        self.backslashes = 0
        self.has_prose = False
        self.orphan = False
        self.next_index = 0
        self.candidate: dict[str, Any] | None = None
        self.optional_outer_close = False

    def _finish_tick_run(self):
        if self.tick_run:
            if self.inline_ticks is None:
                self.inline_ticks = self.tick_run
            elif self.inline_ticks == self.tick_run:
                self.inline_ticks = None
            self.tick_run = 0

    def _literal(self, lexeme) -> list[Event]:
        self.code._feed_fence(lexeme.text)
        for char in lexeme.text:
            if char == "`":
                if self.inline_ticks is not None or self.backslashes % 2 == 0:
                    self.tick_run += 1
                self.backslashes = 0
            else:
                self._finish_tick_run()
                self.backslashes = self.backslashes + 1 if char == "\\" else 0
                if char in "\r\n":
                    self.inline_ticks = None
        cls = ReasoningDelta if self.channel == "reasoning" else ContentDelta
        if lexeme.text.strip():
            self.has_prose = True
        return [cls(lexeme.span, lexeme.text)]

    def consume(self, lexeme: Lexeme) -> list[Event]:
        if self.candidate is not None:
            return cast(list[Event], self._candidate(lexeme))
        kind = lexeme.terminal
        if self.optional_outer_close:
            if kind == "tool_end":
                self.optional_outer_close = False
                return [Control(lexeme.span, "tool_wrapper")]
            if lexeme.text.isspace():
                # The inner function close completes the call. Following
                # whitespace is literal output, even before an optional wrapper.
                # Do not hold an unbounded whitespace run awaiting a closer.
                return self._literal(lexeme)
            self.optional_outer_close = False
        if kind != "backtick":
            self._finish_tick_run()
        if (
            kind in {"backtick", "tilde"}
            or self.code._fence_result()
            or self.inline_ticks is not None
        ):
            return self._literal(lexeme)
        if self.orphan:
            if kind == "tool_end":
                self.orphan = False
                self.channel = "content"
            return [Control(lexeme.span, "orphan_tool_wire")]
        if kind in {"reasoning_start", "reasoning_end"}:
            self.has_prose = False
            return super().consume(lexeme)
        if (
            kind in {"orphan_start", "arg_value_start", "arg_value_end", "arg_key_end"}
            and not self.has_prose
        ):
            self.orphan = True
            return [Control(lexeme.span, "orphan_tool_wire")]
        allowed = self.channel != "reasoning" or self.policy.promote_reasoning
        if allowed and kind in {"muse_start", "muse_end"}:
            return [Control(lexeme.span, "tool_container")]
        if allowed and (
            kind
            in {
                "tool_start",
                "seed_start",
                "bracket_start",
                "function_start",
                "named_function_start",
                "muse_invoke",
            }
            or (kind == "raw_start" and not self.has_prose)
        ):
            self.candidate = {
                "start": lexeme.span.start,
                "body_start": lexeme.span.end,
                "origin": self.channel,
                "opener": lexeme.text,
                "kind": kind,
                "parts": [],
                "size": 0,
                "prefix": "",
                "json": None,
                "value_json": None,
                "xml_body_started": False,
                "committed": False,
                "literal": False,
                "in_value": False,
                "value_close_pending": False,
                "cursor": lexeme.span.start,
                "index": self.next_index,
            }
            if kind in {
                "function_start",
                "named_function_start",
                "raw_start",
                "muse_invoke",
            }:
                self.candidate["body_start"] = lexeme.span.start
                return self._body(lexeme)
            return []
        return self._literal(lexeme)

    def _start(self, name, end):
        c = self.candidate
        c["committed"] = True
        c["name"] = name
        c["cursor"] = end
        self.next_index += 1
        self.channel = "content"
        self.has_prose = False
        return ToolStart(Span(c["start"], end), c["index"], name)

    def _candidate(self, lexeme):
        c = self.candidate
        closer = {
            "tool_start": "</tool_call>",
            "seed_start": "</seed:tool_call>",
            "function_start": "</function>",
            "named_function_start": "</function>",
            "muse_invoke": "</atem:invoke>",
            "bracket_start": ")]",
        }.get(c["kind"])
        if (
            lexeme.pending_terminal
            and c["committed"]
            and closer
            and closer.startswith(lexeme.text)
        ):
            c["partial_closer"] = lexeme.span
            return []
        scanner = c["json"] or c["value_json"]
        if (
            lexeme.terminal == "reasoning_end"
            and c["origin"] == "reasoning"
            and not (
                self.policy.format_name in {"glm47", "glm4"}
                and c["prefix"].strip() in self.context.authorized_tools
            )
            and not c["committed"]
            and not c["in_value"]
            and not (scanner is not None and scanner.in_string)
        ):
            events = (
                []
                if c["literal"]
                else self._reject(lexeme.span.start, "literal_reasoning_example")
            )
            self.candidate = None
            return events + self.consume(lexeme)
        xml_optional = (
            self.policy.format_name
            in {"qwen3_xml", "qwen", "qwen3", "nemotron", "nemotron3"}
            and c["kind"] == "tool_start"
            and c["committed"]
            and c["json"] is None
            and lexeme.terminal == "function_end"
        )
        closes = (
            xml_optional
            or (lexeme.terminal == "bracket_end" and c["kind"] == "bracket_start")
            or (lexeme.terminal == "muse_invoke_end" and c["kind"] == "muse_invoke")
            or (lexeme.terminal == "tool_end" and c["kind"] == "tool_start")
            or (lexeme.terminal == "seed_end" and c["kind"] == "seed_start")
            or (
                lexeme.terminal == "function_end"
                and c["kind"] in {"function_start", "named_function_start"}
            )
        )
        if scanner is None:
            if c["value_close_pending"] and lexeme.text.strip():
                c["value_close_pending"] = False
                c["in_value"] = lexeme.terminal not in {
                    "parameter_start",
                    "named_parameter_start",
                    "function_end",
                    "orphan_start",
                    "tool_end",
                    "seed_end",
                    "muse_parameter",
                    "muse_invoke_end",
                }
            if lexeme.terminal in {
                "parameter_start",
                "named_parameter_start",
                "arg_value_start",
                "muse_parameter",
            }:
                c["in_value"] = True
            elif lexeme.terminal in {
                "parameter_end",
                "arg_value_end",
                "muse_parameter_end",
            }:
                c["value_close_pending"] = True
        if (
            closes
            and not c["in_value"]
            and not (scanner is not None and scanner.in_string)
        ):
            if c["literal"]:
                self.candidate = None
                cls = ReasoningDelta if c["origin"] == "reasoning" else ContentDelta
                return [cls(lexeme.span, lexeme.text)] + self._rejection_diagnostic(
                    c, lexeme.span.end
                )
            if xml_optional or c["kind"] in {
                "function_start",
                "named_function_start",
                "muse_invoke",
            }:
                c["parts"].append(lexeme.text)
                self.optional_outer_close = xml_optional
                # The function closer belongs to converted XML, so completion
                # has a zero-width envelope closer in this bare grammar.
                return self._close(Span(lexeme.span.end, lexeme.span.end))
            return self._close(lexeme.span)
        events = self._body(lexeme)
        if (
            self.candidate is not None
            and c["kind"] == "raw_start"
            and c["json"] is not None
            and c["json"].depth == 0
            and not c["json"].in_string
        ):
            events += self._close(Span(lexeme.span.end, lexeme.span.end))
        return events

    def _body(self, lexeme) -> list[Event]:
        # Resource boundaries are source positions, never input-chunk edges.
        # Processing the owned body once also bounds header lookahead before
        # a large single chunk can commit a header rejected by token streaming.
        events = []
        for offset, char in enumerate(lexeme.text, lexeme.span.start):
            piece = Lexeme(Span(offset, offset + 1), char, None)
            if self.candidate is None:
                events.extend(self.consume(piece))
            else:
                events.extend(self._body_piece(piece))
        return _coalesce_args(events)

    def _body_piece(self, lexeme):
        c = self.candidate
        if c["literal"]:
            if "rejected_parts" in c:
                c["rejected_parts"].append(lexeme.text)
            scanner = c["json"] or c["value_json"]
            if scanner is None and not c["xml_body_started"] and lexeme.text.strip():
                c["xml_body_started"] = True
                if lexeme.text == "{":
                    scanner = c["value_json"] = JsonEnvelope()
                    scanner.track_values = False
            if scanner is not None:
                scanner.feed(lexeme.text, lexeme.span.start)
            c["cursor"] = lexeme.span.end
            cls = ReasoningDelta if c["origin"] == "reasoning" else ContentDelta
            return [cls(lexeme.span, lexeme.text)]
        c["parts"].append(lexeme.text)
        c["size"] += len(lexeme.text)
        if c["json"] is None and not c["prefix"].strip():
            first = lexeme.text.lstrip()
            if first.startswith("{"):
                c["json"] = JsonEnvelope()
        result = []
        if c["committed"] and c["json"] is None:
            if not c["xml_body_started"] and lexeme.text.strip():
                c["xml_body_started"] = True
                if lexeme.text == "{":
                    c["value_json"] = JsonEnvelope()
            if c["value_json"] is not None:
                c["value_json"].feed(lexeme.text, lexeme.span.start)
        scanner = c["json"]
        if scanner is not None:
            arg_chars = []
            arg_start = None
            control_start = None
            for offset, char in enumerate(lexeme.text, lexeme.span.start):
                scanner.feed(char, offset)
                if not c["committed"] and scanner.argument_start is not None:
                    if scanner.name not in self.context.authorized_tools:
                        return self._reject(
                            lexeme.span.end, "unauthorized_tool", keep_open=True
                        )
                    result.append(self._start(scanner.name, scanner.argument_start))
                if c["committed"]:
                    if scanner.argument_end is None or offset < scanner.argument_end:
                        if arg_start is None:
                            arg_start = offset
                        arg_chars.append(char)
                    elif control_start is None:
                        control_start = offset
            if arg_chars:
                text = "".join(arg_chars)
                result.append(
                    ToolArgs(Span(arg_start, arg_start + len(text)), c["index"], text)
                )
                c["cursor"] = arg_start + len(text)
            if control_start is not None:
                result.append(
                    Control(Span(control_start, lexeme.span.end), "tool_metadata")
                )
                c["cursor"] = lexeme.span.end
        elif not c["committed"]:
            # Only the bounded header is examined repeatedly; payloads are
            # stored as chunks and converted once at the enclosing close.
            c["prefix"] += lexeme.text
            prefix = c["prefix"]
            match = re.match(
                r'\s*<function=([^>]+)>|\s*<function name="([^"]+)">|\s*<atem:invoke name="([^"]+)">',
                prefix,
            )
            if match:
                name = (match[1] or match[2] or match[3]).strip()
                if name not in self.context.authorized_tools:
                    return self._reject(
                        lexeme.span.end, "unauthorized_tool", keep_open=True
                    )
                tail = prefix[match.end() :].lstrip()
                # In a private lane, a function-name example alone is not a
                # tool commitment. Require its parameter grammar or validate
                # the complete empty function at the enclosing close.
                if c["origin"] != "reasoning" or tail.startswith(
                    ("<parameter", "<atem:parameter", "{")
                ):
                    result.append(self._start(name, c["body_start"] + match.end()))
                    if tail == "{":
                        c["xml_body_started"] = True
                        c["value_json"] = JsonEnvelope()
                        c["value_json"].feed("{", lexeme.span.end - 1)
            elif c["kind"] == "bracket_start":
                match = re.fullmatch(r"\s*([^()\s]+)\(", prefix)
                if match:
                    name = match[1]
                    if name not in self.context.authorized_tools:
                        return self._reject(
                            lexeme.span.end, "unauthorized_tool", keep_open=True
                        )
                    result.append(self._start(name, c["body_start"] + match.end()))
            elif self.policy.format_name in {"glm47", "glm4"}:
                match = re.match(r"\s*([^<\s]+)(?:\s+|(?=<arg_key>))", prefix)
                if match:
                    name = match[1]
                    if name not in self.context.authorized_tools:
                        return self._reject(
                            lexeme.span.end, "unauthorized_tool", keep_open=True
                        )
                    result.append(self._start(name, c["body_start"] + match.end()))
        if (
            not c["committed"]
            and not (scanner is not None and scanner.argument_seen)
            and c["size"] > self.context.max_header_chars
        ):
            return self._reject(lexeme.span.end, "header_limit", keep_open=True)
        if (
            self.context.max_argument_chars is not None
            and c["size"] > self.context.max_argument_chars
        ):
            if c["committed"]:
                result.append(
                    ParseFailure(
                        Span(c["cursor"], lexeme.span.end), "argument_limit", c["index"]
                    )
                )
                self.candidate = None
                self.orphan = True
            else:
                return self._reject(lexeme.span.end, "argument_limit", keep_open=True)
        return result

    def _reject(self, end, code, *, keep_open=False):
        c = self.candidate
        prefix = (
            c["opener"]
            if c["kind"] in {"tool_start", "seed_start", "bracket_start"}
            else ""
        )
        text = prefix + "".join(c["parts"])
        cls = ReasoningDelta if c["origin"] == "reasoning" else ContentDelta
        self.candidate = c if keep_open else None
        if keep_open:
            if code == "unauthorized_tool" and self.policy.format_name in {
                "glm47",
                "glm4",
            }:
                c["rejected_parts"] = list(c["parts"])
            c["literal"] = True
            c["parts"] = []
            c["prefix"] = ""
            c["cursor"] = end
            if c["json"] is not None:
                c["json"].track_values = False
                c["json"].capture = []
                c["json"].role = None
        self.has_prose = bool(text.strip())
        return [cls(Span(c["start"], end), text), ParseFailure(Span(end, end), code)]

    def _rejection_diagnostic(self, candidate, offset):
        if self.policy.format_name not in {"glm47", "glm4"}:
            return []
        body = "".join(candidate.get("rejected_parts", candidate["parts"]))
        try:
            name, _ = decode_body(body, self.policy.format_name, self.policy.schemas)
        except (ValueError, TypeError, AttributeError, RecursionError):
            return []
        if name in self.context.authorized_tools:
            return []
        match = re.match(r"\s*[^<\s]+\s*(.*)", body, re.S)
        if match is None:
            return []
        return [
            ParseFailure(
                Span(offset, offset),
                "unauthorized_tool",
                tool_name=name,
                raw_arguments=match[1],
            )
        ]

    def _close(self, closer):
        c = self.candidate
        if c["literal"]:
            self.candidate = None
            return []
        body = "".join(c["parts"])
        result = []
        try:
            if any(
                scanner is not None and scanner.invalid
                for scanner in (c["json"], c["value_json"])
            ):
                raise ValueError("invalid JSON structure or nesting limit")
            if c["kind"] == "bracket_start":
                match = re.fullmatch(r"\s*([^()\s]+)\((.*)", body, re.S)
                if match is None:
                    raise ValueError("invalid bracket call")
                name, arguments = match[1], _json(match[2])
            else:
                name, arguments = decode_body(
                    body, self.policy.format_name, self.policy.schemas
                )
            if name not in self.context.authorized_tools:
                raise ValueError("unauthorized tool")
            if not isinstance(arguments, dict):
                raise ValueError("tool arguments must be an object")
            validate_arguments(arguments, self.policy.schemas.get(name, {}))
            arguments_text = json.dumps(arguments, ensure_ascii=False, allow_nan=False)
            if c["committed"] and name != c["name"]:
                raise ValueError("tool name changed after commitment")
        except (ValueError, TypeError, AttributeError, RecursionError):
            if c["committed"]:
                result.append(
                    ParseFailure(
                        Span(c["cursor"], closer.end), "invalid_tool", c["index"]
                    )
                )
                self.candidate = None
                return result
            c["parts"].append(
                "</tool_call>"
                if c["kind"] == "tool_start"
                else ")]"
                if c["kind"] == "bracket_start"
                else "</seed:tool_call>"
                if c["kind"] == "seed_start"
                else ""
            )
            diagnostic = self._rejection_diagnostic({**c, "parts": [body]}, closer.end)
            return self._reject(closer.end, "invalid_tool") + diagnostic
        if not c["committed"]:
            result.append(self._start(name, c["body_start"]))
        if c["json"] is None or c["json"].argument_start is None:
            result.append(
                ToolArgs(Span(c["cursor"], closer.start), c["index"], arguments_text)
            )
        result.append(ToolEnd(closer, c["index"]))
        self.candidate = None
        return result

    def metadata(self, calls, offset, *, channel=None, text=""):
        if channel is not None:
            self.channel = channel
        if text.strip():
            self.has_prose = True
        events = []
        span = Span(offset, offset)
        for call in calls:
            function = call.get("function", call)
            name = function.get("name")
            arguments = function.get("arguments")
            if name not in self.context.authorized_tools:
                index = self.next_index
                self.next_index += 1
                events.append(ParseFailure(span, "unauthorized_tool", index))
                continue
            if isinstance(arguments, dict):
                arguments = json.dumps(arguments, ensure_ascii=False, allow_nan=False)
            if not isinstance(arguments, str) or not isinstance(_json(arguments), dict):
                raise ValueError("invalid structured tool arguments")
            index = self.next_index
            self.next_index += 1
            try:
                validate_arguments(_json(arguments), self.policy.schemas.get(name, {}))
            except ValueError:
                events.append(ParseFailure(span, "invalid_tool", index))
                continue
            events += [
                ToolStart(span, index, name, call.get("id")),
                ToolArgs(span, index, arguments),
                ToolEnd(span, index),
            ]
            self.channel = "content"
            self.has_prose = False
        return events

    def boundary(self, offset):
        return self.finish(offset, "metadata_boundary")

    def finish(self, offset: int, reason: str) -> list[Event]:
        if self.candidate is None:
            return []
        c = self.candidate
        if c["literal"]:
            self.candidate = None
            return []
        if c["committed"]:
            body = "".join(c["parts"]).rstrip()
            complete_body = (
                (
                    c["json"] is not None
                    and c["json"].depth == 0
                    and not c["json"].in_string
                )
                or body.endswith("</function>")
                or (
                    self.policy.format_name in {"glm47", "glm4"}
                    and body.endswith("</arg_value>")
                )
            )
            if complete_body:
                # The argument body is complete; only its outer envelope was
                # interrupted. Conversion validates it without inventing bytes.
                return cast(
                    list[Event],
                    self._close(c.get("partial_closer", Span(offset, offset))),
                )
            self.candidate = None
            return [
                ParseFailure(Span(c["cursor"], offset), "incomplete_tool", c["index"])
            ]
        return cast(list[Event], self._reject(offset, "incomplete_candidate"))
