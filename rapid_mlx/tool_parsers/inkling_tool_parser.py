# SPDX-License-Identifier: Apache-2.0
"""Inkling JSON-block adapter; the wrapper's args become OpenAI arguments."""

import json
import uuid

from ..reasoning.inkling_parser import END, SAMPLING_END, TOOL, InklingBlocks
from .abstract_tool_parser import (
    ExtractedToolCallInformation,
    ToolParser,
    ToolParserManager,
)


def _members(raw):
    """Read complete wrapper members and a prefix-stable args object span."""
    decoder = json.JSONDecoder()
    pos = 0
    name, args = None, ""
    try:
        pos = len(raw) - len(raw.lstrip())
        if raw[pos : pos + 1] != "{":
            return name, args
        pos += 1
        while pos < len(raw):
            while pos < len(raw) and raw[pos] in " \t\r\n,":
                pos += 1
            key, pos = decoder.raw_decode(raw, pos)
            while pos < len(raw) and raw[pos].isspace():
                pos += 1
            if raw[pos : pos + 1] != ":":
                break
            pos += 1
            while pos < len(raw) and raw[pos].isspace():
                pos += 1
            start = pos
            if key == "args" and raw[start : start + 1] == "{":
                depth, quoted, escaped = 0, False, False
                end = start
                for end in range(start, len(raw)):
                    char = raw[end]
                    if escaped:
                        escaped = False
                    elif quoted and char == "\\":
                        escaped = True
                    elif char == '"':
                        quoted = not quoted
                    elif not quoted:
                        depth += (char == "{") - (char == "}")
                        if depth == 0:
                            break
                args = raw[start : end + 1]
            value, pos = decoder.raw_decode(raw, pos)
            if key == "name" and isinstance(value, str):
                name = value
    except ValueError:
        pass
    return name, args


class InklingToolParser(ToolParser):
    SUPPORTS_NATIVE_TOOL_FORMAT = True
    EXPECTED_WIRE_FORMATS = ("inkling_native",)

    def __init__(self, tokenizer=None):
        super().__init__(tokenizer)
        self.reset()

    def reset(self):
        super().reset()
        self.blocks = InklingBlocks()
        self.payload = None
        self.index = 0
        self.sent = ""
        self.call_id = None
        self.streaming = False
        self.pending_deltas = []

    @staticmethod
    def _call(raw):
        try:
            parsed = json.loads(raw)
            if not isinstance(parsed, dict) or not isinstance(parsed.get("name"), str):
                return None
            if not parsed["name"]:
                return None
            args = next(
                (parsed[k] for k in ("args", "arguments", "parameters") if k in parsed),
                {},
            )
            return {
                "id": "call_" + uuid.uuid4().hex[:24],
                "name": parsed["name"],
                "arguments": args
                if isinstance(args, str)
                else json.dumps(args, ensure_ascii=False),
            }
        except (ValueError, TypeError):
            return None

    def _consume(self, events, final=False):
        content, calls = [], []

        def close():
            if self.payload is not None:
                call = self._call(self.payload)
                if call:
                    calls.append(call)
                    if self.streaming:
                        self._stream_call(call)
                elif self.call_id is None:
                    content.append(self.payload)
                elif self.streaming:
                    # A malformed close cannot retract arguments already sent.
                    self.index += 1
                self.payload = None
                self.sent = ""
                self.call_id = None

        for kind, value in events:
            if kind == "marker":
                if value == TOOL:
                    close()
                    self.payload = ""
                elif value in (END, SAMPLING_END):
                    close()
            elif kind == "tool" and self.payload is not None:
                self.payload += value
                if self.streaming:
                    self._stream_call()
            elif kind == "content":
                content.append(value)
        if final:
            close()
        return "".join(content), calls

    def extract_tool_calls(self, model_output, request=None):
        parser = type(self)(self.model_tokenizer)
        content, calls = parser._consume(
            parser.blocks.feed(model_output, final=True), final=True
        )
        return ExtractedToolCallInformation(bool(calls), calls, content or None)

    def _stream_call(self, complete=None):
        name, args = _members(self.payload)
        if complete:
            name = complete["name"]
            args = args or complete["arguments"]
        if not name or (complete is None and not args):
            return
        function = {}
        delta = {"index": self.index, "function": function}
        if self.call_id is None:
            self.call_id = "call_" + uuid.uuid4().hex[:24]
            delta.update(id=self.call_id, type="function")
            function["name"] = name
        if args.startswith(self.sent) and args != self.sent:
            function["arguments"] = args[len(self.sent) :]
            self.sent = args
        if function:
            self.pending_deltas.append(delta)
        if complete:
            self.index += 1

    def _wire(self, content):
        deltas, self.pending_deltas = self.pending_deltas, []
        result = {}
        if content:
            result["content"] = content
        if deltas:
            # Merge fragments from a single input chunk by call index.
            merged = {}
            for delta in deltas:
                index = delta["index"]
                if index not in merged:
                    merged[index] = delta
                    continue
                function = merged[index]["function"]
                function["arguments"] = function.get("arguments", "") + delta[
                    "function"
                ].get("arguments", "")
            result["tool_calls"] = list(merged.values())
            result["preserve_post_tool_content"] = True
        return result or None

    def extract_tool_calls_streaming(
        self,
        previous_text,
        current_text,
        delta_text,
        previous_token_ids=None,
        current_token_ids=None,
        delta_token_ids=None,
        request=None,
    ):
        self.streaming = True
        content, _ = self._consume(self.blocks.feed(delta_text))
        return self._wire(content)

    def finalize_legacy_raw_stream(self, full_text, request=None):
        content, _ = self._consume(self.blocks.feed("", final=True), final=True)
        return self._wire(content)


ToolParserManager.register_module("inkling", InklingToolParser)
