# SPDX-License-Identifier: Apache-2.0
"""Streaming recognition contracts for the parsers' declared wire formats."""

from typing import Protocol, cast

from ..streaming_markers import StreamingMarkerBuffer
from .abstract_tool_parser import ToolParser


class ContentFilter(Protocol):
    def process(self, delta: str) -> str: ...
    def flush(self) -> str: ...


WIRE_OPENERS = {
    "tool_call_json": ("<tool_call>",),
    "tool_call_xml_body": ("<tool_call>",),
    "function_bare": ("<function=",),
    "function_xml_named": ("<function>",),
    "raw_json": (),
    "calling_tool_text": ("[Calling tool",),
    "pythonic_bracket": (),
    "gemma4_native": ("<|tool_call>", "call:"),
    "harmony_commentary": ("<|start|>", "<|channel|>"),
    "mistral_tool_calls": ("[TOOL_CALLS]",),
    "llama_python_tag": ("<|python_tag|>",),
    "glm_named_tool_call": ("<tool_call>",),
    "functionary_native": ("<|from|>", "<|recipient|>"),
    "granite_native": ("<|tool_call|>", "<tool_call>"),
    "kimi_native": (
        "<|tool_calls_section_begin|>",
        "<|tool_call_section_begin|>",
        "<|tool_call_begin|>",
    ),
    "minimax_native": ("<minimax:tool_call>", "<tool_calls>"),
    "seed_oss_native": ("<seed:tool_call>",),
    "deepseek_native": ("<｜tool▁calls▁begin｜>", "<｜tool▁call▁begin｜>"),
    "deepseek_v31_native": ("<｜tool▁calls▁begin｜>", "<｜tool▁call▁begin｜>"),
    "deepseek_v4_dsml": ("<｜DSML｜tool_calls>", "<｜DSML｜r:tool_calls>"),
    "qwen3_coder_xml_named": ("<tool_call>", "<function="),
    "ui_tars_action": ("Action:",),
    "hy3_native": ("<tool_call:", "<tool_call>"),
    "minicpm_native": ("<function ",),
    "muse_atem": ("<atem:function_calls>",),
    "k2_ifm": ("<ifm|tool_calls>",),
}


# Closed envelopes have an explicit content boundary. Using it preserves prose
# following a call even when a parser's incremental API emits only tool deltas.
WIRE_ENVELOPES = {
    "gemma4_native": (("<|tool_call>", "<tool_call|>"),),
    "muse_atem": (("<atem:function_calls>", "</atem:function_calls>"),),
    "minicpm_native": (("<function ", "</function>"),),
    "seed_oss_native": (("<seed:tool_call>", "</seed:tool_call>"),),
    "kimi_native": (
        ("<|tool_calls_section_begin|>", "<|tool_calls_section_end|>"),
        ("<|tool_call_section_begin|>", "<|tool_call_section_end|>"),
        ("<|tool_call_begin|>", "<|tool_call_end|>"),
    ),
    "deepseek_native": (("<｜tool▁calls▁begin｜>", "<｜tool▁calls▁end｜>"),),
    "deepseek_v31_native": (("<｜tool▁calls▁begin｜>", "<｜tool▁calls▁end｜>"),),
    "k2_ifm": (("<ifm|tool_calls>", "</ifm|tool_calls>"),),
    "minimax_native": (("<minimax:tool_call>", "</minimax:tool_call>"),),
}


def tool_openers(parser) -> tuple[str, ...]:
    """Select only the wire formats owned by this request's parser."""
    formats = getattr(parser, "EXPECTED_WIRE_FORMATS", ())
    if formats == ():
        formats = tuple(WIRE_OPENERS)
    return tuple(
        dict.fromkeys(opener for fmt in formats for opener in WIRE_OPENERS.get(fmt, ()))
    )


class StreamingToolContentFilter:
    """Use the configured parser's content decisions on every API surface.

    Structured calls are extracted separately from the untouched generation.
    The prefix buffer prevents old parsers from releasing an incomplete opener.
    """

    def __init__(self, parser: ToolParser, request):
        self.parser = parser
        self.request = request
        self.buffer = StreamingMarkerBuffer(tool_openers(parser))
        self.text = ""
        self._flushed: str | None = None
        self._tools_seen = False
        self._preserve_post_tool_content = False

    def process(self, delta: str) -> str:
        delta = self.buffer.process(delta)
        if not delta:
            return ""
        previous = self.text
        self.text += delta
        result = self.parser.extract_tool_calls_streaming(
            previous, self.text, delta, request=self.request
        )
        if result and result.get("tool_calls"):
            self._tools_seen = True
            self._preserve_post_tool_content = bool(
                result.get("preserve_post_tool_content")
            )
        return cast(str, (result or {}).get("content") or "")

    def flush(self) -> str:
        tail = self.buffer.flush()
        if self._flushed == self.text:
            return "" if self._tools_seen else tail
        self._flushed = self.text
        if self._tools_seen:
            # Match Chat's parser-owned post-call content contract: a suffix
            # may still be tool wire, even after an earlier call was emitted.
            if self._preserve_post_tool_content:
                return self.parser.flush_held_content(self.text + tail)
            return ""
        return self.parser.flush_held_content(self.text) + tail


class StreamingOutputFilter:
    """Recognize native tool wire before applying stateful content cleanup."""

    def __init__(
        self,
        tool_filter: ContentFilter,
        *,
        enabled: bool,
        preserve_think_close: bool = False,
    ):
        from ..api.utils import StreamingContentSanitizer

        self.tool_filter = tool_filter
        self.special = StreamingMarkerBuffer(
            ("<s>", "</s>", "<pad>", "[PAD]", "[SEP]", "[CLS]", "[e~[", "]~!b[")
        )
        self.sanitizer = StreamingContentSanitizer(
            strip_tool_markup=enabled, preserve_think_close=preserve_think_close
        )

    def process(self, delta: str) -> str:
        from ..api.utils import strip_special_tokens

        text = self.special.process(self.tool_filter.process(delta))
        return self.sanitizer.process(strip_special_tokens(text))

    def flush(self) -> str:
        from ..api.utils import strip_special_tokens

        text = self.special.process(self.tool_filter.flush()) + self.special.flush()
        return (
            self.sanitizer.process(strip_special_tokens(text)) + self.sanitizer.flush()
        )


def streaming_output_filter(
    cfg, request, enabled, fallback, *, preserve_think_close=False
):
    """Construct one request-local content lane using the selected parser."""
    from ..api.utils import StreamingToolCallFilter
    from . import ToolParserManager

    # The GLM filter additionally suppresses malformed orphan argument wire.
    if (
        enabled
        and cfg.tool_call_parser
        and cfg.tool_call_parser not in {"glm47", "glm4"}
    ):
        parser_class = ToolParserManager.get_tool_parser(cfg.tool_call_parser)
        if parser_class is not None:
            parser = parser_class(getattr(cfg.engine, "tokenizer", None))
            covered = {
                "tool_call_json",
                "tool_call_xml_body",
                "function_bare",
                "function_xml_named",
                "calling_tool_text",
                "raw_json",
                "qwen3_coder_xml_named",
                "glm_named_tool_call",
                "deepseek_v4_dsml",
            }
            envelopes = [
                pair
                for fmt in parser.EXPECTED_WIRE_FORMATS
                for pair in WIRE_ENVELOPES.get(fmt, ())
            ]
            if "hy3_native" in parser.EXPECTED_WIRE_FORMATS:
                envelopes.append(
                    (parser.tool_call_start_token, parser.tool_call_end_token)
                )
            if envelopes:
                fallback = StreamingToolCallFilter(
                    extra_tags=envelopes,
                    is_complete=lambda text: (
                        parser.extract_tool_calls(text, request=request).tools_called
                    ),
                )
            elif not set(parser.EXPECTED_WIRE_FORMATS).issubset(covered):
                fallback = StreamingToolContentFilter(parser, request)
    return StreamingOutputFilter(
        fallback, enabled=enabled, preserve_think_close=preserve_think_close
    )
