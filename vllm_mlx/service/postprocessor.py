# SPDX-License-Identifier: Apache-2.0
"""Streaming post-processor — unified reasoning + tool call + sanitization pipeline.

Replaces 500+ lines of duplicated logic across stream_chat_completion,
_stream_anthropic_messages, and stream_completion. NOT a filter chain —
one cohesive orchestrator, because reasoning/tool/sanitize are tightly coupled.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from ..api.tool_calling import parse_tool_calls
from ..api.utils import sanitize_output, strip_special_tokens
from ..domain.events import StreamEvent

if TYPE_CHECKING:
    from ..config.server_config import ServerConfig
    from ..engine.base import GenerationOutput

logger = logging.getLogger(__name__)


def _find_json_start(text: str) -> int:
    """Find the first `{` or `[` that is NOT inside `<think>...</think>` tags.

    Returns the index in ``text``, or -1 if no JSON delimiter found outside
    think blocks.  Handles unclosed `<think>` (still accumulating) by
    treating everything after it as inside the block.
    """
    in_think = False
    i = 0
    while i < len(text):
        # Check for <think> open tag
        if text[i : i + 7] == "<think>":
            in_think = True
            i += 7
            continue
        # Check for </think> close tag
        if text[i : i + 8] == "</think>":
            in_think = False
            i += 8
            continue
        # Outside think block — check for JSON delimiter
        if not in_think and text[i] in ("{", "["):
            return i
        i += 1
    return -1


def _has_partial_calling_tool_marker(text: str) -> bool:
    """Return True when a stream tail may become `Calling tool:`."""
    marker = "Calling tool:"
    tail = text.rstrip()
    if tail.endswith("[") and _starts_current_line(tail, len(tail) - 1):
        return True
    for i in range(1, len(marker)):
        partial = marker[:i]
        if tail.endswith(partial):
            start = len(tail) - len(partial)
            if _starts_current_line(tail, start):
                return True
        if tail.endswith(f"[{partial}"):
            start = len(tail) - len(partial) - 1
            if _starts_current_line(tail, start):
                return True
    return False


def _starts_current_line(text: str, start: int) -> bool:
    """Return True when start is preceded only by whitespace on its line."""
    line_start = max(text.rfind("\n", 0, start), text.rfind("\r", 0, start)) + 1
    return text[line_start:start].strip() == ""


def _find_trailing_calling_tool_prefix(text: str) -> int | None:
    marker = "Calling tool:"
    tail_end = len(text.rstrip())
    tail = text[:tail_end]

    if tail.endswith("["):
        start = len(tail) - 1
        if _starts_current_line(tail, start):
            return start

    for i in range(1, len(marker)):
        partial = marker[:i]
        if tail.endswith(partial):
            start = len(tail) - len(partial)
            if _starts_current_line(tail, start):
                return start
        if tail.endswith(f"[{partial}"):
            start = len(tail) - len(partial) - 1
            if _starts_current_line(tail, start):
                return start
    return None


def _strip_trailing_calling_tool_prefix(text: str) -> str | None:
    """Remove a trailing bracket/partial marker that may start a tool call."""
    if not text:
        return None

    start = _find_trailing_calling_tool_prefix(text)
    if start is not None:
        return text[:start].rstrip()
    return None


class StreamingPostProcessor:
    """Processes streaming engine output into StreamEvents.

    Handles:
    1. Channel routing (OutputRouter models like Gemma 4)
    2. Reasoning extraction (text-based parsers for Qwen3, DeepSeek, MiniMax)
    3. Tool call streaming detection (incremental parser)
    4. Output sanitization (strip special tokens, markup)

    Usage::

        processor = StreamingPostProcessor(cfg, request)
        processor.reset()
        async for output in engine.stream_chat(...):
            for event in processor.process_chunk(output):
                yield format_for_my_api_spec(event)
        for event in processor.finalize():
            yield format_for_my_api_spec(event)
    """

    def __init__(
        self,
        cfg: ServerConfig,
        tools_requested: bool = False,
        enable_thinking: bool | None = None,
        json_mode: bool = False,
        request_dict: dict | None = None,
    ):
        self.cfg = cfg
        self.tools_requested = tools_requested
        self.json_mode = json_mode
        self.request_dict = request_dict

        # Per-request parser instances — each streaming request gets its
        # own parser to avoid state corruption under concurrent
        # BatchedEngine requests.
        #
        # Production path: reasoning_parser_name / tool_call_parser are set
        # at startup → _create_*() builds a fresh instance per request.
        #
        # Legacy/test path: cfg.reasoning_parser / cfg.tool_parser_instance
        # may be pre-built (mocks in tests, or singleton from server.py).
        # When reasoning_parser_name is set, always create fresh.
        if cfg.reasoning_parser_name:
            self.reasoning_parser = self._create_reasoning_parser(cfg)
        else:
            self.reasoning_parser = cfg.reasoning_parser  # None or injected mock

        if cfg.tool_call_parser:
            self.tool_parser = self._create_tool_parser(cfg, tools_requested)
        elif cfg.tool_parser_instance:
            self.tool_parser = cfg.tool_parser_instance  # injected mock
        else:
            self.tool_parser = self._create_tool_parser(cfg, tools_requested)

        # State
        self.accumulated_text = ""
        self.tool_accumulated_text = ""
        self.tool_calls_detected = False
        self.tool_markup_possible = False

        # Nemotron thinking prefix
        self._is_thinking_model = False
        self._think_prefix_sent = False

        # JSON mode: suppress thinking preamble before JSON content (#46).
        # When json_mode=True and no reasoning parser, buffer content until
        # the first JSON delimiter ({ or [) is seen, then emit from there.
        self._json_preamble_stripped = False
        self._json_preamble_buffer = ""

    def _tool_call_has_required_args(self, name: str | None, arguments) -> bool:
        if not name or not isinstance(self.request_dict, dict):
            return True
        tools = self.request_dict.get("tools")
        if not isinstance(tools, list):
            return True

        required = []
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            function = tool.get("function")
            if not isinstance(function, dict) or function.get("name") != name:
                continue
            parameters = function.get("parameters")
            if isinstance(parameters, dict):
                required = parameters.get("required") or []
            break
        if not required:
            return True

        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except (TypeError, ValueError):
                return False
        if not isinstance(arguments, dict):
            return False
        return all(key in arguments for key in required)

    def _tool_calls_to_stream_chunks(self, tool_calls) -> list[dict]:
        chunks = []
        for i, tc in enumerate(tool_calls):
            if hasattr(tc, "function"):
                call_id = tc.id
                name = tc.function.name
                arguments = tc.function.arguments
            elif "function" in tc:
                call_id = tc.get("id")
                name = tc["function"]["name"]
                arguments = tc["function"]["arguments"]
            else:
                call_id = tc.get("id")
                name = tc["name"]
                arguments = tc["arguments"]

            if not self._tool_call_has_required_args(name, arguments):
                logger.debug(
                    "Dropping malformed tool call missing required arguments: %s",
                    name,
                )
                continue

            chunks.append(
                {
                    "index": len(chunks),
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": arguments,
                    },
                }
            )
        return chunks

    @staticmethod
    def _create_reasoning_parser(cfg: ServerConfig):
        """Create a per-request reasoning parser instance."""
        if not cfg.reasoning_parser_name:
            return None
        try:
            from ..reasoning import get_parser

            parser_cls = get_parser(cfg.reasoning_parser_name)
            return parser_cls()
        except Exception as e:
            logger.warning(f"Failed to create reasoning parser: {e}")
            return None

    @staticmethod
    def _create_tool_parser(cfg: ServerConfig, tools_requested: bool):
        """Create a per-request tool parser instance."""
        from ..tool_parsers import ToolParserManager

        tokenizer = None
        if cfg.engine is not None and hasattr(cfg.engine, "_tokenizer"):
            tokenizer = cfg.engine._tokenizer

        # Primary: explicit tool parser configured
        if cfg.enable_auto_tool_choice and cfg.tool_call_parser:
            try:
                parser_cls = ToolParserManager.get_tool_parser(cfg.tool_call_parser)
                return parser_cls(tokenizer)
            except Exception as e:
                logger.warning(f"Failed to create tool parser for streaming: {e}")

        # Fallback: auto-infer from reasoning parser
        if tools_requested and cfg.reasoning_parser_name:
            _PARSER_MAP = {"minimax": "minimax"}
            inferred = _PARSER_MAP.get(cfg.reasoning_parser_name)
            if inferred:
                try:
                    parser_cls = ToolParserManager.get_tool_parser(inferred)
                    return parser_cls(tokenizer)
                except Exception as e:
                    logger.debug(f"Auto-infer tool parser for streaming failed: {e}")

        return None

    def set_thinking_model(self, model_name: str):
        """Enable Nemotron-style thinking prefix injection."""
        self._is_thinking_model = (
            "nemotron" in model_name.lower() and not self.reasoning_parser
        )

    def reset(self):
        """Reset all parser states for a new stream.

        Safe for concurrent BatchedEngine requests — each PostProcessor
        instance holds its own parser instances (created in __init__).
        """
        self.accumulated_text = ""
        self.tool_accumulated_text = ""
        self.tool_calls_detected = False
        self.tool_markup_possible = False
        self._think_prefix_sent = False
        self._json_preamble_stripped = False
        self._json_preamble_buffer = ""

        if self.reasoning_parser:
            self.reasoning_parser.reset_state()
        if self.tool_parser:
            self.tool_parser.reset()

    def process_chunk(self, output: GenerationOutput) -> list[StreamEvent]:
        """Process a single engine output chunk.

        Returns a list of StreamEvents (may be empty if content is suppressed).
        """
        delta_text = output.new_text
        if not delta_text:
            # Handle finish-only chunks
            if output.finished:
                return [self._make_finish_event(output)]
            return []

        # Step 1: Separate content from reasoning
        if output.channel:
            return self._process_channel_routed(delta_text, output)
        elif self.reasoning_parser:
            return self._process_with_reasoning(delta_text, output)
        else:
            return self._process_standard(delta_text, output)

    def _process_channel_routed(
        self, delta_text: str, output: GenerationOutput
    ) -> list[StreamEvent]:
        """Handle OutputRouter models (Gemma 4 etc.) with token-level routing."""
        if output.channel == "reasoning":
            content, reasoning = None, delta_text
        elif output.channel == "tool_call":
            content, reasoning = delta_text, None
        else:
            content, reasoning = delta_text, None

        # Tool call detection on content
        if self._should_detect_tool_calls(content):
            result = self._detect_tool_calls(content)
            if result is None:
                return []  # suppressed (inside tool markup)
            if result.get("tool_calls"):
                return [
                    StreamEvent(
                        type="tool_call",
                        tool_calls=result["tool_calls"],
                        finish_reason="tool_calls" if output.finished else None,
                        tool_calls_detected=True,
                    )
                ]
            content = result.get("content", "")

        if self.tool_calls_detected:
            if output.finished:
                return [
                    StreamEvent(
                        type="finish",
                        finish_reason="tool_calls",
                        tool_calls_detected=True,
                    )
                ]
            return []

        # Sanitize
        if content:
            content = strip_special_tokens(content)
        if reasoning:
            reasoning = strip_special_tokens(reasoning)

        finish_reason = self._compute_finish_reason(output)
        if not content and not reasoning and not finish_reason:
            return []

        if content:
            content = sanitize_output(content)
            if not content:
                content = None

        # When finish_reason is set, emit ONE finish event with content/reasoning
        # merged in to avoid double-emission.
        if finish_reason:
            return [
                StreamEvent(
                    type="finish",
                    finish_reason=finish_reason,
                    content=content,
                    reasoning=reasoning,
                    tool_calls_detected=self.tool_calls_detected,
                )
            ]
        events = []
        if content:
            events.append(StreamEvent(type="content", content=content))
        if reasoning:
            events.append(StreamEvent(type="reasoning", reasoning=reasoning))
        return events

    def _should_detect_tool_calls(self, content: str | None) -> bool:
        if not content:
            return False
        if self.tool_parser:
            return True
        return (
            "Calling tool:" in content or "Calling tool:" in self.tool_accumulated_text
            or _has_partial_calling_tool_marker(content)
            or _has_partial_calling_tool_marker(self.tool_accumulated_text + content)
        )

    def _process_with_reasoning(
        self, delta_text: str, output: GenerationOutput
    ) -> list[StreamEvent]:
        """Handle models with text-based reasoning parsers."""
        previous_text = self.accumulated_text
        self.accumulated_text += delta_text
        delta_msg = self.reasoning_parser.extract_reasoning_streaming(
            previous_text, self.accumulated_text, delta_text
        )

        if delta_msg is None:
            # Skip (e.g., <think> token itself)
            if output.finished:
                return [self._make_finish_event(output)]
            return []

        content = delta_msg.content
        reasoning = delta_msg.reasoning

        # MiniMax redirect: tool calls wrapped in <think> blocks
        if self.tool_parser and reasoning:
            _check = self.tool_accumulated_text + reasoning
            if (
                "<minimax:tool_call>" in _check
                or "<tool_call>" in _check
                or '<invoke name="' in _check
            ):
                content = (content or "") + reasoning
                reasoning = None

        # Tool call detection
        if self._should_detect_tool_calls(content):
            result = self._detect_tool_calls(content)
            if result is None:
                return []
            if result.get("tool_calls"):
                return [
                    StreamEvent(
                        type="tool_call",
                        tool_calls=result["tool_calls"],
                        finish_reason="tool_calls" if output.finished else None,
                        tool_calls_detected=True,
                    )
                ]
            content = result.get("content", "")

        if self.tool_calls_detected:
            if output.finished:
                return [
                    StreamEvent(
                        type="finish",
                        finish_reason="tool_calls",
                        tool_calls_detected=True,
                    )
                ]
            return []

        # Sanitize
        if content:
            content = strip_special_tokens(content)
        if reasoning:
            reasoning = strip_special_tokens(reasoning)

        finish_reason = self._compute_finish_reason(output)
        if not content and not reasoning and not finish_reason:
            return []

        if content:
            content = sanitize_output(content)
            if not content:
                content = None

        if finish_reason:
            return [
                StreamEvent(
                    type="finish",
                    finish_reason=finish_reason,
                    content=content,
                    reasoning=reasoning,
                    tool_calls_detected=self.tool_calls_detected,
                )
            ]
        events = []
        if content:
            events.append(StreamEvent(type="content", content=content))
        if reasoning:
            events.append(StreamEvent(type="reasoning", reasoning=reasoning))
        return events

    def _process_standard(
        self, delta_text: str, output: GenerationOutput
    ) -> list[StreamEvent]:
        """Handle standard models (no reasoning parser, no channel router)."""
        content = strip_special_tokens(delta_text)

        # JSON mode preamble stripping (#46): when response_format is set and
        # no reasoning parser is active, the model may emit a thinking preamble
        # (e.g. "Let me think...\n{json}") before the actual JSON. Suppress
        # everything before the first JSON delimiter.
        if (
            self.json_mode
            and not self.reasoning_parser
            and not self._json_preamble_stripped
        ):
            if content:
                self._json_preamble_buffer += content
                json_start = _find_json_start(self._json_preamble_buffer)
                if json_start >= 0:
                    self._json_preamble_stripped = True
                    content = self._json_preamble_buffer[json_start:]
                else:
                    return []

        # Nemotron thinking prefix
        if self._is_thinking_model and not self._think_prefix_sent and content:
            content = "<think>" + content
            self._think_prefix_sent = True

        # Tool call detection
        if self._should_detect_tool_calls(delta_text):
            result = self._detect_tool_calls(delta_text)
            if result is None:
                return []
            if result.get("tool_calls"):
                return [
                    StreamEvent(
                        type="tool_call",
                        tool_calls=result["tool_calls"],
                        finish_reason="tool_calls" if output.finished else None,
                        tool_calls_detected=True,
                    )
                ]
            content = strip_special_tokens(result.get("content", ""))

        if self.tool_calls_detected:
            if output.finished:
                return [
                    StreamEvent(
                        type="finish",
                        finish_reason="tool_calls",
                        tool_calls_detected=True,
                    )
                ]
            return []

        # Filter empty
        if content is not None and content == "":
            content = None

        finish_reason = self._compute_finish_reason(output)

        if not content and not finish_reason:
            return []

        if content:
            content = sanitize_output(content)
            if not content:
                content = None

        # When finish_reason is set, emit ONE finish event with content merged in.
        # Never emit separate content + finish events — that would cause
        # double-emission of the same content and duplicate logprobs.
        if finish_reason:
            return [
                StreamEvent(
                    type="finish",
                    finish_reason=finish_reason,
                    content=content,
                    tool_calls_detected=self.tool_calls_detected,
                )
            ]
        if content:
            return [StreamEvent(type="content", content=content)]
        return []

    def finalize(self) -> list[StreamEvent]:
        """Finalize stream — flush remaining tool calls, emit corrections.

        Call after the engine stream ends.
        """
        events = []

        # Fallback tool call detection: parser accumulated text but never
        # emitted (e.g., closing tag never arrived).
        _fallback_text = self.tool_accumulated_text or self.accumulated_text
        if (
            self.tool_parser
            and _fallback_text
            and not self.tool_calls_detected
            and self.tool_parser.has_pending_tool_call(_fallback_text)
        ):
            result = self.tool_parser.extract_tool_calls(
                _fallback_text, self.request_dict
            )
            if result.tools_called:
                tc_list = [
                    {
                        "index": i,
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": tc["arguments"],
                        },
                    }
                    for i, tc in enumerate(result.tool_calls)
                ]
                events.append(
                    StreamEvent(
                        type="tool_call",
                        tool_calls=tc_list,
                        finish_reason="tool_calls",
                        tool_calls_detected=True,
                    )
                )
                self.tool_calls_detected = True

        if "Calling tool:" in _fallback_text and not self.tool_calls_detected:
            _, tool_calls = parse_tool_calls(_fallback_text, self.request_dict)
            if tool_calls:
                chunks = self._tool_calls_to_stream_chunks(tool_calls)
                if not chunks:
                    return events
                events.append(
                    StreamEvent(
                        type="tool_call",
                        tool_calls=chunks,
                        finish_reason="tool_calls",
                        tool_calls_detected=True,
                    )
                )
                self.tool_calls_detected = True

        return events

    def _detect_tool_calls(self, content: str) -> dict | None:
        """Run incremental tool call detection.

        Returns None if content is suppressed (inside tool markup).
        Returns {"tool_calls": [...]} if tool calls detected.
        Returns {"content": "..."} for normal content pass-through.
        """
        if (
            not self.tool_markup_possible
            and "<" not in content
            and "[" not in content
            and "Calling tool:" not in content
        ):
            self.tool_accumulated_text += content
            return {"content": content}

        if not self.tool_markup_possible:
            self.tool_markup_possible = True

        tool_previous = self.tool_accumulated_text
        self.tool_accumulated_text += content
        if self.tool_parser:
            tool_result = self.tool_parser.extract_tool_calls_streaming(
                tool_previous,
                self.tool_accumulated_text,
                content,
                request=self.request_dict,
            )
        else:
            tool_result = {"content": content}

        if tool_result is None:
            if "Calling tool:" in self.tool_accumulated_text:
                _, tool_calls = parse_tool_calls(
                    self.tool_accumulated_text, self.request_dict
                )
                if tool_calls:
                    chunks = self._tool_calls_to_stream_chunks(tool_calls)
                    if not chunks:
                        return None
                    self.tool_calls_detected = True
                    return {"tool_calls": chunks}
            return None  # inside tool markup

        if "tool_calls" in tool_result:
            self.tool_calls_detected = True
            return tool_result

        if "Calling tool:" in self.tool_accumulated_text:
            _, tool_calls = parse_tool_calls(
                self.tool_accumulated_text, self.request_dict
            )
            if tool_calls:
                chunks = self._tool_calls_to_stream_chunks(tool_calls)
                if not chunks:
                    return None
                self.tool_calls_detected = True
                return {"tool_calls": chunks}
            return None

        if _has_partial_calling_tool_marker(self.tool_accumulated_text):
            content = tool_result.get("content", "")
            stripped = _strip_trailing_calling_tool_prefix(content)
            if stripped:
                return {"content": stripped}
            return None

        return {"content": tool_result.get("content", "")}

    def _compute_finish_reason(self, output: GenerationOutput) -> str | None:
        if not output.finished:
            return None
        if self.tool_calls_detected:
            return "tool_calls"
        return output.finish_reason

    def _make_finish_event(self, output: GenerationOutput) -> StreamEvent:
        return StreamEvent(
            type="finish",
            finish_reason=self._compute_finish_reason(output),
            tool_calls_detected=self.tool_calls_detected,
        )
