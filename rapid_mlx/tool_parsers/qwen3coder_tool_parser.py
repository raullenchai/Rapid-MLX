# SPDX-License-Identifier: Apache-2.0
"""
Qwen3-Coder XML tool call parser for rapid-mlx.

Ported from vLLM upstream (vllm/tool_parsers/qwen3coder_tool_parser.py).

Format:
  <tool_call>                           <- optional wrapper (framing only)
  <function=NAME>                       <- REQUIRED, defines the tool call
  <parameter=KEY>VALUE</parameter>
  </function>                           <- REQUIRED
  </tool_call>                          <- optional wrapper (framing only)

The ``<tool_call>...</tool_call>`` wrapper is OPTIONAL framing. What
structurally defines a tool call is the ``<function=NAME>...</function>``
XML block — so the streaming state machine anchors on ``<function=``
throughout and treats the wrapper as a prefix to strip from content
(issue #978). Anchoring on the wrapper would leak whole tool calls as
raw content when a fine-tune's tokenizer omits the wrapper as a special
token but the model still emits well-formed ``<function=...>`` bodies
(observed with ``Shiftedx/qwopus3.6-35b-a3b-coder-mxfp4-mlx``).

Similar to Seed-OSS but without the seed: namespace prefix.
"""

import ast
import json
import logging
import re
import uuid
from collections.abc import Sequence
from typing import Any, NamedTuple

from ..api.tool_calling import _decode_json_like, _schema_type
from ..tool_call_scan import split_marked_parameters, trim_wrapping_newlines
from .abstract_tool_parser import (
    ExtractedToolCallInformation,
    ToolParser,
    ToolParserManager,
)

logger = logging.getLogger(__name__)

# A JSON-style integer literal: optional sign, ASCII digits, no fraction and
# no exponent. Under a ``number`` schema such a literal is an integer (issue
# #4037); anything else keeps the historical ``float()`` conversion.
_INTEGER_LITERAL_RE = re.compile(r"[+-]?[0-9]+")

# Longest call block to an undeclared tool that is dropped (#4038). Streaming
# must hold the block back until it closes; a longer one is released as
# content (origin/main behaviour) in both paths, well before the
# post-processor's 64 KB suppression budget would release it on its own.
_UNDECLARED_BLOCK_MAX_CHARS = 32768


class _Continue(NamedTuple):
    """Content settled so far plus the rest of the delta still to parse.

    Returned instead of recursing, so one delta holding many dropped blocks
    cannot exhaust the stack (#4038).
    """

    content: str
    tail: str


_StepResult = dict[str, Any] | _Continue | None


_MARKDOWN_FENCE_RE = re.compile(r"[ ]{0,3}(`{3,}|~{3,})")


class _MarkdownCodeTracker:
    """Whether a position lies inside Markdown code: an open fenced block
    (``` or ~~~), or an inline code span opened earlier on the same line.

    Decided from the text before the position alone, so streaming (which
    only has a prefix) and non-streaming reach the same answer (#4038).
    Complete ``<tool_call>…</tool_call>`` spans are skipped, so backticks in
    an earlier call's arguments do not count. Queries must come at
    non-decreasing positions of one growing text; the scan resumes where
    the previous query stopped, so many queries cost linear time.
    """

    _OPEN = "<tool_call>"
    _CLOSE = "</tool_call>"

    def __init__(self) -> None:
        self._committed = 0
        self._fence: str | None = None
        self._line = ""

    def inside(self, text: str, pos: int) -> bool:
        if pos < self._committed:
            # Out-of-order query: answer it from scratch.
            return _MarkdownCodeTracker().inside(text, pos)
        cursor = self._committed
        while cursor < pos:
            start = text.find(self._OPEN, cursor, pos)
            if start == -1:
                self._feed(text[cursor:pos])
                cursor = pos
                break
            self._feed(text[cursor:start])
            close = text.find(self._CLOSE, start + len(self._OPEN), pos)
            if close == -1 or close + len(self._CLOSE) > pos:
                # Unclosed before ``pos``: its text counts for this query
                # only; resume from the opener next time.
                self._committed = start
                fence, line = self._fence, self._line
                self._feed(text[start:pos])
                result = self._result()
                self._fence, self._line = fence, line
                return result
            cursor = close + len(self._CLOSE)
        self._committed = max(self._committed, cursor)
        return self._result()

    def _feed(self, chunk: str) -> None:
        *done, self._line = (self._line + chunk).split("\n")
        for line in done:
            match = _MARKDOWN_FENCE_RE.match(line)
            if self._fence is None:
                if match:
                    self._fence = match.group(1)
            elif (
                match
                and match.group(1)[0] == self._fence[0]
                and len(match.group(1)) >= len(self._fence)
                and not line[match.end() :].strip()
            ):
                self._fence = None

    def _result(self) -> bool:
        if self._fence is not None or _MARKDOWN_FENCE_RE.match(self._line):
            return True
        # A Markdown code span is delimited by matching backtick RUNS, not by
        # an odd number of individual backticks.  In particular, ``code``
        # uses a two-backtick opener; counting characters classified that as
        # ordinary text and let the undeclared-tool scrubber delete examples
        # written inside it (#4038).
        opener_length: int | None = None
        cursor = 0
        while cursor < len(self._line):
            start = self._line.find("`", cursor)
            if start == -1:
                break
            end = start + 1
            while end < len(self._line) and self._line[end] == "`":
                end += 1
            run_length = end - start
            if opener_length is None:
                opener_length = run_length
            elif run_length == opener_length:
                opener_length = None
            cursor = end
        return opener_length is not None


def _merge_content(
    content: str, result: dict[str, Any] | None
) -> dict[str, Any] | None:
    """Prepend settled ``content`` to a streaming result."""
    if not content:
        return result
    if result is None:
        return {"content": content}
    return {**result, "content": content + (result.get("content") or "")}


def _generate_tool_id() -> str:
    return f"call_{uuid.uuid4().hex[:8]}"


def _field(value: Any, name: str, default: Any = None) -> Any:
    """Read a request field from either its wire dict or Pydantic model."""
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _get_arguments_config(func_name: str, tools: list[Any] | None) -> dict:
    """Extract argument config from tools list for type conversion."""
    if tools is None:
        return {}
    for tool in tools:
        func = _field(tool, "function", {})
        if _field(func, "name") == func_name:
            params = _field(func, "parameters", {})
            if isinstance(params, dict) and "properties" in params:
                return params["properties"]
            return {}
    return {}


def _is_string_param(param_name: str, param_config: dict) -> bool:
    """Whether ``param_name`` is explicitly string-typed per the tool schema.

    Unknown / un-configured / typeless params return False so they stay on
    the buffer-then-emit-once path. Non-streaming ``_convert_param_value``
    routes those through ``_decode_json_like()`` which may parse a JSON-
    looking value into an object — streaming it as a raw string would
    break stream/non-stream parity for those cases.
    """
    if param_name not in param_config:
        return False
    param_type = _schema_type(param_config[param_name])
    if param_type is None:
        return False
    return param_type in ("string", "str", "text", "varchar", "char", "enum")


def _convert_param_value(
    param_value: str, param_name: str, param_config: dict, func_name: str
) -> Any:
    """Convert parameter value based on its type in the schema.

    The TYPED scalar branches match against a whitespace-trimmed copy, never
    against the value that gets returned. Values used to arrive here already
    ``.strip()``-ed; now that only the wire's wrapping newline is removed
    (#3401), a model that pads a scalar -- ``<parameter=flag> true </parameter>``
    -- must still resolve to the scalar rather than to ``False``.

    A padded ``null`` is the keyword for every parameter EXCEPT a string-typed
    one, and that boundary is ``_is_string_param`` rather than a per-branch
    test, because it has to hold for the shapes that never reach the type
    dispatch at all: an undeclared parameter, a schema with no ``type`` key,
    and ``{"type": ["null"]}`` / null-only ``anyOf`` / ``oneOf`` (for which
    ``_schema_type`` returns ``None``). v0.14.1 resolved all of them to
    ``None`` because the value arrived pre-stripped.

    The string case is the deliberate exception, twice over. Its padding is
    payload under this wire's contract, and it is also the only set of values
    that reaches ``_close_string_increment``, where this function can be
    applied a second time to an already-decoded value (a pre-existing defect
    left untouched here). Trimming there would turn the string ``" null "``
    into ``None`` in the streamed arguments but not the non-streamed ones.
    """
    keyword = param_value.strip()
    if param_value.lower() == "null":
        return None
    if keyword.lower() == "null" and not _is_string_param(param_name, param_config):
        return None

    if param_name not in param_config:
        return _decode_json_like(param_value)

    cfg = param_config[param_name]
    param_type = _schema_type(cfg)
    if param_type is None:
        return _decode_json_like(param_value)

    if param_type in ("string", "str", "text", "varchar", "char", "enum"):
        try:
            decoded = json.loads(param_value)
        except (json.JSONDecodeError, TypeError):
            decoded = None
        if isinstance(decoded, str):
            return decoded
        return param_value

    if param_type.startswith(("int", "uint", "long", "short", "unsigned")):
        try:
            return int(keyword)
        except (ValueError, TypeError):
            return param_value
    elif param_type.startswith(("num", "float", "double")):
        # ``number`` follows JSON number semantics (issue #4037): ``10000``
        # stays ``10000``. Forcing ``10000.0`` made strictly typed clients
        # reject the call (Codex: "invalid type: floating point `10000.0`,
        # expected usize"). ``float``/``double`` schemas still ask for a float.
        if param_type.startswith("num") and _INTEGER_LITERAL_RE.fullmatch(keyword):
            try:
                return int(keyword)
            except ValueError:
                # Past Python's int-string digit limit: keep float() as before.
                pass
        try:
            return float(keyword)
        except (ValueError, TypeError):
            return param_value
    elif param_type in ("boolean", "bool", "binary"):
        return keyword.lower() == "true"
    else:
        if param_type in ("object", "array", "arr") or param_type.startswith(
            ("dict", "list")
        ):
            decoded = _decode_json_like(param_value)
            if decoded is not param_value:
                return decoded
        try:
            return ast.literal_eval(param_value)
        except (ValueError, SyntaxError):
            return param_value


@ToolParserManager.register_module(["qwen3_coder_xml"])
class Qwen3CoderToolParser(ToolParser):
    """
    Tool call parser for Qwen3-family models using named XML format.

    Supports the XML-based tool call format with <tool_call>/<function=...>
    tags and type conversion from tool schema. The parser name is retained for
    CLI compatibility; routing is based on the checkpoint's wire format, which
    is also used by Qwen3.8 27B.

    Used when --enable-auto-tool-choice --tool-call-parser qwen3_coder_xml are set.
    """

    SUPPORTS_NATIVE_TOOL_FORMAT = True
    EXPECTED_WIRE_FORMATS = ("qwen3_coder_xml_named", "tool_call_xml_body")

    # Grammar-CAPABLE (#558 "做全"): overrides ``structure_info`` below to emit
    # the Qwen3-Coder XML arg body. Only ``<tool_call>``/``</tool_call>`` are
    # single special tokens on the Qwen3-Coder tokenizer (verified on
    # ``mlx-community/Qwen3-Coder-Next-4bit``: ids 151657 / 151658); the inner
    # ``<function=`` / ``<parameter=`` / ``</parameter>`` / ``</function>``
    # markers are ordinary MULTI-token text, emitted as byte literals by the
    # grammar builder.
    _GRAMMAR_SENTINELS = ("<tool_call>", "</tool_call>")

    SUPPORTS_GRAMMAR: bool = True

    def structure_info(self):
        """Grammar-constraint wire triple for the Qwen3-Coder XML tool call (#558).

        Extends #558 grammar constraint from the JSON-body families
        (hermes / qwen / harmony) to the Qwen3-Coder XML wire. The model emits
        (verified byte-for-byte against this model's ``chat_template.jinja``)::

            <tool_call>
            <function=NAME>
            <parameter=KEY>
            VALUE
            </parameter>
            ...
            </function>
            </tool_call>

        The ARGUMENTS are an XML body, NOT a JSON object, so we return a
        ``StructureInfo`` with ``arg_style="xml"``: the builder emits one
        ``<parameter=KEY>\\nVALUE\\n</parameter>`` block per schema property, each
        VALUE constrained per its sub-schema (JSON strings for free-form strings,
        ``%json`` for scalars / objects / arrays, an alternation for enums) — see
        ``rapid_mlx.api.tool_grammar._emit_xml_arg_body``. Those surface forms are
        exactly what ``_parse_xml_function_call`` / ``_convert_param_value``
        round-trip back into JSON ``arguments``.

        ``<tool_call>`` / ``</tool_call>`` are single special tokens on the
        Qwen3-Coder tokenizer, declared as ``sentinels`` (rendered as Lark
        special-token refs); the trigger is ``<tool_call>``. The inner
        ``<function=`` / ``<parameter=`` markers are ordinary text (byte
        literals).

        As on hermes / qwen / harmony, OPT OUT (return ``None`` -> free-form
        fallback) unless the tokenizer proves both sentinels are single special
        tokens — a special-token sentinel on a tokenizer that encodes
        ``<tool_call>`` as multi-token text would build an unenforceable grammar.
        Grammar constraint is a best-effort opt-in, never a hard requirement.
        """
        from rapid_mlx.api.tool_grammar import (  # pragma: no cover - grammar opt-in
            StructureInfo,
            are_single_special_tokens,
        )

        if not are_single_special_tokens(self.model_tokenizer, self._GRAMMAR_SENTINELS):
            return None

        def _info(name: str):
            # The tool NAME is a bare identifier inside the ``<function=NAME>``
            # header (NOT a JSON string — Qwen3-Coder does not quote it), emitted
            # raw as a byte literal by the builder. ``begin`` starts with the
            # ``<tool_call>`` trigger (builder invariant).
            begin = f"<tool_call>\n<function={name}>\n"
            end = "</function>\n</tool_call>"
            return StructureInfo(
                begin=begin,
                end=end,
                trigger="<tool_call>",
                sentinels=self._GRAMMAR_SENTINELS,
                arg_style="xml",
            )

        return _info

    def __init__(self, tokenizer=None):
        super().__init__(tokenizer)

        self.tool_call_start_token = "<tool_call>"
        self.tool_call_end_token = "</tool_call>"
        self.tool_call_prefix = "<function="
        self.function_end_token = "</function>"
        self.parameter_prefix = "<parameter="
        self.parameter_end_token = "</parameter>"

        self.tool_call_complete_regex = re.compile(
            r"<tool_call>(.*?)</tool_call>", re.DOTALL
        )
        self.tool_call_regex = re.compile(
            r"<tool_call>(.*?)</tool_call>|<tool_call>(.*?)$", re.DOTALL
        )
        self.tool_call_function_regex = re.compile(
            r"<function=(.*?)</function>|<function=(.*)$", re.DOTALL
        )
        self.tool_call_parameter_regex = re.compile(
            r"<parameter=(.*?)(?:</parameter>|(?=<parameter=)|(?=</function>)|$)",
            re.DOTALL,
        )

        # Token IDs for streaming (graceful fallback if tokenizer absent)
        self.tool_call_start_token_id = self.vocab.get(self.tool_call_start_token)
        self.tool_call_end_token_id = self.vocab.get(self.tool_call_end_token)

        self._reset_streaming_state()

    def _reset_streaming_state(self):
        self.current_tool_index = 0
        self.is_tool_call_started = False
        self.header_sent = False
        self._current_tool_id = None
        self.current_function_name = None
        self.param_count = 0
        self.in_param = False
        self.in_function = False
        self.accumulated_text = ""
        self.json_started = False
        self.json_closed = False
        self.accumulated_params = {}
        self._streaming_request = None
        self._pending_tool_start: int | None = None
        self._pending_tool_wrapped = False
        self.prev_tool_call_arr = []
        self.in_param_emitted_chars = 0
        self.in_param_opened = False
        self.in_param_name: str | None = None
        self._legacy_raw_stream = False
        self._legacy_raw_param_count = 0
        # Undeclared-tool blocks (#4038). ``_undeclared_offset`` is how much
        # of the stream earlier dropped blocks consumed; the state machine
        # sees only the text after it. ``_undeclared_start`` is set while a
        # block is held back, ``_undeclared_logged`` counts logged drops.
        self._undeclared_offset = 0
        self._undeclared_start: int | None = None
        self._undeclared_name: str | None = None
        self._undeclared_declared: set[str] = set()
        self._undeclared_text = ""
        self._undeclared_dropped = 0
        self._undeclared_logged = 0
        # Full text of the current stream (before ``_undeclared_offset``
        # slicing): the Markdown-code check and the finalize log dedup use it.
        self._stream_text = ""
        self._initial_content_emitted = 0
        # Set once the stream rejects any candidate as content; later blocks
        # are then never dropped (non-streaming: an earlier ``<function=``
        # opener outside dropped blocks) (#4038).
        self._undeclared_blocked = False
        self._code_tracker = _MarkdownCodeTracker()

    def reset(self) -> None:
        """Reset for a new request, including the per-response #4038 state
        (so a reused parser logs every response's drops)."""
        super().reset()
        self._stream_text = ""
        self._undeclared_logged = 0

    def _emit_string_increment(self, param_name: str, value_text: str) -> str:
        """Return a JSON fragment for the safe (already-final) portion of an
        in-flight string param value, or "" if nothing new can be flushed.

        We withhold the last ``len("</parameter>")`` chars of unread tail so
        a partial close tag (e.g. ``</par`` straddling a chunk boundary)
        cannot leak into an emitted JSON fragment.
        """
        keep_back = len(self.parameter_end_token)
        safe_end = len(value_text) - keep_back
        if safe_end <= self.in_param_emitted_chars:
            return ""
        safe = value_text[self.in_param_emitted_chars : safe_end]
        if not safe:
            return ""
        inner = json.dumps(safe, ensure_ascii=False)[1:-1]
        self.in_param_emitted_chars = safe_end
        if not self.in_param_opened:
            self.in_param_opened = True
            prefix = "" if self.param_count == 0 else ", "
            return f'{prefix}"{param_name}": "{inner}'
        return inner

    @staticmethod
    def _decoded_json_string_prefix(value_text: str) -> str:
        """Decode the complete portion of an in-flight JSON string.

        Token boundaries may split an escape (including ``\\uXXXX``), so only
        the prefix consisting of complete JSON characters is returned.
        """
        text = value_text.lstrip()
        if not text.startswith('"'):
            return ""
        i = 1
        safe_end = i
        while i < len(text):
            char = text[i]
            if char == '"':
                break
            if char != "\\":
                i += 1
                safe_end = i
                continue
            if i + 1 >= len(text):
                break
            escape = text[i + 1]
            if escape == "u":
                if i + 6 > len(text):
                    break
                digits = text[i + 2 : i + 6]
                if any(c not in "0123456789abcdefABCDEF" for c in digits):
                    break
                codepoint = int(digits, 16)
                if 0xD800 <= codepoint <= 0xDBFF:
                    # A high surrogate is not independently emit-safe: JSON
                    # decoding combines it with a following low surrogate.
                    if i + 12 > len(text) or text[i + 6 : i + 8] != "\\u":
                        break
                    low_digits = text[i + 8 : i + 12]
                    if any(
                        c not in "0123456789abcdefABCDEF" for c in low_digits
                    ) or not (0xDC00 <= int(low_digits, 16) <= 0xDFFF):
                        break
                    i += 12
                    safe_end = i
                    continue
                i += 6
            elif escape in '"\\/bfnrt':
                i += 2
            else:
                break
            safe_end = i
        encoded = text[1:safe_end]
        try:
            return json.loads(f'"{encoded}"')
        except json.JSONDecodeError:
            return ""

    def _emit_decoded_string_increment(
        self, param_name: str, decoded_value: str
    ) -> str:
        """Emit newly decoded characters from an in-flight JSON string."""
        if len(decoded_value) <= self.in_param_emitted_chars:
            return ""
        safe = decoded_value[self.in_param_emitted_chars :]
        inner = json.dumps(safe, ensure_ascii=False)[1:-1]
        self.in_param_emitted_chars = len(decoded_value)
        if not self.in_param_opened:
            self.in_param_opened = True
            prefix = "" if self.param_count == 0 else ", "
            return f'{prefix}"{param_name}": "{inner}'
        return inner

    def _close_string_increment(
        self, param_name: str, full_value: str, param_config: dict
    ) -> str:
        """Emit the closing fragment for an in-flight string param now that
        ``</parameter>`` has arrived. Handles both the long-string case
        (opener already emitted; emit tail + closing quote) and the short-
        string case (opener never emitted; emit the whole ``"name": "value"``).
        """
        if not self.in_param_opened:
            converted = _convert_param_value(
                full_value,
                param_name,
                param_config,
                self.current_function_name or "",
            )
            serialized = json.dumps(converted, ensure_ascii=False)
            prefix = "" if self.param_count == 0 else ", "
            return f'{prefix}"{param_name}": {serialized}'
        tail = full_value[self.in_param_emitted_chars :]
        inner = json.dumps(tail, ensure_ascii=False)[1:-1]
        return f'{inner}"'

    def finalize_legacy_raw_stream(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> dict | None:
        """Return the un-emitted JSON suffix for a deferred raw parameter."""
        if not self._legacy_raw_stream:
            return None
        result = self.extract_tool_calls(model_output, request=request)
        if not result.tools_called or not result.tool_calls:
            return None
        if self.current_tool_index >= len(result.tool_calls):
            return None
        current = result.tool_calls[self.current_tool_index]
        arguments = json.loads(current["arguments"])
        remaining = list(arguments.items())[self._legacy_raw_param_count :]
        prefix = ", " if self._legacy_raw_param_count else ""
        suffix = prefix + ", ".join(
            f"{json.dumps(name)}: {json.dumps(value, ensure_ascii=False)}"
            for name, value in remaining
        )
        suffix += "}"
        self._legacy_raw_stream = False
        tool_calls = [
            {
                "index": self.current_tool_index,
                "function": {"arguments": suffix},
            }
        ]
        for index, call in enumerate(
            result.tool_calls[self.current_tool_index + 1 :],
            start=self.current_tool_index + 1,
        ):
            tool_calls.append(
                {
                    "index": index,
                    "id": call["id"],
                    "type": "function",
                    "function": {
                        "name": call["name"],
                        "arguments": call["arguments"],
                    },
                }
            )
        return {"tool_calls": tool_calls}

    def _parse_xml_function_call(
        self, function_call_str: str, tools: list[Any] | None
    ) -> dict | None:
        """Parse a single function call from XML and return a tool call dict."""
        try:
            end_index = function_call_str.index(">")
        except ValueError:
            return None
        function_name = function_call_str[:end_index]
        param_config = _get_arguments_config(function_name, tools)
        parameters = function_call_str[end_index + 1 :]
        param_dict = {}
        parsed = (
            split_marked_parameters(
                parameters,
                r"<parameter=([^>]+)>",
                self.parameter_end_token,
                valid_names=set(param_config) or None,
            )
            or []
        )
        for p_name, p_value in parsed:
            param_dict[p_name] = _convert_param_value(
                p_value, p_name, param_config, function_name
            )
        # Preserve the upstream recovery behavior for malformed free-form
        # output that omitted one close tag: the positional scanner correctly
        # protects complete JSON-string payloads, while this fallback recovers
        # later declared parameters that would otherwise be swallowed by the
        # unterminated predecessor. Never overwrite a value the safe scan found.
        for match_text in self.tool_call_parameter_regex.findall(parameters):
            try:
                idx = match_text.index(">")
            except ValueError:
                continue
            p_name = match_text[:idx]
            if p_name in param_dict or p_name not in param_config:
                continue
            p_value = trim_wrapping_newlines(str(match_text[idx + 1 :]))
            param_dict[p_name] = _convert_param_value(
                p_value, p_name, param_config, function_name
            )
        return {
            "id": _generate_tool_id(),
            "name": function_name,
            "arguments": json.dumps(param_dict, ensure_ascii=False),
        }

    def _get_function_calls(self, model_output: str) -> list[str]:
        return [body for body, _, _, _ in self._function_call_candidates(model_output)]

    def _function_call_candidates(
        self, model_output: str
    ) -> list[tuple[str, int, int, bool]]:
        """Return function bodies and the exact framing span each occupies."""
        candidates: list[tuple[str, int, int, bool]] = []
        for start in self._function_start_positions(model_output):
            close = self._top_level_function_close(model_output, start)
            body_start = start + len(self.tool_call_prefix)
            if close == -1:
                body = model_output[body_start:]
                complete_params = (
                    self.parameter_prefix in body
                    and body.rstrip().endswith(self.parameter_end_token)
                )
                wrapper_start = model_output.rfind(self.tool_call_start_token, 0, start)
                wrapper_close = model_output.rfind(self.tool_call_end_token, 0, start)
                wrapped_zero_arg = (
                    wrapper_start > wrapper_close
                    and body.find(">") >= 0
                    and not body[body.find(">") + 1 :].strip()
                )
                next_wrapper = model_output.find(self.tool_call_start_token, body_start)
                recovery_end = next_wrapper if next_wrapper >= 0 else len(model_output)
                trailing_wrapper = model_output.rfind(
                    self.tool_call_end_token, body_start, recovery_end
                )
                if trailing_wrapper >= 0:
                    # EOS recovery for a malformed call missing inner closes.
                    # Use the final wrapper closer so literal closer text in a
                    # raw value remains payload (#1515).
                    body = model_output[body_start:trailing_wrapper]
                    function_end = trailing_wrapper
                else:
                    if not (complete_params or wrapped_zero_arg):
                        continue
                    function_end = len(model_output)
            else:
                body = model_output[body_start:close]
                function_end = close + len(self.function_end_token)

            span_start = start
            wrapper_start = model_output.rfind(self.tool_call_start_token, 0, start)
            wrapper_close_before = model_output.rfind(
                self.tool_call_end_token, 0, start
            )
            is_wrapped = wrapper_start > wrapper_close_before
            if (
                wrapper_start >= 0
                and not model_output[
                    wrapper_start + len(self.tool_call_start_token) : start
                ].strip()
            ):
                span_start = wrapper_start

            span_end = function_end
            wrapper_end = model_output.find(self.tool_call_end_token, function_end)
            if wrapper_end >= 0 and not model_output[function_end:wrapper_end].strip():
                span_end = wrapper_end + len(self.tool_call_end_token)
            candidates.append((body, span_start, span_end, is_wrapped))
        return candidates

    def _content_without_admitted_calls(
        self, model_output: str, admitted_spans: list[tuple[int, int]]
    ) -> str:
        """Remove admitted functions without leaving half of a shared wrapper."""
        ranges: list[tuple[int, int]] = []
        for span_start, span_end in admitted_spans:
            function_start = model_output.find(
                self.tool_call_prefix, span_start, span_end
            )
            function_close = self._top_level_function_close(
                model_output, function_start
            )
            function_end = (
                function_close + len(self.function_end_token)
                if function_close >= 0
                else span_end
            )
            owns_wrapper = model_output[span_start:].startswith(
                self.tool_call_start_token
            )
            wrapper_close = model_output.find(self.tool_call_end_token, function_end)
            if owns_wrapper and (function_close < 0 or wrapper_close < 0):
                function_start = span_start
            ranges.append((function_start, function_end))

        ranges.sort()
        merged_ranges: list[tuple[int, int]] = []
        for start, end in ranges:
            if merged_ranges and start <= merged_ranges[-1][1]:
                merged_ranges[-1] = (
                    merged_ranges[-1][0],
                    max(end, merged_ranges[-1][1]),
                )
            else:
                merged_ranges.append((start, end))
        ranges = merged_ranges

        # A wrapper may frame more than one function. Strip its delimiters only
        # when every non-whitespace byte inside it is already being removed;
        # otherwise both delimiters belong to the rejected content sibling.
        search_from = 0
        range_index = 0
        wrapper_ranges: list[tuple[int, int]] = []
        while True:
            wrapper_start = model_output.find(self.tool_call_start_token, search_from)
            if wrapper_start < 0:
                break
            inner_start = wrapper_start + len(self.tool_call_start_token)
            wrapper_close = model_output.find(self.tool_call_end_token, inner_start)
            if wrapper_close < 0:
                break
            while range_index < len(ranges) and ranges[range_index][1] <= inner_start:
                range_index += 1
            cursor = inner_start
            residual = False
            candidate_index = range_index
            while (
                candidate_index < len(ranges)
                and ranges[candidate_index][0] < wrapper_close
            ):
                start, end = ranges[candidate_index]
                if model_output[cursor : max(cursor, start)].strip():
                    residual = True
                    break
                cursor = max(cursor, min(end, wrapper_close))
                candidate_index += 1
            if not residual and model_output[cursor:wrapper_close].strip():
                residual = True
            if not residual:
                wrapper_ranges.append(
                    (
                        wrapper_start,
                        wrapper_close + len(self.tool_call_end_token),
                    )
                )
            search_from = wrapper_close + len(self.tool_call_end_token)

        content_parts = []
        cursor = 0
        for start, end in sorted([*ranges, *wrapper_ranges]):
            if start > cursor:
                content_parts.append(model_output[cursor:start])
            cursor = max(cursor, end)
        content_parts.append(model_output[cursor:])
        return "".join(content_parts)

    @staticmethod
    def _named_tool_choice(request: dict[str, Any] | None) -> str | None:
        if not isinstance(request, dict):
            return None
        choice = request.get("tool_choice")
        if isinstance(choice, dict) or choice is not None:
            function = _field(choice, "function")
            selected = _field(function, "name") or _field(choice, "name")
            if isinstance(selected, str) and selected:
                return selected
        return None

    @classmethod
    def _declared_tool_names(cls, request: dict[str, Any] | None) -> set[str]:
        """Return executable tool names offered by this request."""
        if not isinstance(request, dict) or request.get("tool_choice") == "none":
            return set()
        names: set[str] = set()
        for tool in request.get("tools") or []:
            function = _field(tool, "function")
            name = _field(function, "name") or _field(tool, "name")
            if isinstance(name, str) and name:
                names.add(name)
        selected = cls._named_tool_choice(request)
        if selected:
            return names.intersection({selected})
        return names

    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> ExtractedToolCallInformation:
        if self.tool_call_prefix not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        try:
            candidates = self._function_call_candidates(model_output)
            if not candidates:
                return ExtractedToolCallInformation(
                    tools_called=False, tool_calls=[], content=model_output
                )

            tools = request.get("tools") if isinstance(request, dict) else None
            declared = self._declared_tool_names(request)
            selected = self._named_tool_choice(request)

            tool_calls: list[dict[str, Any]] = []
            accepted_spans: list[tuple[int, int]] = []
            dropped: list[tuple[int, int, str]] = []
            # #4038 drop gate: every ``<function=`` opener before a dropped
            # block must belong to an earlier dropped block.
            openers = self._function_start_positions(model_output)
            opener_index = 0
            dropped_openers: set[int] = set()
            blocked = False
            code = _MarkdownCodeTracker()
            for fc_str, span_start, span_end, is_wrapped in candidates:
                function_start = model_output.find(self.tool_call_prefix, span_start)
                while (
                    opener_index < len(openers)
                    and openers[opener_index] < function_start
                ):
                    if openers[opener_index] not in dropped_openers:
                        blocked = True
                    opener_index += 1
                candidate_name = fc_str.split(">", 1)[0]
                if (
                    not is_wrapped
                    and self.parameter_prefix not in fc_str
                    and candidate_name != selected
                ):
                    # Wrapper-less calls are supported for #978 models, but a
                    # zero-argument bare span is indistinguishable from prose
                    # documenting the wire format. Require either canonical
                    # framing or parameter structure before model text can
                    # become executable data.
                    continue
                tc = self._parse_xml_function_call(fc_str, tools)
                if not tc:
                    continue
                if tc.get("name") not in declared:
                    # Framing alone cannot distinguish executable wire from
                    # prose documenting that wire. A name the caller did not
                    # offer (including every name under tool_choice=none) is
                    # never executable. Preserve that span as text while still
                    # admitting independent, later candidates -- except a
                    # lone canonical block, which is an attempted call (#4038).
                    if (
                        declared
                        and not tool_calls
                        and not blocked
                        and self._is_lone_canonical_block(
                            model_output, span_start, span_end
                        )
                        and not code.inside(model_output, span_start)
                    ):
                        dropped.append((span_start, span_end, tc["name"]))
                        dropped_openers.add(function_start)
                    continue
                tool_calls.append(tc)
                accepted_spans.append((span_start, span_end))

            # Log each drop once per response. When this re-parses the stream
            # this parser just handled (the post-processor's finalize pass),
            # the streaming path has already logged the first drops.
            same_stream = bool(self._stream_text) and model_output.startswith(
                self._stream_text
            )
            already_logged = self._undeclared_logged if same_stream else 0
            for _, _, name in dropped[already_logged:]:
                self._log_undeclared_drop(name, declared)
            if same_stream:
                self._undeclared_logged = max(already_logged, len(dropped))

            if not tool_calls and not dropped:
                return ExtractedToolCallInformation(
                    tools_called=False, tool_calls=[], content=model_output
                )

            content = self._content_without_admitted_calls(
                model_output,
                accepted_spans + [(start, end) for start, end, _ in dropped],
            )
            if not tool_calls:
                return ExtractedToolCallInformation(
                    tools_called=False, tool_calls=[], content=content
                )

            return ExtractedToolCallInformation(
                tools_called=len(tool_calls) > 0,
                tool_calls=tool_calls,
                content=content if content else None,
            )
        except Exception:
            logger.exception("Error in extracting tool call from response.")
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

    def _is_lone_canonical_block(
        self, text: str, span_start: int, span_end: int
    ) -> bool:
        """``<tool_call>`` + one closed ``<function=…>`` + ``</tool_call>``.

        Only whitespace may separate the wrapper from the function. This is
        the exact shape the model emits to call a tool, so when its name was
        not declared it is an attempted call, not prose about the wire format
        (issue #4038). Bare spans, shared wrappers and unclosed blocks keep
        the preserve-as-text treatment.
        """
        block = text[span_start:span_end]
        if len(block) > _UNDECLARED_BLOCK_MAX_CHARS or not (
            block.startswith(self.tool_call_start_token)
            and block.endswith(self.tool_call_end_token)
        ):
            return False
        function_start = text.find(self.tool_call_prefix, span_start, span_end)
        close = self._top_level_function_close(text, function_start)
        return (
            close != -1
            and not text[
                close + len(self.function_end_token) : span_end
                - len(self.tool_call_end_token)
            ].strip()
        )

    @staticmethod
    def _log_undeclared_drop(name: str, declared: set[str]) -> None:
        logger.warning(
            "qwen3_coder_xml: the model called %r, which this request did not "
            "declare (%d tool(s) declared). The call is not executed and its "
            "markup is removed from the response content (#4038).",
            name,
            len(declared),
        )

    def _lone_block_opener(self, text: str, function_start: int) -> bool:
        """Whether the streaming tool at ``function_start`` may be a droppable
        lone canonical block: wrapped (only whitespace between ``<tool_call>``
        and ``<function=``) and no call emitted before it, mirroring the
        non-streaming ``not tool_calls`` condition."""
        if (
            self.prev_tool_call_arr
            or self.current_tool_index
            or self._undeclared_blocked
        ):
            return False
        wrapper = text.rfind(self.tool_call_start_token, 0, function_start)
        if wrapper < 0 or (
            self._pending_tool_start is not None and wrapper < self._pending_tool_start
        ):
            return False
        if self._code_tracker.inside(
            self._stream_text, self._undeclared_offset + wrapper
        ):
            return False
        return not text[
            wrapper + len(self.tool_call_start_token) : function_start
        ].strip()

    def _advance_past(self, consumed: int) -> None:
        """Restart the state machine after ``consumed`` chars of the
        post-offset text, keeping the per-stream request, offset, drop
        counters and stream text. ``consumed=0`` is the in-place restart
        used when a candidate is rejected as content."""
        saved = (
            self._streaming_request,
            self._undeclared_offset + consumed,
            self._undeclared_dropped,
            self._undeclared_logged,
            self._stream_text,
            self._undeclared_blocked,
            self._code_tracker,
        )
        self._reset_streaming_state()
        (
            self._streaming_request,
            self._undeclared_offset,
            self._undeclared_dropped,
            self._undeclared_logged,
            self._stream_text,
            self._undeclared_blocked,
            self._code_tracker,
        ) = saved

    def _reject_candidate(self) -> None:
        """Restart in place after a candidate was rejected as content; no
        later block in this stream is dropped (#4038)."""
        self._advance_past(0)
        self._undeclared_blocked = True

    def _begin_undeclared_block(
        self,
        text: str,
        function_start: int,
        name: str,
        content_before: str,
        request: dict[str, Any] | None,
        declared: set[str],
    ) -> _StepResult:
        """Hold back a wrapped call to an undeclared tool until it is known
        whether it forms a lone canonical block (#4038)."""
        self._undeclared_start = text.rfind(
            self.tool_call_start_token, 0, function_start
        )
        self._undeclared_name = name
        self._undeclared_declared = declared
        result = self._continue_undeclared_block(text)
        if not content_before:
            return result
        if isinstance(result, _Continue):
            return _Continue(content_before + result.content, result.tail)
        return _merge_content(content_before, result)

    def _continue_undeclared_block(self, text: str) -> _StepResult:
        """Drop a completed lone block, or release a block that turned out not
        to be one exactly as origin/main would have (as content)."""
        start = self._undeclared_start
        assert start is not None
        self._undeclared_text = text
        function_start = text.find(self.tool_call_prefix, start)
        close = self._top_level_function_close(text, function_start)
        if close == -1:
            return self._hold_or_release(text)
        after = close + len(self.function_end_token)
        rest = text[after:]
        trimmed = rest.lstrip()
        if trimmed.startswith(self.tool_call_end_token):
            end = after + (len(rest) - len(trimmed)) + len(self.tool_call_end_token)
            if end - start > _UNDECLARED_BLOCK_MAX_CHARS:
                return self._release_undeclared(text, end)
            self._undeclared_dropped += 1
            if self._undeclared_dropped > self._undeclared_logged:
                self._log_undeclared_drop(
                    self._undeclared_name or "", self._undeclared_declared
                )
                self._undeclared_logged = self._undeclared_dropped
            self._advance_past(end)
            tail = text[end:]
            if not tail:
                return {"content": ""}
            return _Continue("", tail)
        if not trimmed or self.tool_call_end_token.startswith(trimmed):
            return self._hold_or_release(text)
        # A sibling or other text follows inside the wrapper: not a lone block.
        return self._release_undeclared(text, after)

    def _hold_or_release(self, text: str) -> _StepResult:
        """Keep holding an open block unless it is already too long to drop."""
        assert self._undeclared_start is not None
        if len(text) - self._undeclared_start > _UNDECLARED_BLOCK_MAX_CHARS:
            return self._release_undeclared(text, len(text))
        return None

    def _release_undeclared(self, text: str, upto: int) -> _StepResult:
        """Emit the held span up to ``upto`` as content, as origin/main does
        for an undeclared name, and continue on the text after it."""
        assert self._undeclared_start is not None
        released = text[self._undeclared_start : upto]
        rest = text[upto:]
        self._advance_past(upto)
        # The released candidate remains visible content, so its
        # ``<function=`` opener is an earlier non-dropped opener.  Mirror the
        # non-streaming ``blocked`` state: no later canonical block in this
        # response may be deleted after that ambiguity has been exposed.
        self._undeclared_blocked = True
        if not rest:
            return {"content": released}
        return _Continue(released, rest)

    def flush_held_content(self, full_text: str) -> str:
        """Text of an undeclared-tool block still held when the stream ends.

        The block never completed, so it is not a lone canonical block and,
        as in non-streaming, it is returned as content (#4038).
        """
        if self._undeclared_start is None:
            if not self.prev_tool_call_arr:
                held = full_text[self._initial_content_emitted :]
                if held and any(
                    marker.startswith(held) for marker in ("<tool_call>", "<function=")
                ):
                    return held
            return ""
        held: str = self._undeclared_text[self._undeclared_start :]
        self._undeclared_start = None
        return held

    # --- streaming helpers -----------------------------------------------
    #
    # The streaming state machine is anchored on the ``<function=`` /
    # ``</function>`` pair, NOT on the optional ``<tool_call>`` wrapper.
    # These helpers make that decoupling explicit so the wrapper token
    # only appears in the content-before strip logic (where it must, to
    # avoid emitting wrapper framing as user-visible content).

    def _first_opener_pos(self, text: str) -> int:
        """Position of the earliest tool-call framing character in ``text``.

        A tool call may be introduced by either the ``<tool_call>`` wrapper
        or the bare ``<function=`` prefix; both must be stripped from any
        content emitted before the call. Returns ``len(text)`` when no
        opener is present so callers can use it as an unconditional slice
        endpoint.
        """
        tc = text.find(self.tool_call_start_token)
        fn = text.find(self.tool_call_prefix)
        if tc == -1 and fn == -1:
            return len(text)
        if tc == -1:
            return fn
        if fn == -1:
            return tc
        return min(tc, fn)

    def _new_opener_position(
        self, previous_text: str, current_text: str, delta_token_ids: Sequence[int]
    ) -> int:
        """Find an opener completed by this delta, including split markers.

        Accepts the wrapper token via string OR token-id (tokenizers that
        expose ``<tool_call>`` as a special token), and the bare
        ``<function=`` prefix via string. The two openers are equivalent
        as far as the state machine is concerned — either triggers the
        transition out of content-only mode.
        """
        start = max(
            0,
            len(previous_text)
            - max(len(self.tool_call_start_token), len(self.tool_call_prefix))
            + 1,
        )
        positions = [
            pos
            for marker in (self.tool_call_start_token, self.tool_call_prefix)
            if (pos := current_text.find(marker, start)) >= 0
            and pos + len(marker) > len(previous_text)
        ]
        if positions:
            return min(positions)
        if (
            self.tool_call_start_token_id is not None
            and self.tool_call_start_token_id in delta_token_ids
        ):
            return len(previous_text)
        return -1

    def _top_level_function_close(self, text: str, start: int) -> int:
        """Return the position of the top-level ``</function>`` that closes
        the tool opened at ``start`` — i.e. the first ``</function>`` after
        ``start`` that is NOT inside a ``<parameter=…>…</parameter>`` value.

        Returns ``-1`` when the tool hasn't closed yet in the buffer. A
        ``</function>`` embedded in a user's ``code`` parameter (XML
        code samples are the canonical example) MUST NOT be treated as
        the tool boundary; otherwise streaming truncates mid-argument
        (codex review on #978).
        """
        prefix_len = len(self.tool_call_prefix)
        param_open_len = len(self.parameter_prefix)
        param_close_len = len(self.parameter_end_token)
        j = start + prefix_len
        n = len(text)
        while j < n:
            next_param = text.find(self.parameter_prefix, j)
            next_close = text.find(self.function_end_token, j)
            if next_close == -1:
                return -1
            if next_param != -1 and next_param < next_close:
                header_end = text.find(">", next_param + param_open_len)
                if header_end == -1:
                    return -1
                pclose = self._find_parameter_close(text, header_end + 1)
                if pclose == -1:
                    return -1
                j = pclose + param_close_len
                continue
            return next_close
        return -1

    def _find_parameter_close(self, text: str, value_start: int) -> int:
        """Find the structural parameter close, skipping JSON-string payload.

        Constrained XML strings are JSON encoded (#1542), so delimiter-looking
        text inside the quotes is data. Legacy raw model output keeps the old
        first-close behavior.
        """
        i = value_start
        # The XML template permits formatting whitespace between the parameter
        # opener and its JSON-string value.  Treat that whitespace as wire, not
        # as part of a legacy raw string.  In particular, token streaming can
        # expose the whitespace one delta before the opening quote; failing to
        # skip it makes the incremental path escape both JSON wrapper quotes
        # into the user value (``/tmp/x`` becomes ``\"/tmp/x\"``).
        while i < len(text) and text[i].isspace():
            i += 1
        if i >= len(text) or text[i] != '"':
            return text.find(self.parameter_end_token, i)
        i += 1
        escaped = False
        while i < len(text):
            char = text[i]
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                return text.find(self.parameter_end_token, i + 1)
            i += 1
        return -1

    def _top_level_function_close_count(
        self, text: str, top_level_starts: list[int]
    ) -> int:
        """Count ``</function>`` tokens that structurally close a top-level
        ``<function=…>`` opener from ``top_level_starts``.

        Uses ``_top_level_function_close`` per start so a ``</function>``
        inside a ``<parameter=…>…</parameter>`` value (e.g. a user's
        ``code`` argument containing XML) never counts as a tool close.
        """
        return sum(
            1
            for start in top_level_starts
            if self._top_level_function_close(text, start) != -1
        )

    def _function_start_positions(self, text: str) -> list[int]:
        """Positions of TOP-LEVEL ``<function=`` openers in ``text``.

        Skips ``<function=`` substrings that appear inside a
        ``<parameter=…>…</parameter>`` value — those are user data (e.g.
        a ``code`` parameter containing XML), not structural tool-call
        boundaries. Function tags don't nest in Qwen3-Coder XML, so a
        top-level scan alternates between (a) looking for the next
        function opener while skipping parameter-value spans and (b)
        recording found openers. Both the tool-index slicing AND the
        "any more tools?" counter rely on this — using a naive
        ``str.count(...)`` for either would let a bogus in-value
        ``<function=…>`` corrupt streaming state (codex review on #978).

        Incomplete parameter tails (opener without matching
        ``</parameter>``) terminate the scan: everything after an
        unclosed value is potentially user data, so we conservatively
        refuse to promote further ``<function=`` occurrences until the
        value closes.
        """
        positions: list[int] = []
        i = 0
        n = len(text)
        prefix_len = len(self.tool_call_prefix)
        param_open_len = len(self.parameter_prefix)
        param_close_len = len(self.parameter_end_token)
        while i < n:
            next_func = text.find(self.tool_call_prefix, i)
            next_param = text.find(self.parameter_prefix, i)
            if next_func == -1:
                return positions
            if next_param != -1 and next_param < next_func:
                header_end = text.find(">", next_param + param_open_len)
                if header_end == -1:
                    return positions
                close_pos = self._find_parameter_close(text, header_end + 1)
                if close_pos == -1:
                    return positions
                i = close_pos + param_close_len
                continue
            positions.append(next_func)
            i = next_func + prefix_len
        return positions

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int] | None = None,
        current_token_ids: Sequence[int] | None = None,
        delta_token_ids: Sequence[int] | None = None,
        request: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        if not previous_text:
            self._reset_streaming_state()
            self._streaming_request = request
        elif not self._stream_text:
            # The postprocessor can pass through ordinary prose before it
            # starts invoking the tool parser. Those bytes are already on the
            # wire and must not be emitted again when a split opener arrives.
            self._initial_content_emitted = len(previous_text)
        elif request is not None and self._streaming_request is None:
            self._streaming_request = request
        self._stream_text = current_text

        # Drain a chunk that closes the active function and contains another
        # complete function. The single-call state machine returns after the
        # first close; split at its structural close so the remainder is
        # processed with the updated state.
        if self.in_function:
            starts = self._function_start_positions(current_text)
            if self._top_level_function_close_count(current_text, starts) > (
                self.current_tool_index + 1
            ) and self.current_tool_index < len(starts):
                close = self._top_level_function_close(
                    current_text, starts[self.current_tool_index]
                )
                split = close + len(self.function_end_token)
                if close >= 0 and split > len(previous_text):
                    first = self.extract_tool_calls_streaming(
                        previous_text,
                        current_text[:split],
                        current_text[len(previous_text) : split],
                        request=request,
                    )
                    second = self.extract_tool_calls_streaming(
                        current_text[:split],
                        current_text,
                        current_text[split:],
                        request=request,
                    )
                    if first and second:
                        return {
                            "tool_calls": first.get("tool_calls", [])
                            + second.get("tool_calls", []),
                            "content": first.get("content", "")
                            + second.get("content", ""),
                        }
                    return first or second

        # A coarse delta can finish several calls at once. The state machine
        # below normally advances only one call per invocation, so emit every
        # newly completed call here when it is between function bodies.
        if not self.in_function and current_text.count(
            self.function_end_token
        ) > previous_text.count(self.function_end_token):
            complete = self.extract_tool_calls(current_text, request)
            top_level_starts = self._function_start_positions(current_text)
            closed_count = self._top_level_function_close_count(
                current_text, top_level_starts
            )
            already = len(self.prev_tool_call_arr)
            if min(len(complete.tool_calls), closed_count) > already:
                fresh = complete.tool_calls[already:closed_count]
                self.prev_tool_call_arr.extend(fresh)
                self.current_tool_index = already + len(fresh) - 1
                self.json_closed = True
                self.header_sent = True
                self.is_tool_call_started = True
                output = {
                    "tool_calls": [
                        {
                            "index": index,
                            "id": call["id"],
                            "type": "function",
                            "function": {
                                "name": call["name"],
                                "arguments": call["arguments"],
                            },
                        }
                        for index, call in enumerate(fresh, start=already)
                    ]
                }
                if already == 0:
                    first_start = self._first_opener_pos(current_text)
                    prefix = current_text[self._initial_content_emitted : first_start]
                    if prefix:
                        output["content"] = prefix
                        self._initial_content_emitted = first_start
                return output

        if not delta_text:
            return None

        declared = self._declared_tool_names(
            request if request is not None else self._streaming_request
        )
        if not declared:
            # No executable tool exists for this request. Bypass the XML state
            # machine entirely so protocol examples stream byte-for-byte and
            # tool_choice=none cannot be overturned by model-authored markup.
            return {"content": delta_text}

        if self._undeclared_offset:
            previous_text = previous_text[self._undeclared_offset :]
            current_text = current_text[self._undeclared_offset :]
        return self._streaming_step(
            previous_text, current_text, delta_text, delta_token_ids, request, declared
        )

    def _streaming_step(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        delta_token_ids: Sequence[int] | None,
        request: dict[str, Any] | None,
        declared: set[str],
    ) -> dict[str, Any] | None:
        """Run the state machine on the text after any dropped blocks,
        looping (not recursing) over what a dropped block leaves (#4038)."""
        settled = ""
        while True:
            result = self._streaming_step_once(
                previous_text,
                current_text,
                delta_text,
                delta_token_ids,
                request,
                declared,
            )
            if not isinstance(result, _Continue):
                return _merge_content(settled, result)
            settled += result.content
            previous_text, current_text, delta_text = "", result.tail, result.tail
            delta_token_ids = None

    def _streaming_step_once(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        delta_token_ids: Sequence[int] | None,
        request: dict[str, Any] | None,
        declared: set[str],
    ) -> _StepResult:
        if self._undeclared_start is not None:
            return self._continue_undeclared_block(current_text)

        if self._legacy_raw_stream:
            # An escaping-free raw XML string cannot distinguish a literal
            # complete close sequence from structure at an ordinary chunk
            # boundary. Do not make an irreversible streaming decision;
            # StreamingPostProcessor.finalize() runs the last-closer-aware
            # non-streaming parser over the complete output (#1515).
            self.accumulated_text = current_text
            return None

        delta_token_ids = delta_token_ids or []
        self.accumulated_text = current_text

        # Check if we need to advance to next tool. The tool boundary is
        # ``</function>`` — that is the invariant close, whether or not
        # a wrapping ``</tool_call>`` follows. Both the close-count and
        # the "any more tools?" check must ignore ``<function=`` /
        # ``</function>`` substrings inside parameter values — a
        # ``code`` parameter carrying XML code would otherwise trip the
        # advance loop into targeting a bogus next block (codex review
        # on #978).
        if self.json_closed and not self.in_function:
            top_level_starts = self._function_start_positions(current_text)
            tool_count = len(top_level_starts)
            tool_ends = self._top_level_function_close_count(
                current_text, top_level_starts
            )
            if tool_ends > self.current_tool_index:
                self.current_tool_index += 1
                self.header_sent = False
                self.param_count = 0
                self.json_started = False
                self.json_closed = False
                self.accumulated_params = {}
                if self.current_tool_index >= tool_count:
                    self.is_tool_call_started = False
                else:
                    return None

        # Handle content before tool calls. Either opener (wrapper or bare
        # ``<function=``) transitions us out of content-only mode; the
        # content-before-strip position is whichever opener appears first
        # in ``delta_text`` so wrapper framing never leaks to the client.
        if not self.is_tool_call_started:
            opener_start = self._new_opener_position(
                previous_text, current_text, delta_token_ids
            )
            if opener_start >= 0:
                self.is_tool_call_started = True
                opener_pos = max(0, opener_start - len(previous_text))
                self._pending_tool_start = opener_start
                wrapper_start = current_text.find(
                    self.tool_call_start_token, self._pending_tool_start
                )
                function_start = current_text.find(
                    self.tool_call_prefix, self._pending_tool_start
                )
                self._pending_tool_wrapped = wrapper_start >= 0 and (
                    function_start < 0 or wrapper_start <= function_start
                )
                header_start = current_text.find(
                    self.tool_call_prefix, self._pending_tool_start
                )
                if header_start >= 0:
                    name_start = header_start + len(self.tool_call_prefix)
                    header_end = current_text.find(">", name_start)
                    if header_end >= 0:
                        candidate_name = current_text[name_start:header_end]
                        if candidate_name not in declared and (
                            self._lone_block_opener(current_text, header_start)
                        ):
                            return self._begin_undeclared_block(
                                current_text,
                                header_start,
                                candidate_name,
                                delta_text[:opener_pos],
                                request,
                                declared,
                            )
                        if candidate_name not in declared:
                            self._reject_candidate()
                            return {"content": delta_text}
                        candidate_text = current_text[header_start:]
                        if (
                            not self._pending_tool_wrapped
                            and self.function_end_token in candidate_text
                            and self.parameter_prefix not in candidate_text
                        ):
                            self._reject_candidate()
                            return {"content": delta_text}
                if not self.prev_tool_call_arr:
                    content_before = current_text[
                        self._initial_content_emitted : opener_start
                    ]
                    self._initial_content_emitted = opener_start
                else:
                    content_before = delta_text[:opener_pos]
                if content_before:
                    return {"content": content_before}
                # Fall through to header parsing below instead of returning
                # None — the function header may already be in current_text.
            else:
                if not self.prev_tool_call_arr:
                    # Hold a suffix that may become a split opener in the next
                    # token. Only settled prose may reach the content channel.
                    held = max(
                        (
                            n
                            for marker in (
                                self.tool_call_start_token,
                                self.tool_call_prefix,
                            )
                            for n in range(1, len(marker))
                            if current_text.endswith(marker[:n])
                        ),
                        default=0,
                    )
                    safe_end = len(current_text) - held
                    settled = current_text[self._initial_content_emitted : safe_end]
                    self._initial_content_emitted = safe_end
                    return {"content": settled} if settled else None
                # Suppress the trailing-wrapper-close whitespace event so
                # a stream that ends with just ``</tool_call>\n`` doesn't
                # emit an empty tail. ``</function>`` is the actual tool
                # close; ``</tool_call>`` may follow as optional framing.
                trailing = current_text.rstrip()
                if delta_text.strip() == "" and (
                    trailing.endswith(self.tool_call_end_token)
                    or trailing.endswith(self.function_end_token)
                ):
                    return None
                return {"content": delta_text}

        # Find current tool call portion. Slice from the current
        # ``<function=`` opener to the matching top-level ``</function>``
        # close — this is the wrapper-agnostic tool-call block. Both
        # ends use the parameter-aware scanners so a user-visible
        # ``<function=…>`` OR ``</function>`` embedded in a parameter
        # value can't corrupt the slice.
        function_starts = self._function_start_positions(current_text)
        if self.current_tool_index >= len(function_starts):
            return None

        tool_start_idx = function_starts[self.current_tool_index]
        func_close_idx = self._top_level_function_close(current_text, tool_start_idx)
        if func_close_idx == -1:
            tool_text = current_text[tool_start_idx:]
        else:
            tool_text = current_text[
                tool_start_idx : func_close_idx + len(self.function_end_token)
            ]

        # Parse function header
        if not self.header_sent:
            if self.tool_call_prefix in tool_text:
                func_start = tool_text.find(self.tool_call_prefix) + len(
                    self.tool_call_prefix
                )
                func_end = tool_text.find(">", func_start)
                if func_end != -1:
                    self.current_function_name = tool_text[func_start:func_end]
                    if (
                        self.current_function_name not in declared
                        and self._lone_block_opener(current_text, tool_start_idx)
                    ):
                        return self._begin_undeclared_block(
                            current_text,
                            tool_start_idx,
                            self.current_function_name,
                            "",
                            request,
                            declared,
                        )
                    if self.current_function_name not in declared:
                        start = self._pending_tool_start
                        rejected = (
                            current_text[start:] if start is not None else delta_text
                        )
                        self._reject_candidate()
                        return {"content": rejected}
                    if (
                        not self._pending_tool_wrapped
                        and self.parameter_prefix not in tool_text
                        and self.current_function_name
                        != self._named_tool_choice(
                            request if request is not None else self._streaming_request
                        )
                    ):
                        if self.function_end_token not in tool_text:
                            # Wait for a possible first parameter before
                            # emitting an irreversible tool-call header.
                            return None
                        start = self._pending_tool_start
                        rejected = (
                            current_text[start:] if start is not None else delta_text
                        )
                        self._reject_candidate()
                        return {"content": rejected}
                    first_param = tool_text.find(self.parameter_prefix, func_end)
                    if first_param >= 0 and func_close_idx == -1:
                        name_end = tool_text.find(">", first_param)
                        if name_end >= 0:
                            param_name = tool_text[
                                first_param + len(self.parameter_prefix) : name_end
                            ]
                            visible = tool_text[name_end + 1 :].lstrip()
                            request_tools = (
                                self._streaming_request.get("tools")
                                if isinstance(self._streaming_request, dict)
                                else None
                            )
                            config = _get_arguments_config(
                                self.current_function_name, request_tools
                            )
                            if (
                                visible
                                and not visible.startswith('"')
                                and (_is_string_param(param_name, config) or not config)
                            ):
                                self._legacy_raw_stream = True
                                self._legacy_raw_param_count = 0
                    self._current_tool_id = _generate_tool_id()
                    self.header_sent = True
                    self.in_function = True

                    # If the function body is already complete, emit the full
                    # tool call in one chunk to prevent header-only output
                    # when coarse deltas or max_tokens truncation leave no
                    # further parser calls.
                    if func_close_idx != -1:
                        complete = self.extract_tool_calls(current_text, request)
                        if complete.tools_called and len(complete.tool_calls) > 1:
                            calls = complete.tool_calls[self.current_tool_index :]
                            self.prev_tool_call_arr.extend(calls)
                            self.current_tool_index = len(complete.tool_calls) - 1
                            self.json_started = True
                            self.json_closed = True
                            self.in_function = False
                            self.accumulated_params = {}
                            return {
                                "tool_calls": [
                                    {
                                        "index": index,
                                        "id": call["id"],
                                        "type": "function",
                                        "function": {
                                            "name": call["name"],
                                            "arguments": call["arguments"],
                                        },
                                    }
                                    for index, call in enumerate(
                                        calls,
                                        start=self.current_tool_index + 1 - len(calls),
                                    )
                                ]
                            }
                        tools = None
                        if request and isinstance(request, dict):
                            tools = request.get("tools")
                        fc = tool_text[func_start : -len(self.function_end_token)]
                        parsed = self._parse_xml_function_call(fc, tools)
                        args = parsed["arguments"] if parsed else "{}"
                        self.json_started = True
                        self.json_closed = True
                        self.in_function = False
                        self.accumulated_params = {}
                        self.prev_tool_call_arr.append(
                            {"name": self.current_function_name, "arguments": args}
                        )
                        return {
                            "tool_calls": [
                                {
                                    "index": self.current_tool_index,
                                    "id": self._current_tool_id,
                                    "type": "function",
                                    "function": {
                                        "name": self.current_function_name,
                                        "arguments": args,
                                    },
                                }
                            ]
                        }

                    self.prev_tool_call_arr.append(
                        {"name": self.current_function_name, "arguments": "{}"}
                    )
                    self.json_started = True
                    return {
                        "tool_calls": [
                            {
                                "index": self.current_tool_index,
                                "id": self._current_tool_id,
                                "type": "function",
                                "function": {
                                    "name": self.current_function_name,
                                    "arguments": "{",
                                },
                            }
                        ]
                    }
            return None

        # Handle function body
        if self.in_function:
            if not self.json_started:
                self.json_started = True
                return {
                    "tool_calls": [
                        {
                            "index": self.current_tool_index,
                            "function": {"arguments": "{"},
                        }
                    ]
                }

            # Find all parameter start positions
            param_starts = []
            si = 0
            while True:
                si = tool_text.find(self.parameter_prefix, si)
                if si == -1:
                    break
                param_starts.append(si)
                si += len(self.parameter_prefix)

            tools = None
            if self._streaming_request:
                tools = (
                    self._streaming_request.get("tools")
                    if isinstance(self._streaming_request, dict)
                    else None
                )
            param_config = _get_arguments_config(
                self.current_function_name or "", tools
            )

            json_fragments = []

            # In-flight string param from a prior call: drain whatever's
            # now available (close it if </parameter> has arrived, else
            # emit another safe slice). Runs BEFORE the complete-param
            # loop so same-chunk trailing params after the close get
            # picked up in the same call.
            if (
                self.in_param
                and self.in_param_name is not None
                and self.param_count < len(param_starts)
            ):
                param_idx = param_starts[self.param_count]
                param_start = param_idx + len(self.parameter_prefix)
                remaining = tool_text[param_start:]
                if ">" in remaining:
                    name_end = remaining.find(">")
                    value_start = param_start + name_end + 1
                    value_text = tool_text[value_start:]
                    if value_text.startswith("\n"):
                        value_text = value_text[1:]

                    end_idx = self._find_parameter_close(value_text, 0)
                    json_string_pending = value_text.lstrip().startswith('"')
                    if end_idx == -1 and not json_string_pending:
                        # Defensive fallback: model emitted next-param-prefix,
                        # </function>, or </tool_call> without a </parameter>.
                        # Use any of those as the close to avoid hanging
                        # forever in incremental mode (mirrors the existing
                        # complete-param fallback path below).
                        nxt = value_text.find(self.parameter_prefix)
                        fe = value_text.find(self.function_end_token)
                        te = value_text.find(self.tool_call_end_token)
                        candidates = [c for c in (nxt, fe, te) if c != -1]
                        if candidates:
                            end_idx = min(candidates)
                    if end_idx != -1:
                        pv = value_text[:end_idx]
                        if pv.endswith("\n"):
                            pv = pv[:-1]
                        self.accumulated_params[self.in_param_name] = pv
                        close_value = (
                            _convert_param_value(
                                pv,
                                self.in_param_name,
                                param_config,
                                self.current_function_name or "",
                            )
                            if json_string_pending
                            else pv
                        )
                        frag = self._close_string_increment(
                            self.in_param_name, close_value, param_config
                        )
                        if frag:
                            json_fragments.append(frag)
                        self.param_count += 1
                        self.in_param = False
                        self.in_param_name = None
                        self.in_param_emitted_chars = 0
                        self.in_param_opened = False
                    else:
                        frag = (
                            self._emit_decoded_string_increment(
                                self.in_param_name,
                                self._decoded_json_string_prefix(value_text),
                            )
                            if json_string_pending
                            else self._emit_string_increment(
                                self.in_param_name, value_text
                            )
                        )
                        if frag:
                            json_fragments.append(frag)

            # Process complete parameters
            while not self.in_param and self.param_count < len(param_starts):
                param_idx = param_starts[self.param_count]
                param_start = param_idx + len(self.parameter_prefix)
                remaining = tool_text[param_start:]

                if ">" not in remaining:
                    break

                name_end = remaining.find(">")
                current_param_name = remaining[:name_end]
                value_start = param_start + name_end + 1
                value_text = tool_text[value_start:]
                if value_text.startswith("\n"):
                    value_text = value_text[1:]

                if not value_text.strip():
                    break
                is_string = _is_string_param(current_param_name, param_config)
                if not value_text.lstrip().startswith('"') and (
                    is_string or not param_config
                ):
                    # Raw XML has no escaping rule, so its first apparent close
                    # may be payload. Freeze only the un-emitted suffix and let
                    # EOS parsing select the final structural closer.
                    self._legacy_raw_stream = True
                    self._legacy_raw_param_count = self.param_count
                    return None

                param_end_idx = self._find_parameter_close(value_text, 0)
                json_string_pending = value_text.lstrip().startswith('"')
                if param_end_idx == -1 and not json_string_pending:
                    # Try next parameter or function end as delimiter
                    next_param = value_text.find(self.parameter_prefix)
                    func_end = value_text.find(self.function_end_token)
                    if next_param != -1 and (func_end == -1 or next_param < func_end):
                        param_end_idx = next_param
                    elif func_end != -1:
                        param_end_idx = func_end
                    else:
                        tool_end_in_val = value_text.find(self.tool_call_end_token)
                        if tool_end_in_val != -1:
                            param_end_idx = tool_end_in_val
                        else:
                            # Param not yet closed. For string params, emit
                            # incrementally; for non-string types we can't
                            # emit partial JSON (half an int isn't valid),
                            # so fall through to the existing break path.
                            if _is_string_param(current_param_name, param_config):
                                frag = self._emit_string_increment(
                                    current_param_name, value_text
                                )
                                if frag:
                                    json_fragments.append(frag)
                                self.in_param = True
                                self.in_param_name = current_param_name
                            break

                if param_end_idx == -1 and json_string_pending:
                    if _is_string_param(current_param_name, param_config):
                        frag = self._emit_decoded_string_increment(
                            current_param_name,
                            self._decoded_json_string_prefix(value_text),
                        )
                        if frag:
                            json_fragments.append(frag)
                        self.in_param = True
                        self.in_param_name = current_param_name
                    break

                if param_end_idx == -1:
                    break

                pv = value_text[:param_end_idx]
                if pv.endswith("\n"):
                    pv = pv[:-1]

                self.accumulated_params[current_param_name] = pv

                converted = _convert_param_value(
                    pv,
                    current_param_name,
                    param_config,
                    self.current_function_name or "",
                )
                serialized = json.dumps(converted, ensure_ascii=False)

                if self.param_count == 0:
                    frag = f'"{current_param_name}": {serialized}'
                else:
                    frag = f', "{current_param_name}": {serialized}'
                self.param_count += 1
                json_fragments.append(frag)

            # If the function body is now complete, fold the closing ``}``
            # into the same delta so a chunk that batches the last param +
            # </function> emits a self-contained, JSON-valid arguments
            # document instead of leaving the close stranded for a later
            # call (which may never come if the stream ended).
            close_pending = (
                not self.in_param and not self.json_closed and func_close_idx != -1
            )
            if close_pending:
                self.json_closed = True
                tools = None
                if self._streaming_request:
                    tools = (
                        self._streaming_request.get("tools")
                        if isinstance(self._streaming_request, dict)
                        else None
                    )
                func_start = tool_text.find(self.tool_call_prefix) + len(
                    self.tool_call_prefix
                )
                func_content_end = len(tool_text) - len(self.function_end_token)
                if func_content_end != -1:
                    fc = tool_text[func_start:func_content_end]
                    try:
                        parsed = self._parse_xml_function_call(fc, tools)
                        if parsed and self.current_tool_index < len(
                            self.prev_tool_call_arr
                        ):
                            self.prev_tool_call_arr[self.current_tool_index][
                                "arguments"
                            ] = parsed["arguments"]
                    except Exception:
                        pass
                self.in_function = False
                self.accumulated_params = {}
                json_fragments.append("}")

            if json_fragments:
                combined = "".join(json_fragments)
                return {
                    "tool_calls": [
                        {
                            "index": self.current_tool_index,
                            "function": {"arguments": combined},
                        }
                    ]
                }

        return None
