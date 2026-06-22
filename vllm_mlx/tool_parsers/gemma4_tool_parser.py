# SPDX-License-Identifier: Apache-2.0
"""
Gemma 4 tool call parser for rapid-mlx.

Handles Gemma 4's native tool calling format:
  <|tool_call>call:FUNC_NAME{key:<|"|>value<|"|>,...}<tool_call|>
"""

import json
import re
import uuid
from collections.abc import Sequence
from typing import Any

from .abstract_tool_parser import (
    ExtractedToolCallInformation,
    ToolParser,
    ToolParserManager,
)

# Match the gemma4 tool-call wire form. The model trains on
#   <|tool_call>call:name{...}<tool_call|>
# but those outer markers are special tokens that HuggingFace's
# ``tokenizer.decode(..., skip_special_tokens=True)`` (the default
# the mlx-vlm / mlx-lm streaming detokenizer invokes) silently strips
# at decode time even when we kept them in ``skip_special_token_ids``.
# Empirically (PR #558 share probe 2026-06-11 on DiffusionGemma 4-bit):
#   prompt:  weather in palo alto
#   output:  call:weather{location:<|"|>Palo Alto<|"|>}
# i.e. the model emits id=48/49 for the outer wrappers (gets stripped),
# but emits the inner ``<|"|>`` (id=52) as raw BPE bytes that survive
# the same decode call. So in practice we see only the inner body.
#
# Make the outer wrappers OPTIONAL so the parser recognises both the
# pristine wire form AND the post-decode stripped form. The body
# ``call:NAME{...}`` is itself a learned wire token unique to tool
# calling — Gemma 4 does not emit ``call:NAME{...}`` in natural prose,
# so allowing the wrappers to be absent does not introduce false
# positives on regular chat turns.
GEMMA4_TOOL_PATTERN = re.compile(
    r"(?:<\|tool_call>)?call:(\w+)\{(.*?)\}(?:<tool_call\|>)?", re.DOTALL
)

# Match a quoted-string value: <|"|>...<|"|>
GEMMA4_QUOTED_VAL_PATTERN = re.compile(r'<\|"\|>(.*?)<\|"\|>', re.DOTALL)
# Match a bare key:value pair (key, then anything up to , or end-of-string)
GEMMA4_KV_BARE_PATTERN = re.compile(r"(\w+)\s*:\s*([^,]+?)(?=\s*,|\s*$)")

# r5-E F-DGF-V080-B-8: prose-fallback recovery patterns. Gemma 4 at
# low temperature (~0.1) intermittently emits prose describing the
# tool intent ("I should call the `add` tool with a=13 and b=29.")
# instead of emitting the structured ``<|tool_call>call:NAME{...}<tool_call|>``
# wire form. Trace verdict (see commit body): NEITHER parser pattern
# miss (the regex above correctly catches every structured emission)
# NOR template tool injection miss (the chat template renders
# ``<|tool>declaration:NAME{...}<tool|>`` verbatim and the model has
# the schema). The model just chose to think-aloud through the
# ``<|channel>thought`` channel without ever transitioning to the
# tool_call channel — pure LLM decoding edge case.
#
# Defence-in-depth: when the structured form misses AND the request
# carried a ``tools`` array, look for the model's tool-intent prose
# and recover the call. The matcher is gated on three conjunctive
# checks (every false alarm we measured was caught by at least one):
#
#   1. The prose mentions a tool name FROM THE REQUEST (verbatim,
#      with optional backticks/quotes). Natural prose almost never
#      names an arbitrary user-supplied identifier exactly.
#   2. The prose contains ``key=value`` (or ``key: value``)
#      assignments for EVERY required parameter on that tool. This
#      is the strong signal — the model lays out the args even
#      when it forgets to wrap them in the channel form.
#   3. The whole prose passage stays inside a single sentence /
#      80-char window of the tool-name mention so a long natural
#      paragraph that happens to contain ``add`` and an unrelated
#      ``a=`` assignment elsewhere is not collaterally captured.
#
# A miss leaves the prose in ``content`` unchanged — there's no
# silent degradation if the recovery doesn't fire.
GEMMA4_PROSE_KV_PATTERN = re.compile(
    r"(\b\w+)\s*[=:]\s*"
    # value: quoted string, numeric, or bare token up to comma /
    # whitespace boundary. Backticked / quoted strings are unwrapped.
    r"(?:`([^`]+)`|\"([^\"]*)\"|'([^']*)'|(-?\d+(?:\.\d+)?)|([A-Za-z_][\w-]*))"
)


def _parse_gemma4_args(args_str: str) -> dict[str, Any]:
    """Parse Gemma 4's argument format into a dict.

    Gemma 4 uses two value styles inside the {...} block:
      - String values are wrapped in quote tokens:  key:<|"|>value<|"|>
      - Numeric / bool / null values are bare:      key:3   key:true   key:null

    Strategy: replace each quoted string with a placeholder, run a generic
    bare-KV parser over the result, then restore placeholders before
    returning. This lets a single pass handle mixed-type arg dicts.
    """
    # Step 1: stash quoted string values so they can't confuse the bare parser
    stashed: list[str] = []

    def _stash(m: re.Match) -> str:
        stashed.append(m.group(1))
        return f"__Q{len(stashed) - 1}__"

    cleaned = GEMMA4_QUOTED_VAL_PATTERN.sub(_stash, args_str)

    # Step 2: bare KV parse
    result: dict[str, Any] = {}
    for kv in GEMMA4_KV_BARE_PATTERN.finditer(cleaned):
        key = kv.group(1)
        raw_val = kv.group(2).strip()
        # Restore stashed string
        if raw_val.startswith("__Q") and raw_val.endswith("__"):
            try:
                idx = int(raw_val[3:-2])
                result[key] = stashed[idx]
                continue
            except (ValueError, IndexError):
                pass
        # Try to parse as JSON literal (int, float, bool, null)
        try:
            result[key] = json.loads(raw_val)
        except (json.JSONDecodeError, ValueError):
            result[key] = raw_val
    return result


def _generate_tool_id() -> str:
    return f"call_{uuid.uuid4().hex[:8]}"


def _extract_request_tools(request: Any) -> list[dict]:
    """Pull the ``tools`` list off a request in either dict or attr form.

    The parser is called from two paths: the non-streaming finalize
    (request is a ``ChatCompletionRequest`` model_dump dict, see
    ``vllm_mlx/service/helpers.py``) and a few unit-test paths that pass
    raw dicts. Returns ``[]`` when no usable tools list is found.
    """
    if request is None:
        return []
    tools = None
    if isinstance(request, dict):
        tools = request.get("tools")
    else:
        tools = getattr(request, "tools", None)
    if not isinstance(tools, list):
        return []
    return [t for t in tools if isinstance(t, dict)]


def _coerce_prose_value(quoted: tuple) -> Any:
    """Convert a ``GEMMA4_PROSE_KV_PATTERN`` value capture tuple into
    a JSON-friendly Python value.

    The tuple positions are ``(backtick, dquote, squote, number, bare)``
    — at most one is non-empty per match. Numerics are parsed as JSON
    literals so ``a=3`` becomes ``int(3)``, ``rate=0.5`` becomes
    ``float(0.5)``; everything else is returned as a string so the
    JSON emitted to the wire stays valid.
    """
    backtick, dquote, squote, number, bare = quoted
    if number:
        try:
            return json.loads(number)
        except (json.JSONDecodeError, ValueError):
            return number
    # Each alternation arm is either ``None`` (didn't match) or a
    # captured string (matched). Walk the priority order and return
    # the FIRST non-None piece.
    for piece in (backtick, dquote, squote, bare):
        if piece is not None:
            return piece
    return ""


def _try_prose_recover_tool_call(
    text: str, tools: list[dict]
) -> dict | None:
    r"""Recover a structured tool call from prose like
    ``"I should call the \`add\` tool with a=13 and b=29."`` (r5-E
    F-DGF-V080-B-8). Returns ``{"id","name","arguments"}`` on hit, or
    ``None`` on miss / no clear winner.

    Conservative gating (see the rationale block above
    ``GEMMA4_PROSE_KV_PATTERN``):

      * The text must mention a tool by its exact name (within a
        wrapper of optional backticks / single / double quotes).
      * The text must contain a ``key=value`` (or ``key: value``)
        assignment for every required parameter on that tool — a
        partial match is treated as ``None`` (model probably hadn't
        finished the call, or the parameters are unrelated).
      * On multiple candidate tools, take the one whose name appears
        first AND whose required-params are all matched in the
        window after the name. Ambiguity → return None.

    Returns the structured form the caller (``extract_tool_calls``)
    expects: ``{"id": <str>, "name": <str>, "arguments": <json str>}``.
    """
    if not tools or not text:
        return None
    text_lower = text.lower()
    candidates: list[tuple[int, dict, str, list[str]]] = []
    for tool in tools:
        fn = tool.get("function") if isinstance(tool, dict) else None
        if not isinstance(fn, dict):
            continue
        name = fn.get("name")
        if not isinstance(name, str) or not name:
            continue
        # Match the name as a standalone token (after optional
        # backticks / quotes). Don't use a plain substring search:
        # that would match a parameter called ``add`` inside an
        # unrelated tool's signature.
        # Allow surrounding ``\`add\```, ``"add"``, ``'add'``, or
        # the bare word at a word boundary.
        name_re = re.compile(
            r"(?:`|\"|')?" + re.escape(name) + r"(?:`|\"|')?\b",
            re.IGNORECASE,
        )
        match = name_re.search(text)
        if match is None:
            continue
        params = fn.get("parameters") if isinstance(fn, dict) else None
        required = []
        if isinstance(params, dict):
            req = params.get("required")
            if isinstance(req, list):
                required = [r for r in req if isinstance(r, str)]
        candidates.append((match.start(), fn, name, required))

    if not candidates:
        return None
    # Prefer the earliest tool-name mention. If two tools tie on
    # start position (unlikely — names would have to be identical),
    # the first one in ``tools`` wins.
    candidates.sort(key=lambda c: c[0])

    for name_start, fn, name, required in candidates:
        # Search a window AFTER the name mention up to the end of the
        # text (gemma4 prose is short — usually < 300 chars). Scanning
        # forward only ensures the "call X with args" ordering rather
        # than picking up an unrelated ``a=`` from an earlier sentence.
        window = text[name_start:]
        found: dict[str, Any] = {}
        for kv in GEMMA4_PROSE_KV_PATTERN.finditer(window):
            key = kv.group(1)
            value = _coerce_prose_value(kv.groups()[1:])
            # Skip captures whose "key" is actually the tool name
            # itself (``name=add``) — that's the name mention, not an
            # argument assignment.
            if key.lower() == name.lower():
                continue
            # First mention wins: a prose like "a=13" then later
            # "a=14" most likely the second is a correction; pick
            # the first, which matches the model's earliest stated
            # intent (the prose is the model's reasoning, so the
            # earliest commitment is the most reliable).
            found.setdefault(key, value)
        if required and not all(r in found for r in required):
            # Required params missing — not a confident recovery.
            continue
        if not found:
            continue
        return {
            "id": _generate_tool_id(),
            "name": name,
            "arguments": json.dumps(found),
        }
    return None


@ToolParserManager.register_module(["gemma4", "gemma_4"])
class Gemma4ToolParser(ToolParser):
    """
    Tool call parser for Gemma 4 models.

    Format: <|tool_call>call:func_name{key:<|"|>value<|"|>}<tool_call|>
    """

    EXPECTED_WIRE_FORMATS = ("gemma4_native", "calling_tool_text")

    def __init__(self, tokenizer=None):
        super().__init__(tokenizer)
        self._emitted_tool_count = 0

    def reset(self):
        """Reset state for a new request."""
        super().reset()
        self._emitted_tool_count = 0

    def has_pending_tool_call(self, text: str) -> bool:
        """A tool call is in flight as soon as we see the body opener
        ``call:NAME{`` — works for both the pristine wire form
        (``<|tool_call>call:NAME{...}<tool_call|>``) AND the
        post-HF-decode stripped form (``call:NAME{...}``). See the
        comment above ``GEMMA4_TOOL_PATTERN`` for why the wrappers can
        be absent.
        """
        if "<|tool_call>" in text:
            return True
        if re.search(r"call:\w+\{", text):
            return True
        return self.has_text_format_tool_call(text)

    def extract_tool_calls(
        self, model_output: str, request: Any = None
    ) -> ExtractedToolCallInformation:
        matches = list(GEMMA4_TOOL_PATTERN.finditer(model_output))

        if not matches:
            # r5-E F-DGF-V080-B-8: structured form missed. Try the
            # prose-fallback recovery before giving up — gemma4 at
            # low temperature intermittently describes the tool
            # intent in prose ("I should call the `add` tool with
            # a=13 and b=29.") instead of emitting the
            # ``<|tool_call>`` channel form. The recovery is gated
            # by ``request.tools`` so an unrelated chat that happens
            # to contain ``a=13`` prose cannot trigger a false
            # tool_call.
            recovered = _try_prose_recover_tool_call(
                model_output, _extract_request_tools(request)
            )
            if recovered is not None:
                return ExtractedToolCallInformation(
                    tools_called=True,
                    tool_calls=[recovered],
                    # Drop the prose content — keeping it would
                    # double-render the model's stated intent as
                    # both ``message.content`` and the tool call,
                    # surfacing as ``"I should call..."`` text +
                    # the call in the OpenAI response shape. That
                    # shape is wrong: the OpenAI spec is content OR
                    # tool_calls, not both verbatim. (The route
                    # layer ``parallel_tool_calls=false`` and
                    # ``finish_reason=tool_calls`` invariants both
                    # assume ``content`` is empty / falsy on a
                    # tool-call path.)
                    content=None,
                )
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        tool_calls = []
        for match in matches:
            func_name = match.group(1)
            args_str = match.group(2)
            args = _parse_gemma4_args(args_str)

            tool_calls.append(
                {
                    "id": _generate_tool_id(),
                    "name": func_name,
                    "arguments": json.dumps(args),
                }
            )

        # Content is everything outside the tool calls
        content = GEMMA4_TOOL_PATTERN.sub("", model_output).strip() or None

        return ExtractedToolCallInformation(
            tools_called=True, tool_calls=tool_calls, content=content
        )

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence = (),
        current_token_ids: Sequence = (),
        delta_token_ids: Sequence = (),
        request: dict[str, Any] | None = None,
    ) -> dict | None:
        # Check if we're inside a tool call. Either the pristine wire
        # form (``<|tool_call>...<tool_call|>``) or the post-HF-decode
        # stripped form (``call:NAME{...}``) triggers parsing — see the
        # comment above ``GEMMA4_TOOL_PATTERN`` for the empirical
        # justification.
        if "<|tool_call>" in current_text or re.search(r"call:\w+\{", current_text):
            # ``GEMMA4_TOOL_PATTERN`` matches completed bodies (it
            # requires the closing ``}`` and optionally the
            # ``<tool_call|>`` trailer). Count those as completed; if
            # the body opener appears more often than completed bodies,
            # we're still mid-stream and should suppress emission.
            completed_matches = list(GEMMA4_TOOL_PATTERN.finditer(current_text))
            completed = len(completed_matches)
            opener_re = re.compile(r"call:\w+\{")
            open_count = len(list(opener_re.finditer(current_text)))

            # Still accumulating an incomplete tool call
            if completed < open_count:
                return None  # suppress output while inside tool markup

            # Only emit newly completed tool calls (dedup)
            if completed <= self._emitted_tool_count:
                return None

            result = self.extract_tool_calls(current_text)
            if result.tools_called:
                # Only emit tool calls we haven't sent yet
                new_calls = result.tool_calls[self._emitted_tool_count :]
                self._emitted_tool_count = len(result.tool_calls)

                if new_calls:
                    return {
                        "tool_calls": [
                            {
                                "index": self._emitted_tool_count - len(new_calls) + i,
                                "id": tc["id"],
                                "type": "function",
                                "function": {
                                    "name": tc["name"],
                                    "arguments": tc["arguments"],
                                },
                            }
                            for i, tc in enumerate(new_calls)
                        ]
                    }

        # Text-format tool call recovery: catch [Calling tool: name({...})]
        # Models degrade to this format after multiple tool rounds at low quant
        from .abstract_tool_parser import TEXT_TOOL_CALL_ANY, TEXT_TOOL_CALL_FN_PATTERN

        if TEXT_TOOL_CALL_ANY.search(current_text):
            # Check if we have a complete text tool call
            matches = list(TEXT_TOOL_CALL_FN_PATTERN.finditer(current_text))
            new_matches = matches[self._emitted_tool_count :]
            if new_matches:
                self._emitted_tool_count = len(matches)
                return {
                    "tool_calls": [
                        {
                            "index": self._emitted_tool_count - len(new_matches) + i,
                            "id": _generate_tool_id(),
                            "type": "function",
                            "function": {
                                "name": m.group(1),
                                "arguments": m.group(2),
                            },
                        }
                        for i, m in enumerate(new_matches)
                    ]
                }
            # Already emitted or partial — suppress
            return None

        # No tool call markup — pass through as content
        return {"content": delta_text}
