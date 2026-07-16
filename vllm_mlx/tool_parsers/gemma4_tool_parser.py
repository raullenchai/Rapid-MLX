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
from dataclasses import dataclass
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
GEMMA4_TOOL_OPENER_PATTERN = re.compile(r"(?:<\|tool_call>)?call:(\w+)\{")
GEMMA4_TOOL_TRAILER = "<tool_call|>"

# Match a quoted-string value: <|"|>...<|"|>
GEMMA4_QUOTED_VAL_PATTERN = re.compile(r'<\|"\|>(.*?)<\|"\|>', re.DOTALL)
# Match a bare key:value pair (key, then anything up to , or end-of-string)
GEMMA4_KV_BARE_PATTERN = re.compile(r"(\w+)\s*:\s*([^,]+?)(?=\s*,|\s*$)")
# Bare (unquoted) argument key. Compiled once and matched against the full
# buffer with a start position so parsing an N-key object does not slice a
# fresh remaining-input string per key (quadratic allocation).
GEMMA4_KEY_PATTERN = re.compile(r"\w+")


@dataclass(frozen=True)
class _Gemma4ToolCallMatch:
    start: int
    end: int
    name: str
    arguments: str


def _scan_gemma4_tool_calls(
    text: str,
) -> tuple[list[_Gemma4ToolCallMatch], int]:
    """Find tool calls with balanced argument braces.

    A regex cannot distinguish the outer closing brace from a nested
    argument object's closing brace. Scan one call at a time instead, while
    ignoring braces inside Gemma quote tokens and JSON strings. The returned
    opener count lets the streaming path distinguish complete calls from a
    call that is still being generated.
    """
    matches: list[_Gemma4ToolCallMatch] = []
    opener_count = 0
    search_from = 0

    while opener := GEMMA4_TOOL_OPENER_PATTERN.search(text, search_from):
        opener_count += 1
        body_start = opener.end()
        depth = 1
        index = body_start
        in_gemma_string = False
        in_json_string = False
        escaped = False

        while index < len(text):
            char = text[index]
            if in_json_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_json_string = False
                index += 1
                continue
            if text.startswith('<|"|>', index):
                in_gemma_string = not in_gemma_string
                index += len('<|"|>')
                continue
            if in_gemma_string:
                index += 1
                continue
            if char == '"':
                in_json_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    call_end = index + 1
                    if text.startswith(GEMMA4_TOOL_TRAILER, call_end):
                        call_end += len(GEMMA4_TOOL_TRAILER)
                    matches.append(
                        _Gemma4ToolCallMatch(
                            start=opener.start(),
                            end=call_end,
                            name=opener.group(1),
                            arguments=text[body_start:index],
                        )
                    )
                    search_from = call_end
                    break
            index += 1
        else:
            # The final call is incomplete. Its nested contents cannot contain
            # another top-level call, so there is nothing useful left to scan.
            break

    return matches, opener_count


# Best-effort closer for a single opener body. Mirrors the historical
# ``GEMMA4_TOOL_PATTERN`` non-greedy ``call:NAME{(.*?)\}`` semantics: it
# takes the FIRST ``}`` after the opener as the body terminator, without
# tracking Gemma / JSON string state. That is exactly what recovers a
# malformed emission like ``call:f{x:<|"|>unterminated}`` where the
# balanced scanner stalls because the ``<|"|>`` quote is never closed and
# every subsequent ``}`` gets swallowed by the open-string state.
GEMMA4_BESTEFFORT_BODY_PATTERN = re.compile(r"(.*?)\}(?:<tool_call\|>)?", re.DOTALL)


def _recover_incomplete_gemma4_calls(
    text: str,
) -> list[_Gemma4ToolCallMatch]:
    """NON-STREAMING best-effort fallback for malformed tool calls.

    ``_scan_gemma4_tool_calls`` tracks Gemma quote (``<|"|>``) and JSON
    string state so nested ``{}`` inside string values do not close the
    call early. That is correct for well-formed output, but if the model
    emits an UNTERMINATED string (``call:f{x:<|"|>unterminated}`` — the
    closing ``<|"|>`` is missing) the scanner stays in the open-string
    state, swallows the trailing ``}``, and returns zero matches — so the
    whole call would be dropped to raw content.

    This is the historical, well-understood best-effort recovery: it mirrors
    the old ``call:NAME{(.*?)\\}`` non-greedy regex (the first ``}`` after the
    opener closes the body), tracking no string state. It runs ONLY on the
    non-streaming finalize path and ONLY when the balanced scanner found no
    complete call, so a malformed-but-terminated call is still surfaced
    instead of leaking as content.

    It deliberately does NOT try to perfectly reconstruct a mixed
    valid+malformed multi-call sequence: broken tool-call text only happens
    when the model itself emits corrupt output (rare), and it is not worth
    non-linear recovery complexity that historically introduced more bugs
    than the original truncation problem.

    It is NOT wired into the streaming path: a call that is genuinely still
    being generated (``call:f{x:<|"|>hel``) has no closing ``}`` yet, so this
    matcher finds nothing there and the stream stays pending, as before.
    """
    matches: list[_Gemma4ToolCallMatch] = []
    search_from = 0
    while opener := GEMMA4_TOOL_OPENER_PATTERN.search(text, search_from):
        body_start = opener.end()
        body = GEMMA4_BESTEFFORT_BODY_PATTERN.match(text, body_start)
        if not body:
            # No closing ``}`` anywhere after this opener → genuinely
            # incomplete, nothing to recover. Nested contents cannot hold
            # another top-level call, so stop scanning.
            break
        matches.append(
            _Gemma4ToolCallMatch(
                start=opener.start(),
                end=body.end(),
                name=opener.group(1),
                # Only the captured body (group 1) — excludes the closing
                # ``}`` and the optional ``<tool_call|>`` trailer.
                arguments=body.group(1),
            )
        )
        search_from = body.end()
    return matches


# Maximum object/array nesting the recursive argument parser will descend
# before bailing to the flat best-effort fallback. Gemma tool arguments are
# shallow in practice (a couple of nested objects for agent-to-agent calls),
# so 64 is comfortably above any legitimate depth while staying well under
# CPython's default recursion limit (~1000). A model that emits pathologically
# deep ``{a:{a:{...}}}`` output would otherwise recurse past the interpreter
# limit and raise an uncaught ``RecursionError`` that crashes request
# processing — this bound converts that into a controlled ``ValueError`` that
# degrades to the historical flat parser. (codex #1102 round-2 BLOCKING.)
_GEMMA4_MAX_NESTING_DEPTH = 64


class _Gemma4NestingTooDeepError(ValueError):
    """Raised when argument nesting exceeds ``_GEMMA4_MAX_NESTING_DEPTH``.

    Subclasses ``ValueError`` so every existing ``except ValueError`` handler
    (notably ``_parse_gemma4_args``) already routes it to the lenient
    best-effort fallback without a special case.
    """


class _Gemma4ArgumentParser:
    """Recursive parser for Gemma's JSON-like argument syntax."""

    def __init__(self, text: str):
        self.text = text
        self.index = 0
        # Current object/array nesting depth. Guarded against runaway
        # recursion so deeply-nested model output can never crash the request
        # with an uncaught ``RecursionError`` (codex #1102 round-2 BLOCKING).
        self.depth = 0

    def _enter_nesting(self) -> None:
        self.depth += 1
        if self.depth > _GEMMA4_MAX_NESTING_DEPTH:
            raise _Gemma4NestingTooDeepError(
                f"argument nesting exceeds {_GEMMA4_MAX_NESTING_DEPTH}"
            )

    def parse_arguments(self, terminator: str | None = None) -> dict[str, Any]:
        result: dict[str, Any] = {}
        self._skip_space()
        while self.index < len(self.text):
            if terminator and self.text[self.index] == terminator:
                self.index += 1
                return result
            key = self._parse_key()
            self._skip_space()
            self._expect(":")
            result[key] = self._parse_value()
            self._skip_space()
            if self.index >= len(self.text):
                break
            if terminator and self.text[self.index] == terminator:
                self.index += 1
                return result
            self._expect(",")
            self._skip_space()
        if terminator:
            raise ValueError(f"missing closing {terminator!r}")
        return result

    def _parse_key(self) -> str:
        self._skip_space()
        if self.index < len(self.text) and self.text[self.index] == '"':
            value = self._parse_json_string()
            if isinstance(value, str):
                return value
        # Match against the full buffer from ``self.index`` (no per-key
        # ``self.text[self.index:]`` copy → avoids quadratic allocation on
        # many-key objects). ``Pattern.match`` anchors at ``pos``.
        match = GEMMA4_KEY_PATTERN.match(self.text, self.index)
        if not match:
            raise ValueError("expected argument name")
        self.index = match.end()
        return match.group(0)

    def _parse_value(self) -> Any:
        self._skip_space()
        if self.text.startswith('<|"|>', self.index):
            return self._parse_gemma_string()
        if self.index >= len(self.text):
            raise ValueError("expected argument value")

        char = self.text[self.index]
        if char == "{":
            self.index += 1
            self._enter_nesting()
            try:
                return self.parse_arguments("}")
            finally:
                self.depth -= 1
        if char == "[":
            return self._parse_array()
        if char == '"':
            return self._parse_json_string()

        start = self.index
        while self.index < len(self.text) and self.text[self.index] not in ",}]":
            self.index += 1
        raw_value = self.text[start : self.index].strip()
        if not raw_value:
            raise ValueError("expected argument value")
        try:
            return json.loads(raw_value)
        except (json.JSONDecodeError, ValueError):
            return raw_value

    def _parse_array(self) -> list[Any]:
        self.index += 1
        self._enter_nesting()
        try:
            values: list[Any] = []
            self._skip_space()
            while self.index < len(self.text):
                if self.text[self.index] == "]":
                    self.index += 1
                    return values
                values.append(self._parse_value())
                self._skip_space()
                if self.index < len(self.text) and self.text[self.index] == "]":
                    self.index += 1
                    return values
                self._expect(",")
                self._skip_space()
            raise ValueError("missing closing ']'")
        finally:
            self.depth -= 1

    def _parse_gemma_string(self) -> str:
        marker = '<|"|>'
        self.index += len(marker)
        end = self.text.find(marker, self.index)
        if end < 0:
            raise ValueError("unterminated Gemma string")
        value = self.text[self.index : end]
        self.index = end + len(marker)
        return value

    def _parse_json_string(self) -> Any:
        # Decode from ``self.index`` in place. ``raw_decode(s, idx)`` anchors at
        # ``idx`` and returns the ABSOLUTE end offset, so we avoid slicing a
        # fresh ``self.text[self.index:]`` copy per JSON-quoted field — that
        # slice made parsing many quoted fields O(n²) in the buffer length.
        # (codex #1102 round-2 NIT.)
        value, end = json.JSONDecoder().raw_decode(self.text, self.index)
        self.index = end
        return value

    def _skip_space(self) -> None:
        while self.index < len(self.text) and self.text[self.index].isspace():
            self.index += 1

    def _expect(self, expected: str) -> None:
        if self.index >= len(self.text) or self.text[self.index] != expected:
            raise ValueError(f"expected {expected!r}")
        self.index += 1


# r5-E F-DGF-V080-B-8: prose-fallback recovery patterns. Gemma 4 at
# low temperature (~0.1) intermittently emits prose describing the
# tool intent ("I should call the `add` tool with a=13 and b=29.")
# instead of emitting the structured ``<|tool_call>call:NAME{...}<tool_call|>``
# wire form. Trace verdict (see commit body): NEITHER parser scan
# miss (the scanner above correctly catches every structured emission)
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

    Parse the JSON-like object recursively so nested objects and arrays retain
    their structure. If the model emits malformed syntax, fall back to the
    historical flat best-effort parser rather than dropping the entire call.
    """
    try:
        return _Gemma4ArgumentParser(args_str).parse_arguments()
    except (ValueError, json.JSONDecodeError, RecursionError):
        # Keep the historical best-effort behavior for malformed model output.
        # ``RecursionError`` is caught explicitly: the parser's own depth guard
        # converts pathologically nested Gemma objects/arrays into a
        # ``ValueError``, but a deeply-nested JSON string VALUE
        # (``x:"[[[[...]]]]"``) recurses inside the stdlib ``json`` decoder,
        # which has no such guard. Catching it here means deep model output can
        # never crash the request — it degrades to the flat best-effort parse
        # exactly like any other malformed input. (codex #1102 round-2 BLOCKING.)
        pass

    # Stash quoted string values so they can't confuse the fallback bare parser.
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
        # Try to parse as JSON literal (int, float, bool, null). Catch
        # ``RecursionError`` too: a deeply-nested bare value that survived the
        # quoted-string stash (``k:[[[[...]]]]``) would otherwise recurse past
        # the interpreter limit inside ``json.loads``. Falling back to the raw
        # string keeps this best-effort path crash-free. (codex #1102 round-2.)
        try:
            result[key] = json.loads(raw_val)
        except (json.JSONDecodeError, ValueError, RecursionError):
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


def _try_prose_recover_tool_call(text: str, tools: list[dict]) -> dict | None:
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
    # Carry per-mention: (name_start, fn, name, required[], allowed_keys_or_None)
    # We index by EVERY name occurrence (not just the first) so a
    # prose like "using the `add` tool. I should call the `add` tool
    # with a=13 and b=29." can recover from the SECOND mention even
    # when the first mention's bounded window has no key=value pairs.
    # ``allowed_keys_or_None`` is the set of declared parameter keys
    # from ``parameters.properties`` (None means "schema doesn't
    # declare properties — accept anything", which preserves
    # behaviour for tools defined with a free-form schema).
    candidates: list[tuple[int, dict, str, list[str], set[str] | None]] = []
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
        # Case-sensitive: OpenAI tool spec says function names are
        # case-sensitive identifiers, so a model writing ``ADD`` when
        # the registered tool is ``add`` is NOT a confident call.
        # (codex pr_validate NIT-5)
        # Leading negative lookbehind ``(?<![\w-])``: prevent a tool
        # named ``add`` from matching suffixes like ``foo_add`` or
        # ``my-add`` if their identifier text happens to sit near
        # unrelated ``a=`` / ``b=`` prose. (codex round-2 NIT.)
        name_re = re.compile(
            r"(?<![\w-])(?:`|\"|')?" + re.escape(name) + r"(?:`|\"|')?\b",
        )
        matches = list(name_re.finditer(text))
        if not matches:
            continue
        params = fn.get("parameters") if isinstance(fn, dict) else None
        required: list[str] = []
        allowed_keys: set[str] | None = None
        if isinstance(params, dict):
            req = params.get("required")
            if isinstance(req, list):
                required = [r for r in req if isinstance(r, str)]
            props = params.get("properties")
            if isinstance(props, dict):
                allowed_keys = {k for k in props if isinstance(k, str)}
        for match in matches:
            candidates.append((match.start(), fn, name, required, allowed_keys))

    if not candidates:
        return None
    # Try mentions in document order. If two tools tie on start
    # position (unlikely — names would have to be identical), the
    # first one in ``tools`` wins.
    candidates.sort(key=lambda c: c[0])

    # Window bound — codex pr_validate BLOCKING-1: scanning to
    # end-of-text turned multi-sentence answers into false positives.
    # 300 chars after the name mention covers any realistic gemma4
    # tool prose (e.g. "call `add` with a=13 and b=29") while
    # rejecting ramble that mentions ``add`` once and later
    # mentions an unrelated ``a=…`` many sentences later.
    PROSE_WINDOW_BYTES = 300

    # Helper — scan ``window`` for declared key=value pairs, dropping
    # the tool name itself and any key not in ``allowed_keys`` (when
    # the schema declares properties). Returns the recovered
    # ``{key: value}`` dict for THIS window.
    def _scan_window(window: str, name: str, allowed_keys: set[str] | None):
        found: dict[str, Any] = {}
        for kv in GEMMA4_PROSE_KV_PATTERN.finditer(window):
            key = kv.group(1)
            value = _coerce_prose_value(kv.groups()[1:])
            # Skip captures whose "key" is actually the tool name
            # itself (``name=add``) — that's the name mention, not
            # an argument assignment.
            if key.lower() == name.lower():
                continue
            # Drop keys not declared in the tool schema. (codex
            # pr_validate r1 BLOCKING-2: strict tool servers reject
            # unknown args, so silently filter here rather than
            # forward `result=42` from `call add with a=13,
            # result=42`.)
            if allowed_keys is not None and key not in allowed_keys:
                continue
            # First mention wins: a prose like "a=13" then later
            # "a=14" most likely the second is a correction; pick
            # the first, which matches the model's earliest stated
            # intent (the prose is the model's reasoning, so the
            # earliest commitment is the most reliable).
            found.setdefault(key, value)
        return found

    for name_start, fn, name, required, allowed_keys in candidates:
        # Search a BOUNDED window after the name mention, capped at
        # PROSE_WINDOW_BYTES. Forward-only scanning preserves the
        # "call X with args" ordering AND avoids dragging in
        # unrelated key=value pairs far from the call site.
        window_end = min(len(text), name_start + PROSE_WINDOW_BYTES)
        bounded = text[name_start:window_end]

        # Walk sentence boundaries within the bounded window. Start
        # with the first sentence; if required args are still missing
        # AND we are still under PROSE_WINDOW_BYTES, extend to the
        # next sentence boundary, up to a small max-sentence cap.
        # This handles the codex round-2 BLOCKING case
        # ``call add with a=13. b=29.``: the first sentence has only
        # ``a=13`` and we need to peek into the next sentence to find
        # ``b=29``. Capping at 3 sentences keeps the conservative
        # behaviour: "..mentioned add. Long unrelated paragraph.
        # ...a=42, b=99." still won't recover because the boundary
        # walk stops well before the later sentence.
        MAX_SENTENCES = 3
        sentence_ends = [m.end() for m in re.finditer(r"[.!?](?:\s|$)", bounded)]
        # Always include the full bounded window as the final
        # fallback so a single-sentence prose without trailing
        # punctuation (``call add with a=13 and b=29``) still works.
        if not sentence_ends or sentence_ends[-1] < len(bounded):
            sentence_ends.append(len(bounded))
        # Cap.
        sentence_ends = sentence_ends[:MAX_SENTENCES]

        found: dict[str, Any] = {}
        for window_end_idx in sentence_ends:
            window = bounded[:window_end_idx]
            found = _scan_window(window, name, allowed_keys)
            # If all required args have been collected — accept now.
            if required and all(r in found for r in required):
                break
            # If no required args (schema doesn't declare any) and we
            # found at least one key — accept (the prose did commit
            # to a key=value, that's enough for a free-form tool).
            if not required and found:
                break

        if required and not all(r in found for r in required):
            # Required params missing across ALL sentence-expansion
            # attempts — not a confident recovery on THIS mention.
            # Continue to later mentions (e.g. the canonical "using
            # the `add` tool. I should call the `add` tool with
            # a=13 and b=29." has TWO `add` mentions; the second one
            # has the args, so we skip the first and succeed on the
            # second).
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
        comment above ``GEMMA4_TOOL_OPENER_PATTERN`` for why the
        wrappers can be absent.
        """
        if "<|tool_call>" in text:
            return True
        if re.search(r"call:\w+\{", text):
            return True
        return self.has_text_format_tool_call(text)

    def extract_tool_calls(
        self,
        model_output: str,
        request: Any = None,
        *,
        _allow_recovery: bool = True,
    ) -> ExtractedToolCallInformation:
        # ``_allow_recovery`` gates every best-effort recovery path (malformed
        # tool-call closing and prose recovery). It is ``True`` on the
        # NON-STREAMING finalize path — generation is known complete, so it is
        # safe to close a malformed call. The STREAMING path passes ``False``
        # (see ``extract_tool_calls_streaming``): a call that cannot be closed
        # by the balanced scanner might still be mid-generation, so streaming
        # must keep it pending instead of force-emitting a best-effort call.
        matches, _opener_count = _scan_gemma4_tool_calls(model_output)

        if not matches and _allow_recovery:
            # The balanced scanner found no complete call. On the
            # NON-STREAMING finalize path generation is known complete, so a
            # malformed-but-terminated call (e.g. an unterminated ``<|"|>`` /
            # JSON string that keeps the scanner stuck in the open-string
            # state and swallows the trailing ``}``) is recovered with the
            # historical best-effort parser (first ``}`` closes the body)
            # rather than being dropped to raw content. This is deliberately
            # a simple, conservative fallback — it does not attempt to
            # perfectly reconstruct mixed valid+malformed multi-call output.
            #
            # Streaming is untouched: it passes ``_allow_recovery=False``, so
            # a still-generating call stays pending.
            matches = _recover_incomplete_gemma4_calls(model_output)

        if not matches and _allow_recovery:
            # r5-E F-DGF-V080-B-8: structured form missed. Try the
            # prose-fallback recovery before giving up — gemma4 at
            # low temperature intermittently describes the tool
            # intent in prose ("I should call the `add` tool with
            # a=13 and b=29.") instead of emitting the
            # ``<|tool_call>`` channel form. The recovery is gated
            # by ``request.tools`` so an unrelated chat that happens
            # to contain ``a=13`` prose cannot trigger a false
            # tool_call. Also gated on ``_allow_recovery`` so the
            # streaming path never force-recovers mid-generation.
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
            args = _parse_gemma4_args(match.arguments)

            tool_calls.append(
                {
                    "id": _generate_tool_id(),
                    "name": match.name,
                    "arguments": json.dumps(args),
                }
            )

        # Content is everything outside the tool calls
        content_parts: list[str] = []
        content_start = 0
        for match in matches:
            content_parts.append(model_output[content_start : match.start])
            content_start = match.end
        content_parts.append(model_output[content_start:])
        content = "".join(content_parts).strip() or None

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
        # comment above ``GEMMA4_TOOL_OPENER_PATTERN`` for the empirical
        # justification.
        if "<|tool_call>" in current_text or re.search(r"call:\w+\{", current_text):
            # The balanced scanner returns completed bodies and the number of
            # openers seen. If there are more openers than completed calls,
            # the final call is still mid-stream and must be suppressed.
            completed_matches, open_count = _scan_gemma4_tool_calls(current_text)
            completed = len(completed_matches)

            # Still accumulating an incomplete tool call
            if completed < open_count:
                return None  # suppress output while inside tool markup

            # Only emit newly completed tool calls (dedup)
            if completed <= self._emitted_tool_count:
                return None

            # ``_allow_recovery=False``: a call the balanced scanner could not
            # close might still be mid-generation on the stream, so we must
            # never force-emit a best-effort recovery here — an incomplete or
            # malformed call stays pending until generation finalizes.
            result = self.extract_tool_calls(current_text, _allow_recovery=False)
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
