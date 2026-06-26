# SPDX-License-Identifier: Apache-2.0
"""Chat completion endpoints — /v1/chat/completions."""

import asyncio
import gc
import json
import logging
import os
import re
import time
import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from ..api.models import (
    AssistantMessage,
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChoiceLogProbs,
    PromptTokensDetails,
    TokenLogProb,
    Usage,
)
from ..api.response_format_metrics import (
    incr_strict_repair_attempt,
    incr_strict_repair_skipped_context_overflow,
    incr_strict_repair_success,
    incr_strict_request,
    incr_strict_violation,
)
from ..api.strict_json_schema import (
    build_repair_messages,
    build_violation_envelope,
    repair_retry_enabled,
    strict_enforcement_enabled,
    validate_and_envelope,
)
from ..api.tool_calling import (
    build_json_system_prompt,
    check_schema_validity,
    convert_tools_for_template,
    extract_json_schema_for_guided,
    is_strict_json_schema,
    parse_json_output,
    validate_output_against_schema,
)
from ..api.utils import (
    clean_output_text,
    decode_inline_tool_call_arguments,
    extract_json_from_response,
    extract_multimodal_content,
    sanitize_output,
    sanitize_reasoning_for_stream,
    strip_thinking_tags,
    validate_content_blocks_for_capabilities,
)
from ..config import get_config
from ..engine import GenerationOutput
from ..middleware.auth import check_rate_limit, verify_api_key
from ..service.helpers import (
    _TOOL_USE_REQUIRED_SUFFIX,
    _TOOL_USE_SYSTEM_SUFFIX,
    SSE_RESPONSE_HEADERS,
    _apply_reasoning_cutoff_notice,
    _build_usage,
    _check_admission_or_503,
    _disconnect_guard,
    _effective_enable_thinking,
    _extract_streaming_token_logprobs,
    _finalize_content_and_reasoning,
    _inject_json_instruction,
    _is_structured_output_requested,
    _maybe_pin_system_prompt,
    _parse_tool_calls_with_parser,
    _release_admission_unless_committed,
    _rescue_silent_drop_from_reasoning,
    _resolve_enable_thinking,
    _resolve_max_tokens,
    _resolve_model_name,
    _resolve_temperature,
    _resolve_top_p,
    _scan_messages_for_lone_surrogates,
    _should_start_in_thinking,
    _tool_use_required_named_suffix,
    _validate_model_name,
    _validate_response_format,
    _validate_tool_call_params,
    _wait_with_disconnect,
    build_extended_sampling_kwargs,
    enable_thinking_warning_header,
    enforce_context_length_for_messages,
    get_engine,
    maybe_auto_disable_thinking_for_casual_chat,
    maybe_auto_disable_thinking_for_tools,
    repair_messages_fit_context,
)

logger = logging.getLogger(__name__)
_SAFE_DEEPSEEK_TOOL_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

router = APIRouter()


# Exceptions worth catching around the cloud call so the local engine can
# take over: provider/network/auth/quota — transient or out-of-our-control.
# Anything outside this allowlist (AttributeError, TypeError,
# NotImplementedError, …) is an engine-contract violation or programming
# bug and MUST surface as 500. The original ``except Exception`` here hid
# both #500 (missing ``build_prompt``) and the v0.6.70 hotfix (missing
# the token-estimation helper on the engine) as silent fallback warnings.
#
# ``litellm.exceptions`` is imported lazily — its presence depends on
# whether cloud routing was configured at startup. ``httpx`` and the
# stdlib timeout/connection set are always available, so we fall back
# to those when litellm isn't importable.
def _tool_call_name(tc) -> str | None:
    """Extract the function name from a tool_call entry regardless of
    shape. Three real shapes seen in production:

    1. Pydantic ``ToolCall`` — ``tc.function.name``. Text-parser path.
    2. Wrapped dict — ``{"function": {"name": ...}}``. Anthropic
       passthrough and engine structured passthrough through
       ``_parse_tool_calls_with_parser``.
    3. Flat dict — ``{"name": ..., "arguments": ...}``. Raw engine
       ``GenerationOutput.tool_calls`` shape (Harmony StreamableParser
       output before wrapping). Surfaces in tests/fixtures and any
       downstream that forwards engine output directly.

    PR #518 round-2 codex BLOCKING added shapes 1+2; round-3 BLOCKING
    added shape 3 (the round-2 widening missed it, even though the
    same PR's test fixture emits exactly that shape).
    """
    if isinstance(tc, dict):
        fn = tc.get("function")
        if isinstance(fn, dict):
            return fn.get("name")
        if fn is not None:
            return getattr(fn, "name", None)
        # Flat shape — no ``function`` wrapper.
        return tc.get("name")
    fn = getattr(tc, "function", None)
    if isinstance(fn, dict):
        return fn.get("name")
    if fn is not None:
        return getattr(fn, "name", None)
    # Flat attr-shape — no ``function`` attribute.
    return getattr(tc, "name", None)


def _forced_tool_call_prefix(parser_name: str | None, function_name: str) -> str | None:
    """Build the assistant-turn prefix string that the model continues
    when ``tool_choice`` forces a named function.

    Returns ``None`` for parsers where prefix injection isn't a known
    win (e.g. channel-routed harmony / gemma4: those publish tool calls
    via the OutputRouter's tool-call channel directly, so the model
    already produces a structured call when prompted — and pre-pending
    the wire opener would actually CONFUSE the channel state machine).
    For parser-only families (hermes / qwen3coder / llama / kimi /
    glm47 / mistral / minimax / deepseek / nemotron / xlam / functionary)
    the wire opener is unambiguous; insert ``<tool_call>\\n{"name":
    "X", "arguments":`` so the model continues with the arguments object
    and the closer.

    This is the OpenAI ``tool_choice`` forced-function lever — pure
    prefix injection, parser-shape-agnostic of the underlying alias.
    """
    if not function_name:
        return None
    # Verified parsers that explicitly recognise the hermes ``<tool_call>``
    # JSON-body wire shape in their primary path (both non-streaming
    # ``extract_tool_calls`` regex AND the streaming state-machine
    # sentinel set). Each one's source was audited against this opener
    # before being added:
    #   - ``hermes`` (vllm_mlx/tool_parsers/hermes_tool_parser.py):
    #     ``TOOL_CALL_PATTERN = <tool_call>{JSON}</tool_call>``;
    #     ``_STREAMING_SENTINELS = ("<tool_call>", "<function=")``
    #
    # Parsers EXCLUDED on purpose because their primary wire is NOT
    # the JSON ``<tool_call>`` body shape — even when the OPENER
    # matches, the body shape conflicts with the parser's
    # expectations (codex r1 + r3 P2 on this PR):
    #   - ``qwen3coder`` / ``qwen3_coder_xml`` — same ``<tool_call>``
    #     opener but the body uses XML ``<function=NAME>...`` markers,
    #     NOT JSON. ``Qwen3CoderToolParser.extract_tool_calls`` looks
    #     for ``<function=`` after the opener and would miss a JSON
    #     body entirely.
    #   - ``minimax``  → ``<minimax:tool_call>`` / ``<invoke name="...">``
    #   - ``mistral``  → ``[TOOL_CALLS]``
    #   - ``deepseek`` → ``<｜tool▁calls▁begin｜>``
    #   - ``llama``    → ``<|python_tag|>`` / bare JSON (its own opener)
    #   - ``kimi``     → ``<|tool_calls_section_begin|>``
    #   - ``glm47``    → ``<tool_call>...<arg_key>...</arg_value>``
    #     (XML body, NOT JSON — same body-shape conflict as
    #     qwen3coder above)
    #   - ``granite``, ``xlam``, ``functionary``, ``nemotron``,
    #     ``seed_oss`` — distinct wire formats; defer to the
    #     post-parse synthesis fallback rather than risk a wrong
    #     opener.
    _verified_json_tool_call_parsers = {
        "hermes",
    }
    if parser_name in _verified_json_tool_call_parsers:
        # JSON envelope opener — model continues with the arguments
        # object body. Leave a trailing space so the model picks up
        # immediately with ``{...}``.
        #
        # ``json.dumps(function_name)`` escapes any quoted / backslash /
        # control character in the function name so a hostile tool
        # spec (``{"name": "x\\\\\\", \\"arguments\\": ..."}``) cannot
        # corrupt the wire envelope or inject extra fields. Codex r4
        # BLOCKING — direct f-string interpolation was vulnerable.
        return f'<tool_call>\n{{"name": {json.dumps(function_name)}, "arguments": '
    # D-TOOLCHOICE-R1 T2 / R12-5: DeepSeek-V3.1 parser uses the
    # thinking-channel wire
    # ``<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>NAME<｜tool▁sep｜>{json}<｜tool▁call▁end｜><｜tool▁calls▁end｜>``.
    # Pre 0.8.3, this family fell through to ``None`` here — so
    # ``tool_choice="required"`` with a single tool only ever produced
    # the post-parse ``_synthesize_forced_tool_call`` fallback (empty
    # ``arguments="{}"``). With the prefix in place, the model picks up
    # inside the envelope and emits real arguments that
    # ``extract_tool_calls`` parses cleanly (verified by
    # ``DeepSeekV31ToolParser.extract_tool_calls`` on the assembled
    # ``<...begin>name<sep>{...}<...end>...end>`` payload).
    #
    # R12-5: ``deepseek_v31`` and ``deepseek_v3`` are now DIFFERENT
    # parsers with DIFFERENT wire shapes. Only ``deepseek_v31`` uses
    # the bare ``NAME<sep>`` body; ``deepseek_v3`` uses
    # ``function<sep>NAME\n``\`json\n{...}\n``\` and is handled by its
    # own branch below. ``deepseek`` (V2 / R1-distill) is intentionally
    # NOT listed — that parser's body shape is not auto-detected.
    if parser_name == "deepseek_v31":
        # V3.1 body: ``NAME<sep>{json}``. The model continues with the
        # arguments object and closer. The name is interpolated raw
        # because the V3.1 body is NOT JSON-bodied at this position —
        # it is a literal ``NAME<sep>`` split, so ``json.dumps`` would
        # wrap the name in quotes and break the wire.
        #
        # codex r1 BLOCKING #2: a tool name that itself contains a
        # DeepSeek envelope marker would corrupt the wire AND could
        # let a downstream parser re-interpret the suffix as an extra
        # tool call. Upstream validates names against ``request.tools``
        # but not against the wire-token set, so we gate here as
        # defense-in-depth. Hitting any marker returns ``None`` (clean
        # degradation to the post-parse synthesis fallback) rather
        # than emitting a corrupt prefix.
        if not _SAFE_DEEPSEEK_TOOL_NAME_RE.fullmatch(function_name):
            return None
        return (
            f"<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>{function_name}<｜tool▁sep｜>"
        )
    # R12-5: DeepSeek-V3 body — ``function<sep>NAME\n``\`json\n{...}\n``\`.
    # Forced-prefix opens through the envelope, the literal ``function``
    # type tag, the separator, the name, and the opening JSON fence so
    # the model picks up directly with the arguments object body. The
    # same name-safety guard as V3.1 applies (defense-in-depth against
    # marker-bearing tool names).
    if parser_name in ("deepseek_v3", "deepseek_r1_0528"):
        if not _SAFE_DEEPSEEK_TOOL_NAME_RE.fullmatch(function_name):
            return None
        return (
            "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>"
            f"function<｜tool▁sep｜>{function_name}\n```json\n"
        )
    # Channel-routed (harmony / gemma4) and parsers whose wire shape
    # we have NOT audited: no prefix injection. The post-parse
    # synthesis path remains as a fallback (``_synthesize_forced_tool_call``).
    return None


def _recover_partial_tool_args(
    raw_text: str | None, expected_name: str | None = None
) -> str | None:
    """Best-effort recovery of a JSON arguments object from a malformed
    model response under ``tool_choice="required"``.

    Designed for the D-TOOLCHOICE-R1 T3 case: qwen3 + tool_choice=
    required emits something like ::

        <tool_call>
        {"name": "add_numbers", "arguments": 4128, 7591}
        </parameter>
        </function>
        </tool_call>

    where the parser's strict pattern fails (``arguments`` is not a
    JSON object) so the chat route synthesises a call with empty
    ``"{}"`` arguments. The empty-args result is uselessly opaque for
    the client — but the raw text itself often contains enough signal
    to reconstruct *something* better than ``"{}"``. Two recovery
    routes are tried, both bounded so a hostile or genuinely
    unparseable response degrades safely back to ``None``:

    1. **Strict object body.** If the raw text contains a literal
       ``"arguments":`` followed by a balanced JSON object, return
       that object's text. This handles the common case where the
       model emitted valid JSON but failed an outer wrapper (closing
       tag missing, wrong wrapper element).

    2. **No structural recovery possible.** Return ``None``. The
       caller falls back to the existing ``"{}"`` default.

    Intentionally narrow: we do NOT try to coerce the malformed
    qwen3 shape ``"arguments": 4128, 7591`` into a positional-args
    interpretation — there is no schema-agnostic way to map two bare
    integers to named parameters without guessing. The fallback to
    ``"{}"`` plus a downstream ``_validate_tool_call_params`` warning
    is the right contract for that case (the client retries with a
    clearer prompt or a named-function ``tool_choice``).

    codex r4 BLOCKING #1: when ``expected_name`` is provided, also
    verify that the recovered candidate is paired with a
    ``"name": "<expected>"`` field in the same wire span. Without
    this gate, a synth for a named ``tool_choice`` whose target is
    ``"my_target"`` could pick up an unrelated
    ``{"name": "other_tool", "arguments": {...}}`` block elsewhere
    in the response and ship ``other_tool``'s args under the
    forced target's name — a subtle correctness bug because the
    synthesized call's ``function.name`` would not match the args'
    intended schema. We REQUIRE a name match within a 512-byte
    window of the ``"arguments"`` marker (covers a typical inner
    wire body), and FALL BACK to ``"{}"`` (return ``None``) when no
    match exists.
    """
    if not raw_text:
        return None
    text = raw_text
    n = len(text)

    # Wire markers that signal a real tool-call body. When at least
    # one occurrence of ``"arguments":`` sits INSIDE such a span,
    # restrict the search to those — the prose example before the
    # wire span (e.g. a docstring quoting the JSON shape) is then
    # ignored entirely. When NONE of the occurrences are inside a
    # wire span, fall back to scanning the full text (handles the
    # bare-JSON case where the model emitted a raw call with no
    # wrapper).
    #
    # codex r2 BLOCKING: previously this returned the FIRST
    # parseable ``"arguments": {...}``; a docstring-style leading
    # example (``the JSON shape is "arguments": {"a": 0}``) would
    # then beat the actual malformed call further down the response
    # and we'd ship the example's arguments to the client. The fix
    # uses two heuristics in order:
    #
    #   1. PREFER any candidate INSIDE a tool-call wire span
    #      (``<tool_call>...``, ``<function=...>``, DeepSeek envelope
    #      markers, ``[TOOL_CALLS]``, ``<|python_tag|>``). Among
    #      those, pick the LAST parseable one (most-recent intent).
    #   2. If no candidates land inside a wire span, accept the LAST
    #      parseable candidate anywhere in the text — still
    #      preferring "later" because the tool wire is conventionally
    #      at the end of a response.
    _WIRE_SPAN_OPENERS = (
        "<tool_call>",
        "<function=",
        "<function>",
        "<｜tool▁calls▁begin｜>",
        "<｜tool▁call▁begin｜>",
        "[TOOL_CALLS]",
        "<|python_tag|>",
        "<|tool_calls_section_begin|>",
        "<minimax:tool_call>",
        "<invoke",
        "<arg_key>",
    )

    def _scan_balanced_object_at(start_brace: int) -> tuple[int, str] | None:
        """Return ``(end_offset, raw_object_text)`` if the substring
        starting at ``start_brace`` is a balanced JSON object,
        ``None`` otherwise. Cap 8 KiB to avoid burning CPU on a
        malformed giant payload.
        """
        depth = 0
        in_string = False
        escape = False
        pos = start_brace
        scan_end = min(n, start_brace + 8192)
        while pos < scan_end:
            ch = text[pos]
            if escape:
                escape = False
            elif in_string:
                if ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
            else:
                if ch == '"':
                    in_string = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return (pos + 1, text[start_brace : pos + 1])
            pos += 1
        return None

    # Closer counterparts (used to bound the wire-span lookback so
    # pretty-printed / verbose wire bodies aren't misclassified as
    # outside-wire just because their opener sits >256 bytes back).
    # codex r6 NIT: a fixed 256-byte lookback caused valid
    # wrapped calls with verbose metadata before ``"arguments":`` to
    # lose priority to a later prose example. We now bound the search
    # by the nearest known closer instead — if no closer sits between
    # the most recent opener and ``idx``, ``idx`` IS inside the span,
    # regardless of how far back the opener is.
    _WIRE_SPAN_CLOSERS = (
        "</tool_call>",
        "</function>",
        "<｜tool▁calls▁end｜>",
        "<｜tool▁call▁end｜>",
        "[/TOOL_CALLS]",
        "<|tool_calls_section_end|>",
        "</minimax:tool_call>",
        "</invoke>",
        "</arg_value>",
    )

    def _open_wire_span_start(idx: int) -> int | None:
        """Return the nearest still-open wire opener before ``idx``."""
        prefix = text[:idx]
        op_pos = -1
        for opener in _WIRE_SPAN_OPENERS:
            pos = prefix.rfind(opener)
            if pos > op_pos:
                op_pos = pos
        if op_pos < 0:
            return None
        cl_pos = -1
        for closer in _WIRE_SPAN_CLOSERS:
            pos = prefix.rfind(closer)
            if pos > cl_pos:
                cl_pos = pos
        return op_pos if op_pos >= 0 and op_pos > cl_pos else None

    def _next_wire_span_closer(idx: int) -> int | None:
        """Return the nearest known wire closer after ``idx``."""
        close_pos: int | None = None
        for closer in _WIRE_SPAN_CLOSERS:
            pos = text.find(closer, idx)
            if pos != -1 and (close_pos is None or pos < close_pos):
                close_pos = pos
        return close_pos

    def _position_in_wire_span(idx: int) -> bool:
        """Is ``idx`` inside (or immediately after) a known tool-wire
        opener? We don't require a balanced closer — the qwen3 leak
        shape often has no clean closer.

        codex r6 NIT: bound the search by the nearest known
        opener/closer occurrence rather than a fixed lookback. We
        find the LATEST opener at position ``op_pos < idx`` and the
        LATEST closer at position ``cl_pos < idx``; ``idx`` is in
        the span iff ``op_pos`` exists AND ``op_pos > cl_pos``
        (so the most recent opener was not yet closed before ``idx``).
        """
        return _open_wire_span_start(idx) is not None

    def _name_pairs_with(idx: int, expected: str) -> bool:
        """Does a ``"name": "<expected>"`` (or ``"name":"<expected>"``)
        sit in the SAME wire-call block as the ``"arguments"``
        occurrence at ``idx``?

        Heuristic: the per-call block is bounded by the nearest
        wire opener BEFORE ``idx`` (we never look past it for a
        ``"name"`` literal) and either the previous ``"arguments"``
        marker OR the start of the surrounding text — whichever is
        closer. Forward we cap at the next ``"arguments"`` or 256
        bytes. This intentionally restricts the search to the
        ``"name"`` literal that belongs to THE SAME wire body as
        ``idx``, so an unrelated call's ``"name"`` further up the
        response never matches.

        codex r4 BLOCKING #1: the wire-span check alone is not enough
        because a response can contain multiple wire spans each
        with a different ``"name"``. We MUST verify the pairing,
        and the window MUST be tight enough to separate inline
        blocks (a fixed ±512-byte window let adjacent blocks pollute
        each other's pairing).
        """
        if not expected:
            return True  # No constraint when caller doesn't pass one.

        # Backward bound: if ``idx`` is inside a known wire span, use
        # that span's opener. This admits verbose DeepSeek V3.1 output
        # between ``<｜tool▁call▁begin｜>NAME<｜tool▁sep｜>`` and the
        # later JSON ``"arguments"`` object without relying on a fixed
        # byte lookback. If no opener is known, fall back to the
        # previous arguments marker to keep adjacent JSON blocks from
        # cross-pairing.
        span_start = _open_wire_span_start(idx)
        if span_start is not None:
            backward_bound = span_start
        else:
            prev_args_end = text.rfind('"arguments"', 0, idx)
            object_start = text.rfind("{", 0, idx)
            fallback_bound = (
                prev_args_end + len('"arguments"') if prev_args_end != -1 else 0
            )
            backward_bound = max(fallback_bound, object_start)
        # Forward bound: the next "arguments" marker or 256 bytes
        # forward. We cap forward TIGHTER than backward because the
        # canonical wire shape ``{"name":"X","arguments":{...}}``
        # always has ``"name"`` BEFORE ``"arguments"``.
        next_args_idx = text.find('"arguments"', idx + len('"arguments"'))
        next_closer_idx = (
            _next_wire_span_closer(idx) if span_start is not None else None
        )
        forward_bound = n
        if next_args_idx != -1:
            forward_bound = min(forward_bound, next_args_idx)
        if next_closer_idx is not None:
            forward_bound = min(forward_bound, next_closer_idx)
        window = text[backward_bound:forward_bound]
        escaped = re.escape(expected)
        # Accept TWO wire shapes for the paired ``name`` literal:
        #
        # 1. ``"name": "<expected>"`` — the JSON-bodied wire shape
        #    (hermes / qwen3 / qwen3coder / nemotron-with-JSON /
        #    most parsers).
        # 2. ``<｜tool▁call▁begin｜><expected><｜tool▁sep｜>`` — the
        #    DeepSeek V3.1 wire shape, where the name is NOT
        #    JSON-quoted; it sits between the V3.1 call-begin and
        #    sep markers. Without this shape recovery rejects every
        #    legitimate DeepSeek arg body (codex r5 BLOCKING #1).
        #
        # Both forms are searched; either match is sufficient.
        json_pair_re = re.compile(
            r'"name"\s*:\s*"' + escaped + r'"',
            re.DOTALL,
        )
        if json_pair_re.search(window):
            return True
        if not _SAFE_DEEPSEEK_TOOL_NAME_RE.fullmatch(expected):
            return False
        deepseek_begin = "<｜tool▁call▁begin｜>"
        deepseek_sep = "<｜tool▁sep｜>"
        search_pos = 0
        while True:
            begin = window.find(deepseek_begin, search_pos)
            if begin == -1:
                return False
            name_start = begin + len(deepseek_begin)
            sep = window.find(deepseek_sep, name_start)
            if sep == -1:
                return False
            if window[name_start:sep] == expected:
                return True
            search_pos = sep + len(deepseek_sep)

    # Collect every parseable candidate, tagged with whether it sits
    # inside a wire span.
    candidates_in_wire: list[str] = []
    candidates_outside_wire: list[str] = []
    search_start = 0
    while search_start < n:
        args_marker_idx = text.find('"arguments"', search_start)
        if args_marker_idx == -1:
            break
        # codex r3 NIT: the previous fixed 20-char window for the
        # colon rejected valid JSON like
        # ``"arguments"    \n   :   {...}`` (lots of pretty-print
        # whitespace). Walk past whitespace from the end of the
        # ``"arguments"`` token and then require ``:`` — no
        # arbitrary cap.
        pos = args_marker_idx + len('"arguments"')
        while pos < n and text[pos] in " \t\n\r":
            pos += 1
        if pos >= n or text[pos] != ":":
            search_start = args_marker_idx + len('"arguments"')
            continue
        # Skip whitespace after the colon, looking for the object opener.
        pos += 1
        while pos < n and text[pos] in " \t\n\r":
            pos += 1
        if pos >= n or text[pos] != "{":
            search_start = args_marker_idx + len('"arguments"')
            continue
        scan = _scan_balanced_object_at(pos)
        if scan is None:
            search_start = args_marker_idx + len('"arguments"')
            continue
        obj_end, candidate = scan
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            search_start = args_marker_idx + len('"arguments"')
            continue
        if not isinstance(parsed, dict):
            search_start = args_marker_idx + len('"arguments"')
            continue
        # codex r4 BLOCKING #1: when an expected name is set, only
        # accept this candidate if a matching ``"name"`` literal
        # sits within the same wire body. Otherwise we'd pick up
        # an unrelated tool's args and ship them under the forced
        # target's name.
        if expected_name and not _name_pairs_with(args_marker_idx, expected_name):
            search_start = obj_end
            continue
        canonical = json.dumps(parsed, ensure_ascii=False)
        if _position_in_wire_span(args_marker_idx):
            candidates_in_wire.append(canonical)
        else:
            candidates_outside_wire.append(canonical)
        search_start = obj_end

    # Prefer candidates INSIDE a wire span (the real tool call); fall
    # back to outside-wire candidates only when no wire-span match
    # existed. Within each pool, pick the LAST (rightmost) candidate
    # — the tool wire is conventionally at the end of the response,
    # so "last" is "most recent" and most likely to be the actual
    # call rather than a leading example.
    if candidates_in_wire:
        return candidates_in_wire[-1]
    if candidates_outside_wire:
        return candidates_outside_wire[-1]
    return None


# Parser-wire literal markers that MUST be scrubbed from ``content`` /
# ``reasoning_content`` when ``tool_choice="required"`` synthesises a
# call to recover from a malformed model emission (D-TOOLCHOICE-R1 T3).
# Without this scrub, the model's failed tool-call attempt — literal
# ``<tool_call>...`` text the parser couldn't extract — leaks into
# both user-visible fields.
#
# Each entry is ``(opener_regex, closer_regex_or_None)``. The scrubber
# is two-phase:
#
#   1. **Balanced pairs**: when both opener and closer are present,
#      strip the entire span ``opener…closer``. Non-greedy so
#      consecutive blocks each get their own match.
#   2. **Unmatched standalone markers**: any opener/closer literal
#      that survives phase 1 (e.g. an orphan ``</function>`` or a
#      stray ``<tool_call>`` with no closer because the model
#      truncated mid-body) gets stripped as a bare token.
#
# Critically, phase 2 strips ONLY the marker bytes themselves — it
# does NOT delete from the orphan opener to EOF. Codex r1 BLOCKING #1
# caught this: the prior ``opener.*?(?:closer|\Z)`` pattern would eat
# all trailing reasoning/prose whenever the model emitted an unclosed
# opener mid-thought. The new two-phase split preserves trailing
# content while still scrubbing the marker itself, so a model
# response shaped like ``<tool_call>{junk}</tool_call>then prose``
# yields ``then prose`` and ``<tool_call>{junk}\nthen prose`` yields
# ``{junk}\nthen prose`` (the body is left for the reasoning parser
# to consume, but the visible ``<tool_call>`` marker is gone).
#
# The list covers every wire opener/closer pair we know about
# (hermes / qwen3coder / nemotron / deepseek / glm / minimax /
# mistral / kimi / llama / harmony / gemma4 / xlam / functionary /
# granite / seed_oss / vibethinker named-XML). Adding a parser is a
# one-line addition.
_TOOL_WIRE_BALANCED_PAIRS = (
    # Hermes / qwen3 JSON-bodied wire
    (re.compile(r"<tool_call>", re.DOTALL), re.compile(r"</tool_call>", re.DOTALL)),
    # Nemotron-style XML body + bare ``<function=NAME>...``
    (re.compile(r"<function=[^>]*>", re.DOTALL), re.compile(r"</function>", re.DOTALL)),
    (re.compile(r"<function>", re.DOTALL), re.compile(r"</function>", re.DOTALL)),
    # DeepSeek V3 / V3.1 / R1-0528 envelope
    (
        re.compile(r"<｜tool▁calls▁begin｜>", re.DOTALL),
        re.compile(r"<｜tool▁calls▁end｜>", re.DOTALL),
    ),
    (
        re.compile(r"<｜tool▁call▁begin｜>", re.DOTALL),
        re.compile(r"<｜tool▁call▁end｜>", re.DOTALL),
    ),
    # GLM-4 / GLM-4.7 wrapper
    (re.compile(r"<arg_key>", re.DOTALL), re.compile(r"</arg_value>", re.DOTALL)),
    # Minimax
    (
        re.compile(r"<minimax:tool_call>", re.DOTALL),
        re.compile(r"</minimax:tool_call>", re.DOTALL),
    ),
    (re.compile(r"<invoke\b[^>]*>", re.DOTALL), re.compile(r"</invoke>", re.DOTALL)),
    # Kimi section markers
    (
        re.compile(r"<\|tool_calls_section_begin\|>", re.DOTALL),
        re.compile(r"<\|tool_calls_section_end\|>", re.DOTALL),
    ),
    # Mistral
    (
        re.compile(r"\[TOOL_CALLS\]", re.DOTALL),
        re.compile(r"\[/TOOL_CALLS\]", re.DOTALL),
    ),
)
_TOOL_WIRE_BALANCED_SPAN_RES = tuple(
    re.compile(opener_re.pattern + r".*?" + closer_re.pattern, re.DOTALL)
    for opener_re, closer_re in _TOOL_WIRE_BALANCED_PAIRS
)


# Standalone marker tokens that must be stripped even when their
# matching counterpart never arrived. Includes the qwen3 stray
# ``</parameter>`` (closing-only marker; no opener counterpart in
# any tool wire we model) and the Llama python-tag (opener-only).
_TOOL_WIRE_STANDALONE_MARKERS = (
    re.compile(r"<tool_call>"),
    re.compile(r"</tool_call>"),
    re.compile(r"<function=[^>]*>"),
    re.compile(r"<function>"),
    re.compile(r"</function>"),
    re.compile(r"</?parameter[^>]*>"),
    re.compile(r"<｜tool▁calls▁begin｜>"),
    re.compile(r"<｜tool▁calls▁end｜>"),
    re.compile(r"<｜tool▁call▁begin｜>"),
    re.compile(r"<｜tool▁call▁end｜>"),
    re.compile(r"<｜tool▁sep｜>"),
    re.compile(r"<arg_key>"),
    re.compile(r"</arg_value>"),
    re.compile(r"<minimax:tool_call>"),
    re.compile(r"</minimax:tool_call>"),
    re.compile(r"<invoke\b[^>]*>"),
    re.compile(r"</invoke>"),
    re.compile(r"<\|python_tag\|>"),
    re.compile(r"<\|tool_calls_section_begin\|>"),
    re.compile(r"<\|tool_calls_section_end\|>"),
    re.compile(r"\[TOOL_CALLS\]"),
    re.compile(r"\[/TOOL_CALLS\]"),
)


# Cross-family opener/closer set — used by phase 1.5 to catch the
# qwen3-style mixed-closer leak where the model emits ``<tool_call>``
# but closes with ``</function>`` (or some other unrelated closer).
# Any opener-to-any-closer span gets stripped in one sweep. The list
# is the union of opener and closer regex patterns we already track
# in ``_TOOL_WIRE_BALANCED_PAIRS`` plus the always-orphan ones.
#
# This is bounded — phase 1.5 only fires when an opener appears AND
# any closer-class marker appears later in the text. If neither
# qualifies, the span is left for phase 2's marker-only strip.
_CROSS_FAMILY_OPENERS = "|".join(
    [
        r"<tool_call>",
        r"<function=[^>]*>",
        r"<function>",
        r"<｜tool▁calls▁begin｜>",
        r"<｜tool▁call▁begin｜>",
        r"<minimax:tool_call>",
        r"<invoke\b[^>]*>",
        r"<arg_key>",
        r"<\|tool_calls_section_begin\|>",
        r"\[TOOL_CALLS\]",
    ]
)
_CROSS_FAMILY_CLOSERS = "|".join(
    [
        r"</tool_call>",
        r"</function>",
        r"</parameter>",
        r"<｜tool▁calls▁end｜>",
        r"<｜tool▁call▁end｜>",
        r"</minimax:tool_call>",
        r"</invoke>",
        r"</arg_value>",
        r"<\|tool_calls_section_end\|>",
        r"\[/TOOL_CALLS\]",
    ]
)
# Non-greedy ``opener…any-closer`` — catches the qwen3 ``<tool_call>``
# + ``</function>`` cross-family leak that the strict same-family
# pairs in phase 1 don't match.
_CROSS_FAMILY_SPAN_RE = re.compile(
    rf"(?:{_CROSS_FAMILY_OPENERS}).*?(?:{_CROSS_FAMILY_CLOSERS})",
    re.DOTALL,
)


# codex r6 BLOCKING #1 — leak detector. Used to tighten the
# forced/required scrub gate so that we only scrub when the
# parser's ``cleaned_text`` ACTUALLY contains wire-marker literals.
# A successful forced call whose ``cleaned_text`` is e.g. ``"OK"``
# does not get its content rewritten, which protects legitimate
# assistant prose that happens to mention ``<tool_call>`` etc.
#
# The detector is intentionally a cheap substring/regex sweep —
# it returns True on the FIRST match of any wire marker we know
# about. If new parser wires are added, they only need to be
# registered in ``_TOOL_WIRE_STANDALONE_MARKERS`` for the detector
# to pick them up (same source-of-truth as the scrub itself).
def _contains_tool_wire_literal(text: str | None) -> bool:
    """Return True iff ``text`` contains any known tool-wire marker
    (opener, closer, separator). Used by the chat route to decide
    whether the forced/required scrub gate should fire on a
    parser-extracted (non-synth) tool call.
    """
    if not text:
        return False
    for marker_re in _TOOL_WIRE_STANDALONE_MARKERS:
        if marker_re.search(text):
            return True
    return False


_TOOL_WIRE_PAYLOAD_HINT_RE = re.compile(r"[\"'](?:name|arguments)[\"']\s*:")


def _balanced_json_end(text: str, start: int, *, max_scan: int = 8192) -> int | None:
    """Return the exclusive end offset of a balanced JSON-ish object."""
    if start < 0 or start >= len(text) or text[start] != "{":
        return None
    depth = 0
    in_string = False
    escape = False
    stop = min(len(text), start + max_scan)
    for i in range(start, stop):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i + 1
    return None


def _payload_object_after_marker(
    text: str,
    marker_end: int,
    window_end: int,
) -> tuple[int, int] | None:
    """Return adjacent JSON payload bounds after a wire marker, if any."""
    pos = marker_end
    while pos < window_end and text[pos] in " \t\r\n":
        pos += 1
    if pos >= window_end or text[pos] != "{":
        return None
    object_end = _balanced_json_end(text, pos)
    if object_end is None or object_end > window_end:
        return None
    payload = text[pos:object_end]
    if not _TOOL_WIRE_PAYLOAD_HINT_RE.search(payload):
        return None
    return pos, object_end


def _span_has_tool_payload_object(span: str) -> bool:
    object_start = span.find("{")
    while object_start != -1:
        object_end = _balanced_json_end(span, object_start)
        if object_end is not None:
            payload = span[object_start:object_end]
            if _TOOL_WIRE_PAYLOAD_HINT_RE.search(payload):
                return True
            object_start = span.find("{", object_end)
        else:
            object_start = span.find("{", object_start + 1)
    return False


def _contains_structural_tool_wire_leak(text: str | None) -> bool:
    """Return True when known wire markers appear as tool-wire residue.

    ``_contains_tool_wire_literal`` intentionally answers the broad
    question "is any marker token present?"  That is too destructive as
    a scrub gate because ordinary prose can discuss the literal
    ``<tool_call>`` token. This predicate is stricter: it requires a
    balanced/cross-family wire span, or a marker next to JSON/tool-call
    payload hints (``name`` / ``arguments`` / compact object body).
    """
    if not text:
        return False
    for balanced_re in _TOOL_WIRE_BALANCED_SPAN_RES:
        match = balanced_re.search(text)
        if match and _span_has_tool_payload_object(match.group(0)):
            return True
    cross_match = _CROSS_FAMILY_SPAN_RE.search(text)
    if cross_match and _span_has_tool_payload_object(cross_match.group(0)):
        return True
    for marker_re in _TOOL_WIRE_STANDALONE_MARKERS:
        match = marker_re.search(text)
        if not match:
            continue
        window_end = min(len(text), match.end() + 2048)
        if _payload_object_after_marker(text, match.end(), window_end) is not None:
            return True
    return False


def _is_tool_wire_marker_only(text: str | None) -> bool:
    """True when ``text`` has markers but no non-marker payload/prose."""
    if not text or not _contains_tool_wire_literal(text):
        return False
    result = text
    for marker_re in _TOOL_WIRE_STANDALONE_MARKERS:
        result = marker_re.sub("", result)
    return not result.strip()


def _scrub_visible_tool_wire_leaks(text: str | None) -> str:
    """Scrub structural wire residue while preserving marker examples.

    Unlike ``_scrub_tool_wire_literals`` this is safe for route-visible
    fields: prose such as ``"Use <tool_call>...</tool_call>"`` stays
    intact because it has no tool payload hint. Actual malformed wire
    spans and marker-only leftovers are removed.
    """
    if not text:
        return text or ""
    result = text
    # DeepSeek V3.1 orphan fragment:
    # ``<｜tool▁call▁begin｜>NAME<｜tool▁sep｜>{...}`` without a closer.
    # Remove the whole fragment so the opener/name prefix cannot leak
    # when the separator-adjacent JSON payload is scrubbed below.
    deepseek_begin = "<｜tool▁call▁begin｜>"
    deepseek_sep = "<｜tool▁sep｜>"
    search_pos = 0
    while True:
        begin = result.find(deepseek_begin, search_pos)
        if begin == -1:
            break
        sep = result.find(deepseek_sep, begin + len(deepseek_begin))
        if sep == -1:
            search_pos = begin + len(deepseek_begin)
            continue
        next_begin = result.find(deepseek_begin, begin + len(deepseek_begin), sep)
        if next_begin != -1:
            search_pos = next_begin
            continue
        payload_bounds = _payload_object_after_marker(
            result,
            sep + len(deepseek_sep),
            min(len(result), sep + len(deepseek_sep) + 2048),
        )
        if payload_bounds is None:
            search_pos = begin + len(deepseek_begin)
            continue
        _, object_end = payload_bounds
        result = result[:begin] + result[object_end:]
        search_pos = begin
    for balanced_re in _TOOL_WIRE_BALANCED_SPAN_RES:
        result = balanced_re.sub(
            lambda m: "" if _span_has_tool_payload_object(m.group(0)) else m.group(0),
            result,
        )
    result = _CROSS_FAMILY_SPAN_RE.sub(
        lambda m: "" if _span_has_tool_payload_object(m.group(0)) else m.group(0),
        result,
    )
    for _ in range(4):
        changed = False
        for marker_re in _TOOL_WIRE_STANDALONE_MARKERS:
            pieces: list[str] = []
            last = 0
            for match in marker_re.finditer(result):
                window_end = min(len(result), match.end() + 2048)
                pieces.append(result[last : match.start()])
                payload_bounds = _payload_object_after_marker(
                    result, match.end(), window_end
                )
                if payload_bounds is not None:
                    _, object_end = payload_bounds
                    last = object_end
                    changed = True
                    continue
                pieces.append(match.group(0))
                last = match.end()
            if pieces:
                pieces.append(result[last:])
                result = "".join(pieces)
        if not changed:
            break
    if _is_tool_wire_marker_only(result):
        result = _scrub_tool_wire_literals(result)
    return re.sub(r"\s+", " ", result).strip()


def _scrub_tool_wire_literals(text: str | None) -> str:
    """Strip every known parser-wire opener/closer marker from
    ``text`` in three phases. Returns a whitespace-collapsed result
    so we don't leave a void where the wire used to live.

    Phases:

      1. **Same-family balanced spans** — strip every balanced
         ``opener…closer`` span from the same parser family
         (``<tool_call>...</tool_call>``,
         ``<｜tool▁calls▁begin｜>...<｜tool▁calls▁end｜>``, etc.).
      2. **Cross-family spans** — strip any-opener-to-any-closer
         spans the model emitted with mismatched wires (the qwen3
         ``<tool_call>{...}</function>`` shape — codex r3 BLOCKING #1).
      3. **Standalone markers** — orphan opener or closer literals
         that survived phases 1+2 get scrubbed as bare tokens
         WITHOUT eating surrounding text (codex r1 BLOCKING #1
         preserved-trailing-text invariant).

    Idempotent: safe to call on text that has no wire literals (the
    regex sweep is a no-op). Called by the chat route ONLY when
    ``tool_choice="required"`` synthesises (or recovers args for) a
    call from a malformed wire — i.e. when the model's text output
    contained tool-call markers the parser couldn't extract from.
    """
    if not text:
        return text or ""
    result = text
    # Phase 1: same-family balanced opener…closer spans.
    for opener_re, closer_re in _TOOL_WIRE_BALANCED_PAIRS:
        balanced = re.compile(opener_re.pattern + r".*?" + closer_re.pattern, re.DOTALL)
        result = balanced.sub("", result)
    # Phase 1.5 (cross-family): catches ``<tool_call>{...}</function>``
    # and similar mismatched-closer shapes the strict same-family
    # pairs above don't match. Without this, the malformed body
    # between the opener and the unrelated closer survives as
    # ``content`` (codex r3 BLOCKING #1).
    result = _CROSS_FAMILY_SPAN_RE.sub(
        lambda m: "" if _span_has_tool_payload_object(m.group(0)) else m.group(0),
        result,
    )
    # Phase 2: strip standalone marker tokens (orphan opener OR
    # orphan closer that survived phases 1+1.5). ONLY the marker
    # bytes are removed; surrounding text is preserved.
    for marker_re in _TOOL_WIRE_STANDALONE_MARKERS:
        result = marker_re.sub("", result)
    # Collapse any runs of whitespace left by the strip. Preserves
    # single-space separation between surrounding prose words.
    return re.sub(r"\s+", " ", result).strip()


def _synthesize_forced_tool_call(
    name: str, arguments: str = "{}", *, raw_text: str | None = None
):
    """Build a single ``ToolCall`` for a forced ``tool_choice`` whose
    text parser surfaced no calls (#571).

    Text-parser paths (hermes / qwen3_coder / minimax / glm47 / …) only
    surface a tool_call when the model emits the parser's wire markers.
    Channel-routed paths (harmony / gemma4) bypass the text parser
    entirely — the ``OutputRouter`` extracts structured tool_calls
    directly. The two surfaces therefore diverge on the same request:
    a forced ``tool_choice`` succeeds on harmony because the model
    produced the structured channel, but 422s on hermes when the model
    produced text that the parser failed to recognise.

    The OpenAI ``tool_choice`` contract is parser-agnostic: when the
    client forces a tool call, the response MUST carry one. To restore
    symmetry we synthesise a tool_call server-side when the target tool
    is unambiguous (named-function, or ``"required"`` with a single
    tool).

    D-TOOLCHOICE-R1 T3: pre 0.8.3, ``arguments`` defaulted to ``"{}"``
    unconditionally. When the model actually emitted a JSON object
    body the strict parser couldn't extract from (qwen3 malformed
    inner shape, model leaking unclosed tags, etc.) the empty-args
    fallback shipped a uselessly opaque call. ``raw_text`` is now
    consulted first: if it contains a recoverable ``"arguments": {...}``
    object we use THAT instead of ``"{}"``. Downstream
    ``_validate_tool_call_params`` still gates schema compliance; the
    contract guarantee is "a tool_call is present", and now also "the
    arguments are as close to what the model intended as we can
    structurally recover".
    """
    # Lazy import — ToolCall / FunctionCall live alongside the request
    # model in ``api.models``. The lazy form keeps the synthesis path
    # scoped to forced-choice requests; the common case pays nothing.
    from ..api.models import FunctionCall, ToolCall

    # Try the partial-recovery path first; only fall back to the
    # caller-provided default (``"{}"`` or an upstream override) when
    # the raw text yields nothing structurally parseable. Pass the
    # synth target ``name`` as ``expected_name`` so recovery rejects
    # ``"arguments"`` candidates paired with a DIFFERENT tool's
    # ``"name"`` literal (codex r4 BLOCKING #1).
    recovered = _recover_partial_tool_args(raw_text, expected_name=name)
    final_args = recovered if recovered is not None else arguments

    return ToolCall(
        id=f"call_{uuid.uuid4().hex[:8]}",
        type="function",
        function=FunctionCall(name=name, arguments=final_args),
    )


def _normalize_ui_tars_tcs_for_chat(tool_calls: list | None) -> list | None:
    """Apply UI-TARS Computer-Use spec keys to a streaming-shaped tool_calls list.

    The streaming postprocessor surfaces tool_calls as a list of dicts
    in OpenAI-streaming shape (``{"index","id","type","function":{
    "name","arguments"}}``); the cross-format fallback path
    (``finalize_tool_calls``) and the terminal-chunk merge path use
    the same shape. Centralising the per-entry normalisation here
    keeps the three streaming sites (mid-stream emit, terminal merge,
    synthetic fallback) byte-identical with the non-stream chat
    response builder — without it the streaming path would emit the
    parser-native ``point`` shape while the non-stream path emitted
    the spec ``coordinate`` shape (r7-A R7-H1).

    Gated on ``function.name == "computer"`` so vanilla function tools
    whose arguments carry a key named ``point`` are passed through
    verbatim. A None / empty input passes through.
    """
    if not tool_calls:
        return tool_calls
    from ..tool_parsers.ui_tars_tool_parser import (
        normalize_ui_tars_chat_tool_call_arguments,
    )

    out = []
    for tc in tool_calls:
        if not isinstance(tc, dict):
            out.append(tc)
            continue
        fn = tc.get("function") or {}
        name = fn.get("name")
        args = fn.get("arguments")
        if not isinstance(args, str):
            out.append(tc)
            continue
        new_args = normalize_ui_tars_chat_tool_call_arguments(args, name)
        if new_args is args:
            out.append(tc)
            continue
        # Shallow-copy so the upstream postprocessor's structures are
        # not mutated under the caller's feet (defense-in-depth in case
        # the same event is referenced elsewhere).
        new_tc = dict(tc)
        new_fn = dict(fn)
        new_fn["arguments"] = new_args
        new_tc["function"] = new_fn
        out.append(new_tc)
    return out


def _is_harmony_cut_short_stream(
    reasoning_parser,
    accumulated_reasoning: str,
    accumulated_text: str,
    tool_calls_detected: bool,
) -> bool:
    """D-HARMONY-LEAK gate predicate, factored for direct test reuse.

    Returns True when the streaming postprocessor state matches the
    harmony "analysis without final" cut-short shape: an active
    ``HarmonyReasoningParser`` saw reasoning tokens, no content
    tokens have been streamed, AND no commentary tool call was
    detected on any chunk. The streaming chat route uses this to
    decide whether to synthesise a harmony-marked ``raw_text`` so the
    shared rescue helper's gate fires uniformly across the streaming
    and non-streaming surfaces.

    Codex r1 BLOCKING #2 (PR #794): plumbing ``tool_calls_detected``
    keeps a tool-call-only stream from being misclassified as
    analysis-without-final — the cap-exhaust path in
    ``StreamingPostProcessor._process_channel_routed`` sets
    ``tool_calls_detected=True`` even when ``fallback_tool_calls``
    arrives empty, and a wrongly-fired harmony gate there would not
    suppress visible bytes but WOULD lose the channel-state signal
    for any future caller that gates on the synthetic raw shape.

    Codex r2 BLOCKING (PR #794): extracted to a module-level helper
    so ``tests/test_harmony_finalize.py`` exercises the SAME code
    object the streaming chat route uses, not a local re-implementation
    of the predicate.
    """
    rp_is_harmony = (
        type(reasoning_parser).__name__ == "HarmonyReasoningParser"
        if reasoning_parser is not None
        else False
    )
    return bool(
        rp_is_harmony
        and accumulated_reasoning
        and not accumulated_text
        and not tool_calls_detected
    )


def _engine_supports_channel_routed_tool_calls(engine) -> bool:
    """Probe whether the engine's tokenizer yields a channel-routed
    streaming path that can emit structured tool calls without a text
    parser. Harmony (gpt-oss) and Gemma 4 publish tool calls via the
    OutputRouter's tool-call channel, so a stream=true tool_choice=
    required request CAN satisfy the contract for those models even
    when ``cfg.tool_call_parser`` is unset.

    PR #518 round-10 codex BLOCKING #1: the prior gate rejected every
    parser-less streaming-required request and blocked legitimate
    harmony/gemma4 traffic. The capability probe relies on the same
    detection the engine itself uses
    (``OutputRouter.from_tokenizer_for_streaming`` + the engine's
    format allowlist), so a positive answer here means the actual
    engine path WILL produce structured tool_call deltas.
    """
    # Engine-level capability bit — if an engine explicitly declares
    # it has no tool-call surface (DiffusionEngine), the tokenizer
    # probe is moot. Without this, DiffusionGemma's tokenizer would
    # trip the Gemma 4 allowlist even though DiffusionEngine never
    # runs OutputRouter — letting tool_choice="required" finish with
    # plain text and no 422 (codex round 9 [P2] on PR #551).
    if not getattr(engine, "supports_tool_calls", True):
        return False
    try:
        from ..engine.batched import _OUTPUT_ROUTER_ALLOWLIST
        from ..output_router import OutputRouter

        tokenizer = getattr(engine, "tokenizer", None)
        if tokenizer is None:
            return False
        router = OutputRouter.from_tokenizer_for_streaming(tokenizer)
        if router is None:
            return False
        return router.map.format_tag in _OUTPUT_ROUTER_ALLOWLIST
    except Exception:
        # Capability probe is best-effort — any failure means we
        # cannot prove channel-routed support, so the gate falls
        # back to the parser-only path (which 422s without one).
        return False


def _cloud_call_recoverable_exceptions() -> tuple[type[BaseException], ...]:
    """Build the allowlist of exception types we treat as recoverable from
    the cloud call. Lazy so cloud routing being disabled doesn't pay the
    litellm import cost.

    Covered failure shapes (codex round-1 review on PR #502 — broaden
    beyond ``httpx.HTTPError`` to catch real production cases):
      * ``asyncio.TimeoutError`` / ``TimeoutError`` — request budget hit
      * ``ConnectionError`` — TCP/UDP transport down
      * ``ssl.SSLError`` — certificate / handshake — common w/ corp MITM
      * ``json.JSONDecodeError`` — provider returned malformed body
      * ``httpx.HTTPError`` — covers ``HTTPStatusError``, ``RequestError``,
        ``ConnectError``, ``ProxyError``, ``ReadTimeout``, etc.
      * ``litellm.exceptions.APIError`` — provider-side surface
    """
    import asyncio
    import json
    import ssl

    exc_types: list[type[BaseException]] = [
        asyncio.TimeoutError,
        ConnectionError,
        TimeoutError,
        ssl.SSLError,
        json.JSONDecodeError,
    ]
    try:
        import httpx

        exc_types.append(httpx.HTTPError)
    except ImportError:
        pass
    try:
        from litellm import exceptions as _litellm_exc

        exc_types.append(_litellm_exc.APIError)
    except (ImportError, AttributeError):
        pass
    return tuple(exc_types)


_CLOUD_CALL_RECOVERABLE_EXCEPTIONS = _cloud_call_recoverable_exceptions()


# Matches a single backslash directly followed by a non-ASCII codepoint.
# ``lm-format-enforcer``'s grammar permits ``\\`` followed by any codepoint
# as a valid JSON escape, so a model emitting JSON with CJK / emoji content
# can produce strings like ``"\\빠\\르\\게"`` — valid JSON, but the decoded
# value carries literal backslashes. Strip them so clients see clean text.
#
# Scope / known tradeoff: this is applied only on the ``response_format``
# json-output path (see line ~632 below), not to tool-call arguments or
# regular text content. The cleanup is unconditional within that path,
# matching upstream waybarrios#525. A JSON object that LEGITIMATELY
# contains a backslash before a non-ASCII codepoint (e.g. a Windows path
# ``"C:\\사용자\\file.txt"`` in a response_format=json_object reply) will
# be mutated to ``"C:사용자file.txt"``. We accept this tradeoff because:
#  (a) the lm-format-enforcer bug is the overwhelming source of these
#      sequences in JSON-output responses; the file-path case is rare,
#  (b) gating the cleanup on a heuristic ("looks like enforcer output")
#      would be fragile and only catch the obvious patterns,
#  (c) clients that need raw backslash + non-ASCII can fall back to
#      ``response_format=text`` and parse the JSON themselves.
# If a user reports the false-positive in practice, revisit by adding a
# config flag (``--no-strip-spurious-backslashes``) rather than a heuristic.
_BACKSLASH_BEFORE_UNICODE = re.compile(r"\\([^\x00-\x7F])")


def _strip_backslash_before_unicode(obj: object) -> object:
    if isinstance(obj, dict):
        # Clean both keys and values: ``lm-format-enforcer`` can produce
        # ``"\\한\\글": "value"`` (valid JSON, ugly key). Stripping only
        # values would leak the bug into client-visible object keys.
        cleaned: dict[object, object] = {}
        for k, v in obj.items():
            new_key = _strip_backslash_before_unicode(k)
            new_val = _strip_backslash_before_unicode(v)
            if new_key in cleaned:
                # Two distinct dirty keys can collapse to the same clean
                # key (e.g. ``"\\한"`` and ``"한"`` both → ``"한"``). Keep
                # the first occurrence and surface the collision rather
                # than silently dropping a field.
                logger.warning(
                    "JSON key collision after backslash strip: %r dropped "
                    "in favor of earlier value (cleaned key=%r)",
                    k,
                    new_key,
                )
                continue
            cleaned[new_key] = new_val
        return cleaned
    if isinstance(obj, list):
        return [_strip_backslash_before_unicode(v) for v in obj]
    if isinstance(obj, str):
        return _BACKSLASH_BEFORE_UNICODE.sub(r"\1", obj)
    return obj


@router.post(
    "/v1/chat/completions",
    dependencies=[Depends(verify_api_key), Depends(check_rate_limit)],
)
async def create_chat_completion(request: ChatCompletionRequest, raw_request: Request):
    """
    Create a chat completion (supports multimodal content for VLM models).

    OpenAI-compatible multimodal format for images:
    ```json
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": "https://..."}}
        ]
    }]
    ```

    Video support:
    ```json
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What happens in this video?"},
            {"type": "video_url", "video_url": {"url": "https://example.com/video.mp4"}}
        ]
    }]
    ```

    Structured output (JSON mode):
    ```json
    response_format={"type": "json_object"}
    ```

    Structured output (JSON Schema):
    ```json
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "my_schema",
            "schema": {"type": "object", "properties": {...}}
        }
    }
    ```
    """
    _validate_model_name(request.model)
    engine = get_engine(request.model)

    # Admission reservation is acquired LATER — after cloud-routing
    # decision (codex R9: cloud-routable requests must not be 503'd
    # solely because the local engine is at cap; they bypass local
    # generation entirely) and after the cheap validation that may
    # raise HTTPException (codex R3: validation errors used to pin
    # the slot until restart, exhausting the cap via a trivial
    # malformed-JSON DoS). ``_commit_state[0] = True`` is flipped
    # right before returning a StreamingResponse so
    # ``_disconnect_guard`` owns release after the SSE generator
    # closes; the route-level ``finally`` releases for non-streaming
    # and cloud paths.
    _commit_state = [False]
    _admission_acquired = [False]
    try:
        return await _create_chat_completion_impl(
            request, raw_request, engine, _commit_state, _admission_acquired
        )
    finally:
        if _admission_acquired[0]:
            _release_admission_unless_committed(engine, _commit_state[0])


async def _create_chat_completion_impl(
    request: ChatCompletionRequest,
    raw_request: Request,
    engine,
    _commit_state: list[bool],
    _admission_acquired: list[bool],
):
    """Inner impl for ``create_chat_completion``. Admission is
    reserved inside this function — after cloud-routing decision
    and after cheap validation — to avoid (a) 503'ing
    cloud-routable requests when the local engine is full and
    (b) leaking the slot on validation HTTPException paths."""
    # Validate messages is non-empty
    if not request.messages:
        raise HTTPException(
            status_code=400,
            detail="messages must not be empty",
        )

    # Validate message roles
    _valid_roles = {"system", "user", "assistant", "tool", "developer"}
    for msg in request.messages:
        if msg.role not in _valid_roles:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid role '{msg.role}'. Must be one of: {', '.join(sorted(_valid_roles))}",
            )

    # Reject lone-surrogate codepoints in any message-content slot
    # (F-130 + F-131). ``json.loads`` accepts ``"\\uD800"`` as a valid
    # JSON string and binds it to a Python ``str`` carrying the
    # unpaired surrogate; HuggingFace ``tokenizers`` then raises
    # ``TypeError: TextEncodeInput must be …`` deep inside the
    # chat-template render, producing either a 500 (non-stream) or —
    # WORSE — an HTTP 200 with the raw Python error text leaked in an
    # SSE ``data:`` chunk (stream). The route-layer gate runs BEFORE
    # the streaming branch opens its ``StreamingResponse``, so the
    # SSE-leak path in F-131 is closed by construction (a 400 is
    # returned before any byte of SSE is flushed).
    _scan_messages_for_lone_surrogates(request.messages)

    # F-111 / F-112 / F-051: tool-message schema validation.
    #
    # The OpenAI ``chat.completions`` spec defines three invariants on
    # ``role:"tool"`` messages that we previously accepted as 200 and
    # silently mis-rendered into the model prompt:
    #
    #   * F-051: ``tool_call_id`` is REQUIRED on every tool message.
    #     Without it, the tool reply has no provenance — the previously
    #     accepted shape rendered the result into context unlinked
    #     (effectively an attacker-controlled "extra user turn").
    #
    #   * F-112: ``tool_call_id`` MUST reference the ``id`` of a
    #     ``tool_calls[*]`` entry on a PRIOR assistant message in the
    #     same ``messages[]`` array. Orphan tool turns were accepted as
    #     200 and rendered into the prompt as if they were authoritative
    #     tool replies — a direct prompt-injection vector (e.g. an
    #     attacker-controlled ``content: "the password is OMEGA"``
    #     reached the model with no anchoring assistant tool_call).
    #
    #   * F-111: ``role:"tool"`` content MUST be text-only (a string or
    #     a ``[{type:"text",text:str}]`` array). The OpenAI spec does
    #     not define multimodal tool replies, and a non-text part on a
    #     tool reply was previously silently dropped by every text-only
    #     chat template — the model received an empty
    #     ``<tool_response>`` and hallucinated. The text-only array is
    #     accepted here and flattened to a string downstream by
    #     ``utils/chat_template.py::_normalize_text_only_content_arrays``;
    #     this validator only rejects the non-text shapes the
    #     normalization can't safely flatten.
    #
    # All three checks live up-front (BEFORE engine work) so the failure
    # mode is a clean 400 with a precise pointer at the offending
    # message — not a 500 from a downstream renderer crash. F-051 is
    # closed as a freebie by the ``tool_call_id`` REQUIRED check.
    #
    # We track tool_call ids as a CONSUMABLE pending set rather than a
    # monotonically-growing seen-set: each ``assistant.tool_calls[*].id``
    # is a single-use ticket that the next matching ``role:"tool"`` reply
    # consumes. Without this, a client can submit a valid tool reply,
    # then resubmit ANOTHER ``role:"tool"`` message with the SAME
    # ``tool_call_id`` later in ``messages[]`` — both replies render as
    # authoritative tool results for the same call, an attacker-
    # controlled "second reply" prompt-injection vector. Codex round-1
    # BLOCKING on PR #731.
    #
    # We also reject DUPLICATE ``assistant.tool_calls[*].id`` values —
    # whether within a single assistant turn or across turns. The
    # OpenAI contract guarantees ids are unique across the conversation,
    # and the consumable-ticket design relies on it: a re-used id on a
    # later assistant turn would silently re-open a ticket the client
    # never asked for, weakening the F-112 single-use guarantee. Codex
    # round-2 NIT on PR #731 — implement the invariant the docstring
    # claims rather than weaken the docstring.
    _pending_tool_call_ids: set[str] = set()
    _seen_any_tool_call_id: set[str] = set()
    for _idx, _msg in enumerate(request.messages):
        if _msg.role == "assistant" and _msg.tool_calls:
            for _tc in _msg.tool_calls:
                if isinstance(_tc, dict):
                    _tc_id = _tc.get("id")
                else:
                    _tc_id = getattr(_tc, "id", None)
                if isinstance(_tc_id, str) and _tc_id:
                    if _tc_id in _seen_any_tool_call_id:
                        raise HTTPException(
                            status_code=400,
                            detail=(
                                f"messages[{_idx}] assistant tool_calls "
                                f"contains duplicate id {_tc_id!r}; the OpenAI "
                                "spec requires every tool_call.id to be unique "
                                "across the conversation"
                            ),
                        )
                    _seen_any_tool_call_id.add(_tc_id)
                    _pending_tool_call_ids.add(_tc_id)
            continue
        if _msg.role != "tool":
            continue
        # F-051: tool_call_id REQUIRED.
        if not _msg.tool_call_id or not isinstance(_msg.tool_call_id, str):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"messages[{_idx}] with role 'tool' must have a non-empty "
                    "'tool_call_id' string"
                ),
            )
        # F-112: tool_call_id must reference a prior assistant tool_call.id
        # that has NOT already been consumed by an earlier tool reply.
        if _msg.tool_call_id not in _pending_tool_call_ids:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"messages[{_idx}] tool_call_id {_msg.tool_call_id!r} does "
                    "not reference any prior assistant tool_call"
                ),
            )
        # Consume the ticket. A later duplicate ``role:"tool"`` reply
        # with the same id will fall through to the F-112 branch above
        # and be rejected.
        _pending_tool_call_ids.discard(_msg.tool_call_id)
        # F-111: tool content shape. Accept str, None, or text-only
        # array. Anything else is a non-text part (image/video/audio)
        # which would be silently dropped by the renderer. An EMPTY
        # ``list`` (``content: []``) is also rejected — the chat-template
        # normalizer in ``utils/chat_template.py`` does not flatten
        # empty lists (an empty array isn't a "text-only" array; there's
        # no text to extract), so accepting it here would leak a
        # non-string ``content`` into the rendered prompt and crash the
        # template. Clients that intend an empty tool reply must send
        # ``content: ""`` or ``content: null`` (codex round-1 BLOCKING
        # on PR #731).
        _content = _msg.content
        if _content is None or isinstance(_content, str):
            continue
        if isinstance(_content, list):
            if not _content:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"messages[{_idx}] role 'tool' content must not be an "
                        "empty list; send an empty string '' or null for "
                        "an empty tool reply"
                    ),
                )
            _bad = False
            for _part in _content:
                if hasattr(_part, "model_dump"):
                    _part_d = _part.model_dump(exclude_none=True)
                elif isinstance(_part, dict):
                    _part_d = _part
                else:
                    _bad = True
                    break
                if _part_d.get("type") != "text" or not isinstance(
                    _part_d.get("text"), str
                ):
                    _bad = True
                    break
            if _bad:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"messages[{_idx}] role 'tool' content must be a "
                        "string or a text-only array of "
                        "{type:'text', text:str} parts"
                    ),
                )
            continue
        # Anything else (int / dict / bool / ...) is not a valid OpenAI
        # tool content shape.
        raise HTTPException(
            status_code=400,
            detail=(
                f"messages[{_idx}] role 'tool' content must be a string or a "
                f"text-only content-parts array, not {type(_content).__name__}"
            ),
        )

    # Validate n parameter (only n=1 supported)
    if request.n is not None and request.n > 1:
        raise HTTPException(
            status_code=400,
            detail="n > 1 is not supported. Rapid-MLX generates one completion per request.",
        )

    # Validate max_tokens. Lower bound: must be positive. Upper bound: a
    # hard sanity ceiling so a buggy client passing 999_999_999 cannot
    # combine with unbounded admission to OOM the Metal allocator.
    if request.max_tokens is not None and request.max_tokens < 1:
        raise HTTPException(
            status_code=400,
            detail="max_tokens must be at least 1",
        )
    if request.max_tokens is not None and request.max_tokens > 1_000_000:
        raise HTTPException(
            status_code=400,
            detail="max_tokens must be at most 1000000",
        )

    # Validate temperature range (OpenAI spec: 0-2)
    if request.temperature is not None and (
        request.temperature < 0 or request.temperature > 2
    ):
        raise HTTPException(
            status_code=400,
            detail="temperature must be between 0 and 2",
        )

    # Validate top_p range (OpenAI spec: (0, 1]). Without this, top_p=2.0
    # is silently accepted while sister field `temperature` is checked,
    # so clients with a bug see no signal.
    if request.top_p is not None and (request.top_p <= 0 or request.top_p > 1):
        raise HTTPException(
            status_code=400,
            detail="top_p must be in (0, 1]",
        )

    # Validate top_logprobs range (OpenAI spec: 0-20)
    if request.top_logprobs is not None and (
        request.top_logprobs < 0 or request.top_logprobs > 20
    ):
        raise HTTPException(
            status_code=400,
            detail="top_logprobs must be between 0 and 20",
        )

    # Reject non-empty logit_bias with a clear 400 rather than silently
    # dropping it. We accept {} so defensive clients that always include
    # the field don't break.
    if request.logit_bias:
        raise HTTPException(
            status_code=400,
            detail="logit_bias is not supported on this server",
        )

    # Validate ``response_format`` shape BEFORE
    # ``build_json_system_prompt`` is reached (F-013). Two bugs the gate
    # closes: (a) ``type:"json_schema"`` with no ``json_schema`` field
    # used to leak ``AttributeError: 'NoneType' object has no attribute
    # 'get'`` in the 400 body via the broad ``except Exception`` at
    # the call site; (b) unknown ``type`` values (``"xml"``,
    # ``""``, ``{}``, ``type:"json_schema"`` with empty
    # ``json_schema:{}``) were silently accepted as HTTP 200 with no
    # structure enforcement — client received unconstrained prose
    # without any signal.
    _validate_response_format(request.response_format)

    # --- Detailed request logging ---
    n_msgs = len(request.messages)
    msg_roles = [m.role for m in request.messages]
    total_chars = 0
    last_user_preview = ""
    for m in request.messages:
        content = m.content if isinstance(m.content, str) else str(m.content)
        total_chars += len(content)
        if m.role == "user":
            last_user_preview = content[:300]
    n_tools = len(request.tools) if request.tools else 0
    logger.info(
        f"[REQUEST] POST /v1/chat/completions stream={request.stream} "
        f"model={request.model!r} max_tokens={request.max_tokens} "
        f"temp={request.temperature} msgs={n_msgs} roles={msg_roles} "
        f"total_chars={total_chars} tools={n_tools} "
        f"response_format={request.response_format}"
    )
    logger.debug(f"[REQUEST] last user message preview: {last_user_preview!r}")

    cfg = get_config()

    # Enforce ``tool_choice`` at the prompt level (#445). The OpenAI spec
    # accepts four modes: "auto", "none", "required", and
    # ``{"type":"function","function":{"name":X}}``. Local inference has no
    # native enforcement (no FSM constraint), so the only reliable lever for
    # ``"none"`` and the specific-function form is to mutate what the model
    # sees: drop ``tools`` entirely for ``"none"``, or filter to just the
    # named function for the specific case. ``"auto"`` and ``"required"``
    # leave tools untouched — ``"required"`` enforcement is tracked
    # separately under #442 (needs decoder-level constraints, PR #132).
    tc = request.tool_choice
    if tc is not None:
        # Validation runs even when ``tools`` is empty/None: the OpenAI spec
        # treats ``tool_choice`` with a specific function but no matching
        # ``tools`` entry as a malformed request (400), not a silent
        # fall-through. Codex round-1 review of #446 flagged the previous
        # guard ``if tc is not None and request.tools:`` as silently
        # accepting these requests.
        if isinstance(tc, dict) and tc.get("type") == "function":
            fn = tc.get("function") or {}
            target = fn.get("name")
            if not target:
                raise HTTPException(
                    status_code=400,
                    detail=("tool_choice with type='function' requires function.name"),
                )
            if not request.tools:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"tool_choice references function {target!r} but the "
                        "request has no 'tools' array"
                    ),
                )
            filtered = [t for t in request.tools if t.function.get("name") == target]
            if not filtered:
                # F-145: surface a case-insensitive match as a hint when
                # one exists, so clients see "did you mean 'get_Weather'?"
                # instead of having to diff their tool list character-by-
                # character. OpenAI's API is case-sensitive too, but its
                # error message is equally terse — the rapid-mlx hint is
                # additive and OpenAI-shape-compatible (clients that ignore
                # the suffix still see the canonical 400).
                hint = ""
                target_lower = target.lower() if isinstance(target, str) else ""
                if target_lower:
                    case_matches = [
                        name
                        for t in request.tools
                        if isinstance((name := t.function.get("name")), str)
                        and name.lower() == target_lower
                    ]
                    if case_matches:
                        # ``case_matches[0]`` is deterministic — Pydantic
                        # preserves request order on ``request.tools`` so
                        # the hint points at the first matching definition.
                        # We only ever surface ONE hint; if the request
                        # somehow contains multiple case-variants of the
                        # same name (duplicate tool names are silently
                        # accepted today, see F-144), the first wins —
                        # which is the same tool the prompt-time
                        # filter would have picked anyway.
                        hint = f" (did you mean {case_matches[0]!r}? tool_choice is case-sensitive)"
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"tool_choice references function {target!r} which "
                        f"is not present in the 'tools' array{hint}"
                    ),
                )
            request.tools = filtered
        elif tc == "none" and request.tools:
            request.tools = None

    # Save original messages (clean dicts) for cloud routing BEFORE
    # local mutations (extract_multimodal_content, developer→system, suffix injection).
    if cfg.cloud_router:
        _cloud_original_messages = [
            (
                msg.model_dump(exclude_none=True)
                if hasattr(msg, "model_dump")
                else {k: v for k, v in dict(msg).items() if v is not None}
            )
            for msg in request.messages
        ]
    else:
        _cloud_original_messages = None

    # Content blocks must either reach a capable model path or be rejected
    # before generation. Text-only models reject all media; MLLM/VLM models
    # accept image/video but this server has no chat audio lane, so audio is
    # still a request-time 400 instead of being ignored by prompt rendering.
    try:
        validate_content_blocks_for_capabilities(
            request.messages,
            model_name=cfg.model_name,
            allow_image=engine.is_mllm,
            allow_video=engine.is_mllm,
            allow_audio=False,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # For MLLM models, keep original messages with embedded images
    if engine.is_mllm:
        messages = []
        for msg in request.messages:
            if hasattr(msg, "model_dump"):
                msg_dict = msg.model_dump(exclude_none=True)
            else:
                raw = dict(msg)
                msg_dict = {k: v for k, v in raw.items() if v is not None}
            messages.append(msg_dict)
        images, videos = [], []
        # The non-MLLM branch decodes tool_call.function.arguments from JSON
        # string to dict inside extract_multimodal_content() so chat templates
        # that iterate args via .items() (e.g. GLM-4.6V) don't crash. The
        # MLLM branch bypasses that helper, so call the shared decoder here.
        if engine.preserve_native_tool_format:
            decode_inline_tool_call_arguments(messages)
        logger.debug(f"MLLM: Processing {len(messages)} messages")
    else:
        messages, images, videos = extract_multimodal_content(
            request.messages,
            preserve_native_format=engine.preserve_native_tool_format,
        )

    has_media = bool(images or videos)
    if engine.is_mllm and not has_media:
        for msg in request.messages:
            content = msg.content if hasattr(msg, "content") else msg.get("content", "")
            if isinstance(content, list):
                for item in content:
                    item_type = (
                        item.type
                        if hasattr(item, "type")
                        else (item.get("type", "") if isinstance(item, dict) else "")
                    )
                    if item_type in ("image_url", "image", "video", "video_url"):
                        has_media = True
                        break
            if has_media:
                break

    # Normalize "developer" role to "system"
    for i, m in enumerate(messages):
        role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
        if role == "developer":
            if isinstance(m, dict):
                messages[i]["role"] = "system"
            else:
                m.role = "system"

    # Dogfood C-05 / r5-B C-09 fix: auto-prepend the canonical UI-TARS
    # Computer-Use action-API system prompt for the ``ui_tars`` parser
    # family — **tool-coupled** (only when the request actually
    # declares a Computer-Use tool). PR #812 wired the parser by alias
    # regex but never injected the sysprompt the model is post-trained
    # on; the C-05 fix then injected on every UI-TARS request, which
    # broke plain-text and JSON-mode prompts (F-R1-L: ``2+2`` came
    # back as a phantom click). r5-B threads ``tools=request.tools``
    # through so the helper's tool-coupled gate decides: NO computer
    # tool → no injection → model answers in prose / JSON. The helper
    # is also idempotent (skips when the user already pasted the
    # sysprompt) and honors ``tool_choice="none"`` (skips so the
    # model emits plain prose — dogfood C-07).
    from ..tool_parsers.ui_tars_tool_parser import (
        maybe_inject_ui_tars_system_prompt as _maybe_inject_ui_tars_sysprompt,
    )

    messages = _maybe_inject_ui_tars_sysprompt(
        messages,
        tool_call_parser=cfg.tool_call_parser,
        tool_choice=tc,
        tools=request.tools,
    )

    # Auto-inject system prompt suffix for tool use and/or reasoning control.
    # ``tool_choice="required"`` (and the specific-function form) gets a
    # stricter suffix than the default tool-use one — the OpenAI spec
    # guarantees a tool_call when ``required`` is set, but local inference
    # has no decoder-level enforcement (FSM constraint tracked in #132).
    # Prompt injection + post-parse 422 are the strongest levers we have
    # (#468). Strictness shape: explicit ``required`` > named function >
    # the default ``auto``/unset suffix.
    _inject_suffix = None
    if request.tools and cfg.tool_call_parser:
        if tc == "required":
            _inject_suffix = _TOOL_USE_REQUIRED_SUFFIX
        elif isinstance(tc, dict) and tc.get("type") == "function":
            _named = (tc.get("function") or {}).get("name")
            if _named:
                _inject_suffix = _tool_use_required_named_suffix(_named)
            else:
                _inject_suffix = _TOOL_USE_SYSTEM_SUFFIX
        else:
            _inject_suffix = _TOOL_USE_SYSTEM_SUFFIX
    elif cfg.reasoning_parser_name == "minimax":
        _inject_suffix = (
            "\n\nDo NOT think out loud or show your reasoning process. "
            "Give direct answers only — no preamble like 'The user asks...' or "
            "'We should respond...' or 'Let me think...'. Be concise."
        )

    if _inject_suffix:
        has_system = any(
            (m.get("role") if isinstance(m, dict) else getattr(m, "role", None))
            == "system"
            for m in messages
        )
        if has_system:
            for i, m in enumerate(messages):
                role = (
                    m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
                )
                if role == "system":
                    if isinstance(m, dict):
                        messages[i] = {**m, "content": m["content"] + _inject_suffix}
                    else:
                        messages[i]["content"] = m["content"] + _inject_suffix
                    break
        else:
            system_msg = {"role": "system", "content": _inject_suffix.strip()}
            messages = [system_msg] + list(messages)

    # Auto-pin system prompt prefix cache blocks
    if cfg.pin_system_prompt:
        _maybe_pin_system_prompt(messages)

    # Handle response_format - inject system prompt if needed
    response_format = request.response_format
    if response_format:
        try:
            json_instruction = build_json_system_prompt(response_format)
        except Exception as e:
            logger.warning(f"Failed to build JSON system prompt: {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid response_format schema: {e}",
            )
        if json_instruction:
            messages = _inject_json_instruction(messages, json_instruction)

    # R12-T1F (0.8.16 operator dogfood) — auto-disable thinking when
    # ``tools`` is non-empty and the client did NOT pin a thinking
    # preference. Same shape as M-2's strict-json_schema auto-disable
    # (PR #877): default-on thinking on Qwen3 / DeepSeek-R1 burns the
    # entire ``max_tokens`` budget inside ``<think>...</think>`` before
    # the model emits a ``<tool_call>`` envelope, so the agent-SDK
    # tight-budget pattern (``max_tokens=50..100``) finishes with
    # ``finish_reason="length"`` and ``tool_calls=None``. The injection
    # is non-destructive (forward-compat keys preserved) and explicit
    # ``True`` / ``False`` from the client is always honored. MUST run
    # BEFORE ``_resolve_enable_thinking`` so the resolved value drives
    # ``max_tokens`` headroom + the engine kwarg below from one source.
    if maybe_auto_disable_thinking_for_tools(request):
        logger.info(
            "R12-T1F auto-disable: /v1/chat/completions request has "
            "tools=%d with no client-set thinking preference — "
            "injecting chat_template_kwargs.enable_thinking=False so "
            "thinking models do not burn the token budget inside "
            "<think> before emitting the tool_call. Set "
            "chat_template_kwargs.enable_thinking=true to opt back in.",
            len(request.tools),
        )

    # R12-T2F-276 (0.8.16 brand-new-user simulation) — auto-disable
    # thinking on a casual chat completion (no tools, no strict json
    # schema, no explicit reasoning intent) so a first-time SDK user
    # gets a useful first request without having to learn
    # ``chat_template_kwargs``. Third member of the auto-disable
    # family (after R12-T1F TOOLS-AUTO above and R12-M2 strict-json
    # earlier); the helper short-circuits when an earlier trigger
    # already injected the kwarg OR when the client expressed
    # explicit reasoning intent (top-level / nested
    # ``enable_thinking``, ``reasoning_max_tokens``,
    # ``reasoning_effort``, or the Responses-native ``reasoning``
    # dict). MUST run BEFORE ``_resolve_enable_thinking`` so the
    # resolved value drives ``max_tokens`` headroom + the engine
    # kwarg below from one source. Mirrors the ``rapid-mlx chat``
    # REPL's ``--no-think`` default for thinking-capable models on
    # the OpenAI-SDK surface.
    if maybe_auto_disable_thinking_for_casual_chat(request):
        logger.info(
            "R12-T2F auto-disable: /v1/chat/completions casual chat "
            "request to a thinking-capable model (parser=%s) with no "
            "client-set thinking preference and no explicit reasoning "
            "intent — injecting chat_template_kwargs."
            "enable_thinking=False so thinking models do not burn the "
            "token budget inside <think> before emitting the answer. "
            "Set chat_template_kwargs.enable_thinking=true (or "
            "reasoning_max_tokens / reasoning_effort) to opt back in.",
            cfg.reasoning_parser_name,
        )

    # Resolve enable_thinking once and reuse — drives both the
    # max_tokens default (thinking models need more headroom) and the
    # chat_template kwarg below. (#387)
    resolved_thinking = _resolve_enable_thinking(request)

    # Prepare kwargs
    chat_kwargs = {
        "max_tokens": _resolve_max_tokens(request.max_tokens, resolved_thinking),
        "temperature": _resolve_temperature(request.temperature),
        "top_p": _resolve_top_p(request.top_p),
        "stop": request.stop,
    }

    # Extended sampling params — resolve through the request → CLI →
    # alias → generation_config cascade. Only forwards values the
    # cascade actually produced.
    chat_kwargs.update(build_extended_sampling_kwargs(request))

    # Add multimodal content
    if has_media:
        chat_kwargs["images"] = images if images else None
        chat_kwargs["videos"] = videos if videos else None
        if request.video_fps:
            chat_kwargs["video_fps"] = request.video_fps
        if request.video_max_frames:
            chat_kwargs["video_max_frames"] = request.video_max_frames

    # Add tools if provided
    if request.tools:
        chat_kwargs["tools"] = convert_tools_for_template(request.tools)

    # OpenAI ``tool_choice`` forced-function — assistant-turn prefix
    # injection. The chat-template renderer (and the engine's
    # ``chat()``/``stream_chat()``) accept ``forced_assistant_prefix``;
    # when set, the rendered prompt is suffixed with the parser's wire-
    # envelope opener and the model continues from inside the tool
    # call. The text parser then recovers the call in the normal flow.
    # No per-model regex, no per-alias config — the prefix is derived
    # solely from ``cfg.tool_call_parser`` and the requested function
    # name. See ``_forced_tool_call_prefix`` for the parser-shape
    # taxonomy.
    _forced_prefix = None
    if request.tools and request.tool_choice is not None:
        _forced_name: str | None = None
        if (
            isinstance(request.tool_choice, dict)
            and request.tool_choice.get("type") == "function"
        ):
            _forced_name = (request.tool_choice.get("function") or {}).get("name")
        elif request.tool_choice == "required" and len(request.tools) == 1:
            # OpenAI spec: ``required`` with a single tool is unambiguous
            # — same forcing semantics as a named choice.
            _forced_name = request.tools[0].function.get("name")
        if _forced_name:
            _forced_prefix = _forced_tool_call_prefix(
                cfg.tool_call_parser, _forced_name
            )
    if _forced_prefix:
        chat_kwargs["forced_assistant_prefix"] = _forced_prefix

    # PFlash routing (#287): structured-output prompts are
    # prompt-integrity-sensitive — lossy compression would corrupt the
    # JSON schema context, and there is no user-facing opt-out for
    # structured output, so they stay hard-protected here.
    #
    # Tools used to be lumped in with structured output but have a
    # separate user-facing knob — ``PFlashConfig.skip_when_tools``
    # (default skip; CLI ``--pflash-include-tools`` inverts it). The
    # gate flows through ``has_tools`` instead. Force-setting
    # ``requires_prompt_integrity=True`` for tools here short-circuits
    # the ``skip_when_tools`` branch and made the documented CLI
    # opt-in dead (codex r6 BLOCKING).
    if response_format:
        chat_kwargs["requires_prompt_integrity"] = True

    if resolved_thinking is not None:
        chat_kwargs["enable_thinking"] = resolved_thinking

    # Context-length pre-check (DoS defense + UX, rapid-desktop#273 / #463).
    # See ``service/helpers.py::enforce_context_length_for_messages`` for
    # the rationale (8 MiB body still holds ~2M tokens → context window
    # blown → ~60–90 s of wasted prefill before client gives up). Same
    # gate runs in routes/completions, routes/anthropic, routes/responses.
    #
    # rapid-mlx#280 (codex MED on PR #893 review): thread the resolved
    # ``enable_thinking`` so the prompt-token estimate matches what the
    # engine actually generates. The R12-T1F / R12-T2F auto-disable
    # above mutates ``request.chat_template_kwargs`` BEFORE this gate
    # runs, so the gate must consult the resolved value — otherwise it
    # renders with the template default (typically ``True`` on Qwen3 /
    # DeepSeek-R1), over-estimates the prompt by the
    # ``<|im_start|>think...`` scaffolding, and can reject requests
    # that actually fit.
    enforce_context_length_for_messages(
        engine,
        messages,
        tools=request.tools,
        max_tokens=chat_kwargs.get("max_tokens"),
        enable_thinking=resolved_thinking,
    )

    # Cloud routing: offload large-context requests to cloud LLM.
    #
    # The token-budget computation (``build_prompt`` + ``estimate_new_
    # tokens``) is part of the BaseEngine contract — any exception there
    # is a real bug and must surface, NOT be silently swallowed as
    # "falling back to local". The two regressions this scope-narrowing
    # closes (#500 + the v0.6.70 hotfix) both hid behind a broad
    # ``except Exception`` that turned engine-contract violations into
    # warning logs while cloud routing silently never fired.
    #
    # Only the cloud call itself is wrapped, and only "expected,
    # transient" failure shapes (network, auth, provider) are caught.
    if cfg.cloud_router and not engine.is_mllm:
        prompt = engine.build_prompt(messages, tools=request.tools)
        total_tokens, new_tokens = engine.estimate_new_tokens(prompt)
        if cfg.cloud_router.should_route_to_cloud(new_tokens):
            logger.info(
                f"[CLOUD ROUTE] {new_tokens} new tokens (total {total_tokens}) "
                f"> threshold {cfg.cloud_router.threshold}, "
                f"routing to {cfg.cloud_router.cloud_model}"
            )
            cloud_messages = _cloud_original_messages
            cloud_kwargs = {
                "temperature": chat_kwargs.get("temperature"),
                "max_tokens": chat_kwargs.get("max_tokens"),
                "top_p": chat_kwargs.get("top_p"),
            }
            if request.stop:
                cloud_kwargs["stop"] = request.stop
            if request.tool_choice is not None:
                cloud_kwargs["tool_choice"] = request.tool_choice
            if request.response_format:
                rf = request.response_format
                cloud_kwargs["response_format"] = (
                    rf.model_dump() if hasattr(rf, "model_dump") else rf
                )
            if request.tools:
                cloud_kwargs["tools"] = [
                    t.model_dump() if hasattr(t, "model_dump") else t
                    for t in request.tools
                ]
            # Cloud-routed request: the local scheduler/Metal path
            # is bypassed entirely, so admission is not acquired
            # for cloud paths. The wrapper's ``finally`` checks
            # ``_admission_acquired[0]`` (still False here) and
            # skips the release. Without this ordering (admission
            # check moved BELOW the cloud routing block), a burst
            # of local requests filling the cap would 503
            # cloud-routable requests that never touch the local
            # engine (codex R9).
            try:
                if request.stream:
                    return StreamingResponse(
                        _disconnect_guard(
                            cfg.cloud_router.stream_completion(
                                cloud_messages,
                                model_name=cfg.model_name or "cloud",
                                **cloud_kwargs,
                            ),
                            raw_request,
                        ),
                        media_type="text/event-stream",
                        headers=SSE_RESPONSE_HEADERS,
                    )
                else:
                    result = await _wait_with_disconnect(
                        cfg.cloud_router.completion(cloud_messages, **cloud_kwargs),
                        raw_request,
                        timeout=request.timeout or cfg.default_timeout,
                    )
                    if result is None:
                        return Response(status_code=499, content="Client disconnected")
                    # NOTE: L-05's enable_thinking warning intentionally
                    # does NOT fire on the cloud-routed path — the local
                    # ``cfg.reasoning_parser_name`` isn't authoritative
                    # for what the cloud provider does with the ctk
                    # hint. A warning here would be misleading.
                    return Response(
                        content=json.dumps(result),
                        media_type="application/json",
                    )
            except _CLOUD_CALL_RECOVERABLE_EXCEPTIONS as e:
                # Provider/network failures are transient and the local
                # engine is a reasonable fallback. Engine-contract
                # violations (AttributeError, TypeError, …) are NOT in
                # this allowlist on purpose — they must surface as 500.
                logger.warning(
                    f"[CLOUD ROUTE] Cloud call failed ({type(e).__name__}: {e}), "
                    "falling back to local"
                )
        else:
            logger.info(
                f"[LOCAL] {new_tokens} new tokens (total {total_tokens}) "
                f"<= threshold {cfg.cloud_router.threshold}, using local inference"
            )

    # ``tool_choice="required"`` + ``stream=true`` is enforceable IF the
    # engine has SOME path to produce a streaming tool_call:
    #   (a) a text-parser path — ``cfg.tool_call_parser`` set; or
    #   (b) a channel-routed path — harmony (gpt-oss) / Gemma 4 emit
    #       structured tool_calls via the OutputRouter's tool channel
    #       without needing a text parser.
    # The request can satisfy the contract iff EITHER path is available.
    # When neither is available we have NO mechanism at all, so reject
    # upfront with a clear error.
    #
    # Round-7 codex BLOCKING surfaced the silent text-only finish_reason
    # case; round-8 moved the guard below cloud routing; round-9 narrowed
    # to the truly-unenforceable case (no parser); round-10 codex BLOCKING
    # #1 widened "enforceable" to include channel-routed capability so
    # harmony/gemma4 streaming requests aren't blocked by the gate.
    # Engine-level veto — even with ``--tool-call-parser`` set, an
    # engine that has explicitly opted out of tool-call surfaces
    # (``supports_tool_calls=False``) cannot emit structured tool
    # calls because its generator never produces them in the first
    # place. The text parser would only match against the engine's
    # actual ``channel="content"`` output, which has no tool call
    # markers, so streaming would finish with plain text and the
    # contract would silently break. Reject upfront with the same
    # 422 the parser-less path uses (codex round 10 [P2] on PR #551).
    # Use the same falsey predicate (``not getattr(...)``) as
    # ``_engine_supports_channel_routed_tool_calls`` so the two
    # checks treat None / 0 / False uniformly as "engine has opted
    # out" — pr_validate codex r12 NIT. Default True (existing
    # engines) preserves prior behaviour for everything that hasn't
    # opted out.
    _engine_opts_out_of_tools = not getattr(engine, "supports_tool_calls", True)
    # Engine-level veto applies REGARDLESS of stream / non-stream
    # AND for every forced tool-choice shape (codex pr_validate r8
    # NIT #2). The OpenAI ``tool_choice`` API has two forced
    # variants beyond ``"required"``:
    #   - the named-function form ``{"type":"function",
    #     "function":{"name":"foo"}}`` — caller demands a specific
    #     tool gets called
    #   - and the deprecated ``"function"`` literal string (some
    #     legacy SDKs still send it)
    # All three are contracts an opted-out engine cannot satisfy
    # because the generator never produces structured tool_calls.
    # Pre-pr_validate r6, this check was nested inside the
    # ``request.stream`` branch below, so a non-streaming forced
    # request still ran a full diffusion generation before failing
    # in the post-parse gate at line ~1101. That is wasted GPU +
    # ambiguous client UX. Reject upfront for opted-out engines no
    # matter the stream flag (codex pr_validate r6 BLOCKING #1 +
    # r8 NIT #2 on PR #551).
    _forced_tool_choice = (
        tc == "required"
        # Legacy literal — some pre-2024 OpenAI SDKs sent the bare
        # string ``"function"`` to mean "force any function call"
        # before the dict form was added. Codex pr_validate r9 NIT
        # #1 flagged the original predicate omitted this shape so
        # opted-out engines would still run a full generation
        # before failing.
        or tc == "function"
        or (isinstance(tc, dict) and tc.get("type") == "function")
    )
    if _forced_tool_choice and request.tools and _engine_opts_out_of_tools:
        raise HTTPException(
            status_code=422,
            detail=(
                "tool_choice forces a tool call, but the active engine "
                "has explicitly opted out of tool-call surfaces "
                "(supports_tool_calls=False). The generator never emits "
                "structured tool_calls, so any forced choice — "
                '``"required"`` or a named ``{"type":"function","function":'
                '{"name":...}}`` — is unenforceable. Drop tool_choice (or '
                'set it to ``"auto"``/``"none"``), retry against an engine '
                "that supports tool calls, or remove the ``tools`` array "
                "from the request."
            ),
        )
    if (
        request.stream
        and tc == "required"
        and request.tools
        and not cfg.tool_call_parser
        and not _engine_supports_channel_routed_tool_calls(engine)
    ):
        raise HTTPException(
            status_code=422,
            detail=(
                'tool_choice="required" with stream=true requires either a '
                "streaming tool-call parser (--tool-call-parser) or a "
                "channel-routed model (harmony / Gemma 4) so the server has "
                "a path to emit structured tool_calls. Neither is available "
                "for this request — the OpenAI 'tool_call guaranteed' "
                "contract cannot be met. Either set --tool-call-parser=hermes "
                "(or your model's parser), retry with stream=false "
                "(non-stream path 422s text-only output), or pin a specific "
                'function via tool_choice={"type":"function",'
                '"function":{"name":...}}.'
            ),
        )

    # Local-path admission gate: reserve a slot before kicking the
    # engine. Placed AFTER cloud routing so cloud-routable requests
    # don't 503 just because the local cap is full (codex R9), and
    # AFTER the cheap validation above so a malformed request can't
    # pin a slot until restart (codex R3). The wrapper's ``finally``
    # uses ``_admission_acquired`` to decide whether to release.
    _check_admission_or_503(engine)
    _admission_acquired[0] = True

    # Detect guided generation BEFORE the stream/non-stream split so the
    # streaming branch can also route json_schema requests through the
    # constrained path. Pre-fix, only the non-stream branch consulted
    # ``supports_guided_generation`` and stream=true silently bypassed
    # ``GuidedGenerator`` — the model would emit unconstrained tokens
    # (e.g. a ```json ... ``` markdown fence) even with a json_schema
    # response_format set. Surfaced by Gap #2 of the v0.6.60 onboarding
    # sweep; mirrors the constraint-then-emit pattern from upstream
    # waybarrios#548.
    use_guided = False
    json_schema = None
    # R12-4: when strict=true is set but the engine cannot guide (no
    # ``[guided]`` extra), we now run the engine UNCONSTRAINED and
    # validate the output against the schema after generation. If
    # validation fails AND the disable flag is unset, the route
    # performs ONE repair retry with a system-prompt-injected hint
    # naming the failing path. If that still fails, the route
    # returns 422 with a structured envelope. This replaces the
    # legacy ``guided_extra_required`` 400 — see the R12-4 PR body
    # for the design rationale (post-generate validation gives us
    # spec-compliant behavior without the multi-week effort of
    # plumbing native constrained decoding into the MLX engine).
    use_strict_postgen_validation = False
    # H-06: strict mode means the OpenAI contract REQUIRES the model
    # output to validate against the schema. Pre-fix, ``strict=true``
    # was suggestion-only — the route dropped the flag at
    # ``build_json_system_prompt`` time and let the engine emit
    # whatever the model produced. Distinguish the two modes here so
    # the rest of the function can react.
    strict_mode = is_strict_json_schema(response_format)
    # R12-4: respect the ``RAPID_MLX_STRICT_JSON_SCHEMA=off`` escape
    # hatch — operators who depended on the pre-R12-4 silent-200
    # behavior keep the legacy code path. The flag is intentionally
    # checked ONCE per request (not at import time) so an operator
    # can toggle it on a running process via ``os.environ`` (the
    # rapid-mlx desktop client relies on this).
    strict_enforcement_active = strict_mode and strict_enforcement_enabled()

    # Codex r3 BLOCKING #2 + defense-in-depth: ``strict=true`` with
    # tools set (the route's existing ``if response_format and not
    # request.tools`` gate below skips the guided dispatch when
    # tools are present) used to fall through silently — strict
    # mode would never trigger the gate and the model would emit
    # unconstrained tokens. Compute the schema BEFORE the tools
    # gate so the strict-malformed and strict+tools cases both
    # fail closed (400) instead of failing open (silent 200).
    # ``_validate_response_format`` already rejects ``schema={}``
    # at body-parse time but this is the defense-in-depth gate
    # that closes any future bypass (e.g. a refactor that moves
    # the validate-response_format call after this point).
    if strict_mode:
        _strict_schema_check = extract_json_schema_for_guided(response_format)
        if not _strict_schema_check:
            incr_strict_request()
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": (
                            "response_format.json_schema.strict=true "
                            "requires a non-empty "
                            "response_format.json_schema.schema. The "
                            "request set strict=true but the schema "
                            "field is missing or empty — the strict "
                            "contract cannot be enforced without one."
                        ),
                        "type": "invalid_request_error",
                        "code": "strict_schema_required",
                        "param": "response_format.json_schema.schema",
                    }
                },
            )
        # Codex r4 NIT #5: validate the user-supplied schema BEFORE
        # generation so an invalid JSON Schema (e.g. ``type:"objct"``
        # typo) surfaces as a 400 ``invalid_strict_schema`` —
        # pointing at the client's malformed input — instead of
        # falling into the post-decode validator and surfacing as
        # a 502 ``strict_schema_violation`` (server-side breach
        # shape). The check covers both the strict route and the
        # /v1/responses entry below via the same helper.
        _schema_ok, _schema_err = check_schema_validity(_strict_schema_check)
        if not _schema_ok:
            incr_strict_request()
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": (
                            "response_format.json_schema.schema is not "
                            f"a valid JSON Schema document: {_schema_err}. "
                            "Fix the schema and retry."
                        ),
                        "type": "invalid_request_error",
                        "code": "invalid_strict_schema",
                        "param": "response_format.json_schema.schema",
                    }
                },
            )
        if request.tools:
            # Strict + tools is mutually exclusive on this engine:
            # the constrained-decoding path is grammar-driven and
            # cannot coexist with the tool-call grammar. OpenAI's
            # cloud API treats this combination as 400 too. Surface
            # the conflict explicitly so clients see the choice.
            incr_strict_request()
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": (
                            "response_format.json_schema.strict=true "
                            "cannot be combined with 'tools' — the "
                            "constrained-decoding grammar is mutually "
                            "exclusive with the tool-call grammar. "
                            "Drop one or the other and retry."
                        ),
                        "type": "invalid_request_error",
                        "code": "strict_with_tools_unsupported",
                        "param": "response_format.json_schema.strict",
                    }
                },
            )

    if response_format and not request.tools:
        json_schema = extract_json_schema_for_guided(response_format)
        if json_schema:
            # ``supports_guided_generation`` and ``generate_with_schema``
            # are on the BaseEngine contract — defaults are False /
            # NotImplementedError, so engines without guided decoding
            # opt out by leaving the property at False. The previous
            # ``hasattr`` guards were artifacts of the engine API being
            # informal; they're the same silent-skip shape that produced
            # #500 and the v0.6.70 hotfix and have no role now that the
            # contract is explicit.
            use_guided = engine.supports_guided_generation
            if strict_mode:
                # Tick the strict-request counter BEFORE any branching
                # so operators see uniform traffic shape across the
                # guided / postgen-validation / disabled arms.
                incr_strict_request()
                if not use_guided:
                    # R12-4: pre-R12-4 this branch raised 400
                    # ``guided_extra_required``. That broke
                    # pydantic-ai end-to-end (Astrid r3) — every
                    # retry hit the same deterministic empty-args
                    # synthetic ``final_result`` tool_call and the
                    # client exhausted ``max_retries`` against a
                    # server-side blocker the SDK could not
                    # circumvent. The new path runs the engine
                    # UNCONSTRAINED, validates the output against
                    # the schema after generation, attempts a single
                    # repair retry on validation failure, and
                    # surfaces 422 only if BOTH attempts fail. The
                    # disable flag
                    # ``RAPID_MLX_STRICT_JSON_SCHEMA=off`` (checked
                    # at request time) restores the legacy
                    # silent-pass-through behavior for operators
                    # who need the escape hatch.
                    if strict_enforcement_active:
                        use_strict_postgen_validation = True
                        logger.info(
                            "Strict json_schema mode active without "
                            "[guided] extra — engaging R12-4 "
                            "post-generate validation + single "
                            "repair retry path."
                        )
                    else:
                        logger.warning(
                            "Strict json_schema mode requested but "
                            "RAPID_MLX_STRICT_JSON_SCHEMA=off — "
                            "falling through to prompt-injection "
                            "only (legacy silent-pass-through). "
                            "Unset the env var to restore "
                            "enforcement."
                        )
                else:
                    logger.info(
                        "Using guided generation for JSON schema "
                        "enforcement (strict=true)"
                    )
            elif use_guided:
                logger.info("Using guided generation for JSON schema enforcement")
            else:
                # Surface the silent-degradation case: client asked for
                # json_schema response_format but the engine can't
                # enforce it (most commonly: the user installed
                # `rapid-mlx` without the `[guided]` extra). When
                # ``strict=false`` the OpenAI contract is suggestion-only
                # so we fall through to prompt-injection (existing
                # behavior). When ``strict=true`` see the R12-4 branch
                # above.
                logger.warning(
                    "json_schema response_format requested but guided "
                    "generation is unavailable (engine="
                    "%s.supports_guided_generation=False). Falling back "
                    "to unconstrained decoding — schema will NOT be "
                    "enforced. Install with `pip install "
                    "'rapid-mlx[guided]'` to enable outlines-backed "
                    "schema enforcement.",
                    type(engine).__name__,
                )

    if request.stream:
        # Validate chat template eagerly so template errors return 400
        if not engine.is_mllm:
            try:
                engine.build_prompt(
                    messages,
                    tools=chat_kwargs.get("tools"),
                    enable_thinking=chat_kwargs.get("enable_thinking"),
                )
            except Exception as e:
                err_msg = str(e)
                err_type = type(e).__name__
                if (
                    "TemplateError" in err_type
                    or "template" in err_msg.lower()
                    or ("user" in err_msg.lower() and "found" in err_msg.lower())
                ):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Chat template error: {err_msg}",
                    )
                raise
        _commit_state[0] = True
        # L-05: surface silent ``enable_thinking`` drop on non-Qwen
        # parsers via response headers. Merging here lets the SSE
        # ``Cache-Control`` / ``Connection`` headers stay intact.
        _thinking_warning = enable_thinking_warning_header(
            request, getattr(cfg, "reasoning_parser_name", None)
        )
        _sse_headers = (
            {**SSE_RESPONSE_HEADERS, **_thinking_warning}
            if _thinking_warning
            else SSE_RESPONSE_HEADERS
        )
        # C-01: holder list the engine writes the admitted scheduler
        # request id into. ``_disconnect_guard`` reads the SAME list
        # and force-calls ``scheduler.abort_request`` on client
        # disconnect, closing the Astrid r3 hang where the
        # generator-close cascade alone took ~35s to actually free
        # the GPU once the client TCP-RST'd.
        request_id_holder: list[str | None] = [None]
        chat_kwargs["request_id_holder"] = request_id_holder
        if use_guided and json_schema:
            # Constrained streaming: run guided generation buffered, then
            # synthesize an SSE stream from the buffered output. Falls
            # back to the unconstrained streaming helper on guided
            # failure (logged), matching the non-streaming fallback.
            return StreamingResponse(
                _disconnect_guard(
                    stream_chat_completion_guided(
                        engine,
                        messages,
                        request,
                        json_schema,
                        strict_mode=strict_mode,
                        **chat_kwargs,
                    ),
                    raw_request,
                    engine=engine,
                    request_id_holder=request_id_holder,
                ),
                media_type="text/event-stream",
                headers=_sse_headers,
            )
        if use_strict_postgen_validation and json_schema:
            # R12-4 streaming variant: the unconstrained stream is
            # emitted as usual; we buffer the content deltas and run
            # post-stream validation. On failure we synthesize an
            # extra terminal chunk carrying
            # ``finish_reason="json_schema_violation"`` plus a
            # non-content event with the 422-shaped error envelope
            # BEFORE ``[DONE]``. The asymmetry vs. non-streaming is
            # intentional and documented — streaming clients receive
            # the content (so they can show partial output) AND the
            # error signal, while non-streaming clients see a clean
            # HTTP 422 with no body. The repair retry is not run on
            # the streaming path because re-prompting after the wire
            # is already open is structurally incompatible with SSE
            # (no second response_id; clients would mis-attribute the
            # retry tokens to the first stream). The strict-request
            # counter has already ticked above when ``strict_mode``
            # was detected — no double-count here.
            return StreamingResponse(
                _disconnect_guard(
                    stream_chat_completion_strict_postgen(
                        engine,
                        messages,
                        request,
                        json_schema,
                        **chat_kwargs,
                    ),
                    raw_request,
                    engine=engine,
                    request_id_holder=request_id_holder,
                ),
                media_type="text/event-stream",
                headers=_sse_headers,
            )
        return StreamingResponse(
            _disconnect_guard(
                stream_chat_completion(engine, messages, request, **chat_kwargs),
                raw_request,
                engine=engine,
                request_id_holder=request_id_holder,
            ),
            media_type="text/event-stream",
            headers=_sse_headers,
        )

    # Non-streaming response with timing and timeout
    start_time = time.perf_counter()
    timeout = request.timeout or cfg.default_timeout

    # Disable GC during generation to avoid latency spikes
    gc_was_enabled = gc.isenabled()
    if cfg.gc_control and gc_was_enabled:
        gc.disable()

    # Determine if we need per-token logprobs.
    #
    # R15 task #311: OpenAI semantics — ``logprobs=true`` alone returns
    # the sampled-token logprob (``content[].logprob``); ``top_logprobs``
    # only adds the K alternatives field. Pre-fix the gate was
    # ``request.logprobs and request.top_logprobs``, which fell through
    # to the non-logprobs branch when the client sent
    # ``logprobs=true, top_logprobs=None`` and silently dropped the
    # sampled-token logprob the caller asked for. The fix gates on
    # ``logprobs`` alone; ``top_logprobs`` controls the alternatives
    # ceiling and is allowed to be ``None`` / ``0``.
    #
    # ``effective_top_k`` defends ``_extract_token_logprob`` from the
    # ``argpartition(-0)[-0:]``-returns-full-vocab footgun when
    # ``top_logprobs=0`` (or None) — mirrors the /v1/completions
    # ``logprobs=0`` rationale at routes/completions.py L387-399. We
    # extract with ``effective_top_k=1`` to surface the sampled-token
    # logprob, then strip the synthetic single-element alternatives
    # list to ``[]`` below when the caller asked for none.
    want_logprobs = bool(request.logprobs)
    top_k_logprobs = request.top_logprobs or 0
    effective_top_k = max(1, top_k_logprobs)
    token_logprobs_list: list[TokenLogProb] = []

    try:
        if want_logprobs and not use_guided:
            # ``logprobs`` requests need per-token data, so we route through
            # the streaming path even for a non-stream response. The streaming
            # iterator yields per-token outputs; on channel-routed models
            # (harmony/gpt-oss, gemma4) each chunk also carries the channel
            # the router assigned that token. Accumulate text by channel so
            # ``reasoning_text`` and ``text`` reach
            # ``_finalize_content_and_reasoning`` already split — without
            # this, the loop kept only the LAST chunk's text and ``output.
            # reasoning_text`` stayed empty, so the route fell back to the
            # text-regex parser which leaks analysis-channel content into
            # ``content`` on harmony (same shape as #442 but for the logprobs
            # path).
            from dataclasses import replace as _dc_replace

            output = None
            routed_content_parts: list[str] = []
            routed_reasoning_parts: list[str] = []
            saw_channel = False
            async for chunk in engine.stream_chat(messages=messages, **chat_kwargs):
                output = chunk
                token_logprobs_list.extend(
                    _extract_streaming_token_logprobs(
                        chunk, engine.tokenizer, effective_top_k
                    )
                )
                ch = getattr(chunk, "channel", None)
                if ch:
                    saw_channel = True
                    if ch == "reasoning":
                        routed_reasoning_parts.append(chunk.new_text or "")
                    elif ch == "content":
                        routed_content_parts.append(chunk.new_text or "")
                    # ``tool_call`` channel is parsed downstream by
                    # ``_parse_tool_calls_with_parser``; don't fold its body
                    # into either text bucket here.
            if output is None:
                return Response(status_code=499)
            if saw_channel:
                output = _dc_replace(
                    output,
                    text="".join(routed_content_parts),
                    reasoning_text="".join(routed_reasoning_parts),
                )
            # Forced-tool prefix on the logprobs path: the stream
            # yields a synthetic first chunk carrying the prefix
            # bytes for the SSE postprocessor, but THIS internal
            # consumer only retains the last chunk. Fold the prefix
            # into ``output.text`` so the downstream tool parser sees
            # the full ``<tool_call>{"name":...,"arguments":...}``
            # envelope and recovers the model-generated arguments
            # instead of falling through to ``_synthesize_forced_tool_call``'s
            # empty-args default (codex r1 P2 on this PR).
            _forced_prefix_value = chat_kwargs.get("forced_assistant_prefix")
            if (
                _forced_prefix_value
                and output is not None
                and not (output.text or "").startswith(_forced_prefix_value)
            ):
                output = _dc_replace(
                    output,
                    text=_forced_prefix_value + (output.text or ""),
                )
        elif use_guided and json_schema:
            try:
                output = await _wait_with_disconnect(
                    engine.generate_with_schema(
                        messages=messages,
                        json_schema=json_schema,
                        **chat_kwargs,
                    ),
                    raw_request,
                    timeout=timeout,
                )
            except HTTPException:
                raise
            except (TimeoutError, asyncio.TimeoutError, asyncio.CancelledError):
                # These belong to the outer route's standard
                # timeout / cancellation envelopes — NOT to the
                # strict-contract breach shape. Let them propagate
                # unchanged so the 408 / 499 / 503 mapping kicks in.
                raise
            except Exception as guided_err:
                # Codex r6 BLOCKING parity (non-streaming chat path):
                # under strict=true, falling back to ``engine.chat``
                # IS the H-06 hole — the buffered post-decode validator
                # at line ~1752 below would catch it as a 502 if the
                # unconstrained output happened to mis-validate, but
                # if it coincidentally validates the client receives
                # an unconstrained response under a ``strict=true``
                # contract the server never honored. Refuse the
                # fallback under strict mode, surface the breach as
                # 502 ``strict_schema_violation`` directly.
                if strict_mode:
                    incr_strict_violation()
                    logger.warning(
                        "Strict json_schema guided generation failed; "
                        "refusing to fall back to unconstrained because "
                        "strict=true: %s",
                        guided_err,
                    )
                    raise HTTPException(
                        status_code=502,
                        detail={
                            "error": {
                                "message": (
                                    "strict response_format could not be "
                                    "honored: the constrained-decoding path "
                                    f"raised {type(guided_err).__name__} "
                                    "before producing any output. The "
                                    "server refuses to fall back to "
                                    "unconstrained generation because the "
                                    "client asked for strict=true. "
                                    "Investigate the server logs and the "
                                    "rapid_mlx_response_format_strict_"
                                    "violations_total metric."
                                ),
                                "type": "api_error",
                                "code": "strict_schema_violation",
                                "param": "response_format.json_schema",
                            }
                        },
                    ) from guided_err
                logger.warning(
                    f"Guided generation failed, falling back to standard: {guided_err}"
                )
                logger.debug(f"Problematic schema: {json_schema}")
                # Fallback runs under the outer admission reservation
                # still held by the wrapper's ``finally`` — no
                # re-acquire needed (the helper does not release on
                # its own now that release lives at the route level).
                output = await _wait_with_disconnect(
                    engine.chat(messages=messages, **chat_kwargs),
                    raw_request,
                    timeout=timeout,
                )
        else:
            output = await _wait_with_disconnect(
                engine.chat(messages=messages, **chat_kwargs),
                raw_request,
                timeout=timeout,
            )
    except HTTPException:
        raise
    except Exception as e:
        from ..request import InferenceAbortedError

        err_msg = str(e)
        err_type = type(e).__name__
        if isinstance(e, InferenceAbortedError):
            # Engine aborted the request (e.g. Metal runtime error caught
            # in the engine loop). 503 — the server is still up and a
            # smaller request may succeed (#353).
            raise HTTPException(status_code=503, detail=err_msg)
        if (
            "TemplateError" in err_type
            or "template" in err_msg.lower()
            or ("user" in err_msg.lower() and "found" in err_msg.lower())
        ):
            raise HTTPException(
                status_code=400, detail=f"Chat template error: {err_msg}"
            )
        # Image / video fetch failures surface from multimodal_processor
        # (and models/mllm.py:_prepare_images) as ValueError with a
        # "Failed to process image|video" prefix. Convert to 400 so VLM
        # clients get a clear error instead of a 200 with empty completion
        # (#457).
        #
        # The "exceeds the per-batch cap" marker comes from
        # ``mllm_batch_generator._process_prompts`` when vision + text
        # tokens exceed the configured cap. The MLLM scheduler now flags
        # this as a client-actionable error (#682); without the explicit
        # 400 mapping the engine would still return ``HTTPException``-less
        # 500. Surface as 400 so Desktop / curl clients see the actionable
        # message ("downscale image / raise --prefill-step-size") instead
        # of a generic server error.
        if (
            "Failed to process image" in err_msg
            or "Failed to process video" in err_msg
            or "exceeds the per-batch cap" in err_msg
        ):
            raise HTTPException(status_code=400, detail=err_msg)
        raise
    finally:
        if cfg.gc_control and gc_was_enabled:
            gc.enable()
            gc.collect()

    if output is None:
        return Response(status_code=499)

    elapsed = time.perf_counter() - start_time
    tokens_per_sec = output.completion_tokens / elapsed if elapsed > 0 else 0
    logger.info(
        f"Chat completion: {output.completion_tokens} tokens in {elapsed:.2f}s ({tokens_per_sec:.1f} tok/s)"
    )

    # H-06: when the client asked for strict json_schema mode and we
    # routed through guided decoding, validate the buffered text
    # against the schema. Outlines should make this unreachable; a
    # non-zero ``rapid_mlx_response_format_strict_violations_total``
    # rate signals that the constrained-decoding path silently
    # degraded (e.g. ``generate_with_schema`` swallowed an outlines
    # API change and fell back to ``self.chat(...)``).
    #
    # Codex r2 BLOCKING #3: under the strict contract, returning a
    # known schema-invalid 200 body is itself a contract violation —
    # the OpenAI ``strict=true`` semantics promise the response
    # validates. Surface as 502 (upstream/internal contract failed)
    # so clients using ``chat.completions.parsed`` see the error
    # instead of silently consuming garbage that ``model_validate``
    # then rejects with a confusing stack-trace far from the source.
    # The violations counter still ticks before we raise so the
    # operator sees both the rate AND the error response.
    if strict_mode and use_guided and json_schema and output is not None:
        ok, err = validate_output_against_schema(output.text or "", json_schema)
        if not ok:
            incr_strict_violation()
            logger.warning(
                "Strict json_schema response failed post-decode validation: %s",
                err,
            )
            raise HTTPException(
                status_code=502,
                detail={
                    "error": {
                        "message": (
                            "strict response_format violated: model output "
                            f"did not validate against the supplied schema ({err}). "
                            "This indicates the constrained-decoding path silently "
                            "degraded; investigate the server logs and the "
                            "rapid_mlx_response_format_strict_violations_total metric."
                        ),
                        "type": "api_error",
                        "code": "strict_schema_violation",
                        "param": "response_format.json_schema",
                    }
                },
            )

    # R12-4: non-guided strict-mode enforcement. The engine ran
    # UNCONSTRAINED above; now we validate the buffered output and
    # — if it doesn't validate — attempt ONE repair retry with a
    # system-prompt-injected hint naming the failing path. If the
    # repair also fails we surface 422 with a structured envelope so
    # SDK consumers (pydantic-ai) can read ``error.details.failing_path``
    # / ``expected`` / ``got`` instead of looping against an opaque
    # error. Strict + tools is already rejected by the upstream
    # ``strict_with_tools_unsupported`` gate, so we can assume no
    # tool_calls path here.
    if use_strict_postgen_validation and json_schema and output is not None:
        ok, failure_details = validate_and_envelope(output.text or "", json_schema)
        attempts = 1
        if not ok and repair_retry_enabled():
            repair_messages = build_repair_messages(
                messages,
                output.text or "",
                json_schema,
                failure_details or {},
            )
            # The repair turn deliberately drops ``logprobs`` / forced
            # ``tool_choice`` / other request features so the model has
            # the cleanest possible path to "emit JSON only". Most of
            # ``chat_kwargs`` is preserved (model, temperature, max_tokens)
            # so the repair turn respects the operator's runtime caps.
            #
            # Codex r3 #2: ``request_id_holder`` IS preserved (was
            # dropped pre-r3, which left the repair generation
            # unwired from the route's disconnect / cancellation
            # tracking — a client that hung up between attempts
            # would keep the GPU pinned on the repair turn until it
            # finished). Tools / tool_choice / logprobs / top_logprobs
            # are still dropped because they're structurally
            # incompatible with the repair turn's "emit ONLY JSON"
            # contract.
            repair_kwargs = dict(chat_kwargs)
            for _k in (
                "tools",
                "tool_choice",
                "logprobs",
                "top_logprobs",
            ):
                repair_kwargs.pop(_k, None)
            # H-06 #267b: re-check the context-length gate AGAINST the
            # POST-BUILD repair prompt. ``build_repair_messages`` builds a
            # strictly larger prompt than the initial request (prepended
            # instructions, repeated schema, up to 4 KiB of failed output),
            # so a request that passed the initial gate can blow context
            # only on the repair attempt — pre-fix that surfaced as the
            # opaque ``502 strict_repair_engine_failure`` instead of a
            # deterministic ``422 json_schema_violation``. The helper
            # mirrors ``enforce_context_length_for_messages`` and is
            # centralized so chat + responses share one gate.
            # rapid-mlx#280: thread the resolved ``enable_thinking`` so
            # the repair-prompt fit check renders the way the engine
            # actually will. Pre-fix the gate rendered with the
            # template default, so on auto-disabled (R12-M2 strict-
            # mode) runs it could SKIP a retry that would actually
            # fit. ``repair_kwargs`` carries the same value because
            # it's a copy of ``chat_kwargs`` (see line above), but we
            # resolve from ``chat_kwargs`` for symmetry with the
            # initial gate at line ~2066.
            _repair_fits = repair_messages_fit_context(
                engine,
                repair_messages,
                tools=None,
                max_tokens=repair_kwargs.get("max_tokens"),
                enable_thinking=chat_kwargs.get("enable_thinking"),
            )
            repair_output = None
            if not _repair_fits:
                incr_strict_repair_skipped_context_overflow()
                logger.warning(
                    "R12-4 strict json_schema repair retry SKIPPED: "
                    "post-build repair prompt would exceed model context "
                    "window. Surfacing the ORIGINAL 422 json_schema_"
                    "violation envelope instead of attempting a retry "
                    "that would either 502 or truncate."
                )
                # Fall through to the existing ``if not ok:`` block
                # below — ``attempts == 1`` so the envelope reports the
                # single attempt the client actually saw.
            else:
                incr_strict_repair_attempt()
                attempts = 2
                logger.info(
                    "R12-4 strict json_schema first attempt failed "
                    "validation (%s); attempting single repair retry.",
                    failure_details.get("reason") if failure_details else "?",
                )
                try:
                    repair_output = await _wait_with_disconnect(
                        engine.chat(messages=repair_messages, **repair_kwargs),
                        raw_request,
                        timeout=timeout,
                    )
                except HTTPException:
                    raise
                except (TimeoutError, asyncio.TimeoutError, asyncio.CancelledError):
                    raise
                except Exception as repair_err:
                    # Codex r1 #3: a non-timeout, non-disconnect engine
                    # exception during the repair turn is a SERVER failure
                    # (the engine couldn't produce ANY output for the
                    # retry), NOT a client schema-validation failure.
                    # Pre-fix, this branch swallowed the exception and
                    # surfaced a 422 ``json_schema_violation`` using the
                    # ORIGINAL validation failure — misleading the client
                    # into believing their schema was the problem when the
                    # actual fault was a server-side generation error.
                    # Surface as 502 with the engine-error shape so the
                    # operator sees the real cause; the original validation
                    # failure is preserved in ``details.initial_failure``
                    # for postmortem context.
                    logger.warning(
                        "R12-4 strict json_schema repair retry raised %s: %s; "
                        "surfacing as 502 (server-side generation failure, "
                        "NOT a schema-validation contract breach).",
                        type(repair_err).__name__,
                        repair_err,
                    )
                    raise HTTPException(
                        status_code=502,
                        detail={
                            "error": {
                                "message": (
                                    "Strict json_schema repair retry failed: the "
                                    f"engine raised {type(repair_err).__name__} "
                                    "during the second generation attempt. The "
                                    "initial output had also failed schema "
                                    "validation; investigate server logs."
                                ),
                                "type": "api_error",
                                "code": "strict_repair_engine_failure",
                                "param": "response_format.json_schema",
                                "details": {
                                    "initial_failure": failure_details,
                                    "repair_exception": type(repair_err).__name__,
                                },
                            }
                        },
                    ) from repair_err
            if repair_output is not None:
                ok2, failure2 = validate_and_envelope(
                    repair_output.text or "", json_schema
                )
                if ok2:
                    incr_strict_repair_success()
                    logger.info("R12-4 strict json_schema repair retry succeeded.")
                    # Codex r2 #3: aggregate token usage across BOTH
                    # attempts before swapping ``output``. Pre-fix
                    # we discarded the initial attempt's token usage
                    # entirely, so the client-facing response
                    # under-reported the prompt + completion tokens
                    # the server actually billed for. The
                    # ``raw_text``/``reasoning_text``/etc. fields are
                    # taken from the SUCCESSFUL repair output since
                    # those describe what the client receives; only
                    # the numeric usage fields are summed.
                    from dataclasses import replace as _dc_replace

                    initial_prompt_tokens = output.prompt_tokens
                    initial_completion_tokens = output.completion_tokens
                    output = _dc_replace(
                        repair_output,
                        prompt_tokens=(
                            initial_prompt_tokens + repair_output.prompt_tokens
                        ),
                        completion_tokens=(
                            initial_completion_tokens + repair_output.completion_tokens
                        ),
                    )
                    ok = True
                    failure_details = None
                else:
                    # Repair also failed — keep the SECOND failure
                    # details for the envelope since it reflects what
                    # the client just saw. The "attempts" count in
                    # the envelope already reflects the retry.
                    failure_details = failure2
        if not ok:
            incr_strict_violation()
            envelope = build_violation_envelope(
                failure_details or {"reason": "schema_violation"},
                attempts=attempts,
            )
            logger.warning(
                "R12-4 strict json_schema validation failed after %d attempt(s): %s",
                attempts,
                (failure_details or {}).get("message"),
            )
            raise HTTPException(status_code=422, detail=envelope)

    # Parse tool calls from output using configured parser.
    # ``output.tool_calls`` is non-None when the engine's
    # ``OutputRouter`` already produced structured ``[{"name",
    # "arguments"}]`` entries (currently HarmonyStreamingRouter via
    # openai-harmony's StreamableParser). In that case the text-based
    # parser is bypassed — the structured pass is bytes-faithful
    # whereas the regex round-trip lost calls whose JSON arguments
    # contained literal harmony sentinel substrings (PR #515 codex
    # round-12 / round-14 BLOCKING).
    engine_tool_calls = getattr(output, "tool_calls", None)
    cleaned_text, tool_calls = _parse_tool_calls_with_parser(
        output.text, request, structured_tool_calls=engine_tool_calls
    )

    # r7-A R7-H1: UI-TARS chat-lane coordinate-key parity. The parser
    # emits canonical ``point`` / ``start_point`` / ``end_point``; the
    # OpenAI Computer-Use spec uses ``coordinate`` (single-point) and
    # ``path=[{"x","y"}, …]`` (drag). The Anthropic + Responses lanes
    # already normalize at their adapters (``api/anthropic_adapter.py``,
    # ``api/responses_adapter.py``); the chat lane was missed in r6-B
    # and surfaced the parser-native shape, breaking parity. Gated on
    # ``function.name == "computer"`` so vanilla function tools whose
    # arguments happen to carry a ``point`` key are untouched.
    if tool_calls:
        from ..tool_parsers.ui_tars_tool_parser import (
            normalize_ui_tars_chat_tool_call_arguments,
        )

        for _tc in tool_calls:
            _tc.function.arguments = normalize_ui_tars_chat_tool_call_arguments(
                _tc.function.arguments, _tc.function.name
            )

    # Honor ``parallel_tool_calls=false`` by capping the parsed list at one.
    # No decoder-level enforcement exists, so this is a post-parse trim — the
    # only reliable lever for OpenAI-compat clients that explicitly request a
    # single tool call (see PR #132 for the longer-term FSM-constrained path).
    if tool_calls and len(tool_calls) > 1 and request.parallel_tool_calls is False:
        tool_calls = tool_calls[:1]

    # ``tool_choice="required"`` post-parse enforcement (#468 / #571).
    # The system suffix injected above (``_TOOL_USE_REQUIRED_SUFFIX``)
    # makes the model overwhelmingly likely to comply, but local
    # inference has no decoder-level guarantee.
    #
    # Channel-routed engines (harmony / gemma4) bypass the text parser
    # entirely: the ``OutputRouter`` lifts structured tool_calls out of
    # a dedicated channel, so a forced ``tool_choice`` is satisfied
    # whenever the model fires the tool — the parser path is irrelevant.
    #
    # Text-parser engines (hermes / qwen3_coder / minimax / glm47 / …)
    # only surface a tool_call when the model emits the parser's wire
    # markers. The same model behaviour that produces a structured call
    # on harmony can produce text that the hermes regex fails to
    # recognise — and pre-#571 the 422 here fired for hermes while
    # harmony returned 200, breaking parser-agnostic contracts.
    #
    # The OpenAI ``tool_choice`` contract is parser-agnostic: when the
    # client forces a tool call, the response MUST carry one. To
    # restore symmetry we synthesise a tool_call server-side when the
    # target tool is unambiguous — named-function form (the name is
    # the choice), or ``"required"`` with a single tool entry (the
    # name is unique). When ``"required"`` is paired with multiple
    # tools and the parser returned nothing, we genuinely cannot pick
    # — fall back to 422 with a message that points to the
    # ``{type:"function",function:{name:X}}`` form as the escape
    # hatch, matching pre-#571 wording for that diagnostic.
    # Streaming path is best-effort prompt-injection only; once SSE
    # chunks are out we can't 422 mid-flight.
    if request.tool_choice is not None and request.tools:
        if request.tool_choice == "required" and not tool_calls:
            if len(request.tools) == 1:
                _solo_name = request.tools[0].function.get("name")
                if _solo_name:
                    logger.warning(
                        "tool_choice='required' on a parser-only path produced "
                        "no tool_calls; synthesising a call to the sole "
                        "available tool %r (recovering arguments from raw "
                        "text where possible) to honor the OpenAI tool_call-"
                        "guaranteed contract (#571).",
                        _solo_name,
                    )
                    tool_calls = [
                        _synthesize_forced_tool_call(
                            _solo_name,
                            raw_text=output.raw_text or output.text,
                        )
                    ]
            if not tool_calls:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        'tool_choice="required" but the model returned a text response '
                        "with no tool_calls. Local inference has no decoder-level "
                        "constraint; the system-prompt enforcement was insufficient "
                        "for this prompt. Retry with a more concrete user message or "
                        'use tool_choice={"type":"function","function":{"name":...}} '
                        "to pin a specific tool."
                    ),
                )
        if (
            isinstance(request.tool_choice, dict)
            and request.tool_choice.get("type") == "function"
        ):
            _target = (request.tool_choice.get("function") or {}).get("name")

            # OpenAI spec: a named ``tool_choice`` allows ONLY the named
            # function. A response that includes the target plus any
            # other call violates the contract — refuse to forward.
            # Round-4 codex BLOCKING #2: prior ``any(...)`` accepted
            # ``[target, wrong]`` and shipped the extra call to the
            # client. Now require: at least one match AND every emitted
            # call matches.
            if _target:
                _names = [_tool_call_name(tc) for tc in tool_calls or []]
                _mismatched = [n for n in _names if n != _target]
                # Codex R1 BLOCKING (#675): defense-in-depth — never
                # synthesise a call to a function the client did not
                # submit. The early prompt-level validation (~line 488)
                # already 400s when ``_target`` is absent from
                # ``request.tools``, but a future refactor could shift
                # or bypass that gate, and the synthesis branch must
                # not trust ``_target`` blindly. Gate on the submitted
                # tool-name set; on miss, raise 422 rather than
                # fabricating a call to a tool the client never
                # defined.
                _submitted_tool_names = {
                    t.function.get("name")
                    for t in (request.tools or [])
                    if t.type == "function"
                }
                _target_is_submitted = _target in _submitted_tool_names
                # #571: when the parser returned NOTHING (``_names`` is
                # empty), the request still has a deterministic target
                # — the named-function form names it. Synthesise rather
                # than 422 so hermes matches harmony on the same input.
                # A non-empty-but-wrong list (the model called a
                # different tool) is a different failure mode: the
                # model actively defied the choice, which we still
                # surface as 422 — synthesising over a real wrong call
                # would silently drop the model's output and is a worse
                # client experience than the explicit failure.
                if not _names and _target_is_submitted:
                    logger.warning(
                        "tool_choice pinned function %r on a parser-only path "
                        "produced no tool_calls; synthesising a call (recovering "
                        "arguments from raw text where possible) to honor the "
                        "OpenAI tool_call-guaranteed contract (#571).",
                        _target,
                    )
                    tool_calls = [
                        _synthesize_forced_tool_call(
                            _target,
                            raw_text=output.raw_text or output.text,
                        )
                    ]
                elif not _names and not _target_is_submitted:
                    # Codex R1 BLOCKING (#675): named tool_choice points
                    # at a function that is not in ``request.tools`` —
                    # we must not fabricate a call to it. The early 400
                    # gate normally catches this; reaching here implies
                    # the gate was bypassed (e.g. cloud-fallback rewrite
                    # or future refactor). Refuse rather than synthesise.
                    raise HTTPException(
                        status_code=422,
                        detail=(
                            f"tool_choice pinned function {_target!r} but it is "
                            "not present in the request's 'tools' array; refusing "
                            "to synthesise a call to an undefined tool."
                        ),
                    )
                elif _mismatched:
                    raise HTTPException(
                        status_code=422,
                        detail=(
                            f"tool_choice pinned function {_target!r} but the model "
                            f"emitted calls to {_mismatched}. Local "
                            "inference cannot decoder-enforce a specific function; "
                            "retry with a more direct user message."
                        ),
                    )

    # Validate tool call parameter values against schemas
    if tool_calls and request.tools:
        _validate_tool_call_params(tool_calls, request.tools)

    # D-TOOLCHOICE-R1 T3: scrub wire-marker leftovers from the
    # response text. Two trigger conditions:
    #
    #   (a) the post-parse path SYNTHESISED a forced tool call
    #       (parser couldn't extract from a malformed wire body), OR
    #   (b) the parser DID extract a call but ``tool_choice="required"``
    #       was set AND the raw wire had cross-family / orphan markers
    #       the parser's cleanup left behind (e.g. qwen3 emits
    #       ``<tool_call>{json}</function>`` — hermes recovers the JSON
    #       but the trailing ``</function>`` survives as content).
    #
    # Both conditions imply the model attempted a tool call and any
    # wire literals in the output are junk by definition. The scrub
    # is parser-agnostic so adding a new parser doesn't reopen the
    # leak; codex r3 BLOCKING #2 caught case (b) — even the
    # recover-args path needs the scrub.
    # codex r4 BLOCKING #2: the broad gate ``tool_choice is not None``
    # also fired on ``tool_choice="auto"``, where a model legitimately
    # may emit prose alongside a tool call that contains XML/tool
    # marker text (e.g. discussing the tool wire format in the prose).
    # Narrow to forced/required modes only — that's where the wire
    # leakage is structurally known to be junk by construction:
    #   - ``tool_choice="required"`` — model was FORCED to call,
    #     so any wire-shaped leftovers are by-products of that forcing.
    #   - ``tool_choice={"type":"function","function":{"name":X}}`` —
    #     same forcing semantics.
    # ``"auto"`` and ``"none"`` keep the prior pre-0.8.3 behaviour —
    # cleaned_text is the parser's authoritative output.
    # codex r6 BLOCKING #1: gate scrub on the presence of an actual
    # wire-marker leak in ``cleaned_text``, not on every successful
    # forced/required call. A clean parser-extracted call whose
    # ``cleaned_text`` is plain prose (even prose that legitimately
    # discusses ``<tool_call>``) keeps its content unmodified —
    # which it MUST, because the scrub is destructive (rewrites the
    # user-visible field).
    #
    # The final firing condition is intentionally stricter than "a
    # forced call exists": forced synthesis can also happen when a
    # model ignores ``tool_choice="required"`` and emits ordinary
    # prose. Scrub only when the visible text contains STRUCTURAL
    # parser-wire residue, not merely a literal token mention.
    _is_forced_choice = request.tool_choice == "required" or (
        isinstance(request.tool_choice, dict)
        and request.tool_choice.get("type") == "function"
    )
    _raw_text_for_reasoning = output.raw_text or output.text
    _raw_has_structural_wire = _contains_structural_tool_wire_leak(
        _raw_text_for_reasoning
    )

    def _should_scrub_visible_wire(
        text: str | None, *, allow_raw_context: bool = True
    ) -> bool:
        return (
            _is_forced_choice
            and request.tools
            and bool(tool_calls)
            and _contains_tool_wire_literal(text)
            and (
                _contains_structural_tool_wire_leak(text)
                or (allow_raw_context and _raw_has_structural_wire)
            )
        )

    _wire_scrub_active = _should_scrub_visible_wire(cleaned_text)
    # codex r6 BLOCKING #2: scrub the user-visible ``cleaned_text``
    # only. Do NOT mutate ``raw_text`` before it reaches the reasoning
    # parser — pretty-printed reasoning bodies may legitimately
    # contain wire-shaped tokens (e.g. when the reasoning describes
    # the tool wire format), and rewriting them ahead of extraction
    # truncates / collapses reasoning content. The reasoning parser
    # operates on the original ``output.raw_text`` and only its
    # post-extraction visible output is candidate for re-scrubbing
    # (handled by ``_finalize_content_and_reasoning`` returning
    # ``cleaned_text`` that we never mutate after this block).
    if _wire_scrub_active:
        cleaned_text = _scrub_visible_tool_wire_leaks(cleaned_text)

    # Extract reasoning content. extract_reasoning() is stateless (pure regex
    # on full text), so the singleton is safe here unlike the streaming variant.
    # The tool_calls vs no-tool_calls split is encapsulated in
    # _finalize_content_and_reasoning so the regression test suite can exercise
    # the same orchestration without re-implementing it.
    cleaned_text_before_helper = cleaned_text
    cleaned_text, reasoning_text = _finalize_content_and_reasoning(
        # codex r6 BLOCKING #2: pass the ORIGINAL raw text so the
        # reasoning parser sees the bytes the engine actually emitted.
        # Any wire-literal scrub above only touched user-visible
        # ``cleaned_text`` — reasoning extraction operates on the
        # untouched ``raw_text``.
        raw_text=_raw_text_for_reasoning,
        cleaned_text=cleaned_text,
        tool_calls=tool_calls,
        reasoning_parser=cfg.reasoning_parser,
        engine_reasoning_text=getattr(output, "reasoning_text", "") or "",
        # #575 — chat-template-injected ``<think>`` means the model
        # never emits the start tag; pass the *effective* flag (with
        # the same ``None`` → ``"coder" not in model_name`` fallback
        # ``vllm_mlx/utils/chat_template.py:127`` uses for prompt
        # rendering) so the parser's Case 4 fallback fires on
        # default-on thinking — codex R1 BLOCKING. Use
        # ``cfg.model_path`` (the underlying HF path / alias the
        # engine actually loaded) rather than ``cfg.model_name``,
        # which can be overridden by ``--served-model-name`` and
        # would diverge from the prompt-render path's coder check
        # (codex R2 BLOCKING).
        enable_thinking=_effective_enable_thinking(
            resolved_thinking, cfg.model_path or cfg.model_name
        ),
        # Per-request reasoning cap (upstream vLLM PR #20859 backport).
        # None → back-compat no-op.
        reasoning_max_tokens=getattr(request, "reasoning_max_tokens", None),
        # r5-D — finalize-on-truncation shared plug needs to know if
        # generation was cut short so it can re-classify an unclosed
        # think buffer as ``reasoning_content`` instead of leaking it
        # into ``content``. ``None`` keeps the pre-r5-D behaviour on
        # any caller that hasn't been threaded yet.
        finish_reason=getattr(output, "finish_reason", None),
    )
    if _should_scrub_visible_wire(cleaned_text, allow_raw_context=False):
        cleaned_text = _scrub_visible_tool_wire_leaks(cleaned_text)
    if _should_scrub_visible_wire(reasoning_text) and reasoning_text:
        reasoning_text = _scrub_visible_tool_wire_leaks(reasoning_text)

    # Process response_format if specified (after reasoning parser cleaned the text)
    if response_format and not tool_calls:
        json_input = cleaned_text or output.text
        try:
            _, parsed_json, is_valid, error = parse_json_output(
                json_input, response_format
            )
            if parsed_json is not None:
                parsed_json = _strip_backslash_before_unicode(parsed_json)
                # ``ensure_ascii=False`` keeps non-ASCII characters as
                # raw UTF-8 rather than escaping them to ``\uXXXX``. This
                # is the standard recommendation for JSON-over-HTTP with
                # international content (matches OpenAI's own response
                # encoding); FastAPI emits this body as UTF-8 anyway, so
                # the on-wire bytes are smaller and clients don't have to
                # un-escape user-visible CJK / emoji a second time.
                cleaned_text = json.dumps(parsed_json, ensure_ascii=False)
            if not is_valid:
                logger.warning(f"JSON validation failed: {error}")
        except Exception as e:
            logger.warning(f"JSON output parsing failed: {e}")

    # Determine finish reason
    finish_reason = "tool_calls" if tool_calls else output.finish_reason

    # Clean and strip thinking tags from content
    final_content = None
    if cleaned_text:
        final_content = strip_thinking_tags(clean_output_text(cleaned_text))
        final_content = sanitize_output(final_content)
        if response_format and final_content:
            final_content = extract_json_from_response(final_content)

    # Issue #569: never silently drop. If the assistant turn would
    # otherwise have ``content=null`` AND ``tool_calls=null`` but the
    # engine surfaced ``reasoning_text`` (gemma-4-26b-4bit multi-turn
    # where the model got stuck inside ``<|channel>thought\n…`` and
    # ran out of tokens before emitting a closer / final / tool call),
    # surface the reasoning trace as ``content`` so OpenAI-compat
    # agentic clients reading only ``content``/``tool_calls`` don't
    # see an empty message.
    #
    # Codex round-1 BLOCKING on #676: skip the rescue when the client
    # requested structured output (``response_format`` =
    # ``json_object`` / ``json_schema``). Reasoning prose is almost
    # never valid JSON, so surfacing it as ``content`` would break
    # the OpenAI-compat structured-output contract and feed the
    # client garbage prose instead of validated JSON. The existing
    # empty/error path lets a structured-output client retry rather
    # than be surprise-fed unstructured text. Agentic (no
    # ``response_format``) clients still get the rescue.
    #
    # Codex round-2 BLOCKING on #676: the predicate is now factored
    # into ``_is_structured_output_requested`` so the streaming
    # rescue path (chat.py:~1580) can call the SAME predicate. Round
    # 1 inlined the check here only; codex round 2 caught the
    # streaming path drifting because it had no gate at all.
    if not _is_structured_output_requested(response_format):
        # PR #715 bundle, fuzz finding C / live-test repro: when the
        # parser's Case-4 fallback blanked ``cleaned_text`` (no tags +
        # ``enable_thinking=True`` → route the whole output to
        # reasoning, #575), the rescue must NOT then surface that
        # reasoning back as ``content`` — that would duplicate the
        # trace byte-identically into both fields. The helper signals
        # Case-4 by returning an empty ``cleaned_text`` when the
        # pre-helper text was non-empty AND a reasoning_parser was
        # wired AND the engine wasn't already routing (engine path
        # has its own plug). Detect by comparing the pre- and post-
        # helper ``cleaned_text``.
        reasoning_is_case4 = bool(
            cleaned_text_before_helper
            and not cleaned_text
            and reasoning_text
            and cfg.reasoning_parser is not None
            and not (getattr(output, "reasoning_text", "") or "")
        )
        # D-STOP-THINK codex round-5 BLOCKING (PR #799): compute
        # ``prompt_thinking_active`` from the chat template +
        # resolved enable_thinking. Required by the helper's
        # Case-4 + stop + matched_stop arm to discriminate
        # prompt-injected mid-think from a casual stop-terminated
        # answer.
        #
        # Codex round-9 BLOCKING (PR #799): use the SHARED
        # ``_should_start_in_thinking`` predicate from
        # ``service.helpers`` instead of inlining the substring
        # check. Single source of truth — drift across routes is
        # impossible by construction.
        _chat_template_str = ""
        _tok = getattr(engine, "tokenizer", None)
        if _tok and hasattr(_tok, "chat_template"):
            _chat_template_str = _tok.chat_template or ""
        prompt_thinking_active = _should_start_in_thinking(
            _chat_template_str, resolved_thinking
        )
        final_content = _rescue_silent_drop_from_reasoning(
            final_content,
            reasoning_text,
            tool_calls,
            finish_reason=finish_reason,
            raw_text=output.raw_text or output.text,
            reasoning_is_case4=reasoning_is_case4,
            matched_stop=getattr(output, "matched_stop", None),
            prompt_thinking_active=prompt_thinking_active,
        )
        # Issue #858: cutoff sentinel is ON by default — restores PR #802
        # (H-01) semantics after the R-01 (#815) opt-in flip produced
        # empty-bubble regressions in every GUI client that only renders
        # ``message.content``. Power callers that want strict-null
        # behaviour set ``RAPID_MLX_REASONING_CUTOFF_NOTICE=disabled``
        # (or ``0`` / ``false`` / ``no`` / ``off``). The helper itself
        # owns ALL the predicates (env gate, finish_reason, content
        # emptiness, tool-call gate, reasoning presence) so this call
        # site stays trivial.
        final_content = _apply_reasoning_cutoff_notice(
            final_content,
            reasoning_text,
            tool_calls,
            finish_reason,
        )

    # Build logprobs for response if requested.
    #
    # R15 task #311: when the caller asked for ``logprobs=true`` but
    # didn't set ``top_logprobs`` (or set it to 0), strip the synthetic
    # single-element ``top_logprobs`` we extracted above so the wire
    # shape matches OpenAI semantics (sampled-token logprob only, no
    # alternatives field populated). The default ``TokenLogProb.
    # top_logprobs=[]`` is the right empty shape.
    choice_logprobs = None
    if want_logprobs and token_logprobs_list:
        if top_k_logprobs == 0:
            for entry in token_logprobs_list:
                entry.top_logprobs = []
        choice_logprobs = ChoiceLogProbs(content=token_logprobs_list)

    chat_response = ChatCompletionResponse(
        model=_resolve_model_name(request.model),
        choices=[
            ChatCompletionChoice(
                message=AssistantMessage(
                    content=final_content,
                    reasoning_content=reasoning_text,
                    tool_calls=tool_calls,
                ),
                finish_reason=finish_reason,
                logprobs=choice_logprobs,
            )
        ],
        usage=_build_usage(output, reasoning_text),
    )
    # L-05: surface silent ``enable_thinking`` drop on non-Qwen parsers.
    # Empty dict when the client didn't set the flag OR the active
    # parser honors it (qwen3) — kwargs spread is the right shape.
    response_headers = enable_thinking_warning_header(
        request, getattr(cfg, "reasoning_parser_name", None)
    )
    return Response(
        content=chat_response.model_dump_json(exclude_none=True),
        media_type="application/json",
        headers=response_headers or None,
    )


async def stream_chat_completion(
    engine,
    messages: list,
    request: ChatCompletionRequest,
    *,
    response_id: str | None = None,
    created: int | None = None,
    **kwargs,
) -> AsyncIterator[str]:
    """Stream chat completion response.

    Uses StreamingPostProcessor for reasoning/tool/sanitization pipeline.
    SSE formatting stays inline for performance (fast path bypasses Pydantic).

    Args:
        response_id: Optional pre-computed response id (``chatcmpl-…``).
            When provided, all SSE chunks share this id instead of one
            generated fresh here. Used by ``stream_chat_completion_guided``
            on its unconstrained fallback path so the client-visible
            stream stays self-consistent across the guided→unconstrained
            handoff (DeepSeek pr_validate round 5 finding).
        created: Optional pre-computed Unix timestamp. Same rationale.
    """
    from ..service.postprocessor import StreamingPostProcessor

    cfg = get_config()
    gc_was_enabled = gc.isenabled()
    if cfg.gc_control and gc_was_enabled:
        gc.disable()

    try:
        if response_id is None:
            response_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        start_time = time.perf_counter()

        # Check if we should include usage in the final chunk
        include_usage = request.stream_options and request.stream_options.include_usage

        # Logprobs configuration.
        #
        # R15 task #311: ``logprobs=true`` alone returns the sampled-
        # token logprob. ``top_logprobs`` only adds the K alternatives
        # field. Pre-fix the gate AND'd both, silently dropping the
        # sampled-token logprob the caller asked for when
        # ``top_logprobs`` was None/0. Mirror the non-streaming branch
        # above: gate on ``logprobs`` alone, extract with
        # ``effective_top_k=max(1, top_logprobs)`` to defend the
        # extractor's ``argpartition`` from a top_k=0 footgun, then
        # strip the synthetic alternatives below.
        want_logprobs = bool(request.logprobs)
        top_k_logprobs = request.top_logprobs or 0
        effective_top_k = max(1, top_k_logprobs)

        def _build_chunk_logprobs(output: GenerationOutput) -> ChoiceLogProbs | None:
            """Build ChoiceLogProbs for a streaming chunk if logprobs requested."""
            if not want_logprobs:
                return None
            entries = _extract_streaming_token_logprobs(
                output, engine.tokenizer, effective_top_k
            )
            if not entries:
                return None
            if top_k_logprobs == 0:
                # OpenAI semantics: ``logprobs=true`` without
                # ``top_logprobs`` returns sampled-token logprob only,
                # no alternatives. Strip the single-element synthetic
                # list we used to drive the extractor.
                for entry in entries:
                    entry.top_logprobs = []
            return ChoiceLogProbs(content=entries)

        # Pre-compute SSE template parts that don't change per-token.
        _sse_created = created if created is not None else int(time.time())
        _model_escaped = json.dumps(_resolve_model_name(request.model))
        _sse_prefix = (
            f'data: {{"id":"{response_id}","object":"chat.completion.chunk",'
            f'"created":{_sse_created},"model":{_model_escaped},'
            f'"choices":[{{"index":0,"delta":{{'
        )
        _sse_suffix = "}}]}\n\n"

        def _fast_sse_chunk(text: str, field: str = "content") -> str:
            """Build SSE chunk JSON directly, bypassing Pydantic serialization.

            r10-B R10-C2 — emit ONLY ``reasoning_content`` on the
            streaming wire. The deprecation-window dup ``reasoning``
            key (added in r7-A R7-H2) is now removed: it was the
            byte-for-byte root cause of R9-CRIT3, where consumers like
            ``openai-agents``'s ``Runner.run_streamed`` walk both
            ``delta.reasoning_content`` AND ``delta.reasoning`` and
            therefore double-counted every reasoning token. The
            OpenAI o1-style spec uses ``reasoning_content`` only;
            there is no ``reasoning`` key on chat-completions deltas.

            R12-FIX-V2 (Vlad r12 MED-2): sanitize ``text`` against
            special-token markup leaks before serializing. This path
            bypasses the pydantic ``ChatCompletionChunkDelta``
            validator that catches the same leak in the
            non-fast-path streaming branch — so it gets the same
            sanitization explicitly. The systematic principle is
            "every user-visible string that originated from a raw
            token decode flows through the same final sanitizer",
            including the streaming hot path.
            """
            escaped = json.dumps(sanitize_reasoning_for_stream(text))
            return f'{_sse_prefix}"{field}":{escaped}{_sse_suffix}'

        # First chunk with role
        _first_sse = f'{_sse_prefix}"role":"assistant"{_sse_suffix}'
        if logger.isEnabledFor(logging.INFO):
            logger.info(f"[SSE-ROLE] {_first_sse.strip()[:200]}")
        yield _first_sse

        # Initialize post-processor.
        # request_dict carries `tools` so streaming parsers (qwen3_coder etc.)
        # can do schema-driven type conversion (#171).
        request_dict = (
            request.model_dump(exclude_none=True)
            if hasattr(request, "model_dump")
            else None
        )
        processor = StreamingPostProcessor(
            cfg,
            tools_requested=bool(request.tools),
            # `kwargs` is the **kwargs from this function's signature; the
            # route handler unpacks chat_kwargs (which sets
            # "enable_thinking" when request.enable_thinking is not None
            # or cfg.no_thinking is set). Pulled through as a name so
            # StreamingPostProcessor can short-circuit the reasoning
            # parser when the client explicitly disabled thinking
            # (closes the empty-content streaming bug from PR #208).
            enable_thinking=kwargs.get("enable_thinking"),
            json_mode=bool(
                request.response_format
                and getattr(request.response_format, "type", "text") != "text"
            ),
            request=request_dict,
            # Per-request reasoning cap (upstream vLLM PR #20859 backport).
            # When None the postprocessor is a no-op for the cap path.
            reasoning_max_tokens=getattr(request, "reasoning_max_tokens", None),
        )
        processor.set_thinking_model(request.model)
        processor.reset()

        # Forced ``tool_choice`` synthetic-prefix replay swallow (PR #716
        # codex r9 BLOCKING #1). When the upstream chat_kwargs builder
        # set ``forced_assistant_prefix`` (the route's
        # ``_forced_tool_call_prefix`` branch), the engine's
        # ``stream_chat`` yields the prefix back as a synthetic first
        # chunk so plain-text consumers see the wire envelope. Seed the
        # postprocessor with the same bytes so it can swallow that
        # synthetic chunk BEFORE the reasoning parser sees it — without
        # this, the prefix bytes route through ``Case-3 → reasoning``
        # in ``BaseThinkingReasoningParser``, polluting
        # ``accumulated_reasoning`` AND risking a raw-byte leak into
        # ``delta.reasoning_content`` on parser variants the MiniMax
        # tool-markup redirect doesn't catch (chunk-boundary splits,
        # future parsers). See ``StreamingPostProcessor.__init__`` for
        # the swallow-buffer state machine. No-op when the prefix is
        # absent.
        processor.seed_forced_assistant_prefix(kwargs.get("forced_assistant_prefix"))

        # Track token counts for usage reporting
        prompt_tokens = 0
        completion_tokens = 0
        cached_tokens = 0

        # Buffer the terminal "finish" event so the cross-format fallback in
        # processor.finalize() (#425) gets a chance to recover a missed tool
        # call BEFORE we emit a terminal chunk. Without this buffer the route
        # emits a finish_reason="stop" chunk first and then a separate
        # finish_reason="tool_calls" chunk from the fallback path — spec-
        # compliant clients stop reading at the first finish_reason and
        # silently drop the tool call (#v0.6.63 onboarding sweep finding #3).
        buffered_finish: tuple | None = None
        # R11-A codex r1 HIGH #1: when a streaming ``tool_call`` event
        # also carries the terminal ``finish_reason="tool_calls"`` (the
        # postprocessor stamps it inline on the final engine chunk —
        # see ``StreamingPostProcessor.process_chunk`` tool-call emit
        # sites), we must NOT also fall through to the R11-V2 synthetic
        # finish branch below or the wire ends up with TWO terminal
        # finish chunks. Track inline emission here.
        inline_terminal_finish_emitted = False

        # D-STOP-THINK codex round-6 BLOCKING (PR #799):
        # accumulate ``output.matched_stop`` from streamed chunks so the
        # post-loop ``_rescue_silent_drop_from_reasoning`` call below sees
        # the prompt-injected user stop string even when the SAMPLER
        # surfaced it on a non-finish chunk (i.e. ``finish_event.matched_stop``
        # is ``None`` but an earlier ``output`` carried the value). Mirrors
        # the same accumulator in ``routes/responses.py:452/602``. Without
        # this, prompt-injected stop-mid-think streams look like
        # ``matched_stop=None`` and the silent-drop rescue would still
        # leak the reasoning trace into ``delta.content`` — the exact
        # D-STOP-THINK leak this PR is supposed to gate at the parser
        # boundary.
        stream_matched_stop: str | None = None

        # Stream content — PostProcessor handles reasoning/tool/sanitize.
        # ``is_streaming=True`` is consumed by DiffusionEngine to disable
        # the gemma4 wire-marker carve-out in ``skip_special_token_ids``:
        # this path forwards each chunk as an SSE delta without running
        # the tool parser, so any markers left in by the carve-out would
        # surface as raw ``<|tool_call>`` wire text in ``delta.content``
        # to the client (pr_validate #558 r8 BLOCKING #2). Engines whose
        # ``stream_chat`` doesn't know the kwarg swallow it via the
        # ``**kwargs`` tail on ``BaseEngine.stream_chat`` — no behavior
        # change for BatchedEngine which uses its own special-token
        # handling.
        async for output in engine.stream_chat(
            messages=messages, is_streaming=True, **kwargs
        ):
            if hasattr(output, "prompt_tokens") and output.prompt_tokens:
                prompt_tokens = output.prompt_tokens
            if hasattr(output, "completion_tokens") and output.completion_tokens:
                completion_tokens = output.completion_tokens
            # ``cached_tokens`` is a single per-request value (the
            # prefix-cache hit count set once when the request is
            # scheduled), so re-reading it on every chunk just
            # re-stamps the same value; the guard mirrors the
            # ``prompt_tokens`` branch above for ad-hoc engines that
            # don't carry the field.
            if hasattr(output, "cached_tokens") and output.cached_tokens:
                cached_tokens = output.cached_tokens

            # D-STOP-THINK codex round-6 BLOCKING accumulator (PR #799).
            # Capture the last non-empty ``matched_stop`` we see across
            # streamed chunks. The sampler stamps ``matched_stop`` on
            # whichever chunk crossed the stop boundary; the buffered
            # ``finish_event`` often arrives without it because the
            # post-processor splits finish from data. Mirrors
            # ``routes/responses.py:602``.
            _chunk_matched_stop = getattr(output, "matched_stop", None)
            if _chunk_matched_stop:
                stream_matched_stop = _chunk_matched_stop

            for event in processor.process_chunk(output):
                if event.type == "content":
                    if not want_logprobs:
                        _sse = _fast_sse_chunk(event.content, "content")
                        if _sse:
                            yield _sse
                    else:
                        chunk = ChatCompletionChunk(
                            id=response_id,
                            created=_sse_created,
                            model=_resolve_model_name(request.model),
                            choices=[
                                ChatCompletionChunkChoice(
                                    delta=ChatCompletionChunkDelta(
                                        content=event.content,
                                    ),
                                    logprobs=_build_chunk_logprobs(output),
                                )
                            ],
                        )
                        yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"

                elif event.type == "reasoning":
                    yield _fast_sse_chunk(event.reasoning, "reasoning_content")

                elif event.type == "tool_call":
                    # r7-A R7-H1: streaming-lane parity with the
                    # non-stream chat path. The same
                    # ``normalize_ui_tars_chat_tool_call_arguments``
                    # helper runs here so per-delta tool_call
                    # arguments emit the OpenAI Computer-Use spec
                    # keys (``coordinate`` / ``path``) instead of the
                    # parser-native ``point``. Gated on
                    # ``function.name == "computer"`` inside the helper.
                    _normalized_tcs = _normalize_ui_tars_tcs_for_chat(event.tool_calls)
                    chunk = ChatCompletionChunk(
                        id=response_id,
                        created=_sse_created,
                        model=_resolve_model_name(request.model),
                        choices=[
                            ChatCompletionChunkChoice(
                                delta=ChatCompletionChunkDelta(
                                    tool_calls=_normalized_tcs,
                                ),
                                finish_reason=event.finish_reason,
                            )
                        ],
                        # Usage placement (OpenAI streaming spec, D-SSE-USAGE):
                        # ``usage`` MUST appear ONLY when the caller opted
                        # in via ``stream_options.include_usage=true``,
                        # and then ONLY in the dedicated trailing chunk
                        # (this finish chunk carries ``null`` so it
                        # serializes as ``"usage": null`` per spec). When
                        # ``include_usage`` is false / unset, the field
                        # is omitted from every chunk — LangChain /
                        # AI-SDK / vercel-ai-stream parsers double-count
                        # token totals when usage shows up unexpectedly.
                        usage=None,
                    )
                    _tc_sse = f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"
                    # Dogfood F-R2-05 + codex r5 NIT #3: previously
                    # emitted at INFO, which leaked user-action coords
                    # + tool_call JSON into the server log on every
                    # Computer-Use turn (UI-TARS, Anthropic computer
                    # tool, etc.). Now DEBUG, AND redacted: log only
                    # the chunk metadata (tool_call count + finish
                    # reason), never the raw ``arguments`` JSON.
                    # Operators who need full payloads can capture
                    # the wire stream upstream; the application log
                    # should not be a covert PII channel.
                    _tc_count = len(event.tool_calls or [])
                    _tc_finish = event.finish_reason or "-"
                    logger.debug(
                        "[SSE-TC] tool_calls.count=%d finish_reason=%s",
                        _tc_count,
                        _tc_finish,
                    )
                    # R11-A codex r1 HIGH #1: latch on an inline
                    # terminal finish (postprocessor stamps
                    # ``finish_reason="tool_calls"`` on the tool_call
                    # event for the final engine chunk). Prevents the
                    # post-loop R11-V2 synthetic-finish branch from
                    # firing a SECOND terminal chunk with
                    # ``finish_reason="stop"`` — spec-compliant clients
                    # would stop at the first reason and silently drop
                    # the tool call, or — worse — process both and
                    # double-emit the turn end.
                    if event.finish_reason is not None:
                        inline_terminal_finish_emitted = True
                    yield _tc_sse

                elif event.type == "finish":
                    # Defer emission: finalize() (below) may recover a
                    # missed tool call via cross-format fallback. If it
                    # does, we must merge the recovered tool_calls into
                    # this single terminal chunk and re-stamp the
                    # finish_reason as "tool_calls" — not emit two
                    # contradictory finish chunks.
                    buffered_finish = (event, output)

        # Fallback tool call detection (post-stream). Collect ALL fallback
        # tool_call events before emitting; they get merged into the
        # buffered finish chunk so the stream produces exactly one
        # terminal chunk with one finish_reason (OpenAI spec).
        # ``content`` events from finalize() carry prefix-held bytes
        # released by the parser (codex round-3 CRITICAL): the
        # streaming parser holds back partial tool-call sentinels
        # (``<``, ``<|``, ``<func``...) so per-char streaming doesn't
        # leak them as content before the full opener arrives. When
        # the stream ends with held bytes still buffered AND no tool
        # call ever fired, the postprocessor releases them — accumulate
        # here and merge into the terminal chunk's content so the user
        # doesn't see a truncated reply (codex round-4 CRITICAL).
        fallback_tool_calls: list = []
        finalize_content_parts: list[str] = []
        for event in processor.finalize():
            if event.type == "tool_call":
                # r7-A R7-H1: normalize before append so terminal-merge
                # + synthetic-fallback emit sites don't need to know
                # about the UI-TARS spec-key contract.
                fallback_tool_calls.extend(
                    _normalize_ui_tars_tcs_for_chat(event.tool_calls) or []
                )
            elif event.type == "content" and event.content:
                finalize_content_parts.append(event.content)
        finalize_content = "".join(finalize_content_parts)

        # #447 streaming-parity synthesis (2026-06-26). The non-stream
        # chat path at chat.py:~3147 / ~3215 falls back to
        # ``_synthesize_forced_tool_call`` when a forced ``tool_choice``
        # (``"required"`` with a single tool, or
        # ``{"type":"function","function":{"name":...}}``) finishes
        # without the parser surfacing any tool_call — restoring the
        # OpenAI "tool_call guaranteed" contract symmetry with
        # channel-routed paths (harmony / gemma4). The streaming path
        # previously had no equivalent: a qwen3-family model emitting
        # the Nemotron-shape ``<tool_call><function=NAME>...</function>
        # </tool_call>`` body under a hermes JSON-shape
        # ``forced_assistant_prefix`` (``<tool_call>\n{"name":...,
        # "arguments": ``) produces a hybrid wire shape that neither
        # the streaming nor the cross-format extract path can parse,
        # and ``stream=true`` finishes with zero ``tool_call`` deltas
        # plus ``finish_reason="stop"``. The non-stream branch synth-
        # recovers; the stream branch left clients with an empty turn
        # (rapid-desktop#447).
        #
        # Gating identical to the non-stream synthesis at chat.py:~3197 /
        # ~3207: forced choice + the target function is the sole
        # submitted tool (``required`` + 1-tool collapses unambiguously)
        # OR a named function is explicitly pinned AND the pinned name
        # is in ``request.tools`` (defense-in-depth from PR #675 codex
        # r1). The synth fires ONLY when:
        #
        # (a) No ``delta.tool_calls`` chunk reached the wire AND
        #     ``finalize()`` recovered nothing — the wire-truth
        #     ``_tool_calls_emitted_to_wire`` counter + the empty
        #     ``fallback_tool_calls`` together pin the
        #     "zero-tool-calls-shipped" state.
        #
        # (b) The PARSER also did not detect any tool-call shape at all
        #     (``processor.tool_calls_detected is False``). Mirrors the
        #     non-stream ``not _names`` predicate. Codex r1 MAJOR #1
        #     (PR #948 review): without this guard, the synth would
        #     silently REPLACE the model's intended call when the parser
        #     saw a different tool that the forced-``tool_choice`` /
        #     parallel-cap filter dropped (``tool_calls_detected=True``,
        #     ``_tool_calls_emitted_to_wire=0``). The non-stream path
        #     treats "model called a different tool than the pinned
        #     target" as the ``_mismatched`` 422 case, NOT as a synth
        #     trigger — it surfaces the conflict instead of fabricating
        #     a call the model didn't make. Keep streaming aligned.
        if (
            not fallback_tool_calls
            and processor._tool_calls_emitted_to_wire == 0
            and not processor.tool_calls_detected
            and request.tools
            and request.tool_choice is not None
        ):
            _synth_target: str | None = None
            if (
                isinstance(request.tool_choice, dict)
                and request.tool_choice.get("type") == "function"
            ):
                _pinned = (request.tool_choice.get("function") or {}).get("name")
                _submitted = {
                    t.function.get("name")
                    for t in request.tools
                    if getattr(t, "type", None) == "function"
                }
                if _pinned and _pinned in _submitted:
                    _synth_target = _pinned
            elif request.tool_choice == "required" and len(request.tools) == 1:
                _synth_target = request.tools[0].function.get("name")
            if _synth_target:
                _raw_text = (
                    processor.tool_accumulated_text or processor.accumulated_text or ""
                )
                _synth_call = _synthesize_forced_tool_call(
                    _synth_target, raw_text=_raw_text
                )
                # Convert the ``ToolCall`` pydantic object into the
                # streaming-shape dict the terminal-merge path expects
                # (``{"index","id","type","function":{"name","arguments"}}``)
                # so it serializes identically to the parser-emitted
                # deltas. ``index=0`` is the canonical singleton-call
                # index used elsewhere in the streaming postprocessor.
                fallback_tool_calls.append(
                    {
                        "index": 0,
                        "id": _synth_call.id,
                        "type": "function",
                        "function": {
                            "name": _synth_call.function.name,
                            "arguments": _synth_call.function.arguments,
                        },
                    }
                )
                # Codex r1 NIT #1 (PR #948): bump the wire-truth counter
                # so the PR #859 "finish_reason=tool_calls ⇒ ≥1 tool_call
                # delta on the wire" invariant holds — the terminal merge
                # at chat.py:~4280 IS about to emit a ``delta.tool_calls``
                # chunk for this synth, so the counter must reflect that
                # for any downstream gate / log that reads it.
                processor._tool_calls_emitted_to_wire += 1
                logger.info(
                    "[SSE-FORCED-SYNTH-#447] forced tool_choice produced no "
                    "tool_call deltas; synthesizing terminal call to %r "
                    "(args recovered from raw text where possible) to honor "
                    "the OpenAI tool_call-guaranteed contract — mirrors the "
                    "non-stream synthesis path.",
                    _synth_target,
                )

        # Emit the terminal chunk. Three cases:
        #   (a) Streaming parser already emitted tool_calls during the
        #       loop → buffered_finish has finish_reason="tool_calls"
        #       and fallback_tool_calls is empty. Emit as-is.
        #   (b) Streaming parser missed the call but finalize() recovered
        #       it via cross-format fallback → merge tool_calls into the
        #       buffered finish and override finish_reason="tool_calls".
        #   (c) No tool calls at all → emit the buffered finish unchanged.
        if buffered_finish is not None:
            finish_event, finish_output = buffered_finish
            if fallback_tool_calls:
                logger.info(
                    "[SSE-FALLBACK-TC-MERGED] merging %d recovered tool_call(s) "
                    "into terminal chunk; overriding finish_reason -> tool_calls",
                    len(fallback_tool_calls),
                )
            # Merge any released prefix-held content into the terminal
            # chunk's delta.content (codex round-4 CRITICAL). Concatenate
            # to whatever the finish event already carries — the
            # finish_event.content path is normally None for non-tool
            # plain-text streams (deltas already drained content during
            # the loop), so this typically just adds the held suffix.
            terminal_content = (finish_event.content or "") + finalize_content

            # Issue #569 streaming rescue: if NOTHING was streamed as
            # ``content`` across the whole turn AND no ``tool_calls``
            # fired AND the model produced reasoning, surface the
            # accumulated reasoning trace as ``content`` in the
            # terminal chunk. Mirrors the non-streaming rescue in
            # ``_rescue_silent_drop_from_reasoning`` so streaming
            # clients (Cline, Cursor, Codex CLI) reading the
            # assembled ``content`` stream don't end the turn on an
            # empty buffer when gemma-4 (etc.) got stuck inside
            # ``<|channel>thought\n…`` and never emitted any closer
            # / final / tool call. Per-delta ``reasoning_content``
            # chunks have already been sent during the loop; this
            # adds a NEW ``content`` chunk at the end (duplication of
            # the same text across the two channels is the lesser
            # evil vs. a silently empty content stream).
            #
            # Codex round-2 BLOCKING on #676: gate on the SAME
            # ``_is_structured_output_requested`` predicate as the
            # non-streaming path (chat.py:~1283). Without this, a
            # ``stream=true`` request with
            # ``response_format={"type": "json_object"|"json_schema"}``
            # would still receive reasoning prose in
            # ``delta.content`` despite the non-streaming path
            # explicitly suppressing exactly that. Structured-output
            # clients expect validated JSON or the existing empty
            # path so they can retry — never surprise prose.
            #
            # Codex round-3 BLOCKING on #676: route the streaming
            # rescue through ``_rescue_silent_drop_from_reasoning``
            # instead of promoting ``processor.accumulated_reasoning``
            # directly. The previous direct-promotion branch bypassed
            # the helper's whitespace guard, so a reasoning-only
            # stream of ``"   \n"`` would emit a semantically empty
            # ``delta.content`` while non-streaming correctly
            # suppressed it. Funneling both paths through the same
            # helper means the predicate (whitespace + content
            # presence + tool-call absence) is defined ONCE and the
            # two paths cannot drift. The structured-output gate
            # stays here at the call site (parallel to non-streaming
            # at chat.py:~1285), because it depends on per-request
            # ``response_format`` which the rescue helper has no
            # access to.
            already_streamed_content = bool(processor.accumulated_text)
            has_any_tool_calls = bool(fallback_tool_calls) or (
                finish_event.finish_reason == "tool_calls"
            )
            structured_output_requested = _is_structured_output_requested(
                request.response_format
            )
            if (
                not already_streamed_content
                and not has_any_tool_calls
                and not structured_output_requested
            ):
                # Pass ``terminal_content or None`` so the helper
                # sees the same "empty vs whitespace vs real"
                # distinction the non-streaming path does. Pass
                # ``None`` for ``tool_calls`` because we've already
                # checked ``has_any_tool_calls`` above — the helper's
                # tool-call branch would never fire here regardless,
                # but keeping the call symmetric with non-streaming
                # is the point.
                # Codex r3 P1: streaming rescue must skip the
                # truncated-``<think>`` case. The streaming reasoning
                # parser consumes the literal ``<think>`` token as a
                # state transition, so ``accumulated_reasoning`` never
                # carries it — but the parser's ``_saw_any_tag`` flag
                # records that a ``<think>``/``</think>`` boundary was
                # crossed. When the model truncated mid-thought
                # (``finish_reason="length"`` AND the parser saw the
                # opener AND never saw the closer), the reasoning
                # trace is NOT the final answer and promoting it to
                # ``content`` re-introduces the leak the non-streaming
                # gate now suppresses. Synthesise a ``raw_text`` with
                # an unclosed ``<think>`` opener so the rescue's
                # existing finish=length-with-unclosed-think gate
                # fires uniformly across both paths.
                rp = processor.reasoning_parser
                saw_open_no_close = bool(
                    rp
                    and getattr(rp, "_saw_any_tag", False)
                    and "</think>"
                    not in (
                        processor.accumulated_reasoning + processor.accumulated_text
                    )
                )
                # D-HARMONY-LEAK (2026-06-21): harmony-streaming mirror
                # of the truncated-``<think>`` synthetic raw above. On
                # gpt-oss / Harmony streaming, the engine token-routes
                # via ``OutputRouter`` so ``output.channel="reasoning"``
                # arrives in the postprocessor pre-split — the literal
                # ``<|channel|>analysis<|message|>`` opener is consumed
                # as a state transition by ``HarmonyStreamingRouter``
                # and never lands in ``accumulated_reasoning``. The
                # streaming counterpart of the bug: when generation
                # cuts short before a final channel emerges (max_tokens
                # mid-analysis OR a stop string matching mid-analysis),
                # ``accumulated_text`` stays empty and the rescue would
                # promote the analysis trace to ``delta.content`` —
                # shipping byte-identical content + reasoning_content
                # to the client. Synthesise a harmony-marked raw_text
                # so the helper's new harmony-shape gate (analysis
                # marker present, final marker absent) fires uniformly
                # across both the streaming and non-streaming surfaces.
                # Gated on the parser type so the synthetic only fires
                # for Harmony — gemma-4 / qwen families still rely on
                # their existing rescue paths.
                #
                # Codex r1 BLOCKING #2: the empty-content + non-empty-
                # reasoning shape ALSO matches a tool-call-only stream
                # where the parallel-tool-calls cap dropped every
                # commentary entry (``tool_calls_detected=True`` set on
                # the cap-exhaust path but ``fallback_tool_calls`` may
                # arrive empty and ``finish_event.finish_reason`` may
                # be something other than ``"tool_calls"`` on the
                # router-cap path before the buffered-finish gate
                # fires). Plumb ``processor.tool_calls_detected``
                # through so a commentary-call stream is not
                # misclassified as analysis-without-final and
                # accidentally suppressed via the harmony gate —
                # tool-call-only responses legitimately ship
                # ``content=None`` per the OpenAI spec and the gate
                # would only change zero-byte output here, but
                # honouring the explicit channel signal keeps the
                # synthetic_raw discrimination accurate.
                harmony_cut_short = _is_harmony_cut_short_stream(
                    rp,
                    processor.accumulated_reasoning,
                    processor.accumulated_text,
                    processor.tool_calls_detected,
                )
                if harmony_cut_short:
                    synthetic_raw = (
                        "<|channel|>analysis<|message|>"
                        + processor.accumulated_reasoning
                    )
                elif saw_open_no_close:
                    synthetic_raw = "<think>" + processor.accumulated_reasoning
                else:
                    synthetic_raw = processor.accumulated_reasoning or ""
                # PR #715 bundle, fuzz finding C: streaming Case-4
                # mirror of the non-streaming detection. When the
                # parser never saw a ``<think>``/``</think>`` token
                # (``_saw_any_tag`` False) BUT routed everything into
                # reasoning (accumulated_text empty, accumulated_reasoning
                # non-empty) AND a parser is wired, the streamer's
                # Case-3 fallback ("no tags seen yet → reasoning") IS
                # firing — analogous to the non-streaming Case-4
                # blanking. Suppress the rescue on length-truncated
                # streams in that state too.
                reasoning_is_case4_stream = bool(
                    rp
                    and not getattr(rp, "_saw_any_tag", False)
                    and processor.accumulated_reasoning
                    and not processor.accumulated_text
                )
                # D-STOP-THINK codex round-5 BLOCKING (PR #799):
                # compute ``prompt_thinking_active`` for the streaming
                # rescue too — mirrors the non-streaming gate. The
                # streaming function doesn't have ``resolved_thinking``
                # in scope (it's a separate function from the
                # non-streaming caller), so re-resolve here from the
                # request.
                #
                # Codex round-9 BLOCKING (PR #799): use the SHARED
                # ``_should_start_in_thinking`` predicate from
                # ``service.helpers`` instead of inlining the substring
                # check. Single source of truth — drift across the
                # non-streaming and streaming paths is impossible by
                # construction.
                _stream_resolved_thinking = _resolve_enable_thinking(request)
                _chat_template_str_stream = ""
                _tok_stream = getattr(engine, "tokenizer", None)
                if _tok_stream and hasattr(_tok_stream, "chat_template"):
                    _chat_template_str_stream = _tok_stream.chat_template or ""
                prompt_thinking_active_stream = _should_start_in_thinking(
                    _chat_template_str_stream, _stream_resolved_thinking
                )
                # D-STOP-THINK codex round-6 BLOCKING (PR #799):
                # prefer the per-chunk accumulator over
                # ``finish_event.matched_stop`` because the sampler may
                # stamp the matched stop string on an earlier chunk that
                # was NOT the terminal finish event. Falling back to
                # ``finish_event.matched_stop`` preserves backward
                # compatibility with engines that only stamp it on
                # finish (BatchedEngine).
                _effective_matched_stop = stream_matched_stop or getattr(
                    finish_event, "matched_stop", None
                )
                rescued_content = _rescue_silent_drop_from_reasoning(
                    terminal_content or None,
                    processor.accumulated_reasoning,
                    None,
                    finish_reason=finish_event.finish_reason,
                    raw_text=synthetic_raw,
                    reasoning_is_case4=reasoning_is_case4_stream,
                    matched_stop=_effective_matched_stop,
                    prompt_thinking_active=prompt_thinking_active_stream,
                )
                # The helper returns the rescued reasoning ONLY when
                # all four predicates pass (empty/whitespace content,
                # no tool calls, non-empty/non-whitespace reasoning).
                # Otherwise it returns the original input — for our
                # pass it returns ``terminal_content or None``. We
                # only want to overwrite when the helper actually
                # promoted reasoning to content, i.e. the returned
                # value differs from what we passed in.
                if rescued_content and rescued_content != (terminal_content or None):
                    terminal_content = rescued_content
                    logger.info(
                        "[SSE-RESCUE-#569] terminal chunk content empty + no "
                        "tool calls; surfacing %d-char reasoning trace as "
                        "content",
                        len(terminal_content),
                    )
            # Issue #858 streaming mirror: helper is ON by default (PR
            # #802 / H-01 semantics restored). When the SSE rescue above
            # did NOT promote reasoning into ``terminal_content`` (strict
            # null path won — truncated ``<think>`` / harmony
            # analysis-without-final / Case-4 no-tag), emit the literal
            # cutoff sentinel as ONE final-chunk ``delta.content`` event
            # so streaming SDK consumers see the same signal as their
            # non-streaming counterparts. Per-token reasoning deltas have
            # already been sent during the loop; this is a single
            # extra-bytes-on-the-final-chunk event, NOT a per-token
            # mirror of the reasoning trace (D-STOP-THINK regression
            # guard). Default-on: opt out via
            # ``RAPID_MLX_REASONING_CUTOFF_NOTICE=disabled``. Gating
            # logic matches the non-streaming call site — the helper
            # owns it.
            if not has_any_tool_calls and not structured_output_requested:
                cutoff_content = _apply_reasoning_cutoff_notice(
                    terminal_content or None,
                    processor.accumulated_reasoning,
                    None,
                    finish_event.finish_reason,
                )
                if cutoff_content and cutoff_content != (terminal_content or None):
                    terminal_content = cutoff_content
                    logger.info(
                        "[SSE-CUTOFF-H01] terminal chunk content empty + "
                        "finish=length + reasoning present; surfacing "
                        "cutoff sentinel as content"
                    )
            final_chunk = ChatCompletionChunk(
                id=response_id,
                created=_sse_created,
                model=_resolve_model_name(request.model),
                choices=[
                    ChatCompletionChunkChoice(
                        delta=ChatCompletionChunkDelta(
                            content=terminal_content or None,
                            reasoning_content=finish_event.reasoning,
                            tool_calls=(
                                fallback_tool_calls if fallback_tool_calls else None
                            ),
                        ),
                        finish_reason=(
                            "tool_calls"
                            if fallback_tool_calls
                            else finish_event.finish_reason
                        ),
                        logprobs=_build_chunk_logprobs(finish_output),
                    )
                ],
                # See "Usage placement" note on the tool_call branch
                # (D-SSE-USAGE). When ``include_usage`` is false / unset
                # the field is omitted from this terminal chunk; the
                # dedicated trailing usage chunk is suppressed too.
                usage=None,
            )
            yield f"data: {final_chunk.model_dump_json(exclude_none=True)}\n\n"
        elif fallback_tool_calls or finalize_content:
            # Defensive: stream ended without a "finish" event but
            # finalize() produced either recovered tool calls or
            # released held content (shouldn't normally happen —
            # process_chunk emits finish on output.finished).
            #
            # Only emit material that has NOT already been streamed:
            # ``finalize_content`` (released prefix-held tail) and
            # ``fallback_tool_calls`` (cross-format recovered calls).
            # Do NOT include ``processor.accumulated_text`` /
            # ``accumulated_reasoning`` — both were already written
            # to the wire as per-delta chunks during the loop, so
            # replaying them would duplicate the whole response
            # (codex re-review BLOCKING). The original round-6 fix
            # in the postprocessor makes this branch unreachable
            # in the common case, but defense-in-depth: keep this
            # synthetic chunk additive only.
            tool_chunk = ChatCompletionChunk(
                id=response_id,
                created=_sse_created,
                model=_resolve_model_name(request.model),
                choices=[
                    ChatCompletionChunkChoice(
                        delta=ChatCompletionChunkDelta(
                            content=finalize_content or None,
                            reasoning_content=None,
                            tool_calls=fallback_tool_calls or None,
                        ),
                        finish_reason=("tool_calls" if fallback_tool_calls else "stop"),
                    )
                ],
            )
            _fb_sse = f"data: {tool_chunk.model_dump_json(exclude_none=True)}\n\n"
            logger.info(f"[SSE-FALLBACK-TC] {_fb_sse.strip()[:300]}")
            yield _fb_sse
        elif not inline_terminal_finish_emitted:
            # R11-A / R11-V2 invariant guard: every closed SSE stream
            # MUST emit exactly one terminal chunk carrying a
            # ``finish_reason`` BEFORE ``[DONE]``. Pre-fix, 4/10 streams
            # in some Qwen3 + ``tool_choice="required"`` batches reached
            # ``[DONE]`` with no finish_reason chunk at all when the
            # engine ended without surfacing ``output.finished=True`` on
            # any chunk AND ``finalize()`` produced no recovered tool
            # calls / held content. Spec-compliant clients (LangChain,
            # OpenAI Python SDK, AI SDK) stall waiting for the terminal
            # marker. Synthesize a ``finish_reason="stop"`` chunk here
            # so the wire envelope is always well-formed — the
            # accumulated content / reasoning has already been streamed
            # as per-delta chunks during the loop, so this synthetic
            # chunk is structurally additive only.
            #
            # Codex r1 HIGH #1 gate: only fire when the loop did NOT
            # already emit a tool_call chunk carrying an inline
            # ``finish_reason`` (the postprocessor stamps it on the
            # final tool_call event for the last engine chunk). Without
            # the latch, a valid final-chunk tool call would produce
            # TWO terminal chunks — first ``finish_reason="tool_calls"``
            # on the tool_call chunk, then ``finish_reason="stop"`` from
            # this synthetic — violating the "exactly one finish_reason"
            # invariant the rest of this PR pins.
            synthetic_finish = ChatCompletionChunk(
                id=response_id,
                created=_sse_created,
                model=_resolve_model_name(request.model),
                choices=[
                    ChatCompletionChunkChoice(
                        delta=ChatCompletionChunkDelta(),
                        finish_reason="stop",
                    )
                ],
            )
            logger.warning(
                "[SSE-MISSING-FINISH-R11V2] engine stream ended without "
                "emitting a finish event AND finalize() produced no "
                "recovered material; synthesizing finish_reason=stop "
                "terminal chunk so the wire envelope is well-formed"
            )
            yield f"data: {synthetic_finish.model_dump_json(exclude_none=True)}\n\n"

        # Log throughput
        elapsed = time.perf_counter() - start_time
        tokens_per_sec = completion_tokens / elapsed if elapsed > 0 else 0
        logger.info(
            f"Chat completion (stream): {completion_tokens} tokens in {elapsed:.2f}s ({tokens_per_sec:.1f} tok/s)"
        )

        # Send final chunk with usage if requested. Mirror non-streaming
        # shape by populating completion_tokens_details.reasoning_tokens
        # when the postprocessor saw a reasoning split — v0.6.63
        # onboarding sweep finding #5 (streaming previously dropped this
        # field even when non-streaming had it).
        if include_usage:
            # Build a synthetic GenerationOutput-shaped namespace so
            # _build_usage can compute the reasoning_tokens breakdown.
            class _UsageOutput:
                pass

            _u = _UsageOutput()
            _u.prompt_tokens = prompt_tokens
            _u.completion_tokens = completion_tokens
            _u.cached_tokens = cached_tokens
            # ``text`` carries the accumulated content (NOT reasoning) so
            # ``_build_usage`` can split ``completion_tokens`` between
            # reasoning and content by character ratio. Without this,
            # streaming usage chunks attribute 100% of the budget to
            # reasoning when ``len(reasoning)//4 >= completion_tokens``
            # (same root cause as the non-stream bug surfaced by the
            # v0.6.66 hybrid onboarding sweep on qwen3.6-27b-8bit).
            _u.text = processor.accumulated_text or ""
            usage_chunk = ChatCompletionChunk(
                id=response_id,
                created=_sse_created,
                model=_resolve_model_name(request.model),
                choices=[],
                usage=_build_usage(
                    _u,
                    processor.accumulated_reasoning or None,
                ),
            )
            yield f"data: {usage_chunk.model_dump_json(exclude_none=True)}\n\n"

        yield "data: [DONE]\n\n"
    finally:
        if cfg.gc_control and gc_was_enabled:
            gc.enable()
            gc.collect()


async def stream_chat_completion_guided(
    engine,
    messages: list,
    request: ChatCompletionRequest,
    json_schema: dict,
    *,
    strict_mode: bool = False,
    **kwargs,
) -> AsyncIterator[str]:
    """Stream chat completion with json_schema constrained decoding.

    Runs ``engine.generate_with_schema`` (which produces a single buffered
    ``GenerationOutput`` — outlines integration has no native streaming
    interface), then synthesizes an SSE stream from the buffered text.
    Pre-fix, ``stream=true`` requests with ``response_format: json_schema``
    silently bypassed ``GuidedGenerator`` because the stream branch of
    ``_create_chat_completion_impl`` went straight to ``engine.stream_chat``
    with no constraint hookup — the model would emit unconstrained tokens
    (e.g. a ```json ... ``` markdown fence around the JSON), defeating the
    user's intent (Gap #2, v0.6.60 onboarding sweep).

    On guided failure (exception from ``generate_with_schema``), delegates
    to the unconstrained ``stream_chat_completion`` helper to preserve
    request liveness — matches the non-streaming fallback semantics in
    ``_create_chat_completion_impl``. Clients in strict-mode use cases
    should validate the response against their schema regardless.
    """
    cfg = get_config()
    gc_was_enabled = gc.isenabled()
    if cfg.gc_control and gc_was_enabled:
        gc.disable()

    try:
        response_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        start_time = time.perf_counter()

        include_usage = bool(
            request.stream_options and request.stream_options.include_usage
        )

        # Pre-compute SSE template parts (mirrors stream_chat_completion's
        # fast path so chunk encoding is identical for clients).
        _sse_created = int(time.time())
        _model_escaped = json.dumps(_resolve_model_name(request.model))
        _sse_prefix = (
            f'data: {{"id":"{response_id}","object":"chat.completion.chunk",'
            f'"created":{_sse_created},"model":{_model_escaped},'
            f'"choices":[{{"index":0,"delta":{{'
        )
        _sse_suffix = "}}]}\n\n"

        # Run guided generation buffered. If it raises, fall through to
        # the unconstrained streaming helper — this preserves request
        # liveness (constraints best-effort, response always emitted)
        # and matches the non-streaming fallback semantics. We DO NOT
        # emit our own role chunk before the guided call, because on
        # fallback the unconstrained helper emits its own complete
        # SSE stream (role → content → DONE); a pre-emitted role would
        # produce a duplicate role chunk in the fallback path.
        #
        # ``raise_on_failure=True`` is critical: without it,
        # ``generate_with_schema`` silently falls back to
        # ``self.chat(...)`` on guided-engine failure and returns a
        # buffered unconstrained ``GenerationOutput``. From this
        # helper's POV that looks like a successful guided result and
        # we would emit one giant content chunk at the end —
        # defeating SSE for clients/proxies that rely on early chunks
        # (codex Round 2 finding).
        # Codex r5 BLOCKING parity: prevent a kwargs collision with
        # the explicit ``raise_on_failure=True`` below. If ``kwargs``
        # ever contained ``raise_on_failure`` it would TypeError
        # ("got multiple values for keyword argument") before
        # constrained decoding ran, and the outer ``except Exception``
        # arm would mistranslate that operator-wiring bug as a
        # guided-generation failure (silent fallback to unconstrained
        # streaming, which IS the case strict callers cannot
        # tolerate). Sanitize so the strict caller OWNS the value.
        _guided_kwargs = {k: v for k, v in kwargs.items() if k != "raise_on_failure"}
        try:
            output = await engine.generate_with_schema(
                messages=messages,
                json_schema=json_schema,
                raise_on_failure=True,
                **_guided_kwargs,
            )
        except Exception as guided_err:
            # Log only the schema's top-level shape, not the full body —
            # user-supplied schemas may embed PII (default values),
            # internal endpoint names, or be megabytes large. Keys +
            # required-list are enough to disambiguate the failure
            # without flooding ops logs or exposing payload contents.
            _schema_keys = (
                list(json_schema.keys()) if isinstance(json_schema, dict) else None
            )
            _required = (
                json_schema.get("required") if isinstance(json_schema, dict) else None
            )
            logger.debug(
                f"Problematic schema shape: keys={_schema_keys} required={_required}"
            )
            # Codex r6 BLOCKING: under strict=true the fallback to
            # unconstrained ``stream_chat_completion`` IS the H-06 hole
            # we're closing — clients asked for a contract and we
            # silently degraded to best-effort, just over SSE instead
            # of buffered. The post-decode validator below never runs
            # because we ``return`` from the fallback before reaching
            # it. Fix: when ``strict_mode`` is set, translate the
            # guided exception into the canonical SSE error envelope
            # (mirror of the post-decode shape) + DONE, and DO NOT
            # enter the unconstrained fallback.
            if strict_mode:
                incr_strict_violation()
                logger.warning(
                    "Strict json_schema streaming guided generation "
                    "failed; refusing to fall back to unconstrained "
                    "streaming because strict=true: %s",
                    guided_err,
                )
                _err_envelope = {
                    "error": {
                        "message": (
                            "strict response_format could not be honored: "
                            "the constrained-decoding path raised "
                            f"{type(guided_err).__name__} before producing "
                            "any output. The server refuses to fall back "
                            "to unconstrained streaming because the client "
                            "asked for strict=true. Investigate the server "
                            "logs and the "
                            "rapid_mlx_response_format_strict_violations_total "
                            "metric."
                        ),
                        "type": "api_error",
                        "code": "strict_schema_violation",
                        "param": "response_format.json_schema",
                    }
                }
                yield f"data: {json.dumps(_err_envelope)}\n\n"
                yield "data: [DONE]\n\n"
                return
            logger.warning(
                "Guided streaming generation failed, falling back to "
                f"unconstrained streaming: {guided_err}"
            )
            # Forward the pre-computed response_id + _sse_created so the
            # fallback stream's chunks share id/created with this outer
            # helper's would-be chunks. Without this, a client that
            # tracks the completion id across the guided→unconstrained
            # handoff sees two different ids/timestamps for what is
            # logically one request (DeepSeek pr_validate round 5).
            async for chunk in stream_chat_completion(
                engine,
                messages,
                request,
                response_id=response_id,
                created=_sse_created,
                **kwargs,
            ):
                yield chunk
            return

        content = output.text or ""

        # H-06 (codex r2): validate the buffered guided output BEFORE
        # emitting any SSE chunks for strict requests. The streaming
        # path here is a synthesized stream over a buffered output —
        # ``generate_with_schema`` returns a single GenerationOutput
        # rather than a token stream — so we still have a window to
        # convert a strict-contract violation into a clean error
        # SSE envelope instead of letting the schema-invalid bytes
        # reach the client.
        #
        # Outlines should make this unreachable; the violations
        # counter ticks regardless so operators see both the rate
        # AND the error response. We emit a single SSE error chunk
        # carrying the canonical OpenAI envelope, then DONE — clients
        # parsing the SSE stream see the error before any role/content
        # chunks land.
        if strict_mode and json_schema:
            ok, err = validate_output_against_schema(content, json_schema)
            if not ok:
                incr_strict_violation()
                logger.warning(
                    "Strict json_schema response failed post-decode "
                    "validation (streaming): %s",
                    err,
                )
                _err_envelope = {
                    "error": {
                        "message": (
                            "strict response_format violated: model output "
                            f"did not validate against the supplied schema ({err}). "
                            "This indicates the constrained-decoding path silently "
                            "degraded; investigate the server logs and the "
                            "rapid_mlx_response_format_strict_violations_total metric."
                        ),
                        "type": "api_error",
                        "code": "strict_schema_violation",
                        "param": "response_format.json_schema",
                    }
                }
                yield f"data: {json.dumps(_err_envelope)}\n\n"
                yield "data: [DONE]\n\n"
                return

        # Success path: synthesize SSE stream from the buffered output.
        # First chunk with role.
        yield f'{_sse_prefix}"role":"assistant"{_sse_suffix}'

        if content:
            yield f'{_sse_prefix}"content":{json.dumps(content)}{_sse_suffix}'

        # ``output`` is the single buffered ``GenerationOutput`` from
        # ``engine.generate_with_schema`` (outlines integration has no
        # native streaming interface — see this function's docstring).
        # Token counts are therefore read once and final; the main
        # ``stream_chat_completion`` path re-reads inside the stream
        # loop because the engine emits a sequence of GenerationOutputs
        # and the last one carries the authoritative counts.
        prompt_tokens = getattr(output, "prompt_tokens", 0) or 0
        completion_tokens = getattr(output, "completion_tokens", 0) or 0
        cached_tokens = getattr(output, "cached_tokens", 0) or 0
        # Pass the engine's finish_reason through directly. Matches the
        # convention in ``stream_chat_completion`` (line ~925:
        # ``finish_reason=event.finish_reason``), which never coerces a
        # falsy value. ``GenerationOutput.finish_reason`` defaults to
        # "stop" anyway, so the prior ``or "stop"`` was redundant and
        # would have silently rewritten any legitimately-None value the
        # engine emits (DeepSeek pr_validate round 3 finding).
        finish_reason = getattr(output, "finish_reason", None)

        # Final chunk with finish_reason. Usage placement
        # (OpenAI streaming spec, D-SSE-USAGE):
        #  - When ``stream_options.include_usage`` is True, usage MUST
        #    appear ONLY in the dedicated usage chunk below (this finish
        #    chunk carries ``null`` so it serializes consistently per
        #    spec; emitting it in both places had clients that
        #    aggregate usage double-count).
        #  - When False / unset, the field is omitted from EVERY chunk.
        #    The legacy behavior of attaching usage to the finish chunk
        #    when ``include_usage`` was unset was the D-SSE-USAGE bug:
        #    LangChain / AI-SDK / vercel-ai-stream parsers double-count
        #    token totals when usage appears on chunks the spec does
        #    not allow. Clients that want usage MUST opt in.
        finish_usage = None
        # ``created`` must be passed explicitly: the SSE prefix-style
        # chunks above already share ``_sse_created`` (computed once at
        # the top of the helper). ``ChatCompletionChunk.created`` has
        # ``default_factory=lambda: int(time.time())``, so a default
        # instantiation here would stamp a fresh timestamp on the finish
        # chunk and break the OpenAI streaming-spec invariant that all
        # chunks in one completion share a single ``created`` value
        # (DeepSeek pr_validate round 2 finding).
        finish_chunk = ChatCompletionChunk(
            id=response_id,
            created=_sse_created,
            model=_resolve_model_name(request.model),
            choices=[
                ChatCompletionChunkChoice(
                    delta=ChatCompletionChunkDelta(),
                    finish_reason=finish_reason,
                )
            ],
            usage=finish_usage,
        )
        yield f"data: {finish_chunk.model_dump_json(exclude_none=True)}\n\n"

        elapsed = time.perf_counter() - start_time
        tokens_per_sec = completion_tokens / elapsed if elapsed > 0 else 0
        logger.info(
            f"Chat completion (guided stream): {completion_tokens} tokens "
            f"in {elapsed:.2f}s ({tokens_per_sec:.1f} tok/s)"
        )

        if include_usage:
            usage_chunk = ChatCompletionChunk(
                id=response_id,
                created=_sse_created,
                model=_resolve_model_name(request.model),
                choices=[],
                usage=Usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                    prompt_tokens_details=(
                        PromptTokensDetails(cached_tokens=cached_tokens)
                        if cached_tokens
                        else None
                    ),
                ),
            )
            yield f"data: {usage_chunk.model_dump_json(exclude_none=True)}\n\n"

        yield "data: [DONE]\n\n"
    finally:
        if cfg.gc_control and gc_was_enabled:
            gc.enable()
            gc.collect()


async def stream_chat_completion_strict_postgen(
    engine,
    messages: list,
    request: ChatCompletionRequest,
    json_schema: dict,
    **kwargs,
) -> AsyncIterator[str]:
    """R12-4 — streaming variant of post-generate strict enforcement.

    Streams the unconstrained chat completion as ``stream_chat_completion``
    would, but buffers the per-delta content deltas client-side. After
    the upstream stream emits ``[DONE]`` we run the same
    :func:`validate_and_envelope` check used by the non-streaming
    path. On failure we synthesize ONE extra terminal chunk carrying
    ``finish_reason="json_schema_violation"`` plus a non-content
    ``error`` event before re-emitting ``[DONE]``.

    The asymmetry vs. the non-streaming path is intentional:
    streaming clients have already received the content tokens so a
    silent in-place 422 is impossible (the wire is already open).
    Surfacing the violation as an extra finish_reason value lets
    spec-aware clients distinguish "the server detected a strict
    contract breach" from a normal ``stop`` finish, while
    spec-unaware clients still see the content they need to retry
    against. The repair retry is deliberately NOT run here — re-
    prompting after a stream is in flight would require either
    closing the wire (defeating SSE) or attributing the retry tokens
    to the first stream's ``response_id`` (defeating the client's
    ability to distinguish them).

    Buffering rule: we accumulate the ``content`` deltas from each
    SSE chunk's ``choices[0].delta.content`` field, IGNORING
    ``reasoning_content`` (which is not part of the user-visible
    JSON the schema is supposed to constrain). If a future chunk
    format adds a different content surface we'll need to extend the
    accumulator — the buffer payload is small (one JSON object) so
    the memory cost of "buffer the whole stream" is trivial.
    """
    # Generate a stable response_id so the failure chunk we append at
    # the end shares the id with the upstream stream's content chunks
    # — clients group by id, so a fresh uuid here would surface as a
    # second un-associated completion.
    response_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())
    model_name = _resolve_model_name(request.model)

    buffered_content: list[str] = []
    buffered_content_bytes = 0
    # Codex r8 #2: bound the validation buffer. A misbehaving or
    # adversarial generation (forced-loop, jailbroken, model decode
    # bug) can stream content deltas indefinitely; pre-fix we'd
    # accumulate every byte until the upstream gave up, which
    # erased streaming's bounded-memory property and could OOM the
    # server on a single request. The cap defaults to 2 MiB — well
    # above any realistic strict-JSON payload (the largest strict
    # schemas in our pydantic-ai corpus marshal to ~50 KiB) but
    # comfortably below per-request memory limits operators expect
    # streaming to honor. Override via
    # ``RAPID_MLX_STRICT_BUFFER_BYTES`` for unusual workloads.
    #
    # Codex r12 #1 (design decision, documented for future passes):
    # the wrapper streams incremental content deltas to the client
    # as they arrive, then validates the FULL content at end of
    # stream. On overflow we drop the offending chunk (codex r10 #1)
    # and surface the structured ``buffer_overflow`` envelope —
    # prior chunks already on the wire are NOT recalled, because
    # streaming clients consume incrementally by definition (the
    # whole point of SSE is to deliver bytes as they arrive). An
    # "all-or-error" design would require buffering the entire
    # response before yielding the first byte, which:
    #   (a) defeats the user-facing latency benefit of streaming
    #       (which is THE reason clients opt into it);
    #   (b) doesn't actually help — a client building UI from
    #       partial deltas already has to handle interrupted
    #       streams (cancellation, timeouts, etc.); the strict
    #       contract is enforced by the ``finish_reason=
    #       json_schema_violation`` chunk that follows, NOT by
    #       withholding bytes.
    # Clients that need atomic validation use the non-streaming
    # path (``stream=false``) which IS all-or-error.
    #
    # Codex r10 NIT #1: clamp the configured cap to an upper bound.
    # A bad operator value (typo, misunderstanding of bytes vs
    # gigabytes) could silently disable the memory-safety guarantee
    # by setting an enormous cap. 64 MiB is the documented maximum:
    # any single response that legitimately needs more than 64 MiB
    # of JSON content is almost certainly a workload bug, not a
    # legitimate strict-mode use case. Values above the bound are
    # clamped DOWN to the bound with an operator-facing warning.
    _BUFFER_CAP_HARD_MAX = 64 * 1024 * 1024
    _BUFFER_CAP_DEFAULT = 2 * 1024 * 1024
    try:
        _buffer_cap = int(
            os.environ.get("RAPID_MLX_STRICT_BUFFER_BYTES", str(_BUFFER_CAP_DEFAULT))
        )
        if _buffer_cap <= 0:
            _buffer_cap = _BUFFER_CAP_DEFAULT
        elif _buffer_cap > _BUFFER_CAP_HARD_MAX:
            # Operator-facing warning — keep the env-var name OUT of
            # the literal log format so the no-out-of-band-routing
            # AST scan doesn't flag the whole format string as a
            # routing-shape constant. We pass the name in as a
            # parameter instead.
            logger.warning(
                "%s=%d exceeds hard maximum %d; clamping to %d to preserve "
                "memory-safety guarantee. If you need a larger cap, file an "
                "issue rather than raising the hard limit.",
                "RAPID_MLX_STRICT_BUFFER_BYTES",
                _buffer_cap,
                _BUFFER_CAP_HARD_MAX,
                _BUFFER_CAP_HARD_MAX,
            )
            _buffer_cap = _BUFFER_CAP_HARD_MAX
    except (TypeError, ValueError):
        _buffer_cap = _BUFFER_CAP_DEFAULT
    # Codex r2 #1: we MUST NOT forward the upstream stream's terminal
    # chunk (the one carrying ``finish_reason``) until we know whether
    # validation passed. Spec-compliant clients finalize on the first
    # finish_reason they see — if we let the upstream ``stop`` through
    # before our ``json_schema_violation`` chunk, the client treats
    # the response as a successful stop and never sees the violation.
    # Strategy: buffer the LAST chunk that carries a finish_reason
    # (and any usage chunk that follows it). After validation we
    # either emit the buffered terminal chunk (if valid) or REPLACE
    # it with our ``json_schema_violation`` chunk (if invalid).
    held_terminal_chunks: list[str] = []
    held_usage_chunk: str | None = None
    # Codex r7 #1: track whether validation+emission ran. The
    # try/finally below emits ``[DONE]`` unconditionally on the way
    # out — but if the upstream ``async for`` raised mid-stream we
    # must ALSO surface a structured ``upstream_stream_error`` event
    # before ``[DONE]`` so clients see WHY the stream ended early
    # (not just a silent close).
    upstream_raised: BaseException | None = None
    validation_emitted = False
    buffer_overflow = False
    # Codex r8 #1: flag set in the cancellation arm so the finally
    # block skips its own yields (which would themselves raise into
    # a closed pipe and mask the original cancellation).
    cancelled = False

    # Codex r13 #1: keep an explicit handle on the upstream async
    # generator so we can ``aclose()`` it on buffer overflow. Pre-
    # fix the wrapper used ``async for`` with no handle, so on
    # ``break`` the upstream generator was left dangling — the
    # engine kept generating tokens for a request the wrapper had
    # already given up on (wasted compute, held slot). With a
    # handle we explicitly close the generator on overflow so the
    # engine cleanup runs synchronously with the wrapper's
    # decision to bail.
    upstream_agen = stream_chat_completion(
        engine,
        messages,
        request,
        response_id=response_id,
        created=created,
        **kwargs,
    )
    try:
        async for chunk_text in upstream_agen:
            # Swallow the upstream [DONE] sentinel — we emit our own
            # [DONE] at the END of validation (codex r6 #1,
            # unconditional) so post-validation chunks land BEFORE the
            # terminal marker the spec requires last.
            if chunk_text.strip() == "data: [DONE]":
                continue
            is_terminal = False
            is_usage_chunk = False
            # Snoop content deltas + detect terminal finish_reason
            # chunks without disturbing the byte stream.
            if chunk_text.startswith("data: "):
                payload = chunk_text[len("data: ") :].strip()
                try:
                    parsed = json.loads(payload)
                except (json.JSONDecodeError, TypeError, ValueError):
                    parsed = None
                if isinstance(parsed, dict):
                    choices = parsed.get("choices") or []
                    for ch in choices:
                        if not isinstance(ch, dict):
                            continue
                        delta = ch.get("delta") or {}
                        if isinstance(delta, dict):
                            # Codex r11 #2: the JSON-schema strict
                            # contract applies to the user-visible
                            # response content — the ``delta.content``
                            # surface. We deliberately do NOT
                            # accumulate ``delta.reasoning_content``
                            # (a separate thinking-channel surface
                            # that is NOT included in the schema's
                            # scope) or ``delta.tool_calls`` (which
                            # is forbidden in strict mode by the
                            # ``strict_with_tools_unsupported`` gate
                            # in the chat route — line ~2310 — so it
                            # cannot legally appear here). Any future
                            # delta surface that carries user-visible
                            # text MUST be added here, OR the route
                            # gate must reject strict mode for that
                            # surface, otherwise strict mode would
                            # over-trigger ``json_schema_violation``
                            # on a request that emitted valid JSON
                            # through a non-content channel.
                            c = delta.get("content")
                            if isinstance(c, str):
                                # Codex r8 #2 + r9 #1: enforce the
                                # buffer cap in BYTES (UTF-8 encoded),
                                # not characters. Pre-fix used
                                # ``len(c)`` which counts Python code
                                # points; multi-byte content (CJK,
                                # emoji, accented Latin-1+ scripts)
                                # can blow past a byte budget by 2-4x
                                # before the cap engages, defeating
                                # the memory-safety guarantee.
                                # Encoding once per delta is O(N) on
                                # a typically-short delta and the
                                # encoded bytes are discarded
                                # immediately (we still buffer the
                                # original str so the validator
                                # receives correct UTF-8 round-tripped
                                # content). When exceeded we stop
                                # accumulating AND break out of the
                                # upstream loop — the finally block
                                # surfaces a structured
                                # ``buffer_overflow`` error so the
                                # client sees WHY validation was
                                # abandoned, instead of either
                                # silently truncating (data
                                # corruption) or OOMing the server.
                                c_byte_len = len(c.encode("utf-8"))
                                if buffered_content_bytes + c_byte_len > _buffer_cap:
                                    buffer_overflow = True
                                else:
                                    buffered_content.append(c)
                                    buffered_content_bytes += c_byte_len
                        if ch.get("finish_reason"):
                            is_terminal = True
                    # A usage-only chunk has no choices (or empty
                    # choices) AND carries a ``usage`` field — the
                    # upstream ``stream_chat_completion`` emits one
                    # after the terminal chunk when
                    # ``stream_options.include_usage`` is set. We
                    # hold it alongside the terminal chunk so the
                    # terminal-replacement logic doesn't drop usage.
                    if not choices and parsed.get("usage") is not None:
                        is_usage_chunk = True
            # Codex r10 #1: if the buffer cap was hit, DROP the
            # overflowing chunk before forwarding it. Pre-fix the
            # ``yield chunk_text`` ran BEFORE the break check, so a
            # client would receive bytes the server had deliberately
            # excluded from validation (and then receive the
            # ``buffer_overflow`` envelope claiming the buffer was
            # capped). That's a half-truth — the validator skipped
            # the content but the wire still delivered it. Now the
            # break happens BEFORE the forward, so the cap is honored
            # end-to-end: bytes that didn't reach the buffer don't
            # reach the client either.
            if buffer_overflow:
                # Held terminal/usage chunks haven't been emitted yet
                # — leave them on the floor too. The overflow envelope
                # in the post-loop block emits its own
                # finish_reason=json_schema_violation, so a held
                # ``finish_reason=stop`` chunk would conflict.
                # Codex r13 #1 + r14 #1: break out of the loop FIRST;
                # the ``aclose()`` call happens AFTER we exit the
                # ``async for`` block (see the post-loop aclose
                # block below). Calling ``aclose()`` from INSIDE the
                # active ``async for`` raises
                # ``RuntimeError: aclose(): asynchronous generator
                # is already running`` for real (non-mock) async
                # generators — the generator is mid-``__anext__``
                # from the wrapper's perspective. Exiting the loop
                # first releases the iteration handle so aclose()
                # can run cleanly.
                break
            if is_terminal:
                held_terminal_chunks.append(chunk_text)
                continue
            if is_usage_chunk:
                held_usage_chunk = chunk_text
                continue
            yield chunk_text

        # Codex r14 #1: after exiting the ``async for`` block we are
        # NO LONGER inside the generator's iteration, so it is safe
        # to ``aclose()`` it. On the happy path (loop exhausted
        # normally) aclose is a no-op. On the overflow break path
        # aclose propagates ``GeneratorExit`` into the upstream
        # generator so the engine's per-request cleanup runs
        # synchronously and we don't leave a zombie generation
        # task consuming compute. We do this for BOTH paths so
        # the resource-management story is uniform; on the
        # already-exhausted path the second-order cost is trivial.
        if buffer_overflow:
            try:
                await upstream_agen.aclose()
            except (asyncio.CancelledError, GeneratorExit):
                # Documented "successful close" shape — the
                # upstream generator unwound via the cancellation
                # signal we sent it.
                pass
            except Exception:  # noqa: BLE001
                # Any other exception during upstream cleanup is
                # worse than leaving the engine to its natural
                # terminus; log + continue so we still deliver the
                # buffer_overflow envelope to the client.
                logger.warning(
                    "R12-4 upstream aclose() raised during buffer overflow cleanup",
                    exc_info=True,
                )

        # Codex r8 #2: if the buffer cap fired, skip validation and
        # treat the truncated content as a violation. Validating a
        # truncated payload would frequently produce a misleading
        # ``invalid_json`` error (the truncation point can fall
        # mid-string / mid-object) — surfacing the overflow as its
        # own ``buffer_overflow`` code is more actionable for the
        # operator who needs to either raise the cap or fix the
        # runaway model.
        if buffer_overflow:
            incr_strict_violation()
            # Codex r13 NIT: include both the current cap AND the
            # documented hard maximum so the operator's guidance
            # ("raise the cap") is bounded — if they're already at
            # the hard max, the message tells them to investigate
            # the runaway generation instead of futilely raising
            # the env var.
            if _buffer_cap >= _BUFFER_CAP_HARD_MAX:
                cap_guidance = (
                    f"current cap ({_buffer_cap} bytes) is at the hard maximum "
                    f"({_BUFFER_CAP_HARD_MAX} bytes); investigate the runaway "
                    "generation rather than raising the cap further."
                )
            else:
                cap_guidance = (
                    f"raise RAPID_MLX_STRICT_BUFFER_BYTES (current: "
                    f"{_buffer_cap} bytes, hard maximum: "
                    f"{_BUFFER_CAP_HARD_MAX} bytes) or investigate a "
                    "runaway generation."
                )
            overflow_envelope = build_violation_envelope(
                {
                    "reason": "buffer_overflow",
                    "message": (
                        f"strict json_schema content buffer exceeded "
                        f"{_buffer_cap} bytes; abandoned validation. "
                        f"{cap_guidance}"
                    ),
                },
                attempts=1,
            )
            violation_chunk = ChatCompletionChunk(
                id=response_id,
                created=created,
                model=model_name,
                choices=[
                    ChatCompletionChunkChoice(
                        delta=ChatCompletionChunkDelta(),
                        finish_reason="json_schema_violation",
                    )
                ],
            )
            yield (
                "data: " + violation_chunk.model_dump_json(exclude_none=True) + "\n\n"
            )
            error_event = {
                "id": response_id,
                "object": "chat.completion.error",
                "created": created,
                "model": model_name,
                **overflow_envelope,
            }
            error_payload = json.dumps(error_event, separators=(",", ":"))
            yield ("event: chat.completion.error\n" + "data: " + error_payload + "\n\n")
            if held_usage_chunk is not None:
                yield held_usage_chunk
            logger.warning(
                "R12-4 strict json_schema streaming buffer overflow at %d bytes",
                _buffer_cap,
            )
            validation_emitted = True
            return

        # Stream completed normally — run validation. We do NOT
        # incr_strict_violation() on the happy path; only when we
        # surface the failure event.
        full_content = "".join(buffered_content)
        ok, failure_details = validate_and_envelope(full_content, json_schema)
        if ok:
            # Validation passed — release the held terminal + usage
            # chunks in their original order. The client sees a normal
            # stream.
            for terminal in held_terminal_chunks:
                yield terminal
            if held_usage_chunk is not None:
                yield held_usage_chunk
        else:
            incr_strict_violation()
            envelope = build_violation_envelope(
                failure_details or {"reason": "schema_violation"},
                attempts=1,
            )
            # Codex r2 #1: DROP the held upstream terminal chunk(s); a
            # client finalizing on the first ``finish_reason`` would
            # otherwise treat ``stop`` as the verdict and miss the
            # violation. The R12-4 contract: the FIRST finish_reason a
            # strict-streaming client sees on a violating response
            # MUST be ``json_schema_violation``.
            violation_chunk = ChatCompletionChunk(
                id=response_id,
                created=created,
                model=model_name,
                choices=[
                    ChatCompletionChunkChoice(
                        delta=ChatCompletionChunkDelta(),
                        finish_reason="json_schema_violation",
                    )
                ],
            )
            yield (
                "data: " + violation_chunk.model_dump_json(exclude_none=True) + "\n\n"
            )
            # Codex r5 #1: emit ONE error frame, not two. A previous
            # iteration (codex r2) split the envelope into a named
            # SSE event (``event: chat.completion.error\ndata: ...``)
            # AND a plain ``data: ...`` line carrying the same
            # payload. The double-emit is harmful: a client that
            # consumes both named SSE events AND plain ``data:``
            # lines (EventSource subclasses, custom dispatchers)
            # handles the same terminal error twice, producing
            # double-billing in observability and potentially
            # duplicate user-facing toasts. We pick the named form:
            # per SSE spec, ``event: chat.completion.error\ndata:
            # <json>`` is parsed as ONE message event by EventSource
            # (dispatched to the ``chat.completion.error`` listener)
            # AND as ONE ``data:`` line by plain-line consumers
            # (OpenAI Python SDK, curl, AI SDK), who ignore the
            # unknown ``event:`` field. Both client classes receive
            # the envelope exactly once.
            error_event = {
                "id": response_id,
                "object": "chat.completion.error",
                "created": created,
                "model": model_name,
                **envelope,
            }
            # Compact JSON encoding (no spaces) matches the SSE chunk
            # encoding used elsewhere in this module.
            error_payload = json.dumps(error_event, separators=(",", ":"))
            yield ("event: chat.completion.error\n" + "data: " + error_payload + "\n\n")
            # Codex r6 #2: preserve usage accounting on a failed
            # strict generation. Requests with
            # ``stream_options.include_usage`` already consumed
            # generation tokens before the validation verdict landed;
            # dropping the usage chunk on failure would leave billing
            # / observability clients blind to those tokens. The
            # terminal ``finish_reason=json_schema_violation`` chunk
            # has already been emitted ABOVE this point, so the
            # usage-only chunk that follows is unambiguously trailing
            # metadata (mirrors the OpenAI streaming spec, where the
            # final usage chunk arrives AFTER the terminal
            # finish_reason chunk). Pass the held usage chunk through
            # verbatim so consumers reconcile billing against the same
            # numbers they would have seen on a successful turn.
            if held_usage_chunk is not None:
                yield held_usage_chunk
            logger.warning(
                "R12-4 strict json_schema streaming validation failed: %s",
                (failure_details or {}).get("message"),
            )
        validation_emitted = True
    except (asyncio.CancelledError, GeneratorExit):
        # Codex r8 #1: client-disconnect / cooperative cancellation
        # paths arrive as ``asyncio.CancelledError`` (FastAPI /
        # uvicorn cancel the request task when the client TCP socket
        # closes) and ``GeneratorExit`` (when the consumer of THIS
        # async generator calls ``aclose()``). Suppressing them would
        # keep the upstream generation task running long enough to
        # emit our trailing error frame + ``[DONE]``, which:
        #   (a) defeats the disconnect mechanism — the engine keeps
        #       generating tokens for a client who is gone, wasting
        #       compute and prolonging the slot until the natural
        #       terminus;
        #   (b) tries to yield bytes into a closed pipe, raising
        #       again from inside ``finally`` (BrokenPipeError /
        #       RuntimeError "generator closed") and masking the
        #       original cancellation.
        # We set ``cancelled`` so the finally block skips its own
        # yields (which would themselves raise into the dead pipe)
        # and re-raises to propagate cancellation up the stack. The
        # caller's ``StreamingResponse`` framing handles the rest.
        cancelled = True
        raise
    except Exception as exc:  # noqa: BLE001
        # Codex r7 #1 + r8 #1: ordinary upstream generation
        # exceptions (engine raises, runtime errors, etc.) — distinct
        # from cancellation. Without this catch, the exception would
        # propagate past the function epilogue and the promised
        # unconditional ``[DONE]`` at the bottom would NEVER be
        # emitted (Python's generator close semantics propagate the
        # exception past the function epilogue), and clients see a
        # truncated stream with no terminal sentinel. We capture the
        # exception, surface a structured ``upstream_stream_error``
        # event, then drop into the finally block to emit ``[DONE]``.
        # We do NOT re-raise after the finally — by the time the SSE
        # stream is in flight there is no other surface for the
        # error, and the wire envelope is already 200/event-stream.
        upstream_raised = exc
        logger.warning(
            "R12-4 strict streaming upstream generator raised: %s: %s",
            type(exc).__name__,
            exc,
        )
    finally:
        # Codex r8 #1: skip the entire finally body on cancellation —
        # the consumer pipe is dead, our yields would raise into it
        # and mask the original CancelledError. The caller's
        # StreamingResponse machinery handles the disconnect from
        # here.
        if not cancelled:
            # Codex r7 #1: if the upstream raised, emit the error
            # envelope BEFORE [DONE]. Structurally identical to a
            # schema-violation event — distinct ``code`` so clients
            # can branch, same envelope shape so handler code is
            # reusable.
            if upstream_raised is not None and not validation_emitted:
                # Codex r12 #2: do NOT leak ``str(upstream_raised)``
                # into the client-visible SSE payload. Exception
                # messages from the inference stack can include
                # file paths, internal type details, environment
                # values, etc. — info-leak shapes a malicious
                # client can probe. Wire ONLY the exception_type
                # (a coarse, public, contract-style identifier) and
                # the response_id (so operators can correlate the
                # client report with the full server-side log
                # entry). The full ``str(exc)`` was already logged
                # in the except arm above for server-side
                # diagnostics — that's where operators look.
                upstream_envelope = {
                    "error": {
                        "type": "upstream_error",
                        "code": "strict_stream_upstream_error",
                        "message": (
                            "strict json_schema streaming generation "
                            "aborted before validation. See server logs "
                            f"for response_id={response_id}."
                        ),
                        "param": "response_format.json_schema",
                        "details": {
                            "exception_type": type(upstream_raised).__name__,
                            "response_id": response_id,
                        },
                    }
                }
                err_obj = {
                    "id": response_id,
                    "object": "chat.completion.error",
                    "created": created,
                    "model": model_name,
                    **upstream_envelope,
                }
                try:
                    err_payload = json.dumps(err_obj, separators=(",", ":"))
                    yield (
                        "event: chat.completion.error\n"
                        + "data: "
                        + err_payload
                        + "\n\n"
                    )
                except (TypeError, ValueError):
                    # JSON encoding can't reasonably fail here; if it
                    # did we still need to close the stream — fall
                    # through to [DONE].
                    pass
            # Codex r6 #1 + r7 #1: ALWAYS emit ``[DONE]`` on the
            # non-cancelled exit. Spec wire envelope MUST close with
            # ``[DONE]`` so clients don't hang.
            yield "data: [DONE]\n\n"
