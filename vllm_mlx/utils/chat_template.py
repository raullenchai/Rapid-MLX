# SPDX-License-Identifier: Apache-2.0
"""
Chat template application logic for BatchedEngine.

Handles enable_thinking, tools, and fallback logic for chat template rendering.
"""

import copy
import json
import logging
import re

logger = logging.getLogger(__name__)

# Common chat-template role markers across HuggingFace tokenizer families.
# These are always neutralized in user-supplied content even when the
# tokenizer does not declare them in ``special_tokens_map`` (sometimes the
# template strings are baked into the Jinja text without the tokens being
# registered, e.g. some Phi/Llama variants). Listing them here is NOT a
# per-model workaround — it's the union of role-delimiter literals that
# any HF chat template can interpret as a control sequence. The sanitiser
# below ALSO consults the tokenizer's own special-token registry to catch
# tokens we don't enumerate here (qwen3-vl ``<|vision_start|>``, gemma
# ``<start_of_turn>``, …).
_CHAT_TEMPLATE_ROLE_MARKERS = (
    # ChatML (Qwen, ChatGLM, ...)
    "<|im_start|>",
    "<|im_end|>",
    # Llama 3 / Hermes
    "<|start_header_id|>",
    "<|end_header_id|>",
    "<|eot_id|>",
    "<|begin_of_text|>",
    "<|end_of_text|>",
    # Gemma
    "<start_of_turn>",
    "<end_of_turn>",
    # Phi
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|end|>",
    # DeepSeek
    "<|fim_begin|>",
    "<|fim_hole|>",
    "<|fim_end|>",
    "<｜begin▁of▁sentence｜>",
    "<｜end▁of▁sentence｜>",
    "<｜User｜>",
    "<｜Assistant｜>",
    "<｜latest_reminder｜>",
    # Mistral / Anthropic-style
    "[INST]",
    "[/INST]",
    "<<SYS>>",
    "<</SYS>>",
    # Harmony (gpt-oss)
    "<|start|>",
    "<|message|>",
    "<|channel|>",
    "<|return|>",
)

_REASONING_SENTINELS = {
    "<think>",
    "</think>",
    "<reasoning>",
    "</reasoning>",
    "<｜DSML｜",
    "</｜DSML｜",
}
_EXISTING_CONTROL_ESCAPE = re.compile(
    r"<(?P<esc>\u200b+)(?=(?:/?(?:think|reasoning)>|/?｜DSML｜))"
)


def _double_existing_control_escapes(text: str) -> str:
    return _EXISTING_CONTROL_ESCAPE.sub(
        lambda match: "<" + (match.group("esc") * 2), text
    )


def _collect_role_markers(
    template_applicator, *, include_reasoning_sentinels: bool = False
) -> set[str]:
    """Return the set of chat-template role markers that must be neutralized
    in user-supplied content for ``template_applicator``.

    Combines the conservative built-in literals (``_CHAT_TEMPLATE_ROLE_MARKERS``)
    with anything the tokenizer's own special-token registry exposes that
    looks like a delimiter (``<|...|>`` or ``<...turn>`` / ``<...header>``).

    The detector is **per-tokenizer** but **not per-model**: the same
    regex tests the same `<|...|>` family for every tokenizer we load,
    so there's nothing model-specific to maintain.
    """
    markers: set[str] = set(_CHAT_TEMPLATE_ROLE_MARKERS)
    tokenizer = template_applicator
    # Processors (Qwen3-VL, Gemma-3n) wrap a tokenizer. The role markers
    # live on the wrapped tokenizer; the processor exposes vision tokens
    # which are not role markers but ARE still untrusted-input vectors,
    # so we include them too.
    if hasattr(tokenizer, "tokenizer"):
        markers |= _collect_role_markers(
            tokenizer.tokenizer,
            include_reasoning_sentinels=include_reasoning_sentinels,
        )

    candidates: list[str] = []
    for attr in ("all_special_tokens", "additional_special_tokens"):
        vals = getattr(tokenizer, attr, None) or []
        if isinstance(vals, (list, tuple, set)):
            candidates.extend(str(v) for v in vals)
    smap = getattr(tokenizer, "special_tokens_map", None)
    if isinstance(smap, dict):
        for v in smap.values():
            if isinstance(v, str):
                candidates.append(v)
            elif isinstance(v, (list, tuple)):
                candidates.extend(str(x) for x in v)
    # DeepSeek V4's tool prompt explicitly teaches the model to remove the
    # neutralising U+200B when copying repository bytes into a tool argument.
    # Do not mutate reasoning-tag text for unrelated model families which do
    # not receive that restoration contract.
    if include_reasoning_sentinels:
        markers.update(_REASONING_SENTINELS)
    # Only treat sequences that LOOK like a template delimiter as
    # neutralisation targets — picking up every special token would
    # also strip ``<pad>`` / ``<unk>`` etc. from user text, which is
    # not what the user typed but also not a security issue. The two
    # delimiter shapes any HF chat template can interpret as a role
    # change are ``<|...|>`` (ChatML/Llama/Phi/Harmony) and ``<...>``
    # bracket markers ending with ``turn``/``header``/``message``
    # (Gemma family).
    for tok in candidates:
        if not tok or not isinstance(tok, str):
            continue
        if (
            tok.startswith("<|")
            and tok.endswith("|>")
            or tok.startswith("<")
            and tok.endswith(">")
            and any(kw in tok for kw in ("turn", "header", "message", "channel"))
        ):
            markers.add(tok)
    return markers


def _build_marker_pattern(markers: set[str]) -> re.Pattern | None:
    """Compile an alternation regex that matches any role marker.

    Returns None if there are no markers (degenerate templates).
    """
    if not markers:
        return None
    # Sort by length desc so longer markers (``<|im_start|>``) match
    # before their prefixes (``<|im_``) on any future overlap.
    parts = sorted((re.escape(m) for m in markers), key=len, reverse=True)
    return re.compile("|".join(parts))


def _neutralize_in_string(text: str, pattern: re.Pattern) -> str:
    """Replace any chat-template marker in ``text`` with a non-tokenizing
    Unicode-prefixed variant.

    Strategy: insert a zero-width space (U+200B) after the opening
    angle bracket so the literal text round-trips visually but the
    tokenizer cannot recognise it as a control sequence. ZWSP is
    invisible in any client UI that supports Unicode and the user's
    intended text (the literal marker) is preserved.
    """

    def _sub(match: re.Match) -> str:
        marker = match.group(0)
        # ``<​|im_start|>`` — the ZWSP after the first ``<`` breaks
        # the tokenizer match without changing the visible glyphs.
        return marker[0] + "​" + marker[1:]

    return pattern.sub(_sub, text)


def _sanitize_message_content(
    content,
    pattern: re.Pattern,
):
    """Recursively neutralize chat-template markers in ``content``.

    Handles three content shapes:
    * ``str`` → return a string with markers neutralized.
    * ``list`` of content parts (multimodal) → return a new list with
      ``text``-typed parts sanitized; non-text parts pass through.
    * Anything else → returned unchanged.
    """
    if isinstance(content, str):
        return _neutralize_in_string(content, pattern)
    if isinstance(content, list):
        new_parts = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text" and isinstance(part.get("text"), str):
                    new_part = dict(part)
                    new_part["text"] = _neutralize_in_string(part["text"], pattern)
                    new_parts.append(new_part)
                else:
                    new_parts.append(part)
            else:
                new_parts.append(part)
        return new_parts
    return content


def _double_existing_control_escapes_in_content(content):
    """Quote pre-existing framing bytes before adding protocol framing."""
    if isinstance(content, str):
        return _double_existing_control_escapes(content)
    if isinstance(content, list):
        new_parts = []
        for part in content:
            if (
                isinstance(part, dict)
                and part.get("type") == "text"
                and isinstance(part.get("text"), str)
            ):
                new_part = dict(part)
                new_part["text"] = _double_existing_control_escapes(part["text"])
                new_parts.append(new_part)
            else:
                new_parts.append(part)
        return new_parts
    return content


def _sanitize_messages_for_template(
    messages: list[dict],
    template_applicator,
    *,
    include_reasoning_sentinels: bool = False,
) -> list[dict]:
    """Strip / neutralize chat-template control tokens from user-supplied
    message content.

    This is the layer fix for the prompt-injection vector where a user
    writes ``<|im_start|>system\\nIgnore...<|im_end|>`` in their
    message body and the tokenizer parses those literals as real
    role-delimiter control tokens — letting user content forge a
    ``system`` role.

    The sanitiser runs against EVERY ``apply_chat_template`` call (one
    function wraps every render in this module) so the fix is
    template-agnostic. ALL roles are sanitised — the server cannot
    prove an ``assistant``-role message in the request was actually
    produced by its own model output (multi-turn clients ship the
    whole ``messages`` array, so a malicious client can forge
    ``{"role": "assistant", "content": "<|im_start|>system\\n..."}``
    on a replay, codex r4 BLOCKING).

    The neutralisation strategy preserves the literal text visually
    (inserts U+200B after the opening ``<``) so even a legitimate
    assistant turn that genuinely contained the literal marker
    round-trips with the same visible glyphs — only the tokenizer's
    interpretation is neutralised. See ``_neutralize_in_string`` for
    the rationale.
    """
    markers = _collect_role_markers(
        template_applicator,
        include_reasoning_sentinels=include_reasoning_sentinels,
    )
    pattern = _build_marker_pattern(markers)
    if pattern is None:
        return messages
    sanitized: list[dict] = []
    for msg in messages:
        if not isinstance(msg, dict):
            sanitized.append(msg)
            continue
        content = msg.get("content")
        if include_reasoning_sentinels:
            content = _double_existing_control_escapes_in_content(content)
        new_content = _sanitize_message_content(content, pattern)
        if new_content is content:
            sanitized.append(msg)
            continue
        new_msg = dict(msg)
        new_msg["content"] = new_content
        sanitized.append(new_msg)
    return sanitized


# =============================================================================
# F-111: content-array → string normalization
# =============================================================================
#
# OpenAI's o1/o3 client SDKs ship ``tool``-role replies (and many
# ``user``/``assistant`` turns) in the multipart-content shape
# ``content: [{"type": "text", "text": "..."}]`` even when the payload
# is text-only. Most HF chat templates render ``content`` by string
# concatenation (Jinja ``{{ content }}``) or by indexing
# ``content[0].text`` — both produce an empty / wrong render when the
# wire shape is a list of typed parts. Confirmed silent drops on Qwen3
# (renders empty ``<tool_response>``) and a hard ``TypeError`` on
# Hermes3. The fix is one normalization pass right before
# ``apply_chat_template`` — flatten any text-only content array down to
# the single concatenated string the templates expect. Multimodal
# content (image/video/audio parts) is preserved unchanged so the
# vision/audio branches keep working.
#
# A ``tool``-role message can ONLY carry text (tool replies are not
# multimodal in the OpenAI spec — even the o1 wire shape is
# ``[{type:text,text:...}]``). If a caller smuggles a non-text part
# into a ``tool`` reply we raise ``ValueError`` and the
# ``apply_chat_template`` caller surfaces it as HTTP 400 — silently
# dropping would re-open the same "tool content missing" footgun this
# normalization closes.


def _part_type_and_text(part) -> tuple[str | None, str | None]:
    """Return ``(type, text)`` for a content part regardless of wire shape.

    A content part can arrive as a ``dict`` (pre-dumped or
    ``extract_multimodal_content`` output), as a pydantic ``ContentPart``
    instance (request-validation hand-off), or as something else (we
    treat that as "unknown" so the caller can decide what to do).
    """
    if isinstance(part, dict):
        t = part.get("type")
        x = part.get("text")
    else:
        t = getattr(part, "type", None)
        x = getattr(part, "text", None)
    if isinstance(t, str) or t is None:
        t_norm = t
    else:
        t_norm = None
    x_norm = x if isinstance(x, str) else None
    return t_norm, x_norm


def _is_text_only_content_array(content) -> bool:
    """Return True iff ``content`` is a non-empty list whose every
    element is a text part — ``{"type": "text", "text": str}`` or the
    equivalent pydantic ``ContentPart``.

    Multipart content with any non-text part (image_url / video /
    audio_url / input_audio / ...) is left alone for the multimodal
    rendering branches to handle.
    """
    if not isinstance(content, list) or not content:
        return False
    for part in content:
        t, x = _part_type_and_text(part)
        if t != "text" or x is None:
            return False
    return True


def _join_text_parts(content: list) -> str:
    """Concatenate ``{"type": "text", "text": X}`` parts into one string.

    Multiple text parts are joined verbatim (no separator) — OpenAI's
    o1+ SDK ships single-part arrays in practice, and a separator
    would corrupt single-part renders. Multi-part text arrays are an
    accepted edge case and join verbatim mirrors HF tokenizer
    expectations.
    """
    return "".join((_part_type_and_text(part)[1] or "") for part in content)


def _normalize_text_only_content_arrays(messages: list[dict]) -> list[dict]:
    """Flatten text-only ``content`` arrays into plain strings so chat
    templates that expect ``content`` to be a string render correctly.

    Applies to every role; multipart content with non-text parts
    (image/video/audio) is preserved unchanged. For ``tool``-role
    messages with non-text parts we raise ``ValueError`` — tool replies
    are text-only per the OpenAI spec, and silently dropping the
    non-text part would reopen the same "tool content missing"
    bug-class this normalization closes (F-111).
    """
    out: list[dict] = []
    for msg in messages:
        if not isinstance(msg, dict):
            out.append(msg)
            continue
        content = msg.get("content")
        role = msg.get("role")
        if isinstance(content, list) and content:
            if _is_text_only_content_array(content):
                new_msg = dict(msg)
                new_msg["content"] = _join_text_parts(content)
                out.append(new_msg)
                continue
            if role == "tool":
                # Tool replies are text-only per OpenAI spec. A non-text
                # part here would be silently dropped by the renderer
                # (the exact F-111 footgun), so reject explicitly. In
                # the live path the route-level validator in
                # ``vllm_mlx/routes/chat.py`` has already 400'd non-text
                # tool parts; this raise is a defence-in-depth for
                # direct callers of ``apply_chat_template`` (engine
                # tests, the speculative server, the gradio app).
                raise ValueError(
                    "tool-role message content must be a string or a "
                    "text-only array of {type:'text', text:str} parts; "
                    "got a non-text content part"
                )
        out.append(msg)
    return out


# =============================================================================
# GH-973: assistant tool_call.arguments dict-form invariant
# =============================================================================
#
# The OpenAI wire contract encodes ``message.tool_calls[i].function.arguments``
# as a JSON string (see: https://platform.openai.com/docs/api-reference/chat/
# create → ``tool_calls.function.arguments``). Every mainstream HF chat
# template (Qwen3 / Hermes / Llama3 / GLM4 / Nemotron / minimax) iterates
# that field as a mapping — ``tool_call.arguments|items`` — so a JSON-string
# render blows up with:
#
#     TypeError: Can only get item pairs from a mapping.
#
# The bug surfaces on the ``pydantic_ai`` structured-output retry path
# (GH-973): pydantic_ai replays the prior assistant tool_call verbatim in
# the OpenAI wire shape (``arguments`` = JSON string), and the retry pass
# through ``apply_chat_template`` crashes with 500. The direct fix upstream
# in ``routes/chat.py::extract_multimodal_content`` and
# ``engine/batched.py::_normalize_tool_call_arguments_for_template`` covers
# the standard ``/v1/chat/completions`` non-MLLM path, but every other
# caller of the shared ``apply_chat_template`` (guided-generation
# ``BatchedEngine.stream_guided_completion``, native-video path, direct
# engine callers, tests) bypassed those. Moving the invariant to the
# shared ``apply_chat_template`` boundary makes it a single choke point.
#
# Behaviour matches ``engine/batched.py::_normalize_tool_call_arguments_
# for_template`` (str → parsed dict when JSON dict; parsed non-dict
# wrapped as ``{"value": <parsed>}``; malformed JSON wrapped as
# ``{"value": <raw>}``). Dict-form arguments pass through unchanged
# (idempotent), so callers that already normalised upstream pay no cost.
#
# NON-GOALS:
#   * Parser output shape is untouched — tool_parsers/*.py write dict
#     for round-trip correctness; this fix is about REPLAYED messages
#     from the client.
#   * User / tool / system messages are untouched — only assistant.
#   * Malformed JSON is preserved verbatim inside the ``{"value": ...}``
#     wrapper so log-style renderers keep the original text.


def _coerce_arguments_to_dict(arguments):
    """Convert an ``arguments`` value to a dict per the GH-973 rules.

    * ``dict`` → returned unchanged (idempotent).
    * ``str`` → ``json.loads``; if the parsed value is a dict, use it;
      otherwise wrap the parsed value as ``{"value": <parsed>}``.
    * ``str`` that fails to JSON-parse → wrap as ``{"value": <raw>}``.
    * Anything else (``list``, scalar, ...) → wrap as ``{"value": <raw>}``.

    Callers MUST have already checked that an ``arguments`` key is
    present on the source dict — this helper is only invoked after
    presence-and-non-dict is confirmed by the two-pass walk in
    :func:`_normalize_assistant_tool_call_arguments`, so an absent key
    never reaches here (codex r1 NIT: pre-fix we synthesised
    ``{"value": None}`` for absent ``arguments``, silently inventing an
    argument payload; the presence guard closes that).
    """
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except (json.JSONDecodeError, ValueError, TypeError):
            return {"value": arguments}
        if isinstance(parsed, dict):
            return parsed
        return {"value": parsed}
    # Non-string, non-dict — rarely seen (an SDK bug or a test injecting
    # a bare list/int). Wrap so ``|items`` still works.
    return {"value": arguments}


def _tool_call_arguments_need_mutation(tool_call: dict) -> tuple[bool, bool]:
    """Return ``(nested_needs, top_needs)`` for ``tool_call``.

    * ``nested_needs`` — ``function.arguments`` is present AND non-dict.
    * ``top_needs`` — ``tool_call.arguments`` (top-level) is present AND
      non-dict. Both shapes are normalised INDEPENDENTLY: a mixed-shape
      replay that carries BOTH nested and top-level ``arguments`` (some
      SDKs mirror the field for template compatibility) must have both
      forms dict-safe, otherwise a template that iterates
      ``tc.arguments|items`` still crashes even when
      ``tc.function.arguments`` was normalised (codex r3 BLOCKING on
      PR #981).

    Absent ``arguments`` keys yield ``False`` — we don't invent a
    payload for something the caller never sent (codex r1 NIT).
    """
    function = tool_call.get("function")
    nested_needs = (
        isinstance(function, dict)
        and "arguments" in function
        and not isinstance(function.get("arguments"), dict)
    )
    top_needs = "arguments" in tool_call and not isinstance(
        tool_call.get("arguments"), dict
    )
    return nested_needs, top_needs


def _normalize_assistant_tool_call_arguments(messages: list) -> list:
    """Return ``messages`` with every ``assistant``-role tool_call's
    ``arguments`` normalised to a dict.

    Rules (mirror ``engine/batched.py::_normalize_tool_call_arguments_
    for_template`` so the two normalisers are semantically identical
    and safe to layer):

    * ``dict`` → unchanged.
    * ``str`` → ``json.loads``; if the parsed value is a dict, use it;
      otherwise wrap as ``{"value": <parsed>}``.
    * ``str`` that fails to JSON-parse → wrap as ``{"value": <raw>}``.
    * Every non-assistant role is untouched.
    * ABSENT ``arguments`` key is untouched — we do not invent a
      payload the client never sent (codex r1 NIT).

    Both OpenAI-wire shapes are covered INDEPENDENTLY:

    * Nested — ``tool_call.function.arguments`` (OpenAI ChatCompletion
      canonical shape; pydantic_ai / OpenAI SDK).
    * Top-level — ``tool_call.arguments`` (legacy / MCP / a few chat
      templates that flatten the envelope). Codex r1 BLOCKING: some
      templates access ``tool_call.arguments`` directly without an
      ``if tool_call.function is defined`` unwrap step, so the
      nested-only fix leaked the JSON-string form to those templates
      and the same ``TypeError`` fired.

    A mixed-shape replay (both nested AND top-level ``arguments``
    populated — some SDKs mirror the field for template compatibility)
    normalises BOTH. Codex r3 BLOCKING on PR #981 — a defensive
    "top-level only when nested absent" gate still leaked the
    JSON-string form to ``tc.arguments|items`` templates on the mixed
    replay shape.

    Idempotent: repeated calls after the first are no-ops for
    dict-form arguments, so this can safely layer on top of upstream
    normalisers in ``routes/chat.py`` and ``engine/batched.py`` without
    double-work.

    The scan is O(N) over messages. When nothing needs mutation we
    return the caller's list unchanged (no copy). When at least one
    ``arguments`` needs conversion we materialise a shallow copy of
    the touched messages (and their ``tool_calls``) so the caller's
    message list — which the route layer treats as the API surface
    where ``arguments`` MUST stay a string — is left intact.
    """
    if not isinstance(messages, list) or not messages:
        return messages

    # First pass: detect whether any assistant tool_call has a
    # non-dict ``arguments`` payload (either nested under ``function``
    # or top-level). If none, short-circuit without touching the list.
    needs_mutation = False
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        tool_calls = msg.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            nested_needs, top_needs = _tool_call_arguments_need_mutation(tc)
            if nested_needs or top_needs:
                needs_mutation = True
                break
        if needs_mutation:
            break
    if not needs_mutation:
        return messages

    # Second pass: shallow-copy touched messages + tool_calls + function
    # dicts. Untouched messages are shared by reference (cheap).
    normalized: list = []
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            normalized.append(msg)
            continue
        tool_calls = msg.get("tool_calls")
        if not isinstance(tool_calls, list):
            normalized.append(msg)
            continue
        new_tool_calls: list = []
        touched_any = False
        for tc in tool_calls:
            if not isinstance(tc, dict):
                new_tool_calls.append(tc)
                continue
            nested_needs, top_needs = _tool_call_arguments_need_mutation(tc)
            if not nested_needs and not top_needs:
                new_tool_calls.append(tc)
                continue
            new_tc = dict(tc)
            if nested_needs:
                function = tc["function"]
                new_function = dict(function)
                new_function["arguments"] = _coerce_arguments_to_dict(
                    function["arguments"]
                )
                new_tc["function"] = new_function
            if top_needs:
                new_tc["arguments"] = _coerce_arguments_to_dict(tc["arguments"])
            new_tool_calls.append(new_tc)
            touched_any = True
        if touched_any:
            new_msg = dict(msg)
            new_msg["tool_calls"] = new_tool_calls
            normalized.append(new_msg)
        else:
            normalized.append(msg)
    return normalized


def _serialize_assistant_tool_call_arguments(messages: list) -> list:
    """Return a copy with mapping-form tool arguments encoded as JSON.

    Most Hugging Face templates iterate over ``arguments`` and therefore need
    the internal mapping form produced by
    :func:`_normalize_assistant_tool_call_arguments`.  DeepSeek-R1's shipped
    template is a notable inverse: it concatenates ``arguments`` directly into
    a JSON code block and raises ``TypeError: can only concatenate str (not
    \"dict\") to str`` for a standards-compliant replayed tool call.

    This helper is intentionally used only as a compatibility retry after that
    exact render failure.  It is copy-on-write so neither the OpenAI request nor
    the normalised representation used by other templates is mutated.
    """
    if not isinstance(messages, list):
        return messages

    result = messages
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue

        new_calls = tool_calls
        message_changed = False
        for call_index, tool_call in enumerate(tool_calls):
            if not isinstance(tool_call, dict):
                continue
            new_call = tool_call
            call_changed = False

            function = tool_call.get("function")
            if isinstance(function, dict) and isinstance(
                function.get("arguments"), dict
            ):
                new_function = dict(function)
                new_function["arguments"] = json.dumps(
                    function["arguments"], ensure_ascii=False, separators=(",", ":")
                )
                new_call = dict(new_call)
                new_call["function"] = new_function
                call_changed = True

            if isinstance(tool_call.get("arguments"), dict):
                if not call_changed:
                    new_call = dict(new_call)
                new_call["arguments"] = json.dumps(
                    tool_call["arguments"],
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                call_changed = True

            if call_changed:
                if not message_changed:
                    new_calls = list(tool_calls)
                new_calls[call_index] = new_call
                message_changed = True

        if message_changed:
            if result is messages:
                result = list(messages)
            new_message = dict(message)
            new_message["tool_calls"] = new_calls
            result[index] = new_message

    return result


def _flatten_tool_history_for_alternating_template(messages: list) -> list:
    """Encode OpenAI tool history for templates limited to user/assistant.

    Gemma 3's official template rejects every ``role="tool"`` message and
    requires strict user/assistant alternation.  Preserve the conversation by
    rendering structured assistant calls as text, converting tool results to a
    user turn, and merging the immediately following user follow-up into that
    turn.  Called only after the template explicitly reports its alternation
    constraint, so native tool-aware templates retain their native shape.
    """
    flattened: list = []
    for message in messages:
        if not isinstance(message, dict):
            flattened.append(message)
            continue
        role = message.get("role")
        if role == "assistant" and isinstance(message.get("tool_calls"), list):
            parts: list[str] = []
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                parts.append(content.strip())
            for tool_call in message["tool_calls"]:
                if not isinstance(tool_call, dict):
                    continue
                function = tool_call.get("function")
                if not isinstance(function, dict):
                    continue
                name = function.get("name") or "unknown"
                arguments = function.get("arguments", {})
                if not isinstance(arguments, str):
                    arguments = json.dumps(
                        arguments, ensure_ascii=False, separators=(",", ":")
                    )
                parts.append(f"Tool call {name}: {arguments}")
            new_message = dict(message)
            new_message.pop("tool_calls", None)
            new_message["content"] = "\n".join(parts)
            flattened.append(new_message)
            continue
        if role == "tool":
            name = message.get("name") or message.get("tool_call_id") or "unknown"
            content = message.get("content")
            result_text = content if isinstance(content, str) else json.dumps(content)
            text = f"Tool result {name}: {result_text}"
            if (
                flattened
                and isinstance(flattened[-1], dict)
                and flattened[-1].get("role") == "user"
            ):
                prior = flattened[-1].get("content") or ""
                flattened[-1] = {**flattened[-1], "content": f"{prior}\n{text}"}
            else:
                flattened.append({"role": "user", "content": text})
            continue
        if role == "user" and flattened and isinstance(flattened[-1], dict):
            previous = flattened[-1]
            if previous.get("role") == "user" and str(
                previous.get("content", "")
            ).startswith("Tool result "):
                content = message.get("content") or ""
                flattened[-1] = {
                    **previous,
                    "content": f"{previous.get('content', '')}\n\n{content}",
                }
                continue
        flattened.append(message)
    return flattened


def _baseline_sanitize_messages(messages):
    """Fail-closed fallback for ``_sanitize_messages_for_template``.

    Applies the literal ``_CHAT_TEMPLATE_ROLE_MARKERS`` baseline (no
    tokenizer-registry probe — that's what failed) so a sanitiser
    exception cannot reopen the prompt-injection vector by passing
    raw user content through to ``apply_chat_template`` (codex r7
    BLOCKING). Mirrors the fallback in ``vllm_mlx/models/mllm.py``.
    """
    baseline_pattern = _build_marker_pattern(set(_CHAT_TEMPLATE_ROLE_MARKERS))
    if baseline_pattern is None:
        return messages
    fallback: list = []
    for msg in messages:
        if isinstance(msg, dict) and "content" in msg:
            new_msg = dict(msg)
            new_msg["content"] = _sanitize_message_content(
                msg["content"], baseline_pattern
            )
            fallback.append(new_msg)
        else:
            fallback.append(msg)
    return fallback


def _walk_tools_iter(tools, transform):
    """Iteratively walk a tool definition tree, applying ``transform`` to
    every string leaf and returning a structurally-identical deep copy.

    Both :func:`_baseline_sanitize_tools` and
    :func:`_sanitize_tools_for_template` previously used an inner ``_walk``
    that recursed on ``dict`` / ``list`` / ``tuple`` containers. That
    shape ate one Python frame per level of JSON nesting and crashed
    with ``RecursionError`` (HTTP 500) on a client-supplied
    ``tools[].function.parameters`` payload nested ~1000 deep
    (D-TOOL-RECUR; ~10–30 KB JSON, well under the body-size cap).
    Because the crash propagated out as an unhandled ``RecursionError``
    on every loaded model (parser-agnostic), it was an unauthenticated
    DoS surface.

    An iterative walk with an explicit work stack puts the depth bound
    on the heap instead of the C stack, so the same payload finishes
    in O(N) time and O(N) memory without touching the Python recursion
    limit. The body-depth guard (see ``RAPID_MLX_MAX_BODY_DEPTH``) and
    the per-tool depth validator (see ``RAPID_MLX_MAX_TOOL_SCHEMA_DEPTH``)
    upstream of this walk reject payloads whose nesting is large
    enough to be a memory-pressure concern in the first place; this
    iterative walk is the structural defense-in-depth so a payload
    that somehow slips past the guards still cannot crash the worker.

    ``transform`` is applied to every ``str`` leaf. Containers are
    deep-copied; ``tuple`` containers are preserved as tuples. Non-
    string scalars (``int``/``float``/``bool``/``None``) pass through
    unchanged — same contract as the previous recursive form.
    """
    # The work stack carries ``(parent_container, key_or_index, source_node,
    # depth)`` tuples. We allocate the result container up-front when
    # ``source_node`` is a container, push its children to the stack, and
    # let later iterations fill in the children slots in the result. For
    # tuples we accumulate a list buffer and convert in a second pass at
    # the end — see :func:`_finalize_tuple_buffers` for why the
    # second pass MUST run leaves-first (codex r1 BLOCKING #1).
    if isinstance(tools, str):
        return transform(tools)
    if not isinstance(tools, (dict, list, tuple)):
        return tools

    # ``root_holder`` is a single-slot container so the worker loop can
    # assign the root result via the same ``parent[key] = ...`` shape it
    # uses for every other node, without a special-case branch.
    root_holder: list = [None]
    # Stack entries: (parent, key, source, depth)
    stack: list = [(root_holder, 0, tools, 0)]
    # Track tuple buffers with their depth in the result tree so the
    # second pass can convert leaves-first. Each entry is
    # ``(depth, parent, key, list_buf)``. Sort by depth DESC at close
    # so the innermost buf becomes a tuple BEFORE the parent buf is
    # materialised, otherwise the parent tuple captures the (stale)
    # list reference and the inner tuple replacement is lost.
    tuple_buffers: list = []

    while stack:
        parent, key, src, depth = stack.pop()
        if isinstance(src, str):
            parent[key] = transform(src)
        elif isinstance(src, dict):
            new_dict: dict = {}
            parent[key] = new_dict
            for k, v in src.items():
                if isinstance(v, str):
                    new_dict[k] = transform(v)
                elif isinstance(v, (dict, list, tuple)):
                    new_dict[k] = None  # placeholder filled below
                    stack.append((new_dict, k, v, depth + 1))
                else:
                    new_dict[k] = v
        elif isinstance(src, list):
            new_list: list = [None] * len(src)
            parent[key] = new_list
            for i, v in enumerate(src):
                if isinstance(v, str):
                    new_list[i] = transform(v)
                elif isinstance(v, (dict, list, tuple)):
                    stack.append((new_list, i, v, depth + 1))
                else:
                    new_list[i] = v
        elif isinstance(src, tuple):
            # Allocate a list buffer; the parent slot temporarily holds
            # this list. The final-pass converter (post-order, by
            # descending depth) replaces ``parent[key]`` with
            # ``tuple(buf)`` only AFTER every child tuple beneath it
            # has already been converted in place inside ``buf``.
            buf: list = [None] * len(src)
            parent[key] = buf
            tuple_buffers.append((depth, parent, key, buf))
            for i, v in enumerate(src):
                if isinstance(v, str):
                    buf[i] = transform(v)
                elif isinstance(v, (dict, list, tuple)):
                    stack.append((buf, i, v, depth + 1))
                else:
                    buf[i] = v
        else:
            parent[key] = src

    # Convert tuple buffers back into tuples LEAVES-FIRST (deepest
    # depth processed first). codex r1 BLOCKING #1: insertion order
    # is push order, which for a DFS stack is parent-before-child.
    # If we materialise the outer tuple FIRST, the freshly-created
    # ``tuple(buf_outer)`` captures the inner buf as a LIST reference;
    # the subsequent ``buf_outer[i] = tuple(buf_inner)`` mutates the
    # list buffer but the outer tuple (immutable) still points at the
    # original list object, so the returned outer tuple contains a
    # list where the test expects a tuple. Sorting by ``-depth`` (or
    # equivalently the highest-depth-first descending sort) guarantees
    # the inner buf has already been replaced with its tuple form
    # INSIDE ``buf_outer`` before we materialise the outer tuple.
    tuple_buffers.sort(key=lambda entry: entry[0], reverse=True)
    for _depth, parent, key, buf in tuple_buffers:
        parent[key] = tuple(buf)

    return root_holder[0]


def _baseline_sanitize_tools(tools):
    """Fail-closed fallback for ``_sanitize_tools_for_template``.

    Walks the tool definition tree with the literal baseline marker
    set when the tokenizer-registry-aware sanitiser raises — same
    rationale as ``_baseline_sanitize_messages`` (codex r7 BLOCKING).

    Implemented on top of :func:`_walk_tools_iter` (iterative, explicit
    work-stack) so a client-supplied tool tree nested ~1000 levels deep
    cannot hit Python's recursion limit and crash the worker with HTTP
    500 (D-TOOL-RECUR). The iterative walk is the structural fix; the
    request-time depth validator in :func:`_validate_tool_schema_depth`
    (``RAPID_MLX_MAX_TOOL_SCHEMA_DEPTH``) rejects deep payloads earlier
    with a sanitized 400.
    """
    if not tools:
        return tools
    baseline_pattern = _build_marker_pattern(set(_CHAT_TEMPLATE_ROLE_MARKERS))
    if baseline_pattern is None:
        return tools
    return _walk_tools_iter(tools, lambda s: _neutralize_in_string(s, baseline_pattern))


def _sanitize_tools_for_template(
    tools, template_applicator, *, include_reasoning_sentinels: bool = False
):
    """Neutralise chat-template role markers in user-supplied tool
    definitions (names, descriptions, parameter schemas).

    Tool definitions also come from the request body and are rendered
    into the same prompt either by the native template's ``tools=``
    kwarg or by ``_inject_tools_into_messages``'s system-prompt
    fallback. Pre-fix only ``messages`` was sanitised, so a
    client-controlled tool description containing ``<|im_start|>...``
    re-opened the bypass for tool-using requests. Codex r5 P1.

    The neutralisation walks the tool definition tree iteratively —
    every string leaf is run through ``_neutralize_in_string``. Lists
    and dicts are walked structurally; non-string scalars pass
    through unchanged.

    The walk uses :func:`_walk_tools_iter` (explicit work-stack)
    instead of the previous recursive descent so a client-supplied
    schema nested ~1000 levels deep cannot crash the worker with
    HTTP 500 on Python's recursion-limit (D-TOOL-RECUR). The
    request-time depth validator at
    :data:`MAX_TOOL_SCHEMA_DEPTH_ENV` rejects deeper payloads with a
    sanitized 400 before reaching this sanitiser — this iterative
    form is the structural defense-in-depth.
    """
    if not tools:
        return tools
    markers = _collect_role_markers(
        template_applicator,
        include_reasoning_sentinels=include_reasoning_sentinels,
    )
    pattern = _build_marker_pattern(markers)
    if pattern is None:
        return tools

    def _sanitize_tool_string(value: str) -> str:
        if include_reasoning_sentinels:
            value = _double_existing_control_escapes(value)
        return _neutralize_in_string(value, pattern)

    return _walk_tools_iter(tools, _sanitize_tool_string)


def _build_tool_injection_text(tools: list[dict]) -> str:
    """Build a compact tool definition string for system prompt injection.

    When a chat template doesn't support the ``tools`` parameter natively,
    we inject tool definitions into the system message so the model can
    still see them.

    Args:
        tools: List of tool definitions in OpenAI function-calling format.

    Returns:
        A formatted string describing available tools and calling format.
    """
    lines = ["# Available Tools", ""]
    for tool in tools:
        func = tool.get("function", tool)
        name = func.get("name", "unknown")
        desc = func.get("description", "")
        params = func.get("parameters", {})
        props = params.get("properties", {})
        required = params.get("required", [])

        lines.append(f"## {name}")
        if desc:
            lines.append(f"{desc}")
        if props:
            lines.append(f"Parameters: {json.dumps(props, ensure_ascii=False)}")
        if required:
            lines.append(f"Required: {json.dumps(required)}")
        lines.append("")

    lines.append(
        "When you need to use a tool, respond with a JSON object "
        'containing "name" and "arguments" keys.'
    )

    return "\n".join(lines)


def _inject_tools_into_messages(messages: list[dict], tools: list[dict]) -> list[dict]:
    """Inject tool definitions into the system message.

    If the first message has role ``system``, append to its content.
    Otherwise, prepend a new system message with the tool definitions.

    Args:
        messages: Original messages (not mutated).
        tools: Tool definitions to inject.

    Returns:
        A shallow copy of messages with tool definitions injected.
    """
    injection = _build_tool_injection_text(tools)
    msgs = copy.copy(messages)

    if msgs and msgs[0].get("role") == "system":
        first = dict(msgs[0])
        existing = first.get("content", "")
        # Handle content parts format (multimodal messages)
        if isinstance(existing, list):
            # Append as a new text part
            first["content"] = list(existing) + [
                {"type": "text", "text": "\n\n" + injection}
            ]
        else:
            first["content"] = str(existing) + "\n\n" + injection
        msgs[0] = first
    else:
        msgs.insert(0, {"role": "system", "content": injection})

    return msgs


# Hy3 detection — case-insensitive family-boundary match against the
# alias name, HF path, or local directory. Covers ``hy3-preview-4bit``,
# ``mlx-community/Hy3-preview-4bit``, ``Hunyuan-3-Preview``,
# ``hunyuan3``, ``hy-v3-experimental`` and any future ``Hy3-*`` or
# ``Hunyuan-3-*`` re-upload without a per-repo allowlist.
#
# Codex round-3 NIT (PR #1070 finding #4): earlier form used unanchored
# ``hunyuan.?3`` which happily matched substrings inside unrelated
# names / paths (``not-hunyuanx3-test``, any local path containing
# that character sequence). Tightening to family separators plus
# start / end of string is precise enough for HF repo paths and CLI
# alias forms while rejecting incidental substrings.
#
# codex R13 BLOCKING: the TRAILING class must NOT include ``/`` (mirrors the
# same fix in ``model_auto_config.py`` R11) — else a non-Hy3 repo under an HF
# org / local parent directory named ``hy3`` (``hy3/qwen-model``,
# ``some/hy3/nested-qwen``) had ``reasoning_effort="low"`` injected because the
# ``hy3`` PARENT segment matched. The family root must sit in the FINAL path
# segment (the repo/alias name): a LEADING separator (``/`` ``_`` ``.`` ``-``)
# may precede the root, but the root must be followed by end-of-string OR an
# in-segment continuation (``_`` ``.`` ``-``), never a ``/`` path boundary.
# Still matches ``mlx-community/Hy3-preview-4bit``, bare ``hy3``, ``org/hy3``,
# ``Hunyuan-3-Preview``.
_HY3_MODEL_NAME_RE = re.compile(
    r"(?:^|[/_.\-])(?:hy3|hy-v3|hunyuan[-_]?3)(?:$|[_.\-])",
    re.IGNORECASE,
)
_GPT_OSS_MODEL_NAME_RE = re.compile(
    r"(?:^|[/_.\-])gpt[-_]oss(?:$|[_.\-])",
    re.IGNORECASE,
)


def _looks_like_hy3(model_name: str) -> bool:
    """Return True when the model name is Tencent Hunyuan 3 / Hy3.

    Used to gate the ``reasoning_effort='low'`` chat-template default
    injection (fixes upstream PR #1211 comment 4927711484 factual-recall
    regression). Kept as a narrowly-scoped helper so the eventual PR-3
    (which may add explicit request-side ``reasoning_effort`` plumbing)
    doesn't have to duplicate the pattern.
    """
    if not model_name:
        return False
    return bool(_HY3_MODEL_NAME_RE.search(model_name))


def _looks_like_gpt_oss(model_name: str) -> bool:
    """Return True when the model name is the GPT-OSS / Harmony family."""
    if not model_name:
        return False
    return bool(_GPT_OSS_MODEL_NAME_RE.search(model_name))


def _looks_like_gpt_oss_harmony_template(template: str) -> bool:
    """Return True for Harmony chat templates even under a served alias."""
    return all(
        marker in template for marker in ("<|start|>", "<|channel|>", "<|message|>")
    )


def _chat_template_strings(template, *, tools: list[dict] | None = None) -> list[str]:
    if isinstance(template, str):
        return [template]
    if isinstance(template, dict):
        preferred_keys = ("tool_use", "tools", "default") if tools else ("default",)
        for key in preferred_keys:
            value = template.get(key)
            if isinstance(value, str):
                return [value]
        string_values = [value for value in template.values() if isinstance(value, str)]
        return string_values if len(string_values) == 1 else []
    return []


def _template_uses_reasoning_effort_without_enable_thinking(
    template_applicator,
    model_name: str = "",
    tools: list[dict] | None = None,
) -> bool:
    """Return True for templates such as GPT-OSS/Harmony that expose a
    ``reasoning_effort`` kwarg but do not consult ``enable_thinking``.

    In that shape, passing ``enable_thinking=False`` is silently inert;
    the closest template-native low-reasoning request is
    ``reasoning_effort="low"``.
    """
    templates = _chat_template_strings(
        getattr(template_applicator, "chat_template", None),
        tools=tools,
    )
    if not templates:
        return False
    return any(
        "reasoning_effort" in template
        and "enable_thinking" not in template
        and (
            _looks_like_gpt_oss(model_name)
            or _looks_like_gpt_oss_harmony_template(template)
        )
        for template in templates
    )


def _is_gpt_oss_harmony_template(
    template_applicator,
    *,
    model_name: str = "",
    tools: list[dict] | None = None,
) -> bool:
    """Identify GPT-OSS templates whose wire format is OpenAI Harmony."""
    templates = _chat_template_strings(
        getattr(template_applicator, "chat_template", None), tools=tools
    )
    if templates:
        return any(
            _looks_like_gpt_oss_harmony_template(template) for template in templates
        )
    return _looks_like_gpt_oss(model_name)


def _collapse_harmony_system_messages(messages: list[dict]) -> list[dict]:
    """Make every system instruction visible to leading-only Harmony templates.

    GPT-OSS' published template consumes a system/developer role only at index
    zero and silently skips later system roles.  Harmony has no mid-conversation
    system frame, so preserve authority by joining all system instructions into
    the single leading developer frame the template does support.
    """
    # Harmony consumes at most the first message as an instruction. This also
    # catches two consecutive leading system messages: the second is otherwise
    # just as invisible as one placed after a user turn.
    instruction_roles = {"system", "developer"}
    if not any(
        index > 0 and message.get("role") in instruction_roles
        for index, message in enumerate(messages)
    ):
        return messages

    instruction_messages = [
        message for message in messages if message.get("role") in instruction_roles
    ]
    if any(set(message) - {"role", "content"} for message in instruction_messages):
        raise ValueError(
            "GPT-OSS/Harmony cannot preserve metadata on a conversation "
            "instruction message"
        )

    contents = [message.get("content") for message in instruction_messages]
    if not all(isinstance(content, str) for content in contents):
        raise ValueError(
            "GPT-OSS/Harmony system and developer messages must contain "
            "text-only content"
        )

    instruction_role = instruction_messages[0]["role"]
    if any(message["role"] != instruction_role for message in instruction_messages):
        raise ValueError(
            "GPT-OSS/Harmony cannot preserve mixed system and developer "
            "instruction roles"
        )

    collapsed = [
        message for message in messages if message.get("role") not in instruction_roles
    ]
    # All instructions have the same authority role here, so folding them into
    # the single leading frame supported by Harmony is lossless with respect to
    # role authority.
    collapsed.insert(
        0,
        {
            "role": instruction_role,
            "content": "\n\n".join(contents),
        },
    )
    return collapsed


def apply_chat_template(
    template_applicator,
    messages: list[dict],
    tools: list[dict] | None = None,
    enable_thinking: bool | None = None,
    model_name: str = "",
    add_generation_prompt: bool = True,
) -> str:
    """Apply a chat template to messages with consistent fallback behavior.

    Applies a chat template with consistent fallback for ``enable_thinking``
    and ``tools`` parameters.

    Args:
        template_applicator: Object with ``apply_chat_template`` method
            (tokenizer or processor).
        messages: List of chat messages in OpenAI format.
        tools: Converted tool definitions for the template, or None.
        enable_thinking: Whether to enable thinking mode.
            - True/False: explicit control
            - None: auto-detect (True except for coder models)
        model_name: Model name string, used for auto-detection of
            ``enable_thinking`` when set to None.
        add_generation_prompt: Whether the template should append the
            assistant generation prefix (default True — every serving path).
            Passed False only by the reasoning-budget seed probe
            (``routes/chat.py::_template_generation_prefix``), which renders the
            SAME conversation with and without the generation prompt and takes
            the delta to isolate the template-added prefix exactly.

    Returns:
        The formatted prompt string.  Falls back to a plain
        ``role: content`` format if the applicator has no
        ``apply_chat_template`` method.
    """
    from .deepseek_v4_0731 import encode_messages, is_deepseek_v4_0731
    from .gemma4_chat_template import upgrade_stale_gemma4_chat_template

    # Converted Gemma 4 checkpoints commonly retain the pre-2026-07-09
    # template even though Google fixed null rendering and multi-turn tool
    # continuation upstream.  Upgrade only that recognizable stale template;
    # current canonical and custom templates are preserved.
    upgrade_stale_gemma4_chat_template(template_applicator, model_name)

    is_deepseek_v4 = is_deepseek_v4_0731(model_name)

    # F-111: flatten text-only OpenAI-o1+ content arrays
    # (``content: [{"type":"text","text":"X"}]``) into the plain string
    # the HF chat templates expect. Runs FIRST so the sanitiser and the
    # template itself both see a uniform ``content`` shape. A non-text
    # part on a ``tool``-role message raises ``ValueError`` — surfaced
    # by the caller (``routes/chat.py``) as HTTP 400. NOT wrapped in a
    # try/except: silently dropping a non-text tool part would reopen
    # the same "tool content missing" footgun (Qwen3 rendered an empty
    # ``<tool_response>``, Hermes3 ``TypeError``-d).
    messages = _normalize_text_only_content_arrays(messages)

    # GH-973: enforce the assistant tool_call.arguments = dict invariant
    # BEFORE any Jinja rendering. Every mainstream HF chat template
    # (Qwen3 / Hermes / Llama3 / GLM4 / Nemotron / minimax) iterates
    # ``tool_call.arguments|items`` and blows up with
    # ``TypeError: Can only get item pairs from a mapping`` when the
    # OpenAI-wire JSON-string form leaks through. Upstream normalisers
    # in ``routes/chat.py::extract_multimodal_content`` and
    # ``engine/batched.py::_normalize_tool_call_arguments_for_template``
    # cover the standard ``/v1/chat/completions`` non-MLLM path, but
    # every other caller (guided-generation
    # ``BatchedEngine.stream_guided_completion``, native-video path,
    # direct engine callers, tests) bypasses them. Applying the
    # invariant here — the single ``apply_chat_template`` choke point —
    # closes the gap uniformly. Idempotent: dict-form arguments pass
    # through unchanged, so callers that already normalised pay no cost.
    messages = _normalize_assistant_tool_call_arguments(messages)

    # Neutralize chat-template role markers in untrusted (user/tool)
    # content BEFORE the tokenizer parses them. Runs unconditionally for
    # every template-render path in the project (this is the single
    # wrapper every caller funnels through), so the fix is template-
    # agnostic — no per-model handling. See ``_sanitize_messages_for_template``.
    # Fail CLOSED on sanitiser exceptions — falling back to the literal
    # ``_CHAT_TEMPLATE_ROLE_MARKERS`` baseline. Swallowing the failure
    # and rendering raw input would reopen the exact prompt-injection
    # vector this PR closes (codex r7 BLOCKING — same fallback shape as
    # ``vllm_mlx/models/mllm.py::_apply_native_video_template``).
    try:
        messages = _sanitize_messages_for_template(
            messages,
            template_applicator,
            include_reasoning_sentinels=is_deepseek_v4,
        )
    except Exception as e:
        logger.debug(
            "Chat-template marker sanitisation failed (%s); applying "
            "baseline-marker fallback",
            e,
        )
        messages = _baseline_sanitize_messages(messages)
    # Same defence on tool definitions (codex r5 P1) — they are also
    # client-supplied strings rendered into the prompt via the
    # template's ``tools=`` kwarg or the system-prompt injection
    # fallback (``_inject_tools_into_messages``).
    try:
        tools = _sanitize_tools_for_template(
            tools,
            template_applicator,
            include_reasoning_sentinels=is_deepseek_v4,
        )
    except Exception as e:
        logger.debug(
            "Chat-template tool-marker sanitisation failed (%s); applying "
            "baseline-marker fallback",
            e,
        )
        tools = _baseline_sanitize_tools(tools)

    # The published GPT-OSS template accepts a mid-conversation system role but
    # has no loop branch for it, so rendering succeeds while deleting the
    # instruction.  Error-driven compatibility fallback (#1543) cannot catch a
    # successful lossy render.  Normalize only the Harmony family, whose wire
    # protocol has a single leading developer instruction frame.
    if _is_gpt_oss_harmony_template(
        template_applicator, model_name=model_name, tools=tools
    ):
        messages = _collapse_harmony_system_messages(messages)

    # DeepSeek-V4-Flash-0731 intentionally ships a Python encoder instead of
    # a Jinja template.  Route by model identity before the generic tokenizer
    # fallback (which would otherwise silently apply ChatML).
    if is_deepseek_v4:
        return encode_messages(
            messages,
            tools=tools,
            enable_thinking=enable_thinking is not False,
            add_generation_prompt=add_generation_prompt,
        )

    if not hasattr(template_applicator, "apply_chat_template"):
        # Fallback for models without apply_chat_template.
        # Inject tools into the system prompt so the model still sees
        # function schemas — same treatment as the TypeError fallback
        # below.  Fixes #120.
        if tools:
            messages = _inject_tools_into_messages(messages, tools)
        prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        return prompt + "\nassistant:"

    if enable_thinking is None:
        enable_thinking = "coder" not in model_name.lower()

    template_kwargs: dict = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
        "enable_thinking": enable_thinking,
    }
    if tools:
        template_kwargs["tools"] = tools

    # GPT-OSS / Harmony-style templates do not expose an on/off
    # ``enable_thinking`` switch; they expose ``reasoning_effort`` and default
    # it to ``medium``. When a route already resolved ``enable_thinking=False``
    # (tools / strict-json / casual-chat auto-disable, or explicit client
    # opt-out), request the lowest native effort instead of letting the
    # template silently ignore the off flag and keep ``Reasoning: medium``.
    if (
        enable_thinking is False
        and _template_uses_reasoning_effort_without_enable_thinking(
            template_applicator, model_name=model_name, tools=tools
        )
    ):
        template_kwargs.setdefault("reasoning_effort", "low")

    # Hy3 chat_template.jinja defaults ``reasoning_effort=no_think`` which
    # empirically returns "France" instead of "Paris" on factual-recall
    # questions (upstream PR #1211 comment 4927711484, 2026-07-09 spike).
    # Override the default to ``low`` for Hy3 so out-of-the-box requests
    # produce correct answers without the client having to learn the
    # template kwarg. Fires ONLY when:
    #   * model_name signals Hy3 (separator-bounded, case-insensitive family
    #     match via `_HY3_MODEL_NAME_RE` — not a loose substring)
    #   * ``enable_thinking`` is not False (a client that explicitly
    #     disabled thinking wants no_think — respect that intent)
    # NOTE (codex R12 NIT): there is presently NO request-side
    # ``reasoning_effort`` plumb-through — the value is template-only, and this
    # override is the sole injection point. ``setdefault`` (not direct
    # assignment) is deliberate future-proofing: IF a later revision plumbs a
    # graded effort (``medium`` / ``high``) through and pre-populates
    # ``template_kwargs["reasoning_effort"]`` upstream of this call, the
    # explicit value survives instead of being silently overwritten. Until that
    # plumb-through exists, ``setdefault`` behaves identically to assignment
    # here (the key is never pre-populated). Non-Hy3 models never see the kwarg,
    # so no risk of TypeError on other templates.
    if _looks_like_hy3(model_name) and enable_thinking is not False:
        template_kwargs.setdefault("reasoning_effort", "low")

    def _apply_with_alternating_fallback(
        candidate_messages: list[dict], candidate_kwargs: dict
    ) -> str:
        try:
            return template_applicator.apply_chat_template(
                candidate_messages, **candidate_kwargs
            )
        except Exception as exc:
            if "Conversation roles must alternate user/assistant" not in str(
                exc
            ) or not any(
                isinstance(message, dict) and message.get("role") == "tool"
                for message in candidate_messages
            ):
                raise
            flattened = _flatten_tool_history_for_alternating_template(
                candidate_messages
            )
            alternating_kwargs = dict(candidate_kwargs)
            fallback_tools = alternating_kwargs.pop("tools", None)
            if fallback_tools:
                flattened = _inject_tools_into_messages(flattened, fallback_tools)
            return template_applicator.apply_chat_template(
                flattened, **alternating_kwargs
            )

    def _apply_with_mid_system_fallback(
        candidate_messages: list[dict], candidate_kwargs: dict
    ) -> str:
        """Retry templates that explicitly reject a non-leading system role.

        Render the client's message order first so templates that accept
        mid-conversation system messages keep their native semantics.  Only
        the well-known Qwen/Llama/Gemma guard opts into the compatibility
        retry; unrelated template failures must remain unchanged.
        """
        try:
            return _apply_with_alternating_fallback(
                candidate_messages, candidate_kwargs
            )
        except Exception as original:
            if "System message must be at the beginning." not in str(original):
                raise

            first_body = next(
                (
                    index
                    for index, message in enumerate(candidate_messages)
                    if message.get("role") != "system"
                ),
                len(candidate_messages),
            )
            has_mid_system = any(
                message.get("role") == "system"
                for message in candidate_messages[first_body:]
            )
            if not has_mid_system:
                raise

            system_messages = [
                message
                for message in candidate_messages
                if message.get("role") == "system"
            ]
            # Collapsing multiple system messages cannot faithfully preserve
            # per-message metadata such as ``name``. Refuse that lossy retry
            # and surface the template's original diagnostic instead.
            if any(set(message) - {"role", "content"} for message in system_messages):
                raise

            system_contents = [
                message.get("content")
                for message in system_messages
                if message.get("content")
            ]
            if all(isinstance(content, str) for content in system_contents):
                merged_system_content: str | list = "\n\n".join(system_contents)
            else:
                # Multimodal templates may carry structured content arrays.
                # Preserve those parts instead of stringifying them; inject a
                # text separator between instructions so their boundaries do
                # not disappear when two system messages are combined.
                merged_parts: list = []
                for content in system_contents:
                    if merged_parts:
                        merged_parts.append({"type": "text", "text": "\n\n"})
                    if isinstance(content, list):
                        merged_parts.extend(content)
                    elif isinstance(content, dict):
                        merged_parts.append(content)
                    else:
                        merged_parts.append({"type": "text", "text": str(content)})
                merged_system_content = merged_parts
            collapsed = [
                message
                for message in candidate_messages
                if message.get("role") != "system"
            ]
            if merged_system_content:
                collapsed.insert(
                    0, {"role": "system", "content": merged_system_content}
                )

            try:
                return _apply_with_alternating_fallback(collapsed, candidate_kwargs)
            except Exception:
                # Preserve the first diagnostic: it describes the client input
                # that triggered compatibility handling, not our retry shape.
                raise original

    try:
        return _apply_with_mid_system_fallback(messages, template_kwargs)
    except TypeError as e:
        retry_messages = messages
        # DeepSeek-R1's published template concatenates historical tool-call
        # arguments as text, while the majority of HF templates iterate them as
        # mappings.  The shared boundary normalises to the majority mapping
        # form above; retry the exact inverse incompatibility with JSON strings
        # before treating the TypeError as an unsupported template kwarg.
        if 'can only concatenate str (not "dict") to str' in str(e):
            string_argument_messages = _serialize_assistant_tool_call_arguments(
                messages
            )
            if string_argument_messages is not messages:
                retry_messages = string_argument_messages
                try:
                    return _apply_with_mid_system_fallback(
                        string_argument_messages, template_kwargs
                    )
                except TypeError:
                    # It was not the known argument-shape incompatibility; keep
                    # the existing generic kwarg/tools fallback behaviour.
                    pass
        # Step 1: retry without enable_thinking (many templates don't support it).
        # Codex round-1 NIT fix (PR #1070 finding #4): keep
        # ``reasoning_effort`` on this first retry so a Hy3 checkpoint
        # that supports ``reasoning_effort`` but rejects
        # ``enable_thinking`` still gets the ``low`` override. Only drop
        # ``reasoning_effort`` on the SECOND TypeError below, when we
        # know the retry itself failed.
        logger.debug("Chat template TypeError, retrying without enable_thinking: %s", e)
        template_kwargs.pop("enable_thinking", None)
        try:
            return _apply_with_mid_system_fallback(retry_messages, template_kwargs)
        except TypeError as e2:
            # Second failure. Only drop ``reasoning_effort`` when the error
            # actually names it (codex R8 BLOCKING: unconditionally popping it
            # here loses the load-bearing Hy3 ``reasoning_effort="low"`` override
            # when the REAL culprit is ``tools`` — the template rejects tools,
            # not reasoning_effort, and the prompt-injection tools fallback below
            # would then run without the override, regressing Hy3 factual
            # recall). When the failure is about tools, keep reasoning_effort so
            # the tools fallback preserves it.
            # Match Python's ACTUAL unexpected-kwarg error text rather than a
            # loose substring (codex R9 NIT: a template/user error that merely
            # mentions ``reasoning_effort`` in another context must not trigger
            # the drop). CPython raises: "<fn>() got an unexpected keyword
            # argument 'reasoning_effort'".
            _e2 = str(e2)
            reasoning_effort_is_culprit = (
                "unexpected keyword argument 'reasoning_effort'" in _e2
                or 'unexpected keyword argument "reasoning_effort"' in _e2
            )
            if reasoning_effort_is_culprit:
                logger.debug(
                    "Chat template TypeError persisted, dropping "
                    "reasoning_effort (named as unexpected kwarg): %s",
                    e2,
                )
                template_kwargs.pop("reasoning_effort", None)
            else:
                logger.debug(
                    "Chat template TypeError persisted (not reasoning_effort) — "
                    "keeping reasoning_effort for the tools fallback: %s",
                    e2,
                )

        # Step 2: template also rejects tools — fall back to prompt injection.
        # Restore enable_thinking: the step-1 pop removed it because we
        # didn't know yet whether the failure was about enable_thinking
        # or about tools.  Now we know it was tools, so re-add
        # enable_thinking for the final retry so thinking-capable models
        # (Qwen, DeepSeek) don't silently lose that feature.  Fixes #122.
        template_kwargs.pop("tools", None)
        if enable_thinking is not None:
            template_kwargs["enable_thinking"] = enable_thinking
        if tools:
            logger.info(
                "Chat template doesn't support tools param — "
                "injecting %d tool definitions into system prompt",
                len(tools),
            )
            injected = _inject_tools_into_messages(retry_messages, tools)
            try:
                return _apply_with_mid_system_fallback(injected, template_kwargs)
            except TypeError:
                # enable_thinking also unsupported after all — drop it
                template_kwargs.pop("enable_thinking", None)
                return _apply_with_mid_system_fallback(injected, template_kwargs)

        return _apply_with_mid_system_fallback(retry_messages, template_kwargs)
