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


def _collect_role_markers(template_applicator) -> set[str]:
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
        markers |= _collect_role_markers(tokenizer.tokenizer)

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


def _sanitize_messages_for_template(
    messages: list[dict], template_applicator
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
    markers = _collect_role_markers(template_applicator)
    pattern = _build_marker_pattern(markers)
    if pattern is None:
        return messages
    sanitized: list[dict] = []
    for msg in messages:
        if not isinstance(msg, dict):
            sanitized.append(msg)
            continue
        content = msg.get("content")
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
      non-dict AND there is no nested ``function`` dict (top-level
      arguments is the fallback shape for legacy clients that flatten
      the OpenAI envelope; when ``function`` is present the OpenAI
      convention is that ``function.arguments`` is authoritative and
      the top-level ``arguments`` doesn't exist — a defensive extra
      normalisation there could double-mutate).

    Absent ``arguments`` keys yield ``False`` — we don't invent a
    payload for something the caller never sent (codex r1 NIT).
    """
    function = tool_call.get("function")
    nested_needs = (
        isinstance(function, dict)
        and "arguments" in function
        and not isinstance(function.get("arguments"), dict)
    )
    top_needs = (
        not isinstance(function, dict)
        and "arguments" in tool_call
        and not isinstance(tool_call.get("arguments"), dict)
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

    Both OpenAI-wire shapes are covered:

    * Nested — ``tool_call.function.arguments`` (OpenAI ChatCompletion
      canonical shape; pydantic_ai / OpenAI SDK).
    * Top-level — ``tool_call.arguments`` (legacy / MCP / a few chat
      templates that flatten the envelope). Codex r1 BLOCKING: some
      templates access ``tool_call.arguments`` directly without an
      ``if tool_call.function is defined`` unwrap step, so the
      nested-only fix leaked the JSON-string form to those templates
      and the same ``TypeError`` fired.

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


def _sanitize_tools_for_template(tools, template_applicator):
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
    markers = _collect_role_markers(template_applicator)
    pattern = _build_marker_pattern(markers)
    if pattern is None:
        return tools
    return _walk_tools_iter(tools, lambda s: _neutralize_in_string(s, pattern))


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


def apply_chat_template(
    template_applicator,
    messages: list[dict],
    tools: list[dict] | None = None,
    enable_thinking: bool | None = None,
    model_name: str = "",
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

    Returns:
        The formatted prompt string.  Falls back to a plain
        ``role: content`` format if the applicator has no
        ``apply_chat_template`` method.
    """
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
        messages = _sanitize_messages_for_template(messages, template_applicator)
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
        tools = _sanitize_tools_for_template(tools, template_applicator)
    except Exception as e:
        logger.debug(
            "Chat-template tool-marker sanitisation failed (%s); applying "
            "baseline-marker fallback",
            e,
        )
        tools = _baseline_sanitize_tools(tools)

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
        "add_generation_prompt": True,
        "enable_thinking": enable_thinking,
    }
    if tools:
        template_kwargs["tools"] = tools

    try:
        return template_applicator.apply_chat_template(messages, **template_kwargs)
    except TypeError as e:
        # Step 1: retry without enable_thinking (many templates don't support it)
        logger.debug("Chat template TypeError, retrying without enable_thinking: %s", e)
        template_kwargs.pop("enable_thinking", None)
        try:
            return template_applicator.apply_chat_template(messages, **template_kwargs)
        except TypeError:
            pass

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
            injected = _inject_tools_into_messages(messages, tools)
            try:
                return template_applicator.apply_chat_template(
                    injected, **template_kwargs
                )
            except TypeError:
                # enable_thinking also unsupported after all — drop it
                template_kwargs.pop("enable_thinking", None)
                return template_applicator.apply_chat_template(
                    injected, **template_kwargs
                )

        return template_applicator.apply_chat_template(messages, **template_kwargs)
