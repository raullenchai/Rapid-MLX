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


def _baseline_sanitize_tools(tools):
    """Fail-closed fallback for ``_sanitize_tools_for_template``.

    Walks the tool definition tree with the literal baseline marker
    set when the tokenizer-registry-aware sanitiser raises — same
    rationale as ``_baseline_sanitize_messages`` (codex r7 BLOCKING).
    """
    if not tools:
        return tools
    baseline_pattern = _build_marker_pattern(set(_CHAT_TEMPLATE_ROLE_MARKERS))
    if baseline_pattern is None:
        return tools

    def _walk(obj):
        if isinstance(obj, str):
            return _neutralize_in_string(obj, baseline_pattern)
        if isinstance(obj, dict):
            return {k: _walk(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_walk(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(_walk(v) for v in obj)
        return obj

    return _walk(tools)


def _sanitize_tools_for_template(tools, template_applicator):
    """Neutralise chat-template role markers in user-supplied tool
    definitions (names, descriptions, parameter schemas).

    Tool definitions also come from the request body and are rendered
    into the same prompt either by the native template's ``tools=``
    kwarg or by ``_inject_tools_into_messages``'s system-prompt
    fallback. Pre-fix only ``messages`` was sanitised, so a
    client-controlled tool description containing ``<|im_start|>...``
    re-opened the bypass for tool-using requests. Codex r5 P1.

    The neutralisation is recursive over the tool definition tree —
    every string leaf is run through ``_neutralize_in_string``. Lists
    and dicts are walked structurally; non-string scalars pass
    through unchanged.
    """
    if not tools:
        return tools
    markers = _collect_role_markers(template_applicator)
    pattern = _build_marker_pattern(markers)
    if pattern is None:
        return tools

    def _walk(obj):
        if isinstance(obj, str):
            return _neutralize_in_string(obj, pattern)
        if isinstance(obj, dict):
            return {k: _walk(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_walk(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(_walk(v) for v in obj)
        return obj

    return _walk(tools)


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
