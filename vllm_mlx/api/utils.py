# SPDX-License-Identifier: Apache-2.0
"""
Utility functions for text processing and model detection.
"""

import logging
import re

from .models import Message

logger = logging.getLogger(__name__)

# =============================================================================
# Special Token Patterns
# =============================================================================

# Pattern to match special tokens that should be removed from output
# Keeps <think>...</think> blocks intact for reasoning models
SPECIAL_TOKENS_PATTERN = re.compile(
    r"<\|im_end\|>|<\|im_start\|>|<\|endoftext\|>|"
    r"<\|end\|>|<\|eot_id\|>|<\|eom_id\|>|<\|python_tag\|>|"
    r"<\|start_header_id\|>|<\|end_header_id\|>|"
    r"<\|channel\|>|<\|message\|>|<\|start\|>|<\|return\|>|<\|call\|>|<\|constrain\|>|"
    r"<\|turn>|<turn\|>|"
    r"</s>|<s>|<pad>|\[PAD\]|\[SEP\]|\[CLS\]|"
    r"\[e~\[|\]~b\][a-z]*|\]~!b\["
)

# Fast-path characters that MUST be present for any special token to match.
# If none of these appear in the text, regex can be skipped entirely.
_SPECIAL_TOKEN_CHARS = frozenset("<[]")


def strip_special_tokens(text: str) -> str:
    """Remove special tokens from text with a fast-path bypass.

    Most per-token deltas are plain text without special token markers.
    Checking for marker characters first avoids regex overhead on ~99% of tokens.
    """
    # Fast path: no marker characters → no special tokens possible
    for ch in text:
        if ch in _SPECIAL_TOKEN_CHARS:
            return SPECIAL_TOKENS_PATTERN.sub("", text)
    return text


# =============================================================================
# Final sanitizer — last-mile catch-all before content reaches the client.
# Catches ANY remaining markup that earlier layers missed, including:
# - All <|..> and <..|> asymmetric tokens (Gemma 4 style)
# - All <|..|> symmetric tokens (Qwen, GPT-OSS style)
# - [Calling tool:...] text-format tool calls
# - Stray </think>, </tool_call>, etc.
# =============================================================================

_FINAL_SANITIZER = re.compile(
    # Full Gemma 4 tool call (greedy body): <|tool_call>call:name{...}<tool_call|>
    # MUST be listed BEFORE the bare-token strippers, otherwise the inner
    # `call:name{...}` body would be left orphaned in content.
    r"<\|tool_call>.*?<tool_call\|>"
    # Any <|...> or <...|> token (Gemma 4 asymmetric: <|channel>, <tool_call|>, etc.)
    r"|<\|[a-z_\"]+>|<[a-z_\"]+\|>"
    # Any <|...|> token (symmetric: <|im_end|>, <|channel|>, etc.)
    r"|<\|[a-z_]+\|>"
    # [Calling tool:...] or [Calling tool="..."] or bare "[Calling tool" (Gemma 4 mimicry)
    r"|\[Calling\s+tool[^\]]*\]?"
    # Stray closing tags
    r"|</think>|</tool_call>",
    re.DOTALL,
)


def sanitize_output(text: str) -> str:
    """Final catch-all sanitizer for client-facing content.

    This is the LAST defense against markup leakage. Runs after all
    parsers and filters. Strips anything that looks like a special token
    or internal markup pattern.

    Designed to be aggressive — better to over-strip than to leak.
    """
    if not text:
        return text
    for ch in text:
        if ch in _SPECIAL_TOKEN_CHARS:
            cleaned = _FINAL_SANITIZER.sub("", text).strip()
            return cleaned or None  # collapse empty to None
    return text


# Regex for matching final channel marker with optional constrain token:
#   <|channel|>final<|message|>
#   <|channel|>final <|constrain|>JSON<|message|>
_FINAL_CHANNEL_RE = re.compile(
    r"<\|channel\|>final[^<]*(?:<\|constrain\|>[^<]*)?<\|message\|>"
)


def _clean_gpt_oss_output(text: str) -> str:
    """
    Extract final channel content from GPT-OSS channel-based output.

    When reasoning parser is not enabled, this provides a fallback that
    extracts the 'final' channel content so the API response is usable.

    Handles both standard and extended format with constrain token:
        <|channel|>final<|message|>...
        <|channel|>final <|constrain|>JSON<|message|>...

    Args:
        text: Raw model output containing channel tokens.

    Returns:
        Extracted final content, or text with channel tokens stripped.
    """
    match = _FINAL_CHANNEL_RE.search(text)
    if match:
        content = text[match.end() :]
        # Strip trailing structural tokens (including <|constrain|>)
        content = re.sub(
            r"<\|start\|>|<\|end\|>|<\|channel\|>|<\|return\|>|<\|call\|>|<\|message\|>|<\|constrain\|>",
            "",
            content,
        )
        return content.strip()

    # No final channel — strip all channel/structural tokens (including constrain)
    cleaned = re.sub(
        r"<\|channel\|>[^<]*(?:<\|constrain\|>[^<]*)?<\|message\|>|<\|start\|>[^<]*|<\|return\|>|<\|call\|>|<\|constrain\|>[^<]*",
        "",
        text,
    )
    return cleaned.strip()


def clean_output_text(text: str) -> str:
    """
    Clean model output by removing special tokens.

    Keeps <think>...</think> blocks intact for reasoning models.
    Adds opening <think> tag if missing (happens when thinking is enabled
    in the prompt template but the tag is part of the prompt, not output).
    Handles GPT-OSS channel-based format as fallback when reasoning parser
    is not enabled.

    Args:
        text: Raw model output

    Returns:
        Cleaned text with special tokens removed
    """
    if not text:
        return text

    # GPT-OSS channel format — extract final content before general stripping
    if "<|channel|>" in text and "<|message|>" in text:
        text = _clean_gpt_oss_output(text)
        return text

    text = SPECIAL_TOKENS_PATTERN.sub("", text)
    text = text.strip()

    # Add opening <think> tag if response has closing but not opening
    # This happens when enable_thinking=True in the chat template
    if "</think>" in text and not text.lstrip().startswith("<think>"):
        text = "<think>" + text

    return text


# Pattern to match thinking blocks:
# - <think>...</think> (Qwen, DeepSeek, etc.)
# - <|channel>thought\n...<channel|> (Gemma 4)
THINK_PATTERN = re.compile(
    r"<think>[\s\S]*?</think>\s*"
    r"|<\|channel>thought\n[\s\S]*?<channel\|>\s*",
    re.DOTALL,
)


def strip_thinking_tags(text: str) -> str:
    """
    Remove <think>...</think> blocks from model output.

    Used when the client expects pure content (e.g., JSON) without
    reasoning blocks that would break parsing.

    Args:
        text: Model output that may contain thinking blocks

    Returns:
        Text with thinking blocks removed
    """
    if not text:
        return text
    return THINK_PATTERN.sub("", text).strip()


def extract_json_from_response(text: str) -> str:
    """
    Extract JSON object/array from model response that may contain reasoning text.

    Qwen3 and other reasoning models often output:
    "Let me think... reasoning text... {\"result\": 123}"

    This function extracts just the JSON part when present.

    Args:
        text: Model output that may contain text before/after JSON

    Returns:
        Extracted JSON string if found, otherwise original text
    """
    if not text:
        return text

    text = text.strip()

    # If already valid JSON, return as-is
    if (text.startswith("{") and text.endswith("}")) or (
        text.startswith("[") and text.endswith("]")
    ):
        return text

    # Try to find JSON object at the end of the response
    # Find the last { and match to the end
    last_brace = text.rfind("{")
    if last_brace != -1 and text.endswith("}"):
        potential_json = text[last_brace:]
        # Validate it's balanced
        depth = 0
        for char in potential_json:
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
        if depth == 0:
            return potential_json

    # Try to find JSON array at the end
    last_bracket = text.rfind("[")
    if last_bracket != -1 and text.endswith("]"):
        potential_json = text[last_bracket:]
        depth = 0
        for char in potential_json:
            if char == "[":
                depth += 1
            elif char == "]":
                depth -= 1
        if depth == 0:
            return potential_json

    # No JSON found, return original
    return text


# =============================================================================
# Streaming Tool Call Filter
# =============================================================================

# Safety cap for tool call buffer (bytes). If a tool call block never closes,
# the buffer is capped to prevent unbounded memory growth. In practice, the
# buffer is bounded by max_tokens (~100KB at 32768 tokens), but this cap
# protects against pathological cases.
_MAX_TOOL_BUFFER_BYTES = 1_048_576  # 1 MB

# Tags that delimit tool call blocks in streaming output.
# Content inside these tags should be suppressed during streaming because
# it will be re-emitted as structured tool_use blocks after parsing.
_TOOL_CALL_TAGS = [
    ("<minimax:tool_call>", "</minimax:tool_call>"),
    ("<tool_call>", "</tool_call>"),
    ("<function=", "</function>"),
    ("[TOOL_CALL]", "[/TOOL_CALL]"),
    ("[Calling tool", "\n"),  # Bracket-style tool calls: suppress until newline (covers both ]\n and bare \n)
]


class StreamingToolCallFilter:
    """Buffer streaming text to suppress tool call markup.

    Tool call XML (e.g. <minimax:tool_call>...</minimax:tool_call>) arrives
    split across multiple streaming deltas. This filter detects entry into a
    tool call block, suppresses all output until the block closes, and emits
    only non-tool-call text.

    The full unfiltered text is still accumulated separately for tool call
    parsing at stream end.
    """

    def __init__(self):
        self._buffer = ""
        self._in_block = False
        self._close_tag = ""
        # Longest open tag - used to determine how much buffer to hold back
        self._max_open_len = max(len(t[0]) for t in _TOOL_CALL_TAGS)

    def process(self, delta: str) -> str:
        """Process a streaming delta. Returns text to emit (may be empty)."""
        self._buffer += delta

        if self._in_block:
            return self._consume_block()
        else:
            return self._scan_for_open()

    def _scan_for_open(self) -> str:
        """Scan buffer for tool call open tags. Emit safe text."""
        # Check for complete open tags
        for open_tag, close_tag in _TOOL_CALL_TAGS:
            idx = self._buffer.find(open_tag)
            if idx >= 0:
                # Found an open tag - emit text before it, enter block mode
                emit = self._buffer[:idx]
                self._buffer = self._buffer[idx + len(open_tag) :]
                self._in_block = True
                self._close_tag = close_tag
                # Process remainder in case close tag is already in buffer
                after = self._consume_block()
                return emit + after

        # No complete open tag found. Check if buffer ends with a partial
        # match of any open tag - hold that back to avoid emitting a fragment.
        hold_back = 0
        for open_tag, _ in _TOOL_CALL_TAGS:
            for prefix_len in range(min(len(open_tag), len(self._buffer)), 0, -1):
                if self._buffer.endswith(open_tag[:prefix_len]):
                    hold_back = max(hold_back, prefix_len)
                    break

        if hold_back > 0:
            emit = self._buffer[:-hold_back]
            self._buffer = self._buffer[-hold_back:]
            return emit

        # No partial match - safe to emit everything
        emit = self._buffer
        self._buffer = ""
        return emit

    def _consume_block(self) -> str:
        """Consume content inside a tool call block. Returns empty string
        unless the block closes and there's text after it."""
        idx = self._buffer.find(self._close_tag)
        if idx >= 0:
            # Block closed - discard content up to and including close tag
            self._buffer = self._buffer[idx + len(self._close_tag) :]
            self._in_block = False
            self._close_tag = ""
            # Process remainder - might have more text or another tool call
            if self._buffer:
                return self._scan_for_open()
            return ""
        # Still inside block - suppress everything but cap buffer size
        if len(self._buffer) > _MAX_TOOL_BUFFER_BYTES:
            logger.warning(
                f"Tool call buffer exceeded {_MAX_TOOL_BUFFER_BYTES} bytes, "
                f"discarding and exiting block"
            )
            self._buffer = ""
            self._in_block = False
            self._close_tag = ""
        return ""

    def flush(self) -> str:
        """Flush remaining buffer at end of stream."""
        if self._in_block:
            # Unterminated tool call block - discard
            self._buffer = ""
            self._in_block = False
            return ""
        emit = self._buffer
        self._buffer = ""
        return emit


# =============================================================================
# Streaming Think Block Router
# =============================================================================


class StreamingThinkRouter:
    """Route <think>...</think> content to separate Anthropic thinking blocks.

    Instead of emitting thinking content as plain text (where it's
    indistinguishable from the response), this router yields tagged
    pieces that the streaming handler can emit as proper Anthropic
    content block types.

    Each call to process() returns a list of (block_type, text) tuples:
    - ("thinking", text) for content inside <think>...</think>
    - ("text", text) for content outside think blocks

    Args:
        start_in_thinking: If True, assume the model starts in thinking
            mode (e.g. MiniMax adds <think> to the generation prompt,
            so the tag never appears in the output stream).
    """

    def __init__(self, start_in_thinking: bool = False):
        self._buffer = ""
        self._in_think = start_in_thinking

    def process(self, delta: str) -> list[tuple[str, str]]:
        """Process a delta. Returns list of (block_type, text) pieces."""
        self._buffer += delta
        pieces = []
        self._extract_pieces(pieces)
        return pieces

    def _extract_pieces(self, pieces: list[tuple[str, str]]) -> None:
        """Extract all complete pieces from the buffer."""
        while True:
            if self._in_think:
                idx = self._buffer.find("</think>")
                if idx >= 0:
                    # Emit thinking content, exit think mode
                    thinking = self._buffer[:idx]
                    self._buffer = self._buffer[idx + len("</think>") :]
                    self._in_think = False
                    if thinking:
                        pieces.append(("thinking", thinking))
                    continue  # Process remainder
                else:
                    # Check for partial close tag at end
                    for plen in range(min(len("</think>"), len(self._buffer)), 0, -1):
                        if self._buffer.endswith("</think>"[:plen]):
                            # Hold back partial match
                            emit = self._buffer[:-plen]
                            self._buffer = self._buffer[-plen:]
                            if emit:
                                pieces.append(("thinking", emit))
                            return
                    # No partial match - emit all as thinking
                    if self._buffer:
                        pieces.append(("thinking", self._buffer))
                        self._buffer = ""
                    return
            else:
                idx = self._buffer.find("<think>")
                if idx >= 0:
                    # Emit text before tag, enter think mode
                    before = self._buffer[:idx]
                    self._buffer = self._buffer[idx + len("<think>") :]
                    self._in_think = True
                    if before:
                        pieces.append(("text", before))
                    continue  # Process remainder
                else:
                    # Check for partial open tag at end
                    for plen in range(min(len("<think>"), len(self._buffer)), 0, -1):
                        if self._buffer.endswith("<think>"[:plen]):
                            emit = self._buffer[:-plen]
                            self._buffer = self._buffer[-plen:]
                            if emit:
                                pieces.append(("text", emit))
                            return
                    # No partial match - emit all as text
                    if self._buffer:
                        pieces.append(("text", self._buffer))
                        self._buffer = ""
                    return

    def flush(self) -> list[tuple[str, str]]:
        """Flush remaining buffer at end of stream."""
        pieces = []
        if self._buffer:
            block_type = "thinking" if self._in_think else "text"
            pieces.append((block_type, self._buffer))
            self._buffer = ""
        self._in_think = False
        return pieces


# =============================================================================
# Model Detection
# =============================================================================

# Patterns that indicate a multimodal language model (MLLM/VLM)
MLLM_PATTERNS = [
    "-VL-",
    "-VL/",
    "VL-",  # Qwen-VL, Qwen2-VL, Qwen3-VL, etc.
    "llava",
    "LLaVA",  # LLaVA models
    "idefics",
    "Idefics",  # Idefics models
    "paligemma",
    "PaliGemma",  # PaliGemma
    "gemma-3",
    "gemma3",  # Gemma 3 (multimodal)
    "medgemma",
    "MedGemma",  # MedGemma (medical multimodal with SigLIP vision encoder)
    "pixtral",
    "Pixtral",  # Pixtral
    "molmo",
    "Molmo",  # Molmo
    "phi3-vision",
    "phi-3-vision",  # Phi-3 Vision
    "cogvlm",
    "CogVLM",  # CogVLM
    "internvl",
    "InternVL",  # InternVL
    "deepseek-vl",
    "DeepSeek-VL",  # DeepSeek-VL
]


def is_mllm_model(model_name: str) -> bool:
    """
    Check if model name indicates a multimodal language model.

    Args:
        model_name: HuggingFace model name or local path

    Returns:
        True if model is detected as MLLM/VLM
    """
    model_lower = model_name.lower()
    return any(pattern.lower() in model_lower for pattern in MLLM_PATTERNS)


# Backwards compatibility alias
is_vlm_model = is_mllm_model


# =============================================================================
# Multimodal Content Extraction
# =============================================================================


def _content_to_text(content) -> str:
    """Extract text from content that can be str, list[ContentPart], or None."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if hasattr(item, "model_dump"):
                item = item.model_dump(exclude_none=True)
            elif hasattr(item, "dict"):
                item = {k: v for k, v in item.dict().items() if v is not None}
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "\n".join(parts)
    return str(content)


def extract_multimodal_content(
    messages: list[Message],
    preserve_native_format: bool = False,
) -> tuple[list[dict], list[str], list[str]]:
    """
    Extract text content, images, and videos from OpenAI-format messages.

    Handles:
    - Simple text messages
    - Multimodal messages with images/videos
    - Tool call messages (assistant with tool_calls)
    - Tool response messages (role="tool")

    Args:
        messages: List of Message objects
        preserve_native_format: If True, preserve native tool message format
            (role="tool", tool_calls field) instead of converting to text.
            Required for models with native tool support in chat templates
            (e.g., Mistral, Llama 3+, DeepSeek V3).

    Returns:
        Tuple of (processed_messages, images, videos)
        - processed_messages: List of {"role": str, "content": str}
        - images: List of image URLs/paths/base64
        - videos: List of video URLs/paths/base64
    """
    processed_messages = []
    images = []
    videos = []

    for msg in messages:
        # Handle both dict and Pydantic model messages
        if isinstance(msg, dict):
            role = msg.get("role", "user")
            content = msg.get("content")
        else:
            role = msg.role
            content = msg.content

        # Handle tool response messages (role="tool")
        if role == "tool":
            if isinstance(msg, dict):
                tool_call_id = msg.get("tool_call_id", "") or ""
            else:
                tool_call_id = getattr(msg, "tool_call_id", None) or ""
            tool_content = content if content else ""

            if preserve_native_format:
                # Preserve native tool format for models that support it
                processed_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": tool_content,
                    }
                )
            else:
                # Convert to user role for models without native support
                processed_messages.append(
                    {
                        "role": "user",
                        "content": f"[Tool Result ({tool_call_id})]: {tool_content}",
                    }
                )
            continue

        # Handle assistant messages with tool_calls
        if isinstance(msg, dict):
            tool_calls = msg.get("tool_calls")
        else:
            tool_calls = getattr(msg, "tool_calls", None)

        if role == "assistant" and tool_calls:
            if preserve_native_format:
                # Preserve native tool_calls format
                tool_calls_list = []
                for tc in tool_calls:
                    if isinstance(tc, dict):
                        tc_copy = tc
                    elif hasattr(tc, "model_dump"):
                        tc_copy = tc.model_dump()
                    elif hasattr(tc, "dict"):
                        tc_copy = tc.dict()
                    else:
                        continue

                    # Chat templates (e.g. Qwen3) iterate arguments|items,
                    # but OpenAI API sends arguments as a JSON string.
                    # Parse it into a dict so the template can iterate it.
                    func = tc_copy.get("function") or {}
                    args = func.get("arguments")
                    if isinstance(args, str):
                        try:
                            import json

                            func["arguments"] = json.loads(args)
                        except (json.JSONDecodeError, ValueError):
                            pass

                    tool_calls_list.append(tc_copy)

                msg_dict = {"role": role, "content": _content_to_text(content)}
                if tool_calls_list:
                    msg_dict["tool_calls"] = tool_calls_list
                processed_messages.append(msg_dict)
            else:
                # Convert tool calls to text for models without native support
                tool_calls_text = []
                for tc in tool_calls:
                    if isinstance(tc, dict):
                        func = tc.get("function", {})
                        name = func.get("name", "unknown")
                        args = func.get("arguments", "{}")
                        tool_calls_text.append(f"[Calling tool: {name}({args})]")

                text = _content_to_text(content)
                if tool_calls_text:
                    text = (text + "\n" if text else "") + "\n".join(tool_calls_text)

                processed_messages.append({"role": role, "content": text})
            continue

        # Handle None content
        if content is None:
            processed_messages.append({"role": role, "content": ""})
            continue

        if isinstance(content, str):
            # Simple text message
            processed_messages.append({"role": role, "content": content})
        elif isinstance(content, list):
            # Multimodal message - extract text and media
            text_parts = []
            for item in content:
                # Handle both Pydantic models and dicts
                if hasattr(item, "model_dump"):
                    item = item.model_dump(exclude_none=True)
                elif hasattr(item, "dict"):
                    item = {k: v for k, v in item.dict().items() if v is not None}

                item_type = item.get("type", "")

                if item_type == "text":
                    text_parts.append(item.get("text", ""))

                elif item_type == "image_url":
                    img_url = item.get("image_url", {})
                    if isinstance(img_url, str):
                        images.append(img_url)
                    elif isinstance(img_url, dict):
                        images.append(img_url.get("url", ""))

                elif item_type == "image":
                    images.append(item.get("image", item.get("url", "")))

                elif item_type == "video":
                    videos.append(item.get("video", item.get("url", "")))

                elif item_type == "video_url":
                    vid_url = item.get("video_url", {})
                    if isinstance(vid_url, str):
                        videos.append(vid_url)
                    elif isinstance(vid_url, dict):
                        videos.append(vid_url.get("url", ""))

            # Combine text parts
            combined_text = "\n".join(text_parts) if text_parts else ""
            processed_messages.append({"role": role, "content": combined_text})
        else:
            # Unknown format, try to convert
            processed_messages.append({"role": role, "content": str(content)})

    return processed_messages, images, videos
