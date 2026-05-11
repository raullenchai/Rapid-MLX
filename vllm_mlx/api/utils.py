# SPDX-License-Identifier: Apache-2.0
"""
Utility functions for text processing and model detection.
"""

import json
import logging
import re
from pathlib import Path

from .models import Message

logger = logging.getLogger(__name__)

# =============================================================================
# Special Token Patterns
# =============================================================================

# Pattern to match special tokens that should be removed from output
# Keeps <think>...</think> blocks intact for reasoning models
SPECIAL_TOKENS_PATTERN = re.compile(
    r"<\|im_end\|>|<\|im_start\|>|<\|endoftext\|>|"
    r"<\|end\|>|<\|eot_id\|>|<\|start_header_id\|>|<\|end_header_id\|>|"
    r"<\|channel\|>|<\|message\|>|<\|start\|>|<\|return\|>|<\|call\|>|<\|constrain\|>|"
    r"</s>|<s>|<pad>|\[PAD\]|\[SEP\]|\[CLS\]|"
    r"\[e~\[|\]~b\][a-z]*|\]~!b\[|"
    r"</?tool_call_reasoning>"
)


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
    ("<|tool_call>", "<tool_call|>"),
    ("[TOOL_CALL]", "[/TOOL_CALL]"),
    ("[Calling tool", "]\n"),  # Qwen3 bracket-style: [Calling tool: func({...})]\n
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
    "gemma-4",
    "gemma4",  # Gemma 4 (multimodal: vision + audio)
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
    "Qwen3.5-",
    "qwen3_5",  # Qwen3.5 MoE (natively multimodal, hybrid ArraysCache+KVCache)
]


# Config.json keys that, when present, indicate a multimodal model.
_VLM_CONFIG_KEYS = (
    "vision_config",
    "audio_config",
    "vision_tower",
    "mm_vision_tower",
    "image_token_id",
    "image_token_index",
    "audio_token_id",
    "audio_token_index",
)

# Substrings (case-insensitive) inside `architectures` entries that identify VLMs.
# Covers both ForConditionalGeneration VLMs (Qwen2VL, LLaVA, PaliGemma, Mllama, etc.)
# and the few VLMs that use ForCausalLM (Phi3V, Molmo, CogVLM, InternVL).
_VLM_ARCHITECTURE_KEYWORDS = (
    "VLForCondition",
    "VLForCausal",
    "VisionForCondition",
    "VisionForCausal",
    "MultiModalityCausalLM",
    "Llava",
    "Idefics",
    "PaliGemma",
    "Pixtral",
    "Molmo",
    "Phi3V",
    "Phi4V",
    "CogVLM",
    "InternVL",
    "DeepseekVL",
    "Mllama",
    "Gemma3ForConditional",
    "Gemma4ForConditional",
)

# Defensive cap on config.json size to bound parsing cost.
_MAX_CONFIG_JSON_BYTES = 1 * 1024 * 1024


def _try_read_config_json(name_or_path: str) -> dict | None:
    """Read config.json from a local model directory.

    Returns None when the input is not a local directory, the directory has
    no config.json, the file is too large, or it cannot be parsed.
    """
    try:
        candidate = Path(name_or_path)
    except (TypeError, ValueError):
        return None

    if not candidate.is_dir():
        return None

    config_path = candidate / "config.json"
    if not config_path.is_file():
        return None

    try:
        if config_path.stat().st_size > _MAX_CONFIG_JSON_BYTES:
            return None
        with config_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError, ValueError):
        return None

    return data if isinstance(data, dict) else None


def _config_indicates_vlm(config: dict) -> bool:
    """Inspect a parsed config.json dict for multimodal markers."""
    archs = config.get("architectures") or []
    if isinstance(archs, list):
        for arch in archs:
            if not isinstance(arch, str):
                continue
            arch_lower = arch.lower()
            for keyword in _VLM_ARCHITECTURE_KEYWORDS:
                if keyword.lower() in arch_lower:
                    return True

    for key in _VLM_CONFIG_KEYS:
        if key in config:
            return True

    return False


def _check_legacy_string_patterns(model_name: str) -> bool:
    """Validation 1: substring match of MLLM_PATTERNS against the input string.

    Kept for HF repo IDs (where no local config.json is reachable) and as
    a fallback when config.json cannot be read.
    """
    model_lower = model_name.lower()
    for pattern in MLLM_PATTERNS:
        if pattern.lower() in model_lower:
            return True
    return False


def is_mllm_model(model_name: str) -> bool:
    """Check if a model name or path indicates a multimodal language model.

    Two complementary validations are run:

    1. config.json inspection: when ``model_name`` resolves to a local
       directory containing a readable config.json, inspect the model's
       own metadata (``architectures`` field, ``vision_config``,
       ``audio_config``, etc.). Authoritative when available because it
       reflects what the model actually is, not how it is named on disk.

    2. Legacy substring match against ``MLLM_PATTERNS``: applied when no
       config.json is reachable (e.g., a HuggingFace repo ID before the
       weights are downloaded). Preserves the historical behaviour.

    Args:
        model_name: HuggingFace repo ID or local filesystem path.

    Returns:
        True if the model is detected as multimodal (MLLM/VLM).
    """
    config = _try_read_config_json(model_name)
    if config is not None:
        return _config_indicates_vlm(config)
    return _check_legacy_string_patterns(model_name)


# Backwards compatibility alias
is_vlm_model = is_mllm_model


# =============================================================================
# Media Content Detection
# =============================================================================

MEDIA_CONTENT_TYPES = frozenset(
    {
        "image_url",
        "video_url",
        "audio_url",
        "image",
        "video",
        "audio",
    }
)


def has_media_content(messages: list) -> bool:
    """Check if any message contains media content (images, video, audio).

    Handles both plain dicts (``msg.get("content")``) and Pydantic-style
    objects (``msg.content``) so it works in both engine and server contexts.
    """
    for msg in messages:
        content = (
            msg.get("content")
            if isinstance(msg, dict)
            else getattr(msg, "content", None)
        )
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    part_type = part.get("type")
                else:
                    part_type = getattr(part, "type", None)
                if part_type in MEDIA_CONTENT_TYPES:
                    return True
    return False


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
) -> tuple[list[dict], list[str], list[str], list[str]]:
    """
    Extract text content, images, videos, and audio from OpenAI-format messages.

    Handles:
    - Simple text messages
    - Multimodal messages with images/videos/audio
    - Tool call messages (assistant with tool_calls)
    - Tool response messages (role="tool")

    Args:
        messages: List of Message objects
        preserve_native_format: If True, preserve native tool message format
            (role="tool", tool_calls field) instead of converting to text.
            Required for models with native tool support in chat templates
            (e.g., Mistral, Llama 3+, DeepSeek V3).

    Returns:
        Tuple of (processed_messages, images, videos, audios)
        - processed_messages: List of {"role": str, "content": str}
        - images: List of image URLs/paths/base64
        - videos: List of video URLs/paths/base64
        - audios: List of audio URLs/paths/base64
    """
    processed_messages = []
    images = []
    videos = []
    audios = []

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

                elif item_type == "audio":
                    audios.append(item.get("audio", item.get("url", "")))

                elif item_type == "audio_url":
                    audio_url = item.get("audio_url", {})
                    if isinstance(audio_url, str):
                        audios.append(audio_url)
                    elif isinstance(audio_url, dict):
                        audios.append(audio_url.get("url", ""))

            # Combine text parts
            combined_text = "\n".join(text_parts) if text_parts else ""
            processed_messages.append({"role": role, "content": combined_text})
        else:
            # Unknown format, try to convert
            processed_messages.append({"role": role, "content": str(content)})

    return processed_messages, images, videos, audios
