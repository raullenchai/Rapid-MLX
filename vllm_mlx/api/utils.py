# SPDX-License-Identifier: Apache-2.0
"""
Utility functions for text processing and model detection.
"""

import json
import logging
import os
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

# Commentary-channel tool-call markers (both legacy and current forms).
# If ANY of these are present, the output carries tool-call structure
# that the harmony tool parser needs to see intact — bail out of
# stripping. Matches:
#   <|channel|>commentary to=functions.NAME ... <|message|>...<|call|>
#   to=functions.NAME<|channel|>commentary ... <|message|>...<|call|>
_COMMENTARY_TOOL_CALL_RE = re.compile(
    r"<\|channel\|>commentary\s+to=functions\."
    r"|"
    r"to=functions\.\w+<\|channel\|>commentary"
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
    # Tool-call structure must survive to the harmony tool parser:
    # if the model emitted ``<|channel|>commentary to=functions.X...<|call|>``
    # (which gpt-oss-20b does for every tool invocation), the parser needs
    # those structural tokens intact to extract the call. Stripping them
    # here drops the args into plain text and the parser returns 0 calls.
    # Same regression class as PR #436 but for the tool parser. Final
    # channel is unaffected because the route runs ``clean_output_text``
    # again after parsers run (chat.py / anthropic.py).
    if _COMMENTARY_TOOL_CALL_RE.search(text):
        return text

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
    Extract JSON object/array from model response, handling common wrapping.

    Models often wrap JSON in various ways:
    - Reasoning prefix: "Let me think... {json}"  (Qwen3)
    - Markdown code block: ```json\n{json}\n```    (Gemma 4, Llama)
    - Mixed: "Here's the result:\n```json\n{}\n```\nDone."
    - Plain JSON: {json}

    This is part of the output compensation layer — normalizes model
    output variations so downstream frameworks (PydanticAI, LangChain)
    see clean JSON regardless of model quirks.

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

    # Strip markdown code blocks: ```json\n{...}\n``` or ```\n{...}\n```
    # This is the most common wrapping pattern (Gemma 4, Llama 3.x)
    stripped = _strip_markdown_code_block(text)
    if stripped != text:
        return stripped

    # Try to find JSON object at the end of the response
    # Find the last { and match to the end
    last_brace = text.rfind("{")
    if last_brace != -1 and text.endswith("}"):
        potential_json = text[last_brace:]
        if _is_balanced(potential_json, "{", "}"):
            return potential_json

    # Try to find JSON array at the end
    last_bracket = text.rfind("[")
    if last_bracket != -1 and text.endswith("]"):
        potential_json = text[last_bracket:]
        if _is_balanced(potential_json, "[", "]"):
            return potential_json

    # No JSON found, return original
    return text


def _strip_markdown_code_block(text: str) -> str:
    """Strip markdown code block wrapping from text.

    Handles:
        ```json\n{...}\n```
        ```\n{...}\n```
        Text before ```json\n{...}\n``` text after
    """
    import re

    # Match ```json or ``` followed by content and closing ```
    pattern = re.compile(
        r"```(?:json|JSON)?\s*\n([\s\S]*?)\n\s*```",
    )
    match = pattern.search(text)
    if match:
        inner = match.group(1).strip()
        # Verify it looks like JSON
        if inner and (inner[0] in "{["):
            return inner
    return text


def _is_balanced(text: str, open_char: str, close_char: str) -> bool:
    """Check if brackets/braces are balanced."""
    depth = 0
    for char in text:
        if char == open_char:
            depth += 1
        elif char == close_char:
            depth -= 1
    return depth == 0


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
#
# This list is extensible — agent profiles can inject additional tags via
# register_tool_call_tag() or by passing extra_tags to StreamingToolCallFilter.
_TOOL_CALL_TAGS: list[tuple[str, str]] = [
    ("<minimax:tool_call>", "</minimax:tool_call>"),
    ("<tool_call>", "</tool_call>"),
    ("<function=", "</function>"),
    ("[TOOL_CALL]", "[/TOOL_CALL]"),
    (
        "[Calling tool",
        "\n",
    ),  # Bracket-style tool calls: suppress until newline (covers both ]\n and bare \n)
]


def register_tool_call_tag(open_tag: str, close_tag: str) -> bool:
    """Register an additional tool call tag pair for streaming suppression.

    Use this to extend the filter with agent-specific or model-specific
    markup patterns that should be suppressed during streaming.

    Args:
        open_tag: Opening tag (e.g. "<my_tool>")
        close_tag: Closing tag (e.g. "</my_tool>")

    Returns:
        True if the tag was added, False if it was already registered.
    """
    pair = (open_tag, close_tag)
    if pair not in _TOOL_CALL_TAGS:
        _TOOL_CALL_TAGS.append(pair)
        return True
    return False


def get_tool_call_tags() -> list[tuple[str, str]]:
    """Get the current list of tool call tag pairs (read-only copy)."""
    return list(_TOOL_CALL_TAGS)


class StreamingToolCallFilter:
    """Buffer streaming text to suppress tool call markup.

    Tool call XML (e.g. <minimax:tool_call>...</minimax:tool_call>) arrives
    split across multiple streaming deltas. This filter detects entry into a
    tool call block, suppresses all output until the block closes, and emits
    only non-tool-call text.

    The full unfiltered text is still accumulated separately for tool call
    parsing at stream end.

    Args:
        extra_tags: Additional (open, close) tag pairs to suppress, beyond
                    the global _TOOL_CALL_TAGS. Useful for per-request or
                    per-agent customization without mutating global state.
    """

    def __init__(self, extra_tags: list[tuple[str, str]] | None = None):
        self._buffer = ""
        self._in_block = False
        self._close_tag = ""
        # Merge global tags with per-instance extras
        self._tags = _TOOL_CALL_TAGS
        if extra_tags:
            self._tags = _TOOL_CALL_TAGS + [
                t for t in extra_tags if t not in _TOOL_CALL_TAGS
            ]
        # Longest open tag - used to determine how much buffer to hold back
        self._max_open_len = max(len(t[0]) for t in self._tags)

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
        for open_tag, close_tag in self._tags:
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
        for open_tag, _ in self._tags:
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


def _try_read_hub_config_json(model_name: str) -> dict | None:
    """Read a cached config.json for an HF repo ID without going online.

    Looks up the repo in the local ``huggingface_hub`` cache via
    ``try_to_load_from_cache``. If the model has already been downloaded
    (or even just had its config fetched), the file is on disk and we
    can read it authoritatively. If nothing is cached, returns None —
    the caller falls back to legacy substring matching rather than
    making a network call. Never raises; never reaches the network.

    Only runs for inputs that look like repo IDs (``owner/name``), so
    arbitrary strings and local-path lookalikes can't accidentally
    trigger a cache lookup.
    """
    if not isinstance(model_name, str) or "/" not in model_name:
        return None
    if model_name.startswith(("/", "./", "../", "~")):
        return None

    try:
        from huggingface_hub import _CACHED_NO_EXIST, try_to_load_from_cache
    except ImportError:
        return None

    try:
        cached = try_to_load_from_cache(model_name, "config.json")
    except Exception:
        return None
    if cached is None or cached is _CACHED_NO_EXIST:
        return None
    if not isinstance(cached, (str, os.PathLike)):
        return None

    try:
        config_path = Path(cached)
        if not config_path.is_file():
            return None
        if config_path.stat().st_size > _MAX_CONFIG_JSON_BYTES:
            return None
        with config_path.open(encoding="utf-8") as f:
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


# Tensor-name prefixes that indicate actual vision/audio weights live in
# the checkpoint. Used to distinguish text-only forks of multimodal
# architectures (where config.json declares ``vision_config`` but the
# safetensors ship no ``vision_tower.*`` tensors) from genuine VLMs.
# See issue #393 (Qwen3.6-35B-A3B text-only fork misrouted into MLLM
# batched path because Qwen3.5MoeForConditionalGeneration declares
# vision_config even when the user's safetensors are language-only).
_MULTIMODAL_TENSOR_PREFIXES = (
    "vision_tower",
    "vision_model",
    "visual.",
    "audio_tower",
    "audio_model",
    "mm_projector",
    "patch_embed.",
)


def _local_checkpoint_has_multimodal_weights(model_dir: Path) -> bool | None:
    """Probe a local model dir for actual vision/audio tensor weights.

    Returns:
        ``True`` if at least one tensor name in ``model.safetensors.index.json``
        matches a known multimodal prefix (``vision_tower``, ``visual.``,
        ``mm_projector``, …).
        ``False`` if the index is present, readable, and contains zero
        such tensors — meaning the checkpoint is text-only despite
        whatever ``config.json`` declares.
        ``None`` when we can't authoritatively answer (no index file,
        single-file safetensors, unreadable index). Caller should fall
        back to the config-based decision rather than flipping it.

    The single-file-safetensors branch returns ``None`` rather than
    parsing the file header ourselves — the cost of a wrong False (text
    routing for a real VLM → crash later on first image input) is much
    larger than the cost of a wrong True (current behavior, the bug we
    want to fix only for the multi-file case Tylast reported in #393).
    """
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        # No sharded index. Caller must rely on config-based detection.
        return None
    try:
        if index_path.stat().st_size > _MAX_CONFIG_JSON_BYTES:
            # Unreasonably large index — bail rather than guess.
            return None
        with index_path.open(encoding="utf-8") as f:
            idx = json.load(f)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError, ValueError):
        return None
    weight_map = idx.get("weight_map")
    if not isinstance(weight_map, dict):
        return None
    for tensor_name in weight_map:
        if not isinstance(tensor_name, str):
            continue
        for prefix in _MULTIMODAL_TENSOR_PREFIXES:
            if prefix in tensor_name:
                return True
    return False


def _check_legacy_string_patterns(model_name: str) -> bool:
    """Validation 1: substring match of MLLM_PATTERNS against the input string.

    Kept for HF repo IDs (where no local config.json is reachable) and as
    a fallback when config.json cannot be read.
    """
    model_lower = model_name.lower()
    return any(pattern.lower() in model_lower for pattern in MLLM_PATTERNS)


def is_mllm_model(model_name: str) -> bool:
    """Check if a model name or path indicates a multimodal language model.

    Four complementary validations are run, in order:

    1. Local config.json inspection: when ``model_name`` resolves to a
       local directory containing a readable config.json, inspect the
       model's own metadata (``architectures`` field, ``vision_config``,
       ``audio_config``, etc.).

    2. Local weights-presence override: when (1) said "VLM" but the
       directory ships a ``model.safetensors.index.json`` with NO
       multimodal tensors (``vision_tower``, ``visual.``, ``mm_projector``,
       …), the checkpoint is a text-only fork of a multimodal
       architecture. Flip the answer to False so the model loads
       through the text path instead of crashing in the MLLM batched
       engine on a missing vision tower. Fixes #393 (Qwen3.6-35B-A3B
       text-only fork — config.json declares ``vision_config`` because
       the base ``Qwen3_5MoeForConditionalGeneration`` architecture is
       multimodal-capable, but the user's safetensors only contain
       language tensors).

    3. Legacy substring match against ``MLLM_PATTERNS``: when no local
       config is reachable, the historical name-based heuristic decides.

    4. Cached-config override (false-positive correction): when the
       legacy substring says "MLLM" but the repo's own config (already
       in the local HF cache from a prior download or warmup call)
       disagrees (e.g. ``mlx-community/gemma-3-1b-it-4bit`` matches the
       ``gemma-3`` substring yet ships as ``Gemma3ForCausalLM`` with no
       ``vision_config``), the cached config wins and the result flips
       to False. The cache lookup is purely local — no network call,
       so tests don't slow down or flake on HF outages. We deliberately
       only override the True → False direction: the False direction is
       preserved as-is so existing text-routed models with vision-capable
       architectures keep their routing.

    Args:
        model_name: HuggingFace repo ID or local filesystem path.

    Returns:
        True if the model is detected as multimodal (MLLM/VLM).
    """
    config = _try_read_config_json(model_name)
    if config is not None:
        if not _config_indicates_vlm(config):
            return False
        # Config says VLM. For local directories, verify the checkpoint
        # actually carries vision/audio tensors — text-only forks of
        # multimodal architectures (#393) ship a config that declares
        # vision_config but no vision_tower weights, and routing them
        # to the MLLM batched engine crashes at first request.
        try:
            model_dir = Path(model_name)
        except (TypeError, ValueError):
            return True
        if model_dir.is_dir():
            verdict = _local_checkpoint_has_multimodal_weights(model_dir)
            if verdict is False:
                return False
            # verdict is True or None (unable to authoritatively decide);
            # trust the config in both cases — wrong-True here means an
            # unnecessary text-path attempt that returns a clear error,
            # vs wrong-False which would corrupt every request silently.
        return True

    legacy = _check_legacy_string_patterns(model_name)
    if not legacy:
        return False

    # Substring says "MLLM" — verify via the repo's own config and let it
    # override a name-based false positive (e.g. text-only Gemma 3 1B).
    hub_config = _try_read_hub_config_json(model_name)
    if hub_config is not None and not _config_indicates_vlm(hub_config):
        return False
    return True


# Backwards compatibility alias
is_vlm_model = is_mllm_model


def decode_inline_tool_call_arguments(messages: list[dict]) -> None:
    """Decode `tool_calls[].function.arguments` from JSON string to dict in-place.

    The OpenAI API serializes tool-call arguments as a JSON-encoded string.
    Some chat templates (GLM-4.6V, Qwen3 MLLM variants) iterate the arguments
    dict via `.items()`/`|items` and crash on a string. The non-MLLM path
    handles this inside `extract_multimodal_content()`; the MLLM branch
    bypasses that helper, so callers in the MLLM path call this directly.

    Mutates `messages` in-place. Malformed JSON is left untouched.
    """
    for msg in messages:
        for tc in msg.get("tool_calls") or []:
            func = tc.get("function") or {}
            args = func.get("arguments")
            if isinstance(args, str):
                try:
                    func["arguments"] = json.loads(args)
                except (json.JSONDecodeError, ValueError):
                    pass


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
