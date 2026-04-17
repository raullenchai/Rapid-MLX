"""Auto-detect optimal parser configuration from model name/path.

When users don't specify --tool-call-parser or --reasoning-parser,
this module infers the best configuration from the model name pattern.
"""

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Auto-detected parser configuration for a model family."""

    tool_call_parser: str | None = None
    reasoning_parser: str | None = None
    default_max_tokens: int | None = None  # Per-model default when user omits max_tokens


# Model family patterns → optimal config.
# Order matters: first match wins. More specific patterns go first.
_MODEL_PATTERNS: list[tuple[re.Pattern, ModelConfig]] = [
    # DeepSeek V3.1 / R1-0528 — dedicated parser, before generic deepseek
    (
        re.compile(r"deepseek.*(v3\.1|r1[-_]?0528)", re.IGNORECASE),
        ModelConfig(
            tool_call_parser="deepseek_v31",
            reasoning_parser="deepseek_r1",
        ),
    ),
    # DeepSeek R1 (non-0528) — has reasoning
    (
        re.compile(r"deepseek.*r1", re.IGNORECASE),
        ModelConfig(
            tool_call_parser="deepseek",
            reasoning_parser="deepseek_r1",
        ),
    ),
    # DeepSeek (V3, V2.5, etc.) — no reasoning parser
    (
        re.compile(r"deepseek", re.IGNORECASE),
        ModelConfig(
            tool_call_parser="deepseek",
            reasoning_parser=None,
        ),
    ),
    # Qwopus (Qwen3.5 distilled with Claude Opus reasoning)
    (
        re.compile(r"qwopus", re.IGNORECASE),
        ModelConfig(
            tool_call_parser="hermes",
            reasoning_parser="qwen3",
        ),
    ),
    # Qwen3.6 — XML tool format, before generic Qwen3
    (
        re.compile(r"qwen3\.6", re.IGNORECASE),
        ModelConfig(
            tool_call_parser="qwen3_coder_xml",
            reasoning_parser="qwen3",
        ),
    ),
    # Qwen3-Coder — before generic Qwen3 (non-thinking, no reasoning parser)
    (
        re.compile(r"qwen3[-_]?coder", re.IGNORECASE),
        ModelConfig(
            tool_call_parser="hermes",
            reasoning_parser=None,
        ),
    ),
    # Qwen family (Qwen3, Qwen3.5)
    (
        re.compile(r"qwen3", re.IGNORECASE),
        ModelConfig(
            tool_call_parser="hermes",
            reasoning_parser="qwen3",
        ),
    ),
    # GLM family (GLM-4.5, GLM-4.7)
    (
        re.compile(r"glm[-_]?4", re.IGNORECASE),
        ModelConfig(
            tool_call_parser="glm47",
            reasoning_parser=None,
        ),
    ),
    # MiniMax M2.5
    (
        re.compile(r"minimax", re.IGNORECASE),
        ModelConfig(
            tool_call_parser="minimax",
            reasoning_parser="minimax",
        ),
    ),
    # GPT-OSS
    (
        re.compile(r"gpt[-_]?oss", re.IGNORECASE),
        ModelConfig(
            tool_call_parser="harmony",
            reasoning_parser="harmony",
        ),
    ),
    # Kimi
    (
        re.compile(r"kimi", re.IGNORECASE),
        ModelConfig(
            tool_call_parser="kimi",
            reasoning_parser=None,
        ),
    ),
    # Mistral / Devstral
    (
        re.compile(r"mistral|devstral", re.IGNORECASE),
        ModelConfig(
            tool_call_parser="hermes",
            reasoning_parser=None,
        ),
    ),
    # Gemma 4 (native tool format)
    (
        re.compile(r"gemma[-_]?4", re.IGNORECASE),
        ModelConfig(
            tool_call_parser="gemma4",
            reasoning_parser="gemma4",
        ),
    ),
    # Gemma 2/3 (hermes format)
    (
        re.compile(r"gemma", re.IGNORECASE),
        ModelConfig(
            tool_call_parser="hermes",
            reasoning_parser=None,
        ),
    ),
    # Hermes (fine-tuned Llama etc.)
    (
        re.compile(r"hermes", re.IGNORECASE),
        ModelConfig(
            tool_call_parser="hermes",
            reasoning_parser=None,
        ),
    ),
    # Llama
    (
        re.compile(r"llama", re.IGNORECASE),
        ModelConfig(
            tool_call_parser="llama",
            reasoning_parser=None,
        ),
    ),
    # Phi
    (
        re.compile(r"phi[-_]?[34]", re.IGNORECASE),
        ModelConfig(
            tool_call_parser="hermes",
            reasoning_parser=None,
        ),
    ),
]


def detect_model_config(model_path: str) -> ModelConfig | None:
    """Detect optimal parser config from model name/path.

    Args:
        model_path: Model name or path (e.g. "mlx-community/Qwen3.5-9B-4bit")

    Returns:
        ModelConfig if a pattern matches, None otherwise.
    """
    for pattern, config in _MODEL_PATTERNS:
        if pattern.search(model_path):
            logger.info(
                f"Auto-detected model family '{pattern.pattern}' → "
                f"tool_call_parser={config.tool_call_parser}, "
                f"reasoning_parser={config.reasoning_parser}"
            )
            return config
    return None
