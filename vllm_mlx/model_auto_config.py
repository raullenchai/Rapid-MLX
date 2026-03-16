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


# Model family patterns → optimal config.
# Order matters: first match wins. More specific patterns go first.
_MODEL_PATTERNS: list[tuple[re.Pattern, ModelConfig]] = [
    # DeepSeek (R1, V3) — before Qwen because DeepSeek-R1-Qwen3 distills exist
    (re.compile(r"deepseek", re.IGNORECASE), ModelConfig(
        tool_call_parser="hermes",
        reasoning_parser="deepseek_r1",
    )),
    # Qwen family (Qwen3, Qwen3.5, Qwen3-Coder)
    (re.compile(r"qwen3", re.IGNORECASE), ModelConfig(
        tool_call_parser="hermes",
        reasoning_parser="qwen3",
    )),
    # GLM family (GLM-4.5, GLM-4.7)
    (re.compile(r"glm[-_]?4", re.IGNORECASE), ModelConfig(
        tool_call_parser="glm47",
        reasoning_parser=None,
    )),
    # MiniMax M2.5
    (re.compile(r"minimax", re.IGNORECASE), ModelConfig(
        tool_call_parser="minimax",
        reasoning_parser="minimax",
    )),
    # GPT-OSS
    (re.compile(r"gpt[-_]?oss", re.IGNORECASE), ModelConfig(
        tool_call_parser="harmony",
        reasoning_parser="harmony",
    )),
    # Mistral / Devstral
    (re.compile(r"mistral|devstral", re.IGNORECASE), ModelConfig(
        tool_call_parser="hermes",
        reasoning_parser=None,
    )),
    # Gemma
    (re.compile(r"gemma", re.IGNORECASE), ModelConfig(
        tool_call_parser="hermes",
        reasoning_parser=None,
    )),
    # Hermes (fine-tuned Llama etc.)
    (re.compile(r"hermes", re.IGNORECASE), ModelConfig(
        tool_call_parser="hermes",
        reasoning_parser=None,
    )),
    # Llama
    (re.compile(r"llama", re.IGNORECASE), ModelConfig(
        tool_call_parser="hermes",
        reasoning_parser=None,
    )),
    # Phi
    (re.compile(r"phi[-_]?[34]", re.IGNORECASE), ModelConfig(
        tool_call_parser="hermes",
        reasoning_parser=None,
    )),
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
