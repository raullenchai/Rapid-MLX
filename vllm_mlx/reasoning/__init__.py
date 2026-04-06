# SPDX-License-Identifier: Apache-2.0
"""
Reasoning parser module for vllm-mlx.

This module provides parsers for extracting reasoning/thinking content from
model outputs. Supports models like Qwen3, DeepSeek-R1, etc. that use special
tokens (e.g., <think>...</think>) to separate reasoning from final responses.

Usage:
    from vllm_mlx.reasoning import get_parser, list_parsers

    # Get a parser by name
    parser = get_parser("qwen3")()

    # Extract reasoning from complete output
    reasoning, content = parser.extract_reasoning(model_output)

    # For streaming
    parser.reset_state()
    for delta in stream:
        msg = parser.extract_reasoning_streaming(prev, curr, delta)
        if msg:
            # msg.reasoning and/or msg.content will be populated
            ...
"""

from .base import DeltaMessage, ReasoningParser
from .think_parser import BaseThinkingReasoningParser

# Parser registry
_REASONING_PARSERS: dict[str, type[ReasoningParser]] = {}


def register_parser(name: str, parser_class: type[ReasoningParser]) -> None:
    """
    Register a reasoning parser.

    Args:
        name: Name to register the parser under (e.g., "qwen3").
        parser_class: The parser class to register.
    """
    _REASONING_PARSERS[name] = parser_class


def get_parser(name: str) -> type[ReasoningParser]:
    """
    Get a reasoning parser class by name.

    Args:
        name: Name of the parser (e.g., "qwen3", "deepseek_r1").

    Returns:
        The parser class (not an instance).

    Raises:
        KeyError: If parser name is not found.
    """
    if name not in _REASONING_PARSERS:
        available = list(_REASONING_PARSERS.keys())
        raise KeyError(
            f"Reasoning parser '{name}' not found. Available parsers: {available}"
        )
    return _REASONING_PARSERS[name]


def list_parsers() -> list[str]:
    """
    List available parser names.

    Returns:
        List of registered parser names.
    """
    return list(_REASONING_PARSERS.keys())


def _register_builtin_parsers():
    """Register built-in parsers."""
    from .deepseek_r1_parser import DeepSeekR1ReasoningParser
    from .gpt_oss_parser import GptOssReasoningParser
    from .harmony_parser import HarmonyReasoningParser
    from .gemma4_parser import Gemma4ReasoningParser
    from .minimax_parser import MiniMaxReasoningParser
    from .qwen3_parser import Qwen3ReasoningParser

    register_parser("gemma4", Gemma4ReasoningParser)
    register_parser("qwen3", Qwen3ReasoningParser)
    register_parser("deepseek_r1", DeepSeekR1ReasoningParser)
    register_parser("gpt_oss", GptOssReasoningParser)
    register_parser("harmony", HarmonyReasoningParser)
    register_parser("minimax", MiniMaxReasoningParser)


# Register built-in parsers on module load
_register_builtin_parsers()


__all__ = [
    # Base classes
    "ReasoningParser",
    "DeltaMessage",
    "BaseThinkingReasoningParser",
    # Registry functions
    "register_parser",
    "get_parser",
    "list_parsers",
]
