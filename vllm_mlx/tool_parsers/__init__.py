# SPDX-License-Identifier: Apache-2.0
"""
Tool call parsers for vllm-mlx.

This module provides tool call parsing functionality for various model formats.
Inspired by vLLM's tool parser architecture but simplified for MLX backend.

Available parsers:
- auto: Auto-detecting parser that tries all formats (default)
- mistral: Mistral models ([TOOL_CALLS] format)
- qwen/qwen3: Qwen models (<tool_call> and [Calling tool:] formats)
- llama/llama3/llama4: Llama models (<function=name> format)
- hermes/nous: Hermes/NousResearch models
- deepseek/deepseek_v3/deepseek_r1: DeepSeek models (unicode tokens)
- kimi/kimi_k2/moonshot: Kimi/Moonshot models
- granite/granite3: IBM Granite models
- nemotron/nemotron3: NVIDIA Nemotron models
- xlam: Salesforce xLAM models
- functionary/meetkai: MeetKai Functionary models
- glm47/glm4: GLM-4.7 and GLM-4.7-Flash models
- harmony/gpt-oss: GPT-OSS models (Harmony format with channels)

Usage:
    from vllm_mlx.tool_parsers import ToolParserManager

    # Get a parser by name
    parser_cls = ToolParserManager.get_tool_parser("mistral")
    parser = parser_cls(tokenizer)

    # Parse tool calls
    result = parser.extract_tool_calls(model_output)
    if result.tools_called:
        for tc in result.tool_calls:
            print(f"Tool: {tc['name']}, Args: {tc['arguments']}")

    # List available parsers
    print(ToolParserManager.list_registered())
"""

from .abstract_tool_parser import (
    ExtractedToolCallInformation,
    ToolParser,
    ToolParserManager,
)

# Import parsers to register them
from .auto_tool_parser import AutoToolParser
from .deepseek_tool_parser import DeepSeekToolParser
from .functionary_tool_parser import FunctionaryToolParser
from .granite_tool_parser import GraniteToolParser
from .hermes_tool_parser import HermesToolParser
from .kimi_tool_parser import KimiToolParser
from .llama_tool_parser import LlamaToolParser
from .mistral_tool_parser import MistralToolParser
from .nemotron_tool_parser import NemotronToolParser
from .qwen_tool_parser import QwenToolParser
from .xlam_tool_parser import xLAMToolParser
from .glm47_tool_parser import Glm47ToolParser
from .harmony_tool_parser import HarmonyToolParser
from .minimax_tool_parser import MiniMaxToolParser

__all__ = [
    # Base classes
    "ToolParser",
    "ToolParserManager",
    "ExtractedToolCallInformation",
    # Specific parsers
    "AutoToolParser",
    "MistralToolParser",
    "QwenToolParser",
    "LlamaToolParser",
    "HermesToolParser",
    "DeepSeekToolParser",
    "KimiToolParser",
    "GraniteToolParser",
    "NemotronToolParser",
    "xLAMToolParser",
    "FunctionaryToolParser",
    "Glm47ToolParser",
    "HarmonyToolParser",
    "MiniMaxToolParser",
]
