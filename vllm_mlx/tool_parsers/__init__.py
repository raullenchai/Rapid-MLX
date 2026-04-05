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
- seed_oss/seed/gpt_oss: Seed-OSS / GPT-OSS models (XML format)
- deepseek_v31/deepseek_r1_0528: DeepSeek V3.1 / R1-0528 models
- qwen3_coder_xml/qwen3_xml: Qwen3-Coder models (XML format)

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
from .deepseekv31_tool_parser import DeepSeekV31ToolParser
from .functionary_tool_parser import FunctionaryToolParser
from .gemma4_tool_parser import Gemma4ToolParser
from .glm47_tool_parser import Glm47ToolParser
from .granite_tool_parser import GraniteToolParser
from .harmony_tool_parser import HarmonyToolParser
from .hermes_tool_parser import HermesToolParser
from .kimi_tool_parser import KimiToolParser
from .llama_tool_parser import LlamaToolParser
from .minimax_tool_parser import MiniMaxToolParser
from .mistral_tool_parser import MistralToolParser
from .nemotron_tool_parser import NemotronToolParser
from .qwen3coder_tool_parser import Qwen3CoderToolParser
from .qwen_tool_parser import QwenToolParser
from .seed_oss_tool_parser import SeedOssToolParser
from .xlam_tool_parser import xLAMToolParser

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
    "Gemma4ToolParser",
    "Glm47ToolParser",
    "HarmonyToolParser",
    "MiniMaxToolParser",
    "SeedOssToolParser",
    "DeepSeekV31ToolParser",
    "Qwen3CoderToolParser",
]
