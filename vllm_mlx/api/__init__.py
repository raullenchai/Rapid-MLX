# SPDX-License-Identifier: Apache-2.0
"""
API models, utilities, and tool calling support for vllm-mlx.

This module provides shared components used by the server:
- Pydantic models for OpenAI-compatible API
- Utility functions for text processing and model detection
- Tool calling parsing and conversion
"""

from .models import (
    AssistantMessage,
    AudioSeparationRequest,
    AudioSpeechRequest,
    # Audio
    AudioTranscriptionRequest,
    AudioTranscriptionResponse,
    AudioUrl,
    ChatCompletionChoice,
    # Chat requests/responses
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionChoice,
    # Completion requests/responses
    CompletionRequest,
    CompletionResponse,
    ContentPart,
    EmbeddingData,
    # Embeddings
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingUsage,
    # Tool calling
    FunctionCall,
    # Content types
    ImageUrl,
    MCPExecuteRequest,
    MCPExecuteResponse,
    MCPServerInfo,
    MCPServersResponse,
    # MCP
    MCPToolInfo,
    MCPToolsResponse,
    Message,
    ModelInfo,
    ModelsResponse,
    # Structured output
    ResponseFormat,
    ResponseFormatJsonSchema,
    ToolCall,
    ToolDefinition,
    # Common
    Usage,
    VideoUrl,
)
from .tool_calling import (
    build_json_system_prompt,
    convert_tools_for_template,
    extract_json_from_text,
    # Structured output
    parse_json_output,
    parse_tool_calls,
    validate_json_schema,
)
from .utils import (
    MLLM_PATTERNS,
    SPECIAL_TOKENS_PATTERN,
    StreamingThinkRouter,
    StreamingToolCallFilter,
    clean_output_text,
    extract_multimodal_content,
    get_tool_call_tags,
    is_mllm_model,
    is_vlm_model,
    register_tool_call_tag,
    sanitize_output,
    strip_special_tokens,
)

__all__ = [
    # Models
    "ImageUrl",
    "VideoUrl",
    "AudioUrl",
    "ContentPart",
    "Message",
    "FunctionCall",
    "ToolCall",
    "ToolDefinition",
    "ResponseFormat",
    "ResponseFormatJsonSchema",
    "ChatCompletionRequest",
    "ChatCompletionChoice",
    "ChatCompletionResponse",
    "AssistantMessage",
    "CompletionRequest",
    "CompletionChoice",
    "CompletionResponse",
    "Usage",
    "ModelInfo",
    "ModelsResponse",
    "MCPToolInfo",
    "MCPToolsResponse",
    "MCPServerInfo",
    "MCPServersResponse",
    "MCPExecuteRequest",
    "MCPExecuteResponse",
    # Audio
    "AudioTranscriptionRequest",
    "AudioTranscriptionResponse",
    "AudioSpeechRequest",
    "AudioSeparationRequest",
    # Embeddings
    "EmbeddingRequest",
    "EmbeddingData",
    "EmbeddingUsage",
    "EmbeddingResponse",
    # Utils
    "clean_output_text",
    "is_mllm_model",
    "is_vlm_model",
    "extract_multimodal_content",
    "MLLM_PATTERNS",
    "SPECIAL_TOKENS_PATTERN",
    "sanitize_output",
    "strip_special_tokens",
    "StreamingToolCallFilter",
    "StreamingThinkRouter",
    "register_tool_call_tag",
    "get_tool_call_tags",
    # Tool calling
    "parse_tool_calls",
    "convert_tools_for_template",
    # Structured output
    "parse_json_output",
    "validate_json_schema",
    "extract_json_from_text",
    "build_json_system_prompt",
]
