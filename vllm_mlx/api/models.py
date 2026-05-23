# SPDX-License-Identifier: Apache-2.0
"""
Pydantic models for OpenAI-compatible API.

These models define the request and response schemas for:
- Chat completions
- Text completions
- Tool calling
- MCP (Model Context Protocol) integration
"""

import time
import uuid
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, model_validator

# =============================================================================
# Content Types (for multimodal messages)
# =============================================================================


class ImageUrl(BaseModel):
    """Image URL with optional detail level."""

    url: str
    detail: str | None = None


class VideoUrl(BaseModel):
    """Video URL."""

    url: str


class AudioUrl(BaseModel):
    """Audio URL for audio content."""

    url: str


class ContentPart(BaseModel):
    """
    A part of a multimodal message content.

    Supports:
    - text: Plain text content
    - image_url: Image from URL or base64
    - video: Video from local path
    - video_url: Video from URL or base64
    - audio_url: Audio from URL or base64
    """

    type: str  # "text", "image_url", "video", "video_url", "audio_url"
    text: str | None = None
    image_url: ImageUrl | dict | str | None = None
    video: str | None = None
    video_url: VideoUrl | dict | str | None = None
    audio_url: AudioUrl | dict | str | None = None


# =============================================================================
# Messages
# =============================================================================


class Message(BaseModel):
    """
    A message in a chat conversation.

    Supports:
    - Simple text messages (role + content string)
    - Multimodal messages (role + content list with text/images/videos)
    - Tool call messages (assistant with tool_calls)
    - Tool response messages (role="tool" with tool_call_id)
    """

    role: str
    content: str | list[ContentPart] | list[dict] | None = None
    # For assistant messages with tool calls
    tool_calls: list[dict] | None = None
    # For tool response messages (role="tool")
    tool_call_id: str | None = None


# =============================================================================
# Tool Calling
# =============================================================================


class FunctionCall(BaseModel):
    """A function call with name and arguments."""

    name: str
    arguments: str  # JSON string


class ToolCall(BaseModel):
    """A tool call from the model."""

    id: str
    type: str = "function"
    function: FunctionCall


class ToolDefinition(BaseModel):
    """Definition of a tool that can be called by the model."""

    type: str = "function"
    function: dict


# =============================================================================
# Structured Output (JSON Schema)
# =============================================================================


class ResponseFormatJsonSchema(BaseModel):
    """JSON Schema definition for structured output."""

    name: str
    description: str | None = None
    schema_: dict = Field(alias="schema")  # JSON Schema specification
    strict: bool | None = False

    class Config:
        populate_by_name = True


class ResponseFormat(BaseModel):
    """
    Response format specification for structured output.

    Supports:
    - "text": Default text output (no structure enforcement)
    - "json_object": Forces valid JSON output
    - "json_schema": Forces JSON matching a specific schema
    """

    type: str = "text"  # "text", "json_object", "json_schema"
    json_schema: ResponseFormatJsonSchema | None = None


# =============================================================================
# Logprobs
# =============================================================================


class TopLogProb(BaseModel):
    """A top log probability for a token."""

    token: str
    logprob: float
    bytes: list[int] | None = None


class TokenLogProb(BaseModel):
    """Log probability information for a single token."""

    token: str
    logprob: float
    bytes: list[int] | None = None
    top_logprobs: list[TopLogProb] = []


class ChoiceLogProbs(BaseModel):
    """Log probability information for a choice."""

    content: list[TokenLogProb] | None = None


# =============================================================================
# Chat Completion
# =============================================================================


class StreamOptions(BaseModel):
    """Options for streaming responses."""

    include_usage: bool = False  # Include usage stats in final chunk


class ChatCompletionRequest(BaseModel):
    """Request for chat completion."""

    model: str = "default"
    messages: list[Message]
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    # OpenAI-canonical token cap since Sept 2024 (preferred over max_tokens for
    # reasoning models; newer SDKs >=1.45 send only this field). Normalized to
    # max_tokens by a model_validator so all downstream code keeps reading the
    # single max_tokens field.
    max_completion_tokens: int | None = None
    stream: bool = False
    stream_options: StreamOptions | None = (
        None  # Streaming options (include_usage, etc.)
    )
    stop: list[str] | None = None
    # Extended OpenAI-compatible sampling parameters. Without these declared,
    # Pydantic drops them on parse (#355). top_k / min_p flow through to the
    # mlx-lm sampler; repetition_penalty / presence_penalty / frequency_penalty
    # flow through to mlx-lm's make_logits_processors().
    top_k: int | None = None
    min_p: float | None = None
    repetition_penalty: float | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    # Tool calling
    tools: list[ToolDefinition] | None = None
    tool_choice: str | dict | None = None  # "auto", "none", or specific tool
    # OpenAI extended spec — declared so Pydantic stops silently dropping it.
    # When set to False, the route caps the parsed ``tool_calls`` list at
    # length 1 in the response. Default None == True (model may emit
    # multiple). Cannot rely on decoder-level enforcement; this is a
    # post-generation truncation (the only reliable lever absent FSM
    # constraints — see PR #132 / #442 for the decoder-level path).
    parallel_tool_calls: bool | None = None
    # Legacy OpenAI tool-calling shape (pre-1.0 SDK + LangChain compat layers).
    # When set and the modern ``tools``/``tool_choice`` slots are empty, the
    # post-init validator below normalizes them to the modern equivalent so
    # downstream code keeps reading a single shape. Declared so Pydantic
    # stops silently dropping them (same blind-spot family as #355 /
    # #459 / #464). If a client supplies BOTH shapes, modern wins —
    # OpenAI's documented deprecation behavior — and the legacy slots are
    # ignored.
    functions: list[dict] | None = None
    function_call: str | dict | None = None
    # Structured output
    response_format: ResponseFormat | dict | None = None
    # Logprobs
    logprobs: bool | None = None
    top_logprobs: int | None = None  # 0-20, per OpenAI spec
    # OpenAI extended spec — declared so Pydantic stops silently dropping
    # it. Currently rejected with 400 in routes/chat.py if non-empty;
    # mapping to mlx-lm's logits processor is tracked separately.
    logit_bias: dict[str, float] | None = None
    # MLLM-specific parameters
    video_fps: float | None = None
    video_max_frames: int | None = None
    # Request timeout in seconds (None = use server default)
    timeout: float | None = None
    # Thinking/reasoning control (Qwen3 style).  None = server default.
    enable_thinking: bool | None = None
    # OpenAI extended spec: arbitrary kwargs forwarded to the chat template.
    # We currently honor the ``enable_thinking`` key here; other keys are
    # accepted (no Pydantic drop) but not yet forwarded — see
    # ``_resolve_enable_thinking`` in service/helpers.py for precedence.
    chat_template_kwargs: dict | None = None
    # Number of completions (only n=1 supported)
    n: int | None = None

    @model_validator(mode="after")
    def _normalize_max_completion_tokens(self) -> "ChatCompletionRequest":
        if self.max_completion_tokens is not None:
            if (
                self.max_tokens is not None
                and self.max_tokens != self.max_completion_tokens
            ):
                raise ValueError(
                    "Cannot specify both max_tokens and max_completion_tokens with "
                    "different values; use max_completion_tokens only."
                )
            self.max_tokens = self.max_completion_tokens
        return self

    @model_validator(mode="after")
    def _normalize_legacy_functions(self) -> "ChatCompletionRequest":
        """Translate the pre-1.0 ``functions``/``function_call`` shape into
        the modern ``tools``/``tool_choice`` slots so the route never has
        to know about the legacy form. Modern fields take precedence when
        a client supplies both — matches OpenAI's deprecation behavior."""
        if self.functions and self.tools is None:
            self.tools = [
                ToolDefinition(type="function", function=fn) for fn in self.functions
            ]
        if self.function_call is not None and self.tool_choice is None:
            fc = self.function_call
            if isinstance(fc, str):
                # "auto" / "none" map 1:1; anything else passes through and
                # the existing tool_choice handler will 400 on it.
                self.tool_choice = fc
            elif isinstance(fc, dict) and "name" in fc:
                self.tool_choice = {
                    "type": "function",
                    "function": {"name": fc["name"]},
                }
        return self


class AssistantMessage(BaseModel):
    """Response message from the assistant."""

    role: str = "assistant"
    content: str | None = None
    reasoning_content: str | None = (
        None  # Reasoning/thinking content (when --reasoning-parser is used)
    )
    tool_calls: list[ToolCall] | None = None

    def model_post_init(self, __context) -> None:
        """Add deprecated 'reasoning' alias for backward compatibility."""
        pass

    def model_dump(self, **kwargs) -> dict:
        """Include 'reasoning' as alias of reasoning_content for clients expecting it."""
        d = super().model_dump(**kwargs)
        # Add backward-compat alias — clients may read either field
        if "reasoning_content" in d:
            d["reasoning"] = d["reasoning_content"]
        return d


class ChatCompletionChoice(BaseModel):
    """A single choice in chat completion response."""

    index: int = 0
    message: AssistantMessage
    finish_reason: str | None = "stop"
    logprobs: ChoiceLogProbs | None = None


class CompletionTokensDetails(BaseModel):
    """Breakdown of completion token usage (OpenAI-compatible)."""

    reasoning_tokens: int = 0


class Usage(BaseModel):
    """Token usage statistics."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    completion_tokens_details: CompletionTokensDetails | None = None


class ChatCompletionResponse(BaseModel):
    """Response for chat completion."""

    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:8]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage = Field(default_factory=Usage)


# =============================================================================
# Text Completion
# =============================================================================


class CompletionRequest(BaseModel):
    """Request for text completion."""

    model: str = "default"
    prompt: str | list[str]
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stream: bool = False
    stop: list[str] | None = None
    # Extended OpenAI-compatible sampling parameters — see #355 + the
    # matching block on ChatCompletionRequest for wiring + caveats.
    top_k: int | None = None
    min_p: float | None = None
    repetition_penalty: float | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    # Logprobs
    logprobs: bool | None = None
    top_logprobs: int | None = None  # 0-20, per OpenAI spec
    # OpenAI FIM (fill-in-the-middle) suffix. Declared so Pydantic stops
    # silently dropping it; rejected with 400 in routes/completions.py
    # when non-empty since no MLX engine implements FIM yet (and silently
    # ignoring it produces wrong completions on code-completion clients).
    suffix: str | None = None
    # Request timeout in seconds (None = use server default)
    timeout: float | None = None


class CompletionChoice(BaseModel):
    """A single choice in text completion response."""

    index: int = 0
    text: str
    finish_reason: str | None = "stop"
    logprobs: ChoiceLogProbs | None = None


class CompletionResponse(BaseModel):
    """Response for text completion."""

    id: str = Field(default_factory=lambda: f"cmpl-{uuid.uuid4().hex[:8]}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[CompletionChoice]
    usage: Usage = Field(default_factory=Usage)


# =============================================================================
# Models List
# =============================================================================


class ModelInfo(BaseModel):
    """Information about an available model."""

    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "rapid-mlx"


class ModelsResponse(BaseModel):
    """Response for listing models."""

    object: str = "list"
    data: list[ModelInfo]


# =============================================================================
# MCP (Model Context Protocol)
# =============================================================================


class MCPToolInfo(BaseModel):
    """Information about an MCP tool."""

    name: str
    description: str
    server: str
    parameters: dict = Field(default_factory=dict)


class MCPToolsResponse(BaseModel):
    """Response for listing MCP tools."""

    tools: list[MCPToolInfo]
    count: int


class MCPServerInfo(BaseModel):
    """Information about an MCP server."""

    name: str
    state: str
    transport: str
    tools_count: int
    error: str | None = None


class MCPServersResponse(BaseModel):
    """Response for listing MCP servers."""

    servers: list[MCPServerInfo]


class MCPExecuteRequest(BaseModel):
    """Request to execute an MCP tool."""

    tool_name: str
    arguments: dict = Field(default_factory=dict)


class MCPExecuteResponse(BaseModel):
    """Response from executing an MCP tool."""

    tool_name: str
    content: str | list | dict | None = None
    is_error: bool = False
    error_message: str | None = None


# =============================================================================
# Audio (STT/TTS)
# =============================================================================


class AudioTranscriptionRequest(BaseModel):
    """Request for audio transcription (STT)."""

    model: str = "whisper-large-v3"
    language: str | None = None
    response_format: str = "json"
    temperature: float = 0.0
    timestamp_granularities: list[str] | None = None


class AudioTranscriptionResponse(BaseModel):
    """Response from audio transcription."""

    text: str
    language: str | None = None
    duration: float | None = None
    segments: list[dict] | None = None


class AudioSpeechRequest(BaseModel):
    """Request for text-to-speech."""

    model: str = "kokoro"
    input: str
    voice: str = "af_heart"
    speed: float = 1.0
    response_format: str = "wav"


class AudioSeparationRequest(BaseModel):
    """Request for audio source separation."""

    model: str = "htdemucs"
    stems: list[str] = Field(default_factory=lambda: ["vocals", "accompaniment"])


# =============================================================================
# Embeddings
# =============================================================================


class EmbeddingRequest(BaseModel):
    """Request for text embeddings (OpenAI compatible)."""

    # extra="forbid" turns silent-drop into a 422 with a clear field name.
    # Without it, fields like `dimensions` or `encoding_format` typos pass
    # through and the user only notices when the response shape is wrong.
    # protected_namespaces=() suppresses the Pydantic v2 warning about
    # the `model` field colliding with the reserved `model_` prefix; a
    # future Pydantic point release could otherwise promote that warning
    # to an error and 500 every embeddings request.
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    # OpenAI spec lists 4 input shapes: ``str``, ``list[str]``,
    # ``list[int]`` (single pre-tokenized input), and
    # ``list[list[int]]`` (batch of pre-tokenized inputs). Production
    # pipelines that pre-tokenize with a shared HF tokenizer send the
    # latter two forms — refusing them broke LangChain / LlamaIndex
    # integrations that hard-code the spec shape (R10 sweep H6).
    #
    # ``StrictInt`` / ``StrictStr`` so Pydantic does NOT silently
    # coerce ``"123"`` → 123 (would be treated as token id 123, a
    # different embedding from the word "123") or ``True`` → 1
    # (Python ``bool`` is an ``int`` subclass; without ``StrictInt``
    # a JSON ``true`` would pass as token id 1).
    input: StrictStr | list[StrictStr] | list[StrictInt] | list[list[StrictInt]]
    model: str
    # Literal so an unknown value (typo like "base65" or "BASE64") 422s
    # at parse time rather than silently falling back to float — that
    # silent fallback is the same class of bug this PR exists to close.
    encoding_format: Literal["float", "base64"] | None = "float"
    # OpenAI spec: per-vector truncation. Common for MRL-style models
    # (text-embedding-3-large, nomic-embed-text-v1.5). Implemented in
    # the route as a post-embed slice + L2 renormalization (required
    # for the truncated vector to remain a valid embedding for cosine
    # similarity per the OpenAI cookbook).
    dimensions: int | None = None
    # OpenAI abuse-tracking field. Accepted (not validated) so clients
    # using the upstream SDK don't see a 422 on unknown field.
    user: str | None = None


class EmbeddingData(BaseModel):
    """A single embedding result."""

    object: str = "embedding"
    index: int
    # `list[float]` for encoding_format="float"; base64-encoded float32
    # little-endian bytes (as ASCII string) for encoding_format="base64".
    embedding: list[float] | str


class EmbeddingUsage(BaseModel):
    """Token usage for embedding requests."""

    prompt_tokens: int = 0
    total_tokens: int = 0


class EmbeddingResponse(BaseModel):
    """Response for embeddings endpoint (OpenAI compatible)."""

    object: str = "list"
    data: list[EmbeddingData]
    model: str
    usage: EmbeddingUsage = Field(default_factory=EmbeddingUsage)


# =============================================================================
# Streaming (for SSE responses)
# =============================================================================


class ChatCompletionChunkDelta(BaseModel):
    """Delta content in a streaming chunk."""

    role: str | None = None
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[dict] | None = None


class ChatCompletionChunkChoice(BaseModel):
    """A single choice in a streaming chunk."""

    index: int = 0
    delta: ChatCompletionChunkDelta
    finish_reason: str | None = None
    logprobs: ChoiceLogProbs | None = None


class ChatCompletionChunk(BaseModel):
    """A streaming chunk for chat completion."""

    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:8]}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionChunkChoice]
    usage: Usage | None = None  # Included when stream_options.include_usage=true
