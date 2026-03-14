# SPDX-License-Identifier: Apache-2.0
"""
Tests for Pydantic API models.

Tests all request/response models in vllm_mlx/api/models.py.
These are pure Pydantic models with no MLX dependency.
"""

import json
import time

from vllm_mlx.api.models import (
    AssistantMessage,
    AudioSeparationRequest,
    AudioSpeechRequest,
    AudioTranscriptionRequest,
    AudioTranscriptionResponse,
    AudioUrl,
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionChoice,
    CompletionRequest,
    CompletionResponse,
    ContentPart,
    EmbeddingData,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingUsage,
    FunctionCall,
    ImageUrl,
    MCPExecuteRequest,
    MCPExecuteResponse,
    MCPServerInfo,
    MCPToolInfo,
    MCPToolsResponse,
    Message,
    ModelInfo,
    ModelsResponse,
    ResponseFormat,
    ResponseFormatJsonSchema,
    StreamOptions,
    ToolCall,
    ToolDefinition,
    Usage,
    VideoUrl,
)


class TestContentTypes:
    """Tests for multimodal content type models."""

    def test_image_url_basic(self):
        img = ImageUrl(url="https://example.com/img.png")
        assert img.url == "https://example.com/img.png"
        assert img.detail is None

    def test_image_url_with_detail(self):
        img = ImageUrl(url="https://example.com/img.png", detail="high")
        assert img.detail == "high"

    def test_video_url(self):
        vid = VideoUrl(url="https://example.com/vid.mp4")
        assert vid.url == "https://example.com/vid.mp4"

    def test_audio_url(self):
        audio = AudioUrl(url="https://example.com/audio.wav")
        assert audio.url == "https://example.com/audio.wav"

    def test_content_part_text(self):
        part = ContentPart(type="text", text="Hello world")
        assert part.type == "text"
        assert part.text == "Hello world"
        assert part.image_url is None

    def test_content_part_image(self):
        part = ContentPart(
            type="image_url",
            image_url=ImageUrl(url="data:image/png;base64,abc123"),
        )
        assert part.type == "image_url"
        assert part.image_url.url == "data:image/png;base64,abc123"

    def test_content_part_video(self):
        part = ContentPart(type="video", video="/path/to/video.mp4")
        assert part.type == "video"
        assert part.video == "/path/to/video.mp4"

    def test_content_part_video_url(self):
        part = ContentPart(
            type="video_url",
            video_url=VideoUrl(url="https://example.com/vid.mp4"),
        )
        assert part.type == "video_url"
        assert part.video_url.url == "https://example.com/vid.mp4"

    def test_content_part_audio_url(self):
        part = ContentPart(
            type="audio_url",
            audio_url=AudioUrl(url="https://example.com/audio.wav"),
        )
        assert part.type == "audio_url"


class TestMessage:
    """Tests for Message model."""

    def test_simple_text_message(self):
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.tool_calls is None
        assert msg.tool_call_id is None

    def test_system_message(self):
        msg = Message(role="system", content="You are helpful.")
        assert msg.role == "system"

    def test_assistant_message_with_tool_calls(self):
        msg = Message(
            role="assistant",
            content=None,
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather"},
                }
            ],
        )
        assert msg.tool_calls is not None
        assert len(msg.tool_calls) == 1

    def test_tool_response_message(self):
        msg = Message(role="tool", content="72F and sunny", tool_call_id="call_1")
        assert msg.role == "tool"
        assert msg.tool_call_id == "call_1"

    def test_multimodal_message(self):
        msg = Message(
            role="user",
            content=[
                ContentPart(type="text", text="What is this?"),
                ContentPart(
                    type="image_url",
                    image_url=ImageUrl(url="https://example.com/img.png"),
                ),
            ],
        )
        assert isinstance(msg.content, list)
        assert len(msg.content) == 2

    def test_none_content(self):
        msg = Message(role="assistant", content=None)
        assert msg.content is None


class TestToolCalling:
    """Tests for tool calling models."""

    def test_function_call(self):
        fc = FunctionCall(name="get_weather", arguments='{"city": "NYC"}')
        assert fc.name == "get_weather"
        assert fc.arguments == '{"city": "NYC"}'

    def test_tool_call(self):
        tc = ToolCall(
            id="call_abc123",
            type="function",
            function=FunctionCall(name="get_weather", arguments='{"city": "NYC"}'),
        )
        assert tc.id == "call_abc123"
        assert tc.type == "function"
        assert tc.function.name == "get_weather"

    def test_tool_call_default_type(self):
        tc = ToolCall(
            id="call_1",
            function=FunctionCall(name="test", arguments="{}"),
        )
        assert tc.type == "function"

    def test_tool_definition(self):
        td = ToolDefinition(
            function={
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            }
        )
        assert td.type == "function"
        assert td.function["name"] == "get_weather"


class TestResponseFormat:
    """Tests for structured output models."""

    def test_default_text_format(self):
        rf = ResponseFormat()
        assert rf.type == "text"
        assert rf.json_schema is None

    def test_json_object_format(self):
        rf = ResponseFormat(type="json_object")
        assert rf.type == "json_object"

    def test_json_schema_format(self):
        schema = ResponseFormatJsonSchema(
            name="person",
            description="A person",
            schema={"type": "object", "properties": {"name": {"type": "string"}}},
        )
        rf = ResponseFormat(type="json_schema", json_schema=schema)
        assert rf.type == "json_schema"
        assert rf.json_schema.name == "person"
        assert rf.json_schema.schema_ == {
            "type": "object",
            "properties": {"name": {"type": "string"}},
        }

    def test_json_schema_strict(self):
        schema = ResponseFormatJsonSchema(
            name="test",
            schema={"type": "object"},
            strict=True,
        )
        assert schema.strict is True

    def test_json_schema_default_strict(self):
        schema = ResponseFormatJsonSchema(
            name="test",
            schema={"type": "object"},
        )
        assert schema.strict is False


class TestChatCompletion:
    """Tests for chat completion request/response models."""

    def test_minimal_request(self):
        req = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
        )
        assert req.model == "test-model"
        assert len(req.messages) == 1
        assert req.stream is False
        assert req.temperature is None
        assert req.tools is None

    def test_full_request(self):
        req = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
            temperature=0.5,
            top_p=0.9,
            max_tokens=100,
            stream=True,
            stream_options=StreamOptions(include_usage=True),
            stop=["END"],
            tools=[ToolDefinition(function={"name": "test", "description": "test"})],
            tool_choice="auto",
            response_format=ResponseFormat(type="json_object"),
            timeout=30.0,
        )
        assert req.temperature == 0.5
        assert req.stream is True
        assert req.stream_options.include_usage is True
        assert req.tools is not None
        assert req.timeout == 30.0

    def test_mllm_request_params(self):
        req = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
            video_fps=1.0,
            video_max_frames=16,
        )
        assert req.video_fps == 1.0
        assert req.video_max_frames == 16

    def test_assistant_message_reasoning(self):
        msg = AssistantMessage(
            content="The answer is 42.",
            reasoning="I thought about it carefully.",
        )
        assert msg.content == "The answer is 42."
        assert msg.reasoning == "I thought about it carefully."
        assert msg.reasoning_content == "I thought about it carefully."

    def test_assistant_message_no_reasoning(self):
        msg = AssistantMessage(content="Hello")
        assert msg.reasoning is None
        assert msg.reasoning_content is None

    def test_assistant_message_with_tool_calls(self):
        msg = AssistantMessage(
            tool_calls=[
                ToolCall(
                    id="call_1",
                    function=FunctionCall(
                        name="get_weather", arguments='{"city": "NYC"}'
                    ),
                )
            ]
        )
        assert msg.content is None
        assert len(msg.tool_calls) == 1

    def test_chat_completion_choice(self):
        choice = ChatCompletionChoice(
            index=0,
            message=AssistantMessage(content="Hello!"),
            finish_reason="stop",
        )
        assert choice.index == 0
        assert choice.message.content == "Hello!"
        assert choice.finish_reason == "stop"

    def test_usage(self):
        usage = Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        assert usage.prompt_tokens == 10
        assert usage.total_tokens == 30

    def test_usage_defaults(self):
        usage = Usage()
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 0

    def test_chat_completion_response(self):
        resp = ChatCompletionResponse(
            model="test-model",
            choices=[
                ChatCompletionChoice(
                    message=AssistantMessage(content="Hi!"),
                )
            ],
        )
        assert resp.object == "chat.completion"
        assert resp.model == "test-model"
        assert resp.id.startswith("chatcmpl-")
        assert resp.created > 0
        assert len(resp.choices) == 1

    def test_chat_completion_response_auto_fields(self):
        before = int(time.time())
        resp = ChatCompletionResponse(
            model="test",
            choices=[ChatCompletionChoice(message=AssistantMessage(content="x"))],
        )
        after = int(time.time())
        assert before <= resp.created <= after
        assert resp.usage.prompt_tokens == 0


class TestTextCompletion:
    """Tests for text completion models."""

    def test_completion_request_string_prompt(self):
        req = CompletionRequest(model="test-model", prompt="Once upon a time")
        assert req.prompt == "Once upon a time"
        assert req.stream is False

    def test_completion_request_list_prompt(self):
        req = CompletionRequest(model="test-model", prompt=["Hello", "World"])
        assert isinstance(req.prompt, list)
        assert len(req.prompt) == 2

    def test_completion_choice(self):
        choice = CompletionChoice(text="the end.", finish_reason="stop")
        assert choice.text == "the end."
        assert choice.index == 0

    def test_completion_response(self):
        resp = CompletionResponse(
            model="test-model",
            choices=[CompletionChoice(text="Hello!")],
        )
        assert resp.object == "text_completion"
        assert resp.id.startswith("cmpl-")
        assert len(resp.choices) == 1


class TestModelsEndpoint:
    """Tests for models list models."""

    def test_model_info(self):
        info = ModelInfo(id="mlx-community/Llama-3.2-3B-Instruct-4bit")
        assert info.id == "mlx-community/Llama-3.2-3B-Instruct-4bit"
        assert info.object == "model"
        assert info.owned_by == "vllm-mlx"

    def test_models_response(self):
        resp = ModelsResponse(
            data=[
                ModelInfo(id="model-1"),
                ModelInfo(id="model-2"),
            ]
        )
        assert resp.object == "list"
        assert len(resp.data) == 2


class TestMCPModels:
    """Tests for MCP models."""

    def test_mcp_tool_info(self):
        tool = MCPToolInfo(
            name="search",
            description="Search the web",
            server="brave-search",
            parameters={"query": {"type": "string"}},
        )
        assert tool.name == "search"
        assert tool.server == "brave-search"

    def test_mcp_tools_response(self):
        resp = MCPToolsResponse(
            tools=[MCPToolInfo(name="t1", description="d1", server="s1")],
            count=1,
        )
        assert resp.count == 1

    def test_mcp_server_info(self):
        info = MCPServerInfo(
            name="test-server",
            state="connected",
            transport="stdio",
            tools_count=3,
        )
        assert info.state == "connected"
        assert info.error is None

    def test_mcp_server_info_with_error(self):
        info = MCPServerInfo(
            name="broken",
            state="error",
            transport="sse",
            tools_count=0,
            error="Connection refused",
        )
        assert info.error == "Connection refused"

    def test_mcp_execute_request(self):
        req = MCPExecuteRequest(
            tool_name="search",
            arguments={"query": "python"},
        )
        assert req.tool_name == "search"

    def test_mcp_execute_request_default_args(self):
        req = MCPExecuteRequest(tool_name="ping")
        assert req.arguments == {}

    def test_mcp_execute_response(self):
        resp = MCPExecuteResponse(
            tool_name="search",
            content="Results here",
            is_error=False,
        )
        assert resp.content == "Results here"
        assert resp.is_error is False

    def test_mcp_execute_response_error(self):
        resp = MCPExecuteResponse(
            tool_name="search",
            is_error=True,
            error_message="Not found",
        )
        assert resp.is_error is True
        assert resp.error_message == "Not found"


class TestAudioModels:
    """Tests for audio API models."""

    def test_transcription_request_defaults(self):
        req = AudioTranscriptionRequest()
        assert req.model == "whisper-large-v3"
        assert req.temperature == 0.0
        assert req.response_format == "json"

    def test_transcription_request_custom(self):
        req = AudioTranscriptionRequest(
            model="parakeet-tdt-0.6b-v2",
            language="en",
            response_format="verbose_json",
        )
        assert req.model == "parakeet-tdt-0.6b-v2"
        assert req.language == "en"

    def test_transcription_response(self):
        resp = AudioTranscriptionResponse(
            text="Hello world",
            language="en",
            duration=2.5,
        )
        assert resp.text == "Hello world"
        assert resp.duration == 2.5

    def test_speech_request_defaults(self):
        req = AudioSpeechRequest(input="Hello world")
        assert req.model == "kokoro"
        assert req.voice == "af_heart"
        assert req.speed == 1.0
        assert req.response_format == "wav"

    def test_speech_request_custom(self):
        req = AudioSpeechRequest(
            model="chatterbox",
            input="Test speech",
            voice="custom_voice",
            speed=1.5,
        )
        assert req.speed == 1.5

    def test_separation_request_defaults(self):
        req = AudioSeparationRequest()
        assert req.model == "htdemucs"
        assert req.stems == ["vocals", "accompaniment"]


class TestEmbeddingModels:
    """Tests for embedding API models."""

    def test_embedding_request_string(self):
        req = EmbeddingRequest(input="Hello world", model="bert-base")
        assert req.input == "Hello world"
        assert req.encoding_format == "float"

    def test_embedding_request_list(self):
        req = EmbeddingRequest(input=["Hello", "World"], model="bert-base")
        assert isinstance(req.input, list)
        assert len(req.input) == 2

    def test_embedding_data(self):
        data = EmbeddingData(index=0, embedding=[0.1, 0.2, 0.3])
        assert data.object == "embedding"
        assert len(data.embedding) == 3

    def test_embedding_usage(self):
        usage = EmbeddingUsage(prompt_tokens=5, total_tokens=5)
        assert usage.prompt_tokens == 5

    def test_embedding_response(self):
        resp = EmbeddingResponse(
            data=[EmbeddingData(index=0, embedding=[0.1, 0.2])],
            model="bert-base",
        )
        assert resp.object == "list"
        assert resp.model == "bert-base"
        assert len(resp.data) == 1


class TestStreamingModels:
    """Tests for streaming chunk models."""

    def test_chunk_delta_content(self):
        delta = ChatCompletionChunkDelta(content="Hello")
        assert delta.content == "Hello"
        assert delta.role is None

    def test_chunk_delta_role(self):
        delta = ChatCompletionChunkDelta(role="assistant")
        assert delta.role == "assistant"
        assert delta.content is None

    def test_chunk_delta_no_reasoning(self):
        """Streaming chunks should not have reasoning fields (OpenAI compat)."""
        delta = ChatCompletionChunkDelta(content="hello")
        data = json.loads(delta.model_dump_json(exclude_none=True))
        assert "reasoning" not in data
        assert "reasoning_content" not in data

    def test_chunk_delta_tool_calls(self):
        delta = ChatCompletionChunkDelta(
            tool_calls=[{"index": 0, "function": {"name": "test"}}]
        )
        assert len(delta.tool_calls) == 1

    def test_chunk_choice(self):
        choice = ChatCompletionChunkChoice(
            delta=ChatCompletionChunkDelta(content="Hi"),
            finish_reason=None,
        )
        assert choice.index == 0
        assert choice.finish_reason is None

    def test_chunk_choice_finished(self):
        choice = ChatCompletionChunkChoice(
            delta=ChatCompletionChunkDelta(),
            finish_reason="stop",
        )
        assert choice.finish_reason == "stop"

    def test_chat_completion_chunk(self):
        chunk = ChatCompletionChunk(
            model="test-model",
            choices=[
                ChatCompletionChunkChoice(
                    delta=ChatCompletionChunkDelta(content="Hi"),
                )
            ],
        )
        assert chunk.object == "chat.completion.chunk"
        assert chunk.id.startswith("chatcmpl-")
        assert chunk.model == "test-model"

    def test_chat_completion_chunk_with_usage(self):
        chunk = ChatCompletionChunk(
            model="test-model",
            choices=[
                ChatCompletionChunkChoice(
                    delta=ChatCompletionChunkDelta(),
                    finish_reason="stop",
                )
            ],
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
        assert chunk.usage.total_tokens == 15


class TestModelSerialization:
    """Tests for model serialization (model_dump / JSON)."""

    def test_assistant_message_serializes_reasoning_content(self):
        msg = AssistantMessage(content="Answer", reasoning="Thought")
        data = msg.model_dump()
        assert data["reasoning_content"] == "Thought"
        assert data["reasoning"] == "Thought"

    def test_chat_completion_response_json(self):
        resp = ChatCompletionResponse(
            model="test-model",
            choices=[
                ChatCompletionChoice(
                    message=AssistantMessage(content="Hi!"),
                )
            ],
        )
        json_str = resp.model_dump_json()
        assert "test-model" in json_str
        assert "Hi!" in json_str

    def test_chunk_delta_reasoning_field(self):
        """Streaming chunk delta supports optional reasoning field."""
        delta = ChatCompletionChunkDelta(content="hello")
        data = delta.model_dump()
        assert data["reasoning"] is None
        assert "reasoning_content" not in data

        delta_with = ChatCompletionChunkDelta(reasoning="thinking...")
        data_with = delta_with.model_dump()
        assert data_with["reasoning"] == "thinking..."

    def test_response_format_json_schema_alias(self):
        schema = ResponseFormatJsonSchema(
            name="test",
            schema={"type": "object"},
        )
        data = schema.model_dump(by_alias=True)
        assert "schema" in data
