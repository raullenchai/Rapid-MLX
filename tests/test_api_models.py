# SPDX-License-Identifier: Apache-2.0
"""
Tests for Pydantic API models.

Tests all request/response models in vllm_mlx/api/models.py.
These are pure Pydantic models with no MLX dependency.
"""

import json
import time
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

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
    _normalize_stop,
    _validate_timeout,
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

    @pytest.mark.parametrize("role", ["user", "system", "developer"])
    def test_null_content_rejected_for_input_roles(self, role):
        """R15 #175 item 3 (adversarial fuzz a9c828): a request body
        carrying ``{"role":"user","content":null}`` (or "system" /
        "developer") used to return HTTP 200 with garbled output (the
        chat-template flattened ``None`` to the literal string
        ``"None"`` on some templates, dropped the turn on others).
        Post-fix the schema layer rejects with a clear error pointing
        at the content field, so the route surfaces 400 via the
        ``RequestValidationError`` envelope handler."""
        with pytest.raises(ValidationError) as excinfo:
            Message(role=role, content=None)
        # Field path is surfaced so the OpenAI-shaped envelope can
        # populate ``error.param`` with ``content``.
        assert "content" in str(excinfo.value)
        # Error mentions the role so a single-message-array client
        # knows which turn to fix.
        assert role in str(excinfo.value)

    def test_null_content_allowed_for_assistant(self):
        """OpenAI spec allows assistant ``content=null`` when
        ``tool_calls`` is present (the response IS the call). We also
        preserve the pre-fix behaviour for plain assistant turns —
        the #175 fuzz only exposed the user-role case, and tightening
        the assistant path would break the response→follow-up
        roundtrip where THIS server's AssistantMessage shape (which
        emits ``content=null`` for reasoning-only turns) gets re-sent
        as a Message."""
        msg = Message(role="assistant", content=None)
        assert msg.content is None
        msg_tc = Message(
            role="assistant",
            content=None,
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "noop"},
                }
            ],
        )
        assert msg_tc.content is None
        assert msg_tc.tool_calls is not None

    def test_null_content_allowed_for_tool(self):
        """The route-layer F-111 validator already normalises tool
        ``content=null`` to an empty string. Keep the schema-layer
        gate loose so the existing normalization path keeps working."""
        msg = Message(role="tool", content=None, tool_call_id="call_1")
        assert msg.content is None

    def test_empty_string_content_allowed(self):
        """An empty string is a legitimate empty turn — OpenAI accepts
        it and downstream chat templates handle it gracefully."""
        msg = Message(role="user", content="")
        assert msg.content == ""

    def test_empty_list_content_allowed(self):
        """An empty multimodal array is legitimate: the engine decides
        whether to skip the turn or surface a validation error. The
        schema layer must NOT reject ``[]`` — only the explicit null
        shape that the fuzz exposed."""
        msg = Message(role="user", content=[])
        assert msg.content == []

    def test_multimodal_content_allowed_with_image_only(self):
        """A multimodal user turn with only an image_url part — no
        text — must pass. R15 #175 fix gates on ``content is None``
        not on list contents."""
        msg = Message(
            role="user",
            content=[
                ContentPart(
                    type="image_url",
                    image_url=ImageUrl(url="https://example.com/x.png"),
                )
            ],
        )
        assert isinstance(msg.content, list)
        assert len(msg.content) == 1


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

    @pytest.mark.parametrize(
        "bad_name",
        [
            "",  # F-035: empty string - hermes parser leaks <tool_call> to content
            "a" * 65,  # F-146: >64 chars - exceeds OpenAI spec cap
            "a" * 10000,  # F-146: 10k-char fuzz - context-window waster
            "weather_\U0001f324",  # F-146: emoji - non-ASCII
            "$(whoami)",  # F-146: shell metachars
            "foo\nbar",  # F-146: control char (newline)
            "foo bar",  # F-146: space
            "foo.bar",  # F-146: dot
            "foo/bar",  # F-146: slash
        ],
    )
    def test_tool_definition_rejects_malformed_name(self, bad_name):
        """F-035 / F-146: ``function.name`` must match the OpenAI spec
        regex ``^[a-zA-Z0-9_-]{1,64}$``. Pre-fix the schema accepted
        anything (typed as ``dict``), and empty / emoji / 10k-char /
        shell-metachar names all 200'd - on hermes-parser models the
        empty-name case leaked literal ``<tool_call>{"name":"",...}</tool_call>``
        into ``content`` because the tool-call detector keyed off a
        non-empty name. A single regex constraint at the schema layer
        covers the whole class.
        """
        with pytest.raises(ValidationError) as excinfo:
            ToolDefinition(
                function={"name": bad_name, "parameters": {"type": "object"}}
            )
        assert "function.name" in str(excinfo.value)

    def test_tool_definition_rejects_missing_name(self):
        """F-035 / F-146: ``function`` dict missing the ``name`` key is
        also malformed - the OpenAI spec mandates ``name`` as required.
        """
        with pytest.raises(ValidationError) as excinfo:
            ToolDefinition(function={"description": "no name", "parameters": {}})
        assert "function.name" in str(excinfo.value)

    def test_tool_definition_accepts_full_64_char_name(self):
        """Boundary check - exactly 64 chars is the OpenAI spec maximum
        and must pass cleanly. Guards against off-by-one regressions in
        the regex bound."""
        name = "a" * 64
        td = ToolDefinition(function={"name": name, "parameters": {"type": "object"}})
        assert td.function["name"] == name


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

    def test_logit_bias_declared(self):
        """logit_bias must round-trip through the schema. Without an
        explicit declaration Pydantic silently drops the field on parse,
        so any rejection in the route handler never sees it."""
        req = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
            logit_bias={"50000": -100.0, "12345": 5.5},
        )
        assert req.logit_bias == {"50000": -100.0, "12345": 5.5}

    def test_logit_bias_defaults_none(self):
        """logit_bias is optional and defaults to None when omitted."""
        req = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
        )
        assert req.logit_bias is None

    def test_max_completion_tokens_normalized_to_max_tokens(self):
        """OpenAI deprecated max_tokens for reasoning models in Sept 2024;
        SDKs >=1.45 send max_completion_tokens only. Without the validator
        Pydantic silently drops it and the request runs uncapped."""
        req = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="hi")],
            max_completion_tokens=42,
        )
        assert req.max_completion_tokens == 42
        assert req.max_tokens == 42

    def test_max_completion_tokens_equal_to_max_tokens_allowed(self):
        req = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="hi")],
            max_tokens=42,
            max_completion_tokens=42,
        )
        assert req.max_tokens == 42

    def test_max_completion_tokens_conflicts_with_max_tokens(self):
        with pytest.raises(ValidationError) as excinfo:
            ChatCompletionRequest(
                model="test-model",
                messages=[Message(role="user", content="hi")],
                max_tokens=10,
                max_completion_tokens=42,
            )
        assert "max_completion_tokens" in str(excinfo.value)

    def test_assistant_message_reasoning(self):
        msg = AssistantMessage(
            content="The answer is 42.",
            reasoning_content="I thought about it carefully.",
        )
        assert msg.content == "The answer is 42."
        assert msg.reasoning_content == "I thought about it carefully."

    def test_assistant_message_no_reasoning(self):
        msg = AssistantMessage(content="Hello")
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


def _chat(**kwargs):
    return ChatCompletionRequest(
        model="test-model", messages=[Message(role="user", content="Hi")], **kwargs
    )


def _completion(**kwargs):
    return CompletionRequest(model="test-model", prompt="Once upon a time", **kwargs)


class TestStopScalar:
    """``stop`` must accept a scalar string as well as a list on both the
    chat and legacy-completion surfaces, normalizing once to a list in the
    request schema (OpenAI request shape: ``stop`` is ``str | list[str]``).
    Downstream code (``SamplingParams.stop: list[str]``) reads only lists."""

    @pytest.mark.parametrize("factory", [_chat, _completion])
    def test_scalar_string_normalized_to_list(self, factory):
        req = factory(stop="END")
        assert req.stop == ["END"]

    @pytest.mark.parametrize("factory", [_chat, _completion])
    def test_list_preserved(self, factory):
        req = factory(stop=["END", "STOP"])
        assert req.stop == ["END", "STOP"]

    @pytest.mark.parametrize("factory", [_chat, _completion])
    def test_empty_list_ok(self, factory):
        assert factory(stop=[]).stop == []

    @pytest.mark.parametrize("factory", [_chat, _completion])
    def test_none_default(self, factory):
        assert factory().stop is None

    @pytest.mark.parametrize("factory", [_chat, _completion])
    def test_non_string_element_rejected(self, factory):
        with pytest.raises(ValidationError):
            factory(stop=["END", 42])

    @pytest.mark.parametrize("factory", [_chat, _completion])
    def test_stop_sequences_returns_normalized_list(self, factory):
        """``stop_sequences()`` — the typed accessor used at the engine
        boundary — returns the canonical normalized list even when the wire
        input was a scalar string, so downstream ``SamplingParams.stop``
        always sees a flat ``list[str]``."""
        assert factory(stop="END").stop_sequences() == ["END"]
        assert factory(stop=["END", "STOP"]).stop_sequences() == ["END", "STOP"]
        assert factory().stop_sequences() is None

    @pytest.mark.parametrize("req_model", [ChatCompletionRequest, CompletionRequest])
    def test_openapi_schema_advertises_scalar_or_array(self, req_model):
        """The OpenAPI/JSON schema for ``stop`` must advertise both a
        scalar string and an array of strings (the accepted wire shapes),
        so generated clients accept the newly supported scalar form."""
        stop_prop = req_model.model_json_schema()["properties"]["stop"]
        assert "anyOf" in stop_prop, f"stop schema is not a union: {stop_prop}"
        type_set = {arm.get("type") for arm in stop_prop["anyOf"]}
        # string or array are the OpenAI-accepted wire shapes; ``null`` is
        # present because the field is optional (None = server default).
        assert type_set == {"string", "array", "null"}, (
            f"stop schema wrong: {stop_prop}"
        )


class TestTimeoutValidation:
    """A non-positive / non-finite ``timeout`` must be rejected with a
    4xx naming the field (schema layer) instead of firing an instant 504
    when it reaches the request-timeout guard in the routes."""

    @pytest.mark.parametrize("factory", [_chat, _completion])
    @pytest.mark.parametrize("bad", [0, 0.0, -1, -0.5, -1.0])
    def test_non_positive_rejected(self, factory, bad):
        with pytest.raises(ValidationError) as excinfo:
            factory(timeout=bad)
        assert "timeout" in str(excinfo.value)

    @pytest.mark.parametrize("factory", [_chat, _completion])
    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_rejected(self, factory, bad):
        with pytest.raises(ValidationError) as excinfo:
            factory(timeout=bad)
        assert "timeout" in str(excinfo.value)

    @pytest.mark.parametrize("factory", [_chat, _completion])
    @pytest.mark.parametrize("good", [0.1, 1, 30.0, 300])
    def test_positive_accepted(self, factory, good):
        assert factory(timeout=good).timeout == good

    @pytest.mark.parametrize("factory", [_chat, _completion])
    def test_none_default(self, factory):
        assert factory().timeout is None


class TestStopAndTimeoutHelpers:
    """Direct coverage of the shared ``_normalize_stop`` /
    ``_validate_timeout`` helpers. Pydantic skips field validators for
    an *absent* optional field, so the ``None`` guards inside the
    helpers are only exercised when called directly."""

    def test_validate_timeout_none_passthrough(self):
        assert _validate_timeout(None) is None

    def test_validate_timeout_nonfinite_rejected(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            with pytest.raises(ValueError):
                _validate_timeout(bad)

    def test_normalize_stop_none_passthrough(self):
        assert _normalize_stop(None) is None

    def test_normalize_stop_non_string_type_rejected(self):
        for bad in (42, [1, 2], True, 1.5):
            with pytest.raises(ValueError):
                _normalize_stop(bad)


class TestModelsEndpoint:
    """Tests for models list models."""

    def test_model_info(self):
        info = ModelInfo(id="mlx-community/Llama-3.2-3B-Instruct-4bit")
        assert info.id == "mlx-community/Llama-3.2-3B-Instruct-4bit"
        assert info.object == "model"
        assert info.owned_by == "rapid-mlx"

    def test_models_response(self):
        resp = ModelsResponse(
            data=[
                ModelInfo(id="model-1"),
                ModelInfo(id="model-2"),
            ]
        )
        assert resp.object == "list"
        assert resp.models == []
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

    def test_embedding_data_accepts_base64_string(self):
        """embedding may also be a base64-encoded string when the request
        used encoding_format=base64."""
        data = EmbeddingData(index=0, embedding="AAAAAA==")
        assert data.embedding == "AAAAAA=="

    def test_embedding_request_accepts_dimensions(self):
        """OpenAI MRL-style truncation field must round-trip; previously
        the field was silently dropped by the schema."""
        req = EmbeddingRequest(input="hi", model="bert", dimensions=128)
        assert req.dimensions == 128

    def test_embedding_request_accepts_user(self):
        """OpenAI abuse-tracking field must be accepted (not validated)."""
        req = EmbeddingRequest(input="hi", model="bert", user="account-42")
        assert req.user == "account-42"

    def test_embedding_request_rejects_unknown_fields(self):
        """extra='forbid' surfaces unknown fields as 422 errors rather
        than silently dropping them. Catches typos like ``encodingformat``
        or fields the server hasn't implemented yet."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            EmbeddingRequest(input="hi", model="bert", unknown_field="oops")

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

    @pytest.mark.asyncio
    async def test_guided_stream_helper_generates_default_response_id(self):
        """Direct helper callers retain the existing generated-ID contract."""
        from vllm_mlx.config import reset_config
        from vllm_mlx.engine.base import GenerationOutput
        from vllm_mlx.routes.chat import stream_chat_completion_guided

        class _GuidedEngine:
            async def generate_with_schema(self, **_kwargs):
                return GenerationOutput(
                    text="{}",
                    prompt_tokens=1,
                    completion_tokens=1,
                    finished=True,
                    finish_reason="stop",
                )

        cfg = reset_config()
        cfg.model_name = "test-model"
        request = ChatCompletionRequest(
            model="test-model",
            stream=True,
            messages=[{"role": "user", "content": "hi"}],
        )

        chunks = [
            chunk
            async for chunk in stream_chat_completion_guided(
                _GuidedEngine(),
                request.messages,
                request,
                {"type": "object"},
            )
        ]
        first = json.loads(chunks[0].removeprefix("data: "))
        assert first["id"].startswith("chatcmpl-")

    @pytest.mark.asyncio
    async def test_guided_buffer_exposes_no_id_while_generation_is_live(self):
        """Buffered guided work has no public cancellation window."""
        import asyncio

        from vllm_mlx.config import reset_config
        from vllm_mlx.routes.chat import stream_chat_completion_guided

        started = asyncio.Event()

        class _BlockedGuidedEngine:
            async def generate_with_schema(self, **_kwargs):
                started.set()
                await asyncio.Event().wait()

        cfg = reset_config()
        cfg.model_name = "test-model"
        request = ChatCompletionRequest(
            model="test-model",
            stream=True,
            messages=[{"role": "user", "content": "hi"}],
        )
        stream = stream_chat_completion_guided(
            _BlockedGuidedEngine(),
            request.messages,
            request,
            {"type": "object"},
            response_id="chatcmpl-" + "e" * 32,
        )
        first_wire_event = asyncio.create_task(anext(stream))
        await asyncio.wait_for(started.wait(), timeout=0.2)

        assert first_wire_event.done() is False
        first_wire_event.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first_wire_event
        await stream.aclose()

    @pytest.mark.asyncio
    async def test_strict_stream_helper_generates_default_response_id(self):
        """Strict helper direct callers also retain generated IDs."""
        from vllm_mlx.config import reset_config
        from vllm_mlx.engine.base import GenerationOutput
        from vllm_mlx.routes.chat import stream_chat_completion_strict_postgen

        class _StreamEngine:
            preserve_native_tool_format = False
            tokenizer = None

            async def stream_chat(self, messages, **_kwargs):
                del messages
                yield GenerationOutput(
                    text="{}",
                    new_text="{}",
                    prompt_tokens=1,
                    completion_tokens=1,
                    finished=True,
                    finish_reason="stop",
                )

        cfg = reset_config()
        cfg.model_name = "test-model"
        request = ChatCompletionRequest(
            model="test-model",
            stream=True,
            messages=[{"role": "user", "content": "hi"}],
        )

        chunks = [
            chunk
            async for chunk in stream_chat_completion_strict_postgen(
                _StreamEngine(),
                request.messages,
                request,
                {"type": "object"},
            )
        ]
        first = json.loads(chunks[0].removeprefix("data: "))
        assert first["id"].startswith("chatcmpl-")

    @pytest.mark.asyncio
    async def test_forced_prefix_waits_for_scheduler_admission(self):
        """Synthetic tool prefixes cannot expose an unadmitted request id."""
        from vllm_mlx.engine.base import GenerationOutput
        from vllm_mlx.engine.batched import BatchedEngine

        engine = BatchedEngine.__new__(BatchedEngine)
        engine._loaded = True
        engine._prepare_cache_stable_messages = lambda messages: (messages, None)
        engine._apply_chat_template = lambda *_args, **_kwargs: "prompt"
        engine._prepare_harmony_no_thinking_prompt = lambda prompt, **_kwargs: (
            prompt,
            None,
        )
        engine._needs_prefix_boundary_snapshot = lambda: False
        engine._create_output_router = lambda: None
        admitted = False

        async def _stream_generate(**_kwargs):
            nonlocal admitted
            admitted = True
            yield GenerationOutput(
                text="answer",
                new_text="answer",
                prompt_tokens=1,
                completion_tokens=1,
                finished=True,
                finish_reason="stop",
            )

        engine.stream_generate = _stream_generate
        stream = engine.stream_chat(
            [{"role": "user", "content": "hi"}],
            forced_assistant_prefix="<tool_call>",
        )

        first = await anext(stream)

        assert admitted is True
        assert first.new_text == "<tool_call>"
        await stream.aclose()

    @pytest.mark.asyncio
    async def test_role_frame_waits_for_admission_not_first_token(self):
        """Slow prefill still exposes a cancellable ID immediately after admit."""
        import asyncio

        from vllm_mlx.config import reset_config
        from vllm_mlx.engine.base import GenerationOutput
        from vllm_mlx.routes.chat import stream_chat_completion

        class _SlowPrefillEngine:
            preserve_native_tool_format = False
            tokenizer = None

            async def stream_chat(self, messages, **kwargs):
                del messages
                kwargs["request_admitted_event"].set()
                await asyncio.Event().wait()
                yield GenerationOutput(text="unreachable")

        cfg = reset_config()
        cfg.model_name = "test-model"
        request = ChatCompletionRequest(
            model="test-model",
            stream=True,
            messages=[{"role": "user", "content": "hi"}],
        )
        stream = stream_chat_completion(
            _SlowPrefillEngine(),
            request.messages,
            request,
            response_id="chatcmpl-" + "d" * 32,
            request_id="chatcmpl-" + "d" * 32,
        )

        first = await asyncio.wait_for(anext(stream), timeout=0.2)

        assert '"role":"assistant"' in first
        await stream.aclose()

    @pytest.mark.asyncio
    async def test_stream_helper_handles_empty_engine_before_and_after_admission(self):
        """Both immediate and post-admission exhaustion close cleanly."""
        import asyncio

        from vllm_mlx.config import reset_config
        from vllm_mlx.engine.base import GenerationOutput
        from vllm_mlx.routes.chat import stream_chat_completion

        cfg = reset_config()
        cfg.model_name = "test-model"
        request = ChatCompletionRequest(
            model="test-model",
            stream=True,
            messages=[{"role": "user", "content": "hi"}],
        )

        class _ImmediatelyEmptyEngine:
            preserve_native_tool_format = False
            tokenizer = None

            async def stream_chat(self, messages, **_kwargs):
                del messages
                if False:
                    yield None

        immediate = [
            chunk
            async for chunk in stream_chat_completion(
                _ImmediatelyEmptyEngine(), request.messages, request
            )
        ]
        assert any('"role":"assistant"' in chunk for chunk in immediate)

        release = asyncio.Event()

        class _AdmittedThenEmptyEngine:
            preserve_native_tool_format = False
            tokenizer = None

            async def stream_chat(self, messages, **kwargs):
                del messages
                kwargs["request_admitted_event"].set()
                await release.wait()
                if False:
                    yield None

        delayed_stream = stream_chat_completion(
            _AdmittedThenEmptyEngine(), request.messages, request
        )
        first = await anext(delayed_stream)
        assert '"role":"assistant"' in first
        release.set()
        assert [chunk async for chunk in delayed_stream]

        class _TwoOutputEngine:
            preserve_native_tool_format = False
            tokenizer = None

            async def stream_chat(self, messages, **_kwargs):
                del messages
                yield GenerationOutput(text="one", new_text="one")
                yield GenerationOutput(
                    text="onetwo",
                    new_text="two",
                    finished=True,
                    finish_reason="stop",
                )

        outputs = [
            chunk
            async for chunk in stream_chat_completion(
                _TwoOutputEngine(), request.messages, request
            )
        ]
        assert any('"content":"two"' in chunk for chunk in outputs)

    def test_stream_route_generates_one_public_scheduler_identity(self, monkeypatch):
        """The route-generated wire ID is passed unchanged to the engine."""
        import sys
        import types

        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from vllm_mlx.config import reset_config
        from vllm_mlx.engine.base import GenerationOutput
        from vllm_mlx.routes import chat as chat_route

        scheduler_stub = types.ModuleType("vllm_mlx.scheduler")
        scheduler_stub.BackpressureError = type("BackpressureError", (Exception,), {})
        monkeypatch.setitem(sys.modules, "vllm_mlx.scheduler", scheduler_stub)

        class _Engine:
            is_mllm = False
            preserve_native_tool_format = False
            tokenizer = None

            def __init__(self):
                self.stream_kwargs = None

            def build_prompt(self, *_args, **_kwargs):
                return "prompt"

            async def stream_chat(self, messages, **kwargs):
                del messages
                self.stream_kwargs = kwargs
                kwargs["request_admitted_event"].set()
                yield GenerationOutput(
                    text="ok",
                    new_text="ok",
                    finished=True,
                    finish_reason="stop",
                )

        engine = _Engine()
        cfg = reset_config()
        cfg.engine = engine
        cfg.model_name = "test-model"
        cfg.model_registry = None
        cfg.no_thinking = True
        monkeypatch.setattr(chat_route, "get_engine", lambda *_args: engine)
        app = FastAPI()
        app.include_router(chat_route.router)

        response = TestClient(app).post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

        assert response.status_code == 200
        wire_id = engine.stream_kwargs["request_id"]
        assert wire_id.startswith("chatcmpl-")
        assert f'"id":"{wire_id}"' in response.text

    @pytest.mark.asyncio
    async def test_stream_helper_cancels_pending_admission_on_caller_cancel(self):
        """Cancellation while prefill is pending retires both helper tasks."""
        import asyncio

        from vllm_mlx.config import reset_config
        from vllm_mlx.routes.chat import stream_chat_completion

        started = asyncio.Event()

        class _BlockedEngine:
            preserve_native_tool_format = False
            tokenizer = None

            async def stream_chat(self, messages, **_kwargs):
                del messages
                started.set()
                await asyncio.Event().wait()
                if False:
                    yield None

        cfg = reset_config()
        cfg.model_name = "test-model"
        request = ChatCompletionRequest(
            model="test-model",
            stream=True,
            messages=[{"role": "user", "content": "hi"}],
        )
        stream = stream_chat_completion(_BlockedEngine(), request.messages, request)
        pending = asyncio.create_task(anext(stream))
        await started.wait()
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        await stream.aclose()

    def test_stream_request_ids_keep_full_entropy_past_shared_prefix(self, monkeypatch):
        """Two UUIDs sharing the old 32-bit prefix remain distinct IDs."""
        from vllm_mlx.routes import chat

        values = iter(
            [
                SimpleNamespace(hex="12345678" + "a" * 24),
                SimpleNamespace(hex="12345678" + "b" * 24),
            ]
        )
        monkeypatch.setattr(chat.uuid, "uuid4", lambda: next(values))

        first = chat._new_stream_request_id()
        second = chat._new_stream_request_id()

        assert first == "chatcmpl-12345678" + "a" * 24
        assert second == "chatcmpl-12345678" + "b" * 24
        assert first != second

    @pytest.mark.asyncio
    async def test_diffusion_stream_registers_public_id_for_cancellation(self):
        """The diffusion worker's existing cancel event is publicly addressable."""
        import asyncio
        import queue
        import threading

        from vllm_mlx.runtime.diffusion_lane import _STREAM_DONE, DiffusionEngine

        engine = DiffusionEngine.__new__(DiffusionEngine)
        engine._max_tokens = 32
        engine._scheduler_config = None
        engine._generation_lock = asyncio.Lock()
        engine._worker_stuck = False
        engine._jobs = queue.Queue()
        engine._request_cancels = {}
        engine._request_cancels_lock = threading.Lock()
        public_id = "chatcmpl-" + "c" * 32
        admitted = asyncio.Event()

        async def _consume():
            return [
                chunk
                async for chunk in engine._stream_prompt_raw(
                    "prompt",
                    max_tokens=8,
                    temperature=0.0,
                    request_id=public_id,
                    request_admitted_event=admitted,
                )
            ]

        consumer = asyncio.create_task(_consume())
        while engine._jobs.empty():
            await asyncio.sleep(0)
        _, _, _, worker_output, cancel_event, done_event = engine._jobs.get_nowait()

        assert admitted.is_set()
        assert await engine.abort_request(public_id) is True
        assert cancel_event.is_set()
        worker_output.put(_STREAM_DONE)
        done_event.set()
        assert await consumer == []
        assert public_id not in engine._request_cancels
        assert await engine.abort_request(public_id) is False

    @pytest.mark.asyncio
    async def test_diffusion_lifecycle_cancels_and_clears_public_requests(self):
        """Stop signals every public request and resets the identity registry."""
        import threading

        from vllm_mlx.runtime.diffusion_lane import DiffusionEngine

        engine = DiffusionEngine(model_name="test-model")
        active = threading.Event()
        public = threading.Event()
        engine._active_cancel = active
        engine._request_cancels["chatcmpl-public"] = public

        await engine.stop()

        assert active.is_set()
        assert public.is_set()
        assert engine._request_cancels == {}

    @pytest.mark.asyncio
    @pytest.mark.parametrize("duplicate", [False, True])
    async def test_diffusion_setup_failure_removes_public_request(self, duplicate):
        """Admission setup is transactional for duplicate and queue failures."""
        import asyncio
        import queue
        import threading

        from vllm_mlx.runtime.diffusion_lane import DiffusionEngine

        class _FailingJobs(queue.Queue):
            def put(self, item, *args, **kwargs):
                raise RuntimeError("queue unavailable")

        engine = DiffusionEngine.__new__(DiffusionEngine)
        engine._max_tokens = 32
        engine._scheduler_config = None
        engine._generation_lock = asyncio.Lock()
        engine._worker_stuck = False
        engine._jobs = _FailingJobs() if not duplicate else queue.Queue()
        engine._request_cancels = {}
        engine._request_cancels_lock = threading.Lock()
        public_id = "chatcmpl-duplicate"
        if duplicate:
            engine._request_cancels[public_id] = threading.Event()

        stream = engine._stream_prompt_raw(
            "prompt",
            max_tokens=8,
            temperature=0.0,
            request_id=public_id,
        )
        expected = "already exists" if duplicate else "queue unavailable"
        with pytest.raises((ValueError, RuntimeError), match=expected):
            await anext(stream)
        await stream.aclose()

        if duplicate:
            assert public_id in engine._request_cancels
        else:
            assert public_id not in engine._request_cancels


class TestModelSerialization:
    """Tests for model serialization (model_dump / JSON)."""

    def test_assistant_message_serializes_reasoning_content(self):
        msg = AssistantMessage(content="Answer", reasoning_content="Thought")
        data = msg.model_dump()
        assert data["reasoning_content"] == "Thought"

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
        """Streaming chunk delta supports optional reasoning_content field."""
        delta = ChatCompletionChunkDelta(content="hello")
        data = delta.model_dump()
        assert data["reasoning_content"] is None

        delta_with = ChatCompletionChunkDelta(reasoning_content="thinking...")
        data_with = delta_with.model_dump()
        assert data_with["reasoning_content"] == "thinking..."

    def test_response_format_json_schema_alias(self):
        schema = ResponseFormatJsonSchema(
            name="test",
            schema={"type": "object"},
        )
        data = schema.model_dump(by_alias=True)
        assert "schema" in data

    # F-040 / D-MISSING-CONTENT-KEY regression: ``content`` MUST always
    # be present on the wire, even when the route serializes the response
    # with ``exclude_none=True`` and the underlying value is ``None``
    # (e.g. reasoning consumed the whole budget and we truncated
    # mid-``<think>``, or the model produced empty content +
    # ``finish_reason=stop``). Standard OpenAI clients read
    # ``resp["choices"][0]["message"]["content"]`` and crash with
    # ``KeyError: 'content'`` if the field is missing. D-MISSING-CONTENT-KEY
    # additionally enforces that the fill-in value is the empty string
    # ``""`` (not ``null``) so the wire-level ``string`` type
    # discriminator is preserved — Swift Codable / Rust serde decode
    # cleanly into a non-Optional ``String``.
    def test_assistant_message_content_always_present_in_json(self):
        """``content`` is REQUIRED in OpenAI's chat.completion schema (string|null|array)."""
        msg = AssistantMessage(
            content=None,
            reasoning_content="I was thinking but ran out of budget mid-thought.",
        )
        data = json.loads(msg.model_dump_json(exclude_none=True))
        assert "content" in data, (
            "F-040 / D-MISSING-CONTENT-KEY: 'content' must always appear "
            "on the wire (OpenAI spec); any standard SDK client crashes "
            "with KeyError otherwise."
        )
        # D-MISSING-CONTENT-KEY (r12-7): empty string, not null, so the
        # wire ``string`` type discriminator survives strongly-typed
        # decoders (Swift Codable, pydantic strict, Rust serde).
        assert data["content"] == ""
        # r10-B R10-C2: only ``reasoning_content`` is emitted; the
        # deprecation-window ``reasoning`` alias (r7-A R7-H2) was the
        # byte-for-byte root cause of R9-CRIT3 (openai-agents
        # text_delta doubling) and is removed.
        assert "reasoning" not in data
        assert data["reasoning_content"] == (
            "I was thinking but ran out of budget mid-thought."
        )

    def test_assistant_message_content_present_via_model_dump(self):
        """Direct ``model_dump(exclude_none=True)`` also surfaces ``content``."""
        msg = AssistantMessage(content=None, reasoning_content="…")
        data = msg.model_dump(exclude_none=True)
        assert "content" in data
        # D-MISSING-CONTENT-KEY: parity with the wire shape — ``""``.
        assert data["content"] == ""

    def test_chat_completion_response_content_always_present(self):
        """Parent ``ChatCompletionResponse.model_dump_json(exclude_none=True)``
        must preserve ``message.content`` (the actual production path in
        ``routes/chat.py`` non-streaming responses)."""
        resp = ChatCompletionResponse(
            model="reasoning-model",
            choices=[
                ChatCompletionChoice(
                    message=AssistantMessage(
                        content=None,
                        reasoning_content="Truncated mid-think.",
                    ),
                    finish_reason="length",
                )
            ],
        )
        payload = json.loads(resp.model_dump_json(exclude_none=True))
        message = payload["choices"][0]["message"]
        assert "content" in message
        # D-MISSING-CONTENT-KEY: ``""`` (preferred), not ``null``.
        assert message["content"] == ""
        # Make sure we did not regress the role / reasoning shape.
        assert message["role"] == "assistant"
        assert message["reasoning_content"] == "Truncated mid-think."

    def test_assistant_message_empty_stop_emits_empty_content_key(self):
        """D-MISSING-CONTENT-KEY (r12-7) reproducer: empty generation +
        ``finish_reason="stop"`` — the OpenAI canonical shape REQUIRES
        ``content`` to be present (Swift Codable, pydantic strict, Rust
        serde all hard-fail on a missing required key). The serializer
        emits ``content: ""`` (preferred over ``null``)."""
        resp = ChatCompletionResponse(
            model="granite4-h-micro-4bit",
            choices=[
                ChatCompletionChoice(
                    message=AssistantMessage(content=None),
                    finish_reason="stop",
                )
            ],
        )
        payload = json.loads(resp.model_dump_json(exclude_none=True))
        message = payload["choices"][0]["message"]
        assert "content" in message, (
            "D-MISSING-CONTENT-KEY: empty completion + stop MUST still "
            "carry the ``content`` key (regression r12-7, tracked in the "
            "0.8-era local TODO, since removed — see git history)."
        )
        assert message["content"] == ""
        assert message["role"] == "assistant"
        # No reasoning, no tool_calls — exact wire shape is the
        # canonical "{role,content}" doublet.
        assert set(message.keys()) == {"role", "content"}

    def test_assistant_message_tool_calls_only_emits_empty_content(self):
        """D-MISSING-CONTENT-KEY (r12-7): tool-call-only assistant
        messages serialize as ``{role,tool_calls,content:""}`` — the
        canonical OpenAI tool-call envelope. Without the explicit
        ``content`` key strict clients reject the message shape."""
        tc = ToolCall(
            id="call_1",
            type="function",
            function={"name": "get_weather", "arguments": "{}"},
        )
        msg = AssistantMessage(content=None, tool_calls=[tc])
        data = json.loads(msg.model_dump_json(exclude_none=True))
        assert "content" in data
        assert data["content"] == ""
        assert data["tool_calls"][0]["id"] == "call_1"
        assert data["role"] == "assistant"

    def test_chunk_delta_terminal_with_reasoning_has_content_empty(self):
        """F-040 / D-MISSING-CONTENT-KEY (streaming): the terminal chunk
        delta for a reasoning-only finish must expose ``content: ""``
        so clients reading ``chunk.choices[0].delta.content`` on the
        last chunk do not crash.
        """
        delta = ChatCompletionChunkDelta(reasoning_content="…", content=None)
        data = json.loads(delta.model_dump_json(exclude_none=True))
        assert "content" in data
        # D-MISSING-CONTENT-KEY: parity with the non-stream surface.
        assert data["content"] == ""

    def test_chunk_delta_pure_content_stays_lean(self):
        """Per-token content-only deltas keep their minimal shape — we do
        NOT inject ``reasoning_content: null`` or other noise on every
        chunk (per-token streaming budget must stay tight)."""
        delta = ChatCompletionChunkDelta(content="Hello")
        data = json.loads(delta.model_dump_json(exclude_none=True))
        assert data == {"content": "Hello"}

    def test_chunk_delta_empty_stays_empty(self):
        """A truly empty terminal delta (e.g. guided-decoding finish chunk)
        still serializes to ``{}`` — we do not pollute pure
        finish-marker chunks with a spurious ``content: null``.
        """
        delta = ChatCompletionChunkDelta()
        data = json.loads(delta.model_dump_json(exclude_none=True))
        assert data == {}
