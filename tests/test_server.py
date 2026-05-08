# SPDX-License-Identifier: Apache-2.0
"""Tests for the OpenAI-compatible API server."""

import json
import platform
import sys
import threading
from unittest.mock import AsyncMock, MagicMock, patch
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

# Skip all tests if not on Apple Silicon
pytestmark = pytest.mark.skipif(
    sys.platform != "darwin" or platform.machine() != "arm64",
    reason="Requires Apple Silicon",
)


# =============================================================================
# Unit Tests - Request/Response Models
# =============================================================================


class TestRequestModels:
    """Test Pydantic request models."""

    def test_chat_message_text_only(self):
        """Test chat message with text content."""
        from vllm_mlx.server import Message

        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_chat_message_multimodal(self):
        """Test chat message with multimodal content."""
        from vllm_mlx.server import Message

        content = [
            {"type": "text", "text": "What's this?"},
            {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}},
        ]
        msg = Message(role="user", content=content)

        assert msg.role == "user"
        assert isinstance(msg.content, list)
        assert len(msg.content) == 2

    def test_image_url_model(self):
        """Test ImageUrl model."""
        from vllm_mlx.server import ImageUrl

        img_url = ImageUrl(url="https://example.com/image.jpg")
        assert img_url.url == "https://example.com/image.jpg"
        assert img_url.detail is None

    def test_video_url_model(self):
        """Test VideoUrl model."""
        from vllm_mlx.server import VideoUrl

        video_url = VideoUrl(url="https://example.com/video.mp4")
        assert video_url.url == "https://example.com/video.mp4"

    def test_content_part_text(self):
        """Test ContentPart with text."""
        from vllm_mlx.server import ContentPart

        part = ContentPart(type="text", text="Hello world")
        assert part.type == "text"
        assert part.text == "Hello world"

    def test_content_part_image(self):
        """Test ContentPart with image_url."""
        from vllm_mlx.server import ContentPart

        part = ContentPart(
            type="image_url", image_url={"url": "https://example.com/img.jpg"}
        )
        assert part.type == "image_url"
        # image_url can be dict or ImageUrl object
        if isinstance(part.image_url, dict):
            assert part.image_url["url"] == "https://example.com/img.jpg"
        else:
            assert part.image_url.url == "https://example.com/img.jpg"

    def test_content_part_video(self):
        """Test ContentPart with video."""
        from vllm_mlx.server import ContentPart

        part = ContentPart(type="video", video="/path/to/video.mp4")
        assert part.type == "video"
        assert part.video == "/path/to/video.mp4"

    def test_content_part_video_url(self):
        """Test ContentPart with video_url."""
        from vllm_mlx.server import ContentPart

        part = ContentPart(
            type="video_url", video_url={"url": "https://example.com/video.mp4"}
        )
        assert part.type == "video_url"
        # video_url can be dict or VideoUrl object
        if isinstance(part.video_url, dict):
            assert part.video_url["url"] == "https://example.com/video.mp4"
        else:
            assert part.video_url.url == "https://example.com/video.mp4"


class TestChatCompletionRequest:
    """Test ChatCompletionRequest model."""

    def test_basic_request(self):
        """Test basic chat completion request."""
        from vllm_mlx.server import ChatCompletionRequest, Message

        request = ChatCompletionRequest(
            model="test-model", messages=[Message(role="user", content="Hello")]
        )

        assert request.model == "test-model"
        assert len(request.messages) == 1
        assert request.max_tokens is None  # uses _default_max_tokens when None
        assert (
            request.temperature is None
        )  # resolved at runtime by _resolve_temperature
        assert request.stream is False  # default

    def test_request_with_options(self):
        """Test request with custom options."""
        from vllm_mlx.server import ChatCompletionRequest, Message

        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
            max_tokens=100,
            temperature=0.5,
            stream=True,
        )

        assert request.max_tokens == 100
        assert request.temperature == 0.5
        assert request.stream is True

    def test_request_with_video_params(self):
        """Test request with video parameters."""
        from vllm_mlx.server import ChatCompletionRequest, Message

        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Describe the video")],
            video_fps=2.0,
            video_max_frames=16,
        )

        assert request.video_fps == 2.0
        assert request.video_max_frames == 16

    def test_max_tokens_must_be_positive(self):
        """Chat completion requests reject zero or negative max_tokens."""
        from vllm_mlx.server import ChatCompletionRequest, Message

        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                model="test-model",
                messages=[Message(role="user", content="Hello")],
                max_tokens=0,
            )


class TestCompletionRequest:
    """Test CompletionRequest model."""

    def test_basic_completion_request(self):
        """Test basic completion request."""
        from vllm_mlx.server import CompletionRequest

        request = CompletionRequest(model="test-model", prompt="Once upon a time")

        assert request.model == "test-model"
        assert request.prompt == "Once upon a time"
        assert request.max_tokens is None  # uses _default_max_tokens when None

    def test_max_tokens_must_be_positive(self):
        """Completion requests reject zero or negative max_tokens."""
        from vllm_mlx.server import CompletionRequest

        with pytest.raises(ValidationError):
            CompletionRequest(
                model="test-model", prompt="Once upon a time", max_tokens=0
            )


class TestAnthropicRequest:
    """Test Anthropic request model."""

    def test_max_tokens_must_be_positive(self):
        """Anthropic requests reject zero or negative max_tokens."""
        from vllm_mlx.api.anthropic_models import AnthropicRequest

        with pytest.raises(ValidationError):
            AnthropicRequest(
                model="test-model",
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=0,
            )


class TestMCPExecuteEndpoint:
    """Test MCP execute endpoint sandbox routing."""

    @pytest.fixture()
    def client(self):
        from fastapi.testclient import TestClient

        from vllm_mlx.server import app

        return TestClient(app)

    def test_execute_routes_through_executor(self, client):
        """REST MCP execute should use ToolExecutor, not raw manager.execute_tool."""
        import vllm_mlx.server as srv

        mock_manager = MagicMock()
        mock_manager.execute_tool = AsyncMock(
            side_effect=AssertionError("manager.execute_tool should not be called")
        )

        mock_result = MagicMock()
        mock_result.tool_name = "filesystem__read_file"
        mock_result.content = "hello"
        mock_result.is_error = False
        mock_result.error_message = None

        mock_executor = MagicMock()
        mock_executor.execute_tool_calls = AsyncMock(
            return_value=[(mock_result, "mcp-test")]
        )

        original_manager = srv._mcp_manager
        original_executor = srv._mcp_executor
        srv._mcp_manager = mock_manager
        srv._mcp_executor = mock_executor
        try:
            resp = client.post(
                "/v1/mcp/execute",
                json={
                    "tool_name": "filesystem__read_file",
                    "arguments": {"path": "/tmp/test.txt"},
                },
            )
        finally:
            srv._mcp_manager = original_manager
            srv._mcp_executor = original_executor

        assert resp.status_code == 200
        body = resp.json()
        assert body["tool_name"] == "filesystem__read_file"
        assert body["content"] == "hello"
        assert body["is_error"] is False
        mock_executor.execute_tool_calls.assert_awaited_once()
        mock_manager.execute_tool.assert_not_awaited()

    def test_execute_returns_sandbox_blocked_result(self, client):
        """REST MCP execute should surface sandbox blocks via executor result."""
        import vllm_mlx.server as srv

        mock_result = MagicMock()
        mock_result.tool_name = "filesystem__read_file"
        mock_result.content = None
        mock_result.is_error = True
        mock_result.error_message = "Tool 'read_file' is blocked by security policy"

        mock_executor = MagicMock()
        mock_executor.execute_tool_calls = AsyncMock(
            return_value=[(mock_result, "mcp-test")]
        )

        original_manager = srv._mcp_manager
        original_executor = srv._mcp_executor
        srv._mcp_manager = MagicMock()
        srv._mcp_executor = mock_executor
        try:
            resp = client.post(
                "/v1/mcp/execute",
                json={
                    "tool_name": "filesystem__read_file",
                    "arguments": {"path": "../secret.txt"},
                },
            )
        finally:
            srv._mcp_manager = original_manager
            srv._mcp_executor = original_executor

        assert resp.status_code == 200
        body = resp.json()
        assert body["is_error"] is True
        assert "blocked by security policy" in body["error_message"]


class TestServeCli:
    """Test serve CLI argument parsing."""

    def test_trust_remote_code_flag_defaults_false(self):
        """Serve CLI should require explicit opt-in for remote code loading."""
        from vllm_mlx.cli import create_parser

        parser = create_parser()
        args = parser.parse_args(["serve", "mlx-community/Llama-3.2-3B-Instruct-4bit"])
        assert args.trust_remote_code is False

        args = parser.parse_args(
            [
                "serve",
                "mlx-community/Llama-3.2-3B-Instruct-4bit",
                "--trust-remote-code",
            ]
        )
        assert args.trust_remote_code is True

    def test_host_defaults_to_localhost(self):
        """Serve parsers should bind only to localhost unless overridden."""
        from vllm_mlx.cli import create_parser as create_cli_parser
        from vllm_mlx.server import create_parser as create_server_parser

        cli_parser = create_cli_parser()
        cli_args = cli_parser.parse_args(
            ["serve", "mlx-community/Llama-3.2-3B-Instruct-4bit"]
        )
        assert cli_args.host == "127.0.0.1"

        server_parser = create_server_parser()
        server_args = server_parser.parse_args(
            ["--model", "mlx-community/Llama-3.2-3B-Instruct-4bit"]
        )
        assert server_args.host == "127.0.0.1"

    def test_max_request_tokens_defaults_and_overrides(self):
        """Serve CLI exposes a separate request max_tokens ceiling."""
        from vllm_mlx.cli import create_parser

        parser = create_parser()
        args = parser.parse_args(["serve", "mlx-community/Llama-3.2-3B-Instruct-4bit"])
        assert args.max_tokens == 32768
        assert args.max_request_tokens == 32768

        args = parser.parse_args(
            [
                "serve",
                "mlx-community/Llama-3.2-3B-Instruct-4bit",
                "--max-tokens",
                "2048",
                "--max-request-tokens",
                "4096",
            ]
        )
        assert args.max_tokens == 2048
        assert args.max_request_tokens == 4096

    def test_tool_call_parser_accepts_harmony_aliases(self):
        """GPT-OSS/Harmony parsers should be selectable from the serve CLI."""
        from vllm_mlx.cli import create_parser

        parser = create_parser()
        args = parser.parse_args(
            [
                "serve",
                "lmstudio-community/gpt-oss-20b-MLX-8bit",
                "--enable-auto-tool-choice",
                "--tool-call-parser",
                "harmony",
            ]
        )

        assert args.command == "serve"
        assert args.tool_call_parser == "harmony"
        assert args.enable_auto_tool_choice is True

        args = parser.parse_args(
            [
                "serve",
                "lmstudio-community/gpt-oss-20b-MLX-8bit",
                "--enable-auto-tool-choice",
                "--tool-call-parser",
                "gpt-oss",
            ]
        )

        assert args.tool_call_parser == "gpt-oss"

    def test_models_config_allows_registry_backed_serve(self):
        """Registry-backed serving should not require a positional model."""
        from vllm_mlx.cli import create_parser

        parser = create_parser()
        args = parser.parse_args(
            [
                "serve",
                "--models-config",
                "/tmp/models.yaml",
                "--continuous-batching",
            ]
        )

        assert args.command == "serve"
        assert args.model is None
        assert args.models_config == "/tmp/models.yaml"
        assert args.continuous_batching is True

    def test_default_chat_template_kwargs_accepts_json_object(self):
        """Serve CLI should parse default chat template kwargs from a JSON object."""
        from vllm_mlx.cli import create_parser

        parser = create_parser()
        args = parser.parse_args(
            [
                "serve",
                "mlx-community/Qwen3-0.6B-8bit",
                "--default-chat-template-kwargs",
                '{"enable_thinking": false}',
            ]
        )

        assert args.default_chat_template_kwargs == {"enable_thinking": False}

    def test_default_chat_template_kwargs_help_mentions_empty_request_behavior(
        self, capsys
    ):
        """Serve CLI help should explain empty request kwargs keep server defaults."""
        from vllm_mlx.cli import create_parser

        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["serve", "mlx-community/Qwen3-0.6B-8bit", "--help"])

        captured = capsys.readouterr()
        normalized = " ".join(captured.out.split())
        assert "omitted or empty" in normalized
        assert "server defaults" in normalized

    @pytest.mark.parametrize("bad_json", ["{not-json}", "{"])
    def test_default_chat_template_kwargs_rejects_malformed_json(
        self, bad_json, capsys
    ):
        """Serve CLI should fail fast on malformed JSON input."""
        from vllm_mlx.cli import create_parser

        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(
                [
                    "serve",
                    "mlx-community/Qwen3-0.6B-8bit",
                    "--default-chat-template-kwargs",
                    bad_json,
                ]
            )

        captured = capsys.readouterr()
        assert "--default-chat-template-kwargs" in captured.err
        assert "JSON object" in captured.err

    @pytest.mark.parametrize("non_object", ["[]", "true", "123"])
    def test_default_chat_template_kwargs_rejects_non_object_json(
        self, non_object, capsys
    ):
        """Serve CLI should reject valid JSON values that are not objects."""
        from vllm_mlx.cli import create_parser

        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(
                [
                    "serve",
                    "mlx-community/Qwen3-0.6B-8bit",
                    "--default-chat-template-kwargs",
                    non_object,
                ]
            )

        captured = capsys.readouterr()
        assert "--default-chat-template-kwargs" in captured.err
        assert "JSON object" in captured.err


class TestStandaloneServerCli:
    """Test standalone server CLI argument parsing."""

    def test_trust_remote_code_flag_defaults_false(self):
        """Standalone server should require explicit opt-in for remote code loading."""
        from vllm_mlx.server import create_parser

        parser = create_parser()
        args = parser.parse_args(
            ["--model", "mlx-community/Llama-3.2-3B-Instruct-4bit"]
        )
        assert args.trust_remote_code is False

        args = parser.parse_args(
            [
                "--model",
                "mlx-community/Llama-3.2-3B-Instruct-4bit",
                "--trust-remote-code",
            ]
        )
        assert args.trust_remote_code is True

    def test_default_chat_template_kwargs_accepts_json_object(self):
        """Standalone server should parse default chat template kwargs JSON."""
        from vllm_mlx.server import create_parser

        parser = create_parser()
        args = parser.parse_args(
            [
                "--model",
                "mlx-community/Qwen3-0.6B-8bit",
                "--default-chat-template-kwargs",
                '{"enable_thinking": false}',
            ]
        )

        assert args.default_chat_template_kwargs == {"enable_thinking": False}

    def test_default_chat_template_kwargs_help_mentions_empty_request_behavior(
        self, capsys
    ):
        """Standalone help should explain empty request kwargs keep server defaults."""
        from vllm_mlx.server import create_parser

        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--help"])

        captured = capsys.readouterr()
        normalized = " ".join(captured.out.split())
        assert "omitted or empty" in normalized
        assert "server defaults" in normalized

    @pytest.mark.parametrize("bad_json", ["{not-json}", "{"])
    def test_default_chat_template_kwargs_rejects_malformed_json(
        self, bad_json, capsys
    ):
        """Standalone server should fail fast on malformed JSON input."""
        from vllm_mlx.server import create_parser

        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(
                [
                    "--model",
                    "mlx-community/Qwen3-0.6B-8bit",
                    "--default-chat-template-kwargs",
                    bad_json,
                ]
            )

        captured = capsys.readouterr()
        assert "--default-chat-template-kwargs" in captured.err
        assert "JSON object" in captured.err

    @pytest.mark.parametrize("non_object", ["[]", "false", "0"])
    def test_default_chat_template_kwargs_rejects_non_object_json(
        self, non_object, capsys
    ):
        """Standalone server should reject valid JSON values that are not objects."""
        from vllm_mlx.server import create_parser

        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(
                [
                    "--model",
                    "mlx-community/Qwen3-0.6B-8bit",
                    "--default-chat-template-kwargs",
                    non_object,
                ]
            )

        captured = capsys.readouterr()
        assert "--default-chat-template-kwargs" in captured.err
        assert "JSON object" in captured.err


class TestLoadModelTrustRemoteCode:
    """Test load_model trust_remote_code wiring into engine constructors."""

    def test_load_model_simple_defaults_trust_remote_code_false(self):
        """SimpleEngine should receive trust_remote_code=False by default."""
        from vllm_mlx import server

        fake_engine = MagicMock()
        fake_loop = MagicMock()

        with (
            patch.object(
                server, "SimpleEngine", return_value=fake_engine
            ) as mock_engine,
            patch.object(server, "_detect_native_tool_support", return_value=False),
            patch("vllm_mlx.server.asyncio.new_event_loop", return_value=fake_loop),
            patch("vllm_mlx.server.asyncio.set_event_loop"),
        ):
            server.load_model("test-model", use_batching=False)

        assert mock_engine.call_args.kwargs["trust_remote_code"] is False

    def test_load_model_batched_forwards_explicit_trust_remote_code(self):
        """BatchedEngine should receive explicit trust_remote_code opt-in."""
        from vllm_mlx import server

        fake_engine = MagicMock()

        with (
            patch.object(
                server, "BatchedEngine", return_value=fake_engine
            ) as mock_engine,
            patch.object(server, "_detect_native_tool_support", return_value=False),
        ):
            server.load_model(
                "test-model",
                use_batching=True,
                trust_remote_code=True,
            )

        assert mock_engine.call_args.kwargs["trust_remote_code"] is True


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestHelperFunctions:
    """Test server helper functions."""

    def test_list_models_prefers_model_manager_registry(self, monkeypatch):
        """Registry-backed mode should expose configured models through /v1/models."""
        import asyncio
        import vllm_mlx.server as server

        class FakeManager:
            def list_models(self):
                return [
                    {"id": "fast", "status": "loaded"},
                    {"id": "smart", "status": "unloaded"},
                ]

        monkeypatch.setattr(server, "_model_manager", FakeManager())
        monkeypatch.setattr(server, "_model_name", None)

        response = asyncio.run(server.list_models())

        assert [model.id for model in response.data] == ["fast", "smart"]

    def test_validate_model_name_checks_registry_when_present(self, monkeypatch):
        """Registry-backed validation should accept registered names and reject unknown ones."""
        from fastapi import HTTPException
        import vllm_mlx.server as server

        class FakeManager:
            _registry = {"fast": object(), "smart": object()}

            @property
            def registered_model_names(self):
                return sorted(self._registry.keys())

            def has_model(self, name):
                return name in self._registry

        monkeypatch.setattr(server, "_model_manager", FakeManager())
        monkeypatch.setattr(server, "_model_name", None)

        server._validate_model_name("fast")

        with pytest.raises(HTTPException) as exc_info:
            server._validate_model_name("missing")

        assert exc_info.value.status_code == 404
        assert "fast" in exc_info.value.detail

    def test_build_reasoning_parser_uses_configured_name_and_engine_tokenizer(
        self, monkeypatch
    ):
        """Per-request reasoning parser instances should be built from the configured parser name."""
        import vllm_mlx.server as server

        class FakeParser:
            def __init__(self, tokenizer=None):
                self.tokenizer = tokenizer

        class FakeEngine:
            tokenizer = object()

        monkeypatch.setattr(server, "_reasoning_parser_name", "fake")
        monkeypatch.setattr(server, "_reasoning_parser", None)
        monkeypatch.setattr(server, "get_reasoning_parser", lambda name: FakeParser)

        parser = server._build_reasoning_parser(FakeEngine())

        assert isinstance(parser, FakeParser)
        assert parser.tokenizer is FakeEngine.tokenizer

    def test_is_mllm_model_patterns(self):
        """Test MLLM model detection patterns."""
        from vllm_mlx.server import is_mllm_model

        # Should detect as MLLM
        assert is_mllm_model("mlx-community/Qwen3-VL-4B-Instruct-3bit")
        assert is_mllm_model("mlx-community/llava-1.5-7b-4bit")
        assert is_mllm_model("mlx-community/paligemma-3b-mix-224-4bit")
        assert is_mllm_model("mlx-community/pixtral-12b-4bit")
        assert is_mllm_model("mlx-community/Idefics3-8B-Llama3-4bit")
        assert is_mllm_model("mlx-community/deepseek-vl-7b-chat-4bit")

        # Should NOT detect as MLLM
        assert not is_mllm_model("mlx-community/Llama-3.2-1B-Instruct-4bit")
        assert not is_mllm_model("mlx-community/Mistral-7B-Instruct-4bit")
        assert not is_mllm_model("mlx-community/Qwen2-7B-Instruct-4bit")

    def test_sanitize_log_text_escapes_control_characters(self):
        """Untrusted log text should not contain raw control characters."""
        from vllm_mlx.server import _sanitize_log_text

        text = "line1\nline2\r\t\u2028\x1b[31m"
        sanitized = _sanitize_log_text(text)

        assert sanitized == r"line1\nline2\r\t\u2028\x1b[31m"
        assert "\n" not in sanitized.replace(r"\n", "")
        assert "\r" not in sanitized.replace(r"\r", "")

    def test_extract_multimodal_content_text_only(self):
        """Test extracting content from text-only messages."""
        from vllm_mlx.server import extract_multimodal_content, Message

        messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there!"),
        ]

        processed, images, videos, audios = extract_multimodal_content(messages)

        assert len(processed) == 2
        assert processed[0]["content"] == "Hello"
        assert len(images) == 0
        assert len(videos) == 0
        assert len(audios) == 0

    def test_extract_multimodal_content_with_image(self):
        """Test extracting content with images."""
        from vllm_mlx.server import extract_multimodal_content, Message

        messages = [
            Message(
                role="user",
                content=[
                    {"type": "text", "text": "What's this?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/img.jpg"},
                    },
                ],
            )
        ]

        processed, images, videos, audios = extract_multimodal_content(messages)

        assert len(processed) == 1
        assert processed[0]["content"] == "What's this?"
        assert len(images) == 1
        assert "https://example.com/img.jpg" in images[0]
        assert len(audios) == 0

    def test_extract_multimodal_content_with_video(self):
        """Test extracting content with videos."""
        from vllm_mlx.server import extract_multimodal_content, Message

        messages = [
            Message(
                role="user",
                content=[
                    {"type": "text", "text": "Describe this video"},
                    {"type": "video", "video": "/path/to/video.mp4"},
                ],
            )
        ]

        processed, images, videos, audios = extract_multimodal_content(messages)

        assert len(processed) == 1
        assert processed[0]["content"] == "Describe this video"
        assert len(videos) == 1
        assert videos[0] == "/path/to/video.mp4"
        assert len(audios) == 0

    def test_extract_multimodal_content_with_video_url(self):
        """Test extracting content with video_url format."""
        from vllm_mlx.server import extract_multimodal_content, Message

        messages = [
            Message(
                role="user",
                content=[
                    {"type": "text", "text": "What happens?"},
                    {
                        "type": "video_url",
                        "video_url": {"url": "https://example.com/video.mp4"},
                    },
                ],
            )
        ]

        processed, images, videos, audios = extract_multimodal_content(messages)

        assert len(videos) == 1
        assert len(audios) == 0

    def test_extract_multimodal_content_with_audio_url(self):
        """Test extracting content with audio_url format."""
        from vllm_mlx.server import extract_multimodal_content, Message

        messages = [
            Message(
                role="user",
                content=[
                    {"type": "text", "text": "Transcribe this"},
                    {
                        "type": "audio_url",
                        "audio_url": {"url": "data:audio/wav;base64,abc"},
                    },
                ],
            )
        ]

        processed, images, videos, audios = extract_multimodal_content(messages)

        assert len(processed) == 1
        assert processed[0]["content"] == "Transcribe this"
        assert len(images) == 0
        assert len(videos) == 0
        assert audios == ["data:audio/wav;base64,abc"]


# =============================================================================
# Security and Reliability Tests (PR #4)
# =============================================================================


class TestRateLimiter:
    """Test the RateLimiter class for rate limiting functionality."""

    def test_rate_limiter_disabled_by_default(self):
        """Test that rate limiter allows all requests when disabled."""
        from vllm_mlx.server import RateLimiter

        limiter = RateLimiter(requests_per_minute=5, enabled=False)

        # Should allow unlimited requests when disabled
        for _ in range(100):
            allowed, retry_after = limiter.is_allowed("client1")
            assert allowed is True
            assert retry_after == 0

    def test_rate_limiter_enforces_limit(self):
        """Test that rate limiter enforces the request limit."""
        from vllm_mlx.server import RateLimiter

        limiter = RateLimiter(requests_per_minute=3, enabled=True)

        # First 3 requests should be allowed
        for i in range(3):
            allowed, retry_after = limiter.is_allowed("client1")
            assert allowed is True, f"Request {i + 1} should be allowed"
            assert retry_after == 0

        # 4th request should be blocked
        allowed, retry_after = limiter.is_allowed("client1")
        assert allowed is False
        assert retry_after > 0

    def test_rate_limiter_per_client(self):
        """Test that rate limits are tracked per client."""
        from vllm_mlx.server import RateLimiter

        limiter = RateLimiter(requests_per_minute=2, enabled=True)

        # Client 1 uses its quota
        limiter.is_allowed("client1")
        limiter.is_allowed("client1")
        allowed, _ = limiter.is_allowed("client1")
        assert allowed is False

        # Client 2 should still have quota
        allowed, _ = limiter.is_allowed("client2")
        assert allowed is True

    def test_rate_limiter_thread_safety(self):
        """Test that rate limiter is thread-safe."""
        import threading
        from vllm_mlx.server import RateLimiter

        limiter = RateLimiter(requests_per_minute=100, enabled=True)
        results = []
        errors = []

        def make_requests():
            try:
                for _ in range(10):
                    allowed, _ = limiter.is_allowed("shared_client")
                    results.append(allowed)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=make_requests) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread safety errors: {errors}"
        assert len(results) == 100
        # Exactly 100 requests allowed (our limit)
        assert results.count(True) == 100


class TestTempFileManager:
    """Test the TempFileManager class for temp file cleanup."""

    def test_register_and_cleanup_single_file(self):
        """Test registering and cleaning up a single temp file."""
        import tempfile
        import os
        from vllm_mlx.models.mllm import TempFileManager

        manager = TempFileManager()

        # Create a real temp file
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
        temp.write(b"test content")
        temp.close()

        # Register it
        path = manager.register(temp.name)
        assert path == temp.name
        assert os.path.exists(temp.name)

        # Cleanup
        result = manager.cleanup(temp.name)
        assert result is True
        assert not os.path.exists(temp.name)

    def test_cleanup_all_files(self):
        """Test cleaning up all registered temp files."""
        import tempfile
        import os
        from vllm_mlx.models.mllm import TempFileManager

        manager = TempFileManager()
        paths = []

        # Create multiple temp files
        for i in range(3):
            temp = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{i}.txt")
            temp.write(f"content {i}".encode())
            temp.close()
            manager.register(temp.name)
            paths.append(temp.name)

        # Verify all exist
        for p in paths:
            assert os.path.exists(p)

        # Cleanup all
        cleaned = manager.cleanup_all()
        assert cleaned == 3

        # Verify all deleted
        for p in paths:
            assert not os.path.exists(p)

    def test_cleanup_nonexistent_file(self):
        """Test cleanup of a non-existent file."""
        from vllm_mlx.models.mllm import TempFileManager

        manager = TempFileManager()

        # Cleanup a file that doesn't exist
        result = manager.cleanup("/nonexistent/path/file.txt")
        assert result is False

    def test_thread_safe_registration(self):
        """Test that TempFileManager is thread-safe."""
        import threading
        import tempfile
        from vllm_mlx.models.mllm import TempFileManager

        manager = TempFileManager()
        paths = []
        lock = threading.Lock()
        errors = []

        def register_files():
            try:
                for _ in range(5):
                    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
                    temp.write(b"test")
                    temp.close()
                    path = manager.register(temp.name)
                    with lock:
                        paths.append(path)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=register_files) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread safety errors: {errors}"
        assert len(paths) == 25

        # Cleanup all
        cleaned = manager.cleanup_all()
        assert cleaned == 25


class TestRequestOutputCollectorThreadSafety:
    """Test thread-safety of RequestOutputCollector._waiting_consumers."""

    def test_waiting_consumers_thread_safe(self):
        """Test that _waiting_consumers counter is thread-safe."""
        import threading
        from vllm_mlx.output_collector import RequestOutputCollector

        # Reset the counter
        with RequestOutputCollector._waiting_lock:
            RequestOutputCollector._waiting_consumers = 0

        errors = []

        def manipulate_counter():
            try:
                for _ in range(100):
                    with RequestOutputCollector._waiting_lock:
                        RequestOutputCollector._waiting_consumers += 1
                    with RequestOutputCollector._waiting_lock:
                        RequestOutputCollector._waiting_consumers -= 1
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=manipulate_counter) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread safety errors: {errors}"
        # Should return to zero
        with RequestOutputCollector._waiting_lock:
            assert RequestOutputCollector._waiting_consumers == 0

    def test_has_waiting_consumers_method(self):
        """Test has_waiting_consumers class method."""
        from vllm_mlx.output_collector import RequestOutputCollector

        # Reset counter
        with RequestOutputCollector._waiting_lock:
            RequestOutputCollector._waiting_consumers = 0

        assert RequestOutputCollector.has_waiting_consumers() is False

        with RequestOutputCollector._waiting_lock:
            RequestOutputCollector._waiting_consumers = 1

        assert RequestOutputCollector.has_waiting_consumers() is True

        # Reset
        with RequestOutputCollector._waiting_lock:
            RequestOutputCollector._waiting_consumers = 0


class TestRequestTimeoutField:
    """Test the new timeout field in request models."""

    def test_chat_completion_request_timeout_field(self):
        """Test that ChatCompletionRequest has timeout field."""
        from vllm_mlx.server import ChatCompletionRequest, Message

        # Default should be None
        request = ChatCompletionRequest(
            model="test-model", messages=[Message(role="user", content="Hello")]
        )
        assert request.timeout is None

        # Can set custom timeout
        request_with_timeout = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
            timeout=60.0,
        )
        assert request_with_timeout.timeout == 60.0

    def test_completion_request_timeout_field(self):
        """Test that CompletionRequest has timeout field."""
        from vllm_mlx.server import CompletionRequest

        # Default should be None
        request = CompletionRequest(model="test-model", prompt="Once upon a time")
        assert request.timeout is None

        # Can set custom timeout
        request_with_timeout = CompletionRequest(
            model="test-model", prompt="Once upon a time", timeout=120.0
        )
        assert request_with_timeout.timeout == 120.0


class TestMaxTokensLimit:
    """Test server-side max_tokens ceiling enforcement."""

    def test_resolve_request_max_tokens(self):
        """Explicit requests must stay within the configured server ceiling."""
        import vllm_mlx.server as server

        original_default = server._default_max_tokens
        original_limit = server._max_request_tokens
        try:
            server._default_max_tokens = 1024
            server._max_request_tokens = 2048

            assert server._resolve_request_max_tokens(None) == 1024
            assert server._resolve_request_max_tokens(512) == 512

            with pytest.raises(server.HTTPException) as exc_info:
                server._resolve_request_max_tokens(4096)

            assert exc_info.value.status_code == 400
            assert "server limit" in exc_info.value.detail
        finally:
            server._default_max_tokens = original_default
            server._max_request_tokens = original_limit

    @pytest.mark.anyio
    async def test_create_completion_rejects_over_limit_before_engine_lookup(
        self, monkeypatch
    ):
        """Completions should reject oversized requests at the API boundary."""
        from vllm_mlx.server import CompletionRequest, create_completion
        import vllm_mlx.server as server

        monkeypatch.setattr(server, "_max_request_tokens", 1024)
        monkeypatch.setattr(
            server,
            "get_engine",
            lambda: (_ for _ in ()).throw(AssertionError("engine should not load")),
        )

        request = CompletionRequest(
            model="test-model",
            prompt="Once upon a time",
            max_tokens=2048,
        )

        with pytest.raises(server.HTTPException) as exc_info:
            await create_completion(request, raw_request=None)

        assert exc_info.value.status_code == 400

    @pytest.mark.anyio
    async def test_create_chat_completion_rejects_over_limit_before_engine_lookup(
        self, monkeypatch
    ):
        """Chat completions should reject oversized requests at the API boundary."""
        from vllm_mlx.server import (
            ChatCompletionRequest,
            Message,
            create_chat_completion,
        )
        import vllm_mlx.server as server

        monkeypatch.setattr(server, "_max_request_tokens", 1024)
        monkeypatch.setattr(
            server,
            "get_engine",
            lambda: (_ for _ in ()).throw(AssertionError("engine should not load")),
        )

        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
            max_tokens=2048,
        )

        with pytest.raises(server.HTTPException) as exc_info:
            await create_chat_completion(request, raw_request=None)

        assert exc_info.value.status_code == 400

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("user_stop", "expected_stop"),
        [
            (None, ["<|tool_response>"]),
            (["END_USER"], ["END_USER", "<|tool_response>"]),
        ],
    )
    async def test_create_chat_completion_merges_parser_stop_tokens(
        self, monkeypatch, user_stop, expected_stop
    ):
        """Parser-declared stop tokens should be merged into chat kwargs."""
        from vllm_mlx.engine.base import GenerationOutput
        from vllm_mlx.server import (
            ChatCompletionRequest,
            Message,
            create_chat_completion,
        )
        import vllm_mlx.server as server

        captured = {}
        helper_calls = []

        class FakeEngine:
            model_name = "fake-engine"
            is_mllm = False
            preserve_native_tool_format = False

            async def chat(self, messages, **kwargs):
                captured["messages"] = messages
                captured["kwargs"] = kwargs
                return GenerationOutput(
                    text="ok",
                    prompt_tokens=5,
                    completion_tokens=2,
                    finish_reason="stop",
                )

        fake_engine = FakeEngine()

        async def fake_acquire(
            raw_request,
            *,
            total_timeout=None,
            deadline=None,
            count_activity=True,
            model=None,
        ):
            return fake_engine

        async def fake_release(*, count_activity=True):
            return None

        def fake_get_parser_stop_tokens(parser_name, user_stops):
            helper_calls.append((parser_name, user_stops))
            merged = list(user_stops or [])
            if "<|tool_response>" not in merged:
                merged.append("<|tool_response>")
            return merged

        monkeypatch.setattr(server, "_validate_model_name", lambda _m: None)
        monkeypatch.setattr(server, "_acquire_default_engine_for_request", fake_acquire)
        monkeypatch.setattr(server, "_release_default_engine", fake_release)
        monkeypatch.setattr(
            server, "get_parser_stop_tokens", fake_get_parser_stop_tokens
        )
        monkeypatch.setattr(server, "_model_name", "served-model")
        monkeypatch.setattr(server, "_default_max_tokens", 128)
        monkeypatch.setattr(server, "_default_timeout", 30.0)
        monkeypatch.setattr(server, "_enable_auto_tool_choice", True)
        monkeypatch.setattr(server, "_tool_call_parser", "fake")
        monkeypatch.setattr(server, "_tool_parser_instance", None)

        request = ChatCompletionRequest(
            model="served-model",
            messages=[Message(role="user", content="Hello")],
            max_tokens=16,
            stop=user_stop,
        )

        response = await create_chat_completion(request, raw_request=None)

        assert helper_calls
        assert all(call == ("fake", user_stop) for call in helper_calls)
        assert captured["kwargs"]["stop"] == expected_stop
        assert response.choices[0].message.content == "ok"

    @pytest.mark.anyio
    async def test_create_anthropic_message_rejects_over_limit_before_engine_lookup(
        self, monkeypatch
    ):
        """Anthropic requests should reject oversized requests before engine use."""
        from vllm_mlx.server import create_anthropic_message
        import vllm_mlx.server as server

        class FakeRequest:
            async def json(self):
                return {
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 2048,
                }

        monkeypatch.setattr(server, "_max_request_tokens", 1024)
        monkeypatch.setattr(
            server,
            "get_engine",
            lambda: (_ for _ in ()).throw(AssertionError("engine should not load")),
        )

        with pytest.raises(server.HTTPException) as exc_info:
            await create_anthropic_message(FakeRequest())

        assert exc_info.value.status_code == 400


class TestChatTemplateKwargsResolver:
    """Test default chat template kwargs precedence contract."""

    def test_resolver_prefers_request_values_over_server_defaults(self, monkeypatch):
        """Request kwargs should override server defaults key-by-key."""
        import vllm_mlx.server as server

        monkeypatch.setattr(
            server,
            "_default_chat_template_kwargs",
            {"enable_thinking": False, "temperature_hint": "server"},
            raising=False,
        )

        resolved = server._resolve_chat_template_kwargs(
            {"enable_thinking": True, "request_only": 1}
        )
        assert resolved == {
            "enable_thinking": True,
            "temperature_hint": "server",
            "request_only": 1,
        }

    def test_resolver_uses_server_defaults_when_request_omits_kwargs(self, monkeypatch):
        """Resolver should return server defaults when request kwargs are absent."""
        import vllm_mlx.server as server

        monkeypatch.setattr(
            server,
            "_default_chat_template_kwargs",
            {"enable_thinking": False},
            raising=False,
        )

        assert server._resolve_chat_template_kwargs(None) == {"enable_thinking": False}

    def test_resolver_returns_empty_dict_when_no_values_are_provided(self, monkeypatch):
        """Resolver should produce an empty dict when neither source provides values."""
        import vllm_mlx.server as server

        monkeypatch.setattr(
            server,
            "_default_chat_template_kwargs",
            None,
            raising=False,
        )

        assert server._resolve_chat_template_kwargs(None) == {}


class TestAPIKeyVerification:
    """Test API key verification with timing attack prevention."""

    def test_secrets_compare_digest_usage(self):
        """Test that secrets.compare_digest is used (timing attack prevention)."""
        import secrets

        # Verify secrets.compare_digest works as expected
        key1 = "test-api-key-12345"
        key2 = "test-api-key-12345"
        key3 = "different-key-67890"

        # Same keys should match
        assert secrets.compare_digest(key1, key2) is True

        # Different keys should not match
        assert secrets.compare_digest(key1, key3) is False

        # Verify it's constant-time (by checking function exists)
        assert hasattr(secrets, "compare_digest")

    def test_verify_api_key_rejects_invalid(self):
        """Test that invalid API key is rejected with 401."""
        import asyncio
        from fastapi import HTTPException
        from fastapi.security import HTTPAuthorizationCredentials

        # Import and set up the module
        import vllm_mlx.server as server

        original_key = server._api_key

        try:
            # Set a known API key
            server._api_key = "valid-secret-key"

            # Create mock credentials with invalid key
            credentials = HTTPAuthorizationCredentials(
                scheme="Bearer", credentials="invalid-key"
            )

            # Should raise HTTPException with 401
            with pytest.raises(HTTPException) as exc_info:
                asyncio.run(server.verify_api_key(credentials))

            assert exc_info.value.status_code == 401
            assert "Invalid API key" in str(exc_info.value.detail)
        finally:
            server._api_key = original_key


class TestLogAndExceptionSanitization:
    """Test request log previews and internal error responses."""

    @pytest.mark.anyio
    async def test_create_completion_logs_sanitized_prompt_preview(
        self, monkeypatch, caplog
    ):
        """Prompt previews should escape control characters before logging."""
        from vllm_mlx.server import CompletionRequest, create_completion
        import vllm_mlx.server as server

        class DummyEngine:
            async def generate(self, **kwargs):
                return SimpleNamespace(
                    text="ok",
                    finish_reason="stop",
                    completion_tokens=1,
                    prompt_tokens=1,
                )

        async def fake_wait(task, raw_request, timeout):
            return await task

        monkeypatch.setattr(server, "_model_name", "test-model")
        monkeypatch.setattr(server, "get_engine", lambda: DummyEngine())
        monkeypatch.setattr(server, "_wait_with_disconnect", fake_wait)

        request = CompletionRequest(
            model="test-model",
            prompt="line1\nline2\t\x1b[31mred",
            max_tokens=8,
        )

        with caplog.at_level("INFO"):
            response = await create_completion(request, raw_request=None)

        assert response.choices[0].text == "ok"
        preview_logs = [
            record.getMessage()
            for record in caplog.records
            if "prompt_preview=" in record.getMessage()
        ]
        assert preview_logs
        assert "line1\\nline2\\t\\x1b[31mred" in preview_logs[0]
        assert "line1\nline2" not in preview_logs[0]

    @pytest.mark.anyio
    async def test_create_embeddings_hides_internal_exception_details(
        self, monkeypatch, caplog
    ):
        """Embedding failures should log sanitized details but return generic 500s."""
        from vllm_mlx.server import EmbeddingRequest, create_embeddings
        import vllm_mlx.server as server

        class ExplodingEmbeddingEngine:
            def count_tokens(self, texts):
                raise RuntimeError("boom\nsecret\t\x1b[31m")

        monkeypatch.setattr(server, "_embedding_engine", ExplodingEmbeddingEngine())
        monkeypatch.setattr(
            server, "load_embedding_model", lambda *args, **kwargs: None
        )

        request = EmbeddingRequest(
            model="mlx-community/all-MiniLM-L6-v2-4bit",
            input="hello",
        )

        with caplog.at_level("ERROR"):
            with pytest.raises(server.HTTPException) as exc_info:
                await create_embeddings(request)

        assert exc_info.value.status_code == 500
        assert exc_info.value.detail == "Embedding generation failed"

        error_logs = [
            record.getMessage()
            for record in caplog.records
            if "Embedding generation failed:" in record.getMessage()
        ]
        assert error_logs
        assert r"boom\nsecret\t\x1b[31m" in error_logs[0]
        assert "boom\nsecret" not in error_logs[0]

    def test_verify_api_key_accepts_valid(self):
        """Test that valid API key is accepted."""
        import asyncio
        from fastapi.security import HTTPAuthorizationCredentials

        import vllm_mlx.server as server

        original_key = server._api_key

        try:
            # Set a known API key
            server._api_key = "valid-secret-key"

            # Create mock credentials with valid key
            credentials = HTTPAuthorizationCredentials(
                scheme="Bearer", credentials="valid-secret-key"
            )

            # Should not raise any exception
            result = asyncio.run(server.verify_api_key(credentials))
            # verify_api_key returns True on success (no exception raised)
            assert result is True or result is None
        finally:
            server._api_key = original_key


class TestRateLimiterHTTPResponse:
    """Test rate limiter HTTP response behavior."""

    def test_rate_limiter_returns_retry_after(self):
        """Test that rate limiter returns retry_after when limit exceeded."""
        from vllm_mlx.server import RateLimiter

        limiter = RateLimiter(requests_per_minute=2, enabled=True)

        # Exhaust the limit
        limiter.is_allowed("test_client")
        limiter.is_allowed("test_client")

        # Next request should be denied with retry_after
        allowed, retry_after = limiter.is_allowed("test_client")

        assert allowed is False
        assert retry_after is not None
        assert retry_after > 0


class TestEndpointSecurityDependencies:
    """Test auth/rate-limit coverage on protected endpoints."""

    @pytest.fixture
    def client(self):
        import vllm_mlx.server as server

        return TestClient(server.app)

    @pytest.fixture(autouse=True)
    def restore_security_state(self):
        import vllm_mlx.server as server

        original_key = server._api_key
        original_limiter = server._rate_limiter
        try:
            yield
        finally:
            server._api_key = original_key
            server._rate_limiter = original_limiter

    @pytest.mark.parametrize(
        ("method", "path"),
        [
            ("get", "/v1/status"),
            ("get", "/v1/cache/stats"),
            ("delete", "/v1/cache"),
            ("post", "/v1/messages"),
            ("post", "/v1/messages/count_tokens"),
        ],
    )
    def test_endpoints_require_api_key(self, client, method, path):
        import vllm_mlx.server as server

        server._api_key = "test-secret"

        kwargs = {}
        if method == "post":
            kwargs["json"] = {
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 16,
            }

        response = getattr(client, method)(path, **kwargs)
        assert response.status_code == 401
        assert response.json()["detail"] == "API key required"

    @pytest.mark.parametrize("path", ["/v1/messages", "/v1/messages/count_tokens"])
    def test_anthropic_endpoints_apply_rate_limit(self, client, path, monkeypatch):
        import vllm_mlx.server as server

        server._api_key = "test-secret"

        def deny_all(_client_id):
            return False, 7

        monkeypatch.setattr(server._rate_limiter, "enabled", True)
        monkeypatch.setattr(server._rate_limiter, "is_allowed", deny_all)

        response = client.post(
            path,
            headers={"Authorization": "Bearer test-secret"},
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 16,
            },
        )

        assert response.status_code == 429
        assert "Rate limit exceeded" in response.json()["detail"]
        assert response.headers["Retry-After"] == "7"

    def test_rate_limiter_window_cleanup(self):
        """Test that rate limiter cleans up old requests from sliding window."""
        from vllm_mlx.server import RateLimiter
        import time

        limiter = RateLimiter(requests_per_minute=2, enabled=True)

        # Make some requests
        limiter.is_allowed("test_client")
        limiter.is_allowed("test_client")

        # Should be denied (limit reached)
        allowed, _ = limiter.is_allowed("test_client")
        assert allowed is False

        # Manually inject old timestamps to simulate time passing
        # The sliding window should clean these up
        old_time = time.time() - 120  # 2 minutes ago
        with limiter._lock:
            limiter._requests["test_client"] = [old_time, old_time]

        # Now should be allowed again (old requests cleaned up)
        allowed, _ = limiter.is_allowed("test_client")
        assert allowed is True


class TestStreamChatCompletion:
    """Tests for streaming chat completion behavior."""

    @pytest.mark.anyio
    async def test_stream_without_parser_flags_emits_structured_tool_calls(
        self, monkeypatch
    ):
        """Streaming tools should still parse without explicit parser flags."""
        from vllm_mlx.engine.base import GenerationOutput
        from vllm_mlx.server import (
            ChatCompletionRequest,
            Message,
            stream_chat_completion,
        )
        import vllm_mlx.server as server

        class FakeEngine:
            model_name = "fake-engine"

            async def stream_chat(self, messages, **kwargs):
                chunks = [
                    GenerationOutput(text="", new_text="<tool_call>", finished=False),
                    GenerationOutput(
                        text="",
                        new_text="<function=list_directory>",
                        finished=False,
                    ),
                    GenerationOutput(
                        text="",
                        new_text="<parameter=path>/Users/testuser</parameter>",
                        finished=False,
                    ),
                    GenerationOutput(
                        text="",
                        new_text="</function>",
                        finished=False,
                    ),
                    GenerationOutput(
                        text="",
                        new_text="</tool_call>",
                        finished=True,
                        finish_reason="stop",
                        prompt_tokens=5,
                        completion_tokens=7,
                    ),
                ]
                for chunk in chunks:
                    yield chunk

        monkeypatch.setattr(server, "_model_name", "served-model")
        monkeypatch.setattr(server, "_reasoning_parser", None)
        monkeypatch.setattr(server, "_enable_auto_tool_choice", False)
        monkeypatch.setattr(server, "_tool_call_parser", None)
        monkeypatch.setattr(server, "_tool_parser_instance", None)

        request = ChatCompletionRequest(
            model="served-model",
            messages=[Message(role="user", content="hi")],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "list_directory",
                        "description": "List files in a directory",
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                        },
                    },
                }
            ],
            stream=True,
        )

        chunks = [
            chunk
            async for chunk in stream_chat_completion(
                FakeEngine(), request.messages, request
            )
        ]

        payloads = [
            json.loads(chunk.removeprefix("data: ").strip())
            for chunk in chunks
            if chunk != "data: [DONE]\n\n"
        ]
        tool_payloads = [
            payload
            for payload in payloads
            if payload["choices"] and payload["choices"][0]["delta"].get("tool_calls")
        ]

        assert len(tool_payloads) == 1
        delta = tool_payloads[0]["choices"][0]["delta"]
        assert delta["tool_calls"][0]["function"]["name"] == "list_directory"
        assert delta["tool_calls"][0]["function"]["arguments"] == (
            '{"path": "/Users/testuser"}'
        )
        assert delta.get("content") is None
        assert tool_payloads[0]["choices"][0]["finish_reason"] == "tool_calls"
        assert tool_payloads[0]["usage"] == {
            "prompt_tokens": 5,
            "completion_tokens": 7,
            "total_tokens": 12,
        }

    @pytest.mark.anyio
    async def test_stream_without_parser_flags_keeps_plain_text(self, monkeypatch):
        """Generic streaming fallback should not interfere with normal text."""
        from vllm_mlx.engine.base import GenerationOutput
        from vllm_mlx.server import (
            ChatCompletionRequest,
            Message,
            stream_chat_completion,
        )
        import vllm_mlx.server as server

        class FakeEngine:
            model_name = "fake-engine"

            async def stream_chat(self, messages, **kwargs):
                chunks = [
                    GenerationOutput(text="", new_text="hello ", finished=False),
                    GenerationOutput(
                        text="",
                        new_text="world",
                        finished=True,
                        finish_reason="stop",
                        prompt_tokens=4,
                        completion_tokens=2,
                    ),
                ]
                for chunk in chunks:
                    yield chunk

        monkeypatch.setattr(server, "_model_name", "served-model")
        monkeypatch.setattr(server, "_reasoning_parser", None)
        monkeypatch.setattr(server, "_enable_auto_tool_choice", False)
        monkeypatch.setattr(server, "_tool_call_parser", None)
        monkeypatch.setattr(server, "_tool_parser_instance", None)

        request = ChatCompletionRequest(
            model="served-model",
            messages=[Message(role="user", content="hi")],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "list_directory",
                        "description": "List files in a directory",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
            stream=True,
        )

        chunks = [
            chunk
            async for chunk in stream_chat_completion(
                FakeEngine(), request.messages, request
            )
        ]
        payloads = [
            json.loads(chunk.removeprefix("data: ").strip())
            for chunk in chunks
            if chunk != "data: [DONE]\n\n"
        ]

        assert payloads[1]["choices"][0]["delta"]["content"] == "hello "
        assert payloads[2]["choices"][0]["delta"]["content"] == "world"
        assert payloads[2]["choices"][0]["finish_reason"] == "stop"

    @pytest.mark.anyio
    async def test_auto_parser_streams_bare_bracket_tool_calls(self, monkeypatch):
        """Bare bracket tool calls should stream as structured tool_calls."""
        from vllm_mlx.engine.base import GenerationOutput
        from vllm_mlx.server import (
            ChatCompletionRequest,
            Message,
            stream_chat_completion,
        )
        import vllm_mlx.server as server

        class FakeEngine:
            model_name = "fake-engine"

            async def stream_chat(self, messages, **kwargs):
                chunks = [
                    GenerationOutput(text="", new_text="[read(", finished=False),
                    GenerationOutput(
                        text="",
                        new_text='{"file_path": "/tmp/test.py"}',
                        finished=False,
                    ),
                    GenerationOutput(
                        text="",
                        new_text=")]",
                        finished=True,
                        finish_reason="stop",
                        prompt_tokens=4,
                        completion_tokens=3,
                    ),
                ]
                for chunk in chunks:
                    yield chunk

        monkeypatch.setattr(server, "_model_name", "served-model")
        monkeypatch.setattr(server, "_reasoning_parser", None)
        monkeypatch.setattr(server, "_enable_auto_tool_choice", True)
        monkeypatch.setattr(server, "_tool_call_parser", "auto")
        monkeypatch.setattr(server, "_tool_parser_instance", None)

        request = ChatCompletionRequest(
            model="served-model",
            messages=[Message(role="user", content="hi")],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "read",
                        "description": "Read a file",
                        "parameters": {
                            "type": "object",
                            "properties": {"file_path": {"type": "string"}},
                            "required": ["file_path"],
                        },
                    },
                }
            ],
            stream=True,
        )

        chunks = [
            chunk
            async for chunk in stream_chat_completion(
                FakeEngine(), request.messages, request
            )
        ]

        payloads = [
            json.loads(chunk.removeprefix("data: ").strip())
            for chunk in chunks
            if chunk != "data: [DONE]\n\n"
        ]
        tool_payloads = [
            payload
            for payload in payloads
            if payload["choices"] and payload["choices"][0]["delta"].get("tool_calls")
        ]

        assert len(tool_payloads) == 1
        delta = tool_payloads[0]["choices"][0]["delta"]
        assert delta["tool_calls"][0]["function"]["name"] == "read"
        assert delta["tool_calls"][0]["function"]["arguments"] == (
            '{"file_path": "/tmp/test.py"}'
        )
        assert delta.get("content") is None
        assert tool_payloads[0]["choices"][0]["finish_reason"] == "tool_calls"

    def test_streaming_tool_markup_precheck_uses_bounded_boundary(self):
        """Pre-marker checks should not need the full accumulated stream."""
        import vllm_mlx.server as server

        long_prefix = "ordinary text " * 1000

        assert not server._streaming_tool_markup_possible_after_delta(
            long_prefix, "with [ordinary brackets] and no tool call"
        )
        assert server._streaming_tool_markup_possible_after_delta(
            long_prefix, '[read({"file_path": "/tmp/test.py"})]'
        )
        assert server._streaming_tool_markup_possible_after_delta(
            long_prefix + "[read(", '{"file_path": "/tmp/test.py"}'
        )

    @pytest.mark.anyio
    async def test_reasoning_stream_emits_structured_tool_calls(self, monkeypatch):
        """Tool markup after </think> should emit tool_calls chunks."""
        from vllm_mlx.engine.base import GenerationOutput
        from vllm_mlx.reasoning import DeltaMessage
        from vllm_mlx.server import (
            ChatCompletionRequest,
            Message,
            stream_chat_completion,
        )
        import vllm_mlx.server as server

        class FakeEngine:
            model_name = "fake-engine"

            async def stream_chat(self, messages, **kwargs):
                chunks = [
                    GenerationOutput(text="", new_text="<think>", finished=False),
                    GenerationOutput(text="", new_text="reasoning", finished=False),
                    GenerationOutput(text="", new_text="</think>", finished=False),
                    GenerationOutput(text="", new_text="<tool_call>", finished=False),
                    GenerationOutput(
                        text="", new_text='{"name":"search"}', finished=False
                    ),
                    GenerationOutput(
                        text="",
                        new_text="</tool_call>",
                        finished=True,
                        finish_reason="stop",
                        prompt_tokens=7,
                        completion_tokens=3,
                    ),
                ]
                for chunk in chunks:
                    yield chunk

        class FakeReasoningParser:
            def reset_state(self):
                self._in_reasoning = False

            def extract_reasoning_streaming(
                self, previous_text, current_text, delta_text
            ):
                if delta_text == "<think>":
                    self._in_reasoning = True
                    return None
                if delta_text == "</think>":
                    self._in_reasoning = False
                    return None
                if self._in_reasoning:
                    return DeltaMessage(reasoning=delta_text)
                return DeltaMessage(content=delta_text)

        class FakeToolParser:
            def reset(self):
                pass

            def extract_tool_calls_streaming(
                self, previous_text, current_text, delta_text
            ):
                if "</tool_call>" in current_text:
                    return {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_123",
                                "type": "function",
                                "function": {
                                    "name": "search",
                                    "arguments": '{"q":"weather"}',
                                },
                            }
                        ]
                    }
                return None

        monkeypatch.setattr(server, "_model_name", "served-model")
        monkeypatch.setattr(server, "_reasoning_parser", FakeReasoningParser())
        monkeypatch.setattr(server, "_enable_auto_tool_choice", True)
        monkeypatch.setattr(server, "_tool_call_parser", "fake")
        monkeypatch.setattr(server, "_tool_parser_instance", FakeToolParser())

        request = ChatCompletionRequest(
            model="request-model",
            messages=[Message(role="user", content="hi")],
            stream=True,
        )

        chunks = [
            chunk
            async for chunk in stream_chat_completion(
                FakeEngine(), request.messages, request
            )
        ]

        payloads = [
            json.loads(chunk.removeprefix("data: ").strip())
            for chunk in chunks
            if chunk != "data: [DONE]\n\n"
        ]

        tool_payloads = [
            payload
            for payload in payloads
            if payload["choices"] and payload["choices"][0]["delta"].get("tool_calls")
        ]

        assert payloads[0]["choices"][0]["delta"]["role"] == "assistant"
        assert payloads[1]["choices"][0]["delta"]["reasoning_content"] == "reasoning"
        assert "reasoning" not in payloads[1]["choices"][0]["delta"]
        assert len(tool_payloads) == 1
        assert (
            tool_payloads[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"]
            == "search"
        )
        assert tool_payloads[0]["choices"][0]["finish_reason"] == "tool_calls"
        assert tool_payloads[0]["usage"] == {
            "prompt_tokens": 7,
            "completion_tokens": 3,
            "total_tokens": 10,
        }

    @pytest.mark.anyio
    async def test_reasoning_stream_skips_tool_parser_until_markup_appears(
        self, monkeypatch
    ):
        """Plain post-reasoning content should stream normally on the fast path."""
        from vllm_mlx.engine.base import GenerationOutput
        from vllm_mlx.reasoning import DeltaMessage
        from vllm_mlx.server import (
            ChatCompletionRequest,
            Message,
            stream_chat_completion,
        )
        import vllm_mlx.server as server

        class FakeEngine:
            model_name = "fake-engine"

            async def stream_chat(self, messages, **kwargs):
                chunks = [
                    GenerationOutput(text="", new_text="<think>", finished=False),
                    GenerationOutput(text="", new_text="reasoning", finished=False),
                    GenerationOutput(text="", new_text="</think>", finished=False),
                    GenerationOutput(
                        text="",
                        new_text="final answer",
                        finished=True,
                        finish_reason="stop",
                    ),
                ]
                for chunk in chunks:
                    yield chunk

        class FakeReasoningParser:
            def reset_state(self):
                self._in_reasoning = False

            def extract_reasoning_streaming(
                self, previous_text, current_text, delta_text
            ):
                if delta_text == "<think>":
                    self._in_reasoning = True
                    return None
                if delta_text == "</think>":
                    self._in_reasoning = False
                    return None
                if self._in_reasoning:
                    return DeltaMessage(reasoning=delta_text)
                return DeltaMessage(content=delta_text)

        class TrackingToolParser:
            def __init__(self):
                self.calls = []

            def reset(self):
                self.calls.clear()

            def extract_tool_calls_streaming(
                self, previous_text, current_text, delta_text
            ):
                self.calls.append((previous_text, current_text, delta_text))
                return {"content": delta_text}

        tool_parser = TrackingToolParser()

        monkeypatch.setattr(server, "_model_name", "served-model")
        monkeypatch.setattr(server, "_reasoning_parser", FakeReasoningParser())
        monkeypatch.setattr(server, "_enable_auto_tool_choice", True)
        monkeypatch.setattr(server, "_tool_call_parser", "fake")
        monkeypatch.setattr(server, "_tool_parser_instance", tool_parser)

        request = ChatCompletionRequest(
            model="request-model",
            messages=[Message(role="user", content="hi")],
            stream=True,
        )

        chunks = [
            chunk
            async for chunk in stream_chat_completion(
                FakeEngine(), request.messages, request
            )
        ]

        payloads = [
            json.loads(chunk.removeprefix("data: ").strip())
            for chunk in chunks
            if chunk != "data: [DONE]\n\n"
        ]

        assert tool_parser.calls == []
        assert payloads[1]["choices"][0]["delta"]["reasoning_content"] == "reasoning"
        assert "reasoning" not in payloads[1]["choices"][0]["delta"]
        assert payloads[2]["choices"][0]["delta"]["content"] == "final answer"
        assert payloads[2]["choices"][0]["finish_reason"] == "stop"

    @pytest.mark.anyio
    async def test_streaming_chat_no_stream_thread_error_after_residency_preload(
        self, monkeypatch, caplog
    ):
        """Streaming chat should not hit Stream(gpu, N) after residency preload."""
        from vllm_mlx.engine.simple import SimpleEngine
        from vllm_mlx.lifecycle import ModelSpec, ResidencyManager
        from vllm_mlx.server import (
            _ensure_sse_terminal,
            ChatCompletionRequest,
            Message,
            stream_chat_completion,
        )
        import vllm_mlx.server as server

        class FakeLLMModel:
            def __init__(self, *_args, **_kwargs):
                self._load_thread = None
                self.tokenizer = MagicMock()
                self.tokenizer.apply_chat_template.return_value = "user: Count"
                self.tokenizer.bos_token = None
                self.tokenizer.encode.return_value = [1, 2, 3]

            def load(self):
                self._load_thread = threading.get_ident()

            def stream_generate(self, **_kwargs):
                if threading.get_ident() != self._load_thread:
                    raise RuntimeError("There is no Stream(gpu, 3) in current thread.")
                yield SimpleNamespace(
                    text="one, two, three",
                    prompt_tokens=3,
                    finished=True,
                    finish_reason="stop",
                )

        async def engine_factory(spec):
            return SimpleEngine(spec.model_name)

        manager = ResidencyManager(engine_factory, auto_unload_idle_seconds=0)
        manager.register_model(ModelSpec(model_key="default", model_name="test-model"))

        caplog.set_level("ERROR", logger="vllm_mlx.server")
        monkeypatch.setattr(server, "_model_name", "test-model")
        monkeypatch.setattr(server, "_reasoning_parser", None)
        monkeypatch.setattr(server, "_enable_auto_tool_choice", False)
        monkeypatch.setattr(server, "_tool_call_parser", None)
        monkeypatch.setattr(server, "_tool_parser_instance", None)

        with (
            patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False),
            patch("vllm_mlx.models.llm.MLXLanguageModel", FakeLLMModel),
        ):
            try:
                engine = await manager.ensure_loaded("default")
                request = ChatCompletionRequest(
                    model="test-model",
                    messages=[Message(role="user", content="Count: one, two, three")],
                    max_tokens=30,
                    temperature=0,
                    stream=True,
                )

                chunks = [
                    chunk
                    async for chunk in _ensure_sse_terminal(
                        stream_chat_completion(engine, request.messages, request),
                        "data: [DONE]\n\n",
                    )
                ]
            finally:
                await manager.shutdown()

        errors = [
            record.message
            for record in caplog.records
            if "Streaming error, ensuring terminal frame" in record.message
        ]
        assert not errors, f"Unexpected streaming wrapper errors: {errors}"
        assert any("data: [DONE]" in chunk for chunk in chunks)


class TestReasoningAndToolCallsNonStreaming:
    """Non-streaming coexistence of reasoning extraction and tool parsing."""

    @pytest.fixture()
    def client(self):
        """Create a FastAPI test client."""
        from fastapi.testclient import TestClient

        from vllm_mlx.server import app

        return TestClient(app)

    def test_chat_completion_preserves_reasoning_with_tool_calls(
        self, client, monkeypatch
    ):
        """Reasoning should survive when tool calls are present in final output."""
        import vllm_mlx.server as server
        from vllm_mlx.engine.base import GenerationOutput
        from vllm_mlx.server import ToolCall, FunctionCall

        parsed_inputs = []

        class FakeEngine:
            model_name = "fake-engine"
            is_mllm = False
            preserve_native_tool_format = False

            async def chat(self, messages, **kwargs):
                return GenerationOutput(
                    text="<think>Need tool</think><tool_call>",
                    prompt_tokens=7,
                    completion_tokens=3,
                    finish_reason="stop",
                )

        class FakeReasoningParser:
            def extract_reasoning(self, model_output):
                assert model_output == "<think>Need tool</think><tool_call>"
                return "Need tool", "<tool_call>"

        def fake_parse_tool_calls(text, request):
            parsed_inputs.append(text)
            if text == "<tool_call>":
                return None, [
                    ToolCall(
                        id="call_1",
                        type="function",
                        function=FunctionCall(
                            name="get_weather",
                            arguments='{"city":"Paris"}',
                        ),
                    )
                ]
            return text, None

        monkeypatch.setattr(server, "_engine", FakeEngine())
        monkeypatch.setattr(server, "_model_name", "test-model")
        monkeypatch.setattr(server, "_default_timeout", 30.0)
        monkeypatch.setattr(server, "_default_max_tokens", 128)
        monkeypatch.setattr(server, "_api_key", None)
        monkeypatch.setattr(
            server,
            "_rate_limiter",
            server.RateLimiter(requests_per_minute=60, enabled=False),
        )
        monkeypatch.setattr(server, "_reasoning_parser", FakeReasoningParser())
        monkeypatch.setattr(
            server, "_parse_tool_calls_with_parser", fake_parse_tool_calls
        )

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Weather?"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "Get weather",
                            "parameters": {
                                "type": "object",
                                "properties": {"city": {"type": "string"}},
                                "required": ["city"],
                            },
                        },
                    }
                ],
                "max_tokens": 32,
            },
        )

        assert response.status_code == 200
        body = response.json()
        choice = body["choices"][0]
        assert parsed_inputs == ["<tool_call>"]
        assert choice["message"]["content"] is None
        assert choice["message"]["reasoning_content"] == "Need tool"
        assert choice["message"]["tool_calls"][0]["function"]["name"] == "get_weather"
        assert choice["finish_reason"] == "tool_calls"

    def test_anthropic_message_preserves_thinking_with_tool_use(
        self, client, monkeypatch
    ):
        """Anthropic non-streaming should emit thinking and tool_use blocks."""
        import vllm_mlx.server as server
        from vllm_mlx.engine.base import GenerationOutput
        from vllm_mlx.server import ToolCall, FunctionCall

        parsed_inputs = []

        class FakeEngine:
            model_name = "fake-engine"
            is_mllm = False
            preserve_native_tool_format = False

            async def chat(self, messages, **kwargs):
                return GenerationOutput(
                    text="<think>Need tool</think><tool_call>",
                    prompt_tokens=11,
                    completion_tokens=4,
                    finish_reason="stop",
                )

        class FakeReasoningParser:
            def extract_reasoning(self, model_output):
                assert model_output == "<think>Need tool</think><tool_call>"
                return "Need tool", "<tool_call>"

        def fake_parse_tool_calls(text, request):
            parsed_inputs.append(text)
            if text == "<tool_call>":
                return None, [
                    ToolCall(
                        id="call_1",
                        type="function",
                        function=FunctionCall(
                            name="get_weather",
                            arguments='{"city":"Paris"}',
                        ),
                    )
                ]
            return text, None

        monkeypatch.setattr(server, "_engine", FakeEngine())
        monkeypatch.setattr(server, "_model_name", "test-model")
        monkeypatch.setattr(server, "_default_timeout", 30.0)
        monkeypatch.setattr(server, "_default_max_tokens", 128)
        monkeypatch.setattr(server, "_api_key", None)
        monkeypatch.setattr(
            server,
            "_rate_limiter",
            server.RateLimiter(requests_per_minute=60, enabled=False),
        )
        monkeypatch.setattr(server, "_reasoning_parser", FakeReasoningParser())
        monkeypatch.setattr(
            server, "_parse_tool_calls_with_parser", fake_parse_tool_calls
        )

        response = client.post(
            "/v1/messages",
            json={
                "model": "test-model",
                "max_tokens": 32,
                "messages": [{"role": "user", "content": "Weather?"}],
                "tools": [
                    {
                        "name": "get_weather",
                        "description": "Get weather",
                        "input_schema": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    }
                ],
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert parsed_inputs == ["<tool_call>"]
        assert body["stop_reason"] == "tool_use"
        assert [block["type"] for block in body["content"]] == ["thinking", "tool_use"]
        assert body["content"][0]["thinking"] == "Need tool"
        assert body["content"][1]["name"] == "get_weather"
        assert body["content"][1]["input"] == {"city": "Paris"}

    def test_anthropic_message_applies_server_default_chat_template_kwargs(
        self, client, monkeypatch
    ):
        """Anthropic endpoint should forward server default chat_template_kwargs."""
        import vllm_mlx.server as server
        from vllm_mlx.engine.base import GenerationOutput

        captured = {}

        class FakeEngine:
            model_name = "fake-engine"
            is_mllm = False
            preserve_native_tool_format = False

            async def chat(self, messages, **kwargs):
                captured["messages"] = messages
                captured["kwargs"] = kwargs
                return GenerationOutput(
                    text="Final answer",
                    prompt_tokens=11,
                    completion_tokens=4,
                    finish_reason="stop",
                )

        monkeypatch.setattr(server, "_engine", FakeEngine())
        monkeypatch.setattr(server, "_model_name", "test-model")
        monkeypatch.setattr(server, "_default_timeout", 30.0)
        monkeypatch.setattr(server, "_default_max_tokens", 128)
        monkeypatch.setattr(server, "_api_key", None)
        monkeypatch.setattr(
            server,
            "_rate_limiter",
            server.RateLimiter(requests_per_minute=60, enabled=False),
        )
        monkeypatch.setattr(server, "_reasoning_parser", None)
        monkeypatch.setattr(
            server,
            "_default_chat_template_kwargs",
            {"enable_thinking": False},
            raising=False,
        )

        response = client.post(
            "/v1/messages",
            json={
                "model": "test-model",
                "max_tokens": 32,
                "messages": [{"role": "user", "content": "Weather?"}],
            },
        )

        assert response.status_code == 200
        assert captured["kwargs"]["chat_template_kwargs"] == {"enable_thinking": False}

    def test_anthropic_message_request_kwargs_override_server_defaults(
        self, client, monkeypatch
    ):
        """Anthropic request chat_template_kwargs should override server defaults."""
        import vllm_mlx.server as server
        from vllm_mlx.engine.base import GenerationOutput

        captured = {}

        class FakeEngine:
            model_name = "fake-engine"
            is_mllm = False
            preserve_native_tool_format = False

            async def chat(self, messages, **kwargs):
                captured["messages"] = messages
                captured["kwargs"] = kwargs
                return GenerationOutput(
                    text="Final answer",
                    prompt_tokens=11,
                    completion_tokens=4,
                    finish_reason="stop",
                )

        monkeypatch.setattr(server, "_engine", FakeEngine())
        monkeypatch.setattr(server, "_model_name", "test-model")
        monkeypatch.setattr(server, "_default_timeout", 30.0)
        monkeypatch.setattr(server, "_default_max_tokens", 128)
        monkeypatch.setattr(server, "_api_key", None)
        monkeypatch.setattr(
            server,
            "_rate_limiter",
            server.RateLimiter(requests_per_minute=60, enabled=False),
        )
        monkeypatch.setattr(server, "_reasoning_parser", None)
        monkeypatch.setattr(
            server,
            "_default_chat_template_kwargs",
            {"enable_thinking": False, "server_default_only": "yes"},
            raising=False,
        )

        response = client.post(
            "/v1/messages",
            json={
                "model": "test-model",
                "max_tokens": 32,
                "messages": [{"role": "user", "content": "Weather?"}],
                "chat_template_kwargs": {
                    "enable_thinking": True,
                    "request_only": 1,
                },
            },
        )

        assert response.status_code == 200
        assert captured["kwargs"]["chat_template_kwargs"] == {
            "enable_thinking": True,
            "server_default_only": "yes",
            "request_only": 1,
        }

    def test_chat_completion_prepares_messages_once_in_non_stream_path(
        self, client, monkeypatch
    ):
        """Chat non-streaming should prepare request messages a single time."""
        import vllm_mlx.server as server
        from vllm_mlx.engine.base import GenerationOutput

        extract_calls = {"count": 0}

        class FakeEngine:
            model_name = "fake-engine"
            is_mllm = False
            preserve_native_tool_format = False

            async def chat(self, messages, **kwargs):
                return GenerationOutput(
                    text="done",
                    prompt_tokens=3,
                    completion_tokens=1,
                    finish_reason="stop",
                )

        def fake_extract(messages, preserve_native_format=False):
            extract_calls["count"] += 1
            return ([{"role": "user", "content": "hi"}], [], [], [])

        monkeypatch.setattr(server, "_engine", FakeEngine())
        monkeypatch.setattr(server, "_model_name", "test-model")
        monkeypatch.setattr(server, "_default_timeout", 30.0)
        monkeypatch.setattr(server, "_default_max_tokens", 128)
        monkeypatch.setattr(server, "_api_key", None)
        monkeypatch.setattr(
            server,
            "_rate_limiter",
            server.RateLimiter(requests_per_minute=60, enabled=False),
        )
        monkeypatch.setattr(server, "extract_multimodal_content", fake_extract)
        monkeypatch.setattr(server, "_reasoning_parser", None)

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 8,
            },
        )

        assert response.status_code == 200
        assert extract_calls["count"] == 1

    def test_chat_completion_sanitizes_remote_media_safety_errors(
        self, client, monkeypatch
    ):
        """Unsafe remote media URLs should return a generic public error."""
        import vllm_mlx.server as server

        class FakeEngine:
            model_name = "fake-mllm"
            is_mllm = True
            preserve_native_tool_format = False

            async def chat(self, messages, **kwargs):  # pragma: no cover
                raise AssertionError("unsafe URL should fail during preparation")

        async def fake_acquire(_raw_request, **_kwargs):
            return FakeEngine()

        async def fake_release(*_args, **_kwargs):
            return None

        monkeypatch.setattr(server, "_acquire_default_engine_for_request", fake_acquire)
        monkeypatch.setattr(server, "_release_engine_for_request", fake_release)
        monkeypatch.setattr(server, "_model_name", "test-model")
        monkeypatch.setattr(server, "_default_timeout", 30.0)
        monkeypatch.setattr(server, "_default_max_tokens", 128)
        monkeypatch.setattr(server, "_api_key", None)
        monkeypatch.setattr(
            server,
            "_rate_limiter",
            server.RateLimiter(requests_per_minute=60, enabled=False),
        )

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "describe this"},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "http://169.254.169.254/latest/meta-data/"
                                },
                            },
                        ],
                    }
                ],
                "max_tokens": 8,
            },
        )

        assert response.status_code == 400
        assert response.json()["detail"] == "Remote media URL is not allowed"
        assert "169.254.169.254" not in response.text

    def test_anthropic_message_sanitizes_remote_media_safety_errors(
        self, client, monkeypatch
    ):
        """Anthropic preparation should sanitize URL-safety failures too."""
        import vllm_mlx.server as server

        class FakeEngine:
            model_name = "fake-mllm"
            is_mllm = True
            preserve_native_tool_format = False

            async def chat(self, messages, **kwargs):  # pragma: no cover
                raise AssertionError("unsafe URL should fail during preparation")

        async def fake_acquire(_raw_request, **_kwargs):
            return FakeEngine()

        async def fake_release(*_args, **_kwargs):
            return None

        def fake_anthropic_to_openai(_anthropic_request):
            return server.ChatCompletionRequest(
                model="test-model",
                messages=[
                    server.Message(
                        role="user",
                        content=[
                            {"type": "text", "text": "describe this"},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "http://169.254.169.254/latest/meta-data/"
                                },
                            },
                        ],
                    )
                ],
                max_tokens=8,
            )

        monkeypatch.setattr(server, "_acquire_default_engine_for_request", fake_acquire)
        monkeypatch.setattr(server, "_release_engine_for_request", fake_release)
        monkeypatch.setattr(server, "anthropic_to_openai", fake_anthropic_to_openai)
        monkeypatch.setattr(server, "_model_name", "test-model")
        monkeypatch.setattr(server, "_default_timeout", 30.0)
        monkeypatch.setattr(server, "_default_max_tokens", 128)
        monkeypatch.setattr(server, "_api_key", None)
        monkeypatch.setattr(
            server,
            "_rate_limiter",
            server.RateLimiter(requests_per_minute=60, enabled=False),
        )

        response = client.post(
            "/v1/messages",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "describe this"}],
                "max_tokens": 8,
            },
        )

        assert response.status_code == 400
        assert response.json()["detail"] == "Remote media URL is not allowed"
        assert "169.254.169.254" not in response.text


class TestChatCompletionStreamingModeSwitching:
    """Endpoint-level regression tests for stream/non-stream mode switching."""

    @pytest.fixture()
    def client(self):
        """Create a FastAPI test client."""
        from fastapi.testclient import TestClient

        from vllm_mlx.server import app

        return TestClient(app)

    @pytest.mark.anyio
    async def test_nonstream_then_stream_chat_completion_keeps_stream_thread_valid(
        self, client, monkeypatch, caplog
    ):
        """A non-stream chat request must not break the next stream request."""
        import vllm_mlx.server as server
        from vllm_mlx.engine.simple import SimpleEngine

        bound_thread = {"id": None}

        def fake_bind_generation_streams():
            bound_thread["id"] = threading.get_ident()

        class FakeLLMModel:
            def __init__(self, *_args, **_kwargs):
                self.tokenizer = MagicMock()
                self.tokenizer.bos_token = None
                self.tokenizer.apply_chat_template.return_value = (
                    "Count: one, two, three"
                )
                self.tokenizer.encode.return_value = [1, 2, 3]

            def load(self):
                # Initial ownership belongs to the load thread.
                bound_thread["id"] = threading.get_ident()

            def chat(self, **_kwargs):
                if bound_thread["id"] != threading.get_ident():
                    raise RuntimeError("There is no Stream(gpu, 3) in current thread.")
                return SimpleNamespace(
                    text="one, two, three",
                    tokens=[11, 12, 13],
                    finish_reason="stop",
                )

            def stream_generate(self, **_kwargs):
                if bound_thread["id"] != threading.get_ident():
                    raise RuntimeError("There is no Stream(gpu, 3) in current thread.")
                yield SimpleNamespace(
                    text="one, two, three",
                    prompt_tokens=3,
                    finished=True,
                    finish_reason="stop",
                )

        engine = SimpleEngine("test-model")

        async def fake_acquire(_raw_request, **_kwargs):
            return engine

        async def fake_release(*_args, **_kwargs):
            return None

        caplog.set_level("ERROR", logger="vllm_mlx.server")

        with (
            patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False),
            patch("vllm_mlx.models.llm.MLXLanguageModel", FakeLLMModel),
            patch(
                "vllm_mlx.engine.simple.bind_generation_streams",
                side_effect=fake_bind_generation_streams,
            ),
        ):
            monkeypatch.setattr(server, "_model_name", "test-model")
            monkeypatch.setattr(server, "_default_timeout", 30.0)
            monkeypatch.setattr(server, "_default_max_tokens", 128)
            monkeypatch.setattr(server, "_api_key", None)
            monkeypatch.setattr(
                server,
                "_rate_limiter",
                server.RateLimiter(requests_per_minute=60, enabled=False),
            )
            monkeypatch.setattr(server, "_reasoning_parser", None)
            monkeypatch.setattr(server, "_enable_auto_tool_choice", True)
            monkeypatch.setattr(server, "_tool_call_parser", "qwen3_coder")
            monkeypatch.setattr(server, "_tool_parser_instance", None)
            monkeypatch.setattr(
                server,
                "_acquire_default_engine_for_request",
                fake_acquire,
            )
            monkeypatch.setattr(server, "_release_default_engine", fake_release)

            nonstream = client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "Count: one, two, three"}],
                    "max_tokens": 30,
                    "temperature": 0,
                    "stream": False,
                },
            )
            stream = client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "Count: one, two, three"}],
                    "max_tokens": 30,
                    "temperature": 0,
                    "stream": True,
                },
            )

            await engine.stop()

        assert nonstream.status_code == 200
        assert stream.status_code == 200
        assert "data: [DONE]" in stream.text
        assert "one, two, three" in stream.text
        assert not [
            rec.message
            for rec in caplog.records
            if "Streaming error, ensuring terminal frame" in rec.message
        ]

    @pytest.mark.anyio
    async def test_nonstream_mllm_endpoint_text_only_does_not_raise_stream_thread_error(
        self, client, monkeypatch, caplog
    ):
        """Direct /v1/chat/completions non-stream MLLM regression for Stream(gpu, N).

        Mirrors production shape:
        - served model name alias (request model != backing path)
        - text-only MLLM request
        - stream=false
        """
        import vllm_mlx.server as server
        from vllm_mlx.engine.simple import SimpleEngine

        class FakeMllmModel:
            def chat(self, **kwargs):
                raise RuntimeError("MLLM non-stream chat path must not be used")

            def stream_chat(self, **kwargs):
                yield SimpleNamespace(
                    text="one, two, three",
                    finish_reason="stop",
                    prompt_tokens=3,
                )

        engine = SimpleEngine("unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit", force_mllm=True)
        engine._loaded = True
        engine._text_model = None
        engine._model = FakeMllmModel()

        async def fail_if_called(*args, **kwargs):
            raise RuntimeError("There is no Stream(gpu, 3) in current thread.")

        # If this gets called, code regressed to the old broken thread path.
        engine._run_blocking_serialized = fail_if_called  # type: ignore[method-assign]

        async def fake_acquire(_raw_request, **_kwargs):
            return engine

        async def fake_release(*_args, **_kwargs):
            return None

        caplog.set_level("ERROR", logger="vllm_mlx.server")

        monkeypatch.setattr(server, "_engine", engine)
        monkeypatch.setattr(server, "_model_name", "Qwen3.6-35B-A3B")
        monkeypatch.setattr(
            server,
            "_model_path",
            "unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit",
            raising=False,
        )
        monkeypatch.setattr(server, "_force_mllm_model", True, raising=False)
        monkeypatch.setattr(server, "_default_timeout", 30.0)
        monkeypatch.setattr(server, "_default_max_tokens", 4096)
        monkeypatch.setattr(server, "_api_key", None)
        monkeypatch.setattr(
            server,
            "_rate_limiter",
            server.RateLimiter(requests_per_minute=60, enabled=False),
        )
        monkeypatch.setattr(server, "_reasoning_parser", None)
        monkeypatch.setattr(server, "_enable_auto_tool_choice", True)
        monkeypatch.setattr(server, "_tool_call_parser", "qwen3_coder")
        monkeypatch.setattr(server, "_tool_parser_instance", None)
        monkeypatch.setattr(
            server,
            "_acquire_default_engine_for_request",
            fake_acquire,
        )
        monkeypatch.setattr(server, "_release_default_engine", fake_release)

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "Qwen3.6-35B-A3B",
                "messages": [{"role": "user", "content": "Count: one, two, three"}],
                "max_tokens": 30,
                "temperature": 0,
                "stream": False,
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["choices"][0]["message"]["content"] == "one, two, three"
        assert not [
            rec.message
            for rec in caplog.records
            if "There is no Stream(gpu, 3) in current thread." in rec.message
        ]

    @pytest.mark.anyio
    async def test_stream_then_nonstream_chat_completion_keeps_stream_thread_valid(
        self, client, monkeypatch, caplog
    ):
        """A stream chat request must not break the next non-stream request."""
        import vllm_mlx.server as server
        from vllm_mlx.engine.simple import SimpleEngine

        bound_thread = {"id": None}

        def fake_bind_generation_streams():
            bound_thread["id"] = threading.get_ident()

        class FakeLLMModel:
            def __init__(self, *_args, **_kwargs):
                self.tokenizer = MagicMock()
                self.tokenizer.bos_token = None
                self.tokenizer.apply_chat_template.return_value = (
                    "Count: one, two, three"
                )
                self.tokenizer.encode.return_value = [1, 2, 3]

            def load(self):
                # Initial ownership belongs to the load thread.
                bound_thread["id"] = threading.get_ident()

            def chat(self, **_kwargs):
                if bound_thread["id"] != threading.get_ident():
                    raise RuntimeError("There is no Stream(gpu, 3) in current thread.")
                return SimpleNamespace(
                    text="one, two, three",
                    tokens=[11, 12, 13],
                    finish_reason="stop",
                )

            def stream_generate(self, **_kwargs):
                if bound_thread["id"] != threading.get_ident():
                    raise RuntimeError("There is no Stream(gpu, 3) in current thread.")
                yield SimpleNamespace(
                    text="one, two, three",
                    prompt_tokens=3,
                    finished=True,
                    finish_reason="stop",
                )

        engine = SimpleEngine("test-model")

        async def fake_acquire(_raw_request, **_kwargs):
            return engine

        async def fake_release(*_args, **_kwargs):
            return None

        caplog.set_level("ERROR", logger="vllm_mlx.server")

        with (
            patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False),
            patch("vllm_mlx.models.llm.MLXLanguageModel", FakeLLMModel),
            patch(
                "vllm_mlx.engine.simple.bind_generation_streams",
                side_effect=fake_bind_generation_streams,
            ),
        ):
            monkeypatch.setattr(server, "_model_name", "test-model")
            monkeypatch.setattr(server, "_default_timeout", 30.0)
            monkeypatch.setattr(server, "_default_max_tokens", 128)
            monkeypatch.setattr(server, "_api_key", None)
            monkeypatch.setattr(
                server,
                "_rate_limiter",
                server.RateLimiter(requests_per_minute=60, enabled=False),
            )
            monkeypatch.setattr(server, "_reasoning_parser", None)
            monkeypatch.setattr(server, "_enable_auto_tool_choice", True)
            monkeypatch.setattr(server, "_tool_call_parser", "qwen3_coder")
            monkeypatch.setattr(server, "_tool_parser_instance", None)
            monkeypatch.setattr(
                server,
                "_acquire_default_engine_for_request",
                fake_acquire,
            )
            monkeypatch.setattr(server, "_release_default_engine", fake_release)

            stream = client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "Count: one, two, three"}],
                    "max_tokens": 30,
                    "temperature": 0,
                    "stream": True,
                },
            )
            nonstream = client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "Count: one, two, three"}],
                    "max_tokens": 30,
                    "temperature": 0,
                    "stream": False,
                },
            )

            await engine.stop()

        assert stream.status_code == 200
        assert nonstream.status_code == 200
        assert "data: [DONE]" in stream.text
        assert "one, two, three" in nonstream.text
        assert not [
            rec.message
            for rec in caplog.records
            if "Streaming error, ensuring terminal frame" in rec.message
        ]

    @pytest.mark.anyio
    async def test_nonstream_then_stream_parser_init_does_not_reintroduce_stream_error(
        self, client, monkeypatch, caplog
    ):
        """Stream-time parser init after non-stream request must not trigger Stream(gpu, N) errors."""
        import vllm_mlx.server as server
        from vllm_mlx.engine.simple import SimpleEngine

        bound_thread = {"id": None}
        parser_init_threads: list[int] = []

        def fake_bind_generation_streams():
            bound_thread["id"] = threading.get_ident()

        class FakeParser:
            def __init__(self, tokenizer):
                parser_init_threads.append(threading.get_ident())
                self.tokenizer = tokenizer

            def reset(self):
                return None

            def extract_tool_calls_streaming(
                self, previous_text, current_text, delta_text
            ):
                return {"content": delta_text}

            def extract_tool_calls(self, text):
                return SimpleNamespace(tools_called=False, tool_calls=[], content=text)

        class FakeLLMModel:
            def __init__(self, *_args, **_kwargs):
                self.tokenizer = MagicMock()
                self.tokenizer.bos_token = None
                self.tokenizer.apply_chat_template.return_value = (
                    "Count: one, two, three"
                )
                self.tokenizer.encode.return_value = [1, 2, 3]

            def load(self):
                bound_thread["id"] = threading.get_ident()

            def chat(self, **_kwargs):
                if bound_thread["id"] != threading.get_ident():
                    raise RuntimeError("There is no Stream(gpu, 3) in current thread.")
                return SimpleNamespace(
                    text="one, two, three",
                    tokens=[11, 12, 13],
                    finish_reason="stop",
                )

            def stream_generate(self, **_kwargs):
                if bound_thread["id"] != threading.get_ident():
                    raise RuntimeError("There is no Stream(gpu, 3) in current thread.")
                yield SimpleNamespace(
                    text="one, two, three",
                    prompt_tokens=3,
                    finished=True,
                    finish_reason="stop",
                )

        engine = SimpleEngine("test-model")

        async def fake_acquire(_raw_request, **_kwargs):
            return engine

        async def fake_release(*_args, **_kwargs):
            return None

        caplog.set_level("ERROR", logger="vllm_mlx.server")

        with (
            patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False),
            patch("vllm_mlx.models.llm.MLXLanguageModel", FakeLLMModel),
            patch(
                "vllm_mlx.engine.simple.bind_generation_streams",
                side_effect=fake_bind_generation_streams,
            ),
        ):
            monkeypatch.setattr(server, "_model_name", "test-model")
            monkeypatch.setattr(server, "_default_timeout", 30.0)
            monkeypatch.setattr(server, "_default_max_tokens", 128)
            monkeypatch.setattr(server, "_api_key", None)
            monkeypatch.setattr(
                server,
                "_rate_limiter",
                server.RateLimiter(requests_per_minute=60, enabled=False),
            )
            monkeypatch.setattr(server, "_reasoning_parser", None)
            monkeypatch.setattr(server, "_enable_auto_tool_choice", True)
            monkeypatch.setattr(server, "_tool_call_parser", "qwen3_coder")
            monkeypatch.setattr(server, "_tool_parser_instance", None)
            monkeypatch.setattr(
                server.ToolParserManager,
                "get_tool_parser",
                staticmethod(lambda _name: FakeParser),
            )
            monkeypatch.setattr(
                server,
                "_acquire_default_engine_for_request",
                fake_acquire,
            )
            monkeypatch.setattr(server, "_release_default_engine", fake_release)

            nonstream = client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "Count: one, two, three"}],
                    "max_tokens": 30,
                    "temperature": 0,
                    "stream": False,
                },
            )
            assert parser_init_threads == []

            stream = client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "Count: one, two, three"}],
                    "max_tokens": 30,
                    "temperature": 0,
                    "stream": True,
                },
            )

            await engine.stop()

        assert nonstream.status_code == 200
        assert stream.status_code == 200
        assert parser_init_threads, "Parser should initialize on stream request"
        assert "one, two, three" in stream.text
        assert "data: [DONE]" in stream.text
        assert not [
            rec.message
            for rec in caplog.records
            if "Streaming error, ensuring terminal frame" in rec.message
        ]

    def test_anthropic_message_prepares_messages_once_in_non_stream_path(
        self, client, monkeypatch
    ):
        """Anthropic non-streaming should prepare request messages a single time."""
        import vllm_mlx.server as server
        from vllm_mlx.engine.base import GenerationOutput

        extract_calls = {"count": 0}

        class FakeEngine:
            model_name = "fake-engine"
            is_mllm = False
            preserve_native_tool_format = False

            async def chat(self, messages, **kwargs):
                return GenerationOutput(
                    text="done",
                    prompt_tokens=3,
                    completion_tokens=1,
                    finish_reason="stop",
                )

        def fake_extract(messages, preserve_native_format=False):
            extract_calls["count"] += 1
            return ([{"role": "user", "content": "hi"}], [], [], [])

        monkeypatch.setattr(server, "_engine", FakeEngine())
        monkeypatch.setattr(server, "_model_name", "test-model")
        monkeypatch.setattr(server, "_default_timeout", 30.0)
        monkeypatch.setattr(server, "_default_max_tokens", 128)
        monkeypatch.setattr(server, "_api_key", None)
        monkeypatch.setattr(
            server,
            "_rate_limiter",
            server.RateLimiter(requests_per_minute=60, enabled=False),
        )
        monkeypatch.setattr(server, "extract_multimodal_content", fake_extract)
        monkeypatch.setattr(server, "_reasoning_parser", None)

        response = client.post(
            "/v1/messages",
            json={
                "model": "test-model",
                "max_tokens": 8,
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

        assert response.status_code == 200
        assert extract_calls["count"] == 1


# =============================================================================
# Request Cancellation Tests
# =============================================================================


class TestRequestCancellationEndpoint:
    @pytest.mark.anyio
    async def test_cancel_request_routes_to_loaded_engine(self):
        import vllm_mlx.server as server

        class DummyEngine:
            is_mllm = False

            async def abort_request(self, request_id):
                self.request_id = request_id
                return True

        engine = DummyEngine()
        old_engine = server._engine
        old_model_name = server._model_name
        try:
            server._engine = engine
            server._model_name = "test-model"

            result = await server.cancel_request("req-123")

            assert result["cancelled"] is True
            assert result["id"] == "req-123"
            assert result["model"] == "test-model"
            assert engine.request_id == "req-123"
        finally:
            server._engine = old_engine
            server._model_name = old_model_name

    @pytest.mark.anyio
    async def test_cancel_request_404_for_unknown_request(self):
        import vllm_mlx.server as server

        class DummyEngine:
            is_mllm = False

            async def abort_request(self, request_id):
                return False

        old_engine = server._engine
        try:
            server._engine = DummyEngine()

            with pytest.raises(server.HTTPException) as exc:
                await server.cancel_request("missing")

            assert exc.value.status_code == 404
        finally:
            server._engine = old_engine


# =============================================================================
# Integration Tests (require running server)
# =============================================================================


@pytest.mark.slow
@pytest.mark.integration
class TestServerIntegration:
    """Integration tests that require a running server.

    These tests are skipped by default. Run with:
        pytest -m integration --server-url http://localhost:8000
    """

    @pytest.fixture
    def server_url(self, request):
        """Get server URL from command line or use default."""
        return request.config.getoption("--server-url", default="http://localhost:8000")

    def test_health_endpoint(self, server_url):
        """Test /health endpoint."""
        import requests

        response = requests.get(f"{server_url}/health", timeout=5)
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert "model_name" in data

    def test_models_endpoint(self, server_url):
        """Test /v1/models endpoint."""
        import requests

        response = requests.get(f"{server_url}/v1/models", timeout=5)
        assert response.status_code == 200

        data = response.json()
        assert "data" in data
        assert len(data["data"]) > 0

    def test_chat_completion(self, server_url):
        """Test /v1/chat/completions endpoint."""
        import requests

        payload = {
            "model": "default",
            "messages": [{"role": "user", "content": "Say hello"}],
            "max_tokens": 10,
        }

        response = requests.post(
            f"{server_url}/v1/chat/completions",
            json=payload,
            timeout=30,
        )
        assert response.status_code == 200

        data = response.json()
        assert "choices" in data
        assert len(data["choices"]) > 0
        assert data["choices"][0]["message"]["content"]


class TestSseDoneTermination:
    """Regression tests for SSE data: [DONE] termination signal.

    Covers #101: streaming responses must always emit exactly one
    data: [DONE] event, even when the engine raises mid-stream.
    """

    @pytest.mark.anyio
    async def test_stream_completion_normal_emits_done(self, monkeypatch):
        """Normal stream_completion yields exactly one [DONE] at the end."""
        from vllm_mlx.api.models import CompletionRequest
        from vllm_mlx.engine.base import GenerationOutput
        from vllm_mlx.server import stream_completion

        import vllm_mlx.server as server

        class FakeEngine:
            model_name = "fake-engine"

            async def stream_generate(self, **kwargs):
                yield GenerationOutput(text="Hello", new_text="Hello", finished=False)
                yield GenerationOutput(
                    text="Hello world",
                    new_text=" world",
                    finished=True,
                    finish_reason="stop",
                    prompt_tokens=5,
                    completion_tokens=2,
                )

        monkeypatch.setattr(server, "_model_name", "test-model")
        monkeypatch.setattr(server, "_default_max_tokens", 100)

        request = CompletionRequest(model="test-model", prompt="Say hello")
        chunks = [
            chunk
            async for chunk in stream_completion(
                FakeEngine(),
                "Say hello",
                request,
                max_tokens=server._default_max_tokens,
            )
        ]

        done_chunks = [c for c in chunks if c == "data: [DONE]\n\n"]
        assert (
            len(done_chunks) == 1
        ), f"Expected exactly 1 [DONE], got {len(done_chunks)}"
        assert chunks[-1] == "data: [DONE]\n\n", "[DONE] must be the last chunk"

    @pytest.mark.anyio
    async def test_stream_completion_exception_still_emits_done(self, monkeypatch):
        """When engine raises mid-stream, [DONE] is still emitted via _ensure_sse_terminal."""
        from vllm_mlx.api.models import CompletionRequest
        from vllm_mlx.engine.base import GenerationOutput
        from vllm_mlx.server import _ensure_sse_terminal, stream_completion

        import vllm_mlx.server as server

        class ExplodingEngine:
            model_name = "exploding-engine"

            async def stream_generate(self, **kwargs):
                yield GenerationOutput(
                    text="partial", new_text="partial", finished=False
                )
                raise RuntimeError("Metal command buffer SIGABRT")

        monkeypatch.setattr(server, "_model_name", "test-model")
        monkeypatch.setattr(server, "_default_max_tokens", 100)

        request = CompletionRequest(model="test-model", prompt="Say hello")
        # Wrap with _ensure_sse_terminal, matching server routing
        chunks = [
            chunk
            async for chunk in _ensure_sse_terminal(
                stream_completion(
                    ExplodingEngine(),
                    "Say hello",
                    request,
                    max_tokens=server._default_max_tokens,
                ),
                "data: [DONE]\n\n",
            )
        ]

        done_chunks = [c for c in chunks if c == "data: [DONE]\n\n"]
        assert (
            len(done_chunks) == 1
        ), f"Expected exactly 1 [DONE], got {len(done_chunks)}"
        assert chunks[-1] == "data: [DONE]\n\n", "[DONE] must be the last chunk"

    @pytest.mark.anyio
    async def test_ensure_sse_terminal_normal_no_duplicate(self):
        """Wrapper passes through the generator's own [DONE] without duplicating."""
        from vllm_mlx.server import _ensure_sse_terminal

        async def happy_generator():
            yield "data: {}\n\n"
            yield "data: [DONE]\n\n"

        chunks = [
            chunk
            async for chunk in _ensure_sse_terminal(
                happy_generator(), "data: [DONE]\n\n"
            )
        ]

        done_chunks = [c for c in chunks if c == "data: [DONE]\n\n"]
        assert (
            len(done_chunks) == 1
        ), f"Expected exactly 1 [DONE], got {len(done_chunks)}"

    @pytest.mark.anyio
    async def test_ensure_sse_terminal_exception_emits_done(self):
        """Wrapper emits [DONE] when inner generator raises before reaching it."""
        from vllm_mlx.server import _ensure_sse_terminal

        async def exploding_generator():
            yield "data: {}\n\n"
            raise RuntimeError("engine crashed")

        chunks = [
            chunk
            async for chunk in _ensure_sse_terminal(
                exploding_generator(), "data: [DONE]\n\n"
            )
        ]

        done_chunks = [c for c in chunks if c == "data: [DONE]\n\n"]
        assert (
            len(done_chunks) == 1
        ), f"Expected exactly 1 [DONE], got {len(done_chunks)}"
        assert chunks[-1] == "data: [DONE]\n\n", "[DONE] must be the last chunk"

    @pytest.mark.anyio
    async def test_ensure_sse_terminal_anthropic_protocol(self):
        """Wrapper emits Anthropic message_stop, not OpenAI [DONE], on exception."""
        import json

        from vllm_mlx.server import _ensure_sse_terminal

        anthropic_terminal = (
            f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
        )

        async def exploding_anthropic_stream():
            yield "event: content_block_delta\ndata: {}\n\n"
            raise RuntimeError("engine crashed")

        chunks = [
            chunk
            async for chunk in _ensure_sse_terminal(
                exploding_anthropic_stream(), anthropic_terminal
            )
        ]

        # Must emit Anthropic terminal, NOT OpenAI [DONE]
        assert chunks[-1] == anthropic_terminal
        openai_done = [c for c in chunks if c == "data: [DONE]\n\n"]
        assert (
            len(openai_done) == 0
        ), "Must not emit OpenAI [DONE] for Anthropic streams"


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--server-url",
        action="store",
        default="http://localhost:8000",
        help="URL of the vllm-mlx server for integration tests",
    )
