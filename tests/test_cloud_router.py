# SPDX-License-Identifier: Apache-2.0
"""
Tests for cloud routing feature.

Tests cover:
- CloudRouter class (vllm_mlx/cloud_router.py)
"""

import json
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# CloudRouter tests
# ---------------------------------------------------------------------------


class TestCloudRouterShouldRoute:
    """Tests for CloudRouter.should_route_to_cloud method."""

    def test_below_threshold(self):
        """Returns False when new_tokens < threshold."""
        from vllm_mlx.cloud_router import CloudRouter

        router = CloudRouter(cloud_model="test-model", threshold=1000)
        assert router.should_route_to_cloud(500) is False

    def test_at_threshold(self):
        """Returns False when new_tokens == threshold (not exceeding)."""
        from vllm_mlx.cloud_router import CloudRouter

        router = CloudRouter(cloud_model="test-model", threshold=1000)
        assert router.should_route_to_cloud(1000) is False

    def test_above_threshold(self):
        """Returns True when new_tokens > threshold."""
        from vllm_mlx.cloud_router import CloudRouter

        router = CloudRouter(cloud_model="test-model", threshold=1000)
        assert router.should_route_to_cloud(1001) is True

    def test_threshold_plus_one(self):
        """Returns True when new_tokens == threshold + 1."""
        from vllm_mlx.cloud_router import CloudRouter

        router = CloudRouter(cloud_model="test-model", threshold=5000)
        assert router.should_route_to_cloud(5001) is True

    def test_zero_tokens(self):
        """Returns False when new_tokens == 0."""
        from vllm_mlx.cloud_router import CloudRouter

        router = CloudRouter(cloud_model="test-model", threshold=100)
        assert router.should_route_to_cloud(0) is False

    def test_large_threshold(self):
        """Returns True for large token counts exceeding large threshold."""
        from vllm_mlx.cloud_router import CloudRouter

        router = CloudRouter(cloud_model="test-model", threshold=50000)
        assert router.should_route_to_cloud(60000) is True
        assert router.should_route_to_cloud(50000) is False


class TestCloudRouterBuildCallKwargs:
    """Tests for CloudRouter._build_call_kwargs method."""

    def test_basic_kwargs(self):
        """Correctly builds kwargs with basic parameters."""
        from vllm_mlx.cloud_router import CloudRouter

        router = CloudRouter(cloud_model="anthropic/claude-sonnet-4-5", threshold=1000)
        messages = [{"role": "user", "content": "Hello"}]

        kwargs = router._build_call_kwargs(
            messages=messages,
            stream=True,
            temperature=0.8,
            max_tokens=100,
        )

        assert kwargs["model"] == "anthropic/claude-sonnet-4-5"
        assert kwargs["messages"] == messages
        assert kwargs["stream"] is True
        assert kwargs["temperature"] == 0.8
        assert kwargs["max_tokens"] == 100

    def test_passes_through_top_p(self):
        """Passes through top_p parameter."""
        from vllm_mlx.cloud_router import CloudRouter

        router = CloudRouter(cloud_model="test-model", threshold=1000)
        messages = [{"role": "user", "content": "Hello"}]

        kwargs = router._build_call_kwargs(
            messages=messages,
            stream=False,
            top_p=0.95,
        )

        assert kwargs["top_p"] == 0.95

    def test_passes_through_tools(self):
        """Passes through tools parameter."""
        from vllm_mlx.cloud_router import CloudRouter

        router = CloudRouter(cloud_model="test-model", threshold=1000)
        messages = [{"role": "user", "content": "Hello"}]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {},
                },
            }
        ]

        kwargs = router._build_call_kwargs(
            messages=messages,
            stream=False,
            tools=tools,
        )

        assert kwargs["tools"] == tools

    def test_omits_none_values(self):
        """Omits parameters that are None."""
        from vllm_mlx.cloud_router import CloudRouter

        router = CloudRouter(cloud_model="test-model", threshold=1000)
        messages = [{"role": "user", "content": "Hello"}]

        kwargs = router._build_call_kwargs(
            messages=messages,
            stream=False,
            temperature=None,
            max_tokens=None,
            top_p=None,
            tools=None,
        )

        # Should only have model, messages, stream
        assert "temperature" not in kwargs
        assert "max_tokens" not in kwargs
        assert "top_p" not in kwargs
        assert "tools" not in kwargs
        assert kwargs["model"] == "test-model"
        assert kwargs["messages"] == messages
        assert kwargs["stream"] is False

    def test_ignores_unsupported_kwargs(self):
        """Ignores kwargs not in the supported list."""
        from vllm_mlx.cloud_router import CloudRouter

        router = CloudRouter(cloud_model="test-model", threshold=1000)
        messages = [{"role": "user", "content": "Hello"}]

        kwargs = router._build_call_kwargs(
            messages=messages,
            stream=False,
            unsupported_param="should_be_ignored",
        )

        assert "unsupported_param" not in kwargs

    def test_passes_through_response_format(self):
        """response_format is forwarded to litellm (regression: was silently dropped)."""
        from vllm_mlx.cloud_router import CloudRouter

        router = CloudRouter(cloud_model="test-model", threshold=1000)
        messages = [{"role": "user", "content": "Hello"}]
        rf = {
            "type": "json_schema",
            "json_schema": {"name": "out", "schema": {"type": "object"}},
        }

        kwargs = router._build_call_kwargs(
            messages=messages,
            stream=False,
            response_format=rf,
        )

        assert kwargs["response_format"] == rf

    def test_response_format_none_omitted(self):
        """response_format=None is not included in kwargs."""
        from vllm_mlx.cloud_router import CloudRouter

        router = CloudRouter(cloud_model="test-model", threshold=1000)
        messages = [{"role": "user", "content": "Hello"}]

        kwargs = router._build_call_kwargs(
            messages=messages,
            stream=False,
            response_format=None,
        )

        assert "response_format" not in kwargs


class TestCloudRouterLazyImport:
    """Tests for CloudRouter lazy litellm import."""

    def test_litellm_none_initially(self):
        """_litellm is None until first use."""
        from vllm_mlx.cloud_router import CloudRouter

        router = CloudRouter(cloud_model="test-model", threshold=1000)
        assert router._litellm is None

    def test_get_litellm_imports(self):
        """_get_litellm imports litellm on first call."""
        from vllm_mlx.cloud_router import CloudRouter

        router = CloudRouter(cloud_model="test-model", threshold=1000)

        # Mock the litellm module in sys.modules
        mock_litellm = MagicMock()
        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            result = router._get_litellm()
            assert result is mock_litellm
            assert router._litellm is mock_litellm

    def test_get_litellm_cached(self):
        """Subsequent _get_litellm calls return cached instance."""
        from vllm_mlx.cloud_router import CloudRouter

        router = CloudRouter(cloud_model="test-model", threshold=1000)
        mock_lib = MagicMock()
        router._litellm = mock_lib

        # Should return cached instance without re-importing
        result = router._get_litellm()
        assert result is mock_lib


class TestCloudRouterCompletion:
    """Tests for CloudRouter.completion method."""

    @pytest.mark.asyncio
    async def test_completion_returns_dict(self):
        """completion() returns a dict from litellm response."""
        from vllm_mlx.cloud_router import CloudRouter

        router = CloudRouter(cloud_model="test-model", threshold=1000)

        # Mock litellm response
        mock_response = MagicMock()
        mock_response.model_dump.return_value = {
            "id": "resp-123",
            "choices": [{"message": {"content": "Hello!"}}],
        }

        mock_litellm = MagicMock()
        mock_litellm.acompletion = AsyncMock(return_value=mock_response)
        router._litellm = mock_litellm

        messages = [{"role": "user", "content": "Hi"}]
        result = await router.completion(messages, temperature=0.7)

        assert isinstance(result, dict)
        assert result["id"] == "resp-123"
        mock_litellm.acompletion.assert_called_once()

    @pytest.mark.asyncio
    async def test_completion_passes_kwargs(self):
        """completion() passes kwargs to litellm."""
        from vllm_mlx.cloud_router import CloudRouter

        router = CloudRouter(cloud_model="gpt-4", threshold=1000)

        mock_response = MagicMock()
        mock_response.model_dump.return_value = {}

        mock_litellm = MagicMock()
        mock_litellm.acompletion = AsyncMock(return_value=mock_response)
        router._litellm = mock_litellm

        messages = [{"role": "user", "content": "Hi"}]
        await router.completion(
            messages,
            temperature=0.5,
            max_tokens=200,
            top_p=0.9,
        )

        call_args = mock_litellm.acompletion.call_args[1]
        assert call_args["model"] == "gpt-4"
        assert call_args["temperature"] == 0.5
        assert call_args["max_tokens"] == 200
        assert call_args["top_p"] == 0.9
        assert call_args["stream"] is False


class TestCloudRouterStreamCompletion:
    """Tests for CloudRouter.stream_completion method."""

    @pytest.mark.asyncio
    async def test_stream_yields_sse_chunks(self):
        """stream_completion() yields SSE-formatted chunks."""
        from vllm_mlx.cloud_router import CloudRouter

        router = CloudRouter(cloud_model="test-model", threshold=1000)

        # Mock streaming response chunks
        @dataclass
        class MockDelta:
            role: str = None
            content: str = None
            tool_calls: list = None

        @dataclass
        class MockChoice:
            delta: MockDelta
            finish_reason: str = None

        @dataclass
        class MockChunk:
            choices: list

        chunks = [
            MockChunk(choices=[MockChoice(delta=MockDelta(role="assistant"))]),
            MockChunk(choices=[MockChoice(delta=MockDelta(content="Hello"))]),
            MockChunk(choices=[MockChoice(delta=MockDelta(content=" world"))]),
            MockChunk(choices=[MockChoice(delta=MockDelta(), finish_reason="stop")]),
        ]

        async def mock_stream():
            for chunk in chunks:
                yield chunk

        mock_litellm = MagicMock()
        mock_litellm.acompletion = AsyncMock(return_value=mock_stream())
        router._litellm = mock_litellm

        messages = [{"role": "user", "content": "Hi"}]
        result_chunks = []
        async for chunk in router.stream_completion(messages):
            result_chunks.append(chunk)

        # Should have chunks + [DONE]
        assert len(result_chunks) > 0
        assert result_chunks[-1] == "data: [DONE]\n\n"

        # Check SSE format
        for chunk in result_chunks[:-1]:
            assert chunk.startswith("data: ")
            assert chunk.endswith("\n\n")

    @pytest.mark.asyncio
    async def test_stream_formats_sse_correctly(self):
        """stream_completion() formats SSE chunks with proper structure."""
        from vllm_mlx.cloud_router import CloudRouter

        router = CloudRouter(cloud_model="test-model", threshold=1000)

        # Mock minimal streaming response
        @dataclass
        class MockDelta:
            content: str = "test"

        @dataclass
        class MockChoice:
            delta: MockDelta
            finish_reason: str = None

        @dataclass
        class MockChunk:
            choices: list

        async def mock_stream():
            yield MockChunk(choices=[MockChoice(delta=MockDelta())])

        mock_litellm = MagicMock()
        mock_litellm.acompletion = AsyncMock(return_value=mock_stream())
        router._litellm = mock_litellm

        messages = [{"role": "user", "content": "Hi"}]
        result_chunks = []
        async for chunk in router.stream_completion(
            messages, model_name="custom-model"
        ):
            if chunk != "data: [DONE]\n\n":
                result_chunks.append(chunk)

        # Parse first SSE chunk
        if result_chunks:
            sse_data = result_chunks[0].replace("data: ", "").strip()
            parsed = json.loads(sse_data)

            assert parsed["object"] == "chat.completion.chunk"
            assert parsed["model"] == "custom-model"
            assert "choices" in parsed
            assert isinstance(parsed["choices"], list)

    @pytest.mark.asyncio
    async def test_stream_empty_choices_skipped(self):
        """stream_completion() skips chunks with empty choices."""
        from vllm_mlx.cloud_router import CloudRouter

        router = CloudRouter(cloud_model="test-model", threshold=1000)

        @dataclass
        class MockChunk:
            choices: list

        async def mock_stream():
            yield MockChunk(choices=[])  # Empty choices
            yield MockChunk(choices=[])

        mock_litellm = MagicMock()
        mock_litellm.acompletion = AsyncMock(return_value=mock_stream())
        router._litellm = mock_litellm

        messages = [{"role": "user", "content": "Hi"}]
        result_chunks = []
        async for chunk in router.stream_completion(messages):
            result_chunks.append(chunk)

        # Should only have [DONE] since all chunks have empty choices
        assert result_chunks == ["data: [DONE]\n\n"]


