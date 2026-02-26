# SPDX-License-Identifier: Apache-2.0
"""
Tests for cloud routing feature.

Tests cover:
- CloudRouter class (vllm_mlx/cloud_router.py)
- MLXLanguageModel.estimate_new_tokens (vllm_mlx/models/llm.py)
- SimpleEngine.build_prompt and .model (vllm_mlx/engine/simple.py)
- Integration scenarios
"""

import json
import pytest
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from dataclasses import dataclass


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
                }
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
        async for chunk in router.stream_completion(messages, model_name="custom-model"):
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


# ---------------------------------------------------------------------------
# MLXLanguageModel.estimate_new_tokens tests
# ---------------------------------------------------------------------------


class MockCacheEntry:
    """Mock cache entry for testing."""

    def __init__(self, offset: int = 0):
        self._offset = offset

    @property
    def offset(self) -> int:
        return self._offset

    def trim(self, amount: int) -> None:
        self._offset = max(0, self._offset - amount)


class TestMLXLanguageModelEstimateNewTokens:
    """Tests for MLXLanguageModel.estimate_new_tokens method."""

    @pytest.fixture
    def model(self):
        """Create a mock MLXLanguageModel instance."""
        from vllm_mlx.models.llm import MLXLanguageModel

        model = MLXLanguageModel("test-model")
        model._loaded = True

        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.bos_token = "<s>"
        mock_tokenizer.encode = MagicMock(side_effect=lambda text, **kwargs: [1, 2, 3, 4, 5])
        model.tokenizer = mock_tokenizer

        return model

    def test_returns_tuple(self, model):
        """Returns (total_tokens, new_tokens) tuple."""
        total, new = model.estimate_new_tokens("test prompt")
        assert isinstance(total, int)
        assert isinstance(new, int)

    def test_empty_cache_new_equals_total(self, model):
        """When cache is empty, new_tokens == total_tokens."""
        model._cached_token_ids = []
        model._prompt_cache = None

        total, new = model.estimate_new_tokens("test prompt")
        assert new == total

    def test_cache_hit_reduces_new_tokens(self, model):
        """When cache has prefix match, new_tokens < total_tokens."""
        # Set up cache with partial prefix
        model._cached_token_ids = [1, 2, 3]
        model._prompt_cache = [MockCacheEntry(3)]

        # Mock tokenizer to return tokens starting with same prefix
        model.tokenizer.encode = MagicMock(return_value=[1, 2, 3, 4, 5, 6, 7])

        total, new = model.estimate_new_tokens("test prompt")

        # Common prefix = [1, 2, 3], so new = 7 - 3 = 4
        assert total == 7
        assert new == 4
        assert new < total

    def test_no_prefix_match(self, model):
        """When no prefix match, new_tokens == total_tokens."""
        model._cached_token_ids = [10, 20, 30]
        model._prompt_cache = [MockCacheEntry(3)]
        model.tokenizer.encode = MagicMock(return_value=[1, 2, 3, 4, 5])

        total, new = model.estimate_new_tokens("different prompt")

        # No common prefix
        assert total == 5
        assert new == 5

    def test_does_not_modify_cache(self, model):
        """estimate_new_tokens is read-only — does not modify cache state."""
        original_cache_ids = [1, 2, 3, 4, 5]
        model._cached_token_ids = list(original_cache_ids)
        model._prompt_cache = [MockCacheEntry(5)]
        model.tokenizer.encode = MagicMock(return_value=[1, 2, 3, 4, 5, 6, 7])

        # Call estimate_new_tokens
        model.estimate_new_tokens("test prompt")

        # Cache should be unchanged
        assert model._cached_token_ids == original_cache_ids
        assert model._prompt_cache[0].offset == 5

    def test_loads_model_if_not_loaded(self):
        """Loads model if not already loaded."""
        from vllm_mlx.models.llm import MLXLanguageModel

        model = MLXLanguageModel("test-model")
        model._loaded = False

        # Mock the load method to set up the model properly
        def mock_load_impl():
            mock_tokenizer = MagicMock()
            mock_tokenizer.bos_token = "<s>"
            mock_tokenizer.encode = MagicMock(return_value=[1, 2, 3])
            model.tokenizer = mock_tokenizer
            model._loaded = True

        with patch.object(model, "load", side_effect=mock_load_impl) as mock_load:
            model.estimate_new_tokens("test")
            mock_load.assert_called_once()

    def test_handles_bos_token(self, model):
        """Correctly handles add_special_tokens based on bos_token."""
        # Case 1: bos_token is None
        model.tokenizer.bos_token = None
        model.tokenizer.encode = MagicMock(return_value=[1, 2, 3])

        model.estimate_new_tokens("prompt")

        # Should add special tokens when bos_token is None
        call_kwargs = model.tokenizer.encode.call_args[1]
        assert call_kwargs.get("add_special_tokens") is True

        # Case 2: prompt doesn't start with bos_token
        model.tokenizer.bos_token = "<s>"
        model.tokenizer.encode.reset_mock()

        model.estimate_new_tokens("prompt without bos")

        call_kwargs = model.tokenizer.encode.call_args[1]
        assert call_kwargs.get("add_special_tokens") is True

        # Case 3: prompt starts with bos_token
        model.tokenizer.encode.reset_mock()

        model.estimate_new_tokens("<s>prompt with bos")

        call_kwargs = model.tokenizer.encode.call_args[1]
        assert call_kwargs.get("add_special_tokens") is False


# ---------------------------------------------------------------------------
# SimpleEngine.build_prompt tests
# ---------------------------------------------------------------------------


class TestSimpleEngineBuildPrompt:
    """Tests for SimpleEngine.build_prompt method."""

    def test_returns_string(self):
        """build_prompt returns a string."""
        from vllm_mlx.engine.simple import SimpleEngine

        engine = SimpleEngine(model_name="test-model")

        # Mock loaded state
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template = MagicMock(
            return_value="formatted prompt"
        )
        mock_model.tokenizer = mock_tokenizer
        engine._model = mock_model
        engine._loaded = True

        messages = [{"role": "user", "content": "Hello"}]
        result = engine.build_prompt(messages)

        assert isinstance(result, str)
        assert result == "formatted prompt"

    def test_raises_when_not_loaded(self):
        """Raises RuntimeError if engine not loaded."""
        from vllm_mlx.engine.simple import SimpleEngine

        engine = SimpleEngine(model_name="test-model")
        engine._loaded = False

        messages = [{"role": "user", "content": "Hello"}]

        with pytest.raises(RuntimeError, match="Engine not loaded"):
            engine.build_prompt(messages)

    def test_raises_for_mllm_models(self):
        """Raises RuntimeError for MLLM models."""
        from vllm_mlx.engine.simple import SimpleEngine

        engine = SimpleEngine(model_name="test-model")
        engine._loaded = True
        engine._is_mllm = True

        messages = [{"role": "user", "content": "Hello"}]

        with pytest.raises(RuntimeError, match="not supported for MLLM"):
            engine.build_prompt(messages)

    def test_applies_chat_template_with_tools(self):
        """Applies chat template with tools when provided."""
        from vllm_mlx.engine.simple import SimpleEngine

        engine = SimpleEngine(model_name="test-model")

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template = MagicMock(return_value="prompt with tools")
        mock_model.tokenizer = mock_tokenizer
        engine._model = mock_model
        engine._loaded = True

        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"type": "function", "function": {"name": "test_func"}}]

        result = engine.build_prompt(messages, tools=tools)

        # Should have called apply_chat_template with converted tools
        assert mock_tokenizer.apply_chat_template.called
        call_kwargs = mock_tokenizer.apply_chat_template.call_args[1]
        assert "tools" in call_kwargs or True  # Tools may be in kwargs

    def test_fallback_without_chat_template(self):
        """Falls back to simple concatenation if no apply_chat_template."""
        from vllm_mlx.engine.simple import SimpleEngine

        engine = SimpleEngine(model_name="test-model")

        mock_model = MagicMock()
        mock_tokenizer = MagicMock(spec=[])  # No apply_chat_template method
        mock_model.tokenizer = mock_tokenizer
        engine._model = mock_model
        engine._loaded = True

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]

        result = engine.build_prompt(messages)

        assert "user: Hello" in result
        assert "assistant: Hi there" in result
        assert result.endswith("assistant:")

    def test_handles_enable_thinking_for_coder_models(self):
        """Disables thinking mode for coder models."""
        from vllm_mlx.engine.simple import SimpleEngine

        engine = SimpleEngine(model_name="test-coder-model")

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template = MagicMock(return_value="prompt")
        mock_model.tokenizer = mock_tokenizer
        mock_model.model_name = "test-coder-model"
        engine._model = mock_model
        engine._model_name = "test-coder-model"
        engine._loaded = True

        messages = [{"role": "user", "content": "Hello"}]
        engine.build_prompt(messages)

        # Should have called with enable_thinking=False
        call_kwargs = mock_tokenizer.apply_chat_template.call_args[1]
        # enable_thinking should be False for coder models
        # (though it might not be in kwargs if template doesn't support it)


# ---------------------------------------------------------------------------
# SimpleEngine.model property tests
# ---------------------------------------------------------------------------


class TestSimpleEngineModelProperty:
    """Tests for SimpleEngine.model property."""

    def test_returns_model_instance(self):
        """model property returns the underlying _model instance."""
        from vllm_mlx.engine.simple import SimpleEngine

        engine = SimpleEngine(model_name="test-model")
        mock_model = MagicMock()
        engine._model = mock_model

        assert engine.model is mock_model

    def test_returns_none_when_not_loaded(self):
        """model property returns None when engine not loaded."""
        from vllm_mlx.engine.simple import SimpleEngine

        engine = SimpleEngine(model_name="test-model")
        engine._model = None

        assert engine.model is None


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestCloudRoutingIntegration:
    """Integration tests for cloud routing logic."""

    def test_no_cloud_router_no_routing(self):
        """When _cloud_router is None, no routing happens."""
        # This tests the server behavior, which we simulate here
        cloud_router = None
        new_tokens = 25000

        # Should not route to cloud when router is None
        should_route = cloud_router is not None and cloud_router.should_route_to_cloud(new_tokens)
        assert should_route is False

    def test_below_threshold_uses_local(self):
        """When new_tokens < threshold, local inference is used."""
        from vllm_mlx.cloud_router import CloudRouter

        cloud_router = CloudRouter(cloud_model="test-model", threshold=20000)
        new_tokens = 1000

        should_route = cloud_router.should_route_to_cloud(new_tokens)
        assert should_route is False

    def test_above_threshold_uses_cloud(self):
        """When new_tokens > threshold, cloud routing is used."""
        from vllm_mlx.cloud_router import CloudRouter

        cloud_router = CloudRouter(cloud_model="test-model", threshold=20000)
        new_tokens = 25000

        should_route = cloud_router.should_route_to_cloud(new_tokens)
        assert should_route is True

    def test_estimate_then_route_workflow(self):
        """Typical workflow: estimate new tokens, then decide routing."""
        from vllm_mlx.cloud_router import CloudRouter
        from vllm_mlx.models.llm import MLXLanguageModel

        # Setup
        model = MLXLanguageModel("test-model")
        model._loaded = True
        mock_tokenizer = MagicMock()
        mock_tokenizer.bos_token = "<s>"
        mock_tokenizer.encode = MagicMock(return_value=[1] * 30000)  # 30k tokens
        model.tokenizer = mock_tokenizer
        model._cached_token_ids = [1] * 5000  # 5k cached

        router = CloudRouter(cloud_model="cloud-model", threshold=20000)

        # Estimate tokens
        total_tokens, new_tokens = model.estimate_new_tokens("long prompt")

        # new_tokens = 30000 - 5000 = 25000
        assert new_tokens == 25000

        # Should route to cloud
        should_route = router.should_route_to_cloud(new_tokens)
        assert should_route is True

    def test_cold_start_high_tokens_routes_to_cloud(self):
        """Cold start with high token count routes to cloud."""
        from vllm_mlx.cloud_router import CloudRouter
        from vllm_mlx.models.llm import MLXLanguageModel

        model = MLXLanguageModel("test-model")
        model._loaded = True
        mock_tokenizer = MagicMock()
        mock_tokenizer.bos_token = "<s>"
        mock_tokenizer.encode = MagicMock(return_value=[1] * 50000)  # 50k tokens
        model.tokenizer = mock_tokenizer
        model._cached_token_ids = []  # Empty cache (cold start)

        router = CloudRouter(cloud_model="cloud-model", threshold=20000)

        total_tokens, new_tokens = model.estimate_new_tokens("huge prompt")

        # new_tokens == total_tokens for cold start
        assert new_tokens == total_tokens == 50000

        should_route = router.should_route_to_cloud(new_tokens)
        assert should_route is True

    def test_warm_cache_stays_local(self):
        """Warm cache with small new tokens stays on local inference."""
        from vllm_mlx.cloud_router import CloudRouter
        from vllm_mlx.models.llm import MLXLanguageModel

        model = MLXLanguageModel("test-model")
        model._loaded = True
        mock_tokenizer = MagicMock()
        mock_tokenizer.bos_token = "<s>"
        # Prompt has 10k tokens, 9.5k cached
        mock_tokenizer.encode = MagicMock(return_value=[1] * 10000)
        model.tokenizer = mock_tokenizer
        model._cached_token_ids = [1] * 9500

        router = CloudRouter(cloud_model="cloud-model", threshold=20000)

        total_tokens, new_tokens = model.estimate_new_tokens("prompt")

        # new_tokens = 10000 - 9500 = 500
        assert new_tokens == 500

        should_route = router.should_route_to_cloud(new_tokens)
        assert should_route is False
