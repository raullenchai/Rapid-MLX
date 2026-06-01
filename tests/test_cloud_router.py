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


# ---------------------------------------------------------------------------
# Engine contract — regression gates for #500
# ---------------------------------------------------------------------------


class TestEngineCloudRoutingContract:
    """Pins the engine API that ``routes/chat.py`` cloud-routing depends on.

    These tests would have caught issue #500 (cloud routing silently never
    fires) where ``BatchedEngine`` did not implement ``build_prompt``. The
    bug was a regression introduced by #155 (deletion of ``SimpleEngine``,
    which previously hosted the method). Because the call sites in
    ``routes/chat.py`` were guarded by ``hasattr(engine, "build_prompt")``,
    the failure was silent: cloud routing was disabled for every user with
    no log line, while the startup banner still printed
    ``"Cloud routing enabled: ..."``.
    """

    def test_batched_engine_exposes_build_prompt(self):
        """BatchedEngine MUST expose ``build_prompt`` — cloud routing in
        ``routes/chat.py`` depends on the method existing on the live engine
        (not just on test mocks). #500.
        """
        from vllm_mlx.engine.batched import BatchedEngine

        assert hasattr(BatchedEngine, "build_prompt"), (
            "BatchedEngine.build_prompt is required for cloud routing "
            "(routes/chat.py) and streaming chat-template validation. "
            "If you removed it, also remove the call sites in routes/chat.py "
            "and the --cloud-model CLI flag."
        )
        # And it must be a callable, not a class attribute placeholder.
        assert callable(BatchedEngine.build_prompt)

    def test_chat_route_does_not_guard_on_build_prompt_existence(self):
        """``routes/chat.py`` must NOT wrap ``engine.build_prompt`` in
        ``hasattr(engine, "build_prompt")``. That guard was the mechanism
        that silently disabled cloud routing in #500 — when ``SimpleEngine``
        was deleted, the guard turned false and there was no signal at
        runtime that anything had broken.

        If a future engine genuinely doesn't support prompt rendering, fail
        loudly at engine construction or at the call site — not by silently
        disabling cloud routing.
        """
        import pathlib

        src = pathlib.Path("vllm_mlx/routes/chat.py").read_text()
        assert 'hasattr(engine, "build_prompt")' not in src, (
            'Found `hasattr(engine, "build_prompt")` in routes/chat.py. '
            "This guard silently disables cloud routing if the engine class "
            "doesn't expose the method — exactly the failure mode of #500. "
            "Remove the guard; require all production engines to implement "
            "build_prompt (it's now on the BaseEngine contract)."
        )

    def test_batched_engine_exposes_estimate_new_tokens(self):
        """BatchedEngine MUST expose ``estimate_new_tokens`` — cloud routing
        in ``routes/chat.py`` calls it right after ``build_prompt`` to decide
        whether ``new_tokens > cloud_threshold``.

        Pre-#500-followup the route called ``engine.model.estimate_new_tokens``
        — that path raised ``AttributeError: 'BatchedEngine' object has no
        attribute 'model'`` and the try/except around the cloud branch
        silently logged "falling back to local". Same silent-skip pattern
        as the original #500 hasattr trap, one layer deeper.
        """
        from vllm_mlx.engine.batched import BatchedEngine

        assert hasattr(BatchedEngine, "estimate_new_tokens"), (
            "BatchedEngine.estimate_new_tokens is required for cloud routing "
            "(routes/chat.py uses it to compute new-token count vs threshold). "
            "If you removed it, cloud routing falls back to local on every "
            "request — see #500 follow-up."
        )
        assert callable(BatchedEngine.estimate_new_tokens)

    def test_chat_route_calls_engine_estimate_not_engine_model_estimate(self):
        """``routes/chat.py`` must call ``engine.estimate_new_tokens(...)``
        directly, NOT ``engine.model.estimate_new_tokens(...)``.

        BatchedEngine does not expose ``.model`` — that was a SimpleEngine
        attribute deleted in #155. The wrapper try/except in the route
        catches the AttributeError and logs a warning instead of routing,
        which is functionally identical to the cloud branch never firing.
        """
        import pathlib

        src = pathlib.Path("vllm_mlx/routes/chat.py").read_text()
        assert "engine.model.estimate_new_tokens" not in src, (
            "Found `engine.model.estimate_new_tokens` in routes/chat.py. "
            "BatchedEngine has no .model attribute (SimpleEngine convention, "
            "deleted in #155). Use `engine.estimate_new_tokens(prompt)` — the "
            "method is now part of the engine contract."
        )


# ---------------------------------------------------------------------------
# Live cloud-routing repro — would have caught #500 + the v0.6.70 hotfix
# ---------------------------------------------------------------------------


from vllm_mlx.engine.base import BaseEngine, GenerationOutput


class _ContractEngine(BaseEngine):
    """Real ``BaseEngine`` subclass — instantiation enforces the abstract
    contract, so a missing route-layer method (``build_prompt``,
    ``estimate_new_tokens``) raises ``TypeError`` at construction instead of
    a silent runtime degradation.

    This is the property the previous regression tests lacked: every
    existing route test mocks the engine with ``MagicMock``, which
    auto-satisfies any attribute access and so let #500 and the v0.6.70
    hotfix ship green. Subclassing ``BaseEngine`` here means a future PR
    can't remove ``build_prompt`` without this file failing to import.

    Only the slice the cloud-routing branch needs is wired up; the
    remaining abstracts return placeholders so the ABC check passes.
    """

    preserve_native_tool_format = False

    def __init__(self, *, prompt_tokens: int):
        self._prompt_tokens = prompt_tokens
        self.build_prompt_calls: list[dict] = []
        self.estimate_calls: list[str] = []
        self.chat_calls: list[dict] = []

    @property
    def model_name(self) -> str:
        return "test-model"

    @property
    def is_mllm(self) -> bool:
        return False

    @property
    def tokenizer(self):
        return None

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    def build_prompt(
        self,
        messages,
        tools=None,
        enable_thinking=None,
    ) -> str:
        self.build_prompt_calls.append(
            {
                "messages": messages,
                "tools": tools,
                "enable_thinking": enable_thinking,
            }
        )
        return "RENDERED_PROMPT"

    def estimate_new_tokens(self, prompt: str) -> tuple[int, int]:
        self.estimate_calls.append(prompt)
        return self._prompt_tokens, self._prompt_tokens

    async def generate(self, prompt, **kwargs):  # pragma: no cover
        raise NotImplementedError

    async def stream_generate(self, prompt, **kwargs):  # pragma: no cover
        if False:
            yield None

    async def chat(self, messages, **kwargs):
        self.chat_calls.append({"messages": messages, "kwargs": kwargs})
        return GenerationOutput(
            text="local",
            new_text="local",
            tokens=[1],
            prompt_tokens=4,
            completion_tokens=1,
            finished=True,
            finish_reason="stop",
            channel=None,
        )

    async def stream_chat(self, messages, **kwargs):  # pragma: no cover
        if False:
            yield None


def _make_cloud_routed_client(
    *,
    prompt_tokens: int,
    threshold: int,
    cloud_response: dict | None = None,
):
    """Wire ``routes/chat.py`` against a ``_ContractEngine`` + a stubbed
    ``CloudRouter`` whose ``completion()`` returns ``cloud_response``.

    Returns ``(client, engine, cloud_router)`` so tests can inspect both the
    HTTP response AND whether the engine methods were actually called (the
    silent-skip bug signature).
    """
    from unittest.mock import AsyncMock, MagicMock

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from vllm_mlx.cloud_router import CloudRouter
    from vllm_mlx.config import reset_config
    from vllm_mlx.routes.chat import router as chat_router

    cfg = reset_config()
    engine = _ContractEngine(prompt_tokens=prompt_tokens)
    cfg.engine = engine
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.reasoning_parser = None
    cfg.tool_parser = None
    cfg.no_thinking = True

    cloud_router = CloudRouter(cloud_model="test/cloud", threshold=threshold)
    if cloud_response is not None:
        mock_litellm = MagicMock()
        mock_resp = MagicMock()
        mock_resp.model_dump.return_value = cloud_response
        mock_litellm.acompletion = AsyncMock(return_value=mock_resp)
        cloud_router._litellm = mock_litellm
    cfg.cloud_router = cloud_router

    app = FastAPI()
    app.include_router(chat_router)
    return TestClient(app), engine, cloud_router


@pytest.fixture
def _reset_after():
    yield
    from vllm_mlx.config import reset_config

    reset_config()


class TestCloudRoutingFiresEndToEnd:
    """Repro for #500 + the v0.6.70 hotfix.

    Before the fixes, this class of test would have failed in two ways:

    * #500 / pre-v0.6.69: ``hasattr(engine, "build_prompt")`` returned False
      → cloud branch skipped, response came from the local engine.
    * v0.6.70 hotfix / pre-3839a1b: route called ``engine.model.estimate_
      new_tokens`` → AttributeError, try/except logged
      ``[CLOUD ROUTE] Error during routing check ... falling back to local``,
      response came from the local engine.

    Both failure modes produce the same observable: the response payload
    comes from the LOCAL engine instead of the cloud stub. This test
    asserts the cloud branch is reached, the methods are called in order,
    and the cloud response is returned.
    """

    def test_above_threshold_routes_to_cloud(self, _reset_after):
        client, engine, _ = _make_cloud_routed_client(
            prompt_tokens=500,
            threshold=10,
            cloud_response={
                "id": "cloud-resp-1",
                "model": "test/cloud",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "ROUTED_TO_CLOUD",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 500,
                    "completion_tokens": 3,
                    "total_tokens": 503,
                },
            },
        )

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "doesn't matter"}],
                "max_tokens": 16,
            },
        )

        assert resp.status_code == 200, resp.text
        body = resp.json()
        # The cloud branch fired → response carries the cloud content,
        # not the local "local" string.
        assert body["choices"][0]["message"]["content"] == "ROUTED_TO_CLOUD", (
            "cloud-routing branch did not return the cloud response — "
            "the route silently fell back to local (see #500, v0.6.70 hotfix). "
            f"got: {body['choices'][0]['message']}"
        )
        # The contract methods were actually called.
        assert engine.build_prompt_calls, (
            "engine.build_prompt was NOT called — the cloud branch was skipped "
            "before reaching the body (the #500 silent-skip shape)."
        )
        assert engine.estimate_calls == ["RENDERED_PROMPT"], (
            "engine.estimate_new_tokens was NOT called with the rendered prompt "
            "— the cloud branch crashed silently before reaching this line "
            "(the v0.6.70 hotfix shape)."
        )
        # And the local chat() path was NOT exercised.
        assert engine.chat_calls == [], (
            "engine.chat was called even though the request should have routed "
            "to cloud — the cloud branch fell through to local."
        )

    def test_below_threshold_stays_local(self, _reset_after):
        client, engine, _ = _make_cloud_routed_client(
            prompt_tokens=5,
            threshold=10,
            cloud_response=None,
        )

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 16,
            },
        )

        assert resp.status_code == 200, resp.text
        # Local path → content is "local" from the stub.
        assert resp.json()["choices"][0]["message"]["content"] == "local"
        # estimate_new_tokens still ran (the route always evaluates the
        # threshold before deciding), but the cloud branch chose local.
        assert engine.estimate_calls == ["RENDERED_PROMPT"]
        assert engine.chat_calls, "local engine.chat was not called"
