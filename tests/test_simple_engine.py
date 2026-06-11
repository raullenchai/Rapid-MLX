# SPDX-License-Identifier: Apache-2.0
"""Tests for SimpleEngine concurrency handling."""

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import mlx.core as mx
import pytest

pytestmark = pytest.mark.anyio


class TestSimpleEngineConcurrency:
    """Test SimpleEngine lock behavior with concurrent requests."""

    @pytest.fixture
    def anyio_backend(self):
        return "asyncio"

    @pytest.fixture
    def mock_model(self):
        """Create a mock model that tracks concurrent calls."""
        model = MagicMock()
        model.tokenizer = MagicMock()
        model.tokenizer.encode = MagicMock(return_value=[1, 2, 3])

        # Track concurrent executions
        model._concurrent_count = 0
        model._max_concurrent = 0

        def generate_side_effect(**kwargs):
            model._concurrent_count += 1
            model._max_concurrent = max(model._max_concurrent, model._concurrent_count)
            # Simulate some work
            import time

            time.sleep(0.05)
            model._concurrent_count -= 1
            result = MagicMock()
            result.text = "test response"
            result.tokens = [1, 2, 3]
            result.finish_reason = "stop"
            return result

        model.generate = MagicMock(side_effect=generate_side_effect)

        # stream_generate tracks concurrency the same way so tests that
        # exercise SimpleEngine.generate() (which is now an accumulator
        # over stream_generate) see the same serialization behavior.
        def stream_generate_side_effect(**kwargs):
            model._concurrent_count += 1
            model._max_concurrent = max(model._max_concurrent, model._concurrent_count)
            import time

            time.sleep(0.05)
            model._concurrent_count -= 1
            chunk = MagicMock()
            chunk.text = "test response"
            chunk.tokens = [1, 2, 3]
            chunk.finished = True
            chunk.finish_reason = "stop"
            chunk.prompt_tokens = 3
            chunk.completion_tokens = 3
            yield chunk

        model.stream_generate = MagicMock(side_effect=stream_generate_side_effect)
        return model

    @pytest.fixture
    def mock_llm_model(self):
        """Create a mock LLM model."""
        model = MagicMock()
        model.tokenizer = MagicMock()
        model.tokenizer.encode = MagicMock(return_value=[1, 2, 3])

        # Track concurrent executions
        model._concurrent_count = 0
        model._max_concurrent = 0

        def chat_side_effect(**kwargs):
            model._concurrent_count += 1
            model._max_concurrent = max(model._max_concurrent, model._concurrent_count)
            import time

            time.sleep(0.05)
            model._concurrent_count -= 1
            result = MagicMock()
            result.text = "test response"
            result.tokens = [1, 2, 3]
            result.finish_reason = "stop"
            return result

        model.chat = MagicMock(side_effect=chat_side_effect)
        return model

    @pytest.mark.anyio
    async def test_lock_prevents_concurrent_generate(self, mock_model):
        """Test that the lock prevents concurrent generate calls."""
        from vllm_mlx.engine.simple import SimpleEngine

        with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False):
            engine = SimpleEngine("test-model")
            engine._model = mock_model
            engine._loaded = True
            engine._generation_lock_admission = "wait"

            # Launch multiple concurrent generate calls
            tasks = [
                engine.generate(prompt=f"test prompt {i}", max_tokens=10)
                for i in range(5)
            ]

            await asyncio.gather(*tasks)

            # With the lock, max concurrent should be 1
            assert mock_model._max_concurrent == 1, (
                f"Expected max concurrent to be 1, but got {mock_model._max_concurrent}. "
                "The lock is not working correctly."
            )

    @pytest.mark.anyio
    async def test_lock_prevents_concurrent_chat(self, mock_llm_model):
        """Test that the lock prevents concurrent chat calls."""
        from vllm_mlx.engine.simple import SimpleEngine

        with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False):
            engine = SimpleEngine("test-model")
            engine._model = mock_llm_model
            engine._loaded = True
            engine._generation_lock_admission = "wait"

            # Launch multiple concurrent chat calls
            tasks = [
                engine.chat(
                    messages=[{"role": "user", "content": f"test {i}"}], max_tokens=10
                )
                for i in range(5)
            ]

            await asyncio.gather(*tasks)

            # With the lock, max concurrent should be 1
            assert mock_llm_model._max_concurrent == 1, (
                f"Expected max concurrent to be 1, but got {mock_llm_model._max_concurrent}. "
                "The lock is not working correctly."
            )

    @pytest.mark.anyio
    async def test_default_admission_rejects_second_serialized_request(self):
        """Default SimpleEngine admission fails fast instead of queueing."""
        from vllm_mlx.engine.base import EngineBusy
        from vllm_mlx.engine.simple import SimpleEngine

        with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False):
            engine = SimpleEngine("test-model")
            engine._loaded = True

            started = threading.Event()
            release = threading.Event()

            def slow_call():
                started.set()
                release.wait(timeout=1.0)
                return "ok"

            first = asyncio.create_task(
                engine._run_blocking_serialized(
                    slow_call,
                    request_id="first-serialized-request",
                )
            )
            await asyncio.to_thread(started.wait, 1.0)

            with pytest.raises(EngineBusy) as excinfo:
                await engine._run_blocking_serialized(
                    lambda: "late",
                    request_id="second-serialized-request",
                )

            assert "text_generation_busy" == excinfo.value.code
            assert "active=first-serialized-request" in str(excinfo.value)
            assert engine._generation_busy_rejections == 1

            release.set()
            await first

    @pytest.mark.anyio
    async def test_blocking_serialized_tracks_active_request(self):
        """Busy errors include the current blocking serialized holder."""
        from vllm_mlx.engine.simple import SimpleEngine

        with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False):
            engine = SimpleEngine("test-model")
            engine._loaded = True

            def slow_call():
                import time

                time.sleep(0.05)
                return "ok"

            first = asyncio.create_task(
                engine._run_blocking_serialized(
                    slow_call,
                    request_id="probe-active-holder",
                )
            )
            for _ in range(100):
                if engine._generation_lock.locked():
                    break
                await asyncio.sleep(0.001)

            assert "probe-active-holder" in engine._generation_lock_holder_summary()

            await first
            assert engine._generation_lock_holder_summary() == "none"

    async def test_chat_with_tools_aggregates_streaming_path(self, mock_llm_model):
        """Tool-enabled non-stream chat should use the streaming path."""
        from vllm_mlx.engine.simple import SimpleEngine

        async def fake_stream_chat(*args, **kwargs):
            yield MagicMock(
                text="partial",
                tokens=[1],
                prompt_tokens=11,
                completion_tokens=1,
                finish_reason=None,
                finished=False,
            )
            yield MagicMock(
                text='<|im_end|><tool_call>{"name":"bash","arguments":{"command":"pwd"}}</tool_call>',
                tokens=[7, 8, 9],
                prompt_tokens=11,
                completion_tokens=4,
                finish_reason="stop",
                finished=True,
            )

        with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False):
            engine = SimpleEngine("test-model")
            engine._model = mock_llm_model
            engine._loaded = True
            engine.stream_chat = fake_stream_chat  # type: ignore[method-assign]

            output = await engine.chat(
                messages=[{"role": "user", "content": "run pwd"}],
                max_tokens=16,
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "bash",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
            )

            assert (
                output.text
                == '<tool_call>{"name":"bash","arguments":{"command":"pwd"}}</tool_call>'
            )
            assert output.tokens == [7, 8, 9]
            assert output.prompt_tokens == 11
            assert output.completion_tokens == 4
            assert output.finish_reason == "stop"
            mock_llm_model.chat.assert_not_called()

    @pytest.mark.anyio
    async def test_stream_chat_cache_path_accepts_pydantic_message_objects(self):
        """`stream_chat`'s declared signature is ``list[dict]`` but real callers
        (``server.py``'s streaming endpoint, ``test_server.py``'s direct
        invocations) pass Pydantic ``Message`` objects. The system-prefix
        KV-cache eligibility check on this path uses ``.get('role')`` /
        ``dict(m)`` semantics; without normalisation the iteration raises
        ``'Message' object has no attribute 'get'`` before the call ever
        reaches the underlying ``stream_generate``."""
        from vllm_mlx.api.models import Message
        from vllm_mlx.engine.simple import SimpleEngine

        # ``apply_chat_template`` returns identical strings for both
        # probe-divergence renders → boundary stays at 0 → the cache path
        # is correctly skipped and execution falls through to
        # ``self.stream_generate``. The test's value is asserting no
        # ``AttributeError`` leaks out of the message-normalisation step.
        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = (
            "<|im_start|>system\nyou are an assistant<|im_end|>\n"
            "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
        )
        tokenizer.bos_token = None
        tokenizer.encode = MagicMock(return_value=[1, 2, 3])

        model = MagicMock()
        model.tokenizer = tokenizer

        captured_stream_generate = []

        async def fake_stream_generate(*, prompt, **kwargs):
            captured_stream_generate.append({"prompt": prompt, "kwargs": kwargs})
            out = MagicMock(
                text="hi back",
                new_text="hi back",
                prompt_tokens=3,
                completion_tokens=1,
                finished=True,
                finish_reason="stop",
            )
            yield out

        with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False):
            engine = SimpleEngine("test-model")
            engine._model = model
            engine._loaded = True
            engine._supports_system_kv_cache = True
            engine.stream_generate = fake_stream_generate  # type: ignore[method-assign]

            messages = [
                Message(role="system", content="You are a helpful assistant."),
                Message(role="user", content="hi"),
            ]

            chunks = [c async for c in engine.stream_chat(messages=messages)]

        # No AttributeError was raised → normalisation worked.
        # apply_chat_template was called at least 3 times: once for the
        # initial ``prompt`` build and twice more for the Alpha/Bravo
        # probe-divergence renders.
        assert tokenizer.apply_chat_template.call_count >= 3
        assert len(captured_stream_generate) == 1
        assert chunks and chunks[0].text == "hi back"

    @pytest.mark.anyio
    async def test_stream_chat_skips_cache_path_when_no_system_message(self):
        """If the message list has no system role, the cache-eligibility
        check must short-circuit ``has_system = False`` without entering the
        probe-divergence step or the cache-aware streaming branch."""
        from vllm_mlx.engine.simple import SimpleEngine

        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = (
            "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n"
        )
        tokenizer.bos_token = None
        tokenizer.encode = MagicMock(return_value=[1, 2, 3])

        model = MagicMock()
        model.tokenizer = tokenizer

        called_stream_generate = []

        async def fake_stream_generate(*, prompt, **kwargs):
            called_stream_generate.append(prompt)
            out = MagicMock(
                text="hi",
                new_text="hi",
                prompt_tokens=3,
                completion_tokens=1,
                finished=True,
                finish_reason="stop",
            )
            yield out

        with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False):
            engine = SimpleEngine("test-model")
            engine._model = model
            engine._loaded = True
            engine.stream_generate = fake_stream_generate  # type: ignore[method-assign]

            chunks = [
                c
                async for c in engine.stream_chat(
                    messages=[{"role": "user", "content": "hello"}],
                )
            ]

        # No system → cache path skipped → only the initial apply_chat_template
        # call (for ``prompt``) happens, no probe renders.
        assert tokenizer.apply_chat_template.call_count == 1
        assert called_stream_generate
        assert chunks and chunks[0].text == "hi"

    @pytest.mark.anyio
    async def test_stream_chat_cache_path_falls_back_when_mlx_raises(self):
        """When the cache-aware ``_run_with_cache`` body raises *before* the
        first generated token, the path must surface the failure as a
        pre-first-token error and fall back to the uncached
        ``self.stream_generate`` instead of bubbling the exception out."""
        from vllm_mlx.engine.simple import SimpleEngine

        # Probe-divergence renders that DO diverge, so the cache path is
        # entered. The boundary lies after the system block, well past the
        # 16-char minimum.
        def apply_chat_template_side_effect(messages, **kwargs):
            # Find the last user message content to make probes diverge.
            user_content = ""
            for m in reversed(messages):
                role = (
                    m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
                )
                if role == "user":
                    content = (
                        m.get("content")
                        if isinstance(m, dict)
                        else getattr(m, "content", "")
                    )
                    user_content = content or ""
                    break
            return (
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                f"<|im_start|>user\n{user_content}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )

        tokenizer = MagicMock()
        tokenizer.apply_chat_template.side_effect = apply_chat_template_side_effect
        tokenizer.bos_token = None
        # Return long enough token lists that the system-prefix slice is
        # a proper prefix of the full sequence and ``kv_cache_eligible``
        # becomes True.
        tokenizer.encode = MagicMock(
            side_effect=[
                list(range(50)),  # full prompt
                list(range(20)),  # system prefix (proper prefix of above)
            ]
        )

        model = MagicMock()
        model.tokenizer = tokenizer
        # ``self._model.model`` is dereferenced inside _run_with_cache.
        model.model = MagicMock()

        fallback_calls = []

        async def fake_stream_generate(*, prompt, **kwargs):
            fallback_calls.append(prompt)
            out = MagicMock(
                text="fallback-response",
                new_text="fallback-response",
                prompt_tokens=50,
                completion_tokens=1,
                finished=True,
                finish_reason="stop",
            )
            yield out

        # Force the cache-aware path to raise before the first emit so we
        # exercise the pre-first-token error → uncached fallback branch.
        def make_prompt_cache_raises(*args, **kwargs):
            raise RuntimeError("simulated mlx-lm failure")

        with (
            patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False),
            patch(
                "mlx_lm.models.cache.make_prompt_cache",
                side_effect=make_prompt_cache_raises,
            ),
            patch("mlx_lm.sample_utils.make_sampler", return_value=MagicMock()),
        ):
            engine = SimpleEngine("test-model")
            engine._model = model
            engine._loaded = True
            engine.stream_generate = fake_stream_generate  # type: ignore[method-assign]

            chunks = [
                c
                async for c in engine.stream_chat(
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "hello"},
                    ],
                )
            ]

        # The cache-path failure must NOT propagate; we should see the
        # fallback's chunk instead.
        assert fallback_calls, "uncached stream_generate fallback was not invoked"
        assert chunks and chunks[0].text == "fallback-response"

    @pytest.mark.anyio
    async def test_stream_chat_skips_cache_path_when_decode_controls_present(self):
        """If the request carries ``stop`` or ``logits_processors`` (or any non-default
        ``top_k`` / ``min_p`` / ``presence_penalty`` / ``repetition_penalty``), the
        cache branch must be skipped so those controls flow through
        ``self.stream_generate``.
        The cache branch drives ``mlx_lm.stream_generate`` directly with only
        prompt/max_tokens/sampler/prompt_cache, silently dropping any other decode
        controls.
        Gating here keeps cache-eligible and uncached requests on identical decode
        semantics."""
        from vllm_mlx.engine.simple import SimpleEngine

        tokenizer = MagicMock()
        # Render contains a system block so ``has_system`` would otherwise send us
        # into the cache branch.
        tokenizer.apply_chat_template.return_value = (
            "<|im_start|>system\nYou are helpful.<|im_end|>\n"
            "<|im_start|>user\nhello<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        tokenizer.bos_token = None
        tokenizer.encode = MagicMock(return_value=[1, 2, 3])

        model = MagicMock()
        model.tokenizer = tokenizer

        fallback_kwargs: list[dict] = []

        async def fake_stream_generate(*, prompt, **kw):
            fallback_kwargs.append(kw)
            out = MagicMock(
                text="ok",
                new_text="ok",
                prompt_tokens=3,
                completion_tokens=1,
                finished=True,
                finish_reason="stop",
            )
            yield out

        sentinel_processor = MagicMock(name="logits_processor")

        with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False):
            engine = SimpleEngine("test-model")
            engine._model = model
            engine._loaded = True
            engine.stream_generate = fake_stream_generate  # type: ignore[method-assign]

            chunks = [
                c
                async for c in engine.stream_chat(
                    messages=[
                        {"role": "system", "content": "You are helpful."},
                        {"role": "user", "content": "hello"},
                    ],
                    stop=["<|im_end|>"],
                    logits_processors=[sentinel_processor],
                )
            ]

        # Cache-path probe (``apply_chat_template`` called a second + third time for
        # probe-divergence) must NOT happen — only the initial prompt render.
        assert tokenizer.apply_chat_template.call_count == 1
        # The uncached fallback must have been invoked.
        # The decode-control kwargs must have been threaded through.
        assert fallback_kwargs, "uncached stream_generate fallback was not invoked"
        assert fallback_kwargs[0].get("stop") == ["<|im_end|>"]
        assert fallback_kwargs[0].get("logits_processors") == [sentinel_processor]
        assert chunks and chunks[0].text == "ok"

    @pytest.mark.anyio
    async def test_stream_chat_takes_cache_path_when_decode_controls_are_no_ops(self):
        """server.py always sets ``top_k=0``, ``min_p=0.0``, ``presence_penalty=0.0``,
        ``repetition_penalty=1.0`` (no-ops) in ``chat_kwargs``.
        The gate must compare against those defaults so the common path still hits the
        cache.
        Only *active* controls should block."""
        from vllm_mlx.engine.simple import SimpleEngine

        def apply_chat_template_side_effect(messages, **kwargs):
            user_content = ""
            for m in reversed(messages):
                role = (
                    m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
                )
                if role == "user":
                    content = (
                        m.get("content")
                        if isinstance(m, dict)
                        else getattr(m, "content", "")
                    )
                    user_content = content or ""
                    break
            return (
                "<|im_start|>system\nYou are helpful.<|im_end|>\n"
                f"<|im_start|>user\n{user_content}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )

        tokenizer = MagicMock()
        tokenizer.apply_chat_template.side_effect = apply_chat_template_side_effect
        tokenizer.bos_token = None
        tokenizer.encode = MagicMock(side_effect=[list(range(50)), list(range(20))])

        model = MagicMock()
        model.tokenizer = tokenizer

        with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False):
            engine = SimpleEngine("test-model")
            engine._model = model
            engine._loaded = True
            # Probe normally runs in start(); short-circuit it here so the
            # gate doesn't trip on the synthetic engine state.
            engine._supports_system_kv_cache = True

            # No fallback needed since cache path should be exercised.
            # Patch _run_blocking_serialized to short-circuit cache execution cleanly
            # without needing a real mlx_lm.
            async def short_circuit(func, *args, on_cancel=None, **kw):
                # Simulate immediate completion with no responses.
                # The producer-task harness will fire _emit_done().
                return None

            engine._run_blocking_serialized = short_circuit  # type: ignore[method-assign]

            _ = [
                c
                async for c in engine.stream_chat(
                    messages=[
                        {"role": "system", "content": "You are helpful."},
                        {"role": "user", "content": "hello"},
                    ],
                    top_k=0,
                    min_p=0.0,
                    presence_penalty=0.0,
                    repetition_penalty=1.0,
                )
            ]

        # Probe-divergence ran ⇒ apply_chat_template called for prompt + 2 probes.
        assert tokenizer.apply_chat_template.call_count == 3

    @pytest.mark.anyio
    async def test_stream_chat_skips_cache_path_when_mtp_active(self):
        """When ``self._mtp`` is configured, the cache branch must be skipped.
        The branch calls ``mlx_lm.stream_generate`` directly with no ``mtp`` /
        ``num_draft_tokens`` kwargs, while ``MLXLanguageModel.stream_generate``
        attaches them from ``self._mtp`` / ``self._mtp_num_draft_tokens``.
        Running the same request through the cache branch would silently drop
        speculative decoding for cache-eligible turns while keeping it on
        uncached turns — different engine semantics for the same request."""
        from vllm_mlx.engine.simple import SimpleEngine

        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = (
            "<|im_start|>system\nYou are helpful.<|im_end|>\n"
            "<|im_start|>user\nhello<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        tokenizer.bos_token = None
        tokenizer.encode = MagicMock(return_value=[1, 2, 3])

        model = MagicMock()
        model.tokenizer = tokenizer

        fallback_kwargs: list[dict] = []

        async def fake_stream_generate(*, prompt, **kw):
            fallback_kwargs.append(kw)
            out = MagicMock(
                text="ok",
                new_text="ok",
                prompt_tokens=3,
                completion_tokens=1,
                finished=True,
                finish_reason="stop",
            )
            yield out

        with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False):
            engine = SimpleEngine("test-model", mtp=True, mtp_num_draft_tokens=4)
            engine._model = model
            engine._loaded = True
            # Isolate the gate to the feature under test.
            engine._supports_system_kv_cache = True
            engine.stream_generate = fake_stream_generate  # type: ignore[method-assign]

            chunks = [
                c
                async for c in engine.stream_chat(
                    messages=[
                        {"role": "system", "content": "You are helpful."},
                        {"role": "user", "content": "hello"},
                    ],
                )
            ]

        # Cache-path probes must NOT run — only the initial prompt render.
        assert tokenizer.apply_chat_template.call_count == 1
        # The uncached wrapper must have been invoked.
        # MTP kwargs are layered on inside ``MLXLanguageModel.stream_generate``,
        # not at this seam, so this test only proves the cache branch was
        # bypassed; the wrapper attaches MTP itself when ``self._mtp`` is set.
        assert fallback_kwargs, "uncached stream_generate fallback was not invoked"
        assert chunks and chunks[0].text == "ok"

    @pytest.mark.anyio
    async def test_stream_chat_skips_cache_path_when_specprefill_loaded(self):
        """A loaded SpecPrefill draft model (``self._draft_model is not None``)
        triggers ``_stream_generate_specprefill`` routing inside the wrapper for
        large prompts.
        The cache branch has no equivalent routing, so it must be skipped
        whenever a draft model is loaded so all requests go through the
        wrapper's SpecPrefill decision."""
        from vllm_mlx.engine.simple import SimpleEngine

        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = (
            "<|im_start|>system\nYou are helpful.<|im_end|>\n"
            "<|im_start|>user\nhello<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        tokenizer.bos_token = None
        tokenizer.encode = MagicMock(return_value=[1, 2, 3])

        model = MagicMock()
        model.tokenizer = tokenizer

        fallback_kwargs: list[dict] = []

        async def fake_stream_generate(*, prompt, **kw):
            fallback_kwargs.append(kw)
            out = MagicMock(
                text="ok",
                new_text="ok",
                prompt_tokens=3,
                completion_tokens=1,
                finished=True,
                finish_reason="stop",
            )
            yield out

        with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False):
            engine = SimpleEngine("test-model")
            engine._model = model
            engine._draft_model = MagicMock(name="specprefill_draft_model")
            engine._loaded = True
            engine._supports_system_kv_cache = True
            engine.stream_generate = fake_stream_generate  # type: ignore[method-assign]

            chunks = [
                c
                async for c in engine.stream_chat(
                    messages=[
                        {"role": "system", "content": "You are helpful."},
                        {"role": "user", "content": "hello"},
                    ],
                )
            ]

        assert tokenizer.apply_chat_template.call_count == 1
        assert fallback_kwargs, "uncached stream_generate fallback was not invoked"
        assert chunks and chunks[0].text == "ok"

    @pytest.mark.anyio
    async def test_stream_chat_skips_cache_path_when_max_kv_size_set(self):
        """Configured ``max_kv_size`` caps the prompt cache.
        The cache branch builds its cache with ``make_prompt_cache(model)``
        without forwarding ``max_kv_size``, so a non-zero engine-level bound
        must force the uncached path."""
        from vllm_mlx.engine.simple import SimpleEngine

        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = (
            "<|im_start|>system\nYou are helpful.<|im_end|>\n"
            "<|im_start|>user\nhello<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        tokenizer.bos_token = None
        tokenizer.encode = MagicMock(return_value=[1, 2, 3])

        model = MagicMock()
        model.tokenizer = tokenizer

        fallback_kwargs: list[dict] = []

        async def fake_stream_generate(*, prompt, **kw):
            fallback_kwargs.append(kw)
            out = MagicMock(
                text="ok",
                new_text="ok",
                prompt_tokens=3,
                completion_tokens=1,
                finished=True,
                finish_reason="stop",
            )
            yield out

        with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False):
            engine = SimpleEngine("test-model", max_kv_size=4096)
            engine._model = model
            engine._loaded = True
            engine._supports_system_kv_cache = True
            engine.stream_generate = fake_stream_generate  # type: ignore[method-assign]

            chunks = [
                c
                async for c in engine.stream_chat(
                    messages=[
                        {"role": "system", "content": "You are helpful."},
                        {"role": "user", "content": "hello"},
                    ],
                )
            ]

        assert tokenizer.apply_chat_template.call_count == 1
        assert fallback_kwargs, "uncached stream_generate fallback was not invoked"
        assert chunks and chunks[0].text == "ok"

    @pytest.mark.anyio
    async def test_stream_chat_skips_cache_path_when_model_has_non_kv_cache(self):
        """Models whose ``make_prompt_cache`` returns ``RotatingKVCache``
        (sliding-window models like gemma3_text, olmo3, recurrent_gemma) cannot
        be safely snapshotted: ``.state`` aliases buffers that
        ``update_and_fetch`` mutates in place, so restoring a captured snapshot
        on the next turn would silently desynchronize from the running cache.
        ``start()`` probes ``make_prompt_cache`` once and sets
        ``_supports_system_kv_cache=False`` for those models; the gate must
        then skip the cache branch."""
        from vllm_mlx.engine.simple import SimpleEngine

        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = (
            "<|im_start|>system\nYou are helpful.<|im_end|>\n"
            "<|im_start|>user\nhello<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        tokenizer.bos_token = None
        tokenizer.encode = MagicMock(return_value=[1, 2, 3])

        model = MagicMock()
        model.tokenizer = tokenizer

        fallback_kwargs: list[dict] = []

        async def fake_stream_generate(*, prompt, **kw):
            fallback_kwargs.append(kw)
            out = MagicMock(
                text="ok",
                new_text="ok",
                prompt_tokens=3,
                completion_tokens=1,
                finished=True,
                finish_reason="stop",
            )
            yield out

        with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False):
            engine = SimpleEngine("test-model")
            engine._model = model
            engine._loaded = True
            # Simulate the probe finding non-KVCache entries (rotating cache).
            engine._supports_system_kv_cache = False
            engine.stream_generate = fake_stream_generate  # type: ignore[method-assign]

            chunks = [
                c
                async for c in engine.stream_chat(
                    messages=[
                        {"role": "system", "content": "You are helpful."},
                        {"role": "user", "content": "hello"},
                    ],
                )
            ]

        # Cache-path probes must NOT run when the model isn't snapshot-safe.
        assert tokenizer.apply_chat_template.call_count == 1
        assert fallback_kwargs, "uncached stream_generate fallback was not invoked"
        assert chunks and chunks[0].text == "ok"

    @pytest.mark.anyio
    async def test_stream_generate_text_skips_cache_path_when_text_model_unsafe(self):
        """Same snapshot-safety contract as the LLM path, but for MLLM text
        routing: ``_stream_generate_text`` must not enter the system-KV cache
        branch when ``start()``'s probe of the derived ``_text_model`` returned
        non-KVCache entries (e.g. a sliding-window text head). The gate must
        fall back to the uncached path — no encode of the system prefix, no
        snapshot store, no LRU mutation."""
        from collections import OrderedDict

        from vllm_mlx.engine.simple import SimpleEngine

        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = (
            "<|im_start|>system\nYou are helpful.<|im_end|>\n"
            "<|im_start|>user\nhello<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        tokenizer.bos_token = None
        tokenizer.encode = MagicMock(return_value=[1, 2, 3])
        tokenizer.decode = MagicMock(return_value="")
        tokenizer.eos_token_id = 99

        text_model = MagicMock()
        text_model.mtp = None

        engine = SimpleEngine("test-model", force_mllm=True)
        engine._loaded = True
        engine._text_model = text_model
        engine._text_tokenizer = tokenizer
        # Probe at start() decided this text model is NOT snapshot-safe.
        engine._supports_system_kv_cache = False
        engine._system_kv_cache = OrderedDict()
        # Seed counters to verify they don't move on the uncached path.
        engine._system_kv_cache_stats = {
            "hits": 0,
            "misses": 0,
            "stores": 0,
            "evictions": 0,
        }

        def fake_stream_generate(*args, **kw):
            # _run_all iterates this synchronously (`for resp in ...`), so the
            # mock must be a sync generator, not an async one.
            yield SimpleNamespace(
                text="ok",
                new_text="ok",
                prompt_tokens=3,
                completion_tokens=1,
                finished=True,
                finish_reason="stop",
                token=99,
            )

        with (
            patch("vllm_mlx.engine.simple._bind_worker_generation_streams"),
            patch(
                "mlx_lm.models.cache.make_prompt_cache",
                return_value=["backbone-cache"],
            ),
            patch("mlx_lm.sample_utils.make_sampler", return_value=MagicMock()),
            patch(
                "mlx_lm.sample_utils.make_logits_processors",
                return_value=[],
            ),
            patch("mlx_lm.stream_generate", side_effect=fake_stream_generate),
        ):
            chunks = [
                c
                async for c in engine._stream_generate_text(
                    messages=[
                        {"role": "system", "content": "You are helpful."},
                        {"role": "user", "content": "hello"},
                    ],
                    max_tokens=4,
                    temperature=0.7,
                    top_p=0.95,
                )
            ]

        assert chunks, "expected uncached fallback to emit at least one chunk"
        # System-prefix tokenization must NOT have been invoked: the only
        # encode() call is the full-prompt one for the uncached path. The
        # cache branch would have issued a second encode() for the system
        # prefix before storing.
        assert tokenizer.encode.call_count <= 1, (
            "snapshot path ran despite _supports_system_kv_cache=False: "
            f"encode called {tokenizer.encode.call_count} times"
        )
        # And nothing should have landed in the LRU or moved counters.
        assert len(engine._system_kv_cache) == 0
        assert engine._system_kv_cache_stats == {
            "hits": 0,
            "misses": 0,
            "stores": 0,
            "evictions": 0,
        }

    @pytest.mark.anyio
    async def test_stream_generate_text_skips_cache_path_under_bounded_kv(self):
        """Bounded-KV snapshot-safety contract for the MLLM text route.

        When ``_max_kv_size > 0`` the runtime cache branch builds the prompt
        cache via ``make_prompt_cache(model, max_kv_size=N)``. For models
        without a custom ``make_cache``, ``mlx_lm.models.cache.make_prompt_cache``
        returns ``RotatingKVCache`` whose ``.state`` aliases buffers that
        ``update_and_fetch`` mutates in place — snapshot capture would corrupt
        the cached prefix on the next decode.

        The startup probe is now wired with the same ``max_kv_size`` arg as
        the runtime constructor, so under bounded KV the probe sees
        ``RotatingKVCache`` and sets ``_supports_system_kv_cache=False``.
        This test locks in the runtime side of that chain: with the flag set
        False (the correct post-probe state for ``_max_kv_size > 0``),
        ``_stream_generate_text`` must fall back to the uncached path — no
        encode of the system prefix, no LRU store, no counter movement.

        Regression for PR #541 review (Thump604, 2026-05-17 16:04 UTC).
        """
        from collections import OrderedDict

        from vllm_mlx.engine.simple import SimpleEngine

        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = (
            "<|im_start|>system\nYou are helpful.<|im_end|>\n"
            "<|im_start|>user\nhello<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        tokenizer.bos_token = None
        tokenizer.encode = MagicMock(return_value=[1, 2, 3])
        tokenizer.decode = MagicMock(return_value="")
        tokenizer.eos_token_id = 99

        text_model = MagicMock()
        text_model.mtp = None

        engine = SimpleEngine("test-model", force_mllm=True)
        engine._loaded = True
        engine._text_model = text_model
        engine._text_tokenizer = tokenizer
        # Bounded-KV serving: the startup probe (called with the same
        # max_kv_size as runtime) would have built RotatingKVCache and
        # flipped this flag to False.
        engine._max_kv_size = 2048
        engine._supports_system_kv_cache = False
        engine._system_kv_cache = OrderedDict()
        engine._system_kv_cache_stats = {
            "hits": 0,
            "misses": 0,
            "stores": 0,
            "evictions": 0,
        }

        def fake_stream_generate(*args, **kw):
            yield SimpleNamespace(
                text="ok",
                new_text="ok",
                prompt_tokens=3,
                completion_tokens=1,
                finished=True,
                finish_reason="stop",
                token=99,
            )

        with (
            patch("vllm_mlx.engine.simple._bind_worker_generation_streams"),
            patch(
                "mlx_lm.models.cache.make_prompt_cache",
                return_value=["backbone-cache"],
            ),
            patch("mlx_lm.sample_utils.make_sampler", return_value=MagicMock()),
            patch(
                "mlx_lm.sample_utils.make_logits_processors",
                return_value=[],
            ),
            patch("mlx_lm.stream_generate", side_effect=fake_stream_generate),
        ):
            chunks = [
                c
                async for c in engine._stream_generate_text(
                    messages=[
                        {"role": "system", "content": "You are helpful."},
                        {"role": "user", "content": "hello"},
                    ],
                    max_tokens=4,
                    temperature=0.7,
                    top_p=0.95,
                )
            ]

        assert chunks, "expected uncached fallback to emit at least one chunk"
        # The cache branch would have issued a second encode() for the system
        # prefix before storing; under bounded KV the gate must skip that.
        assert tokenizer.encode.call_count <= 1, (
            "snapshot path ran under _max_kv_size>0: "
            f"encode called {tokenizer.encode.call_count} times"
        )
        assert len(engine._system_kv_cache) == 0
        assert engine._system_kv_cache_stats == {
            "hits": 0,
            "misses": 0,
            "stores": 0,
            "evictions": 0,
        }

    @pytest.mark.anyio
    async def test_stream_chat_system_cache_copies_arrays_cache_state(self):
        """Hybrid cache snapshots must not alias ``ArraysCache.state`` lists."""
        import mlx.core as mx

        from vllm_mlx.engine.simple import SimpleEngine

        def apply_chat_template_side_effect(messages, **kwargs):
            user_content = ""
            for m in reversed(messages):
                if m.get("role") == "user":
                    user_content = m.get("content") or ""
                    break
            return (
                "<|im_start|>system\nYou are helpful.<|im_end|>\n"
                f"<|im_start|>user\n{user_content}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )

        tokenizer = MagicMock()
        tokenizer.apply_chat_template.side_effect = apply_chat_template_side_effect
        tokenizer.bos_token = None
        tokenizer.encode = MagicMock(side_effect=[list(range(10)), list(range(5))])

        model = MagicMock()
        model.tokenizer = tokenizer

        class MutableListCache:
            def __init__(self):
                self.cache = [None]

            @property
            def state(self):
                return self.cache

            @state.setter
            def state(self, value):
                self.cache = value

            @property
            def nbytes(self):
                return sum(c.nbytes for c in self.cache if c is not None)

        cache_entry = MutableListCache()

        def fake_model(tokens, cache):
            cache[0].cache[0] = mx.array([[1, 2, 3]])
            return MagicMock()

        model.model = fake_model

        def fake_stream_generate(model_arg, tokenizer_arg, prompt, **kwargs):
            # Simulate suffix/decode state advancing after the system-prefix
            # snapshot has been stored. The saved snapshot must not follow this
            # mutable list change.
            prompt_cache = kwargs["prompt_cache"]
            prompt_cache[0].cache[0] = mx.array([[9, 9, 9]])
            yield SimpleNamespace(text="ok", token=7, finish_reason="stop")

        with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False):
            engine = SimpleEngine("test-model")
            engine._model = model
            engine._loaded = True
            engine._supports_system_kv_cache = True

            with (
                patch("mlx_lm.stream_generate", side_effect=fake_stream_generate),
                patch(
                    "mlx_lm.models.cache.make_prompt_cache",
                    return_value=[cache_entry],
                ),
                patch("mlx_lm.sample_utils.make_sampler"),
            ):
                chunks = [
                    c
                    async for c in engine.stream_chat(
                        messages=[
                            {"role": "system", "content": "You are helpful."},
                            {"role": "user", "content": "hello"},
                        ],
                    )
                ]

        assert chunks[-1].text == "ok"
        assert engine._system_kv_cache
        saved_snapshot, saved_token_count = next(iter(engine._system_kv_cache.values()))
        assert saved_token_count == 5
        saved = saved_snapshot[0][0]
        assert saved.tolist() == [[1, 2, 3]]
        assert cache_entry.cache[0].tolist() == [[9, 9, 9]]

    def test_system_cache_probe_allows_arrays_cache_and_rejects_rotating(self):
        """Hybrid ArraysCache is snapshot-safe; rotating cache still is not."""
        from mlx_lm.models.cache import ArraysCache, KVCache, RotatingKVCache

        from vllm_mlx.engine.simple import SimpleEngine

        assert SimpleEngine._cache_class_is_system_snapshot_safe(KVCache())
        assert SimpleEngine._cache_class_is_system_snapshot_safe(ArraysCache(size=2))
        assert not SimpleEngine._cache_class_is_system_snapshot_safe(
            RotatingKVCache(max_size=128)
        )

    @pytest.mark.anyio
    async def test_stream_chat_uses_gate_time_snapshot_under_concurrent_mutation(
        self,
    ):
        """A concurrent MISS that mutates ``self._system_kv_cache`` between
        the cache-hit gate (which runs outside ``_run_blocking_serialized``) and
        the snapshot restore (which runs inside the serialized worker) must not
        corrupt the HIT. The restore must use the snapshot reference captured at
        gate time, not re-read ``self._system_kv_cache`` later — otherwise a
        different system prefix's KV would be loaded under the hash that decided
        HIT.

        Simulates the race by replacing the cache entry for the same hash inside
        the ``_run_blocking_serialized`` hook (executed after the gate but
        before the worker enters the cache branch), then asserts the restore
        loop wrote the gate-time entries, not the post-gate intruder."""
        import hashlib
        from collections import OrderedDict

        from vllm_mlx.engine.simple import SimpleEngine

        # Same template the positive test uses: divergence falls at the user
        # content so the detected system prefix is the leading frame up through
        # ``<|im_start|>user\n``.
        def apply_chat_template_side_effect(messages, **kwargs):
            user_content = ""
            for m in reversed(messages):
                role = (
                    m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
                )
                if role == "user":
                    content = (
                        m.get("content")
                        if isinstance(m, dict)
                        else getattr(m, "content", "")
                    )
                    user_content = content or ""
                    break
            return (
                "<|im_start|>system\nYou are helpful.<|im_end|>\n"
                f"<|im_start|>user\n{user_content}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )

        expected_prefix = (
            "<|im_start|>system\nYou are helpful.<|im_end|>\n" "<|im_start|>user\n"
        )
        expected_hash = hashlib.sha256(expected_prefix.encode()).hexdigest()[:16]

        tokenizer = MagicMock()
        tokenizer.apply_chat_template.side_effect = apply_chat_template_side_effect
        tokenizer.bos_token = None
        # First encode = full prompt tokens; second = system prefix tokens.
        # range(20) is a prefix of range(50), so prefix-match validation passes.
        tokenizer.encode = MagicMock(side_effect=[list(range(50)), list(range(20))])

        model = MagicMock()
        model.tokenizer = tokenizer

        original_snapshot = [("ORIGINAL_K", "ORIGINAL_V")]
        intruder_snapshot = [("INTRUDER_K", "INTRUDER_V")]

        captured_states: list = []

        class MockCacheEntry:
            def __init__(self) -> None:
                self._state = None

            @property
            def state(self):
                return self._state

            @state.setter
            def state(self, value) -> None:
                captured_states.append(value)
                self._state = value

        with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False):
            engine = SimpleEngine("test-model")
            engine._model = model
            engine._loaded = True
            engine._supports_system_kv_cache = True
            # Pre-seed HIT state matching the divergence-detected prefix.
            engine._system_kv_cache = OrderedDict(
                [(expected_hash, (original_snapshot, 20))]
            )

            async def serialized_with_race(func, *args, on_cancel=None, **kw):
                # Simulate a concurrent MISS replacing the cache entry for the
                # same hash AFTER the gate's HIT decision but BEFORE the worker
                # restores. The gate captured a reference to the original
                # tuple; replacement here must not affect that capture.
                engine._system_kv_cache[expected_hash] = (intruder_snapshot, 20)
                await asyncio.to_thread(func)
                return None

            engine._run_blocking_serialized = (
                serialized_with_race  # type: ignore[method-assign]
            )

            with (
                patch("mlx_lm.stream_generate", return_value=iter([])),
                patch(
                    "mlx_lm.models.cache.make_prompt_cache",
                    return_value=[MockCacheEntry()],
                ),
                patch("mlx_lm.sample_utils.make_sampler"),
            ):
                _ = [
                    c
                    async for c in engine.stream_chat(
                        messages=[
                            {"role": "system", "content": "You are helpful."},
                            {"role": "user", "content": "hello"},
                        ],
                    )
                ]

        # Restore wrote the gate-time snapshot exactly once.
        # If the worker had re-read ``self._system_kv_cache`` we would see
        # ``("INTRUDER_K", "INTRUDER_V")`` instead — that's the TOCTOU bug.
        assert captured_states == [("ORIGINAL_K", "ORIGINAL_V")], (
            "Snapshot restore did not use the gate-time reference; "
            f"captured={captured_states}"
        )

    @pytest.mark.anyio
    async def test_lock_serializes_stream_generate(self, mock_model):
        """Test that stream_generate uses the same lock as other methods."""
        from vllm_mlx.engine.simple import SimpleEngine

        def stream_generate_side_effect(**kwargs):
            # Yield a few chunks
            for i in range(3):
                chunk = MagicMock()
                chunk.text = f"chunk{i}"
                chunk.prompt_tokens = 5
                chunk.finished = i == 2
                chunk.finish_reason = "stop" if i == 2 else None
                yield chunk

        mock_model.stream_generate = MagicMock(side_effect=stream_generate_side_effect)

        with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False):
            engine = SimpleEngine("test-model")
            engine._model = mock_model
            engine._loaded = True
            engine._generation_lock_admission = "wait"

            # Test that stream_generate acquires the lock
            # by checking if it blocks when lock is already held
            lock_acquired = asyncio.Event()
            stream_started = asyncio.Event()

            async def hold_lock():
                async with engine._generation_lock:
                    lock_acquired.set()
                    # Wait until stream tries to start
                    await asyncio.sleep(0.1)

            async def try_stream():
                # Wait for lock to be held
                await lock_acquired.wait()
                stream_started.set()
                # This should block until hold_lock releases
                result = []
                async for chunk in engine.stream_generate(prompt="test", max_tokens=10):
                    result.append(chunk)
                return result

            # Start both tasks
            hold_task = asyncio.create_task(hold_lock())
            stream_task = asyncio.create_task(try_stream())

            # Wait a bit for stream to try to acquire lock
            await asyncio.sleep(0.05)

            # Stream should have started but be blocked on the lock
            assert stream_started.is_set(), "Stream should have attempted to start"

            # Stream task should not be done yet (blocked on lock)
            assert not stream_task.done(), "Stream should be blocked waiting for lock"

            # Let hold_lock finish
            await hold_task

            # Now stream should complete
            result = await stream_task
            assert len(result) == 3, f"Expected 3 chunks, got {len(result)}"

    @pytest.mark.anyio
    async def test_engine_initialization_creates_lock(self):
        """Test that SimpleEngine creates a lock on initialization."""
        from vllm_mlx.engine.simple import SimpleEngine

        with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False):
            engine = SimpleEngine("test-model")

            assert hasattr(engine, "_generation_lock")
            assert isinstance(engine._generation_lock, asyncio.Lock)

    @pytest.mark.anyio
    async def test_run_blocking_serialized_rebinds_worker_generation_streams(self):
        """Worker-thread MLX generation should get fresh thread-local streams."""
        import importlib

        from vllm_mlx.engine.simple import SimpleEngine

        mlx_lm_generate = importlib.import_module("mlx_lm.generate")
        sentinel_stream = object()

        with (
            patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False),
            patch("vllm_mlx.mlx_streams.mx.default_device", return_value="gpu"),
            patch(
                "vllm_mlx.mlx_streams.mx.new_stream",
                return_value=sentinel_stream,
            ),
            patch("vllm_mlx.mlx_streams.mx.set_default_stream"),
        ):
            engine = SimpleEngine("test-model")
            observed = await engine._run_blocking_serialized(
                lambda: mlx_lm_generate.generation_stream
            )

        assert observed is sentinel_stream

    @pytest.mark.anyio
    async def test_llm_stream_generate_stays_on_model_load_thread(self):
        """SimpleEngine must load and stream on the same thread for MLX streams."""
        from vllm_mlx.engine.simple import SimpleEngine

        class FakeLLMModel:
            def __init__(self, *_args, **_kwargs):
                self._load_thread = None
                self.tokenizer = MagicMock()
                self.tokenizer.encode.return_value = [1, 2, 3]

            def load(self):
                self._load_thread = threading.get_ident()

            def stream_generate(self, **_kwargs):
                if threading.get_ident() != self._load_thread:
                    raise RuntimeError("There is no Stream(gpu, 0) in current thread.")
                yield SimpleNamespace(
                    text="ok",
                    prompt_tokens=3,
                    finished=True,
                    finish_reason="stop",
                )

        with (
            patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False),
            patch("vllm_mlx.models.llm.MLXLanguageModel", FakeLLMModel),
        ):
            engine = SimpleEngine("test-model")

            outputs = [
                chunk
                async for chunk in engine.stream_generate(
                    prompt="hello",
                    max_tokens=1,
                    temperature=0.0,
                    top_p=1.0,
                )
            ]

        assert outputs
        assert outputs[-1].new_text == "ok"
        assert outputs[-1].finished is True

    @pytest.mark.anyio
    async def test_start_keeps_text_routing_for_mllm_without_mtp(self):
        """MLLM text-only routing must stay available when MTP is disabled."""
        from vllm_mlx.engine.simple import SimpleEngine

        text_model = MagicMock()
        text_model.mtp = None
        tokenizer = MagicMock()
        tokenizer.convert_tokens_to_ids.return_value = 42

        mock_mllm = MagicMock()
        mock_mllm.model = MagicMock()
        mock_mllm.get_tokenizer.return_value = tokenizer

        with (
            patch(
                "vllm_mlx.models.mllm.MLXMultimodalLM",
                return_value=mock_mllm,
            ),
            patch(
                "vllm_mlx.text_model_from_vlm.build_text_model",
                return_value=text_model,
            ),
        ):
            engine = SimpleEngine("qwen3.6-27b", force_mllm=True, mtp=False)
            await engine.start()

        assert engine._text_model is text_model
        assert engine._text_tokenizer is tokenizer

    @pytest.mark.anyio
    async def test_mllm_nonstream_text_only_routes_without_mtp(self):
        """Non-stream text-only MLLM chat must aggregate the TextModel route."""
        from vllm_mlx.engine.simple import SimpleEngine

        async def fake_stream_chat(*args, **kwargs):
            yield MagicMock(
                text="Hello",
                tokens=[1],
                prompt_tokens=5,
                completion_tokens=1,
                finish_reason="stop",
                finished=True,
            )

        engine = SimpleEngine("test-model", force_mllm=True, mtp=False)
        engine._loaded = True
        engine._text_model = MagicMock()
        engine._model = MagicMock()
        engine.stream_chat = fake_stream_chat  # type: ignore[method-assign]

        output = await engine.chat(
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=16,
        )

        assert output.text == "Hello"
        assert output.tokens == [1]
        assert output.prompt_tokens == 5
        assert output.completion_tokens == 1
        assert output.finish_reason == "stop"
        engine._model.chat.assert_not_called()

    @pytest.mark.anyio
    async def test_mllm_nonstream_text_only_without_text_model_uses_stream_path(self):
        """When TextModel is unavailable, text-only MLLM non-stream chat should
        aggregate stream_chat to avoid mlx_vlm chat thread-stream mismatches.
        """
        from types import SimpleNamespace

        from vllm_mlx.engine.simple import SimpleEngine

        class FakeMllmModel:
            def chat(self, **kwargs):
                raise RuntimeError("There is no Stream(gpu, 3) in current thread.")

            def stream_chat(self, **kwargs):
                yield SimpleNamespace(text="one, two, three", finish_reason="stop")

        engine = SimpleEngine("test-model", force_mllm=True, mtp=False)
        engine._loaded = True
        engine._text_model = None
        engine._model = FakeMllmModel()

        output = await engine.chat(
            messages=[{"role": "user", "content": "Count: one, two, three"}],
            max_tokens=16,
        )

        assert output.text == "one, two, three"
        assert output.finish_reason == "stop"

    @pytest.mark.anyio
    async def test_mllm_nonstream_text_only_without_text_model_keeps_stream_thread_owner(
        self,
    ):
        """MLLM text-only non-stream path must keep stream_chat on model thread.

        Regression: aggregate_stream_chat -> stream_chat used _run_blocking_serialized,
        which moved mlx_vlm stream generation to a worker thread and could raise
        "There is no Stream(gpu, N) in current thread".
        """
        from types import SimpleNamespace

        from vllm_mlx.engine.simple import SimpleEngine

        class FakeMllmModel:
            def __init__(self):
                self._owner_thread = threading.get_ident()

            def stream_chat(self, **kwargs):
                if threading.get_ident() != self._owner_thread:
                    raise RuntimeError("There is no Stream(gpu, 3) in current thread.")
                yield SimpleNamespace(
                    text="one, two, three",
                    finish_reason="stop",
                    prompt_tokens=3,
                )

        engine = SimpleEngine("test-model", force_mllm=True, mtp=False)
        engine._loaded = True
        engine._text_model = None
        engine._model = FakeMllmModel()

        output = await engine.chat(
            messages=[{"role": "user", "content": "Count: one, two, three"}],
            max_tokens=16,
        )

        assert output.text == "one, two, three"
        assert output.finish_reason == "stop"

    @pytest.mark.anyio
    async def test_llm_nonstream_with_logits_processors_uses_stream_path(self):
        """Constrained non-stream chat must not call the blocking chat API.

        ``response_format`` is implemented by request-local logits processors.
        If a non-stream request goes through the blocking model.chat() path,
        the server cannot observe token progress or cancel at token boundaries
        when a client/proxy disconnects.  Aggregating stream_chat keeps the
        constrained and unconstrained chat paths on the same cancellable stream
        implementation.
        """
        from types import SimpleNamespace

        from vllm_mlx.engine.simple import SimpleEngine

        captured_stream_kwargs = {}

        class FakeTokenizer:
            bos_token = None

            def apply_chat_template(self, messages, **kwargs):
                return "<|im_start|>user\nhello"

            def encode(self, text, **kwargs):
                return [1, 2, 3]

        class FakeModel:
            tokenizer = FakeTokenizer()

            def chat(self, **kwargs):
                raise AssertionError("blocking chat path should not be used")

            def stream_generate(self, **kwargs):
                captured_stream_kwargs.update(kwargs)
                yield SimpleNamespace(
                    text="{}",
                    finish_reason="stop",
                    finished=True,
                    prompt_tokens=3,
                )

        engine = SimpleEngine("test-model", force_mllm=False, mtp=False)
        engine._loaded = True
        engine._model = FakeModel()
        sentinel_processor = object()

        output = await engine.chat(
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=16,
            logits_processors=[sentinel_processor],
        )

        assert output.text == "{}"
        assert output.finish_reason == "stop"
        assert captured_stream_kwargs["logits_processors"] == [sentinel_processor]

    @pytest.mark.anyio
    async def test_requests_complete_in_order(self, mock_model):
        """Test that concurrent requests complete (may be in any order due to lock)."""
        from vllm_mlx.engine.simple import SimpleEngine

        with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False):
            engine = SimpleEngine("test-model")
            engine._model = mock_model
            engine._loaded = True

            # Launch multiple concurrent generate calls
            results = await asyncio.gather(
                *[
                    engine.generate(prompt=f"test prompt {i}", max_tokens=10)
                    for i in range(3)
                ]
            )

            # All requests should complete
            assert len(results) == 3
            for result in results:
                assert result.text == "test response"

    @pytest.mark.anyio
    async def test_generate_accumulates_over_stream_generate(self):
        """generate() should iterate stream_generate() and return the last
        yielded GenerationOutput, forwarding per-request kwargs (including
        SpecPrefill overrides) through so they reach _stream_generate_specprefill.
        """
        from vllm_mlx.engine.base import GenerationOutput
        from vllm_mlx.engine.simple import SimpleEngine

        captured_kwargs = {}

        async def fake_stream_generate(**kwargs):
            captured_kwargs.update(kwargs)
            # First chunk: mid-generation
            yield GenerationOutput(
                text="partial",
                new_text="partial",
                tokens=[1, 2],
                prompt_tokens=11,
                completion_tokens=2,
                finished=False,
                finish_reason=None,
            )
            # Final chunk: finished
            yield GenerationOutput(
                text="partial final",
                new_text=" final",
                tokens=[1, 2, 3],
                prompt_tokens=11,
                completion_tokens=3,
                finished=True,
                finish_reason="stop",
            )

        with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False):
            engine = SimpleEngine("test-model")
            engine._loaded = True
            engine.stream_generate = fake_stream_generate  # type: ignore[method-assign]

            output = await engine.generate(
                prompt="say hi",
                max_tokens=16,
                temperature=0.6,
                top_p=0.95,
                specprefill=True,
                specprefill_keep_pct=0.2,
            )

        # Accumulator returns the last GenerationOutput's fields
        assert output.text == "partial final"
        assert output.tokens == [1, 2, 3]
        assert output.prompt_tokens == 11
        assert output.completion_tokens == 3
        assert output.finish_reason == "stop"
        assert output.finished is True

        # Per-request SpecPrefill overrides reach stream_generate
        assert captured_kwargs.get("prompt") == "say hi"
        assert captured_kwargs.get("max_tokens") == 16
        assert captured_kwargs.get("specprefill") is True
        assert captured_kwargs.get("specprefill_keep_pct") == 0.2

    @pytest.mark.anyio
    async def test_generate_empty_stream_returns_safe_default(self):
        """If stream_generate yields nothing, generate() returns an empty
        stop-reason GenerationOutput rather than raising.
        """
        from vllm_mlx.engine.simple import SimpleEngine

        async def empty_stream_generate(**kwargs):
            return
            yield  # unreachable; makes this a generator

        with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False):
            engine = SimpleEngine("test-model")
            engine._loaded = True
            engine.stream_generate = empty_stream_generate  # type: ignore[method-assign]

            output = await engine.generate(prompt="anything", max_tokens=5)

        assert output.text == ""
        assert output.finish_reason == "stop"

    def test_seed_logits_processors_prepends_prompt_tokens(self):
        """Continuation decode processors must see the original prompt prefix."""
        from vllm_mlx.engine.simple import _seed_logits_processors

        seen = {}

        def processor(tokens, logits):
            seen["tokens"] = tokens.tolist()
            return logits

        seeded = _seed_logits_processors(
            mx.array([10, 11], dtype=mx.uint32), [processor]
        )

        logits = mx.zeros((1, 8), dtype=mx.float32)
        seeded[0](mx.array([12, 13], dtype=mx.uint32), logits)

        assert seen["tokens"] == [10, 11, 12, 13]

    @pytest.mark.anyio
    async def test_specprefill_success_preserves_mtp_path(self):
        """Successful sparse prefill should continue through the normal MTP path."""
        from types import SimpleNamespace

        from vllm_mlx.engine.simple import SimpleEngine

        captured = {}

        def fake_make_sampler(**kwargs):
            captured["sampler_kwargs"] = kwargs

            def _sample(_logprobs):
                return mx.array([17], dtype=mx.uint32)

            return _sample

        def fake_stream_generate(model, tokenizer, prompt, **kwargs):
            captured["prompt"] = prompt.tolist()
            captured["kwargs"] = kwargs
            yield SimpleNamespace(text="B", finish_reason="stop")

        def fake_select_chunks(_importance, **kwargs):
            captured["select_chunks_kwargs"] = kwargs
            return mx.array([0, 1, 2], dtype=mx.int32)

        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "<|im_start|>user\nhello"
        tokenizer.bos_token = None
        tokenizer.eos_token_id = 99
        tokenizer.encode.return_value = [5, 6, 7]
        tokenizer.decode.side_effect = lambda ids: "".join(
            {17: "A", 99: ""}.get(tok, f"<{tok}>") for tok in ids
        )

        text_model = MagicMock()
        text_model.mtp = object()
        text_model.make_mtp_cache.return_value = ["mtp-cache"]

        engine = SimpleEngine(
            "test-model",
            force_mllm=True,
            mtp=True,
            mtp_num_draft_tokens=4,
            specprefill_enabled=True,
            specprefill_threshold=1,
        )
        engine._loaded = True
        engine._text_model = text_model
        engine._text_tokenizer = tokenizer
        engine._draft_model = object()

        with (
            patch("vllm_mlx.engine.simple._bind_worker_generation_streams"),
            patch(
                "mlx_lm.models.cache.make_prompt_cache",
                return_value=["backbone-cache"],
            ),
            patch("mlx_lm.sample_utils.make_sampler", side_effect=fake_make_sampler),
            patch(
                "mlx_lm.sample_utils.make_logits_processors",
                return_value=[],
            ),
            patch("mlx_lm.stream_generate", side_effect=fake_stream_generate),
            patch(
                "vllm_mlx.specprefill.score_tokens",
                return_value=mx.array([1.0, 0.9, 0.8], dtype=mx.float32),
            ),
            patch(
                "vllm_mlx.specprefill.select_chunks",
                side_effect=fake_select_chunks,
            ),
            patch(
                "vllm_mlx.specprefill.sparse_prefill",
                return_value=mx.zeros((1, 3, 32), dtype=mx.float32),
            ),
            patch("vllm_mlx.specprefill.cleanup_rope"),
        ):
            outputs = [
                chunk
                async for chunk in engine._stream_generate_text(
                    messages=[{"role": "user", "content": "hello"}],
                    max_tokens=4,
                    temperature=0.6,
                    top_p=0.95,
                    specprefill_backbone_pct=0.25,
                )
            ]

        assert [chunk.new_text for chunk in outputs] == ["A", "B"]
        assert captured["sampler_kwargs"] == {
            "temp": 0.6,
            "top_p": 0.95,
            "top_k": 0,
            "min_p": 0.0,
        }
        assert captured["prompt"] == [17]
        assert captured["kwargs"]["mtp"] is True
        assert captured["kwargs"]["prompt_cache"] == ["backbone-cache", "mtp-cache"]
        assert captured["kwargs"]["max_tokens"] == 3
        assert captured["kwargs"]["logits_processors"] is None
        assert captured["select_chunks_kwargs"]["backbone_pct"] == 0.25

    @pytest.mark.anyio
    async def test_stream_generate_text_forwards_logits_processors_and_sampler_args(
        self,
    ):
        """Text routing must preserve request-local decoding controls."""
        from types import SimpleNamespace

        from vllm_mlx.engine.simple import SimpleEngine

        captured_kwargs = {}
        sampler_calls = []
        penalty_calls = []
        user_processor = MagicMock()
        penalty_processor = MagicMock()

        def fake_stream_generate(model, tokenizer, prompt, **kwargs):
            captured_kwargs.update(kwargs)
            yield SimpleNamespace(text="Hello", finish_reason="stop")

        def fake_make_sampler(**kwargs):
            sampler_calls.append(kwargs)
            return MagicMock()

        def fake_make_logits_processors(**kwargs):
            penalty_calls.append(kwargs)
            return [penalty_processor]

        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "<|im_start|>user\nhello"
        tokenizer.bos_token = None
        tokenizer.eos_token_id = 42

        engine = SimpleEngine("test-model", force_mllm=True, mtp=False)
        engine._loaded = True
        engine._text_model = MagicMock()
        engine._text_model.mtp = None
        engine._text_tokenizer = tokenizer

        with (
            patch("mlx_lm.stream_generate", side_effect=fake_stream_generate),
            patch("mlx_lm.sample_utils.make_sampler", side_effect=fake_make_sampler),
            patch(
                "mlx_lm.sample_utils.make_logits_processors",
                side_effect=fake_make_logits_processors,
            ),
        ):
            outputs = [
                chunk
                async for chunk in engine._stream_generate_text(
                    messages=[{"role": "user", "content": "hello"}],
                    max_tokens=16,
                    temperature=0.3,
                    top_p=0.8,
                    top_k=40,
                    min_p=0.1,
                    presence_penalty=1.5,
                    repetition_penalty=1.2,
                    logits_processors=[user_processor],
                )
            ]

        assert outputs[-1].text == "Hello"
        assert sampler_calls == [{"temp": 0.3, "top_p": 0.8, "top_k": 40, "min_p": 0.1}]
        assert penalty_calls == [{"repetition_penalty": 1.2, "presence_penalty": 1.5}]
        assert captured_kwargs["logits_processors"] == [
            user_processor,
            penalty_processor,
        ]

    @pytest.mark.anyio
    async def test_stream_generate_text_skips_system_cache_when_text_model_not_safe(
        self,
    ):
        """MLLM TextModel routing must not snapshot hybrid ArraysCache state."""
        from types import SimpleNamespace

        from vllm_mlx.engine.simple import SimpleEngine

        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = (
            "<|im_start|>system\nYou are helpful.<|im_end|>\n"
            "<|im_start|>user\nhello<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        tokenizer.bos_token = None
        tokenizer.eos_token_id = 42
        tokenizer.encode = MagicMock(
            side_effect=[
                list(range(10)),  # full prompt
                list(range(5)),  # system prefix
            ]
        )

        captured_prompts = []

        def fake_stream_generate(model, tokenizer, prompt, **kwargs):
            captured_prompts.append(prompt)
            yield SimpleNamespace(text="Hello", finish_reason="stop")

        def fail_if_manual_cache_path_runs(*args, **kwargs):
            raise AssertionError("manual system-cache path must be skipped")

        engine = SimpleEngine("test-model", force_mllm=True, mtp=False)
        engine._loaded = True
        engine._text_model = MagicMock()
        engine._text_model.mtp = None
        engine._text_tokenizer = tokenizer
        engine._supports_system_kv_cache = False

        with (
            patch("mlx_lm.stream_generate", side_effect=fake_stream_generate),
            patch(
                "mlx_lm.models.cache.make_prompt_cache",
                side_effect=fail_if_manual_cache_path_runs,
            ),
        ):
            outputs = [
                chunk
                async for chunk in engine._stream_generate_text(
                    messages=[
                        {"role": "system", "content": "You are helpful."},
                        {"role": "user", "content": "hello"},
                    ],
                    max_tokens=16,
                    temperature=0.3,
                    top_p=0.8,
                )
            ]

        assert outputs[-1].text == "Hello"
        assert captured_prompts == [tokenizer.apply_chat_template.return_value]

    @pytest.mark.anyio
    async def test_stream_generate_text_disables_mtp_when_logits_processors_active(
        self,
    ):
        """Custom logits processors must fail closed to non-MTP decoding."""
        from types import SimpleNamespace

        from vllm_mlx.engine.simple import SimpleEngine

        captured_kwargs = {}
        user_processor = MagicMock()

        def fake_stream_generate(model, tokenizer, prompt, **kwargs):
            captured_kwargs.update(kwargs)
            yield SimpleNamespace(text="Hello", finish_reason="stop")

        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "<|im_start|>user\nhello"
        tokenizer.bos_token = None
        tokenizer.eos_token_id = 42

        engine = SimpleEngine("test-model", force_mllm=True, mtp=True)
        engine._loaded = True
        engine._text_model = MagicMock()
        engine._text_model.mtp = MagicMock()
        engine._text_tokenizer = tokenizer

        with patch("mlx_lm.stream_generate", side_effect=fake_stream_generate):
            outputs = [
                chunk
                async for chunk in engine._stream_generate_text(
                    messages=[{"role": "user", "content": "hello"}],
                    max_tokens=16,
                    temperature=0.7,
                    top_p=0.9,
                    logits_processors=[user_processor],
                )
            ]

        assert outputs[-1].text == "Hello"
        assert "mtp" not in captured_kwargs
        assert captured_kwargs["logits_processors"][0] is user_processor

    @pytest.mark.anyio
    async def test_stream_generate_text_disables_mtp_for_thinking_processor(
        self,
    ):
        """Thinking-budget processors must fail closed to non-MTP decoding."""
        from types import SimpleNamespace

        from vllm_mlx.engine.simple import SimpleEngine

        captured_kwargs = {}

        def fake_stream_generate(model, tokenizer, prompt, **kwargs):
            captured_kwargs.update(kwargs)
            yield SimpleNamespace(text="Hello", finish_reason="stop")

        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "<|im_start|>user\nhello"
        tokenizer.bos_token = None
        tokenizer.eos_token_id = 42

        engine = SimpleEngine("test-model", force_mllm=True, mtp=True)
        engine._loaded = True
        engine._text_model = MagicMock()
        engine._text_model.mtp = MagicMock()
        engine._text_tokenizer = tokenizer

        thinking_proc = MagicMock()

        with patch("mlx_lm.stream_generate", side_effect=fake_stream_generate):
            outputs = [
                chunk
                async for chunk in engine._stream_generate_text(
                    messages=[{"role": "user", "content": "hello"}],
                    max_tokens=16,
                    temperature=0.7,
                    top_p=0.9,
                    logits_processors=[thinking_proc],
                )
            ]

        assert outputs[-1].text == "Hello"
        assert "mtp" not in captured_kwargs
        assert captured_kwargs["logits_processors"][0] is thinking_proc

    @pytest.mark.anyio
    async def test_stream_generate_text_passes_num_draft_tokens(self):
        """Text routing should forward configured MTP draft depth."""
        from types import SimpleNamespace

        from vllm_mlx.engine.simple import SimpleEngine

        captured_kwargs = {}

        def fake_stream_generate(model, tokenizer, prompt, **kwargs):
            captured_kwargs.update(kwargs)
            yield SimpleNamespace(text="Hello", finish_reason="stop")

        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "<|im_start|>user\nhello"
        tokenizer.bos_token = None
        tokenizer.eos_token_id = 42

        engine = SimpleEngine(
            "test-model",
            force_mllm=True,
            mtp=True,
            mtp_num_draft_tokens=4,
        )
        engine._loaded = True
        engine._text_model = MagicMock()
        engine._text_model.mtp = MagicMock()
        engine._text_tokenizer = tokenizer

        with patch("mlx_lm.stream_generate", side_effect=fake_stream_generate):
            outputs = [
                chunk
                async for chunk in engine._stream_generate_text(
                    messages=[{"role": "user", "content": "hello"}],
                    max_tokens=16,
                    temperature=0.7,
                    top_p=0.9,
                )
            ]

        assert outputs[-1].text == "Hello"
        assert "mtp" not in captured_kwargs
        assert captured_kwargs["num_draft_tokens"] == 4

    @pytest.mark.anyio
    async def test_stream_generate_text_reenables_mtp_after_retired_processor_when_enabled(
        self,
    ):
        """Retired thinking processor handoff is an explicit opt-in path."""
        from types import SimpleNamespace

        from vllm_mlx.engine.simple import SimpleEngine

        calls = []

        class RetiringProcessor:
            def __init__(self):
                self.is_retired = False

            def __call__(self, tokens, logits):
                return logits

        processor = RetiringProcessor()

        def fake_stream_generate(model, tokenizer, prompt, **kwargs):
            calls.append({"prompt": prompt, **kwargs})
            if len(calls) == 1:
                processor.is_retired = True
                yield SimpleNamespace(token=11, text="Hello", finish_reason=None)
            else:
                yield SimpleNamespace(token=12, text=" world", finish_reason="stop")

        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "<|im_start|>user\nhello"
        tokenizer.bos_token = None
        tokenizer.eos_token_id = 42
        tokenizer.encode.return_value = [11]

        engine = SimpleEngine(
            "test-model",
            force_mllm=True,
            mtp=True,
            mtp_num_draft_tokens=4,
        )
        engine._loaded = True
        engine._text_model = MagicMock()
        engine._text_model.mtp = MagicMock()
        engine._text_model.make_mtp_cache.return_value = []
        engine._text_tokenizer = tokenizer

        with (
            patch.dict(
                "os.environ",
                {"VLLM_MLX_ENABLE_THINKING_RETIREMENT_RESUME": "1"},
            ),
            patch("mlx_lm.stream_generate", side_effect=fake_stream_generate),
            patch("mlx_lm.models.cache.make_prompt_cache", return_value=[]),
            patch("mlx_lm.models.cache.trim_prompt_cache"),
        ):
            outputs = [
                chunk
                async for chunk in engine._stream_generate_text(
                    messages=[{"role": "user", "content": "hello"}],
                    max_tokens=16,
                    temperature=0.7,
                    top_p=0.9,
                    logits_processors=[processor],
                )
            ]

        assert outputs[-1].text == "Hello world"
        assert len(calls) == 2
        assert "mtp" not in calls[0]
        assert calls[0]["logits_processors"][0] is processor
        assert "mtp" not in calls[1]
        assert calls[1]["num_draft_tokens"] == 4
        assert "prompt_cache" in calls[1]
        assert "logits_processors" not in calls[1]

    @pytest.mark.anyio
    async def test_stream_generate_text_specprefill_reenables_mtp_after_retirement(
        self,
    ):
        """SpecPrefill retirement-to-MTP continuation is explicit opt-in."""
        from types import SimpleNamespace

        from vllm_mlx.engine.simple import SimpleEngine

        calls = []

        class RetiringProcessor:
            def __init__(self):
                self.is_retired = False

            def __call__(self, tokens, logits):
                return logits

        processor = RetiringProcessor()

        def fake_stream_generate(model, tokenizer, prompt, **kwargs):
            calls.append({"prompt": prompt, **kwargs})
            yield SimpleNamespace(token=12, text=" world", finish_reason="stop")

        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "<|im_start|>user\nhello"
        tokenizer.bos_token = None
        tokenizer.eos_token_id = 42
        tokenizer.encode.return_value = [1, 2, 3, 4]
        tokenizer.decode.side_effect = lambda toks: "Hello" if toks == [11] else ""

        engine = SimpleEngine(
            "test-model",
            force_mllm=True,
            mtp=True,
            mtp_num_draft_tokens=4,
            specprefill_enabled=True,
        )
        engine._loaded = True
        engine._draft_model = MagicMock()
        engine._text_model = MagicMock()
        engine._text_model.mtp = MagicMock()
        engine._text_model.make_mtp_cache.return_value = []
        engine._text_tokenizer = tokenizer

        def fake_sample(tokens, logits, sampler, logits_processors):
            processor.is_retired = True
            return mx.array(11, dtype=mx.uint32), logits

        with (
            patch.dict(
                "os.environ",
                {"VLLM_MLX_ENABLE_THINKING_RETIREMENT_RESUME": "1"},
            ),
            patch("mlx_lm.stream_generate", side_effect=fake_stream_generate),
            patch("mlx_lm.models.cache.make_prompt_cache", return_value=[]),
            patch(
                "vllm_mlx.specprefill.score_tokens", return_value=mx.array([0.1, 0.2])
            ),
            patch("vllm_mlx.specprefill.select_chunks", return_value=mx.array([0, 1])),
            patch(
                "vllm_mlx.specprefill.sparse_prefill",
                return_value=mx.zeros((1, 1, 32)),
            ),
            patch("vllm_mlx.specprefill.cleanup_rope"),
            patch(
                "vllm_mlx.engine.simple._sample_with_processors",
                side_effect=fake_sample,
            ),
        ):
            outputs = [
                chunk
                async for chunk in engine._stream_generate_text(
                    messages=[{"role": "user", "content": "hello"}],
                    max_tokens=16,
                    temperature=0.7,
                    top_p=0.9,
                    specprefill=True,
                    logits_processors=[processor],
                )
            ]

        assert outputs[-1].text == "Hello world"
        assert len(calls) == 1
        assert "mtp" not in calls[0]
        assert calls[0]["num_draft_tokens"] == 4
        assert "prompt_cache" in calls[0]
        assert "logits_processors" not in calls[0]

    @pytest.mark.anyio
    async def test_cancellation_does_not_release_lock_before_worker_finishes(
        self, mock_llm_model
    ):
        """A cancelled blocking chat call must not overlap the next worker."""
        from threading import Event, Lock

        from vllm_mlx.engine.simple import SimpleEngine

        first_started = Event()
        release_workers = Event()
        call_count = 0
        call_lock = Lock()

        def chat_side_effect(**kwargs):
            nonlocal call_count
            with call_lock:
                call_count += 1
                current_call = call_count
                mock_llm_model._concurrent_count += 1
                mock_llm_model._max_concurrent = max(
                    mock_llm_model._max_concurrent,
                    mock_llm_model._concurrent_count,
                )
                if current_call == 1:
                    first_started.set()

            try:
                release_workers.wait(timeout=1.0)
                result = MagicMock()
                result.text = f"response-{current_call}"
                result.tokens = [1, 2, 3]
                result.finish_reason = "stop"
                return result
            finally:
                with call_lock:
                    mock_llm_model._concurrent_count -= 1

        mock_llm_model.chat = MagicMock(side_effect=chat_side_effect)

        with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False):
            engine = SimpleEngine("test-model")
            engine._model = mock_llm_model
            engine._loaded = True
            engine._generation_lock_admission = "wait"

            task1 = asyncio.create_task(
                engine.chat(
                    messages=[{"role": "user", "content": "first"}], max_tokens=8
                )
            )
            await asyncio.to_thread(first_started.wait, 1.0)

            task1.cancel()
            task2 = asyncio.create_task(
                engine.chat(
                    messages=[{"role": "user", "content": "second"}], max_tokens=8
                )
            )

            await asyncio.sleep(0.05)
            release_workers.set()

            with pytest.raises(asyncio.CancelledError):
                await task1
            result2 = await task2

            assert result2.text == "response-2"
            assert mock_llm_model._max_concurrent == 1

    @pytest.mark.anyio
    async def test_specprefill_path_does_not_prelock_serialized_runner(self):
        """SpecPrefill should let _run_blocking_serialized own the lock."""
        from vllm_mlx.engine.simple import SimpleEngine

        async def fake_serialized(func, *args, **kwargs):
            assert not engine._generation_lock.locked()
            return []

        with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False):
            engine = SimpleEngine("test-model")
            engine._loaded = True
            engine._model = MagicMock()
            engine._model.model = MagicMock()
            engine._model.tokenizer = MagicMock()
            engine._draft_model = MagicMock()
            engine._run_blocking_serialized = fake_serialized  # type: ignore[method-assign]

            outputs = []
            async for chunk in engine._stream_generate_specprefill(
                prompt="hello",
                tokens=[1, 2, 3, 4],
                max_tokens=4,
                temperature=0.7,
                top_p=0.9,
            ):
                outputs.append(chunk)

            assert len(outputs) == 1
            assert outputs[0].finished
            assert outputs[0].completion_tokens == 0

    @pytest.mark.anyio
    async def test_text_mtp_path_does_not_prelock_serialized_runner(self):
        """Text-only MTP path should let _run_blocking_serialized own the lock."""
        from vllm_mlx.engine.simple import SimpleEngine

        async def fake_serialized(func, *args, **kwargs):
            assert not engine._generation_lock.locked()
            return []

        with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=True):
            engine = SimpleEngine("test-model")
            engine._loaded = True
            engine._text_model = MagicMock()
            engine._text_model.make_mtp_cache = MagicMock(return_value=[])
            engine._text_tokenizer = MagicMock()
            engine._text_tokenizer.apply_chat_template = MagicMock(return_value="hello")
            engine._text_tokenizer.bos_token = None
            engine._draft_model = None
            engine._run_blocking_serialized = fake_serialized  # type: ignore[method-assign]

            outputs = []
            async for chunk in engine._stream_generate_text(
                messages=[{"role": "user", "content": "hello"}],
                max_tokens=4,
                temperature=0.7,
                top_p=0.9,
            ):
                outputs.append(chunk)

            assert len(outputs) == 1
            assert outputs[0].finished
            assert outputs[0].completion_tokens == 0

    @pytest.mark.anyio
    async def test_specprefill_threads_same_cancel_check_to_helpers(self):
        """SpecPrefill worker should pass one cooperative cancel hook through both phases."""
        from vllm_mlx.engine.simple import SimpleEngine

        captured = {}

        def fake_score_tokens(*args, cancel_check=None, **kwargs):
            captured["score"] = cancel_check
            return mx.array([0.5], dtype=mx.float32)

        def fake_sparse_prefill(*args, cancel_check=None, **kwargs):
            captured["prefill"] = cancel_check
            return mx.zeros((1, 1, 8), dtype=mx.float32)

        def fake_select_chunks(_importance, **kwargs):
            captured["select_chunks_kwargs"] = kwargs
            return mx.array([0], dtype=mx.int32)

        with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False):
            engine = SimpleEngine("test-model")
            engine._loaded = True
            engine._draft_model = MagicMock()
            engine._model = MagicMock()
            engine._model.model = MagicMock()
            engine._model.tokenizer = MagicMock()
            engine._model.tokenizer.decode = MagicMock(return_value="A")
            engine._model.tokenizer.eos_token_id = 0

            outputs = []
            with (
                patch("mlx_lm.models.cache.make_prompt_cache", return_value=[]),
                patch(
                    "mlx_lm.sample_utils.make_sampler",
                    return_value=lambda logits: mx.array([0], dtype=mx.int32),
                ),
                patch(
                    "vllm_mlx.specprefill.score_tokens", side_effect=fake_score_tokens
                ),
                patch(
                    "vllm_mlx.specprefill.select_chunks",
                    side_effect=fake_select_chunks,
                ),
                patch(
                    "vllm_mlx.specprefill.sparse_prefill",
                    side_effect=fake_sparse_prefill,
                ),
                patch("vllm_mlx.specprefill.cleanup_rope"),
            ):
                async for chunk in engine._stream_generate_specprefill(
                    prompt="hello",
                    tokens=[1, 2, 3, 4],
                    max_tokens=4,
                    temperature=0.7,
                    top_p=0.9,
                    specprefill_backbone_pct=0.25,
                ):
                    outputs.append(chunk.new_text)

        assert outputs == ["A"]
        assert callable(captured["score"])
        assert captured["score"] is captured["prefill"]
        assert captured["select_chunks_kwargs"]["backbone_pct"] == 0.25

    @pytest.mark.anyio
    async def test_cancelling_specprefill_request_stops_during_scoring(self):
        """Cancelling SpecPrefill should signal the blocking scorer and exit without output."""
        import time
        from threading import Event

        from vllm_mlx.engine.simple import SimpleEngine, _SpecPrefillCancelled

        score_started = Event()
        score_cancelled = Event()

        def fake_score_tokens(*args, cancel_check=None, **kwargs):
            assert callable(cancel_check)
            score_started.set()
            while True:
                try:
                    cancel_check()
                except _SpecPrefillCancelled:
                    score_cancelled.set()
                    raise
                time.sleep(0.01)

        with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False):
            engine = SimpleEngine("test-model")
            engine._loaded = True
            engine._draft_model = MagicMock()
            engine._model = MagicMock()
            engine._model.model = MagicMock()
            engine._model.tokenizer = MagicMock()

            async def consume():
                async for _chunk in engine._stream_generate_specprefill(
                    prompt="hello",
                    tokens=[1, 2, 3, 4],
                    max_tokens=4,
                    temperature=0.7,
                    top_p=0.9,
                ):
                    pytest.fail("Cancelled SpecPrefill request should not emit output")

            with (
                patch("mlx_lm.models.cache.make_prompt_cache", return_value=[]),
                patch(
                    "vllm_mlx.specprefill.score_tokens",
                    side_effect=fake_score_tokens,
                ),
                patch("vllm_mlx.specprefill.cleanup_rope"),
            ):
                task = asyncio.create_task(consume())
                assert await asyncio.to_thread(score_started.wait, 1.0)
                task.cancel()

                with pytest.raises(asyncio.CancelledError):
                    await task

        assert await asyncio.to_thread(score_cancelled.wait, 1.0)

    @pytest.mark.anyio
    async def test_cancelling_specprefill_request_stops_during_sparse_prefill(self):
        """Cancelling SpecPrefill should signal the sparse-prefill loop and exit without output."""
        import time
        from threading import Event

        from vllm_mlx.engine.simple import SimpleEngine, _SpecPrefillCancelled

        prefill_started = Event()
        prefill_cancelled = Event()

        def fake_sparse_prefill(*args, cancel_check=None, **kwargs):
            assert callable(cancel_check)
            prefill_started.set()
            while True:
                try:
                    cancel_check()
                except _SpecPrefillCancelled:
                    prefill_cancelled.set()
                    raise
                time.sleep(0.01)

        with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False):
            engine = SimpleEngine("test-model")
            engine._loaded = True
            engine._draft_model = MagicMock()
            engine._model = MagicMock()
            engine._model.model = MagicMock()
            engine._model.tokenizer = MagicMock()

            async def consume():
                async for _chunk in engine._stream_generate_specprefill(
                    prompt="hello",
                    tokens=[1, 2, 3, 4],
                    max_tokens=4,
                    temperature=0.7,
                    top_p=0.9,
                ):
                    pytest.fail("Cancelled SpecPrefill request should not emit output")

            with (
                patch("mlx_lm.models.cache.make_prompt_cache", return_value=[]),
                patch(
                    "vllm_mlx.specprefill.score_tokens",
                    return_value=mx.array([0.5], dtype=mx.float32),
                ),
                patch(
                    "vllm_mlx.specprefill.select_chunks",
                    return_value=mx.array([0], dtype=mx.int32),
                ),
                patch(
                    "vllm_mlx.specprefill.sparse_prefill",
                    side_effect=fake_sparse_prefill,
                ),
                patch("vllm_mlx.specprefill.cleanup_rope"),
            ):
                task = asyncio.create_task(consume())
                assert await asyncio.to_thread(prefill_started.wait, 1.0)
                task.cancel()

                with pytest.raises(asyncio.CancelledError):
                    await task
        assert await asyncio.to_thread(prefill_cancelled.wait, 1.0)


class TestSimpleEngineClearRuntimeCaches:
    """Operational reset (DELETE /v1/cache) must actually release the
    multi-slot system-prompt KV cache state introduced in the LRU patch —
    otherwise multi-GB Metal-heap snapshots can survive a reset while
    /v1/cache/stats still reports them.
    """

    def test_clear_runtime_caches_drops_lru_and_resets_counters(self):
        from collections import OrderedDict

        from vllm_mlx.engine.simple import SimpleEngine

        engine = SimpleEngine("test-model")
        # Seed a populated LRU + non-zero counters as if the engine had been
        # serving traffic with two distinct system prefixes.
        engine._system_kv_cache = OrderedDict(
            [
                ("hash_a", ([b"snap_a_layer_0", b"snap_a_layer_1"], 28000)),
                ("hash_b", ([b"snap_b_layer_0", b"snap_b_layer_1"], 6500)),
            ]
        )
        engine._system_kv_cache_stats = {
            "hits": 5,
            "misses": 2,
            "stores": 2,
            "evictions": 0,
        }

        result = engine.clear_runtime_caches()

        assert len(engine._system_kv_cache) == 0, "LRU must be empty after clear"
        assert engine._system_kv_cache_stats == {
            "hits": 0,
            "misses": 0,
            "stores": 0,
            "evictions": 0,
        }, "counters must reset so /v1/cache/stats reflects cleared state"
        assert result is not None
        assert result["system_kv_cache"] == {"dropped_entries": 2}

    def test_clear_runtime_caches_no_op_when_lru_empty_and_counters_zero(self):
        from collections import OrderedDict

        from vllm_mlx.engine.simple import SimpleEngine

        engine = SimpleEngine("test-model")
        engine._system_kv_cache = OrderedDict()
        engine._system_kv_cache_stats = {
            "hits": 0,
            "misses": 0,
            "stores": 0,
            "evictions": 0,
        }

        result = engine.clear_runtime_caches()

        # Non-MLLM, empty LRU, zeroed counters → nothing to report.
        assert result is None
