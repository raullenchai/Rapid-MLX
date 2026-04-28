# SPDX-License-Identifier: Apache-2.0
"""
Tests for continuous batching system.

These tests verify the scheduler, engine, and request handling
for the vLLM-style continuous batching implementation.
"""

import asyncio
import sys
import threading
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm_mlx.request import (
    Request,
    RequestOutput,
    RequestStatus,
    SamplingParams,
)
from vllm_mlx.scheduler import (
    Scheduler,
    SchedulerConfig,
    SchedulingPolicy,
)


class TestRequest:
    """Tests for Request class."""

    def test_request_creation(self):
        """Test basic request creation."""
        params = SamplingParams(max_tokens=100, temperature=0.8)
        request = Request(
            request_id="test-1",
            prompt="Hello, world!",
            sampling_params=params,
        )

        assert request.request_id == "test-1"
        assert request.prompt == "Hello, world!"
        assert request.sampling_params.max_tokens == 100
        assert request.status == RequestStatus.WAITING
        assert not request.is_finished()

    def test_request_status_transitions(self):
        """Test request status transitions."""
        request = Request(
            request_id="test-1",
            prompt="Hello",
            sampling_params=SamplingParams(),
        )

        assert request.status == RequestStatus.WAITING
        assert not request.is_finished()

        request.status = RequestStatus.RUNNING
        assert not request.is_finished()

        request.set_finished(RequestStatus.FINISHED_STOPPED)
        assert request.is_finished()
        assert request.get_finish_reason() == "stop"

    def test_request_output_tokens(self):
        """Test appending output tokens."""
        request = Request(
            request_id="test-1",
            prompt="Hello",
            sampling_params=SamplingParams(),
        )
        request.prompt_token_ids = [1, 2, 3]
        request.num_prompt_tokens = 3

        assert request.num_output_tokens == 0
        assert request.num_tokens == 3

        request.append_output_token(100)
        request.append_output_token(101)

        assert request.num_output_tokens == 2
        assert request.num_tokens == 5
        assert request.output_token_ids == [100, 101]

    def test_request_comparison(self):
        """Test request comparison for priority queue."""
        req1 = Request(
            request_id="req-1",
            prompt="Hello",
            sampling_params=SamplingParams(),
            priority=0,
            arrival_time=1.0,
        )
        req2 = Request(
            request_id="req-2",
            prompt="World",
            sampling_params=SamplingParams(),
            priority=1,
            arrival_time=0.5,
        )
        req3 = Request(
            request_id="req-3",
            prompt="Test",
            sampling_params=SamplingParams(),
            priority=0,
            arrival_time=2.0,
        )

        # Lower priority value = higher priority
        assert req1 < req2
        # Same priority, earlier arrival = higher priority
        assert req1 < req3


class TestSamplingParams:
    """Tests for SamplingParams."""

    def test_default_params(self):
        """Test default sampling parameters."""
        params = SamplingParams()

        assert params.max_tokens == 256
        assert params.temperature == 0.7
        assert params.top_p == 0.9
        assert params.stop == []
        assert params.stop_token_ids == []

    def test_custom_params(self):
        """Test custom sampling parameters."""
        params = SamplingParams(
            max_tokens=100,
            temperature=0.5,
            top_p=0.95,
            top_k=50,
            stop=["END"],
            stop_token_ids=[1, 2],
        )

        assert params.max_tokens == 100
        assert params.temperature == 0.5
        assert params.top_p == 0.95
        assert params.top_k == 50
        assert params.stop == ["END"]
        assert params.stop_token_ids == [1, 2]


class TestRequestOutput:
    """Tests for RequestOutput."""

    def test_output_creation(self):
        """Test output creation."""
        output = RequestOutput(
            request_id="test-1",
            new_token_ids=[100, 101],
            new_text="Hello",
            output_token_ids=[100, 101],
            output_text="Hello",
            finished=True,
            finish_reason="stop",
            prompt_tokens=10,
            completion_tokens=2,
        )

        assert output.request_id == "test-1"
        assert output.finished
        assert output.finish_reason == "stop"

        usage = output.usage
        assert usage["prompt_tokens"] == 10
        assert usage["completion_tokens"] == 2
        assert usage["total_tokens"] == 12


class TestSchedulerConfig:
    """Tests for SchedulerConfig."""

    def test_default_config(self):
        """Test default scheduler config."""
        config = SchedulerConfig()

        assert config.max_num_seqs == 256
        assert config.policy == SchedulingPolicy.FCFS
        assert config.prefill_batch_size == 8
        assert config.completion_batch_size == 32

    def test_custom_config(self):
        """Test custom scheduler config."""
        config = SchedulerConfig(
            max_num_seqs=64,
            policy=SchedulingPolicy.PRIORITY,
            prefill_batch_size=4,
            completion_batch_size=16,
        )

        assert config.max_num_seqs == 64
        assert config.policy == SchedulingPolicy.PRIORITY


class TestSchedulerBasic:
    """Basic tests for Scheduler (without real model)."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        tokenizer = MagicMock()
        tokenizer.encode = lambda x: list(range(len(x.split())))
        tokenizer.decode = lambda x: " ".join(str(t) for t in x)
        tokenizer.eos_token_id = 0
        tokenizer.eos_token_ids = {0}
        return tokenizer

    @pytest.fixture
    def mock_model(self):
        """Create a mock model."""
        return MagicMock()

    def test_scheduler_creation(self, mock_model, mock_tokenizer):
        """Test scheduler creation."""
        scheduler = Scheduler(
            model=mock_model,
            tokenizer=mock_tokenizer,
            config=SchedulerConfig(max_num_seqs=10),
        )

        assert scheduler.get_num_waiting() == 0
        assert scheduler.get_num_running() == 0
        assert not scheduler.has_requests()

    def test_add_request(self, mock_model, mock_tokenizer):
        """Test adding requests to scheduler."""
        scheduler = Scheduler(
            model=mock_model,
            tokenizer=mock_tokenizer,
        )

        request = Request(
            request_id="test-1",
            prompt="Hello world",
            sampling_params=SamplingParams(max_tokens=10),
        )

        scheduler.add_request(request)

        assert scheduler.get_num_waiting() == 1
        assert scheduler.has_requests()
        assert scheduler.get_request("test-1") is not None

    def test_add_duplicate_request(self, mock_model, mock_tokenizer):
        """Test adding duplicate request raises error."""
        scheduler = Scheduler(
            model=mock_model,
            tokenizer=mock_tokenizer,
        )

        request = Request(
            request_id="test-1",
            prompt="Hello",
            sampling_params=SamplingParams(),
        )

        scheduler.add_request(request)

        with pytest.raises(ValueError, match="already exists"):
            scheduler.add_request(request)

    def test_abort_waiting_request(self, mock_model, mock_tokenizer):
        """Test aborting a waiting request (deferred abort pattern)."""
        scheduler = Scheduler(
            model=mock_model,
            tokenizer=mock_tokenizer,
        )

        request = Request(
            request_id="test-1",
            prompt="Hello",
            sampling_params=SamplingParams(),
        )

        scheduler.add_request(request)
        assert scheduler.get_num_waiting() == 1

        # abort_request() enqueues for deferred processing
        result = scheduler.abort_request("test-1")
        assert result is True

        # Process pending aborts (normally happens inside step())
        scheduler._process_pending_aborts()

        assert scheduler.get_num_waiting() == 0
        assert "test-1" in scheduler.finished_req_ids

    def test_abort_nonexistent_request(self, mock_model, mock_tokenizer):
        """Test aborting non-existent request (deferred abort always enqueues)."""
        scheduler = Scheduler(
            model=mock_model,
            tokenizer=mock_tokenizer,
        )

        # abort_request() always returns True (enqueue is always successful)
        result = scheduler.abort_request("nonexistent")
        assert result is True

    def test_get_stats(self, mock_model, mock_tokenizer):
        """Test getting scheduler stats."""
        scheduler = Scheduler(
            model=mock_model,
            tokenizer=mock_tokenizer,
        )

        stats = scheduler.get_stats()

        assert "num_waiting" in stats
        assert "num_running" in stats
        assert "num_requests_processed" in stats
        assert stats["num_waiting"] == 0
        assert stats["num_running"] == 0

    def test_reset(self, mock_model, mock_tokenizer):
        """Test resetting scheduler."""
        scheduler = Scheduler(
            model=mock_model,
            tokenizer=mock_tokenizer,
        )

        # Add some requests
        for i in range(5):
            request = Request(
                request_id=f"test-{i}",
                prompt=f"Hello {i}",
                sampling_params=SamplingParams(),
            )
            scheduler.add_request(request)

        assert scheduler.get_num_waiting() == 5

        scheduler.reset()

        assert scheduler.get_num_waiting() == 0
        assert scheduler.get_num_running() == 0
        assert not scheduler.has_requests()


# Integration tests require actual MLX model
@pytest.mark.integration
class TestSchedulerIntegration:
    """Integration tests that require a real model."""

    @pytest.fixture
    def model_and_tokenizer(self):
        """Load a small test model."""
        try:
            from mlx_lm import load

            model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
            return model, tokenizer
        except Exception as e:
            pytest.skip(f"Could not load test model: {e}")

    def test_scheduler_with_real_model(self, model_and_tokenizer):
        """Test scheduler with real model."""
        model, tokenizer = model_and_tokenizer

        scheduler = Scheduler(
            model=model,
            tokenizer=tokenizer,
            config=SchedulerConfig(
                max_num_seqs=4,
                prefill_batch_size=2,
                completion_batch_size=4,
            ),
        )

        # Add a request
        request = Request(
            request_id="test-1",
            prompt="What is 2+2?",
            sampling_params=SamplingParams(max_tokens=10),
        )
        scheduler.add_request(request)

        # Run a few steps
        outputs = []
        for _ in range(20):
            output = scheduler.step()
            if output.outputs:
                outputs.extend(output.outputs)
            if output.finished_request_ids:
                break

        assert len(outputs) > 0
        # Check we got at least one output
        final_output = outputs[-1]
        assert final_output.request_id == "test-1"

    def test_multiple_concurrent_requests(self, model_and_tokenizer):
        """Test handling multiple concurrent requests."""
        model, tokenizer = model_and_tokenizer

        scheduler = Scheduler(
            model=model,
            tokenizer=tokenizer,
            config=SchedulerConfig(
                max_num_seqs=8,
                prefill_batch_size=4,
                completion_batch_size=8,
            ),
        )

        # Add multiple requests
        prompts = [
            "What is 1+1?",
            "What is 2+2?",
            "What is 3+3?",
            "What is 4+4?",
        ]

        for i, prompt in enumerate(prompts):
            request = Request(
                request_id=f"test-{i}",
                prompt=prompt,
                sampling_params=SamplingParams(max_tokens=10),
            )
            scheduler.add_request(request)

        # Run until all complete
        finished = set()
        max_steps = 100
        steps = 0

        while len(finished) < len(prompts) and steps < max_steps:
            output = scheduler.step()
            finished.update(output.finished_request_ids)
            steps += 1

        assert len(finished) == len(prompts), f"Only {len(finished)} requests finished"


class TestEngineThreading:
    """Threading tests for EngineCore."""

    def test_mlx_step_thread_initializer_rebinds_generation_stream(self, monkeypatch):
        """The executor thread must own mlx-lm's generation stream."""
        from vllm_mlx import engine_core

        fake_generate = types.SimpleNamespace(generation_stream="old-stream")
        monkeypatch.setitem(sys.modules, "mlx_lm.generate", fake_generate)
        monkeypatch.setattr(engine_core.mx, "default_device", lambda: "gpu")
        monkeypatch.setattr(
            engine_core.mx, "new_stream", lambda device: f"stream:{device}"
        )

        engine_core._init_mlx_step_thread()

        assert fake_generate.generation_stream == "stream:gpu"


@pytest.mark.asyncio
class TestEngineAsync:
    """Async tests for the engine."""

    @pytest.fixture
    def mock_model_and_tokenizer(self):
        """Create mock model and tokenizer."""
        model = MagicMock()
        tokenizer = MagicMock()
        tokenizer.encode = lambda x: list(range(len(x.split())))
        tokenizer.decode = lambda x: " ".join(str(t) for t in x)
        tokenizer.eos_token_id = 0
        tokenizer.eos_token_ids = {0}
        return model, tokenizer

    async def test_engine_loop_keeps_all_scheduler_steps_on_mlx_thread(
        self, mock_model_and_tokenizer
    ):
        """Prefill and decode steps must run on the same MLX worker thread."""
        from vllm_mlx import engine_core
        from vllm_mlx.engine import EngineConfig, EngineCore

        model, tokenizer = mock_model_and_tokenizer
        engine = EngineCore(model, tokenizer, EngineConfig(step_interval=0.001))

        class FakeScheduler:
            batch_generator = None

            def __init__(self):
                self.calls = 0
                self.thread_names = []

            def has_requests(self):
                return self.calls < 2

            def step(self):
                self.thread_names.append(threading.current_thread().name)
                self.calls += 1
                if self.calls >= 2:
                    engine._running = False
                return SimpleNamespace(outputs=[], finished_request_ids=[])

            def deep_reset(self):
                pass

        fake_scheduler = FakeScheduler()
        engine.scheduler = fake_scheduler

        # Mirror what start() does — create the mlx-step worker executor so
        # _engine_loop() picks it up. Tests can't call start() directly here
        # because start() spawns a task and returns immediately.
        import concurrent.futures

        engine._mlx_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="mlx-step",
            initializer=engine_core._init_mlx_step_thread,
        )
        engine._running = True

        try:
            await asyncio.wait_for(engine._engine_loop(), timeout=2)
        finally:
            engine._running = False
            engine._mlx_executor.shutdown(wait=True)
            engine._mlx_executor = None
            engine.close()

        assert fake_scheduler.thread_names
        assert all(name.startswith("mlx-step") for name in fake_scheduler.thread_names)

    async def test_engine_lifecycle(self, mock_model_and_tokenizer):
        """Test engine start/stop lifecycle."""
        from vllm_mlx.engine import AsyncEngineCore, EngineConfig

        model, tokenizer = mock_model_and_tokenizer

        engine = AsyncEngineCore(model, tokenizer, EngineConfig())

        assert not engine.engine.is_running()

        # Use async context manager
        async with engine:
            assert engine.engine.is_running()
            await asyncio.sleep(0.05)

        assert not engine.engine.is_running()

    async def test_engine_context_manager(self, mock_model_and_tokenizer):
        """Test engine as async context manager."""
        from vllm_mlx.engine import AsyncEngineCore

        model, tokenizer = mock_model_and_tokenizer

        async with AsyncEngineCore(model, tokenizer) as engine:
            assert engine.engine.is_running()

        assert not engine.engine.is_running()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
