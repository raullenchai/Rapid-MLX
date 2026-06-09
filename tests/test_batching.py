# SPDX-License-Identifier: Apache-2.0
"""
Tests for continuous batching system.

These tests verify the scheduler, engine, and request handling
for the vLLM-style continuous batching implementation.
"""

import asyncio
import importlib
import pytest
from unittest.mock import MagicMock
import mlx.core as mx

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
    _install_chunked_prefill,
)

mlx_generate = importlib.import_module("mlx_lm.generate")


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

    def test_chunked_prefill_accepts_prompt_checkpoints(self, monkeypatch):
        """Chunked prefill must match mlx-lm's 7-field prompt tuples."""

        class FakeCacheEntry:
            def empty(self):
                return True

        class FakePromptCache:
            def __init__(self):
                self.state = mx.array([0])

            def finalize(self):
                return None

        class FakeStats:
            prompt_tokens = 0
            prompt_time = 0.0
            generation_time = 0.0

        class FakeBatchGenerator:
            def __init__(self):
                self._stats = FakeStats()
                self._partial = None
                self.active_batch = None
                self.unprocessed_prompts = [
                    (
                        7,
                        [1, 2, 3, 4, 5],
                        16,
                        [FakeCacheEntry()],
                        None,
                        [None],
                        2,
                    )
                ]
                self.prefill_batch_size = 1
                self.completion_batch_size = 1
                self.max_kv_size = None
                self.stop_tokens = set()
                self.prompt_progress_callback = lambda _progress: None
                self.prompt_checkpoint_callback = None
                self._next = lambda: []
                self.remove = lambda _uids: None
                self._process_prompts = lambda _prompts: None
                self.model = lambda _inputs, cache=None: None

        monkeypatch.setattr(
            mlx_generate,
            "_left_pad_prompts",
            lambda prompts, max_length=None: mx.array(prompts),
        )
        monkeypatch.setattr(
            mlx_generate,
            "_make_cache",
            lambda _model, _padding, _max_kv_size=None: [FakePromptCache()],
        )

        batch_gen = FakeBatchGenerator()
        _install_chunked_prefill(batch_gen, budget=4)

        responses = batch_gen._next()

        assert responses == []
        assert batch_gen._partial is not None
        assert batch_gen._partial["prompt_checkpoint"] == 3
        assert batch_gen._partial["processed"] == 2

    def test_chunked_prefill_invokes_checkpoint_callback(self, monkeypatch):
        """prompt_checkpoint_callback must fire after finalization."""

        class FakeCacheEntry:
            def empty(self):
                return True

        class FakePromptCache:
            def __init__(self):
                self.state = mx.array([0])

            def finalize(self):
                return None

            def extract(self, idx):
                return self

        class FakeStats:
            prompt_tokens = 0
            prompt_time = 0.0
            generation_time = 0.0
            generation_tokens = 0

        callback_payloads = []

        from collections import namedtuple

        _Response = namedtuple(
            "Response", ["uid", "token", "logprobs", "finish_reason", "cache"]
        )

        class FakeBatchGenerator:
            Response = _Response

            def __init__(self):
                self._stats = FakeStats()
                self._partial = None
                self.active_batch = None
                self.unprocessed_prompts = [
                    (
                        7,
                        [1, 2, 3],
                        16,
                        [FakeCacheEntry()],
                        None,
                        [None],
                        2,
                    )
                ]
                self.prefill_batch_size = 1
                self.completion_batch_size = 1
                self.max_kv_size = None
                self.stop_tokens = set()
                self.prompt_progress_callback = lambda _progress: None
                self.prompt_checkpoint_callback = (
                    lambda entries: callback_payloads.extend(entries)
                )
                self._next = lambda: []
                self.remove = lambda _uids: None
                self._process_prompts = lambda _prompts: None
                self.model = lambda _inputs, cache=None: None

            def _step(self, inputs, cache, samplers, logits_processors, tokens):
                return mx.array([99]), mx.array([-1.0])

            def _generation_step(self):
                if self.active_batch is not None:
                    self.active_batch = None
                return []

        monkeypatch.setattr(
            mlx_generate,
            "_left_pad_prompts",
            lambda prompts, max_length=None: mx.array(prompts),
        )
        monkeypatch.setattr(
            mlx_generate,
            "_make_cache",
            lambda _model, _padding, _max_kv_size=None: [FakePromptCache()],
        )

        batch_gen = FakeBatchGenerator()
        batch_gen.stop_tokens = {99}
        _install_chunked_prefill(batch_gen, budget=1)

        # First _next: starts partial prefill (processes 1 token)
        batch_gen._next()
        assert batch_gen._partial is not None

        # Second _next: finishes prefill, fires checkpoint callback,
        # then runs generation step which completes (stop token).
        batch_gen._next()

        assert len(callback_payloads) == 1
        uid, checkpoint, _cache_gen = callback_payloads[0]
        assert uid == 7
        assert checkpoint == 1

    def test_chunked_prefill_replays_checkpoint_tail_before_step(self, monkeypatch):
        """checkpoint tails >1 must be replayed after finalize before _step."""

        class FakeCacheEntry:
            def empty(self):
                return True

        class FakePromptCache:
            def __init__(self):
                self.state = mx.array([0])

            def finalize(self):
                return None

            def extract(self, idx):
                return self

        class FakeStats:
            prompt_tokens = 0
            prompt_time = 0.0
            generation_time = 0.0
            generation_tokens = 0

        callback_payloads = []
        model_calls = []
        step_inputs = []

        from collections import namedtuple

        _Response = namedtuple(
            "Response", ["uid", "token", "logprobs", "finish_reason", "cache"]
        )

        class FakeBatchGenerator:
            Response = _Response

            def __init__(self):
                self._stats = FakeStats()
                self._partial = None
                self.active_batch = None
                self.unprocessed_prompts = [
                    (
                        7,
                        [1, 2, 3, 4, 5],
                        16,
                        [FakeCacheEntry()],
                        None,
                        [None],
                        2,
                    )
                ]
                self.prefill_batch_size = 1
                self.completion_batch_size = 1
                self.max_kv_size = None
                self.stop_tokens = {99}
                self.prompt_progress_callback = lambda _progress: None
                self.prompt_checkpoint_callback = (
                    lambda entries: callback_payloads.extend(entries)
                )
                self._next = lambda: []
                self.remove = lambda _uids: None
                self._process_prompts = lambda _prompts: None

            def model(self, inputs, cache=None):
                model_calls.append(inputs.tolist())

            def _step(self, inputs, cache, samplers, logits_processors, tokens):
                step_inputs.append(inputs.tolist())
                return mx.array([99]), mx.array([-1.0])

        monkeypatch.setattr(
            mlx_generate,
            "_left_pad_prompts",
            lambda prompts, max_length=None: mx.array(prompts),
        )
        monkeypatch.setattr(
            mlx_generate,
            "_make_cache",
            lambda _model, _padding, _max_kv_size=None: [FakePromptCache()],
        )

        batch_gen = FakeBatchGenerator()
        _install_chunked_prefill(batch_gen, budget=2)

        # First _next: process the first chunk and leave a 3-token checkpoint tail.
        batch_gen._next()
        assert batch_gen._partial is not None
        assert batch_gen._partial["prompt_checkpoint"] == 3

        # Second _next: finalize, fire callback, replay the checkpoint tail, step.
        batch_gen._next()

        assert model_calls == [[[1, 2]], [[3, 4]]]
        assert step_inputs[0] == [[3, 4, 5]]
        assert len(callback_payloads) == 1
        uid, checkpoint, _cache_gen = callback_payloads[0]
        assert uid == 7
        assert checkpoint == 3

    def test_chunked_prefill_works_without_private_mlx_generate_exports(
        self, monkeypatch
    ):
        """Chunked prefill should tolerate missing private mlx_lm.generate exports."""

        class FakeCacheEntry:
            def empty(self):
                return True

        class FakePromptCache:
            def __init__(self):
                self.state = mx.array([0])

            def finalize(self):
                return None

            def extract(self, idx):
                return self

        class FakeStats:
            prompt_tokens = 0
            prompt_time = 0.0
            generation_time = 0.0
            generation_tokens = 0

        from collections import namedtuple

        _Response = namedtuple(
            "Response", ["uid", "token", "logprobs", "finish_reason", "cache"]
        )

        class FakeBatchGenerator:
            Response = _Response

            def __init__(self):
                self._stats = FakeStats()
                self._partial = None
                self.active_batch = None
                self.unprocessed_prompts = [
                    (
                        7,
                        [1, 2, 3],
                        16,
                        [FakeCacheEntry()],
                        None,
                        [None],
                        2,
                    )
                ]
                self.prefill_batch_size = 1
                self.completion_batch_size = 1
                self.max_kv_size = None
                self.stop_tokens = {99}
                self.prompt_progress_callback = lambda _progress: None
                self.prompt_checkpoint_callback = None
                self._next = lambda: []
                self.remove = lambda _uids: None
                self._process_prompts = lambda _prompts: None
                self.model = lambda _inputs, cache=None: None

            def _step(self, inputs, cache, samplers, logits_processors, tokens):
                return mx.array([99]), mx.array([-1.0])

            def _generation_step(self):
                if self.active_batch is not None:
                    self.active_batch = None
                return []

        monkeypatch.delattr(mlx_generate, "Batch", raising=False)
        monkeypatch.delattr(mlx_generate, "_lazy_extract_cache", raising=False)
        monkeypatch.setattr(
            mlx_generate,
            "_left_pad_prompts",
            lambda prompts, max_length=None: mx.array(prompts),
        )
        monkeypatch.setattr(
            mlx_generate,
            "_make_cache",
            lambda _model, _padding, _max_kv_size=None: [FakePromptCache()],
        )

        batch_gen = FakeBatchGenerator()
        _install_chunked_prefill(batch_gen, budget=1)

        batch_gen._next()
        assert batch_gen._partial is not None
        batch_gen._next()
        assert batch_gen.active_batch is None

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

    def test_step_normalizes_stale_none_logits_processor_slots(
        self, mock_model, mock_tokenizer
    ):
        """Scheduler should sanitize stale BatchGenerator per-slot None values."""

        class FakeActiveBatch:
            logits_processors = [None]

        class FakeBatchGenerator:
            def __init__(self):
                self.active_batch = FakeActiveBatch()
                self.called = False

            def next(self):
                self.called = True
                assert self.active_batch.logits_processors == [[]]
                return []

        scheduler = Scheduler(
            model=mock_model,
            tokenizer=mock_tokenizer,
        )
        request = Request(
            request_id="test-1",
            prompt="Hello",
            sampling_params=SamplingParams(max_tokens=2),
        )
        request.status = RequestStatus.RUNNING
        scheduler.requests[request.request_id] = request
        scheduler.running[request.request_id] = request
        batch_generator = FakeBatchGenerator()
        scheduler.batch_generator = batch_generator

        output = scheduler.step()

        assert batch_generator.called is True
        assert output.finished_request_ids == set()
        assert request.request_id in scheduler.running


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


@pytest.mark.anyio
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

    async def test_stream_outputs_consumer_break_after_finished_does_not_abort(self):
        """Breaking after a finished output is normal consumption, not orphaning."""
        from vllm_mlx.engine_core import EngineCore
        from vllm_mlx.output_collector import RequestOutputCollector
        from vllm_mlx.request import RequestOutput

        engine = EngineCore.__new__(EngineCore)
        engine._output_collectors = {"req-1": RequestOutputCollector()}
        engine._stream_states = {}
        engine._finished_events = {}
        engine.scheduler = MagicMock()
        engine.scheduler.abort_request = MagicMock(return_value=True)
        engine.scheduler.remove_finished_request = MagicMock()

        engine._output_collectors["req-1"].put(
            RequestOutput(
                request_id="req-1",
                output_text="done",
                finished=True,
                finish_reason="stop",
            )
        )

        stream = EngineCore.stream_outputs(engine, "req-1")
        output = await stream.__anext__()
        assert output.finished is True
        await stream.aclose()

        engine.scheduler.abort_request.assert_not_called()
        engine.scheduler.remove_finished_request.assert_called_once_with("req-1")


class TestChunkedPrefillConfig:
    """Regression tests for chunked prefill configuration (#178)."""

    def test_prompt_cache_save_installed_without_chunked_prefill(self):
        """When chunked_prefill_tokens=0 but memory_aware_cache is active,
        _install_prompt_cache_save should still patch _process_prompts."""
        from vllm_mlx.scheduler import _install_prompt_cache_save

        calls = []

        class FakeBatchGen:
            def _process_prompts(self, prompts):
                class FakeBatch:
                    uids = [42]
                    num_tokens = [0]

                    def extract_cache(self, idx):
                        return f"cache-{idx}"

                return FakeBatch()

        bg = FakeBatchGen()
        orig_fn = bg._process_prompts
        _install_prompt_cache_save(bg, lambda uid, cache: calls.append((uid, cache)))

        # _process_prompts was patched
        assert bg._process_prompts is not orig_fn
        bg._process_prompts([])
        assert calls == [(42, "cache-0")], f"Expected callback to fire, got {calls}"

    def test_chunked_prefill_zero_does_not_install_chunked_next(self):
        """chunked_prefill_tokens=0 must not install the chunked _next patch,
        even when use_memory_aware_cache=True."""
        config = SchedulerConfig(chunked_prefill_tokens=0, use_memory_aware_cache=True)
        need_chunked = config.chunked_prefill_tokens > 0
        assert not need_chunked, (
            "chunked_prefill_tokens=0 must not enable the chunked prefill "
            "monkey-patch even when use_memory_aware_cache=True"
        )

    def test_chunked_prefill_positive_enables(self):
        """Positive chunked_prefill_tokens should enable chunked prefill."""
        config = SchedulerConfig(chunked_prefill_tokens=4096)
        need_chunked = config.chunked_prefill_tokens > 0
        assert need_chunked


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
