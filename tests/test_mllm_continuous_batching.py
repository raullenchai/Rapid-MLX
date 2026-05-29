# SPDX-License-Identifier: Apache-2.0
"""
Tests for MLLM (Multimodal Language Model) continuous batching.

These tests verify that the MLLM batch generator and scheduler work correctly
for batching multiple multimodal requests together.

Test Cases:
- Single MLLM request works correctly
- 2, 4, 8 concurrent requests with batching
- Vision cache hits/misses
- Streaming with batching
- Mixed text-only and multimodal requests
"""

import asyncio
import base64
import os
import threading
import tempfile
from contextlib import nullcontext
from unittest.mock import MagicMock

import pytest

# Skip all tests if MLX is not available
try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not available")


# Test image (small PNG)
TEST_IMAGE_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="


class TestMLLMPromptCacheEval:
    def test_collects_kv_and_arrays_cache_tensors(self):
        from vllm_mlx.mllm_batch_generator import _cache_eval_tensors

        kv_keys = object()
        kv_values = object()
        state_a = object()
        state_b = object()

        class KVLikeCache:
            keys = kv_keys
            values = kv_values

            @property
            def state(self):
                raise AssertionError("KV cache state should not be read")

        class ArraysLikeCache:
            state = [state_a, None, state_b]

        class EmptyKVLikeCache:
            keys = None
            values = None

            @property
            def state(self):
                raise AttributeError("empty KV cache has no state")

        assert _cache_eval_tensors(
            [KVLikeCache(), ArraysLikeCache(), EmptyKVLikeCache()]
        ) == [kv_keys, kv_values, state_a, state_b]

    def test_eval_prompt_cache_skips_empty_cache(self, monkeypatch):
        from vllm_mlx.mllm_batch_generator import _eval_prompt_cache

        eval_mock = MagicMock()
        monkeypatch.setattr(mx, "eval", eval_mock)

        class EmptyCache:
            keys = None
            values = None
            state = [None]

        _eval_prompt_cache([EmptyCache()])

        eval_mock.assert_not_called()

    def test_eval_prompt_cache_flattens_cache_tensors(self, monkeypatch):
        from vllm_mlx.mllm_batch_generator import _eval_prompt_cache

        kv_keys = object()
        kv_values = object()
        state = object()
        eval_mock = MagicMock()
        monkeypatch.setattr(mx, "eval", eval_mock)

        class KVLikeCache:
            keys = kv_keys
            values = kv_values

        class ArraysLikeCache:
            pass

        ArraysLikeCache.state = [state]

        _eval_prompt_cache([KVLikeCache(), ArraysLikeCache()])

        eval_mock.assert_called_once_with(kv_keys, kv_values, state)


class TestMLLMPendingRemovals:
    def test_process_pending_removals_atomic_swap_preserves_new_enqueues(self):
        from vllm_mlx.mllm_batch_generator import MLLMBatchGenerator

        class _InjectOnExhaustionIterator:
            def __init__(self, base_iter, on_exhaustion):
                self._base_iter = base_iter
                self._on_exhaustion = on_exhaustion
                self._injected = False

            def __iter__(self):
                return self

            def __next__(self):
                try:
                    return next(self._base_iter)
                except StopIteration:
                    if not self._injected:
                        self._injected = True
                        self._on_exhaustion()
                    raise

        class _InjectOnExhaustionSet(set):
            def __init__(self, values, on_exhaustion):
                super().__init__(values)
                self._on_exhaustion = on_exhaustion

            def __iter__(self):
                return _InjectOnExhaustionIterator(
                    super().__iter__(),
                    self._on_exhaustion,
                )

        batch_generator = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
        batch_generator._pending_removal_lock = threading.Lock()
        removed: list[int] = []

        def _remove(uids: list[int]) -> None:
            removed.extend(uids)

        batch_generator.remove = _remove
        batch_generator._pending_removal_uids = _InjectOnExhaustionSet(
            {1},
            lambda: batch_generator.schedule_removal([2]),
        )

        batch_generator.process_pending_removals()

        assert removed == [1]
        assert batch_generator._pending_removal_uids == {2}

        batch_generator.process_pending_removals()

        assert removed == [1, 2]
        assert batch_generator._pending_removal_uids == set()


def create_test_image(path: str, size: tuple = (32, 32)) -> str:
    """Create a test image file."""
    try:
        from PIL import Image
        import numpy as np

        img = Image.fromarray(np.random.randint(0, 255, (*size, 3), dtype=np.uint8))
        img.save(path)
        return path
    except ImportError:
        # Fallback: write a minimal valid PNG
        png_data = base64.b64decode(TEST_IMAGE_B64)
        with open(path, "wb") as f:
            f.write(png_data)
        return path


class TestMLLMBatchRequest:
    """Tests for MLLMBatchRequest dataclass."""

    def test_create_request(self):
        """Test creating a basic request."""
        from vllm_mlx.mllm_batch_generator import MLLMBatchRequest

        req = MLLMBatchRequest(
            uid=0,
            request_id="test-1",
            prompt="What's in this image?",
            images=["test.jpg"],
            max_tokens=100,
        )

        assert req.uid == 0
        assert req.request_id == "test-1"
        assert req.prompt == "What's in this image?"
        assert req.images == ["test.jpg"]
        assert req.max_tokens == 100
        assert req.num_tokens == 0
        assert req.vision_encoded is False

    def test_request_defaults(self):
        """Test default values."""
        from vllm_mlx.mllm_batch_generator import MLLMBatchRequest

        req = MLLMBatchRequest(
            uid=1,
            request_id="test-2",
            prompt="Hello",
        )

        assert req.images is None
        assert req.videos is None
        assert req.max_tokens == 256
        assert req.temperature == 0.7
        assert req.top_p == 0.9
        assert req.output_tokens == []


class TestMLLMBatchResponse:
    """Tests for MLLMBatchResponse dataclass."""

    def test_create_response(self):
        """Test creating a response."""
        from vllm_mlx.mllm_batch_generator import MLLMBatchResponse

        logprobs = mx.array([0.1, 0.2, 0.3])

        resp = MLLMBatchResponse(
            uid=0,
            request_id="test-1",
            token=42,
            logprobs=logprobs,
            finish_reason=None,
        )

        assert resp.uid == 0
        assert resp.request_id == "test-1"
        assert resp.token == 42
        assert resp.finish_reason is None

    def test_finished_response(self):
        """Test response with finish reason."""
        from vllm_mlx.mllm_batch_generator import MLLMBatchResponse

        resp = MLLMBatchResponse(
            uid=0,
            request_id="test-1",
            token=2,  # EOS
            logprobs=mx.array([0.1]),
            finish_reason="stop",
        )

        assert resp.finish_reason == "stop"

    def test_error_response_skips_decoding(self):
        """Error responses must not decode token=0 as content."""
        from unittest.mock import MagicMock

        from vllm_mlx.mllm_batch_generator import MLLMBatchResponse
        from vllm_mlx.mllm_scheduler import MLLMScheduler
        from vllm_mlx.request import RequestStatus

        # Build a minimal scheduler with mocked internals
        scheduler = MLLMScheduler.__new__(MLLMScheduler)
        scheduler._detokenizer_pool = {}
        scheduler.uid_to_request_id = {0: "req-err"}
        scheduler.total_completion_tokens = 0
        scheduler.num_requests_processed = 0

        mock_tokenizer = MagicMock()
        mock_tokenizer.decode.return_value = ""
        mock_processor = MagicMock()
        mock_processor.tokenizer = mock_tokenizer
        scheduler.processor = mock_processor

        # Create a running request
        mock_request = MagicMock()
        mock_request.request_id = "req-err"
        mock_request.output_tokens = []
        mock_request.num_output_tokens = 0
        mock_request.num_prompt_tokens = 10
        mock_request.status = RequestStatus.RUNNING
        scheduler.running = {"req-err": mock_request}

        error_resp = MLLMBatchResponse(
            uid=0,
            request_id="req-err",
            token=0,
            logprobs=mx.array([0.0]),
            finish_reason="error",
        )

        outputs, finished = scheduler._process_batch_responses([error_resp])

        assert "req-err" in finished
        assert mock_request.status == RequestStatus.FINISHED_ABORTED
        # token=0 should not have been decoded through a detokenizer
        assert "req-err" not in scheduler._detokenizer_pool
        assert len(outputs) == 1
        assert outputs[0].new_text == ""


class TestMLLMBatch:
    """Tests for MLLMBatch class."""

    def test_batch_length(self):
        """Test batch length calculation."""
        from vllm_mlx.mllm_batch_generator import MLLMBatch, MLLMBatchRequest

        requests = [
            MLLMBatchRequest(uid=i, request_id=f"req-{i}", prompt=f"prompt {i}")
            for i in range(3)
        ]

        batch = MLLMBatch(
            uids=[0, 1, 2],
            request_ids=["req-0", "req-1", "req-2"],
            y=mx.array([100, 200, 300]),
            logprobs=[mx.array([0.1]), mx.array([0.2]), mx.array([0.3])],
            max_tokens=[100, 100, 100],
            num_tokens=[0, 0, 0],
            cache=[],
            requests=requests,
        )

        assert len(batch) == 3

    def test_batch_filter(self):
        """Test filtering a batch."""
        from vllm_mlx.mllm_batch_generator import MLLMBatch, MLLMBatchRequest

        requests = [
            MLLMBatchRequest(uid=i, request_id=f"req-{i}", prompt=f"prompt {i}")
            for i in range(4)
        ]

        batch = MLLMBatch(
            uids=[0, 1, 2, 3],
            request_ids=["req-0", "req-1", "req-2", "req-3"],
            y=mx.array([100, 200, 300, 400]),
            logprobs=[
                mx.array([0.1]),
                mx.array([0.2]),
                mx.array([0.3]),
                mx.array([0.4]),
            ],
            max_tokens=[100, 100, 100, 100],
            num_tokens=[0, 0, 0, 0],
            cache=[],
            requests=requests,
        )

        # Keep only indices 1 and 3
        batch.filter([1, 3])

        assert len(batch) == 2
        assert batch.uids == [1, 3]
        assert batch.request_ids == ["req-1", "req-3"]

    def test_batch_extend_handles_empty_protocol_caches_without_keys(self):
        """Caches with empty()/extend() but no .keys still need batch extension."""
        from vllm_mlx.mllm_batch_generator import MLLMBatch, MLLMBatchRequest

        class OpaqueCache:
            def __init__(self):
                self.extend_calls = 0
                self.extended_with = None

            def empty(self):
                return False

            def extend(self, other):
                self.extend_calls += 1
                self.extended_with = other

        primary_cache = OpaqueCache()
        other_cache = OpaqueCache()
        primary = MLLMBatch(
            uids=[1],
            request_ids=["req-1"],
            y=mx.array([100]),
            logprobs=[mx.array([0.1])],
            max_tokens=[100],
            num_tokens=[0],
            cache=[primary_cache],
            requests=[MLLMBatchRequest(uid=1, request_id="req-1", prompt="one")],
        )
        other = MLLMBatch(
            uids=[2],
            request_ids=["req-2"],
            y=mx.array([200]),
            logprobs=[mx.array([0.2])],
            max_tokens=[100],
            num_tokens=[0],
            cache=[other_cache],
            requests=[MLLMBatchRequest(uid=2, request_id="req-2", prompt="two")],
        )

        primary.extend(other)

        assert primary.y.shape == (2,)
        assert primary_cache.extend_calls == 1
        assert primary_cache.extended_with is other_cache


class TestMLLMBatchStats:
    """Tests for MLLMBatchStats."""

    def test_stats_initialization(self):
        """Test stats initialization."""
        from vllm_mlx.mllm_batch_generator import MLLMBatchStats

        stats = MLLMBatchStats()

        assert stats.prompt_tokens == 0
        assert stats.generation_tokens == 0
        assert stats.prompt_time == 0
        assert stats.generation_time == 0
        assert stats.num_images_processed == 0

    def test_tps_calculation(self):
        """Test tokens per second calculation."""
        from vllm_mlx.mllm_batch_generator import MLLMBatchStats

        stats = MLLMBatchStats()
        stats.prompt_tokens = 100
        stats.prompt_time = 2.0
        stats.generation_tokens = 50
        stats.generation_time = 1.0

        assert stats.prompt_tps == 50.0
        assert stats.generation_tps == 50.0

    def test_tps_zero_time(self):
        """Test TPS with zero time."""
        from vllm_mlx.mllm_batch_generator import MLLMBatchStats

        stats = MLLMBatchStats()

        assert stats.prompt_tps == 0
        assert stats.generation_tps == 0


class TestMLLMSchedulerConfig:
    """Tests for MLLMSchedulerConfig."""

    def test_default_config(self):
        """Test default configuration."""
        from vllm_mlx.mllm_scheduler import MLLMSchedulerConfig

        config = MLLMSchedulerConfig()

        assert config.max_num_seqs == 16
        # prefill_batch_size set equal to max_num_seqs to avoid batch extend issues
        assert config.prefill_batch_size == 16
        assert config.completion_batch_size == 16
        assert config.enable_vision_cache is True
        assert config.vision_cache_size == 100

    def test_custom_config(self):
        """Test custom configuration."""
        from vllm_mlx.mllm_scheduler import MLLMSchedulerConfig

        config = MLLMSchedulerConfig(
            max_num_seqs=8,
            prefill_batch_size=2,
            completion_batch_size=8,
            enable_vision_cache=False,
        )

        assert config.max_num_seqs == 8
        assert config.prefill_batch_size == 2
        assert config.completion_batch_size == 8
        assert config.enable_vision_cache is False


class TestMLLMRequest:
    """Tests for MLLMRequest dataclass."""

    def test_create_request(self):
        """Test creating an MLLM request."""
        from vllm_mlx.mllm_scheduler import MLLMRequest
        from vllm_mlx.request import RequestStatus

        req = MLLMRequest(
            request_id="req-1",
            prompt="Describe this image",
            images=["image.jpg"],
        )

        assert req.request_id == "req-1"
        assert req.prompt == "Describe this image"
        assert req.images == ["image.jpg"]
        assert req.status == RequestStatus.WAITING
        assert req.output_text == ""


class TestMLLMSchedulerOutput:
    """Tests for MLLMSchedulerOutput."""

    def test_empty_output(self):
        """Test empty scheduler output."""
        from vllm_mlx.mllm_scheduler import MLLMSchedulerOutput

        output = MLLMSchedulerOutput()

        assert output.scheduled_request_ids == []
        assert output.num_scheduled_tokens == 0
        assert output.finished_request_ids == set()
        assert output.outputs == []
        assert output.has_work is False


class TestMultimodalProcessorBatch:
    """Tests for MultimodalProcessor batch methods."""

    def test_batch_pixel_values_empty(self):
        """Test batching empty pixel values."""
        from vllm_mlx.multimodal_processor import MultimodalProcessor

        # Create mock processor
        mock_model = MagicMock()
        mock_processor = MagicMock()

        processor = MultimodalProcessor(mock_model, mock_processor)

        result = processor.batch_pixel_values([None, None])
        assert result is None

    def test_batch_pixel_values_single(self):
        """Test batching single pixel value."""
        from vllm_mlx.multimodal_processor import MultimodalProcessor

        mock_model = MagicMock()
        mock_processor = MagicMock()

        processor = MultimodalProcessor(mock_model, mock_processor)

        pixels = mx.ones((1, 3, 32, 32))
        result = processor.batch_pixel_values([pixels])

        assert result is not None
        assert result.shape == (1, 3, 32, 32)

    def test_batch_pixel_values_multiple(self):
        """Test batching multiple pixel values."""
        from vllm_mlx.multimodal_processor import MultimodalProcessor

        mock_model = MagicMock()
        mock_processor = MagicMock()

        processor = MultimodalProcessor(mock_model, mock_processor)

        pixels1 = mx.ones((1, 3, 32, 32))
        pixels2 = mx.ones((1, 3, 32, 32)) * 2

        result = processor.batch_pixel_values([pixels1, pixels2])

        assert result is not None
        assert result.shape == (2, 3, 32, 32)

    def test_batch_image_grid_thw(self):
        """Test batching image grid thw."""
        from vllm_mlx.multimodal_processor import MultimodalProcessor

        mock_model = MagicMock()
        mock_processor = MagicMock()

        processor = MultimodalProcessor(mock_model, mock_processor)

        grid1 = mx.array([[1, 4, 4]])
        grid2 = mx.array([[1, 8, 8]])

        result = processor.batch_image_grid_thw([grid1, grid2])

        assert result is not None
        assert result.shape[0] == 2

    def test_prepare_for_batch(self):
        """Test prepare_for_batch method."""
        from vllm_mlx.multimodal_processor import (
            MultimodalProcessor,
            ProcessedMultimodalInput,
        )

        mock_model = MagicMock()
        mock_processor = MagicMock()

        processor = MultimodalProcessor(mock_model, mock_processor)

        # Create processed inputs
        inputs = [
            ProcessedMultimodalInput(
                input_ids=mx.array([1, 2, 3]),
                pixel_values=mx.ones((1, 3, 32, 32)),
                num_images=1,
                num_tokens=3,
            ),
            ProcessedMultimodalInput(
                input_ids=mx.array([4, 5, 6, 7, 8]),
                pixel_values=mx.ones((1, 3, 32, 32)),
                num_images=1,
                num_tokens=5,
            ),
        ]

        input_ids, batch_kwargs, padding = processor.prepare_for_batch(inputs)

        # Check left-padding
        assert input_ids.shape == (2, 5)  # max length is 5
        assert padding == [2, 0]  # first input needs 2 padding

    def test_compute_vision_hash(self):
        """Test vision hash computation."""
        from vllm_mlx.multimodal_processor import MultimodalProcessor

        mock_model = MagicMock()
        mock_processor = MagicMock()

        processor = MultimodalProcessor(mock_model, mock_processor)

        pixels = mx.ones((1, 3, 32, 32))
        hash1 = processor.compute_vision_hash(pixels)
        hash2 = processor.compute_vision_hash(pixels)

        # Same input should give same hash
        assert hash1 == hash2
        assert len(hash1) == 16  # SHA256 truncated to 16 chars


class TestVisionCache:
    """Tests for VLM cache functionality."""

    def test_cache_creation(self):
        """Test VLM cache creation."""
        from vllm_mlx.mllm_cache import MLLMCacheManager

        cache = MLLMCacheManager(max_entries=10)

        assert len(cache) == 0
        assert cache.max_size == 10

    def test_cache_miss(self):
        """Test cache miss."""
        from vllm_mlx.mllm_cache import MLLMCacheManager

        cache = MLLMCacheManager()

        result, hit = cache.fetch_cache(["image.jpg"], "prompt")

        assert result is None
        assert hit is False
        assert cache.stats.misses == 1

    def test_cache_store_and_fetch(self):
        """Test storing and fetching from cache."""
        from vllm_mlx.mllm_cache import MLLMCacheManager

        cache = MLLMCacheManager()

        # Store cache
        test_cache = [{"key": "value"}]
        cache.store_cache(["image.jpg"], "prompt", test_cache, num_tokens=100)

        # Fetch cache
        result, hit = cache.fetch_cache(["image.jpg"], "prompt")

        assert result is not None
        assert hit is True
        assert cache.stats.hits == 1
        assert cache.stats.tokens_saved == 100

    def test_cache_eviction(self):
        """Test cache eviction when full."""
        from vllm_mlx.mllm_cache import MLLMCacheManager

        cache = MLLMCacheManager(max_entries=2)

        # Fill cache
        cache.store_cache(["img1.jpg"], "prompt1", [1], num_tokens=10)
        cache.store_cache(["img2.jpg"], "prompt2", [2], num_tokens=20)

        assert len(cache) == 2

        # Add one more (should evict oldest)
        cache.store_cache(["img3.jpg"], "prompt3", [3], num_tokens=30)

        assert len(cache) == 2
        assert cache.stats.evictions == 1

        # img1 should be evicted
        _, hit = cache.fetch_cache(["img1.jpg"], "prompt1")
        assert hit is False


# Integration tests (require model loading)
@pytest.mark.slow
@pytest.mark.skipif(not os.environ.get("RUN_SLOW_TESTS"), reason="Slow tests disabled")
class TestMLLMSchedulerIntegration:
    """Integration tests for MLLMScheduler with real models."""

    @pytest.fixture
    def test_image_path(self):
        """Create a test image."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            path = create_test_image(f.name)
            yield path
            os.unlink(path)

    async def test_single_request(self, test_image_path):
        """Test single MLLM request."""
        from vllm_mlx.mllm_scheduler import MLLMScheduler, MLLMSchedulerConfig
        from mlx_vlm import load

        # Load a small model
        model, processor = load("mlx-community/Qwen3-VL-4B-Instruct-3bit")

        config = MLLMSchedulerConfig(max_num_seqs=4)
        scheduler = MLLMScheduler(model, processor, config)

        await scheduler.start()

        try:
            request_id = scheduler.add_request(
                prompt="What's in this image?",
                images=[test_image_path],
                max_tokens=50,
            )

            # Run until complete
            while scheduler.has_requests():
                output = scheduler.step()
                if request_id in output.finished_request_ids:
                    break

            # Check result
            request = scheduler.get_request(request_id)
            assert request is not None
            assert len(request.output_tokens) > 0

        finally:
            await scheduler.stop()

    async def test_concurrent_requests(self, test_image_path):
        """Test multiple concurrent MLLM requests."""
        from vllm_mlx.mllm_scheduler import MLLMScheduler, MLLMSchedulerConfig
        from mlx_vlm import load

        model, processor = load("mlx-community/Qwen3-VL-4B-Instruct-3bit")

        config = MLLMSchedulerConfig(max_num_seqs=4)
        scheduler = MLLMScheduler(model, processor, config)

        await scheduler.start()

        try:
            # Add multiple requests
            request_ids = []
            for i in range(4):
                req_id = scheduler.add_request(
                    prompt=f"Describe image {i}",
                    images=[test_image_path],
                    max_tokens=30,
                )
                request_ids.append(req_id)

            # Run until all complete
            finished = set()
            while len(finished) < len(request_ids):
                output = scheduler.step()
                finished.update(output.finished_request_ids)

            # Check all completed
            assert len(finished) == 4

            # Check stats show batching
            stats = scheduler.get_stats()
            assert stats["num_requests_processed"] == 4

        finally:
            await scheduler.stop()

    async def test_streaming(self, test_image_path):
        """Test streaming MLLM generation."""
        from vllm_mlx.mllm_scheduler import MLLMScheduler, MLLMSchedulerConfig
        from mlx_vlm import load

        model, processor = load("mlx-community/Qwen3-VL-4B-Instruct-3bit")

        config = MLLMSchedulerConfig()
        scheduler = MLLMScheduler(model, processor, config)

        await scheduler.start()

        try:
            request_id = await scheduler.add_request_async(
                prompt="Describe this image briefly",
                images=[test_image_path],
                max_tokens=30,
            )

            tokens_received = 0
            async for output in scheduler.stream_outputs(request_id):
                tokens_received += len(output.new_token_ids)
                if output.finished:
                    break

            assert tokens_received > 0

        finally:
            await scheduler.stop()

    async def test_stream_outputs_consumer_break_after_finished_does_not_abort(self):
        """Breaking after a finished output is normal consumption, not orphaning."""
        from vllm_mlx.mllm_scheduler import MLLMScheduler
        from vllm_mlx.request import RequestOutput

        scheduler = MLLMScheduler.__new__(MLLMScheduler)
        scheduler.output_queues = {"req-1": asyncio.Queue()}
        scheduler.abort_request = MagicMock(return_value=True)

        await scheduler.output_queues["req-1"].put(
            RequestOutput(
                request_id="req-1",
                output_text="done",
                finished=True,
                finish_reason="stop",
            )
        )

        stream = MLLMScheduler.stream_outputs(scheduler, "req-1")
        output = await stream.__anext__()
        assert output.finished is True
        await stream.aclose()

        scheduler.abort_request.assert_not_called()
        assert "req-1" not in scheduler.output_queues


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestMLLMBatchGeneratorMTPGuards:
    def test_process_prompts_applies_request_sampling_to_first_token(self, monkeypatch):
        from vllm_mlx.mllm_batch_generator import (
            MLLMBatchGenerator,
            MLLMBatchRequest,
            MLLMBatchStats,
        )

        class FakeCache:
            def merge(self, caches):
                return self

        class RecordingProcessor:
            def __init__(self):
                self.calls = []

            def __call__(self, tokens, logits):
                self.calls.append(tokens.tolist())
                return logits

        processor = RecordingProcessor()
        request_sampler = MagicMock(return_value=mx.array([3], dtype=mx.uint32))
        fallback_sampler = MagicMock(return_value=mx.array([1], dtype=mx.uint32))
        sampler_calls = []

        def fake_make_sampler(**kwargs):
            sampler_calls.append(kwargs)
            return request_sampler

        monkeypatch.setattr(mx, "stream", lambda stream: nullcontext())
        monkeypatch.setattr(
            "mlx_lm.models.cache.make_prompt_cache",
            lambda model, **kwargs: [FakeCache()],
        )
        monkeypatch.setattr("mlx_lm.sample_utils.make_sampler", fake_make_sampler)
        monkeypatch.setattr(
            "mlx_lm.sample_utils.make_logits_processors", lambda **_: []
        )

        generator = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
        generator.max_kv_size = 0
        generator._stats = MLLMBatchStats()
        generator._pending_error_responses = []
        generator._aborted_request_ids = set()
        generator._prefill_progress = {}
        generator.prefix_cache = None
        generator.prefill_step_size = 512
        generator.language_model = object()
        generator.model = MagicMock()
        generator.sampler = fallback_sampler
        generator._trim_rotating_caches = lambda cache: None
        generator._preprocess_request = lambda req: None
        generator._run_chunked_text_prefill = lambda req, cache: mx.array(
            [[[0.0, 1.0, 2.0, 3.0]]]
        )

        request = MLLMBatchRequest(
            uid=7,
            request_id="req-7",
            prompt="hello",
            temperature=0.3,
            top_p=0.8,
            top_k=0,
            min_p=0.0,
            logits_processors=[processor],
        )
        request.input_ids = mx.array([[42]], dtype=mx.uint32)
        request.is_text_only = True

        batch = MLLMBatchGenerator._process_prompts(generator, [request])

        assert batch.y.tolist() == [3]
        assert batch.samplers == [request_sampler]
        assert batch.logits_processors == [[processor]]
        assert processor.calls == [[]]
        assert request_sampler.call_count == 1
        fallback_sampler.assert_not_called()
        assert sampler_calls == [{"temp": 0.3, "top_p": 0.8, "top_k": 0, "min_p": 0.0}]

    def test_next_passes_current_token_to_logits_processor_prefix(self):
        from vllm_mlx.mllm_batch_generator import (
            MLLMBatch,
            MLLMBatchGenerator,
            MLLMBatchRequest,
            MLLMBatchStats,
        )

        captured = {}

        def fake_step(input_tokens, cache, logits_processors, output_tokens, samplers):
            captured["input_tokens"] = input_tokens.tolist()
            captured["output_tokens"] = output_tokens
            return mx.array([11]), [mx.array([0.2, 0.8])]

        generator = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
        generator.max_kv_size = 0
        generator._stats = MLLMBatchStats()
        generator.stop_tokens = set()
        generator.unprocessed_requests = []
        generator._pending_error_responses = []
        generator._prefill_progress = {}
        generator.prefix_cache = None
        generator._maybe_store_prefix_cache = lambda batch, end_idx: None
        generator._step = fake_step

        processor = MagicMock()
        request = MLLMBatchRequest(uid=1, request_id="req-1", prompt="hello")
        request.output_tokens = [5]
        generator.active_batch = MLLMBatch(
            uids=[1],
            request_ids=["req-1"],
            y=mx.array([7]),
            logprobs=[mx.array([0.5, 0.5])],
            max_tokens=[8],
            num_tokens=[1],
            cache=[],
            requests=[request],
            logits_processors=[[processor]],
            samplers=None,
        )

        responses = MLLMBatchGenerator._next(generator)

        assert [r.token for r in responses] == [7]
        assert captured["input_tokens"] == [[7]]
        assert captured["output_tokens"] == [[5, 7]]
        assert request.output_tokens == [5, 7]

    def test_install_mtp_mllm_disables_mtp_when_logits_processors_active(self):
        from vllm_mlx.mllm_batch_generator import install_mtp_mllm

        expected_tokens = mx.array([7])
        expected_logprobs = [mx.array([0.1, 0.9])]
        original_step = MagicMock(return_value=(expected_tokens, expected_logprobs))

        class FakeBatchGen:
            def __init__(self):
                self._step = original_step
                self._next = MagicMock(return_value=[])
                self.active_batch = MagicMock()
                self.active_batch.__len__.return_value = 1
                self.active_batch.requests = [
                    MagicMock(
                        temperature=0.0,
                        top_p=1.0,
                        top_k=0,
                        min_p=0.0,
                    )
                ]
                self.sampler = MagicMock()

        batch_gen = FakeBatchGen()
        language_model = MagicMock()

        install_mtp_mllm(batch_gen, language_model, num_draft_tokens=4)

        stats = batch_gen.get_mtp_stats()
        assert stats["enabled"] is True
        assert stats["requested_draft_tokens"] == 4
        assert stats["effective_draft_tokens"] == 1
        assert stats["attempted"] == 0
        assert (
            stats["bypass_counts_semantics"]
            == "per_condition_overlapping_not_total_steps"
        )
        assert stats["bypass_counts"] == {
            "prefill": 0,
            "no_active_batch": 0,
            "concurrent_batch": 0,
            "non_greedy_sampling": 0,
            "logits_processors": 0,
        }

        logits_processor = MagicMock()
        tokens, logprobs = batch_gen._step(
            mx.array([[123]]),
            cache=[],
            logits_processors=[[logits_processor]],
            output_tokens=[[1, 2]],
            samplers=[None],
        )

        assert tokens.tolist() == expected_tokens.tolist()
        assert [lp.tolist() for lp in logprobs] == [
            lp.tolist() for lp in expected_logprobs
        ]
        original_step.assert_called_once()
        language_model.assert_not_called()
        language_model.mtp_forward.assert_not_called()
        stats = batch_gen.get_mtp_stats()
        assert stats["attempted"] == 0
        assert stats["bypass_counts"]["logits_processors"] == 1

    @pytest.mark.parametrize(
        "sampling_kwargs",
        [
            {"temperature": 0.6, "top_p": 1.0, "top_k": 0, "min_p": 0.0},
            {"temperature": 0.0, "top_p": 0.95, "top_k": 0, "min_p": 0.0},
            {"temperature": 0.0, "top_p": 1.0, "top_k": 20, "min_p": 0.0},
            {"temperature": 0.0, "top_p": 1.0, "top_k": 0, "min_p": 0.05},
        ],
    )
    def test_install_mtp_mllm_disables_mtp_for_non_greedy_sampling(
        self, sampling_kwargs
    ):
        from vllm_mlx.mllm_batch_generator import install_mtp_mllm

        expected_tokens = mx.array([11])
        expected_logprobs = [mx.array([0.3, 0.7])]
        original_step = MagicMock(return_value=(expected_tokens, expected_logprobs))

        class FakeBatchGen:
            def __init__(self):
                self._step = original_step
                self._next = MagicMock(return_value=[])
                self.active_batch = MagicMock()
                self.active_batch.__len__.return_value = 1
                self.active_batch.requests = [MagicMock(**sampling_kwargs)]
                self.sampler = MagicMock()

        batch_gen = FakeBatchGen()
        language_model = MagicMock()

        install_mtp_mllm(batch_gen, language_model, num_draft_tokens=4)

        tokens, logprobs = batch_gen._step(
            mx.array([[321]]),
            cache=[],
            logits_processors=None,
            output_tokens=None,
            samplers=[MagicMock()],
        )

        assert tokens.tolist() == expected_tokens.tolist()
        assert [lp.tolist() for lp in logprobs] == [
            lp.tolist() for lp in expected_logprobs
        ]
        original_step.assert_called_once()
        language_model.assert_not_called()
        language_model.mtp_forward.assert_not_called()
        stats = batch_gen.get_mtp_stats()
        assert stats["attempted"] == 0
        assert stats["bypass_counts"]["non_greedy_sampling"] == 1

    def test_install_mtp_mllm_counts_structural_bypasses(self):
        from vllm_mlx.mllm_batch_generator import install_mtp_mllm

        expected_tokens = mx.array([11])
        expected_logprobs = [mx.array([0.3, 0.7])]
        original_step = MagicMock(return_value=(expected_tokens, expected_logprobs))

        class FakeBatchGen:
            def __init__(self):
                self._step = original_step
                self._next = MagicMock(return_value=[])
                self.active_batch = MagicMock()
                self.active_batch.__len__.return_value = 1
                self.active_batch.requests = [
                    MagicMock(
                        temperature=0.0,
                        top_p=1.0,
                        top_k=0,
                        min_p=0.0,
                    )
                ]
                self.sampler = MagicMock()

        batch_gen = FakeBatchGen()
        language_model = MagicMock()

        install_mtp_mllm(batch_gen, language_model, num_draft_tokens=4)

        batch_gen._step(
            mx.array([[1, 2]]),
            cache=[],
            logits_processors=None,
            output_tokens=None,
            samplers=None,
        )
        batch_gen.active_batch = None
        batch_gen._step(
            mx.array([[3]]),
            cache=[],
            logits_processors=None,
            output_tokens=None,
            samplers=None,
        )
        batch_gen.active_batch = MagicMock()
        batch_gen.active_batch.__len__.return_value = 2
        batch_gen.active_batch.requests = [
            MagicMock(temperature=0.0, top_p=1.0, top_k=0, min_p=0.0),
            MagicMock(temperature=0.0, top_p=1.0, top_k=0, min_p=0.0),
        ]
        batch_gen._step(
            mx.array([[4]]),
            cache=[],
            logits_processors=None,
            output_tokens=None,
            samplers=None,
        )

        assert original_step.call_count == 3
        language_model.assert_not_called()
        language_model.mtp_forward.assert_not_called()
        stats = batch_gen.get_mtp_stats()
        assert stats["attempted"] == 0
        assert stats["bypass_counts"]["prefill"] == 1
        assert stats["bypass_counts"]["no_active_batch"] == 1
        assert stats["bypass_counts"]["concurrent_batch"] == 1

    def test_install_mtp_mllm_accepted_drafts_bypass_request_sampler(self):
        from vllm_mlx.mllm_batch_generator import MLLMBatchResponse, install_mtp_mllm

        class FakeBatchGen:
            def __init__(self):
                self._step = MagicMock()
                self._next = MagicMock(
                    return_value=[
                        MLLMBatchResponse(
                            uid=7,
                            request_id="req-7",
                            token=1,
                            logprobs=mx.array([0.0, 0.0, 0.0, 0.0, 0.0]),
                            finish_reason=None,
                        )
                    ]
                )
                self.active_batch = MagicMock()
                self.active_batch.__len__.return_value = 1
                self.active_batch.uids = [7]
                request = MagicMock(
                    request_id="req-7",
                    temperature=0.0,
                    top_p=1.0,
                    top_k=0,
                    min_p=0.0,
                    output_tokens=[],
                )
                self.active_batch.requests = [request]
                self.active_batch.num_tokens = [0]
                self.active_batch.max_tokens = [16]
                self.stop_tokens = set()
                self.sampler = MagicMock(return_value=mx.array([1], dtype=mx.uint32))
                self._maybe_store_prefix_cache = MagicMock()

        batch_gen = FakeBatchGen()
        request_sampler = MagicMock(return_value=mx.array([1], dtype=mx.uint32))

        class FakeLanguageModel:
            def mtp_forward(self, hidden_states, next_token_ids, mtp_cache=None):
                logits = mx.full((1, 1, 5), -1000.0)
                logits[:, :, 2] = 0.0
                return logits

            def __call__(self, verify_input, cache=None, return_hidden=False):
                logits = mx.full((1, 2, 5), -1000.0)
                logits[:, 0, 2] = 0.0
                logits[:, 1, 3] = 0.0
                return logits, mx.zeros((1, 2, 4))

        install_mtp_mllm(batch_gen, FakeLanguageModel(), num_draft_tokens=1)

        batch_gen._step(
            mx.array([[123]], dtype=mx.uint32),
            cache=[],
            logits_processors=None,
            output_tokens=[[]],
            samplers=[request_sampler],
        )
        responses = batch_gen._next()

        assert [r.token for r in responses] == [1, 2]
        assert request_sampler.call_count == 1
        assert batch_gen.sampler.call_count == 0
        assert batch_gen.get_mtp_stats()["attempted"] == 1
        assert batch_gen.get_mtp_stats()["accepted"] == 1
        assert batch_gen.get_mtp_stats()["acceptance_rate"] == 1.0

    def test_next_keeps_retired_processors_by_default(self, monkeypatch):
        from vllm_mlx.mllm_batch_generator import (
            MLLMBatch,
            MLLMBatchGenerator,
            MLLMBatchRequest,
            MLLMBatchStats,
        )

        monkeypatch.delenv("VLLM_MLX_ENABLE_THINKING_RETIREMENT_RESUME", raising=False)

        class RetiredProcessor:
            is_retired = True

            def __call__(self, tokens, logits):
                return logits

        processor = RetiredProcessor()
        generator = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
        generator.max_kv_size = 0
        generator._stats = MLLMBatchStats()
        generator.stop_tokens = set()
        generator.unprocessed_requests = []
        generator._pending_error_responses = []
        generator._prefill_progress = {}
        generator.prefix_cache = None
        generator._maybe_store_prefix_cache = lambda batch, end_idx: None
        generator._step = lambda *args, **kwargs: (
            mx.array([11]),
            [mx.array([0.2, 0.8])],
        )

        request = MLLMBatchRequest(uid=1, request_id="req-1", prompt="hello")
        generator.active_batch = MLLMBatch(
            uids=[1],
            request_ids=["req-1"],
            y=mx.array([7]),
            logprobs=[mx.array([0.5, 0.5])],
            max_tokens=[4],
            num_tokens=[0],
            cache=[],
            requests=[request],
            logits_processors=[[processor]],
            samplers=None,
        )

        responses = MLLMBatchGenerator._next(generator)

        assert len(responses) == 1
        assert generator.active_batch is not None
        assert generator.active_batch.logits_processors == [[processor]]

    def test_next_drops_retired_processors_only_when_enabled(self, monkeypatch):
        from vllm_mlx.mllm_batch_generator import (
            MLLMBatch,
            MLLMBatchGenerator,
            MLLMBatchRequest,
            MLLMBatchStats,
        )

        monkeypatch.setenv("VLLM_MLX_ENABLE_THINKING_RETIREMENT_RESUME", "1")

        class RetiredProcessor:
            is_retired = True

            def __call__(self, tokens, logits):
                return logits

        generator = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
        generator.max_kv_size = 0
        generator._stats = MLLMBatchStats()
        generator.stop_tokens = set()
        generator.unprocessed_requests = []
        generator._pending_error_responses = []
        generator._prefill_progress = {}
        generator.prefix_cache = None
        generator._maybe_store_prefix_cache = lambda batch, end_idx: None
        generator._step = lambda *args, **kwargs: (
            mx.array([11]),
            [mx.array([0.2, 0.8])],
        )

        request = MLLMBatchRequest(uid=1, request_id="req-1", prompt="hello")
        generator.active_batch = MLLMBatch(
            uids=[1],
            request_ids=["req-1"],
            y=mx.array([7]),
            logprobs=[mx.array([0.5, 0.5])],
            max_tokens=[4],
            num_tokens=[0],
            cache=[],
            requests=[request],
            logits_processors=[[RetiredProcessor()]],
            samplers=None,
        )

        responses = MLLMBatchGenerator._next(generator)

        assert len(responses) == 1
        assert generator.active_batch is not None
        assert generator.active_batch.logits_processors == [None]


class TestBatchedMLLMConfigWiring:
    def test_batched_engine_forwards_prefill_step_size_to_mllm_scheduler(
        self, monkeypatch
    ):
        from vllm_mlx.engine.batched import BatchedEngine
        from vllm_mlx.scheduler import SchedulerConfig

        captured = {}

        class FakeMLXMultimodalLM:
            def __init__(self, model_name, trust_remote_code=True, **kwargs):
                self.model_name = model_name
                self.model = object()
                self.processor = object()

            def load(self):
                return None

        class FakeMLLMSchedulerConfig:
            def __init__(self, **kwargs):
                captured["config_kwargs"] = kwargs
                self.__dict__.update(kwargs)

        class FakeMLLMScheduler:
            def __init__(self, model, processor, config):
                captured["scheduler_config"] = config

            async def start(self):
                return None

        import vllm_mlx.engine.batched as batched_mod
        import vllm_mlx.mllm_scheduler as mllm_sched_mod
        import vllm_mlx.models.mllm as mllm_model_mod

        monkeypatch.setattr(mllm_model_mod, "MLXMultimodalLM", FakeMLXMultimodalLM)
        monkeypatch.setattr(mllm_sched_mod, "MLLMScheduler", FakeMLLMScheduler)
        monkeypatch.setattr(
            mllm_sched_mod, "MLLMSchedulerConfig", FakeMLLMSchedulerConfig
        )
        monkeypatch.setattr(
            batched_mod.BatchedEngine, "_inject_mtp_mllm", lambda self: None
        )

        cfg = SchedulerConfig(
            prefill_batch_size=4,
            completion_batch_size=8,
            prefill_step_size=256,
            enable_prefix_cache=False,
            use_memory_aware_cache=False,
            cache_memory_mb=123,
            enable_mtp=False,
        )
        engine = BatchedEngine(
            model_name="fake-qwen",
            scheduler_config=cfg,
            force_mllm=True,
        )

        asyncio.run(engine._start_mllm())

        assert captured["config_kwargs"]["prefill_step_size"] == 256
        assert captured["config_kwargs"]["enable_prefix_cache"] is False
        assert captured["config_kwargs"]["use_memory_aware_cache"] is False
        assert captured["config_kwargs"]["cache_memory_mb"] == 123
        assert captured["config_kwargs"]["prefix_cache_memory_mb"] == 123


class TestPreprocessIdempotent:
    """_preprocess_request must be idempotent for text-only requests.

    The scheduler offloads preprocessing to a thread-pool executor so
    the event loop stays responsive.  _process_prompts then calls
    _preprocess_request again — the second call must be a no-op.
    """

    def test_text_only_not_preprocessed_twice(self):
        """When input_ids is already set (executor did it), skip."""
        from vllm_mlx.mllm_batch_generator import MLLMBatchRequest

        req = MLLMBatchRequest(
            uid=0,
            prompt="Hello",
            request_id="test-idem",
        )
        # Simulate executor having already set input_ids
        req.input_ids = mx.array([[1, 2, 3]])

        # Build a minimal batch generator with the method
        from vllm_mlx.mllm_batch_generator import MLLMBatchGenerator

        gen = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
        gen.max_kv_size = 0
        gen._preprocess_request = MLLMBatchGenerator._preprocess_request.__get__(
            gen, MLLMBatchGenerator
        )

        # Must return immediately without touching prepare_inputs
        gen._preprocess_request(req)
        assert req.input_ids.shape == (1, 3)

    def test_vision_request_not_skipped(self):
        """Vision requests should NOT be skipped even with input_ids set."""
        from vllm_mlx.mllm_batch_generator import MLLMBatchRequest

        req = MLLMBatchRequest(
            uid=0,
            prompt="Describe",
            request_id="test-vis",
            images=["fake.png"],
        )
        req.input_ids = mx.array([[1, 2, 3]])

        from vllm_mlx.mllm_batch_generator import MLLMBatchGenerator

        gen = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
        gen.max_kv_size = 0
        gen._preprocess_request = MLLMBatchGenerator._preprocess_request.__get__(
            gen, MLLMBatchGenerator
        )

        # Should NOT return early — will try to import prepare_inputs
        with pytest.raises(Exception):
            gen._preprocess_request(req)


class TestChunkedPrefillCacheHandling:
    """Tests for chunked prefill prefix cache handling paths."""

    def _make_fake_batch_gen(self):
        """Build a minimal fake MLLMBatchGenerator for chunked prefill tests."""
        from vllm_mlx.mllm_batch_generator import MLLMBatchGenerator, MLLMBatchStats

        gen = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
        gen._stats = MLLMBatchStats()
        gen._pending_error_responses = []
        gen._aborted_request_ids = set()
        gen._prefill_progress = {}
        gen.active_batch = None
        gen.stop_tokens = set()
        gen.unprocessed_requests = []
        gen._think_suffix_len = 0
        gen.max_kv_size = 0

        # _has_empty_rotating_cache — always False for test caches
        gen._has_empty_rotating_cache = lambda cache: False

        return gen

    def _make_fake_kv_cache(self, offset=100):
        """Create a real KVCache with the given offset."""
        from mlx_lm.models.cache import KVCache

        cache = KVCache()
        # Populate cache by updating with dummy data
        if offset > 0:
            k = mx.zeros((1, 1, offset, 4))
            v = mx.zeros((1, 1, offset, 4))
            cache.update_and_fetch(k, v)
            mx.eval(cache.keys, cache.values)
        return cache

    def test_exact_hit_uses_trim_cache_offset(self, monkeypatch):
        """Exact prefix-cache hit must use _trim_cache_offset(kv, 1),
        NOT _copy_prefix_cache, to reduce the cache offset by 1."""
        from vllm_mlx.mllm_batch_generator import (
            MLLMBatchRequest,
            install_chunked_prefill_mllm,
        )

        gen = self._make_fake_batch_gen()

        # Track which functions are called
        trim_calls = []
        copy_calls = []

        import vllm_mlx.mllm_batch_generator as bg_mod

        orig_trim = bg_mod._trim_cache_offset

        def tracking_trim(cache, n):
            trim_calls.append(n)
            return orig_trim(cache, n)

        monkeypatch.setattr(bg_mod, "_trim_cache_offset", tracking_trim)

        gen._copy_prefix_cache = lambda kv: (copy_calls.append(1), kv)[1]
        gen._trim_rotating_caches = lambda cache: None

        # Fake prefix cache: exact hit → remaining_ids = [] (empty)
        fake_kv = [self._make_fake_kv_cache(offset=50)]

        class FakePrefixCache:
            def fetch(self, ids):
                return fake_kv, []  # exact hit

        gen.prefix_cache = FakePrefixCache()

        # Fake language model and _orig_next
        gen.language_model = MagicMock()
        orig_next_calls = []
        gen._next = lambda: orig_next_calls.append(1) or []

        # Install chunked prefill (budget large enough that 1 token falls through)
        install_chunked_prefill_mllm(gen, budget=1024)

        # Create a request
        req = MLLMBatchRequest(uid=1, request_id="req-exact", prompt="hello")
        req.input_ids = mx.array([[10, 20, 30, 40, 50]])
        req.is_text_only = True
        req.images = None
        req.videos = None
        gen.unprocessed_requests.append(req)

        # Preprocess is a no-op (input_ids already set)
        gen._preprocess_request = lambda r: None

        gen._next()

        # _trim_cache_offset should have been called with trim_by=1
        assert trim_calls == [
            1
        ], f"Expected _trim_cache_offset(kv, 1), got {trim_calls}"
        # _copy_prefix_cache should NOT have been called
        assert copy_calls == [], "Exact hit should NOT call _copy_prefix_cache"

    def test_partial_hit_uses_copy_prefix_cache(self, monkeypatch):
        """Partial prefix-cache hit must use _copy_prefix_cache (not _trim_cache_offset)."""
        from vllm_mlx.mllm_batch_generator import (
            MLLMBatchRequest,
            install_chunked_prefill_mllm,
        )

        gen = self._make_fake_batch_gen()

        trim_calls = []
        copy_calls = []

        import vllm_mlx.mllm_batch_generator as bg_mod

        orig_trim = bg_mod._trim_cache_offset

        def tracking_trim(cache, n):
            trim_calls.append(n)
            return orig_trim(cache, n)

        monkeypatch.setattr(bg_mod, "_trim_cache_offset", tracking_trim)

        gen._copy_prefix_cache = lambda kv: (copy_calls.append(1), kv)[1]
        gen._trim_rotating_caches = lambda cache: None

        # Fake prefix cache: partial hit → remaining_ids has tokens
        fake_kv = [self._make_fake_kv_cache(offset=3)]

        class FakePrefixCache:
            def fetch(self, ids):
                return fake_kv, [40, 50]  # partial hit

        gen.prefix_cache = FakePrefixCache()

        gen.language_model = MagicMock()
        orig_next_calls = []
        gen._next = lambda: orig_next_calls.append(1) or []

        install_chunked_prefill_mllm(gen, budget=1024)

        req = MLLMBatchRequest(uid=2, request_id="req-partial", prompt="hello world")
        req.input_ids = mx.array([[10, 20, 30, 40, 50]])
        req.is_text_only = True
        req.images = None
        req.videos = None
        gen.unprocessed_requests.append(req)
        gen._preprocess_request = lambda r: None

        gen._next()

        # Partial hit uses _copy_prefix_cache, NOT _trim_cache_offset
        assert copy_calls == [
            1
        ], f"Expected _copy_prefix_cache called once, got {copy_calls}"
        assert (
            trim_calls == []
        ), f"Partial hit should NOT call _trim_cache_offset, got {trim_calls}"

    def test_abort_cleans_up_partial_prefill(self):
        """Aborting a request during chunked prefill must clean up _partial."""
        from vllm_mlx.mllm_batch_generator import (
            MLLMBatchRequest,
            install_chunked_prefill_mllm,
        )

        gen = self._make_fake_batch_gen()
        gen.prefix_cache = None
        gen.language_model = MagicMock()
        gen._next = lambda: []

        install_chunked_prefill_mllm(gen, budget=1024)

        # Simulate an in-progress partial prefill
        req = MLLMBatchRequest(uid=3, request_id="req-abort", prompt="long text")
        req.input_ids = mx.array([[1, 2, 3, 4, 5]])
        req.is_text_only = True
        gen._partial = {
            "request": req,
            "cache": [self._make_fake_kv_cache(offset=2)],
            "remaining_ids": mx.array([[3, 4, 5]]),
            "processed": 2,
            "total": 5,
            "cached_count": 0,
            "chunk_count": 1,
        }
        gen._prefill_progress["req-abort"] = (2, 5)

        # Mark request as aborted
        gen._aborted_request_ids.add("req-abort")

        responses = gen._next()

        # Partial should be cleared
        assert gen._partial is None
        # Prefill progress should be cleaned up
        assert "req-abort" not in gen._prefill_progress
        # Should get an abort response (from _pending_error_responses via _generation_step)
        abort_responses = [r for r in responses if r.finish_reason == "abort"]
        assert len(abort_responses) == 1
        assert abort_responses[0].request_id == "req-abort"

    def test_short_prompt_falls_through_to_orig_next(self):
        """Short prompts (< budget) with no prefix cache must fall through
        to _orig_next, not be handled by the chunked prefill path."""
        from vllm_mlx.mllm_batch_generator import (
            MLLMBatchRequest,
            install_chunked_prefill_mllm,
        )

        gen = self._make_fake_batch_gen()
        gen.prefix_cache = None
        gen.language_model = MagicMock()

        orig_next_called = []
        gen._next = lambda: orig_next_called.append(1) or []

        install_chunked_prefill_mllm(gen, budget=1024)

        req = MLLMBatchRequest(uid=4, request_id="req-short", prompt="hi")
        req.input_ids = mx.array([[10, 20, 30]])  # 3 tokens < 1024 budget
        req.is_text_only = True
        req.images = None
        req.videos = None
        gen.unprocessed_requests.append(req)
        gen._preprocess_request = lambda r: None

        gen._next()

        # Should have fallen through to _orig_next
        assert orig_next_called == [
            1
        ], f"Short prompt should fall through to _orig_next, got {orig_next_called}"
